// Twin world package (M2, docs/twin/m2-world.md): the immutable, pointer-free
// compiled form of one Stadium map's static world. FV-free.
//
// One contiguous little-endian blob: a PackHeader followed by 64-byte aligned
// sections of fixed-size PODs. Nothing in the blob is a pointer; every
// cross-reference is an index. The blob carries a format version and a
// 64-bit content hash over every byte (header included, hash field zeroed),
// so it can be saved, mapped, shared between threads and compared by hash.
//
// Produced by tw::fv::CompileWorldPack (src/twin/fv/compile_world.cpp) from
// FV's flat scene export. Read through WorldPackView (typed spans into the
// blob, built and validated by OpenWorldPack).
//
// Orders and identities:
//   - Triangles (`Tris`) are in FV surface order: actor installation order,
//     then the actor's collision-tree surfaces in FV's tree order, then the
//     mesh's triangle index order. Vertices keep FV's stored order
//     (front = (v1-v0)x(v2-v0)). Positions are FV's own world transform
//     (GmVec3::SetMult(local, surface localToWorld)), bit for bit.
//   - `TriInfo.fvOrder` is the triangle's rank in FV's detection order
//     (target groups 1, 3, 4; per group the moving-pair corpora, then FV's
//     static record tree preorder; the mesh octree leaf order): the stable
//     tie-break of board 20 §3.5.
//   - An instance is one FV collision surface placed in the world: a static
//     record, or a surface of a moving-pair corpus (kInstMovingPair).
//     Instances are in surface order, so each actor's instances and
//     triangles are contiguous ranges.
//   - A model is one distinct mesh-local surface (vertices, triangles,
//     stored planes, material ids and octree leaf order all equal). Instance
//     triangle k is model triangle k transformed by the instance iso.
//   - Actor index = FV CudaHostScene actor index = installation order; it is
//     the actor field of Tri::meta.
//   - Trigger id = index into `Triggers` = FV race-trigger installation order
//     (TwRaceMap::triggers).
#ifndef TW_WORLD_PACK_H
#define TW_WORLD_PACK_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "tw_race.h"   // race::TwTrigger, race::TwSpawn, race::TwRaceMap (header-only, FV-free)
#include "tw_world.h"  // tw::Tri (64 B), meta bits, tw::World (B's bench form)

namespace tw::world {

constexpr char kPackMagic[8] = {'T', 'W', 'W', 'P', 'A', 'C', 'K', '\0'};
constexpr std::uint32_t kPackVersion = 1u;
constexpr std::uint32_t kPackAlign = 64u;

// ---- sections ------------------------------------------------------------------
enum class Section : std::uint32_t {
    Tris = 0,          // tw::Tri                      world triangles, surface order
    TriPlanes,         // TriPlane                     FV stored plane, rotated to world
    TriInfo,           // TriInfo                      instance, model triangle, FV order, trigger id
    Instances,         // Instance                     one per FV static mesh record
    Models,            // Model                        distinct mesh-local surfaces
    ModelTris,         // ModelTri                     mesh-local triangles of all models
    ModelVerts,        // Vec3f                        mesh-local vertices of all models
    ModelInstances,    // u32                          instance ids grouped by model (Model::firstRef)
    Actors,            // Actor                        one per FV scene actor (installation order)
    Triggers,          // race::TwTrigger              race trigger table (TwRaceMap::triggers)
    TriggerVolumes,    // TriggerVolume                geometry of each trigger id
    Spawns,            // race::TwSpawn                [0] = race start spawn (unseeded), then CP spawns
    Specials,          // SpecialSurface               booster / turbo / free-wheel surfaces per actor
    WaterCells,        // u8                           water grid cells (PackHeader::water)
    Count
};
constexpr std::uint32_t kSectionCount = static_cast<std::uint32_t>(Section::Count);

struct SectionEntry {           // 24 B
    std::uint32_t elemBytes;    // sizeof the element type, checked on open
    std::uint32_t reserved;
    std::uint64_t offset;       // from the start of the blob, multiple of kPackAlign
    std::uint64_t count;        // elements
};
static_assert(sizeof(SectionEntry) == 24, "SectionEntry layout");

// ---- element types (no implicit padding anywhere) ---------------------------------
struct Vec3f {
    float x, y, z;
};
static_assert(sizeof(Vec3f) == 12, "Vec3f layout");

// FV's stored per-triangle plane (GmSurfMeshTriangle normal/planeDistance,
// convention n.x + d = 0), with the normal rotated to world by FV's own
// GmVec3::SetMult(n, rotation) and d recomputed in world on v0 with FV's
// ComputePlane expression. FV's ellipsoid path does not read it (board 20
// §2.2); it is for culls, patches and plane classes.
struct TriPlane {
    float n[3];
    float d;
};
static_assert(sizeof(TriPlane) == 16, "TriPlane layout");

constexpr std::uint16_t kNoTrigger = 0xFFFFu;

struct TriInfo {                // 16 B, parallel to Tris
    std::uint32_t instance;     // Instances index
    std::uint32_t modelTri;     // triangle index inside the instance's model (= FV mesh triangle index)
    std::uint32_t fvOrder;      // rank in FV detection order (0..triCount-1, a permutation)
    std::uint16_t trigger;      // trigger id (Triggers index), kNoTrigger for solid triangles
    std::uint8_t group;         // FV collision group of the owning actor (1 trigger, 4 block)
    std::uint8_t flags;         // kTriDegenerate
};
static_assert(sizeof(TriInfo) == 16, "TriInfo layout");
constexpr std::uint8_t kTriDegenerate = 1u << 0;  // FV skips it: |(v1-v0)x(v2-v0)|^2 <= 1e-10 in unit space is shape dependent;
                                                  // flagged here when the world cross product is exactly 0

constexpr std::uint8_t kRotGeneral = 0xFFu;       // Instance::rotation: not a signed permutation
constexpr std::uint8_t kYawNone = 0xFFu;          // Instance::yaw: not a pure rotation about +y

struct Instance {               // 128 B
    float iso[12];              // FV localToWorld: basisX, basisY, basisZ (rotation columns), translation
    float lo[3], hi[3];         // world AABB of the instance triangles
    std::uint32_t model;        // Models index
    std::uint32_t actor;        // Actors index
    std::uint32_t firstTri;     // first world triangle (Tris index)
    std::uint32_t triCount;     // = Models[model].triCount
    std::uint32_t fvSurface;    // FV CudaHostScene surface index
    std::uint32_t fvRecord;     // leaf rank of this record in its group's FV static record tree (preorder);
                                // kNoRecord for a moving-pair instance (kInstMovingPair)
    std::int16_t cell[3];       // Stadium grid cell (32 x 8 x 32 m) of the AABB centre: floor(c / size)
    std::uint8_t rotation;      // signed-permutation code (RotationCode), kRotGeneral otherwise
    std::uint8_t yaw;           // quarter turns about +y (0..3) when the rotation is a pure yaw, else kYawNone
    std::uint8_t group;         // FV collision group of the actor
    std::uint8_t flags;         // kInst*
    std::uint16_t reserved0;
    std::uint32_t reserved[5];
};
static_assert(sizeof(Instance) == 128, "Instance layout");
constexpr std::uint8_t kInstTrigger = 1u << 0;       // group-1 race trigger surface
constexpr std::uint8_t kInstPermExact = 1u << 1;     // signed permutation AND model -> world (perm(v) + t) is bit-equal to FV's SetMult
constexpr std::uint8_t kInstMirrored = 1u << 2;      // determinant -1
// The actor is a non-static corpus of a target group (Stadium: one "Generated"
// environment object, ~5.8k triangles at y 193-229 m). FV's Reference
// detection tests it through the moving-pair path, before the group's static
// records; it is solid like any group-4 block. FV's CUDA scene omits it.
constexpr std::uint8_t kInstMovingPair = 1u << 3;
constexpr std::uint32_t kNoRecord = 0xFFFFFFFFu;

struct Model {                  // 64 B
    std::uint64_t hash;         // content hash of the mesh-local data
    std::uint32_t firstTri;     // ModelTris range
    std::uint32_t triCount;
    std::uint32_t firstVert;    // ModelVerts range (ModelTri::v is relative to it)
    std::uint32_t vertCount;
    std::uint32_t firstRef;     // ModelInstances range: the instances of this model
    std::uint32_t instanceCount;
    float lo[3], hi[3];         // mesh-local AABB of the vertices used by triangles
    std::uint32_t materialMask; // bit m set when a triangle has material id m (< 32)
    std::uint32_t flags;        // kModel*
};
static_assert(sizeof(Model) == 64, "Model layout");
constexpr std::uint32_t kModelTrigger = 1u << 0;      // used by a trigger instance
constexpr std::uint32_t kModelSolid = 1u << 1;        // used by a solid (group 4) instance

struct ModelTri {               // 32 B
    std::uint32_t v[3];         // vertex indices relative to Model::firstVert, FV stored order
    float n[3];                 // FV stored mesh-local normal (GmSurfMeshTriangle::normal)
    float d;                    // FV stored planeDistance (n.x + d = 0)
    std::uint32_t meta;         // material id (bits 0-7) | octree leaf rank inside the mesh << 8
};
static_assert(sizeof(ModelTri) == 32, "ModelTri layout");
constexpr std::uint32_t kModelTriMaterialMask = 0xFFu;
constexpr int kModelTriOctreeShift = 8;

struct Actor {                  // 32 B; index = Tri::meta >> kMetaActorShift
    float peerZ[3];             // FV corpus world +z (actor worldPose basisZ): booster axis, w.peer_z
    std::uint32_t peerId;       // FV corpus id = installationOrder + 1: w.peer_id, turbo source id
    std::uint16_t trigger;      // trigger id for race triggers, kNoTrigger otherwise
    std::uint8_t group;         // FV collision group (CHmsItem collisionGroup)
    std::uint8_t purpose;       // FV StaticScenePurpose
    std::uint32_t firstInstance;
    std::uint32_t instanceCount;
    std::uint32_t flags;        // kActor*
};
static_assert(sizeof(Actor) == 32, "Actor layout");
constexpr std::uint32_t kActorCollisionStatic = 1u << 0;
constexpr std::uint32_t kActorHasCheckpoint = 1u << 1;
constexpr std::uint32_t kActorHasSpawn = 1u << 2;
constexpr std::uint32_t kActorInWorld = 1u << 3;   // owns triangles of the collision set
// The Stadium environment's enclosing shell: two hemispheres of 480 triangles,
// +-28 km wide, 7.6 km or more from the map centre, unreachable by the car.
// Left out of the package by default (PackHeader::enclosureTris counts the
// triangles left out); WorldCompileInputs::includeWorldEnclosure keeps it.
constexpr std::uint32_t kActorWorldEnclosure = 1u << 4;

struct TriggerVolume {          // 48 B, parallel to Triggers
    std::uint32_t actor;
    std::uint32_t firstTri;     // the actor's triangles (contiguous)
    std::uint32_t triCount;
    std::uint32_t firstInstance;
    std::uint32_t instanceCount;
    std::uint32_t reserved;
    float lo[3], hi[3];         // world AABB (empty box when triCount == 0)
};
static_assert(sizeof(TriggerVolume) == 48, "TriggerVolume layout");

// Material ids with a special effect when a wheel touches them (board 20 §4,
// board 30 R6): one entry per (actor, material). The force side reads the
// material from the wheel contact; this table is an index for predictors,
// tools and scope checks.
constexpr std::uint8_t kMatTurbo = 7u, kMatTurbo2 = 26u, kMatFreeWheeling = 29u, kMatTurboRoulette = 30u;
constexpr std::uint8_t kMatWater = 13u, kMatGolfBall = 23u;   // ignored by the car (Tri meta kMetaCarIgnores)
struct SpecialSurface {         // 48 B
    std::uint32_t actor;
    std::uint32_t triCount;
    std::uint8_t material;
    std::uint8_t reserved[3];
    float axisZ[3];             // = Actors[actor].peerZ (booster forward axis)
    float lo[3], hi[3];         // world AABB of these triangles
};
static_assert(sizeof(SpecialSurface) == 48, "SpecialSurface layout");

// FV water zone (CSceneVehicleWaterZone, board 20 §4 / board 30 R6.6), pointer
// free; cells in Section::WaterCells (dims[0] * dims[1], row-major in y, then
// kWaterCellPadding zero bytes for the batch kernel's byte gather).
// Same fields and semantics as tw::model::WaterZone (model/car_force_consts.h).
constexpr std::uint32_t kWaterCellPadding = 4u;
struct WaterGrid {              // 48 B
    std::uint32_t known;        // 1: compiled from FV's static configuration; 0: not supplied (enabled = 0)
    std::uint32_t enabled;
    float cellSize[2];
    float origin[2];
    std::uint32_t dims[2];
    float surfaceHeight;
    float secondaryCullHeight;
    std::uint8_t outsideValue;
    std::uint8_t reserved[7];
};
static_assert(sizeof(WaterGrid) == 48, "WaterGrid layout");

constexpr std::uint32_t kRaceShortcutMode = 1u << 0;
constexpr std::uint32_t kPackEnclosureExcluded = 1u << 0;   // PackHeader::flags: the world enclosure was left out
constexpr std::uint32_t kPackMovingPairsExcluded = 1u << 1;  // PackHeader::flags: moving-pair corpora left out (opt-in)

struct PackHeader {             // 1024 B
    char magic[8];
    std::uint32_t version;
    std::uint32_t headerBytes;  // sizeof(PackHeader)
    std::uint64_t totalBytes;   // blob size
    std::uint64_t contentHash;  // PackHash over the blob with this field zeroed
    std::uint64_t fvSceneHash;  // FV CudaHostScene::deterministicHash (provenance)
    std::uint32_t sectionCount; // kSectionCount
    std::uint32_t flags;        // kPack*
    // race (TwRaceMap scalars)
    std::uint32_t requiredCps;
    std::uint32_t requiredLaps;
    std::uint32_t raceFlags;    // kRaceShortcutMode
    std::uint32_t excludedMovingPairTris;  // moving-pair ("Flags") triangles left out (kPackMovingPairsExcluded)
    // world summary
    float boundsLo[3], boundsHi[3];  // world AABB of all triangles
    std::uint32_t actorCount, surfaceCount;       // FV scene totals
    std::uint32_t triggerTris, solidTris;
    std::uint32_t skippedSurfaces;                // FV surfaces outside the collision set (other groups, non-static, no append)
    std::uint32_t generalInstances;               // instances whose rotation is not a signed permutation
    std::uint32_t inexactPermInstances;           // signed permutations whose perm(v) + t differs from FV in some bit
    std::uint32_t nonTriggerGroup1Actors;         // group-1 actors that are not race triggers (trigger id kNoTrigger)
    std::uint32_t movingPairInstances;            // instances flagged kInstMovingPair
    std::uint32_t enclosureTris;                  // world-enclosure triangles left out (kPackEnclosureExcluded)
    WaterGrid water;
    std::uint8_t reserved1[512];
    SectionEntry sections[kSectionCount];
};
static_assert(kSectionCount == 14, "a new section bumps kPackVersion and the header layout");
static_assert(sizeof(PackHeader) == 1024, "PackHeader layout");

// Signed-permutation code: column j of R (basisX/Y/Z) is sign_j * e_{p_j};
// code = permIndex(p_0, p_1, p_2) * 8 + (sign_0 < 0) + 2 (sign_1 < 0) + 4 (sign_2 < 0),
// permIndex over (012, 021, 102, 120, 201, 210). Returns kRotGeneral otherwise.
std::uint8_t RotationCode(const float iso[12]);
// Rows of R for a code (R[i][j], column j = basis j); false for kRotGeneral.
bool RotationFromCode(std::uint8_t code, float R[3][3]);

// ---- the package --------------------------------------------------------------------
// Owns the blob (64-byte aligned). Immutable once built; share it by
// shared_ptr<const WorldPack>.
class WorldPack {
public:
    WorldPack() = default;
    WorldPack(WorldPack &&) noexcept = default;
    WorldPack &operator=(WorldPack &&) noexcept = default;
    const std::uint8_t *data() const { return bytes_.get(); }
    std::size_t size() const { return size_; }
    const PackHeader &Header() const { return *reinterpret_cast<const PackHeader *>(bytes_.get()); }
    std::uint64_t Hash() const { return size_ ? Header().contentHash : 0u; }

    // Allocates a zeroed blob (used by the compiler and by Load).
    bool Allocate(std::size_t bytes);
    std::uint8_t *MutableData() { return bytes_.get(); }

private:
    struct Free {
        void operator()(std::uint8_t *p) const;
    };
    std::unique_ptr<std::uint8_t[], Free> bytes_;
    std::size_t size_ = 0u;
};

// 64-bit content hash (word-wise multiply-xorshift, endian-fixed, FV-free).
std::uint64_t PackHash(const std::uint8_t *data, std::size_t bytes);
// Recomputes and stores the header hash (compiler side).
void SealWorldPack(WorldPack *pack);

bool SaveWorldPack(const char *path, const WorldPack &pack, std::string *error);
bool LoadWorldPack(const char *path, WorldPack *pack, std::string *error);  // validates (OpenWorldPack)

// ---- typed read view ----------------------------------------------------------------
template <class T>
struct PackSpan {
    const T *ptr = nullptr;
    std::size_t count = 0u;
    const T &operator[](std::size_t i) const { return ptr[i]; }
    const T *begin() const { return ptr; }
    const T *end() const { return ptr + count; }
    std::size_t size() const { return count; }
    bool empty() const { return count == 0u; }
};

struct WorldPackView {
    const PackHeader *header = nullptr;
    PackSpan<Tri> tris;
    PackSpan<TriPlane> triPlanes;
    PackSpan<TriInfo> triInfo;
    PackSpan<Instance> instances;
    PackSpan<Model> models;
    PackSpan<ModelTri> modelTris;
    PackSpan<Vec3f> modelVerts;
    PackSpan<std::uint32_t> modelInstances;
    PackSpan<Actor> actors;
    PackSpan<race::TwTrigger> triggers;
    PackSpan<TriggerVolume> triggerVolumes;
    PackSpan<race::TwSpawn> spawns;
    PackSpan<SpecialSurface> specials;
    PackSpan<std::uint8_t> waterCells;

    // TwRaceMap pointing into the pack (valid while the pack lives).
    race::TwRaceMap RaceMap() const;
    // Actor of a triangle (Tri::meta).
    const Actor &ActorOf(const Tri &t) const { return actors[t.meta >> kMetaActorShift]; }
};

// Checks magic, version, header and section sizes, alignment, bounds, the
// content hash (unless skipHash) and every cross-reference; fills `view`.
bool OpenWorldPack(const WorldPack &pack, WorldPackView *view, std::string *error, bool skipHash = false);

// B's bench form (tw_world.h World): the same triangles and meta, TriExtra
// with B's double-precision unit normal and the FV surface index. With
// `source`, also B's mesh-local SourceScene (one surface per instance, as
// B's prototype CompileTwinWorld writes it; architect C's PatchWorld input).
void MakeBenchWorld(const WorldPackView &view, World *out, SourceScene *source = nullptr);

}  // namespace tw::world

#endif
