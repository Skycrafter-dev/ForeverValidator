// tw_record: record FV substep streams for the M2 collision bake-off.
// Usage: tw_record PACKS REPLAY OUT.twrec [maxTicks]
//
// Runs the replay through FV's Reference backend and captures, per collision
// detection (one per substep): the car pose and velocity, the 8 car
// ellipsoid world poses, and every raw narrow-phase contact FV produces
// (GmCollision_Ellipsoid_Mesh output, i.e. before the per-shape merge). The
// static world is captured from FV's own flat export (BuildCudaHostScene) when
// the session installs its scene. Every contact (raw and merged) carries its
// static actor (FV installation order = world-pack actor index; TWREC003).
// Hooks use GNU ld --wrap on four FV symbols, so FV itself is unmodified
// (needs a non-IPO FV build).
#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/native.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/collision/hms_collision.h"
#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_dyna.h"
#include "engine/physics/geometry/gm_surface.h"
#include "engine/physics/geometry/plug_surface.h"
#include "engine/rendering/plug_tree.h"
#include "engine/scene/replay_static_scene_corpuses.h"
#include "engine/scene/static_scene_model.h"
#include "simulation/backends/cuda/cuda_scene_layout.h"
#include "simulation/runtime/replay_simulation_session.h"
#include "tw_world_compile.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/runtime/replay_simulation_runtime.h"
#include "compile_car_shapes.h"
#include "vec3.h"

// Private member access for the record -> corpus map (explicit instantiation
// ignores access checks; FV stays unmodified).
namespace rob {
template <typename Tag, typename Tag::type M>
struct Robber {
    friend typename Tag::type Get(Tag) { return M; }
};
using Zone = CHmsCollisionManager::SZone;
using Group = CHmsCollisionManager::SGroup;
struct ZoneGroups {
    using type = std::array<Group, CHmsCollisionManager_GroupCount> Zone::*;
    friend type Get(ZoneGroups);
};
struct GroupTrees {
    using type = GmOctree<CHmsCollisionManager::SColOctreeCell> Group::*;
    friend type Get(GroupTrees);
};
template struct Robber<ZoneGroups, &Zone::groups>;
template struct Robber<GroupTrees, &Group::staticTrees>;
}  // namespace rob

namespace {
// Static actor of a contact = FV installation order (= CudaHostScene actor index = world-pack actor).
std::unordered_map<const CHmsCorpus *, std::uint32_t> g_corpusOrder;   // from the corpus collection
struct SurfaceOwner {
    const CHmsCorpus *corpus;
    const CPlugSurface *surface;
};
std::unordered_map<const GmIso4 *, SurfaceOwner> g_recordCorpus;  // static record location -> owner (raw contacts)
std::unordered_map<const GmSurf *, SurfaceOwner> g_movingSurf;   // moving-pair corpus geometry -> owner (raw contacts)
const void *g_recordZone = nullptr;
std::size_t g_unknownActor = 0;
std::uint32_t ActorOf(const CHmsCorpus *c) {
    const auto it = g_corpusOrder.find(c);
    if (it == g_corpusOrder.end()) { ++g_unknownActor; return tw::kNoActor; }
    return it->second;
}
void MapTree(const CPlugTree *t, const CHmsCorpus *corpus) {
    if (const CPlugSurface *s = t->Surface())
        if (s->Geometry()) g_movingSurf[s->Geometry()] = {corpus, s};
    for (unsigned long i = 0; i < t->GetChildCount(); ++i)
        if (const CPlugTree *c = t->GetChild(i)) MapTree(c, corpus);
}
void MapRecords(const rob::Zone *zone, const CHmsCorpus *car) {
    g_recordCorpus.clear();
    g_movingSurf.clear();
    g_recordZone = zone;
    for (const rob::Group &g : zone->*Get(rob::ZoneGroups{})) {
        const auto &trees = g.*Get(rob::GroupTrees{});
        for (std::uint32_t i = 0; i < trees.GetCount(); ++i)
            if (trees[i].ContainsSurface())
                g_recordCorpus[&trees[i].SurfaceData().location] = {trees[i].SurfaceData().corpus, trees[i].SurfaceData().surface};
        // non-static corpora: FV tests them through the moving-pair path (stack isos, so keyed by geometry)
        for (std::uint32_t i = 0; i < g.NonStaticCorpusCount(); ++i) {
            const CHmsCorpus *c = g.NonStaticCorpusAt(i);
            if (c != nullptr && c != car && c->CollisionTree() != nullptr) MapTree(c->CollisionTree(), c);
        }
    }
}
tw::Recording g_rec;
bool g_haveWorld = false;
bool g_recording = false;
bool g_inDetect = false;
std::uint32_t g_tick = 0;
std::size_t g_shapeChecks = 0, g_shapeMismatch = 0, g_unknownShape = 0, g_detectCalls = 0;
float g_maxShapeDev = 0.0f;
std::string g_compileDiag;

void ToShape(const GmIso4 &iso, const GmVec3 &radii, tw::Shape *s) {
    s->t[0] = iso.translation.x; s->t[1] = iso.translation.y; s->t[2] = iso.translation.z;
    const GmVec3 e[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int j = 0; j < 3; ++j) {
        GmVec3 a; a.SetMult(e[j], iso.rotation);
        s->axis[j][0] = a.x; s->axis[j][1] = a.y; s->axis[j][2] = a.z;
    }
    s->r[0] = radii.x; s->r[1] = radii.y; s->r[2] = radii.z;
}
}  // namespace

extern "C" {
bool __real__ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection(ReplaySimulationSession *, StaticSceneModelCollection *);
bool __wrap__ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection(ReplaySimulationSession *self,
                                                                                           StaticSceneModelCollection *models) {
    forevervalidator::simulation::CudaHostScene scene;
    // The CUDA device budget (FV's default limits) is too small for the largest scenery maps.
    forevervalidator::simulation::CudaSceneBuildLimits limits;
    limits.maximumActors = 1u << 22;
    limits.maximumVertices = limits.maximumTriangles = 1u << 28;
    limits.maximumOctreeCells = 1u << 29;
    limits.maximumSurfaces = 1u << 24;
    limits.maximumMaterials = limits.maximumAccelerationCells = 1u << 26;
    const auto r = forevervalidator::simulation::BuildCudaHostScene(*models, &scene, limits);
    if (r == forevervalidator::simulation::CudaSceneBuildResult::Success) {
        g_rec.source = tw::SourceScene{};
        g_haveWorld = tw::CompileTwinWorld(scene, &g_rec.world, &g_compileDiag, &g_rec.source);
    } else {
        g_compileDiag = std::string("BuildCudaHostScene failed: ") + forevervalidator::simulation::CudaSceneBuildResultName(r);
    }
    return __real__ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection(self, models);
}

std::uint32_t __real__ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver(
        ReplayStaticCorpusCollection *, ReplayCheckpointContactObserver &);
std::uint32_t __wrap__ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver(
        ReplayStaticCorpusCollection *self, ReplayCheckpointContactObserver &observer) {
    const std::uint32_t n =
            __real__ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver(self, observer);
    g_corpusOrder.clear();
    for (std::uint32_t i = 0; i < self->Count(); ++i)
        if (const ReplayStaticCorpus *c = self->CorpusAt(i)) g_corpusOrder[&c->Corpus()] = c->InstallationOrder();
    return n;
}

// M3 pose path check: the twin's CarShapes (from FV's static configuration) and R(quat) against FV's
// live car tree at every detection (body shapes: shape-to-car pose bits; wheels: rest rotation and x/z).
tw::collide::CarShapes g_carShapes;
bool g_haveCarShapes = false;
std::size_t g_poseChecks = 0, g_quatMismatch = 0, g_bodyMismatch = 0, g_wheelMismatch = 0;
ReplaySimulationRunResult
__real__ZN23ReplaySimulationRuntime5StartERK26ReplaySimulationDefinitionR14ReplayMapSceneRK6GmIso4RK17ReplayControlTickj(
        ReplaySimulationRuntime *, const ReplaySimulationDefinition &, ReplayMapScene &, const GmIso4 &, const ReplayControlTick &, unsigned);
ReplaySimulationRunResult
__wrap__ZN23ReplaySimulationRuntime5StartERK26ReplaySimulationDefinitionR14ReplayMapSceneRK6GmIso4RK17ReplayControlTickj(
        ReplaySimulationRuntime *self, const ReplaySimulationDefinition &definition, ReplayMapScene &scene, const GmIso4 &spawn,
        const ReplayControlTick &first, unsigned seed) {
    forevervalidator::simulation::CudaHostStaticConfiguration cfg;
    std::string diag;
    g_haveCarShapes = forevervalidator::simulation::BuildCudaHostStaticConfiguration(definition, &cfg) ==
                              forevervalidator::simulation::CudaStaticConfigurationBuildResult::Success &&
                      tw::fv::CompileCarShapes(cfg, &g_carShapes, &diag);
    if (!g_haveCarShapes) std::fprintf(stderr, "car shapes: %s\n", diag.c_str());
    return __real__ZN23ReplaySimulationRuntime5StartERK26ReplaySimulationDefinitionR14ReplayMapSceneRK6GmIso4RK17ReplayControlTickj(
            self, definition, scene, spawn, first, seed);
}
bool SameIsoBits(const GmIso4 &a, const tw::collide::Iso &b, bool translationXZOnly) {
    const GmVec3 *cols[3] = {&a.rotation.basisX, &a.rotation.basisY, &a.rotation.basisZ};
    for (int c = 0; c < 3; ++c)
        if (std::memcmp(&cols[c]->x, &b.r[3 * c], 4) || std::memcmp(&cols[c]->y, &b.r[3 * c + 1], 4) || std::memcmp(&cols[c]->z, &b.r[3 * c + 2], 4)) return false;
    if (std::memcmp(&a.translation.x, &b.t[0], 4) || std::memcmp(&a.translation.z, &b.t[2], 4)) return false;
    return translationXZOnly || std::memcmp(&a.translation.y, &b.t[1], 4) == 0;
}

void __real__ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus(CHmsCollisionManagerSZone *,
                                                                                                         CHmsCollisionBuffer &, CHmsCorpus *);
void __wrap__ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus(CHmsCollisionManagerSZone *self,
                                                                                                         CHmsCollisionBuffer &buffer,
                                                                                                         CHmsCorpus *corpus) {
    CHmsDyna *dyna = corpus != nullptr ? corpus->Dynamics() : nullptr;
    CPlugTree *tree = corpus != nullptr ? corpus->CollisionTree() : nullptr;
    if (!g_recording || dyna == nullptr || tree == nullptr || tree->GetChildCount() < tw::kShapes) {
        __real__ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus(self, buffer, corpus);
        return;
    }
    ++g_detectCalls;
    if (g_recordZone != self) MapRecords(self, corpus);
    tw::Substep s{};
    s.tick = g_tick;
    const auto &st = dyna->CurrentState();
    GmIso4 car;
    car.rotation = st.rotation;
    car.translation = st.position;
    s.carPos[0] = st.position.x; s.carPos[1] = st.position.y; s.carPos[2] = st.position.z;
    const GmVec3 *cols[3] = {&st.rotation.basisX, &st.rotation.basisY, &st.rotation.basisZ};
    for (int j = 0; j < 3; ++j) { s.carRot[3 * j] = cols[j]->x; s.carRot[3 * j + 1] = cols[j]->y; s.carRot[3 * j + 2] = cols[j]->z; }
    s.vel[0] = st.linearSpeed.x; s.vel[1] = st.linearSpeed.y; s.vel[2] = st.linearSpeed.z;
    s.angVel[0] = st.angularSpeed.x; s.angVel[1] = st.angularSpeed.y; s.angVel[2] = st.angularSpeed.z;
    if (g_haveCarShapes) {
        ++g_poseChecks;
        const float q[4] = {st.rotationQuat.x, st.rotationQuat.y, st.rotationQuat.z, st.rotationQuat.w};
        const tw::model::M3 R = tw::model::rot_from_quat(q);
        const float rq[9] = {R.c0.x, R.c0.y, R.c0.z, R.c1.x, R.c1.y, R.c1.z, R.c2.x, R.c2.y, R.c2.z};
        if (std::memcmp(rq, s.carRot, sizeof rq) != 0) ++g_quatMismatch;
    }
    for (int k = 0; k < tw::kShapes; ++k) {
        const CPlugTree *child = tree->GetChild(static_cast<unsigned long>(k));
        GmIso4 c2r; c2r.SetIdentity();
        child->GetThisToRootTransfo(c2r, 1, nullptr);
        if (g_haveCarShapes) {
            const tw::collide::CarShape &cs = g_carShapes.shape[k];
            if (cs.wheel < 0) g_bodyMismatch += !SameIsoBits(c2r, cs.bodyPose, false);
            else g_wheelMismatch += !SameIsoBits(c2r, g_carShapes.wheelRest[cs.wheel], true);
        }
        GmIso4 world;
        world.translation.SetMult(c2r.translation, car);
        const GmVec3 e[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        GmVec3 radii{1, 1, 1};
        const CPlugSurface *surf = child->Surface();
        if (surf && surf->Geometry())
            if (const auto *el = dynamic_cast<const GmSurfEllipsoid *>(surf->Geometry())) radii = el->radii;
        tw::Shape &sh = s.shapes[k];
        sh.t[0] = world.translation.x; sh.t[1] = world.translation.y; sh.t[2] = world.translation.z;
        for (int j = 0; j < 3; ++j) {
            GmVec3 a; a.SetMult(e[j], c2r.rotation);
            GmVec3 b; b.SetMult(a, car.rotation);
            sh.axis[j][0] = b.x; sh.axis[j][1] = b.y; sh.axis[j][2] = b.z;
        }
        sh.r[0] = std::fabs(radii.x); sh.r[1] = std::fabs(radii.y); sh.r[2] = std::fabs(radii.z);
    }
    s.firstContact = static_cast<std::uint32_t>(g_rec.contacts.size());
    g_rec.substeps.push_back(s);
    g_inDetect = true;
    __real__ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus(self, buffer, corpus);
    g_inDetect = false;
    g_rec.substeps.back().contactCount = static_cast<std::uint32_t>(g_rec.contacts.size()) - g_rec.substeps.back().firstContact;
    // FV's merged per-shape contacts (the buffer after MergeQueuedSphereContacts)
    g_rec.substeps.back().firstMerged = static_cast<std::uint32_t>(g_rec.merged.size());
    for (std::uint32_t i = 0; i < buffer.PhysicalCollisionCount(); ++i) {
        const SHmsPhysicalCollision *c = buffer.PhysicalCollisionAtOrNull(i);
        if (!c) continue;
        int shape = 255;
        for (int k = 0; k < tw::kShapes; ++k) if (tree->GetChild(static_cast<unsigned long>(k)) == c->TreeA()) shape = k;
        tw::FvContact fc{};
        fc.shape = static_cast<std::uint8_t>(shape);
        fc.primary = c->sphereMergePrimary ? 1 : 0;
        fc.material = static_cast<std::uint8_t>(c->materialB);
        fc.P[0] = c->contactPoint.x; fc.P[1] = c->contactPoint.y; fc.P[2] = c->contactPoint.z;
        fc.N[0] = c->impulseNormal.x; fc.N[1] = c->impulseNormal.y; fc.N[2] = c->impulseNormal.z;
        fc.Sep[0] = c->separation.x; fc.Sep[1] = c->separation.y; fc.Sep[2] = c->separation.z;
        fc.K[0] = c->extraNegated.x; fc.K[1] = c->extraNegated.y; fc.K[2] = c->extraNegated.z;
        fc.actor = ActorOf(c->CorpusB());
        g_rec.merged.push_back(fc);
    }
    g_rec.substeps.back().mergedCount = static_cast<std::uint32_t>(g_rec.merged.size()) - g_rec.substeps.back().firstMerged;
}

int __real__Z26GmCollision_Ellipsoid_MeshRK13LocatedGmSurfS1_R18CGmCollisionBuffer(const LocatedGmSurf &, const LocatedGmSurf &,
                                                                                    CGmCollisionBuffer &);
int __wrap__Z26GmCollision_Ellipsoid_MeshRK13LocatedGmSurfS1_R18CGmCollisionBuffer(const LocatedGmSurf &ellipsoid,
                                                                                    const LocatedGmSurf &mesh,
                                                                                    CGmCollisionBuffer &buffer) {
    if (!g_inDetect || g_rec.substeps.empty()) {
        return __real__Z26GmCollision_Ellipsoid_MeshRK13LocatedGmSurfS1_R18CGmCollisionBuffer(ellipsoid, mesh, buffer);
    }
    tw::Substep &s = g_rec.substeps.back();
    // which shape: nearest recorded centre; cross-check the pose
    tw::Shape obs;
    GmIso4 iso; iso.SetIdentity();
    if (ellipsoid.enabled && ellipsoid.iso) iso = *ellipsoid.iso;
    ToShape(iso, ellipsoid.EllipsoidRadii(), &obs);
    int best = -1; float bestD = 1e30f;
    for (int k = 0; k < tw::kShapes; ++k) {
        float d = 0;
        for (int a = 0; a < 3; ++a) d += (s.shapes[k].t[a] - obs.t[a]) * (s.shapes[k].t[a] - obs.t[a]);
        for (int a = 0; a < 3; ++a) d += (s.shapes[k].r[a] - std::fabs(obs.r[a])) * (s.shapes[k].r[a] - std::fabs(obs.r[a]));
        if (d < bestD) { bestD = d; best = k; }
    }
    if (best < 0 || bestD > 1e-4f) { ++g_unknownShape; }
    else {
        float dev = std::sqrt(bestD);
        for (int j = 0; j < 3; ++j)
            for (int a = 0; a < 3; ++a) dev = std::fmax(dev, std::fabs(s.shapes[best].axis[j][a] - obs.axis[j][a]));
        ++g_shapeChecks;
        if (dev > g_maxShapeDev) g_maxShapeDev = dev;
        if (dev > 1e-5f) ++g_shapeMismatch;
        s.flags |= 1u;
    }
    const unsigned long n0 = buffer.GetCurrentCount();
    const int hit = __real__Z26GmCollision_Ellipsoid_MeshRK13LocatedGmSurfS1_R18CGmCollisionBuffer(ellipsoid, mesh, buffer);
    const unsigned long n1 = buffer.GetCurrentCount();
    for (unsigned long k = n0; k < n1; ++k) {
        const GmCollision &c = buffer.GetCollision(k);
        tw::FvContact fc{};
        fc.shape = static_cast<std::uint8_t>(best < 0 ? 255 : best);
        fc.primary = c.sphereMergePrimary ? 1 : 0;
        fc.material = static_cast<std::uint8_t>(c.materialB);
        fc.P[0] = c.contactPoint.x; fc.P[1] = c.contactPoint.y; fc.P[2] = c.contactPoint.z;
        fc.N[0] = c.impulseNormal.x; fc.N[1] = c.impulseNormal.y; fc.N[2] = c.impulseNormal.z;
        fc.Sep[0] = c.separation.x; fc.Sep[1] = c.separation.y; fc.Sep[2] = c.separation.z;
        fc.K[0] = c.extraNegated.x; fc.K[1] = c.extraNegated.y; fc.K[2] = c.extraNegated.z;
        // static actor and material id (CPlugSurface::ComputeCollision maps the local index only after this call)
        const SurfaceOwner *owner = nullptr;
        const auto rc = g_recordCorpus.find(mesh.iso);
        if (rc != g_recordCorpus.end()) owner = &rc->second;
        else {
            const auto ms = g_movingSurf.find(mesh.surf);
            if (ms != g_movingSurf.end()) owner = &ms->second;
        }
        if (owner != nullptr) {
            fc.actor = ActorOf(owner->corpus);
            fc.material = static_cast<std::uint8_t>(owner->surface->SurfaceMaterialIdFromLocalIndex(c.localMaterialB));
        } else {
            fc.actor = tw::kNoActor;
            ++g_unknownActor;
        }
        g_rec.contacts.push_back(fc);
    }
    return hit;
}
}  // extern "C"

using namespace forevervalidator;
using namespace forevervalidator::experimental;

int main(int argc, char **argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: tw_record PACKS REPLAY OUT [maxTicks]\n"); return 2; }
    const std::uint32_t maxTicks = argc > 4 ? static_cast<std::uint32_t>(std::atoi(argv[4])) : 1u << 30;
    auto source = OpenInstalledPackDirectory(argv[1]);
    auto bytes = ReadNativeReplayFile(argv[2], ReplayIdentity{argv[2]});
    if (!source || !bytes) { std::fprintf(stderr, "open failed\n"); return 1; }
    AssetBytes replay = std::move(bytes).Value();
    PhysicsSandboxOptions options;
    options.backend = SimulationBackend::Reference;
    auto sb = CreatePhysicsSandbox(std::move(source).Value(), options);
    if (!sb) { std::fprintf(stderr, "sandbox failed\n"); return 1; }
    PhysicsSandbox sandbox = std::move(sb).Value();
    auto loaded = sandbox.LoadReplay({replay.data(), replay.size()}, ReplayIdentity{argv[2]});
    if (!loaded) { std::fprintf(stderr, "load failed: %s\n", loaded.Error().diagnostic.c_str()); return 1; }
    if (!g_haveWorld) { std::fprintf(stderr, "world capture failed: %s\n", g_compileDiag.c_str()); return 1; }
    auto scene = sandbox.ReadScene();
    std::string name = argv[2];
    const std::size_t slash = name.find_last_of('/');
    if (slash != std::string::npos) name = name.substr(slash + 1);
    g_rec.name = name.substr(0, name.find('.'));
    g_recording = true;
    std::uint32_t ticks = 0;
    while (ticks < maxTicks) {
        g_tick = ticks;
        if (!sandbox.AdvanceTicks(1)) break;
        ++ticks;
    }
    g_recording = false;
    g_rec.ticks = ticks;
    if (!tw::SaveRecording(argv[3], g_rec)) { std::fprintf(stderr, "save failed\n"); return 1; }
    std::printf("%s: ticks=%u substeps=%zu (%.2f/tick) raw contacts=%zu merged=%zu tris=%zu (ReadScene %zu) triggerTris=%u %s | "
                "shape pose checks %zu, mismatches %zu (max dev %.2e), unmatched ellipsoid calls %zu, contacts without actor %zu\n",
                g_rec.name.c_str(), ticks, g_rec.substeps.size(), ticks ? double(g_rec.substeps.size()) / ticks : 0.0,
                g_rec.contacts.size(), g_rec.merged.size(), g_rec.world.tris.size(), scene ? scene.Value().collisionTriangles.size() : 0u,
                g_rec.world.triggerTris, g_compileDiag.c_str(), g_shapeChecks, g_shapeMismatch, g_maxShapeDev, g_unknownShape, g_unknownActor);
    std::printf("M3 pose path: %zu detections, car shapes %s; R(quat) != FV rotation: %zu; body shape-to-car != CarShapes: %zu; "
                "wheel rotation/x/z != rest pose: %zu\n",
                g_poseChecks, g_haveCarShapes ? "compiled" : "MISSING", g_quatMismatch, g_bodyMismatch, g_wheelMismatch);
    return 0;
}
