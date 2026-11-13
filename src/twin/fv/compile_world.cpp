// Twin FV adapter (M2): CudaHostScene -> tw::world::WorldPack.
// docs/twin/m2-world.md describes the package and the acceptance check.
#include "compile_world.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "compile_car.h"
#include "engine/physics/dynamics/hms_item.h"
#include "engine/scene/static_scene_model.h"
#include "simulation/backends/cuda/cuda_scene_layout.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"

namespace tw::fv {

namespace w = tw::world;
using forevervalidator::simulation::CudaHostScene;
using forevervalidator::simulation::CudaSceneActor;
using forevervalidator::simulation::CudaSceneSurface;

namespace {

bool Refuse(std::string *diag, const std::string &why) {
    if (diag) *diag = "twin world: " + why;
    return false;
}

std::uint32_t Bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

void PutIso(float out[12], const GmIso4 &iso) {
    const GmVec3 *c[4] = {&iso.rotation.basisX, &iso.rotation.basisY, &iso.rotation.basisZ, &iso.translation};
    for (int k = 0; k < 4; ++k) {
        out[3 * k] = c[k]->x;
        out[3 * k + 1] = c[k]->y;
        out[3 * k + 2] = c[k]->z;
    }
}

// FNV-1a over raw bytes (model identity; compared exactly afterwards).
struct Fnv {
    std::uint64_t h = 1469598103934665603ull;
    void Add(const void *p, std::size_t n) {
        const unsigned char *b = static_cast<const unsigned char *>(p);
        for (std::size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= 1099511628211ull;
        }
    }
};

// Mesh-local surface data in the package's model form.
struct LocalModel {
    std::vector<w::Vec3f> verts;
    std::vector<w::ModelTri> tris;
    std::uint64_t hash = 0u;
    bool operator==(const LocalModel &o) const {
        return verts.size() == o.verts.size() && tris.size() == o.tris.size() &&
               std::memcmp(verts.data(), o.verts.data(), verts.size() * sizeof(w::Vec3f)) == 0 &&
               std::memcmp(tris.data(), o.tris.data(), tris.size() * sizeof(w::ModelTri)) == 0;
    }
};

bool BuildLocalModel(const CudaHostScene &scene, const CudaSceneSurface &s, std::uint32_t surfaceIndex, LocalModel *m,
                     std::string *diag) {
    m->verts.resize(s.vertexCount);
    for (std::uint32_t i = 0; i < s.vertexCount; ++i) {
        const GmVec3 &v = scene.vertices[s.firstVertex + i];
        m->verts[i] = {v.x, v.y, v.z};
    }
    // octree leaf rank of every triangle (FV detection order inside the mesh)
    std::vector<std::uint32_t> rank(s.triangleCount, UINT32_MAX);
    std::uint32_t r = 0u;
    for (std::uint32_t c = 0; c < s.octreeCellCount; ++c) {
        const auto &cell = scene.octreeCells[s.firstOctreeCell + c];
        if (!cell.containsTriangle) continue;
        if (cell.triangleIndex >= s.triangleCount || rank[cell.triangleIndex] != UINT32_MAX)
            return Refuse(diag, "surface " + std::to_string(surfaceIndex) + ": mesh octree does not hold each triangle once");
        rank[cell.triangleIndex] = r++;
    }
    if (r != s.triangleCount)
        return Refuse(diag, "surface " + std::to_string(surfaceIndex) + ": mesh octree misses triangles");
    if (s.triangleCount >= (1u << (32 - w::kModelTriOctreeShift)))
        return Refuse(diag, "surface " + std::to_string(surfaceIndex) + ": too many triangles");
    m->tris.resize(s.triangleCount);
    for (std::uint32_t k = 0; k < s.triangleCount; ++k) {
        const auto &src = scene.triangles[s.firstTriangle + k];
        w::ModelTri t{};
        for (int c = 0; c < 3; ++c) {
            if (src.vertexIndices[c] >= s.vertexCount)
                return Refuse(diag, "surface " + std::to_string(surfaceIndex) + ": vertex index out of range");
            t.v[c] = src.vertexIndices[c];
        }
        t.n[0] = src.normal.x;
        t.n[1] = src.normal.y;
        t.n[2] = src.normal.z;
        t.d = src.planeDistance;
        const std::uint16_t local = src.material.Index();
        const std::uint32_t mat = local < s.materialCount ? scene.materials[s.firstMaterial + local] : 0u;
        if (mat > w::kModelTriMaterialMask)
            return Refuse(diag, "surface " + std::to_string(surfaceIndex) + ": material id " + std::to_string(mat));
        t.meta = mat | (rank[k] << w::kModelTriOctreeShift);
        m->tris[k] = t;
    }
    Fnv f;
    const std::uint32_t counts[2] = {s.vertexCount, s.triangleCount};
    f.Add(counts, sizeof counts);
    f.Add(m->verts.data(), m->verts.size() * sizeof(w::Vec3f));
    f.Add(m->tris.data(), m->tris.size() * sizeof(w::ModelTri));
    m->hash = f.h;
    return true;
}

void ExpandBox(float lo[3], float hi[3], const float p[3]) {
    for (int a = 0; a < 3; ++a) {
        lo[a] = std::fmin(lo[a], p[a]);
        hi[a] = std::fmax(hi[a], p[a]);
    }
}
void EmptyBox(float lo[3], float hi[3]) {
    for (int a = 0; a < 3; ++a) {
        lo[a] = 1e30f;
        hi[a] = -1e30f;
    }
}

std::uint8_t YawQuarter(const float iso[12]) {
    // basisX = (cos q90, 0, -sin q90), basisY = (0, 1, 0), basisZ = (sin q90, 0, cos q90)
    static const float X[4][3] = {{1, 0, 0}, {0, 0, -1}, {-1, 0, 0}, {0, 0, 1}};
    static const float Z[4][3] = {{0, 0, 1}, {1, 0, 0}, {0, 0, -1}, {-1, 0, 0}};
    if (!(iso[3] == 0.0f && iso[4] == 1.0f && iso[5] == 0.0f)) return w::kYawNone;
    for (std::uint8_t q = 0; q < 4; ++q) {
        bool same = true;
        for (int a = 0; a < 3; ++a) same = same && iso[a] == X[q][a] && iso[6 + a] == Z[q][a];
        if (same) return q;
    }
    return w::kYawNone;
}

std::int16_t CellOf(float c, float size) {
    const float f = std::floor(c / size);
    return static_cast<std::int16_t>(std::max(-32768.0f, std::min(32767.0f, f)));
}

struct SpawnTable {
    std::vector<race::TwSpawn> spawns;
    std::uint16_t For(const GmIso4 &iso) {
        std::uint32_t words[12];
        float f[12];
        PutIso(f, iso);
        std::memcpy(words, f, sizeof words);
        for (std::size_t i = 0; i < spawns.size(); ++i)
            if (std::memcmp(spawns[i].iso, words, sizeof words) == 0) return static_cast<std::uint16_t>(i);
        race::TwSpawn s{};
        GmQuat q;
        q.Set(iso.rotation);  // FV matrix -> quaternion (as SetLocation does on respawn)
        s.pos[0] = iso.translation.x;
        s.pos[1] = iso.translation.y;
        s.pos[2] = iso.translation.z;
        s.quat[0] = q.x;
        s.quat[1] = q.y;
        s.quat[2] = q.z;
        s.quat[3] = q.w;
        std::memcpy(s.iso, words, sizeof words);
        spawns.push_back(s);
        return static_cast<std::uint16_t>(spawns.size() - 1u);
    }
};

template <class T>
void PutSection(w::WorldPack *pack, w::Section s, const std::vector<T> &v) {
    const w::SectionEntry &e = pack->Header().sections[static_cast<std::uint32_t>(s)];
    if (!v.empty()) std::memcpy(pack->MutableData() + e.offset, v.data(), v.size() * sizeof(T));
}

}  // namespace

bool CompileWorldPack(const WorldCompileInputs &in, w::WorldPack *out, std::string *diag) {
    if (in.scene == nullptr || in.startSpawn == nullptr || out == nullptr) return Refuse(diag, "missing input");
    const CudaHostScene &scene = *in.scene;
    if (scene.schemaVersion != CudaHostScene::SchemaVersion) return Refuse(diag, "FV scene schema version");
    const std::uint32_t nActors = static_cast<std::uint32_t>(scene.actors.size());
    const std::uint32_t nSurfaces = static_cast<std::uint32_t>(scene.surfaces.size());
    if (nActors >= (1u << (32 - kMetaActorShift))) return Refuse(diag, "too many actors");

    // ---- actors, race triggers ------------------------------------------------------
    std::vector<w::Actor> actors(nActors);
    std::vector<race::TwTrigger> triggers;
    std::vector<std::uint32_t> triggerActor;
    SpawnTable spawns;
    spawns.For(*in.startSpawn);  // spawn 0: race start spawn, unseeded
    std::uint32_t slots = 0u, maxSlot = 0u, nonTriggerGroup1 = 0u;
    for (std::uint32_t a = 0; a < nActors; ++a) {
        const CudaSceneActor &src = scene.actors[a];
        w::Actor &dst = actors[a];
        dst.peerZ[0] = src.worldPose.rotation.basisZ.x;
        dst.peerZ[1] = src.worldPose.rotation.basisZ.y;
        dst.peerZ[2] = src.worldPose.rotation.basisZ.z;
        dst.peerId = src.installationOrder + 1u;
        dst.group = static_cast<std::uint8_t>(src.itemProperties.collisionGroup);
        dst.purpose = static_cast<std::uint8_t>(src.purpose);
        dst.trigger = w::kNoTrigger;
        dst.flags = (src.itemProperties.collisionStatic ? w::kActorCollisionStatic : 0u) |
                    (src.hasCheckpoint ? w::kActorHasCheckpoint : 0u) |
                    (src.hasCheckpointSpawn ? w::kActorHasSpawn : 0u);
        if (src.installationOrder != a) return Refuse(diag, "actor installation order is not the actor index");
        const bool isTrigger = src.purpose == static_cast<std::uint32_t>(StaticScenePurpose::CheckpointTrigger);
        if (!isTrigger) {
            if (src.hasCheckpoint) return Refuse(diag, "checkpoint identity on a non-trigger actor " + std::to_string(a));
            if (src.itemProperties.collisionGroup == 1u) ++nonTriggerGroup1;
            continue;
        }
        if (src.itemProperties.collisionGroup != 1u || !src.itemProperties.collisionStatic)
            return Refuse(diag, "race trigger actor " + std::to_string(a) + " outside static group 1");
        race::TwTrigger t{};
        t.role = src.hasCheckpoint ? static_cast<std::uint8_t>(src.checkpointRole) : static_cast<std::uint8_t>(race::kRoleNone);
        t.currentTransform = src.respawnUsesCurrentTransform ? 1u : 0u;
        t.slot = src.checkpointSlot == UINT32_MAX ? race::kNoSlot : static_cast<std::uint16_t>(src.checkpointSlot);
        t.spawn = src.hasCheckpointSpawn ? spawns.For(src.checkpointSpawn) : race::kNoSpawn;
        if (src.checkpointSlot != UINT32_MAX) {
            if (src.checkpointSlot >= 0xFFFFu) return Refuse(diag, "checkpoint slot overflow");
            ++slots;
            maxSlot = std::max(maxSlot, src.checkpointSlot + 1u);
        }
        if (triggers.size() + 1u >= w::kNoTrigger) return Refuse(diag, "too many race triggers");
        dst.trigger = static_cast<std::uint16_t>(triggers.size());
        triggers.push_back(t);
        triggerActor.push_back(a);
    }
    if (slots != maxSlot) return Refuse(diag, "checkpoint slots are not 0..n-1");
    if (spawns.spawns.size() >= 0xFFFFu) return Refuse(diag, "too many spawns");

    // ---- collision set, models, instances, world triangles ----------------------------
    std::vector<w::Instance> instances;
    std::vector<w::Model> models;
    std::vector<w::ModelTri> modelTris;
    std::vector<w::Vec3f> modelVerts;
    std::vector<std::vector<std::uint32_t>> modelRefs;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> modelsByHash;
    std::vector<Tri> tris;
    std::vector<w::TriPlane> planes;
    std::vector<w::TriInfo> info;
    std::vector<std::uint32_t> instanceOfSurface(nSurfaces, UINT32_MAX);
    std::uint32_t skipped = 0u, general = 0u, inexact = 0u, triggerTris = 0u, movingInstances = 0u;
    float boundsLo[3], boundsHi[3];
    EmptyBox(boundsLo, boundsHi);
    {
        std::size_t triTotal = 0u;
        for (const CudaSceneSurface &s : scene.surfaces) triTotal += s.triangleCount;
        tris.reserve(triTotal);
        planes.reserve(triTotal);
        info.reserve(triTotal);
        instances.reserve(nSurfaces);
    }
    // (a) FV's static records of the car's target groups (CUDA BuildAcceleration + cuda_collision TargetGroups);
    // (b) non-static corpora of a target group: FV's Reference detection tests them through the
    //     moving-pair path (DetectCollisionBetween, pair always enabled against the car), before the
    //     group's static records. FV's CUDA scene leaves them out (collisionStatic is required there).
    // 0: not in the set, 1: static record, 2: moving pair
    auto membership = [&](const CudaSceneSurface &s) -> int {
        const CudaSceneActor &actor = scene.actors[s.actorIndex];
        const std::uint32_t group = static_cast<std::uint32_t>(actor.itemProperties.collisionGroup);
        const bool targetGroup = group == 1u || group == 3u || group == 4u;
        if (actor.itemProperties.collisionStatic && s.allowsStaticCollisionRecordAppend && targetGroup) return 1;
        if (!actor.itemProperties.collisionStatic && targetGroup) return 2;
        return 0;
    };
    for (const CudaSceneSurface &s : scene.surfaces)
        if (s.actorIndex >= nActors) return Refuse(diag, "surface actor out of range");

    // ---- world enclosure: an Environment actor (static, group 4) with a > 8 km surface whose every
    // triangle stays > 1 km from the box of the rest of the collision set, and whose box contains it.
    // Left out unless in.includeWorldEnclosure (the car cannot reach it; user decision 2026-11-08).
    std::vector<std::uint8_t> enclosureActor(nActors, 0u);
    std::uint32_t enclosureTris = 0u, excludedMovingTris = 0u;
    {
        std::vector<std::uint8_t> candidate(nActors, 0u);
        for (const CudaSceneSurface &s : scene.surfaces) {
            const CudaSceneActor &a = scene.actors[s.actorIndex];
            const GmVec3 &h = s.worldBounds.halfExtents;
            if (membership(s) == 1 && a.purpose == static_cast<std::uint32_t>(StaticScenePurpose::Environment) &&
                a.itemProperties.collisionGroup == 4u && (h.x > 4096.0f || h.z > 4096.0f))
                candidate[s.actorIndex] = 1u;
        }
        float restLo[3], restHi[3], candLo[3], candHi[3];
        EmptyBox(restLo, restHi);
        EmptyBox(candLo, candHi);
        for (const CudaSceneSurface &s : scene.surfaces) {
            if (membership(s) == 0) continue;
            const GmBoxAligned &b = s.worldBounds;
            const float lo[3] = {b.center.x - b.halfExtents.x, b.center.y - b.halfExtents.y, b.center.z - b.halfExtents.z};
            const float hi[3] = {b.center.x + b.halfExtents.x, b.center.y + b.halfExtents.y, b.center.z + b.halfExtents.z};
            ExpandBox(candidate[s.actorIndex] ? candLo : restLo, candidate[s.actorIndex] ? candHi : restHi, lo);
            ExpandBox(candidate[s.actorIndex] ? candLo : restLo, candidate[s.actorIndex] ? candHi : restHi, hi);
        }
        bool encloses = true;
        for (int a = 0; a < 3; ++a) encloses = encloses && candLo[a] <= restLo[a] && candHi[a] >= restHi[a];
        std::vector<std::uint8_t> far(nActors, 1u);
        for (const CudaSceneSurface &s : scene.surfaces) {
            if (!candidate[s.actorIndex] || membership(s) == 0) continue;
            for (std::uint32_t k = 0; k < s.triangleCount && far[s.actorIndex]; ++k) {
                const auto &src = scene.triangles[s.firstTriangle + k];
                float lo[3], hi[3];
                EmptyBox(lo, hi);
                for (int c = 0; c < 3; ++c) {
                    if (src.vertexIndices[c] >= s.vertexCount) return Refuse(diag, "vertex index out of range");
                    GmVec3 wv;
                    wv.SetMult(scene.vertices[s.firstVertex + src.vertexIndices[c]], s.localToWorld);
                    const float p[3] = {wv.x, wv.y, wv.z};
                    ExpandBox(lo, hi, p);
                }
                bool apart = false;
                for (int a = 0; a < 3; ++a) apart = apart || lo[a] > restHi[a] + 1000.0f || hi[a] < restLo[a] - 1000.0f;
                if (!apart) far[s.actorIndex] = 0u;
            }
        }
        for (std::uint32_t a = 0; a < nActors; ++a) {
            if (!candidate[a] || !far[a] || !encloses) continue;
            enclosureActor[a] = 1u;
            actors[a].flags |= w::kActorWorldEnclosure;
        }
    }
    std::vector<std::uint8_t> excludedSurface(nSurfaces, 0u);

    LocalModel local;
    for (std::uint32_t si = 0; si < nSurfaces; ++si) {
        const CudaSceneSurface &s = scene.surfaces[si];
        const CudaSceneActor &actor = scene.actors[s.actorIndex];
        const std::uint32_t group = static_cast<std::uint32_t>(actor.itemProperties.collisionGroup);
        const int kind = membership(s);
        const bool movingPair = kind == 2;
        if (kind == 0) {
            ++skipped;
            continue;
        }
        if (enclosureActor[s.actorIndex] && !in.includeWorldEnclosure) {
            excludedSurface[si] = 1u;
            enclosureTris += s.triangleCount;
            continue;
        }
        if (movingPair && in.excludeMovingPairCorpora) {
            excludedMovingTris += s.triangleCount;
            continue;
        }
        if (group == 3u) return Refuse(diag, "collision surface in group 3 (actor " + std::to_string(s.actorIndex) + ")");
        if (movingPair && group != 4u)
            return Refuse(diag, "non-static corpus in trigger group 1 (actor " + std::to_string(s.actorIndex) + ")");
        if (s.type != static_cast<std::uint32_t>(GmSurf::EGmSurfType::Mesh))
            return Refuse(diag, "non-mesh collision surface " + std::to_string(si) + " (type " + std::to_string(s.type) + ")");
        if (s.firstVertex + std::uint64_t(s.vertexCount) > scene.vertices.size() ||
            s.firstTriangle + std::uint64_t(s.triangleCount) > scene.triangles.size() ||
            s.firstOctreeCell + std::uint64_t(s.octreeCellCount) > scene.octreeCells.size() ||
            s.firstMaterial + std::uint64_t(s.materialCount) > scene.materials.size())
            return Refuse(diag, "surface " + std::to_string(si) + " ranges");
        if (!BuildLocalModel(scene, s, si, &local, diag)) return false;

        // model dedup
        std::uint32_t model = UINT32_MAX;
        auto &bucket = modelsByHash[local.hash];
        for (std::uint32_t mi : bucket) {
            const w::Model &m = models[mi];
            if (m.vertCount == local.verts.size() && m.triCount == local.tris.size() &&
                std::memcmp(&modelVerts[m.firstVert], local.verts.data(), local.verts.size() * sizeof(w::Vec3f)) == 0 &&
                std::memcmp(&modelTris[m.firstTri], local.tris.data(), local.tris.size() * sizeof(w::ModelTri)) == 0) {
                model = mi;
                break;
            }
        }
        if (model == UINT32_MAX) {
            w::Model m{};
            m.hash = local.hash;
            m.firstTri = static_cast<std::uint32_t>(modelTris.size());
            m.triCount = static_cast<std::uint32_t>(local.tris.size());
            m.firstVert = static_cast<std::uint32_t>(modelVerts.size());
            m.vertCount = static_cast<std::uint32_t>(local.verts.size());
            EmptyBox(m.lo, m.hi);
            for (const w::ModelTri &t : local.tris) {
                for (int c = 0; c < 3; ++c) ExpandBox(m.lo, m.hi, &local.verts[t.v[c]].x);
                const std::uint32_t mat = t.meta & w::kModelTriMaterialMask;
                if (mat < 32u) m.materialMask |= 1u << mat;
            }
            modelTris.insert(modelTris.end(), local.tris.begin(), local.tris.end());
            modelVerts.insert(modelVerts.end(), local.verts.begin(), local.verts.end());
            model = static_cast<std::uint32_t>(models.size());
            models.push_back(m);
            modelRefs.emplace_back();
            bucket.push_back(model);
        }
        const bool trigger = group == 1u;
        models[model].flags |= trigger ? w::kModelTrigger : w::kModelSolid;

        w::Instance inst{};
        PutIso(inst.iso, s.localToWorld);
        inst.model = model;
        inst.actor = s.actorIndex;
        inst.firstTri = static_cast<std::uint32_t>(tris.size());
        inst.triCount = s.triangleCount;
        inst.fvSurface = si;
        inst.fvRecord = UINT32_MAX;
        inst.rotation = w::RotationCode(inst.iso);
        inst.yaw = YawQuarter(inst.iso);
        inst.group = static_cast<std::uint8_t>(group);
        const GmMat3 &R = s.localToWorld.rotation;
        const float det = R.basisX.x * (R.basisY.y * R.basisZ.z - R.basisZ.y * R.basisY.z) -
                          R.basisY.x * (R.basisX.y * R.basisZ.z - R.basisZ.y * R.basisX.z) +
                          R.basisZ.x * (R.basisX.y * R.basisY.z - R.basisY.y * R.basisX.z);
        inst.flags = (trigger ? w::kInstTrigger : 0u) | (det < 0.0f ? w::kInstMirrored : 0u) | (movingPair ? w::kInstMovingPair : 0u);
        if (movingPair) ++movingInstances;
        if (inst.rotation == w::kRotGeneral) ++general;
        float Rp[3][3];
        const bool perm = w::RotationFromCode(inst.rotation, Rp);
        bool exact = perm;
        EmptyBox(inst.lo, inst.hi);
        const std::uint32_t instIndex = static_cast<std::uint32_t>(instances.size());
        const std::uint16_t triggerId = actors[s.actorIndex].trigger;
        for (std::uint32_t k = 0; k < s.triangleCount; ++k) {
            const auto &src = scene.triangles[s.firstTriangle + k];
            const w::ModelTri &mt = local.tris[k];
            Tri t{};
            for (int c = 0; c < 3; ++c) {
                const GmVec3 &lv = scene.vertices[s.firstVertex + src.vertexIndices[c]];
                GmVec3 wv;
                wv.SetMult(lv, s.localToWorld);  // FV's own transform (board 20 §2.2, B: bit-exact model -> world)
                t.v[c][0] = wv.x;
                t.v[c][1] = wv.y;
                t.v[c][2] = wv.z;
                if (perm) {
                    const float l[3] = {lv.x, lv.y, lv.z};
                    for (int i = 0; i < 3 && exact; ++i) {
                        float p = 0.0f;
                        for (int j = 0; j < 3; ++j)
                            if (Rp[i][j] != 0.0f) p = Rp[i][j] * l[j] + inst.iso[9 + i];
                        exact = Bits(p) == Bits(t.v[c][i]);
                    }
                }
            }
            for (int a = 0; a < 3; ++a) {
                t.lo[a] = std::fmin(t.v[0][a], std::fmin(t.v[1][a], t.v[2][a]));
                t.hi[a] = std::fmax(t.v[0][a], std::fmax(t.v[1][a], t.v[2][a]));
                boundsLo[a] = std::fmin(boundsLo[a], t.lo[a]);
                boundsHi[a] = std::fmax(boundsHi[a], t.hi[a]);
                inst.lo[a] = std::fmin(inst.lo[a], t.lo[a]);
                inst.hi[a] = std::fmax(inst.hi[a], t.hi[a]);
            }
            const std::uint32_t mat = mt.meta & w::kModelTriMaterialMask;
            t.meta = mat | (trigger ? kMetaTrigger : 0u) | ((mat == w::kMatWater || mat == w::kMatGolfBall) ? kMetaCarIgnores : 0u) |
                     (s.actorIndex << kMetaActorShift);
            // FV stored plane, rotated to world
            w::TriPlane pl{};
            GmVec3 n;
            n.SetMult(src.normal, s.localToWorld.rotation);
            pl.n[0] = n.x;
            pl.n[1] = n.y;
            pl.n[2] = n.z;
            pl.d = (-n.x * t.v[0][0] - n.y * t.v[0][1]) - n.z * t.v[0][2];
            w::TriInfo x{};
            x.instance = instIndex;
            x.modelTri = k;
            x.fvOrder = UINT32_MAX;
            x.trigger = trigger ? triggerId : w::kNoTrigger;
            x.group = static_cast<std::uint8_t>(group);
            const double e1[3] = {double(t.v[1][0]) - t.v[0][0], double(t.v[1][1]) - t.v[0][1], double(t.v[1][2]) - t.v[0][2]};
            const double e2[3] = {double(t.v[2][0]) - t.v[0][0], double(t.v[2][1]) - t.v[0][1], double(t.v[2][2]) - t.v[0][2]};
            const double cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2], cz = e1[0] * e2[1] - e1[1] * e2[0];
            x.flags = (cx == 0.0 && cy == 0.0 && cz == 0.0) ? w::kTriDegenerate : 0u;
            tris.push_back(t);
            planes.push_back(pl);
            info.push_back(x);
            if (trigger) ++triggerTris;
        }
        if (exact) inst.flags |= w::kInstPermExact;
        else if (perm) ++inexact;
        float c[3];
        for (int a = 0; a < 3; ++a) c[a] = s.triangleCount ? 0.5f * (inst.lo[a] + inst.hi[a]) : inst.iso[9 + a];
        inst.cell[0] = CellOf(c[0], 32.0f);
        inst.cell[1] = CellOf(c[1], 8.0f);
        inst.cell[2] = CellOf(c[2], 32.0f);
        w::Actor &act = actors[s.actorIndex];
        if (act.instanceCount == 0u) act.firstInstance = instIndex;
        else if (act.firstInstance + act.instanceCount != instIndex) return Refuse(diag, "actor surfaces are not contiguous");
        ++act.instanceCount;
        act.flags |= w::kActorInWorld;
        modelRefs[model].push_back(instIndex);
        instanceOfSurface[si] = instIndex;
        instances.push_back(inst);
    }
    if (tris.size() >= (std::size_t(1) << 31)) return Refuse(diag, "too many triangles");

    // ---- FV detection order: groups 1, 3, 4; per group the moving-pair corpora (actor order, their
    // surfaces in FV's flattening order), then the static record tree preorder; mesh octree leaves ----
    std::uint32_t next = 0u;
    auto orderInstance = [&](const w::Instance &inst) {
        const w::Model &m = models[inst.model];
        std::vector<std::uint32_t> byRank(m.triCount);  // triangles in octree leaf order
        for (std::uint32_t k = 0; k < m.triCount; ++k) byRank[modelTris[m.firstTri + k].meta >> w::kModelTriOctreeShift] = k;
        for (std::uint32_t r = 0; r < m.triCount; ++r) info[inst.firstTri + byRank[r]].fvOrder = next++;
    };
    std::vector<std::uint8_t> ordered(instances.size(), 0u);
    for (const std::uint32_t g : {1u, 3u, 4u}) {
        for (std::size_t i = 0; i < instances.size(); ++i)
            if ((instances[i].flags & w::kInstMovingPair) && instances[i].group == g) {
                instances[i].fvRecord = w::kNoRecord;
                ordered[i] = 1u;
                orderInstance(instances[i]);
            }
        const auto &range = scene.accelerationGroups[g - 1u];
        if (range.firstCell + std::uint64_t(range.cellCount) > scene.accelerationCells.size())
            return Refuse(diag, "acceleration range");
        std::uint32_t leaf = 0u;
        for (std::uint32_t c = 0; c < range.cellCount; ++c) {
            const auto &cell = scene.accelerationCells[range.firstCell + c];
            if (cell.surfaceIndex == UINT32_MAX) continue;
            if (cell.surfaceIndex < nSurfaces && excludedSurface[cell.surfaceIndex]) {  // world enclosure left out
                ++leaf;                                                                     // fvRecord keeps FV's rank
                continue;
            }
            if (cell.surfaceIndex >= nSurfaces || instanceOfSurface[cell.surfaceIndex] == UINT32_MAX)
                return Refuse(diag, "acceleration leaf outside the collision set");
            const std::uint32_t ii = instanceOfSurface[cell.surfaceIndex];
            w::Instance &inst = instances[ii];
            if (ordered[ii] || inst.group != g) return Refuse(diag, "acceleration leaf repeated or in the wrong group");
            ordered[ii] = 1u;
            inst.fvRecord = leaf++;
            orderInstance(inst);
        }
    }
    for (std::size_t i = 0; i < instances.size(); ++i)
        if (!ordered[i]) return Refuse(diag, "collision surface missing from FV's record tree");
    if (next != tris.size()) return Refuse(diag, "FV order does not cover every triangle");

    // ---- model -> instance refs ---------------------------------------------------------
    std::vector<std::uint32_t> modelInstances;
    for (std::size_t m = 0; m < models.size(); ++m) {
        models[m].firstRef = static_cast<std::uint32_t>(modelInstances.size());
        models[m].instanceCount = static_cast<std::uint32_t>(modelRefs[m].size());
        modelInstances.insert(modelInstances.end(), modelRefs[m].begin(), modelRefs[m].end());
    }

    // ---- trigger volumes ------------------------------------------------------------------
    std::vector<w::TriggerVolume> volumes(triggers.size());
    for (std::size_t t = 0; t < triggers.size(); ++t) {
        const w::Actor &a = actors[triggerActor[t]];
        w::TriggerVolume &v = volumes[t];
        v.actor = triggerActor[t];
        v.firstInstance = a.firstInstance;
        v.instanceCount = a.instanceCount;
        v.firstTri = a.instanceCount ? instances[a.firstInstance].firstTri : 0u;
        EmptyBox(v.lo, v.hi);
        for (std::uint32_t i = 0; i < a.instanceCount; ++i) {
            const w::Instance &in = instances[a.firstInstance + i];
            v.triCount += in.triCount;
            for (int k = 0; k < 3; ++k) {
                v.lo[k] = std::fmin(v.lo[k], in.lo[k]);
                v.hi[k] = std::fmax(v.hi[k], in.hi[k]);
            }
        }
    }

    // ---- special surfaces (boosters, turbo, free wheel) ----------------------------------------
    std::vector<w::SpecialSurface> specials;
    {
        std::unordered_map<std::uint64_t, std::uint32_t> at;  // (actor, material) -> entry
        for (std::size_t i = 0; i < tris.size(); ++i) {
            const std::uint32_t mat = tris[i].meta & kMetaMaterialMask;
            if (mat != w::kMatTurbo && mat != w::kMatTurbo2 && mat != w::kMatFreeWheeling && mat != w::kMatTurboRoulette) continue;
            if (tris[i].meta & kMetaTrigger) continue;
            const std::uint32_t a = tris[i].meta >> kMetaActorShift;
            const std::uint64_t key = (std::uint64_t(a) << 8) | mat;
            auto it = at.find(key);
            if (it == at.end()) {
                w::SpecialSurface s{};
                s.actor = a;
                s.material = static_cast<std::uint8_t>(mat);
                for (int k = 0; k < 3; ++k) s.axisZ[k] = actors[a].peerZ[k];
                EmptyBox(s.lo, s.hi);
                it = at.emplace(key, static_cast<std::uint32_t>(specials.size())).first;
                specials.push_back(s);
            }
            w::SpecialSurface &s = specials[it->second];
            ++s.triCount;
            for (int k = 0; k < 3; ++k) {
                s.lo[k] = std::fmin(s.lo[k], tris[i].lo[k]);
                s.hi[k] = std::fmax(s.hi[k], tris[i].hi[k]);
            }
        }
    }

    // ---- water --------------------------------------------------------------------------------
    w::WaterGrid water{};
    std::vector<std::uint8_t> waterCells;
    if (in.config != nullptr) {
        tw::model::WaterZone zone;
        std::string why;
        if (!CompileWaterZone(*in.config, &zone, &waterCells, &why)) return Refuse(diag, "water: " + why);
        water.known = 1u;
        water.enabled = zone.enabled;
        water.cellSize[0] = zone.cellSize[0];
        water.cellSize[1] = zone.cellSize[1];
        water.origin[0] = zone.origin[0];
        water.origin[1] = zone.origin[1];
        water.dims[0] = zone.dims[0];
        water.dims[1] = zone.dims[1];
        water.surfaceHeight = zone.surfaceHeight;
        water.secondaryCullHeight = zone.secondaryCullHeight;
        water.outsideValue = zone.outsideValue;
        if (!zone.enabled) waterCells.clear();
        else waterCells.resize(std::size_t(zone.dims[0]) * zone.dims[1] + w::kWaterCellPadding, 0u);
    }

    // ---- layout and seal ------------------------------------------------------------------------
    const std::size_t counts[w::kSectionCount] = {
        tris.size(),     planes.size(),  info.size(),        instances.size(), models.size(),
        modelTris.size(), modelVerts.size(), modelInstances.size(), actors.size(), triggers.size(),
        volumes.size(),  spawns.spawns.size(), specials.size(), waterCells.size()};
    const std::uint32_t elem[w::kSectionCount] = {
        sizeof(Tri),      sizeof(w::TriPlane), sizeof(w::TriInfo), sizeof(w::Instance), sizeof(w::Model),
        sizeof(w::ModelTri), sizeof(w::Vec3f), sizeof(std::uint32_t), sizeof(w::Actor), sizeof(race::TwTrigger),
        sizeof(w::TriggerVolume), sizeof(race::TwSpawn), sizeof(w::SpecialSurface), sizeof(std::uint8_t)};
    w::PackHeader h{};
    std::memcpy(h.magic, w::kPackMagic, 8);
    h.version = w::kPackVersion;
    h.headerBytes = sizeof(w::PackHeader);
    h.sectionCount = w::kSectionCount;
    std::uint64_t offset = sizeof(w::PackHeader);
    for (std::uint32_t s = 0; s < w::kSectionCount; ++s) {
        offset = (offset + w::kPackAlign - 1u) / w::kPackAlign * w::kPackAlign;
        h.sections[s].elemBytes = elem[s];
        h.sections[s].offset = offset;
        h.sections[s].count = counts[s];
        offset += std::uint64_t(counts[s]) * elem[s];
    }
    offset = (offset + w::kPackAlign - 1u) / w::kPackAlign * w::kPackAlign;
    h.totalBytes = offset;
    h.fvSceneHash = scene.deterministicHash;
    h.requiredCps = slots;
    h.requiredLaps = in.requiredLaps;
    h.raceFlags = in.shortcutMode ? w::kRaceShortcutMode : 0u;
    for (int a = 0; a < 3; ++a) {
        h.boundsLo[a] = boundsLo[a];
        h.boundsHi[a] = boundsHi[a];
    }
    h.actorCount = nActors;
    h.surfaceCount = nSurfaces;
    h.triggerTris = triggerTris;
    h.solidTris = static_cast<std::uint32_t>(tris.size()) - triggerTris;
    h.skippedSurfaces = skipped;
    h.generalInstances = general;
    h.inexactPermInstances = inexact;
    h.nonTriggerGroup1Actors = nonTriggerGroup1;
    h.movingPairInstances = movingInstances;
    h.enclosureTris = enclosureTris;
    if (enclosureTris != 0u) h.flags |= w::kPackEnclosureExcluded;
    h.excludedMovingPairTris = excludedMovingTris;
    if (in.excludeMovingPairCorpora) h.flags |= w::kPackMovingPairsExcluded;
    h.water = water;

    w::WorldPack pack;
    if (!pack.Allocate(static_cast<std::size_t>(h.totalBytes))) return Refuse(diag, "allocation failed");
    std::memcpy(pack.MutableData(), &h, sizeof h);
    PutSection(&pack, w::Section::Tris, tris);
    PutSection(&pack, w::Section::TriPlanes, planes);
    PutSection(&pack, w::Section::TriInfo, info);
    PutSection(&pack, w::Section::Instances, instances);
    PutSection(&pack, w::Section::Models, models);
    PutSection(&pack, w::Section::ModelTris, modelTris);
    PutSection(&pack, w::Section::ModelVerts, modelVerts);
    PutSection(&pack, w::Section::ModelInstances, modelInstances);
    PutSection(&pack, w::Section::Actors, actors);
    PutSection(&pack, w::Section::Triggers, triggers);
    PutSection(&pack, w::Section::TriggerVolumes, volumes);
    PutSection(&pack, w::Section::Spawns, spawns.spawns);
    PutSection(&pack, w::Section::Specials, specials);
    PutSection(&pack, w::Section::WaterCells, waterCells);
    w::SealWorldPack(&pack);
    w::WorldPackView view;
    std::string why;
    if (!w::OpenWorldPack(pack, &view, &why, /*skipHash=*/true)) return Refuse(diag, why);
    *out = std::move(pack);
    return true;
}

forevervalidator::simulation::CudaSceneBuildLimits TwinSceneBuildLimits() {
    forevervalidator::simulation::CudaSceneBuildLimits l;
    l.maximumActors = (1u << (32 - kMetaActorShift)) - 1u;  // Tri meta actor field (22 bits)
    l.maximumSurfaces = 1u << 24;
    l.maximumMaterials = 1u << 26;
    l.maximumVertices = 1u << 28;
    l.maximumTriangles = 1u << 28;
    l.maximumOctreeCells = 1u << 29;
    l.maximumAccelerationCells = 1u << 26;
    l.maximumCheckpoints = 0xFFFEu;   // the race map refuses > 63 with its own reason
    return l;
}

bool CompileWorldPackFromModels(const StaticSceneModelCollection &models, WorldCompileInputs in, w::WorldPack *out,
                                std::string *diag) {
    CudaHostScene scene;
    const auto r = forevervalidator::simulation::BuildCudaHostScene(models, &scene, TwinSceneBuildLimits());
    if (r != forevervalidator::simulation::CudaSceneBuildResult::Success)
        return Refuse(diag, std::string("BuildCudaHostScene failed: ") + forevervalidator::simulation::CudaSceneBuildResultName(r));
    in.scene = &scene;
    return CompileWorldPack(in, out, diag);
}

}  // namespace tw::fv
