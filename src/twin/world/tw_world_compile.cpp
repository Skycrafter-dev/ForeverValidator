// FV scene -> twin world. Uses FV's own flat export (BuildCudaHostScene, called
// in place from forevervalidator_core; no CUDA needed).
#include "tw_world_compile.h"

#include <cmath>

#include "engine/scene/static_scene_model.h"

namespace tw {

bool CompileTwinWorld(const forevervalidator::simulation::CudaHostScene &scene, World *out, std::string *diag,
                      SourceScene *source) {
    using forevervalidator::simulation::CudaSceneSurface;
    World w;
    w.actorCount = static_cast<std::uint32_t>(scene.actors.size());
    w.surfaceCount = static_cast<std::uint32_t>(scene.surfaces.size());
    std::uint32_t nonMesh = 0, noAppend = 0, otherGroup = 0;
    for (std::uint32_t si = 0; si < scene.surfaces.size(); ++si) {
        const CudaSceneSurface &s = scene.surfaces[si];
        if (s.type != static_cast<std::uint32_t>(GmSurf::EGmSurfType::Mesh)) { ++nonMesh; continue; }
        if (!s.allowsStaticCollisionRecordAppend) { ++noAppend; continue; }
        if (s.actorIndex >= scene.actors.size()) { if (diag) *diag = "surface actor out of range"; return false; }
        const auto &actor = scene.actors[s.actorIndex];
        const bool trigger = actor.hasCheckpoint ||
                             actor.purpose == static_cast<std::uint32_t>(StaticScenePurpose::CheckpointTrigger);
        if (!trigger && actor.itemProperties.collisionGroup != CHmsItem::ECollisionGroup_Static) { ++otherGroup; continue; }
        if (source) {
            SourceSurface ss{};
            const GmVec3 *c[4] = {&s.localToWorld.rotation.basisX, &s.localToWorld.rotation.basisY, &s.localToWorld.rotation.basisZ,
                                  &s.localToWorld.translation};
            for (int k = 0; k < 4; ++k) { ss.iso[3 * k] = c[k]->x; ss.iso[3 * k + 1] = c[k]->y; ss.iso[3 * k + 2] = c[k]->z; }
            ss.firstVertex = static_cast<std::uint32_t>(source->vertices.size() / 3);
            ss.vertexCount = s.vertexCount;
            ss.firstTri = static_cast<std::uint32_t>(source->tris.size());
            ss.triCount = s.triangleCount;
            ss.actor = s.actorIndex; ss.trigger = trigger ? 1u : 0u;
            for (std::uint32_t vi = 0; vi < s.vertexCount; ++vi) {
                const GmVec3 &v = scene.vertices[s.firstVertex + vi];
                source->vertices.push_back(v.x); source->vertices.push_back(v.y); source->vertices.push_back(v.z);
            }
            for (std::uint32_t ti = 0; ti < s.triangleCount; ++ti) {
                const auto &src = scene.triangles[s.firstTriangle + ti];
                const std::uint16_t local = src.material.Index();
                SourceTri st{{src.vertexIndices[0], src.vertexIndices[1], src.vertexIndices[2]},
                             local < s.materialCount ? scene.materials[s.firstMaterial + local] : 0u};
                source->tris.push_back(st);
            }
            source->surfaces.push_back(ss);
        }
        for (std::uint32_t ti = 0; ti < s.triangleCount; ++ti) {
            const auto &src = scene.triangles[s.firstTriangle + ti];
            Tri t{};
            for (int k = 0; k < 3; ++k) {
                const std::uint32_t vi = src.vertexIndices[k];
                if (vi >= s.vertexCount) { if (diag) *diag = "vertex index out of range"; return false; }
                GmVec3 wv;
                wv.SetMult(scene.vertices[s.firstVertex + vi], s.localToWorld);
                t.v[k][0] = wv.x; t.v[k][1] = wv.y; t.v[k][2] = wv.z;
            }
            for (int a = 0; a < 3; ++a) {
                t.lo[a] = std::fmin(t.v[0][a], std::fmin(t.v[1][a], t.v[2][a]));
                t.hi[a] = std::fmax(t.v[0][a], std::fmax(t.v[1][a], t.v[2][a]));
                w.bounds.lo[a] = std::fmin(w.bounds.lo[a], t.lo[a]);
                w.bounds.hi[a] = std::fmax(w.bounds.hi[a], t.hi[a]);
            }
            const std::uint16_t local = src.material.Index();
            const std::uint32_t mat = local < s.materialCount ? scene.materials[s.firstMaterial + local] : 0u;
            t.meta = (mat & kMetaMaterialMask) | (trigger ? kMetaTrigger : 0u) |
                     ((mat == 13u || mat == 23u) ? kMetaCarIgnores : 0u) | (s.actorIndex << kMetaActorShift);
            TriExtra x{};
            const double e1[3] = {double(t.v[1][0]) - t.v[0][0], double(t.v[1][1]) - t.v[0][1], double(t.v[1][2]) - t.v[0][2]};
            const double e2[3] = {double(t.v[2][0]) - t.v[0][0], double(t.v[2][1]) - t.v[0][1], double(t.v[2][2]) - t.v[0][2]};
            const double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
            const double l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (l > 1e-12) for (int a = 0; a < 3; ++a) x.n[a] = static_cast<float>(n[a] / l);
            x.surface = si;
            w.tris.push_back(t);
            w.extra.push_back(x);
            if (trigger) ++w.triggerTris;
        }
    }
    w.skippedSurfaces = nonMesh + noAppend + otherGroup;
    if (diag) *diag = "skipped surfaces: non-mesh " + std::to_string(nonMesh) + ", no-append " + std::to_string(noAppend) +
                      ", other group " + std::to_string(otherGroup);
    *out = std::move(w);
    return true;
}

}  // namespace tw
