#include <algorithm>
#include <cmath>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/physics_tolerances.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_inverse_mesh_queries.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_mesh_triangle_sidecar.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_triangle_mesh_query.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_native_binary32_collision.h"
#include "simulation/runtime/replay_deterministic_execution.h"

namespace {

bool failNextAllocation = false;

void *Allocate(std::size_t size) {
    if (failNextAllocation) {
        failNextAllocation = false;
        throw std::bad_alloc();
    }
    if (void *allocation = std::malloc(size)) {
        return allocation;
    }
    throw std::bad_alloc();
}

}  // namespace

void *operator new(std::size_t size) {
    return Allocate(size);
}

void *operator new[](std::size_t size) {
    return Allocate(size);
}

void operator delete(void *allocation) noexcept {
    std::free(allocation);
}

void operator delete[](void *allocation) noexcept {
    std::free(allocation);
}

void operator delete(void *allocation, std::size_t) noexcept {
    std::free(allocation);
}

void operator delete[](void *allocation, std::size_t) noexcept {
    std::free(allocation);
}

namespace {

class VectorCollisionBuffer final : public CGmCollisionBuffer {
public:
    GmCollision &GetCollision(unsigned long index) override {
        return collisions_[index];
    }

    GmCollision &AddCollision(void) override {
        collisions_.emplace_back();
        return collisions_.back();
    }

    unsigned long GetCurrentCount(void) override {
        return static_cast<unsigned long>(collisions_.size());
    }

    const std::vector<GmCollision> &Collisions(void) const {
        return collisions_;
    }

private:
    std::vector<GmCollision> collisions_;
};

struct XorShift32 {
    std::uint32_t state = 0x7f4a7c15u;

    std::uint32_t Next(void) {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return state;
    }
};

bool SameBits(float lhs, float rhs) {
    std::uint32_t lhsBits = 0u;
    std::uint32_t rhsBits = 0u;
    std::memcpy(&lhsBits, &lhs, sizeof(lhsBits));
    std::memcpy(&rhsBits, &rhs, sizeof(rhsBits));
    return lhsBits == rhsBits;
}

bool SameBits(const GmVec3 &lhs, const GmVec3 &rhs) {
    return SameBits(lhs.x, rhs.x) &&
           SameBits(lhs.y, rhs.y) &&
           SameBits(lhs.z, rhs.z);
}

bool SameBits(const GmBoxAligned &lhs, const GmBoxAligned &rhs) {
    return SameBits(lhs.center, rhs.center) &&
           SameBits(lhs.halfExtents, rhs.halfExtents);
}

bool SameCollision(const GmCollision &lhs, const GmCollision &rhs) {
    return SameBits(lhs.separation, rhs.separation) &&
           SameBits(lhs.impulseNormal, rhs.impulseNormal) &&
           SameBits(lhs.contactPoint, rhs.contactPoint) &&
           lhs.localMaterialA.Index() == rhs.localMaterialA.Index() &&
           lhs.localMaterialB.Index() == rhs.localMaterialB.Index() &&
           lhs.materialA == rhs.materialA &&
           lhs.materialB == rhs.materialB &&
           lhs.sphereMergePrimary == rhs.sphereMergePrimary &&
           SameBits(lhs.extraNegated, rhs.extraNegated);
}

bool SameBuffers(const VectorCollisionBuffer &lhs,
                 const VectorCollisionBuffer &rhs) {
    const auto &lhsCollisions = lhs.Collisions();
    const auto &rhsCollisions = rhs.Collisions();
    if (lhsCollisions.size() != rhsCollisions.size()) {
        return false;
    }
    for (std::size_t index = 0u; index < lhsCollisions.size(); ++index) {
        if (!SameCollision(lhsCollisions[index], rhsCollisions[index])) {
            return false;
        }
    }
    return true;
}

GmSurfMeshTriangle Triangle(u32 a, u32 b, u32 c) {
    GmSurfMeshTriangle triangle{};
    triangle.normal = {0.0f, 0.0f, 1.0f};
    triangle.vertexIndex = {a, b, c};
    triangle.material = GmLocalMaterialIndex::FromIndex(7u);
    return triangle;
}

bool BuildMesh(GmSurfMesh *mesh,
               std::vector<GmVec3> vertices,
               std::vector<GmSurfMeshTriangle> triangles) {
    mesh->material = GmLocalMaterialIndex::FromIndex(31u);
    return mesh->SetGeometry(
            std::move(vertices),
            std::move(triangles),
            {},
            GmSurfMesh::PlaneSource::Archived);
}

bool BuildUnitMesh(GmSurfMesh *mesh) {
    return BuildMesh(
            mesh,
            {{0.0f, 0.0f, 0.0f},
             {1.0f, 0.0f, 0.0f},
             {0.0f, 1.0f, 0.0f}},
            {Triangle(0u, 1u, 2u)});
}

GmIso4 IdentityAt(GmVec3 translation) {
    GmIso4 iso;
    iso.SetIdentity();
    iso.translation = translation;
    return iso;
}

bool CheckPrecomputedEdges(
        const char *caseName,
        const GmSurfMesh &mesh,
        const OptimizedCpuStaticMeshTriangleSidecar &sidecar) {
    if (!sidecar.IsFor(mesh)) {
        std::fprintf(stderr, "%s sidecar identity differs\n", caseName);
        return false;
    }
    for (u32 triangleIndex = 0u;
         triangleIndex < mesh.TriangleCount();
         ++triangleIndex) {
        const GmSurfMeshTriangle &source = mesh.Triangle(triangleIndex);
        const OptimizedCpuStaticMeshTriangleData &cached =
                sidecar.TriangleAt(triangleIndex);
        if (!SameBits(source.normal, cached.normal) ||
            source.material.Index() != cached.material.Index()) {
            std::fprintf(stderr, "%s triangle metadata differs\n", caseName);
            return false;
        }
        for (u32 edgeIndex = 0u; edgeIndex < 3u; ++edgeIndex) {
            const u32 nextIndex = edgeIndex == 2u
                    ? 0u
                    : edgeIndex + 1u;
            const GmVec3 edgeStart =
                    mesh.Vertex(source.vertexIndex[edgeIndex]);
            const GmVec3 edgeEnd =
                    mesh.Vertex(source.vertexIndex[nextIndex]);
            const GmVec3 edgeDirection =
                    edgeEnd.SubtractForCollision(edgeStart)
                            .NormalizeForCollision(
                                    PhysicsTolerance::
                                            SurfaceDirectionLengthSquared);
            const GmVec3 edgeNormal = edgeDirection.Cross(source.normal);
            if (!SameBits(edgeStart, cached.vertices[edgeIndex]) ||
                !SameBits(edgeDirection,
                          cached.edgeDirections[edgeIndex]) ||
                !SameBits(edgeNormal, cached.edgeNormals[edgeIndex])) {
                std::fprintf(stderr,
                             "%s triangle %u edge %u precompute differs\n",
                             caseName,
                             triangleIndex,
                             edgeIndex);
                return false;
            }
        }
    }
    u32 postingIndex = 0u;
    for (u32 cellIndex = 0u;
         cellIndex < mesh.OctreeCellCount();
         ++cellIndex) {
        const GmMeshOctreeCell &cell = mesh.OctreeCell(cellIndex);
        if (!cell.ContainsTriangle()) {
            continue;
        }
        if (postingIndex >= sidecar.DirectTriangleCount()) {
            std::fprintf(stderr,
                         "%s direct posting count is short\n",
                         caseName);
            return false;
        }
        const OptimizedCpuStaticMeshDirectTrianglePosting &posting =
                sidecar.DirectTriangleAt(postingIndex);
        if (posting.triangleIndex != cell.TriangleIndex() ||
            !SameBits(posting.bounds, cell.Bounds())) {
            std::fprintf(stderr,
                         "%s direct posting %u differs from source cell %u\n",
                         caseName,
                         postingIndex,
                         cellIndex);
            return false;
        }
        ++postingIndex;
    }
    if (postingIndex != sidecar.DirectTriangleCount()) {
        std::fprintf(stderr,
                     "%s direct posting count has trailing records\n",
                     caseName);
        return false;
    }
    return true;
}

bool RunQueryCase(
        const char *caseName,
        const GmSurfMesh &mesh,
        const OptimizedCpuStaticMeshTriangleSidecar &sidecar,
        GmVec3 localCenter,
        float radius,
        const GmIso4 &meshIso) {
    GmSurfSphere sphere;
    sphere.material = GmLocalMaterialIndex::FromIndex(5u);
    sphere.radius = radius;
    GmVec3 worldCenter = localCenter;
    worldCenter.Mult(meshIso);
    GmIso4 sphereIso = IdentityAt(worldCenter);
    const LocatedGmSurf locatedSphere = {&sphere, &sphereIso, true};
    const LocatedGmSurf locatedMesh = {&mesh, &meshIso, true};
    GmIso4 meshInverse;
    meshInverse.SetInverse(meshIso);

    VectorCollisionBuffer baselineFirst;
    VectorCollisionBuffer sidecarSecond;
    const int baselineFirstHit =
            GmCollision_Sphere_Mesh_OptimizedCpuWithMeshInverse(
                    locatedSphere,
                    locatedMesh,
                    meshInverse,
                    baselineFirst);
    const int sidecarSecondHit =
            GmCollision_Sphere_Mesh_OptimizedCpuWithStaticTriangleSidecar(
                    locatedSphere,
                    locatedMesh,
                    meshInverse,
                    sidecar,
                    sidecarSecond);
    if (baselineFirstHit != sidecarSecondHit ||
        !SameBuffers(baselineFirst, sidecarSecond)) {
        std::fprintf(stderr, "%s baseline-first query differs\n", caseName);
        return false;
    }

    VectorCollisionBuffer sidecarFirst;
    VectorCollisionBuffer baselineSecond;
    const int sidecarFirstHit =
            GmCollision_Sphere_Mesh_OptimizedCpuWithStaticTriangleSidecar(
                    locatedSphere,
                    locatedMesh,
                    meshInverse,
                    sidecar,
                    sidecarFirst);
    const int baselineSecondHit =
            GmCollision_Sphere_Mesh_OptimizedCpuWithMeshInverse(
                    locatedSphere,
                    locatedMesh,
                    meshInverse,
                    baselineSecond);
    if (sidecarFirstHit != baselineSecondHit ||
        baselineFirstHit != baselineSecondHit ||
        !SameBuffers(sidecarFirst, baselineSecond) ||
        !SameBuffers(baselineFirst, baselineSecond)) {
        std::fprintf(stderr, "%s sidecar-first query differs\n", caseName);
        return false;
    }

    VectorCollisionBuffer nativeUncached;
    VectorCollisionBuffer nativeCached;
    const int nativeUncachedHit =
            GmCollision_Sphere_Mesh_InlineMathOptimizedCpuNativeBinary32(
                    locatedSphere,
                    locatedMesh,
                    nativeUncached);
    const int nativeCachedHit =
            GmCollision_Sphere_Mesh_InlineMathOptimizedCpuNativeBinary32WithStaticCache(
                    locatedSphere,
                    locatedMesh,
                    meshInverse,
                    sidecar,
                    nativeCached);
    if (baselineFirstHit != nativeUncachedHit ||
        nativeUncachedHit != nativeCachedHit ||
        !SameBuffers(baselineFirst, nativeUncached) ||
        !SameBuffers(nativeUncached, nativeCached)) {
        std::fprintf(stderr, "%s native cached query differs\n", caseName);
        return false;
    }
    return true;
}

bool RunFocusedEdgeCases(void) {
    const float directionLength =
            PhysicsTolerance::SurfaceDirectionLength;
    const float belowDirectionLength = std::nextafter(
            directionLength, 0.0f);
    const float aboveDirectionLength = std::nextafter(
            directionLength, std::numeric_limits<float>::infinity());
    const struct {
        const char *name;
        float edgeLength;
        bool normalized;
    } cases[] = {
        {"zero-edge", 0.0f, false},
        {"direction-threshold-below", belowDirectionLength, false},
        {"direction-threshold-exact", directionLength, false},
        {"direction-threshold-above", aboveDirectionLength, true},
    };

    const GmIso4 identity = IdentityAt({0.0f, 0.0f, 0.0f});
    for (const auto &edgeCase : cases) {
        const bool normalized =
                PhysicsTolerance::SurfaceDirectionLengthSquared <
                edgeCase.edgeLength * edgeCase.edgeLength;
        if (normalized != edgeCase.normalized) {
            std::fprintf(stderr,
                         "%s did not straddle the direction threshold\n",
                         edgeCase.name);
            return false;
        }
        GmSurfMesh mesh;
        if (!BuildMesh(
                    &mesh,
                    {{0.0f, 0.0f, 0.0f},
                     {edgeCase.edgeLength, 0.0f, 0.0f},
                     {0.0f, 1.0f, 0.0f}},
                    {Triangle(0u, 1u, 2u)})) {
            std::fprintf(stderr, "%s mesh construction failed\n",
                         edgeCase.name);
            return false;
        }
        OptimizedCpuStaticMeshTriangleSidecar sidecar;
        if (!sidecar.TryBuild(mesh) ||
            !CheckPrecomputedEdges(edgeCase.name, mesh, sidecar) ||
            !RunQueryCase(edgeCase.name,
                          mesh,
                          sidecar,
                          {edgeCase.edgeLength * 0.25f, 0.25f, 0.25f},
                          0.5f,
                          identity)) {
            return false;
        }
    }
    return true;
}

bool RunShuffledPostingOrderCase(void) {
    GmSurfMesh mesh;
    std::vector<GmMeshOctreeCell> cells = {
        GmMeshOctreeCell::Branch(
                {{1.5f, 0.5f, 0.0f}, {1.5f, 0.5f, 0.0f}}),
        GmMeshOctreeCell::Triangle(
                {{2.5f, 0.5f, 0.0f}, {0.5f, 0.5f, 0.0f}}, 1u),
        GmMeshOctreeCell::Triangle(
                {{0.5f, 0.5f, 0.0f}, {0.5f, 0.5f, 0.0f}}, 0u),
    };
    cells[0].SetSubtreeEntryCount(3u);
    if (!mesh.SetGeometry(
                {{0.0f, 0.0f, 0.0f},
                 {1.0f, 0.0f, 0.0f},
                 {0.0f, 1.0f, 0.0f},
                 {2.0f, 0.0f, 0.0f},
                 {3.0f, 0.0f, 0.0f},
                 {2.0f, 1.0f, 0.0f}},
                {Triangle(0u, 1u, 2u), Triangle(3u, 4u, 5u)},
                std::move(cells),
                GmSurfMesh::PlaneSource::Archived)) {
        return false;
    }
    OptimizedCpuStaticMeshTriangleSidecar sidecar;
    const GmIso4 identity = IdentityAt({0.0f, 0.0f, 0.0f});
    return sidecar.TryBuild(mesh) &&
           CheckPrecomputedEdges(
                   "shuffled-triangle-posting-order", mesh, sidecar) &&
           RunQueryCase(
                   "shuffled-triangle-posting-order",
                   mesh,
                   sidecar,
                   {2.25f, 0.25f, 0.25f},
                   0.5f,
                   identity);
}

bool RunAllocationAndInvalidationCases(const GmSurfMesh &unit) {
    OptimizedCpuStaticMeshTriangleSidecar allocationFallback;
    failNextAllocation = true;
    if (allocationFallback.TryBuild(unit) || failNextAllocation ||
        allocationFallback.IsFor(unit)) {
        std::fprintf(stderr, "allocation fallback differs\n");
        return false;
    }
    if (!allocationFallback.TryBuild(unit) ||
        !allocationFallback.IsFor(unit)) {
        std::fprintf(stderr, "post-allocation rebuild differs\n");
        return false;
    }

    GmSurfMesh mutableMesh;
    if (!BuildUnitMesh(&mutableMesh)) {
        return false;
    }
    OptimizedCpuStaticMeshTriangleSidecar invalidated;
    if (!invalidated.TryBuild(mutableMesh) ||
        !invalidated.IsFor(mutableMesh)) {
        return false;
    }
    mutableMesh.TranslateVerticesAbove(-1.0f, 0.25f);
    if (invalidated.IsFor(mutableMesh)) {
        std::fprintf(stderr, "mutated mesh retained sidecar identity\n");
        return false;
    }
    if (!invalidated.TryBuild(mutableMesh) ||
        !invalidated.IsFor(mutableMesh)) {
        std::fprintf(stderr, "mutated mesh sidecar rebuild differs\n");
        return false;
    }
    mutableMesh.ClearGeometry();
    if (invalidated.IsFor(mutableMesh)) {
        std::fprintf(stderr, "cleared mesh retained sidecar identity\n");
        return false;
    }
    invalidated.Clear();
    if (invalidated.IsFor(unit)) {
        std::fprintf(stderr, "cleared sidecar retained identity\n");
        return false;
    }
    return true;
}

bool RunFloatingEnvironmentPreservationCase(const GmSurfMesh &unit) {
    if (std::feclearexcept(FE_ALL_EXCEPT) != 0 ||
        std::feraiseexcept(FE_INVALID | FE_DIVBYZERO) != 0) {
        std::fprintf(stderr, "could not prepare floating environment\n");
        return false;
    }
    const int exceptionsBefore = std::fetestexcept(FE_ALL_EXCEPT);
    const int roundingBefore = std::fegetround();
#if defined(__i386__) || defined(__x86_64__)
    const unsigned int mxcsrBefore = _mm_getcsr();
#endif

    OptimizedCpuStaticMeshTriangleSidecar sidecar;
    const bool built = sidecar.TryBuild(unit);
    const int exceptionsAfter = std::fetestexcept(FE_ALL_EXCEPT);
    const int roundingAfter = std::fegetround();
#if defined(__i386__) || defined(__x86_64__)
    const unsigned int mxcsrAfter = _mm_getcsr();
#endif

    const bool identical = built && sidecar.IsFor(unit) &&
            exceptionsBefore == exceptionsAfter &&
            roundingBefore == roundingAfter
#if defined(__i386__) || defined(__x86_64__)
            && mxcsrBefore == mxcsrAfter
#endif
            ;
    std::feclearexcept(FE_ALL_EXCEPT);
    if (!identical) {
        std::fprintf(stderr,
                     "sidecar build changed floating environment\n");
        return false;
    }
    return true;
}

bool RunMismatchFallback(
        const GmSurfMesh &unit,
        const OptimizedCpuStaticMeshTriangleSidecar &unitSidecar) {
    GmSurfMesh secondMesh;
    if (!BuildUnitMesh(&secondMesh)) {
        return false;
    }
    GmIso4 meshIso = IdentityAt({2.0f, -3.0f, 4.0f});
    return RunQueryCase(
            "mesh-identity-fallback",
            secondMesh,
            unitSidecar,
            {0.25f, 0.25f, 0.25f},
            0.5f,
            meshIso) &&
           unitSidecar.IsFor(unit);
}

bool RunDeterministicFuzz(
        const GmSurfMesh &mesh,
        const OptimizedCpuStaticMeshTriangleSidecar &sidecar) {
    XorShift32 random;
    for (u32 caseIndex = 0u; caseIndex < 2048u; ++caseIndex) {
        const GmVec3 localCenter = {
            static_cast<float>(static_cast<int>(random.Next() % 513u) -
                               128) /
                    128.0f,
            static_cast<float>(static_cast<int>(random.Next() % 513u) -
                               128) /
                    128.0f,
            static_cast<float>(static_cast<int>(random.Next() % 513u) -
                               256) /
                    128.0f,
        };
        const float radius =
                static_cast<float>(1u + random.Next() % 128u) / 64.0f;
        GmIso4 meshIso;
        meshIso.SetIdentity();
        meshIso.rotation.SetRotateQuarterY(random.Next() & 3u);
        meshIso.translation = {
            static_cast<float>(static_cast<int>(random.Next() % 17u) - 8) /
                    4.0f,
            static_cast<float>(static_cast<int>(random.Next() % 17u) - 8) /
                    4.0f,
            static_cast<float>(static_cast<int>(random.Next() % 17u) - 8) /
                    4.0f,
        };
        char caseName[32];
        std::snprintf(caseName, sizeof(caseName), "sidecar-fuzz-%04u",
                      caseIndex);
        if (!RunQueryCase(
                    caseName, mesh, sidecar, localCenter, radius, meshIso)) {
            return false;
        }
    }
    return true;
}

bool RunUniformGridCoverageFuzz(void) {
    XorShift32 random;
    std::vector<OptimizedCpuStaticUniformGrid::Entry> entries;
    entries.reserve(257u);
    for (u32 entryIndex = 0u; entryIndex < 257u; ++entryIndex) {
        const GmVec3 center = {
            static_cast<float>(static_cast<int>(random.Next() % 8193u) -
                               4096) /
                    128.0f,
            static_cast<float>(static_cast<int>(random.Next() % 2049u) -
                               1024) /
                    128.0f,
            static_cast<float>(static_cast<int>(random.Next() % 8193u) -
                               4096) /
                    128.0f,
        };
        const GmVec3 half = {
            static_cast<float>(1u + random.Next() % 257u) / 128.0f,
            static_cast<float>(1u + random.Next() % 129u) / 128.0f,
            static_cast<float>(1u + random.Next() % 257u) / 128.0f,
        };
        entries.push_back({entryIndex * 2u + 1u, {center, half}});
    }

    OptimizedCpuStaticUniformGrid grid;
    if (!grid.TryBuild(entries, 514u, 4096u)) {
        std::fprintf(stderr, "uniform grid construction failed\n");
        return false;
    }

    u32 indexedQueryCount = 0u;
    for (u32 caseIndex = 0u; caseIndex < 4096u; ++caseIndex) {
        const GmBoxAligned query = {
            {
                static_cast<float>(
                        static_cast<int>(random.Next() % 10241u) - 5120) /
                        128.0f,
                static_cast<float>(
                        static_cast<int>(random.Next() % 3073u) - 1536) /
                        128.0f,
                static_cast<float>(
                        static_cast<int>(random.Next() % 10241u) - 5120) /
                        128.0f,
            },
            {
                static_cast<float>(1u + random.Next() % 513u) / 128.0f,
                static_cast<float>(1u + random.Next() % 257u) / 128.0f,
                static_cast<float>(1u + random.Next() % 513u) / 128.0f,
            },
        };
        OptimizedCpuStaticUniformGrid::CandidateSpan span;
        if (!grid.DirectCandidateSpan(query, &span)) {
            continue;
        }
        ++indexedQueryCount;
        for (std::size_t candidateIndex = 1u;
             candidateIndex < span.size;
             ++candidateIndex) {
            if (span.data[candidateIndex - 1u] >=
                span.data[candidateIndex]) {
                std::fprintf(stderr,
                             "uniform grid candidate order differs\n");
                return false;
            }
        }
        for (const auto &entry : entries) {
            if (!query.TestInter(entry.bounds)) {
                continue;
            }
            if (span.size == 0u ||
                !std::binary_search(
                        span.data, span.data + span.size, entry.sourceIndex)) {
                std::fprintf(stderr,
                             "uniform grid omitted intersecting entry in "
                             "case %u\n",
                             caseIndex);
                return false;
            }
        }
    }
    if (indexedQueryCount < 4000u) {
        std::fprintf(stderr, "uniform grid used too few fuzz queries\n");
        return false;
    }

    GmBoxAligned invalid{};
    invalid.center.x = std::numeric_limits<float>::quiet_NaN();
    OptimizedCpuStaticUniformGrid::CandidateSpan invalidSpan;
    if (grid.DirectCandidateSpan(invalid, &invalidSpan)) {
        std::fprintf(stderr, "uniform grid accepted non-finite query\n");
        return false;
    }
    return true;
}

}  // namespace

int main(void) {
    GmSurfMesh inactiveMesh;
    OptimizedCpuStaticMeshTriangleSidecar inactiveSidecar;
    if (inactiveSidecar.TryBuild(inactiveMesh)) {
        std::fprintf(stderr,
                     "sidecar built outside deterministic execution\n");
        return 1;
    }

    tmnf::simulation::DeterministicExecutionScope deterministicScope;
    if (!deterministicScope.Established()) {
        std::fprintf(stderr, "could not establish deterministic execution\n");
        return 1;
    }

    GmSurfMesh unit;
    if (!BuildUnitMesh(&unit)) {
        std::fprintf(stderr, "could not construct unit mesh\n");
        return 1;
    }
    OptimizedCpuStaticMeshTriangleSidecar unitSidecar;
    if (!unitSidecar.TryBuild(unit) ||
        !CheckPrecomputedEdges("unit", unit, unitSidecar) ||
        !RunFocusedEdgeCases() ||
        !RunShuffledPostingOrderCase() ||
        !RunAllocationAndInvalidationCases(unit) ||
        !RunFloatingEnvironmentPreservationCase(unit) ||
        !RunMismatchFallback(unit, unitSidecar) ||
        !RunDeterministicFuzz(unit, unitSidecar) ||
        !RunUniformGridCoverageFuzz()) {
        return 1;
    }

    std::printf(
            "static_triangle_sidecar_query_cases=2054 "
            "uniform_grid_coverage_cases=4097 "
            "lifecycle_cases=8 floating_environment_cases=1 "
            "result=identical\n");
    std::printf(
            "triangle_payload_bytes=%zu sidecar_object_bytes=%zu "
            "direct_triangle_posting_bytes=%zu record_pointer_bytes=%zu\n",
            sizeof(OptimizedCpuStaticMeshTriangleData),
            sizeof(OptimizedCpuStaticMeshTriangleSidecar),
            sizeof(OptimizedCpuStaticMeshDirectTrianglePosting),
            sizeof(const OptimizedCpuStaticMeshTriangleSidecar *));
    return 0;
}
