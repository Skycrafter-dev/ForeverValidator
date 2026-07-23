#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/geometry/gm_surface.h"
#include "engine/physics/geometry/gmsurf_collision.h"
#include "engine/physics/geometry/plug_surface.h"
#include "engine/rendering/plug_material.h"
#include "simulation/runtime/replay_deterministic_execution.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_binary32_math.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_native_binary32_collision.h"

#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace {

std::size_t referenceEllipsoidMeshCallCount = 0u;

}  // namespace

#if defined(FOREVERVALIDATOR_HAS_GNU_LD_WRAP)
int RealGmCollisionEllipsoidMesh(
        const LocatedGmSurf &ellipsoid,
        const LocatedGmSurf &mesh,
        CGmCollisionBuffer &collisionBuffer)
        asm("__real__Z26GmCollision_Ellipsoid_MeshRK13LocatedGmSurfS1_R18CGmCollisionBuffer");
int WrappedGmCollisionEllipsoidMesh(
        const LocatedGmSurf &ellipsoid,
        const LocatedGmSurf &mesh,
        CGmCollisionBuffer &collisionBuffer)
        asm("__wrap__Z26GmCollision_Ellipsoid_MeshRK13LocatedGmSurfS1_R18CGmCollisionBuffer");

int WrappedGmCollisionEllipsoidMesh(
        const LocatedGmSurf &ellipsoid,
        const LocatedGmSurf &mesh,
        CGmCollisionBuffer &collisionBuffer) {
    ++referenceEllipsoidMeshCallCount;
    return RealGmCollisionEllipsoidMesh(
            ellipsoid, mesh, collisionBuffer);
}
#endif

namespace {

std::size_t completedDirectCaseCount = 0u;
std::size_t completedPlugCaseCount = 0u;
std::size_t completedMxcsrCaseCount = 0u;
forevervalidator::simulation::OptimizedCpuBinary32MathPath binary32MathPath =
        forevervalidator::simulation::OptimizedCpuBinary32MathPath::Reference;

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

    void Seed(const GmCollision &collision) {
        collisions_.push_back(collision);
    }

    const std::vector<GmCollision> &Collisions(void) const {
        return collisions_;
    }

private:
    std::vector<GmCollision> collisions_;
};

struct XorShift32 {
    std::uint32_t state = 0x8f7011eeu;

    std::uint32_t Next(void) {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return state;
    }
};

std::uint32_t FloatBits(float value) {
    std::uint32_t bits = 0u;
    static_assert(sizeof(bits) == sizeof(value), "binary32 size mismatch");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool CompareFloat(const char *caseName,
                  const char *passName,
                  std::size_t collisionIndex,
                  const char *field,
                  float reference,
                  float optimized) {
    const std::uint32_t referenceBits = FloatBits(reference);
    const std::uint32_t optimizedBits = FloatBits(optimized);
    if (referenceBits == optimizedBits) {
        return true;
    }
    std::fprintf(stderr,
                 "%s/%s collision %zu %s differs: reference=%08x "
                 "optimized=%08x\n",
                 caseName,
                 passName,
                 collisionIndex,
                 field,
                 referenceBits,
                 optimizedBits);
    return false;
}

bool CompareVector(const char *caseName,
                   const char *passName,
                   std::size_t collisionIndex,
                   const char *field,
                   const GmVec3 &reference,
                   const GmVec3 &optimized) {
    char component[64];
    std::snprintf(component, sizeof(component), "%s.x", field);
    if (!CompareFloat(caseName,
                      passName,
                      collisionIndex,
                      component,
                      reference.x,
                      optimized.x)) {
        return false;
    }
    std::snprintf(component, sizeof(component), "%s.y", field);
    if (!CompareFloat(caseName,
                      passName,
                      collisionIndex,
                      component,
                      reference.y,
                      optimized.y)) {
        return false;
    }
    std::snprintf(component, sizeof(component), "%s.z", field);
    return CompareFloat(caseName,
                        passName,
                        collisionIndex,
                        component,
                        reference.z,
                        optimized.z);
}

bool CompareCollision(const char *caseName,
                      const char *passName,
                      std::size_t collisionIndex,
                      const GmCollision &reference,
                      const GmCollision &optimized) {
    if (!CompareVector(caseName,
                       passName,
                       collisionIndex,
                       "separation",
                       reference.separation,
                       optimized.separation) ||
        !CompareVector(caseName,
                       passName,
                       collisionIndex,
                       "impulseNormal",
                       reference.impulseNormal,
                       optimized.impulseNormal) ||
        !CompareVector(caseName,
                       passName,
                       collisionIndex,
                       "contactPoint",
                       reference.contactPoint,
                       optimized.contactPoint) ||
        !CompareVector(caseName,
                       passName,
                       collisionIndex,
                       "extraNegated",
                       reference.extraNegated,
                       optimized.extraNegated)) {
        return false;
    }
    if (reference.localMaterialA.Index() !=
                optimized.localMaterialA.Index() ||
        reference.localMaterialB.Index() !=
                optimized.localMaterialB.Index() ||
        reference.materialA != optimized.materialA ||
        reference.materialB != optimized.materialB ||
        reference.sphereMergePrimary != optimized.sphereMergePrimary) {
        std::fprintf(stderr,
                     "%s/%s collision %zu metadata differs: "
                     "localA=%u/%u localB=%u/%u materialA=%u/%u "
                     "materialB=%u/%u primary=%u/%u\n",
                     caseName,
                     passName,
                     collisionIndex,
                     reference.localMaterialA.Index(),
                     optimized.localMaterialA.Index(),
                     reference.localMaterialB.Index(),
                     optimized.localMaterialB.Index(),
                     static_cast<unsigned>(reference.materialA),
                     static_cast<unsigned>(optimized.materialA),
                     static_cast<unsigned>(reference.materialB),
                     static_cast<unsigned>(optimized.materialB),
                     static_cast<unsigned>(reference.sphereMergePrimary),
                     static_cast<unsigned>(optimized.sphereMergePrimary));
        return false;
    }
    return true;
}

bool CompareBuffers(const char *caseName,
                    const char *passName,
                    const VectorCollisionBuffer &reference,
                    const VectorCollisionBuffer &optimized) {
    const std::vector<GmCollision> &referenceCollisions =
            reference.Collisions();
    const std::vector<GmCollision> &optimizedCollisions =
            optimized.Collisions();
    if (referenceCollisions.size() != optimizedCollisions.size()) {
        std::fprintf(stderr,
                     "%s/%s collision count differs: reference=%zu "
                     "optimized=%zu\n",
                     caseName,
                     passName,
                     referenceCollisions.size(),
                     optimizedCollisions.size());
        return false;
    }
    for (std::size_t index = 0u;
         index < referenceCollisions.size();
         ++index) {
        if (!CompareCollision(caseName,
                              passName,
                              index,
                              referenceCollisions[index],
                              optimizedCollisions[index])) {
            return false;
        }
    }
    return true;
}

GmCollision SentinelCollision(void) {
    GmCollision collision{};
    collision.separation = {-0.0f, 1.25f, -2.5f};
    collision.impulseNormal = {3.0f, -4.0f, 5.0f};
    collision.contactPoint = {-6.0f, 7.0f, -8.0f};
    collision.localMaterialA = GmLocalMaterialIndex::FromIndex(101u);
    collision.localMaterialB = GmLocalMaterialIndex::FromIndex(202u);
    collision.materialA = EPlugSurfaceMaterialId_Ice;
    collision.materialB = EPlugSurfaceMaterialId_Wood;
    collision.sphereMergePrimary = true;
    collision.extraNegated = {9.0f, -10.0f, 11.0f};
    return collision;
}

int RunReference(const LocatedGmSurf &ellipsoid,
                 const LocatedGmSurf &mesh,
                 VectorCollisionBuffer *buffer) {
    buffer->Seed(SentinelCollision());
    return GmCollision_Ellipsoid_Mesh(ellipsoid, mesh, *buffer);
}

int RunOptimized(const LocatedGmSurf &ellipsoid,
                 const LocatedGmSurf &mesh,
                 VectorCollisionBuffer *buffer) {
    buffer->Seed(SentinelCollision());
    return GmCollision_Ellipsoid_Mesh_OptimizedCpu(
            ellipsoid, mesh, *buffer);
}

int RunNativeSelected(const LocatedGmSurf &ellipsoid,
                      const LocatedGmSurf &mesh,
                      VectorCollisionBuffer *buffer) {
    buffer->Seed(SentinelCollision());
    if (binary32MathPath == forevervalidator::simulation::
                                    OptimizedCpuBinary32MathPath::X86Sse2) {
        return GmCollision_Ellipsoid_Mesh_InlineMathOptimizedCpuNativeBinary32(
                ellipsoid, mesh, *buffer);
    }
    return GmCollision_Ellipsoid_Mesh_OptimizedCpu(
            ellipsoid, mesh, *buffer);
}

int RunNativeStaticInverseSelected(
        const LocatedGmSurf &ellipsoid,
        const LocatedGmSurf &mesh,
        VectorCollisionBuffer *buffer) {
    buffer->Seed(SentinelCollision());
    if (binary32MathPath == forevervalidator::simulation::
                                    OptimizedCpuBinary32MathPath::X86Sse2) {
        GmIso4 meshInverse;
        meshInverse.SetInverse(*mesh.iso);
        return GmCollision_Ellipsoid_Mesh_InlineMathOptimizedCpuNativeBinary32WithStaticInverse(
                ellipsoid, mesh, meshInverse, *buffer);
    }
    return GmCollision_Ellipsoid_Mesh_OptimizedCpu(
            ellipsoid, mesh, *buffer);
}

bool CheckExpected(const char *caseName,
                   int expectedHit,
                   std::size_t minimumNewCollisions,
                   int referenceHit,
                   const VectorCollisionBuffer &reference) {
    const std::size_t collisionCount = reference.Collisions().size();
    const std::size_t newCollisionCount = collisionCount - 1u;
    if (expectedHit >= 0 &&
        (referenceHit != 0) != (expectedHit != 0)) {
        std::fprintf(stderr,
                     "%s did not exercise expected Reference result: "
                     "expected_hit=%d actual_hit=%d\n",
                     caseName,
                     expectedHit,
                     referenceHit);
        return false;
    }
    if (newCollisionCount < minimumNewCollisions ||
        (referenceHit != 0) != (newCollisionCount != 0u)) {
        std::fprintf(stderr,
                     "%s Reference hit/count differs: hit=%d new=%zu "
                     "minimum=%zu\n",
                     caseName,
                     referenceHit,
                     newCollisionCount,
                     minimumNewCollisions);
        return false;
    }
    return true;
}

bool RunDirectDifferential(const char *caseName,
                           const GmSurfEllipsoid &ellipsoid,
                           const GmSurfMesh &mesh,
                           const GmIso4 &ellipsoidIso,
                           const GmIso4 &meshIso,
                           bool ellipsoidEnabled = true,
                           bool meshEnabled = true,
                           int expectedHit = -1,
                           std::size_t minimumNewCollisions = 0u) {
    const LocatedGmSurf locatedEllipsoid = {
        &ellipsoid,
        &ellipsoidIso,
        ellipsoidEnabled,
    };
    const LocatedGmSurf locatedMesh = {
        &mesh,
        &meshIso,
        meshEnabled,
    };

    VectorCollisionBuffer referenceFirst;
    VectorCollisionBuffer optimizedSecond;
    VectorCollisionBuffer nativeThird;
    VectorCollisionBuffer nativeStaticFourth;
    const int referenceFirstHit =
            RunReference(locatedEllipsoid, locatedMesh, &referenceFirst);
    const int optimizedSecondHit =
            RunOptimized(locatedEllipsoid, locatedMesh, &optimizedSecond);
    const int nativeThirdHit =
            RunNativeSelected(locatedEllipsoid, locatedMesh, &nativeThird);
    const int nativeStaticFourthHit = RunNativeStaticInverseSelected(
            locatedEllipsoid, locatedMesh, &nativeStaticFourth);
    if (!CheckExpected(caseName,
                       expectedHit,
                       minimumNewCollisions,
                       referenceFirstHit,
                       referenceFirst) ||
        referenceFirstHit != optimizedSecondHit ||
        optimizedSecondHit != nativeThirdHit ||
        nativeThirdHit != nativeStaticFourthHit ||
        !CompareBuffers(caseName,
                        "reference-first",
                        referenceFirst,
                        optimizedSecond) ||
        !CompareBuffers(caseName,
                        "e005-before-native",
                        optimizedSecond,
                        nativeThird) ||
        !CompareBuffers(caseName,
                        "native-before-static-inverse",
                        nativeThird,
                        nativeStaticFourth)) {
        if (referenceFirstHit != optimizedSecondHit ||
            optimizedSecondHit != nativeThirdHit) {
            std::fprintf(stderr,
                         "%s/reference-first hit differs: reference=%d "
                         "optimized=%d native=%d\n",
                         caseName,
                         referenceFirstHit,
                         optimizedSecondHit,
                         nativeThirdHit);
        }
        return false;
    }

    VectorCollisionBuffer nativeFirst;
    VectorCollisionBuffer nativeStaticSecond;
    VectorCollisionBuffer optimizedSecondPass;
    VectorCollisionBuffer referenceSecond;
    const int nativeFirstHit =
            RunNativeSelected(locatedEllipsoid, locatedMesh, &nativeFirst);
    const int nativeStaticSecondHit = RunNativeStaticInverseSelected(
            locatedEllipsoid, locatedMesh, &nativeStaticSecond);
    const int optimizedSecondPassHit =
            RunOptimized(locatedEllipsoid, locatedMesh, &optimizedSecondPass);
    const int referenceSecondHit =
            RunReference(locatedEllipsoid, locatedMesh, &referenceSecond);
    if (nativeFirstHit != optimizedSecondPassHit ||
        nativeFirstHit != nativeStaticSecondHit ||
        optimizedSecondPassHit != referenceSecondHit ||
        referenceFirstHit != referenceSecondHit ||
        !CompareBuffers(caseName,
                        "native-before-e005",
                        optimizedSecondPass,
                        nativeFirst) ||
        !CompareBuffers(caseName,
                        "native-static-repeat",
                        nativeFirst,
                        nativeStaticSecond) ||
        !CompareBuffers(caseName,
                        "e005-repeat",
                        optimizedSecond,
                        optimizedSecondPass) ||
        !CompareBuffers(caseName,
                        "reference-repeat",
                        referenceFirst,
                        referenceSecond)) {
        std::fprintf(stderr,
                     "%s/native-first result differs: native=%d "
                     "optimized=%d reference=%d repeated_reference=%d\n",
                     caseName,
                     nativeFirstHit,
                     optimizedSecondPassHit,
                     referenceSecondHit,
                     referenceFirstHit);
        return false;
    }
    ++completedDirectCaseCount;
    return true;
}

GmSurfMeshTriangle Triangle(u32 a,
                            u32 b,
                            u32 c,
                            std::uint16_t material) {
    GmSurfMeshTriangle triangle{};
    triangle.normal = {0.0f, 0.0f, 1.0f};
    triangle.planeDistance = 0.0f;
    triangle.vertexIndex = {a, b, c};
    triangle.material = GmLocalMaterialIndex::FromIndex(material);
    return triangle;
}

bool BuildUnitMesh(GmSurfMesh *mesh) {
    mesh->material = GmLocalMaterialIndex::FromIndex(31u);
    return mesh->SetGeometry(
            {{0.0f, 0.0f, 0.0f},
             {1.0f, 0.0f, 0.0f},
             {0.0f, 1.0f, 0.0f}},
            {Triangle(0u, 1u, 2u, 7u)},
            {},
            GmSurfMesh::PlaneSource::Archived);
}

bool BuildSeparatedBranchMesh(GmSurfMesh *mesh) {
    const GmBoxAligned nearBounds = {
        {0.5f, 0.5f, 0.0f},
        {0.5f, 0.5f, 0.0f},
    };
    const GmBoxAligned farBounds = {
        {10.5f, 0.5f, 0.0f},
        {0.5f, 0.5f, 0.0f},
    };
    const GmBoxAligned rootBounds = {
        {5.5f, 0.5f, 0.0f},
        {5.5f, 0.5f, 0.0f},
    };
    GmMeshOctreeCell root = GmMeshOctreeCell::Branch(rootBounds);
    root.SetSubtreeEntryCount(5u);
    GmMeshOctreeCell nearBranch = GmMeshOctreeCell::Branch(nearBounds);
    nearBranch.SetSubtreeEntryCount(2u);
    GmMeshOctreeCell farBranch = GmMeshOctreeCell::Branch(farBounds);
    farBranch.SetSubtreeEntryCount(2u);

    mesh->material = GmLocalMaterialIndex::FromIndex(41u);
    return mesh->SetGeometry(
            {{0.0f, 0.0f, 0.0f},
             {1.0f, 0.0f, 0.0f},
             {0.0f, 1.0f, 0.0f},
             {10.0f, 0.0f, 0.0f},
             {11.0f, 0.0f, 0.0f},
             {10.0f, 1.0f, 0.0f}},
            {Triangle(0u, 1u, 2u, 11u),
             Triangle(3u, 4u, 5u, 13u)},
            {root,
             nearBranch,
             GmMeshOctreeCell::Triangle(nearBounds, 0u),
             farBranch,
             GmMeshOctreeCell::Triangle(farBounds, 1u)},
            GmSurfMesh::PlaneSource::Archived);
}

bool BuildOverlappingOrderMesh(GmSurfMesh *mesh) {
    const GmBoxAligned bounds = {
        {0.5f, 0.5f, 0.0f},
        {0.5f, 0.5f, 0.0f},
    };
    GmMeshOctreeCell root = GmMeshOctreeCell::Branch(bounds);
    root.SetSubtreeEntryCount(3u);
    mesh->material = GmLocalMaterialIndex::FromIndex(51u);
    return mesh->SetGeometry(
            {{0.0f, 0.0f, 0.0f},
             {1.0f, 0.0f, 0.0f},
             {0.0f, 1.0f, 0.0f}},
            {Triangle(0u, 1u, 2u, 17u),
             Triangle(0u, 1u, 2u, 19u)},
            {root,
             GmMeshOctreeCell::Triangle(bounds, 1u),
             GmMeshOctreeCell::Triangle(bounds, 0u)},
            GmSurfMesh::PlaneSource::Archived);
}

bool BuildDegenerateMesh(GmSurfMesh *mesh) {
    const GmBoxAligned bounds = {
        {0.5f, 0.0f, 0.0f},
        {0.5f, 0.0f, 0.0f},
    };
    mesh->material = GmLocalMaterialIndex::FromIndex(61u);
    return mesh->SetGeometry(
            {{0.0f, 0.0f, 0.0f},
             {0.5f, 0.0f, 0.0f},
             {1.0f, 0.0f, 0.0f}},
            {Triangle(0u, 1u, 2u, 23u)},
            {GmMeshOctreeCell::Triangle(bounds, 0u)},
            GmSurfMesh::PlaneSource::Archived);
}

GmIso4 IdentityAt(const GmVec3 &translation) {
    GmIso4 result;
    result.SetIdentity();
    result.translation = translation;
    return result;
}

bool RunCoreCases(const GmSurfMesh &unit,
                  const GmSurfMesh &separated,
                  const GmSurfMesh &overlapping,
                  const GmSurfMesh &degenerate) {
    const GmIso4 identity = IdentityAt({0.0f, 0.0f, 0.0f});
    GmSurfEllipsoid ellipsoid;
    ellipsoid.material = GmLocalMaterialIndex::FromIndex(5u);
    ellipsoid.radii = {0.8f, 0.5f, 0.4f};

    if (!RunDirectDifferential("anisotropic-face-hit",
                               ellipsoid,
                               unit,
                               IdentityAt({0.25f, 0.25f, 0.2f}),
                               identity,
                               true,
                               true,
                               1,
                               1u) ||
        !RunDirectDifferential("anisotropic-behind-plane",
                               ellipsoid,
                               unit,
                               IdentityAt({0.25f, 0.25f, -0.2f}),
                               identity,
                               true,
                               true,
                               0)) {
        return false;
    }

    GmIso4 meshIso = IdentityAt({3.25f, -1.5f, 5.75f});
    meshIso.rotation.RotateX(0.37f);
    meshIso.rotation.RotateY(-0.21f);
    meshIso.rotation.RotateZ(0.13f);
    GmVec3 worldCenter = {0.25f, 0.25f, 0.2f};
    worldCenter.Mult(meshIso);
    GmIso4 ellipsoidIso = meshIso;
    ellipsoidIso.translation = worldCenter;
    if (!RunDirectDifferential("co-rotated-world-transform",
                               ellipsoid,
                               unit,
                               ellipsoidIso,
                               meshIso,
                               true,
                               true,
                               1,
                               1u)) {
        return false;
    }

    GmIso4 differentlyRotatedEllipsoid =
            IdentityAt({0.25f, 0.25f, 0.1f});
    differentlyRotatedEllipsoid.rotation.RotateX(-0.43f);
    differentlyRotatedEllipsoid.rotation.RotateY(0.29f);
    differentlyRotatedEllipsoid.rotation.RotateZ(0.17f);
    if (!RunDirectDifferential("anisotropic-independent-rotation",
                               ellipsoid,
                               unit,
                               differentlyRotatedEllipsoid,
                               identity,
                               true,
                               true,
                               1,
                               1u)) {
        return false;
    }

    GmSurfEllipsoid branchEllipsoid;
    branchEllipsoid.material = GmLocalMaterialIndex::FromIndex(9u);
    branchEllipsoid.radii = {0.5f, 0.35f, 0.3f};
    if (!RunDirectDifferential("near-subtree",
                               branchEllipsoid,
                               separated,
                               IdentityAt({0.25f, 0.25f, 0.15f}),
                               identity,
                               true,
                               true,
                               1,
                               1u) ||
        !RunDirectDifferential("far-subtree",
                               branchEllipsoid,
                               separated,
                               IdentityAt({10.25f, 0.25f, 0.15f}),
                               identity,
                               true,
                               true,
                               1,
                               1u) ||
        !RunDirectDifferential("ordered-overlap",
                               branchEllipsoid,
                               overlapping,
                               IdentityAt({0.25f, 0.25f, 0.15f}),
                               identity,
                               true,
                               true,
                               1,
                               2u) ||
        !RunDirectDifferential("degenerate-normal",
                               branchEllipsoid,
                               degenerate,
                               IdentityAt({0.25f, 0.0f, 0.0f}),
                               identity,
                               true,
                               true,
                               0)) {
        return false;
    }

    GmSurfEllipsoid boundaryEllipsoid;
    boundaryEllipsoid.material = GmLocalMaterialIndex::FromIndex(15u);
    boundaryEllipsoid.radii = {0.25f, 0.2f, 0.15f};
    const float touchingX = 1.0f + boundaryEllipsoid.radii.x;
    if (!RunDirectDifferential("bounds-touch",
                               boundaryEllipsoid,
                               unit,
                               IdentityAt({touchingX, 0.0f, 0.0f}),
                               identity) ||
        !RunDirectDifferential(
                "bounds-inside-ulp",
                boundaryEllipsoid,
                unit,
                IdentityAt({std::nextafter(
                                    touchingX,
                                    -std::numeric_limits<float>::infinity()),
                            0.0f,
                            0.0f}),
                identity) ||
        !RunDirectDifferential(
                "bounds-outside-ulp",
                boundaryEllipsoid,
                unit,
                IdentityAt({std::nextafter(
                                    touchingX,
                                    std::numeric_limits<float>::infinity()),
                            0.0f,
                            0.0f}),
                identity)) {
        return false;
    }

    GmIso4 signedZeroMesh = identity;
    signedZeroMesh.rotation.basisX.y = -0.0f;
    signedZeroMesh.rotation.basisX.z = -0.0f;
    signedZeroMesh.rotation.basisY.x = -0.0f;
    signedZeroMesh.rotation.basisY.z = -0.0f;
    signedZeroMesh.rotation.basisZ.x = -0.0f;
    signedZeroMesh.rotation.basisZ.y = -0.0f;
    signedZeroMesh.translation = {-0.0f, 0.0f, -0.0f};
    GmIso4 signedZeroEllipsoid =
            IdentityAt({0.25f, 0.25f, 0.1f});
    signedZeroEllipsoid.rotation.basisX.y = -0.0f;
    signedZeroEllipsoid.rotation.basisY.z = -0.0f;
    signedZeroEllipsoid.rotation.basisZ.x = -0.0f;
    if (!RunDirectDifferential("signed-zero-transform",
                               ellipsoid,
                               unit,
                               signedZeroEllipsoid,
                               signedZeroMesh,
                               true,
                               true,
                               1,
                               1u)) {
        return false;
    }

    GmIso4 nanEllipsoid = IdentityAt({0.25f, 0.25f, 0.1f});
    nanEllipsoid.translation.x =
            std::numeric_limits<float>::quiet_NaN();
    if (!RunDirectDifferential("nan-transform",
                               ellipsoid,
                               unit,
                               nanEllipsoid,
                               identity,
                               true,
                               true,
                               0)) {
        return false;
    }

    GmIso4 ignoredEllipsoidIso =
            IdentityAt({100.0f, 100.0f, 100.0f});
    ignoredEllipsoidIso.rotation.RotateY(0.77f);
    const GmIso4 translatedMesh = IdentityAt({0.0f, 0.0f, -0.15f});
    if (!RunDirectDifferential("ellipsoid-location-disabled",
                               branchEllipsoid,
                               unit,
                               ignoredEllipsoidIso,
                               translatedMesh,
                               false,
                               true,
                               1,
                               1u)) {
        return false;
    }

    GmIso4 ignoredMeshIso = IdentityAt({7.0f, 8.0f, 9.0f});
    ignoredMeshIso.rotation.SetRotateQuarterY(2u);
    return RunDirectDifferential("mesh-location-disabled",
                                 branchEllipsoid,
                                 unit,
                                 IdentityAt({0.25f, 0.25f, 0.15f}),
                                 ignoredMeshIso,
                                 true,
                                 false,
                                 1,
                                 1u);
}

bool RunDeterministicFuzz(const GmSurfMesh &mesh) {
    XorShift32 random;
    for (u32 caseIndex = 0u; caseIndex < 2048u; ++caseIndex) {
        GmSurfEllipsoid ellipsoid;
        ellipsoid.material = GmLocalMaterialIndex::FromIndex(
                static_cast<std::uint16_t>(random.Next() % 64u));
        ellipsoid.radii = {
            static_cast<float>(8u + random.Next() % 121u) / 64.0f,
            static_cast<float>(8u + random.Next() % 121u) / 64.0f,
            static_cast<float>(8u + random.Next() % 121u) / 64.0f,
        };

        GmIso4 meshIso = IdentityAt({
            static_cast<float>(static_cast<int>(random.Next() % 17u) - 8) /
                    4.0f,
            static_cast<float>(static_cast<int>(random.Next() % 17u) - 8) /
                    4.0f,
            static_cast<float>(static_cast<int>(random.Next() % 17u) - 8) /
                    4.0f,
        });
        meshIso.rotation.RotateX(
                static_cast<float>(static_cast<int>(random.Next() % 101u) -
                                   50) /
                100.0f);
        meshIso.rotation.RotateY(
                static_cast<float>(static_cast<int>(random.Next() % 101u) -
                                   50) /
                100.0f);
        meshIso.rotation.RotateZ(
                static_cast<float>(static_cast<int>(random.Next() % 101u) -
                                   50) /
                100.0f);

        const GmVec3 localCenter = {
            static_cast<float>(static_cast<int>(random.Next() % 1537u) -
                               128) /
                    128.0f,
            static_cast<float>(static_cast<int>(random.Next() % 513u) -
                               128) /
                    128.0f,
            static_cast<float>(static_cast<int>(random.Next() % 513u) -
                               256) /
                    128.0f,
        };
        GmVec3 worldCenter = localCenter;
        worldCenter.Mult(meshIso);
        GmIso4 ellipsoidIso = IdentityAt(worldCenter);
        ellipsoidIso.rotation.RotateX(
                static_cast<float>(static_cast<int>(random.Next() % 101u) -
                                   50) /
                100.0f);
        ellipsoidIso.rotation.RotateY(
                static_cast<float>(static_cast<int>(random.Next() % 101u) -
                                   50) /
                100.0f);
        ellipsoidIso.rotation.RotateZ(
                static_cast<float>(static_cast<int>(random.Next() % 101u) -
                                   50) /
                100.0f);

        char caseName[32];
        std::snprintf(caseName, sizeof(caseName), "fuzz-%04u", caseIndex);
        if (!RunDirectDifferential(
                    caseName, ellipsoid, mesh, ellipsoidIso, meshIso)) {
            return false;
        }
    }
    return true;
}

class TrackingEllipsoid final : public GmSurfEllipsoid {
public:
    std::size_t CollisionCallCount(void) const {
        return collisionCallCount_;
    }

protected:
    int Collide(const LocatedGmSurf &self,
                const LocatedGmSurf &other,
                CGmCollisionBuffer &collisionBuffer) const override {
        ++collisionCallCount_;
        return GmSurfEllipsoid::Collide(self, other, collisionBuffer);
    }

private:
    mutable std::size_t collisionCallCount_ = 0u;
};

class DerivedMesh final : public GmSurfMesh {};

class TrackingMesh final : public GmSurfMesh {
public:
    std::size_t CollisionCallCount(void) const {
        return collisionCallCount_;
    }

protected:
    int Collide(const LocatedGmSurf &self,
                const LocatedGmSurf &other,
                CGmCollisionBuffer &collisionBuffer) const override {
        ++collisionCallCount_;
        return GmSurfMesh::Collide(self, other, collisionBuffer);
    }

private:
    mutable std::size_t collisionCallCount_ = 0u;
};

bool ConfigurePlugSurface(CPlugSurface *surface,
                          std::unique_ptr<GmSurf> geometry,
                          u32 materialCount,
                          u32 selectedLocalMaterial,
                          EPlugSurfaceMaterialId selectedSurfaceMaterial) {
    if (geometry != nullptr) {
        CMwNodRef<CPlugSurfaceGeom> geometryNode =
                MakeMwNod<CPlugSurfaceGeom>();
        geometryNode->SetGmSurf(std::move(geometry));
        surface->SetGeometry(geometryNode.Get());
    }
    if (!surface->SetMaterialCount(materialCount)) {
        return false;
    }
    for (u32 materialIndex = 0u;
         materialIndex < materialCount;
         ++materialIndex) {
        CMwNodRef<CPlugMaterial> material = MakeMwNod<CPlugMaterial>();
        if (materialIndex == selectedLocalMaterial) {
            material->SetSurfaceMaterialId(selectedSurfaceMaterial);
        }
        if (!surface->SetMaterialAt(materialIndex, material.Get())) {
            return false;
        }
    }
    return true;
}

int RunPlugReference(const SPlugSurfaceLocatedPair &pair,
                     VectorCollisionBuffer *buffer) {
    buffer->Seed(SentinelCollision());
    return CPlugSurface::ComputeCollision(pair, *buffer);
}

int RunPlugOptimized(const SPlugSurfaceLocatedPair &pair,
                     VectorCollisionBuffer *buffer) {
    buffer->Seed(SentinelCollision());
    return CPlugSurface::ComputeCollisionOptimizedCpu(pair, *buffer);
}

int RunPlugNativeSelected(const SPlugSurfaceLocatedPair &pair,
                          VectorCollisionBuffer *buffer) {
    buffer->Seed(SentinelCollision());
    if (binary32MathPath == forevervalidator::simulation::
                                    OptimizedCpuBinary32MathPath::X86Sse2) {
        return ComputePlugSurfaceCollisionInlineMathOptimizedCpuNativeBinary32(
                pair, *buffer);
    }
    return CPlugSurface::ComputeCollisionOptimizedCpu(pair, *buffer);
}

bool CheckMappedMaterials(const char *caseName,
                          const VectorCollisionBuffer &buffer,
                          std::uint16_t localMaterialA,
                          std::uint16_t localMaterialB,
                          EPlugSurfaceMaterialId materialA,
                          EPlugSurfaceMaterialId materialB) {
    const std::vector<GmCollision> &collisions = buffer.Collisions();
    for (std::size_t collisionIndex = 1u;
         collisionIndex < collisions.size();
         ++collisionIndex) {
        const GmCollision &collision = collisions[collisionIndex];
        if (collision.localMaterialA.Index() != localMaterialA ||
            collision.localMaterialB.Index() != localMaterialB ||
            collision.materialA != materialA ||
            collision.materialB != materialB) {
            std::fprintf(stderr,
                         "%s mapped material differs at collision %zu\n",
                         caseName,
                         collisionIndex);
            return false;
        }
    }
    return true;
}

bool RunPlugDifferential(const char *caseName,
                         CPlugSurface &surfaceA,
                         const GmIso4 &isoA,
                         CPlugSurface &surfaceB,
                         const GmIso4 &isoB,
                         int expectedHit,
                         std::size_t minimumNewCollisions,
                         bool checkMaterials = false,
                         std::uint16_t localMaterialA = 0u,
                         std::uint16_t localMaterialB = 0u,
                         EPlugSurfaceMaterialId materialA =
                                 EPlugSurfaceMaterialId_Concrete,
                         EPlugSurfaceMaterialId materialB =
                                 EPlugSurfaceMaterialId_Concrete) {
    const SPlugSurfaceLocatedPair pair(surfaceA, isoA, surfaceB, isoB);
    VectorCollisionBuffer referenceFirst;
    VectorCollisionBuffer optimizedSecond;
    VectorCollisionBuffer nativeThird;
    const int referenceFirstHit = RunPlugReference(pair, &referenceFirst);
    const int optimizedSecondHit = RunPlugOptimized(pair, &optimizedSecond);
    const int nativeThirdHit = RunPlugNativeSelected(pair, &nativeThird);
    if (!CheckExpected(caseName,
                       expectedHit,
                       minimumNewCollisions,
                       referenceFirstHit,
                       referenceFirst) ||
        referenceFirstHit != optimizedSecondHit ||
        optimizedSecondHit != nativeThirdHit ||
        !CompareBuffers(caseName,
                        "plug-reference-first",
                        referenceFirst,
                        optimizedSecond) ||
        !CompareBuffers(caseName,
                        "plug-e005-before-native",
                        optimizedSecond,
                        nativeThird) ||
        (checkMaterials &&
         !CheckMappedMaterials(caseName,
                               referenceFirst,
                               localMaterialA,
                               localMaterialB,
                               materialA,
                               materialB))) {
        return false;
    }

    VectorCollisionBuffer nativeFirst;
    VectorCollisionBuffer optimizedSecondPass;
    VectorCollisionBuffer referenceSecond;
    const int nativeFirstHit = RunPlugNativeSelected(pair, &nativeFirst);
    const int optimizedSecondPassHit =
            RunPlugOptimized(pair, &optimizedSecondPass);
    const int referenceSecondHit = RunPlugReference(pair, &referenceSecond);
    if (nativeFirstHit != optimizedSecondPassHit ||
        optimizedSecondPassHit != referenceSecondHit ||
        referenceFirstHit != referenceSecondHit ||
        !CompareBuffers(caseName,
                        "plug-native-before-e005",
                        optimizedSecondPass,
                        nativeFirst) ||
        !CompareBuffers(caseName,
                        "plug-e005-repeat",
                        optimizedSecond,
                        optimizedSecondPass) ||
        !CompareBuffers(caseName,
                        "plug-reference-repeat",
                        referenceFirst,
                        referenceSecond)) {
        return false;
    }
    ++completedPlugCaseCount;
    return true;
}

bool RunPlugRoutingCases(void) {
    const GmIso4 identity = IdentityAt({0.0f, 0.0f, 0.0f});
    const GmIso4 ellipsoidHit = IdentityAt({0.25f, 0.25f, 0.2f});

    {
        CPlugSurface ellipsoidSurface;
        CPlugSurface meshSurface;
        auto ellipsoid = std::make_unique<GmSurfEllipsoid>();
        ellipsoid->material = GmLocalMaterialIndex::FromIndex(5u);
        ellipsoid->radii = {0.8f, 0.5f, 0.4f};
        auto mesh = std::make_unique<GmSurfMesh>();
        if (!BuildUnitMesh(mesh.get()) ||
            !ConfigurePlugSurface(&ellipsoidSurface,
                                  std::move(ellipsoid),
                                  6u,
                                  5u,
                                  EPlugSurfaceMaterialId_Turbo2) ||
            !ConfigurePlugSurface(&meshSurface,
                                  std::move(mesh),
                                  8u,
                                  7u,
                                  EPlugSurfaceMaterialId_Snow) ||
            !RunPlugDifferential("exact-ellipsoid-mesh-material-remap",
                                 ellipsoidSurface,
                                 ellipsoidHit,
                                 meshSurface,
                                 identity,
                                 1,
                                 1u,
                                 true,
                                 5u,
                                 7u,
                                 EPlugSurfaceMaterialId_Turbo2,
                                 EPlugSurfaceMaterialId_Snow)) {
            return false;
        }
    }

    {
        CPlugSurface ellipsoidSurface;
        CPlugSurface meshSurface;
        auto ellipsoid = std::make_unique<TrackingEllipsoid>();
        TrackingEllipsoid *trackingEllipsoid = ellipsoid.get();
        ellipsoid->material = GmLocalMaterialIndex::FromIndex(5u);
        ellipsoid->radii = {0.8f, 0.5f, 0.4f};
        auto mesh = std::make_unique<GmSurfMesh>();
        if (!BuildUnitMesh(mesh.get()) ||
            !ConfigurePlugSurface(&ellipsoidSurface,
                                  std::move(ellipsoid),
                                  6u,
                                  5u,
                                  EPlugSurfaceMaterialId_Ice) ||
            !ConfigurePlugSurface(&meshSurface,
                                  std::move(mesh),
                                  8u,
                                  7u,
                                  EPlugSurfaceMaterialId_Wood) ||
            !RunPlugDifferential("derived-ellipsoid-fallback",
                                 ellipsoidSurface,
                                 ellipsoidHit,
                                 meshSurface,
                                 identity,
                                 1,
                                 1u) ||
            trackingEllipsoid->CollisionCallCount() != 6u) {
            std::fprintf(stderr,
                         "derived-ellipsoid-fallback generic calls=%zu\n",
                         trackingEllipsoid->CollisionCallCount());
            return false;
        }
    }

    {
        CPlugSurface ellipsoidSurface;
        CPlugSurface meshSurface;
        auto ellipsoid = std::make_unique<GmSurfEllipsoid>();
        ellipsoid->material = GmLocalMaterialIndex::FromIndex(5u);
        ellipsoid->radii = {0.8f, 0.5f, 0.4f};
        auto mesh = std::make_unique<DerivedMesh>();
        if (!BuildUnitMesh(mesh.get()) ||
            !ConfigurePlugSurface(&ellipsoidSurface,
                                  std::move(ellipsoid),
                                  6u,
                                  5u,
                                  EPlugSurfaceMaterialId_Rubber) ||
            !ConfigurePlugSurface(&meshSurface,
                                  std::move(mesh),
                                  8u,
                                  7u,
                                  EPlugSurfaceMaterialId_Grass)) {
            return false;
        }
        const std::size_t callsBefore = referenceEllipsoidMeshCallCount;
        if (!RunPlugDifferential("derived-mesh-fallback",
                                 ellipsoidSurface,
                                 ellipsoidHit,
                                 meshSurface,
                                 identity,
                                 1,
                                 1u)
#if defined(FOREVERVALIDATOR_HAS_GNU_LD_WRAP)
            ||
            referenceEllipsoidMeshCallCount - callsBefore != 6u) {
#else
            ) {
#endif
            std::fprintf(stderr,
                         "derived-mesh-fallback reference calls=%zu\n",
                         referenceEllipsoidMeshCallCount - callsBefore);
            return false;
        }
    }

    {
        CPlugSurface meshSurface;
        CPlugSurface ellipsoidSurface;
        auto mesh = std::make_unique<TrackingMesh>();
        TrackingMesh *trackingMesh = mesh.get();
        auto ellipsoid = std::make_unique<GmSurfEllipsoid>();
        ellipsoid->material = GmLocalMaterialIndex::FromIndex(5u);
        ellipsoid->radii = {0.8f, 0.5f, 0.4f};
        if (!BuildUnitMesh(mesh.get()) ||
            !ConfigurePlugSurface(&meshSurface,
                                  std::move(mesh),
                                  8u,
                                  7u,
                                  EPlugSurfaceMaterialId_Danger) ||
            !ConfigurePlugSurface(&ellipsoidSurface,
                                  std::move(ellipsoid),
                                  6u,
                                  5u,
                                  EPlugSurfaceMaterialId_Pavement) ||
            !RunPlugDifferential("reversed-pair-fallback",
                                 meshSurface,
                                 identity,
                                 ellipsoidSurface,
                                 ellipsoidHit,
                                 1,
                                 1u) ||
            trackingMesh->CollisionCallCount() != 6u) {
            std::fprintf(stderr,
                         "reversed-pair-fallback generic calls=%zu\n",
                         trackingMesh->CollisionCallCount());
            return false;
        }
    }

    {
        CPlugSurface nullSurface;
        CPlugSurface ellipsoidSurface;
        CPlugSurface meshSurface;
        auto ellipsoid = std::make_unique<GmSurfEllipsoid>();
        ellipsoid->material = GmLocalMaterialIndex::FromIndex(5u);
        ellipsoid->radii = {0.8f, 0.5f, 0.4f};
        auto mesh = std::make_unique<GmSurfMesh>();
        if (!BuildUnitMesh(mesh.get()) ||
            !ConfigurePlugSurface(&nullSurface,
                                  nullptr,
                                  0u,
                                  0u,
                                  EPlugSurfaceMaterialId_Concrete) ||
            !ConfigurePlugSurface(&ellipsoidSurface,
                                  std::move(ellipsoid),
                                  6u,
                                  5u,
                                  EPlugSurfaceMaterialId_Ice) ||
            !ConfigurePlugSurface(&meshSurface,
                                  std::move(mesh),
                                  8u,
                                  7u,
                                  EPlugSurfaceMaterialId_Wood) ||
            !RunPlugDifferential("null-first-fallback",
                                 nullSurface,
                                 identity,
                                 meshSurface,
                                 identity,
                                 0,
                                 0u) ||
            !RunPlugDifferential("null-second-fallback",
                                 ellipsoidSurface,
                                 ellipsoidHit,
                                 nullSurface,
                                 identity,
                                 0,
                                 0u)) {
            return false;
        }
    }

    return true;
}

bool RunMxcsrFallbackCases(void) {
#if defined(__i386__) || defined(__x86_64__)
    if (binary32MathPath != forevervalidator::simulation::
                                    OptimizedCpuBinary32MathPath::X86Sse2) {
        return true;
    }
    const unsigned int base = _mm_getcsr();

    _mm_setcsr(base | 0x3fu);
    ++completedMxcsrCaseCount;
    if (forevervalidator::simulation::
                SelectOptimizedCpuBinary32MathPathForActiveExecution() !=
        binary32MathPath) {
        std::fprintf(stderr, "sticky MXCSR flags disabled native path\n");
        _mm_setcsr(base);
        return false;
    }
    _mm_setcsr(base);

    const unsigned int incompatible[] = {
        base | 0x0040u,
        base & ~0x0080u,
        base | 0x2000u,
        base | 0x4000u,
        base | 0x6000u,
        base | 0x8000u,
    };
    for (unsigned int value : incompatible) {
        _mm_setcsr(value);
        ++completedMxcsrCaseCount;
        if (forevervalidator::simulation::
                    SelectOptimizedCpuBinary32MathPathForActiveExecution() !=
            forevervalidator::simulation::
                    OptimizedCpuBinary32MathPath::Reference) {
            std::fprintf(stderr,
                         "incompatible MXCSR selected native path: %08x\n",
                         value);
            _mm_setcsr(base);
            return false;
        }
        _mm_setcsr(base);
    }

    if (std::fesetround(FE_DOWNWARD) != 0) {
        std::fprintf(stderr, "could not establish FE_DOWNWARD\n");
        return false;
    }
    ++completedMxcsrCaseCount;
    if (forevervalidator::simulation::
                SelectOptimizedCpuBinary32MathPathForActiveExecution() !=
        forevervalidator::simulation::
                OptimizedCpuBinary32MathPath::Reference) {
        std::fprintf(stderr, "non-RNE fenv selected native path\n");
        std::fesetround(FE_TONEAREST);
        _mm_setcsr(base);
        return false;
    }
    if (std::fesetround(FE_TONEAREST) != 0) {
        std::fprintf(stderr, "could not restore FE_TONEAREST\n");
        _mm_setcsr(base);
        return false;
    }
    _mm_setcsr(base);
#endif
    return true;
}

}  // namespace

int main(void) {
    if (forevervalidator::simulation::
                SelectOptimizedCpuBinary32MathPathForActiveExecution() !=
        forevervalidator::simulation::
                OptimizedCpuBinary32MathPath::Reference) {
        std::fprintf(stderr, "native path selected outside deterministic scope\n");
        return 1;
    }
    tmnf::simulation::DeterministicExecutionScope deterministicScope;
    if (!deterministicScope.Established()) {
        std::fprintf(stderr, "could not establish deterministic execution\n");
        return 1;
    }
    binary32MathPath = forevervalidator::simulation::
            SelectOptimizedCpuBinary32MathPathForActiveExecution();
    if (!RunMxcsrFallbackCases()) {
        return 1;
    }

    GmSurfMesh unit;
    GmSurfMesh separated;
    GmSurfMesh overlapping;
    GmSurfMesh degenerate;
    if (!BuildUnitMesh(&unit) ||
        !BuildSeparatedBranchMesh(&separated) ||
        !BuildOverlappingOrderMesh(&overlapping) ||
        !BuildDegenerateMesh(&degenerate)) {
        std::fprintf(stderr, "could not construct differential meshes\n");
        return 1;
    }
    if (!RunCoreCases(unit, separated, overlapping, degenerate) ||
        !RunDeterministicFuzz(separated) ||
        !RunPlugRoutingCases()) {
        return 1;
    }
    std::printf("ellipsoid_mesh_cases=%zu result=identical\n",
                completedDirectCaseCount);
    std::printf("plug_surface_cases=%zu result=identical\n",
                completedPlugCaseCount);
    std::printf("binary32_path=%s mxcsr_cases=%zu result=valid\n",
                binary32MathPath == forevervalidator::simulation::
                                            OptimizedCpuBinary32MathPath::X86Sse2
                        ? "x86_sse2"
                        : "reference",
                completedMxcsrCaseCount);
    return 0;
}
