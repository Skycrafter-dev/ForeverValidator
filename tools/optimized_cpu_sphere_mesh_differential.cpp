#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "engine/core/binary32_math.h"
#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/collision/hms_collision.h"
#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/geometry/gm_surface.h"
#include "engine/physics/geometry/gmsurf_collision.h"
#include "engine/physics/geometry/plug_surface.h"
#include "engine/rendering/plug_material.h"
#include "engine/rendering/plug_tree.h"
#include "simulation/runtime/replay_deterministic_execution.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_binary32_math.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_native_binary32_collision.h"

#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace {

std::size_t completedCaseCount = 0u;
std::size_t completedPlugSurfaceCaseCount = 0u;
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
    std::uint32_t state = 0x6d2b79f5u;

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
    if (!CompareFloat(caseName, passName, collisionIndex, component,
                      reference.x, optimized.x)) {
        return false;
    }
    std::snprintf(component, sizeof(component), "%s.y", field);
    if (!CompareFloat(caseName, passName, collisionIndex, component,
                      reference.y, optimized.y)) {
        return false;
    }
    std::snprintf(component, sizeof(component), "%s.z", field);
    return CompareFloat(caseName, passName, collisionIndex, component,
                        reference.z, optimized.z);
}

bool CompareCollision(const char *caseName,
                      const char *passName,
                      std::size_t collisionIndex,
                      const GmCollision &reference,
                      const GmCollision &optimized) {
    if (!CompareVector(caseName, passName, collisionIndex, "separation",
                       reference.separation, optimized.separation) ||
        !CompareVector(caseName, passName, collisionIndex, "impulseNormal",
                       reference.impulseNormal, optimized.impulseNormal) ||
        !CompareVector(caseName, passName, collisionIndex, "contactPoint",
                       reference.contactPoint, optimized.contactPoint) ||
        !CompareVector(caseName, passName, collisionIndex, "extraNegated",
                       reference.extraNegated, optimized.extraNegated)) {
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
    for (std::size_t index = 0u; index < referenceCollisions.size(); ++index) {
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

int RunReference(const LocatedGmSurf &sphere,
                 const LocatedGmSurf &mesh,
                 VectorCollisionBuffer *buffer) {
    buffer->Seed(SentinelCollision());
    return GmCollision_Sphere_Mesh(sphere, mesh, *buffer);
}

int RunOptimized(const LocatedGmSurf &sphere,
                 const LocatedGmSurf &mesh,
                 VectorCollisionBuffer *buffer) {
    buffer->Seed(SentinelCollision());
    return GmCollision_Sphere_Mesh_OptimizedCpu(sphere, mesh, *buffer);
}

int RunNativeSelected(const LocatedGmSurf &sphere,
                      const LocatedGmSurf &mesh,
                      VectorCollisionBuffer *buffer) {
    buffer->Seed(SentinelCollision());
    if (binary32MathPath == forevervalidator::simulation::
                                    OptimizedCpuBinary32MathPath::X86Sse2) {
        return GmCollision_Sphere_Mesh_InlineMathOptimizedCpuNativeBinary32(
                sphere, mesh, *buffer);
    }
    return GmCollision_Sphere_Mesh_OptimizedCpu(sphere, mesh, *buffer);
}

bool CheckExpected(const char *caseName,
                   int expectedHit,
                   std::size_t minimumNewCollisions,
                   int referenceHit,
                   const VectorCollisionBuffer &reference) {
    const std::size_t collisionCount = reference.Collisions().size();
    const std::size_t newCollisionCount = collisionCount == 0u
            ? 0u
            : collisionCount - 1u;
    if (expectedHit >= 0 && (referenceHit != 0) != (expectedHit != 0)) {
        std::fprintf(stderr,
                     "%s did not exercise the expected Reference result: "
                     "expected_hit=%d actual_hit=%d\n",
                     caseName,
                     expectedHit,
                     referenceHit);
        return false;
    }
    if (newCollisionCount < minimumNewCollisions) {
        std::fprintf(stderr,
                     "%s emitted too few Reference collisions: expected_at_least=%zu "
                     "actual=%zu\n",
                     caseName,
                     minimumNewCollisions,
                     newCollisionCount);
        return false;
    }
    if ((referenceHit != 0) != (newCollisionCount != 0u)) {
        std::fprintf(stderr,
                     "%s Reference hit/count contract differs: hit=%d new=%zu\n",
                     caseName,
                     referenceHit,
                     newCollisionCount);
        return false;
    }
    return true;
}

bool RunDifferential(const char *caseName,
                     const GmSurfSphere &sphere,
                     const GmSurfMesh &mesh,
                     const GmIso4 &sphereIso,
                     const GmIso4 &meshIso,
                     bool sphereEnabled = true,
                     bool meshEnabled = true,
                     int expectedHit = -1,
                     std::size_t minimumNewCollisions = 0u) {
    const LocatedGmSurf locatedSphere = {
        &sphere,
        &sphereIso,
        sphereEnabled,
    };
    const LocatedGmSurf locatedMesh = {
        &mesh,
        &meshIso,
        meshEnabled,
    };

    VectorCollisionBuffer referenceFirst;
    VectorCollisionBuffer optimizedSecond;
    VectorCollisionBuffer nativeThird;
    const int referenceFirstHit =
            RunReference(locatedSphere, locatedMesh, &referenceFirst);
    const int optimizedSecondHit =
            RunOptimized(locatedSphere, locatedMesh, &optimizedSecond);
    const int nativeThirdHit =
            RunNativeSelected(locatedSphere, locatedMesh, &nativeThird);
    if (!CheckExpected(caseName,
                       expectedHit,
                       minimumNewCollisions,
                       referenceFirstHit,
                       referenceFirst) ||
        referenceFirstHit != optimizedSecondHit ||
        optimizedSecondHit != nativeThirdHit ||
        !CompareBuffers(caseName,
                        "reference-first",
                        referenceFirst,
                        optimizedSecond) ||
        !CompareBuffers(caseName,
                        "e005-before-native",
                        optimizedSecond,
                        nativeThird)) {
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
    VectorCollisionBuffer optimizedSecondPass;
    VectorCollisionBuffer referenceSecond;
    const int nativeFirstHit =
            RunNativeSelected(locatedSphere, locatedMesh, &nativeFirst);
    const int optimizedSecondPassHit =
            RunOptimized(locatedSphere, locatedMesh, &optimizedSecondPass);
    const int referenceSecondHit =
            RunReference(locatedSphere, locatedMesh, &referenceSecond);
    if (nativeFirstHit != optimizedSecondPassHit ||
        optimizedSecondPassHit != referenceSecondHit ||
        referenceFirstHit != referenceSecondHit ||
        !CompareBuffers(caseName,
                        "native-before-e005",
                        optimizedSecondPass,
                        nativeFirst) ||
        !CompareBuffers(caseName,
                        "e005-repeat",
                        optimizedSecond,
                        optimizedSecondPass) ||
        !CompareBuffers(caseName,
                        "reference-repeat",
                        referenceFirst,
                        referenceSecond)) {
        if (nativeFirstHit != optimizedSecondPassHit ||
            optimizedSecondPassHit != referenceSecondHit ||
            referenceFirstHit != referenceSecondHit) {
            std::fprintf(stderr,
                         "%s/native-first hit differs: native=%d "
                         "optimized=%d reference=%d repeated_reference=%d\n",
                         caseName,
                         nativeFirstHit,
                         optimizedSecondPassHit,
                         referenceSecondHit,
                         referenceFirstHit);
        }
        return false;
    }
    ++completedCaseCount;
    return true;
}

GmSurfMeshTriangle Triangle(u32 a, u32 b, u32 c, std::uint16_t material) {
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

GmIso4 IdentityAt(const GmVec3 &translation) {
    GmIso4 result;
    result.SetIdentity();
    result.translation = translation;
    return result;
}

bool RunCoreCases(const GmSurfMesh &unit,
                  const GmSurfMesh &separated,
                  const GmSurfMesh &overlapping) {
    GmSurfSphere sphere;
    sphere.material = GmLocalMaterialIndex::FromIndex(5u);
    sphere.radius = 1.0f;
    const GmIso4 identity = IdentityAt({0.0f, 0.0f, 0.0f});

    if (!RunDifferential("face-hit",
                         sphere,
                         unit,
                         IdentityAt({0.25f, 0.25f, 0.5f}),
                         identity,
                         true,
                         true,
                         1,
                         1u) ||
        !RunDifferential("behind-plane",
                         sphere,
                         unit,
                         IdentityAt({0.25f, 0.25f, -0.25f}),
                         identity,
                         true,
                         true,
                         0) ||
        !RunDifferential("outside-radius",
                         sphere,
                         unit,
                         IdentityAt({0.25f, 0.25f,
                                     std::nextafter(
                                             1.0f,
                                             std::numeric_limits<float>::
                                                     infinity())}),
                         identity,
                         true,
                         true,
                         0)) {
        return false;
    }

    const float planeDistance = 0.25f;
    const float edgeReach = CIsqrt(
            sphere.radius * sphere.radius -
            planeDistance * planeDistance);
    const float belowReach = std::nextafter(
            edgeReach, -std::numeric_limits<float>::infinity());
    const float aboveReach = std::nextafter(
            edgeReach, std::numeric_limits<float>::infinity());
    if (!RunDifferential("edge-inside-ulp",
                         sphere,
                         unit,
                         IdentityAt({0.5f, -belowReach, planeDistance}),
                         identity,
                         true,
                         true,
                         1,
                         1u) ||
        !RunDifferential("edge-exact",
                         sphere,
                         unit,
                         IdentityAt({0.5f, -edgeReach, planeDistance}),
                         identity) ||
        !RunDifferential("edge-outside-ulp",
                         sphere,
                         unit,
                         IdentityAt({0.5f, -aboveReach, planeDistance}),
                         identity,
                         true,
                         true,
                         0)) {
        return false;
    }

    GmSurfSphere endpointSphere = sphere;
    endpointSphere.radius = 0.5f;
    if (!RunDifferential("endpoint-a",
                         endpointSphere,
                         unit,
                         IdentityAt({-0.1f, -0.1f, 0.2f}),
                         identity,
                         true,
                         true,
                         1,
                         1u) ||
        !RunDifferential("endpoint-b",
                         endpointSphere,
                         unit,
                         IdentityAt({1.1f, -0.1f, 0.2f}),
                         identity,
                         true,
                         true,
                         1,
                         1u)) {
        return false;
    }

    GmSurfSphere branchSphere = sphere;
    branchSphere.radius = 0.5f;
    if (!RunDifferential("near-subtree",
                         branchSphere,
                         separated,
                         IdentityAt({0.25f, 0.25f, 0.25f}),
                         identity,
                         true,
                         true,
                         1,
                         1u) ||
        !RunDifferential("far-subtree",
                         branchSphere,
                         separated,
                         IdentityAt({10.25f, 0.25f, 0.25f}),
                         identity,
                         true,
                         true,
                         1,
                         1u) ||
        !RunDifferential("ordered-overlap",
                         branchSphere,
                         overlapping,
                         IdentityAt({0.25f, 0.25f, 0.25f}),
                         identity,
                         true,
                         true,
                         1,
                         2u)) {
        return false;
    }

    GmIso4 rotatedMesh;
    rotatedMesh.SetIdentity();
    rotatedMesh.rotation.SetRotateQuarterY(1u);
    rotatedMesh.translation = {3.0f, -2.0f, 5.0f};
    GmVec3 rotatedSphereCenter = {0.25f, 0.25f, 0.5f};
    rotatedSphereCenter.Mult(rotatedMesh);
    GmIso4 rotatedSphere = IdentityAt(rotatedSphereCenter);
    rotatedSphere.rotation.SetRotateQuarterY(3u);
    if (!RunDifferential("rotated-translated",
                         sphere,
                         unit,
                         rotatedSphere,
                         rotatedMesh,
                         true,
                         true,
                         1,
                         1u)) {
        return false;
    }

    GmSurfSphere boundarySphere = sphere;
    boundarySphere.radius = 0.25f;
    const float touchingX = 1.0f + boundarySphere.radius;
    if (!RunDifferential("bounds-touch",
                         boundarySphere,
                         unit,
                         IdentityAt({touchingX, 0.0f, 0.0f}),
                         identity) ||
        !RunDifferential("bounds-inside-ulp",
                         boundarySphere,
                         unit,
                         IdentityAt({std::nextafter(
                                             touchingX,
                                             -std::numeric_limits<float>::
                                                     infinity()),
                                     0.0f,
                                     0.0f}),
                         identity) ||
        !RunDifferential("bounds-outside-ulp",
                         boundarySphere,
                         unit,
                         IdentityAt({std::nextafter(
                                             touchingX,
                                             std::numeric_limits<float>::
                                                     infinity()),
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
    GmIso4 signedZeroSphere = IdentityAt({0.25f, 0.25f, 0.5f});
    signedZeroSphere.rotation.basisX.y = -0.0f;
    signedZeroSphere.rotation.basisY.z = -0.0f;
    signedZeroSphere.rotation.basisZ.x = -0.0f;
    if (!RunDifferential("signed-zero-transform",
                         sphere,
                         unit,
                         signedZeroSphere,
                         signedZeroMesh,
                         true,
                         true,
                         1,
                         1u)) {
        return false;
    }

    GmIso4 nanSphere = IdentityAt({0.25f, 0.25f, 0.5f});
    nanSphere.translation.x = std::numeric_limits<float>::quiet_NaN();
    if (!RunDifferential("nan-transform",
                         sphere,
                         unit,
                         nanSphere,
                         identity,
                         true,
                         true,
                         0)) {
        return false;
    }

    GmIso4 ignoredSphereIso = IdentityAt({100.0f, 100.0f, 100.0f});
    GmIso4 translatedMesh = IdentityAt({0.0f, 0.0f, -0.25f});
    if (!RunDifferential("sphere-location-disabled",
                         sphere,
                         unit,
                         ignoredSphereIso,
                         translatedMesh,
                         false,
                         true,
                         1,
                         1u)) {
        return false;
    }

    GmIso4 ignoredMeshIso = identity;
    ignoredMeshIso.rotation.SetRotateQuarterY(2u);
    ignoredMeshIso.translation = {7.0f, 8.0f, 9.0f};
    return RunDifferential("mesh-location-disabled",
                           sphere,
                           unit,
                           IdentityAt({0.25f, 0.25f, 0.5f}),
                           ignoredMeshIso,
                           true,
                           false,
                           1,
                           1u);
}

bool RunDeterministicFuzz(const GmSurfMesh &mesh) {
    XorShift32 random;
    for (u32 caseIndex = 0u; caseIndex < 2048u; ++caseIndex) {
        GmSurfSphere sphere;
        sphere.material = GmLocalMaterialIndex::FromIndex(
                static_cast<std::uint16_t>(random.Next() % 64u));
        sphere.radius = static_cast<float>(1u + random.Next() % 128u) /
                64.0f;

        const GmVec3 localCenter = {
            static_cast<float>(static_cast<int>(random.Next() % 1665u) -
                               128) /
                    128.0f,
            static_cast<float>(static_cast<int>(random.Next() % 513u) -
                               128) /
                    128.0f,
            static_cast<float>(static_cast<int>(random.Next() % 513u) -
                               256) /
                    128.0f,
        };

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

        GmVec3 worldCenter = localCenter;
        worldCenter.Mult(meshIso);
        GmIso4 sphereIso = IdentityAt(worldCenter);
        sphereIso.rotation.SetRotateQuarterY(random.Next() & 3u);

        char caseName[32];
        std::snprintf(caseName, sizeof(caseName), "fuzz-%04u", caseIndex);
        if (!RunDifferential(
                    caseName, sphere, mesh, sphereIso, meshIso)) {
            return false;
        }
    }
    return true;
}

struct ExpectedPlugCollisionMetadata {
    bool enabled = false;
    std::uint16_t localMaterialA = 0u;
    std::uint16_t localMaterialB = 0u;
    EPlugSurfaceMaterialId materialA = EPlugSurfaceMaterialId_Concrete;
    EPlugSurfaceMaterialId materialB = EPlugSurfaceMaterialId_Concrete;
};

class TrackingSphere final : public GmSurfSphere {
public:
    std::size_t CollisionCallCount(void) const {
        return collisionCallCount_;
    }

protected:
    int Collide(const LocatedGmSurf &self,
                const LocatedGmSurf &other,
                CGmCollisionBuffer &collisionBuffer) const override {
        ++collisionCallCount_;
        return GmSurfSphere::Collide(self, other, collisionBuffer);
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

bool CheckPlugMetadata(const char *caseName,
                       const VectorCollisionBuffer &buffer,
                       const ExpectedPlugCollisionMetadata &expected) {
    if (!expected.enabled) {
        return true;
    }
    const std::vector<GmCollision> &collisions = buffer.Collisions();
    for (std::size_t collisionIndex = 1u;
         collisionIndex < collisions.size();
         ++collisionIndex) {
        const GmCollision &collision = collisions[collisionIndex];
        if (collision.localMaterialA.Index() != expected.localMaterialA ||
            collision.localMaterialB.Index() != expected.localMaterialB ||
            collision.materialA != expected.materialA ||
            collision.materialB != expected.materialB) {
            std::fprintf(
                    stderr,
                    "%s mapped metadata differs at collision %zu: "
                    "localA=%u/%u localB=%u/%u materialA=%u/%u "
                    "materialB=%u/%u\n",
                    caseName,
                    collisionIndex,
                    collision.localMaterialA.Index(),
                    expected.localMaterialA,
                    collision.localMaterialB.Index(),
                    expected.localMaterialB,
                    static_cast<unsigned>(collision.materialA),
                    static_cast<unsigned>(expected.materialA),
                    static_cast<unsigned>(collision.materialB),
                    static_cast<unsigned>(expected.materialB));
            return false;
        }
    }
    return true;
}

bool RunPlugDifferential(
        const char *caseName,
        CPlugSurface &surfaceA,
        const GmIso4 &isoA,
        CPlugSurface &surfaceB,
        const GmIso4 &isoB,
        int expectedHit,
        std::size_t minimumNewCollisions,
        const ExpectedPlugCollisionMetadata &expectedMetadata = {}) {
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
        !CheckPlugMetadata(caseName, referenceFirst, expectedMetadata)) {
        if (referenceFirstHit != optimizedSecondHit ||
            optimizedSecondHit != nativeThirdHit) {
            std::fprintf(stderr,
                         "%s/plug-reference-first hit differs: "
                         "reference=%d optimized=%d native=%d\n",
                         caseName,
                         referenceFirstHit,
                         optimizedSecondHit,
                         nativeThirdHit);
        }
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
                        referenceSecond) ||
        !CheckPlugMetadata(caseName, referenceSecond, expectedMetadata)) {
        if (nativeFirstHit != optimizedSecondPassHit ||
            optimizedSecondPassHit != referenceSecondHit ||
            referenceFirstHit != referenceSecondHit) {
            std::fprintf(stderr,
                         "%s/plug-native-first hit differs: native=%d "
                         "optimized=%d reference=%d repeated_reference=%d\n",
                         caseName,
                         nativeFirstHit,
                         optimizedSecondPassHit,
                         referenceSecondHit,
                         referenceFirstHit);
        }
        return false;
    }
    ++completedPlugSurfaceCaseCount;
    return true;
}

bool RunStaticContactTaggingCase(CPlugSurface &movingSurface,
                                 const GmIso4 &movingIso,
                                 CPlugSurface &staticSurface,
                                 const GmIso4 &staticIso) {
    const char *caseName = "static-contact-tagging";
    const SPlugSurfaceLocatedPair pair(
            movingSurface, movingIso, staticSurface, staticIso);
    CHmsCollisionBuffer reference;
    CHmsCollisionBuffer optimized;
    SHmsPhysicalCollision sentinel{};
    static_cast<GmCollision &>(sentinel) = SentinelCollision();
    reference.AddPhysicalCollision(sentinel);
    optimized.AddPhysicalCollision(sentinel);

    const int referenceHit = CPlugSurface::ComputeCollision(pair, reference);
    const int optimizedHit =
            CPlugSurface::ComputeCollisionOptimizedCpu(pair, optimized);
    if (referenceHit != 1 || optimizedHit != referenceHit ||
        reference.PhysicalCollisionCount() < 2u ||
        reference.PhysicalCollisionCount() !=
                optimized.PhysicalCollisionCount()) {
        std::fprintf(stderr,
                     "%s collision result differs: reference_hit=%d "
                     "optimized_hit=%d reference_count=%u "
                     "optimized_count=%u\n",
                     caseName,
                     referenceHit,
                     optimizedHit,
                     reference.PhysicalCollisionCount(),
                     optimized.PhysicalCollisionCount());
        return false;
    }
    for (u32 collisionIndex = 0u;
         collisionIndex < reference.PhysicalCollisionCount();
         ++collisionIndex) {
        const SHmsPhysicalCollision *referenceCollision =
                reference.PhysicalCollisionAtOrNull(collisionIndex);
        const SHmsPhysicalCollision *optimizedCollision =
                optimized.PhysicalCollisionAtOrNull(collisionIndex);
        if (referenceCollision == nullptr || optimizedCollision == nullptr ||
            !CompareCollision(caseName,
                              "before-static-tagging",
                              collisionIndex,
                              *referenceCollision,
                              *optimizedCollision)) {
            return false;
        }
    }

    CHmsCorpus movingCorpus;
    CHmsCorpus staticCorpus;
    CPlugTree movingTree;
    CPlugTree staticTree;
    const CHmsCollisionManager::SColOctreeCell staticRecord =
            CHmsCollisionManager::SColOctreeCell::Surface(
                    staticSurface.GeomBox(),
                    staticIso,
                    staticSurface,
                    staticTree,
                    staticCorpus);
    CHmsItem::SHmsCollisionGroupPair collisionGroupPair = {
        CHmsItem::ECollisionGroup_Dynamic,
        CHmsItem::ECollisionGroup_Static,
        true,
        true,
        false,
    };
    CHmsCollisionManager::SGroup::SAgainstGroup against;
    against.collisionGroupPair = &collisionGroupPair;
    CHmsCollisionManager::SZone zone(17u, nullptr);
    zone.SelectReplayStaticCollisionTarget(against);
    zone.BeginReplayStaticCollisionPass(&reference, &movingCorpus);
    zone.TagNewStaticCollisions(
            &reference, 1u, &movingTree, &staticRecord);
    zone.BeginReplayStaticCollisionPass(&optimized, &movingCorpus);
    zone.TagNewStaticCollisions(
            &optimized, 1u, &movingTree, &staticRecord);

    const SHmsPhysicalCollision *referenceSentinel =
            reference.PhysicalCollisionAtOrNull(0u);
    const SHmsPhysicalCollision *optimizedSentinel =
            optimized.PhysicalCollisionAtOrNull(0u);
    if (referenceSentinel == nullptr || optimizedSentinel == nullptr ||
        referenceSentinel->CorpusA() != nullptr ||
        referenceSentinel->CorpusB() != nullptr ||
        referenceSentinel->TreeA() != nullptr ||
        referenceSentinel->TreeB() != nullptr ||
        optimizedSentinel->CorpusA() != nullptr ||
        optimizedSentinel->CorpusB() != nullptr ||
        optimizedSentinel->TreeA() != nullptr ||
        optimizedSentinel->TreeB() != nullptr) {
        std::fprintf(stderr, "%s modified the prefix collision\n", caseName);
        return false;
    }

    for (u32 collisionIndex = 1u;
         collisionIndex < reference.PhysicalCollisionCount();
         ++collisionIndex) {
        const SHmsPhysicalCollision *referenceCollision =
                reference.PhysicalCollisionAtOrNull(collisionIndex);
        const SHmsPhysicalCollision *optimizedCollision =
                optimized.PhysicalCollisionAtOrNull(collisionIndex);
        if (referenceCollision == nullptr || optimizedCollision == nullptr ||
            referenceCollision->CorpusA() != &movingCorpus ||
            referenceCollision->CorpusB() != &staticCorpus ||
            referenceCollision->TreeA() != &movingTree ||
            referenceCollision->TreeB() != &staticTree ||
            &referenceCollision->CollisionGroupPair() != &collisionGroupPair ||
            optimizedCollision->CorpusA() != &movingCorpus ||
            optimizedCollision->CorpusB() != &staticCorpus ||
            optimizedCollision->TreeA() != &movingTree ||
            optimizedCollision->TreeB() != &staticTree ||
            &optimizedCollision->CollisionGroupPair() != &collisionGroupPair ||
            !CompareCollision(caseName,
                              "after-static-tagging",
                              collisionIndex,
                              *referenceCollision,
                              *optimizedCollision)) {
            std::fprintf(stderr,
                         "%s actor binding differs at collision %u\n",
                         caseName,
                         collisionIndex);
            return false;
        }
    }
    ++completedPlugSurfaceCaseCount;
    return true;
}

bool RunPlugSurfaceRoutingCases(void) {
    const GmIso4 identity = IdentityAt({0.0f, 0.0f, 0.0f});
    const GmIso4 sphereHit = IdentityAt({0.25f, 0.25f, 0.5f});

    {
        CPlugSurface sphereSurface;
        CPlugSurface meshSurface;
        auto sphere = std::make_unique<GmSurfSphere>();
        sphere->material = GmLocalMaterialIndex::FromIndex(5u);
        sphere->radius = 1.0f;
        auto mesh = std::make_unique<GmSurfMesh>();
        if (!BuildUnitMesh(mesh.get()) ||
            !ConfigurePlugSurface(&sphereSurface,
                                  std::move(sphere),
                                  6u,
                                  5u,
                                  EPlugSurfaceMaterialId_Turbo2) ||
            !ConfigurePlugSurface(&meshSurface,
                                  std::move(mesh),
                                  8u,
                                  7u,
                                  EPlugSurfaceMaterialId_Snow) ||
            !RunPlugDifferential(
                    "exact-sphere-mesh-material-remap",
                    sphereSurface,
                    sphereHit,
                    meshSurface,
                    identity,
                    1,
                    1u,
                    {true,
                     5u,
                     7u,
                     EPlugSurfaceMaterialId_Turbo2,
                     EPlugSurfaceMaterialId_Snow}) ||
            !RunStaticContactTaggingCase(
                    sphereSurface, sphereHit, meshSurface, identity)) {
            return false;
        }
    }

    {
        CPlugSurface sphereSurface;
        CPlugSurface meshSurface;
        auto sphere = std::make_unique<TrackingSphere>();
        TrackingSphere *trackingSphere = sphere.get();
        sphere->material = GmLocalMaterialIndex::FromIndex(5u);
        sphere->radius = 1.0f;
        auto mesh = std::make_unique<GmSurfMesh>();
        if (!BuildUnitMesh(mesh.get()) ||
            !ConfigurePlugSurface(&sphereSurface,
                                  std::move(sphere),
                                  6u,
                                  5u,
                                  EPlugSurfaceMaterialId_Ice) ||
            !ConfigurePlugSurface(&meshSurface,
                                  std::move(mesh),
                                  8u,
                                  7u,
                                  EPlugSurfaceMaterialId_Wood) ||
            !RunPlugDifferential(
                    "derived-sphere-fallback",
                    sphereSurface,
                    sphereHit,
                    meshSurface,
                    identity,
                    1,
                    1u,
                    {true,
                     5u,
                     7u,
                     EPlugSurfaceMaterialId_Ice,
                     EPlugSurfaceMaterialId_Wood}) ||
            trackingSphere->CollisionCallCount() != 6u) {
            std::fprintf(stderr,
                         "derived-sphere-fallback did not use generic "
                         "dispatch four times: calls=%zu\n",
                         trackingSphere->CollisionCallCount());
            return false;
        }
    }

    {
        CPlugSurface sphereSurface;
        CPlugSurface meshSurface;
        auto sphere = std::make_unique<GmSurfSphere>();
        sphere->material = GmLocalMaterialIndex::FromIndex(5u);
        sphere->radius = 1.0f;
        auto mesh = std::make_unique<DerivedMesh>();
        if (!BuildUnitMesh(mesh.get()) ||
            !ConfigurePlugSurface(&sphereSurface,
                                  std::move(sphere),
                                  6u,
                                  5u,
                                  EPlugSurfaceMaterialId_Rubber) ||
            !ConfigurePlugSurface(&meshSurface,
                                  std::move(mesh),
                                  8u,
                                  7u,
                                  EPlugSurfaceMaterialId_Grass) ||
            !RunPlugDifferential(
                    "derived-mesh-fallback",
                    sphereSurface,
                    sphereHit,
                    meshSurface,
                    identity,
                    1,
                    1u,
                    {true,
                     5u,
                     7u,
                     EPlugSurfaceMaterialId_Rubber,
                     EPlugSurfaceMaterialId_Grass})) {
            return false;
        }
    }

    {
        CPlugSurface meshSurface;
        CPlugSurface sphereSurface;
        auto mesh = std::make_unique<TrackingMesh>();
        TrackingMesh *trackingMesh = mesh.get();
        auto sphere = std::make_unique<GmSurfSphere>();
        sphere->material = GmLocalMaterialIndex::FromIndex(5u);
        sphere->radius = 1.0f;
        if (!BuildUnitMesh(mesh.get()) ||
            !ConfigurePlugSurface(&meshSurface,
                                  std::move(mesh),
                                  8u,
                                  7u,
                                  EPlugSurfaceMaterialId_Danger) ||
            !ConfigurePlugSurface(&sphereSurface,
                                  std::move(sphere),
                                  6u,
                                  5u,
                                  EPlugSurfaceMaterialId_Pavement) ||
            !RunPlugDifferential(
                    "reversed-derived-mesh-fallback",
                    meshSurface,
                    identity,
                    sphereSurface,
                    sphereHit,
                    1,
                    1u,
                    {true,
                     7u,
                     5u,
                     EPlugSurfaceMaterialId_Danger,
                     EPlugSurfaceMaterialId_Pavement}) ||
            trackingMesh->CollisionCallCount() != 6u) {
            std::fprintf(stderr,
                         "reversed-derived-mesh-fallback did not use generic "
                         "dispatch four times: calls=%zu\n",
                         trackingMesh->CollisionCallCount());
            return false;
        }
    }

    {
        CPlugSurface sphereSurface;
        CPlugSurface boxSurface;
        auto sphere = std::make_unique<TrackingSphere>();
        TrackingSphere *trackingSphere = sphere.get();
        sphere->material = GmLocalMaterialIndex::FromIndex(5u);
        sphere->radius = 1.0f;
        auto box = std::make_unique<GmSurfBox>();
        box->material = GmLocalMaterialIndex::FromIndex(9u);
        box->center = {0.0f, 0.0f, 0.0f};
        box->halfExtents = {1.0f, 1.0f, 1.0f};
        if (!ConfigurePlugSurface(&sphereSurface,
                                  std::move(sphere),
                                  6u,
                                  5u,
                                  EPlugSurfaceMaterialId_Metal) ||
            !ConfigurePlugSurface(&boxSurface,
                                  std::move(box),
                                  10u,
                                  9u,
                                  EPlugSurfaceMaterialId_Sand) ||
            !RunPlugDifferential(
                    "nonmatching-sphere-box-fallback",
                    sphereSurface,
                    IdentityAt({0.0f, 0.0f, 1.5f}),
                    boxSurface,
                    identity,
                    1,
                    1u,
                    {true,
                     5u,
                     9u,
                     EPlugSurfaceMaterialId_Metal,
                     EPlugSurfaceMaterialId_Sand}) ||
            trackingSphere->CollisionCallCount() != 6u) {
            std::fprintf(stderr,
                         "nonmatching-sphere-box-fallback did not use generic "
                         "dispatch four times: calls=%zu\n",
                         trackingSphere->CollisionCallCount());
            return false;
        }
    }

    {
        CPlugSurface nullSurface;
        CPlugSurface sphereSurface;
        CPlugSurface meshSurface;
        auto sphere = std::make_unique<GmSurfSphere>();
        sphere->material = GmLocalMaterialIndex::FromIndex(5u);
        sphere->radius = 1.0f;
        auto mesh = std::make_unique<GmSurfMesh>();
        if (!BuildUnitMesh(mesh.get()) ||
            !ConfigurePlugSurface(&nullSurface,
                                  nullptr,
                                  0u,
                                  0u,
                                  EPlugSurfaceMaterialId_Concrete) ||
            !ConfigurePlugSurface(&sphereSurface,
                                  std::move(sphere),
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
                                 sphereSurface,
                                 sphereHit,
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
    if (!BuildUnitMesh(&unit) ||
        !BuildSeparatedBranchMesh(&separated) ||
        !BuildOverlappingOrderMesh(&overlapping)) {
        std::fprintf(stderr, "could not construct differential meshes\n");
        return 1;
    }
    if (!RunCoreCases(unit, separated, overlapping) ||
        !RunDeterministicFuzz(separated) ||
        !RunPlugSurfaceRoutingCases()) {
        return 1;
    }
    std::printf("sphere_mesh_cases=%zu result=identical\n",
                completedCaseCount);
    std::printf("plug_surface_cases=%zu result=identical\n",
                completedPlugSurfaceCaseCount);
    std::printf("binary32_path=%s mxcsr_cases=%zu result=valid\n",
                binary32MathPath == forevervalidator::simulation::
                                            OptimizedCpuBinary32MathPath::X86Sse2
                        ? "x86_sse2"
                        : "reference",
                completedMxcsrCaseCount);
    return 0;
}
