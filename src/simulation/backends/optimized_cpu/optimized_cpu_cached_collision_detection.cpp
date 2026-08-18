// OptimizedCpu static collision traversal with immutable transform sidecars.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "engine/physics/collision/gm_collision_buffer.h"
#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_item.h"
#include "engine/physics/geometry/plug_surface.h"
#include "engine/rendering/plug_tree.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_surface_transform_cache.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_triangle_mesh_query.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_native_binary32_collision.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_ellipsoid_mesh_packet.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_bounds_overlap.h"

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

using forevervalidator::simulation::OptimizedCpuStaticBoundsOverlap;

namespace {

constexpr std::size_t EllipsoidPacketWidth = 8u;

struct EllipsoidPacketTraversalLane {
    CPlugTree *tree = nullptr;
    CPlugSurface *surface = nullptr;
    GmIso4 location{};
    GmBoxAligned bounds{};
    LocatedGmSurf located{};
    SHmsSphereBufferContact *sphereContact = nullptr;
    CHmsCollisionBuffer *buffer = nullptr;
    u32 temporalSlotOrdinal = 0u;

    EllipsoidPacketTraversalLane(
            CPlugTree *treeValue,
            CPlugSurface *surfaceValue,
            const GmIso4 &locationValue,
            const GmBoxAligned &boundsValue,
            u32 temporalSlotOrdinalValue) noexcept
        : tree(treeValue),
          surface(surfaceValue),
          location(locationValue),
          bounds(boundsValue),
          located{},
          sphereContact(nullptr),
          buffer(nullptr),
          temporalSlotOrdinal(temporalSlotOrdinalValue) {}
};

template <typename T, std::size_t Count>
class UninitializedObjectArray {
    static_assert(std::is_trivially_destructible_v<T>);

    struct alignas(T) Slot {
        std::byte bytes[sizeof(T)];
    };

    static_assert(sizeof(Slot) == sizeof(T));
    static_assert(alignof(Slot) == alignof(T));

public:
    UninitializedObjectArray(void) noexcept {}

    template <typename... Arguments>
    T &ConstructAt(std::size_t index, Arguments &&...arguments) noexcept {
        return *::new (static_cast<void *>(slots_[index].bytes)) T(
                std::forward<Arguments>(arguments)...);
    }

    T &operator[](std::size_t index) noexcept {
        return *std::launder(
                reinterpret_cast<T *>(slots_[index].bytes));
    }

    const T &operator[](std::size_t index) const noexcept {
        return *std::launder(
                reinterpret_cast<const T *>(slots_[index].bytes));
    }

private:
    std::array<Slot, Count> slots_;
};

using EllipsoidPacketTraversalLanes = UninitializedObjectArray<
        EllipsoidPacketTraversalLane,
        EllipsoidPacketWidth>;

#if defined(__i386__) || defined(__x86_64__)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2"), always_inline))
#endif
inline __m256 DirectLaneDot3(
        __m256 x,
        __m256 y,
        __m256 z,
        float coefficientX,
        float coefficientY,
        float coefficientZ) noexcept {
    const __m256 xy = _mm256_add_ps(
            _mm256_mul_ps(_mm256_set1_ps(coefficientX), x),
            _mm256_mul_ps(_mm256_set1_ps(coefficientY), y));
    return _mm256_add_ps(
            xy,
            _mm256_mul_ps(_mm256_set1_ps(coefficientZ), z));
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2"), always_inline))
#endif
inline __m256 DirectLaneAbsoluteBroadcast(float value) noexcept {
    return _mm256_and_ps(
            _mm256_set1_ps(value),
            _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2"), always_inline))
#endif
inline __m256 DirectLaneAbsoluteDot3(
        __m256 x,
        __m256 y,
        __m256 z,
        float coefficientX,
        float coefficientY,
        float coefficientZ) noexcept {
    const __m256 xy = _mm256_add_ps(
            _mm256_mul_ps(DirectLaneAbsoluteBroadcast(coefficientX), x),
            _mm256_mul_ps(DirectLaneAbsoluteBroadcast(coefficientY), y));
    return _mm256_add_ps(
            xy,
            _mm256_mul_ps(
                    DirectLaneAbsoluteBroadcast(coefficientZ), z));
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2"), always_inline))
#endif
inline __m256i DirectLaneBoundedValueMask(__m256 values) noexcept {
    const __m256i magnitude = _mm256_and_si256(
            _mm256_castps_si256(values),
            _mm256_set1_epi32(0x7fffffff));
    const __m256i exponent = _mm256_srli_epi32(magnitude, 23);
    const __m256i zero = _mm256_cmpeq_epi32(
            magnitude, _mm256_setzero_si256());
    const __m256i atLeastMinimum = _mm256_cmpgt_epi32(
            exponent, _mm256_set1_epi32(66));
    const __m256i atMostMaximum = _mm256_cmpgt_epi32(
            _mm256_set1_epi32(188), exponent);
    return _mm256_or_si256(
            zero,
            _mm256_and_si256(atLeastMinimum, atMostMaximum));
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2"), always_inline))
#endif
inline bool DirectLaneBoxValuesAreBounded(
        __m256 centerX,
        __m256 centerY,
        __m256 centerZ,
        __m256 halfX,
        __m256 halfY,
        __m256 halfZ) noexcept {
    __m256i bounded = DirectLaneBoundedValueMask(centerX);
    bounded = _mm256_and_si256(
            bounded, DirectLaneBoundedValueMask(centerY));
    bounded = _mm256_and_si256(
            bounded, DirectLaneBoundedValueMask(centerZ));
    bounded = _mm256_and_si256(
            bounded, DirectLaneBoundedValueMask(halfX));
    bounded = _mm256_and_si256(
            bounded, DirectLaneBoundedValueMask(halfY));
    bounded = _mm256_and_si256(
            bounded, DirectLaneBoundedValueMask(halfZ));
    return _mm256_movemask_epi8(bounded) == -1;
}
#endif

bool CollectEllipsoidPacketTraversalLanes(
        const GmIso4 &parentIso,
        const CPlugTree &tree,
        EllipsoidPacketTraversalLanes *lanes,
        std::size_t *laneCount,
        u32 *nextTemporalSlotOrdinal) {
    const u32 temporalSlotOrdinal = (*nextTemporalSlotOrdinal)++;
    if (!tree.HasWorldBox()) {
        return true;
    }

    GmIso4 localIso;
    tree.ComposeCollisionIso(parentIso, localIso);
    const u32 childCount = tree.GetChildCount();
    for (u32 childIndex = 0u; childIndex < childCount; ++childIndex) {
        if (!CollectEllipsoidPacketTraversalLanes(
                    localIso,
                    *tree.GetChild(childIndex),
                    lanes,
                    laneCount,
                    nextTemporalSlotOrdinal)) {
            return false;
        }
    }

    CPlugSurface *surface = tree.Surface();
    if (surface == nullptr) {
        return true;
    }
    const GmSurf *geometry = surface->Geometry();
    if (geometry == nullptr || typeid(*geometry) != typeid(GmSurfEllipsoid) ||
        !surface->UsesSphereContactBuffer() ||
        *laneCount >= EllipsoidPacketWidth) {
        return false;
    }

    GmBoxAligned bounds;
    tree.GetTransformedCollisionBox(parentIso, bounds);
    lanes->ConstructAt(
            (*laneCount)++,
            const_cast<CPlugTree *>(&tree),
            surface,
            localIso,
            bounds,
            temporalSlotOrdinal);
    return true;
}

void CompletePacketCollisionMaterials(
        CHmsCollisionBuffer &buffer,
        u32 firstNew,
        CPlugSurface &movingSurface,
        CPlugSurface &staticSurface) {
    const u32 count = buffer.PhysicalCollisionCount();
    for (u32 collisionIndex = firstNew;
         collisionIndex < count;
         ++collisionIndex) {
        GmCollision &collision = buffer.GetCollision(collisionIndex);
        collision.materialA =
                movingSurface.SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialA);
        collision.materialB =
                staticSurface.SurfaceMaterialIdFromLocalIndex(
                        collision.localMaterialB);
    }
}

inline void PopulateCertifiedPacketLane(
        OptimizedCpuPreparedEllipsoidMeshPacket &prepared,
        std::size_t laneIndex,
        const GmIso4 &ellipsoidWorld,
        const GmSurfEllipsoid &ellipsoid,
        CHmsCollisionBuffer *buffer) noexcept {
    const GmVec3 radii = ellipsoid.radii;
    prepared.worldXx.values[laneIndex] =
            ellipsoidWorld.rotation.basisX.x;
    prepared.worldXy.values[laneIndex] =
            ellipsoidWorld.rotation.basisY.x;
    prepared.worldXz.values[laneIndex] =
            ellipsoidWorld.rotation.basisZ.x;
    prepared.worldYx.values[laneIndex] =
            ellipsoidWorld.rotation.basisX.y;
    prepared.worldYy.values[laneIndex] =
            ellipsoidWorld.rotation.basisY.y;
    prepared.worldYz.values[laneIndex] =
            ellipsoidWorld.rotation.basisZ.y;
    prepared.worldZx.values[laneIndex] =
            ellipsoidWorld.rotation.basisX.z;
    prepared.worldZy.values[laneIndex] =
            ellipsoidWorld.rotation.basisY.z;
    prepared.worldZz.values[laneIndex] =
            ellipsoidWorld.rotation.basisZ.z;
    prepared.worldTx.values[laneIndex] = ellipsoidWorld.translation.x;
    prepared.worldTy.values[laneIndex] = ellipsoidWorld.translation.y;
    prepared.worldTz.values[laneIndex] = ellipsoidWorld.translation.z;
    prepared.radiiX.values[laneIndex] = radii.x;
    prepared.radiiY.values[laneIndex] = radii.y;
    prepared.radiiZ.values[laneIndex] = radii.z;
    prepared.inverseRadiiX.values[laneIndex] = 1.0f / radii.x;
    prepared.inverseRadiiY.values[laneIndex] = 1.0f / radii.y;
    prepared.inverseRadiiZ.values[laneIndex] = 1.0f / radii.z;
    prepared.materials[laneIndex] = ellipsoid.material;
    prepared.buffers[laneIndex] = buffer;
}

#if defined(__i386__) || defined(__x86_64__)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((hot, noinline, target("avx2"),
               optimize("no-inline-functions,no-unroll-loops")))
#elif defined(__clang__)
__attribute__((hot, noinline, target("avx2")))
#endif
std::size_t CollectDirectLaneStarTraversalLanesAvx2(
        const GmIso4 &rootLocation,
        const OptimizedCpuMovingEllipsoidPacketPlan &movingPlan,
        EllipsoidPacketTraversalLanes *lanes) {
    const auto *planNodes = movingPlan.NodeData();
    const auto *planLanes = movingPlan.LaneData();
    alignas(32) float sourceCenterX[EllipsoidPacketWidth];
    alignas(32) float sourceCenterY[EllipsoidPacketWidth];
    alignas(32) float sourceCenterZ[EllipsoidPacketWidth];
    alignas(32) float sourceHalfX[EllipsoidPacketWidth];
    alignas(32) float sourceHalfY[EllipsoidPacketWidth];
    alignas(32) float sourceHalfZ[EllipsoidPacketWidth];
    for (std::size_t laneIndex = 0u;
         laneIndex < EllipsoidPacketWidth;
         ++laneIndex) {
        const GmBoxAligned &source = planLanes[laneIndex].tree->Box();
        sourceCenterX[laneIndex] = source.center.x;
        sourceCenterY[laneIndex] = source.center.y;
        sourceCenterZ[laneIndex] = source.center.z;
        sourceHalfX[laneIndex] = source.halfExtents.x;
        sourceHalfY[laneIndex] = source.halfExtents.y;
        sourceHalfZ[laneIndex] = source.halfExtents.z;
    }

    const __m256 centerX = _mm256_load_ps(sourceCenterX);
    const __m256 centerY = _mm256_load_ps(sourceCenterY);
    const __m256 centerZ = _mm256_load_ps(sourceCenterZ);
    const __m256 halfX = _mm256_load_ps(sourceHalfX);
    const __m256 halfY = _mm256_load_ps(sourceHalfY);
    const __m256 halfZ = _mm256_load_ps(sourceHalfZ);
    if (!DirectLaneBoxValuesAreBounded(
                centerX,
                centerY,
                centerZ,
                halfX,
                halfY,
                halfZ)) {
        return 0u;
    }
    const GmMat3 &rotation = rootLocation.rotation;

    __m256 transformedCenterX = DirectLaneDot3(
            centerX, centerY, centerZ,
            rotation.basisX.x, rotation.basisY.x, rotation.basisZ.x);
    __m256 transformedCenterY = DirectLaneDot3(
            centerX, centerY, centerZ,
            rotation.basisX.y, rotation.basisY.y, rotation.basisZ.y);
    __m256 transformedCenterZ = DirectLaneDot3(
            centerX, centerY, centerZ,
            rotation.basisX.z, rotation.basisY.z, rotation.basisZ.z);
    transformedCenterX = _mm256_add_ps(
            transformedCenterX,
            _mm256_set1_ps(rootLocation.translation.x));
    transformedCenterY = _mm256_add_ps(
            transformedCenterY,
            _mm256_set1_ps(rootLocation.translation.y));
    transformedCenterZ = _mm256_add_ps(
            transformedCenterZ,
            _mm256_set1_ps(rootLocation.translation.z));
    const __m256 transformedHalfX = DirectLaneAbsoluteDot3(
            halfX, halfY, halfZ,
            rotation.basisX.x, rotation.basisY.x, rotation.basisZ.x);
    const __m256 transformedHalfY = DirectLaneAbsoluteDot3(
            halfX, halfY, halfZ,
            rotation.basisX.y, rotation.basisY.y, rotation.basisZ.y);
    const __m256 transformedHalfZ = DirectLaneAbsoluteDot3(
            halfX, halfY, halfZ,
            rotation.basisX.z, rotation.basisY.z, rotation.basisZ.z);

    _mm256_store_ps(sourceCenterX, transformedCenterX);
    _mm256_store_ps(sourceCenterY, transformedCenterY);
    _mm256_store_ps(sourceCenterZ, transformedCenterZ);
    _mm256_store_ps(sourceHalfX, transformedHalfX);
    _mm256_store_ps(sourceHalfY, transformedHalfY);
    _mm256_store_ps(sourceHalfZ, transformedHalfZ);

    for (std::size_t laneIndex = 0u;
         laneIndex < EllipsoidPacketWidth;
         ++laneIndex) {
        const OptimizedCpuMovingEllipsoidPacketPlan::Lane &planLane =
                planLanes[laneIndex];
        const OptimizedCpuMovingEllipsoidPacketPlan::Node &node =
                planNodes[laneIndex + 1u];
        GmIso4 location = node.usesLocalTransform
                ? node.tree->LocalIso()
                : rootLocation;
        if (node.usesLocalTransform) {
            location.Mult(rootLocation);
        }
        const GmBoxAligned bounds = {
            {
                sourceCenterX[laneIndex],
                sourceCenterY[laneIndex],
                sourceCenterZ[laneIndex],
            },
            {
                sourceHalfX[laneIndex],
                sourceHalfY[laneIndex],
                sourceHalfZ[laneIndex],
            },
        };
        lanes->ConstructAt(
                laneIndex,
                planLane.tree,
                planLane.surface,
                location,
                bounds,
                planLane.temporalSlotOrdinal);
    }
    return EllipsoidPacketWidth;
}
#endif

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((hot, noinline,
               optimize("no-inline-functions,no-unroll-loops")))
#elif defined(__clang__)
__attribute__((hot, noinline))
#endif
std::size_t CollectDirectLaneStarTraversalLanes(
        const GmIso4 &movingIso,
        const OptimizedCpuMovingEllipsoidPacketPlan &movingPlan,
        bool boundsArithmeticIsBounded,
        EllipsoidPacketTraversalLanes *lanes) {
    const auto *planNodes = movingPlan.NodeData();
    const auto *planLanes = movingPlan.LaneData();
    const OptimizedCpuMovingEllipsoidPacketPlan::Node &rootNode =
            planNodes[0u];
    GmIso4 rootLocation = rootNode.usesLocalTransform
            ? rootNode.tree->LocalIso()
            : movingIso;
    if (rootNode.usesLocalTransform) {
        rootLocation.Mult(movingIso);
    }

    const std::size_t laneCount = movingPlan.LaneCount();
#if defined(__i386__) || defined(__x86_64__)
    if (boundsArithmeticIsBounded &&
        laneCount == EllipsoidPacketWidth) {
        return CollectDirectLaneStarTraversalLanesAvx2(
                rootLocation, movingPlan, lanes);
    }
#else
    (void)boundsArithmeticIsBounded;
#endif
    for (std::size_t laneIndex = 0u;
         laneIndex < laneCount;
         ++laneIndex) {
        const OptimizedCpuMovingEllipsoidPacketPlan::Lane &planLane =
                planLanes[laneIndex];
        const OptimizedCpuMovingEllipsoidPacketPlan::Node &node =
                planNodes[laneIndex + 1u];
        GmIso4 location = node.usesLocalTransform
                ? node.tree->LocalIso()
                : rootLocation;
        if (node.usesLocalTransform) {
            location.Mult(rootLocation);
        }
        GmBoxAligned bounds;
        planLane.tree->GetTransformedCollisionBox(rootLocation, bounds);
        lanes->ConstructAt(
                laneIndex,
                planLane.tree,
                planLane.surface,
                location,
                bounds,
                planLane.temporalSlotOrdinal);
    }
    return laneCount;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((hot, noinline))
#endif
std::size_t CollectCompiledPlanTraversalLanes(
        const GmIso4 &movingIso,
        const OptimizedCpuMovingEllipsoidPacketPlan &movingPlan,
        EllipsoidPacketTraversalLanes *lanes) {
    UninitializedObjectArray<
            GmIso4,
            OptimizedCpuMovingEllipsoidPacketPlan::MaxNodeCount>
            nodeLocations;
    const auto *planNodes = movingPlan.NodeData();
    const auto *planLanes = movingPlan.LaneData();
    const auto *operations = movingPlan.OperationData();
    std::size_t laneCount = 0u;
    for (std::size_t operationIndex = 0u;
         operationIndex < movingPlan.OperationCount();
         ++operationIndex) {
        const OptimizedCpuMovingEllipsoidPacketPlan::Operation &operation =
                operations[operationIndex];
        if (operation.kind ==
            OptimizedCpuMovingEllipsoidPacketPlan::OperationKind::
                    ComposeNode) {
            const OptimizedCpuMovingEllipsoidPacketPlan::Node &node =
                    planNodes[operation.index];
            const GmIso4 &parentLocation =
                    node.parentNodeIndex ==
                                    OptimizedCpuMovingEllipsoidPacketPlan::
                                            NoParent
                            ? movingIso
                            : nodeLocations[node.parentNodeIndex];
            if (node.usesLocalTransform) {
                GmIso4 &location = nodeLocations.ConstructAt(
                        operation.index,
                        node.tree->LocalIso());
                location.Mult(parentLocation);
            } else {
                nodeLocations.ConstructAt(
                        operation.index,
                        parentLocation);
            }
            continue;
        }

        const OptimizedCpuMovingEllipsoidPacketPlan::Lane &planLane =
                planLanes[operation.index];
        const OptimizedCpuMovingEllipsoidPacketPlan::Node &node =
                planNodes[planLane.nodeIndex];
        const GmIso4 &parentLocation =
                node.parentNodeIndex ==
                                OptimizedCpuMovingEllipsoidPacketPlan::NoParent
                        ? movingIso
                        : nodeLocations[node.parentNodeIndex];
        GmBoxAligned bounds;
        planLane.tree->GetTransformedCollisionBox(parentLocation, bounds);
        lanes->ConstructAt(
                laneCount++,
                planLane.tree,
                planLane.surface,
                nodeLocations[planLane.nodeIndex],
                bounds,
                planLane.temporalSlotOrdinal);
    }
    return laneCount;
}

bool DetectEllipsoidPacketAgainstStaticGroup(
        CHmsCollisionManagerSZone &zone,
        const GmIso4 &movingIso,
        const CPlugTree &movingTree,
        const OptimizedCpuMovingEllipsoidPacketPlan *movingPlan,
        const OptimizedCpuStaticSurfaceTransformGroup &transforms,
        GmOctree<CHmsCollisionManagerSColOctreeCell> &staticTrees) {
    if (!OptimizedCpuEllipsoidMeshPacketAvailable()) {
        return false;
    }

    // The certified operation stream constructs every node and lane exactly
    // once before it is read. Raw slots avoid clearing the unused tail of the
    // fixed-capacity arrays on every packet pass while keeping normal C++
    // object lifetimes for the active entries.
    EllipsoidPacketTraversalLanes lanes;
    std::size_t laneCount = 0u;
    const bool directLaneStar =
            movingPlan != nullptr && movingPlan->IsDirectLaneStar();
    const bool directLaneFullPlan = directLaneStar &&
            movingPlan->LaneCount() == EllipsoidPacketWidth;
#if defined(__i386__) || defined(__x86_64__)
    const bool directLaneBoundsArithmeticIsBounded =
            directLaneFullPlan &&
            transforms.BroadPhaseArithmeticIsBoundedFor(
                    movingTree, movingIso);
#else
    const bool directLaneBoundsArithmeticIsBounded = false;
#endif
    if (directLaneFullPlan &&
        !directLaneBoundsArithmeticIsBounded) {
        return false;
    }
    if (movingPlan != nullptr) {
        laneCount = directLaneStar
                ? CollectDirectLaneStarTraversalLanes(
                          movingIso,
                          *movingPlan,
                          directLaneBoundsArithmeticIsBounded,
                          &lanes)
                : CollectCompiledPlanTraversalLanes(
                          movingIso, *movingPlan, &lanes);
    } else {
        u32 nextTemporalSlotOrdinal = 0u;
        if (!CollectEllipsoidPacketTraversalLanes(
                    movingIso,
                    movingTree,
                    &lanes,
                    &laneCount,
                    &nextTemporalSlotOrdinal) ||
            laneCount < 2u) {
            return false;
        }
    }
    if (laneCount < 2u) {
        return false;
    }

    const bool certifiedFullPacket =
            directLaneStar && laneCount == EllipsoidPacketWidth;
    const u32 *sharedCandidateCurrent = nullptr;
    std::size_t sharedCandidateRemaining = 0u;
    std::array<const u32 *, EllipsoidPacketWidth> candidateCurrent;
    std::array<std::size_t, EllipsoidPacketWidth> candidateRemaining;
    if (certifiedFullPacket) {
        // A direct lane star never emits a surface for its root, so temporal
        // ordinal zero is reserved. The refreshed root box contains every
        // direct lane and therefore yields one ordered conservative candidate
        // span for the whole packet.
        GmBoxAligned packetBounds;
        movingTree.GetTransformedCollisionBox(movingIso, packetBounds);
        OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateSpan span;
        if (!transforms.TemporalCandidateSpanFor(
                    movingTree,
                    0u,
                    packetBounds,
                    &span)) {
            return false;
        }
        sharedCandidateCurrent = span.data;
        sharedCandidateRemaining = span.size;
    } else {
        for (std::size_t laneIndex = 0u;
             laneIndex < laneCount;
             ++laneIndex) {
            OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateSpan span;
            if (!transforms.TemporalCandidateSpanFor(
                        *lanes[laneIndex].tree,
                        lanes[laneIndex].temporalSlotOrdinal,
                        lanes[laneIndex].bounds,
                        &span)) {
                return false;
            }
            candidateCurrent[laneIndex] = span.data;
            candidateRemaining[laneIndex] = span.size;
        }
    }

    std::array<OptimizedCpuEllipsoidMeshPacketLane, EllipsoidPacketWidth>
            packetLanes;
    OptimizedCpuPreparedEllipsoidMeshPacket preparedPacket;
    forevervalidator::simulation::OptimizedCpuStaticBoundsPacket8
            packedLaneBounds;
    for (std::size_t laneIndex = 0u;
         laneIndex < laneCount;
         ++laneIndex) {
        EllipsoidPacketTraversalLane &lane = lanes[laneIndex];
        if (!certifiedFullPacket &&
            !lane.surface->UsesSphereContactBuffer()) {
            return false;
        }
        lane.sphereContact = zone.EnsureTreeSphereContact(lane.tree);
        lane.buffer = lane.sphereContact;
        if (lane.buffer == nullptr) {
            return false;
        }
        lane.located = {
            lane.surface->Geometry(),
            &lane.location,
            1,
        };
        if (certifiedFullPacket) {
            packedLaneBounds.SetLane(laneIndex, lane.bounds);
            PopulateCertifiedPacketLane(
                    preparedPacket,
                    laneIndex,
                    lane.location,
                    static_cast<const GmSurfEllipsoid &>(
                            *lane.located.surf),
                    lane.buffer);
        } else {
            packetLanes[laneIndex] = {&lane.located, lane.buffer};
        }
    }
    const std::uint32_t allLaneMask =
            (1u << static_cast<std::uint32_t>(laneCount)) - 1u;
    if (certifiedFullPacket) {
        preparedPacket.laneCount = EllipsoidPacketWidth;
        preparedPacket.preparedMask = allLaneMask;
    } else {
        if (!PrepareOptimizedCpuEllipsoidMeshPacket(
                    packetLanes.data(),
                    laneCount,
                    allLaneMask,
                    &preparedPacket)) {
            return false;
        }
    }
    std::uint32_t collidedMask = 0u;
#if defined(__i386__) || defined(__x86_64__)
    // The packed test evaluates all three axes for every lane instead of
    // scalar z/y short-circuiting. The bounded-arithmetic certificate excludes
    // invalid, overflowing, and underflowing operands, leaving only FE_INEXACT
    // as a possible extra sticky status bit. Use SIMD only after that bit is
    // already set, so observable floating status remains unchanged.
    const bool usePacketBoundsOverlap = certifiedFullPacket &&
            (_mm_getcsr() & _MM_EXCEPT_INEXACT) != 0u &&
            directLaneBoundsArithmeticIsBounded;
#else
    const bool usePacketBoundsOverlap = false;
#endif

    for (;;) {
        u32 staticTreeIndex;
        if (certifiedFullPacket) {
            if (sharedCandidateRemaining == 0u) {
                break;
            }
            staticTreeIndex = *sharedCandidateCurrent++;
            --sharedCandidateRemaining;
        } else {
            staticTreeIndex = std::numeric_limits<u32>::max();
            for (std::size_t laneIndex = 0u;
                 laneIndex < laneCount;
                 ++laneIndex) {
                if (candidateRemaining[laneIndex] != 0u) {
                    staticTreeIndex = std::min(
                            staticTreeIndex, *candidateCurrent[laneIndex]);
                }
            }
            if (staticTreeIndex == std::numeric_limits<u32>::max()) {
                break;
            }
        }

        CHmsCollisionManagerSColOctreeCell *record =
                &staticTrees[staticTreeIndex];
        std::uint32_t activeMask = 0u;
        if (usePacketBoundsOverlap) {
            activeMask = forevervalidator::simulation::
                    OptimizedCpuStaticBoundsOverlapPacket8(
                            packedLaneBounds, record->Bounds());
        } else if (certifiedFullPacket) {
            for (std::size_t laneIndex = 0u;
                 laneIndex < laneCount;
                 ++laneIndex) {
                if (OptimizedCpuStaticBoundsOverlap(
                            lanes[laneIndex].bounds,
                            record->Bounds())) {
                    activeMask |= 1u << laneIndex;
                }
            }
        } else {
            for (std::size_t laneIndex = 0u;
                 laneIndex < laneCount;
                 ++laneIndex) {
                std::size_t &remaining = candidateRemaining[laneIndex];
                const u32 *&candidate = candidateCurrent[laneIndex];
                if (remaining == 0u || *candidate != staticTreeIndex) {
                    continue;
                }
                ++candidate;
                --remaining;
                if (OptimizedCpuStaticBoundsOverlap(
                            lanes[laneIndex].bounds,
                            record->Bounds())) {
                    activeMask |= 1u << laneIndex;
                }
            }
        }
        if (activeMask == 0u) {
            continue;
        }

        const CHmsCollisionManagerSColOctreeCell::StaticSurface &staticSurface =
                record->SurfaceData();
        const OptimizedCpuStaticMeshTriangleSidecar *triangleSidecar =
                transforms.TriangleSidecarAt(staticTreeIndex);
        const OptimizedCpuCertifiedStaticMeshPacket *certifiedMesh =
                transforms.CertifiedMeshPacketAt(staticTreeIndex);
        bool packetHandled = false;
        if (certifiedMesh != nullptr) {
            std::array<u32, EllipsoidPacketWidth> firstNew;
            for (std::size_t laneIndex = 0u;
                 laneIndex < laneCount;
                 ++laneIndex) {
                if ((activeMask & (1u << laneIndex)) == 0u) {
                    continue;
                }
                firstNew[laneIndex] =
                        lanes[laneIndex].buffer->PhysicalCollisionCount();
            }
            std::uint32_t hitMask = 0u;
            packetHandled =
                    GmCollision_PreparedEllipsoidPacket_Mesh_InlineMathOptimizedCpuNativeBinary32WithCertifiedStaticMesh(
                            preparedPacket,
                            activeMask,
                            *certifiedMesh,
                            &hitMask);
            if (packetHandled) {
                collidedMask |= hitMask;
                for (std::size_t laneIndex = 0u;
                     laneIndex < laneCount;
                     ++laneIndex) {
                    if ((hitMask & (1u << laneIndex)) == 0u) {
                        continue;
                    }
                    EllipsoidPacketTraversalLane &lane = lanes[laneIndex];
                    CompletePacketCollisionMaterials(
                            *lane.buffer,
                            firstNew[laneIndex],
                            *lane.surface,
                            *staticSurface.surface);
                    zone.TagNewStaticCollisions(
                            lane.buffer,
                            firstNew[laneIndex],
                            lane.tree,
                            record);
                }
            }
        }

        if (!packetHandled) {
            for (std::size_t laneIndex = 0u;
                 laneIndex < laneCount;
                 ++laneIndex) {
                if ((activeMask & (1u << laneIndex)) == 0u) {
                    continue;
                }
                EllipsoidPacketTraversalLane &lane = lanes[laneIndex];
                const u32 firstNew =
                        lane.buffer->PhysicalCollisionCount();
                const SPlugSurfaceLocatedPair surfacePair = {
                    *lane.surface,
                    lane.location,
                    *staticSurface.surface,
                    staticSurface.location,
                };
                const int collided =
                        ComputePlugSurfaceCollisionInlineMathOptimizedCpuNativeBinary32WithStaticCache(
                                surfacePair,
                                transforms.InverseAt(staticTreeIndex),
                                triangleSidecar,
                                *lane.buffer);
                if (collided == 0) {
                    continue;
                }
                zone.TagNewStaticCollisions(
                        lane.buffer,
                        firstNew,
                        lane.tree,
                        record);
                collidedMask |= 1u << laneIndex;
            }
        }
    }

    while (collidedMask != 0u) {
        unsigned int laneIndex = 0u;
        unsigned int firstSetBit = collidedMask;
        while ((firstSetBit & 1u) == 0u) {
            firstSetBit >>= 1u;
            ++laneIndex;
        }
        zone.AddSphereContactOnce(lanes[laneIndex].sphereContact);
        collidedMask &= collidedMask - 1u;
    }
    return true;
}

bool TrySkipWholeTreeBoundsEmpty(
        const GmIso4 &movingIso,
        const CPlugTree &movingTree,
        const OptimizedCpuStaticSurfaceTransformGroup &transforms) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    const unsigned int savedMxcsr = _mm_getcsr();
    constexpr unsigned int MxcsrControlMask = 0xffc0u;
    constexpr unsigned int DeterministicMxcsrControl = 0x1f80u;
    constexpr unsigned int InexactStatus = 0x20u;
    if ((savedMxcsr & MxcsrControlMask) != DeterministicMxcsrControl ||
        (savedMxcsr & InexactStatus) == 0u ||
        !transforms.BroadPhaseArithmeticIsBoundedFor(
                movingTree, movingIso)) {
        return false;
    }
    GmBoxAligned movingBounds;
    movingTree.GetTransformedCollisionBox(movingIso, movingBounds);
    const bool empty = !transforms.WholeTreeBoundsOverlapAnySurface(
            movingBounds);
    return empty;
#else
    (void)movingIso;
    (void)movingTree;
    (void)transforms;
    return false;
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((hot, noinline))
#endif
void DetectNativeBinary32CachedTemporalSpan(
        CHmsCollisionManagerSZone &zone,
        const GmBoxAligned &movingBox,
        CPlugTree &movingTree,
        CPlugSurface &movingSurface,
        const GmIso4 &movingLocation,
        const OptimizedCpuStaticSurfaceTransformGroup &transforms,
        const OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateSpan
                &temporalCandidates) {
    const u32 *candidate = temporalCandidates.data;
    const u32 *const end = candidate + temporalCandidates.size;
    const CHmsCollisionManagerSColOctreeCell *const records =
            transforms.RecordData();
    const GmIso4 *const inverses = transforms.InverseData();
    const OptimizedCpuStaticMeshTriangleSidecar *const *const
            triangleSidecars = transforms.TriangleSidecarData();
    for (; candidate != end; ++candidate) {
        const u32 staticTreeIndex = *candidate;
        const CHmsCollisionManagerSColOctreeCell *record =
                &records[staticTreeIndex];
        if (!OptimizedCpuStaticBoundsOverlap(
                    movingBox, record->Bounds())) {
            continue;
        }

        // The span contains only surface records, and the advance certificate
        // proves every static surface tree is collision-enabled.
        const CHmsCollisionManagerSColOctreeCell::StaticSurface
                &staticSurface = record->SurfaceData();
        SHmsSphereBufferContact *sphereContact = nullptr;
        CHmsCollisionBuffer *buffer = zone.ChooseCollisionOutputBuffer(
                &movingTree, &movingSurface, &sphereContact);
        const u32 firstNew = buffer->PhysicalCollisionCount();
        const SPlugSurfaceLocatedPair surfacePair = {
            movingSurface,
            movingLocation,
            *staticSurface.surface,
            staticSurface.location,
        };
        const int collided =
                ComputePlugSurfaceCollisionInlineMathOptimizedCpuNativeBinary32WithStaticCache(
                        surfacePair,
                        inverses[staticTreeIndex],
                        triangleSidecars[staticTreeIndex],
                        *buffer);
        if (collided == 0) {
            continue;
        }
        if (sphereContact != nullptr) {
            zone.AddSphereContactOnce(sphereContact);
        }
        zone.TagNewStaticCollisions(
                buffer, firstNew, &movingTree, record);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((cold, noinline))
#endif
void DetectNativeBinary32CachedColdFallback(
        CHmsCollisionManagerSZone &zone,
        const GmBoxAligned &movingBox,
        CPlugTree &movingTree,
        CPlugSurface &movingSurface,
        const GmIso4 &movingLocation,
        const OptimizedCpuStaticSurfaceTransformGroup &transforms,
        GmOctree<CHmsCollisionManagerSColOctreeCell> &staticTrees) {
    const auto processSurfaceRecord = [&](u32 staticTreeIndex) {
        CHmsCollisionManagerSColOctreeCell *record =
                &staticTrees[staticTreeIndex];
        if (!OptimizedCpuStaticBoundsOverlap(
                    movingBox, record->Bounds()) ||
            !record->ContainsSurface()) {
            return;
        }

        const CHmsCollisionManagerSColOctreeCell::StaticSurface
                &staticSurface = record->SurfaceData();
        SHmsSphereBufferContact *sphereContact = nullptr;
        CHmsCollisionBuffer *buffer = zone.ChooseCollisionOutputBuffer(
                &movingTree, &movingSurface, &sphereContact);
        const u32 firstNew = buffer->PhysicalCollisionCount();
        const SPlugSurfaceLocatedPair surfacePair = {
            movingSurface,
            movingLocation,
            *staticSurface.surface,
            staticSurface.location,
        };
        const int collided =
                ComputePlugSurfaceCollisionInlineMathOptimizedCpuNativeBinary32WithStaticCache(
                        surfacePair,
                        transforms.InverseAt(staticTreeIndex),
                        transforms.TriangleSidecarAt(staticTreeIndex),
                        *buffer);
        if (collided == 0) {
            return;
        }
        if (sphereContact != nullptr) {
            zone.AddSphereContactOnce(sphereContact);
        }
        zone.TagNewStaticCollisions(
                buffer, firstNew, &movingTree, record);
    };

    const u32 staticTreeCount = staticTrees.GetCount();
    for (u32 staticTreeIndex = 0u;
         staticTreeIndex < staticTreeCount;) {
        CHmsCollisionManagerSColOctreeCell *record =
                &staticTrees[staticTreeIndex];
        if (!OptimizedCpuStaticBoundsOverlap(
                    movingBox, record->Bounds())) {
            staticTreeIndex += record->SubtreeEntryCount();
            continue;
        }
        processSurfaceRecord(staticTreeIndex);
        ++staticTreeIndex;
    }
}

}  // namespace

bool OptimizedCpuCollectDirectLaneStarBoundsForDifferential(
        const GmIso4 &movingIso,
        const OptimizedCpuMovingEllipsoidPacketPlan &movingPlan,
        bool boundsArithmeticIsBounded,
        std::array<GmBoxAligned,
                   OptimizedCpuMovingEllipsoidPacketPlan::MaxLaneCount>
                *bounds) noexcept {
    if (bounds == nullptr || !boundsArithmeticIsBounded ||
        !movingPlan.IsDirectLaneStar() ||
        movingPlan.LaneCount() !=
                OptimizedCpuMovingEllipsoidPacketPlan::MaxLaneCount ||
        !OptimizedCpuEllipsoidMeshPacketAvailable()) {
        return false;
    }
    EllipsoidPacketTraversalLanes lanes;
    if (CollectDirectLaneStarTraversalLanes(
                movingIso,
                movingPlan,
                boundsArithmeticIsBounded,
                &lanes) !=
        OptimizedCpuMovingEllipsoidPacketPlan::MaxLaneCount) {
        return false;
    }
    for (std::size_t laneIndex = 0u;
         laneIndex < OptimizedCpuMovingEllipsoidPacketPlan::MaxLaneCount;
         ++laneIndex) {
        (*bounds)[laneIndex] = lanes[laneIndex].bounds;
    }
    return true;
}

void CHmsCollisionManager::SZone::DetectCollisionsCorpusOptimizedCpuCached(
        CHmsCollisionBuffer &collisionBuffer,
        CHmsCorpus *corpus,
        const OptimizedCpuStaticSurfaceTransformCache &transforms) {
    activeCollisionBuffer = &collisionBuffer;

    const u32 groupIndex = corpus->Item()->CollisionGroup();
    CHmsCollisionManagerSGroup *group = &groups[groupIndex - 1u];

    for (CHmsCollisionManagerSAgainstGroup &againstEntry :
         group->againstGroups) {
        CHmsCollisionManagerSAgainstGroup *against = &againstEntry;
        activeCollisionGroupPair = against->collisionGroupPair;

        for (const SGroup::MovingCorpusState &target :
             against->targetGroup->movingCorpuses) {
            CHmsCorpus *other = target.corpus;
            if (against->collisionSchedule.IsEnabled(*corpus, *other)) {
                DetectCollisionBetween(corpus, other);
            }
        }

        activeStaticTargetGroup = against->targetGroup;
        if (against->targetGroup->StaticTreeCount() > 1u) {
            activeCorpusA = corpus;
            const OptimizedCpuStaticSurfaceTransformGroup *groupTransforms =
                    transforms.GroupFor(*against->targetGroup);
            if (groupTransforms == nullptr) {
                DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpu(
                        *corpus->LocationIso(),
                        *corpus->CollisionTree());
            } else {
                u32 nextTemporalSlotOrdinal = 0u;
                DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuCached(
                        *corpus->LocationIso(),
                        *corpus->CollisionTree(),
                        nextTemporalSlotOrdinal,
                        *groupTransforms);
            }
        }
    }

    MergeQueuedSphereContacts(collisionBuffer);
}

void CHmsCollisionManager::SZone::
DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuCached(
        const GmIso4 &movingIsoRef,
        const CPlugTree &movingTree,
        u32 &nextTemporalSlotOrdinal,
        const OptimizedCpuStaticSurfaceTransformGroup &transforms) {
    CHmsCollisionManagerSZone *zone = this;
    const GmIso4 *movingIso = &movingIsoRef;
    const u32 temporalSlotOrdinal = nextTemporalSlotOrdinal++;
    if (!movingTree.HasWorldBox()) {
        return;
    }

    GmIso4 localIso;
    movingTree.ComposeCollisionIso(*movingIso, localIso);

    const u32 childCount = movingTree.GetChildCount();
    for (u32 childIndex = 0u; childIndex < childCount; ++childIndex) {
        zone->DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuCached(
                localIso,
                *movingTree.GetChild(childIndex),
                nextTemporalSlotOrdinal,
                transforms);
    }

    CPlugSurface *movingSurface = movingTree.Surface();
    if (movingSurface == nullptr) {
        return;
    }
    CPlugTree *movingTreeNode = const_cast<CPlugTree *>(&movingTree);

    GmBoxAligned movingBox;
    movingTree.GetTransformedCollisionBox(*movingIso, movingBox);

    GmOctree<CHmsCollisionManagerSColOctreeCell> &staticTrees =
            zone->activeStaticTargetGroup->staticTrees;
    const u32 staticTreeCount = staticTrees.GetCount();
    const auto processSurfaceRecord = [&](u32 staticTreeIndex,
                                          bool surfaceIsKnown) {
        CHmsCollisionManagerSColOctreeCell *record =
                &staticTrees[staticTreeIndex];
        if (!OptimizedCpuStaticBoundsOverlap(movingBox, record->Bounds()) ||
            (!surfaceIsKnown && !record->ContainsSurface())) {
            return;
        }

        const SColOctreeCell::StaticSurface &staticSurface =
                record->SurfaceData();
        SHmsSphereBufferContact *sphereContact = nullptr;
        CHmsCollisionBuffer *buffer = zone->ChooseCollisionOutputBuffer(
                movingTreeNode, movingSurface, &sphereContact);
        const u32 firstNew = buffer->PhysicalCollisionCount();
        const SPlugSurfaceLocatedPair surfacePair = {
            *movingSurface,
            localIso,
            *staticSurface.surface,
            staticSurface.location,
        };

        const int surfaceCollisionResult =
                ComputeCollisionOptimizedCpuWithStaticMeshTriangleSidecar(
                        surfacePair,
                        transforms.InverseAt(staticTreeIndex),
                        transforms.TriangleSidecarAt(staticTreeIndex),
                        *buffer);
        if (surfaceCollisionResult) {
            if (sphereContact != nullptr) {
                zone->AddSphereContactOnce(sphereContact);
            }
            zone->TagNewStaticCollisions(
                    buffer, firstNew, movingTreeNode, record);
        }
    };

    OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateSpan
            temporalCandidates;
    if (transforms.TemporalCandidateSpanFor(
                movingTree,
                temporalSlotOrdinal,
                movingBox,
                &temporalCandidates)) {
        for (std::size_t candidateIndex = 0u;
             candidateIndex < temporalCandidates.size;
             ++candidateIndex) {
            processSurfaceRecord(
                    temporalCandidates.data[candidateIndex], true);
        }
        return;
    }

    for (u32 staticTreeIndex = 0u;
         staticTreeIndex < staticTreeCount;) {
        CHmsCollisionManagerSColOctreeCell *record =
                &staticTrees[staticTreeIndex];
        if (!OptimizedCpuStaticBoundsOverlap(movingBox, record->Bounds())) {
            staticTreeIndex += record->SubtreeEntryCount();
            continue;
        }

        processSurfaceRecord(staticTreeIndex, false);

        ++staticTreeIndex;
    }
}

void CHmsCollisionManager::SZone::
DetectCollisionsCorpusOptimizedCpuNativeBinary32Cached(
        CHmsCollisionBuffer &collisionBuffer,
        CHmsCorpus *corpus,
        const OptimizedCpuStaticSurfaceTransformCache &transforms) {
    activeCollisionBuffer = &collisionBuffer;

    const u32 groupIndex = corpus->Item()->CollisionGroup();
    CHmsCollisionManagerSGroup *group = &groups[groupIndex - 1u];

    for (CHmsCollisionManagerSAgainstGroup &againstEntry :
         group->againstGroups) {
        CHmsCollisionManagerSAgainstGroup *against = &againstEntry;
        activeCollisionGroupPair = against->collisionGroupPair;

        for (const SGroup::MovingCorpusState &target :
             against->targetGroup->movingCorpuses) {
            CHmsCorpus *other = target.corpus;
            if (against->collisionSchedule.IsEnabled(*corpus, *other)) {
                DetectCollisionBetween(corpus, other);
            }
        }

        activeStaticTargetGroup = against->targetGroup;
        if (against->targetGroup->StaticTreeCount() > 1u) {
            activeCorpusA = corpus;
            const OptimizedCpuStaticSurfaceTransformGroup *groupTransforms =
                    transforms.GroupFor(*against->targetGroup);
            if (groupTransforms == nullptr) {
                DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32(
                        *corpus->LocationIso(),
                        *corpus->CollisionTree());
            } else {
                bool empty = false;
                if (groupTransforms->ShouldRefreshWholePassPrediction(
                            *corpus->CollisionTree())) {
                    empty = TrySkipWholeTreeBoundsEmpty(
                            *corpus->LocationIso(),
                            *corpus->CollisionTree(),
                            *groupTransforms);
                    groupTransforms->ObserveWholePassResult(
                            *corpus->CollisionTree(), empty);
                }
                if (empty) {
                    continue;
                }
                if (!DetectEllipsoidPacketAgainstStaticGroup(
                            *this,
                            *corpus->LocationIso(),
                            *corpus->CollisionTree(),
                            transforms.MovingEllipsoidPacketPlanFor(
                                    *corpus->CollisionTree()),
                            *groupTransforms,
                            against->targetGroup->staticTrees)) {
                    u32 nextTemporalSlotOrdinal = 0u;
                    DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32Cached(
                            *corpus->LocationIso(),
                            *corpus->CollisionTree(),
                            nextTemporalSlotOrdinal,
                            *groupTransforms);
                }
            }
        }
    }

    MergeQueuedSphereContacts(collisionBuffer);
}

void CHmsCollisionManager::SZone::
DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32Cached(
        const GmIso4 &movingIsoRef,
        const CPlugTree &movingTree,
        u32 &nextTemporalSlotOrdinal,
        const OptimizedCpuStaticSurfaceTransformGroup &transforms) {
    CHmsCollisionManagerSZone *zone = this;
    const GmIso4 *movingIso = &movingIsoRef;
    const u32 temporalSlotOrdinal = nextTemporalSlotOrdinal++;
    if (!movingTree.HasWorldBox()) {
        return;
    }

    GmIso4 localIso;
    movingTree.ComposeCollisionIso(*movingIso, localIso);

    const u32 childCount = movingTree.GetChildCount();
    for (u32 childIndex = 0u; childIndex < childCount; ++childIndex) {
        zone->DetectCollisionBetweenTreeAndStaticCollisionTreeOptimizedCpuNativeBinary32Cached(
                localIso,
                *movingTree.GetChild(childIndex),
                nextTemporalSlotOrdinal,
                transforms);
    }

    CPlugSurface *movingSurface = movingTree.Surface();
    if (movingSurface == nullptr) {
        return;
    }
    CPlugTree *movingTreeNode = const_cast<CPlugTree *>(&movingTree);

    GmBoxAligned movingBox;
    movingTree.GetTransformedCollisionBox(*movingIso, movingBox);

    GmOctree<CHmsCollisionManagerSColOctreeCell> &staticTrees =
            zone->activeStaticTargetGroup->staticTrees;

    OptimizedCpuStaticSurfaceTransformGroup::TemporalCandidateSpan
            temporalCandidates;
    if (transforms.TemporalCandidateSpanFor(
                movingTree,
                temporalSlotOrdinal,
                movingBox,
                &temporalCandidates)) {
        if (temporalCandidates.size != 0u) {
            DetectNativeBinary32CachedTemporalSpan(
                    *zone,
                    movingBox,
                    *movingTreeNode,
                    *movingSurface,
                    localIso,
                    transforms,
                    temporalCandidates);
        }
        return;
    }

    DetectNativeBinary32CachedColdFallback(
            *zone,
            movingBox,
            *movingTreeNode,
            *movingSurface,
            localIso,
            transforms,
            staticTrees);
}
