#ifndef FOREVERVALIDATOR_CUDA_SCENE_LAYOUT_H
#define FOREVERVALIDATOR_CUDA_SCENE_LAYOUT_H

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

#include "engine/physics/geometry/gm_surface.h"
#include "engine/scene/static_scene_model.h"

namespace forevervalidator::simulation {

enum class CudaSceneBuildResult : std::uint8_t {
    Success,
    InvalidSource,
    PrototypeConstructionFailed,
    UnsupportedGeometry,
    ActorOverflow,
    SurfaceOverflow,
    MaterialOverflow,
    VertexOverflow,
    TriangleOverflow,
    OctreeOverflow,
    AccelerationOverflow,
    CheckpointOverflow,
    AllocationFailed,
};

struct CudaSceneBuildLimits {
    std::uint32_t maximumActors = 65536u;
    std::uint32_t maximumSurfaces = 262144u;
    std::uint32_t maximumMaterials = 1048576u;
    std::uint32_t maximumVertices = 4194304u;
    std::uint32_t maximumTriangles = 4194304u;
    std::uint32_t maximumOctreeCells = 8388608u;
    std::uint32_t maximumAccelerationCells = 1048576u;
    std::uint32_t maximumCheckpoints = 1024u;
};

struct CudaSceneActor {
    GmIso4 worldPose{};
    CHmsItem::Properties itemProperties{};
    std::uint32_t installationOrder = 0u;
    std::uint32_t purpose = 0u;
    bool hasCheckpoint = false;
    std::uint32_t checkpointRole = 0u;
    std::uint32_t raceBlockId = 0u;
    std::uint32_t checkpointSlot = UINT32_MAX;
    bool respawnUsesCurrentTransform = false;
    bool hasCheckpointSpawn = false;
    GmIso4 checkpointSpawn{};
};

struct CudaSceneSurface {
    std::uint32_t actorIndex = 0u;
    std::uint32_t type = 0u;
    GmIso4 localToWorld{};
    GmIso4 worldToLocal{};
    GmBoxAligned localBounds{};
    GmBoxAligned worldBounds{};
    std::uint32_t firstMaterial = 0u;
    std::uint32_t materialCount = 0u;
    std::uint32_t firstVertex = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t firstTriangle = 0u;
    std::uint32_t triangleCount = 0u;
    std::uint32_t firstOctreeCell = 0u;
    std::uint32_t octreeCellCount = 0u;
    GmLocalMaterialIndex primitiveMaterial{};
    float sphereRadius = 0.0f;
    GmVec3 ellipsoidRadii{};
    GmVec3 boxCenter{};
    GmVec3 boxHalfExtents{};
    GmVec3 polygonVertices[4]{};
    std::uint32_t polygonVertexCount = 0u;
    GmVec3 polygonNormal{};
    bool polygonBackSide = false;
    bool usesSphereContactBuffer = false;
    bool allowsStaticCollisionRecordAppend = false;
};

struct CudaSceneTriangle {
    GmVec3 normal{};
    float planeDistance = 0.0f;
    std::uint32_t vertexIndices[3]{};
    GmLocalMaterialIndex material{};
    // Duplicate the immutable vertex payload beside the triangle.  Collision
    // traversal reaches a triangle through an octree-cell index, so keeping
    // the three positions here removes three further dependent global-memory
    // lookups from the latency-bound CUDA hot path.  Keep vertexIndices for
    // layout diagnostics and any consumers that need the source topology.
    GmVec3 vertices[3]{};
};

struct CudaSceneOctreeCell {
    GmBoxAligned bounds{};
    std::uint32_t subtreeEntryCount = 1u;
    std::uint32_t containsTriangle = 0u;
    std::uint32_t triangleIndex = 0u;
};

struct CudaSceneAccelerationCell {
    GmBoxAligned bounds{};
    std::uint32_t subtreeEntryCount = 1u;
    std::uint32_t surfaceIndex = UINT32_MAX;
};

struct CudaSceneAccelerationRange {
    std::uint32_t firstCell = 0u;
    std::uint32_t cellCount = 0u;
};

struct CudaHostScene {
    static constexpr std::uint32_t SchemaVersion = 2u;

    std::uint32_t schemaVersion = SchemaVersion;
    std::uint64_t deterministicHash = 0u;
    std::vector<CudaSceneActor> actors;
    std::vector<CudaSceneSurface> surfaces;
    std::vector<std::uint32_t> materials;
    std::vector<GmVec3> vertices;
    std::vector<CudaSceneTriangle> triangles;
    std::vector<CudaSceneOctreeCell> octreeCells;
    std::array<CudaSceneAccelerationRange, 5u> accelerationGroups{};
    std::vector<CudaSceneAccelerationCell> accelerationCells;

    void Clear() noexcept;
    bool Valid(const CudaSceneBuildLimits &limits = {}) const noexcept;
};

CudaSceneBuildResult BuildCudaHostScene(
        const StaticSceneModelCollection &source,
        CudaHostScene *destination,
        const CudaSceneBuildLimits &limits = {}) noexcept;
const char *CudaSceneBuildResultName(
        CudaSceneBuildResult result) noexcept;

}  // namespace forevervalidator::simulation

#endif
