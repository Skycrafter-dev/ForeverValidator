// Generated from src/simulation/backends/cuda/cuda_scene_layout.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_SCENE_LAYOUT_H
#define FOREVERVALIDATOR_HIP_SCENE_LAYOUT_H

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

#include "engine/physics/geometry/gm_surface.h"
#include "engine/scene/static_scene_model.h"

namespace forevervalidator::simulation {

enum class HipSceneBuildResult : std::uint8_t {
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

struct HipSceneBuildLimits {
    std::uint32_t maximumActors = 65536u;
    std::uint32_t maximumSurfaces = 262144u;
    std::uint32_t maximumMaterials = 1048576u;
    std::uint32_t maximumVertices = 4194304u;
    std::uint32_t maximumTriangles = 4194304u;
    std::uint32_t maximumOctreeCells = 8388608u;
    std::uint32_t maximumAccelerationCells = 1048576u;
    std::uint32_t maximumCheckpoints = 1024u;
};

struct HipSceneActor {
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

struct HipSceneSurface {
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

struct HipSceneTriangle {
    GmVec3 normal{};
    float planeDistance = 0.0f;
    std::uint32_t vertexIndices[3]{};
    GmLocalMaterialIndex material{};
};

struct HipSceneOctreeCell {
    GmBoxAligned bounds{};
    std::uint32_t subtreeEntryCount = 1u;
    std::uint32_t containsTriangle = 0u;
    std::uint32_t triangleIndex = 0u;
};

struct HipSceneAccelerationCell {
    GmBoxAligned bounds{};
    std::uint32_t subtreeEntryCount = 1u;
    std::uint32_t surfaceIndex = UINT32_MAX;
};

struct HipSceneAccelerationRange {
    std::uint32_t firstCell = 0u;
    std::uint32_t cellCount = 0u;
};

struct HipHostScene {
    static constexpr std::uint32_t SchemaVersion = 2u;

    std::uint32_t schemaVersion = SchemaVersion;
    std::uint64_t deterministicHash = 0u;
    std::vector<HipSceneActor> actors;
    std::vector<HipSceneSurface> surfaces;
    std::vector<std::uint32_t> materials;
    std::vector<GmVec3> vertices;
    std::vector<HipSceneTriangle> triangles;
    std::vector<HipSceneOctreeCell> octreeCells;
    std::array<HipSceneAccelerationRange, 5u> accelerationGroups{};
    std::vector<HipSceneAccelerationCell> accelerationCells;

    void Clear() noexcept;
    bool Valid(const HipSceneBuildLimits &limits = {}) const noexcept;
};

HipSceneBuildResult BuildHipHostScene(
        const StaticSceneModelCollection &source,
        HipHostScene *destination,
        const HipSceneBuildLimits &limits = {}) noexcept;
const char *HipSceneBuildResultName(
        HipSceneBuildResult result) noexcept;

}  // namespace forevervalidator::simulation

#endif
