#include "simulation/backends/cuda/cuda_scene_layout.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <new>
#include <utility>

#include "engine/scene/plug_solid.h"

namespace forevervalidator::simulation {
namespace {

template<typename T>
void ClearPadding(T &value) {
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12
    __builtin_clear_padding(&value);
#else
    static_cast<void>(value);
#endif
}

class Hash {
public:
    template<typename T>
    void Add(const T &value) {
        const auto *bytes =
                reinterpret_cast<const unsigned char *>(&value);
        for (std::size_t index = 0u; index < sizeof(value); ++index) {
            value_ ^= bytes[index];
            value_ *= 1099511628211ull;
        }
    }

    template<typename T>
    void Add(const std::vector<T> &values) {
        Add(values.size());
        for (const T &value : values) {
            Add(value);
        }
    }

    std::uint64_t Value() const noexcept { return value_; }

private:
    std::uint64_t value_ = 1469598103934665603ull;
};

CHmsItem::Properties DefaultStaticProperties() {
    CHmsItem item;
    item.ApplyBlockMobilDefaults();
    return item.GetProperties();
}

bool WouldOverflow(std::size_t current,
                   std::size_t additional,
                   std::uint32_t maximum) {
    return current > maximum || additional > maximum - current;
}

enum class AccelerationPartition {
    Overlapping,
    Below,
    Above,
};

GmBoxAligned AccelerationSpanBounds(
        const CudaHostScene &scene,
        const std::vector<std::uint32_t> &surfaces) {
    GmBoxAligned bounds =
            scene.surfaces[surfaces.front()].worldBounds;
    for (std::size_t index = 1u; index < surfaces.size(); ++index) {
        bounds.Union(
                scene.surfaces[surfaces[index]].worldBounds);
    }
    return bounds;
}

AccelerationPartition PartitionAccelerationSurface(
        const CudaSceneSurface &surface,
        GmAxis axis,
        float splitPlane) {
    const float center = surface.worldBounds.center.Component(axis);
    const float halfExtent =
            surface.worldBounds.halfExtents.Component(axis);
    if (center + halfExtent <= splitPlane) {
        return AccelerationPartition::Below;
    }
    if (center - halfExtent >= splitPlane) {
        return AccelerationPartition::Above;
    }
    return AccelerationPartition::Overlapping;
}

void ExtractAccelerationPartition(
        std::vector<std::uint32_t> &remaining,
        std::vector<AccelerationPartition> &partitions,
        AccelerationPartition selectedPartition,
        std::vector<std::uint32_t> &selected) {
    selected.clear();
    std::size_t index = 0u;
    while (index < remaining.size()) {
        if (partitions[index] != selectedPartition) {
            ++index;
            continue;
        }
        selected.push_back(remaining[index]);
        remaining[index] = remaining.back();
        partitions[index] = partitions.back();
        remaining.pop_back();
        partitions.pop_back();
    }
}

bool AccelerationShouldSplit(
        const GmBoxAligned &bounds,
        std::uint32_t count) {
    if (count <= 1u) return false;
    const float extentSquared =
            bounds.halfExtents.x * bounds.halfExtents.x +
            bounds.halfExtents.y * bounds.halfExtents.y +
            bounds.halfExtents.z * bounds.halfExtents.z;
    return extentSquared != 0.0f;
}

bool AppendAccelerationLeaf(
        CudaHostScene &scene,
        std::uint32_t surface,
        const CudaSceneBuildLimits &limits) {
    if (scene.accelerationCells.size() >=
        limits.maximumAccelerationCells) {
        return false;
    }
    CudaSceneAccelerationCell cell;
    cell.bounds = scene.surfaces[surface].worldBounds;
    cell.surfaceIndex = surface;
    ClearPadding(cell);
    scene.accelerationCells.push_back(cell);
    return true;
}

bool AppendAccelerationBranch(
        CudaHostScene &scene,
        const GmBoxAligned &bounds,
        const CudaSceneBuildLimits &limits,
        std::uint32_t *index) {
    if (index == nullptr ||
        scene.accelerationCells.size() >=
                limits.maximumAccelerationCells) {
        return false;
    }
    *index = static_cast<std::uint32_t>(
            scene.accelerationCells.size());
    CudaSceneAccelerationCell cell;
    cell.bounds = bounds;
    ClearPadding(cell);
    scene.accelerationCells.push_back(cell);
    return true;
}

bool BuildAccelerationRecurse(
        CudaHostScene &scene,
        const std::vector<std::uint32_t> &source,
        const CudaSceneBuildLimits &limits,
        std::uint32_t branchIndex,
        std::uint32_t *emittedCount) {
    if (emittedCount == nullptr) return false;
    if (source.empty()) {
        *emittedCount = 0u;
        return true;
    }
    const GmBoxAligned span =
            AccelerationSpanBounds(scene, source);
    scene.accelerationCells[branchIndex].bounds = span;
    if (!AccelerationShouldSplit(
                span, static_cast<std::uint32_t>(source.size()))) {
        for (std::uint32_t surface : source) {
            if (!AppendAccelerationLeaf(scene, surface, limits)) {
                return false;
            }
        }
        *emittedCount =
                static_cast<std::uint32_t>(source.size()) + 1u;
        return true;
    }

    const GmAxis axis = span.LongestAxis();
    const float splitPlane = span.center.Component(axis);
    std::vector<std::uint32_t> remaining = source;
    std::vector<AccelerationPartition> partitions;
    partitions.reserve(source.size());
    for (std::uint32_t surface : source) {
        partitions.push_back(PartitionAccelerationSurface(
                scene.surfaces[surface], axis, splitPlane));
    }

    const std::size_t originalCount = remaining.size();
    std::vector<std::uint32_t> selected;
    std::uint32_t total = 1u;
    for (AccelerationPartition partition :
         {AccelerationPartition::Below,
          AccelerationPartition::Above}) {
        ExtractAccelerationPartition(
                remaining, partitions, partition, selected);
        if (selected.size() == 1u) {
            if (!AppendAccelerationLeaf(
                        scene, selected.front(), limits)) {
                return false;
            }
            ++total;
        } else if (selected.size() > 1u) {
            std::uint32_t childIndex = 0u;
            if (!AppendAccelerationBranch(
                        scene,
                        AccelerationSpanBounds(scene, selected),
                        limits, &childIndex)) {
                return false;
            }
            std::uint32_t childCount = 0u;
            if (!BuildAccelerationRecurse(
                        scene, selected, limits, childIndex,
                        &childCount)) {
                return false;
            }
            scene.accelerationCells[childIndex].
                    subtreeEntryCount = childCount;
            total += childCount;
        }
    }

    if (remaining.size() != originalCount &&
        remaining.size() > 1u) {
        std::uint32_t childIndex = 0u;
        if (!AppendAccelerationBranch(
                    scene,
                    AccelerationSpanBounds(scene, remaining),
                    limits, &childIndex)) {
            return false;
        }
        std::uint32_t childCount = 0u;
        if (!BuildAccelerationRecurse(
                    scene, remaining, limits, childIndex,
                    &childCount)) {
            return false;
        }
        scene.accelerationCells[childIndex].
                subtreeEntryCount = childCount;
        total += childCount;
    } else {
        for (std::uint32_t surface : remaining) {
            if (!AppendAccelerationLeaf(scene, surface, limits)) {
                return false;
            }
        }
        total += static_cast<std::uint32_t>(remaining.size());
    }
    *emittedCount = total;
    return true;
}

bool BuildAcceleration(
        CudaHostScene &scene,
        const CudaSceneBuildLimits &limits) {
    for (std::uint32_t group = 1u; group <= 5u; ++group) {
        std::vector<std::uint32_t> surfaces;
        for (std::uint32_t index = 0u;
             index < scene.surfaces.size(); ++index) {
            const CudaSceneSurface &surface =
                    scene.surfaces[index];
            const CudaSceneActor &actor =
                    scene.actors[surface.actorIndex];
            if (actor.itemProperties.collisionStatic &&
                surface.allowsStaticCollisionRecordAppend &&
                static_cast<std::uint32_t>(
                        actor.itemProperties.collisionGroup) ==
                        group) {
                surfaces.push_back(index);
            }
        }
        CudaSceneAccelerationRange &range =
                scene.accelerationGroups[group - 1u];
        range.firstCell = static_cast<std::uint32_t>(
                scene.accelerationCells.size());
        std::uint32_t root = 0u;
        if (!AppendAccelerationBranch(
                    scene, {}, limits, &root)) {
            return false;
        }
        if (!surfaces.empty()) {
            std::uint32_t count = 0u;
            if (!BuildAccelerationRecurse(
                        scene, surfaces, limits, root, &count)) {
                return false;
            }
            scene.accelerationCells[root].subtreeEntryCount =
                    count;
        }
        range.cellCount = static_cast<std::uint32_t>(
                scene.accelerationCells.size()) -
                range.firstCell;
    }
    return true;
}

struct Builder {
    CudaHostScene &output;
    const CudaSceneBuildLimits &limits;
    const CHmsItem::Properties defaultProperties =
            DefaultStaticProperties();
    std::uint32_t checkpointSlot = 0u;

    CudaSceneBuildResult AddMaterials(
            const CPlugSurface &surface,
            CudaSceneSurface &record) {
        const std::uint32_t count = surface.MaterialCount();
        if (WouldOverflow(output.materials.size(), count,
                          limits.maximumMaterials)) {
            return CudaSceneBuildResult::MaterialOverflow;
        }
        record.firstMaterial =
                static_cast<std::uint32_t>(output.materials.size());
        record.materialCount = count;
        for (std::uint32_t index = 0u; index < count; ++index) {
            output.materials.push_back(static_cast<std::uint32_t>(
                    surface.SurfaceMaterialIdFromLocalIndex(
                            GmLocalMaterialIndex::FromIndex(
                                    static_cast<std::uint16_t>(index)))));
        }
        return CudaSceneBuildResult::Success;
    }

    CudaSceneBuildResult AddMesh(
            const GmSurfMesh &mesh,
            CudaSceneSurface &record) {
        if (WouldOverflow(output.vertices.size(), mesh.VertexCount(),
                          limits.maximumVertices)) {
            return CudaSceneBuildResult::VertexOverflow;
        }
        if (WouldOverflow(output.triangles.size(), mesh.TriangleCount(),
                          limits.maximumTriangles)) {
            return CudaSceneBuildResult::TriangleOverflow;
        }
        if (WouldOverflow(output.octreeCells.size(),
                          mesh.OctreeCellCount(),
                          limits.maximumOctreeCells)) {
            return CudaSceneBuildResult::OctreeOverflow;
        }
        record.firstVertex =
                static_cast<std::uint32_t>(output.vertices.size());
        record.vertexCount = mesh.VertexCount();
        for (std::uint32_t index = 0u;
             index < mesh.VertexCount();
             ++index) {
            output.vertices.push_back(mesh.Vertex(index));
        }
        record.firstTriangle =
                static_cast<std::uint32_t>(output.triangles.size());
        record.triangleCount = mesh.TriangleCount();
        for (std::uint32_t index = 0u;
             index < mesh.TriangleCount();
             ++index) {
            const GmSurfMeshTriangle &source = mesh.Triangle(index);
            CudaSceneTriangle triangle;
            triangle.normal = source.normal;
            triangle.planeDistance = source.planeDistance;
            std::copy(source.vertexIndex.begin(),
                      source.vertexIndex.end(),
                      triangle.vertexIndices);
            triangle.material = source.material;
            for (std::uint32_t vertex = 0u; vertex < 3u; ++vertex) {
                triangle.vertices[vertex] =
                        mesh.Vertex(source.vertexIndex[vertex]);
            }
            ClearPadding(triangle);
            output.triangles.push_back(triangle);
        }
        record.firstOctreeCell =
                static_cast<std::uint32_t>(output.octreeCells.size());
        record.octreeCellCount = mesh.OctreeCellCount();
        for (std::uint32_t index = 0u;
             index < mesh.OctreeCellCount();
             ++index) {
            const GmMeshOctreeCell &source = mesh.OctreeCell(index);
            CudaSceneOctreeCell cell;
            cell.bounds = source.Bounds();
            cell.subtreeEntryCount = source.SubtreeEntryCount();
            cell.containsTriangle = source.ContainsTriangle();
            if (cell.containsTriangle) {
                cell.triangleIndex = source.TriangleIndex();
            }
            ClearPadding(cell);
            output.octreeCells.push_back(cell);
        }
        return CudaSceneBuildResult::Success;
    }

    CudaSceneBuildResult AddSurface(
            std::uint32_t actorIndex,
            const CPlugTree &tree,
            const GmIso4 &pose,
            bool collisionPathEnabled) {
        if (output.surfaces.size() >= limits.maximumSurfaces) {
            return CudaSceneBuildResult::SurfaceOverflow;
        }
        const CPlugSurface *surface = tree.Surface();
        const GmSurf *geometry =
                surface != nullptr ? surface->Geometry() : nullptr;
        if (surface == nullptr || geometry == nullptr) {
            return CudaSceneBuildResult::InvalidSource;
        }
        CudaSceneSurface record;
        record.actorIndex = actorIndex;
        record.localToWorld = pose;
        record.worldToLocal.SetInverse(pose);
        geometry->GetBoundingBox(record.localBounds);
        record.worldBounds.SetMult(record.localBounds, pose);
        record.primitiveMaterial = geometry->material;
        record.usesSphereContactBuffer =
                geometry->UsesSphereContactBuffer() != 0;
        record.allowsStaticCollisionRecordAppend =
                collisionPathEnabled &&
                surface->AllowsStaticCollisionRecordAppend() != 0;
        CudaSceneBuildResult result = AddMaterials(*surface, record);
        if (result != CudaSceneBuildResult::Success) {
            return result;
        }
        if (const auto *sphere =
                    dynamic_cast<const GmSurfSphere *>(geometry)) {
            record.type = static_cast<std::uint32_t>(
                    GmSurf::EGmSurfType::Sphere);
            record.sphereRadius = sphere->radius;
        } else if (const auto *ellipsoid =
                           dynamic_cast<const GmSurfEllipsoid *>(
                                   geometry)) {
            record.type = static_cast<std::uint32_t>(
                    GmSurf::EGmSurfType::Ellipsoid);
            record.ellipsoidRadii = ellipsoid->radii;
        } else if (const auto *box =
                           dynamic_cast<const GmSurfBox *>(geometry)) {
            record.type = static_cast<std::uint32_t>(
                    GmSurf::EGmSurfType::Box);
            record.boxCenter = box->center;
            record.boxHalfExtents = box->halfExtents;
        } else if (const auto *polygon =
                           dynamic_cast<const GmSurfPolygon *>(
                                   geometry)) {
            record.type = static_cast<std::uint32_t>(
                    GmSurf::EGmSurfType::Polygon);
            std::copy(polygon->vertices.begin(),
                      polygon->vertices.end(),
                      record.polygonVertices);
            record.polygonVertexCount = polygon->vertexCount;
            record.polygonNormal = polygon->normal;
            record.polygonBackSide = polygon->backSide;
        } else if (const auto *mesh =
                           dynamic_cast<const GmSurfMesh *>(geometry)) {
            record.type = static_cast<std::uint32_t>(
                    GmSurf::EGmSurfType::Mesh);
            result = AddMesh(*mesh, record);
            if (result != CudaSceneBuildResult::Success) {
                return result;
            }
        } else {
            return CudaSceneBuildResult::UnsupportedGeometry;
        }
        ClearPadding(record);
        output.surfaces.push_back(record);
        return CudaSceneBuildResult::Success;
    }

    CudaSceneBuildResult AddTree(
            std::uint32_t actorIndex,
            const CPlugTree &tree,
            const GmIso4 &parentPose,
            bool parentCollisionPathEnabled) {
        GmIso4 pose;
        tree.ComposeCollisionIso(parentPose, pose);
        const bool collisionPathEnabled =
                parentCollisionPathEnabled &&
                tree.HasWorldBox() != 0;
        for (std::uint32_t index = 0u;
             index < tree.GetChildCount();
             ++index) {
            const CPlugTree *child = tree.GetChild(index);
            if (child == nullptr) {
                continue;
            }
            const CudaSceneBuildResult result =
                    AddTree(actorIndex, *child, pose,
                            collisionPathEnabled);
            if (result != CudaSceneBuildResult::Success) {
                return result;
            }
        }
        if (tree.AllowsSurfaceCollision()) {
            const CPlugSurface *surface = tree.Surface();
            if (surface != nullptr && surface->Geometry() != nullptr) {
                const CudaSceneBuildResult result =
                        AddSurface(actorIndex, tree, pose,
                                   collisionPathEnabled);
                if (result != CudaSceneBuildResult::Success) {
                    return result;
                }
            }
        }
        return CudaSceneBuildResult::Success;
    }

    CudaSceneBuildResult AddModel(
            const StaticSceneModel &model,
            std::uint32_t installationOrder) {
        if (output.actors.size() >= limits.maximumActors) {
            return CudaSceneBuildResult::ActorOverflow;
        }
        CudaSceneActor actor;
        actor.worldPose = model.WorldIso();
        actor.itemProperties =
                model.ItemProperties().value_or(defaultProperties);
        if (model.Purpose() == StaticScenePurpose::CheckpointTrigger) {
            actor.itemProperties.collisionStatic = true;
        }
        actor.installationOrder = installationOrder;
        actor.purpose = static_cast<std::uint32_t>(model.Purpose());
        if (model.CheckpointIdentity().has_value()) {
            const auto &identity = *model.CheckpointIdentity();
            actor.hasCheckpoint = true;
            actor.checkpointRole =
                    static_cast<std::uint32_t>(identity.raceRole);
            actor.raceBlockId = identity.raceBlockId;
            actor.respawnUsesCurrentTransform =
                    identity.respawnUsesCurrentTransform;
            if (identity.raceRole == BlockRaceRole::Checkpoint) {
                if (checkpointSlot >= limits.maximumCheckpoints) {
                    return CudaSceneBuildResult::CheckpointOverflow;
                }
                actor.checkpointSlot = checkpointSlot++;
            }
        }
        if (model.CheckpointSpawnIso().has_value()) {
            actor.hasCheckpointSpawn = true;
            actor.checkpointSpawn = *model.CheckpointSpawnIso();
        }
        const std::uint32_t actorIndex =
                static_cast<std::uint32_t>(output.actors.size());
        ClearPadding(actor);
        output.actors.push_back(actor);
        CMwNodRef<CPlugSolid> solid = model.Prototype().CreateInstance();
        if (!solid) {
            return CudaSceneBuildResult::PrototypeConstructionFailed;
        }
        const CPlugTree *tree =
                solid->CollisionTree();
        if (tree == nullptr) {
            return CudaSceneBuildResult::Success;
        }
        return AddTree(actorIndex, *tree, model.WorldIso(), true);
    }
};

std::uint64_t ComputeHash(const CudaHostScene &scene) {
    Hash hash;
    hash.Add(scene.schemaVersion);
    hash.Add(scene.actors);
    hash.Add(scene.surfaces);
    hash.Add(scene.materials);
    hash.Add(scene.vertices);
    hash.Add(scene.triangles);
    hash.Add(scene.octreeCells);
    hash.Add(scene.accelerationGroups);
    hash.Add(scene.accelerationCells);
    return hash.Value();
}

}  // namespace

void CudaHostScene::Clear() noexcept {
    schemaVersion = SchemaVersion;
    deterministicHash = 0u;
    actors.clear();
    surfaces.clear();
    materials.clear();
    vertices.clear();
    triangles.clear();
    octreeCells.clear();
    accelerationGroups = {};
    accelerationCells.clear();
}

bool CudaHostScene::Valid(
        const CudaSceneBuildLimits &limits) const noexcept {
    if (schemaVersion != SchemaVersion ||
        actors.size() > limits.maximumActors ||
        surfaces.size() > limits.maximumSurfaces ||
        materials.size() > limits.maximumMaterials ||
        vertices.size() > limits.maximumVertices ||
        triangles.size() > limits.maximumTriangles ||
        octreeCells.size() > limits.maximumOctreeCells ||
        accelerationCells.size() >
                limits.maximumAccelerationCells) {
        return false;
    }
    for (const CudaSceneAccelerationRange &range :
         accelerationGroups) {
        if (range.firstCell > accelerationCells.size() ||
            range.cellCount >
                    accelerationCells.size() - range.firstCell ||
            range.cellCount == 0u) {
            return false;
        }
    }
    for (const CudaSceneAccelerationCell &cell :
         accelerationCells) {
        if (cell.surfaceIndex != UINT32_MAX &&
            cell.surfaceIndex >= surfaces.size()) {
            return false;
        }
    }
    for (const CudaSceneSurface &surface : surfaces) {
        if (surface.actorIndex >= actors.size() ||
            surface.firstMaterial > materials.size() ||
            surface.materialCount >
                    materials.size() - surface.firstMaterial ||
            surface.firstVertex > vertices.size() ||
            surface.vertexCount >
                    vertices.size() - surface.firstVertex ||
            surface.firstTriangle > triangles.size() ||
            surface.triangleCount >
                    triangles.size() - surface.firstTriangle ||
            surface.firstOctreeCell > octreeCells.size() ||
            surface.octreeCellCount >
                    octreeCells.size() - surface.firstOctreeCell) {
            return false;
        }
    }
    return deterministicHash == ComputeHash(*this);
}

CudaSceneBuildResult BuildCudaHostScene(
        const StaticSceneModelCollection &source,
        CudaHostScene *destination,
        const CudaSceneBuildLimits &limits) noexcept {
    if (destination == nullptr || !source.IsComplete() ||
        source.Empty()) {
        return CudaSceneBuildResult::InvalidSource;
    }
    try {
        CudaHostScene result;
        result.actors.reserve(std::min<std::size_t>(
                source.Models().size(), limits.maximumActors));
        Builder builder{result, limits};
        for (std::uint32_t index = 0u;
             index < source.Models().size();
             ++index) {
            const CudaSceneBuildResult status =
                    builder.AddModel(source.Models()[index], index);
            if (status != CudaSceneBuildResult::Success) {
                return status;
            }
        }
        if (!BuildAcceleration(result, limits)) {
            return CudaSceneBuildResult::AccelerationOverflow;
        }
        result.deterministicHash = ComputeHash(result);
        if (!result.Valid(limits)) {
            return CudaSceneBuildResult::InvalidSource;
        }
        *destination = std::move(result);
        return CudaSceneBuildResult::Success;
    } catch (const std::bad_alloc &) {
        return CudaSceneBuildResult::AllocationFailed;
    }
}

const char *CudaSceneBuildResultName(
        CudaSceneBuildResult result) noexcept {
    switch (result) {
    case CudaSceneBuildResult::Success: return "success";
    case CudaSceneBuildResult::InvalidSource: return "invalid_source";
    case CudaSceneBuildResult::PrototypeConstructionFailed:
        return "prototype_construction_failed";
    case CudaSceneBuildResult::UnsupportedGeometry:
        return "unsupported_geometry";
    case CudaSceneBuildResult::ActorOverflow: return "actor_overflow";
    case CudaSceneBuildResult::SurfaceOverflow: return "surface_overflow";
    case CudaSceneBuildResult::MaterialOverflow:
        return "material_overflow";
    case CudaSceneBuildResult::VertexOverflow: return "vertex_overflow";
    case CudaSceneBuildResult::TriangleOverflow:
        return "triangle_overflow";
    case CudaSceneBuildResult::OctreeOverflow: return "octree_overflow";
    case CudaSceneBuildResult::AccelerationOverflow:
        return "acceleration_overflow";
    case CudaSceneBuildResult::CheckpointOverflow:
        return "checkpoint_overflow";
    case CudaSceneBuildResult::AllocationFailed:
        return "allocation_failed";
    }
    return "unknown";
}

}  // namespace forevervalidator::simulation
