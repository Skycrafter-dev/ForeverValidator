#include "simulation/runtime/replay_simulation_session.h"

#include "engine/game/game_random_sequence.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "engine/core/binary32_math.h"
#include "engine/physics/geometry/gm_surface.h"
#include "engine/physics/geometry/plug_surface.h"
#include "engine/scene/plug_solid.h"
#include "engine/rendering/plug_material.h"
#include "engine/rendering/plug_tree.h"
#include "engine/rendering/plug_visual.h"
#include "format/archive/archive_class_ids.h"
#include "simulation/replay/replay_map_scene.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_binary32_math.h"
#include "simulation/backends/cuda/cuda_scene_layout.h"
#include "simulation/backends/cuda/cuda_scene_storage.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/backends/cuda/cuda_static_configuration_storage.h"
#include "simulation/backends/cuda/cuda_backend.h"
#include "simulation/backends/cuda/cuda_collision_certification.h"
#include "simulation/backends/cuda/cuda_physics_step_certification.h"
#include "simulation/backends/cuda/cuda_state_layout.h"
#include "simulation/backends/cuda/cuda_stunt_certification.h"
#include "simulation/backends/cuda/cuda_vehicle_prefix_certification.h"
#include "simulation/backends/cuda/cuda_vehicle_force_certification.h"
#include "simulation/backends/cuda/cuda_timeline_executor.h"
#if FOREVERVALIDATOR_HAS_CUDA
#include <cuda_runtime.h>
#include "simulation/backends/cuda/cuda_session_specialization.h"
#endif
#include "simulation/backends/speculative_ticking/speculative_ticking_backend.h"
#include "simulation/runtime/replay_environment.h"
#include "simulation/runtime/replay_physics_world.h"
#include "simulation/runtime/replay_simulation_runtime.h"
#include "simulation/runtime/replay_vehicle_body.h"
#include "simulation/runtime/replay_vehicle_simulation.h"
#include "engine/game/trackmania_race.h"
#include "simulation/runtime/replay_deterministic_execution.h"
namespace {

GmVec3 TransformPoint(const GmVec3 &point, const GmIso4 &iso) {
    GmVec3 transformed;
    transformed.SetMult(point, iso);
    return transformed;
}

void AppendTriangle(std::vector<ReplayStaticCollisionTriangle> &triangles,
                    const GmVec3 &a,
                    const GmVec3 &b,
                    const GmVec3 &c,
                    const GmIso4 &iso) {
    triangles.push_back({
            TransformPoint(a, iso),
            TransformPoint(b, iso),
            TransformPoint(c, iso)});
}

void AppendBox(std::vector<ReplayStaticCollisionTriangle> &triangles,
               const GmSurfBox &box,
               const GmIso4 &iso) {
    const GmVec3 &c = box.center;
    const GmVec3 &h = box.halfExtents;
    const std::array<GmVec3, 8u> vertices{{
            {c.x - h.x, c.y - h.y, c.z - h.z},
            {c.x + h.x, c.y - h.y, c.z - h.z},
            {c.x + h.x, c.y + h.y, c.z - h.z},
            {c.x - h.x, c.y + h.y, c.z - h.z},
            {c.x - h.x, c.y - h.y, c.z + h.z},
            {c.x + h.x, c.y - h.y, c.z + h.z},
            {c.x + h.x, c.y + h.y, c.z + h.z},
            {c.x - h.x, c.y + h.y, c.z + h.z},
    }};
    constexpr std::array<std::array<unsigned, 3u>, 12u> Faces{{
            {{0u, 2u, 1u}}, {{0u, 3u, 2u}},
            {{4u, 5u, 6u}}, {{4u, 6u, 7u}},
            {{0u, 1u, 5u}}, {{0u, 5u, 4u}},
            {{3u, 7u, 6u}}, {{3u, 6u, 2u}},
            {{0u, 4u, 7u}}, {{0u, 7u, 3u}},
            {{1u, 2u, 6u}}, {{1u, 6u, 5u}},
    }};
    for (const auto &face : Faces) {
        AppendTriangle(triangles,
                       vertices[face[0]],
                       vertices[face[1]],
                       vertices[face[2]],
                       iso);
    }
}

void AppendEllipsoid(std::vector<ReplayStaticCollisionTriangle> &triangles,
                     const GmVec3 &radii,
                     const GmIso4 &iso) {
    constexpr unsigned Latitudes = 8u;
    constexpr unsigned Longitudes = 12u;
    constexpr float Pi = 3.14159265358979323846f;
    const auto point = [&](unsigned latitude, unsigned longitude) {
        const float phi = -0.5f * Pi +
                Pi * static_cast<float>(latitude) /
                        static_cast<float>(Latitudes);
        const float theta = 2.0f * Pi * static_cast<float>(longitude) /
                static_cast<float>(Longitudes);
        const float ring = std::cos(phi);
        return GmVec3{
                radii.x * ring * std::cos(theta),
                radii.y * std::sin(phi),
                radii.z * ring * std::sin(theta)};
    };
    for (unsigned latitude = 0u; latitude < Latitudes; ++latitude) {
        for (unsigned longitude = 0u; longitude < Longitudes; ++longitude) {
            const unsigned nextLongitude = (longitude + 1u) % Longitudes;
            const GmVec3 a = point(latitude, longitude);
            const GmVec3 b = point(latitude, nextLongitude);
            const GmVec3 c = point(latitude + 1u, nextLongitude);
            const GmVec3 d = point(latitude + 1u, longitude);
            if (latitude != 0u) {
                AppendTriangle(triangles, a, c, b, iso);
            }
            if (latitude + 1u != Latitudes) {
                AppendTriangle(triangles, a, d, c, iso);
            }
        }
    }
}

void AppendSurface(std::vector<ReplayStaticCollisionTriangle> &triangles,
                   const GmSurf &surface,
                   const GmIso4 &iso) {
    if (const auto *mesh = dynamic_cast<const GmSurfMesh *>(&surface)) {
        for (u32 index = 0u; index < mesh->TriangleCount(); ++index) {
            const GmSurfMeshTriangle &triangle = mesh->Triangle(index);
            AppendTriangle(triangles,
                           mesh->Vertex(triangle.vertexIndex[0]),
                           mesh->Vertex(triangle.vertexIndex[1]),
                           mesh->Vertex(triangle.vertexIndex[2]),
                           iso);
        }
        return;
    }
    if (const auto *polygon = dynamic_cast<const GmSurfPolygon *>(&surface)) {
        for (std::uint8_t index = 1u;
             index + 1u < polygon->vertexCount;
             ++index) {
            AppendTriangle(triangles,
                           polygon->vertices[0],
                           polygon->vertices[index],
                           polygon->vertices[index + 1u],
                           iso);
        }
        return;
    }
    if (const auto *box = dynamic_cast<const GmSurfBox *>(&surface)) {
        AppendBox(triangles, *box, iso);
        return;
    }
    if (const auto *ellipsoid =
                dynamic_cast<const GmSurfEllipsoid *>(&surface)) {
        AppendEllipsoid(triangles, ellipsoid->radii, iso);
        return;
    }
    if (const auto *sphere = dynamic_cast<const GmSurfSphere *>(&surface)) {
        AppendEllipsoid(triangles,
                        {sphere->radius, sphere->radius, sphere->radius},
                        iso);
    }
}

void AppendTree(std::vector<ReplayStaticCollisionTriangle> &triangles,
                const CPlugTree &tree,
                const GmIso4 &parentIso) {
    GmIso4 iso;
    tree.ComposeCollisionIso(parentIso, iso);
    if (tree.AllowsSurfaceCollision()) {
        const CPlugSurface *surface = tree.Surface();
        if (surface != nullptr && surface->Geometry() != nullptr) {
            AppendSurface(triangles, *surface->Geometry(), iso);
        }
    }
    for (u32 index = 0u; index < tree.GetChildCount(); ++index) {
        const CPlugTree *child = tree.GetChild(index);
        if (child != nullptr) {
            AppendTree(triangles, *child, iso);
        }
    }
}

bool BuildStaticCollisionTriangles(
        const StaticSceneModelCollection &models,
        std::vector<ReplayStaticCollisionTriangle> &triangles) {
    try {
        for (const StaticSceneModel &model : models.Models()) {
            if (model.Purpose() ==
                    StaticScenePurpose::DedicatedInitialCollision) {
                continue;
            }
            CPlugSolid *solid = model.Prototype().SourceSolid();
            CPlugTree *tree = solid != nullptr ? solid->CollisionTree() : nullptr;
            if (tree != nullptr) {
                AppendTree(triangles, *tree, model.WorldIso());
            }
        }
    } catch (const std::bad_alloc &) {
        triangles.clear();
        return false;
    }
    return true;
}

namespace sandbox = forevervalidator::experimental;

sandbox::PhysicsSandboxScenePurpose PublicPurpose(
        StaticScenePurpose purpose) {
    switch (purpose) {
    case StaticScenePurpose::PlacedBlock:
        return sandbox::PhysicsSandboxScenePurpose::PlacedBlock;
    case StaticScenePurpose::SubMobil:
        return sandbox::PhysicsSandboxScenePurpose::SubMobil;
    case StaticScenePurpose::Clip:
        return sandbox::PhysicsSandboxScenePurpose::Clip;
    case StaticScenePurpose::Helper:
        return sandbox::PhysicsSandboxScenePurpose::Helper;
    case StaticScenePurpose::CheckpointTrigger:
        return sandbox::PhysicsSandboxScenePurpose::CheckpointTrigger;
    case StaticScenePurpose::DedicatedInitialCollision:
        return sandbox::PhysicsSandboxScenePurpose::
                DedicatedInitialCollision;
    case StaticScenePurpose::Pylon:
        return sandbox::PhysicsSandboxScenePurpose::Pylon;
    case StaticScenePurpose::Decoration:
        return sandbox::PhysicsSandboxScenePurpose::Decoration;
    case StaticScenePurpose::Terrain:
        return sandbox::PhysicsSandboxScenePurpose::Terrain;
    case StaticScenePurpose::Generated:
        return sandbox::PhysicsSandboxScenePurpose::Generated;
    case StaticScenePurpose::Environment:
    default:
        return sandbox::PhysicsSandboxScenePurpose::Environment;
    }
}

sandbox::PhysicsSandboxRenderProvenance PublicProvenance(
        const StaticSceneProvenance &source) {
    sandbox::PhysicsSandboxRenderProvenance result;
    result.blockName = source.blockName;
    result.collection = source.collection;
    result.descriptorPath = source.descriptorPath;
    result.sceneObjectId = source.sceneObjectId;
    result.placementIdentity = source.placementIdentity;
    result.blockInstanceId = source.blockInstanceId;
    result.variant = source.variant;
    result.componentIndex = source.componentIndex;
    result.authored = source.authored;
    return result;
}

sandbox::PhysicsSandboxTransform PublicTransform(const GmIso4 &source) {
    return {
            {source.rotation.basisX.x,
             source.rotation.basisX.y,
             source.rotation.basisX.z},
            {source.rotation.basisY.x,
             source.rotation.basisY.y,
             source.rotation.basisY.z},
            {source.rotation.basisZ.x,
             source.rotation.basisZ.y,
             source.rotation.basisZ.z},
            {source.translation.x,
             source.translation.y,
             source.translation.z}};
}

namespace {

struct RenderBounds {
    forevervalidator::Vector3 minimum{};
    forevervalidator::Vector3 maximum{};
    bool valid = false;
};

forevervalidator::Vector3 TransformRenderPoint(
        const sandbox::PhysicsSandboxTransform &transform,
        const forevervalidator::Vector3 &point) {
    return {
            transform.translation.x +
                    transform.basisX.x * point.x +
                    transform.basisY.x * point.y +
                    transform.basisZ.x * point.z,
            transform.translation.y +
                    transform.basisX.y * point.x +
                    transform.basisY.y * point.y +
                    transform.basisZ.y * point.z,
            transform.translation.z +
                    transform.basisX.z * point.x +
                    transform.basisY.z * point.y +
                    transform.basisZ.z * point.z};
}

void IncludePoint(RenderBounds &bounds,
                  const forevervalidator::Vector3 &point) {
    if (!bounds.valid) {
        bounds.minimum = point;
        bounds.maximum = point;
        bounds.valid = true;
        return;
    }
    bounds.minimum.x = std::min(bounds.minimum.x, point.x);
    bounds.minimum.y = std::min(bounds.minimum.y, point.y);
    bounds.minimum.z = std::min(bounds.minimum.z, point.z);
    bounds.maximum.x = std::max(bounds.maximum.x, point.x);
    bounds.maximum.y = std::max(bounds.maximum.y, point.y);
    bounds.maximum.z = std::max(bounds.maximum.z, point.z);
}

void IncludeBounds(RenderBounds &target, const RenderBounds &source) {
    if (!source.valid) {
        return;
    }
    IncludePoint(target, source.minimum);
    IncludePoint(target, source.maximum);
}

RenderBounds InstanceRenderBounds(
        const sandbox::PhysicsSandboxRenderScene &scene,
        const sandbox::PhysicsSandboxRenderInstance &instance) {
    RenderBounds bounds;
    if (instance.meshIndex >= scene.meshes.size()) {
        return bounds;
    }
    const sandbox::PhysicsSandboxRenderMesh &mesh =
            scene.meshes[instance.meshIndex];
    for (unsigned corner = 0u; corner < 8u; ++corner) {
        const forevervalidator::Vector3 local{
                (corner & 1u) != 0u ? mesh.boundsMax.x : mesh.boundsMin.x,
                (corner & 2u) != 0u ? mesh.boundsMax.y : mesh.boundsMin.y,
                (corner & 4u) != 0u ? mesh.boundsMax.z : mesh.boundsMin.z};
        IncludePoint(
                bounds,
                TransformRenderPoint(instance.worldTransform, local));
    }
    return bounds;
}

float MaximumSpan(const RenderBounds &bounds) {
    return std::max({
            bounds.maximum.x - bounds.minimum.x,
            bounds.maximum.y - bounds.minimum.y,
            bounds.maximum.z - bounds.minimum.z});
}

float MinimumSpan(const RenderBounds &bounds) {
    return std::min({
            bounds.maximum.x - bounds.minimum.x,
            bounds.maximum.y - bounds.minimum.y,
            bounds.maximum.z - bounds.minimum.z});
}

bool ContainsBounds(const RenderBounds &outer,
                    const RenderBounds &inner,
                    float tolerance) {
    return outer.valid && inner.valid &&
            outer.minimum.x <= inner.minimum.x + tolerance &&
            outer.minimum.y <= inner.minimum.y + tolerance &&
            outer.minimum.z <= inner.minimum.z + tolerance &&
            outer.maximum.x >= inner.maximum.x - tolerance &&
            outer.maximum.y >= inner.maximum.y - tolerance &&
            outer.maximum.z >= inner.maximum.z - tolerance;
}

bool IsBackgroundPurpose(sandbox::PhysicsSandboxScenePurpose purpose) {
    return purpose == sandbox::PhysicsSandboxScenePurpose::Environment ||
            purpose == sandbox::PhysicsSandboxScenePurpose::Decoration;
}

struct RenderGroupKey {
    sandbox::PhysicsSandboxScenePurpose purpose =
            sandbox::PhysicsSandboxScenePurpose::Environment;
    std::string blockName;
    std::string descriptorPath;
    std::string sceneObjectId;
    std::optional<std::uint64_t> placementIdentity;

    bool operator<(const RenderGroupKey &other) const {
        return std::tie(
                       purpose, blockName, descriptorPath, sceneObjectId,
                       placementIdentity) <
                std::tie(
                       other.purpose, other.blockName,
                       other.descriptorPath, other.sceneObjectId,
                       other.placementIdentity);
    }
};

struct RenderGroup {
    RenderBounds bounds;
    std::vector<std::size_t> instances;
};

}  // namespace

void ClassifyRenderLayers(
        sandbox::PhysicsSandboxRenderScene &scene) {
    RenderBounds foregroundBounds;
    for (const sandbox::PhysicsSandboxRenderInstance &instance :
         scene.instances) {
        if (!instance.visible || instance.lodLevel != 0u ||
            IsBackgroundPurpose(instance.purpose)) {
            continue;
        }
        IncludeBounds(
                foregroundBounds,
                InstanceRenderBounds(scene, instance));
    }
    if (!foregroundBounds.valid) {
        return;
    }

    std::map<RenderGroupKey, RenderGroup> groups;
    for (std::size_t index = 0u; index < scene.instances.size(); ++index) {
        const sandbox::PhysicsSandboxRenderInstance &instance =
                scene.instances[index];
        if (!instance.visible || instance.lodLevel != 0u ||
            !IsBackgroundPurpose(instance.purpose)) {
            continue;
        }
        const sandbox::PhysicsSandboxRenderProvenance &provenance =
                instance.provenance;
        RenderGroup &group = groups[{
                instance.purpose,
                provenance.blockName,
                provenance.descriptorPath,
                provenance.sceneObjectId,
                provenance.placementIdentity}];
        IncludeBounds(group.bounds, InstanceRenderBounds(scene, instance));
        group.instances.push_back(index);
    }

    const float foregroundSpan = MaximumSpan(foregroundBounds);
    if (!(foregroundSpan > 0.0f) || !std::isfinite(foregroundSpan)) {
        return;
    }
    constexpr float BackgroundScaleRatio = 4.0f;
    const float tolerance = foregroundSpan * 0.01f;
    for (const auto &[key, group] : groups) {
        static_cast<void>(key);
        if (!ContainsBounds(group.bounds, foregroundBounds, tolerance) ||
            MaximumSpan(group.bounds) <
                    foregroundSpan * BackgroundScaleRatio ||
            MinimumSpan(group.bounds) <
                    foregroundSpan * BackgroundScaleRatio * 0.5f) {
            continue;
        }
        for (std::size_t index : group.instances) {
            scene.instances[index].renderLayer =
                    sandbox::PhysicsSandboxRenderLayer::Background;
            scene.instances[index].castsShadows = false;
        }
    }
}

std::string PreferredPath(const std::string &selected,
                          const std::string &plain) {
    return !selected.empty() ? selected : plain;
}

class StaticRenderSceneBuilder {
public:
    sandbox::PhysicsSandboxRenderSceneHandle Build(
            const StaticSceneModelCollection &models) {
        scene_ = std::make_shared<sandbox::PhysicsSandboxRenderScene>();
        for (const StaticSceneModel &model : models.Models()) {
            if (model.Purpose() ==
                    StaticScenePurpose::DedicatedInitialCollision) {
                AddDiagnostic(
                        sandbox::PhysicsSandboxRenderDiagnosticCode::
                                CollisionOnlyObjectSkipped,
                        "collision-only object omitted from visual scene",
                        model.Provenance());
                continue;
            }
            CPlugSolid *solid = model.Prototype().SourceSolid();
            CPlugTree *root =
                    solid != nullptr ? solid->CollisionTree() : nullptr;
            if (root == nullptr) {
                AddDiagnostic(
                        sandbox::PhysicsSandboxRenderDiagnosticCode::
                                UnsupportedVisual,
                        "scene model has no visual tree",
                        model.Provenance());
                continue;
            }
            AppendTree(model, *root, model.WorldIso(), nullptr, true, 0u,
                       0.0f);
        }
        ClassifyRenderLayers(*scene_);
        return scene_;
    }

private:
    void AddDiagnostic(
            sandbox::PhysicsSandboxRenderDiagnosticCode code,
            std::string message,
            const StaticSceneProvenance &provenance) {
        scene_->diagnostics.push_back(
                {code, std::move(message), PublicProvenance(provenance)});
    }

    std::uint32_t MaterialIndex(const CPlugMaterial *material) {
        const auto found = materialIndices_.find(material);
        if (found != materialIndices_.end()) {
            return found->second;
        }
        sandbox::PhysicsSandboxRenderMaterial output;
        output.id = scene_->materials.size() + 1u;
        if (material != nullptr) {
            const MaterialRenderDefinition &definition =
                    material->ReplayRenderDefinition();
            output.sourcePath = PreferredPath(
                    definition.MaterialSelectedPath(),
                    definition.MaterialPlainPath());
            output.modelPath = PreferredPath(
                    definition.MaterialModelSelectedPath(),
                    definition.MaterialModelPlainPath());
            output.shaderPath = PreferredPath(
                    definition.ShaderSelectedPath(),
                    definition.ShaderPlainPath());
            output.shaderFlags = definition.ShaderFlags();
            output.surfaceMaterialId =
                    static_cast<std::uint8_t>(
                            material->SurfaceMaterialId());
            output.water = definition.HasBitmapRenderWater();
            for (const MaterialRenderBitmapDefinition &bitmap :
                 definition.Bitmaps()) {
                output.bitmaps.push_back({
                        bitmap.samplerName,
                        PreferredPath(bitmap.selectedPath,
                                      bitmap.plainPath),
                        bitmap.bitmapClassId,
                        bitmap.renderClassId});
                output.cubeMap = output.cubeMap ||
                        bitmap.renderClassId ==
                                TMNF_CLASS_CPlugBitmapRenderCubeMap;
                output.renderTarget = output.renderTarget ||
                        bitmap.renderClassId ==
                                TMNF_CLASS_CPlugBitmapRenderScene3d;
            }
        }
        const std::uint32_t index =
                static_cast<std::uint32_t>(scene_->materials.size());
        scene_->materials.push_back(std::move(output));
        materialIndices_.emplace(material, index);
        return index;
    }

    std::optional<std::uint32_t> MeshIndex(
            CPlugVisual &visual,
            const StaticSceneProvenance &provenance) {
        const auto found = meshIndices_.find(&visual);
        if (found != meshIndices_.end()) {
            return found->second;
        }
        const unsigned long vertexCount = visual.GetTotalVertexCount();
        std::vector<GxVertex> source =
                visual.CanonicalVertices(1, 1, 1);
        if (vertexCount == 0u || source.size() != vertexCount) {
            AddDiagnostic(
                    sandbox::PhysicsSandboxRenderDiagnosticCode::
                            UnsupportedVisual,
                    "visual has no supported vertex stream",
                    provenance);
            return std::nullopt;
        }

        sandbox::PhysicsSandboxRenderMesh mesh;
        mesh.id = scene_->meshes.size() + 1u;
        mesh.vertices.resize(vertexCount);

        GxTexCoordSet uv0;
        GxTexCoordSet uv1;
        mesh.hasUv0 = visual.VStreamOrClassic_GetTexCoordSet(
                uv0, 0u, nullptr) != 0 && uv0.Count() == vertexCount;
        mesh.hasUv1 = visual.VStreamOrClassic_GetTexCoordSet(
                uv1, 1u, nullptr) != 0 && uv1.Count() == vertexCount;
        const auto *visual3d = dynamic_cast<const CPlugVisual3D *>(&visual);
        mesh.hasTangents = visual3d != nullptr &&
                visual3d->TangentCount() == vertexCount;
        mesh.hasNormals = visual.HasVertexNormal();
        mesh.hasVertexColors = visual.HasVertexColor();

        for (u32 index = 0u; index < vertexCount; ++index) {
            const GxVertex &input = source[index];
            auto &output = mesh.vertices[index];
            output.position = {
                    input.position.x, input.position.y, input.position.z};
            output.normal = {
                    input.normal.x, input.normal.y, input.normal.z};
            output.color = {
                    input.color[0], input.color[1],
                    input.color[2], input.color[3]};
            if (mesh.hasTangents) {
                const GmVec3 &tangent = visual3d->tangents[index];
                output.tangent = {
                        tangent.x, tangent.y, tangent.z, 1.0f};
            }
            if (mesh.hasUv0) {
                const GxTexCoord4 uv = uv0.Coordinate4At(index);
                output.uv0 = {uv.u, uv.v};
            }
            if (mesh.hasUv1) {
                const GxTexCoord4 uv = uv1.Coordinate4At(index);
                output.uv1 = {uv.u, uv.v};
            }
        }

        unsigned long indexCount = 0u;
        unsigned short *indices = nullptr;
        visual.GetVertexIndexation(indexCount, indices);
        if (indexCount == 0u) {
            indexCount = vertexCount;
            mesh.indices.reserve(indexCount);
            for (u32 index = 0u; index < indexCount; ++index) {
                mesh.indices.push_back(index);
            }
        } else if (indices == nullptr) {
            AddDiagnostic(
                    sandbox::PhysicsSandboxRenderDiagnosticCode::
                            InvalidTopology,
                    "visual index stream has no backing storage",
                    provenance);
            return std::nullopt;
        } else {
            mesh.indices.reserve(indexCount);
            for (u32 index = 0u; index < indexCount; ++index) {
                mesh.indices.push_back(indices[index]);
            }
        }
        bool topologyValid = mesh.indices.size() % 3u == 0u;
        for (std::uint32_t index : mesh.indices) {
            topologyValid = topologyValid && index < vertexCount;
        }
        if (!topologyValid) {
            AddDiagnostic(
                    sandbox::PhysicsSandboxRenderDiagnosticCode::
                            InvalidTopology,
                    "visual index stream is not a valid triangle list",
                    provenance);
            return std::nullopt;
        }

        const GmBoxAligned &box = visual.BoundingBox();
        mesh.boundsMin = {
                box.center.x - std::fabs(box.halfExtents.x),
                box.center.y - std::fabs(box.halfExtents.y),
                box.center.z - std::fabs(box.halfExtents.z)};
        mesh.boundsMax = {
                box.center.x + std::fabs(box.halfExtents.x),
                box.center.y + std::fabs(box.halfExtents.y),
                box.center.z + std::fabs(box.halfExtents.z)};
        mesh.subsets.push_back(
                {0u, static_cast<std::uint32_t>(mesh.indices.size()), 0u});
        const std::uint32_t result =
                static_cast<std::uint32_t>(scene_->meshes.size());
        scene_->meshes.push_back(std::move(mesh));
        meshIndices_.emplace(&visual, result);
        return result;
    }

    void AppendTree(
            const StaticSceneModel &model,
            CPlugTree &tree,
            const GmIso4 &parentIso,
            const CPlugMaterial *inheritedMaterial,
            bool inheritedVisibility,
            std::uint32_t lodLevel,
            float lodFarDistance) {
        GmIso4 worldIso;
        tree.ComposeCollisionIso(parentIso, worldIso);
        const bool visible = inheritedVisibility && tree.IsVisible();
        const CPlugMaterial *material =
                tree.Material() != nullptr
                ? tree.Material()
                : inheritedMaterial;
        if (CPlugVisual *visual = tree.Visual()) {
            const std::optional<std::uint32_t> meshIndex =
                    MeshIndex(*visual, model.Provenance());
            if (meshIndex.has_value()) {
                sandbox::PhysicsSandboxRenderInstance instance;
                instance.id = scene_->instances.size() + 1u;
                instance.meshIndex = *meshIndex;
                instance.materialIndex = MaterialIndex(material);
                instance.worldTransform = PublicTransform(worldIso);
                instance.provenance =
                        PublicProvenance(model.Provenance());
                instance.purpose = PublicPurpose(model.Purpose());
                instance.lodLevel = lodLevel;
                instance.lodFarDistance = lodFarDistance;
                instance.visible = visible;
                instance.castsShadows = tree.IsShadowCaster();
                scene_->instances.push_back(std::move(instance));
                if (!scene_->meshes[*meshIndex].hasUv0) {
                    AddDiagnostic(
                            sandbox::PhysicsSandboxRenderDiagnosticCode::
                                    MissingUv,
                            "visual has no authored UV0 stream",
                            model.Provenance());
                }
                if (!scene_->meshes[*meshIndex].hasTangents) {
                    AddDiagnostic(
                            sandbox::PhysicsSandboxRenderDiagnosticCode::
                                    MissingTangent,
                            "visual has no authored tangent stream",
                            model.Provenance());
                }
                if (material == nullptr) {
                    AddDiagnostic(
                            sandbox::PhysicsSandboxRenderDiagnosticCode::
                                    MissingMaterial,
                            "visual has no final material assignment",
                            model.Provenance());
                }
            }
        }

        auto *mip = dynamic_cast<CPlugTreeVisualMip *>(&tree);
        for (u32 childIndex = 0u;
             childIndex < tree.GetChildCount();
             ++childIndex) {
            CPlugTree *child = tree.GetChild(childIndex);
            if (child == nullptr) {
                continue;
            }
            std::uint32_t childLod = lodLevel;
            float childFar = lodFarDistance;
            if (mip != nullptr) {
                for (u32 level = 0u; level < mip->LevelCount(); ++level) {
                    if (mip->LevelTree(level) == child) {
                        childLod = level;
                        childFar = mip->LevelFarZ(level);
                        break;
                    }
                }
            }
            AppendTree(model, *child, worldIso, material, visible,
                       childLod, childFar);
        }
    }

    std::shared_ptr<sandbox::PhysicsSandboxRenderScene> scene_;
    std::unordered_map<const CPlugVisual *, std::uint32_t> meshIndices_;
    std::unordered_map<const CPlugMaterial *, std::uint32_t>
            materialIndices_;
};

sandbox::PhysicsSandboxRenderSceneHandle BuildStaticRenderScene(
        const StaticSceneModelCollection &models) {
    try {
        return StaticRenderSceneBuilder().Build(models);
    } catch (const std::bad_alloc &) {
        return {};
    }
}

ReplayTrajectoryObservation ObserveReplayTrajectory(
        const ReplaySimulationStepExecution &execution,
        const ReplayControlTick &tick) {
    ReplayTrajectoryObservation observation;
    observation.simulatedPosition = execution.simulatedFrame.position;
    observation.writePosition = execution.writeFrame.position;
    observation.finishTickMs = execution.finishTickMs;
    observation.finishTime = execution.finishTime;
    if (!tick.comparisonTarget.has_value()) {
        return observation;
    }

    ReplayTrajectoryDeviation comparison;
    comparison.targetPosition = *tick.comparisonTarget;
    comparison.delta = {
            observation.writePosition.x - comparison.targetPosition.x,
            observation.writePosition.y - comparison.targetPosition.y,
            observation.writePosition.z - comparison.targetPosition.z};
    const float horizontalDistanceSquared =
            comparison.delta.x * comparison.delta.x +
            comparison.delta.y * comparison.delta.y;
    comparison.distance = CIsqrt(
            horizontalDistanceSquared +
            comparison.delta.z * comparison.delta.z);
    observation.comparison = comparison;
    return observation;
}

ReplayTrajectoryObservation ObserveCudaTrajectory(
        const forevervalidator::simulation::
                CudaTimelineObservation &source) {
    ReplayTrajectoryObservation result;
    result.simulatedPosition = source.simulatedPosition;
    result.writePosition = source.writePosition;
    if (source.hasComparison) {
        result.comparison = ReplayTrajectoryDeviation{
                source.comparisonTarget,
                source.comparisonDelta,
                source.comparisonDistance};
    }
    if (source.hasFinishTick) {
        result.finishTickMs = source.finishTickMs;
    }
    return result;
}

#if FOREVERVALIDATOR_HAS_CUDA
ReplayControlTick CandidateControlTick(
        const ReplayControlTick &source,
        std::uint32_t candidate,
        bool mutateControls) {
    ReplayControlTick result = source;
    if (mutateControls && candidate != 0u) {
        const float offset =
                static_cast<float>(
                        static_cast<std::int32_t>(candidate % 9u) - 4) *
                0.015625f;
        result.controls.steering =
                std::clamp(
                        result.controls.steering + offset,
                        -1.0f, 1.0f);
    }
    return result;
}
#endif

}  // namespace

void ClassifyPhysicsSandboxRenderLayers(
        sandbox::PhysicsSandboxRenderScene &scene) {
    ClassifyRenderLayers(scene);
}

struct ReplaySimulationInstance {
    CTrackManiaRace race;
    std::unique_ptr<ReplaySimulationRuntime> runtime;
    std::uint32_t incrementalRespawnCount = 0u;

    void ResetRuntime() {
        runtime.reset();
        incrementalRespawnCount = 0u;
    }
};

struct ReplaySimulationSession::Impl {
    explicit Impl(forevervalidator::SimulationBackend requestedBackend)
        : backend(requestedBackend) {}

    forevervalidator::SimulationBackend backend;
    ReplayMapScene mapScene;
    ReplaySimulationInstance instance;
    std::shared_ptr<const std::vector<ReplayStaticCollisionTriangle>>
            staticCollisionTriangles;
    sandbox::PhysicsSandboxRenderSceneHandle staticRenderScene;
    forevervalidator::simulation::CudaHostScene cudaHostScene;
    forevervalidator::simulation::CudaDeviceScene cudaDeviceScene;
    std::optional<
            forevervalidator::simulation::CudaSceneTransferMetrics>
            cudaSceneTransfer;
    forevervalidator::simulation::CudaHostStaticConfiguration
            cudaHostConfiguration;
    forevervalidator::simulation::CudaDeviceStaticConfiguration
            cudaDeviceConfiguration;
    std::optional<forevervalidator::simulation::
                          CudaStaticConfigurationTransferMetrics>
            cudaConfigurationTransfer;
    std::string cudaInitializationDiagnostic;
    std::shared_ptr<forevervalidator::simulation::cuda::specialization::
                            SessionModule>
            cudaSearchSpecialization;
    std::string cudaSearchSpecializationDiagnostic;
    bool cudaSearchSpecializationAttempted = false;
    std::optional<forevervalidator::simulation::
                          CudaTimelineExecutionMetrics>
            cudaTimelineMetrics;
    std::uint32_t incrementalValidationSeed = 0u;

    void ResetRuntime() { instance.ResetRuntime(); }

    ReplaySimulationRunResult PrepareCudaConfiguration(
            const ReplaySimulationDefinition &definition) {
        if (backend != forevervalidator::SimulationBackend::Cuda) {
            return ReplaySimulationRunResult::Success;
        }
        if (!forevervalidator::simulation::
                    QueryCudaRuntimeDiagnostics().IsReady()) {
            cudaInitializationDiagnostic =
                    forevervalidator::simulation::
                            QueryCudaRuntimeDiagnostics().diagnostic;
            return ReplaySimulationRunResult::CudaUnavailable;
        }
        if (!cudaSceneTransfer.has_value() ||
            !cudaSceneTransfer->success || !cudaDeviceScene.Ready()) {
            cudaInitializationDiagnostic =
                    cudaSceneTransfer.has_value()
                    ? cudaSceneTransfer->diagnostic
                    : "CUDA immutable scene was not uploaded";
            return ReplaySimulationRunResult::CudaInitializationFailed;
        }
        forevervalidator::simulation::CudaHostStaticConfiguration built;
        const auto buildResult = forevervalidator::simulation::
                BuildCudaHostStaticConfiguration(definition, &built);
        if (buildResult !=
            forevervalidator::simulation::
                    CudaStaticConfigurationBuildResult::Success) {
            cudaInitializationDiagnostic =
                    "CUDA vehicle/environment flattening failed with code " +
                    std::to_string(
                            static_cast<unsigned>(buildResult));
            return ReplaySimulationRunResult::CudaInitializationFailed;
        }
        if (cudaDeviceConfiguration.Ready() &&
            cudaDeviceConfiguration.ConfigurationHash() ==
                    built.deterministicHash) {
            cudaHostConfiguration = std::move(built);
            return ReplaySimulationRunResult::Success;
        }
        cudaConfigurationTransfer =
                cudaDeviceConfiguration.Upload(built);
        if (!cudaConfigurationTransfer->success) {
            cudaInitializationDiagnostic =
                    cudaConfigurationTransfer->diagnostic;
            cudaHostConfiguration.Clear();
            return ReplaySimulationRunResult::CudaInitializationFailed;
        }
        cudaHostConfiguration = std::move(built);
        cudaInitializationDiagnostic =
                "CUDA immutable scene and configuration are ready";
        return ReplaySimulationRunResult::Success;
    }

    ReplaySimulationTimelineResult ExecuteCudaTimeline(
            const std::vector<ReplayControlTick> &controlTicks,
            std::uint32_t validationSeed,
            std::uint64_t controlCursor) {
        ReplaySimulationTimelineResult result;
        if (!instance.runtime || !cudaDeviceScene.Ready() ||
            !cudaDeviceConfiguration.Ready()) {
            cudaInitializationDiagnostic =
                    "CUDA timeline prerequisites are not ready";
            result.result =
                    ReplaySimulationRunResult::CudaInitializationFailed;
            return result;
        }
        std::optional<ReplaySimulationRuntime::RuntimeClone> runtime =
                instance.runtime->CaptureRuntimeClone();
        if (!runtime.has_value()) {
            cudaInitializationDiagnostic =
                    "CUDA initial runtime state capture failed";
            result.result =
                    ReplaySimulationRunResult::CudaInitializationFailed;
            return result;
        }
        ReplaySimulationInstanceClone clone;
        clone.race = instance.race.CaptureRuntimeClone();
        clone.runtime = std::move(*runtime);
        clone.incrementalRespawnCount =
                instance.incrementalRespawnCount;

        forevervalidator::simulation::CudaCandidateTimelineInput input;
        const auto conversion = forevervalidator::simulation::
                EncodeCudaCandidateState(
                        clone, validationSeed, controlCursor, 0u,
                        tmnf::simulation::CaptureGameRandomState(),
                        &input.initialState);
        if (conversion != forevervalidator::simulation::
                                  CudaStateConversionResult::Success) {
            cudaInitializationDiagnostic =
                    "CUDA initial state conversion failed with code " +
                    std::to_string(
                            static_cast<unsigned>(conversion));
            result.result =
                    ReplaySimulationRunResult::CudaInitializationFailed;
            return result;
        }
        try {
            input.ticks.reserve(controlTicks.size());
            for (const ReplayControlTick &tick : controlTicks) {
                input.ticks.push_back(
                        forevervalidator::simulation::
                                FlattenCudaControlTick(tick));
            }
        } catch (const std::bad_alloc &) {
            cudaInitializationDiagnostic =
                    "CUDA control timeline allocation failed";
            result.result =
                    ReplaySimulationRunResult::CudaInitializationFailed;
            return result;
        }

        forevervalidator::simulation::CudaTimelineBatchResult executed =
                forevervalidator::simulation::ExecuteCudaTimelineBatch(
                        cudaDeviceScene.DeviceData(),
                        cudaDeviceConfiguration.DeviceData(),
                        {std::move(input)});
        cudaTimelineMetrics = executed.metrics;
        cudaInitializationDiagnostic = executed.diagnostic;
        if (executed.status != forevervalidator::simulation::
                                       CudaTimelineStatus::Success ||
            executed.candidates.size() != 1u ||
            executed.candidates[0].status !=
                    forevervalidator::simulation::
                            CudaTimelineStatus::Success) {
            if (executed.candidates.size() == 1u) {
                cudaInitializationDiagnostic +=
                        " candidate_status=" +
                        std::string(
                                forevervalidator::simulation::
                                        CudaTimelineStatusName(
                                                executed.candidates[0].
                                                        status)) +
                        " failure_tick=" +
                        std::to_string(
                                executed.candidates[0].failureTick) +
                        " executed_ticks=" +
                        std::to_string(
                                executed.candidates[0].
                                        executedTickCount) +
                        " failure_detail=" +
                        std::to_string(
                                executed.candidates[0].
                                        failureDetail);
                const std::uint32_t failure =
                        executed.candidates[0].failureTick;
                if (failure < controlTicks.size()) {
                    const auto flattened =
                            forevervalidator::simulation::
                                    FlattenCudaControlTick(
                                            controlTicks[failure]);
                    cudaInitializationDiagnostic +=
                            " action_flags=" +
                            std::to_string(
                                    flattened.actionFlags) +
                            " respawns=" +
                            std::to_string(
                                    flattened.
                                            respawnAtCheckpointCount);
                }
            }
            result.result =
                    ReplaySimulationRunResult::CudaExecutionFailed;
            return result;
        }

        ReplaySimulationInstanceClone restored;
        const auto decode = forevervalidator::simulation::
                DecodeCudaCandidateState(
                        executed.candidates[0].finalState, &restored);
        if (decode != forevervalidator::simulation::
                              CudaStateConversionResult::Success ||
            !instance.race.PrepareRuntimeCloneRestore(restored.race) ||
            !instance.runtime->PrepareRuntimeCloneRestore(
                    restored.runtime)) {
            cudaInitializationDiagnostic =
                    "CUDA final state restoration validation failed";
            result.result =
                    ReplaySimulationRunResult::CudaExecutionFailed;
            return result;
        }
        instance.race.RestoreRuntimeClone(std::move(restored.race));
        instance.runtime->RestoreRuntimeClone(
                std::move(restored.runtime));
        instance.incrementalRespawnCount =
                restored.incrementalRespawnCount;
        tmnf::simulation::RestoreGameRandomState(
                executed.candidates[0].finalState.randomState);
        try {
            result.observations.reserve(
                    executed.candidates[0].observations.size());
            for (const auto &observation :
                 executed.candidates[0].observations) {
                result.observations.push_back(
                        ObserveCudaTrajectory(observation));
            }
        } catch (const std::bad_alloc &) {
            result.result =
                    ReplaySimulationRunResult::
                            ObservationAllocationFailed;
            return result;
        }
        result.executedRespawnCount =
                executed.candidates[0].executedRespawnCount;
        result.finishTimeMs = instance.runtime->FinishTimeMs();
        result.finishTime = instance.runtime->FinishTime();
        result.stuntsScore = instance.runtime->StuntsScore();
        result.raceCompleted = result.finishTimeMs.has_value();
        result.result = ReplaySimulationRunResult::Success;
        return result;
    }
};

ReplaySimulationSession::ReplaySimulationSession(
        forevervalidator::SimulationBackend backend)
    : impl(std::make_unique<Impl>(backend)) {}

ReplaySimulationSession::~ReplaySimulationSession() = default;

std::unique_ptr<ReplaySimulationSession>
ReplaySimulationSession::ClonePrepared() const {
    try {
        auto clone = std::make_unique<ReplaySimulationSession>(impl->backend);
        if (!clone->impl->mapScene.ClonePreparedFrom(impl->mapScene)) {
            return {};
        }
        clone->impl->staticCollisionTriangles =
                impl->staticCollisionTriangles;
        clone->impl->staticRenderScene = impl->staticRenderScene;
        clone->impl->incrementalValidationSeed =
                impl->incrementalValidationSeed;
        return clone;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

void ReplaySimulationSession::Reset() {
    impl->ResetRuntime();
    impl->mapScene.Reset(impl->instance.race);
    impl->staticCollisionTriangles.reset();
    impl->staticRenderScene.reset();
    impl->cudaSearchSpecialization.reset();
    impl->cudaSearchSpecializationDiagnostic.clear();
    impl->cudaSearchSpecializationAttempted = false;
    impl->cudaHostScene.Clear();
    impl->cudaDeviceScene.Reset();
    impl->cudaSceneTransfer.reset();
    impl->cudaHostConfiguration.Clear();
    impl->cudaDeviceConfiguration.Reset();
    impl->cudaConfigurationTransfer.reset();
    impl->cudaInitializationDiagnostic.clear();
    impl->cudaTimelineMetrics.reset();
    impl->incrementalValidationSeed = 0u;
}

bool ReplaySimulationSession::PreloadChallenge(
        CGameCtnChallengeConstruction &construction) {
    return impl->mapScene.PreloadChallenge(construction) ==
           ReplayMapSceneResult::Ready;
}

bool ReplaySimulationSession::InstallStaticScene(
        StaticSceneModelCollection models) {
    std::vector<ReplayStaticCollisionTriangle> triangles;
    sandbox::PhysicsSandboxRenderSceneHandle renderScene =
            BuildStaticRenderScene(models);
    forevervalidator::simulation::CudaHostScene cudaScene;
    if (impl->backend == forevervalidator::SimulationBackend::Cuda) {
        const auto cudaBuild =
                forevervalidator::simulation::BuildCudaHostScene(
                        models, &cudaScene);
        if (cudaBuild != forevervalidator::simulation::
                                 CudaSceneBuildResult::Success) {
            impl->cudaInitializationDiagnostic =
                    std::string("CUDA scene flattening failed: ") +
                    forevervalidator::simulation::
                            CudaSceneBuildResultName(cudaBuild);
            return false;
        }
    }
    if (!renderScene ||
        !BuildStaticCollisionTriangles(models, triangles) ||
        impl->mapScene.InstallModels(std::move(models)) !=
                ReplayMapSceneResult::Ready) {
        return false;
    }
    try {
        impl->staticCollisionTriangles =
                std::make_shared<const std::vector<
                        ReplayStaticCollisionTriangle>>(
                        std::move(triangles));
    } catch (const std::bad_alloc &) {
        return false;
    }
    impl->staticRenderScene = std::move(renderScene);
    impl->cudaHostScene = std::move(cudaScene);
    return true;
}

void ReplaySimulationSession::ActivateStaticScene() {
    impl->mapScene.Activate();
    if (impl->backend == forevervalidator::SimulationBackend::Cuda) {
        impl->cudaSceneTransfer =
                impl->cudaDeviceScene.Upload(impl->cudaHostScene);
        impl->cudaInitializationDiagnostic =
                impl->cudaSceneTransfer->diagnostic;
    }
}

void ReplaySimulationSession::ConfigureReplayRace(
        EChallengePlayMode playMode,
        bool isLapRace,
        std::uint32_t lapCount) {
    impl->instance.race.SetReplayChallengePlayMode(playMode);
    impl->instance.race.InitNbLapsAndCheckpoints(
            isLapRace ? lapCount : 1u);
}

const std::vector<ReplayStaticCollisionTriangle> &
ReplaySimulationSession::StaticCollisionTriangles() const noexcept {
    static const std::vector<ReplayStaticCollisionTriangle> empty;
    return impl->staticCollisionTriangles
            ? *impl->staticCollisionTriangles
            : empty;
}

sandbox::PhysicsSandboxRenderSceneHandle
ReplaySimulationSession::StaticRenderScene() const noexcept {
    return impl->staticRenderScene;
}

ReplaySimulationTimelineResult ReplaySimulationSession::SimulateTimeline(
        const ReplaySimulationDefinition &simulationDefinition,
        const std::vector<ReplayControlTick> &controlTicks,
        std::uint32_t validationSeed) {
    ReplaySimulationTimelineResult result;
    if (controlTicks.empty()) {
        return result;
    }

    if (!tmnf::simulation::DeterministicExecutionScope::IsActive()) {
        result.result =
                ReplaySimulationRunResult::DeterministicExecutionUnavailable;
        return result;
    }
    const std::size_t observationCount = static_cast<std::size_t>(
            std::count_if(
                    controlTicks.begin(),
                    controlTicks.end(),
                    [](const ReplayControlTick &tick) {
                        return tick.observe;
                    }));
    if (observationCount != 0u) {
        try {
            result.observations.reserve(observationCount);
        } catch (const std::bad_alloc &) {
            result.result =
                    ReplaySimulationRunResult::ObservationAllocationFailed;
            return result;
        }
    }
    impl->ResetRuntime();

    result.result = impl->PrepareCudaConfiguration(simulationDefinition);
    if (result.result != ReplaySimulationRunResult::Success) {
        return result;
    }

    const ReplayMapSceneResult readyResult =
            impl->mapScene.EnsureReady(impl->instance.race);
    if (readyResult != ReplayMapSceneResult::Ready) {
        result.result = MapReplaySceneResult(readyResult);
        return result;
    }
    GmIso4 startLocation;
    if (!impl->mapScene.FirstStartLineSpawnLocation(startLocation)) {
        result.result = ReplaySimulationRunResult::MapStartUnavailable;
        return result;
    }

    impl->instance.runtime = std::make_unique<ReplaySimulationRuntime>(
            impl->instance.race, impl->backend);
    result.result = impl->instance.runtime->Start(
            simulationDefinition,
            impl->mapScene,
            startLocation,
            controlTicks.front(),
            validationSeed);
    if (result.result != ReplaySimulationRunResult::Success) {
        return result;
    }

    if (impl->backend == forevervalidator::SimulationBackend::Cuda) {
        return impl->ExecuteCudaTimeline(
                controlTicks, validationSeed, 0u);
    } else if (impl->backend ==
               forevervalidator::SimulationBackend::OptimizedCpu) {
        impl->instance.runtime->PrepareOptimizedCpuStaticTransforms();
        impl->instance.runtime->
                CertifyOptimizedCpuStaticTransformsForAdvance();
        const forevervalidator::simulation::OptimizedCpuBinary32MathPath
                binary32MathPath = forevervalidator::simulation::
                        SelectOptimizedCpuBinary32MathPathForActiveExecution();
        if (binary32MathPath == forevervalidator::simulation::
                                        OptimizedCpuBinary32MathPath::X86Sse2) {
            for (const ReplayControlTick &tick : controlTicks) {
                const ReplaySimulationStepExecution execution =
                        impl->instance.runtime->
                                StepOptimizedCpuNativeBinary32(tick);
                if (execution.result != ReplaySimulationRunResult::Success) {
                    result.result = execution.result;
                    return result;
                }
                result.executedRespawnCount +=
                        execution.respawnExecutedCount;

                if (tick.observe) {
                    try {
                        result.observations.push_back(
                                ObserveReplayTrajectory(execution, tick));
                    } catch (const std::bad_alloc &) {
                        result.result =
                                ReplaySimulationRunResult::ObservationAllocationFailed;
                        return result;
                    }
                }
            }
        } else {
            for (const ReplayControlTick &tick : controlTicks) {
                const ReplaySimulationStepExecution execution =
                        impl->instance.runtime->StepOptimizedCpu(tick);
                if (execution.result != ReplaySimulationRunResult::Success) {
                    result.result = execution.result;
                    return result;
                }
                result.executedRespawnCount +=
                        execution.respawnExecutedCount;

                if (tick.observe) {
                    try {
                        result.observations.push_back(
                                ObserveReplayTrajectory(execution, tick));
                    } catch (const std::bad_alloc &) {
                        result.result =
                                ReplaySimulationRunResult::ObservationAllocationFailed;
                        return result;
                    }
                }
            }
        }
    } else if (impl->backend ==
               forevervalidator::SimulationBackend::SpeculativeTicking) {
        forevervalidator::simulation::PrepareSpeculativeTicking(
                *impl->instance.runtime);
        forevervalidator::simulation::CertifySpeculativeTickingForAdvance(
                *impl->instance.runtime);
        for (const ReplayControlTick &tick : controlTicks) {
            const ReplaySimulationStepExecution execution =
                    forevervalidator::simulation::StepSpeculativeTicking(
                            *impl->instance.runtime, tick);
            if (execution.result != ReplaySimulationRunResult::Success) {
                result.result = execution.result;
                return result;
            }
            result.executedRespawnCount += execution.respawnExecutedCount;

            if (tick.observe) {
                try {
                    result.observations.push_back(
                            ObserveReplayTrajectory(execution, tick));
                } catch (const std::bad_alloc &) {
                    result.result =
                            ReplaySimulationRunResult::ObservationAllocationFailed;
                    return result;
                }
            }
        }
    } else {
        for (const ReplayControlTick &tick : controlTicks) {
            const ReplaySimulationStepExecution execution =
                    impl->instance.runtime->Step(tick);
            if (execution.result != ReplaySimulationRunResult::Success) {
                result.result = execution.result;
                return result;
            }
            result.executedRespawnCount +=
                    execution.respawnExecutedCount;

            if (tick.observe) {
                try {
                    result.observations.push_back(
                            ObserveReplayTrajectory(execution, tick));
                } catch (const std::bad_alloc &) {
                    result.result =
                            ReplaySimulationRunResult::ObservationAllocationFailed;
                    return result;
                }
            }
        }
    }
    result.finishTimeMs = impl->instance.runtime->FinishTimeMs();
    result.finishTime = impl->instance.runtime->FinishTime();
    result.stuntsScore = impl->instance.runtime->StuntsScore();
    result.raceCompleted = result.finishTimeMs.has_value();
    result.result = ReplaySimulationRunResult::Success;
    return result;
}

ReplaySimulationRunResult ReplaySimulationSession::StartIncremental(
        const ReplaySimulationDefinition &simulationDefinition,
        const ReplayControlTick &firstTick,
        std::uint32_t validationSeed) {
    if (!tmnf::simulation::DeterministicExecutionScope::IsActive()) {
        return ReplaySimulationRunResult::DeterministicExecutionUnavailable;
    }
    impl->ResetRuntime();
    const ReplaySimulationRunResult cudaPreparation =
            impl->PrepareCudaConfiguration(simulationDefinition);
    if (cudaPreparation != ReplaySimulationRunResult::Success) {
        return cudaPreparation;
    }
    const ReplayMapSceneResult readyResult =
            impl->mapScene.EnsureReady(impl->instance.race);
    if (readyResult != ReplayMapSceneResult::Ready) {
        return MapReplaySceneResult(readyResult);
    }
    GmIso4 startLocation;
    if (!impl->mapScene.FirstStartLineSpawnLocation(startLocation)) {
        return ReplaySimulationRunResult::MapStartUnavailable;
    }
    impl->instance.runtime = std::make_unique<ReplaySimulationRuntime>(
            impl->instance.race, impl->backend);
    const ReplaySimulationRunResult result = impl->instance.runtime->Start(
            simulationDefinition,
            impl->mapScene,
            startLocation,
            firstTick,
            validationSeed);
    if (result == ReplaySimulationRunResult::Success) {
        impl->incrementalValidationSeed = validationSeed;
        if (impl->backend ==
            forevervalidator::SimulationBackend::SpeculativeTicking) {
            forevervalidator::simulation::PrepareSpeculativeTicking(
                    *impl->instance.runtime);
        } else if (impl->backend ==
                   forevervalidator::SimulationBackend::OptimizedCpu) {
            impl->instance.runtime->PrepareOptimizedCpuStaticTransforms();
        }
    }
    return result;
}

ReplaySimulationTimelineResult ReplaySimulationSession::AdvanceIncremental(
        const std::vector<ReplayControlTick> &controlTicks,
        std::size_t begin,
        std::size_t count) {
    ReplaySimulationTimelineResult result;
    if (!impl->instance.runtime || begin > controlTicks.size() ||
        count > controlTicks.size() - begin) {
        return result;
    }
    if (impl->backend == forevervalidator::SimulationBackend::OptimizedCpu) {
        impl->instance.runtime->
                CertifyOptimizedCpuStaticTransformsForAdvance();
        const forevervalidator::simulation::OptimizedCpuBinary32MathPath
                binary32MathPath = forevervalidator::simulation::
                        SelectOptimizedCpuBinary32MathPathForActiveExecution();
        if (binary32MathPath == forevervalidator::simulation::
                                        OptimizedCpuBinary32MathPath::X86Sse2) {
            for (std::size_t index = begin; index < begin + count; ++index) {
                const ReplayControlTick &tick = controlTicks[index];
                const ReplaySimulationStepExecution execution =
                        impl->instance.runtime->
                                StepOptimizedCpuNativeBinary32(tick);
                if (execution.result != ReplaySimulationRunResult::Success) {
                    result.result = execution.result;
                    return result;
                }
                impl->instance.incrementalRespawnCount +=
                        execution.respawnExecutedCount;
            }
        } else {
            for (std::size_t index = begin; index < begin + count; ++index) {
                const ReplayControlTick &tick = controlTicks[index];
                const ReplaySimulationStepExecution execution =
                        impl->instance.runtime->StepOptimizedCpu(tick);
                if (execution.result != ReplaySimulationRunResult::Success) {
                    result.result = execution.result;
                    return result;
                }
                impl->instance.incrementalRespawnCount +=
                        execution.respawnExecutedCount;
            }
        }
    } else if (impl->backend == forevervalidator::SimulationBackend::Cuda) {
        std::vector<ReplayControlTick> cudaTicks;
        try {
            cudaTicks.assign(
                    controlTicks.begin() + begin,
                    controlTicks.begin() + begin + count);
        } catch (const std::bad_alloc &) {
            result.result =
                    ReplaySimulationRunResult::
                            ObservationAllocationFailed;
            return result;
        }
        return impl->ExecuteCudaTimeline(
                cudaTicks, impl->incrementalValidationSeed, begin);
    } else if (impl->backend ==
               forevervalidator::SimulationBackend::SpeculativeTicking) {
        forevervalidator::simulation::CertifySpeculativeTickingForAdvance(
                *impl->instance.runtime);
        for (std::size_t index = begin; index < begin + count; ++index) {
            const ReplayControlTick &tick = controlTicks[index];
            const ReplaySimulationStepExecution execution =
                    forevervalidator::simulation::StepSpeculativeTicking(
                            *impl->instance.runtime, tick);
            if (execution.result != ReplaySimulationRunResult::Success) {
                result.result = execution.result;
                return result;
            }
            impl->instance.incrementalRespawnCount +=
                    execution.respawnExecutedCount;
        }
    } else {
        for (std::size_t index = begin; index < begin + count; ++index) {
            const ReplayControlTick &tick = controlTicks[index];
            const ReplaySimulationStepExecution execution =
                    impl->instance.runtime->Step(tick);
            if (execution.result != ReplaySimulationRunResult::Success) {
                result.result = execution.result;
                return result;
            }
            impl->instance.incrementalRespawnCount +=
                    execution.respawnExecutedCount;
        }
    }
    result.finishTimeMs = impl->instance.runtime->FinishTimeMs();
    result.finishTime = impl->instance.runtime->FinishTime();
    result.stuntsScore = impl->instance.runtime->StuntsScore();
    result.raceCompleted = result.finishTimeMs.has_value();
    result.executedRespawnCount = impl->instance.incrementalRespawnCount;
    result.result = ReplaySimulationRunResult::Success;
    return result;
}

std::optional<ReplaySimulationStateView>
ReplaySimulationSession::CurrentState() const {
    if (!impl->instance.runtime) {
        return std::nullopt;
    }
    ReplaySimulationStateView result;
    result.frame = impl->instance.runtime->CurrentFrame();
    result.controls = impl->instance.runtime->CurrentControls();
    result.race = impl->instance.runtime->RaceProgress();
    result.finishTimeMs = impl->instance.runtime->FinishTimeMs();
    result.finishTime = impl->instance.runtime->FinishTime();
    result.stuntsScore = impl->instance.runtime->StuntsScore();
    result.respawnCount = impl->instance.incrementalRespawnCount;
    const ReplayRaceCameraVehicleState camera =
            impl->instance.runtime->CurrentRaceCameraState();
    result.signedSpeed = camera.signedSpeed;
    result.turbo = camera.turbo;
    result.cameraFlightTransition = camera.cameraFlightTransition;
    result.burning = camera.burning;
    result.gearChanged = camera.gearChanged;
    result.wheelContact = camera.wheelContact;
    result.wheelHasSurface = camera.wheelHasSurface;
    result.cameraSupportUp = camera.cameraSupportUp;
    const CSceneVehicleCar::SConditionState condition =
            impl->instance.runtime->CurrentConditionState();
    result.localSpeed = condition.localSpeed;
    result.freeWheeling = condition.freeWheeling;
    result.lateralContact = condition.lateralContact;
    result.sliding = condition.sliding;
    result.gear = condition.gear;
    result.rpm = condition.rpm;
    result.turningRate = condition.turningRate;
    result.turboType = condition.turboType;
    result.turboBoostFactor = condition.turboBoostFactor;
    result.wheelContact = condition.wheelGroundContact;
    result.wheelSliding = condition.wheelSliding;
    result.wheelSurface = condition.wheelSurface;
    return result;
}


std::optional<std::uint32_t>
ReplaySimulationSession::ApplyReplayStuntTimePenalty(
        std::uint32_t overtimeMs) {
    if (!impl->instance.runtime) {
        return std::nullopt;
    }
    if (impl->backend == forevervalidator::SimulationBackend::Cuda) {
#if !FOREVERVALIDATOR_HAS_CUDA
        impl->cudaInitializationDiagnostic =
                "CUDA support is not compiled into this build";
        return std::nullopt;
#else
        if (!impl->instance.runtime->StuntsScore().has_value()) {
            return std::nullopt;
        }
        forevervalidator::simulation::CudaRaceState encoded;
        const auto encode = forevervalidator::simulation::
                EncodeCudaRaceState(
                        impl->instance.race.CaptureRuntimeClone(),
                        &encoded);
        if (encode != forevervalidator::simulation::
                              CudaStateConversionResult::Success) {
            impl->cudaInitializationDiagnostic =
                    "CUDA stunt penalty state conversion failed";
            return std::nullopt;
        }
        forevervalidator::simulation::CudaStuntCommand command;
        command.kind = forevervalidator::simulation::
                CudaStuntCommandKind::TimePenalty;
        command.overtimeMs = overtimeMs;
        const auto executed = forevervalidator::simulation::
                ExecuteCudaStuntCommandsForCertification(
                        encoded, {command});
        if (!executed.success) {
            impl->cudaInitializationDiagnostic =
                    executed.diagnostic;
            return std::nullopt;
        }
        CTrackManiaRace::RuntimeClone restored;
        const auto decode = forevervalidator::simulation::
                DecodeCudaRaceState(executed.finalState, &restored);
        if (decode != forevervalidator::simulation::
                              CudaStateConversionResult::Success ||
            !impl->instance.race.PrepareRuntimeCloneRestore(restored)) {
            impl->cudaInitializationDiagnostic =
                    "CUDA stunt penalty restoration failed";
            return std::nullopt;
        }
        impl->instance.race.RestoreRuntimeClone(std::move(restored));
        return impl->instance.runtime->StuntsScore();
#endif
    }
    return impl->instance.runtime->ApplyReplayStuntTimePenalty(overtimeMs);
}

std::shared_ptr<const ReplaySimulationInstanceClone>
ReplaySimulationSession::CaptureRuntimeClone() const {
    if (!impl->instance.runtime ||
        impl->instance.runtime->CurrentPhase() !=
                                  ReplaySimulationRuntime::Phase::Idle) {
        return {};
    }
    std::optional<ReplaySimulationRuntime::RuntimeClone> runtime =
            impl->instance.runtime->CaptureRuntimeClone();
    if (!runtime.has_value()) {
        return {};
    }
    auto clone = std::make_shared<ReplaySimulationInstanceClone>();
    clone->race = impl->instance.race.CaptureRuntimeClone();
    clone->runtime = std::move(*runtime);
    clone->incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    clone->randomState =
            tmnf::simulation::CaptureGameRandomState();
    return clone;
}

bool ReplaySimulationSession::PrepareCudaSearchSpecialization(
        std::string *diagnostic) {
#if !FOREVERVALIDATOR_HAS_CUDA
    if (diagnostic != nullptr) {
        *diagnostic = "CUDA support is not compiled into this build";
    }
    return false;
#else
    if (impl->backend != forevervalidator::SimulationBackend::Cuda ||
        !impl->cudaDeviceScene.Ready() ||
        !impl->cudaDeviceConfiguration.Ready()) {
        impl->cudaSearchSpecializationDiagnostic =
                "CUDA map data is not ready for fast mode";
        if (diagnostic != nullptr) {
            *diagnostic = impl->cudaSearchSpecializationDiagnostic;
        }
        return false;
    }
    const forevervalidator::CudaBackendDiagnostics cudaDiagnostics =
            forevervalidator::QueryCudaBackendDiagnostics();
    if (cudaDiagnostics.IsReady() &&
        !forevervalidator::simulation::
                IsCudaSessionSpecializationSupported(
                        cudaDiagnostics.computeCapabilityMajor,
                        cudaDiagnostics.computeCapabilityMinor)) {
        impl->cudaSearchSpecializationDiagnostic =
                "Optional CUDA session specialization requires compute "
                "capability 7.5 or newer; the regular precompiled CUDA "
                "search kernel remains available";
        if (diagnostic != nullptr) {
            *diagnostic = impl->cudaSearchSpecializationDiagnostic;
        }
        return false;
    }
    if (impl->cudaSearchSpecialization &&
        impl->cudaSearchSpecialization->Ready()) {
        if (diagnostic != nullptr) {
            diagnostic->clear();
        }
        return true;
    }
    if (impl->cudaSearchSpecializationAttempted) {
        if (diagnostic != nullptr) {
            *diagnostic = impl->cudaSearchSpecializationDiagnostic;
        }
        return false;
    }
    impl->cudaSearchSpecializationAttempted = true;

    forevervalidator::simulation::CudaPackedStaticConfigurationHeader
            configuration{};
    forevervalidator::simulation::CudaPackedSceneHeader scene{};
    cudaError_t error = cudaMemcpy(
            &configuration,
            impl->cudaDeviceConfiguration.DeviceData(),
            sizeof(configuration),
            cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) {
        error = cudaMemcpy(
                &scene,
                impl->cudaDeviceScene.DeviceData(),
                sizeof(scene),
                cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) {
        impl->cudaSearchSpecializationDiagnostic =
                "Could not read CUDA map data for fast mode: " +
                std::string(cudaGetErrorString(error));
        if (diagnostic != nullptr) {
            *diagnostic = impl->cudaSearchSpecializationDiagnostic;
        }
        return false;
    }

    auto module = std::make_shared<
            forevervalidator::simulation::cuda::specialization::
                    SessionModule>();
    std::string buildDiagnostic;
    if (!module->Build(
                configuration,
                reinterpret_cast<std::uintptr_t>(
                        impl->cudaDeviceConfiguration.DeviceData()),
                scene,
                reinterpret_cast<std::uintptr_t>(
                        impl->cudaDeviceScene.DeviceData()),
                &buildDiagnostic)) {
        impl->cudaSearchSpecializationDiagnostic =
                buildDiagnostic.empty()
                ? "The optional fast CUDA kernel could not be built"
                : std::move(buildDiagnostic);
        if (diagnostic != nullptr) {
            *diagnostic = impl->cudaSearchSpecializationDiagnostic;
        }
        return false;
    }
    impl->cudaSearchSpecialization = std::move(module);
    impl->cudaSearchSpecializationDiagnostic.clear();
    if (diagnostic != nullptr) {
        diagnostic->clear();
    }
    return true;
#endif
}

std::shared_ptr<const forevervalidator::simulation::cuda::specialization::
                        SessionModule>
ReplaySimulationSession::CudaSearchSpecialization() const noexcept {
    return impl->cudaSearchSpecialization;
}

const std::string &
ReplaySimulationSession::CudaSearchSpecializationDiagnostic() const noexcept {
    return impl->cudaSearchSpecializationDiagnostic;
}

std::unique_ptr<forevervalidator::simulation::CudaSearchExecutor>
ReplaySimulationSession::CreateCudaSearchExecutor(
        forevervalidator::simulation::CudaSearchExecutorConfiguration
                configuration,
        std::uint64_t initialControlCursor,
        std::string *diagnostic) const {
#if !FOREVERVALIDATOR_HAS_CUDA
    (void)configuration;
    (void)initialControlCursor;
    if (diagnostic != nullptr) {
        *diagnostic = "CUDA support is not compiled into this build";
    }
    return {};
#else
    if (impl->backend != forevervalidator::SimulationBackend::Cuda ||
        !impl->instance.runtime || !impl->cudaDeviceScene.Ready() ||
        !impl->cudaDeviceConfiguration.Ready()) {
        if (diagnostic != nullptr) {
            *diagnostic = "CUDA search prerequisites are not ready";
        }
        return {};
    }
    const std::shared_ptr<const ReplaySimulationInstanceClone> initial =
            CaptureRuntimeClone();
    if (!initial) {
        if (diagnostic != nullptr) {
            *diagnostic = "CUDA search branch state capture failed";
        }
        return {};
    }
    const auto conversion =
            forevervalidator::simulation::EncodeCudaCandidateState(
                    *initial,
                    impl->incrementalValidationSeed,
                    initialControlCursor,
                    0u,
                    initial->randomState,
                    &configuration.branchState);
    if (conversion != forevervalidator::simulation::
                              CudaStateConversionResult::Success) {
        if (diagnostic != nullptr) {
            *diagnostic = "CUDA search branch state conversion failed";
        }
        return {};
    }
    configuration.deviceScene =
            impl->cudaDeviceScene.DeviceData();
    configuration.deviceStaticConfiguration =
            impl->cudaDeviceConfiguration.DeviceData();
    return forevervalidator::simulation::CudaSearchExecutor::Create(
            configuration, diagnostic);
#endif
}

bool ReplaySimulationSession::PrepareRuntimeCloneRestore(
        const ReplaySimulationInstanceClone &clone) {
    return impl->instance.runtime &&
           impl->instance.race.PrepareRuntimeCloneRestore(clone.race) &&
           impl->instance.runtime->PrepareRuntimeCloneRestore(clone.runtime);
}

void ReplaySimulationSession::RestoreRuntimeClone(
        ReplaySimulationInstanceClone clone) noexcept {
    impl->instance.race.RestoreRuntimeClone(std::move(clone.race));
    impl->instance.runtime->RestoreRuntimeClone(std::move(clone.runtime));
    impl->instance.incrementalRespawnCount = clone.incrementalRespawnCount;
    tmnf::simulation::RestoreGameRandomState(clone.randomState);
}

std::optional<OptimizedCpuStaticSceneFingerprint>
ReplaySimulationSession::
        CaptureOptimizedCpuStaticSceneFingerprintForTesting(
                void) const noexcept {
    if (!impl->instance.runtime) {
        return std::nullopt;
    }
    return impl->instance.runtime->
            CaptureOptimizedCpuStaticSceneFingerprintForTesting(
                    impl->mapScene.PersistentCollisionZoneForTesting());
}

std::optional<forevervalidator::simulation::CudaSceneTransferMetrics>
ReplaySimulationSession::CudaSceneTransferMetricsForTesting(void) const {
    return impl->cudaSceneTransfer;
}

std::optional<forevervalidator::simulation::
                      CudaStaticConfigurationTransferMetrics>
ReplaySimulationSession::
        CudaStaticConfigurationTransferMetricsForTesting(void) const {
    return impl->cudaConfigurationTransfer;
}

const std::string &
ReplaySimulationSession::CudaInitializationDiagnostic() const noexcept {
    return impl->cudaInitializationDiagnostic;
}

std::optional<forevervalidator::simulation::CudaTimelineExecutionMetrics>
ReplaySimulationSession::CudaTimelineMetricsForTesting(void) const {
    return impl->cudaTimelineMetrics;
}

forevervalidator::simulation::CudaTimelineBatchResult
ReplaySimulationSession::ExecuteCudaCandidateBatchForTesting(
        const std::vector<ReplayControlTick> &ticks,
        std::uint32_t candidateCount,
        bool mutateControls,
        std::uint64_t initialControlCursor,
        bool cancellationRequested) {
    forevervalidator::simulation::CudaTimelineBatchResult result;
#if !FOREVERVALIDATOR_HAS_CUDA
    (void)ticks;
    (void)candidateCount;
    (void)mutateControls;
    (void)initialControlCursor;
    (void)cancellationRequested;
    result.status =
            forevervalidator::simulation::CudaTimelineStatus::DeviceFailure;
    result.diagnostic =
            "CUDA support is not compiled into this build";
    return result;
#else
    if (impl->backend != forevervalidator::SimulationBackend::Cuda ||
        !impl->instance.runtime || ticks.empty() ||
        candidateCount == 0u ||
        !impl->cudaDeviceScene.Ready() ||
        !impl->cudaDeviceConfiguration.Ready()) {
        result.status =
                forevervalidator::simulation::
                        CudaTimelineStatus::InvalidArgument;
        result.diagnostic =
                "CUDA candidate batch prerequisites are not ready";
        return result;
    }
    const auto runtime =
            impl->instance.runtime->CaptureRuntimeClone();
    if (!runtime.has_value()) {
        result.status =
                forevervalidator::simulation::
                        CudaTimelineStatus::InvalidArgument;
        result.diagnostic =
                "CUDA candidate batch state capture failed";
        return result;
    }
    ReplaySimulationInstanceClone initial;
    initial.race = impl->instance.race.CaptureRuntimeClone();
    initial.runtime = *runtime;
    initial.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    try {
        std::vector<forevervalidator::simulation::
                            CudaCandidateTimelineInput>
                candidates(candidateCount);
        const std::uint32_t randomState =
                tmnf::simulation::CaptureGameRandomState();
        for (std::uint32_t candidate = 0u;
             candidate < candidateCount; ++candidate) {
            auto &destination = candidates[candidate];
            const auto conversion = forevervalidator::simulation::
                    EncodeCudaCandidateState(
                            initial, impl->incrementalValidationSeed,
                            initialControlCursor, candidate, randomState,
                            &destination.initialState);
            if (conversion != forevervalidator::simulation::
                                      CudaStateConversionResult::Success) {
                result.status =
                        forevervalidator::simulation::
                                CudaTimelineStatus::InvalidArgument;
                result.diagnostic =
                        "CUDA candidate batch state conversion failed";
                return result;
            }
            destination.ticks.reserve(ticks.size());
            for (const ReplayControlTick &tick : ticks) {
                const ReplayControlTick candidateTick =
                        CandidateControlTick(
                                tick, candidate, mutateControls);
                auto flattened = forevervalidator::simulation::
                        FlattenCudaControlTick(candidateTick);
                destination.ticks.push_back(flattened);
            }
        }
        return forevervalidator::simulation::ExecuteCudaTimelineBatch(
                impl->cudaDeviceScene.DeviceData(),
                impl->cudaDeviceConfiguration.DeviceData(),
                candidates, cancellationRequested);
    } catch (const std::bad_alloc &) {
        result.status =
                forevervalidator::simulation::
                        CudaTimelineStatus::CapacityExceeded;
        result.diagnostic =
                "CUDA candidate batch host allocation failed";
        return result;
    }
#endif
}

ReplayCudaVehiclePrefixDifferential
ReplaySimulationSession::
        RunCudaCandidateBatchDifferentialForTesting(
                const std::vector<ReplayControlTick> &ticks,
                std::uint32_t candidateCount,
                bool mutateControls,
                std::uint64_t initialControlCursor) {
    ReplayCudaVehiclePrefixDifferential result;
#if !FOREVERVALIDATOR_HAS_CUDA
    (void)ticks;
    (void)candidateCount;
    (void)mutateControls;
    (void)initialControlCursor;
    result.diagnostic =
            "CUDA support is not compiled into this build";
    return result;
#else
    if (!impl->instance.runtime || ticks.empty() ||
        candidateCount == 0u) {
        result.diagnostic =
                "CUDA candidate batch differential prerequisites are not ready";
        return result;
    }
    const auto capturedRuntime =
            impl->instance.runtime->CaptureRuntimeClone();
    if (!capturedRuntime.has_value()) {
        result.diagnostic =
                "CUDA candidate batch initial state capture failed";
        return result;
    }
    ReplaySimulationInstanceClone initial;
    initial.race = impl->instance.race.CaptureRuntimeClone();
    initial.runtime = *capturedRuntime;
    initial.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    const std::uint32_t initialRandomState =
            tmnf::simulation::CaptureGameRandomState();
    const auto restoreInitial = [&]() {
        if (!impl->instance.race.PrepareRuntimeCloneRestore(
                    initial.race) ||
            !impl->instance.runtime->PrepareRuntimeCloneRestore(
                    initial.runtime)) {
            return false;
        }
        impl->instance.race.RestoreRuntimeClone(initial.race);
        impl->instance.runtime->RestoreRuntimeClone(initial.runtime);
        impl->instance.incrementalRespawnCount =
                initial.incrementalRespawnCount;
        tmnf::simulation::RestoreGameRandomState(initialRandomState);
        return true;
    };
    const auto gpu = ExecuteCudaCandidateBatchForTesting(
            ticks, candidateCount, mutateControls,
            initialControlCursor, false);
    if (gpu.status != forevervalidator::simulation::
                              CudaTimelineStatus::Success ||
        gpu.candidates.size() != candidateCount) {
        result.diagnostic =
                "CUDA candidate batch differential launch failed: " +
                gpu.diagnostic;
        return result;
    }
    std::vector<forevervalidator::simulation::
                        CudaCandidateTimelineOutput>
            cpuCandidates;
    try {
        cpuCandidates.resize(candidateCount);
        for (std::uint32_t candidate = 0u;
             candidate < candidateCount; ++candidate) {
            if (!restoreInitial()) {
                result.diagnostic =
                        "CUDA candidate batch reference restoration failed";
                return result;
            }
            auto &cpuOutput = cpuCandidates[candidate];
            cpuOutput.status = forevervalidator::simulation::
                    CudaTimelineStatus::Success;
            for (const ReplayControlTick &sourceTick : ticks) {
                const ReplayControlTick tick = CandidateControlTick(
                        sourceTick, candidate, mutateControls);
                const ReplaySimulationStepExecution execution =
                        impl->instance.runtime->
                                StepOptimizedCpuNativeBinary32(tick);
                if (execution.result !=
                    ReplaySimulationRunResult::Success) {
                    restoreInitial();
                    result.diagnostic =
                            "CPU candidate batch reference transition failed";
                    return result;
                }
                impl->instance.incrementalRespawnCount +=
                        execution.respawnExecutedCount;
                cpuOutput.executedRespawnCount +=
                        execution.respawnExecutedCount;
                ++cpuOutput.executedTickCount;
                if (tick.observe) {
                    forevervalidator::simulation::
                            CudaTimelineObservation observation;
                    observation.simulatedPosition =
                            execution.simulatedFrame.position;
                    observation.writePosition =
                            execution.writeFrame.position;
                    observation.hasComparison =
                            tick.comparisonTarget.has_value();
                    if (tick.comparisonTarget.has_value()) {
                        observation.comparisonTarget =
                                *tick.comparisonTarget;
                        observation.comparisonDelta = {
                                execution.writeFrame.position.x -
                                        tick.comparisonTarget->x,
                                execution.writeFrame.position.y -
                                        tick.comparisonTarget->y,
                                execution.writeFrame.position.z -
                                        tick.comparisonTarget->z,
                        };
                        const GmVec3 &delta =
                                observation.comparisonDelta;
                        observation.comparisonDistance = CIsqrt(
                                (delta.x * delta.x +
                                 delta.y * delta.y) +
                                delta.z * delta.z);
                    }
                    const auto race =
                            impl->instance.race.CaptureRuntimeClone();
                    observation.hasFinishTick =
                            race.progress.raceCompleted;
                    observation.finishTickMs =
                            race.progress.lastPrepareTimeMs;
                    cpuOutput.observations.push_back(observation);
                }
            }
            const auto finalRuntime =
                    impl->instance.runtime->CaptureRuntimeClone();
            if (!finalRuntime.has_value()) {
                restoreInitial();
                result.diagnostic =
                        "CPU candidate batch final state capture failed";
                return result;
            }
            ReplaySimulationInstanceClone final;
            final.race = impl->instance.race.CaptureRuntimeClone();
            final.runtime = *finalRuntime;
            final.incrementalRespawnCount =
                    impl->instance.incrementalRespawnCount;
            const auto conversion = forevervalidator::simulation::
                    EncodeCudaCandidateState(
                            final, impl->incrementalValidationSeed,
                            initialControlCursor + ticks.size(),
                            candidate,
                            tmnf::simulation::CaptureGameRandomState(),
                            &cpuOutput.finalState);
            if (conversion != forevervalidator::simulation::
                                      CudaStateConversionResult::Success) {
                restoreInitial();
                result.diagnostic =
                        "CPU candidate batch final state conversion failed";
                return result;
            }
            const auto &gpuOutput = gpu.candidates[candidate];
            const auto *cpuBytes =
                    reinterpret_cast<const std::uint8_t *>(
                            &cpuOutput.finalState);
            const auto *gpuBytes =
                    reinterpret_cast<const std::uint8_t *>(
                            &gpuOutput.finalState);
            std::size_t mismatch = 0u;
            while (mismatch < sizeof(cpuOutput.finalState) &&
                   cpuBytes[mismatch] == gpuBytes[mismatch]) {
                ++mismatch;
            }
            result.checkedBytes += sizeof(cpuOutput.finalState);
            if (mismatch != sizeof(cpuOutput.finalState)) {
                restoreInitial();
                result.firstMismatchByte = mismatch;
                result.cpuByte = cpuBytes[mismatch];
                result.gpuByte = gpuBytes[mismatch];
                result.diagnostic =
                        "CUDA candidate batch final state diverged for candidate " +
                        std::to_string(candidate) + " at byte " +
                        std::to_string(mismatch) +
                        " cpu_replacements=" +
                        std::to_string(
                                cpuOutput.finalState.body.
                                        collisionReplacements.count) +
                        "+" +
                        std::to_string(
                                cpuOutput.finalState.
                                        collisionReplacementOverflow.count) +
                        " gpu_replacements=" +
                        std::to_string(
                                gpuOutput.finalState.body.
                                        collisionReplacements.count) +
                        "+" +
                        std::to_string(
                                gpuOutput.finalState.
                                        collisionReplacementOverflow.count) +
                        " cpu_overflow0=(" +
                        std::to_string(
                                cpuOutput.finalState.
                                        collisionReplacementOverflow.
                                        values[0].x) +
                        "," +
                        std::to_string(
                                cpuOutput.finalState.
                                        collisionReplacementOverflow.
                                        values[0].y) +
                        "," +
                        std::to_string(
                                cpuOutput.finalState.
                                        collisionReplacementOverflow.
                                        values[0].z) +
                        ") gpu_overflow0=(" +
                        std::to_string(
                                gpuOutput.finalState.
                                        collisionReplacementOverflow.
                                        values[0].x) +
                        "," +
                        std::to_string(
                                gpuOutput.finalState.
                                        collisionReplacementOverflow.
                                        values[0].y) +
                        "," +
                        std::to_string(
                                gpuOutput.finalState.
                                        collisionReplacementOverflow.
                                        values[0].z) +
                        ")";
                return result;
            }
            if (cpuOutput.observations.size() !=
                gpuOutput.observations.size()) {
                restoreInitial();
                result.diagnostic =
                        "CUDA candidate batch observation count diverged";
                return result;
            }
            for (std::size_t observation = 0u;
                 observation < cpuOutput.observations.size();
                 ++observation) {
                if (std::memcmp(
                            &cpuOutput.observations[observation],
                            &gpuOutput.observations[observation],
                            sizeof(forevervalidator::simulation::
                                           CudaTimelineObservation)) != 0) {
                    restoreInitial();
                    result.diagnostic =
                            "CUDA candidate batch observation diverged";
                    return result;
                }
                result.checkedBytes +=
                        sizeof(forevervalidator::simulation::
                                       CudaTimelineObservation);
            }
        }
    } catch (const std::bad_alloc &) {
        restoreInitial();
        result.diagnostic =
                "CUDA candidate batch differential allocation failed";
        return result;
    }
    const auto cpuWinner =
            forevervalidator::simulation::SelectCudaTimelineWinner(
                    cpuCandidates);
    if (cpuWinner != gpu.winnerCandidateIndex ||
        (cpuWinner.has_value() &&
         gpu.winnerCandidateId !=
                 std::optional<std::uint32_t>(
                         cpuCandidates[*cpuWinner].
                                 finalState.candidateId))) {
        restoreInitial();
        result.diagnostic =
                "CUDA candidate batch winner diverged";
        return result;
    }
    if (!restoreInitial()) {
        result.diagnostic =
                "CUDA candidate batch final restoration failed";
        return result;
    }
    result.success = true;
    result.firstMismatchByte = SIZE_MAX;
    result.diagnostic =
            "CUDA candidate batch states, observations, and winner are bit-exact";
    return result;
#endif
}

ReplayCudaVehiclePrefixDifferential
ReplaySimulationSession::RunCudaVehiclePrefixDifferentialForTesting(
        float dt) {
    ReplayCudaVehiclePrefixDifferential result;
#if !FOREVERVALIDATOR_HAS_CUDA
    (void)dt;
    result.diagnostic =
            "CUDA support is not compiled into this build";
    return result;
#else
    if (impl->backend != forevervalidator::SimulationBackend::Cuda ||
        !impl->instance.runtime ||
        !impl->cudaDeviceConfiguration.Ready() ||
        !(dt > 0.0f)) {
        result.diagnostic =
                "CUDA vehicle-prefix differential prerequisites are not ready";
        return result;
    }
    std::optional<ReplaySimulationRuntime::RuntimeClone> initialRuntime =
            impl->instance.runtime->CaptureRuntimeClone();
    std::optional<ReplaySimulationRuntime::RuntimeClone> cpuRuntime =
            impl->instance.runtime->
                    CaptureVehiclePrefixReferenceForTesting(dt);
    if (!initialRuntime.has_value() || !cpuRuntime.has_value()) {
        result.diagnostic =
                "CPU vehicle-prefix reference capture failed";
        return result;
    }
    ReplaySimulationInstanceClone initial;
    initial.race = impl->instance.race.CaptureRuntimeClone();
    initial.runtime = std::move(*initialRuntime);
    initial.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    ReplaySimulationInstanceClone cpu = initial;
    cpu.runtime = std::move(*cpuRuntime);

    forevervalidator::simulation::CudaCandidateState encoded;
    const auto conversion = forevervalidator::simulation::
            EncodeCudaCandidateState(
                    initial, impl->incrementalValidationSeed, 0u, 0u,
                    1u, &encoded);
    if (conversion != forevervalidator::simulation::
                              CudaStateConversionResult::Success) {
        result.diagnostic =
                "CUDA vehicle-prefix initial-state conversion failed";
        return result;
    }
    const auto gpu = forevervalidator::simulation::
            ExecuteCudaVehiclePrefixForCertification(
                    impl->cudaDeviceConfiguration.DeviceData(),
                    encoded, dt);
    if (!gpu.success) {
        result.diagnostic = gpu.diagnostic;
        return result;
    }
    ReplaySimulationInstanceClone decoded;
    if (forevervalidator::simulation::DecodeCudaCandidateState(
                gpu.finalState, &decoded) !=
        forevervalidator::simulation::
                CudaStateConversionResult::Success) {
        result.diagnostic =
                "CUDA vehicle-prefix final-state conversion failed";
        return result;
    }
    forevervalidator::simulation::CudaCandidateState cpuEncoded;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                cpu, impl->incrementalValidationSeed, 0u, 0u, 1u,
                &cpuEncoded) !=
        forevervalidator::simulation::
                CudaStateConversionResult::Success) {
        result.diagnostic =
                "CPU vehicle-prefix result conversion failed";
        return result;
    }
    const auto *cpuCandidateBytes =
            reinterpret_cast<const std::uint8_t *>(&cpuEncoded);
    const auto *gpuCandidateBytes =
            reinterpret_cast<const std::uint8_t *>(&gpu.finalState);
    std::size_t candidateMismatch = 0u;
    while (candidateMismatch < sizeof(cpuEncoded) &&
           cpuCandidateBytes[candidateMismatch] ==
                   gpuCandidateBytes[candidateMismatch]) {
        ++candidateMismatch;
    }
    const std::uint64_t cpuRaceHash =
            ReplayRaceRuntimeSemanticHash(cpu.race);
    const std::uint64_t gpuRaceHash =
            ReplayRaceRuntimeSemanticHash(decoded.race);
    const std::uint64_t cpuRuntimeHash =
            ReplaySimulationInstanceSemanticHash(cpu);
    const std::uint64_t gpuRuntimeHash =
            ReplaySimulationInstanceSemanticHash(decoded);
    constexpr std::size_t CandidateSemanticExtent =
            offsetof(
                    forevervalidator::simulation::
                            CudaCandidatePhysicsState,
                    stuntsEnabled) +
            sizeof(bool);
    if (candidateMismatch >= CandidateSemanticExtent &&
        cpuRaceHash == gpuRaceHash &&
        cpuRuntimeHash == gpuRuntimeHash) {
        result.success = true;
        result.checkedBytes = CandidateSemanticExtent;
        result.firstMismatchByte = SIZE_MAX;
        result.diagnostic =
                "CUDA vehicle-prefix state is bit-exact";
        return result;
    }

    result.checkedBytes = CandidateSemanticExtent;
    result.firstMismatchByte = candidateMismatch;
    if (candidateMismatch < sizeof(cpuEncoded)) {
        result.cpuByte = cpuCandidateBytes[candidateMismatch];
        result.gpuByte = gpuCandidateBytes[candidateMismatch];
    }
    result.diagnostic =
            "CUDA vehicle-prefix state diverged at candidate byte " +
            std::to_string(candidateMismatch) +
            " race_hashes=" + std::to_string(cpuRaceHash) +
            "/" + std::to_string(gpuRaceHash) +
            " state_hashes=" + std::to_string(cpuRuntimeHash) +
            "/" + std::to_string(gpuRuntimeHash) +
            " input_speed_z=" +
            std::to_string(encoded.body.current.linearSpeed.z) +
            " cpu_wheel0_speed=" +
            std::to_string(
                    cpuEncoded.vehicle.wheels.values[0].
                            realTime.wheelAngularSpeed) +
            " gpu_wheel0_speed=" +
            std::to_string(
                    gpu.finalState.vehicle.wheels.values[0].
                            realTime.wheelAngularSpeed) +
            " cpu_wheel0_contact=" +
            std::to_string(
                    cpuEncoded.vehicle.wheels.values[0].
                            realTime.contactPresent) +
            " gpu_wheel0_contact=" +
            std::to_string(
                    gpu.finalState.vehicle.wheels.values[0].
                            realTime.contactPresent) +
            " cpu_radius=" +
            std::to_string(
                    cpuEncoded.vehicle.wheels.values[0].rollingRadius) +
            " gpu_radius=" +
            std::to_string(
                    gpu.finalState.vehicle.wheels.values[0].rollingRadius);
    return result;
#endif
}

ReplayCudaVehiclePrefixDifferential
ReplaySimulationSession::RunCudaVehicleForceDifferentialForTesting(
        float dt) {
    ReplayCudaVehiclePrefixDifferential result;
#if !FOREVERVALIDATOR_HAS_CUDA
    (void)dt;
    result.diagnostic =
            "CUDA support is not compiled into this build";
    return result;
#else
    if (impl->backend != forevervalidator::SimulationBackend::Cuda ||
        !impl->instance.runtime ||
        !impl->cudaDeviceConfiguration.Ready() ||
        !(dt > 0.0f)) {
        result.diagnostic =
                "CUDA vehicle-force differential prerequisites are not ready";
        return result;
    }
    auto initialRuntime =
            impl->instance.runtime->CaptureRuntimeClone();
    auto cpuRuntime = impl->instance.runtime->
            CaptureVehicleForceReferenceForTesting(dt);
    if (!initialRuntime.has_value() || !cpuRuntime.has_value()) {
        result.diagnostic =
                "CPU vehicle-force reference capture failed";
        return result;
    }
    ReplaySimulationInstanceClone initial;
    initial.race = impl->instance.race.CaptureRuntimeClone();
    initial.runtime = std::move(*initialRuntime);
    initial.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    ReplaySimulationInstanceClone cpu = initial;
    cpu.runtime = std::move(*cpuRuntime);
    forevervalidator::simulation::CudaCandidateState encoded;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                initial, impl->incrementalValidationSeed, 0u, 0u,
                1u, &encoded) !=
        forevervalidator::simulation::
                CudaStateConversionResult::Success) {
        result.diagnostic =
                "CUDA vehicle-force initial-state conversion failed";
        return result;
    }
    const auto gpu = forevervalidator::simulation::
            ExecuteCudaVehicleForceForCertification(
                    impl->cudaDeviceConfiguration.DeviceData(),
                    encoded, dt);
    if (!gpu.success || !gpu.supported) {
        result.diagnostic = gpu.diagnostic;
        return result;
    }
    ReplaySimulationInstanceClone decoded;
    if (forevervalidator::simulation::DecodeCudaCandidateState(
                gpu.finalState, &decoded) !=
        forevervalidator::simulation::
                CudaStateConversionResult::Success) {
        result.diagnostic =
                "CUDA vehicle-force final-state conversion failed";
        return result;
    }
    forevervalidator::simulation::CudaCandidateState cpuEncoded;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                cpu, impl->incrementalValidationSeed, 0u, 0u, 1u,
                &cpuEncoded) !=
        forevervalidator::simulation::
                CudaStateConversionResult::Success) {
        result.diagnostic =
                "CPU vehicle-force result conversion failed";
        return result;
    }
    const auto *cpuBytes =
            reinterpret_cast<const std::uint8_t *>(&cpuEncoded);
    const auto *gpuBytes =
            reinterpret_cast<const std::uint8_t *>(&gpu.finalState);
    std::size_t mismatch = 0u;
    while (mismatch < sizeof(cpuEncoded) &&
           cpuBytes[mismatch] == gpuBytes[mismatch]) {
        ++mismatch;
    }
    constexpr std::size_t SemanticExtent =
            offsetof(
                    forevervalidator::simulation::
                            CudaCandidatePhysicsState,
                    stuntsEnabled) +
            sizeof(bool);
    result.checkedBytes = SemanticExtent;
    if (mismatch >= SemanticExtent &&
        ReplayRaceRuntimeSemanticHash(cpu.race) ==
                ReplayRaceRuntimeSemanticHash(decoded.race) &&
        ReplaySimulationInstanceSemanticHash(cpu) ==
                ReplaySimulationInstanceSemanticHash(decoded)) {
        result.success = true;
        result.firstMismatchByte = SIZE_MAX;
        result.diagnostic =
                "CUDA vehicle-force state is bit-exact";
        return result;
    }
    result.firstMismatchByte = mismatch;
    if (mismatch < sizeof(cpuEncoded)) {
        result.cpuByte = cpuBytes[mismatch];
        result.gpuByte = gpuBytes[mismatch];
    }
    result.diagnostic =
            "CUDA vehicle-force state diverged at candidate byte " +
            std::to_string(mismatch) +
            " cpu_torque=(" +
            std::to_string(cpuEncoded.body.current.torque.x) + "," +
            std::to_string(cpuEncoded.body.current.torque.y) + "," +
            std::to_string(cpuEncoded.body.current.torque.z) + ")" +
            " gpu_torque=(" +
            std::to_string(gpu.finalState.body.current.torque.x) + "," +
            std::to_string(gpu.finalState.body.current.torque.y) + "," +
            std::to_string(gpu.finalState.body.current.torque.z) + ")" +
            " cpu_force=(" +
            std::to_string(cpuEncoded.body.current.force.x) + "," +
            std::to_string(cpuEncoded.body.current.force.y) + "," +
            std::to_string(cpuEncoded.body.current.force.z) + ")" +
            " gpu_force=(" +
            std::to_string(gpu.finalState.body.current.force.x) + "," +
            std::to_string(gpu.finalState.body.current.force.y) + "," +
            std::to_string(gpu.finalState.body.current.force.z) + ")";
    return result;
#endif
}

ReplayCudaVehiclePrefixDifferential
ReplaySimulationSession::RunCudaCollisionDifferentialForTesting(void) {
    ReplayCudaVehiclePrefixDifferential result;
#if !FOREVERVALIDATOR_HAS_CUDA
    result.diagnostic =
            "CUDA support is not compiled into this build";
    return result;
#else
    if (impl->backend != forevervalidator::SimulationBackend::Cuda ||
        !impl->instance.runtime ||
        !impl->cudaDeviceConfiguration.Ready() ||
        !impl->cudaDeviceScene.Ready()) {
        result.diagnostic =
                "CUDA collision differential prerequisites are not ready";
        return result;
    }
    const auto runtime =
            impl->instance.runtime->CaptureRuntimeClone();
    const auto cpu =
            impl->instance.runtime->
                    CaptureCollisionReferenceForTesting();
    if (!runtime.has_value() || !cpu.has_value()) {
        result.diagnostic =
                "CPU collision reference capture failed";
        return result;
    }
    ReplaySimulationInstanceClone initial;
    initial.race = impl->instance.race.CaptureRuntimeClone();
    initial.runtime = *runtime;
    initial.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    if (!impl->instance.runtime->
                ApplyCollisionResponseReferenceForTesting()) {
        result.diagnostic =
                "CPU collision-response reference failed";
        return result;
    }
    ReplaySimulationInstanceClone cpuResponse;
    cpuResponse.race =
            impl->instance.race.CaptureRuntimeClone();
    const auto cpuResponseRuntime =
            impl->instance.runtime->CaptureRuntimeClone();
    cpuResponse.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    if (!cpuResponseRuntime.has_value() ||
        !impl->instance.race.PrepareRuntimeCloneRestore(
                initial.race) ||
        !impl->instance.runtime->PrepareRuntimeCloneRestore(
                initial.runtime)) {
        result.diagnostic =
                "CPU collision-response capture or restoration failed";
        return result;
    }
    cpuResponse.runtime = *cpuResponseRuntime;
    impl->instance.race.RestoreRuntimeClone(initial.race);
    impl->instance.runtime->RestoreRuntimeClone(initial.runtime);
    impl->instance.incrementalRespawnCount =
            initial.incrementalRespawnCount;
    forevervalidator::simulation::CudaCandidateState encoded;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                initial, impl->incrementalValidationSeed, 0u, 0u,
                tmnf::simulation::CaptureGameRandomState(), &encoded) !=
        forevervalidator::simulation::
                CudaStateConversionResult::Success) {
        result.diagnostic =
                "CUDA collision initial-state conversion failed";
        return result;
    }
    const auto gpu = forevervalidator::simulation::
            ExecuteCudaCollisionForCertification(
                    impl->cudaDeviceScene.DeviceData(),
                    impl->cudaDeviceConfiguration.DeviceData(),
                    encoded);
    if (!gpu.success) {
        result.diagnostic = gpu.diagnostic;
        return result;
    }
    constexpr std::size_t CollisionSemanticBytes =
            sizeof(GmVec3) * 4u +
            sizeof(std::uint32_t) * 2u +
            sizeof(bool);
    result.checkedBytes =
            std::min(cpu->size(), gpu.collisions.size()) *
            CollisionSemanticBytes;
    if (cpu->size() != gpu.collisions.size()) {
        result.firstMismatchByte = result.checkedBytes;
        result.diagnostic =
                "CUDA collision count diverged cpu=" +
                std::to_string(cpu->size()) + " gpu=" +
                std::to_string(gpu.collisions.size()) +
                " accel_cells=" +
                std::to_string(gpu.accelerationCellVisits) +
                " accel_surfaces=" +
                std::to_string(gpu.accelerationSurfaceVisits) +
                " mesh_cells=" +
                std::to_string(gpu.meshCellVisits) +
                " mesh_intersections=" +
                std::to_string(gpu.meshCellIntersections) +
                " mesh_triangle_cells=" +
                std::to_string(gpu.meshTriangleCells) +
                " triangles=" +
                std::to_string(gpu.triangleTests) +
                " hits=" +
                std::to_string(gpu.triangleHits) +
                " shape=" +
                std::to_string(gpu.firstVisitedShape) +
                " surface=" +
                std::to_string(gpu.firstVisitedSurface) +
                " shape_pos=(" +
                std::to_string(
                        gpu.firstShapeWorld.translation.x) + "," +
                std::to_string(
                        gpu.firstShapeWorld.translation.y) + "," +
                std::to_string(
                        gpu.firstShapeWorld.translation.z) + ")" +
                " unit_box=(" +
                std::to_string(gpu.firstEllipsoidBox.center.x) + "," +
                std::to_string(gpu.firstEllipsoidBox.center.y) + "," +
                std::to_string(gpu.firstEllipsoidBox.center.z) +
                "; " +
                std::to_string(
                        gpu.firstEllipsoidBox.halfExtents.x) + "," +
                std::to_string(
                        gpu.firstEllipsoidBox.halfExtents.y) + "," +
                std::to_string(
                        gpu.firstEllipsoidBox.halfExtents.z) + ")" +
                " mesh_root=(" +
                std::to_string(gpu.firstMeshRootBounds.center.x) + "," +
                std::to_string(gpu.firstMeshRootBounds.center.y) + "," +
                std::to_string(gpu.firstMeshRootBounds.center.z) +
                "; " +
                std::to_string(
                        gpu.firstMeshRootBounds.halfExtents.x) + "," +
                std::to_string(
                        gpu.firstMeshRootBounds.halfExtents.y) + "," +
                std::to_string(
                        gpu.firstMeshRootBounds.halfExtents.z) + ")";
        if (!cpu->empty()) {
            const GmCollision &first = cpu->front();
            result.diagnostic +=
                    " cpu_first=(" +
                    std::to_string(first.contactPoint.x) + "," +
                    std::to_string(first.contactPoint.y) + "," +
                    std::to_string(first.contactPoint.z) +
                    ") cpu_materials=" +
                    std::to_string(static_cast<std::uint32_t>(
                            first.materialA)) + "," +
                    std::to_string(static_cast<std::uint32_t>(
                            first.materialB));
        }
        result.diagnostic += " cpu_contacts=[";
        for (std::size_t index = 0u; index < cpu->size(); ++index) {
            const GmCollision &collision = (*cpu)[index];
            if (index != 0u) result.diagnostic += ";";
            result.diagnostic +=
                    std::to_string(collision.contactPoint.x) + "," +
                    std::to_string(collision.contactPoint.y) + "," +
                    std::to_string(collision.contactPoint.z) + "|" +
                    std::to_string(collision.impulseNormal.x) + "," +
                    std::to_string(collision.impulseNormal.y) + "," +
                    std::to_string(collision.impulseNormal.z) + "|" +
                    std::to_string(collision.sphereMergePrimary) + "|" +
                    std::to_string(static_cast<std::uint32_t>(
                            collision.materialA)) + "," +
                    std::to_string(static_cast<std::uint32_t>(
                            collision.materialB));
        }
        result.diagnostic += "] gpu_contacts=[";
        for (std::size_t index = 0u;
             index < gpu.collisions.size(); ++index) {
            const auto &collision = gpu.collisions[index];
            if (index != 0u) result.diagnostic += ";";
            result.diagnostic +=
                    std::to_string(collision.contactPoint.x) + "," +
                    std::to_string(collision.contactPoint.y) + "," +
                    std::to_string(collision.contactPoint.z) + "|" +
                    std::to_string(collision.impulseNormal.x) + "," +
                    std::to_string(collision.impulseNormal.y) + "," +
                    std::to_string(collision.impulseNormal.z) + "|" +
                    std::to_string(collision.sphereMergePrimary) + "|" +
                    std::to_string(collision.materialA) + "," +
                    std::to_string(collision.materialB) + "|" +
                    std::to_string(collision.movingShapeIndex) + "," +
                    std::to_string(collision.staticSurfaceIndex) + "," +
                    std::to_string(collision.staticActorIndex);
        }
        result.diagnostic += "]";
        if (!gpu.collisions.empty()) {
            result.diagnostic += " gpu_actor_purposes=[";
            for (std::size_t index = 0u;
                 index < gpu.collisions.size(); ++index) {
                if (index != 0u) result.diagnostic += ";";
                const std::uint32_t actorIndex =
                        gpu.collisions[index].staticActorIndex;
                if (actorIndex >= impl->cudaHostScene.actors.size()) {
                    result.diagnostic += "invalid";
                    continue;
                }
                const auto &actor =
                        impl->cudaHostScene.actors[actorIndex];
                result.diagnostic +=
                        std::to_string(actor.purpose) + "," +
                        std::to_string(static_cast<std::uint32_t>(
                                actor.itemProperties.collisionGroup)) + "," +
                        std::to_string(
                                actor.itemProperties.collisionStatic) + "," +
                        std::to_string(actor.itemProperties.active) + "," +
                        std::to_string(actor.itemProperties.zombie);
            }
            result.diagnostic += "]";
        }
        return result;
    }
    const auto sameVector = [](const GmVec3 &left,
                               const GmVec3 &right) {
        return std::memcmp(
                       &left, &right, sizeof(GmVec3)) == 0;
    };
    for (std::size_t index = 0u; index < cpu->size(); ++index) {
        const GmCollision &left = (*cpu)[index];
        const auto &right = gpu.collisions[index];
        if (!sameVector(left.separation, right.separation) ||
            !sameVector(left.impulseNormal, right.impulseNormal) ||
            !sameVector(left.contactPoint, right.contactPoint) ||
            static_cast<std::uint32_t>(left.materialA) !=
                    right.materialA ||
            static_cast<std::uint32_t>(left.materialB) !=
                    right.materialB ||
            left.sphereMergePrimary !=
                    right.sphereMergePrimary ||
            !sameVector(left.extraNegated, right.extraNegated)) {
            result.firstMismatchByte =
                    index * CollisionSemanticBytes;
            result.diagnostic =
                    "CUDA collision diverged at record " +
                    std::to_string(index) +
                    " cpu_point=(" +
                    std::to_string(left.contactPoint.x) + "," +
                    std::to_string(left.contactPoint.y) + "," +
                    std::to_string(left.contactPoint.z) + ")" +
                    " gpu_point=(" +
                    std::to_string(right.contactPoint.x) + "," +
                    std::to_string(right.contactPoint.y) + "," +
                    std::to_string(right.contactPoint.z) + ")" +
                    " cpu_separation=(" +
                    std::to_string(left.separation.x) + "," +
                    std::to_string(left.separation.y) + "," +
                    std::to_string(left.separation.z) + ")" +
                    " gpu_separation=(" +
                    std::to_string(right.separation.x) + "," +
                    std::to_string(right.separation.y) + "," +
                    std::to_string(right.separation.z) + ")" +
                    " cpu_normal=(" +
                    std::to_string(left.impulseNormal.x) + "," +
                    std::to_string(left.impulseNormal.y) + "," +
                    std::to_string(left.impulseNormal.z) + ")" +
                    " gpu_normal=(" +
                    std::to_string(right.impulseNormal.x) + "," +
                    std::to_string(right.impulseNormal.y) + "," +
                    std::to_string(right.impulseNormal.z) + ")" +
                    " cpu_extra=(" +
                    std::to_string(left.extraNegated.x) + "," +
                    std::to_string(left.extraNegated.y) + "," +
                    std::to_string(left.extraNegated.z) + ")" +
                    " gpu_extra=(" +
                    std::to_string(right.extraNegated.x) + "," +
                    std::to_string(right.extraNegated.y) + "," +
                    std::to_string(right.extraNegated.z) + ")" +
                    " cpu_materials=" +
                    std::to_string(static_cast<std::uint32_t>(
                            left.materialA)) + "," +
                    std::to_string(static_cast<std::uint32_t>(
                            left.materialB)) +
                    " gpu_materials=" +
                    std::to_string(right.materialA) + "," +
                    std::to_string(right.materialB) +
                    " cpu_primary=" +
                    std::to_string(left.sphereMergePrimary) +
                    " gpu_primary=" +
                    std::to_string(right.sphereMergePrimary);
            return result;
        }
    }
    forevervalidator::simulation::CudaCandidateState
            cpuResponseEncoded;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                cpuResponse, impl->incrementalValidationSeed,
                0u, 0u,
                tmnf::simulation::CaptureGameRandomState(),
                &cpuResponseEncoded) !=
        forevervalidator::simulation::
                CudaStateConversionResult::Success) {
        result.diagnostic =
                "CPU collision-response state conversion failed";
        return result;
    }
    const auto *cpuResponseBytes =
            reinterpret_cast<const std::uint8_t *>(
                    &cpuResponseEncoded);
    const auto *gpuResponseBytes =
            reinterpret_cast<const std::uint8_t *>(
                    &gpu.finalState);
    std::size_t responseMismatch = 0u;
    while (responseMismatch < sizeof(cpuResponseEncoded) &&
           cpuResponseBytes[responseMismatch] ==
                   gpuResponseBytes[responseMismatch]) {
        ++responseMismatch;
    }
    result.checkedBytes += sizeof(cpuResponseEncoded);
    if (responseMismatch != sizeof(cpuResponseEncoded)) {
        result.firstMismatchByte =
                responseMismatch;
        result.cpuByte =
                cpuResponseBytes[responseMismatch];
        result.gpuByte =
                gpuResponseBytes[responseMismatch];
        result.diagnostic =
                "CUDA collision response diverged at candidate byte " +
                std::to_string(responseMismatch) +
                " cpu_replacements=" +
                std::to_string(
                        cpuResponseEncoded.body.
                                collisionReplacements.count) +
                    " gpu_replacements=" +
                    std::to_string(
                            gpu.finalState.body.
                                    collisionReplacements.count) +
                    " cpu_prepared_time=" +
                    std::to_string(
                            cpuResponseEncoded.race.preparedEventTimeMs) +
                    " gpu_prepared_time=" +
                    std::to_string(
                            gpu.finalState.race.preparedEventTimeMs) +
                    " cpu_prepared_count=" +
                    std::to_string(
                            cpuResponseEncoded.race.progress.
                                    preparedEventCount) +
                    " gpu_prepared_count=" +
                    std::to_string(
                            gpu.finalState.race.progress.
                                    preparedEventCount) +
                    " wheel_force_mode=" +
                    std::to_string(
                            impl->cudaHostConfiguration.tuning.
                                    wheelForceMode) +
                    " damper_max=" +
                    std::to_string(
                            impl->cudaHostConfiguration.tuning.
                                    suspension.
                                    damperModulationMaxAbsorb) +
                    " gpu_response_wheel=" +
                    std::to_string(
                            gpu.firstResponseWheelIndex) +
                    " gpu_local_before=(" +
                    std::to_string(
                            gpu.firstResponseReplacementBefore.x) + "," +
                    std::to_string(
                            gpu.firstResponseReplacementBefore.y) + "," +
                    std::to_string(
                            gpu.firstResponseReplacementBefore.z) + ")" +
                    " gpu_local_after=(" +
                    std::to_string(
                            gpu.firstResponseReplacementAfter.x) + "," +
                    std::to_string(
                            gpu.firstResponseReplacementAfter.y) + "," +
                    std::to_string(
                            gpu.firstResponseReplacementAfter.z) + ")";
        if (!gpu.collisions.empty()) {
            result.diagnostic +=
                    " first_shape=" +
                    std::to_string(
                            gpu.collisions[0].movingShapeIndex);
        }
        for (std::uint32_t wheel = 0u;
             wheel < encoded.vehicle.wheels.count; ++wheel) {
            result.diagnostic +=
                    " wheel" + std::to_string(wheel) +
                    "_damper=" +
                    std::to_string(
                            encoded.vehicle.wheels.values[wheel].
                                    realTime.damperAbsorb);
        }
        result.diagnostic +=
                " cpu_wheel_contacts=" +
                std::to_string(
                        cpuResponseEncoded.vehicle.contacts.
                                wheelContactCount) +
                " gpu_wheel_contacts=" +
                std::to_string(
                        gpu.finalState.vehicle.contacts.
                                wheelContactCount) +
                " cpu_w3_max_replacement=" +
                std::to_string(
                        cpuResponseEncoded.vehicle.wheels.values[3].
                                realTime.maxReplacementY) +
                " gpu_w3_max_replacement=" +
                std::to_string(
                        gpu.finalState.vehicle.wheels.values[3].
                                realTime.maxReplacementY);
        if (cpuResponseEncoded.body.collisionReplacements.count !=
                    0u &&
            gpu.finalState.body.collisionReplacements.count !=
                    0u) {
            const GmVec3 &cpuReplacement =
                    cpuResponseEncoded.body.
                            collisionReplacements.values[0];
            const GmVec3 &gpuReplacement =
                    gpu.finalState.body.
                            collisionReplacements.values[0];
            result.diagnostic +=
                    " cpu_first=(" +
                    std::to_string(cpuReplacement.x) + "," +
                    std::to_string(cpuReplacement.y) + "," +
                    std::to_string(cpuReplacement.z) + ")" +
                    " gpu_first=(" +
                    std::to_string(gpuReplacement.x) + "," +
                    std::to_string(gpuReplacement.y) + "," +
                    std::to_string(gpuReplacement.z) + ")";
        }
        return result;
    }
    result.success = true;
    result.firstMismatchByte = SIZE_MAX;
    result.diagnostic =
            "CUDA collision sequence and response are bit-exact count=" +
            std::to_string(cpu->size()) +
            " accel_cells=" +
            std::to_string(gpu.accelerationCellVisits) +
            " accel_surfaces=" +
            std::to_string(gpu.accelerationSurfaceVisits) +
            " mesh_cells=" +
            std::to_string(gpu.meshCellVisits) +
            " mesh_intersections=" +
            std::to_string(gpu.meshCellIntersections) +
            " triangle_cells=" +
            std::to_string(gpu.meshTriangleCells) +
            " triangles=" +
            std::to_string(gpu.triangleTests) +
            " hits=" +
            std::to_string(gpu.triangleHits);
    return result;
#endif
}

ReplayCudaVehiclePrefixDifferential
ReplaySimulationSession::RunCudaPhysicsStepDifferentialForTesting(void) {
    ReplayCudaVehiclePrefixDifferential result;
#if !FOREVERVALIDATOR_HAS_CUDA
    result.diagnostic =
            "CUDA support is not compiled into this build";
    return result;
#else
    if (impl->backend != forevervalidator::SimulationBackend::Cuda ||
        !impl->instance.runtime ||
        !impl->cudaDeviceConfiguration.Ready() ||
        !impl->cudaDeviceScene.Ready()) {
        result.diagnostic =
                "CUDA physics-step differential prerequisites are not ready";
        return result;
    }
    const auto initialRuntime =
            impl->instance.runtime->CaptureRuntimeClone();
    if (!initialRuntime.has_value()) {
        result.diagnostic =
                "CPU physics-step initial capture failed";
        return result;
    }
    ReplaySimulationInstanceClone initial;
    initial.race = impl->instance.race.CaptureRuntimeClone();
    initial.runtime = *initialRuntime;
    initial.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    ReplayControlTick tick;
    tick.periodMs = initial.runtime.world.schemePeriodMs != 0u
            ? initial.runtime.world.schemePeriodMs
            : 10u;
    tick.timeMs = initial.runtime.world.tickTimeMs;
    tick.controls =
            impl->instance.runtime->CurrentControls();
    const std::uint32_t randomState =
            tmnf::simulation::CaptureGameRandomState();
    if (!impl->instance.runtime->
                StepPhysicsKernelReferenceForTesting(tick)) {
        result.diagnostic =
                "CPU physics-step reference execution failed";
        return result;
    }
    const auto cpuRuntime =
            impl->instance.runtime->CaptureRuntimeClone();
    ReplaySimulationInstanceClone cpu;
    cpu.race = impl->instance.race.CaptureRuntimeClone();
    cpu.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    if (!cpuRuntime.has_value() ||
        !impl->instance.race.PrepareRuntimeCloneRestore(
                initial.race) ||
        !impl->instance.runtime->PrepareRuntimeCloneRestore(
                initial.runtime)) {
        result.diagnostic =
                "CPU physics-step final capture or restoration failed";
        return result;
    }
    cpu.runtime = *cpuRuntime;
    impl->instance.race.RestoreRuntimeClone(initial.race);
    impl->instance.runtime->RestoreRuntimeClone(initial.runtime);
    impl->instance.incrementalRespawnCount =
            initial.incrementalRespawnCount;

    forevervalidator::simulation::CudaCandidateState
            initialEncoded;
    forevervalidator::simulation::CudaCandidateState cpuEncoded;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                initial, impl->incrementalValidationSeed,
                0u, 0u, randomState, &initialEncoded) !=
                forevervalidator::simulation::
                        CudaStateConversionResult::Success ||
        forevervalidator::simulation::EncodeCudaCandidateState(
                cpu, impl->incrementalValidationSeed,
                0u, 0u, randomState, &cpuEncoded) !=
                forevervalidator::simulation::
                        CudaStateConversionResult::Success) {
        result.diagnostic =
                "CUDA physics-step state conversion failed";
        return result;
    }
    initialEncoded.world.schemePeriodMs = tick.periodMs;
    initialEncoded.world.tickTimeMs = tick.timeMs;
    const auto gpu = forevervalidator::simulation::
            ExecuteCudaPhysicsStepForCertification(
                    impl->cudaDeviceScene.DeviceData(),
                    impl->cudaDeviceConfiguration.DeviceData(),
                    initialEncoded);
    if (!gpu.success) {
        result.diagnostic = gpu.diagnostic;
        return result;
    }
    const auto *cpuBytes =
            reinterpret_cast<const std::uint8_t *>(&cpuEncoded);
    const auto *gpuBytes =
            reinterpret_cast<const std::uint8_t *>(
                    &gpu.finalState);
    const std::size_t semanticExtent =
            reinterpret_cast<const std::uint8_t *>(
                    &cpuEncoded.collisionReplacementOverflow) -
            cpuBytes +
            sizeof(cpuEncoded.collisionReplacementOverflow);
    std::size_t mismatch = 0u;
    while (mismatch < semanticExtent &&
           cpuBytes[mismatch] == gpuBytes[mismatch]) {
        ++mismatch;
    }
    result.checkedBytes = semanticExtent;
    if (mismatch != semanticExtent) {
        result.firstMismatchByte = mismatch;
        result.cpuByte = cpuBytes[mismatch];
        result.gpuByte = gpuBytes[mismatch];
        result.diagnostic =
                "CUDA complete physics step diverged at candidate byte " +
                std::to_string(mismatch) +
                " cpu_position=(" +
                std::to_string(cpuEncoded.body.current.position.x) + "," +
                std::to_string(cpuEncoded.body.current.position.y) + "," +
                std::to_string(cpuEncoded.body.current.position.z) + ")" +
                " gpu_position=(" +
                std::to_string(
                        gpu.finalState.body.current.position.x) + "," +
                std::to_string(
                        gpu.finalState.body.current.position.y) + "," +
                std::to_string(
                        gpu.finalState.body.current.position.z) + ")" +
                " cpu_speed=(" +
                std::to_string(
                        cpuEncoded.body.current.linearSpeed.x) + "," +
                std::to_string(
                        cpuEncoded.body.current.linearSpeed.y) + "," +
                std::to_string(
                        cpuEncoded.body.current.linearSpeed.z) + ")" +
                " gpu_speed=(" +
                std::to_string(
                        gpu.finalState.body.current.linearSpeed.x) + "," +
                std::to_string(
                        gpu.finalState.body.current.linearSpeed.y) + "," +
                std::to_string(
                        gpu.finalState.body.current.linearSpeed.z) + ")" +
                " cpu_temp_force=(" +
                std::to_string(
                        cpuEncoded.body.temporary.force.x) + "," +
                std::to_string(
                        cpuEncoded.body.temporary.force.y) + "," +
                std::to_string(
                        cpuEncoded.body.temporary.force.z) + ")" +
                " gpu_temp_force=(" +
                std::to_string(
                        gpu.finalState.body.temporary.force.x) + "," +
                std::to_string(
                        gpu.finalState.body.temporary.force.y) + "," +
                std::to_string(
                        gpu.finalState.body.temporary.force.z) + ")" +
                " cpu_current_force=(" +
                std::to_string(
                        cpuEncoded.body.current.force.x) + "," +
                std::to_string(
                        cpuEncoded.body.current.force.y) + "," +
                std::to_string(
                        cpuEncoded.body.current.force.z) + ")" +
                " gpu_current_force=(" +
                std::to_string(
                        gpu.finalState.body.current.force.x) + "," +
                std::to_string(
                        gpu.finalState.body.current.force.y) + "," +
                std::to_string(
                        gpu.finalState.body.current.force.z) + ")" +
                " initial_active=" +
                std::to_string(initialEncoded.body.dynamicActive) +
                " cpu_active=" +
                std::to_string(cpuEncoded.body.dynamicActive) +
                " gpu_active=" +
                std::to_string(
                        gpu.finalState.body.dynamicActive) +
                " cpu_dynamic_group_count=" +
                std::to_string(
                        impl->instance.runtime->
                                DynamicCollisionCorpusCountForTesting());
        return result;
    }
    result.success = true;
    result.firstMismatchByte = SIZE_MAX;
    result.diagnostic =
            "CUDA complete physics step is bit-exact";
    return result;
#endif
}

ReplayCudaVehiclePrefixDifferential
ReplaySimulationSession::
RunCudaCollisionSubstepDifferentialForTesting(float dt) {
    ReplayCudaVehiclePrefixDifferential result;
#if !FOREVERVALIDATOR_HAS_CUDA
    static_cast<void>(dt);
    result.diagnostic =
            "CUDA support is not compiled into this build";
    return result;
#else
    if (impl->backend != forevervalidator::SimulationBackend::Cuda ||
        !impl->instance.runtime ||
        !impl->cudaDeviceConfiguration.Ready() ||
        !impl->cudaDeviceScene.Ready() || !(dt > 0.0f)) {
        result.diagnostic =
                "CUDA collision-substep differential prerequisites are not ready";
        return result;
    }
    const auto initialRuntime =
            impl->instance.runtime->CaptureRuntimeClone();
    const auto cpuRuntime =
            impl->instance.runtime->
                    CaptureCollisionSubstepReferenceForTesting(dt);
    if (!initialRuntime.has_value() || !cpuRuntime.has_value()) {
        result.diagnostic =
                "CPU collision-substep reference failed";
        return result;
    }
    ReplaySimulationInstanceClone initial;
    initial.race = impl->instance.race.CaptureRuntimeClone();
    initial.runtime = *initialRuntime;
    initial.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    ReplaySimulationInstanceClone cpu = initial;
    cpu.runtime = *cpuRuntime;
    const std::uint32_t randomState =
            tmnf::simulation::CaptureGameRandomState();
    forevervalidator::simulation::CudaCandidateState
            initialEncoded;
    forevervalidator::simulation::CudaCandidateState cpuEncoded;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                initial, impl->incrementalValidationSeed,
                0u, 0u, randomState, &initialEncoded) !=
                        forevervalidator::simulation::
                                CudaStateConversionResult::Success ||
        forevervalidator::simulation::EncodeCudaCandidateState(
                cpu, impl->incrementalValidationSeed,
                0u, 0u, randomState, &cpuEncoded) !=
                        forevervalidator::simulation::
                                CudaStateConversionResult::Success) {
        result.diagnostic =
                "CUDA collision-substep state conversion failed";
        return result;
    }
    const auto gpu = forevervalidator::simulation::
            ExecuteCudaCollisionSubstepForCertification(
                    impl->cudaDeviceScene.DeviceData(),
                    impl->cudaDeviceConfiguration.DeviceData(),
                    initialEncoded, dt);
    if (!gpu.success) {
        result.diagnostic = gpu.diagnostic;
        return result;
    }
    const auto *cpuBytes =
            reinterpret_cast<const std::uint8_t *>(&cpuEncoded);
    const auto *gpuBytes =
            reinterpret_cast<const std::uint8_t *>(
                    &gpu.finalState);
    std::size_t mismatch = 0u;
    while (mismatch < sizeof(cpuEncoded) &&
           cpuBytes[mismatch] == gpuBytes[mismatch]) {
        ++mismatch;
    }
    result.checkedBytes = sizeof(cpuEncoded);
    if (mismatch != sizeof(cpuEncoded)) {
        result.firstMismatchByte = mismatch;
        result.cpuByte = cpuBytes[mismatch];
        result.gpuByte = gpuBytes[mismatch];
        result.diagnostic =
                "CUDA collision substep diverged at candidate byte " +
                std::to_string(mismatch) + " dt=" +
                std::to_string(dt) +
                " cpu_position=(" +
                std::to_string(cpuEncoded.body.current.position.x) + "," +
                std::to_string(cpuEncoded.body.current.position.y) + "," +
                std::to_string(cpuEncoded.body.current.position.z) + ")" +
                " gpu_position=(" +
                std::to_string(gpu.finalState.body.current.position.x) + "," +
                std::to_string(gpu.finalState.body.current.position.y) + "," +
                std::to_string(gpu.finalState.body.current.position.z) + ")" +
                " cpu_replacements=" +
                std::to_string(
                        cpuEncoded.body.collisionReplacements.count) +
                " gpu_replacements=" +
                std::to_string(
                        gpu.finalState.body.
                                collisionReplacements.count);
        return result;
    }
    result.success = true;
    result.firstMismatchByte = SIZE_MAX;
    result.diagnostic =
            "CUDA collision substep is bit-exact";
    return result;
#endif
}

ReplayCudaVehiclePrefixDifferential
ReplaySimulationSession::RunCudaPreCollisionDifferentialForTesting(
        float dt) {
    ReplayCudaVehiclePrefixDifferential result;
#if !FOREVERVALIDATOR_HAS_CUDA
    static_cast<void>(dt);
    result.diagnostic =
            "CUDA support is not compiled into this build";
    return result;
#else
    if (impl->backend != forevervalidator::SimulationBackend::Cuda ||
        !impl->instance.runtime ||
        !impl->cudaDeviceConfiguration.Ready() || !(dt > 0.0f)) {
        result.diagnostic =
                "CUDA pre-collision differential prerequisites are not ready";
        return result;
    }
    const auto initialRuntime =
            impl->instance.runtime->CaptureRuntimeClone();
    const auto cpuRuntime =
            impl->instance.runtime->
                    CapturePreCollisionReferenceForTesting(dt);
    const auto cpuForceRuntime =
            impl->instance.runtime->
                    CaptureForcePassReferenceForTesting(dt);
    if (!initialRuntime.has_value() || !cpuRuntime.has_value() ||
        !cpuForceRuntime.has_value()) {
        result.diagnostic =
                "CPU pre-collision reference failed";
        return result;
    }
    ReplaySimulationInstanceClone initial;
    initial.race = impl->instance.race.CaptureRuntimeClone();
    initial.runtime = *initialRuntime;
    initial.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    ReplaySimulationInstanceClone cpu = initial;
    cpu.runtime = *cpuRuntime;
    ReplaySimulationInstanceClone cpuForce = initial;
    cpuForce.runtime = *cpuForceRuntime;
    const std::uint32_t randomState =
            tmnf::simulation::CaptureGameRandomState();
    forevervalidator::simulation::CudaCandidateState initialEncoded;
    forevervalidator::simulation::CudaCandidateState cpuEncoded;
    forevervalidator::simulation::CudaCandidateState cpuForceEncoded;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                initial, impl->incrementalValidationSeed,
                0u, 0u, randomState, &initialEncoded) !=
                        forevervalidator::simulation::
                                CudaStateConversionResult::Success ||
        forevervalidator::simulation::EncodeCudaCandidateState(
                cpu, impl->incrementalValidationSeed,
                0u, 0u, randomState, &cpuEncoded) !=
                        forevervalidator::simulation::
                                CudaStateConversionResult::Success ||
        forevervalidator::simulation::EncodeCudaCandidateState(
                cpuForce, impl->incrementalValidationSeed,
                0u, 0u, randomState, &cpuForceEncoded) !=
                        forevervalidator::simulation::
                                CudaStateConversionResult::Success) {
        result.diagnostic =
                "CUDA pre-collision state conversion failed";
        return result;
    }
    initialEncoded.vehicle.mobil.absorbContactEnabled = true;
    initialEncoded.vehicle.mobil.physicsUpdatesEnabled = true;
    const auto gpuForce = forevervalidator::simulation::
            ExecuteCudaVehicleForcePassForCertification(
                    impl->cudaDeviceConfiguration.DeviceData(),
                    initialEncoded, dt);
    const auto gpu = forevervalidator::simulation::
            ExecuteCudaPreCollisionForCertification(
                    impl->cudaDeviceConfiguration.DeviceData(),
                    initialEncoded, dt);
    if (!gpu.success) {
        result.diagnostic = gpu.diagnostic;
        return result;
    }
    const auto *cpuBytes =
            reinterpret_cast<const std::uint8_t *>(&cpuEncoded);
    const auto *gpuBytes =
            reinterpret_cast<const std::uint8_t *>(&gpu.finalState);
    std::size_t mismatch = 0u;
    while (mismatch < sizeof(cpuEncoded) &&
           cpuBytes[mismatch] == gpuBytes[mismatch]) {
        ++mismatch;
    }
    result.checkedBytes = sizeof(cpuEncoded);
    if (mismatch != sizeof(cpuEncoded)) {
        result.firstMismatchByte = mismatch;
        result.cpuByte = cpuBytes[mismatch];
        result.gpuByte = gpuBytes[mismatch];
        const auto localSpeed = [](
                const forevervalidator::simulation::
                        CudaCandidateState &state) {
            const GmMat3 &rotation = state.body.current.rotation;
            const GmVec3 &speed = state.body.current.linearSpeed;
            return GmVec3{
                    rotation.basisX.x * speed.x +
                            rotation.basisX.y * speed.y +
                            rotation.basisX.z * speed.z,
                    rotation.basisY.x * speed.x +
                            rotation.basisY.y * speed.y +
                            rotation.basisY.z * speed.z,
                    rotation.basisZ.x * speed.x +
                            rotation.basisZ.y * speed.y +
                            rotation.basisZ.z * speed.z};
        };
        const GmVec3 cpuForceLocal = localSpeed(cpuForceEncoded);
        const GmVec3 gpuForceLocal = localSpeed(gpuForce.finalState);
        const GmVec3 initialLocal = localSpeed(initialEncoded);
        result.diagnostic =
                "CUDA pre-collision state diverged at candidate byte " +
                std::to_string(mismatch) + " dt=" +
                std::to_string(dt) +
                " cpu_position=(" +
                std::to_string(cpuEncoded.body.current.position.x) + "," +
                std::to_string(cpuEncoded.body.current.position.y) + "," +
                std::to_string(cpuEncoded.body.current.position.z) + ")" +
                " gpu_position=(" +
                std::to_string(gpu.finalState.body.current.position.x) + "," +
                std::to_string(gpu.finalState.body.current.position.y) + "," +
                std::to_string(gpu.finalState.body.current.position.z) + ")" +
                " initial_linear=(" +
                std::to_string(
                        initialEncoded.body.current.linearSpeed.x) + "," +
                std::to_string(
                        initialEncoded.body.current.linearSpeed.y) + "," +
                std::to_string(
                        initialEncoded.body.current.linearSpeed.z) + ")" +
                " cpu_forcepass_linear=(" +
                std::to_string(
                        cpuForceEncoded.body.current.linearSpeed.x) + "," +
                std::to_string(
                        cpuForceEncoded.body.current.linearSpeed.y) + "," +
                std::to_string(
                        cpuForceEncoded.body.current.linearSpeed.z) + ")" +
                " cpu_forcepass_angular=(" +
                std::to_string(
                        cpuForceEncoded.body.current.angularSpeed.x) + "," +
                std::to_string(
                        cpuForceEncoded.body.current.angularSpeed.y) + "," +
                std::to_string(
                        cpuForceEncoded.body.current.angularSpeed.z) + ")" +
                " cpu_linear=(" +
                std::to_string(
                        cpuEncoded.body.current.linearSpeed.x) + "," +
                std::to_string(
                        cpuEncoded.body.current.linearSpeed.y) + "," +
                std::to_string(
                        cpuEncoded.body.current.linearSpeed.z) + ")" +
                " gpu_linear=(" +
                std::to_string(
                        gpu.finalState.body.current.linearSpeed.x) + "," +
                std::to_string(
                        gpu.finalState.body.current.linearSpeed.y) + "," +
                std::to_string(
                        gpu.finalState.body.current.linearSpeed.z) + ")" +
                " cpu_force=(" +
                std::to_string(cpuEncoded.body.current.force.x) + "," +
                std::to_string(cpuEncoded.body.current.force.y) + "," +
                std::to_string(cpuEncoded.body.current.force.z) + ")" +
                " gpu_force=(" +
                std::to_string(
                        gpu.finalState.body.current.force.x) + "," +
                std::to_string(
                        gpu.finalState.body.current.force.y) + "," +
                std::to_string(
                        gpu.finalState.body.current.force.z) + ")" +
                " gpu_forcepass_linear=(" +
                std::to_string(
                        gpuForce.finalState.body.current.linearSpeed.x) + "," +
                std::to_string(
                        gpuForce.finalState.body.current.linearSpeed.y) + "," +
                std::to_string(
                        gpuForce.finalState.body.current.linearSpeed.z) + ")" +
                " gpu_forcepass_angular=(" +
                std::to_string(
                        gpuForce.finalState.body.current.angularSpeed.x) + "," +
                std::to_string(
                        gpuForce.finalState.body.current.angularSpeed.y) + "," +
                std::to_string(
                        gpuForce.finalState.body.current.angularSpeed.z) + ")" +
                " cpu_forcepass_local=(" +
                std::to_string(cpuForceLocal.x) + "," +
                std::to_string(cpuForceLocal.y) + "," +
                std::to_string(cpuForceLocal.z) + ")" +
                " gpu_forcepass_local=(" +
                std::to_string(gpuForceLocal.x) + "," +
                std::to_string(gpuForceLocal.y) + "," +
                std::to_string(gpuForceLocal.z) + ")" +
                " initial_local=(" +
                std::to_string(initialLocal.x) + "," +
                std::to_string(initialLocal.y) + "," +
                std::to_string(initialLocal.z) + ")" +
                " speed_cap=" +
                std::to_string(initialEncoded.vehicle.linearSpeedCap) +
                " cpu_mass=" +
                std::to_string(cpuForceEncoded.body.parameters.mass) +
                " gpu_mass=" +
                std::to_string(
                        gpuForce.finalState.body.parameters.mass) +
                " cpu_accum_impulse=(" +
                std::to_string(
                        cpuForceEncoded.vehicle.forceAccumulators.
                                impulse.x) + "," +
                std::to_string(
                        cpuForceEncoded.vehicle.forceAccumulators.
                                impulse.y) + "," +
                std::to_string(
                        cpuForceEncoded.vehicle.forceAccumulators.
                                impulse.z) + ")" +
                " gpu_accum_impulse=(" +
                std::to_string(
                        gpuForce.finalState.vehicle.forceAccumulators.
                                impulse.x) + "," +
                std::to_string(
                        gpuForce.finalState.vehicle.forceAccumulators.
                                impulse.y) + "," +
                std::to_string(
                        gpuForce.finalState.vehicle.forceAccumulators.
                                impulse.z) + ")" +
                " special=(" +
                std::to_string(
                        initialEncoded.vehicle.controls.
                                specialContactResponseGate) + "," +
                std::to_string(static_cast<unsigned>(
                        initialEncoded.vehicle.controls.
                                specialContactResponseMode)) + ")" +
                " cpu_splash=(" +
                std::to_string(
                        cpuForceEncoded.vehicle.water.splashPending) + "," +
                std::to_string(
                        cpuForceEncoded.vehicle.water.
                                splashLocalSpeed.x) + "," +
                std::to_string(
                        cpuForceEncoded.vehicle.water.
                                splashLocalSpeed.y) + "," +
                std::to_string(
                        cpuForceEncoded.vehicle.water.
                                splashLocalSpeed.z) + ")" +
                " gpu_splash=(" +
                std::to_string(
                        gpuForce.finalState.vehicle.water.
                                splashPending) + "," +
                std::to_string(
                        gpuForce.finalState.vehicle.water.
                                splashLocalSpeed.x) + "," +
                std::to_string(
                        gpuForce.finalState.vehicle.water.
                                splashLocalSpeed.y) + "," +
                std::to_string(
                        gpuForce.finalState.vehicle.water.
                                splashLocalSpeed.z) + ")";
        return result;
    }
    result.success = true;
    result.firstMismatchByte = SIZE_MAX;
    result.diagnostic =
            "CUDA pre-collision state is bit-exact";
    return result;
#endif
}

ReplayCudaVehiclePrefixDifferential
ReplaySimulationSession::RunCudaTimelineTickDifferentialForTesting(
        const ReplayControlTick &tick) {
    ReplayCudaVehiclePrefixDifferential result;
#if !FOREVERVALIDATOR_HAS_CUDA
    static_cast<void>(tick);
    result.diagnostic =
            "CUDA support is not compiled into this build";
    return result;
#else
    if (impl->backend != forevervalidator::SimulationBackend::Cuda ||
        !impl->instance.runtime ||
        !impl->cudaDeviceConfiguration.Ready() ||
        !impl->cudaDeviceScene.Ready()) {
        result.diagnostic =
                "CUDA timeline-tick differential prerequisites are not ready";
        return result;
    }
    const auto runtime =
            impl->instance.runtime->CaptureRuntimeClone();
    if (!runtime.has_value()) {
        result.diagnostic =
                "CPU timeline-tick initial capture failed";
        return result;
    }
    ReplaySimulationInstanceClone initial;
    initial.race = impl->instance.race.CaptureRuntimeClone();
    initial.runtime = *runtime;
    initial.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    const std::uint32_t initialRandomState =
            tmnf::simulation::CaptureGameRandomState();

    const ReplaySimulationStepExecution execution =
            impl->instance.runtime->
                    StepOptimizedCpuNativeBinary32(tick);
    if (execution.result != ReplaySimulationRunResult::Success) {
        impl->instance.race.RestoreRuntimeClone(initial.race);
        impl->instance.runtime->RestoreRuntimeClone(initial.runtime);
        tmnf::simulation::RestoreGameRandomState(
                initialRandomState);
        result.diagnostic =
                "CPU timeline-tick reference execution failed";
        return result;
    }
    impl->instance.incrementalRespawnCount +=
            execution.respawnExecutedCount;
    const auto cpuRuntime =
            impl->instance.runtime->CaptureRuntimeClone();
    ReplaySimulationInstanceClone cpu;
    cpu.race = impl->instance.race.CaptureRuntimeClone();
    cpu.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    const std::uint32_t finalRandomState =
            tmnf::simulation::CaptureGameRandomState();
    if (!cpuRuntime.has_value() ||
        !impl->instance.race.PrepareRuntimeCloneRestore(
                initial.race) ||
        !impl->instance.runtime->PrepareRuntimeCloneRestore(
                initial.runtime)) {
        result.diagnostic =
                "CPU timeline-tick final capture or restoration failed";
        return result;
    }
    cpu.runtime = *cpuRuntime;
    impl->instance.race.RestoreRuntimeClone(initial.race);
    impl->instance.runtime->RestoreRuntimeClone(initial.runtime);
    impl->instance.incrementalRespawnCount =
            initial.incrementalRespawnCount;
    tmnf::simulation::RestoreGameRandomState(initialRandomState);

    forevervalidator::simulation::CudaCandidateState initialEncoded;
    forevervalidator::simulation::CudaCandidateState cpuEncoded;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                initial, impl->incrementalValidationSeed,
                0u, 0u, initialRandomState,
                &initialEncoded) !=
                        forevervalidator::simulation::
                                CudaStateConversionResult::Success ||
        forevervalidator::simulation::EncodeCudaCandidateState(
                cpu, impl->incrementalValidationSeed,
                1u, 0u, finalRandomState,
                &cpuEncoded) !=
                        forevervalidator::simulation::
                                CudaStateConversionResult::Success) {
        result.diagnostic =
                "CUDA timeline-tick state conversion failed";
        return result;
    }
    forevervalidator::simulation::CudaCandidateTimelineInput input;
    input.initialState = initialEncoded;
    input.ticks.push_back(
            forevervalidator::simulation::
                    FlattenCudaControlTick(tick));
    const forevervalidator::simulation::CudaTimelineBatchResult gpu =
            forevervalidator::simulation::ExecuteCudaTimelineBatch(
                    impl->cudaDeviceScene.DeviceData(),
                    impl->cudaDeviceConfiguration.DeviceData(),
                    {input});
    if (gpu.status !=
                forevervalidator::simulation::
                        CudaTimelineStatus::Success ||
        gpu.candidates.size() != 1u ||
        gpu.candidates[0].status !=
                forevervalidator::simulation::
                        CudaTimelineStatus::Success) {
        result.diagnostic =
                "CUDA timeline-tick execution failed: " +
                gpu.diagnostic;
        return result;
    }
    const forevervalidator::simulation::CudaCandidateState &gpuState =
            gpu.candidates[0].finalState;
    const auto *cpuBytes =
            reinterpret_cast<const std::uint8_t *>(&cpuEncoded);
    const auto *gpuBytes =
            reinterpret_cast<const std::uint8_t *>(&gpuState);
    const std::size_t semanticExtent =
            reinterpret_cast<const std::uint8_t *>(
                    &cpuEncoded.collisionReplacementOverflow) -
            cpuBytes +
            sizeof(cpuEncoded.collisionReplacementOverflow);
    std::size_t mismatch = 0u;
    while (mismatch < semanticExtent &&
           cpuBytes[mismatch] == gpuBytes[mismatch]) {
        ++mismatch;
    }
    result.checkedBytes = semanticExtent;
    if (mismatch != semanticExtent) {
        result.firstMismatchByte = mismatch;
        result.cpuByte = cpuBytes[mismatch];
        result.gpuByte = gpuBytes[mismatch];
        result.diagnostic =
                "CUDA complete timeline tick diverged at candidate byte " +
                std::to_string(mismatch) +
                " time_ms=" + std::to_string(tick.timeMs) +
                " controls=(" +
                std::to_string(tick.controls.lowSpeedGateA) + "," +
                std::to_string(tick.controls.lowSpeedGateB) + "," +
                std::to_string(tick.controls.steering) + ")" +
                " actions=" +
                std::to_string(
                        forevervalidator::simulation::
                                FlattenCudaControlTick(tick).
                                        actionFlags) +
                " cpu_position=(" +
                std::to_string(cpuEncoded.body.current.position.x) + "," +
                std::to_string(cpuEncoded.body.current.position.y) + "," +
                std::to_string(cpuEncoded.body.current.position.z) + ")" +
                " gpu_position=(" +
                std::to_string(gpuState.body.current.position.x) + "," +
                std::to_string(gpuState.body.current.position.y) + "," +
                std::to_string(gpuState.body.current.position.z) + ")" +
                " cpu_burnout=(" +
                std::to_string(
                        static_cast<std::uint32_t>(
                                cpuEncoded.vehicle.gearedDrive.
                                        burnoutPhase)) + "," +
                std::to_string(
                        cpuEncoded.vehicle.gearedDrive.
                                burnoutStartTick) + "," +
                std::to_string(
                        cpuEncoded.vehicle.gearedDrive.
                                burnoutExitStartTick) + ")" +
                " gpu_burnout=(" +
                std::to_string(
                        static_cast<std::uint32_t>(
                                gpuState.vehicle.gearedDrive.
                                        burnoutPhase)) + "," +
                std::to_string(
                        gpuState.vehicle.gearedDrive.
                                burnoutStartTick) + "," +
                std::to_string(
                        gpuState.vehicle.gearedDrive.
                                burnoutExitStartTick) + ")";
        result.diagnostic +=
                " cpu_splash=(" +
                std::to_string(
                        cpuEncoded.vehicle.frameHistory.physicsPrevious.
                                waterSplashEventCounter) + "," +
                std::to_string(
                        cpuEncoded.vehicle.frameHistory.physicsCurrent.
                                waterSplashEventCounter) + "," +
                std::to_string(
                        cpuEncoded.vehiclePassthrough.asyncCurrent.
                                waterSplashEventCounter) + "," +
                std::to_string(
                        cpuEncoded.vehiclePassthrough.asyncPrevious.
                                waterSplashEventCounter) + ")" +
                " gpu_splash=(" +
                std::to_string(
                        gpuState.vehicle.frameHistory.physicsPrevious.
                                waterSplashEventCounter) + "," +
                std::to_string(
                        gpuState.vehicle.frameHistory.physicsCurrent.
                                waterSplashEventCounter) + "," +
                std::to_string(
                        gpuState.vehiclePassthrough.asyncCurrent.
                                waterSplashEventCounter) + "," +
                std::to_string(
                        gpuState.vehiclePassthrough.asyncPrevious.
                                waterSplashEventCounter) + ")";
        return result;
    }
    result.success = true;
    result.firstMismatchByte = SIZE_MAX;
    result.diagnostic =
            "CUDA complete timeline tick is bit-exact";
    return result;
#endif
}

bool ReplaySimulationSession::StageCudaTimelinePrefixForTesting(
        const ReplayControlTick &tick) {
    return impl->backend ==
                    forevervalidator::SimulationBackend::Cuda &&
            impl->instance.runtime &&
            impl->instance.runtime->PrepareStepForTesting(tick);
}

bool ReplaySimulationSession::StageCollisionSubstepForTesting(
        float dt) {
    if (impl->backend !=
                forevervalidator::SimulationBackend::Cuda ||
        !impl->instance.runtime || !(dt > 0.0f)) {
        return false;
    }
    const auto after =
            impl->instance.runtime->
                    CaptureCollisionSubstepReferenceForTesting(dt);
    if (!after.has_value() ||
        !impl->instance.runtime->PrepareRuntimeCloneRestore(*after)) {
        return false;
    }
    impl->instance.runtime->RestoreRuntimeClone(*after);
    return true;
}

bool ReplaySimulationSession::StageCudaPreCollisionForTesting(float dt) {
#if !FOREVERVALIDATOR_HAS_CUDA
    (void)dt;
    return false;
#else
    if (impl->backend != forevervalidator::SimulationBackend::Cuda ||
        !impl->instance.runtime ||
        !impl->cudaDeviceConfiguration.Ready()) {
        return false;
    }
    const auto runtime =
            impl->instance.runtime->CaptureRuntimeClone();
    if (!runtime.has_value()) return false;
    ReplaySimulationInstanceClone initial;
    initial.race = impl->instance.race.CaptureRuntimeClone();
    initial.runtime = *runtime;
    initial.incrementalRespawnCount =
            impl->instance.incrementalRespawnCount;
    forevervalidator::simulation::CudaCandidateState encoded;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                initial, impl->incrementalValidationSeed,
                0u, 0u,
                tmnf::simulation::CaptureGameRandomState(),
                &encoded) != forevervalidator::simulation::
                        CudaStateConversionResult::Success) {
        return false;
    }
    encoded.vehicle.mobil.absorbContactEnabled = true;
    encoded.vehicle.mobil.physicsUpdatesEnabled = true;
    if (!(dt > 0.0f)) {
        dt = static_cast<float>(
                     encoded.world.schemePeriodMs) *
                0.001f;
    }
    const forevervalidator::simulation::CudaPhysicsStepExecution
            staged = forevervalidator::simulation::
                    ExecuteCudaPreCollisionForCertification(
                    impl->cudaDeviceConfiguration.DeviceData(),
                    encoded, dt);
    if (!staged.success) return false;
    ReplaySimulationInstanceClone decoded;
    if (forevervalidator::simulation::DecodeCudaCandidateState(
                staged.finalState, &decoded) !=
            forevervalidator::simulation::
                    CudaStateConversionResult::Success ||
        !PrepareRuntimeCloneRestore(decoded)) {
        return false;
    }
    RestoreRuntimeClone(std::move(decoded));
    return true;
#endif
}

std::uint64_t ReplaySimulationInstanceSemanticHash(
        const ReplaySimulationInstanceClone &clone) {
    constexpr std::uint64_t Offset = 1469598103934665603ull;
    constexpr std::uint64_t Prime = 1099511628211ull;
    std::uint64_t hash = Offset;
    const std::array<std::uint64_t, 2u> components = {
            ReplayRaceRuntimeSemanticHash(clone.race),
            ReplaySimulationRuntimeSemanticHash(clone.runtime)};
    for (std::uint64_t component : components) {
        for (unsigned shift = 0u; shift < 64u; shift += 8u) {
            hash ^= static_cast<std::uint8_t>(component >> shift);
            hash *= Prime;
        }
    }
    std::uint32_t respawns = clone.incrementalRespawnCount;
    for (unsigned shift = 0u; shift < 32u; shift += 8u) {
        hash ^= static_cast<std::uint8_t>(respawns >> shift);
        hash *= Prime;
    }
    for (unsigned shift = 0u; shift < 32u; shift += 8u) {
        hash ^= static_cast<std::uint8_t>(
                clone.randomState >> shift);
        hash *= Prime;
    }
    return hash;
}
