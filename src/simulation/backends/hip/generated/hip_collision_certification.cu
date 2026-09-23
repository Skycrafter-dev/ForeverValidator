// Generated from src/simulation/backends/cuda/cuda_collision_certification.cu by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_collision_certification.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <new>
#include <string>

#include "simulation/backends/hip/generated/hip_collision_response.cuh"

namespace forevervalidator::simulation {
namespace {

__global__ void ExecuteCollisionKernel(
        const void *sceneData,
        const void *configurationData,
        HipCandidateState *state,
        hip::collision::HipCollisionScratch *scratch,
        std::uint32_t *status) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    const auto *scene =
            static_cast<const HipPackedSceneHeader *>(sceneData);
    const auto *configuration =
            static_cast<
                    const HipPackedStaticConfigurationHeader *>(
                    configurationData);
    hip::collision::Status collisionStatus =
            hip::collision::Detect(
                    scene, configuration, *state, *scratch);
    if (collisionStatus ==
        hip::collision::Status::Success) {
        collisionStatus = hip::collision::Respond(
                scene, configuration, *state, *scratch);
    }
    *status = static_cast<std::uint32_t>(collisionStatus);
}

__global__ void ExecuteCollisionOrderingKernel(
        hip::collision::HipCollisionScratch *scratch) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    hip::collision::detail::SortForResponse(*scratch);
}

__global__ void ExecuteShapeWorldPoseKernel(
        const HipVehicleCollisionShape *shapes,
        const HipCandidateState *state,
        std::uint32_t shapeIndex,
        GmIso4 *worldPose) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    *worldPose = hip::collision::detail::ShapeWorldPose(
            shapeIndex, shapes, *state,
            hip::collision::detail::BodyPose(state->body));
}

std::string Failure(const char *operation, hipError_t error) {
    return std::string(operation) + " failed: " +
           hipGetErrorName(error) + " (" +
           hipGetErrorString(error) + ")";
}

}  // namespace

HipCollisionExecution ExecuteHipCollisionForCertification(
        const void *deviceScene,
        const void *deviceStaticConfiguration,
        const HipCandidateState &state) noexcept {
    HipCollisionExecution result;
    if (deviceScene == nullptr ||
        deviceStaticConfiguration == nullptr ||
        state.schemaVersion != HipCandidateState::SchemaVersion) {
        result.diagnostic =
                "invalid HIP collision certification request";
        return result;
    }
    HipCandidateState *deviceState = nullptr;
    hip::collision::HipCollisionScratch *deviceScratch = nullptr;
    std::uint32_t *deviceStatus = nullptr;
    auto cleanup = [&]() {
        if (deviceStatus != nullptr) hipFree(deviceStatus);
        if (deviceScratch != nullptr) hipFree(deviceScratch);
        if (deviceState != nullptr) hipFree(deviceState);
    };
    hipError_t error = hipMalloc(
            reinterpret_cast<void **>(&deviceState),
            sizeof(HipCandidateState));
    if (error == hipSuccess) {
        error = hipMalloc(
                reinterpret_cast<void **>(&deviceScratch),
                sizeof(hip::collision::HipCollisionScratch));
    }
    if (error == hipSuccess) {
        error = hipMalloc(
                reinterpret_cast<void **>(&deviceStatus),
                sizeof(std::uint32_t));
    }
    if (error != hipSuccess) {
        result.diagnostic = Failure("hipMalloc(collision)", error);
        cleanup();
        return result;
    }
    error = hipMemcpy(
            deviceState, &state, sizeof(state),
            hipMemcpyHostToDevice);
    if (error != hipSuccess) {
        result.diagnostic = Failure("hipMemcpy(state H2D)", error);
        cleanup();
        return result;
    }
    ExecuteCollisionKernel<<<1u, 1u>>>(
            deviceScene, deviceStaticConfiguration, deviceState,
            deviceScratch, deviceStatus);
    error = hipGetLastError();
    if (error != hipSuccess) {
        result.diagnostic = Failure("collision launch", error);
        cleanup();
        return result;
    }
    std::uint32_t status = UINT32_MAX;
    hip::collision::HipCollisionScratch hostScratch;
    error = hipMemcpy(
            &status, deviceStatus, sizeof(status),
            hipMemcpyDeviceToHost);
    if (error == hipSuccess) {
        error = hipMemcpy(
                &hostScratch, deviceScratch, sizeof(hostScratch),
                hipMemcpyDeviceToHost);
    }
    if (error == hipSuccess) {
        error = hipMemcpy(
                &result.finalState, deviceState,
                sizeof(result.finalState),
                hipMemcpyDeviceToHost);
    }
    cleanup();
    if (error != hipSuccess) {
        result.diagnostic = Failure("hipMemcpy(collision D2H)", error);
        return result;
    }
    result.status =
            static_cast<hip::collision::Status>(status);
    result.accelerationCellVisits =
            hostScratch.accelerationCellVisits;
    result.accelerationSurfaceVisits =
            hostScratch.accelerationSurfaceVisits;
    result.meshCellVisits = hostScratch.meshCellVisits;
    result.meshCellIntersections =
            hostScratch.meshCellIntersections;
    result.meshTriangleCells = hostScratch.meshTriangleCells;
    result.triangleTests = hostScratch.triangleTests;
    result.triangleHits = hostScratch.triangleHits;
    result.firstVisitedShape = hostScratch.firstVisitedShape;
    result.firstVisitedSurface = hostScratch.firstVisitedSurface;
    result.firstShapeWorld = hostScratch.firstShapeWorld;
    result.firstMovingBounds = hostScratch.firstMovingBounds;
    result.firstEllipsoidBox = hostScratch.firstEllipsoidBox;
    result.firstSurfaceWorldBounds =
            hostScratch.firstSurfaceWorldBounds;
    result.firstMeshRootBounds = hostScratch.firstMeshRootBounds;
    result.firstResponseWheelIndex =
            hostScratch.firstResponseWheelIndex;
    result.firstResponseReplacementBefore =
            hostScratch.firstResponseReplacementBefore;
    result.firstResponseReplacementAfter =
            hostScratch.firstResponseReplacementAfter;
    if (result.status != hip::collision::Status::Success) {
        result.diagnostic =
                "HIP collision kernel returned status " +
                std::to_string(status);
        return result;
    }
    try {
        result.collisions.assign(
                hostScratch.collisions,
                hostScratch.collisions +
                        hostScratch.collisionCount);
    } catch (...) {
        result.diagnostic =
                "HIP collision result allocation failed";
        return result;
    }
    result.success = true;
    result.diagnostic =
            "HIP collision certification kernel completed";
    return result;
}

HipCollisionOrderingExecution
ExecuteHipCollisionOrderingForCertification(
        const std::vector<hip::collision::HipCollision> &collisions)
        noexcept {
    HipCollisionOrderingExecution result;
    if (collisions.size() >
        hip::collision::CollisionCapacity) {
        result.diagnostic =
                "HIP collision ordering input exceeds capacity";
        return result;
    }
    hip::collision::HipCollisionScratch host;
    host.collisionCount =
            static_cast<std::uint32_t>(collisions.size());
    std::copy(
            collisions.begin(), collisions.end(),
            host.collisions);
    hip::collision::HipCollisionScratch *device = nullptr;
    hipError_t error = hipMalloc(
            reinterpret_cast<void **>(&device), sizeof(host));
    if (error == hipSuccess) {
        error = hipMemcpy(
                device, &host, sizeof(host),
                hipMemcpyHostToDevice);
    }
    if (error == hipSuccess) {
        ExecuteCollisionOrderingKernel<<<1u, 1u>>>(device);
        error = hipGetLastError();
    }
    if (error == hipSuccess) {
        error = hipMemcpy(
                &host, device, sizeof(host),
                hipMemcpyDeviceToHost);
    }
    if (device != nullptr) hipFree(device);
    if (error != hipSuccess) {
        result.diagnostic =
                Failure("HIP collision ordering", error);
        return result;
    }
    try {
        result.collisions.assign(
                host.collisions,
                host.collisions + host.collisionCount);
    } catch (const std::bad_alloc &) {
        result.diagnostic =
                "HIP collision ordering result allocation failed";
        return result;
    }
    result.success = true;
    result.diagnostic =
            "HIP collision response ordering completed";
    return result;
}

HipShapeWorldPoseExecution ExecuteHipShapeWorldPoseForCertification(
        const std::vector<HipVehicleCollisionShape> &shapes,
        const HipCandidateState &state,
        std::uint32_t shapeIndex) noexcept {
    HipShapeWorldPoseExecution result;
    if (shapes.empty() || shapeIndex >= shapes.size() ||
        state.schemaVersion != HipCandidateState::SchemaVersion) {
        result.diagnostic =
                "invalid HIP shape-world certification request";
        return result;
    }
    HipVehicleCollisionShape *deviceShapes = nullptr;
    HipCandidateState *deviceState = nullptr;
    GmIso4 *devicePose = nullptr;
    hipError_t error = hipMalloc(
            reinterpret_cast<void **>(&deviceShapes),
            shapes.size() * sizeof(shapes.front()));
    if (error == hipSuccess) {
        error = hipMalloc(
                reinterpret_cast<void **>(&deviceState),
                sizeof(HipCandidateState));
    }
    if (error == hipSuccess) {
        error = hipMalloc(
                reinterpret_cast<void **>(&devicePose),
                sizeof(GmIso4));
    }
    if (error == hipSuccess) {
        error = hipMemcpy(
                deviceShapes, shapes.data(),
                shapes.size() * sizeof(shapes.front()),
                hipMemcpyHostToDevice);
    }
    if (error == hipSuccess) {
        error = hipMemcpy(
                deviceState, &state, sizeof(HipCandidateState),
                hipMemcpyHostToDevice);
    }
    if (error == hipSuccess) {
        ExecuteShapeWorldPoseKernel<<<1u, 1u>>>(
                deviceShapes, deviceState, shapeIndex, devicePose);
        error = hipGetLastError();
    }
    if (error == hipSuccess) {
        error = hipMemcpy(
                &result.worldPose, devicePose, sizeof(GmIso4),
                hipMemcpyDeviceToHost);
    }
    if (devicePose != nullptr) hipFree(devicePose);
    if (deviceState != nullptr) hipFree(deviceState);
    if (deviceShapes != nullptr) hipFree(deviceShapes);
    if (error != hipSuccess) {
        result.diagnostic =
                Failure("HIP shape-world certification", error);
        return result;
    }
    result.success = true;
    result.diagnostic =
            "HIP hierarchical shape pose is bit-exact";
    return result;
}

}  // namespace forevervalidator::simulation
