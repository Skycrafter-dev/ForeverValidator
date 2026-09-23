// Generated from src/simulation/backends/cuda/cuda_physics_step_certification.cu by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_physics_step_certification.h"

#include <hip/hip_runtime.h>

#include "simulation/backends/hip/generated/hip_physics_step.cuh"

namespace forevervalidator::simulation {
namespace {

__global__ void ExecutePhysicsStepKernel(
        const HipPackedSceneHeader *scene,
        const HipPackedStaticConfigurationHeader *configuration,
        HipCandidateState *state,
        hip::collision::HipCollisionScratch *scratch,
        std::uint32_t *status) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    state->vehicle.mobil.absorbContactEnabled = true;
    state->vehicle.mobil.physicsUpdatesEnabled = true;
    *status = static_cast<std::uint32_t>(
            hip::physics::Step(
                    scene, configuration, *state, *scratch));
}

__global__ void ExecutePreCollisionKernel(
        const HipPackedStaticConfigurationHeader *configuration,
        HipCandidateState *state,
        float dt,
        std::uint32_t *status) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    hip::environment::BeginForcePass(
            state->body, configuration);
    hip::vehicle::ForceStatus forceStatus =
            hip::vehicle::ForceStatus::Success;
    if (state->vehicle.mobil.physicsUpdatesEnabled) {
        forceStatus = hip::vehicle::ComputeForcesModel6(
                *state, configuration, dt);
    }
    if (forceStatus != hip::vehicle::ForceStatus::Success) {
        *status = static_cast<std::uint32_t>(forceStatus) + 1u;
        return;
    }
    HipFixedArray<
            GmVec3,
            HipCollisionReplacementOverflowCapacity>
            overflowReplacements{};
    hip::dynamics::PreCollision(
            state->body, overflowReplacements, dt);
    *status = 0u;
}

__global__ void ExecuteCollisionSubstepKernel(
        const HipPackedSceneHeader *scene,
        const HipPackedStaticConfigurationHeader *configuration,
        HipCandidateState *state,
        hip::collision::HipCollisionScratch *scratch,
        float dt,
        std::uint32_t *status) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    *status = static_cast<std::uint32_t>(
            hip::physics::CollisionSubstep(
                    scene, configuration, *state, dt, *scratch));
}

std::string Failure(const char *operation, hipError_t error) {
    return std::string(operation) + " failed: " +
           hipGetErrorName(error) + " (" +
           hipGetErrorString(error) + ")";
}

}  // namespace

HipPhysicsStepExecution ExecuteHipPhysicsStepForCertification(
        const void *deviceScene,
        const void *deviceStaticConfiguration,
        const HipCandidateState &state) noexcept {
    HipPhysicsStepExecution result;
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
        result.diagnostic =
                Failure("hipMalloc(physics step)", error);
        cleanup();
        return result;
    }
    error = hipMemcpy(
            deviceState, &state, sizeof(state),
            hipMemcpyHostToDevice);
    if (error != hipSuccess) {
        result.diagnostic =
                Failure("hipMemcpy(physics state H2D)", error);
        cleanup();
        return result;
    }
    ExecutePhysicsStepKernel<<<1u, 1u>>>(
            static_cast<const HipPackedSceneHeader *>(
                    deviceScene),
            static_cast<
                    const HipPackedStaticConfigurationHeader *>(
                    deviceStaticConfiguration),
            deviceState, deviceScratch, deviceStatus);
    error = hipGetLastError();
    if (error != hipSuccess) {
        result.diagnostic =
                Failure("physics step launch", error);
        cleanup();
        return result;
    }
    std::uint32_t status = UINT32_MAX;
    error = hipMemcpy(
            &status, deviceStatus, sizeof(status),
            hipMemcpyDeviceToHost);
    if (error == hipSuccess) {
        error = hipMemcpy(
                &result.finalState, deviceState,
                sizeof(result.finalState),
                hipMemcpyDeviceToHost);
    }
    cleanup();
    if (error != hipSuccess) {
        result.diagnostic =
                Failure("hipMemcpy(physics state D2H)", error);
        return result;
    }
    if (status != static_cast<std::uint32_t>(
                          hip::physics::Status::Success)) {
        result.diagnostic =
                "HIP physics step returned status " +
                std::to_string(status);
        return result;
    }
    result.success = true;
    result.diagnostic =
            "HIP physics step completed";
    return result;
}

HipPhysicsStepExecution ExecuteHipPreCollisionForCertification(
        const void *deviceStaticConfiguration,
        const HipCandidateState &state,
        float dt) noexcept {
    HipPhysicsStepExecution result;
    if (deviceStaticConfiguration == nullptr || !(dt > 0.0f)) {
        result.diagnostic =
                "invalid HIP pre-collision certification request";
        return result;
    }
    HipCandidateState *deviceState = nullptr;
    std::uint32_t *deviceStatus = nullptr;
    auto cleanup = [&]() {
        if (deviceStatus != nullptr) hipFree(deviceStatus);
        if (deviceState != nullptr) hipFree(deviceState);
    };
    hipError_t error = hipMalloc(
            reinterpret_cast<void **>(&deviceState),
            sizeof(HipCandidateState));
    if (error == hipSuccess) {
        error = hipMalloc(
                reinterpret_cast<void **>(&deviceStatus),
                sizeof(std::uint32_t));
    }
    if (error == hipSuccess) {
        error = hipMemcpy(
                deviceState, &state, sizeof(state),
                hipMemcpyHostToDevice);
    }
    if (error != hipSuccess) {
        result.diagnostic =
                Failure("HIP pre-collision allocation/copy", error);
        cleanup();
        return result;
    }
    ExecutePreCollisionKernel<<<1u, 1u>>>(
            static_cast<const
                    HipPackedStaticConfigurationHeader *>(
                    deviceStaticConfiguration),
            deviceState, dt, deviceStatus);
    error = hipGetLastError();
    std::uint32_t status = UINT32_MAX;
    if (error == hipSuccess) {
        error = hipMemcpy(
                &status, deviceStatus, sizeof(status),
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
        result.diagnostic =
                Failure("HIP pre-collision execution", error);
        return result;
    }
    if (status != 0u) {
        result.diagnostic =
                "HIP pre-collision force status " +
                std::to_string(status);
        return result;
    }
    result.success = true;
    result.diagnostic =
            "HIP pre-collision state completed";
    return result;
}

HipPhysicsStepExecution
ExecuteHipCollisionSubstepForCertification(
        const void *deviceScene,
        const void *deviceStaticConfiguration,
        const HipCandidateState &state,
        float dt) noexcept {
    HipPhysicsStepExecution result;
    if (deviceScene == nullptr ||
        deviceStaticConfiguration == nullptr || !(dt > 0.0f)) {
        result.diagnostic =
                "invalid HIP collision-substep certification request";
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
    if (error == hipSuccess) {
        error = hipMemcpy(
                deviceState, &state, sizeof(state),
                hipMemcpyHostToDevice);
    }
    if (error != hipSuccess) {
        result.diagnostic =
                Failure("HIP collision-substep allocation/copy", error);
        cleanup();
        return result;
    }
    ExecuteCollisionSubstepKernel<<<1u, 1u>>>(
            static_cast<const HipPackedSceneHeader *>(deviceScene),
            static_cast<const HipPackedStaticConfigurationHeader *>(
                    deviceStaticConfiguration),
            deviceState, deviceScratch, dt, deviceStatus);
    error = hipGetLastError();
    std::uint32_t status = UINT32_MAX;
    if (error == hipSuccess) {
        error = hipMemcpy(
                &status, deviceStatus, sizeof(status),
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
        result.diagnostic =
                Failure("HIP collision-substep execution", error);
        return result;
    }
    if (status != static_cast<std::uint32_t>(
                          hip::physics::Status::Success)) {
        result.diagnostic =
                "HIP collision substep returned status " +
                std::to_string(status);
        return result;
    }
    result.success = true;
    result.diagnostic =
            "HIP collision substep completed";
    return result;
}

}  // namespace forevervalidator::simulation
