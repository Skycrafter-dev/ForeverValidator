// Generated from src/simulation/backends/cuda/cuda_race_certification.cu by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_race_certification.h"

#include <hip/hip_runtime.h>

#include <string>

#include "simulation/backends/hip/generated/hip_race.cuh"

namespace forevervalidator::simulation {
namespace {

__global__ void ExecuteRaceContactKernel(
        HipCandidateState *state,
        const HipSceneActor *actor) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    hip::race::OnTriggerContact(*state, *actor);
}

std::string Failure(const char *operation, hipError_t error) {
    return std::string(operation) + " failed: " +
           hipGetErrorName(error) + " (" +
           hipGetErrorString(error) + ")";
}

}  // namespace

HipRaceContactExecution ExecuteHipRaceContactForCertification(
        const HipCandidateState &initialState,
        const HipSceneActor &actor) noexcept {
    HipRaceContactExecution result;
    HipCandidateState *deviceState = nullptr;
    HipSceneActor *deviceActor = nullptr;
    auto cleanup = [&]() {
        if (deviceActor != nullptr) hipFree(deviceActor);
        if (deviceState != nullptr) hipFree(deviceState);
    };

    hipError_t error = hipMalloc(
            reinterpret_cast<void **>(&deviceState),
            sizeof(HipCandidateState));
    if (error == hipSuccess) {
        error = hipMalloc(
                reinterpret_cast<void **>(&deviceActor),
                sizeof(HipSceneActor));
    }
    if (error == hipSuccess) {
        error = hipMemcpy(
                deviceState, &initialState,
                sizeof(HipCandidateState), hipMemcpyHostToDevice);
    }
    if (error == hipSuccess) {
        error = hipMemcpy(
                deviceActor, &actor,
                sizeof(HipSceneActor), hipMemcpyHostToDevice);
    }
    if (error != hipSuccess) {
        result.diagnostic = Failure("HIP race contact setup", error);
        cleanup();
        return result;
    }

    ExecuteRaceContactKernel<<<1u, 1u>>>(deviceState, deviceActor);
    error = hipGetLastError();
    if (error == hipSuccess) {
        error = hipMemcpy(
                &result.finalState, deviceState,
                sizeof(HipCandidateState), hipMemcpyDeviceToHost);
    }
    cleanup();
    if (error != hipSuccess) {
        result.diagnostic = Failure("HIP race contact execution", error);
        return result;
    }
    result.success = true;
    result.diagnostic = "HIP race contact certification kernel completed";
    return result;
}

}  // namespace forevervalidator::simulation
