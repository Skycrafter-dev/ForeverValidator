// Generated from src/simulation/backends/cuda/cuda_stunt_certification.cu by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_stunt_certification.h"

#include <hip/hip_runtime.h>

#include <string>

#include "simulation/backends/hip/generated/hip_stunts.cuh"

namespace forevervalidator::simulation {
namespace {

struct DeviceResult {
    std::uint32_t failureCommand = UINT32_MAX;
    std::uint32_t failureDetail = 0u;
};

__global__ void ExecuteStuntCommandsKernel(
        HipRaceState *race,
        const HipStuntCommand *commands,
        std::uint32_t commandCount,
        DeviceResult *result) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    for (std::uint32_t index = 0u; index < commandCount; ++index) {
        hip::stunts::Status status =
                hip::stunts::Status::Success;
        switch (commands[index].kind) {
        case HipStuntCommandKind::Update:
            status = hip::stunts::UpdateState(
                    *race, commands[index].state);
            break;
        case HipStuntCommandKind::RespawnPenalty:
            hip::stunts::ApplyRespawnPenalty(race->stunts);
            break;
        case HipStuntCommandKind::TimePenalty:
            status = hip::stunts::ApplyTimePenalty(
                    *race, commands[index].overtimeMs);
            break;
        default:
            result->failureCommand = index;
            result->failureDetail = UINT32_MAX;
            return;
        }
        if (status != hip::stunts::Status::Success) {
            result->failureCommand = index;
            result->failureDetail =
                    static_cast<std::uint32_t>(status);
            return;
        }
    }
}

std::string Failure(const char *operation, hipError_t error) {
    return std::string(operation) + " failed: " +
           hipGetErrorName(error) + " (" +
           hipGetErrorString(error) + ")";
}

}  // namespace

HipStuntExecution ExecuteHipStuntCommandsForCertification(
        const HipRaceState &initialState,
        const std::vector<HipStuntCommand> &commands) noexcept {
    HipStuntExecution result;
    if (commands.empty() ||
        commands.size() > UINT32_MAX) {
        result.diagnostic =
                "invalid HIP stunt certification request";
        return result;
    }
    HipRaceState *deviceRace = nullptr;
    HipStuntCommand *deviceCommands = nullptr;
    DeviceResult *deviceResult = nullptr;
    auto cleanup = [&]() {
        if (deviceResult != nullptr) hipFree(deviceResult);
        if (deviceCommands != nullptr) hipFree(deviceCommands);
        if (deviceRace != nullptr) hipFree(deviceRace);
    };
    hipError_t error = hipMalloc(
            reinterpret_cast<void **>(&deviceRace),
            sizeof(HipRaceState));
    if (error == hipSuccess) {
        error = hipMalloc(
                reinterpret_cast<void **>(&deviceCommands),
                commands.size() * sizeof(HipStuntCommand));
    }
    if (error == hipSuccess) {
        error = hipMalloc(
                reinterpret_cast<void **>(&deviceResult),
                sizeof(DeviceResult));
    }
    DeviceResult emptyResult;
    if (error == hipSuccess) {
        error = hipMemcpy(
                deviceRace, &initialState, sizeof(HipRaceState),
                hipMemcpyHostToDevice);
    }
    if (error == hipSuccess) {
        error = hipMemcpy(
                deviceCommands, commands.data(),
                commands.size() * sizeof(HipStuntCommand),
                hipMemcpyHostToDevice);
    }
    if (error == hipSuccess) {
        error = hipMemcpy(
                deviceResult, &emptyResult, sizeof(DeviceResult),
                hipMemcpyHostToDevice);
    }
    if (error != hipSuccess) {
        result.diagnostic = Failure("HIP stunt setup", error);
        cleanup();
        return result;
    }

    ExecuteStuntCommandsKernel<<<1u, 1u>>>(
            deviceRace, deviceCommands,
            static_cast<std::uint32_t>(commands.size()),
            deviceResult);
    error = hipGetLastError();
    DeviceResult hostResult;
    if (error == hipSuccess) {
        error = hipMemcpy(
                &hostResult, deviceResult, sizeof(DeviceResult),
                hipMemcpyDeviceToHost);
    }
    if (error == hipSuccess) {
        error = hipMemcpy(
                &result.finalState, deviceRace, sizeof(HipRaceState),
                hipMemcpyDeviceToHost);
    }
    cleanup();
    if (error != hipSuccess) {
        result.diagnostic = Failure("HIP stunt execution", error);
        return result;
    }
    result.failureCommand = hostResult.failureCommand;
    result.failureDetail = hostResult.failureDetail;
    result.success = hostResult.failureCommand == UINT32_MAX;
    result.diagnostic = result.success
            ? "HIP stunt certification kernel completed"
            : "HIP stunt certification command failed";
    return result;
}

}  // namespace forevervalidator::simulation
