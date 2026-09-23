// Generated from src/simulation/backends/cuda/cuda_vehicle_force_certification.cu by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_vehicle_force_certification.h"

#include <hip/hip_runtime.h>

#include <string>

#include "simulation/backends/hip/generated/hip_static_configuration.h"
#include "simulation/backends/hip/generated/hip_environment.cuh"
#include "simulation/backends/hip/generated/hip_vehicle_forces.cuh"

namespace forevervalidator::simulation {
namespace {

__global__ void ExecuteVehicleForceKernel(
        const void *configurationData,
        HipCandidateState *state,
        float dt,
        bool beginEnvironment,
        std::uint32_t *status) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    const auto *configuration =
            static_cast<const HipPackedStaticConfigurationHeader *>(
                    configurationData);
    if (configuration == nullptr ||
        configuration->magic !=
                HipPackedStaticConfigurationHeader::Magic ||
        configuration->schemaVersion !=
                HipPackedStaticConfigurationHeader::SchemaVersion ||
        state == nullptr ||
        state->schemaVersion != HipCandidateState::SchemaVersion ||
        !(dt > 0.0f)) {
        *status = UINT32_MAX;
        return;
    }
    if (beginEnvironment) {
        hip::environment::BeginForcePass(
                state->body, configuration);
    }
    *status = static_cast<std::uint32_t>(
            hip::vehicle::ComputeForcesModel6(
                    *state, configuration, dt));
}

std::string Failure(const char *operation, hipError_t error) {
    return std::string(operation) + " failed: " +
           hipGetErrorName(error) + " (" +
           hipGetErrorString(error) + ")";
}

}  // namespace

HipVehicleForceExecution ExecuteHipVehicleForceForCertificationImpl(
        const void *deviceStaticConfiguration,
        const HipCandidateState &initialState,
        float dt,
        bool beginEnvironment) noexcept {
    HipVehicleForceExecution result;
    if (deviceStaticConfiguration == nullptr ||
        initialState.schemaVersion !=
                HipCandidateState::SchemaVersion ||
        !(dt > 0.0f)) {
        result.diagnostic =
                "invalid HIP vehicle-force certification request";
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
    if (error != hipSuccess) {
        result.diagnostic = Failure("hipMalloc(state)", error);
        return result;
    }
    error = hipMalloc(
            reinterpret_cast<void **>(&deviceStatus),
            sizeof(std::uint32_t));
    if (error != hipSuccess) {
        result.diagnostic = Failure("hipMalloc(status)", error);
        cleanup();
        return result;
    }
    error = hipMemcpy(
            deviceState, &initialState, sizeof(HipCandidateState),
            hipMemcpyHostToDevice);
    if (error != hipSuccess) {
        result.diagnostic = Failure("hipMemcpy(state H2D)", error);
        cleanup();
        return result;
    }
    ExecuteVehicleForceKernel<<<1u, 1u>>>(
            deviceStaticConfiguration, deviceState, dt,
            beginEnvironment, deviceStatus);
    error = hipGetLastError();
    if (error != hipSuccess) {
        result.diagnostic = Failure("vehicle-force launch", error);
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
                sizeof(HipCandidateState),
                hipMemcpyDeviceToHost);
    }
    cleanup();
    if (error != hipSuccess) {
        result.diagnostic = Failure("hipMemcpy(D2H)", error);
        return result;
    }
    result.success = status != UINT32_MAX;
    result.supported =
            status == static_cast<std::uint32_t>(
                    hip::vehicle::ForceStatus::Success);
    result.diagnostic = result.supported
            ? "HIP vehicle-force certification kernel completed"
            : "HIP vehicle-force transition is not implemented";
    return result;
}

HipVehicleForceExecution ExecuteHipVehicleForceForCertification(
        const void *deviceStaticConfiguration,
        const HipCandidateState &initialState,
        float dt) noexcept {
    return ExecuteHipVehicleForceForCertificationImpl(
            deviceStaticConfiguration, initialState, dt, false);
}

HipVehicleForceExecution ExecuteHipVehicleForcePassForCertification(
        const void *deviceStaticConfiguration,
        const HipCandidateState &initialState,
        float dt) noexcept {
    return ExecuteHipVehicleForceForCertificationImpl(
            deviceStaticConfiguration, initialState, dt, true);
}

}  // namespace forevervalidator::simulation
