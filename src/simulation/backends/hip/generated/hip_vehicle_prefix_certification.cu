// Generated from src/simulation/backends/cuda/cuda_vehicle_prefix_certification.cu by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_vehicle_prefix_certification.h"

#include <hip/hip_runtime.h>

#include <string>

#include "simulation/backends/hip/generated/hip_static_configuration.h"
#include "simulation/backends/hip/generated/hip_vehicle_wheels.cuh"

namespace forevervalidator::simulation {
namespace {

__global__ void ExecuteVehiclePrefixKernel(
        const void *configurationData,
        HipCandidateState *state,
        float dt,
        std::uint32_t *status) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
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
        *status = 1u;
        return;
    }
    hip::vehicle::IntegrateVehiclePrefix(
            *state, configuration, dt);
    *status = 0u;
}

std::string Failure(const char *operation, hipError_t error) {
    return std::string(operation) + " failed: " +
           hipGetErrorName(error) + " (" +
           hipGetErrorString(error) + ")";
}

}  // namespace

HipVehiclePrefixExecution ExecuteHipVehiclePrefixForCertification(
        const void *deviceStaticConfiguration,
        const HipCandidateState &initialState,
        float dt) noexcept {
    HipVehiclePrefixExecution result;
    if (deviceStaticConfiguration == nullptr ||
        initialState.schemaVersion !=
                HipCandidateState::SchemaVersion ||
        !(dt > 0.0f)) {
        result.diagnostic =
                "invalid HIP vehicle-prefix certification request";
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

    ExecuteVehiclePrefixKernel<<<1u, 1u>>>(
            deviceStaticConfiguration, deviceState, dt, deviceStatus);
    error = hipGetLastError();
    if (error != hipSuccess) {
        result.diagnostic = Failure("vehicle-prefix launch", error);
        cleanup();
        return result;
    }

    std::uint32_t status = 1u;
    error = hipMemcpy(
            &status, deviceStatus, sizeof(status),
            hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
        result.diagnostic = Failure("hipMemcpy(status D2H)", error);
        cleanup();
        return result;
    }
    error = hipMemcpy(
            &result.finalState, deviceState,
            sizeof(HipCandidateState), hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
        result.diagnostic = Failure("hipMemcpy(state D2H)", error);
        cleanup();
        return result;
    }
    cleanup();
    if (status != 0u) {
        result.diagnostic =
                "HIP vehicle-prefix kernel rejected the request";
        return result;
    }
    result.success = true;
    result.diagnostic =
            "HIP vehicle-prefix certification kernel completed";
    return result;
}

}  // namespace forevervalidator::simulation
