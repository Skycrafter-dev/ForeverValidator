#include "simulation/backends/simulation_backend.h"

#include "simulation/backends/vulkan/vulkan_compute_runtime.h"

namespace forevervalidator::simulation {

bool IsVulkanBackendReady() noexcept {
    return vulkan::QueryRuntimeDiagnostics().IsReady();
}

}  // namespace forevervalidator::simulation

namespace forevervalidator {

VulkanBackendDiagnostics QueryVulkanBackendDiagnostics() noexcept {
    const simulation::vulkan::RuntimeDiagnostics source =
            simulation::vulkan::QueryRuntimeDiagnostics();
    VulkanBackendDiagnostics result;
    result.status = static_cast<VulkanBackendStatus>(source.status);
    result.exactSearchPhysics = source.IsReady() &&
            simulation::vulkan::SupportsExactSearchPhysics();
    result.apiVersion = source.apiVersion;
    result.driverVersion = source.driverVersion;
    result.vendorId = source.vendorId;
    result.deviceId = source.deviceId;
    result.deviceLocalMemoryBytes = source.deviceLocalMemoryBytes;
    result.subgroupSize = source.subgroupSize;
    result.deviceName = source.deviceName;
    result.driverName = source.driverName;
    result.diagnostic = source.diagnostic;
    return result;
}

}  // namespace forevervalidator
