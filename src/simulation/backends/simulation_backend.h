#ifndef FOREVERVALIDATOR_SIMULATION_BACKEND_H
#define FOREVERVALIDATOR_SIMULATION_BACKEND_H

#include <cstddef>
#include <functional>

#include <forevervalidator/validation.h>

namespace forevervalidator::simulation {

bool IsSimulationBackendSupported(SimulationBackend backend) noexcept;
SimulationBackend ResolveLeafBackend(SimulationBackend backend) noexcept;
bool UsesOptimizedCpuFoundation(SimulationBackend backend) noexcept;
bool IsCudaBackendReady() noexcept;
bool IsVulkanBackendReady() noexcept;
void ExecuteBatched(std::size_t count,
                    const std::function<void(std::size_t)> &operation);

}  // namespace forevervalidator::simulation

#endif
