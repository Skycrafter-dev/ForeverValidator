#include "simulation/backends/simulation_backend.h"

namespace forevervalidator::simulation {

SimulationBackend ResolveLeafBackend(SimulationBackend backend) noexcept {
    switch (backend) {
    case SimulationBackend::OptimizedCpu:
        return SimulationBackend::OptimizedCpu;
    case SimulationBackend::SpeculativeTicking:
        return SimulationBackend::SpeculativeTicking;
    case SimulationBackend::Cuda:
        return SimulationBackend::Cuda;
    case SimulationBackend::Vulkan:
        return SimulationBackend::Vulkan;
    case SimulationBackend::Batched:
        return SimulationBackend::Reference;
    case SimulationBackend::Reference:
        return SimulationBackend::Reference;
    }
    return SimulationBackend::Reference;
}

bool UsesOptimizedCpuFoundation(SimulationBackend backend) noexcept {
    return backend == SimulationBackend::OptimizedCpu ||
           backend == SimulationBackend::SpeculativeTicking;
}

}  // namespace forevervalidator::simulation
