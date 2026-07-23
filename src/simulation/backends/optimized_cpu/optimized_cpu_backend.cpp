#include "simulation/backends/simulation_backend.h"

namespace forevervalidator::simulation {

SimulationBackend ResolveLeafBackend(SimulationBackend backend) noexcept {
    switch (backend) {
    case SimulationBackend::OptimizedCpu:
        return SimulationBackend::OptimizedCpu;
    case SimulationBackend::Batched:
        return SimulationBackend::Reference;
    case SimulationBackend::Reference:
        return SimulationBackend::Reference;
    }
    return SimulationBackend::Reference;
}

}  // namespace forevervalidator::simulation
