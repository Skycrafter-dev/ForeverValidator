#include "simulation/backends/simulation_backend.h"
#include "simulation/backends/cuda/cuda_backend.h"

#include <forevervalidator/validation.h>

#include <iostream>

int main() {
    using forevervalidator::SimulationBackend;
    using forevervalidator::simulation::IsSimulationBackendSupported;
    using forevervalidator::simulation::IsCudaComputeCapabilitySupported;
    using forevervalidator::simulation::
            IsCudaSessionSpecializationSupported;
    using forevervalidator::simulation::ResolveLeafBackend;
    using forevervalidator::simulation::UsesOptimizedCpuFoundation;

    static_assert(static_cast<std::uint8_t>(SimulationBackend::Reference) ==
                  0u);
    static_assert(static_cast<std::uint8_t>(SimulationBackend::OptimizedCpu) ==
                  1u);
    static_assert(static_cast<std::uint8_t>(SimulationBackend::Batched) ==
                  2u);
    static_assert(static_cast<std::uint8_t>(
                          SimulationBackend::SpeculativeTicking) == 3u);
    static_assert(static_cast<std::uint8_t>(SimulationBackend::Cuda) == 4u);
    static_assert(!IsCudaComputeCapabilitySupported(6, 0));
    static_assert(IsCudaComputeCapabilitySupported(6, 1));
    static_assert(IsCudaComputeCapabilitySupported(12, 0));
    static_assert(!IsCudaSessionSpecializationSupported(6, 1));
    static_assert(!IsCudaSessionSpecializationSupported(7, 0));
    static_assert(!IsCudaSessionSpecializationSupported(7, 4));
    static_assert(IsCudaSessionSpecializationSupported(7, 5));
    static_assert(IsCudaSessionSpecializationSupported(12, 0));

    if (ResolveLeafBackend(SimulationBackend::Reference) !=
        SimulationBackend::Reference) {
        std::cerr << "Reference did not resolve to the reference backend\n";
        return 1;
    }
    if (ResolveLeafBackend(SimulationBackend::OptimizedCpu) !=
        SimulationBackend::OptimizedCpu) {
        std::cerr << "OptimizedCpu silently resolved to Reference\n";
        return 1;
    }
    if (ResolveLeafBackend(SimulationBackend::SpeculativeTicking) !=
        SimulationBackend::SpeculativeTicking) {
        std::cerr << "SpeculativeTicking silently resolved to another backend\n";
        return 1;
    }
    if (ResolveLeafBackend(SimulationBackend::Cuda) !=
        SimulationBackend::Cuda) {
        std::cerr << "Cuda silently resolved to a CPU backend\n";
        return 1;
    }
    if (!IsSimulationBackendSupported(SimulationBackend::Cuda)) {
        std::cerr << "Cuda was not registered as selectable\n";
        return 1;
    }
    const forevervalidator::CudaBackendDiagnostics cuda =
            forevervalidator::QueryCudaBackendDiagnostics();
#if FOREVERVALIDATOR_HAS_CUDA
    if (cuda.status == forevervalidator::CudaBackendStatus::NotCompiled) {
        std::cerr << "CUDA-enabled build reported NotCompiled\n";
        return 1;
    }
#else
    if (cuda.status != forevervalidator::CudaBackendStatus::NotCompiled ||
        cuda.diagnostic.empty()) {
        std::cerr << "CPU-only CUDA diagnostics are not explicit\n";
        return 1;
    }
#endif
    if (!IsSimulationBackendSupported(
                SimulationBackend::SpeculativeTicking)) {
        std::cerr << "SpeculativeTicking was not registered as supported\n";
        return 1;
    }
    if (!UsesOptimizedCpuFoundation(SimulationBackend::OptimizedCpu) ||
        !UsesOptimizedCpuFoundation(
                SimulationBackend::SpeculativeTicking) ||
        UsesOptimizedCpuFoundation(SimulationBackend::Reference) ||
        UsesOptimizedCpuFoundation(SimulationBackend::Batched)) {
        std::cerr << "optimized CPU foundation routing is incorrect\n";
        return 1;
    }
    return 0;
}
