// Twin backend routing (docs/twin/integration.md): the enum value, routing
// helpers, error names and availability diagnostics. No game assets needed.
#include "simulation/backends/simulation_backend.h"

#include <forevervalidator/validation.h>

#include <cstring>
#include <iostream>

namespace {

int g_fail = 0;

void Check(bool ok, const char *what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_fail;
    }
}

}  // namespace

int main() {
    using forevervalidator::SimulationBackend;
    using forevervalidator::ValidationErrorCode;
    using forevervalidator::ValidationFailureReason;
    namespace sim = forevervalidator::simulation;

    static_assert(static_cast<std::uint8_t>(SimulationBackend::Twin) == 7u,
                  "Twin is appended after Hip");
    static_assert(static_cast<std::uint16_t>(ValidationFailureReason::TwinNotCompiled) == 730u, "reason 730");
    static_assert(static_cast<std::uint16_t>(ValidationFailureReason::TwinExecutionFailed) == 733u, "reason 733");

    Check(sim::IsSimulationBackendSupported(SimulationBackend::Twin), "Twin is selectable");
    Check(sim::ResolveLeafBackend(SimulationBackend::Twin) == SimulationBackend::Twin,
          "Twin resolves to itself (not silently to a CPU backend)");
    Check(!sim::UsesOptimizedCpuFoundation(SimulationBackend::Twin),
          "the runtime never sees Twin: the session runs as optimized_cpu");

    Check(std::strcmp(forevervalidator::ValidationFailureReasonName(
                              ValidationFailureReason::TwinUnsupportedSimulationScope),
                      "twin_unsupported_simulation_scope") == 0,
          "scope reason name");
    Check(std::strcmp(forevervalidator::ValidationFailureReasonName(ValidationFailureReason::TwinNotImplemented),
                      "twin_not_implemented") == 0,
          "not-implemented reason name");
    Check(std::strcmp(forevervalidator::ValidationErrorCodeName(ValidationErrorCode::TwinUnavailable),
                      "twin_unavailable") == 0,
          "error code name");
    forevervalidator::ValidationError error;
    error.code = ValidationErrorCode::TwinUnavailable;
    Check(forevervalidator::ValidationErrorExitCode(error) == 69, "twin unavailable exits 69 like CUDA");
    error.code = ValidationErrorCode::TwinExecutionFailed;
    Check(forevervalidator::ValidationErrorExitCode(error) == 71, "twin execution failure exits 71 like CUDA");

    const forevervalidator::ValidationOptions defaults;
    Check(defaults.backend == SimulationBackend::Reference && !defaults.twin.debugDelegateToOptimizedCpu &&
                  defaults.twin.policy == forevervalidator::TwinPolicy::FvSubsteps,
          "defaults: reference backend, no delegate, FV substeps");

    const forevervalidator::TwinBackendDiagnostics twin = forevervalidator::QueryTwinBackendDiagnostics();
    Check(!twin.diagnostic.empty(), "twin diagnostics are explicit");
#if FOREVERVALIDATOR_HAS_TWIN
    Check(twin.status != forevervalidator::TwinBackendStatus::NotCompiled, "twin build reports compiled");
    Check(!twin.modelId.empty(), "twin build reports a model id");
#else
    Check(twin.status == forevervalidator::TwinBackendStatus::NotCompiled, "build without twin reports NotCompiled");
#endif
    std::cout << "twin backend: status " << static_cast<int>(twin.status) << ", " << twin.diagnostic << '\n';
    return g_fail == 0 ? 0 : 1;
}
