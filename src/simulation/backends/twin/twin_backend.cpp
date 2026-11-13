#include "simulation/backends/twin/twin_backend.h"

#if FOREVERVALIDATOR_HAS_TWIN
#include <forevervalidator/experimental/twin.h>
#endif

namespace forevervalidator {

TwinBackendDiagnostics QueryTwinBackendDiagnostics() noexcept {
    TwinBackendDiagnostics diagnostics;
    try {
#if FOREVERVALIDATOR_HAS_TWIN
        const experimental::twin::TwinEngineInfo engine =
                experimental::twin::QueryTwinEngine();
        diagnostics.modelId = engine.modelId;
        if (engine.stepImplemented) {
            diagnostics.status = TwinBackendStatus::Ready;
            diagnostics.diagnostic = "twin engine ready (model " +
                                     engine.modelId + ")";
        } else {
            diagnostics.status = TwinBackendStatus::StepNotImplemented;
            diagnostics.diagnostic =
                    "twin engine is not implemented yet (model " +
                    engine.modelId + "; missing: " + engine.missing +
                    "); for routing tests only, "
                    "--twin-debug-delegate-optimized-cpu runs the ticks "
                    "after the prestart on optimized_cpu";
        }
#else
        diagnostics.status = TwinBackendStatus::NotCompiled;
        diagnostics.diagnostic =
                "ForeverValidator was built without the twin engine "
                "(configure with -DFOREVERVALIDATOR_BUILD_TWIN=ON)";
#endif
    } catch (...) {
    }
    return diagnostics;
}

}  // namespace forevervalidator

#if !FOREVERVALIDATOR_HAS_TWIN
namespace forevervalidator::simulation {

TwinTimelineSegmentResult ExecuteTwinTimelineSegment(
        const TwinTimelineSegmentRequest &) {
    TwinTimelineSegmentResult result;
    result.result = ReplaySimulationRunResult::TwinUnavailable;
    result.diagnostic = QueryTwinBackendDiagnostics().diagnostic;
    return result;
}

std::unique_ptr<TwinSearchExecutor> CreateTwinSearchExecutor(
        CudaSearchExecutorConfiguration, TwinSearchExecutorSource,
        std::string *diagnostic) {
    if (diagnostic != nullptr) {
        *diagnostic = QueryTwinBackendDiagnostics().diagnostic;
    }
    return nullptr;
}

}  // namespace forevervalidator::simulation
#endif
