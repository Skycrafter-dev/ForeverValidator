#include "simulation/backends/hip/generated/hip_session_specialization.h"

namespace forevervalidator::simulation::hip::specialization {

std::uint64_t SessionModuleBuildCountForTesting() noexcept { return 0u; }

SessionModule::~SessionModule() = default;

bool SessionModule::Build(
        const HipPackedStaticConfigurationHeader &, std::uint64_t,
        const HipPackedSceneHeader &, std::uint64_t,
        std::string *diagnostic) {
    if (diagnostic != nullptr) {
        *diagnostic = "HIP session specialization is unavailable";
    }
    return false;
}

bool SessionModule::Ready() const noexcept { return false; }

hipFunction_t SessionModule::Kernel(std::uint32_t) const noexcept {
    return nullptr;
}

const KernelMetrics &SessionModule::Metrics(std::uint32_t) const noexcept {
    return throughput_.metrics;
}

void SessionModule::Reset() noexcept {}

}  // namespace forevervalidator::simulation::hip::specialization
