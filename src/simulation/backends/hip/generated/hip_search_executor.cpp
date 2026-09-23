// Generated from src/simulation/backends/cuda/cuda_search_executor.cpp by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_search_executor.h"

namespace forevervalidator::simulation {

const char *HipSearchStatusName(HipSearchStatus status) noexcept {
    switch (status) {
    case HipSearchStatus::Success: return "success";
    case HipSearchStatus::InvalidArgument: return "invalid_argument";
    case HipSearchStatus::UnsupportedConfiguration:
        return "unsupported_configuration";
    case HipSearchStatus::CapacityExceeded: return "capacity_exceeded";
    case HipSearchStatus::Cancelled: return "cancelled";
    case HipSearchStatus::DeviceFailure: return "device_failure";
    case HipSearchStatus::UnsupportedPhysicsTransition:
        return "unsupported_physics_transition";
    }
    return "unknown";
}

#if !FOREVERVALIDATOR_HAS_HIP

struct HipSearchExecutor::Impl {};

HipSearchExecutor::HipSearchExecutor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
HipSearchExecutor::~HipSearchExecutor() = default;
HipSearchExecutor::HipSearchExecutor(HipSearchExecutor &&) noexcept =
        default;
HipSearchExecutor &HipSearchExecutor::operator=(
        HipSearchExecutor &&) noexcept = default;

std::unique_ptr<HipSearchExecutor> HipSearchExecutor::Create(
        const HipSearchExecutorConfiguration &,
        std::string *diagnostic) noexcept {
    if (diagnostic != nullptr) {
        *diagnostic =
                "HIP search is unavailable in a CPU-only build";
    }
    return {};
}

HipSearchBatchExecution HipSearchExecutor::EvaluateBaseline() noexcept {
    HipSearchBatchExecution result;
    result.status = HipSearchStatus::DeviceFailure;
    result.diagnostic =
            "HIP search is unavailable in a CPU-only build";
    return result;
}

HipSearchBatchExecution HipSearchExecutor::EvaluateBaseline(
        const std::function<bool()> &) noexcept {
    return EvaluateBaseline();
}

HipSearchBatchExecution HipSearchExecutor::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        bool) noexcept {
    HipSearchBatchExecution result;
    result.status = HipSearchStatus::DeviceFailure;
    result.firstCandidateId = firstCandidateId;
    result.candidateCount = candidateCount;
    result.diagnostic =
            "HIP search is unavailable in a CPU-only build";
    return result;
}

HipSearchBatchExecution HipSearchExecutor::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        const std::function<bool()> &) noexcept {
    return RunBatch(firstCandidateId, candidateCount, false);
}

bool HipSearchExecutor::ReserveBatchCapacity(
        std::uint32_t,
        std::string *diagnostic) noexcept {
    if (diagnostic != nullptr) {
        *diagnostic =
                "HIP search is unavailable in a CPU-only build";
    }
    return false;
}

bool HipSearchExecutor::UpdateConditionTimes(double, double) noexcept {
    return false;
}

std::uint32_t HipSearchExecutor::BatchCapacity() const noexcept {
    return 0u;
}

#endif

}  // namespace forevervalidator::simulation
