#include "simulation/backends/cuda/cuda_search_executor.h"

namespace forevervalidator::simulation {

const char *CudaSearchStatusName(CudaSearchStatus status) noexcept {
    switch (status) {
    case CudaSearchStatus::Success: return "success";
    case CudaSearchStatus::InvalidArgument: return "invalid_argument";
    case CudaSearchStatus::UnsupportedConfiguration:
        return "unsupported_configuration";
    case CudaSearchStatus::CapacityExceeded: return "capacity_exceeded";
    case CudaSearchStatus::Cancelled: return "cancelled";
    case CudaSearchStatus::DeviceFailure: return "device_failure";
    case CudaSearchStatus::UnsupportedPhysicsTransition:
        return "unsupported_physics_transition";
    }
    return "unknown";
}

#if !FOREVERVALIDATOR_HAS_CUDA

struct CudaSearchExecutor::Impl {};

CudaSearchExecutor::CudaSearchExecutor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CudaSearchExecutor::~CudaSearchExecutor() = default;
CudaSearchExecutor::CudaSearchExecutor(CudaSearchExecutor &&) noexcept =
        default;
CudaSearchExecutor &CudaSearchExecutor::operator=(
        CudaSearchExecutor &&) noexcept = default;

std::unique_ptr<CudaSearchExecutor> CudaSearchExecutor::Create(
        const CudaSearchExecutorConfiguration &,
        std::string *diagnostic) noexcept {
    if (diagnostic != nullptr) {
        *diagnostic =
                "CUDA search is unavailable in a CPU-only build";
    }
    return {};
}

CudaSearchBatchExecution CudaSearchExecutor::EvaluateBaseline() noexcept {
    CudaSearchBatchExecution result;
    result.status = CudaSearchStatus::DeviceFailure;
    result.diagnostic =
            "CUDA search is unavailable in a CPU-only build";
    return result;
}

CudaSearchBatchExecution CudaSearchExecutor::EvaluateBaseline(
        const std::function<bool()> &) noexcept {
    return EvaluateBaseline();
}

CudaSearchBatchExecution CudaSearchExecutor::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        bool) noexcept {
    CudaSearchBatchExecution result;
    result.status = CudaSearchStatus::DeviceFailure;
    result.firstCandidateId = firstCandidateId;
    result.candidateCount = candidateCount;
    result.diagnostic =
            "CUDA search is unavailable in a CPU-only build";
    return result;
}

CudaSearchBatchExecution CudaSearchExecutor::RunBatch(
        std::uint64_t firstCandidateId,
        std::uint32_t candidateCount,
        const std::function<bool()> &) noexcept {
    return RunBatch(firstCandidateId, candidateCount, false);
}

bool CudaSearchExecutor::ReserveBatchCapacity(
        std::uint32_t,
        std::string *diagnostic) noexcept {
    if (diagnostic != nullptr) {
        *diagnostic =
                "CUDA search is unavailable in a CPU-only build";
    }
    return false;
}

bool CudaSearchExecutor::UpdateChangedSegmentCount(std::uint32_t) noexcept {
    return false;
}

bool CudaSearchExecutor::UpdateConditionTimes(double, double) noexcept {
    return false;
}

std::uint32_t CudaSearchExecutor::BatchCapacity() const noexcept {
    return 0u;
}

#endif

}  // namespace forevervalidator::simulation
