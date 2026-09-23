#ifndef FOREVERVALIDATOR_HIP_STATE_BRIDGE_H
#define FOREVERVALIDATOR_HIP_STATE_BRIDGE_H

#include <cstring>
#include <type_traits>

#include "simulation/backends/cuda/cuda_timeline_executor.h"
#include "simulation/backends/cuda/cuda_search_executor.h"
#include "simulation/backends/hip/generated/hip_timeline_executor.h"
#include "simulation/backends/hip/generated/hip_search_executor.h"

namespace forevervalidator::simulation::hip_bridge {

template<typename Destination, typename Source>
Destination BitCopy(const Source &source) noexcept {
    static_assert(sizeof(Destination) == sizeof(Source));
    static_assert(alignof(Destination) == alignof(Source));
    static_assert(std::is_trivially_copyable_v<Destination>);
    static_assert(std::is_trivially_copyable_v<Source>);
    Destination result;
    std::memcpy(&result, &source, sizeof(result));
    return result;
}

CudaTimelineBatchResult ToCudaResult(
        const HipTimelineBatchResult &source);
HipSearchExecutorConfiguration ToHipConfiguration(
        const CudaSearchExecutorConfiguration &source);
CudaSearchBatchExecution ToCudaResult(
        const HipSearchBatchExecution &source);

}  // namespace forevervalidator::simulation::hip_bridge

#endif
