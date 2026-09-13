#ifndef FOREVERVALIDATOR_CUDA_SEARCH_WINNER_SELECTION_CUH
#define FOREVERVALIDATOR_CUDA_SEARCH_WINNER_SELECTION_CUH

#include <cstdint>

namespace forevervalidator::simulation::cuda_search_detail {

constexpr std::uint32_t InvalidCandidateSlot = UINT32_MAX;

struct DeviceSample {
    double score = 0.0;
    double timeMs = 0.0;
    double detail0 = 0.0;
    double detail1 = 0.0;
    std::uint64_t candidateId = 0u;
    std::uint64_t logicalOrder = UINT64_MAX;
    std::uint32_t candidateSlot = InvalidCandidateSlot;
    std::uint32_t evaluationTick = 0u;
    std::uint32_t eventCount = UINT32_MAX;
    bool valid = false;
    bool mutation = false;
    bool preciseFinish = false;
    std::uint8_t reserved = 0u;
};

struct BetterSample {
    bool maximize = false;

    __host__ __device__ DeviceSample operator()(
            const DeviceSample &left,
            const DeviceSample &right) const {
        if (left.valid != right.valid) {
            return left.valid ? left : right;
        }
        if (!left.valid) {
            return left;
        }
        if (left.score != right.score) {
            if (maximize) {
                return left.score > right.score ? left : right;
            }
            return left.score < right.score ? left : right;
        }
        if (left.eventCount != right.eventCount) {
            return left.eventCount < right.eventCount ? left : right;
        }
        return left.logicalOrder <= right.logicalOrder ? left : right;
    }
};

__host__ __device__ inline bool StrictlyBetter(
        const DeviceSample &candidate,
        const DeviceSample &incumbent,
        bool maximize) {
    if (!candidate.valid) {
        return false;
    }
    if (!incumbent.valid) {
        return true;
    }
    if (candidate.score != incumbent.score) {
        return maximize ? candidate.score > incumbent.score
                        : candidate.score < incumbent.score;
    }
    return candidate.eventCount < incumbent.eventCount;
}

}  // namespace forevervalidator::simulation::cuda_search_detail

#endif
