#ifndef FOREVERVALIDATOR_CUDA_SEARCH_PROGRESS_CUH
#define FOREVERVALIDATOR_CUDA_SEARCH_PROGRESS_CUH

#include <cstdint>

namespace forevervalidator::simulation::cuda_search_progress_detail {

constexpr double InvalidClosestTargetDistanceSquared =
        1.7976931348623157e+308;

__host__ __device__ inline double SquaredDistanceToTargetVolume(
        const double *bounds,
        double x,
        double y,
        double z) {
    const double values[3]{x, y, z};
    double result = 0.0;
#pragma unroll
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const double distance =
                values[axis] < bounds[axis]
                ? bounds[axis] - values[axis]
                : values[axis] > bounds[axis + 3u]
                ? values[axis] - bounds[axis + 3u]
                : 0.0;
        result += distance * distance;
    }
    return result;
}

__host__ __device__ inline double UpdateClosestTargetDistanceSquared(
        double currentMinimum,
        double sampledDistanceSquared,
        bool enteredTarget) {
    if (enteredTarget) {
        return 0.0;
    }
    return currentMinimum < sampledDistanceSquared
            ? currentMinimum : sampledDistanceSquared;
}

__host__ __device__ inline bool IsQualifyingSearchCandidate(
        bool active,
        bool sampleValid) {
    return active && sampleValid;
}

__device__ inline void AtomicMinNonnegativeDouble(
        double value,
        double *minimum) {
    atomicMin(
            reinterpret_cast<unsigned long long *>(minimum),
            static_cast<unsigned long long>(
                    __double_as_longlong(value)));
}

}  // namespace forevervalidator::simulation::cuda_search_progress_detail

#endif
