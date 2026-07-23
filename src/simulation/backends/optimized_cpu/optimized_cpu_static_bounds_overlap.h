#ifndef FOREVERVALIDATOR_OPTIMIZED_CPU_STATIC_BOUNDS_OVERLAP_H
#define FOREVERVALIDATOR_OPTIMIZED_CPU_STATIC_BOUNDS_OVERLAP_H

#include <cmath>

#include "engine/core/gm_types.h"

#if defined(_MSC_VER)
#define FV_E026_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FV_E026_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define FV_E026_ALWAYS_INLINE inline
#endif

namespace forevervalidator::simulation {

namespace optimized_cpu_static_bounds_detail {

FV_E026_ALWAYS_INLINE bool AxisIntervalsOverlap(
        float centerA,
        float extentA,
        float centerB,
        float extentB) noexcept {
    const float distance = std::fabs(centerB - centerA);
    const float extentSum = extentB + extentA;
    return !(extentSum < distance);
}

}  // namespace optimized_cpu_static_bounds_detail

FV_E026_ALWAYS_INLINE int OptimizedCpuStaticBoundsOverlap(
        const GmBoxAligned &movingBox,
        const GmBoxAligned &staticBounds) noexcept {
    using optimized_cpu_static_bounds_detail::AxisIntervalsOverlap;
    if (!AxisIntervalsOverlap(
                movingBox.center.z,
                movingBox.halfExtents.z,
                staticBounds.center.z,
                staticBounds.halfExtents.z)) {
        return 0;
    }
    if (!AxisIntervalsOverlap(
                movingBox.center.y,
                movingBox.halfExtents.y,
                staticBounds.center.y,
                staticBounds.halfExtents.y)) {
        return 0;
    }
    return AxisIntervalsOverlap(
                   movingBox.center.x,
                   movingBox.halfExtents.x,
                   staticBounds.center.x,
                   staticBounds.halfExtents.x)
            ? 1
            : 0;
}

}  // namespace forevervalidator::simulation

#undef FV_E026_ALWAYS_INLINE

#endif
