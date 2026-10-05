#ifndef FOREVERVALIDATOR_VALIDATION_PLANNING_SEARCH_INPUT_BOUNDARY_H
#define FOREVERVALIDATOR_VALIDATION_PLANNING_SEARCH_INPUT_BOUNDARY_H

#include <forevervalidator/experimental/physics_sandbox.h>

#include <algorithm>
#include <limits>

namespace forevervalidator::experimental::search_limits {

inline std::vector<PhysicsSandboxInputEvent>::const_iterator SearchInputBaselineEnd(
        const std::vector<PhysicsSandboxInputEvent> &inputs,
        std::int64_t simulationHorizonMs,
        const std::vector<PhysicsSandboxCudaMutationSegment> &segments) {
    std::int64_t endTimeMs = simulationHorizonMs;
    for (const auto &segment : segments) {
        // RestoreSegmentControls uses the next two milliseconds to restore
        // analog/digital priority. Keep existing resets visible on promotion.
        const auto restorationEnd = segment.maximumTimeMs >
                        std::numeric_limits<std::int64_t>::max() - 2
                ? std::numeric_limits<std::int64_t>::max()
                : segment.maximumTimeMs + 2;
        endTimeMs = std::max(endTimeMs, restorationEnd);
    }
    return std::upper_bound(
            inputs.begin(), inputs.end(), endTimeMs,
            [](std::int64_t timeMs, const PhysicsSandboxInputEvent &event) {
                return timeMs < event.timeMs;
            });
}

}  // namespace forevervalidator::experimental::search_limits

#endif
