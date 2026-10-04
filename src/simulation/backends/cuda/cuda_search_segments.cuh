#ifndef FOREVERVALIDATOR_CUDA_SEARCH_SEGMENTS_CUH
#define FOREVERVALIDATOR_CUDA_SEARCH_SEGMENTS_CUH

#include "simulation/backends/cuda/cuda_search_branch_state.cuh"

#if defined(__CUDACC__)
#define FOREVERVALIDATOR_SEGMENT_HD __host__ __device__
#else
#define FOREVERVALIDATOR_SEGMENT_HD
#endif

namespace forevervalidator::simulation::cuda_search_detail {

// Stateless replay keeps segment selection independent of modifier RNG usage.
struct SegmentSelection {
    std::uint64_t seed;
    std::uint32_t count;
    std::uint32_t changed;

    FOREVERVALIDATOR_SEGMENT_HD bool Selected(std::uint32_t index) const {
        std::uint64_t state = seed;
        std::uint32_t needed = changed;
        for (std::uint32_t current = 0u; current <= index; ++current) {
            state += UINT64_C(0x9e3779b97f4a7c15);
            std::uint64_t value = state;
            value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
            value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
            value ^= value >> 31u;
            const bool selected = value % (count - current) < needed;
            if (current == index) return selected;
            if (selected) --needed;
        }
        return false;
    }

    FOREVERVALIDATOR_SEGMENT_HD bool Contains(
            const CudaSearchMutationSegment* segments,
            std::int64_t timeMs) const {
        for (std::uint32_t index = 0u; index < count; ++index) {
            if (timeMs < segments[index].minimumTimeMs) return false;
            if (timeMs <= segments[index].maximumTimeMs) {
                return Selected(index);
            }
        }
        return false;
    }
};

FOREVERVALIDATOR_SEGMENT_HD inline bool AnalogSteeringWins(
        const DeviceControlState& state) {
    const auto digitalTime = state.steerLeftTime > state.steerRightTime
                                     ? state.steerLeftTime
                                     : state.steerRightTime;
    return state.steerTime > digitalTime ||
           (state.steerTime == digitalTime && !state.steerLeft &&
            !state.steerRight &&
            (state.steerValue > 655 || state.steerValue < -655));
}

FOREVERVALIDATOR_SEGMENT_HD inline bool AnalogGasWins(
        const DeviceControlState& state) {
    const auto digitalTime = state.accelerateTime > state.brakeTime
                                     ? state.accelerateTime
                                     : state.brakeTime;
    return state.gasTime > digitalTime ||
           (state.gasTime == digitalTime && !state.accelerate && !state.brake);
}

// Restore only changed channel groups, before the following simulation tick.
// Distinct timestamps preserve analog/digital priority without replacing any
// baseline input event at the next tick.
FOREVERVALIDATOR_SEGMENT_HD inline bool RestoreSegmentControls(
        const CudaSearchInputEvent* baseline, std::uint32_t baselineCount,
        CudaSearchInputEvent* events, std::uint32_t* eventCount,
        std::uint32_t capacity, DeviceControlState initial,
        std::int64_t endTimeMs) {
    DeviceControlState expected = initial;
    DeviceControlState actual = initial;
    for (std::uint32_t index = 0u; index < baselineCount; ++index) {
        if (baseline[index].timeMs <= endTimeMs + 2)
            ApplyControlEvent(expected, baseline[index]);
    }
    for (std::uint32_t index = 0u; index < *eventCount; ++index) {
        if (events[index].timeMs <= endTimeMs + 2)
            ApplyControlEvent(actual, events[index]);
    }
    const bool steerChanged =
            expected.steerValue != actual.steerValue ||
            expected.steerLeft != actual.steerLeft ||
            expected.steerRight != actual.steerRight ||
            AnalogSteeringWins(expected) != AnalogSteeringWins(actual);
    const bool gasChanged = expected.gasValue != actual.gasValue ||
                            expected.accelerate != actual.accelerate ||
                            expected.brake != actual.brake ||
                            AnalogGasWins(expected) != AnalogGasWins(actual);
    const std::uint32_t growth =
            3u * (static_cast<std::uint32_t>(steerChanged) +
                  static_cast<std::uint32_t>(gasChanged));
    if (growth > capacity - *eventCount) return false;
    if (steerChanged) {
        const auto analogTime = static_cast<std::int32_t>(
                endTimeMs + (AnalogSteeringWins(expected) ? 2 : 1));
        const auto digitalTime = static_cast<std::int32_t>(
                endTimeMs + (AnalogSteeringWins(expected) ? 1 : 2));
        events[(*eventCount)++] = {analogTime, 4u, 2u, expected.steerValue};
        events[(*eventCount)++] = {digitalTime, 5u, 1u,
                                   expected.steerLeft ? 1 : 0};
        events[(*eventCount)++] = {digitalTime, 6u, 1u,
                                   expected.steerRight ? 1 : 0};
    }
    if (gasChanged) {
        const auto analogTime = static_cast<std::int32_t>(
                endTimeMs + (AnalogGasWins(expected) ? 2 : 1));
        const auto digitalTime = static_cast<std::int32_t>(
                endTimeMs + (AnalogGasWins(expected) ? 1 : 2));
        events[(*eventCount)++] = {analogTime, 2u, 2u, expected.gasValue};
        events[(*eventCount)++] = {digitalTime, 1u, 1u,
                                   expected.accelerate ? 1 : 0};
        events[(*eventCount)++] = {digitalTime, 3u, 1u, expected.brake ? 1 : 0};
    }
    return true;
}

}  // namespace forevervalidator::simulation::cuda_search_detail

#undef FOREVERVALIDATOR_SEGMENT_HD
#endif
