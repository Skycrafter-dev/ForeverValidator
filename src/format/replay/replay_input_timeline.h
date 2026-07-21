#ifndef REPLAY_INPUT_TIMELINE_H
#define REPLAY_INPUT_TIMELINE_H

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

enum class ReplayInputActionKind {
    Unmapped,
    Accelerate,
    Gas,
    Brake,
    Steer,
    SteerLeft,
    SteerRight,
    RaceRunning,
    FinishLine,
    Respawn,
};

enum class ReplayInputSwitchState {
    Released,
    Pressed,
    NonCanonicalActive,
};

enum class ReplayInputActionValueKind {
    None,
    Switch,
    Analog,
};

class ReplayInputActionValue {
public:
    static ReplayInputActionValue None() {
        return ReplayInputActionValue{};
    }

    static ReplayInputActionValue Switch(ReplayInputSwitchState state) {
        ReplayInputActionValue value;
        value.kind_ = ReplayInputActionValueKind::Switch;
        value.switchState_ = state;
        return value;
    }

    static ReplayInputActionValue Analog(float analog) {
        ReplayInputActionValue value;
        value.kind_ = ReplayInputActionValueKind::Analog;
        value.analog_ = analog;
        return value;
    }

    ReplayInputActionValueKind Kind() const {
        return kind_;
    }

    bool IsActive() const {
        return kind_ == ReplayInputActionValueKind::Switch &&
               switchState_ != ReplayInputSwitchState::Released;
    }

    bool IsCanonicalPress() const {
        return kind_ == ReplayInputActionValueKind::Switch &&
               switchState_ == ReplayInputSwitchState::Pressed;
    }

    float AnalogValue() const {
        return analog_;
    }

private:
    ReplayInputActionValueKind kind_ = ReplayInputActionValueKind::None;
    ReplayInputSwitchState switchState_ = ReplayInputSwitchState::Released;
    float analog_ = 0.0f;
};

struct ReplayInputEvent {
    uint32_t timeMs = 0u;
    ReplayInputActionKind action = ReplayInputActionKind::Unmapped;
    ReplayInputActionValue value = ReplayInputActionValue::None();
};

struct ReplayInputMetadata {
    uint32_t durationMs = 0u;
    uint32_t validationSeed = 0u;
    std::optional<int32_t> raceTimeMs;
    std::optional<uint32_t> respawnCount;
    std::optional<uint32_t> stuntScore;
};

enum class ReplayInputProvenance {
    Unmarked,
    TMInterface,
};

enum class ReplayInputTimelineCreateResult {
    Success,
    MissingOutput,
    TimeOutOfRange,
    EventsOutOfOrder,
    InvalidValue,
};

class ReplayInputTimeline {
public:
    ReplayInputTimeline() = default;

    static ReplayInputTimelineCreateResult Create(
            ReplayInputMetadata metadata,
            std::vector<ReplayInputActionKind> definedActions,
            std::vector<ReplayInputEvent> events,
            ReplayInputTimeline *out,
            ReplayInputProvenance provenance =
                    ReplayInputProvenance::Unmarked) {
        if (out == nullptr) {
            return ReplayInputTimelineCreateResult::MissingOutput;
        }
        const uint32_t signedTimeLimit = static_cast<uint32_t>(
                std::numeric_limits<int32_t>::max());
        if (metadata.durationMs > signedTimeLimit ||
            (metadata.respawnCount.has_value() &&
             *metadata.respawnCount > signedTimeLimit) ||
            (metadata.stuntScore.has_value() &&
             *metadata.stuntScore > signedTimeLimit)) {
            return ReplayInputTimelineCreateResult::TimeOutOfRange;
        }

        uint32_t previousTimeMs = 0u;
        bool hasPreviousEvent = false;
        for (const ReplayInputEvent &event : events) {
            if (event.timeMs > signedTimeLimit) {
                return ReplayInputTimelineCreateResult::TimeOutOfRange;
            }
            if (hasPreviousEvent && event.timeMs < previousTimeMs) {
                return ReplayInputTimelineCreateResult::EventsOutOfOrder;
            }
            previousTimeMs = event.timeMs;
            hasPreviousEvent = true;

            const ReplayInputActionValueKind expectedValueKind =
                    ValueKindForAction(event.action);
            if (event.value.Kind() != expectedValueKind ||
                (expectedValueKind == ReplayInputActionValueKind::Analog &&
                 !std::isfinite(event.value.AnalogValue()))) {
                return ReplayInputTimelineCreateResult::InvalidValue;
            }
        }

        ReplayInputTimeline timeline;
        timeline.metadata_ = std::move(metadata);
        timeline.definedActions_ = std::move(definedActions);
        timeline.events_ = std::move(events);
        timeline.provenance_ = provenance;
        *out = std::move(timeline);
        return ReplayInputTimelineCreateResult::Success;
    }

    const ReplayInputMetadata &Metadata() const {
        return metadata_;
    }

    const std::vector<ReplayInputEvent> &Events() const {
        return events_;
    }

    const std::vector<ReplayInputActionKind> &DefinedActions() const {
        return definedActions_;
    }

    size_t DefinedActionCount() const {
        return definedActions_.size();
    }

    size_t EventCount() const {
        return events_.size();
    }

    ReplayInputProvenance Provenance() const {
        return provenance_;
    }

private:
    static ReplayInputActionValueKind ValueKindForAction(
            ReplayInputActionKind action) {
        switch (action) {
        case ReplayInputActionKind::Gas:
        case ReplayInputActionKind::Steer:
            return ReplayInputActionValueKind::Analog;
        case ReplayInputActionKind::Unmapped:
            return ReplayInputActionValueKind::None;
        default:
            return ReplayInputActionValueKind::Switch;
        }
    }

    ReplayInputMetadata metadata_;
    std::vector<ReplayInputActionKind> definedActions_;
    std::vector<ReplayInputEvent> events_;
    ReplayInputProvenance provenance_ = ReplayInputProvenance::Unmarked;
};

#endif
