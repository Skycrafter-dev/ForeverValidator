// Twin control stream: replay input events -> one TwinControl per tick.
//
// Port of FV's BuildTimelinePlan / ApplyEvent / ControlsFromState
// (src/simulation/control/replay_control_plan.cpp:76-181, 200-345) for the
// race path (board 30 R2). The state machine runs once when the stream is
// built, never in the tick loop. Stunts-only outputs (input change history,
// recorded FinishLine flag) and the validation tail tick (R1.10) are not
// produced: Stunts is M14 and the tail tick is a validation-mode feature.
#ifndef TW_CONTROLS_H
#define TW_CONTROLS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tw_race.h"

namespace tw {
namespace race {

// Same numbering as forevervalidator::experimental::PhysicsSandboxInputAction.
enum class TwAction : std::uint8_t {
    Unmapped, Accelerate, Gas, Brake, Steer, SteerLeft, SteerRight, RaceRunning, FinishLine, Respawn,
};
enum class TwValueKind : std::uint8_t { None, Switch, Analog };
enum class TwSwitch : std::uint8_t { Released, Pressed, NonCanonicalActive };

struct TwInputEvent {         // = PhysicsSandboxInputEvent
    std::int32_t timeMs;      // race time (0 = race start), ordered
    TwAction action;
    TwValueKind kind;
    TwSwitch state;
    std::int32_t analog;      // [-65536, 65536]
};

struct TwTimeline {           // FV sandbox defaults (validation_api.cpp:1582-1596)
    std::uint32_t tickMs = 10u;
    std::uint32_t prestartMs = 2600u;
    std::uint32_t inputTimeBaseMs = 100000u;
    std::int32_t enableRaceAfterMs = 2600;  // begin race / respawns / race-start reset from here
    std::int32_t establishSpawnAtMs = 0;    // race spawn on the first tick at or after
};

struct TwControlStream {
    std::uint32_t firstTickMs = 0u;
    std::vector<TwinControl> ticks;  // ticks[i] is tick firstTickMs + i * tickMs
    std::vector<std::pair<std::uint32_t, std::uint32_t>> respawnOverflow;  // (i, count) for escape ticks

    std::uint32_t Respawns(std::size_t i) const {
        const TwinControl c = ticks[i];
        if (!(c & kCtlRespawnEscape)) return TwCtlRespawnField(c);
        for (const auto &o : respawnOverflow)
            if (o.first == i) return o.second;
        return TwCtlRespawnField(c);
    }
};

// Builds ticks firstTickMs .. firstTickMs + (tickCount-1)*tickMs. The input
// state machine always starts at the first tick (tickMs), so a window that
// starts at the race start sees the held state of the prestart.
// Returns false (and *error) on unordered events, out-of-range analog values
// or a firstTickMs that is not a positive multiple of tickMs.
bool TwBuildControls(const TwInputEvent *events, std::size_t count, const TwTimeline &timeline,
                     std::uint32_t firstTickMs, std::uint32_t tickCount, TwControlStream *out,
                     std::string *error = nullptr);

}  // namespace race
}  // namespace tw

#endif
