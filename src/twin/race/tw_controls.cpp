#include "tw_controls.h"

#include <cstdlib>

namespace tw {
namespace race {

namespace {

struct ControlState {  // FV ReplayControlState; times are FV's absolute input times
    bool raceRunning = false, accelerate = false, brake = false, steerRight = false, steerLeft = false;
    std::int32_t steerLeftTime = 0, steerRightTime = 0, steerTime = 0, steerValue = 0;
    std::int32_t accelerateTime = 0, brakeTime = 0, gasTime = 0, gasValue = 0;
};

bool IsActive(const TwInputEvent &e) { return e.kind == TwValueKind::Switch && e.state != TwSwitch::Released; }
std::int32_t AnalogOf(const TwInputEvent &e) { return e.kind == TwValueKind::Analog ? e.analog : 0; }

void ApplyEvent(ControlState &s, const TwInputEvent &e, std::int32_t time) {
    const bool active = IsActive(e);
    switch (e.action) {
    case TwAction::Accelerate: s.accelerate = active; s.accelerateTime = time; break;
    case TwAction::Gas: s.gasValue = AnalogOf(e); s.gasTime = time; break;
    case TwAction::Brake: s.brake = active; s.brakeTime = time; break;
    case TwAction::Steer: s.steerValue = AnalogOf(e); s.steerTime = time; break;
    case TwAction::SteerLeft: s.steerLeft = active; s.steerLeftTime = time; break;
    case TwAction::SteerRight: s.steerRight = active; s.steerRightTime = time; break;
    case TwAction::RaceRunning: s.raceRunning = active; break;
    default: break;
    }
}

// ControlsFromState (replay_control_plan.cpp:145-181), steer kept as the raw
// 1/65536 integer the float is made from.
TwinControl ControlsFromState(const ControlState &s) {
    std::int32_t steer = 0;
    const std::int32_t digitalSteerTime = s.steerLeftTime > s.steerRightTime ? s.steerLeftTime : s.steerRightTime;
    const bool analogSteer = s.steerTime > digitalSteerTime ||
                             (s.steerTime == digitalSteerTime && !s.steerLeft && !s.steerRight &&
                              std::abs(s.steerValue) > 655);
    if (analogSteer)
        steer = s.steerValue;
    else if (s.steerLeft)
        steer = -kAnalogScale;
    else if (s.steerRight)
        steer = kAnalogScale;
    bool gas = false, brake = false;
    const std::int32_t digitalGateTime = s.accelerateTime > s.brakeTime ? s.accelerateTime : s.brakeTime;
    const bool analogGas = s.gasTime > digitalGateTime || (s.gasTime == digitalGateTime && !s.accelerate && !s.brake);
    if (analogGas) {
        if (s.gasValue <= -19661)
            gas = true;
        else if (s.gasValue >= 19661)
            brake = true;
    } else {
        gas = s.accelerate;
        brake = s.brake;
    }
    return TwCtlMake(steer, gas, brake);
}

bool Fail(std::string *error, const char *what) {
    if (error) *error = what;
    return false;
}

}  // namespace

bool TwBuildControls(const TwInputEvent *events, std::size_t count, const TwTimeline &tl,
                     std::uint32_t firstTickMs, std::uint32_t tickCount, TwControlStream *out,
                     std::string *error) {
    if (out == nullptr || tl.tickMs == 0u) return Fail(error, "bad arguments");
    if (firstTickMs == 0u || firstTickMs % tl.tickMs != 0u) return Fail(error, "first tick is not a tick time");
    // FV's sandbox conversion (validation_api.cpp:1549-1564) and timeline checks.
    std::vector<std::uint32_t> absolute(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::int64_t a = static_cast<std::int64_t>(tl.inputTimeBaseMs) + events[i].timeMs;
        if (a < 0 || a > 0x7FFFFFFFll) return Fail(error, "input time out of range");
        if (events[i].kind == TwValueKind::Analog && (events[i].analog < -kAnalogScale || events[i].analog > kAnalogScale))
            return Fail(error, "analog value out of range");
        if (i > 0 && events[i].timeMs < events[i - 1].timeMs) return Fail(error, "events out of order");
        absolute[i] = static_cast<std::uint32_t>(a);
    }

    out->firstTickMs = firstTickMs;
    out->ticks.clear();
    out->respawnOverflow.clear();
    out->ticks.reserve(tickCount);
    ControlState state;
    std::size_t cursor = 0u;
    bool previousRaceRunning = false, raceStartResetDone = false, spawnEstablished = false;
    const std::uint64_t lastTick = static_cast<std::uint64_t>(firstTickMs) +
                                   (tickCount == 0u ? 0u : static_cast<std::uint64_t>(tickCount - 1u) * tl.tickMs);
    for (std::uint64_t T = tl.tickMs; tickCount != 0u && T <= lastTick; T += tl.tickMs) {
        const std::int32_t timeMs = static_cast<std::int32_t>(T);
        const std::uint32_t sample = tl.inputTimeBaseMs - tl.prestartMs + static_cast<std::uint32_t>(T);
        std::uint32_t respawns = 0u;
        while (cursor < count && absolute[cursor] <= sample) {
            const TwInputEvent &e = events[cursor];
            if (e.action == TwAction::Respawn && state.raceRunning && IsActive(e)) ++respawns;  // R2.4
            ApplyEvent(state, e, static_cast<std::int32_t>(absolute[cursor]));
            ++cursor;
        }
        TwinControl c = ControlsFromState(state);
        if (!spawnEstablished && timeMs >= tl.establishSpawnAtMs) {
            c |= kCtlEstablishSpawn;
            spawnEstablished = true;
        }
        if (timeMs >= tl.enableRaceAfterMs) {
            c |= kCtlBeginRace;
            if (respawns >= 255u) {
                c |= (255u << kCtlRespawnShift) | kCtlRespawnEscape;
                if (T >= firstTickMs)
                    out->respawnOverflow.push_back({static_cast<std::uint32_t>((T - firstTickMs) / tl.tickMs), respawns});
            } else {
                c |= respawns << kCtlRespawnShift;
            }
            if (!raceStartResetDone && !previousRaceRunning && state.raceRunning) {
                c |= kCtlRaceStartReset;
                raceStartResetDone = true;
            }
        }
        previousRaceRunning = state.raceRunning;
        if (T >= firstTickMs) out->ticks.push_back(c);
    }
    return true;
}

}  // namespace race
}  // namespace tw
