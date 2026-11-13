// Twin FV adapter: the validation timeline after FV's prestart
// (design.md §8.2 "Validate", milestone M7). Compiled into
// forevervalidator_core when -DFOREVERVALIDATOR_BUILD_TWIN=ON.
//
//   1. TwinWorld for the session: world package, car pack, water, race map
//      (fv/twin_world_build.cpp) from the request's static models,
//      definition and start spawn;                                     done
//   2. request.start (FV runtime clone after T = 2600) -> TwinState
//      (fv/import.cpp: ExtractState -> ImportGolden);                 done
//   3. request.ticks -> TwinControl (FlattenTwinControlTick);         done
//   4. TwinStep per tick from T = 2610. Today the first tick stops at the
//      missing collision stage (M3) and the run reports TwinNotImplemented
//      with what was built. Once collision exists: one
//      ReplayTrajectoryObservation per observed tick (simulated and write
//      positions, comparison against tick.comparisonTarget exactly as
//      ObserveReplayTrajectory computes it), finish and respawns (M7).
#include "simulation/backends/twin/twin_backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "import.h"
#include "simulation/runtime/replay_simulation_session.h"
#include "tw_race.h"
#include "tw_step.h"
#include "twin_api_internal.h"
#include "twin_world_build.h"

namespace forevervalidator::simulation {

namespace {

std::uint32_t FloatBits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// One FV control-plan tick as a TwinControl (race/tw_race.h bit layout).
// Refuses values the control word cannot carry exactly.
bool FlattenTwinControlTick(const ReplayControlTick &t, tw::race::TwinControl *out, std::uint32_t *respawns,
                            std::string *why) {
    const ReplayVehicleControlState &c = t.controls;
    const auto digital = [](float v) { return FloatBits(v) == FloatBits(0.0f) || FloatBits(v) == FloatBits(1.0f); };
    if (!digital(c.lowSpeedGateA) || !digital(c.lowSpeedGateB)) {
        *why = "analog gas/brake at T=" + std::to_string(t.timeMs);
        return false;
    }
    const float scaled = c.steering * static_cast<float>(tw::race::kAnalogScale);
    const std::int32_t raw = static_cast<std::int32_t>(scaled);
    if (static_cast<float>(raw) != scaled || raw < -131072 || raw > 131071) {
        *why = "steering not a multiple of 1/65536 at T=" + std::to_string(t.timeMs);
        return false;
    }
    tw::race::TwinControl w = tw::race::TwCtlMake(raw, c.lowSpeedGateA != 0.0f, c.lowSpeedGateB != 0.0f);
    if (FloatBits(tw::race::TwCtlSteer(w)) != FloatBits(c.steering)) {
        *why = "steering does not round-trip at T=" + std::to_string(t.timeMs);
        return false;
    }
    const ReplayRaceTransitionActions &a = t.actions;
    if (a.enableStuntsSimulation || a.finishRace || a.suppressVehicleForceCallbacks) {
        *why = "unsupported race transition (stunts, finish or callback suppression) at T=" + std::to_string(t.timeMs);
        return false;
    }
    const std::uint32_t n = a.respawnAtCheckpointCount;
    w |= (n >= 255u ? 255u : n) << tw::race::kCtlRespawnShift;
    if (n >= 255u) w |= tw::race::kCtlRespawnEscape;
    if (a.resetAtRaceStart) w |= tw::race::kCtlRaceStartReset;
    if (a.enableRaceSimulation) w |= tw::race::kCtlBeginRace;
    if (a.establishRaceSpawn) w |= tw::race::kCtlEstablishSpawn;
    *out = w;
    *respawns = n;
    return true;
}

TwinTimelineSegmentResult Fail(ReplaySimulationRunResult code, std::string diagnostic) {
    TwinTimelineSegmentResult r;
    r.result = code;
    r.diagnostic = std::move(diagnostic);
    return r;
}

}  // namespace

TwinTimelineSegmentResult ExecuteTwinTimelineSegment(const TwinTimelineSegmentRequest &request) {
    if (request.start == nullptr || (request.ticks == nullptr && request.tickCount != 0u))
        return Fail(ReplaySimulationRunResult::TwinExecutionFailed, "twin timeline: invalid hand-off request");
    if (request.models == nullptr || request.definition == nullptr || !request.haveStartSpawn)
        return Fail(ReplaySimulationRunResult::TwinExecutionFailed,
                    "twin timeline: the hand-off request carries no static models, definition or start spawn");
    try {
        // 1. world
        tw::fv::TwinWorldSource src;
        src.models = request.models;
        src.definition = request.definition;
        src.startSpawn = request.startSpawn;
        src.requiredLaps = request.start->race.progress.requiredLapCount;
        src.collision = request.collision;
        tw::fv::TwinWorldLabel label;
        std::string diag;
        const auto world = tw::fv::BuildTwinWorld(src, request.policy, label, &diag);
        if (!world) return Fail(ReplaySimulationRunResult::TwinExecutionFailed, diag);
        const tw::fv::TwinWorldPacks &packs = *world->Internal().packs;

        // 2. start state (after T = 2600)
        tw::model::TwinState state;
        if (!tw::fv::ImportFvClone(*world, *request.start, request.validationSeed, &state, &diag))
            return Fail(ReplaySimulationRunResult::TwinExecutionFailed, diag);
        if (request.tickCount == 0u) {
            TwinTimelineSegmentResult done;
            done.result = ReplaySimulationRunResult::Success;
            return done;
        }
        if (request.ticks[0].timeMs != state.tickMs + 10u)
            return Fail(ReplaySimulationRunResult::TwinExecutionFailed,
                        "twin timeline: the first twin tick T=" + std::to_string(request.ticks[0].timeMs) +
                                " does not follow the imported state's T=" + std::to_string(state.tickMs));

        // 3. controls, flattened as the ticks run. The validation tail tick
        // (board 30 R1.10: finish race / callback suppression) is not a
        // twin tick; M7 handles it (m6a-race.md "Scope notes").
        const std::size_t runCount = 1u;   // until collision (M3) lands: the first twin tick only
        std::vector<tw::race::TwinControl> controls(runCount);
        std::vector<std::uint32_t> respawns(runCount);
        for (std::size_t i = 0; i < runCount; ++i) {
            if (!FlattenTwinControlTick(request.ticks[i], &controls[i], &respawns[i], &diag))
                return Fail(ReplaySimulationRunResult::TwinExecutionFailed, "twin controls: " + diag);
        }

        // 4. ticks
        const std::uint32_t handoffMs = state.tickMs;
        tw::step::Events ev;
        tw::step::StopRule stop;
        // Through the world's ISA variant (isa/tw_isa.h).
        namespace twin_internal = forevervalidator::experimental::twin::internal;
        const twin_internal::CollisionCachePtr cache = twin_internal::NewCollisionCache(world->Internal());
        const std::uint32_t done = twin_internal::StepTicks(world->Internal(), state, controls.data(), runCount, stop,
                                                            &ev, respawns.data(), cache.get());
        char built[512];
        std::snprintf(built, sizeof built,
                      "hand-off imported at T=%u; world package %zu triangles (%u triggers, %zu spawns, %zu instances, "
                      "%u enclosure triangles left out), car pack, water %s, race map (%u checkpoints, %u laps); "
                      "compile: scene %.0f ms, world %.0f ms, car %.0f ms",
                      handoffMs, packs.view.tris.size(), static_cast<unsigned>(packs.view.triggers.size()),
                      packs.view.spawns.size(), packs.view.instances.size(), packs.view.header->enclosureTris,
                      packs.water.enabled ? "on" : "off", packs.race.requiredCps, packs.race.requiredLaps, packs.sceneMs,
                      packs.worldMs, packs.carMs);
        if (done == 0u && ev.status == tw::step::Status::NotImplemented) {
            return Fail(ReplaySimulationRunResult::TwinNotImplemented,
                        std::string("twin: ") + built + "; the first twin tick T=" +
                                std::to_string(request.ticks[0].timeMs) + " stopped: " +
                                (world->Internal().model.collision == nullptr ? "collision stage (M3) not built"
                                                            : "a tick stage is missing") +
                                " (engine missing: " + tw::step::Info().missing + ")");
        }
        if (done == 0u)
            return Fail(ReplaySimulationRunResult::TwinExecutionFailed,
                        std::string("twin: ") + built + "; the first twin tick failed: " + tw::step::StatusName(ev.status));
        return Fail(ReplaySimulationRunResult::TwinNotImplemented,
                    std::string("twin: ") + built + "; ran the first tick; per-tick observations are not implemented "
                                                    "yet (M7, src/twin/fv/twin_timeline.cpp)");
    } catch (const std::bad_alloc &) {
        return Fail(ReplaySimulationRunResult::TwinExecutionFailed, "twin timeline: allocation failed");
    }
}

}  // namespace forevervalidator::simulation
