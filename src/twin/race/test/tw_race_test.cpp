// M6a unit tests (FV-independent). Ports of FV's tests/checkpoint_freewheel_tests.cpp
// and tests/replay_control_plan_tests.cpp to the twin race layer, plus rule
// cases from board 30 that FV's tests do not cover.
#include <cstdio>
#include <cstring>
#include <vector>

#include "../tw_controls.h"
#include "../tw_race.h"

using namespace tw::race;
using tw::model::TwinState;

namespace {

int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                    \
    do {                                                    \
        ++g_checks;                                         \
        if (!(cond)) {                                      \
            ++g_fail;                                       \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);              \
            std::fprintf(stderr, "\n");                     \
        }                                                   \
    } while (0)

constexpr float kRest = 0.0125f;

struct MapFixture {
    std::vector<TwTrigger> triggers;
    std::vector<TwSpawn> spawns;
    TwRaceMap map;
    // trigger ids: 0..cps-1 checkpoints (slot i, spawn i+1), cps = finish, cps+1 = start line, cps+2 = role None
    MapFixture(std::uint32_t cps, std::uint32_t laps) {
        spawns.resize(cps + 1u);
        for (std::uint32_t i = 0; i <= cps; ++i) {
            std::memset(&spawns[i], 0, sizeof(TwSpawn));
            spawns[i].pos[0] = 100.0f * static_cast<float>(i);
            spawns[i].quat[3] = 1.0f;
        }
        for (std::uint32_t i = 0; i < cps; ++i)
            triggers.push_back({kRoleCheckpoint, 0u, static_cast<std::uint16_t>(i), static_cast<std::uint16_t>(i + 1u), 0u});
        triggers.push_back({kRoleFinish, 0u, kNoSlot, kNoSpawn, 0u});
        triggers.push_back({kRoleStart, 0u, kNoSlot, kNoSpawn, 0u});
        triggers.push_back({kRoleNone, 0u, kNoSlot, kNoSpawn, 0u});
        Bind(cps, laps);
    }
    void Bind(std::uint32_t cps, std::uint32_t laps) {
        map.triggers = triggers.data();
        map.triggerCount = static_cast<std::uint32_t>(triggers.size());
        map.spawns = spawns.data();
        map.spawnCount = static_cast<std::uint32_t>(spawns.size());
        map.requiredCps = cps;
        map.requiredLaps = laps;
    }
    std::uint32_t Finish() const { return map.requiredCps; }
};

TwinState RaceState() {  // after the race spawn: start spawn current and previous
    TwinState s;
    tw::model::TwinStateClear(&s);
    s.spawnInitialized = 1u;
    return s;
}

std::uint32_t Contact(const TwRaceMap &m, TwinState &s, std::uint32_t trigger, TwRaceEvents *out = nullptr) {
    TwRaceEvents ev;
    TwRaceEventsClear(ev);
    TwRaceOnTriggerContact(m, s, trigger, ev);
    if (out) *out = ev;
    return ev.cpAccepted;
}

// --- port: TestCurrentTransformCheckpointContactClearsFreewheel -------------
void TestCurrentTransformCheckpointContactClearsFreewheel() {
    // FV's test blocks belong to no course: no slot, the contact can only clear free wheel.
    std::vector<TwTrigger> t = {{kRoleCheckpoint, 1u, kNoSlot, kNoSpawn, 0u}, {kRoleCheckpoint, 0u, kNoSlot, kNoSpawn, 0u}};
    TwSpawn sp{};
    TwRaceMap m;
    m.triggers = t.data();
    m.triggerCount = 2u;
    m.spawns = &sp;
    m.spawnCount = 1u;
    TwinState s = RaceState();
    TwRaceEvents ev;
    s.freewheel = 1u;
    Contact(m, s, 0u, &ev);
    CHECK(s.freewheel == 0u, "current-transform checkpoint contact retained freewheel");
    CHECK(ev.freewheelClears == 1u, "current-transform checkpoint contact was not counted");
    m.clearFreewheelOnCurrentTransformContact = false;
    s.freewheel = 1u;
    Contact(m, s, 0u, &ev);
    CHECK(s.freewheel == 1u && ev.freewheelClears == 0u, "disabled rule changed the free wheel");
    m.clearFreewheelOnCurrentTransformContact = true;
    s.freewheel = 1u;
    Contact(m, s, 1u, &ev);
    CHECK(s.freewheel == 1u && ev.freewheelClears == 0u, "ordinary checkpoint contact cleared freewheel");
    // TestParityBackendRouting: the twin is a parity backend, the rule is on by default.
    CHECK(TwRaceMap{}.clearFreewheelOnCurrentTransformContact, "default map does not clear on current-transform contact");
}

// --- port: TestLapCountersRemainSeparate -----------------------------------
void TestLapCountersRemainSeparate() {
    for (const std::uint32_t cps : {0u, 2u}) {
        MapFixture f(cps, 2u);
        TwinState s = RaceState();
        for (std::uint32_t lap = 0; lap < 2u; ++lap) {
            for (std::uint32_t cp = 0; cp < cps; ++cp) {
                TwRaceEvents ev;
                TwRaceEventsClear(ev);
                CHECK(detail::InternalOnCheckpoint(f.map, s, cp, cp, cp + 1u, ev), "ordinary checkpoint was not accepted");
                CHECK(!detail::InternalOnCheckpoint(f.map, s, cp, cp, cp + 1u, ev), "duplicate ordinary checkpoint was accepted");
            }
            Contact(f.map, s, f.Finish());
            const TwinState saved = s;  // snapshot = plain copy
            s = saved;
            CHECK(s.laps == lap + 1u && s.cpCount == cps * (lap + 1u) && s.raceCompleted == (lap == 1u ? 1u : 0u),
                  "finish passage changed ordinary counts or snapshot lost laps (cps %u lap %u)", cps, lap);
        }
        Contact(f.map, s, f.Finish());
        CHECK(s.laps == 2u, "completed race counted another finish passage");
    }
}

// --- port: TestAcceptedCheckpointJournal -----------------------------------
void TestAcceptedCheckpointJournal() {
    MapFixture f(2u, 2u);
    TwinState s = RaceState();
    for (std::uint32_t lap = 1; lap <= 2; ++lap) {
        TwRaceEvents ev;
        TwRaceEventsClear(ev);
        TwRaceOnTriggerContact(f.map, s, f.Finish(), ev);
        CHECK(ev.journalCount == 0u, "premature finish was journaled");
        for (std::uint32_t index = 0; index < 2; ++index) {
            const std::uint32_t slot = 1u - index;
            TwRaceEvents a;
            TwRaceEventsClear(a);
            const bool first = detail::InternalOnCheckpoint(f.map, s, index, slot, slot + 1u, a);
            const bool second = detail::InternalOnCheckpoint(f.map, s, index, slot, slot + 1u, a);
            CHECK(first && !second, "journal changed acceptance rules");
            const TwAccepted &e = a.journal[0];
            CHECK(a.journalCount == 1u && e.slot == slot && e.index == index && e.lap == lap &&
                      e.eventIndex == (lap - 1u) * 3u + index + 1u && !e.finish,
                  "journal lost identity, lap, ordinal or global sequence");
        }
        TwRaceEventsClear(ev);
        TwRaceOnTriggerContact(f.map, s, f.Finish(), ev);
        CHECK(ev.journalCount == 1u && ev.journal[0].finish && ev.journal[0].lap == lap && ev.journal[0].slot == 2u &&
                  ev.journal[0].eventIndex == lap * 3u && ev.finish == 1u && ev.cpAccepted == 1u && ev.cpLastSlot == 2u,
              "finish event lost slot, lap or sequence");
        const TwinState saved = s;
        CHECK(saved.acceptedEvents == lap * 3u, "snapshot lost the accepted event sequence");
    }
    TwRaceEvents ev;
    TwRaceEventsClear(ev);
    TwRaceOnTriggerContact(f.map, s, f.Finish(), ev);
    CHECK(ev.journalCount == 0u && s.raceCompleted, "completed race produced another finish event");
    // new session = fresh state
    s = RaceState();
    CHECK(s.acceptedEvents == 0u, "new session retained accepted history");

    // An accepted unusual ordinal is not a checkpoint-count delta or a finish.
    TwRaceEventsClear(ev);
    CHECK(detail::InternalOnCheckpoint(f.map, s, 2u, 0u, 1u, ev) && s.cpCount == 0u && ev.journalCount == 1u &&
              !ev.journal[0].finish,
          "accepted bug-checkpoint event was inferred from counters instead of acceptance");
    const std::uint32_t before = ev.journalCount;
    detail::InternalOnCheckpoint(f.map, s, 0u, 99u, 1u, ev);
    CHECK(ev.journalCount == before, "invalid checkpoint request appeared in the journal");

    // Many same-tick transitions: the twin's slot limit is 64 (63 checkpoints + finish).
    MapFixture big(63u, 1u);
    CHECK(TwRaceMapValid(big.map), "63-checkpoint map rejected");
    MapFixture tooBig(64u, 1u);
    CHECK(!TwRaceMapValid(tooBig.map), "64-checkpoint map accepted (slot bitset holds 64 slots)");
    s = RaceState();
    TwRaceEventsClear(ev);
    for (std::uint32_t i = 0; i < 63u; ++i) TwRaceOnTriggerContact(big.map, s, i, ev);
    TwRaceOnTriggerContact(big.map, s, big.Finish(), ev);
    CHECK(ev.journalCount == 64u && ev.cpAccepted == 64u && s.acceptedEvents == 64u && s.raceCompleted,
          "same-tick transitions lost (journal %u)", ev.journalCount);

    // Shortcut finish bypasses the slot test but is journaled.
    MapFixture sc(0u, 1u);
    sc.map.shortcutMode = true;
    s = RaceState();
    TwRaceEventsClear(ev);
    TwRaceOnTriggerContact(sc.map, s, sc.Finish(), ev);
    CHECK(ev.journalCount == 1u && ev.journal[0].finish && ev.journal[0].eventIndex == 1u && s.raceCompleted,
          "shortcut finish bypassed the accepted-event journal");
}

// --- board 30 cases FV's unit tests do not cover ---------------------------
void TestRules() {
    MapFixture f(2u, 1u);
    TwinState s = RaceState();
    TwRaceEvents ev;
    // R3.4: start line is prepared but changes nothing; role None is ignored entirely
    TwRaceEventsClear(ev);
    TwRaceOnTriggerContact(f.map, s, f.Finish() + 1u, ev);
    TwRaceOnTriggerContact(f.map, s, f.Finish() + 2u, ev);
    CHECK(ev.contacts == 1u && ev.cpAccepted == 0u, "start line / none dispatch");
    // R3.7: finish with missing checkpoints is rejected
    CHECK(Contact(f.map, s, f.Finish()) == 0u && !s.raceCompleted, "finish accepted with 0/2 checkpoints");
    // R3.5: any order, R5.1 spawn = CP spawn, free wheel cleared on accept
    s.freewheel = 1u;
    CHECK(Contact(f.map, s, 1u, &ev) == 1u && s.spawnIndex == 2u && s.prevSpawnIndex == 0u && s.freewheel == 0u &&
              ev.freewheelClears == 1u && ev.cpLastSlot == 1u,
          "checkpoint 1 first");
    CHECK(Contact(f.map, s, 1u) == 0u, "slot accepted twice in a lap");
    CHECK(Contact(f.map, s, 0u) == 1u && s.spawnIndex == 1u && s.lapCpCount == 2u, "checkpoint 0 second");
    // finish: spawn back to the race start spawn, race completes
    CHECK(Contact(f.map, s, f.Finish(), &ev) == 1u && s.raceCompleted && ev.raceCompleted && ev.finish &&
              s.spawnIndex == 0u && s.laps == 1u && s.finishCount == 1u && s.cpEvents == 3u && s.cpCount == 2u,
          "finish");

    // current-transform checkpoint keeps the respawn location; CP without spawn goes back to previous
    std::vector<TwTrigger> t = {{kRoleCheckpoint, 1u, 0u, 1u, 0u}, {kRoleCheckpoint, 0u, 1u, kNoSpawn, 0u},
                                {kRoleCheckpoint, 0u, 2u, 3u, 0u}, {kRoleFinish, 0u, kNoSlot, kNoSpawn, 0u}};
    f.triggers = t;
    f.spawns.resize(4u);
    f.Bind(3u, 1u);
    s = RaceState();
    Contact(f.map, s, 2u);
    CHECK(s.spawnIndex == 3u, "ordinary spawn");
    s.freewheel = 1u;
    Contact(f.map, s, 0u, &ev);
    CHECK(s.spawnIndex == 3u && s.freewheel == 0u && ev.freewheelClears == 2u, "current-transform CP: spawn kept, 2 clears");
    Contact(f.map, s, 0u, &ev);
    CHECK(ev.cpAccepted == 0u && ev.freewheelClears == 1u, "repeated current-transform contact clears again");
    Contact(f.map, s, 1u);
    CHECK(s.spawnIndex == 0u, "CP without a spawn iso falls back to the previous spawn");

    // laps: slots reset at the lap end, 0-CP lap map counts two finish contacts in one substep as two laps
    MapFixture lap0(0u, 3u);
    s = RaceState();
    const TwTriggerHit hits[2] = {{static_cast<std::uint16_t>(lap0.Finish()), 4u, 0u},
                                  {static_cast<std::uint16_t>(lap0.Finish()), 5u, 0u}};
    TwRaceEventsClear(ev);
    TwRaceOnTriggerHits(lap0.map, s, hits, 2u, ev);
    CHECK(s.laps == 2u && !s.raceCompleted && ev.cpAccepted == 2u, "0-CP lap map: two finish contacts = two laps");
    MapFixture lap2(2u, 2u);
    s = RaceState();
    Contact(lap2.map, s, 0u);
    Contact(lap2.map, s, 1u);
    Contact(lap2.map, s, lap2.Finish());
    CHECK(s.laps == 1u && s.lapCpCount == 0u && s.cpSlots[0] == 0u && !s.raceCompleted, "lap end resets slots");
    CHECK(Contact(lap2.map, s, 0u) == 1u && s.cpCount == 3u, "checkpoint counts again in lap 2");
    // requiredLaps == 0: never completes
    MapFixture never(0u, 0u);
    s = RaceState();
    Contact(never.map, s, never.Finish());
    CHECK(s.laps == 1u && !s.raceCompleted, "requiredLaps 0 completed");
}

void TestTransitions() {
    MapFixture f(1u, 1u);
    TwinState s;
    tw::model::TwinStateClear(&s);
    s.speedBlocked = s.speedBlocked2 = 1u;
    TwRaceEvents ev;
    TwRaceEventsClear(ev);
    // R5.6: respawn without a location is a no-op and not counted
    TwRaceTickBegin(f.map, s, 0u, 1u, 10u, kRest, ev);
    CHECK(ev.respawned == 0u && s.respawns == 0u, "respawn without a location counted");
    // R1.5: establish spawn
    s.slipActive = 1u;
    TwRaceTickBegin(f.map, s, kCtlEstablishSpawn, 0u, 20u, kRest, ev);
    CHECK(s.spawnInitialized && s.spawnIndex == 0u && s.prevSpawnIndex == 0u && !s.slipActive &&
              s.slipLastTickMs == 0xFFFFFFFFu && s.tickMs == 20u,
          "establish spawn");
    // R1.3/R1.4: begin race + race-start reset
    s.freewheel = 1u;
    s.trbType = 1u;
    s.gear = 3u;
    s.vel[0] = 5.0f;
    TwRaceTickBegin(f.map, s, kCtlBeginRace | kCtlRaceStartReset | kCtlGas, 0u, 2600u, kRest, ev);
    CHECK(!s.speedBlocked && !s.speedBlocked2 && s.trbRouletteOriginMs == 2600u && !s.freewheel && !s.trbType &&
              s.gear == 1u && s.vel[0] == 5.0f && s.absorb[0] == kRest,
          "race-start reset");
    const TwTickInputs in = TwCtlPhysicsInputs(kCtlBeginRace | kCtlRaceStartReset | kCtlGas | TwCtlMake(-65536, true, true));
    CHECK(in.gas == 0.0f && in.brake == 0.0f && in.steer == 0.0f, "race-start tick must run with zero input");
    // R5: standing respawn at the current spawn, counted per respawn
    Contact(f.map, s, 0u);
    s.vel[1] = -3.0f;
    s.angVel[2] = 1.0f;
    s.pos[0] = 7.0f;
    s.groundedPrev = 0u;
    s.speedBlocked2 = 1u;
    s.prevForce[0] = 2.0f;
    TwRaceEventsClear(ev);
    TwRaceTickBegin(f.map, s, kCtlBeginRace | (2u << kCtlRespawnShift), 2u, 3000u, kRest, ev);
    CHECK(ev.respawned == 2u && s.respawns == 2u && s.pos[0] == 100.0f && s.vel[1] == 0.0f && s.angVel[2] == 0.0f &&
              s.groundedPrev == 1u && !s.speedBlocked2 && s.prevForce[0] == 2.0f && s.cpCount == 1u,
          "standing respawn");
}

void TestControlDecode() {
    for (const std::int32_t v : {-65536, -65535, -656, -655, -1, 0, 1, 655, 656, 32768, 65535, 65536}) {
        const TwinControl c = TwCtlMake(v, v > 0, v < 0);
        CHECK(TwCtlSteerRaw(c) == v, "steer raw %d", v);
        CHECK(TwCtlSteer(c) == static_cast<float>(v) / 65536.0f, "steer float %d", v);
        CHECK(TwCtlGas(c) == (v > 0 ? 1.0f : 0.0f) && TwCtlBrake(c) == (v < 0 ? 1.0f : 0.0f), "gates %d", v);
    }
}

TwInputEvent Sw(std::int32_t t, TwAction a, bool press) {
    return {t, a, TwValueKind::Switch, press ? TwSwitch::Pressed : TwSwitch::Released, 0};
}
TwInputEvent An(std::int32_t t, TwAction a, std::int32_t v) { return {t, a, TwValueKind::Analog, TwSwitch::Released, v}; }

// --- port: TestRaceValidationAppendsUnobservedTick (minus the validation tail tick) ---
void TestControlPlanPort() {
    const std::vector<TwInputEvent> e = {Sw(10, TwAction::Respawn, true), Sw(20, TwAction::Accelerate, true),
                                         Sw(20, TwAction::RaceRunning, true), Sw(20, TwAction::FinishLine, true),
                                         Sw(20, TwAction::Respawn, true)};
    TwTimeline tl;
    tl.tickMs = 10u;
    tl.prestartMs = 10u;
    tl.inputTimeBaseMs = 0u;
    tl.enableRaceAfterMs = 30;
    tl.establishSpawnAtMs = 30;
    TwControlStream cs;
    std::string err;
    CHECK(TwBuildControls(e.data(), e.size(), tl, 10u, 3u, &cs, &err), "build: %s", err.c_str());
    CHECK(cs.ticks.size() == 3u, "tick count");
    CHECK(cs.ticks[0] == 0u && cs.ticks[1] == 0u, "ticks before the race carry nothing");
    const TwinControl c = cs.ticks[2];
    CHECK((c & kCtlEstablishSpawn) && (c & kCtlRaceStartReset) && (c & kCtlBeginRace) && cs.Respawns(2) == 1u &&
              TwCtlGas(c) == 1.0f,
          "horizon tick lost its transitions (respawn before RaceRunning must not count)");
}

void TestControlStateMachine() {
    TwTimeline tl;  // FV sandbox defaults
    TwControlStream cs;
    // steering: analog vs digital by last change, tie rules
    std::vector<TwInputEvent> e = {
        Sw(0, TwAction::RaceRunning, true),
        An(100, TwAction::Steer, 30000),     // tick 2700
        Sw(200, TwAction::SteerLeft, true),  // 2800: digital later -> -1
        An(300, TwAction::Steer, 500),       // 2900: analog later -> 500
        Sw(400, TwAction::SteerLeft, false), // 3000: digital later, none held -> 0
        An(500, TwAction::Gas, -20000),      // 3100: analog gas wins -> gas
        Sw(600, TwAction::Brake, true),      // 3200: digital later -> brake only
        Sw(600, TwAction::Accelerate, true), //        same time -> gas + brake (bugslide)
        An(700, TwAction::Gas, 19661),       // 3300: analog later -> brake
        An(800, TwAction::Gas, 19660),       // 3400: dead zone -> none
        Sw(900, TwAction::Respawn, true), Sw(900, TwAction::Respawn, false), Sw(900, TwAction::Respawn, true),  // 3500: 2
        {905, TwAction::Respawn, TwValueKind::Switch, TwSwitch::NonCanonicalActive, 0},                        // 3510: 1
    };
    CHECK(TwBuildControls(e.data(), e.size(), tl, 2600u, 92u, &cs), "build");
    auto at = [&](std::uint32_t T) { return cs.ticks[(T - 2600u) / 10u]; };
    CHECK(at(2600) & kCtlRaceStartReset, "race start reset at 2600");
    CHECK(!(at(2610) & kCtlRaceStartReset), "race start reset once");
    CHECK(TwCtlSteerRaw(at(2690)) == 0 && TwCtlSteerRaw(at(2700)) == 30000, "analog steer from its tick");
    CHECK(TwCtlSteerRaw(at(2800)) == -65536, "digital later wins");
    CHECK(TwCtlSteerRaw(at(2900)) == 500, "analog later wins even below the dead test");
    CHECK(TwCtlSteerRaw(at(3000)) == 0, "digital release later: centred");
    CHECK(TwCtlGas(at(3100)) == 1.0f && TwCtlBrake(at(3100)) == 0.0f, "analog gas");
    CHECK(TwCtlGas(at(3200)) == 1.0f && TwCtlBrake(at(3200)) == 1.0f, "gas + brake together");
    CHECK(TwCtlGas(at(3300)) == 0.0f && TwCtlBrake(at(3300)) == 1.0f, "analog brake threshold");
    CHECK(TwCtlGas(at(3400)) == 0.0f && TwCtlBrake(at(3400)) == 0.0f, "analog dead zone");
    CHECK(cs.Respawns((3500 - 2600) / 10) == 2u && cs.Respawns((3510 - 2600) / 10) == 1u, "respawn counts");
    // a window starting later sees the same held state
    TwControlStream w;
    CHECK(TwBuildControls(e.data(), e.size(), tl, 3200u, 3u, &w) && w.ticks[0] == at(3200), "window = full stream");
    // escape: 300 respawns in one tick
    std::vector<TwInputEvent> many = {Sw(0, TwAction::RaceRunning, true)};
    for (int i = 0; i < 300; ++i) many.push_back(Sw(50, TwAction::Respawn, true));
    CHECK(TwBuildControls(many.data(), many.size(), tl, 2600u, 10u, &cs), "build many");
    CHECK((cs.ticks[5] & kCtlRespawnEscape) && cs.Respawns(5) == 300u, "respawn escape");
    // errors
    std::vector<TwInputEvent> bad = {Sw(10, TwAction::Accelerate, true), Sw(5, TwAction::Brake, true)};
    CHECK(!TwBuildControls(bad.data(), bad.size(), tl, 10u, 5u, &cs), "unordered events accepted");
    std::vector<TwInputEvent> range = {An(10, TwAction::Steer, 65537)};
    CHECK(!TwBuildControls(range.data(), range.size(), tl, 10u, 5u, &cs), "out-of-range analog accepted");
    // RaceRunning before the race start: no reset ever (FV: previousRaceRunning updates every tick)
    std::vector<TwInputEvent> early = {Sw(-500, TwAction::RaceRunning, true)};
    CHECK(TwBuildControls(early.data(), early.size(), tl, 10u, 300u, &cs), "build early");
    bool reset = false;
    for (TwinControl c : cs.ticks) reset |= (c & kCtlRaceStartReset) != 0u;
    CHECK(!reset && (cs.ticks[0] & kCtlEstablishSpawn) && !(cs.ticks[1] & kCtlEstablishSpawn), "early RaceRunning");
}

void TestFinishBracket() {
    // monotone probe: completes iff dt >= threshold
    for (const float threshold : {0.0001f, 0.00337f, 0.0099f, 0.01f}) {
        const TwFinishSubstep sub{4570u, 0.0, 0.01f};
        std::uint64_t lo = 0, hi = 0;
        int probes = 0;
        const bool ok = TwRefineFinish(sub, [&](float dt) { ++probes; return dt >= threshold; }, &lo, &hi);
        const double thrNs = 4570.0e6 + static_cast<double>(threshold) * 1e9;
        CHECK(ok && hi == lo + 1u && static_cast<double>(lo) <= thrNs + 2.0 && static_cast<double>(hi) >= thrNs - 2.0 &&
                  probes <= 26,
              "bracket thr %g: (%llu, %llu] probes %d", threshold, (unsigned long long)lo, (unsigned long long)hi, probes);
    }
    std::uint64_t lo, hi;
    CHECK(!TwRefineFinish(TwFinishSubstep{10u, 0.0, 0.01f}, [](float) { return false; }, &lo, &hi),
          "probe that never completes gave a bracket");
    TwinState s;
    tw::model::TwinStateClear(&s);
    TwStoreFinish(s, 0x123456789ull);
    CHECK(s.finishPresent && s.finishLowerNs[0] == 0x23456789u && s.finishLowerNs[1] == 1u, "finish store");
}

}  // namespace

int main() {
    TestCurrentTransformCheckpointContactClearsFreewheel();
    TestLapCountersRemainSeparate();
    TestAcceptedCheckpointJournal();
    TestRules();
    TestTransitions();
    TestControlDecode();
    TestControlPlanPort();
    TestControlStateMachine();
    TestFinishBracket();
    std::printf("tw_race_test: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
