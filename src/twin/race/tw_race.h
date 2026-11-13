// Twin race layer (milestone M6a, docs/twin/m6a-race.md; rules: docs/twin/board/30-race-rules.md).
//
// Header-only, FV-independent, scalar. Everything here runs on one TwinState
// (src/twin/model/tw_state.h); batch code gathers the few lanes with a race
// event (respawn, trigger contact, finish) with TwinLoadLane / TwinStoreLane.
//
// Contents
//   TwinControl     the per-tick u32 (design.md §3.3) and its decoders
//   TwRaceMap       per-map race data: trigger table, spawn table, required counts
//   TwTriggerHit    THE INTERFACE COLLISION PROVIDES (one entry per surviving
//                   trigger contact of a substep, sort order, see below)
//   TwRaceTickBegin tick transitions: race spawn, begin race, race-start reset,
//                   respawns (board 30 R1.4, R1.5, R5)
//   TwRaceOnTriggerHits  checkpoint / finish / lap rules (R3.4-R3.8)
//   TwVehicleReset, TwRespawnPlace   VehicleReset and the standing respawn on TwinState
//   TwRefineFinish  the sub-tick finish bracket driver (R4.2), probe supplied by the step
//
// Not here (owned by the car-force kernel, src/twin/model/forces.h C17 / water):
// turbo and roulette activation, the free-wheel SET from wheel materials, water
// entry. The race layer only clears free wheel and sets the roulette origin.
#ifndef TW_RACE_H
#define TW_RACE_H

#include <cmath>  // std::isfinite only (compiler builtin, no libm call)
#include <cstdint>
#include <cstring>
#include <limits>

#include "tw_math_config.h"   // TW_HD
#include "tw_state.h"

namespace tw {
namespace race {

using model::TwinState;

// ---------------------------------------------------------------------------
// TwinControl (design.md §3.3)
//   bits 0-17  signed steer in 1/65536 (analog value or +-65536 digital)
//   bit 18     gas gate      bit 19 brake gate
//   bits 20-27 respawns executed before physics this tick (255 = see escape)
//   bit 28     escape: respawn count >= 255, the exact count is in TwControlStream::respawnOverflow
//   bit 29     race-start reset (VehicleReset after the controls: the tick runs with zero input)
//   bit 30     begin race (BeginRaceSimulation: speed blocks off; every tick >= prestart)
//   bit 31     establish race spawn (first tick)
using TwinControl = std::uint32_t;
constexpr std::uint32_t kCtlSteerMask = 0x3FFFFu;
constexpr std::uint32_t kCtlGas = 1u << 18;
constexpr std::uint32_t kCtlBrake = 1u << 19;
constexpr int kCtlRespawnShift = 20;
constexpr std::uint32_t kCtlRespawnMask = 0xFFu << kCtlRespawnShift;
constexpr std::uint32_t kCtlRespawnEscape = 1u << 28;
constexpr std::uint32_t kCtlRaceStartReset = 1u << 29;
constexpr std::uint32_t kCtlBeginRace = 1u << 30;
constexpr std::uint32_t kCtlEstablishSpawn = 1u << 31;
constexpr std::int32_t kAnalogScale = 65536;

TW_HD inline std::int32_t TwCtlSteerRaw(TwinControl c) {
    std::int32_t v = static_cast<std::int32_t>(c & kCtlSteerMask);
    return (v & 0x20000) ? v - 0x40000 : v;
}
// Same arithmetic as FV's ControlsFromState: value / 65536 (digital: -1 / +1 exactly).
TW_HD inline float TwCtlSteer(TwinControl c) {
    return static_cast<float>(TwCtlSteerRaw(c)) / static_cast<float>(kAnalogScale);
}
TW_HD inline float TwCtlGas(TwinControl c) { return (c & kCtlGas) ? 1.0f : 0.0f; }
TW_HD inline float TwCtlBrake(TwinControl c) { return (c & kCtlBrake) ? 1.0f : 0.0f; }
TW_HD inline std::uint32_t TwCtlRespawnField(TwinControl c) { return (c & kCtlRespawnMask) >> kCtlRespawnShift; }
TW_HD inline TwinControl TwCtlMake(std::int32_t steerRaw, bool gas, bool brake) {
    return (static_cast<std::uint32_t>(steerRaw) & kCtlSteerMask) | (gas ? kCtlGas : 0u) | (brake ? kCtlBrake : 0u);
}

// The inputs the forces see this tick. The race-start reset runs after the
// tick's controls were applied and zeroes them (board 30 R1.4, 40:T7).
struct TwTickInputs {
    float gas, brake, steer;
};
TW_HD inline TwTickInputs TwCtlPhysicsInputs(TwinControl c) {
    if (c & kCtlRaceStartReset) return {0.0f, 0.0f, 0.0f};
    return {TwCtlGas(c), TwCtlBrake(c), TwCtlSteer(c)};
}

// ---------------------------------------------------------------------------
// Per-map race data (compiled with the world, M2; built from FV's
// ReplayStaticCorpusCollection, see test/tw_race_fvcheck.cpp BuildRaceMap).
enum : std::uint8_t {  // engine/game/block_race_role.h
    kRoleStart = 0,
    kRoleFinish = 1,
    kRoleCheckpoint = 2,
    kRoleNone = 3,
    kRoleStartFinish = 4,
};
constexpr std::uint16_t kNoSlot = 0xFFFFu;
constexpr std::uint16_t kNoSpawn = 0xFFFFu;
constexpr std::uint32_t kMaxSlots = 64u;  // cpSlots is a u64 bitset: requiredCps + 1 <= 64

struct TwTrigger {             // one per race-trigger corpus, in FV installation order (= trigger id)
    std::uint8_t role;         // kRole*
    std::uint8_t currentTransform;  // block info RespawnUsesCurrentTransform
    std::uint16_t slot;        // checkpoint slot (CheckpointSlotForBlock), kNoSlot if none
    std::uint16_t spawn;       // spawn-table index (SpawnIsoForCheckpointBlock), kNoSpawn if none
    std::uint16_t reserved;
};

struct TwSpawn {               // a respawn location
    float pos[3];              // iso translation
    float quat[4];             // x y z w, FV's GmQuat::Set(rotation) (computed at compile)
    std::uint32_t iso[12];     // FV GmIso4 words: basisX, basisY, basisZ, translation (golden matching)
};

struct TwRaceMap {
    const TwTrigger *triggers = nullptr;
    std::uint32_t triggerCount = 0u;
    const TwSpawn *spawns = nullptr;  // [0] = race start spawn, UNSEEDED (board 30 R1.6)
    std::uint32_t spawnCount = 0u;
    std::uint32_t requiredCps = 0u;   // number of checkpoint triggers (R3.6)
    std::uint32_t requiredLaps = 1u;  // map laps if lap race, else 1 (R3.8)
    bool clearFreewheelOnCurrentTransformContact = true;  // FV parity backends (R3.5.1)
    bool shortcutMode = false;        // EChallengePlayMode::Shortcut (R3.9), not used by Stadium Race
};

// Checks the limits the compact state imposes (slot bitset, u16 indices).
TW_HD inline bool TwRaceMapValid(const TwRaceMap &m) {
    if (m.requiredCps + 1u > kMaxSlots || m.spawnCount == 0u || m.spawnCount > 0xFFFEu) return false;
    for (std::uint32_t i = 0; i < m.triggerCount; ++i) {
        const TwTrigger &t = m.triggers[i];
        if (t.role > kRoleStartFinish) return false;
        if (t.slot != kNoSlot && t.slot >= m.requiredCps) return false;
        if (t.spawn != kNoSpawn && t.spawn >= m.spawnCount) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// THE COLLISION INTERFACE (what M3/M5 must hand the race layer).
//
// For every substep, after the per-shape merge and the sort (board 20 §3.4,
// collision_sorting.cpp), the contacts whose static triangle belongs to a
// race trigger, in sort order, ONE ENTRY PER SURVIVING CONTACT (not
// de-duplicated: FV runs the trigger callback per contact; a second finish
// contact in the same substep can count a lap on a 0-checkpoint lap map).
// Triggers stay inside the merge with the solid contacts (R3.2): a trigger
// edge contact dropped by a road face contact is not a hit, a trigger face
// contact can drop a solid edge contact. Trigger contacts apply no impulse.
// The call point is collision response of the substep (after forces and
// integration of that substep): the free-wheel clear lands after C17 of the
// same substep set it. Role-None triggers may be omitted or passed.
struct TwTriggerHit {
    std::uint16_t trigger;  // index into TwRaceMap::triggers
    std::uint8_t shape;     // car shape 0..7 (informative)
    std::uint8_t flags;     // reserved
};

// ---------------------------------------------------------------------------
// Events of one tick (zero them at tick start with TwRaceEventsClear).
struct TwAccepted {           // FV's accepted-checkpoint journal entry
    std::uint32_t slot, index, lap, eventIndex;
    bool finish;
};
constexpr std::uint32_t kJournalMax = 8u;

struct TwRaceEvents {
    std::uint32_t contacts;        // role-carrying trigger contacts (FV preparedEventCount)
    std::uint32_t cpAccepted;      // accepted checkpoint AND finish passages (harness ev.cp_accepted)
    std::uint32_t cpLastSlot;      // slot of the last accepted passage, 0xFFFFFFFF = none
    std::uint32_t finish;          // a finish passage was accepted (lap end or race end)
    std::uint32_t raceCompleted;   // the race completed in this tick
    std::uint32_t respawned;       // respawns executed (counted) this tick
    std::uint32_t freewheelClears; // FV freewheelClearCount delta
    std::uint32_t journalCount;    // accepted passages (may exceed kJournalMax)
    TwAccepted journal[kJournalMax];
};
TW_HD inline void TwRaceEventsClear(TwRaceEvents &e) {
    std::memset(&e, 0, sizeof e);
    e.cpLastSlot = 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// VehicleReset on the twin state (FV scene_vehicle_car.cpp:524-542 and the
// functions it calls). Fields not listed survive, as in FV: prevForce
// (gearedDrive.scaledCurrentForce), burnout centre/radii/direction, turbo end /
// source / phase / roulette origin, wheel surface material, peer axis / id,
// latest contact point, position and velocities.
TW_HD inline void TwVehicleReset(TwinState &s, float wheelRestAbsorb) {
    // ResetPlayerControls (:410-418); gas/brake/steer are the tick input, see TwCtlPhysicsInputs
    s.steerCur = 0.0f;
    s.freewheel = 0u;
    // ResetTurbo (:442-446)
    s.trbType = 0u;
    s.trbScale = 0.0f;
    // ResetAirControl (:436-440)
    s.airRefresh = 0u;
    s.airMemTickMs = 0u;
    s.airMem[0] = s.airMem[1] = s.airMem[2] = 0.0f;
    // ResetGearedDrive (:454-468)
    s.slipActive = 0u;
    s.slipLastTickMs = 0xFFFFFFFFu;
    s.brnStartMs = 0xFFFFFFFFu;
    s.brnExitStartMs = 0xFFFFFFFFu;
    s.brnNormal[0] = s.brnNormal[1] = s.brnNormal[2] = 0.0f;
    s.localSpeedPrev[0] = s.localSpeedPrev[1] = s.localSpeedPrev[2] = 0.0f;
    s.brnPhase = 0u;     // Inactive
    s.engineState = 0u;  // Steady
    // ResetImpacts (:470-489)
    s.bodyContact = 0u;
    s.latContact = 0u;
    s.latTickMs = 0xFFFFFFFFu;
    s.inWater = 0u;      // noGroundFrictionGuard
    // WheelReset (scene_vehicle_car_wheels.cpp:1003-1034)
    for (int w = 0; w < 4; ++w) {
        s.maxRepl[w] = 0.0f;
        s.absorbVel[w] = 0.0f;
        s.absorb[w] = wheelRestAbsorb;
        s.contact[w] = 0u;
        s.slipping[w] = 0u;
        s.normalSum[3 * w] = s.normalSum[3 * w + 1] = s.normalSum[3 * w + 2] = 0.0f;
    }
    // SEngine::Reset (:546-554)
    s.rpm = 0.0f;
    s.gear = 1u;
    s.shiftCooldown = 0.0f;
    s.slipRpmScale = 1.0f;
    s.reverse = 0u;      // useLowSpeedGateB
}

// Standing respawn (replay_vehicle_simulation.cpp:203-225): VehicleReset, dyna
// states reset (zero velocities), speed block 2 off, dyna parameters
// reinstalled (force scale = groundedSolidFeedback1, linear damping 0: the
// grounded pair), car placed at the spawn iso. Held controls survive (input).
TW_HD inline void TwRespawnPlace(TwinState &s, const TwSpawn &spawn, float wheelRestAbsorb) {
    TwVehicleReset(s, wheelRestAbsorb);
    for (int k = 0; k < 3; ++k) {
        s.pos[k] = spawn.pos[k];
        s.vel[k] = 0.0f;
        s.angVel[k] = 0.0f;
    }
    for (int k = 0; k < 4; ++k) s.quat[k] = spawn.quat[k];
    s.speedBlocked2 = 0u;
    s.groundedPrev = 1u;
}

// ---------------------------------------------------------------------------
// Tick transitions and respawns: everything FV does for the race between the
// tick start and the physics step (replay_simulation_runtime.cpp:519-539,
// replay_vehicle_simulation.cpp:118-137). `respawnCount` is the decoded count
// (TwCtlRespawnField, or the overflow table when the escape bit is set).
TW_HD inline void TwRaceTickBegin(const TwRaceMap &m, TwinState &s, TwinControl c, std::uint32_t respawnCount,
                            std::uint32_t tickMs, float wheelRestAbsorb, TwRaceEvents &ev) {
    s.tickMs = tickMs;
    if (c & kCtlEstablishSpawn) {
        // EstablishRaceSpawnFrame (scene_vehicle_car.cpp:377-382) + SetInitialSpawnLocation
        s.slipActive = 0u;
        s.slipLastTickMs = 0xFFFFFFFFu;
        s.spawnIndex = 0u;
        s.prevSpawnIndex = 0u;
        s.spawnInitialized = 1u;
    }
    if (c & kCtlBeginRace) {  // BeginRaceSimulation (:348-355)
        s.speedBlocked = 0u;
        s.speedBlocked2 = 0u;
    }
    if (c & kCtlRaceStartReset) {  // replay_vehicle_simulation.cpp:129-136
        s.trbRouletteOriginMs = tickMs;
        s.speedBlocked = 0u;
        TwVehicleReset(s, wheelRestAbsorb);
    }
    for (std::uint32_t i = 0; i < respawnCount; ++i) {
        if (!s.spawnInitialized || s.spawnIndex >= m.spawnCount) continue;  // R5.6: no location, not counted
        TwRespawnPlace(s, m.spawns[s.spawnIndex], wheelRestAbsorb);
        s.respawns = static_cast<std::uint16_t>(s.respawns + 1u);
        ++ev.respawned;
    }
}

// ---------------------------------------------------------------------------
// Checkpoint / finish rules (trackmania_race.cpp:114-315).
namespace detail {

// slot < kMaxSlots (64) is checked by the caller; the mask keeps the index provably in range.
TW_HD inline bool SlotPassed(const TwinState &s, std::uint32_t slot) {
    return (s.cpSlots[(slot >> 5) & 1u] >> (slot & 31u)) & 1u;
}
TW_HD inline void SetSlot(TwinState &s, std::uint32_t slot) { s.cpSlots[(slot >> 5) & 1u] |= 1u << (slot & 31u); }

TW_HD inline void ClearFreewheel(TwinState &s, TwRaceEvents &ev) {
    s.freewheel = 0u;
    ++ev.freewheelClears;
}

TW_HD inline void Record(const TwinState &s, TwRaceEvents &ev, std::uint32_t index, std::uint32_t slot, bool finish) {
    if (ev.journalCount < kJournalMax)
        ev.journal[ev.journalCount] = {slot, index, static_cast<std::uint32_t>(s.laps) + 1u,
                                       static_cast<std::uint32_t>(s.acceptedEvents) + 1u, finish};
    ++ev.journalCount;
    ++ev.cpAccepted;
    ev.cpLastSlot = slot;
    if (finish) ev.finish = 1u;
}

// InternalOnCheckpoint (:199-243). `spawn` = kNoSpawn takes FV's null-iso
// branch (back to the previous spawn).
TW_HD inline bool InternalOnCheckpoint(const TwRaceMap &m, TwinState &s, std::uint32_t index, std::uint32_t slot,
                                 std::uint32_t spawn, TwRaceEvents &ev) {
    const std::uint32_t size = m.requiredCps + 1u;
    if (index >= size || slot >= size || slot >= kMaxSlots || SlotPassed(s, slot)) return false;
    Record(s, ev, index, slot, slot == m.requiredCps);
    s.acceptedEvents = static_cast<std::uint16_t>(s.acceptedEvents + 1u);
    SetSlot(s, slot);
    if (index != m.requiredCps) {
        s.lapCpCount = static_cast<std::uint16_t>(s.lapCpCount + 1u);
        s.cpCount = static_cast<std::uint16_t>(s.cpCount + 1u);
    }
    s.cpEvents = static_cast<std::uint16_t>(s.cpEvents + 1u);
    if (spawn != kNoSpawn) {
        if (!s.spawnInitialized) {  // EnsurePreviousSpawnLocationInitialized
            s.prevSpawnIndex = static_cast<std::uint16_t>(spawn);
            s.spawnInitialized = 1u;
        }
        s.spawnIndex = static_cast<std::uint16_t>(spawn);
    } else {
        s.spawnIndex = s.prevSpawnIndex;
    }
    ClearFreewheel(s, ev);
    return true;
}

TW_HD inline void OnCheckpoint(const TwRaceMap &m, TwinState &s, const TwTrigger &t, TwRaceEvents &ev) {
    if (m.clearFreewheelOnCurrentTransformContact && t.currentTransform) ClearFreewheel(s, ev);  // R3.5.1
    if (t.slot == kNoSlot) return;
    const std::uint32_t spawn = t.currentTransform ? s.spawnIndex : t.spawn;  // R5.1
    InternalOnCheckpoint(m, s, s.lapCpCount, t.slot, spawn, ev);
}

TW_HD inline void OnFinishLine(const TwRaceMap &m, TwinState &s, TwRaceEvents &ev) {
    if (s.raceCompleted || s.lapCpCount < m.requiredCps) return;
    if (m.shortcutMode) {
        Record(s, ev, m.requiredCps, m.requiredCps, true);
        s.acceptedEvents = static_cast<std::uint16_t>(s.acceptedEvents + 1u);
        s.finishCount = static_cast<std::uint16_t>(s.finishCount + 1u);
        s.laps = 1u;
        s.raceCompleted = 1u;
        ev.raceCompleted = 1u;
        return;
    }
    if (!InternalOnCheckpoint(m, s, m.requiredCps, m.requiredCps, kNoSpawn, ev)) return;
    s.finishCount = static_cast<std::uint16_t>(s.finishCount + 1u);
    s.laps = static_cast<std::uint16_t>(s.laps + 1u);
    if (m.requiredLaps != 0u && s.laps >= m.requiredLaps) {
        s.raceCompleted = 1u;
        ev.raceCompleted = 1u;
        return;
    }
    s.cpSlots[0] = s.cpSlots[1] = 0u;  // ResetCheckpointSlots: new lap
    s.lapCpCount = 0u;
}

}  // namespace detail

// One trigger contact (OnCheckpointContact, :279-315). Events carry the tick
// time (R1.8); nothing here depends on the substep.
TW_HD inline void TwRaceOnTriggerContact(const TwRaceMap &m, TwinState &s, std::uint32_t trigger, TwRaceEvents &ev) {
    if (trigger >= m.triggerCount) return;
    const TwTrigger &t = m.triggers[trigger];
    if (t.role == kRoleNone) return;
    ++ev.contacts;  // InternalPrepareEvent
    switch (t.role) {
    case kRoleFinish:
    case kRoleStartFinish: detail::OnFinishLine(m, s, ev); return;
    case kRoleCheckpoint: detail::OnCheckpoint(m, s, t, ev); return;
    default: return;  // StartLine: prepared, nothing else
    }
}

// The substep's hits in collision sort order (R3.3: FV order kept).
TW_HD inline void TwRaceOnTriggerHits(const TwRaceMap &m, TwinState &s, const TwTriggerHit *hits, std::uint32_t count,
                                TwRaceEvents &ev) {
    for (std::uint32_t i = 0; i < count; ++i) TwRaceOnTriggerContact(m, s, hits[i].trigger, ev);
}

// ---------------------------------------------------------------------------
// Sub-tick finish bracket (R4.2; FV replay_finish_time_estimator.cpp:14-101,
// ported loop). `probe(dtSeconds)` reruns the completing substep from its
// start state with the shortened step and returns whether the race completed
// (M5 supplies it). Output (lower, upper] in ns, origin = tick start; FV
// publishes estimated = upper.
struct TwFinishSubstep {
    std::uint32_t tickMs;           // tick time T
    double substepStartSeconds;     // offset of the completing substep inside the tick
    float substepDurationSeconds;   // its full duration
};

namespace detail {
// floor / ceil for 0 <= x < 2^53 without libm (the tick path links no libm,
// design.md §7); exact in that range, so the bracket equals FV's.
TW_HD inline std::uint64_t FloorNs(double x) { return static_cast<std::uint64_t>(x); }
TW_HD inline std::uint64_t CeilNs(double x) {
    const std::uint64_t f = static_cast<std::uint64_t>(x);
    return f + (x > static_cast<double>(f) ? 1u : 0u);
}
}  // namespace detail

template <class Probe>
TW_HD inline bool TwRefineFinish(const TwFinishSubstep &sub, Probe &&probe, std::uint64_t *lowerNs, std::uint64_t *upperNs) {
    constexpr double kNsPerS = 1000000000.0;
    if (!std::isfinite(sub.substepStartSeconds) || sub.substepStartSeconds < 0.0 ||
        !std::isfinite(sub.substepDurationSeconds) || !(sub.substepDurationSeconds > 0.0f))
        return false;
    const std::uint64_t tickStartNs = static_cast<std::uint64_t>(sub.tickMs) * 1000000u;
    double lower = static_cast<double>(tickStartNs) + sub.substepStartSeconds * kNsPerS;
    double upper = lower + static_cast<double>(sub.substepDurationSeconds) * kNsPerS;
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower < 0.0 || !(lower < upper) ||
        upper > static_cast<double>(std::numeric_limits<std::uint64_t>::max() - 1u) ||
        upper >= 9007199254740992.0)  // 2^53: the cast-based floor/ceil below is exact below it
        return false;
    if (!probe(sub.substepDurationSeconds)) return false;
    const double startNs = lower;
    for (;;) {
        const std::uint64_t firstInterior = detail::FloorNs(lower) + 1u;
        const std::uint64_t upperCeil = detail::CeilNs(upper);
        if (upperCeil == 0u || firstInterior >= upperCeil) break;
        const std::uint64_t lastInterior = upperCeil - 1u;
        const std::uint64_t candidate = firstInterior + (lastInterior - firstInterior) / 2u;
        const double partial = (static_cast<double>(candidate) - startNs) / kNsPerS;
        if (partial <= 0.0) {
            lower = static_cast<double>(candidate);
            continue;
        }
        if (probe(static_cast<float>(partial)))
            upper = static_cast<double>(candidate);
        else
            lower = static_cast<double>(candidate);
    }
    *lowerNs = detail::FloorNs(lower);
    *upperNs = detail::CeilNs(upper);
    return *lowerNs < *upperNs;  // FinishTimeEstimate::IsValid
}

// The state keeps the lower bound (the M1 codec stores {present, lower}; FV's
// brackets are 1 ns wide, upper = estimated = lower + 1).
TW_HD inline void TwStoreFinish(TwinState &s, std::uint64_t lowerNs) {
    s.finishPresent = 1u;
    s.finishLowerNs[0] = static_cast<std::uint32_t>(lowerNs);
    s.finishLowerNs[1] = static_cast<std::uint32_t>(lowerNs >> 32);
}

}  // namespace race
}  // namespace tw

#endif
