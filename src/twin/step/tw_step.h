// Twin tick engine entry points (design.md §5, §8.1). FV-independent.
//
// This is the seam between the public API (include/forevervalidator/
// experimental/twin.h, src/twin/api/) and the kernels (model/, world/,
// race/, later collide/ and step/substep.h). Until M5 lands the substep
// loop, Step() is a stub that reports Status::NotImplemented and leaves the
// state untouched; Info() says which pieces are missing. Callers, the FV
// backend adapter, the harness engine and the golden-hash suite are written
// against this interface now.
//
// Determinism: every entry point that runs physics sets the canonical FP
// mode (tw::math::FpGuard) and calls a noinline kernel (math.md §4).
#ifndef TW_STEP_H
#define TW_STEP_H

#include <cstddef>
#include <cstdint>

#include "tw_state.h"

namespace tw {
struct World;
namespace model {
struct CarForceConsts;
struct WaterZone;
}
namespace race {
struct TwRaceMap;
}

namespace step {

class CollisionStage;     // stages.h
class CollisionCache;     // stages.h
class ContactPredictor;   // policy.h
struct PolicyConfig;      // policy.h

using Control = std::uint32_t;  // design.md §3.3 (decoders: race/tw_race.h)

// Part of the model id: each policy has its own golden hashes (design.md §7).
enum class Policy : std::uint8_t {
    FvSubsteps = 0,       // Tier 1: FV's speed-based substep count
    CappedSubsteps3 = 1,  // Tier 2 fallback: at most 3 substeps per tick (M11)
    ContactSubsteps = 2,  // Tier 2 plan of record, "contact" = contact1d0e3w900n0 (substep-policy.md, policy.h
                          // PolicyConfig::Recommended()): 1 substep on a contact-free tick, min(n, 3) when the
                          // tick-start pose touches anything moving along its normal or the contact set changes
                          // by dt, FV's n on those ticks at >= 900 km/h; needs the predictor. Opt-in fast
                          // policy (e.g. search): it changes trick behaviour (63/82 board-40 measures kept,
                          // fails the corpus roulette G3)
    ContactSubstepsFull = 3,  // Tier 2 plan of record, "contact-full" = spec1n0x1b1 (design.md rev 5 §5.1):
                              // event = a tick-start contact moving along its normal, or the contact key
                              // (shape, material, trigger) changing over one speculative substep (L7: run it,
                              // compare, restore the snapshot and rerun on a change); no event -> 1 substep
                              // (airborne: forces + integration FV's n times, L8; otherwise the ballistic
                              // correction pos += F/m dt^2 (n-1)/(2n)); event -> FV's n;
                              // passes M-H2, tricks and corpus G3
};

enum class Status : std::uint8_t {
    Ok = 0,
    NotImplemented = 1,   // the tick kernel (M5) is not built yet
    InvalidArgument = 2,  // null pointer, missing pack the kernel needs
    Stopped = 3,          // StepTicks: the stop rule ended the run early (not an error)
};

const char *PolicyName(Policy policy);
const char *StatusName(Status status);

// The immutable per-map, per-car inputs of the kernel. Packs that are not
// compiled yet stay null; Step() reports InvalidArgument once it needs them.
struct Model {
    Policy policy = Policy::FvSubsteps;
    const model::CarForceConsts *car = nullptr;   // fv/compile_car.cpp (M4)
    const model::WaterZone *water = nullptr;      // fv/compile_car.cpp (M4)
    const World *world = nullptr;                 // fv/compile_world.cpp (M2)
    const race::TwRaceMap *race = nullptr;        // race layer (M6a/M6b)
    const CollisionStage *collision = nullptr;    // detection over `world` (collide/, M3; stages.h)
    const ContactPredictor *predictor = nullptr;  // ContactSubsteps: contact-change predictor (policy.h)
    const PolicyConfig *policyConfig = nullptr;   // optional: an explicit policy (policy.h grammar) instead of `policy`
};

// Per-tick events (accumulated over StepTicks).
struct Events {
    Status status = Status::Ok;
    std::uint32_t ticks = 0u;          // ticks executed
    std::uint32_t substeps = 0u;
    std::uint32_t respawns = 0u;
    std::uint32_t checkpoints = 0u;
    bool finished = false;
    std::uint32_t finishTickMs = 0u;   // tick of the finish (finished only)
    std::uint64_t finishLowerNs = 0u;  // finish bracket lower bound (race.finish_ns, TwStoreFinish);
                                       // orders ties at the finish tick (board 30 R4.2)
    std::uint32_t cpLastSlot = 0xFFFFFFFFu;  // slot of the last accepted passage of the last tick
                                             // (TwRaceEvents::cpLastSlot; 0xFFFFFFFF = none)
    std::uint32_t wheelContacts = 0u;  // the last tick's wheel / body contact counts (RespTick; FV's
    std::uint32_t bodyContacts = 0u;   // AfterContacts snapshot: fr.has_wheel_contact / fr.has_body_contact)
    bool finishPassage = false;        // the last tick accepted a finish passage (lap end or race end;
                                       // TwRaceEvents::finish), unlike `finished` (race completed)
};

struct StopRule {
    bool atFinish = true;              // stop after the tick that finishes the race
};

// What the engine can do in this build.
struct EngineInfo {
    bool stepImplemented = false;
    const char *modelId = "";          // changes whenever the goldens change
    const char *missing = "";          // human-readable list of missing pieces
};

EngineInfo Info();

// The substep policy a model runs (Model::policyConfig, else Model::policy's preset).
PolicyConfig ModelPolicyConfig(const Model &model);

// One 10 ms tick (tick time = state.tickMs + 10). On any status other than Ok
// the state is unchanged. `cache` (nullable): the candidate's collision cache
// (model.collision->NewCache(), stages.h), kept by the caller across ticks;
// the result does not depend on it, so a restore or respawn needs no flush.
Status Step(const Model &model, model::TwinState &state, Control control, Events *events = nullptr,
            CollisionCache *cache = nullptr);

// Up to `count` ticks; returns the number executed. Stops early on a
// non-Ok status (reported in events->status) or when the stop rule fires
// (events->status = Stopped). `respawnCounts` (nullable, one per tick): the
// exact respawn count of each tick (TwControlStream::Respawns); without it
// the control's 8-bit field is used (saturated at 255 on escape ticks).
std::uint32_t StepTicks(const Model &model, model::TwinState &state, const Control *controls,
                        std::size_t count, StopRule stop, Events *events = nullptr,
                        const std::uint32_t *respawnCounts = nullptr, CollisionCache *cache = nullptr);

// TwinStateHash with every NaN canonicalised to 0x7fc00000 first (math.md §2:
// NaN payloads are outside the determinism contract).
std::uint64_t CanonicalHash(const model::TwinState &state);

}  // namespace step
}  // namespace tw

#endif
