// Twin tick engine (tw_step.h): the all-twin tick (tick.h + twin_stages.h)
// once every pack is present, NotImplemented before.
#include "tw_step.h"

#include <cstring>

#include "policy.h"
#include "stages.h"
#include "tick.h"
#include "tw_fpenv.h"
#include "twin_stages.h"

namespace tw {
namespace step {

namespace {

// Bumped whenever a change alters any golden hash (design.md §7). "stub"
// marks builds whose Step() does nothing: the tick (step/tick.h) is wired but
// returns NotImplemented until the packs it needs exist.
constexpr const char *kModelId = "twin-stub-0";
// Info() has no model: once CompileTwinWorld builds car, water and race map,
// collision is the only missing piece.
constexpr const char *kMissing = "collision (M3: Model::collision, a step::CollisionStage over the world pack)";

#if defined(__GNUC__) || defined(__clang__)
#define TW_STEP_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define TW_STEP_NOINLINE __declspec(noinline)
#else
#define TW_STEP_NOINLINE
#endif

// The guarded kernel: a separate non-inlined function so the compiler cannot
// move FP work across the mode switch (math.md §4).
PolicyConfig ConfigOf(const Model &m) {
    if (m.policyConfig != nullptr) return *m.policyConfig;
    switch (m.policy) {
    case Policy::FvSubsteps: return PolicyConfig::Fv();
    case Policy::CappedSubsteps3: return PolicyConfig::Cap(3u);
    case Policy::ContactSubsteps: return PolicyConfig::Recommended();
    case Policy::ContactSubstepsFull: {
        // spec1n0x1b1 (design.md rev 5 §5.1): the speculative substep (L7); airborne
        // 1-substep ticks run forces + integration FV's n times (L8), other 1-substep
        // ticks get the ballistic correction (b1) (substep-policy.md).
        PolicyConfig p;
        if (!ParsePolicy("spec1n0x1b1", &p)) return PolicyConfig::Fv();
        return p;
    }
    }
    return PolicyConfig::Fv();
}

}  // namespace

PolicyConfig ModelPolicyConfig(const Model &m) { return ConfigOf(m); }

namespace {

// `respawns`: exact respawn count, or ~0u = decode the control's field.
TW_STEP_NOINLINE Status StepKernel(const Model &m, model::TwinState &state, Control control,
                                   std::uint32_t respawns, Events *events, CollisionCache *cache) {
    TwinPacks packs;
    packs.car = m.car;
    packs.water = m.water;
    packs.collision = m.collision;
    packs.race = m.race;
    packs.predictor = m.predictor;
    packs.policy = ConfigOf(m);
    if (!TwinPacksComplete(packs)) return Status::NotImplemented;
    TwinStages stages(packs, cache);
    TickContext t;
    t.respawnCount = respawns != ~0u ? respawns : race::TwCtlRespawnField(control);
    RunTick(stages, state, control, t);
    if (events != nullptr) {
        events->substeps += t.substepsRun;
        events->respawns += t.respawned;
        events->checkpoints += t.checkpoints;
        events->cpLastSlot = t.cpLastSlot;           // of this (the last) tick
        events->finishPassage = t.finishPassage;
        events->wheelContacts = t.wheelContacts;
        events->bodyContacts = t.bodyContacts;
        if (t.finished && !events->finished) {
            events->finished = true;
            events->finishTickMs = t.finishTickMs;
            events->finishLowerNs = state.finishPresent
                    ? (static_cast<std::uint64_t>(state.finishLowerNs[1]) << 32) | state.finishLowerNs[0]
                    : 0u;
        }
    }
    return Status::Ok;
}

}  // namespace

const char *PolicyName(Policy policy) {
    switch (policy) {
    case Policy::FvSubsteps: return "fv";
    case Policy::CappedSubsteps3: return "cap3";
    case Policy::ContactSubsteps: return "contact";
    case Policy::ContactSubstepsFull: return "contact-full";
    }
    return "unknown";
}

const char *StatusName(Status status) {
    switch (status) {
    case Status::Ok: return "ok";
    case Status::NotImplemented: return "not_implemented";
    case Status::InvalidArgument: return "invalid_argument";
    case Status::Stopped: return "stopped";
    }
    return "unknown";
}

EngineInfo Info() {
    EngineInfo info;
    info.stepImplemented = false;
    info.modelId = kModelId;
    info.missing = kMissing;
    return info;
}

Status Step(const Model &model, model::TwinState &state, Control control, Events *events, CollisionCache *cache) {
    Status status;
    {
        math::FpGuard guard;
        status = StepKernel(model, state, control, ~0u, events, cache);
    }
    if (events != nullptr) {
        events->status = status;
        if (status == Status::Ok) ++events->ticks;
    }
    return status;
}

std::uint32_t StepTicks(const Model &model, model::TwinState &state, const Control *controls,
                        std::size_t count, StopRule stop, Events *events, const std::uint32_t *respawnCounts,
                        CollisionCache *cache) {
    Events local;
    Events &ev = events != nullptr ? *events : local;
    ev.status = Status::Ok;
    if (controls == nullptr && count != 0u) {
        ev.status = Status::InvalidArgument;
        return 0u;
    }
    std::uint32_t done = 0u;
    math::FpGuard guard;
    for (std::size_t i = 0; i < count; ++i) {
        const Status status =
                StepKernel(model, state, controls[i], respawnCounts != nullptr ? respawnCounts[i] : ~0u, &ev, cache);
        if (status != Status::Ok) {
            ev.status = status;
            return done;
        }
        ++done;
        ++ev.ticks;
        if (stop.atFinish && ev.finished) {
            ev.status = i + 1u < count ? Status::Stopped : Status::Ok;
            return done;
        }
    }
    return done;
}

std::uint64_t CanonicalHash(const model::TwinState &state) {
    model::TwinState copy;
    std::memcpy(&copy, &state, sizeof(copy));
    std::size_t n = 0;
    const model::TwFieldInfo *fields = model::TwinFields(&n);
    unsigned char *base = reinterpret_cast<unsigned char *>(&copy);
    for (std::size_t i = 0; i < n; ++i) {
        if (fields[i].type != model::TwType::F32) continue;
        for (std::uint16_t k = 0; k < fields[i].count; ++k) {
            unsigned char *p = base + fields[i].offset + 4u * k;
            std::uint32_t bits;
            std::memcpy(&bits, p, 4);
            if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
                bits = 0x7fc00000u;
                std::memcpy(p, &bits, 4);
            }
        }
    }
    return model::TwinStateHash(copy);
}

}  // namespace step
}  // namespace tw
