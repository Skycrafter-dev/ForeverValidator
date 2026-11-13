// Twin acceptance harness: FV runtime state <-> flat golden-record words.
//
// FV-dependent. Maps the fields of ReplaySimulationInstanceClone (the
// complete per-tick runtime clone the sandbox captures) to named fields of
// 32-bit words, and back. The field list is the twin's view of board 10 §5
// (tier State), the race state of board 30 §10 (tier Race), extra FV state
// that is not in §5 (tier Extra) and outputs (tier Output).
#ifndef FOREVERVALIDATOR_TWIN_FV_STATE_H
#define FOREVERVALIDATOR_TWIN_FV_STATE_H

#include <cstdint>
#include <string>
#include <vector>

#include <forevervalidator/experimental/physics_sandbox.h>

#include "simulation/runtime/replay_simulation_session.h"
#include "src/twin/common/twin_golden.h"

namespace twin_fv {

enum class Tier : std::uint8_t {
    State = 0,   // board 10 §5 minimal state (the twin's state vector)
    Race = 1,    // board 30 §10 race state carried across ticks
    Extra = 2,   // FV state not listed in board §5 (may or may not matter)
    Output = 3,  // rebuilt every substep or output only
};

inline const char *TierName(Tier tier) {
    switch (tier) {
    case Tier::State: return "state";
    case Tier::Race: return "race";
    case Tier::Extra: return "extra";
    case Tier::Output: return "output";
    }
    return "?";
}

using Clone = ReplaySimulationInstanceClone;

struct StateField {
    const char *name;
    twin::FieldType type;
    std::uint16_t count;
    Tier tier;
    void (*get)(const Clone &, std::uint32_t *);
    void (*set)(Clone &, const std::uint32_t *);
    const char *fvPath;
};

const std::vector<StateField> &StateFields();
std::uint32_t StateWordCount();

// Writes every state field, in table order, to `out` (StateWordCount words).
void ExtractState(const Clone &clone, std::uint32_t *out);

// Overwrites the fields whose tier bit is in `tierMask` (1 << Tier) from
// `words` (laid out as ExtractState writes them).
void ApplyState(Clone &clone, const std::uint32_t *words,
                std::uint32_t tierMask);

// Same, with explicit per-field additions and removals by name.
void ApplyState(Clone &clone, const std::uint32_t *words, std::uint32_t tierMask,
                const std::vector<std::string> &with,
                const std::vector<std::string> &without);

constexpr std::uint32_t TierBit(Tier tier) {
    return 1u << static_cast<std::uint32_t>(tier);
}

// --- sandbox state access -------------------------------------------------
// Through twin_test::PhysicsSandboxTwinTestAccess (a friend of
// PhysicsSandboxState). PeekClone's pointer lives as long as `state`.
// ReplaceClone gives `state` a new private copy with the clone replaced
// (cursor, inputs and compatibility keys kept), for RestoreState.
const Clone *PeekClone(const forevervalidator::experimental::PhysicsSandboxState &state,
                       std::string *error);
bool ReplaceClone(forevervalidator::experimental::PhysicsSandboxState &state,
                  const Clone &clone, std::string *error);

// FV's substep count rule (physics_step.cpp ComputeSubstepCount) from the
// tick-start state; `metric` is (|v|+|w|)*dt/maxStep before truncation.
std::uint32_t SubstepCount(const Clone &clone, float *metric);

}  // namespace twin_fv

#endif
