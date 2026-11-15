// Twin FV adapter: an exact FV runtime state as a TwinState (design.md §8.2,
// m6a-race.md; semantics owned by architect A).
//
// Recipe (A, verified lossless by M1 on 45,144 states): FV's runtime clone
// -> harness state words (tools/twin/twin_fv_state.cpp ExtractState) ->
// tw::model::ImportGolden with the world's context (the car's fs/lds pairs,
// the world package's spawn table, requiredCps/laps). ImportGolden's refusals
// (UnknownSpawn, UnknownFsLds, HighSlots, NonZeroAccumulator,
// MapConstantMismatch, ...) are hard errors.
//
// Hand-off: FV runs the prestart through T = 2600 (race spawn, begin race,
// race-start reset included); the imported state is the one after T = 2600
// and the twin's first tick is T = 2610. The seed field (no FV counterpart,
// informative only) takes the validation seed.
#ifndef TW_FV_IMPORT_H
#define TW_FV_IMPORT_H

#include <cstdint>
#include <string>

#include "tw_state.h"
#include "tw_state_golden.h"

struct ReplaySimulationInstanceClone;

namespace forevervalidator::experimental::twin {
class TwinWorld;
}

namespace tw::fv {

bool ImportFvClone(const ReplaySimulationInstanceClone &clone, const tw::model::TwGoldenContext &context,
                   std::uint32_t seed, tw::model::TwinState *out, std::string *diag);

// Same, with the context of a compiled TwinWorld (fv/twin_world_build.h).
bool ImportFvClone(const forevervalidator::experimental::twin::TwinWorld &world,
                   const ReplaySimulationInstanceClone &clone, std::uint32_t seed, tw::model::TwinState *out,
                   std::string *diag);

}  // namespace tw::fv

#endif
