// Twin API internals shared by the API, the FV adapters (src/twin/fv/) and
// the harness engine (tools/twin/twin_engine_twin.cpp). Not a public header:
// it needs the twin include directories (src/twin/CMakeLists.txt TW_INCLUDE_DIRS).
#ifndef FOREVERVALIDATOR_TWIN_API_INTERNAL_H
#define FOREVERVALIDATOR_TWIN_API_INTERNAL_H

#include <forevervalidator/experimental/twin.h>

#include <memory>

#include "tw_controls.h"
#include "tw_isa.h"
#include "tw_state.h"
#include "tw_step.h"

namespace tw::fv {
struct TwinWorldPacks;  // src/twin/fv/twin_world_packs.h
}

namespace forevervalidator::experimental::twin {

struct TwinWorld::Impl {
    TwinWorldInfo info;
    tw::step::Model model;            // points into `packs`
    tw::race::TwTimeline timeline;    // FV sandbox defaults: 10 ms ticks, 2600 ms prestart
    // World package (M2), CarForceConsts + WaterZone (M4), TwRaceMap, import
    // context; built by tw::fv::BuildTwinWorld (src/twin/fv/twin_world_build.h).
    // Null for a world built without FV (tests): the engine then reports
    // the missing packs.
    std::shared_ptr<const tw::fv::TwinWorldPacks> packs;
    // Runtime ISA dispatch (isa/tw_isa.h): the variant's own tick instance
    // over `packs`, chosen in TwinWorldFactory::Make. Null = the baseline
    // `model` above.
    const tw::isa::TickTable *isa = nullptr;
    std::shared_ptr<void> isaInstance;
};

// Builds a TwinWorld from its parts: CompileTwinWorld, the FV adapters and
// tests (a world without packs runs the engine's argument checks only).
struct TwinWorldFactory {
    static std::shared_ptr<const TwinWorld> Make(std::unique_ptr<TwinWorld::Impl> impl);
};

namespace internal {

// The world's tick, through its ISA variant when it has one. Every twin
// stepping path (TwinStep, TwinStepTicks, the batch runner, the validation
// timeline) goes through these.
// `cache`: from NewCollisionCache(world) for this candidate (or null).
tw::step::Status Step(const TwinWorld::Impl &world, tw::model::TwinState &state, tw::step::Control control,
                      tw::step::Events *events, tw::step::CollisionCache *cache = nullptr);
std::uint32_t StepTicks(const TwinWorld::Impl &world, tw::model::TwinState &state, const tw::step::Control *controls,
                        std::size_t count, tw::step::StopRule stop, tw::step::Events *events,
                        const std::uint32_t *respawnCounts, tw::step::CollisionCache *cache = nullptr);

// A per-candidate collision cache of the world's stage, from the variant the
// world runs (null when the stage has none). Keep one per candidate; it
// need not be flushed when the candidate's state is restored.
using CollisionCachePtr = std::unique_ptr<tw::step::CollisionCache, void (*)(tw::step::CollisionCache *)>;
CollisionCachePtr NewCollisionCache(const TwinWorld::Impl &world);

tw::model::TwinState ToModel(const TwinState &state) noexcept;
TwinState FromModel(const tw::model::TwinState &state) noexcept;
tw::step::Policy ToStepPolicy(TwinPolicy policy) noexcept;
bool IsTwinScope(MapEnvironment environment, VehicleModel vehicle, std::optional<PlayMode> playMode) noexcept;

}  // namespace internal

}  // namespace forevervalidator::experimental::twin

#endif
