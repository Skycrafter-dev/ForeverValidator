// Twin FV adapter: a TwinWorld from an FV session (design.md §8.2).
//
// Compiles, from FV's own exports called in place:
//   - the world package (fv/compile_world.cpp, from BuildCudaHostScene of the
//     session's static models; the world enclosure left out by default);
//   - the car pack and the water zone (fv/compile_car.cpp, from
//     BuildCudaHostStaticConfiguration of the simulation definition);
//   - the race map (the package's TwRaceMap tables) and the import context.
// The tick model (step::Model) points into these packs, its collision stage
// included (TwinWorldSource::collision, isa/tw_collision_choice.h).
#ifndef TW_FV_TWIN_WORLD_BUILD_H
#define TW_FV_TWIN_WORLD_BUILD_H

#include <cstdint>
#include <memory>
#include <string>

#include <forevervalidator/experimental/twin.h>

#include "engine/core/gm_types.h"
#include "twin_world_packs.h"

class StaticSceneModelCollection;
class ReplaySimulationSession;
struct ReplaySimulationDefinition;
struct ReplaySimulationInstanceClone;

namespace tw::fv {

struct TwinWorldSource {
    const StaticSceneModelCollection *models = nullptr;     // the session's installed static models
    const ReplaySimulationDefinition *definition = nullptr;  // car tuning, water
    GmIso4 startSpawn{};                                     // race start spawn (unseeded)
    std::uint32_t requiredLaps = 1u;                         // clone race.progress.requiredLapCount
    bool shortcutMode = false;
    bool includeWorldEnclosure = false;                      // WorldCompileInputs::includeWorldEnclosure
    bool excludeMovingPairCorpora = false;                   // WorldCompileInputs::excludeMovingPairCorpora
    // Collision stage: Noise (C's, the product default) or Exact (B's, the reference);
    // FOREVERVALIDATOR_TWIN_COLLIDE overrides it (isa/tw_collision_choice.h).
    forevervalidator::TwinCollision collision = forevervalidator::TwinCollision::Noise;
};

// Fills `src` from a session, its definition and a runtime clone of it (the
// laps come from the clone; the start spawn from the session's map scene).
bool TwinWorldSourceFromSession(const ReplaySimulationSession &session, const ReplaySimulationDefinition &definition,
                                const ReplaySimulationInstanceClone &clone, TwinWorldSource *src, std::string *diag);

bool BuildTwinWorldPacks(const TwinWorldSource &src, TwinWorldPacks *out, std::string *diag);

struct TwinWorldLabel {
    forevervalidator::MapEnvironment mapEnvironment = forevervalidator::MapEnvironment::Stadium;
    forevervalidator::VehicleModel vehicleModel = forevervalidator::VehicleModel::StadiumCar;
    std::string mapName;
};

// The packs plus the tick model; null on failure (reason in `diag`).
std::shared_ptr<const forevervalidator::experimental::twin::TwinWorld> BuildTwinWorld(
        const TwinWorldSource &src, forevervalidator::TwinPolicy policy, const TwinWorldLabel &label, std::string *diag);

// The public API's routes (src/twin/api/twin_api.cpp): from a loaded sandbox
// (its session, definition and current state), and a sandbox state imported
// into a compiled world (seed = 0: the field is informative only).
std::shared_ptr<const forevervalidator::experimental::twin::TwinWorld> BuildTwinWorldFromSandbox(
        const forevervalidator::experimental::PhysicsSandbox &loaded, forevervalidator::TwinPolicy policy,
        forevervalidator::TwinCollision collision, const TwinWorldLabel &label, std::string *diag);
bool ImportSandboxState(const forevervalidator::experimental::twin::TwinWorld &world,
                        const forevervalidator::experimental::PhysicsSandboxState &state, tw::model::TwinState *out,
                        std::string *diag);

}  // namespace tw::fv

#endif
