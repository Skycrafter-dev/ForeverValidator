// The immutable packs a TwinWorld owns (FV-free; built by
// fv/twin_world_build.cpp from an FV session, read by api/ and step/).
//
//   world   M2 world package (world/tw_world_pack.h) and its view
//   car     CarForceConsts (fv/compile_car.cpp, M4)
//   water   WaterZone over `waterCells` (fv/compile_car.cpp CompileWaterZone)
//   race    TwRaceMap pointing into the world package (race/tw_race.h)
//   collision  M3 ExactCollision over the world package + CarShapes (fv/compile_car_shapes.cpp)
//   golden  the import context (model/tw_state_golden.h): the car's fs/lds
//           pairs and the spawn table's iso words (spawnIsos)
//
// Everything points into this object: keep it alive (TwinWorld::Impl holds
// it by shared_ptr) and never copy it.
#ifndef TW_FV_TWIN_WORLD_PACKS_H
#define TW_FV_TWIN_WORLD_PACKS_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "car_force_consts.h"
#include "atlas_stage.h"
#include "cc_stage.h"
#include "tw_collide.h"
#include "tw_race.h"
#include "tw_state_golden.h"
#include "tw_world_pack.h"

namespace tw::fv {

struct TwinWorldPacks {
    TwinWorldPacks() = default;
    TwinWorldPacks(const TwinWorldPacks &) = delete;
    TwinWorldPacks &operator=(const TwinWorldPacks &) = delete;

    tw::world::WorldPack pack;
    tw::world::WorldPackView view;
    tw::model::CarForceConsts car{};
    tw::model::WaterZone water{};
    std::vector<std::uint8_t> waterCells;
    tw::race::TwRaceMap race{};
    std::vector<std::array<std::uint32_t, 12>> spawnIsos;
    tw::model::TwGoldenContext golden{};
    // M3 collision stage over `view` (collide/tw_collide.h) and the car's ellipsoids
    tw::collide::CarShapes shapes{};
    tw::collide::ExactCollision collision;
    // Architect C's stage over the same view and shapes: the product default ("c", noise-level narrow
    // phase), or a C experiment (c-exact, ...); B's `collision` above is the reference ("b"). See Stage()
    // and isa/tw_collision_choice.h.
    tw::collide_c::CcCollision collisionC;
    bool useC = false;
    std::string collisionName;   // the stage's name (tw_collision_choice.h)
    std::unique_ptr<tw::atlas::AtlasCollision> collisionAtlas;   // E's atlas stage ("atlas", gen2 S1)
    const tw::step::CollisionStage *Stage() const {
        if (collisionAtlas) return collisionAtlas.get();
        return useC ? static_cast<const tw::step::CollisionStage *>(&collisionC) : static_cast<const tw::step::CollisionStage *>(&collision);
    }
    // compile cost (informative)
    double sceneMs = 0.0, worldMs = 0.0, carMs = 0.0;

    const tw::model::WaterZone *Water() const { return water.enabled ? &water : nullptr; }
};

}  // namespace tw::fv

#endif
