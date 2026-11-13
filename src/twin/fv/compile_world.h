// Twin FV adapter (M2): the immutable world package (world/tw_world_pack.h)
// from FV's flat scene export (BuildCudaHostScene, called in place).
// Only src/twin/fv/ includes FV headers (design.md §8.3).
#ifndef TW_FV_COMPILE_WORLD_H
#define TW_FV_COMPILE_WORLD_H

#include <cstdint>
#include <string>

#include "engine/core/gm_types.h"
#include "simulation/backends/cuda/cuda_scene_layout.h"
#include "tw_world_pack.h"

class StaticSceneModelCollection;

namespace forevervalidator::simulation {
struct CudaHostStaticConfiguration;
}

namespace tw::fv {

struct WorldCompileInputs {
    // FV's flat scene (BuildCudaHostScene of the session's static models). Required.
    const forevervalidator::simulation::CudaHostScene *scene = nullptr;
    // The race start spawn, unseeded (board 30 R1.6): FV's initial player
    // spawn, race.player previousSpawnLocation at load. Required.
    const GmIso4 *startSpawn = nullptr;
    // Session lap rule (board 30 R3.8): isLapRace ? map laps : 1.
    std::uint32_t requiredLaps = 1u;
    bool shortcutMode = false;   // EChallengePlayMode::Shortcut (R3.9); not Stadium Race
    // Keep the world's enclosing shell (kActorWorldEnclosure: an Environment
    // actor whose every triangle stays more than 1 km from the rest of the
    // collision set, and whose box contains it). Off by default: the car
    // cannot reach it and its 28 km triangles spoil spatial indexes (user
    // decision, 2026-11-08; m2-world.md §2).
    bool includeWorldEnclosure = false;
    // Leave out the moving-pair corpora (kInstMovingPair: Stadium's "Flags",
    // 24 flag meshes, 5,808 triangles at y 200-229 m outside the map area).
    // Off by default: FV collides with them. For the cost comparison the
    // user asked for (kept unless leaving them out is significantly faster).
    bool excludeMovingPairCorpora = false;
    // FV's static configuration (BuildCudaHostStaticConfiguration of the
    // session definition). Optional: supplies the water grid; without it the
    // package's water grid is marked unknown (WaterGrid::known = 0).
    const forevervalidator::simulation::CudaHostStaticConfiguration *config = nullptr;
};

// Builds and seals the package. Refuses (false, reason in `diag`) what the
// twin does not model, the same set FV's CUDA backend refuses at the scene:
// non-mesh collision surfaces in target groups 1/3/4, group-3 static records
// (board 20 §2.2), race triggers outside group 1, checkpoint identities on
// non-trigger actors, mesh octrees that do not hold each triangle once, and
// the race-map limits (TwRaceMapValid: <= 63 checkpoints).
bool CompileWorldPack(const WorldCompileInputs &in, tw::world::WorldPack *out, std::string *diag);

// Limits for BuildCudaHostScene when the scene feeds the twin. FV's defaults
// are the CUDA device budget (4,194,304 vertices and triangles, 65,536
// actors), which the largest Stadium scenery maps exceed (tmnf-7713641).
// The twin's own limits are what the package can index: actors < 2^22 (the
// Tri meta actor field), < 2^24 triangles per mesh (ModelTri octree rank),
// u32 everywhere else.
forevervalidator::simulation::CudaSceneBuildLimits TwinSceneBuildLimits();

// Same, from the static models (calls BuildCudaHostScene with TwinSceneBuildLimits first).
bool CompileWorldPackFromModels(const StaticSceneModelCollection &models, WorldCompileInputs in,
                                tw::world::WorldPack *out, std::string *diag);

}  // namespace tw::fv

#endif
