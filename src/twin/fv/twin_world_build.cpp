// Twin FV adapter: TwinWorld from an FV session (twin_world_build.h).
#include "twin_world_build.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>

#include "compile_car.h"
#include "compile_car_shapes.h"
#include "compile_world.h"
#include "engine/scene/static_scene_model.h"
#include "simulation/backends/cuda/cuda_scene_layout.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/runtime/replay_simulation_session.h"
#include "import.h"
#include "twin_api_internal.h"
#include "tw_collision_choice.h"
#include "validation/api/physics_sandbox_twin_test_access.h"

namespace tw::fv {

namespace twin = forevervalidator::experimental::twin;
namespace sim = forevervalidator::simulation;

namespace {
double MsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
std::uint32_t Bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
}  // namespace

bool TwinWorldSourceFromSession(const ReplaySimulationSession &session, const ReplaySimulationDefinition &definition,
                                const ReplaySimulationInstanceClone &clone, TwinWorldSource *src, std::string *diag) {
    TwinWorldSource s;
    s.models = &session.StaticModels();
    s.definition = &definition;
    if (!session.StartSpawnLocation(s.startSpawn)) {
        if (diag) *diag = "twin world: the map has no start line spawn";
        return false;
    }
    s.requiredLaps = clone.race.progress.requiredLapCount;
    *src = s;
    return true;
}

bool BuildTwinWorldPacks(const TwinWorldSource &src, TwinWorldPacks *out, std::string *diag) {
    if (src.models == nullptr || src.definition == nullptr || out == nullptr) {
        if (diag) *diag = "twin world: missing static models or simulation definition";
        return false;
    }
    // car pack + water (fv/compile_car.cpp)
    auto t0 = std::chrono::steady_clock::now();
    auto cfg = std::make_unique<sim::CudaHostStaticConfiguration>();
    const auto cr = sim::BuildCudaHostStaticConfiguration(*src.definition, cfg.get());
    if (cr != sim::CudaStaticConfigurationBuildResult::Success) {
        if (diag) *diag = "twin car: BuildCudaHostStaticConfiguration failed (" + std::to_string(static_cast<unsigned>(cr)) + ")";
        return false;
    }
    std::string why;
    if (!CompileCarForceConsts(*cfg, &out->car, &why)) {
        if (diag) *diag = "twin car: " + why;
        return false;
    }
    if (!CompileWaterZone(*cfg, &out->water, &out->waterCells, &why)) {
        if (diag) *diag = "twin water: " + why;
        return false;
    }
    out->water.cells = out->waterCells.empty() ? nullptr : out->waterCells.data();
    out->carMs = MsSince(t0);

    // world package (fv/compile_world.cpp)
    t0 = std::chrono::steady_clock::now();
    auto scene = std::make_unique<sim::CudaHostScene>();
    const auto sr = sim::BuildCudaHostScene(*src.models, scene.get(), TwinSceneBuildLimits());
    if (sr != sim::CudaSceneBuildResult::Success) {
        if (diag) *diag = std::string("twin world: BuildCudaHostScene failed: ") + sim::CudaSceneBuildResultName(sr);
        return false;
    }
    out->sceneMs = MsSince(t0);
    t0 = std::chrono::steady_clock::now();
    WorldCompileInputs in;
    in.scene = scene.get();
    in.startSpawn = &src.startSpawn;
    in.requiredLaps = src.requiredLaps;
    in.shortcutMode = src.shortcutMode;
    in.config = cfg.get();
    in.includeWorldEnclosure = src.includeWorldEnclosure;
    in.excludeMovingPairCorpora = src.excludeMovingPairCorpora;
    if (!CompileWorldPack(in, &out->pack, diag)) return false;
    if (!tw::world::OpenWorldPack(out->pack, &out->view, &why, /*skipHash=*/true)) {
        if (diag) *diag = "twin world: " + why;
        return false;
    }
    // M3 collision stage: the car's ellipsoids + a BVH4 over the package (collide/tw_collide.h)
    if (!CompileCarShapes(*cfg, &out->shapes, &why)) {
        if (diag) *diag = "twin car shapes: " + why;
        return false;
    }
    // The collision stage: src.collision (c = C's noise stage, the product default; b = B's exact
    // stage, the reference), or FOREVERVALIDATOR_TWIN_COLLIDE's experiment (isa/tw_collision_choice.h).
    // B's stage is built in every case (the ISA variants and the reference tools read its options).
    namespace cch = tw::collision_choice;
    const std::string stageName = cch::Effective(src.collision == forevervalidator::TwinCollision::Exact ? cch::kExact : cch::kNoise);
    cch::Choice choice;
    if (!cch::Resolve(stageName, tw::collide::CollideOptions{}, &choice)) {
        if (diag) *diag = "twin collision: " + cch::UnknownDiagnostic(stageName);
        return false;
    }
    if (!out->collision.Init(out->view, out->shapes, &why, choice.b)) {
        if (diag) *diag = "twin collision: " + why;
        return false;
    }
    if (choice.useC) {
        if (!out->collisionC.Init(out->view, out->shapes, &why, choice.c)) {
            if (diag) *diag = "twin collision (C): " + why;
            return false;
        }
    }
    if (choice.useAtlas) {
        out->collisionAtlas.reset(new tw::atlas::AtlasCollision());
        if (!out->collisionAtlas->Init(out->view, out->shapes, &why)) {
            if (diag) *diag = "twin collision (atlas): " + why;
            return false;
        }
    }
    out->useC = choice.useC;
    out->collisionName = choice.name;
    out->worldMs = MsSince(t0);

    // race map + import context
    out->race = out->view.RaceMap();
    if (!tw::race::TwRaceMapValid(out->race)) {
        if (diag) *diag = "twin race map: limits exceeded (TwRaceMapValid)";
        return false;
    }
    out->spawnIsos.resize(out->view.spawns.size());
    for (std::size_t i = 0; i < out->view.spawns.size(); ++i)
        std::memcpy(out->spawnIsos[i].data(), out->view.spawns[i].iso, sizeof out->spawnIsos[i]);
    tw::model::TwGoldenContext &g = out->golden;
    g = tw::model::TwGoldenContext{};
    g.fsGroundedBits = Bits(out->car.fsGrounded);   // bodyAirResponse.groundedSolidFeedback1
    g.ldsGroundedBits = 0u;                          // grounded linear damping scale is 0 (M1 TwCarPack)
    g.fsAirborneBits = Bits(out->car.fsAirborne);   // airborneSolidFeedback1
    g.ldsAirborneBits = Bits(out->car.ldsAirborne); // airborneSolidFeedback0
    g.spawns.iso = reinterpret_cast<const std::uint32_t(*)[12]>(out->spawnIsos.data());
    g.spawns.count = static_cast<std::uint32_t>(out->spawnIsos.size());
    g.requiredCps = out->race.requiredCps;
    g.requiredLaps = out->race.requiredLaps;
    return true;
}

std::shared_ptr<const twin::TwinWorld> BuildTwinWorld(const TwinWorldSource &src, forevervalidator::TwinPolicy policy,
                                                      const TwinWorldLabel &label, std::string *diag) {
    try {
        std::shared_ptr<TwinWorldPacks> packs(new TwinWorldPacks());   // over-aligned (CarForceConsts): aligned new
        if (!BuildTwinWorldPacks(src, packs.get(), diag)) return nullptr;
        auto impl = std::make_unique<twin::TwinWorld::Impl>();
        impl->info.mapEnvironment = label.mapEnvironment;
        impl->info.vehicleModel = label.vehicleModel;
        impl->info.policy = policy;
        impl->info.collision = src.collision;
        impl->info.collisionStage = packs->collisionName;
        impl->info.mapName = label.mapName;
        impl->info.carPackReady = true;
        impl->info.worldPackReady = true;
        impl->model.policy = twin::internal::ToStepPolicy(policy);
        impl->model.car = &packs->car;
        impl->model.water = packs->Water();
        impl->model.race = &packs->race;
        impl->model.world = nullptr;      // step::Model::world is B's bench form; M3's stage reads packs->view
        impl->model.collision = packs->Stage();      // C's noise stage, B's exact one, or an experiment (packs->collisionName)
        impl->packs = std::move(packs);
        return twin::TwinWorldFactory::Make(std::move(impl));
    } catch (const std::bad_alloc &) {
        if (diag) *diag = "twin world: allocation failed";
        return nullptr;
    }
}

std::shared_ptr<const twin::TwinWorld> BuildTwinWorldFromSandbox(const forevervalidator::experimental::PhysicsSandbox &loaded,
                                                                 forevervalidator::TwinPolicy policy,
                                                                 forevervalidator::TwinCollision collision,
                                                                 const TwinWorldLabel &label, std::string *diag) {
    using Access = forevervalidator::experimental::twin_test::PhysicsSandboxTwinTestAccess;
    const ReplaySimulationSession *session = Access::Session(loaded);
    const ReplaySimulationDefinition *definition = Access::Definition(loaded);
    if (session == nullptr || definition == nullptr) {
        if (diag) *diag = "twin world: the sandbox has no loaded replay or map";
        return nullptr;
    }
    auto state = loaded.CaptureState();
    if (!state) {
        if (diag) *diag = "twin world: sandbox state capture failed: " + state.Error().diagnostic;
        return nullptr;
    }
    const std::shared_ptr<const ReplaySimulationInstanceClone> clone = Access::RuntimeClone(state.Value());
    if (clone == nullptr) {
        if (diag) *diag = "twin world: the sandbox state has no runtime clone";
        return nullptr;
    }
    TwinWorldSource src;
    if (!TwinWorldSourceFromSession(*session, *definition, *clone, &src, diag)) return nullptr;
    src.collision = collision;
    return BuildTwinWorld(src, policy, label, diag);
}

bool ImportSandboxState(const twin::TwinWorld &world, const forevervalidator::experimental::PhysicsSandboxState &state,
                        tw::model::TwinState *out, std::string *diag) {
    using Access = forevervalidator::experimental::twin_test::PhysicsSandboxTwinTestAccess;
    const std::shared_ptr<const ReplaySimulationInstanceClone> clone = Access::RuntimeClone(state);
    if (clone == nullptr) {
        if (diag) *diag = "twin import: the sandbox state has no runtime clone";
        return false;
    }
    return ImportFvClone(world, *clone, 0u, out, diag);
}

}  // namespace tw::fv
