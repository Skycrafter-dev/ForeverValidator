#include "simulation/runtime/replay_simulation_runtime.h"
#include <utility>

#include "engine/physics/dynamics/hms_item.h"
#include "simulation/runtime/replay_environment.h"
#include "simulation/replay/replay_map_scene.h"
#include "simulation/runtime/replay_physics_world.h"
#include "simulation/runtime/replay_vehicle_body.h"
#include "simulation/runtime/replay_vehicle_simulation.h"
#include "simulation/runtime/replay_validation_spawn.h"
#include "engine/game/trackmania_race.h"
namespace {

CHmsItem::Properties ReplayVehicleItemProperties() {
    CHmsItem::Properties properties;
    properties.contactInterest = CHmsItem::EContactInterest_Local;
    properties.collisionGroup = CHmsItem::ECollisionGroup_Dynamic;
    properties.dynamicType = CHmsItem::EDynamicType_Normal;
    properties.active = true;
    properties.shadowTexCastedEnabled = true;
    properties.shadowFakeEnabled = true;
    properties.lightLensFlareEnabled = true;
    properties.shadowTexCastedCount = 1u;
    properties.shadowCasterGroupMask = 2u;
    properties.shadowReceiverGroupMask = 0xff6u;
    return properties;
}

}  // namespace

ReplayStuntSimulationState BuildReplayStuntSimulationState(
        const ReplaySimulationStepExecution &execution,
        const CSceneVehicleCar::SVehicleCarState &physics,
        const ReplayControlTick &tick) {
    ReplayStuntSimulationState state;
    state.tickTimeMs = tick.timeMs;
    state.inputQueryTimeOffsetMs = tick.periodMs;
    state.raceStart = tick.actions.resetAtRaceStart;
    state.finishRace = tick.actions.finishRace;
    state.vehicleLocation.Set(
            execution.writeFrame.rotation,
            execution.writeFrame.position);
    state.forwardSpeed = physics.forwardSpeed;
    state.sideSpeed = physics.sideSpeed;
    state.hasWheelContact = physics.hasWheelContact;
    state.hasBodyContact = physics.hasBodyContact;
    state.bodyContactVerticalAngle = physics.bodyContactVerticalAngle;
    state.bodyContactHorizontalAngle = physics.bodyContactHorizontalAngle;
    state.noGroundFrictionGuard = physics.noGroundFrictionGuard;
    state.inputLastChangeTimeMs = tick.stuntsInput.lastChangeTimeMs;
    return state;
}

struct ReplaySimulationRuntime::State {
    explicit State(CTrackManiaRace &race)
        : vehicle(race), race(race) {}

    ReplayPhysicsWorld world;
    ReplayEnvironment environment;
    ReplayVehicleBody body;
    ReplayVehicleSimulation vehicle;
    CTrackManiaRace &race;
    const ReplaySimulationDefinition *definition = nullptr;
    bool staticSceneReady = false;
    bool firstStep = true;
    bool stuntsEnabled = false;
    Phase phase = Phase::Detached;
};

ReplaySimulationRuntime::ReplaySimulationRuntime(CTrackManiaRace &race)
    : state_(std::make_unique<State>(race)) {}

ReplaySimulationRuntime::~ReplaySimulationRuntime() = default;

ReplaySimulationRunResult ReplaySimulationRuntime::Start(
        const ReplaySimulationDefinition &definition,
        ReplayMapScene &mapScene,
        const GmIso4 &spawnLocation,
        const ReplayControlTick &firstTick,
        std::uint32_t validationSeed) {
    State &state = *state_;
    state.definition = &definition;
    const ReplayMapSceneResult sceneResult = state.world.ConnectMapScene(
            mapScene, &state.vehicle.Car(), state.race);
    if (sceneResult != ReplayMapSceneResult::Ready) {
        return MapReplaySceneResult(sceneResult);
    }

    state.body.InitializeAtSpawn(
            definition.vehicle.dynaParameters, spawnLocation);
    state.body.ConstructItem(ReplayVehicleItemProperties());
    state.body.BuildCorpus();
    state.body.InstallEmptyCollisionTree();
    state.environment.Build(definition.environment);
    state.environment.InstallWater(
            state.world.Zone(),
            state.world.CollisionZone(),
            definition.environment.water);
    state.world.AddVehicleBody(state.body.Corpus());
    state.staticSceneReady = mapScene.IsActive();

    const ReplayVehiclePreparationResult vehicleResult = state.vehicle.Start(
            definition, firstTick, state.body, state.staticSceneReady);
    if (vehicleResult != ReplayVehiclePreparationResult::Ready) {
        return ReplaySimulationRunResult::VehicleCollisionModelFailed;
    }
    const std::optional<ReplayDynaParameters> parameters =
            state.vehicle.BuildDynaParameters();
    if (parameters.has_value()) {
        state.body.InstallDynaParameters(*parameters);
    }
    state.stuntsEnabled = firstTick.actions.enableStuntsSimulation;
    state.race.ConfigureReplayStuntsSimulation(
            state.stuntsEnabled,
            firstTick.actions.stuntsTimeLimitMs);
    state.body.SetSpawnLocation(
            BuildReplayValidationSpawnLocation(
                    spawnLocation, validationSeed));
    state.firstStep = true;
    state.phase = Phase::Idle;
    return ReplaySimulationRunResult::Success;
}

ReplaySimulationStepExecution ReplaySimulationRuntime::Step(
        const ReplayControlTick &tick) {
    State &state = *state_;
    ReplaySimulationStepExecution execution;
    if (state.definition == nullptr || state.phase != Phase::Idle) {
        execution.result = ReplaySimulationRunResult::InvalidControlTimeline;
        return execution;
    }
    state.phase = Phase::Stepping;

    if (!state.firstStep) {
        state.vehicle.PrepareStep(tick, state.body);
    }

    CSceneVehicleCar &car = state.vehicle.Car();
    car.EnableAbsorbContactCallback(1);
    car.EnablePhysicsUpdates(!tick.actions.suppressVehicleForceCallbacks);
    state.world.InstallEnvironment(
            state.environment,
            state.definition->environment,
            !tick.actions.suppressVehicleForceCallbacks);
    state.world.SetSimulationTime(tick);

    for (std::uint32_t respawnIndex = 0u;
         respawnIndex < tick.actions.respawnAtCheckpointCount;
         ++respawnIndex) {
        if (state.vehicle.Respawn(state.body)) {
            ++execution.respawnExecutedCount;
            state.race.ApplyReplayStuntRespawnPenalty(tick.timeMs);
        }
    }

    state.world.Step();
    execution.simulatedFrame = state.body.CaptureCurrentFrame();
    execution.writeFrame = state.body.CaptureWriteState();
    if (state.stuntsEnabled) {
        const CSceneVehicleCar::SVehicleCarState &physics =
                car.ReplayPhysicsState();
        const ReplayStuntSimulationState stuntState =
                BuildReplayStuntSimulationState(
                        execution, physics, tick);
        state.race.SetReplayStuntSimulationState(stuntState);
        state.race.UpdateStunts();
    }
    execution.finishTickMs = state.vehicle.FinishTimeMs();
    state.firstStep = false;
    state.phase = Phase::Idle;
    return execution;
}

std::optional<std::uint32_t> ReplaySimulationRuntime::FinishTimeMs() const {
    return state_->vehicle.FinishTimeMs();
}

std::optional<std::uint32_t> ReplaySimulationRuntime::StuntsScore() const {
    if (!state_->stuntsEnabled) {
        return std::nullopt;
    }
    return state_->race.StuntsScore();
}

ReplayDynaFrameState ReplaySimulationRuntime::CurrentFrame() const {
    return state_->body.CaptureCurrentFrame();
}

ReplayVehicleControlState ReplaySimulationRuntime::CurrentControls() const {
    const CSceneVehicleCar::SControlInput input =
            state_->vehicle.Car().ControlInput();
    return {input.lowSpeedGateA, input.lowSpeedGateB, input.steering};
}

const ReplayRaceProgress &ReplaySimulationRuntime::RaceProgress() const {
    return state_->race.Progress();
}

std::optional<std::uint32_t>
ReplaySimulationRuntime::ApplyReplayStuntTimePenalty(
        std::uint32_t overtimeMs) {
    if (!state_->stuntsEnabled) {
        return std::nullopt;
    }
    state_->race.ApplyReplayStuntTimePenalty(overtimeMs);
    return state_->race.StuntsScore();
}

std::optional<ReplaySimulationRuntime::RuntimeClone>
ReplaySimulationRuntime::CaptureRuntimeClone() const {
    if (state_->phase != Phase::Idle || state_->definition == nullptr) {
        return std::nullopt;
    }
    RuntimeClone clone;
    clone.world = state_->world.CaptureRuntimeClone();
    clone.body = state_->body.CaptureRuntimeClone();
    clone.vehicle = state_->vehicle.CaptureRuntimeClone();
    clone.firstStep = state_->firstStep;
    clone.stuntsEnabled = state_->stuntsEnabled;
    return clone;
}

bool ReplaySimulationRuntime::PrepareRuntimeCloneRestore(
        const RuntimeClone &clone) {
    return state_->phase == Phase::Idle &&
           state_->definition != nullptr &&
           state_->body.PrepareRuntimeCloneRestore(clone.body) &&
           state_->vehicle.CanRestoreRuntimeClone(clone.vehicle);
}

void ReplaySimulationRuntime::RestoreRuntimeClone(
        RuntimeClone clone) noexcept {
    state_->world.RestoreRuntimeClone(clone.world);
    state_->body.RestoreRuntimeClone(std::move(clone.body));
    state_->vehicle.RestoreRuntimeClone(clone.vehicle);
    state_->firstStep = clone.firstStep;
    state_->stuntsEnabled = clone.stuntsEnabled;
    state_->phase = Phase::Idle;
}

ReplaySimulationRuntime::Phase
ReplaySimulationRuntime::CurrentPhase() const noexcept {
    return state_->phase;
}
