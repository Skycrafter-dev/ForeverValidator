#include "simulation/runtime/replay_simulation_runtime.h"

#include "simulation/backends/optimized_cpu/optimized_cpu_vehicle_forces.h"
#include "simulation/backends/cuda/cuda_backend.h"
#include "simulation/backends/cuda/cuda_vehicle_cpu_reference.h"
#include "simulation/runtime/replay_finish_time_estimator.h"
#include <cstring>
#include <new>
#include <utility>

#include "engine/physics/dynamics/hms_item.h"
#include "engine/physics/collision/gm_collision_buffer.h"
#include "simulation/backends/simulation_backend.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_surface_transform_cache.h"
#include "simulation/runtime/replay_environment.h"
#include "simulation/replay/replay_map_scene.h"
#include "simulation/runtime/replay_physics_world.h"
#include "simulation/runtime/replay_vehicle_body.h"
#include "simulation/runtime/replay_vehicle_simulation.h"
#include "simulation/runtime/replay_validation_spawn.h"
#include "engine/game/trackmania_race.h"
namespace {

enum FinishProbePhysicsPath : std::uint8_t {
    FinishProbeReference,
    FinishProbeOptimizedCpu,
    FinishProbeOptimizedCpuCached,
    FinishProbeOptimizedCpuNativeBinary32,
    FinishProbeOptimizedCpuNativeBinary32Cached,
};

class ProjectionHash {
public:
    template<typename T>
    void Add(const T &value) noexcept {
        AddBytes(&value, sizeof(value));
    }

    template<typename T>
    void AddOptional(const std::optional<T> &value) noexcept {
        const bool present = value.has_value();
        Add(present);
        if (present) {
            Add(*value);
        }
    }

    template<typename T>
    void AddVector(const std::vector<T> &values) noexcept {
        Add(values.size());
        for (const T &value : values) {
            Add(value);
        }
    }

    void AddBoolVector(const std::vector<bool> &values) noexcept {
        Add(values.size());
        for (bool value : values) {
            Add(value);
        }
    }

    std::uint64_t Value() const noexcept { return value_; }

private:
    void AddBytes(const void *data, std::size_t size) noexcept {
        const auto *bytes = static_cast<const unsigned char *>(data);
        for (std::size_t index = 0u; index < size; ++index) {
            value_ ^= bytes[index];
            value_ *= 1099511628211ull;
        }
    }

    std::uint64_t value_ = 1469598103934665603ull;
};

std::uint64_t HashDynamicBody(
        const ReplayVehicleBody::RuntimeClone &body) {
    ProjectionHash hash;
    hash.AddOptional(body.maxAngularSpeed);
    hash.Add(body.dynaParams);
    hash.Add(body.physicalParameters);
    hash.Add(body.tempState);
    hash.Add(body.writeState);
    hash.Add(body.currentState);
    hash.AddVector(body.pendingCollisionReplacements);
    hash.Add(body.isDynamicActive);
    hash.Add(body.dynamicType);
    hash.Add(body.corpusLocalIso);
    return hash.Value();
}

void AddRaceProgress(ProjectionHash &hash,
                     const ReplayRaceProgress &progress) {
    hash.Add(progress.installedTriggerCount);
    hash.Add(progress.preparedEventCount);
    hash.Add(progress.checkpointCount);
    hash.Add(progress.finishCount);
    hash.Add(progress.freewheelClearCount);
    hash.Add(progress.lastBlockRole);
    hash.Add(progress.lastPrepareTimeMs);
    hash.Add(progress.lastAcceptedBlockId);
    hash.Add(progress.lastContactBlockId);
    hash.Add(progress.currentLapCheckpointCount);
    hash.Add(progress.totalCheckpointEventCount);
    hash.Add(progress.completedLapCount);
    hash.Add(progress.requiredLapCount);
    hash.Add(progress.requiredCheckpointCount);
    hash.Add(progress.raceCompleted);
}

void AddStuntState(ProjectionHash &hash,
                   const ReplayStuntSimulationState &state) {
    hash.Add(state.tickTimeMs);
    hash.Add(state.inputQueryTimeOffsetMs);
    hash.Add(state.raceStart);
    hash.Add(state.finishRace);
    hash.Add(state.vehicleLocation);
    hash.Add(state.forwardSpeed);
    hash.Add(state.sideSpeed);
    hash.Add(state.hasWheelContact);
    hash.Add(state.hasBodyContact);
    hash.Add(state.bodyContactVerticalAngle);
    hash.Add(state.bodyContactHorizontalAngle);
    hash.Add(state.noGroundFrictionGuard);
    hash.Add(state.inputLastChangeTimeMs);
}

void AddStuntEvent(ProjectionHash &hash,
                   const ReplayStuntEvent &event) {
    hash.Add(event.figure);
    hash.Add(event.degree);
    hash.Add(event.score);
    hash.Add(event.bonus);
    hash.Add(event.straightLanding);
    hash.Add(event.reverseLanding);
    hash.Add(event.masterJump);
    hash.Add(event.chain);
}

void AddStuntEvents(
        ProjectionHash &hash,
        const std::vector<ReplayStuntEvent> &events) {
    hash.Add(events.size());
    for (const ReplayStuntEvent &event : events) {
        AddStuntEvent(hash, event);
    }
}

void AddWheel(ProjectionHash &hash,
              const CSceneVehicleCar::SSimulationWheel &wheel) {
    hash.Add(wheel.killsLateralSpeedOnContact);
    hash.Add(wheel.axle);
    hash.Add(wheel.rollingRadius);
    hash.Add(wheel.surfaceHandler.RestPose());
    hash.Add(wheel.surfaceHandler.CurrentPose());
    hash.Add(wheel.forceApplicationPoint);
    hash.Add(wheel.realTimeState);
    hash.Add(wheel.previousPhysicsState);
    hash.Add(wheel.currentPhysicsState);
    hash.Add(wheel.previousAsyncState);
    hash.Add(wheel.asyncState);
}


void AddAirControl(ProjectionHash &hash,
                   const CSceneVehicleCar::SAirControl &air) {
    hash.Add(air.refreshMemory);
    hash.Add(air.memoryTick);
    hash.Add(air.memoryAngular);
}

void AddRadiusSteering(
        ProjectionHash &hash,
        const CSceneVehicleCar::SRadiusSteeringState &radius) {
    hash.Add(radius.steerAngle);
    hash.Add(radius.previousSteerSign);
    hash.Add(radius.phase);
}

void AddSlipMemory(
        ProjectionHash &hash,
        const CSceneVehicleCar::SSlipMemoryState &slip) {
    hash.Add(slip.active);
    hash.Add(slip.lastTick);
    hash.Add(slip.startTick);
    hash.Add(slip.elapsedTicks);
    hash.Add(slip.steeringMemoryTick);
    hash.Add(slip.steeringMemorySlip);
}

std::uint64_t HashPowertrain(
        const CSceneVehicleCar::RuntimeClone &car) {
    ProjectionHash hash;
    hash.Add(car.controls);
    hash.Add(car.feedback);
    hash.Add(car.integration);
    hash.Add(car.engine);
    hash.Add(car.turbo);
    AddAirControl(hash, car.airControl);
    AddRadiusSteering(hash, car.radiusSteering);
    AddSlipMemory(hash, car.slipMemory);
    hash.Add(car.gearedDrive);
    hash.Add(car.lastComputeForcesTick);
    hash.Add(car.dynaPartSprings);
    hash.Add(car.forceAccumulators);
    return hash.Value();
}


std::uint64_t HashVehicle(
        const ReplayVehicleSimulation::RuntimeClone &vehicle) {
    const CSceneVehicleCar::RuntimeClone &car = vehicle.car;
    ProjectionHash hash;
    hash.Add(car.vehicle.mobil);
    hash.Add(car.vehicle.vehicleEvents);
    hash.Add(car.vehicle.water);
    hash.Add(car.vehicle.updateAsync);
    hash.Add(car.vehicle.networked);
    hash.Add(car.vehicle.predictionDelayTicks);
    hash.AddOptional(car.vehicle.stateSampleWindow);
    hash.Add(car.vehicle.asyncPeriodSeconds);
    hash.Add(car.wheels.size());
    for (const CSceneVehicleCar::SSimulationWheel &wheel : car.wheels) {
        AddWheel(hash, wheel);
    }
    hash.Add(HashPowertrain(car));
    hash.Add(car.linearSpeedCap);
    hash.Add(car.frameHistory);
    hash.Add(car.reverseGearSpeedThreshold);
    hash.Add(car.contacts);
    hash.AddBoolVector(vehicle.wheelSurfaces.movedByUpdateSurface);
    return hash.Value();
}


std::uint64_t HashRace(const CTrackManiaRace::RuntimeClone &race) {
    ProjectionHash hash;
    hash.Add(race.player);
    hash.AddVector(race.checkpointSlotsPassed);
    hash.AddOptional(race.playerSpawnLocation);
    hash.AddOptional(race.lastAcceptedSpawnLocation);
    hash.Add(race.currentSpawnLocationInitialized);
    hash.Add(race.preparedEventTimeMs);
    hash.Add(race.replayPlayMode);
    hash.Add(race.replayNbLaps);
    AddRaceProgress(hash, race.progress);
    hash.Add(race.replayStuntsEnabled);
    hash.Add(race.replayStuntStateAvailable);
    hash.Add(race.replayStuntsTimeLimitMs);
    hash.Add(race.replayStuntsRaceStartTimeMs);
    AddStuntState(hash, race.replayStuntState);
    for (const CTrackManiaRace::ReplayStuntInputSnapshot &snapshot :
         race.replayStuntInputHistory) {
        hash.Add(snapshot.tickTimeMs);
        hash.Add(snapshot.lastChangeTimeMs);
    }
    hash.Add(race.replayStuntInputHistorySize);
    hash.Add(race.replayStuntLocationHistory);
    hash.Add(race.replayStuntLocationHistorySize);
    hash.Add(race.replayStuntPreviousLocation);
    hash.Add(race.replayStuntTakeoffLocation);
    hash.Add(race.replayStuntRotation);
    hash.Add(race.replayStuntLandingDirection);
    hash.Add(race.replayStuntTakeoffTick);
    hash.Add(race.replayStuntLandingTick);
    hash.Add(race.replayStuntPreviousLandingTick);
    hash.Add(race.replayStuntChain);
    hash.Add(race.replayStuntComboWindowMs);
    hash.Add(race.replayStuntInProgress);
    hash.Add(race.replayStuntMasterJump);
    hash.Add(race.replayStuntBadLanding);
    hash.AddOptional(race.replayStuntScoreAtTimeLimit);
    hash.Add(race.replayStuntFigureScores);
    hash.Add(race.stuntsScore);
    AddStuntEvents(hash, race.stuntEvents);
    return hash.Value();
}

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

struct ReplaySimulationRuntime::State : CHmsCollisionSubstepObserver {
    struct Snapshot {
        RuntimeClone runtime;
        CTrackManiaRace::RuntimeClone race;
    };

    State(CTrackManiaRace &race,
          forevervalidator::SimulationBackend requestedBackend)
        : vehicle(race),
          race(race),
          backend(forevervalidator::simulation::ResolveLeafBackend(
                  requestedBackend)) {
        race.SetCurrentTransformCheckpointFreewheelClearEnabled(
                backend == forevervalidator::SimulationBackend::Reference ||
                backend == forevervalidator::SimulationBackend::OptimizedCpu ||
                backend == forevervalidator::SimulationBackend::Cuda ||
                backend == forevervalidator::SimulationBackend::Hip ||
                backend == forevervalidator::SimulationBackend::Vulkan);
    }

    void BeforeCollisionSubstep(CHmsCorpus &corpus, float dt) override;
    void AfterCollisionSubstep(CHmsCorpus &corpus, float dt) override;

    ReplayPhysicsWorld world;
    ReplayEnvironment environment;
    ReplayVehicleBody body;
    ReplayVehicleSimulation vehicle;
    CTrackManiaRace &race;
    forevervalidator::SimulationBackend backend;
    const ReplaySimulationDefinition *definition = nullptr;
    bool staticSceneReady = false;
    bool firstStep = true;
    bool stuntsEnabled = false;
    Phase phase = Phase::Detached;
    forevervalidator::simulation::OptimizedCpuVehicleForceContext
            optimizedCpuVehicleForces;
    std::unique_ptr<OptimizedCpuStaticSurfaceTransformCache>
            optimizedCpuStaticTransforms;
    std::optional<forevervalidator::FinishTimeEstimate> finishTime;
    bool captureFinishTransition = false;
    bool finishTransitionFound = false;
    bool finishBeforeSubstep = false;
    double finishElapsedSeconds = 0.0;
    double finishSubstepStartSeconds = 0.0;
    float finishSubstepDurationSeconds = 0.0f;
    std::optional<Snapshot> finishSubstepSnapshot;
    Snapshot finishTickSnapshot;
};

void ReplaySimulationRuntime::State::BeforeCollisionSubstep(
        CHmsCorpus &corpus, float dt) {
    (void)dt;
    if (!captureFinishTransition || finishTransitionFound ||
        &corpus != &body.Corpus()) {
        return;
    }
    Snapshot snapshot;
    snapshot.runtime.world = world.CaptureRuntimeClone();
    snapshot.runtime.body = body.CaptureRuntimeClone();
    snapshot.runtime.vehicle = vehicle.CaptureRuntimeClone();
    snapshot.runtime.finishTime = finishTime;
    snapshot.runtime.firstStep = firstStep;
    snapshot.runtime.stuntsEnabled = stuntsEnabled;
    snapshot.race = race.CaptureRuntimeClone();
    finishSubstepSnapshot = std::move(snapshot);
    finishSubstepStartSeconds = finishElapsedSeconds;
    finishBeforeSubstep = race.Progress().raceCompleted;
}

void ReplaySimulationRuntime::State::AfterCollisionSubstep(
        CHmsCorpus &corpus, float dt) {
    if (!captureFinishTransition || &corpus != &body.Corpus()) {
        return;
    }
    if (!finishTransitionFound && !finishBeforeSubstep &&
        race.Progress().raceCompleted) {
        finishTransitionFound = true;
        finishSubstepDurationSeconds = dt;
    }
    finishElapsedSeconds += static_cast<double>(dt);
}

ReplaySimulationRuntime::ReplaySimulationRuntime(
        CTrackManiaRace &race,
        forevervalidator::SimulationBackend backend)
    : state_(std::make_unique<State>(race, backend)) {}

ReplaySimulationRuntime::~ReplaySimulationRuntime() = default;

ReplaySimulationRunResult ReplaySimulationRuntime::Start(
        const ReplaySimulationDefinition &definition,
        ReplayMapScene &mapScene,
        const GmIso4 &spawnLocation,
        const ReplayControlTick &firstTick,
        std::uint32_t validationSeed) {
    State &state = *state_;
    state.optimizedCpuVehicleForces.Reset();
    state.finishTime.reset();
    state.captureFinishTransition = false;
    state.finishTransitionFound = false;
    state.finishSubstepSnapshot.reset();
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

void ReplaySimulationRuntime::PrepareOptimizedCpuStaticTransforms(
        void) noexcept {
    State &state = *state_;
    state.optimizedCpuStaticTransforms.reset();
    if (!forevervalidator::simulation::UsesOptimizedCpuFoundation(
                state.backend) ||
        state.definition == nullptr || state.phase != Phase::Idle) {
        return;
    }
    try {
        auto transforms =
                std::make_unique<OptimizedCpuStaticSurfaceTransformCache>();
        if (transforms->TryRebuild(state.world.CollisionZone())) {
            state.optimizedCpuStaticTransforms = std::move(transforms);
        }
    } catch (const std::bad_alloc &) {
    }
}

void ReplaySimulationRuntime::
CertifyOptimizedCpuStaticTransformsForAdvance(void) noexcept {
    State &state = *state_;
    if (state.optimizedCpuStaticTransforms != nullptr) {
        CPlugTree *movingTree = state.body.Corpus().CollisionTree();
        state.optimizedCpuStaticTransforms->CertifyForRuntimeAdvance(
                state.world.CollisionZone(), movingTree);
    }
}

ReplaySimulationStepExecution ReplaySimulationRuntime::Step(
        const ReplayControlTick &tick) {
    State &state = *state_;
    ReplaySimulationStepExecution execution;
    if (state.backend == forevervalidator::SimulationBackend::Cuda ||
        state.backend == forevervalidator::SimulationBackend::Hip ||
        state.backend == forevervalidator::SimulationBackend::Vulkan) {
        execution.result =
                ReplaySimulationRunResult::CudaExecutionFailed;
        return execution;
    }
    if (state.definition == nullptr || state.phase != Phase::Idle) {
        execution.result = ReplaySimulationRunResult::InvalidControlTimeline;
        return execution;
    }
    state.race.ClearAcceptedCheckpointEvents();
    const bool finishPreCaptured =
            !state.captureFinishTransition &&
            !state.race.Progress().raceCompleted &&
            CaptureRuntimeClone(state.finishTickSnapshot.runtime);
    if (finishPreCaptured) {
        state.race.CaptureRuntimeClone(state.finishTickSnapshot.race);
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
    if (finishPreCaptured &&
        !state.finishTickSnapshot.race.progress.raceCompleted &&
        state.race.Progress().raceCompleted) {
        EstimateFinishTime(
                tick, FinishProbeReference,
                state.finishTickSnapshot.runtime,
                state.finishTickSnapshot.race);
    }
    execution.finishTime = state.finishTime;
    return execution;
}

ReplaySimulationStepExecution ReplaySimulationRuntime::StepOptimizedCpu(
        const ReplayControlTick &tick) {
    State &state = *state_;
    ReplaySimulationStepExecution execution;
    if (state.definition == nullptr || state.phase != Phase::Idle) {
        execution.result = ReplaySimulationRunResult::InvalidControlTimeline;
        return execution;
    }
    if (state.stuntsEnabled &&
        !state.definition->optimizedCpuStadiumSpecializationsEnabled) {
        return Step(tick);
    }
    state.race.ClearAcceptedCheckpointEvents();
    const bool finishPreCaptured =
            !state.captureFinishTransition &&
            !state.race.Progress().raceCompleted &&
            CaptureRuntimeClone(state.finishTickSnapshot.runtime);
    if (finishPreCaptured) {
        state.race.CaptureRuntimeClone(state.finishTickSnapshot.race);
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

    if (state.optimizedCpuStaticTransforms != nullptr &&
        state.optimizedCpuStaticTransforms->IsCertifiedFor(
                state.world.CollisionZone())) {
        state.world.StepOptimizedCpuCached(
                *state.optimizedCpuStaticTransforms);
    } else {
        state.world.StepOptimizedCpu();
    }
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
    const std::uint8_t finishPath =
            state.optimizedCpuStaticTransforms != nullptr &&
                    state.optimizedCpuStaticTransforms->IsCertifiedFor(
                            state.world.CollisionZone())
            ? FinishProbeOptimizedCpuCached
            : FinishProbeOptimizedCpu;
    if (finishPreCaptured &&
        !state.finishTickSnapshot.race.progress.raceCompleted &&
        state.race.Progress().raceCompleted) {
        EstimateFinishTime(
                tick, finishPath,
                state.finishTickSnapshot.runtime,
                state.finishTickSnapshot.race);
    }
    execution.finishTime = state.finishTime;
    return execution;
}

ReplaySimulationStepExecution
ReplaySimulationRuntime::StepOptimizedCpuNativeBinary32(
        const ReplayControlTick &tick) {
    State &state = *state_;
    ReplaySimulationStepExecution execution;
    if (state.definition == nullptr || state.phase != Phase::Idle) {
        execution.result = ReplaySimulationRunResult::InvalidControlTimeline;
        return execution;
    }
    if (!state.definition->optimizedCpuStadiumSpecializationsEnabled) {
        return StepOptimizedCpu(tick);
    }
    state.race.ClearAcceptedCheckpointEvents();
    const bool finishPreCaptured =
            !state.captureFinishTransition &&
            !state.race.Progress().raceCompleted &&
            CaptureRuntimeClone(state.finishTickSnapshot.runtime);
    if (finishPreCaptured) {
        state.race.CaptureRuntimeClone(state.finishTickSnapshot.race);
    }
    state.phase = Phase::Stepping;

    if (!state.firstStep) {
        state.vehicle.PrepareStep(tick, state.body);
    }

    CSceneVehicleCar &car = state.vehicle.Car();
    car.EnsurePhysicsCallbacks(
            1,
            tick.actions.suppressVehicleForceCallbacks ? 0 : 1);
    CHmsItem *enabledItem = car.HmsItem();
    CHmsItem::CCallback *enabledComputeForcesCallback =
            enabledItem != nullptr
                    ? enabledItem->CallbackGet(
                              CHmsItem::ECallback_ComputeForces)
                    : nullptr;
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

    if (state.definition->optimizedCpuStadiumSpecializationsEnabled) {
        state.optimizedCpuVehicleForces.BeginTick(
                car,
                forevervalidator::simulation::
                        OptimizedCpuBinary32MathPath::X86Sse2,
                enabledComputeForcesCallback);
    }
    if (state.optimizedCpuStaticTransforms != nullptr &&
        state.optimizedCpuStaticTransforms->IsCertifiedFor(
                state.world.CollisionZone())) {
        state.world.StepOptimizedCpuNativeBinary32Cached(
                *state.optimizedCpuStaticTransforms,
                state.optimizedCpuVehicleForces);
    } else {
        state.world.StepOptimizedCpuNativeBinary32(
                state.optimizedCpuVehicleForces);
    }
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
    const std::uint8_t finishPath =
            state.optimizedCpuStaticTransforms != nullptr &&
                    state.optimizedCpuStaticTransforms->IsCertifiedFor(
                            state.world.CollisionZone())
            ? FinishProbeOptimizedCpuNativeBinary32Cached
            : FinishProbeOptimizedCpuNativeBinary32;
    if (finishPreCaptured &&
        !state.finishTickSnapshot.race.progress.raceCompleted &&
        state.race.Progress().raceCompleted) {
        EstimateFinishTime(
                tick, finishPath,
                state.finishTickSnapshot.runtime,
                state.finishTickSnapshot.race);
    }
    execution.finishTime = state.finishTime;
    return execution;
}

std::optional<std::uint32_t> ReplaySimulationRuntime::FinishTimeMs() const {
    return state_->vehicle.FinishTimeMs();
}

std::optional<forevervalidator::FinishTimeEstimate>
ReplaySimulationRuntime::FinishTime() const {
    return state_->finishTime;
}

bool ReplaySimulationRuntime::ProbeFinishSubstep(
        float dt, std::uint8_t physicsPath) {
    State &state = *state_;
    if (state.phase != Phase::Idle || !(dt > 0.0f)) {
        return false;
    }

    CSceneVehicleCar &car = state.vehicle.Car();
    CHmsItem *item = car.HmsItem();
    CHmsItem::CCallback *computeForcesCallback =
            item != nullptr
                    ? item->CallbackGet(
                              CHmsItem::ECallback_ComputeForces)
                    : nullptr;
    CHmsCorpus &corpus = state.body.Corpus();
    CHmsDyna &dyna = state.body.Dyna();
    CHmsCollisionManagerSZone &collisionZone =
            state.world.CollisionZone();
    CHmsCollisionBuffer collisionBuffer;
    state.world.InstallSimulationTimeAsCurrent();
    collisionZone.PrepareCollisions();
    state.phase = Phase::Stepping;

    const bool nativeBinary32 =
            physicsPath == FinishProbeOptimizedCpuNativeBinary32 ||
            physicsPath ==
                    FinishProbeOptimizedCpuNativeBinary32Cached;
    if (nativeBinary32) {
        state.optimizedCpuVehicleForces.BeginTick(
                car,
                forevervalidator::simulation::
                        OptimizedCpuBinary32MathPath::X86Sse2,
                computeForcesCallback);
        state.world.Zone().ComputeCorpusForcesOptimizedCpuVehicle(
                &corpus, dt, state.optimizedCpuVehicleForces);
    } else {
        state.world.Zone().ComputeCorpusForces(&corpus, dt);
    }
    dyna.DoPreCollisionDynamic(dt);

    switch (physicsPath) {
    case FinishProbeOptimizedCpu:
        collisionZone.DetectCollisionsCorpusOptimizedCpu(
                collisionBuffer, &corpus);
        break;
    case FinishProbeOptimizedCpuCached:
        if (state.optimizedCpuStaticTransforms == nullptr) {
            state.phase = Phase::Idle;
            return false;
        }
        collisionZone.DetectCollisionsCorpusOptimizedCpuCached(
                collisionBuffer, &corpus,
                *state.optimizedCpuStaticTransforms);
        break;
    case FinishProbeOptimizedCpuNativeBinary32:
        collisionZone.DetectCollisionsCorpusOptimizedCpuNativeBinary32(
                collisionBuffer, &corpus);
        break;
    case FinishProbeOptimizedCpuNativeBinary32Cached:
        if (state.optimizedCpuStaticTransforms == nullptr) {
            state.phase = Phase::Idle;
            return false;
        }
        collisionZone.
                DetectCollisionsCorpusOptimizedCpuNativeBinary32Cached(
                        collisionBuffer, &corpus,
                        *state.optimizedCpuStaticTransforms);
        break;
    case FinishProbeReference:
    default:
        collisionZone.DetectCollisionsCorpus(
                collisionBuffer, &corpus);
        break;
    }
    state.world.Zone().ComputeCollisionResponse(collisionBuffer);
    dyna.DoPostCollisionDynamic();
    const bool finished = state.race.Progress().raceCompleted;
    state.phase = Phase::Idle;
    return finished;
}

void ReplaySimulationRuntime::EstimateFinishTime(
        const ReplayControlTick &tick,
        std::uint8_t physicsPath,
        RuntimeClone preTickRuntime,
        CTrackManiaRace::RuntimeClone preTickRace) {
    State &state = *state_;
    std::optional<RuntimeClone> finalRuntime = CaptureRuntimeClone();
    if (!finalRuntime.has_value()) {
        return;
    }
    CTrackManiaRace::RuntimeClone finalRace =
            state.race.CaptureRuntimeClone();
    if (!state.race.PrepareRuntimeCloneRestore(preTickRace) ||
        !PrepareRuntimeCloneRestore(preTickRuntime)) {
        return;
    }

    state.race.RestoreRuntimeClone(std::move(preTickRace));
    RestoreRuntimeClone(std::move(preTickRuntime));
    state.captureFinishTransition = true;
    state.finishTransitionFound = false;
    state.finishBeforeSubstep = false;
    state.finishElapsedSeconds = 0.0;
    state.finishSubstepStartSeconds = 0.0;
    state.finishSubstepDurationSeconds = 0.0f;
    state.finishSubstepSnapshot.reset();
    state.world.Zone().SetCollisionSubstepObserver(&state);

    ReplaySimulationStepExecution replay;
    switch (physicsPath) {
    case FinishProbeOptimizedCpu:
    case FinishProbeOptimizedCpuCached:
        replay = StepOptimizedCpu(tick);
        break;
    case FinishProbeOptimizedCpuNativeBinary32:
    case FinishProbeOptimizedCpuNativeBinary32Cached:
        replay = StepOptimizedCpuNativeBinary32(tick);
        break;
    case FinishProbeReference:
    default:
        replay = Step(tick);
        break;
    }
    state.world.Zone().SetCollisionSubstepObserver(nullptr);
    state.captureFinishTransition = false;

    std::optional<State::Snapshot> transition =
            std::move(state.finishSubstepSnapshot);
    const bool transitionReady =
            replay.result == ReplaySimulationRunResult::Success &&
            state.finishTransitionFound && transition.has_value();
    const ReplayFinishSubstep substep{
            tick.timeMs,
            tick.periodMs,
            state.finishSubstepStartSeconds,
            state.finishSubstepDurationSeconds,
    };

    if (!state.race.PrepareRuntimeCloneRestore(finalRace) ||
        !PrepareRuntimeCloneRestore(*finalRuntime)) {
        return;
    }
    state.race.RestoreRuntimeClone(finalRace);
    RestoreRuntimeClone(*finalRuntime);
    if (!transitionReady) {
        return;
    }

    const State::Snapshot exactPreSubstep = std::move(*transition);
    const auto restoreExactPreSubstep = [&]() {
        if (!state.race.PrepareRuntimeCloneRestore(
                    exactPreSubstep.race) ||
            !PrepareRuntimeCloneRestore(
                    exactPreSubstep.runtime)) {
            return false;
        }
        state.race.RestoreRuntimeClone(exactPreSubstep.race);
        RestoreRuntimeClone(exactPreSubstep.runtime);
        return true;
    };
    const std::optional<forevervalidator::FinishTimeEstimate> estimate =
            RefineReplayFinishTime(
                    substep,
                    [&](float partialDt) {
                        return restoreExactPreSubstep() &&
                               ProbeFinishSubstep(
                                       partialDt, physicsPath);
                    });

    if (!state.race.PrepareRuntimeCloneRestore(finalRace) ||
        !PrepareRuntimeCloneRestore(*finalRuntime)) {
        return;
    }
    state.race.RestoreRuntimeClone(std::move(finalRace));
    RestoreRuntimeClone(std::move(*finalRuntime));
    state.finishTime = estimate;
}

std::optional<std::uint32_t> ReplaySimulationRuntime::StuntsScore() const {
    if (!state_->stuntsEnabled) {
        return std::nullopt;
    }
    return state_->race.StuntsScore();
}

std::uint32_t
ReplaySimulationRuntime::DynamicCollisionCorpusCountForTesting(void) const {
    const CHmsCollisionManager::SGroup *group =
            state_->world.CollisionZone().GroupAtOrNull(
                    static_cast<std::uint32_t>(
                            CHmsItem::ECollisionGroup_Dynamic) -
                    1u);
    return group != nullptr ? group->NonStaticCorpusCount() : 0u;
}

bool ReplaySimulationRuntime::StepPhysicsKernelReferenceForTesting(
        const ReplayControlTick &tick) {
    State &state = *state_;
    if (state.definition == nullptr || state.phase != Phase::Idle) {
        return false;
    }
    CSceneVehicleCar &car = state.vehicle.Car();
    car.EnableAbsorbContactCallback(1);
    car.EnablePhysicsUpdates(true);
    CHmsItem *item = car.HmsItem();
    CHmsItem::CCallback *computeForcesCallback =
            item != nullptr
                    ? item->CallbackGet(
                              CHmsItem::ECallback_ComputeForces)
                    : nullptr;
    state.world.InstallEnvironment(
            state.environment, state.definition->environment, true);
    state.world.SetSimulationTime(tick);
    state.optimizedCpuVehicleForces.BeginTick(
            car,
            forevervalidator::simulation::
                    OptimizedCpuBinary32MathPath::X86Sse2,
            computeForcesCallback);
    state.world.StepOptimizedCpuNativeBinary32(
            state.optimizedCpuVehicleForces);
    return true;
}

ReplayDynaFrameState ReplaySimulationRuntime::CurrentFrame() const {
    return state_->body.CaptureCurrentFrame();
}

ReplayVehicleControlState ReplaySimulationRuntime::CurrentControls() const {
    const CSceneVehicleCar::SControlInput input =
            state_->vehicle.Car().ControlInput();
    return {input.lowSpeedGateA, input.lowSpeedGateB, input.steering};
}

ReplayRaceCameraVehicleState
ReplaySimulationRuntime::CurrentRaceCameraState() const {
    ReplayRaceCameraVehicleState result;
    const CSceneVehicleCar &car = state_->vehicle.Car();
    const CSceneVehicleCar::SVehicleCarState &physics =
            car.ReplayPhysicsState();
    result.signedSpeed = physics.forwardSpeed;
    result.turbo = physics.turboActive != 0u ? 1.0f : 0.0f;
    result.gearChanged =
            physics.engineControlState ==
            CSceneVehicleCarEngineControlState_GearShift;
    const std::size_t wheelCount =
            std::min<std::size_t>(car.WheelGetCount(), 4u);
    for (std::size_t index = 0u; index < wheelCount; ++index) {
        const CSceneVehicleCar::SSimulationWheel &wheel =
                car.WheelAt(static_cast<u32>(index));
        result.wheelContact[index] =
                wheel.currentPhysicsState.contactPresent;
        result.wheelHasSurface[index] =
                wheel.asyncState.contactPresent;
    }
    return result;
}

CSceneVehicleCar::SConditionState
ReplaySimulationRuntime::CurrentConditionState() const {
    return state_->vehicle.Car().ConditionState();
}


std::uint64_t ReplaySimulationRuntimeSemanticHash(
        const ReplaySimulationRuntime::RuntimeClone &clone) {
    ProjectionHash hash;
    hash.Add(clone.world);
    hash.Add(HashDynamicBody(clone.body));
    hash.Add(HashVehicle(clone.vehicle));
    hash.AddOptional(clone.finishTime);
    hash.Add(clone.firstStep);
    hash.Add(clone.stuntsEnabled);
    return hash.Value();
}

std::uint64_t ReplayRaceRuntimeSemanticHash(
        const CTrackManiaRace::RuntimeClone &clone) {
    return HashRace(clone);
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
    RuntimeClone clone;
    if (!CaptureRuntimeClone(clone)) {
        return std::nullopt;
    }
    return clone;
}

bool ReplaySimulationRuntime::CaptureRuntimeClone(
        RuntimeClone &clone) const {
    if (state_->phase != Phase::Idle || state_->definition == nullptr) {
        return false;
    }
    clone.world = state_->world.CaptureRuntimeClone();
    state_->body.CaptureRuntimeClone(clone.body);
    state_->vehicle.CaptureRuntimeClone(clone.vehicle);
    clone.finishTime = state_->finishTime;
    clone.firstStep = state_->firstStep;
    clone.stuntsEnabled = state_->stuntsEnabled;
    return true;
}

std::optional<ReplaySimulationRuntime::RuntimeClone>
ReplaySimulationRuntime::CaptureVehiclePrefixReferenceForTesting(
        float dt) {
    State &state = *state_;
    if (state.phase != Phase::Idle || state.definition == nullptr ||
        !(dt > 0.0f)) {
        return std::nullopt;
    }
    std::optional<RuntimeClone> before = CaptureRuntimeClone();
    if (!before.has_value()) {
        return std::nullopt;
    }
    CudaVehicleCpuReferenceAccess::IntegrateVehicle(
            state.vehicle.Car(), dt);
    std::optional<RuntimeClone> after = CaptureRuntimeClone();
    RestoreRuntimeClone(std::move(*before));
    return after;
}

std::optional<ReplaySimulationRuntime::RuntimeClone>
ReplaySimulationRuntime::CaptureVehicleForceReferenceForTesting(
        float dt) {
    State &state = *state_;
    if (state.phase != Phase::Idle || state.definition == nullptr ||
        !(dt > 0.0f)) {
        return std::nullopt;
    }
    std::optional<RuntimeClone> before = CaptureRuntimeClone();
    if (!before.has_value()) {
        return std::nullopt;
    }
    ReplayControlTick tick;
    tick.periodMs = before->world.schemePeriodMs;
    tick.timeMs = before->world.tickTimeMs;
    state.world.SetSimulationTime(tick);
    CudaVehicleCpuReferenceAccess::ComputeForces(
            state.vehicle.Car(), dt);
    std::optional<RuntimeClone> after = CaptureRuntimeClone();
    RestoreRuntimeClone(std::move(*before));
    return after;
}

std::optional<ReplaySimulationRuntime::RuntimeClone>
ReplaySimulationRuntime::CaptureCollisionSubstepReferenceForTesting(
        float dt) {
    State &state = *state_;
    if (state.phase != Phase::Idle || state.definition == nullptr ||
        !(dt > 0.0f)) {
        return std::nullopt;
    }
    std::optional<RuntimeClone> before = CaptureRuntimeClone();
    if (!before.has_value()) {
        return std::nullopt;
    }
    CSceneVehicleCar &car = state.vehicle.Car();
    CHmsItem *item = car.HmsItem();
    CHmsItem::CCallback *callback =
            item != nullptr
                    ? item->CallbackGet(
                              CHmsItem::ECallback_ComputeForces)
                    : nullptr;
    state.world.InstallEnvironment(
            state.environment, state.definition->environment, true);
    state.optimizedCpuVehicleForces.BeginTick(
            car,
            forevervalidator::simulation::
                    OptimizedCpuBinary32MathPath::X86Sse2,
            callback);
    state.world.Zone().ComputeCorpusForcesOptimizedCpuVehicle(
            &state.body.Corpus(), dt,
            state.optimizedCpuVehicleForces);
    state.body.Dyna().DoPreCollisionDynamic(dt);
    CHmsCollisionBuffer buffer;
    state.world.CollisionZone().
            DetectCollisionsCorpusOptimizedCpuNativeBinary32(
                    buffer, &state.body.Corpus());
    state.world.Zone().ComputeCollisionResponse(buffer);
    state.body.Dyna().DoPostCollisionDynamic();
    std::optional<RuntimeClone> after = CaptureRuntimeClone();
    RestoreRuntimeClone(std::move(*before));
    return after;
}

std::optional<ReplaySimulationRuntime::RuntimeClone>
ReplaySimulationRuntime::CapturePreCollisionReferenceForTesting(
        float dt) {
    State &state = *state_;
    if (state.phase != Phase::Idle || state.definition == nullptr ||
        !(dt > 0.0f)) {
        return std::nullopt;
    }
    std::optional<RuntimeClone> before = CaptureRuntimeClone();
    if (!before.has_value()) return std::nullopt;
    CSceneVehicleCar &car = state.vehicle.Car();
    CHmsItem *item = car.HmsItem();
    CHmsItem::CCallback *callback =
            item != nullptr
                    ? item->CallbackGet(
                              CHmsItem::ECallback_ComputeForces)
                    : nullptr;
    state.world.InstallEnvironment(
            state.environment, state.definition->environment, true);
    state.optimizedCpuVehicleForces.BeginTick(
            car,
            forevervalidator::simulation::
                    OptimizedCpuBinary32MathPath::X86Sse2,
            callback);
    state.world.Zone().ComputeCorpusForcesOptimizedCpuVehicle(
            &state.body.Corpus(), dt,
            state.optimizedCpuVehicleForces);
    state.body.Dyna().DoPreCollisionDynamic(dt);
    std::optional<RuntimeClone> after = CaptureRuntimeClone();
    RestoreRuntimeClone(std::move(*before));
    return after;
}

std::optional<ReplaySimulationRuntime::RuntimeClone>
ReplaySimulationRuntime::CaptureForcePassReferenceForTesting(float dt) {
    State &state = *state_;
    if (state.phase != Phase::Idle || state.definition == nullptr ||
        !(dt > 0.0f)) {
        return std::nullopt;
    }
    std::optional<RuntimeClone> before = CaptureRuntimeClone();
    if (!before.has_value()) return std::nullopt;
    CSceneVehicleCar &car = state.vehicle.Car();
    CHmsItem *item = car.HmsItem();
    CHmsItem::CCallback *callback =
            item != nullptr
                    ? item->CallbackGet(
                              CHmsItem::ECallback_ComputeForces)
                    : nullptr;
    state.world.InstallEnvironment(
            state.environment, state.definition->environment, true);
    state.optimizedCpuVehicleForces.BeginTick(
            car,
            forevervalidator::simulation::
                    OptimizedCpuBinary32MathPath::X86Sse2,
            callback);
    state.world.Zone().ComputeCorpusForcesOptimizedCpuVehicle(
            &state.body.Corpus(), dt,
            state.optimizedCpuVehicleForces);
    std::optional<RuntimeClone> after = CaptureRuntimeClone();
    RestoreRuntimeClone(std::move(*before));
    return after;
}

std::optional<std::vector<GmCollision>>
ReplaySimulationRuntime::CaptureCollisionReferenceForTesting(void) {
    State &state = *state_;
    if (state.phase != Phase::Idle || state.definition == nullptr) {
        return std::nullopt;
    }
    try {
        CHmsCollisionBuffer buffer;
        state.world.CollisionZone().
                DetectCollisionsCorpusOptimizedCpuNativeBinary32(
                        buffer, &state.body.Corpus());
        buffer.SortForCollisionResponse();
        std::vector<GmCollision> collisions;
        collisions.reserve(buffer.PhysicalCollisionCount());
        for (std::uint32_t index = 0u;
             index < buffer.PhysicalCollisionCount(); ++index) {
            const SHmsPhysicalCollision *collision =
                    buffer.PhysicalCollisionAtOrNull(index);
            if (collision != nullptr) {
                collisions.push_back(
                        static_cast<const GmCollision &>(*collision));
            }
        }
        return collisions;
    } catch (const std::bad_alloc &) {
        return std::nullopt;
    }
}

bool ReplaySimulationRuntime::
ApplyCollisionResponseReferenceForTesting(void) {
    State &state = *state_;
    if (state.phase != Phase::Idle || state.definition == nullptr) {
        return false;
    }
    state.world.InstallSimulationTimeAsCurrent();
    CHmsCollisionBuffer buffer;
    state.world.CollisionZone().
            DetectCollisionsCorpusOptimizedCpuNativeBinary32(
                    buffer, &state.body.Corpus());
    state.world.Zone().ComputeCollisionResponse(buffer);
    return true;
}

bool ReplaySimulationRuntime::PrepareStepForTesting(
        const ReplayControlTick &tick) {
    State &state = *state_;
    if (state.phase != Phase::Idle || state.definition == nullptr) {
        return false;
    }
    state.race.ClearAcceptedCheckpointEvents();
    if (!state.firstStep) {
        state.vehicle.PrepareStep(tick, state.body);
    }
    state.vehicle.Car().EnableAbsorbContactCallback(1);
    state.vehicle.Car().EnablePhysicsUpdates(
            !tick.actions.suppressVehicleForceCallbacks);
    state.world.SetSimulationTime(tick);
    return true;
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
    if (state_->optimizedCpuStaticTransforms != nullptr) {
        state_->optimizedCpuStaticTransforms->ClearRuntimeTemporalCandidates(
                state_->world.CollisionZone(),
                state_->body.Corpus().CollisionTree());
    }
    state_->firstStep = clone.firstStep;
    state_->stuntsEnabled = clone.stuntsEnabled;
    state_->finishTime = clone.finishTime;
    state_->phase = Phase::Idle;
}

ReplaySimulationRuntime::Phase
ReplaySimulationRuntime::CurrentPhase() const noexcept {
    return state_->phase;
}

std::optional<OptimizedCpuStaticSceneFingerprint>
ReplaySimulationRuntime::
        CaptureOptimizedCpuStaticSceneFingerprintForTesting(
                const CHmsCollisionManagerSZone &expectedPersistentZone)
                const noexcept {
    State &state = *state_;
    if (!forevervalidator::simulation::UsesOptimizedCpuFoundation(
                state.backend) ||
        state.phase != Phase::Idle ||
        state.optimizedCpuStaticTransforms == nullptr ||
        &state.world.CollisionZone() != &expectedPersistentZone ||
        !state.optimizedCpuStaticTransforms->IsFor(
                expectedPersistentZone)) {
        return std::nullopt;
    }
    return state.optimizedCpuStaticTransforms->
            CaptureSourceFingerprintForTesting();
}
