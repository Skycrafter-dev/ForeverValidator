#include "simulation/runtime/replay_vehicle_simulation.h"
#include <cstdint>
#include <memory>

#include "engine/physics/geometry/plug_surface_material_library.h"
#include "simulation/runtime/replay_vehicle_assembly.h"
#include "simulation/runtime/replay_validation_spawn.h"
#include "engine/game/trackmania_race.h"
ReplayVehicleSimulation::ReplayVehicleSimulation(CTrackManiaRace &race)
    : race_(race) {
  tuning_.ResetToDefaults();
  vehicleStruct_.ResetSimulationState();
  car_.BindWheelSurfaceObserver(wheelSurfaces_);
  race_.BindVehicle(&car_);
}

ReplayVehicleSimulation::~ReplayVehicleSimulation() {
  car_.DetachPhysicsItem();
  race_.BindVehicle(nullptr);
  wheelSurfaces_.Reset(&car_);
  car_.ClearWheelSurfaceObserver();
}

void ReplayVehicleSimulation::InstallActiveTuning() {
  tuningContainer_.SetActiveTuning(tuning_);
  car_.TuningsSet(&tuningContainer_);
}

void ReplayVehicleSimulation::ApplyControls(
    const ReplayVehicleControlState &controls) {
  car_.ApplyControlInput(
      {controls.lowSpeedGateA, controls.lowSpeedGateB, controls.steering});
}

void ReplayVehicleSimulation::ApplyWheelMaterialFromTuning() {
  CSceneVehicleCarTuning *activeTuning =
      car_.Tunings() != nullptr ? car_.Tunings()->ActiveTuning() : nullptr;
  if (activeTuning == nullptr || activeTuning->contactResponse.singleMaterial >=
                                     EPlugSurfaceMaterialId_Count) {
    return;
  }

  CPlugMaterial *material = PlugSurfaceMaterials().Default(
      activeTuning->contactResponse.singleMaterial);
  if (material == nullptr) {
    return;
  }

  for (u32 index = 0u; index < car_.WheelGetCount(); ++index) {
    CPlugTree *tree = car_.WheelAt(index).surfaceHandler.Tree();
    CPlugSurface *surface = tree != nullptr ? tree->Surface() : nullptr;
    if (surface == nullptr || surface->MaterialCount() != 1u) {
      continue;
    }
    surface->SetMaterialAt(0u, material);
  }
}

std::optional<ReplayDynaParameters>
ReplayVehicleSimulation::BuildDynaParameters() const {
  const u32 wheelCount = car_.WheelGetCount();
  if (wheelCount == 0u) {
    return std::nullopt;
  }

  GmVec3 minimum = car_.WheelAt(0u).surfaceHandler.RestPoint();
  GmVec3 maximum = minimum;
  float bottomYSum = minimum.y - car_.WheelAt(0u).rollingRadius;
  for (u32 index = 1u; index < wheelCount; ++index) {
    const CSceneVehicleCar::SSimulationWheel &wheel = car_.WheelAt(index);
    const GmVec3 point = wheel.surfaceHandler.RestPoint();
    if (point.x < minimum.x)
      minimum.x = point.x;
    if (point.y < minimum.y)
      minimum.y = point.y;
    if (point.z < minimum.z)
      minimum.z = point.z;
    if (maximum.x < point.x)
      maximum.x = point.x;
    if (maximum.y < point.y)
      maximum.y = point.y;
    if (maximum.z < point.z)
      maximum.z = point.z;
    bottomYSum = bottomYSum + (point.y - wheel.rollingRadius);
  }

  GmBoxAligned wheelBounds;
  wheelBounds.SetMinMax(minimum, maximum);

  ReplayDynaParameters parameters;
  parameters.mass = tuning_.bodyAirResponse.solidPhysicalMass;
  parameters.linearDampingScale = 0.0f;
  parameters.angularDampingScale = 0.0f;
  parameters.maxStepDistance =
      tuning_.bodyAirResponse.solidPhysicalResponseCoefB;
  parameters.forceScale = tuning_.bodyAirResponse.groundedSolidFeedback1;
  parameters.localCenterOfMass = wheelBounds.center;
  parameters.localCenterOfMass.y =
      (1.0f / static_cast<float>(wheelCount)) * bottomYSum +
      tuning_.bodyAirResponse.solidCenterYOffset;
  parameters.localCenterOfMass.z =
      wheelBounds.center.z +
      tuning_.bodyAirResponse.solidCenterZHalfExtentScale *
          wheelBounds.halfExtents.z;

  const float width = tuning_.bodyAirResponse.solidInertiaBoxSize.x * 2.0f;
  const float height = tuning_.bodyAirResponse.solidInertiaBoxSize.y * 2.0f;
  const float length = 2.0f * tuning_.bodyAirResponse.solidInertiaBoxSize.z;
  const float inertiaScale =
      (1.0f / tuning_.bodyAirResponse.solidInertiaMass) * 12.0f;
  parameters.inverseBodyInertia.SetDiagonal(
      inertiaScale / (height * height + length * length),
      inertiaScale / (length * length + width * width),
      inertiaScale / (width * width + height * height));
  return parameters;
}

void ReplayVehicleSimulation::ApplyTransitionActions(
    const ReplayControlTick &tick, ReplayVehicleBody &body) {
  ApplyControls(tick.controls);
  if (tick.actions.establishRaceSpawn) {
    const GmIso4 initialSpawn = body.CaptureCurrentFrame().Location();
    car_.EstablishRaceSpawnFrame(initialSpawn);
    race_.SetInitialSpawnLocation(initialSpawn);
  }
  if (tick.actions.enableRaceSimulation) {
    car_.BeginRaceSimulation();
  }
  if (tick.actions.resetAtRaceStart) {
    car_.SetTurboRouletteTickOrigin(tick.timeMs);
    car_.VehicleBlockSpeedSet(0);
    car_.VehicleReset();
    for (u32 index = 0u; index < car_.WheelGetCount(); ++index) {
      wheelSurfaces_.UpdateLocalIsoFromWheel(index, &car_.WheelAt(index));
    }
  }
}

ReplayVehiclePreparationResult ReplayVehicleSimulation::Start(
    const ReplaySimulationDefinition &definition,
    const ReplayControlTick &tick, ReplayVehicleBody &body,
    bool staticSceneReady) {
  const VehicleSimulationDefinition &vehicleDefinition = definition.vehicle;
  InstallReplayVehicleInitialParameters(vehicleDefinition.initialParameters,
                                        car_);
  ApplyControls(tick.controls);
  if (tick.actions.establishRaceSpawn) {
    const GmIso4 initialSpawn = body.CaptureCurrentFrame().Location();
    car_.EstablishRaceSpawnFrame(initialSpawn);
    race_.SetInitialSpawnLocation(initialSpawn);
  }
  if (tick.actions.enableRaceSimulation) {
    car_.BeginRaceSimulation();
  }
  tuningInstallation_.Install(vehicleDefinition.tuning, tuning_,
                              vehicleStruct_);
  InstallReplayVehicleWheels(vehicleDefinition.wheels, car_, vehicleStruct_);
  InstallActiveTuning();

  for (u32 index = 0u; index < car_.WheelGetCount(); ++index) {
    CSceneVehicleCar::SSimulationWheel &wheel = car_.WheelAt(index);
    wheelSurfaces_.BeginWheelFromReplayState(index, &wheel);
    wheel.realTimeState.peerCorpusId = {};
  }
  if (tick.actions.resetAtRaceStart) {
    car_.SetTurboRouletteTickOrigin(tick.timeMs);
    car_.VehicleBlockSpeedSet(0);
    car_.VehicleReset();
    for (u32 index = 0u; index < car_.WheelGetCount(); ++index) {
      wheelSurfaces_.UpdateLocalIsoFromWheel(index, &car_.WheelAt(index));
    }
  }

  if (staticSceneReady) {
    if (!collisionModel_.Build(vehicleDefinition.collisionModel)) {
      return ReplayVehiclePreparationResult::CollisionModelConstructionFailed;
    }
    std::unique_ptr<CPlugTree> collisionTree = collisionModel_.ReleaseTree();
    if (collisionTree == nullptr) {
      return ReplayVehiclePreparationResult::CollisionTreeUnavailable;
    }
    body.Solid().SetTree(collisionTree.release(), 0);
    wheelSurfaces_.BindCollisionShapes(collisionModel_,
                                       vehicleDefinition.wheels, car_);
  }

  car_.AttachPhysicsItem(body.Item());
  car_.BindVehicleStruct(vehicleStruct_);
  materials_.Install(car_, vehicleDefinition.materials);
  turboSound_.Install(car_);
  car_.OnEnterScene();
  ApplyWheelMaterialFromTuning();
  body.InstallPhysicalParameters(body.CaptureDynaParameters());
  return ReplayVehiclePreparationResult::Ready;
}

void ReplayVehicleSimulation::PrepareStep(
    const ReplayControlTick &tick, ReplayVehicleBody &body) {
  ApplyTransitionActions(tick, body);
  body.InstallPhysicalParameters(body.CaptureDynaParameters());
}

bool ReplayVehicleSimulation::Respawn(ReplayVehicleBody &body) {
  if (!race_.HasRespawnLocation()) {
    return false;
  }

  const CSceneVehicleCar::SControlInput controls = car_.ControlInput();
  car_.VehicleReset();

  CHmsDyna &dyna = body.Dyna();
  dyna.CurrentState().Reset();
  dyna.WriteState().Reset();
  dyna.TemporaryState().Reset();
  dyna.ClearCollisionReplacements();
  car_.VehicleBlockSpeed2Set(0);

  const std::optional<ReplayDynaParameters> parameters = BuildDynaParameters();
  if (parameters.has_value()) {
    body.InstallDynaParameters(*parameters);
  }
  dyna.SetLocation(race_.RespawnLocation());
  car_.ApplyControlInput(controls);
  return true;
}

std::optional<u32> ReplayVehicleSimulation::FinishTimeMs() const {
  const ReplayRaceProgress &progress = race_.Progress();
  return progress.raceCompleted
             ? std::optional<u32>(progress.lastPrepareTimeMs)
             : std::nullopt;
}

ReplayVehicleSimulation::RuntimeClone
ReplayVehicleSimulation::CaptureRuntimeClone() const {
    return {car_.CaptureRuntimeClone(),
            wheelSurfaces_.CaptureRuntimeClone()};
}

bool ReplayVehicleSimulation::CanRestoreRuntimeClone(
        const RuntimeClone &clone) const noexcept {
    return car_.CanRestoreRuntimeClone(clone.car) &&
           wheelSurfaces_.CanRestoreRuntimeClone(
                   clone.wheelSurfaces, car_);
}

void ReplayVehicleSimulation::RestoreRuntimeClone(
        const RuntimeClone &clone) noexcept {
    car_.RestoreRuntimeClone(clone.car);
    wheelSurfaces_.RestoreRuntimeClone(clone.wheelSurfaces, car_);
}
