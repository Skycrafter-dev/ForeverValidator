// Vehicle lifecycle, state interpolation, and reset invariants.

#include "engine/scene/scene_vehicle_car_internal.h"
using namespace SceneVehicleCarDynamics;

unsigned long CSceneVehicleCar::s_TurboRoulettePeriodMs = 1000ul;

void CSceneVehicleCar::SSimulationWheel::SRealTimeState::ResetSimulationState(
    void) {
  damperAbsorb = 0.0f;
  damperVelocity = 0.0f;
  maxReplacementY = 0.0f;
  visualRotation.SetIdentity();
  contactFrame.SetIdentity();
  latestContactPoint = {};
  wheelAngularSpeed = 0.0f;
  contactPresent = false;
  contactMaterial = EPlugSurfaceMaterialId_Concrete;
  slipping = false;
  peerZAxisInCarLocal = {};
  peerCorpusId = {};
  contactNormalSampleCount = 0u;
  accumulatedContactNormal = {};
  wheelSpinAngle = 0.0f;
  currentVisualSteerAngle = 0.0f;
  targetVisualSteerAngle = 0.0f;
  rejectedNormalContact = false;
  rejectedNormalContactPoint = {};
}

void CSceneVehicleCar::SSimulationWheel::ResetSimulationState(void) {
  killsLateralSpeedOnContact = false;
  axle = VehicleWheelAxle::Rear;
  rollingRadius = 1.0f;
  surfaceHandler = CSceneVehicle::SSurfaceHandler{};
  forceApplicationPoint = {};
  realTimeState.ResetSimulationState();
  previousPhysicsState = {};
  currentPhysicsState = {};
  previousAsyncState = {};
  asyncState = {};
}

void CSceneVehicleCar::SSimulationWheel::ClearContactCarryStamp() {
  realTimeState.contactMaterial = EPlugSurfaceMaterialId_Concrete;
}

CSceneVehicleCar::SEngine::SEngine(void)
    : engineInputMax(11000.0f), lowSpeedFeedbackGateScale(1.0f),
      lowSpeedFeedbackFrictionScale(0.0f), lowSpeedFeedbackForce(0.0f),
      engineInputMemory(0.0f), targetTransmissionInput(0.0f),
      slipRpmScale(1.0f), shiftCooldown(0.0f), useLowSpeedGateB(0u),
      gearIndex(1) {}

CSceneVehicleCar::SDynaPart::SDynaPart(void) : id(), spring() {
  spring.stiffness = 120.0f;
  spring.damping = 3.0f;
  spring.ClearVals();
}

CSceneVehicleCar::SDynaPart::~SDynaPart(void) = default;

CSceneVehicleCar::CSceneVehicleCar(void)
    : vehicleStruct{}, turboSoundSource{}, wheels{}, controls{}, feedback{},
      linearSpeedCap(277.77777f), integration{}, frameHistory{}, engine{},
      reverseGearSpeedThreshold(10.0f), turbo{}, airControl{}, contacts{},
      radiusSteering{}, slipMemory{}, gearedDrive{}, lastComputeForcesTick(0u),
      dynaParts{}, forceAccumulators{} {
  wheels.reserve(4u);
  controls.specialContactResponseMode =
      CSceneVehicleCarSpecialContactMode_ImpulseFromForce;
  feedback.forwardSpring.stiffness = 150.0f;
  feedback.forwardSpring.damping = 8.0f;
  feedback.forwardSpring.ClearVals();
  feedback.sideSpring.stiffness = 150.0f;
  feedback.sideSpring.damping = 8.0f;
  feedback.sideSpring.ClearVals();
  feedback.springDriveLimit = 10.0f;
  feedback.springVelocityLimit = 15.0f;
  feedback.springValueLimit = 1.0f;
  gearedDrive.burnoutBaseRadius = 1.0f;
  gearedDrive.burnoutTargetRadius = 2.0f;
  gearedDrive.activeSteerSlowDownScale = 1.0f;
  gearedDrive.wheelLongitudinalSpan = 1.0f;
  VehicleReset();
}

CSceneVehicleCar::~CSceneVehicleCar(void) { DetachPhysicsItem(); }

CSceneVehicleCar::RuntimeClone
CSceneVehicleCar::CaptureRuntimeClone(void) const {
  RuntimeClone clone;
  clone.vehicle = CSceneVehicle::CaptureRuntimeClone();
  clone.wheels = wheels;
  for (SSimulationWheel &wheel : clone.wheels) {
    wheel.surfaceHandler.BindTree(nullptr);
  }
  clone.controls = controls;
  clone.feedback = feedback;
  clone.linearSpeedCap = linearSpeedCap;
  clone.integration = integration;
  clone.frameHistory = frameHistory;
  clone.engine = engine;
  clone.reverseGearSpeedThreshold = reverseGearSpeedThreshold;
  clone.turbo = turbo;
  clone.airControl = airControl;
  clone.contacts = contacts;
  clone.radiusSteering = radiusSteering;
  clone.slipMemory = slipMemory;
  clone.gearedDrive = gearedDrive;
  clone.lastComputeForcesTick = lastComputeForcesTick;
  for (std::size_t index = 0u; index < dynaParts.size(); ++index) {
    clone.dynaPartSprings[index] = dynaParts[index].spring;
  }
  clone.forceAccumulators = forceAccumulators;
  return clone;
}

bool CSceneVehicleCar::CanRestoreRuntimeClone(
    const RuntimeClone &clone) const noexcept {
  return clone.wheels.size() == wheels.size();
}

void CSceneVehicleCar::RestoreRuntimeClone(
    const RuntimeClone &clone) noexcept {
  CSceneVehicle::RestoreRuntimeClone(clone.vehicle);
  for (std::size_t index = 0u; index < wheels.size(); ++index) {
    CPlugTree *targetTree = wheels[index].surfaceHandler.Tree();
    wheels[index] = clone.wheels[index];
    wheels[index].surfaceHandler.BindTree(targetTree);
    if (targetTree != nullptr) {
      targetTree->SetLocation(wheels[index].surfaceHandler.CurrentPose());
    }
  }
  controls = clone.controls;
  feedback = clone.feedback;
  linearSpeedCap = clone.linearSpeedCap;
  integration = clone.integration;
  frameHistory = clone.frameHistory;
  engine = clone.engine;
  reverseGearSpeedThreshold = clone.reverseGearSpeedThreshold;
  turbo = clone.turbo;
  airControl = clone.airControl;
  contacts = clone.contacts;
  radiusSteering = clone.radiusSteering;
  slipMemory = clone.slipMemory;
  gearedDrive = clone.gearedDrive;
  lastComputeForcesTick = clone.lastComputeForcesTick;
  for (std::size_t index = 0u; index < dynaParts.size(); ++index) {
    dynaParts[index].spring = clone.dynaPartSprings[index];
  }
  forceAccumulators = clone.forceAccumulators;
}

void CSceneVehicleCar::SetTurboRouletteTickOrigin(unsigned long tick) {
  turbo.rouletteTickOrigin = static_cast<u32>(tick);
}

void CSceneVehicleCar::HmsComputeForces(float dt) { ComputeForces(dt); }

void CSceneVehicleCar::HmsAfterContacts(void) { AfterContacts(); }

void CSceneVehicleCar::SetWheelCount(u32 wheelCount) {
  wheels.resize(wheelCount);
}

CSceneVehicleCar::SSimulationWheel &CSceneVehicleCar::WheelAt(u32 wheelIndex) {
  return wheels[wheelIndex];
}

const CSceneVehicleCar::SSimulationWheel &
CSceneVehicleCar::WheelAt(u32 wheelIndex) const {
  return wheels[wheelIndex];
}

void CSceneVehicleCar::BindTurboSound(CSceneSoundSource &source,
                                      CHmsSoundSource &sound) {
  source.AttachSound(sound);
  turboSoundSource = std::ref(source);
}

void CSceneVehicleCar::BindWheelSurfaceObserver(
    CSceneVehicleCarWheelSurfaceObserver &observer) {
  wheelSurfaceObserver = &observer;
}

void CSceneVehicleCar::ClearWheelSurfaceObserver(void) {
  wheelSurfaceObserver = nullptr;
}

void CSceneVehicleCar::SVehicleCarState::Reset(void) {
  VehicleStateReset();
  engineInputMemory = 0.0f;
  bodyContactVerticalAngle = 0.0f;
  airControlRefreshMemory = false;
  bodyContactHorizontalAngle = 0.0f;
  engineControlState = CSceneVehicleCarEngineControlState_Steady;
  shiftDirection = CSceneVehicleCarShiftDirection_Up;
  hasWheelContact = false;
  hasBodyContact = false;
  bodyContactZPositive = false;
  noGroundFrictionGuard = false;
}

void CSceneVehicleCar::SVehicleCarState::Set(
    const CSceneVehicleCar::SVehicleCarState &rhs) {
  VehicleStateSet(rhs);
  engineInputMemory = rhs.engineInputMemory;
  airControlRefreshMemory = rhs.airControlRefreshMemory;
  engineControlState = rhs.engineControlState;
  shiftDirection = rhs.shiftDirection;
  hasWheelContact = rhs.hasWheelContact;
  hasBodyContact = rhs.hasBodyContact;
  bodyContactVerticalAngle = rhs.bodyContactVerticalAngle;
  bodyContactZPositive = rhs.bodyContactZPositive;
  bodyContactHorizontalAngle = rhs.bodyContactHorizontalAngle;
  noGroundFrictionGuard = rhs.noGroundFrictionGuard;
}

void CSceneVehicleCar::SVehicleCarState::SetBlend(
    const CSceneVehicleCar::SVehicleCarState &from,
    const CSceneVehicleCar::SVehicleCarState &to, float blend) {
  VehicleStateSetBlend(from, to, blend);
  engineInputMemory =
      from.engineInputMemory * (1.0f - blend) + to.engineInputMemory * blend;
  airControlRefreshMemory = to.airControlRefreshMemory;
  engineControlState = to.engineControlState;
  shiftDirection = to.shiftDirection;
  hasWheelContact = to.hasWheelContact;
  hasBodyContact = to.hasBodyContact;
  bodyContactVerticalAngle = to.bodyContactVerticalAngle;
  bodyContactZPositive = to.bodyContactZPositive;
  bodyContactHorizontalAngle = to.bodyContactHorizontalAngle;
  noGroundFrictionGuard = to.noGroundFrictionGuard;
}

const CSceneVehicle::SVehicleState &
CSceneVehicleCar::VehicleStateAsyncGet(void) const {
  return frameHistory.asyncCurrent;
}

const CSceneVehicle::SVehicleState &
CSceneVehicleCar::VehicleStatePrevAsyncGet(void) const {
  return frameHistory.asyncPrevious;
}

void CSceneVehicleCar::ComputeAsyncState(void) {
  frameHistory.asyncPrevious.Set(frameHistory.asyncCurrent);
  for (SSimulationWheel &wheel : wheels) {
    wheel.previousAsyncState = wheel.asyncState;
  }

  const float blend = VehicleStateComputeBlendVal();
  if (blend <= 0.0f) {
    frameHistory.asyncCurrent.Set(frameHistory.physicsPrevious);
  } else if (blend >= 1.0f) {
    frameHistory.asyncCurrent.Set(frameHistory.physicsCurrent);
  } else {
    frameHistory.asyncCurrent.SetBlend(frameHistory.physicsPrevious,
                                       frameHistory.physicsCurrent, blend);
  }

  for (SSimulationWheel &wheel : wheels) {
    const float previousAsyncSpin = wheel.asyncState.wheelSpinAngle;
    wheel.asyncState.SetBlend(wheel.previousPhysicsState,
                              wheel.currentPhysicsState, blend);
    if (IsNetworked() && AsyncPeriodSeconds() > 0.0f &&
        wheel.rollingRadius != 0.0f) {
      const float advancedSpin =
          previousAsyncSpin +
          (frameHistory.asyncCurrent.forwardSpeed / wheel.rollingRadius) *
              AsyncPeriodSeconds();
      float wrappedSpin =
          CIfmod(advancedSpin, SceneVehicleMath::WheelSpinAnglePeriod);
      if (wrappedSpin < 0.0f) {
        wrappedSpin = wrappedSpin + SceneVehicleMath::WheelSpinAnglePeriod;
      }
      wheel.asyncState.wheelSpinAngle = wrappedSpin;
    }

    wheel.asyncState.localSurfacePoint = wheel.surfaceHandler.CurrentPoint();
    wheel.asyncState.localSurfacePoint.y =
        wheel.asyncState.localSurfacePoint.y - wheel.rollingRadius;
    wheel.asyncState.worldSurfacePoint.SetMult(
        wheel.asyncState.localSurfacePoint,
        frameHistory.asyncCurrent.corpusIso);
    wheel.asyncState.visualRotation = wheel.surfaceHandler.CurrentRotation();
    wheel.asyncState.visualRotation.RotateY(
        wheel.asyncState.currentVisualSteerAngle);
  }
}

void CSceneVehicleCar::VehicleUpdateAsync(void) {
  if (IsUpdateAsync()) {
    ComputeAsyncState();
  }
  CSceneVehicle::VehicleUpdateAsync();
}

void CSceneVehicleCar::VehicleBlockSpeedSet(int isBlocked) {
  integration.speedBlocked = isBlocked != 0;
}

void CSceneVehicleCar::VehicleBlockSpeed2Set(int isBlocked) {
  integration.speedBlockedSecondary = isBlocked != 0;
}

void CSceneVehicleCar::VehicleFreeWheelingSet(int isFreeWheeling) {
  controls.forcedLowSpeedFriction = isFreeWheeling != 0;
}

void CSceneVehicleCar::BeginRaceSimulation(void) {
  integration.updateWheelVisuals = true;
  integration.integrateWheels = true;
  integration.integrateEngine = true;
  integration.zeroHorizontalSpeed = false;
  integration.speedBlocked = false;
  integration.speedBlockedSecondary = false;
}

CSceneVehicleCar::SControlInput CSceneVehicleCar::ControlInput(void) const {
  return {controls.lowSpeedGateA, controls.lowSpeedGateB,
          controls.steeringControl};
}

void CSceneVehicleCar::ApplyControlInput(const SControlInput &input) {
  controls.lowSpeedGateA = input.lowSpeedGateA;
  controls.lowSpeedGateB = input.lowSpeedGateB;
  controls.steeringControl = input.steering;
  frameHistory.physicsCurrent.lowSpeedGateA = input.lowSpeedGateA;
  frameHistory.physicsCurrent.lowSpeedGateB = input.lowSpeedGateB;
  frameHistory.physicsCurrent.steeringControl = input.steering;
}

void CSceneVehicleCar::ConfigureSimulationLimits(float maximumLinearSpeed,
                                                 float reverseTransitionSpeed) {
  linearSpeedCap = maximumLinearSpeed;
  reverseGearSpeedThreshold = reverseTransitionSpeed;
}

void CSceneVehicleCar::EstablishRaceSpawnFrame(const GmIso4 &spawnFrame) {
  gearedDrive.frameIso = spawnFrame;
  slipMemory.active = false;
  slipMemory.lastTick = UINT32_MAX;
  slipMemory.startTick = UINT32_MAX;
}

CSceneVehicleCarTuning *CSceneVehicleCar::ActiveTuningOrNull(void) const {
  CSceneVehicleTunings *container = Tunings();
  return container != nullptr ? container->ActiveTuning() : nullptr;
}

void CSceneVehicleCar::OnEnterScene(void) {
  CSceneVehicle::OnEnterScene();
  UpdateParamsFromTuning();

  CHmsItem *item = HmsItem();
  CHmsDyna *dyna = item != nullptr ? item->FirstDyna() : nullptr;
  if (dyna == nullptr) {
    return;
  }

  const CSceneVehicleCarTuning *tuning = ActiveTuningOrNull();
  if (tuning == nullptr) {
    return;
  }
  const float maximum =
      tuning->contactResponse.pointImpulseAngularSpeedMax;
  dyna->SetAngularSpeedLimit(maximum > ScalarEpsilon
                                 ? std::optional<float>(maximum)
                                 : std::nullopt);
}

void CSceneVehicleCar::ResetPlayerControls(void) {
  controls.lowSpeedGateA = 0.0f;
  controls.lowSpeedGateB = 0.0f;
  controls.steeringControl = 0.0f;
  controls.currentSteering = 0.0f;
  controls.specialContactResponseGate = 0.0f;
  contacts.specialContactImpulseCooldownUntil = 0;
  controls.forcedLowSpeedFriction = false;
}

void CSceneVehicleCar::ResetFrameHistory(void) {
  frameHistory.asyncCurrent.Reset();
  frameHistory.asyncPrevious.Reset();
  frameHistory.physicsPrevious.Reset();
  frameHistory.physicsCurrent.Reset();

  frameHistory.physicsCurrent.vehicleEvent0Value =
      VehicleEventValue(EVehicleEvent_None);
  frameHistory.physicsPrevious.vehicleEvent0Value =
      VehicleEventValue(EVehicleEvent_None);
  frameHistory.physicsCurrent.waterSplashEventCounter =
      VehicleEventValue(EVehicleEvent_WaterSplash);
  frameHistory.physicsPrevious.waterSplashEventCounter =
      VehicleEventValue(EVehicleEvent_WaterSplash);
}

void CSceneVehicleCar::ResetAirControl(void) {
  airControl.refreshMemory = false;
  airControl.memoryTick = 0;
  airControl.memoryAngular = SceneVehicleMath::Zero();
}

void CSceneVehicleCar::ResetTurbo(void) {
  turbo.progressRatio = 0.0f;
  turbo.type = ETurboType_Inactive;
  turbo.impulseScale = 0.0f;
}

void CSceneVehicleCar::ResetRadiusSteering(void) {
  radiusSteering.steerAngle = 0.0f;
  radiusSteering.phase = CSceneVehicleCarRadiusSteeringPhase_Idle;
  radiusSteering.previousSteerSign = 0.0f;
}

void CSceneVehicleCar::ResetGearedDrive(void) {
  slipMemory.active = false;
  slipMemory.lastTick = UINT32_MAX;
  slipMemory.startTick = UINT32_MAX;
  gearedDrive.wheelSpeedOverrideActive = false;
  gearedDrive.burnoutStartTick = UINT32_MAX;
  gearedDrive.burnoutExitStartTick = UINT32_MAX;
  gearedDrive.frameIso.SetIdentity();
  gearedDrive.burnoutContactNormal = SceneVehicleMath::Zero();
  gearedDrive.localSpeed = SceneVehicleMath::Zero();
  gearedDrive.burnoutPhase = CSceneVehicleCarBurnoutPhase_Inactive;
  gearedDrive.engineState = CSceneVehicleCarEngineControlState_Steady;
  gearedDrive.inputWindowExceeded = false;
  gearedDrive.wheelDriveSpeedInhibited = false;
}

void CSceneVehicleCar::ResetImpacts(void) {
  contacts.bodyContactPresent = false;
  contacts.lateralSlowDownContactActive = false;
  contacts.lateralSlowDownLastTick = UINT32_MAX;
  controls.noGroundFrictionGuard = false;
  contacts.frontWheelImpactState = CSceneVehicleCarImpactState_None;
  contacts.rearWheelImpactState = CSceneVehicleCarImpactState_None;
  contacts.bodyImpactState = CSceneVehicleCarImpactState_None;
  contacts.lastWheelContactMaterial = EPlugSurfaceMaterialId_Concrete;
  contacts.lastBodyContactMaterial = EPlugSurfaceMaterialId_Concrete;
  contacts.peakRearWheelImpactState = CSceneVehicleCarImpactState_None;
  contacts.peakFrontWheelImpactState = CSceneVehicleCarImpactState_None;
  contacts.peakBodyImpactState = CSceneVehicleCarImpactState_None;
  contacts.frontWheelImpactBucket = 0.0f;
  contacts.peakWheelImpactMaterial = EPlugSurfaceMaterialId_Concrete;
  contacts.rearWheelImpactBucket = 0.0f;
  contacts.peakBodyImpactMaterial = EPlugSurfaceMaterialId_Concrete;
  contacts.bodyImpactBucket = 0.0f;
  lastComputeForcesTick = 0;
}

void CSceneVehicleCar::ResetForceAccumulators(void) {
  forceAccumulators.force = SceneVehicleMath::Zero();
  forceAccumulators.impulse = SceneVehicleMath::Zero();
  contacts.bodyContactPointSum = SceneVehicleMath::Zero();
  contacts.bodyContactNormalSum = SceneVehicleMath::Zero();
  contacts.bodyContactCount = 0;
  contacts.wheelContactCount = 0;
  feedback.surfaceAccumulator = 0.0f;
}

void CSceneVehicleCar::ResetDynaParts(void) {
  for (SDynaPart &part : dynaParts) {
    part.Reset();
  }
}

void CSceneVehicleCar::ResetSolidFeedback(void) {
  CHmsItem *item = HmsItem();
  CPlugSolid *solid = item != nullptr ? item->Solid() : nullptr;
  if (solid == nullptr) {
    return;
  }

  CSceneVehicleCarTuning *tuning = ActiveTuningOrNull();
  if (tuning == nullptr) {
    return;
  }

  solid->Physical().Parameters().vehicleContactFeedbackScale =
      tuning->bodyAirResponse.groundedSolidFeedback1;
  solid->Physical().Parameters().linearFluidFriction = 0.0f;
}

void CSceneVehicleCar::VehicleReset() {
  ResetPlayerControls();
  CSceneVehicle::VehicleReset();
  ResetFrameHistory();
  ResetTurbo();
  ResetAirControl();
  ResetRadiusSteering();
  (slipMemory.steeringMemoryTick = UINT32_MAX);
  ResetGearedDrive();
  ResetImpacts();
  ResetForceAccumulators();
  for (SSimulationWheel &wheel : wheels) {
    WheelReset(wheel);
  }

  engine.Reset();
  ResetDynaParts();
  ResetSolidFeedback();
}

void CSceneVehicleCar::SDynaPart::Reset(void) { spring.ClearVals(); }

void CSceneVehicleCar::SEngine::Reset(void) {
  useLowSpeedGateB = false;
  engineInputMemory = 0.0f;
  gearIndex = 1;
  targetTransmissionInput = 0.0f;
  lowSpeedFeedbackForce = 0.0f;
  shiftCooldown = 0.0f;
  slipRpmScale = 1.0f;
}

void CSceneVehicleCar::SSimulationWheel::SState::Reset(void) {
  // The material identity survives frame-state resets by design.
  damperAbsorb = 0.0f;
  wheelSpinAngle = 0.0f;
  currentVisualSteerAngle = 0.0f;
  contactPresent = false;
  slipping = false;
  worldSurfacePoint = {};
  localSurfacePoint = {};
  visualRotation.SetIdentity();
  rejectedNormalContact = false;
  rejectedNormalContactPoint = {};
}

void OrderWindowedValues(float &first, float &second, float period) {
  float &lower = second < first ? second : first;
  float &upper = second < first ? first : second;
  const float wrappedLower = lower + period;
  if (wrappedLower - upper <= upper - lower) {
    lower = wrappedLower;
  }
}

void CSceneVehicleCar::SSimulationWheel::SState::SetBlend(
    const CSceneVehicleCar::SSimulationWheel::SState &from,
    const CSceneVehicleCar::SSimulationWheel::SState &to, float blend) {
  const float fromBlend = 1.0f - blend;
  const auto lerp = [fromBlend, blend](float a, float b) {
    return (a * fromBlend) + (b * blend);
  };
  *this = from;
  damperAbsorb = lerp(from.damperAbsorb, to.damperAbsorb);
  float fromSpin = from.wheelSpinAngle;
  float toSpin = to.wheelSpinAngle;
  OrderWindowedValues(fromSpin, toSpin, SceneVehicleMath::WheelSpinAnglePeriod);
  wheelSpinAngle = lerp(fromSpin, toSpin);
  currentVisualSteerAngle =
      lerp(from.currentVisualSteerAngle, to.currentVisualSteerAngle);
  contactPresent = from.contactPresent && to.contactPresent;
  slipping = from.slipping && to.slipping;
}

void CSceneVehicleCar::BuildPointImpulseTorqueSeed(
    GmVec3 &outTorqueSeed, const GmVec3 &worldComToPoint,
    const GmVec3 &worldImpulse) {
  outTorqueSeed.x =
      worldComToPoint.y * worldImpulse.z - worldComToPoint.z * worldImpulse.y;
  outTorqueSeed.y =
      worldComToPoint.z * worldImpulse.x - worldImpulse.z * worldComToPoint.x;
  outTorqueSeed.z =
      worldImpulse.y * worldComToPoint.x - worldComToPoint.y * worldImpulse.x;
}

void CSceneVehicleCar::BuildWorldComToImpulsePoint(GmVec3 &outWorldComToPoint,
                                                   const GmVec3 &worldPoint,
                                                   const GmVec3 &worldCom) {
  outWorldComToPoint.x = worldPoint.x - worldCom.x;
  outWorldComToPoint.y = worldPoint.y - worldCom.y;
  outWorldComToPoint.z = worldPoint.z - worldCom.z;
}

void CSceneVehicleCar::ScaleAndAddPointImpulseAngularDelta(
    CHmsDyna::CHmsStateDyna &state, GmVec3 &pointImpulseAngularDelta,
    const CSceneVehicleCarTuning &tuning) {
  // Scale the full impulse response, then apply the additional yaw response.
  float angularScale = tuning.contactResponse.pointImpulseAngularScale;
  pointImpulseAngularDelta.x = angularScale * pointImpulseAngularDelta.x;
  pointImpulseAngularDelta.y = angularScale * pointImpulseAngularDelta.y;
  pointImpulseAngularDelta.z = angularScale * pointImpulseAngularDelta.z;
  pointImpulseAngularDelta.y =
      (tuning.contactResponse.pointImpulseAngularYScale *
       pointImpulseAngularDelta.y);
  state.angularSpeed.x = state.angularSpeed.x + pointImpulseAngularDelta.x;
  state.angularSpeed.y = state.angularSpeed.y + pointImpulseAngularDelta.y;
  state.angularSpeed.z = state.angularSpeed.z + pointImpulseAngularDelta.z;
}

CSceneVehicleCar::SSimulationWheel::SSimulationWheel(void) {
  ResetSimulationState();
}

CSceneVehicleCar::SSimulationWheel::~SSimulationWheel(void) = default;
