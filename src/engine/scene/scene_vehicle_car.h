#pragma once

#include "engine/core/engine_types.h"
#include <array>
#include <functional>
#include <optional>
#include <vector>

#include "engine/core/func_keys_real.h"
#include "engine/physics/collision/hms_collision.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_dyna.h"
#include "engine/rendering/plug_material.h"
#include "engine/scene/scene_object.h"
#include "engine/scene/scene_vehicle.h"
#include "engine/scene/scene_vehicle_car_tuning.h"
#include "engine/scene/scene_vehicle_car_types.h"
#include "engine/scene/scene_vehicle_material.h"
#include "engine/scene/scene_vehicle_struct.h"
#include "engine/game/vehicle_wheel_axle.h"
void OrderWindowedValues(float &first, float &second, float period);

class CSceneVehicleCarWheelSurfaceObserver;
struct OptimizedCpuModel3VehicleForceAccess;

class CSceneVehicleCar : public CSceneVehicle {
public:
  struct SSimulationWheel;

  enum ETurboType {
    ETurboType_Inactive = 0,
    ETurboType_Direct = 1,
    ETurboType_Roulette = 2,
  };

  struct SControls {
    float lowSpeedGateA = 0.0f;
    float lowSpeedGateB = 0.0f;
    float steeringControl = 0.0f;
    float specialContactResponseGate = 0.0f;
    float currentSteering = 0.0f;
    bool forcedLowSpeedFriction = false;
    CSceneVehicleCarSpecialContactMode specialContactResponseMode =
        CSceneVehicleCarSpecialContactMode_None;
    bool noGroundFrictionGuard = false;
  };

  struct SControlInput {
    float lowSpeedGateA = 0.0f;
    float lowSpeedGateB = 0.0f;
    float steering = 0.0f;
  };

  struct SFeedback {
    GmSpring<float> forwardSpring;
    GmSpring<float> sideSpring;
    float ramp0 = 0.0f;
    float ramp1 = 0.0f;
    float springDriveLimit = 0.0f;
    float springVelocityLimit = 0.0f;
    float springValueLimit = 0.0f;
    float surfaceAccumulator = 0.0f;
  };

  struct STurbo {
    u32 rouletteTickOrigin = 0u;
    float progressRatio = 0.0f;
    float impulseScale = 0.0f;
    u32 startTick = 0u;
    u32 endTick = 0u;
    ETurboType type = ETurboType_Inactive;
    CHmsCorpusId sourceCorpusId;
    float type2Phase = 0.0f;
  };

  struct SAirControl {
    bool refreshMemory = false;
    u32 memoryTick = 0u;
    GmVec3 memoryAngular{};
  };

  struct SContacts {
    CSceneVehicleCarImpactState bodyImpactState =
        CSceneVehicleCarImpactState_None;
    EPlugSurfaceMaterialId lastBodyContactMaterial =
        EPlugSurfaceMaterialId_Concrete;
    EPlugSurfaceMaterialId lastWheelContactMaterial =
        EPlugSurfaceMaterialId_Concrete;
    bool bodyContactPresent = false;
    bool lateralSlowDownContactActive = false;
    u32 lateralSlowDownLastTick = 0u;
    u32 specialContactImpulseCooldownUntil = 0u;
    CSceneVehicleCarImpactState frontWheelImpactState =
        CSceneVehicleCarImpactState_None;
    CSceneVehicleCarImpactState rearWheelImpactState =
        CSceneVehicleCarImpactState_None;
    CSceneVehicleCarImpactState peakRearWheelImpactState =
        CSceneVehicleCarImpactState_None;
    CSceneVehicleCarImpactState peakFrontWheelImpactState =
        CSceneVehicleCarImpactState_None;
    CSceneVehicleCarImpactState peakBodyImpactState =
        CSceneVehicleCarImpactState_None;
    EPlugSurfaceMaterialId peakWheelImpactMaterial =
        EPlugSurfaceMaterialId_Concrete;
    EPlugSurfaceMaterialId peakBodyImpactMaterial =
        EPlugSurfaceMaterialId_Concrete;
    float frontWheelImpactBucket = 0.0f;
    float rearWheelImpactBucket = 0.0f;
    float bodyImpactBucket = 0.0f;
    u32 wheelContactCount = 0u;
    u32 bodyContactCount = 0u;
    GmVec3 bodyContactPointSum{};
    GmVec3 bodyContactNormalSum{};
  };

  struct SRadiusSteeringState {
    float steerAngle = 0.0f;
    float previousSteerSign = 0.0f;
    CSceneVehicleCarRadiusSteeringPhase phase =
        CSceneVehicleCarRadiusSteeringPhase_Idle;
  };

  struct SSlipMemoryState {
    bool active = false;
    u32 lastTick = 0u;
    u32 startTick = 0u;
    u32 elapsedTicks = 0u;
    u32 steeringMemoryTick = 0u;
    bool steeringMemorySlip = false;
  };

  struct SGearedDriveState {
    CSceneVehicleCarEngineControlState engineState =
        CSceneVehicleCarEngineControlState_Steady;
    CSceneVehicleCarBurnoutPhase burnoutPhase =
        CSceneVehicleCarBurnoutPhase_Inactive;
    bool wheelSpeedOverrideActive = false;
    GmIso4 frameIso{};
    GmVec3 scaledCurrentForce{};
    GmVec3 burnoutCenter{};
    float burnoutBaseRadius = 0.0f;
    float burnoutTargetRadius = 0.0f;
    u32 burnoutStartTick = 0u;
    u32 burnoutExitStartTick = 0u;
    GmVec3 burnoutContactNormal{};
    float burnoutDirection = 0.0f;
    GmVec3 localSpeed{};
    float activeSteerSlowDownScale = 0.0f;
    bool wheelDriveSpeedInhibited = false;
    bool inputWindowExceeded = false;
    CSceneVehicleCarShiftDirection shiftDirection =
        CSceneVehicleCarShiftDirection_Up;
    float wheelLongitudinalSpan = 0.0f;
  };

  struct SForceAccumulators {
    GmVec3 force{};
    GmVec3 impulse{};
  };

  struct SIntegration {
    bool updateWheelVisuals = true;
    bool integrateWheels = true;
    bool integrateEngine = true;
    bool zeroHorizontalSpeed = false;
    bool speedBlocked = true;
    bool speedBlockedSecondary = true;
  };

  struct SEngine {
    float engineInputMax;
    float lowSpeedFeedbackGateScale;
    float lowSpeedFeedbackFrictionScale;
    float lowSpeedFeedbackForce;
    float engineInputMemory;
    float targetTransmissionInput;
    float slipRpmScale;
    float shiftCooldown;
    bool useLowSpeedGateB;
    int gearIndex;

    SEngine(void);
    void Reset(void);
  };

  struct SDynaPart {
    CMwId id;
    GmSpring<float> spring;

    SDynaPart(void);
    ~SDynaPart(void);
    void Reset(void);
  };

  struct SVehicleCarState : public CSceneVehicle::SVehicleState {
    float engineInputMemory;
    bool airControlRefreshMemory;
    CSceneVehicleCarEngineControlState engineControlState;
    CSceneVehicleCarShiftDirection shiftDirection;
    bool hasWheelContact;
    bool hasBodyContact;
    float bodyContactVerticalAngle;
    bool bodyContactZPositive;
    float bodyContactHorizontalAngle;
    bool noGroundFrictionGuard;

    void Reset(void);
    void Set(const SVehicleCarState &rhs);
    void SetBlend(const SVehicleCarState &from, const SVehicleCarState &to,
                  float blend);
  };

  struct SFrameHistory {
    SVehicleCarState physicsPrevious;
    SVehicleCarState physicsCurrent;
    SVehicleCarState asyncCurrent;
    SVehicleCarState asyncPrevious;
  };

  struct RuntimeClone {
    CSceneVehicle::RuntimeClone vehicle{};
    std::vector<SSimulationWheel> wheels;
    SControls controls{};
    SFeedback feedback{};
    float linearSpeedCap = 0.0f;
    SIntegration integration{};
    SFrameHistory frameHistory{};
    SEngine engine{};
    float reverseGearSpeedThreshold = 0.0f;
    STurbo turbo{};
    SAirControl airControl{};
    SContacts contacts{};
    SRadiusSteeringState radiusSteering{};
    SSlipMemoryState slipMemory{};
    SGearedDriveState gearedDrive{};
    u32 lastComputeForcesTick = 0u;
    std::array<GmSpring<float>, 4> dynaPartSprings{};
    SForceAccumulators forceAccumulators{};
  };

private:
  friend struct CSceneVehicleCarLegacyEngineTestPeer;
  friend struct CSceneVehicleCarGearedEngineTestPeer;
  friend struct OptimizedCpuModel3VehicleForceAccess;

  std::optional<std::reference_wrapper<CSceneVehicleStruct>> vehicleStruct;
  std::optional<std::reference_wrapper<CSceneSoundSource>> turboSoundSource;
  CSceneVehicleCarWheelSurfaceObserver *wheelSurfaceObserver = nullptr;
  std::vector<SSimulationWheel> wheels;
  SControls controls;
  SFeedback feedback;
  float linearSpeedCap;
  SIntegration integration;
  SFrameHistory frameHistory;
  SEngine engine;
  float reverseGearSpeedThreshold;
  STurbo turbo;
  SAirControl airControl;
  SContacts contacts;
  SRadiusSteeringState radiusSteering;
  SSlipMemoryState slipMemory;
  SGearedDriveState gearedDrive;
  u32 lastComputeForcesTick;
  std::array<SDynaPart, 4> dynaParts;
  SForceAccumulators forceAccumulators;

public:
  CSceneVehicleCar(void);
  ~CSceneVehicleCar(void) override;
  static unsigned long s_TurboRoulettePeriodMs;
  static float GetRouletteValue01(unsigned long tickDelta,
                                  unsigned long period);
  static float GetRouletteBoostFactorFromValue01(float value);
  float GetRouletteCurrentBoostFactor(void) const;
  void SetTurboRouletteTickOrigin(unsigned long tick);
  float GetMaxSpeed(void);
  void OnEnterScene(void) override;
  void VehicleReset(void) override;
  void VehicleBlockSpeedSet(int isBlocked) override;
  void VehicleBlockSpeed2Set(int isBlocked) override;
  virtual void VehicleFreeWheelingSet(int isFreeWheeling);
  void BeginRaceSimulation(void);
  void AfterContacts(void);
  void ComputeForces(float dt);
  virtual void OtherUpdateAsync(void);
  virtual void OtherVehicleForces(void);
  virtual void OtherVehiclePhysics(void);
  void UpdateParamsFromTuning(void) override;
  const CSceneVehicle::SVehicleState &VehicleStateAsyncGet(void) const override;
  const CSceneVehicle::SVehicleState &
  VehicleStatePrevAsyncGet(void) const override;
  void ComputeAndApplyContactImpulse(float restitution, const GmVec3 &speed,
                                     const GmVec3 &normal, const GmVec3 &point);
  void SetWheelCount(u32 wheelCount);
  SSimulationWheel &WheelAt(u32 wheelIndex);
  const SSimulationWheel &WheelAt(u32 wheelIndex) const;
  void BindVehicleStruct(CSceneVehicleStruct &definition) {
    vehicleStruct = std::ref(definition);
  }
  SControlInput ControlInput(void) const;
  void ApplyControlInput(const SControlInput &input);
  const SVehicleCarState &ReplayPhysicsState(void) const {
    return frameHistory.physicsCurrent;
  }
  void ConfigureSimulationLimits(float maximumLinearSpeed,
                                 float reverseTransitionSpeed);
  void EstablishRaceSpawnFrame(const GmIso4 &spawnFrame);
  void BindTurboSound(CSceneSoundSource &source, CHmsSoundSource &sound);
  void BindWheelSurfaceObserver(CSceneVehicleCarWheelSurfaceObserver &observer);
  void ClearWheelSurfaceObserver(void);
  RuntimeClone CaptureRuntimeClone(void) const;
  bool CanRestoreRuntimeClone(const RuntimeClone &clone) const noexcept;
  void RestoreRuntimeClone(const RuntimeClone &clone) noexcept;

protected:
  void HmsComputeForces(float dt) override;
  void HmsAfterContacts(void) override;
  void AddVehicleForce(const GmVec3 &force, const GmVec3 &point);
  void AddVehicleCentralForce(const GmVec3 &force);
  void AddVehicleTorque(const GmVec3 &torque);
  void SetVehicleLinearSpeed(const GmVec3 &speed);
  void SetVehicleAngularSpeed(const GmVec3 &speed);
  void AddVehicleImpulse(const GmVec3 &impulse);
  void AddVehicleImpulse(const GmVec3 &impulse, const GmVec3 &point);
  void AccumulateCentralImpulse(const GmVec3 &impulse);
  void AccumulatePointImpulse(const GmVec3 &impulseLocal);
  GmVec3 BuildLinearSpeedAfterPointImpulse(CHmsDyna::CHmsStateDyna &state,
                                           const GmVec3 &worldImpulse,
                                           float invMass);
  int RejectPointImpulseLinearSpeedGrowth(const CHmsDyna::CHmsStateDyna &state,
                                          const GmVec3 &candidateLinear,
                                          const CSceneVehicleCarTuning &tuning);
  void BuildWorldComToImpulsePoint(GmVec3 &outWorldComToPoint,
                                   const GmVec3 &worldPoint,
                                   const GmVec3 &worldCom);
  void BuildPointImpulseTorqueSeed(GmVec3 &outTorqueSeed,
                                   const GmVec3 &worldComToPoint,
                                   const GmVec3 &worldImpulse);
  void
  ScaleAndAddPointImpulseAngularDelta(CHmsDyna::CHmsStateDyna &state,
                                      GmVec3 &pointImpulseAngularDelta,
                                      const CSceneVehicleCarTuning &tuning);
  void ClampAngularSpeedAfterPointImpulse(CHmsDyna::CHmsStateDyna &state,
                                          const CSceneVehicleCarTuning &tuning);
  void UpdateTurbo(unsigned long tick);
  void EnableTurbo(unsigned long tick, unsigned long duration,
                   float impulseScale, ETurboType turboType,
                   unsigned long sourceCorpusId);
  void WheelReset(SSimulationWheel &wheel);
  void WheelIntegrate(SSimulationWheel &wheel, float dt);
  void EngineIntegrate(float input, float dt);
  void IntegrateVehicle(float dt);
  void WheelAbsorbContact(SSimulationWheel &wheel,
                          CHmsPhysicalContact &contact);
  void WheelAddForceToVehicle(SSimulationWheel &wheel, const GmVec3 &force);
  void ApplyFrictionForces(const GmVec3 &speed);
  void ComputeAirControl(const GmVec3 &angularSpeed, unsigned long tick,
                         int isGroundContact, int resetMemory);
  int ApplyWaterForces(const GmVec3 &forceToSubtract);
  void AbsorbContact(CHmsPhysicalContact &contact) override;
  void ComputeAsyncState(void);
  void VehicleUpdateAsync(void) override;
  void
  ComputeVehicleGroundMaterialVals(CSceneVehicleMaterial::SBlendableVals &out,
                                   int &hasMaterial);
  void GetSlopeAdherence(const GmVec3 &normal, float &outFirst,
                         float &outSecond);
  void ComputeForcesModel3(float dt, const GmVec3 &currentForce,
                           float slopeAdherenceA, float slopeAdherenceB,
                           const GmVec3 &linearSpeed,
                           const GmVec3 &angularSpeed, float visualSteerYaw,
                           int hasGroundMaterial,
                           CSceneVehicleMaterial::SBlendableVals *materialVals,
                           int &outSlipFlag, float &outSurfaceFeedback);
  void ComputeForcesModel4(float dt, const GmVec3 &currentForce,
                           float slopeAdherenceA, float slopeAdherenceB,
                           const GmVec3 &linearSpeed,
                           const GmVec3 &angularSpeed, float visualSteerYaw,
                           int hasGroundMaterial,
                           CSceneVehicleMaterial::SBlendableVals *materialVals,
                           int &outSlipFlag, float &outSurfaceFeedback);
  void ComputeForcesModel5(float dt, const GmVec3 &currentForce,
                           float slopeAdherenceA, float slopeAdherenceB,
                           const GmVec3 &linearSpeed,
                           const GmVec3 &angularSpeed, float visualSteerYaw,
                           int hasGroundMaterial,
                           CSceneVehicleMaterial::SBlendableVals *materialVals,
                           int &outSlipFlag, float &outSurfaceFeedback);
  void ComputeForcesModel6(float dt, const GmVec3 &currentForce,
                           float slopeAdherenceA, float slopeAdherenceB,
                           const GmVec3 &linearSpeed,
                           const GmVec3 &angularSpeed, float visualSteerYaw,
                           int hasGroundMaterial,
                           CSceneVehicleMaterial::SBlendableVals *materialVals,
                           int &outSlipFlag, float &outSurfaceFeedback);
  unsigned long GetWheelFromSurfaceTree(const CPlugTree *surfaceTree);
  int IsGroundContact();
  int IsAllWheelGroundContactId(uint8_t materialId);
  int IsGroundContactId(uint8_t materialId, GmVec3 &outPeerAxis,
                        unsigned long &outPeerCorpusId);
  void CreateFakeContacts();
  void WheelUpdateSpeedFromVehicleSpeed(SSimulationWheel &wheel,
                                        float vehicleForwardSpeed, float dt);

protected:
  void VehicleInitFromSolid(void) override;
  void GetLateralFriction(const GmVec3 &linearSpeed, const GmVec3 &direction,
                          CSceneVehicleMaterial::SBlendableVals *materialVals,
                          float slopeAdherenceA, int alreadySlipping,
                          float &outForce, int &outSlipping);

private:
  struct LegacyForceRequest {
    float dt;
    const GmVec3 &currentForce;
    float slopeAdherenceA;
    float slopeAdherenceB;
    const GmVec3 &linearSpeed;
    const GmVec3 &angularSpeed;
    float visualSteerYaw;
    bool hasGroundMaterial;
    CSceneVehicleMaterial::SBlendableVals &materialVals;
    int &outSlipFlag;
    float &outSurfaceFeedback;
  };

  void ApplyModel3ContactForces(const LegacyForceRequest &request,
                                const CSceneVehicleCarTuning &tuning);
  void ApplyModel3SteeringTorques(const LegacyForceRequest &request,
                                  const CSceneVehicleCarTuning &tuning,
                                  float speedMagnitude);
  void ApplyModel3DriveForces(const LegacyForceRequest &request,
                              const CSceneVehicleCarTuning &tuning);

  struct Model4ForceState {
    float speedMagnitude = 0.0f;
    float steerAngle = 0.0f;
    float steerAngleSin = 0.0f;
    float steerAngleCos = 1.0f;
    GmVec3 steeredSide{1.0f, 0.0f, 0.0f};
    float forwardSpeed = 0.0f;
    float sideSpeedForExit = 0.0f;
    int wasRadiusSteeringActive = 0;
    int slipSeen = 0;
    float driveForce = 0.0f;
    float opposingLongitudinalForce = 0.0f;
  };

  void PrepareModel4GroundForces(const LegacyForceRequest &request,
                                 const CSceneVehicleCarTuning &tuning,
                                 Model4ForceState &state);
  void ApplyModel4SteeringTorque(const CSceneVehicleCarTuning &tuning,
                                 Model4ForceState &state);
  void ComputeModel4Drive(const LegacyForceRequest &request,
                          const CSceneVehicleCarTuning &tuning,
                          Model4ForceState &state);
  void UpdateModel4RadiusState(const LegacyForceRequest &request,
                               const CSceneVehicleCarTuning &tuning,
                               Model4ForceState &state);
  void ApplyModel4LongitudinalForce(const LegacyForceRequest &request,
                                    const CSceneVehicleCarTuning &tuning,
                                    const Model4ForceState &state);

  struct Model5ForceState {
    int waterActive = 0;
    bool wasSlipping = false;
    int slipSeen = 0;
    float sideForceLimitTotal = 0.0f;
    float requestedSideForceTotal = 0.0f;
    u32 tick = 0u;
    float accelBase = 0.0f;
    float driveForce = 0.0f;
  };

  void ApplyModel5ContactForces(const LegacyForceRequest &request,
                                const CSceneVehicleCarTuning &tuning);
  void ApplyModel5SideTorques(const LegacyForceRequest &request,
                              const CSceneVehicleCarTuning &tuning,
                              Model5ForceState &state);
  void ComputeModel5DriveForce(const LegacyForceRequest &request,
                               const CSceneVehicleCarTuning &tuning,
                               Model5ForceState &state);
  void ApplyModel5LongitudinalForce(const LegacyForceRequest &request,
                                    const CSceneVehicleCarTuning &tuning,
                                    Model5ForceState &state);

  struct Model6ForceState {
    float frameY = 0.0f;
    int waterActive = 0;
    bool dirtSlideSurface = false;
    u32 tick = 0u;
    int slipSeen = 0;
    float sideForceLimitTotal = 0.0f;
    float requestedSideForceTotal = 0.0f;
    u32 wheelCount = 0u;
  };

  bool HandleModel6CircularBurnout(const LegacyForceRequest &request,
                                   const CSceneVehicleCarTuning &tuning,
                                   const Model6ForceState &state);
  void ApplyModel6DirtSlide(const LegacyForceRequest &request,
                            const CSceneVehicleCarTuning &tuning,
                            const Model6ForceState &state);
  void ApplyModel6ContactWheel(const LegacyForceRequest &request,
                               const CSceneVehicleCarTuning &tuning,
                               Model6ForceState &state,
                               SSimulationWheel &wheel,
                               const GmVec3 &bodyCenter);
  void ApplyModel6ContactForces(const LegacyForceRequest &request,
                                const CSceneVehicleCarTuning &tuning,
                                Model6ForceState &state);
  void ApplyModel6GroundForces(const LegacyForceRequest &request,
                               const CSceneVehicleCarTuning &tuning,
                               Model6ForceState &state);

  static float BurnoutPhase(u32 elapsedTicks, u32 durationTicks);
  static float BurnoutFadeResult(float scale, float fade) {
    return (scale - 1.0f) * fade + 1.0f;
  }
  CSceneVehicleCarTuning *ActiveTuningOrNull(void) const;
  void ResetPlayerControls(void);
  void ResetFrameHistory(void);
  void ResetAirControl(void);
  void ResetTurbo(void);
  void ResetRadiusSteering(void);
  void ResetGearedDrive(void);
  void ResetImpacts(void);
  void ResetForceAccumulators(void);
  void ResetDynaParts(void);
  void ResetSolidFeedback(void);
  void ApplyWaterSplashImpulse(const CSceneVehicleCarTuning &tuning,
                               const GmIso4 &iso, const GmVec3 &localSpeed,
                               float curveInput);
  int AreAllWheelsAirborne();
  void ClampEngineInput();
  float TransmissionWeightedSpeed();
  void IntegrateLegacyEngine(CSceneVehicleCarTuning *tuning, float input,
                             float dt, int blocked);
  void
  UpdateDirectionalTransitionInput(CSceneVehicleCarTuning *tuning, float dt,
                                   int inputActive,
                                   CSceneVehicleCarEngineControlState state);
  void UpdateTransmissionTransitions(CSceneVehicleCarTuning *tuning,
                                     int inputActive, int burnoutState);
  void IntegrateGearedEngine(CSceneVehicleCarTuning *tuning, float dt,
                             int inputActive, int blocked);
  void RefreshCollisionTree();
  void UpdateWheelVisualState(SSimulationWheel &wheel,
                              CSceneVehicleCarTuning *tuning,
                              float vehicleForwardSpeed, float dt,
                              float visualSpeedDenominator);
  void UpdateCurrentSteering(CSceneVehicleCarTuning *tuning, float dt);
  void UpdateBodyContactSnapshot();
  void ResetContactAccumulators();
  void UpdateWheelSnapshot(SSimulationWheel &wheel);
  void UpdateMaterialFeedback();
  void SetZeroDynamics();
  void UpdateDynaParamsForGroundContact(CSceneVehicleCarTuning *tuning,
                                        int isGroundContact);
  void ClampLinearSpeed(GmVec3 &linearSpeed);
  float ComputeVisualSteerYaw(CSceneVehicleCarTuning *tuning,
                              const GmVec3 &linearSpeed);
  void ComputeSelectedHandlingForces(
      CSceneVehicleCarTuning *tuning, float dt, const GmVec3 &currentForce,
      float slopeAdherenceA, float slopeAdherenceB, const GmVec3 &linearSpeed,
      const GmVec3 &angularSpeed, float visualSteerYaw, int hasGroundMaterial,
      CSceneVehicleMaterial::SBlendableVals &materialVals, int &outSlipFlag,
      float &surfaceFeedback);
  int ScanWheelSideSpeedKillContacts(int &hasSideSpeedKillContact);
  void UpdateLowSpeedFeedback(CSceneVehicleCarTuning *tuning,
                              int hasAnyContact);
  void KillSideSpeedForTaggedContact(CSceneVehicleCarTuning *tuning,
                                     int hasSideSpeedKillContact,
                                     GmVec3 &linearSpeed);
  void ApplySpecialContactResponse(CSceneVehicleCarTuning *tuning,
                                   const GmVec3 &currentForce, u32 tick,
                                   int isGroundContact);
  void UpdateImpactStates(CSceneVehicleCarTuning *tuning);
  void ProcessTurboContacts(CSceneVehicleCarTuning *tuning, u32 tick);
  CFuncKeysReal &FeedbackRamp0Curve() {
    return vehicleStruct->get().FeedbackRamp0Curve();
  }
  CFuncKeysReal &FeedbackRamp1Curve() {
    return vehicleStruct->get().FeedbackRamp1Curve();
  }
  void UpdateFeedbackSpringAxis(GmSpring<float> &spring, float dt,
                                float savedForceAxis, float savedImpulseAxis,
                                int invertDrive);
  void UpdateFeedbackTail(CSceneVehicleCarTuning *tuning, float dt,
                          const GmVec3 &linearSpeed, const GmVec3 &savedForce,
                          const GmVec3 &savedImpulse, float surfaceFeedback);
  void ClearWheelContactScratch();
  void ResetPerTickContactFeedback();
  void SaveAndClearAccumulatedFeedback(GmVec3 &savedForce,
                                       GmVec3 &savedImpulse);
  struct DirtSlideForces {
    GmVec3 front{};
    GmVec3 rear{};
  };
  struct GearedWheelSideForceResult {
    float force = 0.0f;
    float sideLimit = 0.0f;
    float sideRequested = 0.0f;
    bool slipped = false;
  };
  struct OpposingLongitudinalResult {
    float force = 0.0f;
    bool slipped = false;
  };

  CSceneVehicleMaterial *GetWheelMaterial(const SSimulationWheel &wheel);
  float ComputeSlipAccelerationBlend(const CSceneVehicleCarTuning *tuning, u32 tick,
                                     float accumulatedSideForceLimit,
                                     float accumulatedRequestedSideForce);
  float ComputeSlippingWheelDriveScale(const CSceneVehicleCarTuning *tuning);
  void MarkAllWheelsSlipping();
  int AllWheelsContactMaterial(EPlugSurfaceMaterialId materialId);
  void CaptureBurnoutReferenceFrame();
  int AdvanceBurnoutPhases(const CSceneVehicleCarTuning *tuning, u32 tick);
  int CanApplyDirtSlideForces();
  DirtSlideForces BuildDirtSlideForces(
      const CSceneVehicleCarTuning *tuning,
      const GmVec3 &linearSpeed, const GmVec3 &unitSpeed) const;
  float BurnoutDriveFade(const CSceneVehicleCarTuning *tuning, u32 tick);
  float BurnoutSideForceFade(const CSceneVehicleCarTuning *tuning, u32 tick);
  float BurnoutExitAcceleration(const CSceneVehicleCarTuning *tuning, u32 tick);
  void UpdateGearDirection(const GmVec3 &linearSpeed);
  GmVec3 BuildBurnoutRadiusSeed(const GmVec3 &bodyCenter);
  void EnterCircularBurnout(const CSceneVehicleCarTuning *tuning,
                            const GmVec3 &linearSpeed, float visualSteerYaw);
  void TryEnterForwardBurnout(const CSceneVehicleCarTuning *tuning,
                              const GmVec3 &linearSpeed, float visualSteerYaw,
                              float frameY, u32 tick, int hasGroundMaterial);
  void TryEnterReverseBurnout(const CSceneVehicleCarTuning *tuning,
                              const GmVec3 &linearSpeed, float drive,
                              float frameY, u32 tick);
  void ApplyCircularBurnoutForces(const CSceneVehicleCarTuning *tuning,
                                  const GmVec3 &currentForce,
                                  const GmVec3 &linearSpeed,
                                  const GmVec3 &angularSpeed,
                                  float visualSteerYaw, int hasGroundMaterial,
                                  int waterActive);
  GearedWheelSideForceResult ComputeGearedWheelSideForce(
      const CSceneVehicleCarTuning *tuning, SSimulationWheel &wheel,
      const CSceneVehicleMaterial::SBlendableVals &materialVals,
      const GmVec3 &linearSpeed, const GmVec3 &angularSpeed, float steerRamp,
      float dt);
  void UpdateSlipMemory(u32 tick, int slipSeen);
  OpposingLongitudinalResult ComputeOpposingLongitudinalForce(
      const CSceneVehicleCarTuning *tuning,
      const CSceneVehicleMaterial::SBlendableVals &materialVals,
      const GmVec3 &linearSpeed, float driveForce, float frameY, u32 tick,
      int slipFlag);
  float ComputeGearedDriveForce(
      const CSceneVehicleCarTuning *tuning,
      const CSceneVehicleMaterial::SBlendableVals &materialVals,
      const GmVec3 &linearSpeed, u32 tick, int waterActive, float slipAccelMix);

public:
  unsigned long WheelGetCount(void) const override;
  int WheelIsSliding(unsigned long wheelIndex) const override;
  unsigned short
  WheelGetContactMaterial(unsigned long wheelIndex) const override;
  const GmVec3 &
  WheelGetAsyncGroundContactPos(unsigned long wheelIndex) const override;
};

struct CSceneVehicleCar::SSimulationWheel {
  struct SState {
    float damperAbsorb = 0.0f;
    float wheelSpinAngle = 0.0f;
    float currentVisualSteerAngle = 0.0f;
    EPlugSurfaceMaterialId contactMaterial = EPlugSurfaceMaterialId_Concrete;
    bool contactPresent = false;
    bool slipping = false;
    GmVec3 worldSurfacePoint{};
    GmVec3 localSurfacePoint{};
    GmMat3 visualRotation{};
    bool rejectedNormalContact = false;
    GmVec3 rejectedNormalContactPoint{};

    void Reset(void);
    void SetBlend(const SState &from, const SState &to, float blend);
  };

  struct SRealTimeState {
    float damperAbsorb = 0.0f;
    float damperVelocity = 0.0f;
    float maxReplacementY = 0.0f;
    GmMat3 visualRotation{};
    GmMat3 contactFrame{};
    GmVec3 latestContactPoint{};
    float wheelAngularSpeed = 0.0f;
    bool contactPresent = false;
    EPlugSurfaceMaterialId contactMaterial = EPlugSurfaceMaterialId_Concrete;
    bool slipping = false;
    GmVec3 peerZAxisInCarLocal{};
    CHmsCorpusId peerCorpusId;
    u32 contactNormalSampleCount = 0u;
    GmVec3 accumulatedContactNormal{};
    float wheelSpinAngle = 0.0f;
    float currentVisualSteerAngle = 0.0f;
    float targetVisualSteerAngle = 0.0f;
    bool rejectedNormalContact = false;
    GmVec3 rejectedNormalContactPoint{};

    void ResetSimulationState(void);
    void Integrate(float dt);
  };

  bool killsLateralSpeedOnContact = false;
  VehicleWheelAxle axle = VehicleWheelAxle::Rear;
  float rollingRadius = 1.0f;
  CSceneVehicle::SSurfaceHandler surfaceHandler;
  GmVec3 forceApplicationPoint{};
  SRealTimeState realTimeState;
  SState previousPhysicsState;
  SState currentPhysicsState;
  SState previousAsyncState;
  SState asyncState;

  SSimulationWheel(void);
  ~SSimulationWheel(void);
  void ResetSimulationState(void);
  void ClearContactCarryStamp();
  int GetSurfaceTreeRollingRadius(float &outRadius) const;
  void ApplySingleMaterialRefFromTuning(const CSceneVehicleCarTuning &tuning);
  void CopyRestSurfaceIso();
  void OffsetCurrentSurfaceY(float yOffset);
};

class CSceneVehicleCarWheelSurfaceObserver {
public:
  virtual ~CSceneVehicleCarWheelSurfaceObserver() = default;
  virtual void
  OnWheelSurfaceUpdated(CSceneVehicleCar &car,
                        CSceneVehicleCar::SSimulationWheel &wheel) = 0;
};
