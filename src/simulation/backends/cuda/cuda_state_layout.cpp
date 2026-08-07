#include "simulation/backends/cuda/cuda_state_layout.h"

#include <algorithm>
#include <cstring>
#include <new>

#include "simulation/runtime/replay_simulation_session.h"

namespace forevervalidator::simulation {
namespace {

void EncodeFrame(
        const CSceneVehicleCar::SVehicleCarState &source,
        CudaVehicleCarFrameState &destination) {
    destination.forwardSpeed = source.forwardSpeed;
    destination.sideSpeed = source.sideSpeed;
    destination.steeringControl = source.steeringControl;
    destination.lowSpeedGateA = source.lowSpeedGateA;
    destination.lowSpeedGateB = source.lowSpeedGateB;
    destination.turboActive = source.turboActive;
    destination.turboProgressRatio = source.turboProgressRatio;
    destination.wheelSpeedOverrideActive =
            source.wheelSpeedOverrideActive;
    destination.surfaceFeedbackAccumulator =
            source.surfaceFeedbackAccumulator;
    destination.feedbackSideSpringValue =
            source.feedbackSideSpringValue;
    destination.feedbackForwardSpringValue =
            source.feedbackForwardSpringValue;
    destination.feedbackRamp1 = source.feedbackRamp1;
    destination.feedbackRamp0 = source.feedbackRamp0;
    destination.corpusIso = source.corpusIso;
    destination.vehicleEvent0Value = source.vehicleEvent0Value;
    destination.waterSplashEventCounter =
            source.waterSplashEventCounter;
    destination.localLinearSpeed = source.localLinearSpeed;
    destination.materialFeedbackSpeed = source.materialFeedbackSpeed;
    destination.materialFeedbackIntensity =
            source.materialFeedbackIntensity;
    destination.engineInputMemory = source.engineInputMemory;
    destination.airControlRefreshMemory =
            source.airControlRefreshMemory;
    destination.engineControlState =
            static_cast<std::uint32_t>(source.engineControlState);
    destination.shiftDirection =
            static_cast<std::uint32_t>(source.shiftDirection);
    destination.hasWheelContact = source.hasWheelContact;
    destination.hasBodyContact = source.hasBodyContact;
    destination.bodyContactVerticalAngle =
            source.bodyContactVerticalAngle;
    destination.bodyContactZPositive = source.bodyContactZPositive;
    destination.bodyContactHorizontalAngle =
            source.bodyContactHorizontalAngle;
    destination.noGroundFrictionGuard =
            source.noGroundFrictionGuard;
}

void DecodeFrame(
        const CudaVehicleCarFrameState &source,
        CSceneVehicleCar::SVehicleCarState &destination) {
    destination.forwardSpeed = source.forwardSpeed;
    destination.sideSpeed = source.sideSpeed;
    destination.steeringControl = source.steeringControl;
    destination.lowSpeedGateA = source.lowSpeedGateA;
    destination.lowSpeedGateB = source.lowSpeedGateB;
    destination.turboActive = source.turboActive;
    destination.turboProgressRatio = source.turboProgressRatio;
    destination.wheelSpeedOverrideActive =
            source.wheelSpeedOverrideActive;
    destination.surfaceFeedbackAccumulator =
            source.surfaceFeedbackAccumulator;
    destination.feedbackSideSpringValue =
            source.feedbackSideSpringValue;
    destination.feedbackForwardSpringValue =
            source.feedbackForwardSpringValue;
    destination.feedbackRamp1 = source.feedbackRamp1;
    destination.feedbackRamp0 = source.feedbackRamp0;
    destination.corpusIso = source.corpusIso;
    destination.vehicleEvent0Value = source.vehicleEvent0Value;
    destination.waterSplashEventCounter =
            source.waterSplashEventCounter;
    destination.localLinearSpeed = source.localLinearSpeed;
    destination.materialFeedbackSpeed = source.materialFeedbackSpeed;
    destination.materialFeedbackIntensity =
            source.materialFeedbackIntensity;
    destination.engineInputMemory = source.engineInputMemory;
    destination.airControlRefreshMemory =
            source.airControlRefreshMemory;
    destination.engineControlState =
            static_cast<CSceneVehicleCarEngineControlState>(
                    source.engineControlState);
    destination.shiftDirection =
            static_cast<CSceneVehicleCarShiftDirection>(
                    source.shiftDirection);
    destination.hasWheelContact = source.hasWheelContact;
    destination.hasBodyContact = source.hasBodyContact;
    destination.bodyContactVerticalAngle =
            source.bodyContactVerticalAngle;
    destination.bodyContactZPositive = source.bodyContactZPositive;
    destination.bodyContactHorizontalAngle =
            source.bodyContactHorizontalAngle;
    destination.noGroundFrictionGuard =
            source.noGroundFrictionGuard;
}

template<typename T>
void EncodeOptional(const std::optional<T> &source,
                    CudaOptional<T> &destination) {
    destination.present = source.has_value();
    destination.value = source.value_or(T{});
}

template<typename T>
std::optional<T> DecodeOptional(const CudaOptional<T> &source) {
    return source.present ? std::optional<T>(source.value) : std::nullopt;
}

CudaStateConversionResult EncodeBody(
        const ReplayVehicleBody::RuntimeClone &source,
        CudaDynamicBodyState &destination,
        CudaFixedArray<
                GmVec3,
                CudaCollisionReplacementOverflowCapacity>
                &overflowDestination) {
    if (source.pendingCollisionReplacements.size() >
        CudaCollisionReplacementCapacity) {
        return CudaStateConversionResult::CollisionReplacementOverflow;
    }
    EncodeOptional(source.maxAngularSpeed, destination.maxAngularSpeed);
    destination.parameters = source.dynaParams;
    destination.physicalParameters = source.physicalParameters;
    destination.corpusLocalIso = source.corpusLocalIso;
    destination.temporary = source.tempState;
    destination.write = source.writeState;
    destination.current = source.currentState;
    const std::size_t inlineCount = std::min(
            source.pendingCollisionReplacements.size(),
            CudaCollisionReplacementInlineCapacity);
    destination.collisionReplacements.count =
            static_cast<std::uint32_t>(inlineCount);
    std::copy_n(
            source.pendingCollisionReplacements.begin(),
            inlineCount,
            destination.collisionReplacements.values);
    const std::size_t overflowCount =
            source.pendingCollisionReplacements.size() - inlineCount;
    overflowDestination.count =
            static_cast<std::uint32_t>(overflowCount);
    std::copy_n(
            source.pendingCollisionReplacements.begin() + inlineCount,
            overflowCount,
            overflowDestination.values);
    destination.dynamicActive = source.isDynamicActive;
    destination.dynamicType =
            static_cast<std::uint32_t>(source.dynamicType);
    return CudaStateConversionResult::Success;
}

CudaStateConversionResult EncodeVehicle(
        const ReplayVehicleSimulation::RuntimeClone &source,
        CudaVehicleState &destination,
        CudaVehiclePassthroughState &passthrough) {
    const CSceneVehicleCar::RuntimeClone &car = source.car;
    if (car.wheels.size() > std::size(destination.wheels.values) ||
        source.wheelSurfaces.movedByUpdateSurface.size() !=
                car.wheels.size()) {
        return CudaStateConversionResult::WheelOverflow;
    }
    destination.mobil = car.vehicle.mobil;
    std::copy(car.vehicle.vehicleEvents.begin(),
              car.vehicle.vehicleEvents.end(),
              destination.vehicleEvents);
    destination.water = car.vehicle.water;
    passthrough.updateAsync = car.vehicle.updateAsync;
    passthrough.networked = car.vehicle.networked;
    passthrough.predictionDelayTicks =
            car.vehicle.predictionDelayTicks;
    EncodeOptional(car.vehicle.stateSampleWindow,
                   passthrough.stateSampleWindow);
    passthrough.asyncPeriodSeconds = car.vehicle.asyncPeriodSeconds;
    destination.wheels.count =
            static_cast<std::uint32_t>(car.wheels.size());
    for (std::size_t index = 0u; index < car.wheels.size(); ++index) {
        const CSceneVehicleCar::SSimulationWheel &sourceWheel =
                car.wheels[index];
        CudaWheelState &wheel = destination.wheels.values[index];
        wheel.killsLateralSpeedOnContact =
                sourceWheel.killsLateralSpeedOnContact;
        wheel.axle = static_cast<std::uint32_t>(sourceWheel.axle);
        wheel.rollingRadius = sourceWheel.rollingRadius;
        wheel.restPose = sourceWheel.surfaceHandler.RestPose();
        wheel.currentPose = sourceWheel.surfaceHandler.CurrentPose();
        wheel.forceApplicationPoint =
                sourceWheel.forceApplicationPoint;
        wheel.realTime = sourceWheel.realTimeState;
        wheel.previousPhysics = sourceWheel.previousPhysicsState;
        wheel.currentPhysics = sourceWheel.currentPhysicsState;
        passthrough.wheels[index].previousAsync =
                sourceWheel.previousAsyncState;
        passthrough.wheels[index].currentAsync =
                sourceWheel.asyncState;
        wheel.surfaceMovedByUpdate =
                source.wheelSurfaces.movedByUpdateSurface[index];
    }
    destination.controls = car.controls;
    destination.feedback = car.feedback;
    destination.linearSpeedCap = car.linearSpeedCap;
    destination.integration = car.integration;
    EncodeFrame(car.frameHistory.physicsPrevious,
                destination.frameHistory.physicsPrevious);
    EncodeFrame(car.frameHistory.physicsCurrent,
                destination.frameHistory.physicsCurrent);
    EncodeFrame(car.frameHistory.asyncCurrent,
                passthrough.asyncCurrent);
    EncodeFrame(car.frameHistory.asyncPrevious,
                passthrough.asyncPrevious);
    destination.engine = car.engine;
    destination.reverseGearSpeedThreshold =
            car.reverseGearSpeedThreshold;
    destination.turbo = car.turbo;
    destination.airControl = car.airControl;
    destination.contacts = car.contacts;
    destination.radiusSteering = car.radiusSteering;
    destination.slipMemory = car.slipMemory;
    destination.gearedDrive = car.gearedDrive;
    destination.lastComputeForcesTick = car.lastComputeForcesTick;
    std::copy(car.dynaPartSprings.begin(),
              car.dynaPartSprings.end(),
              destination.dynaPartSprings);
    destination.forceAccumulators = car.forceAccumulators;
    return CudaStateConversionResult::Success;
}

CudaStateConversionResult EncodeRace(
        const CTrackManiaRace::RuntimeClone &source,
        CudaRacePhysicsState &destination,
        CudaStuntState &stunts,
        CudaFixedArray<ReplayStuntEvent, 2048u> &stuntEvents) {
    if (source.checkpointSlotsPassed.size() >
        CudaCheckpointSlotCapacity) {
        return CudaStateConversionResult::CheckpointOverflow;
    }
    if (source.stuntEvents.size() >
        std::size(stuntEvents.values)) {
        return CudaStateConversionResult::StuntEventOverflow;
    }
    destination.player = source.player.CaptureRuntimeClone();
    destination.checkpointSlotsPassed.count =
            static_cast<std::uint32_t>(
                    source.checkpointSlotsPassed.size());
    for (std::uint32_t index = 0u;
         index < destination.checkpointSlotsPassed.count; ++index) {
        if (source.checkpointSlotsPassed[index] != 0u) {
            destination.checkpointSlotsPassed.words[
                    index / CudaCheckpointSlots::WordBits] |=
                    1u << (index % CudaCheckpointSlots::WordBits);
        }
    }
    EncodeOptional(source.playerSpawnLocation,
                   destination.playerSpawnLocation);
    EncodeOptional(source.lastAcceptedSpawnLocation,
                   destination.lastAcceptedSpawnLocation);
    destination.currentSpawnLocationInitialized =
            source.currentSpawnLocationInitialized;
    destination.preparedEventTimeMs = source.preparedEventTimeMs;
    destination.replayPlayMode =
            static_cast<std::uint32_t>(source.replayPlayMode);
    destination.replayNbLaps = source.replayNbLaps;
    destination.progress = source.progress;
    stunts.replayStuntsEnabled = source.replayStuntsEnabled;
    stunts.replayStuntStateAvailable =
            source.replayStuntStateAvailable;
    stunts.replayStuntsTimeLimitMs =
            source.replayStuntsTimeLimitMs;
    stunts.replayStuntsRaceStartTimeMs =
            source.replayStuntsRaceStartTimeMs;
    stunts.replayStuntState = source.replayStuntState;
    std::copy(source.replayStuntInputHistory.begin(),
              source.replayStuntInputHistory.end(),
              stunts.stuntInputHistory);
    stunts.stuntInputHistorySize =
            static_cast<std::uint32_t>(
                    source.replayStuntInputHistorySize);
    std::copy(source.replayStuntLocationHistory.begin(),
              source.replayStuntLocationHistory.end(),
              stunts.stuntLocationHistory);
    stunts.stuntLocationHistorySize =
            static_cast<std::uint32_t>(
                    source.replayStuntLocationHistorySize);
    stunts.stuntPreviousLocation =
            source.replayStuntPreviousLocation;
    stunts.stuntTakeoffLocation =
            source.replayStuntTakeoffLocation;
    stunts.stuntRotation = source.replayStuntRotation;
    stunts.stuntLandingDirection =
            source.replayStuntLandingDirection;
    stunts.stuntTakeoffTick = source.replayStuntTakeoffTick;
    stunts.stuntLandingTick = source.replayStuntLandingTick;
    stunts.stuntPreviousLandingTick =
            source.replayStuntPreviousLandingTick;
    stunts.stuntChain = source.replayStuntChain;
    stunts.stuntComboWindowMs = source.replayStuntComboWindowMs;
    stunts.stuntInProgress = source.replayStuntInProgress;
    stunts.stuntMasterJump = source.replayStuntMasterJump;
    stunts.stuntBadLanding = source.replayStuntBadLanding;
    EncodeOptional(source.replayStuntScoreAtTimeLimit,
                   stunts.stuntScoreAtTimeLimit);
    std::copy(source.replayStuntFigureScores.begin(),
              source.replayStuntFigureScores.end(),
              stunts.stuntFigureScores);
    stunts.stuntsScore = source.stuntsScore;
    stuntEvents.count =
            static_cast<std::uint32_t>(source.stuntEvents.size());
    for (std::size_t index = 0u;
         index < source.stuntEvents.size(); ++index) {
        const ReplayStuntEvent &sourceEvent =
                source.stuntEvents[index];
        ReplayStuntEvent &destinationEvent =
                stuntEvents.values[index];
        destinationEvent.figure = sourceEvent.figure;
        destinationEvent.degree = sourceEvent.degree;
        destinationEvent.score = sourceEvent.score;
        destinationEvent.bonus = sourceEvent.bonus;
        destinationEvent.straightLanding =
                sourceEvent.straightLanding;
        destinationEvent.reverseLanding =
                sourceEvent.reverseLanding;
        destinationEvent.masterJump = sourceEvent.masterJump;
        destinationEvent.chain = sourceEvent.chain;
    }
    return CudaStateConversionResult::Success;
}

void DecodeBody(
                const CudaDynamicBodyState &source,
                const CudaFixedArray<
                        GmVec3,
                        CudaCollisionReplacementOverflowCapacity>
                        &overflowSource,
                ReplayVehicleBody::RuntimeClone &destination) {
    destination.maxAngularSpeed = DecodeOptional(source.maxAngularSpeed);
    destination.dynaParams = source.parameters;
    destination.physicalParameters = source.physicalParameters;
    destination.corpusLocalIso = source.corpusLocalIso;
    destination.tempState = source.temporary;
    destination.writeState = source.write;
    destination.currentState = source.current;
    destination.pendingCollisionReplacements.assign(
            source.collisionReplacements.values,
            source.collisionReplacements.values +
                    source.collisionReplacements.count);
    destination.pendingCollisionReplacements.insert(
            destination.pendingCollisionReplacements.end(),
            overflowSource.values,
            overflowSource.values + overflowSource.count);
    destination.isDynamicActive = source.dynamicActive;
    destination.dynamicType =
            static_cast<CHmsDyna::EDynamicType>(source.dynamicType);
}

void DecodeVehicle(const CudaVehicleState &source,
                   const CudaVehiclePassthroughState &passthrough,
                   ReplayVehicleSimulation::RuntimeClone &destination) {
    CSceneVehicleCar::RuntimeClone &car = destination.car;
    car.vehicle.mobil = source.mobil;
    std::copy(std::begin(source.vehicleEvents),
              std::end(source.vehicleEvents),
              car.vehicle.vehicleEvents.begin());
    car.vehicle.water = source.water;
    car.vehicle.updateAsync = passthrough.updateAsync;
    car.vehicle.networked = passthrough.networked;
    car.vehicle.predictionDelayTicks =
            passthrough.predictionDelayTicks;
    car.vehicle.stateSampleWindow =
            DecodeOptional(passthrough.stateSampleWindow);
    car.vehicle.asyncPeriodSeconds =
            passthrough.asyncPeriodSeconds;
    car.wheels.resize(source.wheels.count);
    destination.wheelSurfaces.movedByUpdateSurface.assign(
            source.wheels.count, false);
    for (std::size_t index = 0u; index < source.wheels.count; ++index) {
        const CudaWheelState &wheel = source.wheels.values[index];
        CSceneVehicleCar::SSimulationWheel &target = car.wheels[index];
        target.killsLateralSpeedOnContact =
                wheel.killsLateralSpeedOnContact;
        target.axle = static_cast<VehicleWheelAxle>(wheel.axle);
        target.rollingRadius = wheel.rollingRadius;
        target.surfaceHandler.BindTree(nullptr);
        target.surfaceHandler.SetRestPose(wheel.restPose);
        target.surfaceHandler.SetCurrentPose(wheel.currentPose);
        target.forceApplicationPoint = wheel.forceApplicationPoint;
        target.realTimeState = wheel.realTime;
        target.previousPhysicsState = wheel.previousPhysics;
        target.currentPhysicsState = wheel.currentPhysics;
        target.previousAsyncState =
                passthrough.wheels[index].previousAsync;
        target.asyncState =
                passthrough.wheels[index].currentAsync;
        destination.wheelSurfaces.movedByUpdateSurface[index] =
                wheel.surfaceMovedByUpdate;
    }
    car.controls = source.controls;
    car.feedback = source.feedback;
    car.linearSpeedCap = source.linearSpeedCap;
    car.integration = source.integration;
    DecodeFrame(source.frameHistory.physicsPrevious,
                car.frameHistory.physicsPrevious);
    DecodeFrame(source.frameHistory.physicsCurrent,
                car.frameHistory.physicsCurrent);
    DecodeFrame(passthrough.asyncCurrent,
                car.frameHistory.asyncCurrent);
    DecodeFrame(passthrough.asyncPrevious,
                car.frameHistory.asyncPrevious);
    car.engine = source.engine;
    car.reverseGearSpeedThreshold = source.reverseGearSpeedThreshold;
    car.turbo = source.turbo;
    car.airControl = source.airControl;
    car.contacts = source.contacts;
    car.radiusSteering = source.radiusSteering;
    car.slipMemory = source.slipMemory;
    car.gearedDrive = source.gearedDrive;
    car.lastComputeForcesTick = source.lastComputeForcesTick;
    std::copy(std::begin(source.dynaPartSprings),
              std::end(source.dynaPartSprings),
              car.dynaPartSprings.begin());
    car.forceAccumulators = source.forceAccumulators;
}

void DecodeRace(
                const CudaRacePhysicsState &source,
                const CudaStuntState &stunts,
                const CudaFixedArray<ReplayStuntEvent, 2048u>
                        &stuntEvents,
                CTrackManiaRace::RuntimeClone &destination) {
    destination.player.RestoreRuntimeClone(source.player);
    destination.checkpointSlotsPassed.resize(
            source.checkpointSlotsPassed.count);
    for (std::uint32_t index = 0u;
         index < source.checkpointSlotsPassed.count; ++index) {
        destination.checkpointSlotsPassed[index] =
                (source.checkpointSlotsPassed.words[
                         index / CudaCheckpointSlots::WordBits] &
                 (1u << (index %
                         CudaCheckpointSlots::WordBits))) != 0u
                        ? 1u
                        : 0u;
    }
    destination.playerSpawnLocation =
            DecodeOptional(source.playerSpawnLocation);
    destination.lastAcceptedSpawnLocation =
            DecodeOptional(source.lastAcceptedSpawnLocation);
    destination.currentSpawnLocationInitialized =
            source.currentSpawnLocationInitialized;
    destination.preparedEventTimeMs = source.preparedEventTimeMs;
    destination.replayPlayMode =
            static_cast<EChallengePlayMode>(source.replayPlayMode);
    destination.replayNbLaps = source.replayNbLaps;
    destination.progress = source.progress;
    destination.replayStuntsEnabled = stunts.replayStuntsEnabled;
    destination.replayStuntStateAvailable =
            stunts.replayStuntStateAvailable;
    destination.replayStuntsTimeLimitMs =
            stunts.replayStuntsTimeLimitMs;
    destination.replayStuntsRaceStartTimeMs =
            stunts.replayStuntsRaceStartTimeMs;
    destination.replayStuntState = stunts.replayStuntState;
    std::copy(std::begin(stunts.stuntInputHistory),
              std::end(stunts.stuntInputHistory),
              destination.replayStuntInputHistory.begin());
    destination.replayStuntInputHistorySize =
            stunts.stuntInputHistorySize;
    std::copy(std::begin(stunts.stuntLocationHistory),
              std::end(stunts.stuntLocationHistory),
              destination.replayStuntLocationHistory.begin());
    destination.replayStuntLocationHistorySize =
            stunts.stuntLocationHistorySize;
    destination.replayStuntPreviousLocation =
            stunts.stuntPreviousLocation;
    destination.replayStuntTakeoffLocation =
            stunts.stuntTakeoffLocation;
    destination.replayStuntRotation = stunts.stuntRotation;
    destination.replayStuntLandingDirection =
            stunts.stuntLandingDirection;
    destination.replayStuntTakeoffTick = stunts.stuntTakeoffTick;
    destination.replayStuntLandingTick = stunts.stuntLandingTick;
    destination.replayStuntPreviousLandingTick =
            stunts.stuntPreviousLandingTick;
    destination.replayStuntChain = stunts.stuntChain;
    destination.replayStuntComboWindowMs = stunts.stuntComboWindowMs;
    destination.replayStuntInProgress = stunts.stuntInProgress;
    destination.replayStuntMasterJump = stunts.stuntMasterJump;
    destination.replayStuntBadLanding = stunts.stuntBadLanding;
    destination.replayStuntScoreAtTimeLimit =
            DecodeOptional(stunts.stuntScoreAtTimeLimit);
    std::copy(std::begin(stunts.stuntFigureScores),
              std::end(stunts.stuntFigureScores),
              destination.replayStuntFigureScores.begin());
    destination.stuntsScore = stunts.stuntsScore;
    destination.stuntEvents.assign(
            stuntEvents.values,
            stuntEvents.values + stuntEvents.count);
}

}  // namespace

CudaStateConversionResult EncodeCudaRaceState(
        const CTrackManiaRace::RuntimeClone &source,
        CudaRaceState *destination) noexcept {
    if (destination == nullptr) {
        return CudaStateConversionResult::InvalidArgument;
    }
    // Value assignment initializes members but is not required to overwrite
    // padding. Clear the complete trivially-copyable transport object first
    // so raw state copies and fingerprints are deterministic.
    std::memset(destination, 0, sizeof(*destination));
    ::new (static_cast<void *>(destination)) CudaRaceState{};
    return EncodeRace(
            source, *destination, destination->stunts,
            destination->stuntEvents);
}

CudaStateConversionResult DecodeCudaRaceState(
        const CudaRaceState &source,
        CTrackManiaRace::RuntimeClone *destination) noexcept {
    if (destination == nullptr) {
        return CudaStateConversionResult::InvalidArgument;
    }
    if (source.checkpointSlotsPassed.count >
        CudaCheckpointSlotCapacity) {
        return CudaStateConversionResult::CheckpointOverflow;
    }
    if (source.stuntEvents.count >
        std::size(source.stuntEvents.values)) {
        return CudaStateConversionResult::StuntEventOverflow;
    }
    try {
        CTrackManiaRace::RuntimeClone result;
        DecodeRace(
                source, source.stunts,
                source.stuntEvents, result);
        *destination = std::move(result);
        return CudaStateConversionResult::Success;
    } catch (const std::bad_alloc &) {
        return CudaStateConversionResult::AllocationFailed;
    }
}

CudaStateConversionResult EncodeCudaCandidateState(
        const ReplaySimulationInstanceClone &source,
        std::uint32_t validationSeed,
        std::uint64_t controlCursor,
        std::uint32_t candidateId,
        std::uint32_t randomState,
        CudaCandidateState *destination) noexcept {
    if (destination == nullptr) {
        return CudaStateConversionResult::InvalidArgument;
    }
    *destination = CudaCandidateState{};
    destination->candidateId = candidateId;
    destination->validationSeed = validationSeed;
    destination->randomState = randomState;
    destination->controlCursor = controlCursor;
    destination->world = source.runtime.world;
    destination->incrementalRespawnCount =
            source.incrementalRespawnCount;
    if (source.runtime.finishTime.has_value()) {
        destination->finishTime.present = true;
        destination->finishTime.value = *source.runtime.finishTime;
    }
    destination->firstStep = source.runtime.firstStep;
    destination->stuntsEnabled = source.runtime.stuntsEnabled;
    CudaStateConversionResult result =
            EncodeBody(
                    source.runtime.body, destination->body,
                    destination->collisionReplacementOverflow);
    if (result != CudaStateConversionResult::Success) {
        return result;
    }
    result = EncodeVehicle(
            source.runtime.vehicle,
            destination->vehicle,
            destination->vehiclePassthrough);
    if (result != CudaStateConversionResult::Success) {
        return result;
    }
    return EncodeRace(
            source.race, destination->race,
            destination->stunts,
            destination->stuntEvents);
}

CudaStateConversionResult DecodeCudaCandidateState(
        const CudaCandidateState &source,
        ReplaySimulationInstanceClone *destination) noexcept {
    if (destination == nullptr) {
        return CudaStateConversionResult::InvalidArgument;
    }
    if (source.schemaVersion != CudaCandidateState::SchemaVersion) {
        return CudaStateConversionResult::SchemaMismatch;
    }
    if (source.vehicle.wheels.count >
        std::size(source.vehicle.wheels.values)) {
        return CudaStateConversionResult::WheelOverflow;
    }
    if (source.body.collisionReplacements.count >
        std::size(source.body.collisionReplacements.values)) {
        return CudaStateConversionResult::CollisionReplacementOverflow;
    }
    if (source.collisionReplacementOverflow.count >
        std::size(source.collisionReplacementOverflow.values)) {
        return CudaStateConversionResult::CollisionReplacementOverflow;
    }
    if (source.race.checkpointSlotsPassed.count >
        CudaCheckpointSlotCapacity) {
        return CudaStateConversionResult::CheckpointOverflow;
    }
    if (source.stuntEvents.count >
        std::size(source.stuntEvents.values)) {
        return CudaStateConversionResult::StuntEventOverflow;
    }
    try {
        ReplaySimulationInstanceClone result;
        result.runtime.world = source.world;
        DecodeBody(
                source.body, source.collisionReplacementOverflow,
                result.runtime.body);
        DecodeVehicle(
                source.vehicle,
                source.vehiclePassthrough,
                result.runtime.vehicle);
        DecodeRace(
                source.race, source.stunts,
                source.stuntEvents,
                result.race);
        result.incrementalRespawnCount =
                source.incrementalRespawnCount;
        if (source.finishTime.present) {
            result.runtime.finishTime = source.finishTime.value;
        }
        result.randomState = source.randomState;
        result.runtime.firstStep = source.firstStep;
        result.runtime.stuntsEnabled = source.stuntsEnabled;
        *destination = std::move(result);
        return CudaStateConversionResult::Success;
    } catch (const std::bad_alloc &) {
        return CudaStateConversionResult::AllocationFailed;
    }
}

}  // namespace forevervalidator::simulation
