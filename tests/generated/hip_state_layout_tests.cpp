// Generated from tests/cuda_state_layout_tests.cpp by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_state_layout.h"
#include "simulation/runtime/replay_simulation_session.h"

#include <iostream>

namespace {

ReplaySimulationInstanceClone BuildState() {
    ReplaySimulationInstanceClone state;
    state.runtime.world.schemePeriodMs = 10u;
    state.runtime.world.tickTimeMs = 12340u;
    state.runtime.body.maxAngularSpeed = 99.5f;
    state.runtime.body.currentState.position = {1.0f, 2.0f, 3.0f};
    state.runtime.body.pendingCollisionReplacements = {
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f}};
    state.runtime.body.isDynamicActive = true;
    state.runtime.body.dynamicType =
            CHmsDyna::EDynamicType_FullAngularDynamics;
    state.runtime.body.corpusLocalIso.translation =
            {10.0f, 20.0f, 30.0f};
    state.runtime.vehicle.car.wheels.resize(4u);
    state.runtime.vehicle.wheelSurfaces.movedByUpdateSurface.assign(
            4u, false);
    for (std::size_t index = 0u; index < 4u; ++index) {
        auto &wheel = state.runtime.vehicle.car.wheels[index];
        wheel.rollingRadius = 0.5f + static_cast<float>(index);
        wheel.realTimeState.wheelAngularSpeed =
                20.0f + static_cast<float>(index);
        wheel.realTimeState.contactPresent = (index & 1u) != 0u;
        state.runtime.vehicle.wheelSurfaces.movedByUpdateSurface[index] =
                (index & 1u) == 0u;
    }
    state.runtime.vehicle.car.engine.gearIndex = 4;
    state.runtime.vehicle.car.turbo.startTick = 1000u;
    state.runtime.vehicle.car.controls.currentSteering = 0.25f;
    state.race.checkpointSlotsPassed = {1u, 0u, 1u, 0u};
    state.race.progress.requiredCheckpointCount = 3u;
    state.race.progress.currentLapCheckpointCount = 2u;
    state.race.replayStuntsEnabled = true;
    state.race.stuntsScore = 123u;
    state.race.stuntEvents.push_back(
            {EFigures_Unknown, 360u, 100u, 1.25f,
             true, false, false, 2u});
    state.runtime.firstStep = false;
    state.runtime.stuntsEnabled = true;
    state.runtime.finishTime =
            forevervalidator::FinishTimeEstimate{
                    12339999999u, 12340000000u, 12340000000u};
    state.incrementalRespawnCount = 3u;
    state.randomState = 1234567u;
    return state;
}

}  // namespace

int main() {
    using forevervalidator::simulation::HipCandidateState;
    using forevervalidator::simulation::HipStateConversionResult;
    using forevervalidator::simulation::DecodeHipCandidateState;
    ReplaySimulationInstanceClone original = BuildState();
    HipCandidateState encoded;
    const HipStateConversionResult encode =
            forevervalidator::simulation::EncodeHipCandidateState(
                    original, 42u, 987u, 11u, 1234567u, &encoded);
    if (encode != HipStateConversionResult::Success) {
        std::cerr << "state encode failed\n";
        return 1;
    }
    if (encoded.candidateId != 11u ||
        encoded.validationSeed != 42u ||
        encoded.controlCursor != 987u ||
        encoded.randomState != 1234567u) {
        std::cerr << "candidate-owned metadata was not captured\n";
        return 1;
    }
    ReplaySimulationInstanceClone decoded;
    const HipStateConversionResult decode =
            forevervalidator::simulation::DecodeHipCandidateState(
                    encoded, &decoded);
    if (decode != HipStateConversionResult::Success ||
        ReplaySimulationInstanceSemanticHash(original) !=
                ReplaySimulationInstanceSemanticHash(decoded) ||
        decoded.incrementalRespawnCount != 3u ||
        decoded.runtime.finishTime != original.runtime.finishTime) {
        std::cerr << "state round trip changed future-affecting data\n";
        return 1;
    }
    ReplaySimulationInstanceClone checkpointBoundary = BuildState();
    checkpointBoundary.race.checkpointSlotsPassed.resize(
            forevervalidator::simulation::HipCheckpointSlotCapacity);
    for (std::size_t index = 0u;
         index < checkpointBoundary.race.checkpointSlotsPassed.size();
         index += 3u) {
        checkpointBoundary.race.checkpointSlotsPassed[index] = 1u;
    }
    HipCandidateState checkpointBoundaryEncoded;
    if (forevervalidator::simulation::EncodeHipCandidateState(
                checkpointBoundary, 42u, 987u, 11u, 1234567u,
                &checkpointBoundaryEncoded) !=
                    HipStateConversionResult::Success ||
        DecodeHipCandidateState(checkpointBoundaryEncoded, &decoded) !=
                HipStateConversionResult::Success ||
        ReplaySimulationInstanceSemanticHash(checkpointBoundary) !=
                ReplaySimulationInstanceSemanticHash(decoded)) {
        std::cerr << "checkpoint boundary did not round trip\n";
        return 1;
    }
    ReplaySimulationInstanceClone replacementBoundary = BuildState();
    replacementBoundary.runtime.body.pendingCollisionReplacements.resize(
            forevervalidator::simulation::
                    HipCollisionReplacementCapacity);
    for (std::size_t index = 0u;
         index < replacementBoundary.runtime.body.
                         pendingCollisionReplacements.size();
         ++index) {
        replacementBoundary.runtime.body.
                pendingCollisionReplacements[index] = {
                        static_cast<float>(index),
                        static_cast<float>(index + 1u),
                        static_cast<float>(index + 2u)};
    }
    HipCandidateState boundaryEncoded;
    if (forevervalidator::simulation::EncodeHipCandidateState(
                replacementBoundary, 42u, 987u, 11u, 1234567u,
                &boundaryEncoded) != HipStateConversionResult::Success ||
        DecodeHipCandidateState(boundaryEncoded, &decoded) !=
                HipStateConversionResult::Success ||
        ReplaySimulationInstanceSemanticHash(replacementBoundary) !=
                ReplaySimulationInstanceSemanticHash(decoded)) {
        std::cerr << "collision replacement boundary did not round trip\n";
        return 1;
    }
    const HipCandidateState valid = encoded;
    encoded.vehicle.wheels.count = 5u;
    if (DecodeHipCandidateState(encoded, &decoded) !=
        HipStateConversionResult::WheelOverflow) {
        std::cerr << "state overflow was not rejected\n";
        return 1;
    }
    encoded = valid;
    encoded.collisionReplacementOverflow.count =
            forevervalidator::simulation::
                    HipCollisionReplacementOverflowCapacity + 1u;
    if (DecodeHipCandidateState(encoded, &decoded) !=
        HipStateConversionResult::CollisionReplacementOverflow) {
        std::cerr << "collision replacement overflow was not rejected\n";
        return 1;
    }
    encoded = valid;
    encoded.race.checkpointSlotsPassed.count = 1025u;
    if (DecodeHipCandidateState(encoded, &decoded) !=
        HipStateConversionResult::CheckpointOverflow) {
        std::cerr << "checkpoint overflow was not rejected\n";
        return 1;
    }
    encoded = valid;
    encoded.stuntEvents.count = 2049u;
    if (DecodeHipCandidateState(encoded, &decoded) !=
        HipStateConversionResult::StuntEventOverflow) {
        std::cerr << "stunt event overflow was not rejected\n";
        return 1;
    }
    encoded = valid;
    ++encoded.schemaVersion;
    if (DecodeHipCandidateState(encoded, &decoded) !=
        HipStateConversionResult::SchemaMismatch) {
        std::cerr << "state schema mismatch was not rejected\n";
        return 1;
    }
    return 0;
}
