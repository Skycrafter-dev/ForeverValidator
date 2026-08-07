#include "simulation/backends/cuda/cuda_state_layout.h"
#include "simulation/runtime/replay_simulation_session.h"

#include <cstring>
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
    using forevervalidator::simulation::CudaCandidateState;
    using forevervalidator::simulation::CudaRaceState;
    using forevervalidator::simulation::CudaStateConversionResult;
    using forevervalidator::simulation::DecodeCudaCandidateState;
    using forevervalidator::simulation::EncodeCudaRaceState;
    ReplaySimulationInstanceClone original = BuildState();
    CudaRaceState encodedRaceA;
    CudaRaceState encodedRaceB;
    std::memset(&encodedRaceA, 0x5a, sizeof(encodedRaceA));
    std::memset(&encodedRaceB, 0xa5, sizeof(encodedRaceB));
    if (EncodeCudaRaceState(original.race, &encodedRaceA) !=
                CudaStateConversionResult::Success ||
        EncodeCudaRaceState(original.race, &encodedRaceB) !=
                CudaStateConversionResult::Success ||
        std::memcmp(
                &encodedRaceA,
                &encodedRaceB,
                sizeof(encodedRaceA)) != 0) {
        std::cerr << "race encode left destination-dependent bytes\n";
        return 1;
    }
    CudaCandidateState encodedA;
    CudaCandidateState encodedB;
    std::memset(&encodedA, 0x3c, sizeof(encodedA));
    std::memset(&encodedB, 0xc3, sizeof(encodedB));
    const CudaStateConversionResult encodeA =
            forevervalidator::simulation::EncodeCudaCandidateState(
                    original, 42u, 987u, 11u, 1234567u, &encodedA);
    const CudaStateConversionResult encodeB =
            forevervalidator::simulation::EncodeCudaCandidateState(
                    original, 42u, 987u, 11u, 1234567u, &encodedB);
    if (encodeA != CudaStateConversionResult::Success ||
        encodeB != CudaStateConversionResult::Success ||
        std::memcmp(&encodedA, &encodedB, sizeof(encodedA)) != 0) {
        std::cerr << "candidate encode left destination-dependent bytes\n";
        return 1;
    }
    CudaCandidateState encoded = encodedA;
    if (encoded.candidateId != 11u ||
        encoded.validationSeed != 42u ||
        encoded.controlCursor != 987u ||
        encoded.randomState != 1234567u) {
        std::cerr << "candidate-owned metadata was not captured\n";
        return 1;
    }
    ReplaySimulationInstanceClone decoded;
    const CudaStateConversionResult decode =
            forevervalidator::simulation::DecodeCudaCandidateState(
                    encoded, &decoded);
    CudaCandidateState reencoded;
    std::memset(&reencoded, 0x69, sizeof(reencoded));
    const CudaStateConversionResult reencode =
            forevervalidator::simulation::EncodeCudaCandidateState(
                    decoded, 42u, 987u, 11u, 1234567u,
                    &reencoded);
    if (decode != CudaStateConversionResult::Success ||
        reencode != CudaStateConversionResult::Success ||
        std::memcmp(&encodedA, &reencoded, sizeof(encodedA)) != 0 ||
        decoded.incrementalRespawnCount != 3u ||
        decoded.runtime.finishTime != original.runtime.finishTime) {
        std::cerr << "state round trip changed CUDA transport data\n";
        return 1;
    }
    ReplaySimulationInstanceClone checkpointBoundary = BuildState();
    checkpointBoundary.race.checkpointSlotsPassed.resize(
            forevervalidator::simulation::CudaCheckpointSlotCapacity);
    for (std::size_t index = 0u;
         index < checkpointBoundary.race.checkpointSlotsPassed.size();
         index += 3u) {
        checkpointBoundary.race.checkpointSlotsPassed[index] = 1u;
    }
    CudaCandidateState checkpointBoundaryEncoded;
    CudaCandidateState checkpointBoundaryRoundtrip;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                checkpointBoundary, 42u, 987u, 11u, 1234567u,
                &checkpointBoundaryEncoded) !=
                    CudaStateConversionResult::Success ||
        DecodeCudaCandidateState(checkpointBoundaryEncoded, &decoded) !=
                CudaStateConversionResult::Success ||
        forevervalidator::simulation::EncodeCudaCandidateState(
                decoded, 42u, 987u, 11u, 1234567u,
                &checkpointBoundaryRoundtrip) !=
                    CudaStateConversionResult::Success ||
        std::memcmp(
                &checkpointBoundaryEncoded,
                &checkpointBoundaryRoundtrip,
                sizeof(checkpointBoundaryEncoded)) != 0) {
        std::cerr << "checkpoint boundary did not round trip\n";
        return 1;
    }
    ReplaySimulationInstanceClone replacementBoundary = BuildState();
    replacementBoundary.runtime.body.pendingCollisionReplacements.resize(
            forevervalidator::simulation::
                    CudaCollisionReplacementCapacity);
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
    CudaCandidateState boundaryEncoded;
    CudaCandidateState boundaryRoundtrip;
    if (forevervalidator::simulation::EncodeCudaCandidateState(
                replacementBoundary, 42u, 987u, 11u, 1234567u,
                &boundaryEncoded) != CudaStateConversionResult::Success ||
        DecodeCudaCandidateState(boundaryEncoded, &decoded) !=
                CudaStateConversionResult::Success ||
        forevervalidator::simulation::EncodeCudaCandidateState(
                decoded, 42u, 987u, 11u, 1234567u,
                &boundaryRoundtrip) !=
                    CudaStateConversionResult::Success ||
        std::memcmp(
                &boundaryEncoded, &boundaryRoundtrip,
                sizeof(boundaryEncoded)) != 0) {
        std::cerr << "collision replacement boundary did not round trip\n";
        return 1;
    }
    const CudaCandidateState valid = encoded;
    encoded.vehicle.wheels.count = 5u;
    if (DecodeCudaCandidateState(encoded, &decoded) !=
        CudaStateConversionResult::WheelOverflow) {
        std::cerr << "state overflow was not rejected\n";
        return 1;
    }
    encoded = valid;
    encoded.collisionReplacementOverflow.count =
            forevervalidator::simulation::
                    CudaCollisionReplacementOverflowCapacity + 1u;
    if (DecodeCudaCandidateState(encoded, &decoded) !=
        CudaStateConversionResult::CollisionReplacementOverflow) {
        std::cerr << "collision replacement overflow was not rejected\n";
        return 1;
    }
    encoded = valid;
    encoded.race.checkpointSlotsPassed.count = 1025u;
    if (DecodeCudaCandidateState(encoded, &decoded) !=
        CudaStateConversionResult::CheckpointOverflow) {
        std::cerr << "checkpoint overflow was not rejected\n";
        return 1;
    }
    encoded = valid;
    encoded.stuntEvents.count = 2049u;
    if (DecodeCudaCandidateState(encoded, &decoded) !=
        CudaStateConversionResult::StuntEventOverflow) {
        std::cerr << "stunt event overflow was not rejected\n";
        return 1;
    }
    encoded = valid;
    ++encoded.schemaVersion;
    if (DecodeCudaCandidateState(encoded, &decoded) !=
        CudaStateConversionResult::SchemaMismatch) {
        std::cerr << "state schema mismatch was not rejected\n";
        return 1;
    }
    return 0;
}
