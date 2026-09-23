// Generated from src/simulation/backends/cuda/cuda_timeline_executor.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_TIMELINE_EXECUTOR_H
#define FOREVERVALIDATOR_HIP_TIMELINE_EXECUTOR_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "simulation/backends/hip/generated/hip_state_layout.h"

namespace forevervalidator::simulation {

struct HipControlTick {
    std::uint32_t periodMs = 0u;
    std::uint32_t timeMs = 0u;
    std::uint32_t actionFlags = 0u;
    std::uint32_t stuntsTimeLimitMs = 0u;
    std::uint32_t respawnAtCheckpointCount = 0u;
    ReplayVehicleControlState controls{};
    ReplayStuntInputState stuntsInput{};
    bool observe = false;
    bool hasComparisonTarget = false;
    GmVec3 comparisonTarget{};
};

enum HipControlActionFlag : std::uint32_t {
    HipControlActionEstablishRaceSpawn = 1u << 0u,
    HipControlActionSuppressVehicleForceCallbacks = 1u << 1u,
    HipControlActionEnableRaceSimulation = 1u << 2u,
    HipControlActionResetAtRaceStart = 1u << 3u,
    HipControlActionEnableStuntsSimulation = 1u << 4u,
    HipControlActionFinishRace = 1u << 5u,
};

struct HipTimelineObservation {
    GmVec3 simulatedPosition{};
    GmVec3 writePosition{};
    bool hasComparison = false;
    GmVec3 comparisonTarget{};
    GmVec3 comparisonDelta{};
    float comparisonDistance = 0.0f;
    bool hasFinishTick = false;
    std::uint32_t finishTickMs = 0u;
};

enum class HipTimelineStatus : std::uint32_t {
    Success,
    InvalidArgument,
    SchemaMismatch,
    CapacityExceeded,
    Cancelled,
    DeviceFailure,
    UnsupportedPhysicsTransition,
};

struct HipCandidateTimelineInput {
    HipCandidateState initialState{};
    std::vector<HipControlTick> ticks;
};

struct HipCandidateTimelineOutput {
    HipTimelineStatus status = HipTimelineStatus::InvalidArgument;
    std::uint32_t failureTick = UINT32_MAX;
    std::uint32_t failureDetail = 0u;
    std::uint32_t executedTickCount = 0u;
    std::uint32_t executedRespawnCount = 0u;
    HipCandidateState finalState{};
    std::vector<HipTimelineObservation> observations;
};

struct HipTimelineExecutionMetrics {
    std::uint64_t candidateCount = 0u;
    std::uint64_t tickCount = 0u;
    std::uint64_t observationCapacity = 0u;
    std::uint64_t hostToDeviceBytes = 0u;
    std::uint64_t deviceToHostBytes = 0u;
    std::uint64_t peakDeviceBytes = 0u;
    double allocationMilliseconds = 0.0;
    double transferMilliseconds = 0.0;
    double kernelMilliseconds = 0.0;
    double synchronizationMilliseconds = 0.0;
};

struct HipTimelineBatchResult {
    HipTimelineStatus status = HipTimelineStatus::InvalidArgument;
    std::vector<HipCandidateTimelineOutput> candidates;
    HipTimelineExecutionMetrics metrics{};
    std::optional<std::size_t> winnerCandidateIndex;
    std::optional<std::uint32_t> winnerCandidateId;
    std::string diagnostic;
};

HipControlTick FlattenHipControlTick(
        const ReplayControlTick &source) noexcept;

HipTimelineBatchResult ExecuteHipTimelineBatch(
        const void *deviceScene,
        const void *deviceStaticConfiguration,
        const std::vector<HipCandidateTimelineInput> &candidates,
        bool cancellationRequested = false) noexcept;

const char *HipTimelineStatusName(HipTimelineStatus status) noexcept;

std::optional<std::size_t> SelectHipTimelineWinner(
        const std::vector<HipCandidateTimelineOutput> &candidates) noexcept;

}  // namespace forevervalidator::simulation

#endif
