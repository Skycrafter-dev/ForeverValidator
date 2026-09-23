// Generated from src/simulation/backends/cuda/cuda_stunts.cuh by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#ifndef FOREVERVALIDATOR_HIP_STUNTS_CUH
#define FOREVERVALIDATOR_HIP_STUNTS_CUH

#include <hip/hip_runtime.h>

#include <cstdint>

#include "simulation/backends/hip/generated/hip_exact_math.cuh"
#include "simulation/backends/hip/generated/hip_state_layout.h"
#include "simulation/backends/hip/generated/hip_timeline_executor.h"

namespace forevervalidator::simulation::hip::stunts {

enum class Status : std::uint32_t {
    Success = 0u,
    EventCapacityExceeded = 1u,
};

namespace detail {

constexpr float Pi = 3.1415927410125732f;
constexpr float HalfPi = 1.5707963705062866f;
constexpr float StraightLandingLimit = 0.39269909262657166f;
constexpr float ReverseLandingLimit = 2.7488937377929688f;
constexpr float LandingSpeedEpsilon = 0.000009999999747378752f;
constexpr std::uint32_t InterComboDelay = 2000u;
constexpr std::uint32_t MinimumStuntTime = 5u;
constexpr float FigureRepeatMalus = 0.2f;
constexpr float ChainBonus1 = 0.2f;
constexpr float ChainBonus2 = 0.15000001f;
constexpr float ChainBonus3 = 0.1f;
constexpr float ChainBonus4 = 0.050000001f;
constexpr float ChainBonus5 = 0.02f;
constexpr std::uint32_t RespawnPenalty = 50u;

__device__ inline float Dot(const GmVec3 &left, const GmVec3 &right) {
    const float xy = left.x * right.x + left.y * right.y;
    return xy + left.z * right.z;
}

__device__ inline GmMat3 MultiplyByTranspose(
        const GmMat3 &left,
        const GmMat3 &right) {
    GmMat3 result;
    result.basisX = {
            Dot(right.basisX, left.basisX),
            Dot(right.basisY, left.basisX),
            Dot(right.basisZ, left.basisX),
    };
    result.basisY = {
            Dot(right.basisX, left.basisY),
            Dot(right.basisY, left.basisY),
            Dot(right.basisZ, left.basisY),
    };
    result.basisZ = {
            Dot(right.basisX, left.basisZ),
            Dot(right.basisY, left.basisZ),
            Dot(right.basisZ, left.basisZ),
    };
    return result;
}

__device__ inline float MatrixElement(
        const GmMat3 &matrix,
        std::uint32_t row,
        std::uint32_t column) {
    const GmVec3 &basis = column == 0u
            ? matrix.basisX
            : (column == 1u ? matrix.basisY : matrix.basisZ);
    return row == 0u ? basis.x : (row == 1u ? basis.y : basis.z);
}

__device__ inline void SetQuaternionComponent(
        GmQuat &value, std::uint32_t index, float component) {
    if (index == 0u) value.x = component;
    else if (index == 1u) value.y = component;
    else value.z = component;
}

__device__ inline GmQuat QuaternionFromMatrix(const GmMat3 &matrix) {
    GmQuat result{};
    const float trace =
            (MatrixElement(matrix, 0u, 0u) +
             MatrixElement(matrix, 1u, 1u)) +
            MatrixElement(matrix, 2u, 2u);
    if (trace > 0.0f) {
        const float root = exact::Sqrt(trace + 1.0f);
        const float scale = exact::Divide(0.5f, root);
        result.w = root * 0.5f;
        result.x =
                (MatrixElement(matrix, 2u, 1u) -
                 MatrixElement(matrix, 1u, 2u)) *
                scale;
        result.y =
                (MatrixElement(matrix, 0u, 2u) -
                 MatrixElement(matrix, 2u, 0u)) *
                scale;
        result.z =
                (MatrixElement(matrix, 1u, 0u) -
                 MatrixElement(matrix, 0u, 1u)) *
                scale;
        return result;
    }

    constexpr std::uint32_t NextAxis[3] = {1u, 2u, 0u};
    std::uint32_t dominant = 0u;
    if (MatrixElement(matrix, 0u, 0u) <
        MatrixElement(matrix, 1u, 1u)) {
        dominant = 1u;
    }
    if (MatrixElement(matrix, dominant, dominant) <
        MatrixElement(matrix, 2u, 2u)) {
        dominant = 2u;
    }
    const std::uint32_t next = NextAxis[dominant];
    const std::uint32_t final = NextAxis[next];
    const float root = exact::Sqrt(
            (MatrixElement(matrix, dominant, dominant) -
             (MatrixElement(matrix, final, final) +
              MatrixElement(matrix, next, next))) +
            1.0f);
    const float scale = exact::Divide(0.5f, root);
    SetQuaternionComponent(result, dominant, root * 0.5f);
    result.w =
            (MatrixElement(matrix, final, next) -
             MatrixElement(matrix, next, final)) *
            scale;
    SetQuaternionComponent(result, next,
            (MatrixElement(matrix, dominant, next) +
             MatrixElement(matrix, next, dominant)) *
            scale);
    SetQuaternionComponent(result, final,
            (MatrixElement(matrix, dominant, final) +
             MatrixElement(matrix, final, dominant)) *
            scale);
    return result;
}

__device__ inline void QuaternionRotation(
        const GmQuat &rotation,
        float &angle,
        GmVec3 &axis) {
    axis = {rotation.x, rotation.y, rotation.z};
    const float lengthSquared =
            axis.z * axis.z + axis.y * axis.y + axis.x * axis.x;
    if (lengthSquared <= 9.9999994396249292e-11f) {
        axis = {1.0f, 0.0f, 0.0f};
        angle = 0.0f;
        return;
    }
    const float inverseLength =
            exact::Divide(1.0f, exact::Sqrt(lengthSquared));
    axis.x *= inverseLength;
    axis.y *= inverseLength;
    axis.z *= inverseLength;
    std::uint32_t dominant =
            fabsf(axis.x) < fabsf(axis.y) ? 1u : 0u;
    const float dominantMagnitude =
            dominant == 0u ? fabsf(axis.x) : fabsf(axis.y);
    if (dominantMagnitude < fabsf(axis.z)) {
        dominant = 2u;
    }
    const float quaternionComponent =
            dominant == 0u
                    ? rotation.x
                    : (dominant == 1u ? rotation.y : rotation.z);
    const float axisComponent =
            dominant == 0u
                    ? axis.x
                    : (dominant == 1u ? axis.y : axis.z);
    angle = 2.0f *
            exact::Atan2(
                    exact::Divide(
                            quaternionComponent, axisComponent),
                    rotation.w);
}

__device__ inline GmMat3 IdentityMatrix() {
    return {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
    };
}

__device__ inline GmIso4 IdentityIso() {
    return {IdentityMatrix(), {}};
}

__device__ inline GmVec3 PointInTakeoffFrame(
        const GmIso4 &takeoff,
        const GmVec3 &worldPoint) {
    const GmVec3 relative = {
            worldPoint.x - takeoff.translation.x,
            worldPoint.y - takeoff.translation.y,
            worldPoint.z - takeoff.translation.z,
    };
    return {
            Dot(takeoff.rotation.basisX, relative),
            Dot(takeoff.rotation.basisY, relative),
            Dot(takeoff.rotation.basisZ, relative),
    };
}

__device__ inline std::uint32_t RotationCount(
        float rotation,
        bool badLanding) {
    const float magnitude = fabsf(rotation);
    return static_cast<std::uint32_t>(
            badLanding ? exact::Divide(magnitude, Pi)
                       : exact::Divide(magnitude + HalfPi, Pi));
}

__device__ inline void Reset(HipStuntState &race) {
    race.stuntRotation = {};
    race.stuntTakeoffTick = UINT32_MAX;
    race.stuntLandingTick = UINT32_MAX;
    race.stuntInProgress = false;
    race.stuntMasterJump = false;
    race.stuntBadLanding = false;
    race.stuntLandingDirection = 0.0f;
    if (race.stuntLocationHistorySize != 0u) {
        race.stuntPreviousLocation = race.stuntLocationHistory[0];
    } else if (race.replayStuntStateAvailable) {
        race.stuntPreviousLocation =
                race.replayStuntState.vehicleLocation;
    } else {
        race.stuntPreviousLocation = IdentityIso();
    }
}

__device__ inline void PushInput(HipStuntState &race) {
    CTrackManiaRace::ReplayStuntInputSnapshot snapshot;
    snapshot.tickTimeMs = race.replayStuntState.tickTimeMs;
    snapshot.lastChangeTimeMs =
            race.replayStuntState.inputLastChangeTimeMs;
    constexpr std::uint32_t Capacity = 32u;
    if (race.stuntInputHistorySize == Capacity) {
        for (std::uint32_t index = 1u; index < Capacity; ++index) {
            race.stuntInputHistory[index - 1u] =
                    race.stuntInputHistory[index];
        }
        race.stuntInputHistory[Capacity - 1u] = snapshot;
    } else {
        race.stuntInputHistory[race.stuntInputHistorySize++] = snapshot;
    }
}

__device__ inline void PushLocation(HipStuntState &race) {
    constexpr std::uint32_t Capacity = 20u;
    if (race.stuntLocationHistorySize == Capacity) {
        for (std::uint32_t index = 1u; index < Capacity; ++index) {
            race.stuntLocationHistory[index - 1u] =
                    race.stuntLocationHistory[index];
        }
        race.stuntLocationHistory[Capacity - 1u] =
                race.replayStuntState.vehicleLocation;
    } else {
        race.stuntLocationHistory[race.stuntLocationHistorySize++] =
                race.replayStuntState.vehicleLocation;
    }
}

__device__ inline bool IsStuntTimeOver(
        const HipStuntState &race,
        std::uint32_t tick) {
    if (race.replayStuntsRaceStartTimeMs == UINT32_MAX ||
        race.replayStuntsRaceStartTimeMs >= tick) {
        return false;
    }
    return race.replayStuntsTimeLimitMs <
           tick - race.replayStuntsRaceStartTimeMs;
}

__device__ inline bool IsMasterJump(
        const HipStuntState &race,
        std::uint32_t startTimeMs,
        std::uint32_t endTimeMs) {
    const std::uint32_t queryStart =
            startTimeMs +
            race.replayStuntState.inputQueryTimeOffsetMs;
    const std::uint32_t queryEnd =
            endTimeMs +
            race.replayStuntState.inputQueryTimeOffsetMs;
    CTrackManiaRace::ReplayStuntInputSnapshot selected{};
    bool found = false;
    for (std::uint32_t index = race.stuntInputHistorySize;
         index != 0u; --index) {
        const auto candidate = race.stuntInputHistory[index - 1u];
        if (candidate.tickTimeMs <= queryEnd) {
            selected = candidate;
            found = true;
            break;
        }
    }
    if (!found) {
        return true;
    }
    for (std::uint32_t index = 0u; index < 6u; ++index) {
        if (selected.lastChangeTimeMs[index] > queryStart) {
            return false;
        }
    }
    return true;
}

__device__ inline Status AppendEvent(
        HipFixedArray<ReplayStuntEvent, 2048u> &stuntEvents,
        std::uint32_t figure,
        std::uint32_t degree,
        std::uint32_t score,
        float bonus,
        bool straightLanding,
        bool reverseLanding,
        bool masterJump,
        std::uint32_t chain) {
    if (stuntEvents.count >= 2048u) {
        return Status::EventCapacityExceeded;
    }
    ReplayStuntEvent &event =
            stuntEvents.values[stuntEvents.count++];
    event.figure = static_cast<EFigures>(figure);
    event.degree = degree;
    event.score = score;
    event.bonus = bonus;
    event.straightLanding = straightLanding;
    event.reverseLanding = reverseLanding;
    event.masterJump = masterJump;
    event.chain = chain;
    return Status::Success;
}

__device__ inline void UpdateRotation(HipStuntState &race) {
    const GmMat3 relative = MultiplyByTranspose(
            race.replayStuntState.vehicleLocation.rotation,
            race.stuntPreviousLocation.rotation);
    const GmQuat quaternion = QuaternionFromMatrix(relative);
    float angle = 0.0f;
    GmVec3 axis{};
    QuaternionRotation(quaternion, angle, axis);
    race.stuntRotation.x += axis.x * angle;
    race.stuntRotation.y += axis.y * angle;
    race.stuntRotation.z += axis.z * angle;
    race.stuntPreviousLocation =
            race.replayStuntState.vehicleLocation;
}

__device__ inline Status Compute(
        HipStuntState &race,
        HipFixedArray<ReplayStuntEvent, 2048u> &stuntEvents) {
    if (!race.replayStuntsEnabled ||
        race.stuntLandingTick == UINT32_MAX) {
        return Status::Success;
    }
    std::uint32_t duration = 0u;
    if (race.stuntTakeoffTick != UINT32_MAX) {
        duration =
                (race.stuntLandingTick - race.stuntTakeoffTick) / 100u;
        if (duration < MinimumStuntTime) {
            return Status::Success;
        }
    }

    const float absX = fabsf(race.stuntRotation.x);
    const float absY = fabsf(race.stuntRotation.y);
    const float absZ = fabsf(race.stuntRotation.z);
    const std::uint32_t countX =
            RotationCount(race.stuntRotation.x, race.stuntBadLanding);
    const std::uint32_t countY =
            RotationCount(race.stuntRotation.y, race.stuntBadLanding);
    const std::uint32_t countZ =
            RotationCount(race.stuntRotation.z, race.stuntBadLanding);

    std::uint32_t figure = 1u;
    std::uint32_t primaryCount = 0u;
    if (countZ == 0u) {
        if (countX == 0u) {
            if (countY != 0u) {
                if (fabsf(
                            race.stuntTakeoffLocation.rotation.basisY.y) >
                    0.20000000298023224f) {
                    figure = 4u;
                } else {
                    const GmVec3 relativeLanding = PointInTakeoffFrame(
                            race.stuntTakeoffLocation,
                            race.replayStuntState.vehicleLocation.translation);
                    figure =
                            (relativeLanding.x < 0.0f &&
                             race.stuntRotation.y > 0.0f) ||
                                            (relativeLanding.x > 0.0f &&
                                             race.stuntRotation.y < 0.0f)
                                    ? 6u
                                    : 5u;
                }
                primaryCount = countY;
            }
        } else if (countY == 0u) {
            figure = race.stuntRotation.x > 0.0f ? 2u : 3u;
            primaryCount = countX;
        } else if (absX > absY) {
            figure = 10u;
            primaryCount = countX;
        } else {
            figure = 8u;
            primaryCount = countY;
        }
    } else if (countX == 0u) {
        if (countY == 0u) {
            figure = 7u;
            primaryCount = countZ;
        } else if (absZ > absY) {
            figure = 12u;
            primaryCount = countZ;
        } else {
            figure = 9u;
            primaryCount = countY;
        }
    } else if (countY == 0u) {
        if (absZ <= absX) {
            figure = 11u;
            primaryCount = countX;
        } else {
            figure = 13u;
            primaryCount = countZ;
        }
    } else if (absZ > absX && absZ > absY) {
        figure = 16u;
        primaryCount = countZ;
    } else if (absX > absZ && absX > absY) {
        figure = 15u;
        primaryCount = countX;
    } else {
        figure = 14u;
        primaryCount = countY;
    }

    if (race.stuntBadLanding) {
        figure += 17u;
    }
    bool reverseLanding =
            fabsf(race.stuntLandingDirection) >= ReverseLandingLimit;
    bool straightLanding =
            fabsf(race.stuntLandingDirection) <= StraightLandingLimit;
    bool masterJump = false;
    if (figure >= 2u && figure <= 16u &&
        race.stuntLandingTick >= 100u) {
        masterJump = IsMasterJump(
                race,
                race.stuntTakeoffTick + 100u,
                race.stuntLandingTick - 100u);
        if (!straightLanding && !reverseLanding) {
            masterJump = false;
        }
    }

    const std::uint32_t previousLanding =
            race.stuntPreviousLandingTick;
    const std::uint32_t landingGap =
            previousLanding == UINT32_MAX
                    ? 0u
                    : race.stuntTakeoffTick - previousLanding;
    if (previousLanding == UINT32_MAX ||
        landingGap > race.stuntComboWindowMs) {
        race.stuntChain = 0u;
    } else {
        ++race.stuntChain;
    }
    race.stuntPreviousLandingTick = race.stuntLandingTick;

    if (race.stuntBadLanding) {
        masterJump = false;
        straightLanding = false;
        reverseLanding = false;
    }
    std::uint32_t baseScore =
            duration + 15u * countX +
            10u * (countY + 2u * countZ);
    if (figure == 6u) {
        baseScore += 5u;
    } else if (figure == 1u) {
        baseScore >>= 1u;
    }
    if (race.stuntBadLanding) {
        baseScore >>= 1u;
    }

    float bonus = 1.0f;
    if (reverseLanding || straightLanding) {
        bonus = 1.25f;
    }
    if (masterJump) {
        bonus += 0.25f;
    }
    if (race.stuntChain != 0u) bonus += ChainBonus1;
    if (race.stuntChain > 1u) bonus += ChainBonus2;
    if (race.stuntChain > 2u) bonus += ChainBonus3;
    if (race.stuntChain > 3u) bonus += ChainBonus4;
    if (race.stuntChain > 4u) {
        bonus +=
                static_cast<float>(race.stuntChain - 4u) *
                ChainBonus5;
    }

    float repeatMalus =
            exact::Divide(
                    static_cast<float>(race.stuntFigureScores[figure]) *
                            FigureRepeatMalus,
                    100.0f);
    if (0.75f < repeatMalus) repeatMalus = 0.75f;
    std::uint32_t repeatedScore = static_cast<std::uint32_t>(
            (1.0f - repeatMalus) *
            static_cast<float>(baseScore));
    if (repeatedScore == 0u) repeatedScore = 1u;
    const std::uint32_t score = static_cast<std::uint32_t>(
            static_cast<float>(repeatedScore) * bonus);
    if (IsStuntTimeOver(
                race,
                race.replayStuntState.tickTimeMs +
                        race.replayStuntState.inputQueryTimeOffsetMs)) {
        return Status::Success;
    }

    race.stuntsScore += score;
    std::uint32_t interComboDelay = InterComboDelay;
    if (race.stuntComboWindowMs != 0u &&
        race.stuntComboWindowMs > InterComboDelay + landingGap) {
        interComboDelay = race.stuntComboWindowMs - landingGap;
    }
    race.stuntComboWindowMs = interComboDelay + 20u * score;
    const Status eventStatus = AppendEvent(
            stuntEvents, figure, 180u * primaryCount, score, bonus,
            straightLanding, reverseLanding, masterJump,
            race.stuntChain);
    if (eventStatus != Status::Success) {
        return eventStatus;
    }
    race.stuntFigureScores[figure] += score;
    return Status::Success;
}

}  // namespace detail

__device__ inline void Configure(
        HipRaceState &race,
        bool enabled,
        std::uint32_t timeLimitMs) {
    HipStuntState &stunts = race.stunts;
    stunts.replayStuntsEnabled = enabled;
    stunts.replayStuntStateAvailable = false;
    stunts.replayStuntsTimeLimitMs = timeLimitMs;
    stunts.replayStuntsRaceStartTimeMs = UINT32_MAX;
    stunts.replayStuntState = {};
    stunts.stuntInputHistorySize = 0u;
    stunts.stuntLocationHistorySize = 0u;
    stunts.stuntPreviousLocation = detail::IdentityIso();
    stunts.stuntTakeoffLocation = detail::IdentityIso();
    stunts.stuntPreviousLandingTick = UINT32_MAX;
    stunts.stuntChain = 0u;
    stunts.stuntComboWindowMs = 0u;
    stunts.stuntScoreAtTimeLimit = {};
    for (std::uint32_t index = 0u; index < 39u; ++index) {
        stunts.stuntFigureScores[index] = 0u;
    }
    stunts.stuntsScore = 0u;
    race.stuntEvents = {};
    detail::Reset(stunts);
}

__device__ inline void ApplyRespawnPenalty(HipStuntState &race) {
    if (!race.replayStuntsEnabled) return;
    const std::uint32_t penalty =
            race.stuntsScore < detail::RespawnPenalty
                    ? race.stuntsScore
                    : detail::RespawnPenalty;
    race.stuntsScore -= penalty;
    detail::Reset(race);
}

__device__ inline Status ApplyTimePenalty(
        HipRaceState &race,
        std::uint32_t overtimeMs) {
    HipStuntState &stunts = race.stunts;
    if (!stunts.replayStuntsEnabled) return Status::Success;
    const std::uint32_t penalty = overtimeMs / 100u;
    const std::uint32_t scoreAtTimeLimit =
            stunts.stuntScoreAtTimeLimit.present
                    ? stunts.stuntScoreAtTimeLimit.value
                    : stunts.stuntsScore;
    stunts.stuntsScore =
            penalty < scoreAtTimeLimit
                    ? scoreAtTimeLimit - penalty
                    : 0u;
    if (penalty == 0u) return Status::Success;
    stunts.stuntComboWindowMs = detail::InterComboDelay;
    return detail::AppendEvent(
            race.stuntEvents, 34u, 0u, penalty, 0.0f,
            false, false, false, 0u);
}

__device__ inline Status UpdateState(
        HipStuntState &race,
        HipFixedArray<ReplayStuntEvent, 2048u> &stuntEvents,
        const ReplayStuntSimulationState &state) {
    race.replayStuntState = state;
    race.replayStuntStateAvailable = true;
    if (state.raceStart) {
        race.replayStuntsRaceStartTimeMs = state.tickTimeMs;
        race.stuntScoreAtTimeLimit = {};
        detail::Reset(race);
    }
    if (!race.replayStuntsEnabled) {
        return Status::Success;
    }

    const std::uint32_t currentTick =
            race.replayStuntState.tickTimeMs;
    if (race.replayStuntState.finishRace) {
        if (!race.stuntScoreAtTimeLimit.present &&
            detail::IsStuntTimeOver(race, currentTick)) {
            race.stuntScoreAtTimeLimit.present = true;
            race.stuntScoreAtTimeLimit.value = race.stuntsScore;
        }
        return Status::Success;
    }
    detail::PushInput(race);
    detail::PushLocation(race);

    const bool hasGroundContact =
            race.replayStuntState.hasWheelContact ||
            race.replayStuntState.hasBodyContact;
    if (race.stuntInProgress) {
        if (race.replayStuntState.noGroundFrictionGuard) {
            detail::Reset(race);
        }
        if (hasGroundContact) {
            if (race.replayStuntState.hasBodyContact &&
                (race.replayStuntState.bodyContactHorizontalAngle > 0.5f ||
                 race.replayStuntState.bodyContactVerticalAngle > 0.5f)) {
                race.stuntBadLanding = true;
            }
            race.stuntLandingTick = currentTick;
            const float speedMagnitude = exact::Sqrt(
                    race.replayStuntState.forwardSpeed *
                                    race.replayStuntState.forwardSpeed +
                            race.replayStuntState.sideSpeed *
                                    race.replayStuntState.sideSpeed);
            race.stuntLandingDirection =
                    speedMagnitude <= detail::LandingSpeedEpsilon
                            ? 0.0f
                            : exact::Atan2(
                                      exact::Divide(
                                              race.replayStuntState.sideSpeed,
                                              speedMagnitude),
                                      exact::Divide(
                                              race.replayStuntState.forwardSpeed,
                                              speedMagnitude));
            const Status status =
                    detail::Compute(race, stuntEvents);
            if (status != Status::Success) return status;
            detail::Reset(race);
        } else {
            race.stuntLandingTick = UINT32_MAX;
            const std::uint32_t masterStart =
                    race.stuntTakeoffTick + 100u;
            if (currentTick <= race.stuntTakeoffTick + 200u) {
                race.stuntMasterJump = true;
            } else if (race.stuntMasterJump) {
                race.stuntMasterJump = detail::IsMasterJump(
                        race, masterStart, currentTick - 100u);
            }
            detail::UpdateRotation(race);
        }
    }

    if (!race.stuntInProgress && !hasGroundContact &&
        !race.replayStuntState.noGroundFrictionGuard &&
        race.replayStuntsRaceStartTimeMs != UINT32_MAX &&
        currentTick > race.replayStuntsRaceStartTimeMs) {
        detail::Reset(race);
        race.stuntTakeoffTick = currentTick;
        race.stuntInProgress = true;
        race.stuntTakeoffLocation =
                race.replayStuntState.vehicleLocation;
    }
    if (!race.stuntScoreAtTimeLimit.present &&
        detail::IsStuntTimeOver(race, currentTick)) {
        race.stuntScoreAtTimeLimit.present = true;
        race.stuntScoreAtTimeLimit.value = race.stuntsScore;
    }
    return Status::Success;
}

__device__ inline Status UpdateState(
        HipRaceState &race,
        const ReplayStuntSimulationState &state) {
    return UpdateState(
            race.stunts, race.stuntEvents, state);
}

__device__ inline Status Update(
        HipCandidateState &candidate,
        const HipControlTick &tick) {
    ReplayStuntSimulationState state;
    state.tickTimeMs = tick.timeMs;
    state.inputQueryTimeOffsetMs = tick.periodMs;
    state.raceStart =
            (tick.actionFlags &
             HipControlActionResetAtRaceStart) != 0u;
    state.finishRace =
            (tick.actionFlags &
             HipControlActionFinishRace) != 0u;
    state.vehicleLocation = {
            candidate.body.write.rotation,
            candidate.body.write.position,
    };
    const auto &physics =
            candidate.vehicle.frameHistory.physicsCurrent;
    state.forwardSpeed = physics.forwardSpeed;
    state.sideSpeed = physics.sideSpeed;
    state.hasWheelContact = physics.hasWheelContact;
    state.hasBodyContact = physics.hasBodyContact;
    state.bodyContactVerticalAngle =
            physics.bodyContactVerticalAngle;
    state.bodyContactHorizontalAngle =
            physics.bodyContactHorizontalAngle;
    state.noGroundFrictionGuard =
            physics.noGroundFrictionGuard;
    state.inputLastChangeTimeMs =
            tick.stuntsInput.lastChangeTimeMs;
    return UpdateState(
            candidate.stunts, candidate.stuntEvents, state);
}

}  // namespace forevervalidator::simulation::hip::stunts

#endif
