// Twin GPU engine: the session judge's targets beyond the hand-written ones (tw_gpu_session.cu OnTick): condition
// programs, scripted objectives, checkpoint and distance targets, custom volumes, the entry speed option. The
// evaluation is FV's own CUDA code (extract_cuda_search.py --judge: the evaluator configuration, DeviceSample,
// EvaluateState and the per-tick functions, qualifiers kept) over stand-ins of FV's state types that carry only the
// members it reads, filled from the twin state as the host's TwinPhysics::ReadFull fills FV's. The judge order is
// the kernel's (SimulateSearchCandidatesKernel): condition, scripted, distance reached, checkpoint, EvaluateState.
// FV-free: no FV header; the generated file is FV source copied at build time.
#ifndef TW_GPU_JUDGE_CUH
#define TW_GPU_JUDGE_CUH

#include <cstddef>
#include <cstdint>

#include <forevervalidator/experimental/around_walls_table.h>

#include "tw_gpu_session.h"

namespace tw::gpu::fvjudge {

// ---- FV's state types, the members the evaluator reads ----
struct GmVec3 {
    float x, y, z;
};
struct GmQuat {
    float x, y, z, w;
};
struct GmMat3 {
    GmVec3 basisX, basisY, basisZ;
};
namespace CHmsDyna {
struct CHmsStateDyna {
    GmQuat rotationQuat;
    GmMat3 rotation;
    GmVec3 position, linearSpeed, angularSpeed;
};
}  // namespace CHmsDyna
namespace CSceneVehicleCar {
enum ETurboType { ETurboType_Inactive = 0, ETurboType_Direct = 1, ETurboType_Roulette = 2 };
}
enum class EChallengePlayMode : std::uint32_t { Race = 0u, Platform = 1u, Puzzle = 2u, Crazy = 3u, Shortcut = 4u, Stunts = 5u };
struct ReplayRaceProgress {
    std::uint32_t checkpointCount, finishCount, lastPrepareTimeMs, lastAcceptedBlockId, currentLapCheckpointCount,
            totalCheckpointEventCount, completedLapCount, requiredLapCount, requiredCheckpointCount;
    bool raceCompleted;
};
struct CudaCheckpointSlots {
    static constexpr std::size_t WordBits = 32u;
    static constexpr std::size_t WordCount = 2u;   // the twin's slots (race::kMaxSlots = 64)
    std::uint32_t words[WordCount];
};
struct WheelRealTime {
    bool contactPresent, slipping;
    std::uint32_t contactMaterial;
};
struct Vehicle {
    struct {
        struct {
            GmVec3 localLinearSpeed;
        } physicsCurrent;
    } frameHistory;
    struct {
        bool forcedLowSpeedFriction;
    } controls;
    struct {
        bool lateralSlowDownContactActive;
    } contacts;
    struct {
        std::uint32_t count;
        struct {
            WheelRealTime realTime;
        } values[4];
    } wheels;
    struct {
        bool useLowSpeedGateB;
        int gearIndex;
        float engineInputMemory;
    } engine;
    struct {
        float steerAngle;
    } radiusSteering;
    struct {
        CSceneVehicleCar::ETurboType type;
        float type2Phase, impulseScale;
    } turbo;
};
struct CudaCandidatePhysicsState {
    struct {
        CHmsDyna::CHmsStateDyna current, temporary;
    } body;
    Vehicle vehicle;
    struct {
        CudaCheckpointSlots checkpointSlotsPassed;
        std::uint32_t replayPlayMode;
        ReplayRaceProgress progress;
    } race;
};
namespace cuda::facts {
__device__ inline std::uint32_t WheelCount(const Vehicle &vehicle) { return vehicle.wheels.count; }
}  // namespace cuda::facts
struct CudaPackedSceneHeader {};
// The twin's progress names the slot of the tick's last accepted passage in lastAcceptedBlockId.
__device__ inline std::uint32_t CheckpointSlotOfBlock(const CudaPackedSceneHeader *, std::uint32_t slot) { return slot; }
namespace experimental::around_walls {
using ::forevervalidator::experimental::around_walls::Path;
// The query with a corner it needs not filled yet: marked for the host (as PathTo marks it) and answered without the
// table's fallback search, which every tick of every lane would run until the batch ends; the batch runs again once
// the host has filled the marked corners (the session's FillDeviceMissedCorners), so this answer is never used.
__device__ inline Path PathTo(const unsigned char *bytes, double x, double y, double z) {
    namespace aw = ::forevervalidator::experimental::around_walls;
    const aw::detail::Table table(bytes);
    const aw::Corner *corners = table.Array<aw::Corner>(table.header->corners);
    const aw::detail::V center{x, y, z};
    std::uint64_t around[8];
    table.CornersAround(center, around);
    std::uint64_t floorPoints[aw::kFloorPoints];
    const int floorCount = table.FloorPointsAround(center, floorPoints);
    bool missing = false;
    for (int c = 0; c < 8; ++c) {
        if (corners[around[c]].direct == aw::kCornerUnfilled) {
            aw::detail::MarkMissed(bytes, around[c]);
            missing = true;
        }
    }
    for (int f = 0; f < floorCount; ++f) {
        if (corners[floorPoints[f]].direct == aw::kCornerUnfilled) {
            aw::detail::MarkMissed(bytes, floorPoints[f]);
            missing = true;
        }
    }
    return missing ? Path{} : aw::PathTo(bytes, x, y, z);
}
}  // namespace experimental::around_walls

// Verbatim CUDA source: its switches cover only the kinds each function handles (nvcc does not warn about the others).
#define FV_VOLUME_HD __host__ __device__
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch"
#endif
#include "twin_cuda_judge_extract.inc"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#undef FV_VOLUME_HD

// ---- the twin state as the evaluator's state (TwinPhysics::ReadFull) ----
__device__ inline void SetBody(CHmsDyna::CHmsStateDyna &b, const float pos[3], const float vel[3], const float ang[3],
                               const float q[4]) {
    b.position = {pos[0], pos[1], pos[2]};
    b.linearSpeed = {vel[0], vel[1], vel[2]};
    b.angularSpeed = {ang[0], ang[1], ang[2]};
    b.rotationQuat = {q[0], q[1], q[2], q[3]};
    // GmMat3::Set(GmQuat)
    const float twoX = q[0] * 2.0f, twoY = q[1] * 2.0f, twoZ = q[2] * 2.0f;
    const float xw2 = q[3] * twoX, yw2 = q[3] * twoY, zw2 = q[3] * twoZ;
    const float xx2 = q[0] * twoX, xy2 = q[0] * twoY, xz2 = q[0] * twoZ;
    const float yy2 = q[1] * twoY, yz2 = q[1] * twoZ, zz2 = q[2] * twoZ;
    b.rotation.basisX = {__fsub_rn(__fsub_rn(1.0f, yy2), zz2), __fadd_rn(xy2, zw2), __fsub_rn(xz2, yw2)};
    b.rotation.basisY = {__fsub_rn(xy2, zw2), __fsub_rn(__fsub_rn(1.0f, xx2), zz2), __fadd_rn(yz2, xw2)};
    b.rotation.basisZ = {__fadd_rn(xz2, yw2), __fsub_rn(yz2, xw2), __fsub_rn(__fsub_rn(1.0f, xx2), yy2)};
}

__device__ inline void ReadState(const model::TwinState &s, const PreviousTick &previous, std::uint32_t cpLastSlot,
                                 const JudgeConfig &c, CudaCandidatePhysicsState &state) {
    SetBody(state.body.current, s.pos, s.vel, s.angVel, s.quat);
    SetBody(state.body.temporary, previous.position, previous.velocity, previous.angular, previous.quat);
    const CHmsDyna::CHmsStateDyna &b = state.body.current;
    const auto dot = [](const GmVec3 &a, const GmVec3 &v) {
        return __fadd_rn(__fadd_rn(__fmul_rn(a.x, v.x), __fmul_rn(a.y, v.y)), __fmul_rn(a.z, v.z));
    };
    state.vehicle.frameHistory.physicsCurrent.localLinearSpeed = {
            dot(b.linearSpeed, b.rotation.basisX), dot(b.linearSpeed, b.rotation.basisY), dot(b.linearSpeed, b.rotation.basisZ)};
    state.vehicle.controls.forcedLowSpeedFriction = s.freewheel != 0u;
    state.vehicle.contacts.lateralSlowDownContactActive = s.latContact != 0u;
    state.vehicle.wheels.count = 4u;
    for (int i = 0; i < 4; ++i) {
        state.vehicle.wheels.values[i].realTime = {s.contact[i] != 0u, s.slipping[i] != 0u, s.surface[i]};
    }
    state.vehicle.engine.useLowSpeedGateB = s.reverse != 0u;
    state.vehicle.engine.gearIndex = s.gear;
    state.vehicle.engine.engineInputMemory = s.rpm;
    state.vehicle.radiusSteering.steerAngle = 0.0f;   // the legacy steering model's (not a twin car)
    state.vehicle.turbo.type = static_cast<CSceneVehicleCar::ETurboType>(s.trbType);
    state.vehicle.turbo.type2Phase = s.trbPhase;
    state.vehicle.turbo.impulseScale = s.trbScale;
    state.race.checkpointSlotsPassed.words[0] = s.cpSlots[0];
    state.race.checkpointSlotsPassed.words[1] = s.cpSlots[1];
    state.race.replayPlayMode = c.playMode;
    ReplayRaceProgress &r = state.race.progress;
    r.checkpointCount = s.cpCount;
    r.finishCount = s.finishCount;
    r.lastPrepareTimeMs = s.tickMs;
    r.lastAcceptedBlockId = cpLastSlot;
    r.currentLapCheckpointCount = s.lapCpCount;
    r.totalCheckpointEventCount = s.cpEvents;
    r.completedLapCount = s.laps;
    r.requiredLapCount = c.requiredLaps;
    r.requiredCheckpointCount = c.requiredCheckpoints;
    r.raceCompleted = s.raceCompleted != 0u;
}

__device__ inline void SetBest(JudgeState &j, const DeviceSample &s, std::uint32_t evaluationTick) {
    j.score = s.score;
    j.timeMs = s.timeMs;
    j.detail0 = s.detail0;
    j.detail1 = s.detail1;
    j.evaluationTick = evaluationTick;
    j.valid = 1u;
    j.preciseFinish = s.preciseFinish ? 1u : 0u;
}

__device__ inline std::uint8_t Invalid(JudgeState &j) {   // TickJudge::Invalidate
    j.score = j.timeMs = j.detail0 = j.detail1 = 0.0;
    j.evaluationTick = 0u;
    j.valid = 0u;
    j.preciseFinish = 0u;
    return kJudgeInvalid;
}

// TickJudge::OnTick for the targets above (the kernel's order); scripted: the job's objective scores then metric
// values (2 x CudaSearchMaximumScriptedObjectives doubles).
__device__ inline std::uint8_t OnTickTargets(const JudgeConfig &c, std::uint32_t t, const model::TwinState &s,
                                             const PreviousTick &previous, std::uint32_t cpLastSlot, bool finished,
                                             std::uint64_t finishLowerNs, JudgeState &j, double *scripted) {
    const CudaSearchEvaluatorConfiguration &e = *static_cast<const CudaSearchEvaluatorConfiguration *>(c.evaluator);
    const CudaSearchEvaluatorKind kind = e.kind;
    const std::int64_t publicTime = c.branchTimeMs + static_cast<std::int64_t>(t + 1u) * c.tickDurationMs;
    if (publicTime < c.evaluationStartTimeMs) return kJudgeContinue;
    CudaCandidatePhysicsState state;
    ReadState(s, previous, cpLastSlot, c, state);
    const bool completed = state.race.progress.raceCompleted;
    if (c.conditionCount != 0u &&
        !EvaluateCondition(static_cast<const CudaSearchConditionInstruction *>(c.condition), c.conditionCount, state, 0u,
                           c.clock[0], c.clock[1], c.clock[2])) {
        ++j.evaluationIndex;
        return completed ? kJudgeDone : kJudgeContinue;
    }
    const CheckpointProgressBefore before{previous.counters[0], previous.counters[1], previous.counters[2],
                                          previous.counters[3]};
    if (kind == CudaSearchEvaluatorKind::Scripted) {
        DeviceSample sample;
        sample.valid = j.valid != 0u;
        for (std::uint32_t i = 0u; i < e.scriptedObjectiveCount; ++i) {
            sample.objectiveScores[i] = scripted[i];
            sample.metricValues[i] = scripted[CudaSearchMaximumScriptedObjectives + i];
        }
        if (UpdateScriptedSample(e, state, 0u, c.clock[0], c.clock[1], c.clock[2], static_cast<double>(publicTime), &sample)) {
            SetBest(j, sample, j.evaluationIndex);
            for (std::uint32_t i = 0u; i < e.scriptedObjectiveCount; ++i) {
                scripted[i] = sample.objectiveScores[i];
                scripted[CudaSearchMaximumScriptedObjectives + i] = sample.metricValues[i];
            }
        }
        ++j.evaluationIndex;
        return completed ? kJudgeDone : kJudgeContinue;
    }
    if (kind == CudaSearchEvaluatorKind::Distance && DistanceTargetReached(e, state, before, nullptr)) {
        if (!j.valid || static_cast<double>(publicTime) < j.score) {
            DeviceSample sample;
            sample.score = static_cast<double>(publicTime);
            sample.timeMs = static_cast<double>(publicTime);
            SetBest(j, sample, j.evaluationIndex);
        }
        return kJudgeDone;
    }
    if (kind == CudaSearchEvaluatorKind::CheckpointEvent) {
        if (CheckpointEventAccepted(e, state, before, nullptr)) {
            DeviceSample sample;
            sample.score = static_cast<double>(publicTime);
            sample.timeMs = static_cast<double>(publicTime);
            SetBest(j, sample, j.evaluationIndex);
            return kJudgeDone;
        }
        ++j.evaluationIndex;
        return completed ? kJudgeDone : kJudgeContinue;
    }
    const bool finishTime = kind == CudaSearchEvaluatorKind::FinishTime;
    bool reported = j.reported != 0u;
    if (finishTime && !reported && !completed) {
        ++j.evaluationIndex;
        if (c.incumbentPrecise && static_cast<double>(publicTime) >= c.incumbentTimeMs) return Invalid(j);
        return kJudgeContinue;
    }
    DeviceSample sample = EvaluateState(e, state, state.body.temporary.position, state.body.temporary.linearSpeed,
                                        static_cast<double>(publicTime - c.tickDurationMs), static_cast<double>(publicTime),
                                        0u, &reported);
    j.reported = reported ? 1u : 0u;
    if (finishTime && reported && sample.valid) {
        const std::uint64_t prestartNs = static_cast<std::uint64_t>(c.prestartDurationMs) * 1000000u;
        const double coarseLowerMs = sample.timeMs - static_cast<double>(c.prestartDurationMs) - static_cast<double>(c.tickDurationMs);
        if (c.incumbentPrecise && coarseLowerMs >= c.incumbentTimeMs) return Invalid(j);
        const std::uint64_t lower = finishLowerNs, upper = finishLowerNs + 1u;
        if (!finished || lower < prestartNs || upper < prestartNs) return kJudgeUnsupported;
        if (upper >= c.incumbentUpperNs) return Invalid(j);
        sample.score = static_cast<double>(upper - prestartNs);
        sample.timeMs = sample.score / 1000000.0;
        sample.preciseFinish = true;
    }
    if (sample.valid && (!j.valid || (c.maximize != 0u ? sample.score > j.score : sample.score < j.score))) {
        SetBest(j, sample, j.evaluationIndex);
    }
    ++j.evaluationIndex;
    if (reported && (kind == CudaSearchEvaluatorKind::VolumeEntry || kind == CudaSearchEvaluatorKind::CustomVolumeEntry ||
                     kind == CudaSearchEvaluatorKind::ConditionTimeEarliest ||
                     kind == CudaSearchEvaluatorKind::ConditionTimeLatest)) {
        return kJudgeDone;
    }
    if (finishTime) {
        if (reported) return kJudgeDone;
        if (c.incumbentPrecise && static_cast<double>(publicTime) >= c.incumbentTimeMs) return Invalid(j);
    }
    return kJudgeContinue;
}

}  // namespace tw::gpu::fvjudge

#endif
