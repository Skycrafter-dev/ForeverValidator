// Twin GPU engine: the search session's engine (tw_gpu_session.h SearchEngine). One candidate per thread: its start state,
// its controls, after every tick the session's judge (twin_search_executor.cpp TickJudge::OnTick, FV's CUDA
// EvaluateState for the kinds the twin session supports) with its early stop.
//
// The judge's arithmetic is FV's EvaluateState as written (double, compiled without FMA like the host's), so the
// device computes the host's bits; Pose's atan2 is the exception (CUDA's atan2 is not the host libm's), so for Pose the
// device keeps every tick whose device score lies within kPoseMargin of its device best (a superset of the ticks that
// can be the host's best: both atan2 are within a few ulp of the true value) and the host re-judges them exactly.
//
// The tick is dev::RunTick (tw_gpu_tick.cuh); the kernel is the replaceable piece (a lockstep runner with a per-tick
// hook can take its place).
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "tw_gpu.h"
#include "tw_gpu_packs.cuh"
#include "tw_gpu_session.h"
#include "tw_gpu_util.cuh"
// tw_gpu_tick.cuh defines RunSubstep out of line (__noinline__ without inline): a second translation unit that
// includes it collides with tw_gpu_queue.cu's at the device link, so this one's copies get their own names.
#define RunSubstep RunSubstepSession
#define RunSubstepK RunSubstepKSession
#define RunTick RunTickSession
#include "tw_gpu_tick.cuh"
#include "tw_gpu_lockstep.cuh"
#undef RunTick
#undef RunSubstepK
#undef RunSubstep

namespace tw::gpu {

namespace {

using namespace dev;

// The session's car constants in constant memory (TW_GPU_CONST_CAR, M round 6; SearchEngine::Init uploads them once per
// device): the lane runner's forces and response read them as constant operands and uniform constant loads instead of
// generic loads through DevPacks::k. One car per device at a time (ConstCarSlots), so no launch ever reads another
// engine's constants (the queue runner keeps DevPacks::k).
#ifndef TW_GPU_CONST_CAR
#define TW_GPU_CONST_CAR 1
#endif
__constant__ unsigned char gCarK[sizeof(model::CarForceConsts)] __attribute__((aligned(64)));
// Per device: the car constants' bytes in gCarK and the live engines reading them (SearchEngine::Init).
struct ConstCarSlots {
    struct Slot {
        std::vector<unsigned char> bytes;
        int live = 0;
    };
    std::mutex lock;
    std::map<int, Slot> byDevice;
};
ConstCarSlots &CarSlots() {
    static ConstCarSlots slots;
    return slots;
}
// An engine's hold on its device's car constants: Dev's first member, so it is released after the job sets' streams are done.
struct ConstCarHold {
    int device = -1;
    ConstCarHold() = default;
    ConstCarHold(const ConstCarHold &) = delete;
    ConstCarHold &operator=(const ConstCarHold &) = delete;
    ~ConstCarHold() {
        if (device < 0) return;
        ConstCarSlots &slots = CarSlots();
        const std::lock_guard<std::mutex> hold(slots.lock);
        --slots.byDevice[device].live;
    }
};

// The view the judge reads after a tick (TwinPhysics::ReadTick's fields).
struct View {
    const float *pos, *vel, *quat;
    bool raceCompleted;
    std::uint32_t tickMs;
    bool finished;
    std::uint64_t finishLowerNs;
};

struct Sample {
    double score = 0.0, timeMs = 0.0, detail0 = 0.0, detail1 = 0.0;
    bool valid = false;
};

// ---- FV's CUDA EvaluateState (cuda_search_executor.cu), the kinds the twin session supports ----
__device__ bool ContainsVolume(const JudgeConfig &e, const float p[3]) {
    return static_cast<double>(p[0]) >= e.values[0] && static_cast<double>(p[1]) >= e.values[1] &&
           static_cast<double>(p[2]) >= e.values[2] && static_cast<double>(p[0]) <= e.values[3] &&
           static_cast<double>(p[1]) <= e.values[4] && static_cast<double>(p[2]) <= e.values[5];
}

__device__ bool SegmentEntry(const JudgeConfig &e, const float from[3], const float to[3], double *fraction) {
    const double fromValues[3]{from[0], from[1], from[2]};
    const double toValues[3]{to[0], to[1], to[2]};
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const double sweptMinimum = fromValues[axis] < toValues[axis] ? fromValues[axis] : toValues[axis];
        const double sweptMaximum = fromValues[axis] > toValues[axis] ? fromValues[axis] : toValues[axis];
        if (sweptMaximum < e.values[axis] || sweptMinimum > e.values[axis + 3u]) return false;
    }
    double enter = 0.0;
    double leave = 1.0;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const double delta = toValues[axis] - fromValues[axis];
        if (fabs(delta) <= 1e-12) {
            if (fromValues[axis] < e.values[axis] || fromValues[axis] > e.values[axis + 3u]) return false;
            continue;
        }
        double near = (e.values[axis] - fromValues[axis]) / delta;
        double far = (e.values[axis + 3u] - fromValues[axis]) / delta;
        if (near > far) {
            const double swap = near;
            near = far;
            far = swap;
        }
        enter = enter > near ? enter : near;
        leave = leave < far ? leave : far;
        if (enter > leave) return false;
    }
    if (enter < 0.0 || enter > 1.0) return false;
    *fraction = enter;
    return true;
}

// Pose's weighted position error and rotation error (FV's: conjugate(target) * car as a rotation vector in the
// target's axes, weighted), the rotation with the device's atan2 (the host re-judges the near-best ticks exactly).
__device__ void PoseTerms(const JudgeConfig &e, const float pos[3], const float q[4], double *positionError, double *rotationError) {
    const double x = static_cast<double>(pos[0]) - e.values[0];
    const double y = static_cast<double>(pos[1]) - e.values[1];
    const double z = static_cast<double>(pos[2]) - e.values[2];
    *positionError = sqrt((e.values[8] * x * x + e.values[9] * y * y) + e.values[10] * z * z);
    const double tx = -e.values[3], ty = -e.values[4], tz = -e.values[5], tw = e.values[6];
    double rx = tw * q[0] + tx * q[3] + ty * q[2] - tz * q[1];
    double ry = tw * q[1] - tx * q[2] + ty * q[3] + tz * q[0];
    double rz = tw * q[2] + tx * q[1] - ty * q[0] + tz * q[3];
    double rw = tw * q[3] - tx * q[0] - ty * q[1] - tz * q[2];
    if (rw < 0.0) {
        rx = -rx;
        ry = -ry;
        rz = -rz;
        rw = -rw;
    }
    const double sine = sqrt((rx * rx + ry * ry) + rz * rz);
    double rotation = 0.0;
    if (sine > 1e-12) {
        const double scale = 2.0 * atan2(sine, rw) / sine;
        rx *= scale;
        ry *= scale;
        rz *= scale;
        rotation = sqrt((e.values[12] * rx * rx + e.values[11] * ry * ry) + e.values[13] * rz * rz);
    }
    *rotationError = rotation;
}

__device__ Sample EvaluateState(const JudgeConfig &e, const View &v, const float previous[3], double previousTimeMs,
                                double currentTimeMs, bool *reported) {
    Sample result;
    result.timeMs = currentTimeMs;
    switch (static_cast<JudgeKind>(e.kind)) {
    case JudgeKind::Velocity: {
        const double x = v.vel[0];
        const double y = v.vel[1];
        const double z = v.vel[2];
        const double speed = sqrt((x * x + y * y) + z * z);
        double ax = e.values[0], ay = e.values[1], az = e.values[2];
        if ((e.optionFlags & 4u) != 0u) {   // CudaSearchVelocityTowardsPoint: the unit direction to the point
            ax = e.values[4] - static_cast<double>(v.pos[0]);
            ay = e.values[5] - static_cast<double>(v.pos[1]);
            az = e.values[6] - static_cast<double>(v.pos[2]);
            const double length = sqrt((ax * ax + ay * ay) + az * az);
            if (length <= 1e-12) {
                ax = 0.0;
                ay = 0.0;
                az = 0.0;
            } else {
                ax /= length;
                ay /= length;
                az /= length;
            }
        }
        double alignment = 1.0;
        if ((e.optionFlags & 3u) != 0u) {
            alignment = speed <= 1e-12 ? 0.0 : (x * ax + y * ay + z * az) / speed;
            if (alignment < e.values[3]) return result;
        }
        result.score = (e.optionFlags & 1u) != 0u ? x * ax + y * ay + z * az : speed;
        result.detail0 = speed;
        result.detail1 = alignment;
        result.valid = true;
        break;
    }
    case JudgeKind::Point: {
        const double x = static_cast<double>(v.pos[0]) - e.values[0];
        const double y = static_cast<double>(v.pos[1]) - e.values[1];
        const double z = static_cast<double>(v.pos[2]) - e.values[2];
        result.score = sqrt((x * x + y * y) + z * z);
        result.valid = true;
        break;
    }
    case JudgeKind::Pose: {   // device atan2: the host re-judges the near-best ticks
        double positionError = 0.0, rotationError = 0.0;
        PoseTerms(e, v.pos, v.quat, &positionError, &rotationError);
        result.score = (1.0 - e.values[7]) * positionError + e.values[7] * rotationError;
        result.detail0 = positionError;
        result.detail1 = rotationError;
        result.valid = true;
        break;
    }
    case JudgeKind::VolumeEntry:
        if (*reported || ContainsVolume(e, previous)) return result;
        {
            double fraction = 0.0;
            if (!SegmentEntry(e, previous, v.pos, &fraction)) return result;
            *reported = true;
            result.timeMs = previousTimeMs + fraction * (currentTimeMs - previousTimeMs);
            result.score = result.timeMs;
            result.valid = true;
        }
        break;
    case JudgeKind::StuntPoints:
        result.score = 0.0;   // the twin's view carries no stunt score (EvaluationView::stuntsScore = 0)
        result.valid = true;
        break;
    case JudgeKind::FinishTime:
        if (*reported || !v.raceCompleted) return result;
        *reported = true;
        result.timeMs = v.tickMs;
        result.score = result.timeMs;
        result.valid = true;
        break;
    }
    return result;
}

// Within one candidate StrictlyBetter compares scores only (the same event count).
__device__ __forceinline__ bool Better(const Sample &s, const JudgeState &best, bool maximize) {
    if (!s.valid) return false;
    if (!best.valid) return true;
    if (s.score != best.score) return maximize ? s.score > best.score : s.score < best.score;
    return false;
}

__device__ __forceinline__ void Invalidate(JudgeState &j) {
    j.score = j.timeMs = j.detail0 = j.detail1 = 0.0;
    j.evaluationTick = 0u;
    j.valid = 0u;
    j.preciseFinish = 0u;
}

// Pose's near-best ticks: a superset of the ticks whose host score can be the best (device and host atan2 differ by
// a few ulp; the margin is orders of magnitude wider).
__device__ __forceinline__ double PoseMargin(double best) { return 1e-9 * (1.0 + fabs(best)); }

struct PoseTrack {
    double best;   // device best score
    bool any;
};

// TickJudge::OnTick for timeline tick t.
__device__ std::uint8_t OnTick(const JudgeConfig &c, std::uint32_t t, const View &v, JudgeState &j, PoseTrack &pt,
                               SearchResult &out) {
    const JudgeKind kind = static_cast<JudgeKind>(c.kind);
    const bool finishTime = kind == JudgeKind::FinishTime;
    const std::int64_t publicTime = c.branchTimeMs + static_cast<std::int64_t>(t + 1u) * c.tickDurationMs;
    if (publicTime < c.evaluationStartTimeMs) return kJudgeContinue;
    bool reported = j.reported != 0u;
    if (finishTime && !reported && !v.raceCompleted) {
        ++j.evaluationIndex;
        if (c.incumbentPrecise && static_cast<double>(publicTime) >= c.incumbentTimeMs) {
            Invalidate(j);
            return kJudgeInvalid;
        }
        return kJudgeContinue;
    }
    Sample s = EvaluateState(c, v, j.previous, static_cast<double>(publicTime - c.tickDurationMs), static_cast<double>(publicTime),
                             &reported);
    j.reported = reported ? 1u : 0u;
    bool precise = false;
    if (finishTime && reported && s.valid) {
        const std::uint64_t prestartNs = static_cast<std::uint64_t>(c.prestartDurationMs) * 1000000u;
        const double coarseLowerMs = s.timeMs - static_cast<double>(c.prestartDurationMs) - static_cast<double>(c.tickDurationMs);
        if (c.incumbentPrecise && coarseLowerMs >= c.incumbentTimeMs) {
            Invalidate(j);
            return kJudgeInvalid;
        }
        const std::uint64_t lower = v.finishLowerNs, upper = v.finishLowerNs + 1u;
        if (!v.finished || lower < prestartNs || upper < prestartNs) return kJudgeUnsupported;
        if (upper >= c.incumbentUpperNs) {
            Invalidate(j);
            return kJudgeInvalid;
        }
        s.score = static_cast<double>(upper - prestartNs);
        s.timeMs = s.score / 1000000.0;
        precise = true;
    }
    if (kind == JudgeKind::Pose) {
        // keep the ticks within the margin of the device best (in tick order), drop those the new best leaves behind;
        // a NaN score (StrictlyBetter keeps a first NaN sample) goes to the CPU
        if (s.score != s.score) out.poseOverflow = 1u;
        const double margin = pt.any ? PoseMargin(pt.best) : 0.0;
        if (!pt.any || s.score < pt.best - margin) {
            pt.best = s.score;
            pt.any = true;
            const double m = PoseMargin(pt.best);
            int k = 0;
            for (int i = 0; i < out.poseCount; ++i) {
                double pe = 0.0, re = 0.0;
                PoseTerms(c, out.pose[i].pos, out.pose[i].quat, &pe, &re);
                const double si = (1.0 - c.values[7]) * pe + c.values[7] * re;
                if (si <= pt.best + m) out.pose[k++] = out.pose[i];
            }
            out.poseCount = static_cast<std::uint8_t>(k);
        } else if (s.score < pt.best) {
            pt.best = s.score;
        }
        if (s.score <= pt.best + PoseMargin(pt.best)) {
            if (out.poseCount < kPoseNear) {
                PoseNear &p = out.pose[out.poseCount++];
                p.t = t;
                p.evaluationIndex = j.evaluationIndex;
                for (int a = 0; a < 3; ++a) p.pos[a] = v.pos[a];
                for (int a = 0; a < 4; ++a) p.quat[a] = v.quat[a];
            } else {
                out.poseOverflow = 1u;
            }
        }
    }
    if (Better(s, j, c.maximize != 0u)) {
        j.score = s.score;
        j.timeMs = s.timeMs;
        j.detail0 = s.detail0;
        j.detail1 = s.detail1;
        j.evaluationTick = j.evaluationIndex;
        j.valid = 1u;
        j.preciseFinish = precise ? 1u : 0u;
    }
    ++j.evaluationIndex;
    if (reported && kind == JudgeKind::VolumeEntry) return kJudgeDone;
    if (finishTime) {
        if (reported) return kJudgeDone;
        if (c.incumbentPrecise && static_cast<double>(publicTime) >= c.incumbentTimeMs) {
            Invalidate(j);
            return kJudgeInvalid;
        }
    }
    return kJudgeContinue;
}

struct KernelStats {
    unsigned long long substeps;
    unsigned long long ticks;    // the lane runner's (SubmitJobs: the host does not see the results)
    unsigned long long clk[8];   // FOREVERVALIDATOR_TWIN_GPU_CLOCKS: RunTick's stage cycles [0..5], the judge, the whole thread
};

__global__ void SearchKernel(DevPacks P, JudgeConfig c, const SearchJob *jobs, std::uint32_t n, const model::TwinState *starts,
                             const std::uint32_t *controls, SearchResult *results, KernelStats *stats, bool clocks) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const SearchJob job = jobs[i];
    model::TwinState s = starts[job.start];
    GpuContacts contacts;
    model::TwinState finishStart;
    SearchResult r;
    r.judge = job.judge;
    r.poseCount = 0u;
    r.poseOverflow = 0u;
    PoseTrack pt{0.0, false};
    std::uint32_t substeps = 0u, lost = 0u, ran = 0u;
    unsigned long long clk[8] = {};
    const long long k0 = clocks ? clock64() : 0;
    std::uint8_t verdict = kJudgeContinue;
    bool finished = false;
    std::uint64_t finishLowerNs = 0u;
    for (std::uint32_t k = 0; k < job.ticks; ++k) {
        const bool completedBefore = s.raceCompleted != 0u;
        RunTickSession(P, s, controls[job.controls + k], contacts, finishStart, &substeps, &lost, clocks ? clk : nullptr);
        const long long j0 = clocks ? clock64() : 0;
        ++ran;
        // step::StepKernel's sticky Events::finished: the tick in which the race completed (raceCompleted only goes
        // 0 -> 1, together with the tick's race event)
        if (!finished && !completedBefore && s.raceCompleted != 0u) {
            finished = true;
            finishLowerNs = s.finishPresent != 0u ? (static_cast<std::uint64_t>(s.finishLowerNs[1]) << 32) | s.finishLowerNs[0] : 0u;
        }
        const View v{s.pos, s.vel, s.quat, s.raceCompleted != 0u, s.tickMs, finished, finishLowerNs};
        verdict = OnTick(c, job.firstTick + k, v, r.judge, pt, r);
        for (int a = 0; a < 3; ++a) r.judge.previous[a] = s.pos[a];
        if (clocks) clk[6] += static_cast<unsigned long long>(clock64() - j0);
        if (verdict != kJudgeContinue) break;
    }
    r.ticks = ran;
    r.verdict = verdict;
    r.lost = lost != 0u ? 1u : 0u;
    results[i] = r;
    atomicAdd(&stats->substeps, static_cast<unsigned long long>(substeps));
    if (clocks) {
        clk[7] = static_cast<unsigned long long>(clock64() - k0);
        for (int k = 0; k < 8; ++k) atomicAdd(&stats->clk[k], clk[k]);
    }
}

// FOREVERVALIDATOR_TWIN_GPU_TAIL (measurement): per warp its start and its exit, per job when a lane took it and when it
// ended (globaltimer, ns) and its ticks; the jobs' count. Null pointers: none.
struct TailRec {
    unsigned long long *warp;   // [2 * warps]: start, exit
    unsigned long long *job;    // [2 * jobs]: taken, done
    std::uint32_t *ticks;       // [jobs]
    std::uint32_t *count;       // the jobs' count
};
__device__ __forceinline__ unsigned long long GlobalTimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// A parked job's judge (LaneJobs::Save / Load).
struct alignas(16) LaneSaved {
    SearchResult r;
    PoseTrack pt;
    std::uint64_t finishLowerNs;
    std::uint32_t finished;
};

// The jobs on the lockstep lane runner (tw_gpu_lockstep.cuh RunLanes, FOREVERVALIDATOR_TWIN_GPU_SCHEDULE=lanes, the
// default): persistent lanes, one substep a pass, a lane takes the pool's next job when its own is over; the judge
// is the runner's per-tick hook. The pool is the jobs in the host's order (first change, then controls), so warp
// batches take neighbouring jobs.
struct LaneJobs {
    std::uint32_t count;
    const SearchJob *jobs;
    const model::TwinState *starts;
    const std::uint32_t *controls;
    SearchResult *results;
    const JudgeConfig *c;
    // the lane's current job's judge
    SearchResult r;
    PoseTrack pt;
    std::uint32_t firstTick;
    std::uint8_t verdict;
    bool finished;                 // step::StepKernel's sticky Events::finished
    std::uint64_t finishLowerNs;
    unsigned long long substeps, ticks;
    unsigned long long *tailJob;   // TailRec::job, ticks (null: none)
    std::uint32_t *tailTicks;

    __device__ const model::CarForceConsts &Car(const DevPacks &P) const {
        if constexpr (TW_GPU_CONST_CAR != 0) return *reinterpret_cast<const model::CarForceConsts *>(gCarK);
        return *P.k;
    }
    __device__ JobRef Ref(std::uint32_t j) const {
        const SearchJob &job = jobs[j];
        return JobRef{0u, starts + job.start, controls + job.controls, job.ticks};
    }
    __device__ JobRef Job(std::uint32_t j) const {
        if (tailJob != nullptr) tailJob[2u * j] = GlobalTimer();
        return Ref(j);
    }
    // parking (FOREVERVALIDATOR_TWIN_GPU_PARK): the judge between two ticks
    static constexpr int kSaveQuads = static_cast<int>(sizeof(LaneSaved) / 16u);
    __device__ std::uint32_t Ticks(std::uint32_t j) const { return jobs[j].ticks; }
    __device__ void Save(std::uint32_t, const ParkSlot &p) {
        LaneSaved v;
        v.r = r;
        v.pt = pt;
        v.finishLowerNs = finishLowerNs;
        v.finished = finished ? 1u : 0u;
        const uint4 *w = reinterpret_cast<const uint4 *>(&v);
        for (int q = 0; q < kSaveQuads; ++q) __stcg(p.at(q), w[q]);
    }
    __device__ void Load(std::uint32_t j, const ParkSlot &p) {
        LaneSaved v;
        uint4 *w = reinterpret_cast<uint4 *>(&v);
        for (int q = 0; q < kSaveQuads; ++q) w[q] = __ldcg(p.at(q));
        r = v.r;
        pt = v.pt;
        finishLowerNs = v.finishLowerNs;
        finished = v.finished != 0u;
        firstTick = jobs[j].firstTick;
        verdict = kJudgeContinue;
    }
    __device__ void Begin(std::uint32_t j) {
        const SearchJob &job = jobs[j];
        r.judge = job.judge;
        r.poseCount = 0u;
        r.poseOverflow = 0u;
        pt = PoseTrack{0.0, false};
        firstTick = job.firstTick;
        verdict = kJudgeContinue;
        finished = false;
        finishLowerNs = 0u;
    }
    __device__ bool Tick(std::uint32_t j, std::uint32_t t, model::TwinState &s, const race::TwRaceEvents &ev) {
        if (t == 0u) Begin(j);
        if (!finished && ev.raceCompleted != 0u) {   // TickContext::finished = the tick's raceEv.raceCompleted
            finished = true;
            finishLowerNs = s.finishPresent != 0u ? (static_cast<std::uint64_t>(s.finishLowerNs[1]) << 32) | s.finishLowerNs[0] : 0u;
        }
        const View v{s.pos, s.vel, s.quat, s.raceCompleted != 0u, s.tickMs, finished, finishLowerNs};
        verdict = OnTick(*c, firstTick + t, v, r.judge, pt, r);
        for (int a = 0; a < 3; ++a) r.judge.previous[a] = s.pos[a];
        return verdict == kJudgeContinue;
    }
    __device__ void Done(std::uint32_t j, std::uint32_t ticks, const LaneCounts &counts) {
        if (ticks == 0u) Begin(j);
        r.ticks = ticks;
        r.verdict = verdict;
        r.lost = counts.lost != 0u ? 1u : 0u;
        results[j] = r;
        substeps += counts.substeps;
        this->ticks += ticks;
        if (tailJob != nullptr) {
            tailJob[2u * j + 1u] = GlobalTimer();
            tailTicks[j] = ticks;
        }
    }
};

// packs: the world's packs as a __grid_constant__ parameter (M round 6: the detection's DevWorld fields, the car shapes and
// the world's pointers, are constant-bank operands instead of generic loads through a pointer)
__global__ void SearchLanesKernel(const __grid_constant__ DevPacks packs, const __grid_constant__ JudgeConfig c, const SearchJob *jobs, std::uint32_t n,
                                  const model::TwinState *starts, const std::uint32_t *controls, SearchResult *results,
                                  KernelStats *stats, LaneRows rows, unsigned *next, std::uint32_t chunk, const std::uint32_t *count,
                                  TailRec tail) {
    LaneJobs J{count != nullptr ? *count : n, jobs, starts, controls, results, &c};
    J.substeps = 0u;
    J.ticks = 0u;
    J.tailJob = tail.job;
    J.tailTicks = tail.ticks;
    const std::uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (tail.warp != nullptr && (threadIdx.x & 31u) == 0u) tail.warp[2u * warp] = GlobalTimer();
    if (tail.count != nullptr && blockIdx.x == 0u && threadIdx.x == 0u) *tail.count = J.count;
    RunLanes<LaneJobs, false, false, OnePack>(OnePack{&packs}, J, rows, next, chunk, nullptr);
    if (tail.warp != nullptr && (threadIdx.x & 31u) == 0u) tail.warp[2u * warp + 1u] = GlobalTimer();
    atomicAdd(&stats->substeps, J.substeps);
    atomicAdd(&stats->ticks, J.ticks);
}

// A device buffer that keeps its capacity.
struct Grow {
    DevBuffer b;
    bool Fit(std::size_t n, std::string *error) { return n <= b.bytes || b.Alloc(std::max(n, b.bytes + b.bytes / 2u), error); }
};

// Pinned host memory that keeps its capacity.
struct Pinned {
    void *p = nullptr;
    std::size_t bytes = 0;
    Pinned() = default;
    Pinned(const Pinned &) = delete;
    Pinned &operator=(const Pinned &) = delete;
    ~Pinned() {
        if (p != nullptr) cudaFreeHost(p);
    }
    bool Fit(std::size_t n, std::string *error) {
        if (n <= bytes) return true;
        if (p != nullptr) cudaFreeHost(p);
        p = nullptr;
        bytes = 0;
        const std::size_t want = std::max(n, bytes + bytes / 2u);
        if (!Check(cudaMallocHost(&p, want), "cudaMallocHost", error)) return p = nullptr, false;
        bytes = want;
        return true;
    }
};

// One job set: host staging, device buffers, its stream.
struct JobSet {
    Pinned hJobs, hControls, hResults;
    Grow dJobs, dControls, dResults;
    DevBuffer stats;
    Pinned hStats;   // KernelStats (pinned: a copy to pageable memory would make Submit wait for the kernel)
    LaneSpace space; // the lane runner's lanes, rows and pool counter
    cudaStream_t stream = nullptr;
    cudaStream_t post = nullptr;   // PostStream (null: the set's stream)
    cudaEvent_t e0 = nullptr, e1 = nullptr, up = nullptr;
    cudaEvent_t eP = nullptr;   // MarkPost: the work after the lanes queued (a batch that follows this one waits for it, else e1)
    bool marked = false;
    std::size_t n = 0u;
    std::size_t prepared = 0u;   // ReserveJobs: the lanes are sized for this many jobs
    DevBuffer tail;              // FOREVERVALIDATOR_TWIN_GPU_TAIL: TailRec's arrays for `tailJobs` jobs and the lanes' warps
    std::size_t tailJobs = 0u;
    std::chrono::steady_clock::time_point submitted;
    ~JobSet() {
        if (stream != nullptr) cudaStreamSynchronize(stream);   // a set still in flight (a batch run ahead)
        if (post != nullptr) cudaStreamSynchronize(post);
        if (e0 != nullptr) cudaEventDestroy(e0);
        if (e1 != nullptr) cudaEventDestroy(e1);
        if (up != nullptr) cudaEventDestroy(up);
        if (eP != nullptr) cudaEventDestroy(eP);
        if (stream != nullptr) cudaStreamDestroy(stream);
        if (post != nullptr) cudaStreamDestroy(post);
    }
};

}  // namespace

struct SearchEngine::Dev {
    ConstCarHold carHold;   // first: released last, after the sets' streams (TW_GPU_CONST_CAR)
    CarPacks car;
    CollisionPacks collision;
    DevBuffer starts;
    DevBuffer packs;   // the world's DevPacks (the lane runner reads them through a pointer)
    JobSet sets[kMaxSets];
    int used = 2;                          // SetSets
    unsigned long long tailLastEnd = 0u;   // FOREVERVALIDATOR_TWIN_GPU_TAIL: the last batch's kernel end (globaltimer)
};

SearchEngine::SearchEngine() = default;
SearchEngine::~SearchEngine() {
    if (dev_ && std::getenv("TW_GPU_DETECT_STATS") != nullptr) {   // the session's detections, per warp call
        unsigned long long a[CollisionPacks::kDetectStats];
        std::string error;
        if (dev_->collision.DetectStats(a, &error)) CollisionPacks::PrintDetectStats(a);
    }
}

bool SearchEngine::Init(const model::CarForceConsts &car, const model::WaterZone *water, const race::TwRaceMap &race,
                        const world::WorldPackView &view, const collide::CarShapes &shapes, const model::TwinState *starts,
                        std::size_t startCount, std::string *error) {
    auto d = std::make_unique<Dev>();
    if (!d->car.Upload(car, water, race, error) || !d->collision.Upload(view, shapes, error) ||
        !d->starts.Put(starts, sizeof(model::TwinState) * startCount, error))
        return false;
    {
        const CarPacks::Dev *car = d->car.dev();
        const CollisionPacks::Dev *col = d->collision.dev();
        const DevPacks packs{car->k, car->w, car->r, col->w};
        if (!d->packs.Put(&packs, sizeof packs, error)) return false;
    }
    if (TW_GPU_CONST_CAR != 0) {   // the lane runner's car constants in constant memory (one car per device at a time): the
        // bytes are written only while no live engine on the device reads them (a session after another map's: uploaded again),
        // so no launch ever reads another engine's constants; two live engines with different cars on one device: refused
        ConstCarSlots &slots = CarSlots();
        const std::lock_guard<std::mutex> hold(slots.lock);
        int device = 0;
        if (!Check(cudaGetDevice(&device), "cudaGetDevice", error)) return false;
        ConstCarSlots::Slot &slot = slots.byDevice[device];
        const unsigned char *b = reinterpret_cast<const unsigned char *>(&car);
        if (slot.bytes.size() != sizeof car || !std::equal(slot.bytes.begin(), slot.bytes.end(), b)) {
            if (slot.live != 0) return *error = "SearchEngine: a live engine on this device holds another car's constants (TW_GPU_CONST_CAR)", false;
            if (!Check(cudaMemcpyToSymbol(gCarK, &car, sizeof car), "car constants", error)) return false;
            slot.bytes.assign(b, b + sizeof car);
        }
        ++slot.live;
        d->carHold.device = device;
    }
    // FOREVERVALIDATOR_TWIN_SEARCH_GPU_POST=1: a set's outcomes on a stream of the greatest priority (PostStream)
    const char *postEnv = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_POST");
    const bool post = postEnv != nullptr && std::strcmp(postEnv, "1") == 0;
    int least = 0, greatest = 0;
    if (!Check(cudaDeviceGetStreamPriorityRange(&least, &greatest), "cudaDeviceGetStreamPriorityRange", error)) return false;
    for (JobSet &js : d->sets) {
        if (!js.stats.Alloc(sizeof(KernelStats), error) || !js.hStats.Fit(sizeof(KernelStats), error) ||
            !Check(cudaStreamCreateWithFlags(&js.stream, cudaStreamNonBlocking), "cudaStreamCreate", error) ||
            (post && !Check(cudaStreamCreateWithPriority(&js.post, cudaStreamNonBlocking, greatest), "cudaStreamCreate", error)) ||
            !Check(cudaEventCreate(&js.e0), "cudaEventCreate", error) || !Check(cudaEventCreate(&js.e1), "cudaEventCreate", error) ||
            !Check(cudaEventCreate(&js.up), "cudaEventCreate", error) ||
            !Check(cudaEventCreateWithFlags(&js.eP, cudaEventDisableTiming), "cudaEventCreate", error))
            return false;
    }
    dev_ = std::move(d);
    // one job of no tick on each set: the module's load and the kernel's local memory come with the engine, not with
    // the first batch
    for (int set = 0; set < kMaxSets; ++set) {
        SearchJob *job = Jobs(set, 1u, error);
        if (job == nullptr || Controls(set, 1u, error) == nullptr) return false;
        *job = SearchJob{};
        SearchStats st;
        if (!Submit(set, JudgeConfig{}, 1u, 0u, error) || Collect(set, &st, error) == nullptr) return false;
    }
    return true;
}

std::size_t SearchEngine::WorldBytes() const { return dev_ ? dev_->collision.Bytes() : 0u; }

namespace {
// FOREVERVALIDATOR_TWIN_GPU_SCHEDULE: lanes (default, the lockstep lane runner) | tick (one job a thread, RunTick)
bool LaneSchedule() {
    static const bool lanes = [] {
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_GPU_SCHEDULE");
        return v == nullptr || std::strcmp(v, "tick") != 0;
    }();
    return lanes;
}
constexpr std::uint32_t kSpareBlocks = 8u;   // SubmitJobs' lane kernel: blocks left free (see there)
bool TailOn() {   // FOREVERVALIDATOR_TWIN_GPU_TAIL: per batch, the lane kernel's warps and jobs over time (measurement)
    static const bool on = std::getenv("FOREVERVALIDATOR_TWIN_GPU_TAIL") != nullptr;
    return on;
}
// The set's tail record for its lanes and up to `jobs` jobs (allocated when it grows: allocation synchronizes the device).
bool FitTail(JobSet &js, std::size_t jobs, std::string *error) {
    if (!TailOn() || (js.tailJobs >= jobs && js.tail.bytes != 0u)) return true;
    const std::size_t warps = js.space.lanes / 32u;
    if (!js.tail.Alloc(sizeof(unsigned long long) * (2u * warps + 2u * jobs) + sizeof(std::uint32_t) * (jobs + 1u), error)) return false;
    js.tailJobs = jobs;
    return true;
}
TailRec Tail(const JobSet &js) {
    if (!TailOn() || js.tail.bytes == 0u) return TailRec{nullptr, nullptr, nullptr, nullptr};
    const std::size_t warps = js.space.lanes / 32u;
    unsigned long long *w = static_cast<unsigned long long *>(js.tail.ptr);
    std::uint32_t *t = reinterpret_cast<std::uint32_t *>(w + 2u * warps + 2u * js.tailJobs);
    return TailRec{w, w + 2u * warps, t, t + js.tailJobs};
}
// The batch's lane kernel over time (after the set's stream is done).
void PrintTail(const JobSet &js, unsigned long long *lastEnd) {
    const TailRec r = Tail(js);
    if (r.warp == nullptr) return;
    const std::size_t warps = js.space.lanes / 32u;
    std::vector<unsigned long long> w(2u * warps), j(2u * js.tailJobs);
    std::vector<std::uint32_t> ticks(js.tailJobs + 1u);
    if (cudaMemcpy(w.data(), r.warp, sizeof(unsigned long long) * w.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(j.data(), r.job, sizeof(unsigned long long) * j.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(ticks.data(), r.ticks, sizeof(std::uint32_t) * ticks.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        return;
    const std::size_t count = std::min<std::size_t>(ticks[js.tailJobs], js.tailJobs);
    unsigned long long t0 = ~0ull, t1 = 0u, firstExit = ~0ull, lastTake = 0u;
    double warpNs = 0.0, laneNs = 0.0, tickSum = 0.0, jobNsMax = 0.0;
    std::uint32_t tickMax = 0u;
    std::size_t ran = 0u;   // warps that ran (the spare blocks' do not)
    for (std::size_t k = 0; k < warps; ++k) {
        if (w[2u * k] == 0u) continue;
        ++ran;
        t0 = std::min(t0, w[2u * k]);
        t1 = std::max(t1, w[2u * k + 1u]);
        firstExit = std::min(firstExit, w[2u * k + 1u]);
        warpNs += static_cast<double>(w[2u * k + 1u] - w[2u * k]);
    }
    for (std::size_t k = 0; k < count; ++k) {
        lastTake = std::max(lastTake, j[2u * k]);
        const double d = static_cast<double>(j[2u * k + 1u] - j[2u * k]);
        laneNs += d;
        jobNsMax = std::max(jobNsMax, d);
        tickSum += ticks[k];
        tickMax = std::max(tickMax, ticks[k]);
    }
    const double span = static_cast<double>(t1 - t0);
    unsigned parked = 0u;   // parking: the groups set aside (the rings' pushes)
    if (js.space.Rows().quantum != 0u) {
        unsigned c[lockstep::kRingCounters + 2 * lockstep::kBuckets] = {};
        if (cudaMemcpy(c, js.space.Next(), sizeof c, cudaMemcpyDeviceToHost) == cudaSuccess)
            for (int b = 0; b < lockstep::kBuckets; ++b) parked += c[lockstep::kRingCounters + 2 * b + 1];
    }
    char alive[160];
    int at = 0;
    for (int f = 1; f <= 9; ++f) {   // warps still running at f/10 of the span
        const unsigned long long t = t0 + static_cast<unsigned long long>(span * f / 10.0);
        std::size_t n = 0u;
        for (std::size_t k = 0; k < warps; ++k) n += w[2u * k] != 0u && w[2u * k + 1u] > t ? 1u : 0u;
        at += std::snprintf(alive + at, sizeof alive - at, "%s%.0f", f == 1 ? "" : " ", 100.0 * n / ran);
    }
    if (const char *path = std::getenv("FOREVERVALIDATOR_TWIN_GPU_TAIL_DUMP")) {   // per job: pool index, taken, done (ns from
        // the kernel's first warp start), ticks; a header line per batch with the lanes and the gap after the last kernel
        if (FILE *f = std::fopen(path, "a")) {
            std::fprintf(f, "batch %zu %u %lld\n", count, static_cast<unsigned>(ran * 32u),
                         *lastEnd != 0u ? static_cast<long long>(t0) - static_cast<long long>(*lastEnd) : 0ll);
            for (std::size_t k = 0; k < count; ++k)
                std::fprintf(f, "%zu %lld %lld %u\n", k, static_cast<long long>(j[2u * k]) - static_cast<long long>(t0),
                             static_cast<long long>(j[2u * k + 1u]) - static_cast<long long>(t0), ticks[k]);
            std::fclose(f);
        }
    }
    std::fprintf(stderr,
                 "twin gpu tail: %zu jobs on %u lanes (%.2f a lane), span %.2f ms (starts %+.2f ms after the last kernel's end), "
                 "warps used %.1f%%, lanes used %.1f%%, last job taken at %.1f%%, first warp exit at %.1f%%, warps running at 10..90%%: "
                 "%s; ticks a job mean %.1f max %u, job time mean %.2f max %.2f ms; lane-ms %.0f from %.3f to %.3f ms; %u groups parked\n",
                 count, static_cast<unsigned>(ran * 32u), static_cast<double>(count) / (ran * 32u), span / 1e6,
                 *lastEnd != 0u ? (static_cast<double>(t0) - static_cast<double>(*lastEnd)) / 1e6 : 0.0, 100.0 * warpNs / (ran * span),
                 100.0 * laneNs / (ran * 32u * span), 100.0 * static_cast<double>(lastTake - t0) / span,
                 100.0 * static_cast<double>(firstExit - t0) / span, alive, count ? tickSum / count : 0.0, tickMax,
                 count ? laneNs / count / 1e6 : 0.0, jobNsMax / 1e6, laneNs / 1e6, static_cast<double>(t0 % 1000000000000ull) / 1e6,
                 static_cast<double>(t1 % 1000000000000ull) / 1e6, parked);
    *lastEnd = t1;
}
// FOREVERVALIDATOR_TWIN_GPU_PARK: the lane runner's parking quantum in ticks (default 16; 0: none): a warp's group of jobs is
// set aside at a tick's start when waiting work (the pool's next jobs, a parked group) has more ticks left, so a batch's jobs
// end together instead of in a last round on a few lanes (round 8, E)
constexpr std::uint32_t kParkQuantum = 16u;
std::uint32_t ParkQuantum() {
    static const std::uint32_t q = [] {
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_GPU_PARK");
        return v != nullptr ? static_cast<std::uint32_t>(std::atoi(v)) : kParkQuantum;
    }();
    return q;
}
std::uint32_t ParkMargin() {   // FOREVERVALIDATOR_TWIN_GPU_PARK_MARGIN: ticks (default half the quantum)
    static const std::uint32_t m = [] {
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_GPU_PARK_MARGIN");
        return v != nullptr ? static_cast<std::uint32_t>(std::atoi(v)) : ParkQuantum() / 2u;
    }();
    return m;
}
std::uint32_t ParkShift() {   // FOREVERVALIDATOR_TWIN_GPU_PARK_SHIFT: the margin is at least left >> shift (default 2: a quarter)
    static const std::uint32_t m = [] {
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_GPU_PARK_SHIFT");
        return v != nullptr ? static_cast<std::uint32_t>(std::atoi(v)) : 2u;
    }();
    return m;
}
std::uint32_t ParkRounds() {   // FOREVERVALIDATOR_TWIN_GPU_PARK_ROUNDS: groups park for the pool's jobs in its last rounds only (2)
    static const std::uint32_t m = [] {
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_GPU_PARK_ROUNDS");
        return v != nullptr ? static_cast<std::uint32_t>(std::atoi(v)) : 2u;
    }();
    return m;
}
std::uint32_t ParkMaxRounds() {   // FOREVERVALIDATOR_TWIN_GPU_PARK_JOBS: parking in a batch of at most this many lanes' worth of jobs (3)
    static const std::uint32_t m = [] {
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_GPU_PARK_JOBS");
        return v != nullptr ? static_cast<std::uint32_t>(std::atoi(v)) : 3u;
    }();
    return m;
}
std::uint32_t BlockThreads() {   // FOREVERVALIDATOR_TWIN_GPU_BLOCK: threads per block (measurement)
    static const std::uint32_t threads = [] {
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_GPU_BLOCK");
        const long b = v != nullptr ? std::strtol(v, nullptr, 10) : 0;
        return b >= 32 && b <= 1024 ? static_cast<std::uint32_t>(b) : 64u;
    }();
    return threads;
}
}  // namespace

int SearchEngine::Sets() const { return dev_ ? dev_->used : 2; }

bool SearchEngine::SetSets(int sets, std::string *error) {
    if (!dev_ || sets < 2 || sets > kMaxSets) return *error = "SearchEngine: job sets out of range", false;
    dev_->used = sets;
    return true;
}

std::size_t SearchEngine::SetBytes() const {
    if (!dev_) return 0u;
    const JobSet &js = dev_->sets[0];
    return js.space.tickStart.bytes + js.space.end.bytes + js.space.finishStart.bytes + js.space.park.bytes + js.space.next.bytes +
           js.dResults.b.bytes + js.dJobs.b.bytes +
           js.dControls.b.bytes + js.tail.bytes;
}

bool SearchEngine::Reserve(std::size_t jobs, std::size_t controls, std::string *error) {
    jobs = std::max<std::size_t>(jobs, 1u);
    controls = std::max<std::size_t>(controls, 1u);
    for (int set = 0; set < dev_->used; ++set) {
        JobSet &js = dev_->sets[set];
        // the lanes only grow (as ReserveJobs): a smaller batch (the session's baseline, one slot) does not shrink them, so the
        // next full batch does not allocate them again (allocation synchronizes the device: 6 frees and allocations of
        // 0.2 ms before the first batch's generation)
        if (LaneSchedule() && jobs > js.prepared) {
            if (!js.space.Prepare(SearchLanesKernel, BlockThreads(), jobs, 0u, error) ||
                !js.space.PreparePark(jobs, LaneJobs::kSaveQuads, ParkQuantum(), ParkMargin(), ParkShift(), ParkRounds(), ParkMaxRounds(), error))
                return false;
            js.prepared = jobs;
        }
        if (!js.hJobs.Fit(sizeof(SearchJob) * jobs, error) || !js.hControls.Fit(sizeof(std::uint32_t) * controls, error) ||
            !js.hResults.Fit(sizeof(SearchResult) * jobs, error) || !js.dJobs.Fit(sizeof(SearchJob) * jobs, error) ||
            !js.dControls.Fit(sizeof(std::uint32_t) * controls, error) || !js.dResults.Fit(sizeof(SearchResult) * jobs, error))
            return false;
    }
    return true;
}

SearchJob *SearchEngine::Jobs(int set, std::size_t n, std::string *error) {
    JobSet &js = dev_->sets[set];
    return js.hJobs.Fit(sizeof(SearchJob) * std::max<std::size_t>(n, 1u), error) ? static_cast<SearchJob *>(js.hJobs.p) : nullptr;
}

std::uint32_t *SearchEngine::Controls(int set, std::size_t count, std::string *error) {
    JobSet &js = dev_->sets[set];
    return js.hControls.Fit(sizeof(std::uint32_t) * std::max<std::size_t>(count, 1u), error) ? static_cast<std::uint32_t *>(js.hControls.p)
                                                                                              : nullptr;
}

bool SearchEngine::Submit(int set, const JudgeConfig &config, std::size_t n, std::size_t controlCount, std::string *error, int after) {
    if (!dev_) return *error = "SearchEngine: not initialized", false;
    Dev &d = *dev_;
    JobSet &js = d.sets[set];
    js.n = n;
    js.submitted = std::chrono::steady_clock::now();
    if (n == 0u) return true;
    if (!js.dJobs.Fit(sizeof(SearchJob) * n, error) || !js.dControls.Fit(sizeof(std::uint32_t) * std::max<std::size_t>(controlCount, 1u), error) ||
        !js.dResults.Fit(sizeof(SearchResult) * n, error) || !js.hResults.Fit(sizeof(SearchResult) * n, error))
        return false;
    const CarPacks::Dev *car = d.car.dev();
    const CollisionPacks::Dev *col = d.collision.dev();
    const DevPacks packs{car->k, car->w, car->r, col->w};
    if (!Check(cudaMemcpyAsync(js.dJobs.b.ptr, js.hJobs.p, sizeof(SearchJob) * n, cudaMemcpyHostToDevice, js.stream), "jobs H2D", error) ||
        (controlCount != 0u && !Check(cudaMemcpyAsync(js.dControls.b.ptr, js.hControls.p, sizeof(std::uint32_t) * controlCount,
                                                      cudaMemcpyHostToDevice, js.stream),
                                      "controls H2D", error)) ||
        !Check(cudaMemsetAsync(js.stats.ptr, 0, sizeof(KernelStats), js.stream), "stats", error))
        return false;
    const std::uint32_t threads = BlockThreads();
    static const bool clocks = std::getenv("FOREVERVALIDATOR_TWIN_GPU_CLOCKS") != nullptr;
    const std::uint32_t blocks = static_cast<std::uint32_t>((n + threads - 1u) / threads);
    if (after >= 0 && after != set && !Check(cudaStreamWaitEvent(js.stream, d.sets[after].e1, 0), "cudaStreamWaitEvent", error)) return false;
    if (LaneSchedule()) {
        if (js.space.lanes == 0u && (!js.space.Prepare(SearchLanesKernel, threads, n, 0u, error) ||
                                     !js.space.PreparePark(n, LaneJobs::kSaveQuads, ParkQuantum(), ParkMargin(), ParkShift(), ParkRounds(), ParkMaxRounds(), error)))
            return false;
        if (!js.space.ResetPool(js.stream, error)) return false;
        cudaEventRecord(js.e0, js.stream);
        SearchLanesKernel<<<js.space.blocks, js.space.threads, 0, js.stream>>>(
                packs /* M: by value (__grid_constant__) */, config, static_cast<const SearchJob *>(js.dJobs.b.ptr),
                static_cast<std::uint32_t>(n), static_cast<const model::TwinState *>(d.starts.ptr),
                static_cast<const std::uint32_t *>(js.dControls.b.ptr), static_cast<SearchResult *>(js.dResults.b.ptr),
                static_cast<KernelStats *>(js.stats.ptr), js.space.Rows(), js.space.Next(), 0u, nullptr,
                TailRec{nullptr, nullptr, nullptr, nullptr});
    } else {
    cudaEventRecord(js.e0, js.stream);
    SearchKernel<<<blocks, threads, 0, js.stream>>>(packs, config, static_cast<const SearchJob *>(js.dJobs.b.ptr),
                                                    static_cast<std::uint32_t>(n), static_cast<const model::TwinState *>(d.starts.ptr),
                                                    static_cast<const std::uint32_t *>(js.dControls.b.ptr),
                                                    static_cast<SearchResult *>(js.dResults.b.ptr), static_cast<KernelStats *>(js.stats.ptr),
                                                    clocks);
    }
    cudaEventRecord(js.e1, js.stream);
    if (!Check(cudaGetLastError(), "SearchKernel launch", error) ||
        !Check(cudaMemcpyAsync(js.hResults.p, js.dResults.b.ptr, sizeof(SearchResult) * n, cudaMemcpyDeviceToHost, js.stream),
               "results D2H", error) ||
        !Check(cudaMemcpyAsync(js.hStats.p, js.stats.ptr, sizeof(KernelStats), cudaMemcpyDeviceToHost, js.stream), "stats D2H", error))
        return false;
    return true;
}

const SearchResult *SearchEngine::Collect(int set, SearchStats *stats, std::string *error) {
    *stats = SearchStats{};
    JobSet &js = dev_->sets[set];
    if (js.n == 0u) return nullptr;
    if (!Check(cudaStreamSynchronize(js.stream), "SearchKernel", error)) return nullptr;
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, js.e0, js.e1);
    stats->kernelMs = ms;
    stats->wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - js.submitted).count();
    const KernelStats &ks = *static_cast<const KernelStats *>(js.hStats.p);
    stats->substeps = ks.substeps;
    for (int k = 0; k < 8; ++k) stats->clk[k] = ks.clk[k];
    const SearchResult *out = static_cast<const SearchResult *>(js.hResults.p);
    for (std::size_t i = 0; i < js.n; ++i) stats->ticks += out[i].ticks;
    return out;
}

bool SearchEngine::ReserveJobs(std::size_t jobs, std::string *error) {
    jobs = std::max<std::size_t>(jobs, 1u);
    for (int set = 0; set < dev_->used; ++set) {
        JobSet &js = dev_->sets[set];
        if (jobs > js.prepared) {
            if (!js.space.Prepare(SearchLanesKernel, BlockThreads(), jobs, 0u, error) ||
                !js.space.PreparePark(jobs, LaneJobs::kSaveQuads, ParkQuantum(), ParkMargin(), ParkShift(), ParkRounds(), ParkMaxRounds(), error))
                return false;
            js.prepared = jobs;
        }
        if (!js.dResults.Fit(sizeof(SearchResult) * jobs, error) || !FitTail(js, js.prepared, error)) return false;
    }
    return true;
}

void *SearchEngine::Stream(int set) const { return dev_->sets[set].stream; }
void SearchEngine::MarkPost(int set) {
    JobSet &js = dev_->sets[set];
    cudaEventRecord(js.eP, js.post != nullptr ? js.post : js.stream);
    js.marked = true;
}
void *SearchEngine::PostStream(int set) const {
    const JobSet &js = dev_->sets[set];
    return js.post != nullptr ? js.post : js.stream;
}

const SearchResult *SearchEngine::DeviceResults(int set) const { return static_cast<const SearchResult *>(dev_->sets[set].dResults.b.ptr); }

bool SearchEngine::SubmitJobs(int set, const JudgeConfig &config, const SearchJob *jobs, const std::uint32_t *controls,
                              const std::uint32_t *count, std::size_t maxJobs, std::string *error, int after) {
    if (!dev_) return *error = "SearchEngine: not initialized", false;
    Dev &d = *dev_;
    JobSet &js = d.sets[set];
    js.n = maxJobs;
    js.submitted = std::chrono::steady_clock::now();
    if (maxJobs == 0u) return true;
    if (!ReserveJobs(maxJobs, error) || !Check(cudaMemsetAsync(js.stats.ptr, 0, sizeof(KernelStats), js.stream), "stats", error)) return false;
    // after >= 0: the lanes after set `after`'s lanes and the work queued after them (MarkPost: its outcomes' kernel, else waiting for
    // SM slots behind these lanes' blocks: 8 ms on pressforward)
    if (after >= 0 && after != set &&
        !Check(cudaStreamWaitEvent(js.stream, d.sets[after].marked ? d.sets[after].eP : d.sets[after].e1, 0), "cudaStreamWaitEvent", error))
        return false;
    js.marked = false;
    if (!js.space.ResetPool(js.stream, error)) return false;
    // kSpareBlocks fewer blocks than the device holds (FOREVERVALIDATOR_TWIN_GPU_SPARE: another count): the slots where the
    // kernels around the lanes run while a batch run ahead fills the slots this batch's blocks leave. Without them the batch's
    // outcomes' kernel waited for the next batch's blocks to leave (rpg 32,768 a batch: 47 ms), and the next batch's
    // generation for this batch's.
    static const std::uint32_t spare = [] {
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_GPU_SPARE");
        return v != nullptr ? static_cast<std::uint32_t>(std::atoi(v)) : kSpareBlocks;
    }();
    const std::uint32_t blocks = js.space.blocks > spare + 1u ? js.space.blocks - spare : js.space.blocks;
    if (TailOn() && js.tail.bytes != 0u &&
        !Check(cudaMemsetAsync(js.tail.ptr, 0, sizeof(unsigned long long) * 2u * (js.space.lanes / 32u), js.stream), "tail", error))
        return false;
    cudaEventRecord(js.e0, js.stream);
    const DevPacks packs{d.car.dev()->k, d.car.dev()->w, d.car.dev()->r, d.collision.dev()->w};   // M: by value (__grid_constant__)
    SearchLanesKernel<<<blocks, js.space.threads, 0, js.stream>>>(
            packs, config, jobs, static_cast<std::uint32_t>(maxJobs),
            static_cast<const model::TwinState *>(d.starts.ptr), controls, static_cast<SearchResult *>(js.dResults.b.ptr),
            static_cast<KernelStats *>(js.stats.ptr), js.space.Rows(), js.space.Next(), 0u, count, Tail(js));
    cudaEventRecord(js.e1, js.stream);
    return Check(cudaGetLastError(), "SearchLanesKernel launch", error) &&
           (js.post == nullptr || Check(cudaStreamWaitEvent(js.post, js.e1, 0), "cudaStreamWaitEvent", error)) &&
           Check(cudaMemcpyAsync(js.hStats.p, js.stats.ptr, sizeof(KernelStats), cudaMemcpyDeviceToHost, js.stream), "stats D2H", error);
}

bool SearchEngine::Ready(int set) const {
    const JobSet &js = dev_->sets[set];
    return cudaStreamQuery(js.stream) == cudaSuccess && (js.post == nullptr || cudaStreamQuery(js.post) == cudaSuccess);
}

bool SearchEngine::LanesDone(int set) const { return cudaEventQuery(dev_->sets[set].e1) == cudaSuccess; }

bool SearchEngine::Wait(int set, SearchStats *stats, std::string *error) {
    *stats = SearchStats{};
    JobSet &js = dev_->sets[set];
    if (!Check(cudaStreamSynchronize(js.stream), "SearchLanesKernel", error) ||
        (js.post != nullptr && !Check(cudaStreamSynchronize(js.post), "SearchLanesKernel outcomes", error)))
        return false;
    if (js.n == 0u) return true;
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, js.e0, js.e1);
    stats->kernelMs = ms;
    stats->wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - js.submitted).count();
    const KernelStats &ks = *static_cast<const KernelStats *>(js.hStats.p);
    stats->substeps = ks.substeps;
    stats->ticks = ks.ticks;
    for (int k = 0; k < 8; ++k) stats->clk[k] = ks.clk[k];
    PrintTail(js, &dev_->tailLastEnd);
    return true;
}

}  // namespace tw::gpu
