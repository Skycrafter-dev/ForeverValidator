// Twin GPU engine: the lockstep lane runner (as the CPU's batch runner, src/twin/batch/batch_runner.h): persistent
// threads, each owns a lane; a pass = every live lane runs exactly one substep through the one RunSubstep call
// site; between passes a lane at its tick's end finishes the tick (the finish refinement's substeps are passes
// too), hands the state to the jobs' hook, and begins its next tick or takes the next job from the pool (a global
// counter). Each lane computes exactly what RunTick (tw_gpu_tick.cuh) computes, bit for bit; only the schedule
// differs: warp-mates with different substep counts or tick counts no longer wait for each other.
//
// Jobs (a template parameter, the engine's API for a search, docs/twin/gpu.md "Lane runner"):
//   std::uint32_t count;                                                   jobs in the batch
//   __device__ JobRef Job(std::uint32_t j) const;                          world, start state, controls, ticks
//   __device__ bool Tick(std::uint32_t j, std::uint32_t t, model::TwinState &s, const race::TwRaceEvents &ev);
//       after tick t of job j (TickEnd done): the state, which the hook may read or replace, and the tick's race
//       events; false stops the job
//   __device__ void Done(std::uint32_t j, std::uint32_t ticks, const LaneCounts &c);
//       job j is over after `ticks` ticks (all of them, or the tick whose hook said stop); its counters
//   __device__ const model::CarForceConsts &Car(const DevPacks &P) const;
//       the car's constants for a world's packs (*P.k, or the same bytes in constant memory)
// and, for parking (LaneRows::quantum != 0, chunk 0: a group of jobs set aside between ticks so that a waiting group with
// more ticks left runs, docs/twin/gpu.md "round 8, E"):
//   static constexpr int kSaveQuads;                                       the hook's per-lane state, 16-byte words
//   __device__ JobRef Ref(std::uint32_t j) const;                          as Job, without its side effects (a resume)
//   __device__ std::uint32_t Ticks(std::uint32_t j) const;                 job j's ticks (the pool's next, peeked)
//   __device__ void Save(std::uint32_t j, const ParkSlot &p);              the lane's hook state of job j, parked (word q at
//       p.at(q)); the lane's next job starts afresh
//   __device__ void Load(std::uint32_t j, const ParkSlot &p);              ... resumed
//
// The finish refinement (TickEnd, race::TwRefineFinish) needs the state at the start of the completing substep:
// instead of a copy at every substep start, the lane keeps the tick's start state (before TickBegin, one copy a
// tick, in global rows) and on a finish runs TickBegin and the substeps before the completing one again from it
// (the same inputs, so the same bits), then the bracket's probes, one substep a pass each.
#ifndef TW_GPU_LOCKSTEP_CUH
#define TW_GPU_LOCKSTEP_CUH

#include <algorithm>
#include <cstring>
#include <string>

#include "policy.h"
#include "tw_gpu_finish.h"
#include "tw_gpu_tick.cuh"
#include "tw_gpu_util.cuh"
#include "tw_gpu_arch.cuh"

namespace tw::gpu::dev {

struct JobRef {
    std::uint32_t world;
    const model::TwinState *start;
    const std::uint32_t *controls;   // the job's first tick's control first
    std::uint32_t ticks;
};

// A job's counters: main substeps (the ticks' own), extra substeps (the finish refinement's replay and probes),
// raw contacts lost.
struct LaneCounts {
    std::uint32_t substeps, extra, lost;
    std::uint32_t replayDiffer;   // kVerify: ticks whose replay from the tick-start copy gave other bits
};

// Global rows the runner keeps per lane (laneCount lanes): the tick-start state (16-byte word w of lane l at
// tickStart[w * laneCount + l], uint4), and on a finish tick the tick's end state and the completing substep's start state.
// Parking (quantum != 0): a parked job's rows (job-major by word: word w of job j at park[w * cap + j]): its tick-start
// state (kQuads words), its schedule (t, counts), the hook's words; the parked groups' rings by ticks left (queue[b * qcap +
// k]: base job, lane mask, (t << 16) | ticks left, seq = slot + 1 once written; their counters after the pool's, lockstep::kBuckets).
struct LaneRows {
    std::uint32_t *tickStart;
    model::TwinState *end, *finishStart;
    std::uint32_t laneCount;
    uint4 *park;
    uint4 *queue;
    std::uint32_t cap, qcap, quantum, margin;   // a group with `left` ticks left parks for waiting work with more than
    std::uint32_t marginShift;                  // max(margin, left >> marginShift) more
    std::uint32_t poolWindow;                   // ... and for the pool's jobs once it holds at most this many
    std::uint32_t parkJobs;                     // parking in a launch of at most this many jobs
};

// A parked job's hook words (Jobs::Save / Load): word q at p[q * stride].
struct ParkSlot {
    uint4 *p;
    std::uint32_t stride;
    __device__ __forceinline__ uint4 *at(int q) const { return p + static_cast<std::size_t>(q) * stride; }
};

namespace lockstep {

// The pass's substep with forces + integration and the response inlined (tw_gpu_tick.cuh RunSubstep<true>); 0: the
// stages' modules, as v1.
#ifndef TW_GPU_INLINE_STAGES
#define TW_GPU_INLINE_STAGES 1
#endif
constexpr bool kInlineStages = TW_GPU_INLINE_STAGES != 0;

constexpr int kWords = static_cast<int>(sizeof(model::TwinState) / 4);

__device__ __forceinline__ void CopyState(model::TwinState &d, const model::TwinState &s) {
    const uint4 *a = reinterpret_cast<const uint4 *>(&s);
    uint4 *b = reinterpret_cast<uint4 *>(&d);
#pragma unroll 4
    for (int i = 0; i < static_cast<int>(sizeof(model::TwinState) / 16); ++i) b[i] = a[i];
}

constexpr int kQuads = static_cast<int>(sizeof(model::TwinState) / 16);

// The tick-start row of a lane (16-byte word k of lane l at tickStart[k * laneCount + l], so a warp's accesses coalesce).
__device__ __forceinline__ void LoadTickStart(model::TwinState &s, const LaneRows &rows, std::uint32_t lane) {
    const uint4 *t = reinterpret_cast<const uint4 *>(rows.tickStart) + lane;
    uint4 *x = reinterpret_cast<uint4 *>(&s);
#pragma unroll 4
    for (int k = 0; k < kQuads; ++k) x[k] = t[static_cast<std::size_t>(k) * rows.laneCount];
}

enum : std::uint8_t { kMain = 0, kReplay = 1, kProbe = 2, kTake = 3, kIdle = 4, kParked = 5 };

// One lane's schedule (the job, the tick, the substep, the finish bracket).
struct Lane {
    std::uint32_t job, ticks, t, control;
    const std::uint32_t *controls;
    std::uint32_t world;
    model::CarTickIn in;
    std::uint32_t n, i;       // substeps this tick, the next one
    float split, remaining;   // ForEachSubstepDt: n - 1 splits, then the remainder
    double elapsed;           // seconds of the tick's substeps done
    bool found;               // the completing substep is known: index fin, start offset startSeconds, duration
    std::uint32_t fin;
    double startSeconds;
    float duration;
    std::uint8_t mode;
    float probeH;
    FinishBracket bracket;   // race::TwRefineFinish
    LaneCounts counts;
};

// ---- parking: the parked groups by ticks left, kBuckets rings (bucket b: groups with min(kBuckets - 1, left / quantum) = b;
// one entry a group: base job, lane mask, (t << 16) | ticks left, seq = slot + 1 once written); the counters after the pool's:
// next[kRingCounters + 2b] head, next[kRingCounters + 2b + 1] tail ----
constexpr int kBuckets = 32;   // one a lane: the warp reads all the rings' counters at once
constexpr int kRingCounters = 4;
constexpr std::size_t kRingBytes = sizeof(unsigned) * (kRingCounters + 2 * kBuckets);   // the counters' bytes (16-byte multiple)
__device__ __forceinline__ unsigned Relaxed(const unsigned *p) { return *reinterpret_cast<const volatile unsigned *>(p); }
__device__ __forceinline__ unsigned BucketOf(unsigned left, unsigned quantum) { return min(static_cast<unsigned>(kBuckets - 1), left / quantum); }

// The non-empty rings (bit b: ring b). The whole warp.
__device__ __forceinline__ unsigned RingMask(const unsigned *next, unsigned laneId) {
    const unsigned *c = next + kRingCounters + 2u * laneId;
    const unsigned h = Relaxed(c), t = Relaxed(c + 1);
    return __ballot_sync(0xFFFFFFFFu, h < t);
}

// Ring b's head group's ticks left (0: the ring is empty). One lane.
__device__ __forceinline__ unsigned HeadLeft(const LaneRows &rows, const unsigned *next, unsigned b) {
    const unsigned *c = next + kRingCounters + 2u * b;
    const unsigned h = Relaxed(c);
    if (h >= Relaxed(c + 1)) return 0u;
    const uint4 *e = rows.queue + static_cast<std::size_t>(b) * rows.qcap + h % rows.qcap;
    while (Relaxed(&e->w) < h + 1u) {
    }
    __threadfence();
    return __ldcg(&e->z) & 0xFFFFu;
}

// Ring b's head group, taken (false: the ring is empty). One lane.
__device__ __forceinline__ bool PopGroup(const LaneRows &rows, unsigned *next, unsigned b, uint4 *out) {
    unsigned *c = next + kRingCounters + 2u * b;
    unsigned h = Relaxed(c);
    for (;;) {
        if (h >= Relaxed(c + 1)) return false;
        const unsigned seen = atomicCAS(c, h, h + 1u);
        if (seen == h) break;
        h = seen;
    }
    const uint4 *e = rows.queue + static_cast<std::size_t>(b) * rows.qcap + h % rows.qcap;
    while (Relaxed(&e->w) != h + 1u) {
    }
    __threadfence();
    *out = __ldcg(e);
    return true;
}

// A parked group into its ring (its lanes' rows written and fenced before). One lane.
__device__ __forceinline__ void PushGroup(const LaneRows &rows, unsigned *next, unsigned base, unsigned mask, unsigned t, unsigned left) {
    const unsigned b = BucketOf(left, rows.quantum);
    const unsigned slot = atomicAdd(next + kRingCounters + 2u * b + 1u, 1u);
    uint4 *e = rows.queue + static_cast<std::size_t>(b) * rows.qcap + slot % rows.qcap;
    __stcg(reinterpret_cast<uint2 *>(e), make_uint2(base, mask));
    __stcg(&e->z, (t << 16) | min(left, 0xFFFFu));
    __threadfence();
    *reinterpret_cast<volatile unsigned *>(&e->w) = slot + 1u;
}

}  // namespace lockstep

// The worlds' packs: an array in global memory (JobRef::world indexes it), or one world's packs a kernel parameter
// (__grid_constant__: their fields are constant-bank operands, not loads through a pointer).
__device__ __forceinline__ const DevPacks &PacksAt(const DevPacks *p, std::uint32_t world) { return p[world]; }
struct OnePack {
    const DevPacks *p;   // the kernel's __grid_constant__ parameter
};
__device__ __forceinline__ const DevPacks &PacksAt(const OnePack &o, std::uint32_t) { return *o.p; }

// The runner. packs: the worlds' packs (JobRef::world indexes it); next: the pool's counter, 0 at launch (every lane takes
// its first job from the pool); chunk: 0, warp batches: a warp whose lanes are all free takes the next jobs, one a
// lane (warp-mates stay on neighbouring jobs, alike when the pool is ordered longest first; a free lane waits for
// its warp-mates); 2..31, the same once that many lanes are free; 1, a lane takes the pool's next job alone; >= 32,
// a warp takes `chunk` consecutive jobs at a time and its lanes take them in order; clk: stage clocks
// (QueueStats::stageCycles' layout, per thread; null: none). kVerify (a check, slow): at every tick's end the whole
// tick again from the tick-start copy, as a finish's replay does, compared with the tick's state (LaneCounts::replayDiffer).
template <class Jobs, bool kBarrier, bool kVerify = false, class Packs = const DevPacks *>
__device__ void RunLanes(const Packs packs, Jobs &jobs, const LaneRows rows, unsigned *next, std::uint32_t chunk, unsigned long long *clk) {
    using namespace lockstep;
    const std::uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    model::TwinState s;
    GpuContacts contacts;
    model::RespTick respTick, scratchTick;
    race::TwRaceEvents raceEv, scratchEv;
    Lane L;
    std::uint32_t lost = 0u;   // RunSubstep's counter (the job's share goes to LaneCounts)

    auto tickBegin = [&](model::TwinState &x, race::TwRaceEvents &ev, model::RespTick &rt) {   // TwinStages::TickBegin
        race::TwRaceEventsClear(ev);
        std::memset(&rt, 0, sizeof rt);
        const DevPacks &P = PacksAt(packs, L.world);
        race::TwRaceTickBegin(*P.race, x, L.control, race::TwCtlRespawnField(L.control), x.tickMs + 10u, jobs.Car(P).restAbsorb, ev);
        const race::TwTickInputs ti = race::TwCtlPhysicsInputs(L.control);
        L.in.gas = ti.gas;
        L.in.brake = ti.brake;
        L.in.steer = ti.steer;
        L.in.tickMs = x.tickMs;
    };
    auto beginTick = [&]() {
        L.control = L.controls[L.t];
        {   // the tick's start state, for a finish's replay
            const uint4 *w = reinterpret_cast<const uint4 *>(&s);
            uint4 *t = reinterpret_cast<uint4 *>(rows.tickStart) + lane;
#pragma unroll 4
            for (int k = 0; k < kQuads; ++k) __stcs(t + static_cast<std::size_t>(k) * rows.laneCount, w[k]);
        }
        tickBegin(s, raceEv, respTick);
        std::uint32_t n = step::FvSubstepCount(s, step::kTickDt);
        if (n == 0u) n = 1u;
        L.n = n;
        L.i = 0u;
        L.remaining = step::kTickDt;
        L.split = n > 1u ? step::kTickDt / tw::math::from_u32(n) : 0.0f;
        L.elapsed = 0.0;
        L.found = false;
        L.mode = kMain;
    };
    // job j (past the pool's end: the lane is idle; no ticks: done at once, the lane takes another)
    auto takeJob = [&](std::uint32_t j) {
        if (j >= jobs.count) {
            L.mode = kIdle;
            return;
        }
        const JobRef r = jobs.Job(j);
        if (r.ticks == 0u) {
            jobs.Done(j, 0u, LaneCounts{0u, 0u, 0u, 0u});
            L.mode = kTake;
            return;
        }
        L.job = j;
        L.world = r.world;
        L.controls = r.controls;
        L.ticks = r.ticks;
        L.t = 0u;
        L.counts = LaneCounts{0u, 0u, 0u, 0u};
        CopyState(s, *r.start);
        beginTick();
    };
    auto endTick = [&]() {   // the tick is over (TickEnd done): the hook, then the next tick or job
        const bool go = jobs.Tick(L.job, L.t, s, raceEv);
        ++L.t;
        if (go && L.t < L.ticks) {
            beginTick();
            return;
        }
        jobs.Done(L.job, L.t, L.counts);
        L.mode = kTake;
    };
    auto finishTick = [&](bool store, std::uint64_t lowerNs) {
        CopyState(s, rows.end[lane]);
        if (store) race::TwStoreFinish(s, lowerNs);
        endTick();
    };
    auto startProbes = [&]() {   // race::TwRefineFinish up to its first probe (s: the completing substep's start)
        CopyState(rows.finishStart[lane], s);
        const race::TwFinishSubstep sub{L.in.tickMs, L.startSeconds, L.duration};
        if (L.bracket.Begin(sub, &L.probeH))
            L.mode = kProbe;
        else
            finishTick(false, 0u);
    };

    // parking (rows.quantum != 0, chunk 0): a group of jobs (a warp's lanes, jobs base + lane) set aside at a tick's start when
    // waiting work has more ticks left, so that the batch's jobs end together instead of a last round on a few lanes
    auto parkLane = [&]() {   // at a tick's start (TickBegin done, no substep run): the tick-start row, the schedule, the hook
        const std::uint32_t j = L.job, cap = rows.cap;
        const uint4 *ts = reinterpret_cast<const uint4 *>(rows.tickStart) + lane;
        uint4 *d = rows.park + j;
#pragma unroll 4
        for (int k = 0; k < kQuads; ++k) __stcg(d + static_cast<std::size_t>(k) * cap, ts[static_cast<std::size_t>(k) * rows.laneCount]);
        __stcg(d + static_cast<std::size_t>(kQuads) * cap, make_uint4(L.t, L.counts.substeps, L.counts.extra, L.counts.lost));
        jobs.Save(j, ParkSlot{d + static_cast<std::size_t>(kQuads + 1) * cap, cap});
        L.mode = kParked;
    };
    auto resumeJob = [&](std::uint32_t j) {   // a parked job again, from its tick's start
        const JobRef r = jobs.Ref(j);
        const std::uint32_t cap = rows.cap;
        L.job = j;
        L.world = r.world;
        L.controls = r.controls;
        L.ticks = r.ticks;
        uint4 *d = rows.park + j;
        uint4 *x = reinterpret_cast<uint4 *>(&s);
#pragma unroll 4
        for (int k = 0; k < kQuads; ++k) x[k] = __ldcg(d + static_cast<std::size_t>(k) * cap);
        const uint4 c = __ldcg(d + static_cast<std::size_t>(kQuads) * cap);
        L.t = c.x;
        L.counts = LaneCounts{c.y, c.z, c.w, 0u};
        jobs.Load(j, ParkSlot{d + static_cast<std::size_t>(kQuads + 1) * cap, cap});
        beginTick();
    };
    // parking only in a batch of at most rows.parkJobs jobs (a deeper batch's tail is short beside its span: parking costs more there,
    // offroad 32,768 x 8 in groups of 2 batches -3%)
    const bool parkOn = rows.quantum != 0u && chunk == 0u && !kVerify && !kBarrier && jobs.count <= rows.parkJobs;
    std::uint32_t tq = rows.quantum;   // warp-uniform: the tick at which the warp's group may park
    bool parking = false;              // warp-uniform: the group parks as its lanes reach tq

    const std::uint32_t laneId = threadIdx.x & 31u;
    std::uint32_t chunkNext = 0u, chunkEnd = 0u;   // the warp's chunk of jobs (chunk mode)
    const long long k0 = clock64();
    L.mode = kTake;   // the first job from the pool too: lanes take jobs in the order they start (a block that starts late, behind
                      // another kernel's blocks, takes what is left then)
    for (;;) {
        // lanes whose job is over take the next one
        if (parkOn) {
            constexpr unsigned kAll = 0xFFFFFFFFu;
            const bool fresh = L.mode == kMain && L.i == 0u;   // a tick begun, no substep run yet
            if (__ballot_sync(kAll, fresh && L.t >= tq) != 0u) {
                if (!parking) {   // park if waiting work has more ticks left than the group (by the margin): a parked group, or
                    // the pool's next jobs once the pool is down to its last rounds (poolWindow jobs; a deep pool's jobs start as
                    // the groups that run end: parking for them is churn)
                    const unsigned left = __reduce_max_sync(kAll, L.mode < kTake ? L.ticks - L.t : 0u);
                    const unsigned rings = RingMask(next, laneId);
                    unsigned wait = 0u;
                    if (laneId == 0u) {
                        const unsigned n = Relaxed(next);
                        wait = n < jobs.count && jobs.count - n <= rows.poolWindow ? jobs.Ticks(n) : 0u;
                        if (rings != 0u) wait = max(wait, HeadLeft(rows, next, 31u - __clz(rings)));
                    }
                    wait = __shfl_sync(kAll, wait, 0);
                    parking = wait > left + max(rows.margin, left >> rows.marginShift);
                    if (!parking) tq += rows.quantum;
                }
                if (parking && fresh && L.t >= tq) parkLane();
            }
            if (__ballot_sync(kAll, L.mode < kTake) == 0u) {   // every lane free: the group parks, the warp takes the waiting work
                const unsigned parked = __ballot_sync(kAll, L.mode == kParked);
                if (parked != 0u) {
                    const unsigned left = __reduce_max_sync(kAll, L.mode == kParked ? L.ticks - L.t : 0u);
                    const unsigned base = __shfl_sync(kAll, L.job - laneId, __ffs(parked) - 1);
                    __threadfence();
                    __syncwarp();
                    if (laneId == 0u) PushGroup(rows, next, base, parked, tq, left);
                }
                // the waiting work with the most ticks left: the pool's next 32 jobs or the ring's head group
                uint4 g = make_uint4(0u, 0u, 0u, 0u);
                unsigned kind = 0u;   // 0 none, 1 the pool's jobs g.x.., 2 the ring's group g
                for (unsigned rings = RingMask(next, laneId);; rings = RingMask(next, laneId)) {
                    if (laneId == 0u) {
                        const unsigned n = Relaxed(next);
                        const unsigned poolLeft = n < jobs.count ? jobs.Ticks(n) : 0u;
                        const unsigned top = rings != 0u ? 31u - __clz(rings) : 0u;
                        if (rings != 0u && HeadLeft(rows, next, top) > poolLeft && PopGroup(rows, next, top, &g)) kind = 2u;
                        if (kind == 0u && n < jobs.count) {
                            g.x = atomicAdd(next, 32u);
                            if (g.x < jobs.count) kind = 1u;
                        }
                        if (kind == 0u && rings != 0u && PopGroup(rows, next, top, &g)) kind = 2u;
                        if (kind == 0u && rings != 0u) kind = 3u;   // the top ring emptied meanwhile: look again
                    }
                    kind = __shfl_sync(kAll, kind, 0);
                    if (kind != 3u) break;
                    kind = 0u;
                }
                g.x = __shfl_sync(kAll, g.x, 0);
                g.y = __shfl_sync(kAll, g.y, 0);
                g.z = __shfl_sync(kAll, g.z, 0);
                parking = false;
                if (kind == 1u) {
                    tq = rows.quantum;
                    takeJob(g.x + laneId);
                } else if (kind == 2u) {
                    tq = (g.z >> 16) + rows.quantum;
                    if (((g.y >> laneId) & 1u) != 0u)
                        resumeJob(g.x + laneId);
                    else
                        L.mode = kIdle;
                } else {
                    L.mode = kIdle;
                }
            }
        } else if (chunk == 1u) {
            while (L.mode == kTake) takeJob(atomicAdd(next, 1u));
        } else if (chunk < 32u) {   // warp batches: once all its lanes (or `chunk` of them) are free, they take the next jobs
            const unsigned busy = __ballot_sync(0xFFFFFFFFu, L.mode < kTake), take = __ballot_sync(0xFFFFFFFFu, L.mode == kTake);
            if (take != 0u && (busy == 0u || (chunk != 0u && static_cast<std::uint32_t>(__popc(take)) >= chunk))) {
                const int leader = __ffs(take) - 1;
                std::uint32_t base = 0u;
                if (static_cast<int>(laneId) == leader) base = atomicAdd(next, static_cast<unsigned>(__popc(take)));
                base = __shfl_sync(0xFFFFFFFFu, base, leader);
                if (L.mode == kTake) takeJob(base + static_cast<std::uint32_t>(__popc(take & ((1u << laneId) - 1u))));
            }
        } else {
            for (unsigned need = __ballot_sync(0xFFFFFFFFu, L.mode == kTake); need != 0u; need = __ballot_sync(0xFFFFFFFFu, L.mode == kTake)) {
                if (chunkNext == chunkEnd) {
                    std::uint32_t base = 0u;
                    if (laneId == 0u) base = atomicAdd(next, chunk);
                    base = __shfl_sync(0xFFFFFFFFu, base, 0);
                    chunkNext = base;
                    chunkEnd = base + chunk;
                }
                const std::uint32_t avail = chunkEnd - chunkNext, rank = __popc(need & ((1u << laneId) - 1u));
                if (L.mode == kTake && rank < avail) takeJob(chunkNext + rank);
                chunkNext += min(avail, static_cast<std::uint32_t>(__popc(need)));
            }
        }
        const bool live = L.mode < kTake;   // kTake: a warp batch's lane waiting for its warp-mates
        if (kBarrier) {
            if (__syncthreads_or(L.mode != kIdle) == 0) break;
            if (!live) continue;
        } else if (chunk != 1u) {
            if (!__any_sync(0xFFFFFFFFu, L.mode != kIdle)) break;
            if (!live) continue;
        } else if (!live) {
            break;
        }
        // ---- the pass: one substep ----
        const std::uint8_t mode = L.mode;
        const bool main = mode == kMain;
        const float h = main ? (L.i + 1u < L.n ? L.split : L.remaining) : mode == kReplay ? L.split : L.probeH;
        if (mode == kProbe) {   // a probe: fresh stage state (TickEnd's probe)
            scratchEv = race::TwRaceEvents{};
            scratchTick = model::RespTick{};
        }
        const bool track = main && !L.found && !s.raceCompleted;
        const long long c0 = clock64();
        const std::uint32_t lost0 = lost;
        const DevPacks &P = PacksAt(packs, L.world);
        RunSubstepK<kInlineStages>(P, jobs.Car(P), s, L.in, h, main ? respTick : scratchTick, main ? raceEv : scratchEv, contacts, &lost,
                                   main ? clk : nullptr);
        const long long c1 = clock64();
        if (clk != nullptr) clk[5] += static_cast<unsigned long long>(c1 - c0);
        L.counts.lost += lost - lost0;
        // ---- between passes ----
        if (main) {
            ++L.counts.substeps;
            if (track && s.raceCompleted) {
                L.found = true;
                L.fin = L.i;
                L.startSeconds = L.elapsed;
                L.duration = h;
            }
            L.elapsed += static_cast<double>(h);
            if (L.i + 1u < L.n) L.remaining = L.remaining - L.split;
            ++L.i;
            if (kVerify && L.i == L.n) {   // the finish replay's path on every tick: the tick again from its start copy
                CopyState(rows.end[lane], s);
                LoadTickStart(s, rows, lane);
                const model::CarTickIn in = L.in;
                tickBegin(s, scratchEv, scratchTick);
                float rem = step::kTickDt;
                for (std::uint32_t k = 0; k < L.n; ++k) {
                    const float hk = k + 1u < L.n ? L.split : rem;
                    RunSubstep(PacksAt(packs, L.world), s, L.in, hk, scratchTick, scratchEv, contacts, &lost, nullptr);
                    if (k + 1u < L.n) rem = rem - L.split;
                }
                L.in = in;
                const std::uint32_t *a = reinterpret_cast<const std::uint32_t *>(&s);
                const std::uint32_t *b = reinterpret_cast<const std::uint32_t *>(&rows.end[lane]);
                std::uint32_t d = 0u;
                for (int k = 0; k < kWords; ++k) d |= a[k] ^ b[k];
                if (d != 0u) ++L.counts.replayDiffer;
                CopyState(s, rows.end[lane]);
            }
            if (L.i == L.n) {
                if (raceEv.raceCompleted == 0u || !L.found) {
                    const long long e0 = clock64();
                    endTick();
                    if (clk != nullptr) clk[3] += static_cast<unsigned long long>(clock64() - e0);
                } else {   // TickEnd (R4.2): the completing substep's start state again, then the bracket's probes
                    const long long f0 = clock64();
                    CopyState(rows.end[lane], s);
                    LoadTickStart(s, rows, lane);
                    const model::CarTickIn in = L.in;
                    tickBegin(s, scratchEv, scratchTick);
                    L.in = in;
                    if (L.fin == 0u) {
                        startProbes();
                    } else {
                        L.mode = kReplay;
                        L.i = 0u;
                    }
                    if (clk != nullptr) clk[4] += static_cast<unsigned long long>(clock64() - f0);
                }
            }
        } else if (mode == kReplay) {
            ++L.counts.extra;
            if (++L.i == L.fin) startProbes();
        } else {
            ++L.counts.extra;
            if (L.bracket.Next(s.raceCompleted != 0u, &L.probeH))
                CopyState(s, rows.finishStart[lane]);
            else
                finishTick(L.bracket.Valid(), L.bracket.LowerNs());
        }
    }
    if (clk != nullptr) clk[6] += static_cast<unsigned long long>(clock64() - k0);
}

}  // namespace tw::gpu::dev

namespace tw::gpu {

// Host: a lane runner kernel's lanes (as many as the GPU holds at once, at most the jobs rounded up to a block, or
// `maxLanes`), their rows and the pool's counter. Before each launch: ResetPool; launch <<<blocks, threads>>> with
// Rows() and Next(). Parking (PreparePark, quantum != 0): the parked jobs' rows for up to `jobs` jobs and the ring.
struct LaneSpace {
    DevBuffer tickStart, end, finishStart, next, park;
    std::uint32_t blocks = 0u, threads = 0u, lanes = 0u;
    std::uint32_t cap = 0u, qcap = 0u, quantum = 0u, margin = 0u, marginShift = 31u, windowRounds = 2u, jobRounds = ~0u;

    template <class Kernel>
    bool Prepare(Kernel kernel, std::uint32_t threadsPerBlock, std::size_t jobs, std::size_t maxLanes, std::string *error) {
        int device = 0, sms = 0, perSm = 0;
        if (!Check(cudaGetDevice(&device), "cudaGetDevice", error) ||
            !Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "cudaDeviceGetAttribute", error) ||
            !Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kernel, static_cast<int>(threadsPerBlock), 0), "occupancy", error))
            return false;
        std::size_t want = static_cast<std::size_t>(sms) * static_cast<std::size_t>(perSm) * threadsPerBlock;
        if (maxLanes > 0u && maxLanes < want) want = maxLanes;
        const std::size_t most = (jobs + threadsPerBlock - 1u) / threadsPerBlock * threadsPerBlock;
        if (most < want) want = most;
        threads = threadsPerBlock;
        blocks = static_cast<std::uint32_t>((want + threads - 1u) / threads);
        lanes = blocks * threads;
        if (lanes == 0u) return true;
        const std::size_t bytes = sizeof(model::TwinState) * lanes;
        return (tickStart.bytes == bytes || tickStart.Alloc(bytes, error)) && (end.bytes == bytes || end.Alloc(bytes, error)) &&
               (finishStart.bytes == bytes || finishStart.Alloc(bytes, error)) && (next.bytes != 0u || next.Alloc(16u, error));
    }
    // Parking with `quantum` ticks (0: none) and a `margin` in ticks for up to `jobs` jobs, the hook's words `hookQuads` (after
    // Prepare).
    bool PreparePark(std::size_t jobs, int hookQuads, std::uint32_t q, std::uint32_t m, std::uint32_t shift, std::uint32_t rounds,
                     std::uint32_t maxRounds, std::string *error) {
        quantum = q;
        margin = m;
        marginShift = shift;
        windowRounds = rounds;
        jobRounds = maxRounds;
        if (q == 0u || jobs == 0u) return true;
        if (maxRounds < 0xFFFFu) jobs = std::min<std::size_t>(jobs, static_cast<std::size_t>(maxRounds) * lanes);   // no parking past it
        if (jobs > cap) {
            cap = static_cast<std::uint32_t>((jobs + 31u) / 32u * 32u);
            qcap = cap / 32u + 2u;   // a group is in at most one ring at a time
            if (!park.Alloc(sizeof(uint4) * (static_cast<std::size_t>(dev::lockstep::kQuads + 1 + hookQuads) * cap), error) ||
                !next.Alloc(RingBytes(), error))
                return false;
        }
        return true;
    }
    dev::LaneRows Rows() const {
        const bool on = quantum != 0u && cap != 0u;
        return dev::LaneRows{static_cast<std::uint32_t *>(tickStart.ptr), static_cast<model::TwinState *>(end.ptr),
                             static_cast<model::TwinState *>(finishStart.ptr), lanes,
                             on ? static_cast<uint4 *>(park.ptr) : nullptr,
                             on ? reinterpret_cast<uint4 *>(static_cast<unsigned char *>(next.ptr) + dev::lockstep::kRingBytes) : nullptr, cap, qcap,
                             on ? quantum : 0u, margin, marginShift, windowRounds * lanes,
                             jobRounds >= 0xFFFFu ? cap : std::min(cap, jobRounds * lanes)};
    }
    unsigned *Next() const { return static_cast<unsigned *>(next.ptr); }
    std::size_t RingBytes() const { return dev::lockstep::kRingBytes + sizeof(uint4) * dev::lockstep::kBuckets * qcap; }
    bool ResetPool(cudaStream_t stream, std::string *error) {   // the pool starts at job 0 (and the parked groups' ring empty)
        const std::size_t bytes = quantum != 0u && cap != 0u ? RingBytes() : sizeof(unsigned);
        return Check(cudaMemsetAsync(next.ptr, 0, bytes, stream), "pool reset", error);
    }
};

}  // namespace tw::gpu

#endif
