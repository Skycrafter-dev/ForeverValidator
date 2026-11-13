// Twin GPU engine: the recorded-queue runner (tw_gpu.h RunQueue): every candidate's ticks from its start
// state, one candidate per thread, all worlds in one launch; optionally each state checked bit for bit
// against the recording (a different state is counted and replaced by the recording, as searchq does).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tw_gpu.h"
#include "tw_gpu_lockstep.cuh"
#include "tw_gpu_packs.cuh"
#include "tw_gpu_tick.cuh"
#include "tw_gpu_util.cuh"

namespace tw::gpu {

namespace {

using namespace dev;

struct QueueOut {
    std::uint32_t differ, substeps, lost, extra;
};

__device__ __forceinline__ bool SameState(const model::TwinState &a, const model::TwinState &b) {
    const std::uint32_t *x = reinterpret_cast<const std::uint32_t *>(&a), *y = reinterpret_cast<const std::uint32_t *>(&b);
    std::uint32_t d = 0u;
    for (int i = 0; i < static_cast<int>(sizeof(model::TwinState) / 4); ++i) d |= x[i] ^ y[i];
    return d == 0u;
}

__global__ void QueueKernel(const DevPacks *packs, const QueueCand *cands, std::uint32_t n, const model::TwinState *start,
                            const std::uint32_t *controls, const model::TwinState *after, QueueOut *out, unsigned long long *stageClk) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const QueueCand c = cands[i];
    const DevPacks &P = packs[c.world];
    model::TwinState s = start[i];
    GpuContacts contacts;
    model::TwinState finishStart;
    std::uint32_t substeps = 0u, lost = 0u, differ = 0u;
    unsigned long long clk[6] = {0ull, 0ull, 0ull, 0ull, 0ull, 0ull};
    const long long k0 = clock64();
    for (std::uint32_t t = 0; t < c.ticks; ++t) {
        RunTick(P, s, controls[c.firstTick + t], contacts, finishStart, &substeps, &lost, stageClk != nullptr ? clk : nullptr);
        if (after != nullptr && !SameState(s, after[c.firstTick + t])) {
            ++differ;
            s = after[c.firstTick + t];
        }
    }
    out[i] = QueueOut{differ, substeps, lost, 0u};
    if (stageClk != nullptr) {
        const unsigned long long total = static_cast<unsigned long long>(clock64() - k0);
        for (int k = 0; k < 6; ++k) atomicAdd(&stageClk[k], clk[k]);
        atomicAdd(&stageClk[6], total);
    }
}

// The queue as the lane runner's jobs: candidate j from start[j], controls [firstTick, firstTick + ticks); with
// `after`, each state compared with the recording (a different one counted and replaced).
struct QueueJobs {
    std::uint32_t count;
    const QueueCand *cands;
    const model::TwinState *start;
    const std::uint32_t *controls;
    const model::TwinState *after;
    QueueOut *out;
    const std::uint32_t *order;   // the pool's order (job p is candidate order[p]); null: the queue's
    std::uint32_t differ;         // the thread's current job's

    __device__ std::uint32_t Cand(std::uint32_t p) const { return order != nullptr ? order[p] : p; }
    __device__ JobRef Job(std::uint32_t p) const {
        const std::uint32_t j = Cand(p);
        const QueueCand c = cands[j];
        return JobRef{c.world, start + j, controls + c.firstTick, c.ticks};
    }
    // parking (TW_GPU_PARK): the job's differ count
    static constexpr int kSaveQuads = 1;
    __device__ JobRef Ref(std::uint32_t p) const { return Job(p); }
    __device__ std::uint32_t Ticks(std::uint32_t p) const { return cands[Cand(p)].ticks; }
    __device__ void Save(std::uint32_t, const ParkSlot &s) {
        __stcg(s.at(0), make_uint4(differ, 0u, 0u, 0u));
        differ = 0u;
    }
    __device__ void Load(std::uint32_t, const ParkSlot &s) { differ = __ldcg(s.at(0)).x; }
    __device__ bool Tick(std::uint32_t p, std::uint32_t t, model::TwinState &s, const race::TwRaceEvents &) {
        if (after == nullptr) return true;
        const model::TwinState &a = after[cands[Cand(p)].firstTick + t];
        if (!SameState(s, a)) {
            ++differ;
            lockstep::CopyState(s, a);
        }
        return true;
    }
    __device__ void Done(std::uint32_t p, std::uint32_t, const LaneCounts &c) {
        out[Cand(p)] = QueueOut{differ + c.replayDiffer, c.substeps, c.lost, c.extra};
        differ = 0u;
    }
    __device__ const model::CarForceConsts &Car(const DevPacks &P) const { return *P.k; }
};

template <bool kBarrier, bool kVerify>
__global__ void LaneKernel(const DevPacks *packs, QueueJobs jobs, LaneRows rows, unsigned *next, std::uint32_t chunk, unsigned long long *stageClk) {
    unsigned long long clk[7] = {0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull};
    jobs.differ = 0u;
    RunLanes<QueueJobs, kBarrier, kVerify>(packs, jobs, rows, next, chunk, stageClk != nullptr ? clk : nullptr);
    if (stageClk != nullptr)
        for (int k = 0; k < 7; ++k) atomicAdd(&stageClk[k], clk[k]);
}

}  // namespace

bool RunQueue(const QueueWorld *worlds, std::size_t nWorlds, const QueueCand *cands, std::size_t nCands, const model::TwinState *start,
              const std::uint32_t *controls, const model::TwinState *after, std::size_t nTicks, int reps, QueueStats *stats,
              std::string *error) {
    return RunQueue(worlds, nWorlds, cands, nCands, start, controls, after, nTicks, reps, QueueConfig{}, stats, error);
}

bool RunQueue(const QueueWorld *worlds, std::size_t nWorlds, const QueueCand *cands, std::size_t nCands, const model::TwinState *start,
              const std::uint32_t *controls, const model::TwinState *after, std::size_t nTicks, int reps, const QueueConfig &config,
              QueueStats *stats, std::string *error) {
    *stats = QueueStats{};
    if (nCands == 0u) return true;
    std::vector<DevPacks> hp(nWorlds);
    for (std::size_t m = 0; m < nWorlds; ++m) {
        const CarPacks::Dev *car = worlds[m].car->dev();
        const CollisionPacks::Dev *col = worlds[m].collision->dev();
        if (car == nullptr || col == nullptr) return *error = "RunQueue: packs not uploaded", false;
        hp[m] = DevPacks{car->k, car->w, car->r, col->w};
    }
    DevBuffer dPacks, dCands, dStart, dControls, dAfter, dOut, dClk, dOrder;
    LaneSpace space;
    if (!dPacks.Put(hp.data(), sizeof(DevPacks) * hp.size(), error) || !dCands.Put(cands, sizeof(QueueCand) * nCands, error) ||
        !dStart.Put(start, sizeof(model::TwinState) * nCands, error) || !dControls.Put(controls, sizeof(std::uint32_t) * nTicks, error) ||
        (after != nullptr && !dAfter.Put(after, sizeof(model::TwinState) * nTicks, error)) || !dOut.Alloc(sizeof(QueueOut) * nCands, error) ||
        !dClk.Alloc(sizeof(unsigned long long) * 7u, error))
        return false;
    // the launch: one thread a candidate (v1), or the lane runner's lanes (as many as are resident at once)
    std::uint32_t threads = config.block > 0 ? static_cast<std::uint32_t>(config.block) : 64u, blocks = 0u, lanes = 0u;
    void (*laneKernel)(const DevPacks *, QueueJobs, LaneRows, unsigned *, std::uint32_t, unsigned long long *) =
            std::getenv("TW_GPU_VERIFY") != nullptr && after != nullptr   // the replay check on every tick (identity runs only)
                    ? &LaneKernel<false, true>
                    : config.barrier ? &LaneKernel<true, false> : &LaneKernel<false, false>;
    const std::uint32_t chunk = config.chunk > 0 ? static_cast<std::uint32_t>(config.chunk) : 0u;
    if (config.lockstep) {
        if (!space.Prepare(laneKernel, threads, nCands, config.lanes > 0 ? static_cast<std::size_t>(config.lanes) : 0u, error)) return false;
        // TW_GPU_PARK=q: parking with a quantum of q ticks (default 16, 0: none; the lane runner sets a group aside for waiting work
        // with more ticks left, as the session's)
        const char *parkEnv = std::getenv("TW_GPU_PARK");
        const std::uint32_t quantum = parkEnv != nullptr ? static_cast<std::uint32_t>(std::atoi(parkEnv)) : 16u;
        if (!space.PreparePark(nCands, QueueJobs::kSaveQuads, quantum, quantum / 2u, 2u, 2u, 0xFFFFu, error)) return false;
        blocks = space.blocks;
        lanes = space.lanes;
    } else {
        blocks = static_cast<std::uint32_t>((nCands + threads - 1u) / threads);
        lanes = static_cast<std::uint32_t>(nCands);
    }
    // the pool's order (TW_GPU_ORDER): worldplace (default), by world, then place's order (warps rarely mix worlds:
    // B2 measured a 491 m mean position spread per warp with place); place, lengths in buckets of 32 ticks, longest
    // first (the last jobs taken are the shortest: a short tail), then world, then the start's 32 m cell, so a warp
    // batch starts in one place (candidates of one length mutate the same tick of a session); lpt, longest first
    // only; world, by world then longest first; queue, the queue's.
    // Copies of a candidate (the same controls: searchq --gpu-copies) keep the queue's distance: the k-th copy of
    // every candidate after all (k-1)-th ones, so a warp does not run one candidate's copies side by side.
    const char *orderEnv = std::getenv("TW_GPU_ORDER");
    const std::string orderMode = orderEnv != nullptr ? orderEnv : "worldplace";
    if (config.lockstep && orderMode != "queue") {
        std::vector<std::uint32_t> order(nCands), copy(nCands);
        std::vector<std::uint32_t> seen(nTicks + 1u, 0u);
        for (std::size_t i = 0; i < nCands; ++i) {
            order[i] = static_cast<std::uint32_t>(i);
            copy[i] = cands[i].firstTick < nTicks ? seen[cands[i].firstTick]++ : 0u;
        }
        const bool byWorld = orderMode == "world", byWorldPlace = orderMode == "worldplace", byPlace = orderMode == "place" || byWorldPlace;
        // place: lengths in buckets of TW_GPU_BUCKET ticks (longest bucket first), then world, then the start
        // position's 32 m cell (Morton order), so a warp batch's lanes start near each other
        const char *bucketEnv = std::getenv("TW_GPU_BUCKET");
        const std::uint32_t bucket = bucketEnv != nullptr && std::atoi(bucketEnv) > 0 ? static_cast<std::uint32_t>(std::atoi(bucketEnv)) : 32u;
        const char *cellEnv = std::getenv("TW_GPU_CELL");   // the cell's size in metres (32)
        const float cellM = cellEnv != nullptr && std::atof(cellEnv) > 0.0 ? static_cast<float>(std::atof(cellEnv)) : 32.0f;
        std::vector<std::uint64_t> cell(byPlace ? nCands : 0u);
        for (std::size_t i = 0; byPlace && i < nCands; ++i) {
            auto q = [&](float v) { return static_cast<std::uint32_t>(static_cast<std::int32_t>(std::floor(v / cellM)) + 32768) & 0xFFFFu; };
            const std::uint32_t x = q(start[i].pos[0]), z = q(start[i].pos[2]);
            std::uint64_t m = 0;
            for (int k = 0; k < 16; ++k) m |= static_cast<std::uint64_t>(((x >> k) & 1u) | (((z >> k) & 1u) << 1)) << (2 * k);
            cell[i] = m;
        }
        std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
            if (copy[a] != copy[b]) return copy[a] < copy[b];
            if (byWorldPlace && cands[a].world != cands[b].world) return cands[a].world < cands[b].world;
            if (byPlace) {
                const std::uint32_t ba = cands[a].ticks / bucket, bb = cands[b].ticks / bucket;
                if (ba != bb) return ba > bb;
                if (cands[a].world != cands[b].world) return cands[a].world < cands[b].world;
                if (cell[a] != cell[b]) return cell[a] < cell[b];
            }
            if (byWorld && cands[a].world != cands[b].world) return cands[a].world < cands[b].world;
            return cands[a].ticks > cands[b].ticks;
        });
        if (!dOrder.Put(order.data(), sizeof(std::uint32_t) * nCands, error)) return false;
    }
    QueueJobs jobs{static_cast<std::uint32_t>(nCands), static_cast<const QueueCand *>(dCands.ptr), static_cast<const model::TwinState *>(dStart.ptr),
                   static_cast<const std::uint32_t *>(dControls.ptr), after != nullptr ? static_cast<const model::TwinState *>(dAfter.ptr) : nullptr,
                   static_cast<QueueOut *>(dOut.ptr), static_cast<const std::uint32_t *>(dOrder.ptr), 0u};
    const LaneRows rows = space.Rows();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best = 0.0f;
    cudaMemset(dClk.ptr, 0, sizeof(unsigned long long) * 7u);
    for (int r = 0; r < std::max(1, reps + 1); ++r) {   // launch 0: the stage clocks (instrumented); then the timed reps
        unsigned long long *clk = r == 0 ? static_cast<unsigned long long *>(dClk.ptr) : nullptr;
        if (config.lockstep && !space.ResetPool(nullptr, error)) return false;
        cudaEventRecord(e0);
        if (config.lockstep)
            laneKernel<<<blocks, threads>>>(static_cast<const DevPacks *>(dPacks.ptr), jobs, rows, space.Next(), chunk, clk);
        else
            QueueKernel<<<blocks, threads>>>(static_cast<const DevPacks *>(dPacks.ptr), static_cast<const QueueCand *>(dCands.ptr),
                                             static_cast<std::uint32_t>(nCands), static_cast<const model::TwinState *>(dStart.ptr),
                                             static_cast<const std::uint32_t *>(dControls.ptr), jobs.after, static_cast<QueueOut *>(dOut.ptr), clk);
        cudaEventRecord(e1);
        if (!Check(cudaGetLastError(), "QueueKernel launch", error) || !Check(cudaEventSynchronize(e1), "QueueKernel", error)) {
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            return false;
        }
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        if (config.lockstep && space.Rows().quantum != 0u) {   // parking: the groups parked (the ring's pushes)
            unsigned c[lockstep::kRingCounters + 2 * lockstep::kBuckets] = {};
            if (cudaMemcpy(c, space.Next(), sizeof c, cudaMemcpyDeviceToHost) == cudaSuccess) {
                unsigned pushed = 0u, popped = 0u;
                for (int b = 0; b < lockstep::kBuckets; ++b) {
                    popped += c[lockstep::kRingCounters + 2 * b];
                    pushed += c[lockstep::kRingCounters + 2 * b + 1];
                }
                std::fprintf(stderr, "gpu park: launch %d, %u groups parked, %u resumed, %.1f ms\n", r, pushed, popped, ms);
            }
        }
        if (r == 1 || (r > 1 && ms < best)) best = ms;
        if (reps <= 0) best = ms;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::vector<QueueOut> out(nCands);
    if (!dOut.Get(out.data(), sizeof(QueueOut) * nCands, error) || !dClk.Get(stats->stageCycles, sizeof stats->stageCycles, error)) return false;
    stats->ms = best;
    stats->lanes = lanes;
    for (std::size_t i = 0; i < nCands; ++i) {
        stats->ticks += cands[i].ticks;
        stats->substeps += out[i].substeps;
        stats->differ += out[i].differ;
        stats->rawLost += out[i].lost;
        stats->extraSubsteps += out[i].extra;
    }
    return true;
}

}  // namespace tw::gpu
