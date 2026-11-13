// Twin GPU engine: C's collision (tw_gpu_stages.cuh, tw_gpu_collide.cuh).
#include "tw_gpu_collide.cuh"
#include "tw_gpu_stages.cuh"

namespace tw::gpu::dev {

// TW_GPU_DETECT_STATS (DevWorld::stats): per warp call (the lanes calling together), [0] calls, [1] lanes, [2] the lanes'
// positions' box diagonal (mm), [3] diagonals under 5 m, [4] under 20 m, [5] the slowest lane's cycles, [6] the lanes' cycles;
// what a warp queue of narrow phases would hold: [7] box pairs (sum over the lanes), [8] the most of one lane, [9] plane
// passes, [10] the most of one lane, [11] UnitContact runs (first pass), [12] the most of one lane, [13] raw contacts, [14]
// frames made, [15] non-primaries run again, [16] rounds of 32 per shape (sum over shapes of ceil(plane passes / 32)),
// [17] rounds of 32 across shapes, [18] lanes with a plane pass, [19] tree nodes, [20] the most of one lane; [21] candidate
// ranks, plane prefilter rejects [22] behind and [23] in front, [24] box pairs of candidate ranks the prefilter rejects for
// every shape, [25] those ranks, [26] edge rejects, [27] merged contacts, [28] leaf hits, [29] box pairs the phase-2 plane tests drop, [30] ranks UnionFront drops, [31] pairs the edge planes reject (sums over the lanes).
// One call site of Detect (one inlined copy: two copies of the walk in the hot code cost instruction cache).
__device__ void DetectAt(const DevWorld &w, const float pos[3], const float quat[4], const float absorb[4], GpuContacts &out) {
    const bool st = w.stats != nullptr;
    if (!st) {
        for (int i = 0; i < w.repeat; ++i) Detect(w, pos, quat, absorb, out);   // repeat (measurement): the same output each time
        return;
    }
    const unsigned m = __activemask();
    float d2 = 0.0f;
    for (int a = 0; a < 3; ++a) {
        const float e = FUnkey(__reduce_max_sync(m, FKey(pos[a]))) - FUnkey(__reduce_min_sync(m, FKey(pos[a])));
        d2 += e * e;
    }
    const long long t0 = clock64();
    DetectCount c{};
    DetectT<true>(w, pos, quat, absorb, out, &c);
    const unsigned cyc = static_cast<unsigned>(clock64() - t0);
    const unsigned mx = __reduce_max_sync(m, cyc), sum = __reduce_add_sync(m, cyc);
    unsigned v[32];
    v[7] = __reduce_add_sync(m, c.boxPairs);
    v[8] = __reduce_max_sync(m, c.boxPairs);
    v[9] = __reduce_add_sync(m, c.planePass);
    v[10] = __reduce_max_sync(m, c.planePass);
    v[11] = __reduce_add_sync(m, c.ucRuns);
    v[12] = __reduce_max_sync(m, c.ucRuns);
    v[13] = __reduce_add_sync(m, c.raws);
    v[14] = __reduce_add_sync(m, c.frames);
    v[15] = __reduce_add_sync(m, c.second);
    v[16] = 0u;
    for (int k = 0; k < 8; ++k) v[16] += (__reduce_add_sync(m, c.planeK[k]) + 31u) / 32u;
    v[17] = (v[9] + 31u) / 32u;
    v[18] = __popc(__ballot_sync(m, c.planePass != 0u));
    v[19] = __reduce_add_sync(m, c.nodes);
    v[20] = __reduce_max_sync(m, c.nodes);
    v[21] = __reduce_add_sync(m, c.cands);
    v[22] = __reduce_add_sync(m, c.behind);
    v[23] = __reduce_add_sync(m, c.front);
    v[24] = __reduce_add_sync(m, c.rankRejPairs);
    v[25] = __reduce_add_sync(m, c.rankRej);
    v[26] = __reduce_add_sync(m, c.planePass - c.ucRuns);
    v[27] = __reduce_add_sync(m, c.contacts);
    v[28] = __reduce_add_sync(m, c.leafHits);
    v[29] = __reduce_add_sync(m, c.boxPlaneRej);
    v[30] = __reduce_add_sync(m, c.unionFront);
    v[31] = __reduce_add_sync(m, c.edgeOut);
    if (static_cast<int>(threadIdx.x & 31u) == __ffs(m) - 1) {
        const float d = sqrtf(d2);
        atomicAdd(&w.stats[0], 1ull);
        atomicAdd(&w.stats[1], static_cast<unsigned long long>(__popc(m)));
        atomicAdd(&w.stats[2], static_cast<unsigned long long>(d * 1000.0f));
        atomicAdd(&w.stats[3], d < 5.0f ? 1ull : 0ull);
        atomicAdd(&w.stats[4], d < 20.0f ? 1ull : 0ull);
        atomicAdd(&w.stats[5], static_cast<unsigned long long>(mx));
        atomicAdd(&w.stats[6], static_cast<unsigned long long>(sum));
        for (int i = 7; i < 32; ++i) atomicAdd(&w.stats[i], static_cast<unsigned long long>(v[i]));
    }
}

}  // namespace tw::gpu::dev
