// Twin GPU engine: C's collision stage on the device (tw_gpu_collide.cuh): the device world's upload and the
// stage check (RunDetect).
#include "tw_gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tw_gpu_collide.cuh"
#include "tw_gpu_packs.cuh"
#include "tw_gpu_stages.cuh"
#include "tw_gpu_util.cuh"

namespace tw::gpu {

bool BuildHostWorld(const world::WorldPackView &view, const collide::CarShapes &car, HostWorld *w, std::string *error);

namespace {

using namespace dev;

__global__ void DetectKernel(DevWorld w, const DetectCase *cases, GpuContacts *out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    DetectAt(w, cases[i].pos, cases[i].quat, cases[i].absorb, out[i]);
}

// The benchmark's timed launch: the tick's call (DetectAt, contacts in local memory, the world in global memory as the
// tick's DevPacks), one word written out.
__global__ void BenchKernel(const DevWorld *w, const DetectCase *cases, const std::uint32_t *idx, std::uint32_t n, std::uint32_t *sink) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || idx[i] == kIdle) return;
    const DetectCase &c = cases[idx[i]];
    GpuContacts out;
    DetectAt(*w, c.pos, c.quat, c.absorb, out);
    sink[i] = static_cast<std::uint32_t>(out.n) ^ out.rawLost ^ (out.n > 0 ? __float_as_uint(out.c[out.n - 1].P[1]) : 0u);
}

// The measured launch: kCount false, the SM cycles of DetectAt per thread and the slowest lane's per warp; kCount true, the
// per-call counters (DetectT<true> inline).
template <bool kCount>
__global__ void BenchStatsKernel(DevWorld w, const DevWorld *gw, const DetectCase *cases, const std::uint32_t *idx, std::uint32_t n,
                                 unsigned long long *acc) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool on = i < n && idx[i] != kIdle;
    unsigned cyc = 0u;
    if (on) {
        const DetectCase &c = cases[idx[i]];
        GpuContacts out;
        if constexpr (kCount) {
            DetectCount k{};
            DetectT<true>(w, c.pos, c.quat, c.absorb, out, &k);
            atomicAdd(&acc[3], static_cast<unsigned long long>(k.nodes));
            atomicAdd(&acc[4], static_cast<unsigned long long>(k.childTests));
            atomicAdd(&acc[5], static_cast<unsigned long long>(k.leafHits));
            atomicAdd(&acc[6], static_cast<unsigned long long>(k.boxPairs));
            atomicAdd(&acc[7], static_cast<unsigned long long>(k.planePass));
            atomicAdd(&acc[8], static_cast<unsigned long long>(k.raws));
            atomicAdd(&acc[9], static_cast<unsigned long long>(out.n));
        } else {
            const long long t0 = clock64();
            DetectAt(*gw, c.pos, c.quat, c.absorb, out);
            cyc = static_cast<unsigned>(clock64() - t0);
            if (out.n < 0) acc[0] = 0ull;   // keeps the call
            atomicAdd(&acc[0], 1ull);
            atomicAdd(&acc[1], static_cast<unsigned long long>(cyc));
        }
    }
    if constexpr (!kCount) {
        const unsigned m = __reduce_max_sync(0xFFFFFFFFu, cyc);
        if ((threadIdx.x & 31u) == 0u) atomicAdd(&acc[2], static_cast<unsigned long long>(m));
    }
}

}  // namespace

CollisionPacks::CollisionPacks() = default;
CollisionPacks::~CollisionPacks() = default;

bool CollisionPacks::Upload(const world::WorldPackView &view, const collide::CarShapes &car, std::string *error) {
    HostWorld h;
    if (!BuildHostWorld(view, car, &h, error)) return false;
    if (h.depth + 1u > static_cast<std::uint32_t>(dev::kStack / 16)) return *error = "gpu world: tree too deep for the walk's stack", false;
    if (h.stackNeed > static_cast<std::uint32_t>(dev::kStack)) h.root = 0u;   // the spatial tree too deep for the stack: the FV tree
    if (h.ranks >= (1u << 24)) return *error = "gpu world: more than 2^24 triangles (the candidate list's rank field)", false;
    auto d = std::make_unique<Dev>();
    std::vector<RankN> rankN(h.rank.size());   // plane, RankRef and vertices side by side (Dev::rank holds them; Dev::plane stays empty)
    for (std::size_t r = 0; r < rankN.size(); ++r) {
        for (int a = 0; a < 4; ++a) rankN[r].plane[a] = h.plane[4 * r + a];
        rankN[r].instance = h.rank[r].instance;
        rankN[r].tri = h.rank[r].tri;
        for (int j = 0; j < 9; ++j) rankN[r].v[j] = h.verts[std::size_t(9) * h.rank[r].modelTri + j];
        rankN[r].pad = 0u;
    }
    if (!d->qnodes.Put(h.qnodes.data(), sizeof(NodeQ) * h.qnodes.size(), error) ||
        !d->rankBox.Put(h.rankBox.data(), sizeof(RankBox) * h.rankBox.size(), error) ||
        !d->rank.Put(rankN.data(), sizeof(RankN) * rankN.size(), error) ||
        !d->plane.Put(h.edge.data(), sizeof(RankE) * h.edge.size(), error) ||   // Dev::plane holds the edge planes (RankE)
        !d->iso.Put(h.iso.data(), sizeof(float) * h.iso.size(), error) ||
        !d->outRec.Put(h.outRec.data(), sizeof(OutRecG) * h.outRec.size(), error))
        return false;
    if (std::getenv("TW_GPU_DETECT_STATS") != nullptr) {
        if (!d->stats.Alloc(sizeof(unsigned long long) * kDetectStats, error) ||
            !Check(cudaMemset(d->stats.ptr, 0, sizeof(unsigned long long) * kDetectStats), "stats", error))
            return false;
        d->w.stats = static_cast<unsigned long long *>(d->stats.ptr);
    }
    d->w.car = h.car;
    d->w.k = h.k;
    d->w.qnodes = static_cast<const NodeQ *>(d->qnodes.ptr);
    d->w.rankBox = static_cast<const RankBox *>(d->rankBox.ptr);
    d->w.rankN = static_cast<const RankN *>(d->rank.ptr);
    d->w.rankE = static_cast<const RankE *>(d->plane.ptr);
    d->w.iso = static_cast<const float *>(d->iso.ptr);
    d->w.outRec = static_cast<const OutRecG *>(d->outRec.ptr);
    d->w.root = h.root;
    const char *repeat = std::getenv("TW_GPU_DETECT_REPEAT");
    d->w.repeat = repeat != nullptr ? std::max(1, std::atoi(repeat)) : 1;
    d->bytes = d->qnodes.bytes + d->rankBox.bytes + d->rank.bytes + d->plane.bytes + d->iso.bytes + d->verts.bytes + d->outRec.bytes;
    dev_ = std::move(d);
    return true;
}

std::size_t CollisionPacks::Bytes() const { return dev_ ? dev_->bytes : 0u; }

bool CollisionPacks::DetectStats(unsigned long long out[kDetectStats], std::string *error) const {
    for (int k = 0; k < kDetectStats; ++k) out[k] = 0ull;
    if (!dev_ || dev_->w.stats == nullptr) return true;
    return Check(cudaDeviceSynchronize(), "stats", error) && dev_->stats.Get(out, sizeof(unsigned long long) * kDetectStats, error);
}

void CollisionPacks::PrintDetectStats(const unsigned long long a[kDetectStats]) {
    const double wc = double(std::max(1ull, a[0])), ln = double(std::max(1ull, a[1]));
    std::printf("gpu detect stats: %llu warp calls, %.1f lanes a call, spread %.1f m (%.0f%% < 5 m, %.0f%% < 20 m), cycles a call: lane %.0f, "
                "slowest lane %.0f\n",
                a[0], double(a[1]) / wc, double(a[2]) * 1e-3 / wc, 100.0 * double(a[3]) / wc, 100.0 * double(a[4]) / wc, double(a[6]) / ln,
                double(a[5]) / wc);
    std::printf("gpu detect queue: per lane-call box pairs %.2f, plane passes %.2f, UnitContact %.2f, raws %.2f, frames %.2f, second %.2f, "
                "nodes %.1f; per warp call: box pairs %.1f (max lane %.1f), plane passes %.1f (max lane %.1f, lanes with one %.1f), "
                "UnitContact %.1f (max lane %.1f), nodes max lane %.1f; rounds of 32 per shape %.2f, across shapes %.2f\n",
                double(a[7]) / ln, double(a[9]) / ln, double(a[11]) / ln, double(a[13]) / ln, double(a[14]) / ln, double(a[15]) / ln,
                double(a[19]) / ln, double(a[7]) / wc, double(a[8]) / wc, double(a[9]) / wc, double(a[10]) / wc, double(a[18]) / wc,
                double(a[11]) / wc, double(a[12]) / wc, double(a[20]) / wc, double(a[16]) / wc, double(a[17]) / wc);
    std::printf("gpu detect rejects: per lane-call leaf hits %.2f, candidate ranks %.2f, plane rejects behind %.2f / in front %.2f, box pairs of "
                "ranks rejected for every shape %.2f (%.2f ranks), edge rejects %.2f, contacts %.2f, box pairs dropped in phase 2 %.2f (ranks in front of the union box %.2f), pairs the edge planes reject %.2f\n",
                double(a[28]) / ln, double(a[21]) / ln, double(a[22]) / ln, double(a[23]) / ln, double(a[24]) / ln, double(a[25]) / ln,
                double(a[26]) / ln, double(a[27]) / ln, double(a[29]) / ln, double(a[30]) / ln, double(a[31]) / ln);
}

bool RunDetect(const CollisionPacks &packs, const DetectCase *cases, std::size_t n, GpuContacts *out, std::string *error) {
    if (packs.dev() == nullptr) return *error = "RunDetect: packs not uploaded", false;
    if (n == 0u) return true;
    DevBuffer in, res;
    if (!in.Put(cases, sizeof(DetectCase) * n, error) || !res.Alloc(sizeof(GpuContacts) * n, error)) return false;
    const std::uint32_t threads = 64u, blocks = static_cast<std::uint32_t>((n + threads - 1u) / threads);
    DetectKernel<<<blocks, threads>>>(packs.dev()->w, static_cast<const DetectCase *>(in.ptr), static_cast<GpuContacts *>(res.ptr),
                                      static_cast<std::uint32_t>(n));
    if (!Check(cudaGetLastError(), "DetectKernel launch", error) || !Check(cudaDeviceSynchronize(), "DetectKernel", error)) return false;
    return res.Get(out, sizeof(GpuContacts) * n, error);
}

bool BenchDetect(const CollisionPacks &packs, const DetectCase *cases, std::size_t nCases, const std::uint32_t *idx, std::size_t n, int reps,
                 DetectBenchStats *stats, std::string *error) {
    *stats = DetectBenchStats{};
    if (packs.dev() == nullptr) return *error = "BenchDetect: packs not uploaded", false;
    if (n == 0u) return true;
    DevBuffer dc, di, sink, acc, dw;
    DevWorld w = packs.dev()->w;
    w.stats = nullptr;
    if (!dc.Put(cases, sizeof(DetectCase) * nCases, error) || !di.Put(idx, sizeof(std::uint32_t) * n, error) ||
        !sink.Alloc(sizeof(std::uint32_t) * n, error) || !acc.Alloc(sizeof(unsigned long long) * 16u, error) || !dw.Put(&w, sizeof w, error))
        return false;
    const auto *pw = static_cast<const DevWorld *>(dw.ptr);
    const std::uint32_t threads = 64u, blocks = static_cast<std::uint32_t>((n + threads - 1u) / threads);
    const auto *pc = static_cast<const DetectCase *>(dc.ptr);
    const auto *pi = static_cast<const std::uint32_t *>(di.ptr);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best = 0.0f;
    bool ok = true;
    for (int r = 0; r < std::max(1, reps) + 1 && ok; ++r) {   // launch 0 warms up
        cudaEventRecord(e0);
        BenchKernel<<<blocks, threads>>>(pw, pc, pi, static_cast<std::uint32_t>(n), static_cast<std::uint32_t *>(sink.ptr));
        cudaEventRecord(e1);
        ok = Check(cudaGetLastError(), "BenchKernel launch", error) && Check(cudaEventSynchronize(e1), "BenchKernel", error);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        if (r == 1 || (r > 1 && ms < best)) best = ms;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (!ok) return false;
    unsigned long long a[16] = {};
    cudaMemset(acc.ptr, 0, sizeof a);
    auto *pa = static_cast<unsigned long long *>(acc.ptr);
    BenchStatsKernel<false><<<blocks, threads>>>(w, pw, pc, pi, static_cast<std::uint32_t>(n), pa);
    BenchStatsKernel<true><<<blocks, threads>>>(w, pw, pc, pi, static_cast<std::uint32_t>(n), pa);
    if (!Check(cudaGetLastError(), "BenchStatsKernel launch", error) || !Check(cudaDeviceSynchronize(), "BenchStatsKernel", error) ||
        !acc.Get(a, sizeof a, error))
        return false;
    stats->ms = best;
    stats->calls = a[0];
    stats->cycles = a[1];
    stats->warpCycles = a[2];
    stats->nodes = a[3];
    stats->childTests = a[4];
    stats->leafHits = a[5];
    stats->boxPairs = a[6];
    stats->planePass = a[7];
    stats->raws = a[8];
    stats->contacts = a[9];
    return true;
}

}  // namespace tw::gpu
