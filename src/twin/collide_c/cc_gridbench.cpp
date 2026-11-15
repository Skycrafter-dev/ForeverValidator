// cc_gridbench: CellGrid (block-cell index) against B's BVH4 on the world package's triangles.
//   cc_gridbench PACK REC [PACK REC ...]
// Query boxes come from the recorded FV substeps:
//   (shape) every shape's world AABB, as B's reference DetectShapes builds it;
//   (memo)  B's union-memo refills (tw_collide.cpp: union of the 8 shape boxes, extended along the motion by 6x
//           the last step, inflated by 0.25 m, refilled when the union leaves it).
// Exactness: every query's triangle set equals the BVH4's. Cycles: wall time x measured core clock, best of reps.
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cc_grid.h"
#include "tw_accel.h"
#include "tw_world.h"
#include "tw_world_pack.h"

using Clock = std::chrono::steady_clock;
using tw::Box3;

namespace {
double CoreGHz() {
    std::uint64_t x = 0;
    const auto t0 = Clock::now();
    for (int i = 0; i < 25000000; ++i) {
        asm volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
    }
    return 2e8 / std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

Box3 ShapeBox(const tw::Shape &sh) {
    Box3 q;
    for (int a = 0; a < 3; ++a) {
        const float c = sh.t[a];
        const float h = std::fabs(sh.axis[0][a]) * sh.r[0] + std::fabs(sh.axis[1][a]) * sh.r[1] + std::fabs(sh.axis[2][a]) * sh.r[2];
        const float eps = 1e-3f + 1e-6f * std::fabs(c);
        q.lo[a] = c - h - eps; q.hi[a] = c + h + eps;
    }
    return q;
}
}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: cc_gridbench PACK REC [PACK REC ...]\n"); return 2; }
    const int reps = std::getenv("CC_REPS") ? std::atoi(std::getenv("CC_REPS")) : 5;
    const double ghz = CoreGHz();
    for (int a = 1; a + 1 < argc; a += 2) {
        tw::world::WorldPack pack;
        std::string err;
        if (!tw::world::LoadWorldPack(argv[a], &pack, &err)) { std::fprintf(stderr, "%s: %s\n", argv[a], err.c_str()); return 1; }
        tw::world::WorldPackView v;
        if (!tw::world::OpenWorldPack(pack, &v, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        tw::Recording rec;
        if (!tw::LoadRecording(argv[a + 1], &rec)) { std::fprintf(stderr, "cannot load %s\n", argv[a + 1]); return 1; }
        tw::World w;
        tw::world::MakeBenchWorld(v, &w);
        auto b0 = Clock::now();
        tw::Bvh4 bvh;
        bvh.Build(w);
        const double bvhMs = std::chrono::duration<double, std::milli>(Clock::now() - b0).count();
        b0 = Clock::now();
        cc::CellGrid grid;
        cc::CellGrid::Options opt;
        if (std::getenv("CC_MAXREFS")) opt.maxCellRefs = std::atoi(std::getenv("CC_MAXREFS"));
        if (std::getenv("CC_SUBMIN")) opt.subgridMinTris = static_cast<std::uint32_t>(std::atoi(std::getenv("CC_SUBMIN")));
        if (std::getenv("CC_SUBCELL")) opt.subCell = static_cast<float>(std::atof(std::getenv("CC_SUBCELL")));
        grid.Build(v, opt);
        const double gridMs = std::chrono::duration<double, std::milli>(Clock::now() - b0).count();
        std::printf("## %s: %zu tris, %zu instances, %zu models | grid: %zu instances in %d x %d x %d cells (%zu refs), %zu models (%zu with sub-grids); "
                    "%zu sub-cell registrations; huge: %zu instances, %zu tris | bytes: grid %.2f MB (per-map cell + instance tables %.1f KB), BVH4 %.2f MB | build: grid %.0f ms, BVH4 %.0f ms\n",
                    rec.name.c_str(), v.tris.size(), v.instances.size(), v.models.size(), grid.gridInstances, grid.dim[0], grid.dim[1], grid.dim[2], grid.cellRefs,
                    grid.models, grid.subgridModels, grid.subRegs, grid.hugeInstances, grid.hugeTris, grid.Bytes() / 1e6, grid.HotBytesPerMap() / 1e3, bvh.Bytes() / 1e6, gridMs, bvhMs);
        // query boxes
        std::vector<Box3> shapeQ, memoQ, bigQ;
        Box3 memo = tw::EmptyBox();
        bool have = false;
        float last[3] = {0, 0, 0};
        for (const tw::Substep &s : rec.substeps) {
            Box3 u = tw::EmptyBox();
            for (int k = 0; k < tw::kShapes; ++k) { const Box3 q = ShapeBox(s.shapes[k]); shapeQ.push_back(q); u = tw::Union(u, q); }
            float c[3];
            for (int i = 0; i < 3; ++i) c[i] = 0.5f * (u.lo[i] + u.hi[i]);
            if (!have || !tw::Contains(memo, u)) {
                Box3 big = u;
                if (have)
                    for (int i = 0; i < 3; ++i) {
                        const float d = std::fmax(-4.0f, std::fmin(4.0f, (c[i] - last[i]) * 6.0f));
                        if (d > 0.0f) big.hi[i] += d; else big.lo[i] += d;
                    }
                memo = tw::Inflate(big, 0.25f);
                memoQ.push_back(memo);
                bigQ.push_back(tw::Inflate(memo, 4.0f));  // C's per-candidate query: horizon of several poses at speed
                have = true;
            }
            for (int i = 0; i < 3; ++i) last[i] = c[i];
        }
        // exactness: identical triangle sets
        auto check = [&](const std::vector<Box3> &qs, const char *name) {
            std::size_t bad = 0, missing = 0, extra = 0, total = 0;
            std::vector<std::uint32_t> A, B;
            for (const Box3 &q : qs) {
                A.clear(); B.clear();
                bvh.Query(q, [&](std::uint32_t t) { A.push_back(t); });
                grid.Query(q, [&](std::uint32_t t) { B.push_back(t); });
                std::sort(A.begin(), A.end());
                std::sort(B.begin(), B.end());
                total += A.size();
                if (A != B) {
                    ++bad;
                    std::vector<std::uint32_t> d;
                    std::set_difference(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(d));
                    missing += d.size();
                    d.clear();
                    std::set_difference(B.begin(), B.end(), A.begin(), A.end(), std::back_inserter(d));
                    extra += d.size();
                }
            }
            std::printf("  %s queries: %zu, triangles reported %.1f per query; sets differing from the BVH4: %zu (missing %zu, extra %zu)\n", name, qs.size(),
                        double(total) / std::max<std::size_t>(1, qs.size()), bad, missing, extra);
        };
        check(shapeQ, "shape");
        check(memoQ, "memo ");
        check(bigQ, "large");
        // timing
        auto timeIt = [&](const std::vector<Box3> &qs, auto &&query) {
            double best = 1e30;
            volatile std::uint64_t sinkv = 0;
            std::uint64_t sink = 0;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = Clock::now();
                for (const Box3 &q : qs) query(q, [&](std::uint32_t t) { sink += t; });
                best = std::min(best, std::chrono::duration<double, std::nano>(Clock::now() - t0).count() * ghz / double(qs.size()));
            }
            sinkv = sink;
            return best;
        };
        const double bs = timeIt(shapeQ, [&](const Box3 &q, auto &&f) { bvh.Query(q, f); });
        const double gs = timeIt(shapeQ, [&](const Box3 &q, auto &&f) { grid.Query(q, f); });
        const double bm = timeIt(memoQ, [&](const Box3 &q, auto &&f) { bvh.Query(q, f); });
        const double gm = timeIt(memoQ, [&](const Box3 &q, auto &&f) { grid.Query(q, f); });
        const double bb = timeIt(bigQ, [&](const Box3 &q, auto &&f) { bvh.Query(q, f); });
        const double gb = timeIt(bigQ, [&](const Box3 &q, auto &&f) { grid.Query(q, f); });
#if CC_GRID_STATS
        grid.qCells = grid.qInstances = grid.qBlocks = grid.qHits = 0;
        for (const Box3 &q : memoQ) grid.Query(q, [&](std::uint32_t) {});
        const double nq = double(std::max<std::size_t>(1, memoQ.size()));
        std::printf("  memo query work: cells %.1f, instances %.1f, 16-lane blocks %.1f, hits %.1f\n", grid.qCells / nq, grid.qInstances / nq, grid.qBlocks / nq, grid.qHits / nq);
        grid.qCells = grid.qInstances = grid.qBlocks = grid.qHits = 0;
        for (const Box3 &q : shapeQ) grid.Query(q, [&](std::uint32_t) {});
        const double ns = double(std::max<std::size_t>(1, shapeQ.size()));
        std::printf("  shape query work: cells %.1f, instances %.1f, 16-lane blocks %.1f, hits %.1f\n", grid.qCells / ns, grid.qInstances / ns, grid.qBlocks / ns, grid.qHits / ns);
#endif
        std::printf("  cycles per query: shape box BVH4 %.0f / grid %.0f | memo refill BVH4 %.0f / grid %.0f | large (memo + 4 m) BVH4 %.0f / grid %.0f (core %.2f GHz)\n", bs, gs, bm,
                    gm, bb, gb, ghz);
        grid.skipHuge = true;
        const double gs2 = timeIt(shapeQ, [&](const Box3 &q, auto &&f) { grid.Query(q, f); });
        const double gm2 = timeIt(memoQ, [&](const Box3 &q, auto &&f) { grid.Query(q, f); });
        grid.skipHuge = false; grid.skipGrid = true;
        const double gs3 = timeIt(shapeQ, [&](const Box3 &q, auto &&f) { grid.Query(q, f); });
        const double gm3 = timeIt(memoQ, [&](const Box3 &q, auto &&f) { grid.Query(q, f); });
        grid.skipGrid = false;
        std::printf("  split: grid part only %.0f / %.0f, huge BVH part only %.0f / %.0f (shape / memo)\n", gs2, gm2, gs3, gm3);
        {  // the same memo boxes, each repeated 64 times in a row: cache-warm cost
            std::vector<Box3> rep;
            for (std::size_t i = 0; i < memoQ.size(); i += std::max<std::size_t>(1, memoQ.size() / 64))
                for (int r = 0; r < 64; ++r) rep.push_back(memoQ[i]);
            const double bw = timeIt(rep, [&](const Box3 &q, auto &&f) { bvh.Query(q, f); });
            const double gw = timeIt(rep, [&](const Box3 &q, auto &&f) { grid.Query(q, f); });
            std::printf("  warm (each memo box repeated 64x): BVH4 %.0f / grid %.0f cycles per query\n", bw, gw);
        }
        std::fflush(stdout);
    }
    return 0;
}
