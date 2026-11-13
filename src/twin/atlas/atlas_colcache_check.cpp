// Architect C: the lazy column cache (atlas_colcache.h) against E's full raw grid (atlas_build.h).
//   atlas_colcache_check PACK.twpack REC.twrec [budgetMB]
// Builder: E's Atlas::BuildColumnBlock (CC_COPY=1: a stand-in that copies the block out of the full grid). Checks, at FV's shape poses: (1) one reader, unbounded and small budgets (evictions,
// rebuilds): Entries / Union / EntryMembers bit for bit against the full grid; (2) 6 threads sharing one
// cache with a small budget, each its own order of the queries: the same; (3) the free test's PMU cycles
// through the cache against the full grid; build counts, resident and peak bytes, build TSC.
#define private public
#include "atlas_build.h"
#undef private
#include "atlas_colcache.h"
#include "tw_world.h"
#include "twin_pmu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <x86intrin.h>

using namespace tw;
using namespace tw::atlas;

namespace {
volatile std::size_t g_sink = 0;

// stand-in block builder: the block's slots out of E's full grid, local indices
bool CopyBlock(const Atlas &A, std::uint32_t bx, std::uint32_t bz, ColumnBlock *out) {
    const std::uint32_t bnx = (A.colNx_ + kColBlock - 1) / kColBlock, bnz = (A.colNz_ + kColBlock - 1) / kColBlock;
    if (bx >= bnx || bz >= bnz) return false;
    out->bx = bx; out->bz = bz;
    out->entries.clear(); out->unions.clear(); out->memFirst.clear(); out->members.clear(); out->overlap.clear();
    for (std::uint32_t q = 0; q < kColBlockSlots; ++q) {
        out->first[q] = static_cast<std::uint32_t>(out->entries.size());
        out->unionFirst[q] = static_cast<std::uint32_t>(out->unions.size() / 2);
        const std::uint32_t lc = q / 2, cls = q & 1u;
        const std::uint32_t ix = bx * kColBlock + lc % kColBlock, iz = bz * kColBlock + lc / kColBlock;
        if (ix >= A.colNx_ || iz >= A.colNz_) continue;
        const std::size_t s = (std::size_t(iz) * A.colNx_ + ix) * 2 + cls;
        for (std::uint32_t i = A.colFirst_[s]; i < A.colFirst_[s + 1]; ++i) {
            out->entries.push_back(A.colEntries_[i]);
            out->memFirst.push_back(static_cast<std::uint32_t>(out->members.size()));
            for (std::uint32_t m = A.entMemFirst_[i]; m < A.entMemFirst_[i + 1]; ++m) out->members.push_back(A.entMembers_[m]);
            out->overlap.push_back(A.entOverlap_[i]);
        }
        for (std::uint32_t u = A.colUnionFirst_[s]; u < A.colUnionFirst_[s + 1]; ++u) { out->unions.push_back(A.colUnion_[2 * u]); out->unions.push_back(A.colUnion_[2 * u + 1]); }
    }
    out->first[kColBlockSlots] = static_cast<std::uint32_t>(out->entries.size());
    out->unionFirst[kColBlockSlots] = static_cast<std::uint32_t>(out->unions.size() / 2);
    out->memFirst.push_back(static_cast<std::uint32_t>(out->members.size()));
    return true;
}

struct Q { std::uint32_t slot, handle; std::int32_t lo, hi; };

// one query through the cache against the full grid: entries, union, members bit for bit
bool Same(Atlas &A, ColumnCache &C, int reader, const Q &q) {
    std::uint32_t n1, n2, u1, u2;
    const ColumnEntry *e1 = A.Entries(q.slot, &n1);
    const ColumnEntry *e2 = C.Entries(reader, q.handle, &n2);
    if (n1 != n2 || (n1 && std::memcmp(e1, e2, n1 * sizeof(ColumnEntry)) != 0)) return false;
    const std::int32_t *a1 = A.Union(q.slot, &u1);
    const std::int32_t *a2 = C.Union(reader, q.handle, &u2);
    if (u1 != u2 || (u1 && std::memcmp(a1, a2, 8 * u1) != 0)) return false;
    for (std::uint32_t i = 0; i < n1; ++i) {
        std::uint32_t m1, m2;
        bool o1, o2;
        const std::uint32_t *p1 = A.EntryMembers(A.colFirst_[q.slot] + i, &m1, &o1);
        const std::uint32_t *p2 = C.EntryMembers(reader, q.handle, i, &m2, &o2);
        if (m1 != m2 || o1 != o2 || (m1 && std::memcmp(p1, p2, 4 * m1) != 0)) return false;
    }
    return true;
}
}  // namespace

// Pose-stream mode (twin_step_bench TWB_DUMP *.poses: 8 ShapePose per Detect call): the atlas built with lazy
// columns (no whole grid), the cache fed with every call's 8 shape centres (wheels: class 0, bodies: class 1);
// blocks touched, resident bytes, builds and their TSC, unbounded and under the budget.
int PoseMode(const world::WorldPackView &view, const Bvh4 &bvh, const char *packPath, const char *posePath, std::size_t budget) {
    std::string err;
    Atlas A;
    BuildOptions bo;
    bo.lazyColumns = true;
    const std::uint64_t b0 = __rdtsc();
    if (!A.Build(view, bvh, bo, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const double buildMs = double(__rdtsc() - b0) / 3.893e6;
    float x0, z0;
    std::uint32_t nx, nz;
    A.ColumnGrid(&x0, &z0, &nx, &nz);
    std::FILE *f = std::fopen(posePath, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", posePath); return 1; }
    struct SP { float r[9], t[3], radii[3]; };
    std::vector<SP> poses;
    SP sp[8];
    while (std::fread(sp, sizeof sp, 1, f) == 1) poses.insert(poses.end(), sp, sp + 8);
    std::fclose(f);
    ColumnCache::Builder builder = [&A](std::uint32_t bx, std::uint32_t bz, ColumnBlock *out) { return A.BuildColumnBlock(bx, bz, out); };
    for (int mode = 0; mode < 3; ++mode) {   // unbounded; budget; budget + prefetch along the predicted path
        const std::size_t bud = mode == 0 ? 0 : budget;
        ColumnCache::Options o;
        o.budgetBytes = bud;
        o.prefetchThread = mode == 2;
        ColumnCache C(x0, z0, nx, nz, builder, o);
        const int r = C.Register();
        std::size_t outside = 0;
        float prev[3] = {0, 0, 0};
        for (std::size_t i = 0; i < poses.size(); ++i) {
            if (i % 8 == 0) {
                C.BeginTick(r);
                if (mode == 2) {   // car centre ~ body shape 0; prefetch the blocks 8..64 calls ahead at the current displacement
                    const float *c = poses[i].t;
                    const float d[3] = {c[0] - prev[0], c[1] - prev[1], c[2] - prev[2]};
                    if (i > 0)
                        for (float k : {8.0f, 16.0f, 32.0f, 64.0f}) {
                            const float p[3] = {c[0] + k * d[0], c[1] + k * d[1], c[2] + k * d[2]};
                            C.Prefetch(C.SlotHandle(p, 1));
                        }
                    prev[0] = c[0]; prev[1] = c[1]; prev[2] = c[2];
                }
            }
            const std::uint32_t h = C.SlotHandle(poses[i].t, i % 8 >= 4 ? 0 : 1);
            if (h == 0xFFFFFFFFu) { ++outside; continue; }
            std::uint32_t n;
            (void)C.Union(r, h, &n);
        }
        C.BeginTick(r);
        const ColumnCacheStats s = C.Stats();
        std::printf("%s: %zu tris, %zu calls; atlas (lazy columns) %.0f ms; %s: %llu builds (%llu by the helper), %llu evictions, resident %.1f MB, peak %.1f MB; "
                    "build %.2f ms mean, %.2f ms max; worst call stall %.2f ms; %zu queries outside the grid\n",
                    packPath, view.tris.size(), poses.size() / 8, buildMs,
                    mode == 0 ? "unbounded" : mode == 1 ? ("budget " + std::to_string(bud >> 20) + " MB").c_str() : ("budget " + std::to_string(bud >> 20) + " MB + prefetch").c_str(),
                    (unsigned long long)s.builds, (unsigned long long)s.prefetchBuilds, (unsigned long long)s.evictions, s.residentBytes / 1e6, s.peakResidentBytes / 1e6,
                    s.builds ? double(s.buildTsc) / s.builds / 3.893e6 : 0.0, s.maxBuildTsc / 3.893e6, s.maxTickBuildTsc / 3.893e6, outside);
        std::fflush(stdout);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: atlas_colcache_check PACK REC|POSES [budgetMB]\n"); return 2; }
    const std::size_t smallBudget = (argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 4) << 20;
    world::WorldPack pack;
    std::string err;
    if (!world::LoadWorldPack(argv[1], &pack, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    world::WorldPackView view;
    if (!world::OpenWorldPack(pack, &view, &err, true)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    World w;
    w.tris.assign(view.tris.begin(), view.tris.end());
    Bvh4 bvh;
    bvh.Build(w);
    if (std::strstr(argv[2], ".poses")) return PoseMode(view, bvh, argv[1], argv[2], smallBudget);
    Atlas A;
    if (!A.Build(view, bvh, BuildOptions{}, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    Recording rec;
    if (!LoadRecording(argv[2], &rec)) { std::fprintf(stderr, "cannot load %s\n", argv[2]); return 1; }
    const bool copy = std::getenv("CC_COPY") != nullptr;
    ColumnCache::Builder builder = [&A, copy](std::uint32_t bx, std::uint32_t bz, ColumnBlock *out) {
        return copy ? CopyBlock(A, bx, bz, out) : A.BuildColumnBlock(bx, bz, out);
    };
    std::vector<Q> qs;
    {
        ColumnCache probe(A.colX0_, A.colZ0_, A.colNx_, A.colNz_, builder, ColumnCache::Options{});
        for (const Substep &st : rec.substeps)
            for (int k = 0; k < 8; ++k) {
                const Shape &sh = st.shapes[k];
                const float rmax = std::max(sh.r[0], std::max(sh.r[1], sh.r[2]));
                const std::uint32_t slot = A.ColumnSlot(sh.t, k >= 4 ? 0 : 1);
                if (slot == 0xFFFFFFFFu) continue;
                qs.push_back(Q{slot, probe.SlotHandle(sh.t, k >= 4 ? 0 : 1), std::int32_t(std::floor(double(sh.t[1] - rmax) * 1000.0)),
                               std::int32_t(std::ceil(double(sh.t[1] + rmax) * 1000.0))});
                if (probe.HandleOf(slot) != qs.back().handle) { std::fprintf(stderr, "handle mismatch\n"); return 1; }
            }
    }
    // (1) one reader: unbounded and small budgets
    for (std::size_t budget : {std::size_t(0), smallBudget}) {
        ColumnCache::Options o;
        o.budgetBytes = budget;
        ColumnCache C(A.colX0_, A.colZ0_, A.colNx_, A.colNz_, builder, o);
        const int r = C.Register();
        std::size_t bad = 0;
        for (std::size_t i = 0; i < qs.size(); ++i) {
            if (i % 8 == 0) C.BeginTick(r);
            bad += !Same(A, C, r, qs[i]);
        }
        const ColumnCacheStats s = C.Stats();
        std::printf("%s: %zu queries, 1 reader, budget %s: %zu differ; %llu builds, %llu evictions, %llu freed, resident %.2f MB, peak %.2f MB, "
                    "build %.0f k TSC mean, %.0f k max, worst per tick %.0f k\n",
                    argv[1], qs.size(), budget ? (std::to_string(budget >> 20) + " MB").c_str() : "unbounded", bad, (unsigned long long)s.builds,
                    (unsigned long long)s.evictions, (unsigned long long)s.freed, s.residentBytes / 1e6, s.peakResidentBytes / 1e6,
                    s.builds ? double(s.buildTsc) / s.builds / 1e3 : 0.0, s.maxBuildTsc / 1e3, s.maxTickBuildTsc / 1e3);
    }
    // (2) 6 threads, one shared cache, small budget, each thread its own rotation of the queries
    {
        ColumnCache::Options o;
        o.budgetBytes = smallBudget;
        ColumnCache C(A.colX0_, A.colZ0_, A.colNx_, A.colNz_, builder, o);
        std::atomic<std::size_t> bad{0}, done{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 6; ++t)
            th.emplace_back([&, t] {
                const int r = C.Register();
                std::size_t b = 0;
                const std::size_t start = qs.size() * std::size_t(t) / 6;
                for (std::size_t j = 0; j < qs.size(); ++j) {
                    if (j % 8 == 0) C.BeginTick(r);
                    b += !Same(A, C, r, qs[(start + j) % qs.size()]);
                }
                C.Idle(r);
                bad += b;
                done += qs.size();
            });
        for (auto &x : th) x.join();
        const ColumnCacheStats s = C.Stats();
        std::printf("  6 threads, budget %zu MB: %zu of %zu queries differ; %llu builds (%llu waited for another thread's build), %llu evictions, %llu freed, peak %.2f MB\n",
                    smallBudget >> 20, bad.load(), done.load(), (unsigned long long)s.builds, (unsigned long long)s.raceDuplicates, (unsigned long long)s.evictions,
                    (unsigned long long)s.freed, s.peakResidentBytes / 1e6);
    }
    // (3) free-test cost: full grid vs cache (warm, unbounded), PMU cycles per query, min of 5
    {
        ColumnCache C(A.colX0_, A.colZ0_, A.colNx_, A.colNz_, builder, ColumnCache::Options{0, 64});
        const int r = C.Register();
        C.BeginTick(r);
        for (const Q &q : qs) { std::uint32_t n; (void)C.Union(r, q.handle, &n); }
        double best[3] = {1e30, 1e30, 1e30};
        for (int rep = 0; rep < 5; ++rep)
            for (int mode = 0; mode < 3; ++mode) {
                ColumnCache::SlotRef held[8];   // mode 2: each shape keeps its resolved slot while its handle is unchanged
                std::uint32_t heldH[8];
                for (auto &h : heldH) h = 0xFFFFFFFFu;
                tw::bench::Pmu pmu;
                std::size_t hits = 0;
                const auto t0 = pmu.Read();
                for (std::size_t i = 0; i < qs.size(); ++i) {
                    const Q &q = qs[i];
                    std::uint32_t un;
                    const std::int32_t *u;
                    if (mode == 0) u = A.Union(q.slot, &un);
                    else if (mode == 1) u = C.Union(r, q.handle, &un);
                    else {
                        const int k = static_cast<int>(i % 8);
                        if (heldH[k] != q.handle) { held[k] = C.Resolve(r, q.handle); heldH[k] = q.handle; }
                        u = held[k].unions;
                        un = held[k].unionCount;
                    }
                    bool any = false;
                    for (std::uint32_t k = 0; k < un && !any; ++k) any = u[2 * k] <= q.hi && u[2 * k + 1] >= q.lo;
                    hits += any;
                }
                const auto t1 = pmu.Read();
                g_sink += hits;
                best[mode] = std::min(best[mode], double(tw::bench::Pmu::Delta(t0, t1).cycles) / std::max<std::size_t>(1, qs.size()));
            }
        std::printf("  free test: full grid %.1f, cache %.1f per query, cache with the shape's slot held while unchanged %.1f PMU cycles per query\n", best[0], best[1],
                    best[2]);
    }
    return 0;
}
