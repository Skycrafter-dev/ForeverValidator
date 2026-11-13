// atlas_check: the contact-atlas stage against FV's recorded substeps (B's *.twrec) on the matching world
// package (*.twpack). FV-free.
//
//   atlas_check PACK REC [--max N] [--reps R]
//
// Checks, per recorded (substep, shape):
//   free / T0   FV has no merged contact for the shape
//   T1          FV has exactly one, primary, the same material, P N Sep within the gate floors
//   whole call  the stage's merged contacts per shape against FV's (count, material, primary, |dP| < 5 cm,
//               N.N > 0.9: C's discrete rule) and, bitwise, warm cache = cache flushed before every call
// Then times (PMU, min of R): the decisions alone (fast path, no fallback) and the whole stage, per substep.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <x86intrin.h>

#include "atlas_stage.h"
#include "cc_stage.h"
#include "twin_pmu.h"
#include "tw_world.h"

using namespace tw;

namespace {

inline float kColumnMfix() { return atlas::kColumnM; }

void PoseOf(const Substep &s, collide::ShapePose sp[8]) {
    for (int k = 0; k < 8; ++k) {
        const Shape &sh = s.shapes[k];
        for (int c = 0; c < 3; ++c)
            for (int a = 0; a < 3; ++a) sp[k].world.r[3 * c + a] = sh.axis[c][a];
        for (int a = 0; a < 3; ++a) { sp[k].world.t[a] = sh.t[a]; sp[k].radii[a] = sh.r[a]; }
    }
}

bool SameOut(const step::SubstepContacts &a, const step::SubstepContacts &b) {
    return a.n == b.n && a.overflow == b.overflow && std::memcmp(a.c, b.c, sizeof(a.c[0]) * static_cast<std::size_t>(a.n)) == 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: atlas_check PACK REC [--max N] [--reps R] [--tries T]\n"); return 2; }
    std::size_t maxS = 0, hot = 0; int reps = 3;
    atlas::AtlasOptions opt;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--max")) maxS = std::strtoull(argv[i + 1], nullptr, 10);
        else if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--tries")) opt.maxTries = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--hot")) hot = std::strtoull(argv[i + 1], nullptr, 10);
        else if (!std::strcmp(argv[i], "--exact-fallback")) opt.exactFallback = std::atoi(argv[i + 1]) != 0;
        else if (!std::strcmp(argv[i], "--lazy")) opt.build.lazyColumns = std::atoi(argv[i + 1]) != 0;
        else if (!std::strcmp(argv[i], "--profile")) opt.profile = std::atoi(argv[i + 1]);
    }
    world::WorldPack pack;
    std::string err;
    if (!world::LoadWorldPack(argv[1], &pack, &err)) { std::fprintf(stderr, "pack: %s\n", err.c_str()); return 1; }
    world::WorldPackView view;
    if (!world::OpenWorldPack(pack, &view, &err, true)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    Recording rec;
    if (!LoadRecording(argv[2], &rec)) { std::fprintf(stderr, "cannot load %s\n", argv[2]); return 1; }
    collide::CarShapes car;
    for (int k = 0; k < 8; ++k) car.shape[k].wheel = k >= 4 ? k - 4 : -1;
    atlas::AtlasCollision st;
    if (!st.Init(view, car, &err, opt)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const atlas::Atlas &A = st.GetAtlas();
    std::printf("%s: %zu tris, %zu plates, %zu members, %zu loose (%zu dups); columns %zu, %zu entries, %zu union, %zu entry members; build %.0f ms + columns %.0f ms\n",
                rec.name.c_str(), A.stats.tris, A.stats.plates, A.stats.members, A.stats.loose, A.stats.dups, A.stats.columns,
                A.stats.columnEntries, A.stats.columnUnion, A.stats.entryMembers, A.stats.buildMs, A.stats.gridMs);
    const std::size_t N = maxS ? std::min(maxS, rec.substeps.size()) : rec.substeps.size();
    std::vector<collide::ShapePose> poses(N * 8);
    for (std::size_t i = 0; i < N; ++i) PoseOf(rec.substeps[i], &poses[8 * i]);

    // ---- correctness pass (warm cache) + flushed-cache identity
    atlas::AtlasStats stats;
    atlas::AtlasCache warm, flush;
    flush.flushEveryCall = true;
    std::size_t bad0 = 0, bad1 = 0, bad1mat = 0, discMis = 0, flushDiff = 0, shapeSubs = 0, fastSubs = 0;
    double maxdP = 0, maxdN = 0, maxdS = 0;
    step::SubstepContacts o, of;
    atlas::Decision dec[8];
    for (std::size_t i = 0; i < N; ++i) {
        const Substep &s = rec.substeps[i];
        st.DetectShapes(&poses[8 * i], o, &warm, &stats, dec);
        flush.Flush();
        st.DetectShapes(&poses[8 * i], of, &flush);
        flushDiff += !SameOut(o, of);
        bool fast = true;
        for (int k = 0; k < 8; ++k) {
            if (dec[k] == atlas::kFallback) fast = false;
            std::vector<const FvContact *> F;
            for (std::uint32_t q = s.firstMerged; q < s.firstMerged + s.mergedCount; ++q) if (rec.merged[q].shape == k) F.push_back(&rec.merged[q]);
            std::vector<const model::RespContact *> W;
            for (int q = 0; q < o.n; ++q) if (o.c[q].shape == k) W.push_back(&o.c[q]);
            if ((dec[k] == atlas::kFree || dec[k] == atlas::kT0) && !F.empty()) {
                if (bad0 < 8) std::printf("  T0/free with FV contact: substep %zu shape %d (dec %d, FV %zu)\n", i, k, int(dec[k]), F.size());
                ++bad0;
            }
            if (dec[k] == atlas::kT1) {
                bool ok = F.size() == 1 && F[0]->primary && W.size() == 1;
                if (ok) {
                    const FvContact &f = *F[0];
                    const model::RespContact &w = *W[0];
                    double dP = 0, dN = 0, dS = 0, pP = 0, pS = 0;
                    for (int a = 0; a < 3; ++a) {
                        dP += (double(f.P[a]) - w.P[a]) * (double(f.P[a]) - w.P[a]); pP += double(f.P[a]) * f.P[a];
                        dN += (double(f.N[a]) - w.N[a]) * (double(f.N[a]) - w.N[a]);
                        dS += (double(f.Sep[a]) - w.Sep[a]) * (double(f.Sep[a]) - w.Sep[a]); pS += double(f.Sep[a]) * f.Sep[a];
                    }
                    dP = std::sqrt(dP); dN = std::sqrt(dN); dS = std::sqrt(dS);
                    maxdP = std::max(maxdP, dP); maxdN = std::max(maxdN, dN); maxdS = std::max(maxdS, dS);
                    if (dP > 2e-3 + 1e-5 * std::sqrt(pP) || dN > 1e-3 || dS > 2e-4 + 1e-3 * std::sqrt(pS)) ok = false;
                    if (f.material != w.material) { ++bad1mat; ok = false; }
                }
                if (!ok) {
                    if (bad1 < 8) std::printf("  T1 mismatch: substep %zu shape %d (FV %zu)\n", i, k, F.size());
                    ++bad1;
                }
            }
            if (F.empty() && W.empty()) continue;
            ++shapeSubs;
            bool disc = F.size() == W.size();
            for (std::size_t q = 0; disc && q < F.size(); ++q) {
                const double dx = double(F[q]->P[0]) - W[q]->P[0], dy = double(F[q]->P[1]) - W[q]->P[1], dz = double(F[q]->P[2]) - W[q]->P[2];
                const double nd = double(F[q]->N[0]) * W[q]->N[0] + double(F[q]->N[1]) * W[q]->N[1] + double(F[q]->N[2]) * W[q]->N[2];
                disc = F[q]->material == W[q]->material && (F[q]->primary != 0) == ((W[q]->flags & model::kRespPrimary) != 0) &&
                       dx * dx + dy * dy + dz * dz < 0.05 * 0.05 && nd > 0.9;
            }
            discMis += !disc;
        }
        fastSubs += fast;
    }
    const double tot = double(N) * 8.0;
    std::printf("decisions per shape-substep: free %.2f %%, T0 %.2f %%, T1 %.2f %%, fallback %.2f %%; substeps with no fallback %.2f %%\n",
                100.0 * stats.shapes[atlas::kFree] / tot, 100.0 * stats.shapes[atlas::kT0] / tot, 100.0 * stats.shapes[atlas::kT1] / tot,
                100.0 * stats.shapes[atlas::kFallback] / tot, 100.0 * fastSubs / N);
    const char *wn[atlas::kFbCount] = {"loose", "no tile", "foreign", "reach tie", "member", "face tie", "proof", "tries"};
    std::printf("fallback reasons:");
    for (int w = 0; w < atlas::kFbCount; ++w) std::printf(" %s %.2f%%", wn[w], 100.0 * stats.why[w] / tot);
    std::printf("\ntries per decided shape %.2f, plane tests %.2f per substep, column entries past the search %.1f per shape, field tests %.3f per substep\n", double(stats.tries) / tot,
                double(stats.planeTestsRun) / N, double(stats.colScanned) / tot, double(stats.fieldTests) / N);
    std::printf("checks: T0/free with FV contact %zu, T1 mismatches %zu (material %zu), T1 max |dP| %.2e |dN| %.2e |dSep| %.2e\n", bad0, bad1, bad1mat,
                maxdP, maxdN, maxdS);
    std::printf("        merged sets vs FV (discrete): %zu of %zu (substep, shape) differ; warm vs flushed-every-call: %zu substeps differ\n",
                discMis, shapeSubs, flushDiff);
    std::printf("tiles built %zu, atlas %.1f MB\n", A.TilesBuilt(), A.Bytes() / 1e6);

    // ---- whole-tick proof: on every tick TickClearShapes (from the tick's first recorded pose) declares clear, FV has no
    // contact in any of the tick's substeps
    {
        std::size_t ticks = 0, clear = 0, bad = 0;
        std::size_t i = 0;
        while (i < N) {
            std::size_t j = i;
            while (j < N && rec.substeps[j].tick == rec.substeps[i].tick) ++j;
            const Substep &s0 = rec.substeps[i];
            const bool c = st.TickClearShapes(&poses[8 * i], s0.carPos, s0.vel, s0.angVel, 0.01f);
            bool fv = false;
            for (std::size_t q = i; q < j; ++q) fv = fv || rec.substeps[q].mergedCount > 0;
            ++ticks; clear += c; bad += c && fv;
            i = j;
        }
        std::printf("TickClear: %zu of %zu ticks proved contact-free (%.1f %%), %zu of them with an FV contact\n", clear, ticks, 100.0 * clear / ticks, bad);
    }

    // ---- superblocks the path touches: BuildColumnBlock against the whole grid, slot by slot; build time
    if (std::getenv("ATLAS_BLOCKS") && A.HasColumnGrid()) {   // needs --lazy 0 (the whole grid as the reference)
        float x0, z0; std::uint32_t nx, nz;
        A.ColumnGrid(&x0, &z0, &nx, &nz);
        std::vector<std::uint64_t> blocks;
        for (std::size_t i = 0; i < N; ++i)
            for (int k = 0; k < 8; ++k) {
                const float *t = poses[8 * i + std::size_t(k)].world.t;
                const float fx = (t[0] - x0) / kColumnMfix(), fz = (t[2] - z0) / kColumnMfix();
                if (fx < 0 || fz < 0) continue;
                const std::uint32_t ix = std::uint32_t(fx), iz = std::uint32_t(fz);
                if (ix >= nx || iz >= nz) continue;
                blocks.push_back((std::uint64_t(iz / 16) << 32) | (ix / 16));
            }
        std::sort(blocks.begin(), blocks.end());
        blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
        std::size_t slotDiff = 0, slots = 0;
        double ms = 0, worst = 0;
        for (std::uint64_t b : blocks) {
            const std::uint32_t bx = std::uint32_t(b & 0xFFFFFFFFu), bz = std::uint32_t(b >> 32);
            atlas::ColumnBlock cb;
            const auto t0 = std::chrono::steady_clock::now();
            A.BuildColumnBlock(bx, bz, &cb);
            const double m = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ms += m; worst = std::max(worst, m);
            for (std::uint32_t lz = 0; lz < 16; ++lz)
                for (std::uint32_t lx = 0; lx < 16; ++lx) {
                    const std::uint32_t ix = bx * 16 + lx, iz = bz * 16 + lz;
                    if (ix >= nx || iz >= nz) continue;
                    for (int cls = 0; cls < 2; ++cls) {
                        const float p[3] = {x0 + (float(ix) + 0.5f) * kColumnMfix(), 0.0f, z0 + (float(iz) + 0.5f) * kColumnMfix()};
                        const std::uint32_t slot = A.ColumnSlot(p, cls);
                        const std::uint32_t q = (lz * 16 + lx) * 2 + std::uint32_t(cls);
                        std::uint32_t ne = 0, nu = 0;
                        const atlas::ColumnEntry *e = A.Entries(slot, &ne);
                        const std::int32_t *u = A.Union(slot, &nu);
                        bool same = ne == cb.first[q + 1] - cb.first[q] && nu == cb.unionFirst[q + 1] - cb.unionFirst[q];
                        for (std::uint32_t j = 0; same && j < ne; ++j) {
                            same = std::memcmp(&e[j], &cb.entries[cb.first[q] + j], sizeof(atlas::ColumnEntry)) == 0;
                            std::uint32_t nm = 0; bool ov = false;
                            const std::uint32_t *m = A.EntryMembers(A.EntryIndex(e + j), &nm, &ov);
                            const std::uint32_t bj = cb.first[q] + j;
                            same = same && nm == cb.memFirst[bj + 1] - cb.memFirst[bj] && ov == (cb.overlap[bj] != 0) &&
                                   std::memcmp(m, cb.members.data() + cb.memFirst[bj], 4 * nm) == 0;
                        }
                        for (std::uint32_t j = 0; same && j < 2 * nu; ++j) same = u[j] == cb.unions[2 * cb.unionFirst[q] + j];
                        ++slots;
                        slotDiff += !same;
                    }
                }
        }
        std::printf("superblocks touched %zu: %zu of %zu slots differ from the whole grid; build %.2f ms mean, %.2f ms worst\n", blocks.size(),
                    slotDiff, slots, ms / std::max<std::size_t>(1, blocks.size()), worst);
    }

    // ---- timing: decisions only (the fast path), the whole stage; B's stage for comparison.
    // --hot H: the timed loops repeat the first H substeps (a small working set: compute cost without misses)
    std::vector<std::size_t> order(N);
    for (std::size_t i = 0; i < N; ++i) order[i] = hot ? i % hot : i;
    collide_c::CcCollision cst;
    {
        collide_c::StageOptions copt;
        if (!collide_c::ParseStageSwitch("c", &copt) || !cst.Init(view, car, &err, copt)) { std::fprintf(stderr, "C init: %s\n", err.c_str()); return 1; }
    }
    double bestDec = 1e30, bestAll = 1e30, bestB = 1e30, bestC = 1e30, insDec = 0;
    float sink = 0;
    for (int rep = 0; rep < reps; ++rep) {
        {
            tw::bench::Pmu pmu;
            const auto a = pmu.Read();
            model::RespContact c;
            for (std::size_t ii = 0; ii < N; ++ii)
                for (int k = 0; k < 8; ++k) {
                    const std::size_t i = order[ii];
                    atlas::FallbackWhy why;
                    const atlas::Decision d = st.Decide(poses[8 * i + static_cast<std::size_t>(k)], k, &c, warm, nullptr, &why);
                    sink += float(d) + (d == atlas::kT1 ? c.P[0] : 0.0f);
                }
            const auto b = pmu.Read();
            const auto dl = tw::bench::Pmu::Delta(a, b);
            if (double(dl.cycles) < bestDec) { bestDec = double(dl.cycles); insDec = double(dl.instructions); }
        }
        {
            tw::bench::Pmu pmu;
            const auto a = pmu.Read();
            for (std::size_t ii = 0; ii < N; ++ii) { st.DetectShapes(&poses[8 * order[ii]], o, &warm); sink += float(o.n); }
            const auto b = pmu.Read();
            bestAll = std::min(bestAll, double(tw::bench::Pmu::Delta(a, b).cycles));
        }
        {   // C's noise stage (the product default) on the same calls
            collide_c::CcCache cc;
            tw::bench::Pmu pmu;
            const auto a = pmu.Read();
            for (std::size_t ii = 0; ii < N; ++ii) { cst.DetectShapes(&poses[8 * order[ii]], o, &cc); sink += float(o.n); }
            const auto b = pmu.Read();
            bestC = std::min(bestC, double(tw::bench::Pmu::Delta(a, b).cycles));
        }
        {
            collide::ExactCache bc;
            tw::bench::Pmu pmu;
            const auto a = pmu.Read();
            for (std::size_t ii = 0; ii < N; ++ii) { st.Exact().DetectShapes(&poses[8 * order[ii]], o, nullptr, &bc); sink += float(o.n); }
            const auto b = pmu.Read();
            bestB = std::min(bestB, double(tw::bench::Pmu::Delta(a, b).cycles));
        }
    }
    std::printf("cycles per substep: atlas decisions (8 shapes, fast path) %.0f (%.0f instructions); whole atlas stage %.0f; B's exact stage %.0f; C's noise stage %.0f  [sink %g]\n",
                bestDec / N, insDec / N, bestAll / N, bestB / N, bestC / N, double(sink));
    return 0;
}
