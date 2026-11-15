// cc_bench: architect C collision prototype on B's recorded FV substep streams (tw_record .twrec).
//
//   cc_bench REC...        (stats build: counters, hot bytes, comparisons with FV and with the triangle twin)
//   cc_bench_lean REC...   (no counters: cycles only)
// env: CC_L (query horizon, m), CC_PROMOTE (m), CC_ANG / CC_DIST (patch tolerances), CC_MAXMEM,
//      CC_REPS (timed repetitions), CC_EXC (print N twin-vs-reference differences), CC_MODE (fv,policy)
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cc_core.h"
#include "tw_accel.h"

using namespace cc;
using Clock = std::chrono::steady_clock;

namespace {

double CoreGHz() {  // a chain of dependent adds runs at one add per cycle (B's method)
    std::uint64_t x = 0;
    const auto t0 = Clock::now();
    for (int i = 0; i < 25000000; ++i) {
        asm volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
    }
    return 2e8 / std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

float EnvF(const char *n, float d) { const char *v = std::getenv(n); return v ? static_cast<float>(std::atof(v)) : d; }

// Pose extrapolated like the harness probe (copied from B's tw_voxbench so both predictors probe the
// same poses): constant v and w, gravity -30 m/s^2 on y; car basis rotated about w by |w| t.
void Extrapolate(const tw::Substep &s, float tau, tw::Substep *o) {
    *o = s;
    const double g = -30.0;
    const float *R = s.carRot;
    float Rn[9];
    const double wx = s.angVel[0], wy = s.angVel[1], wz = s.angVel[2], w = std::sqrt(wx * wx + wy * wy + wz * wz);
    for (int c = 0; c < 3; ++c) {
        double v[3] = {R[3 * c], R[3 * c + 1], R[3 * c + 2]};
        if (w * tau > 1e-9) {
            const double ux = wx / w, uy = wy / w, uz = wz / w, a = w * tau, cs = std::cos(a), sn = std::sin(a);
            const double dot = ux * v[0] + uy * v[1] + uz * v[2];
            const double cx = uy * v[2] - uz * v[1], cy = uz * v[0] - ux * v[2], cz = ux * v[1] - uy * v[0];
            const double n0 = v[0] * cs + cx * sn + ux * dot * (1 - cs), n1 = v[1] * cs + cy * sn + uy * dot * (1 - cs), n2 = v[2] * cs + cz * sn + uz * dot * (1 - cs);
            v[0] = n0; v[1] = n1; v[2] = n2;
        }
        for (int i = 0; i < 3; ++i) Rn[3 * c + i] = static_cast<float>(v[i]);
    }
    float p[3];
    for (int i = 0; i < 3; ++i) p[i] = static_cast<float>(s.carPos[i] + s.vel[i] * tau + (i == 1 ? 0.5 * g * tau * tau : 0.0));
    for (int i = 0; i < 3; ++i) o->carPos[i] = p[i];
    for (int i = 0; i < 9; ++i) o->carRot[i] = Rn[i];
    for (int k = 0; k < tw::kShapes; ++k) {
        const tw::Shape &sh = s.shapes[k];
        float lt[3], la[3][3];
        for (int j = 0; j < 3; ++j) {
            lt[j] = R[3 * j] * (sh.t[0] - s.carPos[0]) + R[3 * j + 1] * (sh.t[1] - s.carPos[1]) + R[3 * j + 2] * (sh.t[2] - s.carPos[2]);
            for (int a = 0; a < 3; ++a) la[a][j] = R[3 * j] * sh.axis[a][0] + R[3 * j + 1] * sh.axis[a][1] + R[3 * j + 2] * sh.axis[a][2];
        }
        tw::Shape &d = o->shapes[k];
        for (int i = 0; i < 3; ++i) {
            d.t[i] = p[i] + Rn[i] * lt[0] + Rn[3 + i] * lt[1] + Rn[6 + i] * lt[2];
            for (int a = 0; a < 3; ++a) d.axis[a][i] = Rn[i] * la[a][0] + Rn[3 + i] * la[a][1] + Rn[6 + i] * la[a][2];
        }
    }
}

// Reference: the triangle twin (BVH4 over every world triangle, FV's exact test, FV's merge).
struct Reference {
    const tw::World *w;
    tw::Bvh4 bvh;
    std::vector<Emitted> raw, merged[kShapes];
    void Step(const tw::Substep &s) {
        for (int k = 0; k < kShapes; ++k) {
            tw::ShapeMap sm;
            tw::MakeShapeMap(s.shapes[k], &sm);
            const Box3 b = tw::Inflate(sm.box, 1e-3f);
            raw.clear();
            bvh.Query(b, [&](std::uint32_t t) {
                tw::Contact c;
                if (tw::ExactTest(sm, w->tris[t], &c) != tw::kNone)
                    raw.push_back(Emitted{c, t, static_cast<std::uint8_t>(w->tris[t].meta & tw::kMetaMaterialMask), static_cast<std::uint8_t>(c.kind == tw::kFace)});
            });
            bool prim = false, feat = false;
            for (const Emitted &c : raw) { prim |= c.primary != 0; feat |= c.primary == 0; }
            if (prim && feat) MergeShape(raw, &merged[k]); else merged[k] = raw;
        }
    }
    void KOf(KSet out[kShapes]) const {
        for (int k = 0; k < kShapes; ++k) { out[k] = KSet{}; for (const Emitted &e : merged[k]) KAdd(out[k], e.material); }
    }
};

double Pct(std::vector<float> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * v.size()))];
}

// B's comparison against FV's merged per-shape contacts (m2-broadphase.md §6 step 5).
struct FvCompare {
    std::size_t shapeSubs = 0, countEq = 0, fvTotal = 0, twTotal = 0, matched = 0, matEq = 0, primEq = 0, exceptions = 0;
    std::vector<float> dP, dAng, dSep;
    void Add(const tw::Recording &rec, const tw::Substep &s, int sh, const std::vector<Emitted> &W) {
        std::vector<const tw::FvContact *> F;
        for (std::uint32_t c = s.firstMerged; c < s.firstMerged + s.mergedCount; ++c)
            if (rec.merged[c].shape == sh) F.push_back(&rec.merged[c]);
        if (F.empty() && W.empty()) return;
        ++shapeSubs; fvTotal += F.size(); twTotal += W.size();
        if (F.size() == W.size()) ++countEq;
        std::vector<bool> used(W.size(), false);
        bool exc = F.size() != W.size();
        for (const tw::FvContact *f : F) {
            int best = -1; float bd = 1e30f;
            for (std::size_t k = 0; k < W.size(); ++k) {
                if (used[k]) continue;
                const float *P = W[k].c.P, *N = W[k].c.N, *S = W[k].c.Sep;
                const float d = std::sqrt((P[0] - f->P[0]) * (P[0] - f->P[0]) + (P[1] - f->P[1]) * (P[1] - f->P[1]) + (P[2] - f->P[2]) * (P[2] - f->P[2]));
                const float dn = 1.0f - (N[0] * f->N[0] + N[1] * f->N[1] + N[2] * f->N[2]);
                const float ds = std::fabs(std::sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]) - std::sqrt(f->Sep[0] * f->Sep[0] + f->Sep[1] * f->Sep[1] + f->Sep[2] * f->Sep[2]));
                const float cost = d + dn + ds + (W[k].material != f->material ? 0.01f : 0.0f) + (W[k].primary != f->primary ? 0.01f : 0.0f);
                if (d < 0.05f && cost < bd) { bd = cost; best = static_cast<int>(k); }
            }
            if (best < 0) continue;
            used[best] = true; ++matched;
            const Emitted &w = W[best];
            const float *P = w.c.P, *N = w.c.N, *S = w.c.Sep;
            const float d = std::sqrt((P[0] - f->P[0]) * (P[0] - f->P[0]) + (P[1] - f->P[1]) * (P[1] - f->P[1]) + (P[2] - f->P[2]) * (P[2] - f->P[2]));
            const float dot = std::fmax(-1.0f, std::fmin(1.0f, N[0] * f->N[0] + N[1] * f->N[1] + N[2] * f->N[2]));
            const float ang = std::acos(dot) * 57.29578f;
            const float ds = std::sqrt((S[0] - f->Sep[0]) * (S[0] - f->Sep[0]) + (S[1] - f->Sep[1]) * (S[1] - f->Sep[1]) + (S[2] - f->Sep[2]) * (S[2] - f->Sep[2]));
            dP.push_back(d); dAng.push_back(ang); dSep.push_back(ds);
            matEq += w.material == f->material; primEq += w.primary == f->primary;
            if (d > 1e-3f || ang > 0.5f || ds > 1e-4f || w.material != f->material || w.primary != f->primary) exc = true;
        }
        exceptions += exc;
    }
    void Print(const char *what) const {
        std::printf("  %s vs FV merged: (substep,shape) with contacts %zu, count equal %.3f%%, exceptions %zu (%.3f%%); contacts FV %zu twin %zu matched %zu; "
                    "material eq %.3f%%, FACE/feature eq %.3f%%; |dP| p99 %.2e max %.2e m, normal p99 %.3f max %.3f deg, |dSep| p99 %.2e max %.2e m\n",
                    what, shapeSubs, 100.0 * countEq / std::max<std::size_t>(1, shapeSubs), exceptions, 100.0 * exceptions / std::max<std::size_t>(1, shapeSubs),
                    fvTotal, twTotal, matched, 100.0 * matEq / std::max<std::size_t>(1, matched), 100.0 * primEq / std::max<std::size_t>(1, matched),
                    Pct(dP, 0.99), Pct(dP, 1.0), Pct(dAng, 0.99), Pct(dAng, 1.0), Pct(dSep, 0.99), Pct(dSep, 1.0));
    }
};

// Twin vs triangle twin, per (pose, shape): same multiset of contacts within float noise?
struct TwinCompare {
    std::size_t shapePoses = 0, withContacts = 0, differ = 0, printed = 0, maxPrint = 0;
    bool Same(const std::vector<Emitted> &A, const std::vector<Emitted> &B) {
        if (A.size() != B.size()) return false;
        std::vector<bool> used(B.size(), false);
        for (const Emitted &a : A) {
            bool ok = false;
            for (std::size_t k = 0; k < B.size() && !ok; ++k) {
                if (used[k] || B[k].material != a.material || B[k].primary != a.primary) continue;
                float d = 0, dn = 0;
                for (int i = 0; i < 3; ++i) { d = std::max(d, std::fabs(B[k].c.P[i] - a.c.P[i])); dn = std::max(dn, std::fabs(B[k].c.N[i] - a.c.N[i])); }
                float ds = 0;
                for (int i = 0; i < 3; ++i) ds = std::max(ds, std::fabs(B[k].c.Sep[i] - a.c.Sep[i]));
                if (d < 1e-3f && dn < 1e-3f && ds < 1e-4f) { used[k] = true; ok = true; }
            }
            if (!ok) return false;
        }
        return true;
    }
    void Add(std::size_t pose, int sh, const std::vector<Emitted> &mine, const std::vector<Emitted> &ref) {
        ++shapePoses;
        if (!mine.empty() || !ref.empty()) ++withContacts;
        if (Same(mine, ref)) return;
        ++differ;
        if (printed < maxPrint) {
            ++printed;
            std::printf("    DIFF pose %zu shape %d: twin %zu, reference %zu\n", pose, sh, mine.size(), ref.size());
            for (const Emitted &e : mine)
                std::printf("      C   %s tri %u mat %u P(%.5f %.5f %.5f) N(%.4f %.4f %.4f) |Sep| %.6f\n", e.primary ? "FACE" : "feat", e.tri, e.material, e.c.P[0], e.c.P[1], e.c.P[2], e.c.N[0], e.c.N[1], e.c.N[2],
                            std::sqrt(e.c.Sep[0] * e.c.Sep[0] + e.c.Sep[1] * e.c.Sep[1] + e.c.Sep[2] * e.c.Sep[2]));
            for (const Emitted &e : ref)
                std::printf("      ref %s tri %u mat %u P(%.5f %.5f %.5f) N(%.4f %.4f %.4f) |Sep| %.6f\n", e.primary ? "FACE" : "feat", e.tri, e.material, e.c.P[0], e.c.P[1], e.c.P[2], e.c.N[0], e.c.N[1], e.c.N[2],
                            std::sqrt(e.c.Sep[0] * e.c.Sep[0] + e.c.Sep[1] * e.c.Sep[1] + e.c.Sep[2] * e.c.Sep[2]));
        }
    }
};

struct Ticks {
    std::vector<std::size_t> first;
    std::vector<int> n;
    explicit Ticks(const tw::Recording &rec) {
        for (std::size_t i = 0; i < rec.substeps.size(); ++i) {
            if (i == 0 || rec.substeps[i].tick != rec.substeps[i - 1].tick) { first.push_back(i); n.push_back(0); }
            ++n.back();
        }
    }
};

// Odometer to keep in hand at the tick's first pose so that both probe poses (dt/2, dt) stay inside the
// query horizon: translation |v| dt plus rotation |w| dt times (shape offset + ||M||_F) <= 4 m, +10 %.
float Reserve(const tw::Substep &s) {
    const float v = std::sqrt(s.vel[0] * s.vel[0] + s.vel[1] * s.vel[1] + s.vel[2] * s.vel[2]);
    const float w = std::sqrt(s.angVel[0] * s.angVel[0] + s.angVel[1] * s.angVel[1] + s.angVel[2] * s.angVel[2]);
    return EnvF("CC_RESERVE", 1.0f) * ((v + 4.0f * w) * 0.011f + 0.02f);
}

void Configure(CarCache &c, const PatchWorld &pw, const tw::World &w) {
    c.pw = &pw; c.world = &w;
    c.L = EnvF("CC_L", 1.0f);
    c.Lposes = EnvF("CC_LPOSES", 8.0f);
    c.SetEarlyOut(EnvF("CC_EARLY", 4.0f));
    c.promote = EnvF("CC_PROMOTE", 3.0f);
    c.Reset();
}

#if CC_STATS
void PrintStats(const char *what, const Stats &st, double poses, double ticks) {
    const double P = std::max(1.0, poses), SP = P * kShapes;
    std::printf("  %s: resets (teleports) %llu; body proxy evals %.3f per pose (certified %.3f, expanded to body shapes %.4f), proxy containment violations %llu\n", what,
                static_cast<unsigned long long>(st.resets), st.proxyEvals / std::max(1.0, poses), st.proxyCertified / std::max(1.0, poses), st.proxyExpand / std::max(1.0, poses),
                static_cast<unsigned long long>(st.proxyViolations));
    std::printf("  %s per pose: queries %.4f (%.3f/tick, %.0f patches, %.0f nodes each), expiry scans %.2f, vector AABB re-certs %.3f, margin evals %.3f (%.3f per slot), promotions %.3f, demotions %.3f\n",
                what, st.queries / P, st.queries / std::max(1.0, ticks), double(st.queryPatches) / std::max<std::uint64_t>(1, st.queries), double(st.queryNodes) / std::max<std::uint64_t>(1, st.queries),
                st.expireScans / P, st.boxRecerts / P, st.marginEvals / P, st.marginEvals / (P * 5), st.promotions / P, st.demotions / P);
    std::printf("  %s poses with: query %.1f%%, car-level scan %.1f%%, slot scan %.1f%%, wheel full evaluation %.1f%%, proxy evaluation %.1f%%, body full evaluation %.1f%%\n", what,
                100.0 * st.busyBy[0] / P, 100.0 * st.busyBy[1] / P, 100.0 * st.busyBy[2] / P, 100.0 * st.busyBy[3] / P, 100.0 * st.busyBy[4] / P, 100.0 * st.busyBy[5] / P);
    std::printf("  %s per pose: steady (no query, scan, full evaluation or proxy work) %.1f%%; sweeps %.4f, proxy patches certified by the 4 body slabs %.3f\n", what,
                100.0 * st.steadyPoses / P, st.sweeps / P, st.proxyBodySlab / P);
    std::printf("  %s per pose: car level held %.1f, scans %.3f, vector re-certs %.3f, handed to the slots %.3f\n", what, st.carHeld / P, st.carScans / P, st.carRecerts / P, st.carWakes / P);
    std::printf("  %s per pose: FACE kernel %.3f, slab pass certified %.3f; wheel axes off the car columns %llu\n", what, st.faceFast / P, st.slabCertified / P,
                static_cast<unsigned long long>(st.axisMismatch));
    std::printf("  %s per pose: sleepers held %.1f, tracked held %.2f (max %llu), tracked evals %.2f -> behind %.2f, out of reach %.2f, outside outline %.3f, closed FACE %.3f (+ angle bound %.3f), walk steps %.2f\n",
                what, st.sleepersHeld / P, st.trackedHeld / P, static_cast<unsigned long long>(st.maxTracked), st.trackedEvals / P, st.emptyBehind / P, st.emptyFar / P,
                st.emptyOutside / P, st.faceClosed / P, st.faceAngle / P, st.walkSteps / P);
    const double fb = double(st.fbBorder + st.fbOutside + st.fbWalk + st.fbMember + st.fbDeep + st.fbExactOnly);
    std::printf("  %s per pose: exact fallbacks %.3f (slab border %.3f, near outline %.3f, walk %.4f, member border %.3f, deep near edge %.3f, sliver %.3f) -> exact tests %.3f, hits %.3f; merges %.3f\n",
                what, fb / P, st.fbBorder / P, st.fbOutside / P, st.fbWalk / P, st.fbMember / P, st.fbDeep / P, st.fbExactOnly / P, st.exactTests / P, st.exactHits / P, st.merges / P);
    std::printf("  %s per slot (0 = body proxy, 4..7 wheels; 1..3 unused) tracked held / fallbacks (by body/wheel shape) / margin evals per pose:", what);
    for (int k = 0; k < kShapes; ++k) std::printf(" %.2f/%.3f/%.3f", st.perShapeTracked[k] / P, st.perShapeFallback[k] / P, st.perShapeMargins[k] / P);
    std::printf("\n");
}
#endif

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: cc_bench REC...\n"); return 2; }
    const std::string mode = std::getenv("CC_MODE") ? std::getenv("CC_MODE") : "fv,policy";
    const int reps = std::max(1, static_cast<int>(EnvF("CC_REPS", 3)));
    for (int a = 1; a < argc; ++a) {
        tw::Recording rec;
        if (!tw::LoadRecording(argv[a], &rec)) { std::fprintf(stderr, "cannot load %s\n", argv[a]); return 1; }
        const std::size_t N = rec.substeps.size();
        PatchWorld pw;
        pw.Build(rec.world, &rec.source, EnvF("CC_ANG", 1e-4f), EnvF("CC_DIST", 1e-4f), static_cast<std::uint32_t>(EnvF("CC_MAXMEM", 64)));
        const Ticks ticks(rec);
        const double T = static_cast<double>(ticks.first.size());
        std::printf("## %s: %zu substeps, %zu ticks (%.2f/tick); %zu tris -> %zu patches (%zu multi-triangle, %zu slivers, max %zu members; %zu enclosure tris left out), %zu outline edges; "
                    "patch data %.1f MB (+ world tris %.1f MB), build %.0f ms; L %.2f m, promote %.2f poses\n",
                    rec.name.c_str(), N, ticks.first.size(), N / T, rec.world.tris.size(), pw.patches.size(), pw.multi, pw.exactOnly, pw.maxMembers, pw.enclosure, pw.edges.size(),
                    pw.Bytes() / 1e6, rec.world.Bytes() / 1e6, pw.buildMs, EnvF("CC_L", 1.0f), EnvF("CC_PROMOTE", 3.0f));
        std::fflush(stdout);
        double ghz = CoreGHz();
#if CC_STATS
        Reference ref;
        ref.w = &rec.world;
        ref.bvh.Build(rec.world);
#endif
        // ---------------------------------------------------------------- calibration: the triangle twin (B's BVH4 + exact test)
        if (mode.find("ref") != std::string::npos) {
            Reference rt;
            rt.w = &rec.world;
            rt.bvh.Build(rec.world);
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                const auto w0 = Clock::now();
                for (std::size_t i = 0; i < N; ++i) rt.Step(rec.substeps[i]);
                best = std::min(best, std::chrono::duration<double, std::nano>(Clock::now() - w0).count() * ghz / N);
            }
            std::printf("REFERENCE (BVH4 per shape + exact test + merge): %.0f cyc/substep = %.0f cyc/tick at FV's count\n", best, best * N / T);
        }
        // ---------------------------------------------------------------- FV substep count
        if (mode.find("fv") != std::string::npos) {
            double best = 1e30, firstCyc = 0;
            for (int r = 0; r < reps; ++r) {
                CarCache c;
                Configure(c, pw, rec.world);
                const auto w0 = Clock::now();
                for (std::size_t i = 0; i < N; ++i) c.Step(rec.substeps[i]);
                const double cyc = std::chrono::duration<double, std::nano>(Clock::now() - w0).count() * ghz / N;
                if (r == 0) firstCyc = cyc;
                best = std::min(best, cyc);
            }
            std::printf("FV-COUNT: %.0f cyc/substep (first pass %.0f) = %.0f cyc/tick at FV's %.2f substeps/tick%s\n", best, firstCyc, best * N / T, N / T,
                        CC_STATS ? "  [stats build: counters on, not a timing]" : "");
#if CC_MICROB
            {
                CarCache c2;
                Configure(c2, pw, rec.world);
                const std::size_t stop = std::getenv("CC_MICRO") ? std::strtoul(std::getenv("CC_MICRO"), nullptr, 10) : N / 2;
                for (std::size_t i = 0; i <= stop && i < N; ++i) c2.Step(rec.substeps[i]);
                c2.MicroBench(ghz);
            }
#endif
#if CC_SPLIT
            {
                CarCache c;
                Configure(c, pw, rec.world);
                std::vector<float> per(N);
                double sum = 0;
                double sSteady = 0, sBusy = 0;
                std::size_t nSteady = 0;
                for (std::size_t i = 0; i < N; ++i) {
                    const std::uint64_t a0 = __rdtsc();
                    c.Step(rec.substeps[i]);
                    per[i] = static_cast<float>(__rdtsc() - a0);
                    sum += per[i];
                    if (c.busy_) sBusy += per[i]; else { sSteady += per[i]; ++nSteady; }
                }
                std::printf("  FV poses: steady %.1f%% at mean %.0f TSC, the rest mean %.0f TSC (%.1f%% of the time)\n", 100.0 * nSteady / N, sSteady / std::max<std::size_t>(1, nSteady),
                            sBusy / std::max<std::size_t>(1, N - nSteady), 100.0 * sBusy / std::max(1.0, sum));
                std::vector<float> srt = per;
                std::sort(srt.begin(), srt.end());
                double top = 0;
                for (std::size_t i = srt.size() - srt.size() / 100; i < srt.size(); ++i) top += srt[i];
                std::printf("  per-pose TSC: mean %.0f p50 %.0f p90 %.0f p99 %.0f max %.0f; top 1%% of poses = %.1f%% of the time\n", sum / N, Pct(per, 0.5), Pct(per, 0.9),
                            Pct(per, 0.99), Pct(per, 1.0), 100.0 * top / sum);
                std::size_t worst = 0;
                for (std::size_t i = 0; i < N; ++i) if (per[i] > per[worst]) worst = i;
                std::printf("  worst pose %zu (tick %u)\n", worst, rec.substeps[worst].tick);
                CarCache c2;
                Configure(c2, pw, rec.world);
                const std::size_t stop = std::getenv("CC_MICRO") ? std::strtoul(std::getenv("CC_MICRO"), nullptr, 10) : N / 2;
                for (std::size_t i = 0; i <= stop && i < N; ++i) c2.Step(rec.substeps[i]);
                c2.MicroBench(ghz);
            }
            {
                const double tot = double(g_split[0] + g_split[1] + g_split[2] + g_split[3]);
                std::printf("  split (rdtsc share): setup+odometers %.1f%%, query %.1f%%, expiry %.1f%%, tracked+merge %.1f%%; rdtsc per substep %.0f\n",
                            100 * g_split[0] / tot, 100 * g_split[1] / tot, 100 * g_split[2] / tot, 100 * g_split[3] / tot, tot / (double(N) * reps));
                for (auto &x : g_split) x = 0;
            }
#endif
#if CC_STATS
            CarCache c;
            Configure(c, pw, rec.world);
            FvCompare fvMine, fvRef;
            TwinCompare tc;
            tc.maxPrint = static_cast<std::size_t>(EnvF("CC_EXC", 0));
            std::vector<float> hot, query, exact;
            std::uint32_t tick = 0xffffffffu;
            auto flush = [&] {
                if (tick != 0xffffffffu) { hot.push_back(g_touch.lines[0] * 64.0f); query.push_back(g_touch.lines[1] * 64.0f); exact.push_back(g_touch.lines[2] * 64.0f); }
                g_touch.Clear();
            };
            for (std::size_t i = 0; i < N; ++i) {
                const tw::Substep &s = rec.substeps[i];
                if (s.tick != tick) { flush(); tick = s.tick; }
                c.Step(s);
                if (std::getenv("CC_DUMP") && i == static_cast<std::size_t>(std::atol(std::getenv("CC_DUMP")))) {
                    std::printf("DUMP substep %zu tick %u v (%.2f %.2f %.2f)\n", i, s.tick, s.vel[0], s.vel[1], s.vel[2]);
                    for (int k = 0; k < kShapes; ++k) c.Dump(k);
                }
                ref.Step(s);
                for (int k = 0; k < kShapes; ++k) {
                    fvMine.Add(rec, s, k, c.s[k].merged);
                    fvRef.Add(rec, s, k, ref.merged[k]);
                    tc.Add(i, k, c.s[k].merged, ref.merged[k]);
                }
            }
            flush();
            PrintStats("fv", c.st, double(N), T);
            fvMine.Print("twin C       ");
            fvRef.Print("triangle twin");
            std::printf("  twin C vs triangle twin: %zu of %zu (pose,shape) with contacts differ (%.4f%%)\n", tc.differ, tc.withContacts, 100.0 * tc.differ / std::max<std::size_t>(1, tc.withContacts));
            auto mean = [](const std::vector<float> &v) { double s = 0; for (float x : v) s += x; return v.empty() ? 0.0 : s / v.size(); };
            std::printf("  bytes touched per tick (distinct 64 B lines): cache/hot mean %.0f p99 %.0f max %.0f | re-query mean %.0f p99 %.0f max %.0f | exact fallback mean %.0f p99 %.0f max %.0f\n",
                        mean(hot), Pct(hot, 0.99), Pct(hot, 1.0), mean(query), Pct(query, 0.99), Pct(query, 1.0), mean(exact), Pct(exact, 0.99), Pct(exact, 1.0));
#endif
            std::fflush(stdout);
        }
        // ---------------------------------------------------------------- substep policy (contact1d0e3w900) + predictor
        if (mode.find("policy") != std::string::npos) {
            double best = 1e30, firstCyc = 0;
            std::size_t subsTotal = 0, events = 0, probed = 0;
            for (int r = 0; r < reps; ++r) {
                CarCache c;
                Configure(c, pw, rec.world);
                std::size_t subs = 0, ev = 0, pr = 0;
                const auto w0 = Clock::now();
                for (std::size_t t = 0; t < ticks.first.size(); ++t) {
                    const tw::Substep &s0 = rec.substeps[ticks.first[t]];
                    const int n = ticks.n[t];
                    c.reserve = Reserve(s0);
                    c.Step(s0); ++subs;
                    int kk = 1;
                    if (n > 1) {
                        ++pr;
                        KSet K0[kShapes];
                        c.KOf(K0);
                        tw::Substep pose[2];
                        Extrapolate(s0, 0.005f, &pose[0]);
                        Extrapolate(s0, 0.01f, &pose[1]);
                        const bool e = c.ProbeDiffers(pose[0], K0, nullptr) || c.ProbeDiffers(pose[1], K0, nullptr);
                        const float sp = std::sqrt(s0.vel[0] * s0.vel[0] + s0.vel[1] * s0.vel[1] + s0.vel[2] * s0.vel[2]);
                        if (e) { ++ev; kk = sp * 3.6f >= 900.0f ? n : std::min(n, 3); }
                    }
                    for (int j = 1; j < kk; ++j) { c.Step(rec.substeps[ticks.first[t] + static_cast<std::size_t>(j * n / kk)]); ++subs; }
                }
                const double cyc = std::chrono::duration<double, std::nano>(Clock::now() - w0).count() * ghz / T;
                if (r == 0) firstCyc = cyc;
                best = std::min(best, cyc);
                subsTotal = subs; events = ev; probed = pr;
            }
            std::printf("POLICY: %.0f cyc/tick all collision + predictor (first pass %.0f), %.2f substeps/tick, events %.1f%% of probed ticks (%zu probed of %.0f)%s\n",
                        best, firstCyc, subsTotal / T, 100.0 * events / std::max<std::size_t>(1, probed), probed, T,
                        CC_STATS ? "  [stats build]" : "");
#if CC_SPLIT
            {
                CarCache c;
                Configure(c, pw, rec.world);
                std::uint64_t tS = 0, tP = 0, pS = 0, pB = 0, nS = 0, nB = 0;
                double probedSplit = 0;
                std::vector<float> perTick;
                for (auto &x : g_split) x = 0;
                for (std::size_t t = 0; t < ticks.first.size(); ++t) {
                    const tw::Substep &s0 = rec.substeps[ticks.first[t]];
                    const int n = ticks.n[t];
                    c.reserve = Reserve(s0);
                    std::uint64_t a0 = __rdtsc();
                    c.Step(s0);
                    std::uint64_t a1 = __rdtsc();
                    tS += a1 - a0;
                    if (c.busy_) { pB += a1 - a0; ++nB; } else { pS += a1 - a0; ++nS; }
                    std::uint64_t tick = a1 - a0;
                    int kk = 1;
                    if (n > 1) {
                        ++probedSplit;
                        KSet K0[kShapes];
                        c.KOf(K0);
                        tw::Substep pose[2];
                        Extrapolate(s0, 0.005f, &pose[0]);
                        Extrapolate(s0, 0.01f, &pose[1]);
                        a0 = __rdtsc();
                        const bool e = c.ProbeDiffers(pose[0], K0, nullptr) || c.ProbeDiffers(pose[1], K0, nullptr);
                        a1 = __rdtsc();
                        tP += a1 - a0;
                        tick += a1 - a0;
                        const float sp = std::sqrt(s0.vel[0] * s0.vel[0] + s0.vel[1] * s0.vel[1] + s0.vel[2] * s0.vel[2]);
                        if (e) kk = sp * 3.6f >= 900.0f ? n : std::min(n, 3);
                    }
                    for (int j = 1; j < kk; ++j) {
                        a0 = __rdtsc();
                        c.Step(rec.substeps[ticks.first[t] + static_cast<std::size_t>(j * n / kk)]);
                        a1 = __rdtsc();
                        tS += a1 - a0;
                        tick += a1 - a0;
                        if (c.busy_) { pB += a1 - a0; ++nB; } else { pS += a1 - a0; ++nS; }
                    }
                    perTick.push_back(static_cast<float>(tick));
                }
                const double tot = double(tS + tP), sp4 = double(g_split[0] + g_split[1] + g_split[2] + g_split[3]);
                std::vector<float> srt = perTick;
                std::sort(srt.begin(), srt.end());
                double top = 0, sum = 0;
                for (float x : perTick) sum += x;
                for (std::size_t i = srt.size() - srt.size() / 100; i < srt.size(); ++i) top += srt[i];
                std::printf("  queries: %llu (%llu with a sweep): BVH %.0f TSC per query, sweep %.0f TSC per sweep\n", (unsigned long long)g_split[6], (unsigned long long)g_split[7],
                            double(g_split[4]) / std::max<std::uint64_t>(1, g_split[6]), double(g_split[5]) / std::max<std::uint64_t>(1, g_split[7]));
                std::printf("  policy poses: steady %.1f%% at mean %.0f TSC, the rest mean %.0f TSC; probes mean %.0f TSC\n", 100.0 * nS / std::max<std::uint64_t>(1, nS + nB),
                            double(pS) / std::max<std::uint64_t>(1, nS), double(pB) / std::max<std::uint64_t>(1, nB), double(tP) / std::max(1.0, 2.0 * probedSplit));
                std::printf("  policy split (TSC): collision passes %.1f%% (setup %.1f, query %.1f, expiry %.1f, tracked+merge %.1f), predictor %.1f%%; per tick p50 %.0f p90 %.0f p99 %.0f max %.0f TSC, top 1%% of ticks = %.1f%%\n",
                            100 * tS / tot, 100 * g_split[0] / sp4, 100 * g_split[1] / sp4, 100 * g_split[2] / sp4, 100 * g_split[3] / sp4, 100 * tP / tot,
                            Pct(perTick, 0.5), Pct(perTick, 0.9), Pct(perTick, 0.99), Pct(perTick, 1.0), 100.0 * top / sum);
            }
#endif
#if CC_STATS
            // instrumented policy run: counters, split, comparisons, predictor vs the triangle twin's exact probe
            CarCache c;
            Configure(c, pw, rec.world);
            Reference rp;  // exact probe reference
            rp.w = &rec.world;
            rp.bvh.Build(rec.world);
            FvCompare fvMine;
            TwinCompare tc;
            tc.maxPrint = static_cast<std::size_t>(EnvF("CC_EXC", 0));
            std::size_t falseNo = 0, falseChange = 0, evMine = 0, evRef = 0, prb = 0, kShapeMis = 0, kShapeTot = 0;
            std::uint64_t tStep = 0, tProbe = 0;
            std::size_t subs = 0;
            std::vector<float> hot, query, exact;
            int heavyPrinted = 0;
            for (std::size_t t = 0; t < ticks.first.size(); ++t) {
                g_touch.Clear();
                const tw::Substep &s0 = rec.substeps[ticks.first[t]];
                const int n = ticks.n[t];
                const Stats before = c.st, pbefore = c.pst;
                struct Heavy {
                    CarCache &c; const Stats &b, &pb; int &printed; std::size_t t; const tw::Substep &s0;
                    ~Heavy() {
                        const std::uint64_t me = c.st.marginEvals - b.marginEvals + c.pst.probeMargins - pb.probeMargins;
                        const std::uint64_t qp = c.st.queryPatches - b.queryPatches;
                        if ((me > 150 || qp > 400) && printed < 6) {
                            ++printed;
                            const float v = std::sqrt(s0.vel[0] * s0.vel[0] + s0.vel[1] * s0.vel[1] + s0.vel[2] * s0.vel[2]);
                            std::printf("    HEAVY tick %zu |v| %.0f km/h: queries %llu (%llu patches), margin evals %llu (probe %llu), proxy expands %llu, tracked evals %llu, sleepers now",
                                        t, v * 3.6, (unsigned long long)(c.st.queries - b.queries), (unsigned long long)qp, (unsigned long long)(c.st.marginEvals - b.marginEvals),
                                        (unsigned long long)(c.pst.probeMargins - pb.probeMargins), (unsigned long long)(c.st.proxyExpand - b.proxyExpand + c.pst.proxyExpand - pb.proxyExpand),
                                        (unsigned long long)(c.st.trackedEvals - b.trackedEvals));
                            for (int k = 4; k < kSlots; ++k) std::printf(" %zu", c.s[k].sleep.n);
                            std::printf("\n");
                        }
                    }
                } heavy{c, before, pbefore, heavyPrinted, t, s0};
                std::uint64_t r0 = __rdtsc();
                c.reserve = Reserve(s0);
                c.Step(s0); ++subs;
                tStep += __rdtsc() - r0;
                if (std::getenv("CC_DUMP") && ticks.first[t] + 3 >= static_cast<std::size_t>(std::atol(std::getenv("CC_DUMP"))) &&
                    ticks.first[t] <= static_cast<std::size_t>(std::atol(std::getenv("CC_DUMP"))) + 1) {
                    std::printf("DUMP policy pose %zu (tick start)\n", ticks.first[t]);
                    for (int k = 4; k < kShapes; ++k) c.Dump(k);
                }
                ref.Step(s0);
                for (int k = 0; k < kShapes; ++k) { fvMine.Add(rec, s0, k, c.s[k].merged); tc.Add(ticks.first[t], k, c.s[k].merged, ref.merged[k]); }
                int kk = 1;
                if (n > 1) {
                    ++prb;
                    KSet K0[kShapes], K0r[kShapes];
                    c.KOf(K0);
                    ref.KOf(K0r);
                    tw::Substep pose[2];
                    Extrapolate(s0, 0.005f, &pose[0]);
                    Extrapolate(s0, 0.01f, &pose[1]);
                    r0 = __rdtsc();
                    const bool e = c.ProbeDiffers(pose[0], K0, nullptr) || c.ProbeDiffers(pose[1], K0, nullptr);
                    tProbe += __rdtsc() - r0;
                    // exact probe: triangle twin at both poses; and C's full K at both poses vs it
                    bool er = false;
                    for (int pz = 0; pz < 2; ++pz) {
                        rp.Step(pose[pz]);
                        KSet Kr[kShapes], Km[kShapes];
                        rp.KOf(Kr);
                        c.ProbeDiffers(pose[pz], K0, Km);
                        for (int k = 0; k < kShapes; ++k) { er |= Kr[k] != K0r[k]; kShapeMis += Kr[k] != Km[k]; ++kShapeTot; }
                    }
                    evMine += e; evRef += er;
                    falseNo += !e && er; falseChange += e && !er;
                    const float sp = std::sqrt(s0.vel[0] * s0.vel[0] + s0.vel[1] * s0.vel[1] + s0.vel[2] * s0.vel[2]);
                    if (e) kk = sp * 3.6f >= 900.0f ? n : std::min(n, 3);
                }
                for (int j = 1; j < kk; ++j) {
                    const std::size_t i = ticks.first[t] + static_cast<std::size_t>(j * n / kk);
                    r0 = __rdtsc();
                    c.Step(rec.substeps[i]); ++subs;
                    tStep += __rdtsc() - r0;
                    if (std::getenv("CC_DUMP") && i + 1 >= static_cast<std::size_t>(std::atol(std::getenv("CC_DUMP"))) && i <= static_cast<std::size_t>(std::atol(std::getenv("CC_DUMP"))) + 1) {
                        std::printf("DUMP policy pose %zu\n", i);
                        for (int k = 4; k < kShapes; ++k) c.Dump(k);
                    }
                    ref.Step(rec.substeps[i]);
                    for (int k = 0; k < kShapes; ++k) { fvMine.Add(rec, rec.substeps[i], k, c.s[k].merged); tc.Add(i, k, c.s[k].merged, ref.merged[k]); }
                }
                hot.push_back(g_touch.lines[0] * 64.0f); query.push_back(g_touch.lines[1] * 64.0f); exact.push_back(g_touch.lines[2] * 64.0f);
            }
            const double tot = double(tStep + tProbe);
            std::printf("  policy split (rdtsc, stats build): collision passes %.1f%%, predictor %.1f%%; %.2f substeps/tick\n", 100.0 * tStep / tot, 100.0 * tProbe / tot, subs / T);
            PrintStats("policy", c.st, double(subs), T);
            PrintStats("probe ", c.pst, double(c.pst.probes), T);
            std::printf("  probes: beyond query horizon %.3f%% of shape-probes, sleeper margins %.3f per probe\n",
                        100.0 * c.pst.probeHorizon / std::max(1.0, double(c.pst.probes) * kShapes), double(c.pst.probeMargins) / std::max<std::uint64_t>(1, c.pst.probes));
            std::printf("  predictor: %zu probed ticks, events C %.1f%% / exact probe %.1f%%, false no-change %zu, false change %zu; per-shape K at probe poses differs from exact probe %zu of %zu\n",
                        prb, 100.0 * evMine / std::max<std::size_t>(1, prb), 100.0 * evRef / std::max<std::size_t>(1, prb), falseNo, falseChange, kShapeMis, kShapeTot);
            fvMine.Print("twin C (policy poses)");
            std::printf("  twin C vs triangle twin at policy poses: %zu of %zu (pose,shape) with contacts differ\n", tc.differ, tc.withContacts);
            auto mean = [](const std::vector<float> &v) { double s = 0; for (float x : v) s += x; return v.empty() ? 0.0 : s / v.size(); };
            std::printf("  bytes touched per tick (policy, incl. probes): cache/hot mean %.0f p99 %.0f max %.0f | re-query mean %.0f p99 %.0f | exact fallback mean %.0f p99 %.0f\n",
                        mean(hot), Pct(hot, 0.99), Pct(hot, 1.0), mean(query), Pct(query, 0.99), mean(exact), Pct(exact, 0.99));
#endif
            std::fflush(stdout);
        }
        (void)ghz;
    }
    return 0;
}
