// Architect C: rival collision stage (cc_stage.h) against B's stage (tw_collide.h) on the real
// engine's Detect calls (pose streams: twin_step_bench TWB_DUMP).
//
//   cc_stagebench PACK POSES [PACK POSES ...]
//     env CC_REPS=n (timing repetitions, default 3), CC_NOREF (skip the DetectShapesRef check),
//         CC_SCALAR (C without SSE lanes), CC_ONLY=b|c (time one stage)
//
// Per stream: (1) bit identity per call: C (warm cache), C (cache flushed every call) and B's
// stage against B's DetectShapesRef; (2) cost per call, best of n, each pass from a cold cache,
// both stages with the same last-answer cache in front (B's DetectPose has it; the stream holds
// shape poses, so the key here is the 8 shape poses' bits); (3) C's split (stats pass).
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "twin_pmu.h"
#include "cc_stage.h"
#include "tw_collide.h"
#include "tw_world_pack.h"

using namespace tw;
using Clock = std::chrono::steady_clock;

namespace {

double CoreGHz() {
    std::uint64_t x = 0;
    const auto t0 = Clock::now();
    for (int i = 0; i < 25000000; ++i)
        asm volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
    return 2e8 / std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

bool Same(const step::SubstepContacts &a, const step::SubstepContacts &b) {
    if (a.n != b.n || a.overflow != b.overflow) return false;
    for (int c = 0; c < a.n; ++c)
        if (std::memcmp(&a.c[c], &b.c[c], sizeof a.c[c]) != 0) return false;
    return true;
}

// B's stage with the last-answer cache of its DetectPose (keyed here by the shape poses)
struct BRun {
    const collide::ExactCollision *stage;
    collide::ExactCache cache;
    collide::ShapePose key[8];
    bool have = false;
    step::SubstepContacts last;
    void Call(const collide::ShapePose *sp, step::SubstepContacts &out) {
        if (have && std::memcmp(key, sp, sizeof key) == 0) {
            out.n = last.n;
            out.overflow = last.overflow;
            std::memcpy(out.c, last.c, sizeof(out.c[0]) * static_cast<std::size_t>(last.n));
            return;
        }
        stage->DetectShapes(sp, out, nullptr, &cache);
        std::memcpy(key, sp, sizeof key);
        have = true;
        last.n = out.n;
        last.overflow = out.overflow;
        std::memcpy(last.c, out.c, sizeof(out.c[0]) * static_cast<std::size_t>(out.n));
    }
};

bool Bits(const float *a, const float *b, int n) { return std::memcmp(a, b, sizeof(float) * static_cast<std::size_t>(n)) == 0; }

// Recorder mode (CC_REC): FV's recorded poses (twrec) -> C's merged contacts against FV's recorded
// merged contacts (tw_m3check's bit-identity rule: per shape, same count, P N Sep bits, material,
// primary), and against B's stage.
int Recorded(const char *packPath, const char *recPath) {
    world::WorldPack pack;
    std::string err;
    if (!world::LoadWorldPack(packPath, &pack, &err)) { std::fprintf(stderr, "pack: %s\n", err.c_str()); return 1; }
    world::WorldPackView view;
    if (!world::OpenWorldPack(pack, &view, &err, true)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    Recording rec;
    if (!LoadRecording(recPath, &rec)) { std::fprintf(stderr, "cannot load %s\n", recPath); return 1; }
    collide::CarShapes car;
    for (int k = 0; k < 8; ++k) car.shape[k].wheel = k >= 4 ? k - 4 : -1;
    collide::ExactCollision b;
    collide_c::CcCollision c;
    collide_c::StageOptions copt;
    copt.noise = std::getenv("CC_NOISE") != nullptr;
    if (const char *sw = std::getenv("CC_SWITCH")) if (!collide_c::ParseStageSwitch(sw, &copt)) { std::fprintf(stderr, "bad CC_SWITCH\n"); return 1; }
    if (!b.Init(view, car, &err) || !c.Init(view, car, &err, copt)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int showMis = std::getenv("CC_SHOW") ? std::atoi(std::getenv("CC_SHOW")) : 0;
    int shown = 0;
    std::vector<double> eP, eN, eS;   // continuous errors of discretely matched contacts
    std::size_t overP = 0, overN = 0, overS = 0, discShape = 0, matchedContacts = 0, ties = 0;
    collide_c::CcCache warm, flush;
    flush.flushEveryCall = true;
    collide::ExactCache cb;
    std::size_t N = rec.substeps.size(), subsIdent = 0, flushDiff = 0, bDiff = 0, shapeSubs = 0, shapeExc = 0, contacts = 0;
    step::SubstepContacts oc, of, ob;
    collide::ShapePose sp[8];
    for (std::size_t i = 0; i < N; ++i) {
        const Substep &s = rec.substeps[i];
        for (int k = 0; k < 8; ++k) {
            const Shape &sh = s.shapes[k];
            for (int cc = 0; cc < 3; ++cc)
                for (int a = 0; a < 3; ++a) sp[k].world.r[3 * cc + a] = sh.axis[cc][a];
            for (int a = 0; a < 3; ++a) { sp[k].world.t[a] = sh.t[a]; sp[k].radii[a] = sh.r[a]; }
        }
        c.DetectShapes(sp, oc, &warm);
        c.DetectShapes(sp, of, &flush);
        b.DetectShapes(sp, ob, nullptr, &cb);
        flushDiff += !Same(oc, of);
        bDiff += !Same(oc, ob);
        contacts += static_cast<std::size_t>(oc.n);
        bool ident = !oc.overflow;
        for (int k = 0; k < 8; ++k) {
            std::vector<const FvContact *> F;
            for (std::uint32_t q = s.firstMerged; q < s.firstMerged + s.mergedCount; ++q) if (rec.merged[q].shape == k) F.push_back(&rec.merged[q]);
            std::vector<const model::RespContact *> W;
            for (int q = 0; q < oc.n; ++q) if (oc.c[q].shape == k) W.push_back(&oc.c[q]);
            if (F.empty() && W.empty()) continue;
            ++shapeSubs;
            bool si = F.size() == W.size();
            for (std::size_t q = 0; si && q < F.size(); ++q) {
                const FvContact &f = *F[q];
                const model::RespContact &w = *W[q];
                si = Bits(f.P, w.P, 3) && Bits(f.N, w.N, 3) && Bits(f.Sep, w.Sep, 3) && f.material == w.material &&
                     (f.primary != 0) == ((w.flags & model::kRespPrimary) != 0);
            }
            shapeExc += !si;
            ident = ident && si;
            bool disc = F.size() == W.size();   // discrete: count, and per index (order) the same contact: material, FACE flag, near the same place
            for (std::size_t q = 0; disc && q < F.size(); ++q) {
                const float *a = F[q]->P, *bb = W[q]->P;
                const double dx = double(a[0]) - bb[0], dy = double(a[1]) - bb[1], dz = double(a[2]) - bb[2];
                const double nd = double(F[q]->N[0]) * W[q]->N[0] + double(F[q]->N[1]) * W[q]->N[1] + double(F[q]->N[2]) * W[q]->N[2];
                disc = F[q]->material == W[q]->material && (F[q]->primary != 0) == ((W[q]->flags & model::kRespPrimary) != 0) &&
                       dx * dx + dy * dy + dz * dz < 0.05 * 0.05 && nd > 0.9;
            }
            if (!disc) {
                ++discShape;
                // near-tie test: does FV's own decision for this shape flip when the poses move by 1-2 ulp
                // (B's exact stage = FV's bits; 12 seeded perturbations of every pose float)?
                bool tie = false;
                std::uint32_t seed = 0x9e3779b9u * static_cast<std::uint32_t>(i + 1) + static_cast<std::uint32_t>(k);
                for (int t = 0; t < 12 && !tie; ++t) {
                    collide::ShapePose pp[8];
                    std::memcpy(pp, sp, sizeof pp);
                    for (int kk = 0; kk < 8; ++kk)
                        for (int f = 0; f < 12; ++f) {
                            seed = seed * 1664525u + 1013904223u;
                            float &x = f < 9 ? pp[kk].world.r[f] : pp[kk].world.t[f - 9];
                            const int steps = 1 + int(seed >> 31);   // 1 or 2 ulp
                            for (int st = 0; st < steps; ++st) x = std::nextafter(x, (seed >> 30 & 1u) ? 1e30f : -1e30f);
                        }
                    step::SubstepContacts op;
                    b.DetectShapesRef(pp, op);
                    std::vector<const model::RespContact *> X;
                    for (int qq = 0; qq < op.n; ++qq) if (op.c[qq].shape == k) X.push_back(&op.c[qq]);
                    bool same = X.size() == F.size();
                    for (std::size_t qq = 0; same && qq < F.size(); ++qq)
                        same = X[qq]->material == F[qq]->material && ((X[qq]->flags & model::kRespPrimary) != 0) == (F[qq]->primary != 0);
                    tie = !same;
                }
                ties += tie;
                if (!tie && shown < showMis) {
                    ++shown;
                    std::printf("  mismatch substep %zu shape %d: FV %zu, C %zu\n", i, k, F.size(), W.size());
                    for (const FvContact *f : F)
                        std::printf("    FV %s m%u P(%.5f %.5f %.5f) N(%.5f %.5f %.5f) |Sep| %.3e\n", f->primary ? "FACE" : "feat", f->material, f->P[0], f->P[1], f->P[2],
                                    f->N[0], f->N[1], f->N[2], std::sqrt(double(f->Sep[0]) * f->Sep[0] + double(f->Sep[1]) * f->Sep[1] + double(f->Sep[2]) * f->Sep[2]));
                    for (const model::RespContact *w : W)
                        std::printf("    C  %s m%u P(%.5f %.5f %.5f) N(%.5f %.5f %.5f) |Sep| %.3e\n", (w->flags & 1) ? "FACE" : "feat", w->material, w->P[0], w->P[1], w->P[2],
                                    w->N[0], w->N[1], w->N[2], std::sqrt(double(w->Sep[0]) * w->Sep[0] + double(w->Sep[1]) * w->Sep[1] + double(w->Sep[2]) * w->Sep[2]));
                }
                continue;
            }
            for (std::size_t q = 0; q < F.size(); ++q) {
                const FvContact &f = *F[q];
                const model::RespContact &w = *W[q];
                auto dist = [](const float *a, const float *b) {
                    const double x = double(a[0]) - b[0], y = double(a[1]) - b[1], z = double(a[2]) - b[2];
                    return std::sqrt(x * x + y * y + z * z);
                };
                auto norm = [](const float *a) { return std::sqrt(double(a[0]) * a[0] + double(a[1]) * a[1] + double(a[2]) * a[2]); };
                const double dP = dist(f.P, w.P), dN = dist(f.N, w.N), dS = dist(f.Sep, w.Sep);
                eP.push_back(dP); eN.push_back(dN); eS.push_back(dS);
                ++matchedContacts;
                // gate-1 floors (harness.md): w.latest_point 2e-3 m + 1e-5 rel, w.normal_sum 1e-3 + 1e-4 rel, push-out (w.max_repl_y) 2e-4 m + 1e-3 rel
                overP += dP > 2e-3 + 1e-5 * norm(f.P);
                overN += dN > 1e-3 + 1e-4;
                overS += dS > 2e-4 + 1e-3 * norm(f.Sep);
            }
        }
        subsIdent += ident;
    }
    std::printf("## %s (FV recording%s): %zu substeps, %.2f contacts/substep; bit-identical to FV %.4f%% (%zu differ), (substep, shape) exceptions %zu of %zu; "
                "C flushed-every-call vs warm: %zu differ; C vs B: %zu differ\n",
                recPath, copt.noise ? ", noise-level C" : "", N, double(contacts) / double(N ? N : 1), 100.0 * double(subsIdent) / double(N ? N : 1), N - subsIdent,
                shapeExc, shapeSubs, flushDiff, bDiff);
    auto q = [](std::vector<double> &v, double p) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, std::size_t(p * double(v.size())))]; };
    std::printf("  discrete: %zu of %zu (substep, shape) differ (%.4f%%), of which FV itself flips under 1-2 ulp pose noise: %zu; continuous over %zu matched contacts: |dP| p50 %.1e p99.99 %.1e max %.1e m, |dN| p50 %.1e p99.99 %.1e max %.1e, "
                "|dSep| p50 %.1e p99.99 %.1e max %.1e m; over gate-1 floors P %zu N %zu Sep %zu\n",
                discShape, shapeSubs, 100.0 * double(discShape) / double(shapeSubs ? shapeSubs : 1), ties, matchedContacts, q(eP, 0.5), q(eP, 0.9999), q(eP, 1.0),
                q(eN, 0.5), q(eN, 0.9999), q(eN, 1.0), q(eS, 0.5), q(eS, 0.9999), q(eS, 1.0), overP, overN, overS);
    std::fflush(stdout);
    return 0;
}

int Stream(const char *packPath, const char *posePath) {
    world::WorldPack pack;
    std::string err;
    if (!world::LoadWorldPack(packPath, &pack, &err)) { std::fprintf(stderr, "pack: %s\n", err.c_str()); return 1; }
    world::WorldPackView view;
    if (!world::OpenWorldPack(pack, &view, &err, true)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    std::FILE *f = std::fopen(posePath, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", posePath); return 1; }
    std::vector<collide::ShapePose> poses;
    collide::ShapePose sp[8];
    while (std::fread(sp, sizeof sp, 1, f) == 1) poses.insert(poses.end(), sp, sp + 8);
    std::fclose(f);
    const std::size_t N = poses.size() / 8;
    if (N == 0) { std::fprintf(stderr, "%s: empty\n", posePath); return 1; }
    collide::CarShapes car;   // the stream has no shape table: StadiumCar tree order (0-3 body, 4-7 wheels)
    for (int k = 0; k < 8; ++k) car.shape[k].wheel = k >= 4 ? k - 4 : -1;
    collide::ExactCollision b;
    if (!b.Init(view, car, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    collide_c::StageOptions opt;
    if (std::getenv("CC_SCALAR")) { opt.simdFrames = false; opt.simdNarrow = false; }
    if (std::getenv("CC_NOSIMDNARROW")) opt.simdNarrow = false;
    if (std::getenv("CC_GATE")) opt.bodyGate = true;
    if (std::getenv("CC_NOSKIP")) opt.skipAbsorbed = false;
    if (std::getenv("CC_NOISE")) opt.noise = true;
    if (const char *sw = std::getenv("CC_SWITCH")) if (!collide_c::ParseStageSwitch(sw, &opt)) { std::fprintf(stderr, "bad CC_SWITCH\n"); return 1; }
    if (std::getenv("CC_LEAD")) opt.leadPoses = static_cast<float>(std::atof(std::getenv("CC_LEAD")));
    if (std::getenv("CC_MARGIN")) opt.unionMargin = static_cast<float>(std::atof(std::getenv("CC_MARGIN")));
    collide_c::CcCollision c;
    if (!c.Init(view, car, &err, opt)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    if (std::getenv("CC_ABL")) {   // ablation: cost to the end of each phase (best of 3), B: to its filter
        double t[5] = {1e30, 1e30, 1e30, 1e30, 1e30};
        collide_c::CcCollision ab[3];
        for (int a = 0; a < 3; ++a) {
            collide_c::StageOptions o2 = opt;
            o2.stopAfter = a + 1;
            o2.lastCache = false;
            if (!ab[a].Init(view, car, &err, o2)) return 1;
        }
        collide::CollideOptions bo;
        bo.profileStop = 1;
        collide::ExactCollision bf;
        if (!bf.Init(view, car, &err, bo)) return 1;
        step::SubstepContacts x;
        bench::Pmu pm;
        for (int rep = 0; rep < 3; ++rep)
            for (int a = 0; a < 4; ++a) {
                collide_c::CcCache cc;
                collide::ExactCache cb;
                const auto c0 = pm.Read();
                for (std::size_t i = 0; i < N; ++i) {
                    if (a < 3) ab[a].DetectShapes(&poses[8 * i], x, &cc);
                    else bf.DetectShapes(&poses[8 * i], x, nullptr, &cb);
                }
                t[a] = std::min(t[a], double(bench::Pmu::Delta(c0, pm.Read()).cycles));
            }
        const double g = 1.0;
        std::printf("  ablation per call (no last-answer cache): C to memo %.0f, to filter %.0f, to frames %.0f; B to its filter %.0f\n", t[0] * g / N, t[1] * g / N,
                    t[2] * g / N, t[3] * g / N);
    }
    // (1) bit identity
    std::size_t dWarm = 0, dFlush = 0, dB = 0, contacts = 0;
    {
        const bool ref = std::getenv("CC_NOREF") == nullptr;
        collide::ExactCache cb;
        collide_c::CcCache warm, flush;
        flush.flushEveryCall = true;
        step::SubstepContacts ob, oc, of, orf;
        for (std::size_t i = 0; i < N; ++i) {
            const collide::ShapePose *p = &poses[8 * i];
            b.DetectShapes(p, ob, nullptr, &cb);
            c.DetectShapes(p, oc, &warm);
            c.DetectShapes(p, of, &flush);
            const step::SubstepContacts *truth = &ob;
            if (ref) {
                b.DetectShapesRef(p, orf);
                truth = &orf;
                dB += !Same(ob, orf);
            }
            dWarm += !Same(oc, *truth);
            dFlush += !Same(of, *truth);
            contacts += static_cast<std::size_t>(truth->n);
        }
    }
    // (2) cost per call
    const int reps = std::getenv("CC_REPS") ? std::max(1, std::atoi(std::getenv("CC_REPS"))) : 5;
    const char *only = std::getenv("CC_ONLY");
    double best[2] = {1e30, 1e30};   // PMU core cycles of the pass (every call of the pass is the stage's)
    std::uint64_t lastHits = 0, refills = 0;
    bench::Pmu pmu;
    if (!pmu.ok()) { std::fprintf(stderr, "PMU (rdpmc) unavailable\n"); return 1; }
    step::SubstepContacts o;
    for (int r = 0; r < reps; ++r) {
        if (!only || *only == 'b') {
            BRun br{&b};
            const auto c0 = pmu.Read();
            for (std::size_t i = 0; i < N; ++i) br.Call(&poses[8 * i], o);
            best[0] = std::min(best[0], double(bench::Pmu::Delta(c0, pmu.Read()).cycles));
        }
        if (!only || *only == 'c') {
            collide_c::CcCache cc;
            const auto c0 = pmu.Read();
            for (std::size_t i = 0; i < N; ++i) c.DetectShapes(&poses[8 * i], o, &cc);
            best[1] = std::min(best[1], double(bench::Pmu::Delta(c0, pmu.Read()).cycles));
            lastHits = cc.lastHits;
            refills = cc.refills;
        }
    }
    const double ghz = CoreGHz();   // only for the stats pass's TSC split
    // (3) C's split
    collide_c::StageStats st;
    double memoN = 0, nearN = 0, pairsN = 0, planesN = 0;
    {
        collide_c::CcCache cc;
        for (std::size_t i = 0; i < N; ++i) { c.DetectShapes(&poses[8 * i], o, &cc, &st); memoN += cc.m.n; }
        nearN = double(cc.m.nearEntries); pairsN = double(cc.m.pairs); planesN = double(cc.m.planesMade);
        const double cn = double(st.calls - st.lastHits);
        std::printf("  pairs / survivors per computed call by shape:");
        for (int k = 0; k < 8; ++k) std::printf(" %d: %.1f/%.2f", k, cc.m.pairsOf[k] / cn, cc.m.survOf[k] / cn);
        std::printf("; gated entries %.2f\n", cc.m.gated / cn);
        std::printf("  filter fine (TSC per computed call): pass0 %.0f, pass1 %.0f, planes %.0f, prefilter %.0f, sort %.0f\n", cc.m.fine[0] / cn,
                    cc.m.fine[1] / cn, cc.m.fine[2] / cn, cc.m.fine[3] / cn, cc.m.fine[4] / cn);
    }
    const double n = double(N), kk = ghz / 3.893, nc = double(st.calls - st.lastHits);
    std::printf("## %s: %zu calls, %.2f contacts/call; differ vs reference: C warm %zu, C flushed %zu, B %zu\n", posePath, N, double(contacts) / n,
                dWarm, dFlush, dB);
    std::printf("  PMU core cycles/call (best of %d): B %.0f, C %.0f (%.2fx); last-answer hits %.3f, refills %.3f per call\n", reps,
                best[0] / n, best[1] / n, best[0] / best[1], double(lastHits) / n, double(refills) / n);
    std::printf("  C per computed call: survivors %.2f (body %.2f), frames %.2f, decide reject %.2f face %.2f fv %.2f, raw %.2f (body %.2f), merged %.2f (body %.2f); FV-form runs %.2f, skipped %.2f\n",
                st.survivors / nc, st.bodySurvivors / nc, st.frames / nc, st.reject / nc, st.face / nc, st.fvForm / nc, st.raw / nc, st.bodyRaw / nc,
                st.merged / nc, st.bodyMerged / nc, st.fvRuns / nc, st.skipped / nc);
    std::printf("  C memo per call: entries %.1f, near the union %.1f, (entry, shape) pairs %.1f, planes made %.2f\n", memoN / n, nearN / nc, pairsN / nc, planesN / nc);
    std::printf("  C split per computed call (stats pass, core cycles): boxes %.0f, memo %.0f, filter %.0f, frames %.0f, narrow %.0f, merge+out %.0f\n",
                st.tBoxes * kk / nc, st.tMemo * kk / nc, st.tFilter * kk / nc, st.tFrames * kk / nc, st.tNarrow * kk / nc, st.tMerge * kk / nc);
    std::fflush(stdout);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || (argc - 1) % 2) { std::fprintf(stderr, "usage: cc_stagebench PACK POSES [PACK POSES ...]\n"); return 2; }
    const bool recMode = std::getenv("CC_REC") != nullptr;
    for (int a = 1; a + 1 < argc; a += 2)
        if (recMode ? Recorded(argv[a], argv[a + 1]) : Stream(argv[a], argv[a + 1])) return 1;
    return 0;
}
