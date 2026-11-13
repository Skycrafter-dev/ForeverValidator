// tw_m3check: the M3 collision stage (src/twin/collide/) against FV's merged per-shape
// contacts recorded at FV's own shape poses.
//   tw_m3check PACK REC [PACK REC ...]      env M3_EXC=n prints the first n differing substeps
// Per substep: bit-identical (every shape's merged list equal in order, P/N/Sep bits, material,
// primary flag), and the §6 metrics (count, |dP| <= 1 mm, angle <= 0.5 deg, |dSep| <= 0.1 mm,
// material, flag). TWREC003 recordings also check the peer actor.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <x86intrin.h>
#include <vector>

#include "../common/twin_pmu.h"
#include "tw_collide.h"
#include "tw_collide_ref.h"
#include "cc_stage.h"
#include "tw_world.h"
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

bool Bits(const float *a, const float *b, int n) { return std::memcmp(a, b, sizeof(float) * static_cast<std::size_t>(n)) == 0; }

void Poses(const tw::Substep &s, collide::ShapePose out[8]) {
    for (int k = 0; k < 8; ++k) {
        const Shape &sh = s.shapes[k];
        for (int c = 0; c < 3; ++c)
            for (int i = 0; i < 3; ++i) out[k].world.r[3 * c + i] = sh.axis[c][i];
        for (int i = 0; i < 3; ++i) { out[k].world.t[i] = sh.t[i]; out[k].radii[i] = sh.r[i]; }
    }
}

}  // namespace

// M3_REFD=1: arguments in triples PACK CARSHAPES CPOSES (twin_step_bench TWB_DUMP): the float stage (DetectPose, at
// the car pose) against the double-precision reference (tw_collide_ref.h) at the same pose, widened. Per call the
// merged lists are matched (nearest point, same shape); reported: discrete mismatches (count, material, flags), the
// largest |dP|, normal angle and |dSep| over matched contacts, and for every mismatch the reference's closest
// decision margin (a near-threshold decision is where float and double may legitimately part).
int RefDouble(const char *packPath, const char *carPath, const char *posePath) {
    world::WorldPack pack;
    std::string err;
    if (!world::LoadWorldPack(packPath, &pack, &err)) { std::fprintf(stderr, "pack: %s\n", err.c_str()); return 1; }
    world::WorldPackView view;
    if (!world::OpenWorldPack(pack, &view, &err, true)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    collide::CarShapes car;
    {
        std::FILE *f = std::fopen(carPath, "rb");
        if (!f || std::fread(&car, sizeof car, 1, f) != 1) { std::fprintf(stderr, "cannot read %s\n", carPath); if (f) std::fclose(f); return 1; }
        std::fclose(f);
    }
    std::vector<float> cp;
    {
        std::FILE *f = std::fopen(posePath, "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", posePath); return 1; }
        float b[11];
        while (std::fread(b, sizeof b, 1, f) == 1) cp.insert(cp.end(), b, b + 11);
        std::fclose(f);
    }
    const std::size_t N = cp.size() / 11;
    collide::ExactCollision stage;
    collide::CollideOptions sopt;
    if (std::getenv("M3_REFD_NOISE")) sopt.narrow = &collide_c::noise::Narrow;   // b-noise: C's noise-level narrow phase
    if (std::getenv("M3_R7")) sopt.closedWheelFace = true;
    if (!stage.Init(view, car, &err, sopt)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    collide::ref::DetectionD dd;
    step::SubstepContacts sc;
    std::size_t calls = 0, mism = 0, mismNear = 0, shapePairs = 0, contactsF = 0, contactsD = 0;
    double maxDP = 0, maxAng = 0, maxDS = 0, minMarginMism = 1e300, maxUlp = 0;
    std::vector<double> dPs;
    std::size_t byKind[8] = {};
    const double kNear = 1e-4;   // a decision margin below this (unit-space units) counts as near a threshold
    for (std::size_t i = 0; i < N; ++i) {
        const float *pf = &cp[11 * i];
        stage.DetectPose(pf, pf + 3, pf + 7, sc);
        const double pd[3] = {pf[0], pf[1], pf[2]}, qd[4] = {pf[3], pf[4], pf[5], pf[6]}, ad[4] = {pf[7], pf[8], pf[9], pf[10]};
        collide::ref::Detect(stage, pd, qd, ad, &dd);
        ++calls;
        contactsF += static_cast<std::size_t>(sc.n);
        contactsD += dd.contacts.size();
        bool callBad = false;
        for (int k = 0; k < 8; ++k) {
            std::vector<const model::RespContact *> F;
            for (int c = 0; c < sc.n; ++c) if (sc.c[c].shape == k) F.push_back(&sc.c[c]);
            std::vector<const collide::ref::ContactD *> D;
            for (const auto &c : dd.contacts) if (c.shape == k) D.push_back(&c);
            if (F.empty() && D.empty()) continue;
            ++shapePairs;
            bool bad = F.size() != D.size();
            std::vector<bool> used(F.size(), false);
            for (std::size_t di = 0; di < D.size(); ++di) {
                const collide::ref::ContactD *d = D[di];
                int best = -1; double bc = 1e300;
                if (F.size() == D.size()) {   // same count: both lists are in FV's order
                    best = static_cast<int>(di);
                    const double dx = F[di]->P[0] - d->P[0], dy = F[di]->P[1] - d->P[1], dz = F[di]->P[2] - d->P[2];
                    bc = std::sqrt(dx * dx + dy * dy + dz * dz);
                } else {
                    for (std::size_t q = 0; q < F.size(); ++q) {
                        if (used[q]) continue;
                        const double dx = F[q]->P[0] - d->P[0], dy = F[q]->P[1] - d->P[1], dz = F[q]->P[2] - d->P[2];
                        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (dist < 0.05 && dist < bc) { bc = dist; best = static_cast<int>(q); }
                    }
                }
                if (best < 0) { bad = true; continue; }
                used[static_cast<std::size_t>(best)] = true;
                const model::RespContact &f = *F[static_cast<std::size_t>(best)];
                const double dot = std::max(-1.0, std::min(1.0, f.N[0] * d->N[0] + f.N[1] * d->N[1] + f.N[2] * d->N[2]));
                const double ds = std::sqrt((f.Sep[0] - d->Sep[0]) * (f.Sep[0] - d->Sep[0]) + (f.Sep[1] - d->Sep[1]) * (f.Sep[1] - d->Sep[1]) +
                                            (f.Sep[2] - d->Sep[2]) * (f.Sep[2] - d->Sep[2]));
                if (f.material != d->material || f.flags != d->flags || bc > 1e-3) { bad = true; continue; }   // another contact: discrete
                if (ds > 1e-4 && std::getenv("M3_REFD_SHOW"))
                    std::printf("    call %zu shape %d %s: Sep float (%.6e %.6e %.6e) double (%.6e %.6e %.6e); |P| diff %.2e; shape margin %.2e kind %d\n", i, k,
                                (f.flags & model::kRespPrimary) ? "FACE" : "feature", f.Sep[0], f.Sep[1], f.Sep[2], d->Sep[0], d->Sep[1], d->Sep[2], bc,
                                dd.perShape[k].value, int(dd.perShape[k].kind));
                maxDP = std::max(maxDP, bc);
                dPs.push_back(bc);
                {   // relative to the float spacing at the point's largest coordinate (the map-scale ulp)
                    const float r = static_cast<float>(std::max(std::fabs(d->P[0]), std::max(std::fabs(d->P[1]), std::fabs(d->P[2]))));
                    const double ulp = double(std::nextafter(r, 1e30f) - r);
                    maxUlp = std::max(maxUlp, bc / ulp);
                }
                maxAng = std::max(maxAng, std::acos(dot) * 57.29577951308232);
                maxDS = std::max(maxDS, ds);
            }
            if (bad) callBad = true;
        }
        if (callBad) {
            ++mism;
            minMarginMism = std::min(minMarginMism, dd.closest.value);
            // FV's float error in unit space at this position: its composed transforms round at ~4 eps (|t_e| + |t_m|),
            // over the smallest radius (0.182 m, the wheels); a decision within 2x that is a near-threshold one
            const double errScale = std::max(kNear, 2.0 * 4.0 * 1.1920929e-7 * 2.0 * (std::fabs(pf[0]) + std::fabs(pf[1]) + std::fabs(pf[2])) / 0.182);
            if (dd.closest.value < errScale) { ++mismNear; ++byKind[dd.closest.kind]; }
            else if (std::getenv("M3_REFD_SHOW")) {
                std::printf("    call %zu: mismatch away from thresholds (closest margin %.2e kind %d shape %d); float %d contacts, double %zu\n", i,
                            dd.closest.value, int(dd.closest.kind), int(dd.closest.shape), sc.n, dd.contacts.size());
                for (int c = 0; c < sc.n; ++c)
                    std::printf("      F sh %d %s m%u P(%.4f %.4f %.4f) N(%.4f %.4f %.4f)\n", sc.c[c].shape, (sc.c[c].flags & 1) ? "FACE" : "feat", sc.c[c].material,
                                sc.c[c].P[0], sc.c[c].P[1], sc.c[c].P[2], sc.c[c].N[0], sc.c[c].N[1], sc.c[c].N[2]);
                for (const auto &c : dd.contacts)
                    std::printf("      D sh %d %s m%u P(%.4f %.4f %.4f) N(%.4f %.4f %.4f) margin %.2e kind %d\n", c.shape, (c.flags & 1) ? "FACE" : "feat", c.material,
                                c.P[0], c.P[1], c.P[2], c.N[0], c.N[1], c.N[2], dd.perShape[c.shape].value, int(dd.perShape[c.shape].kind));
            }
        }
    }
    std::printf("## %s: %zu calls, contacts float %zu / double %zu; discrete mismatches %zu calls (%.4f%%), of which near a threshold (within FV's float error at that position) %zu\n",
                posePath, calls, contactsF, contactsD, mism, 100.0 * mism / std::max<std::size_t>(1, calls), mismNear);
    std::printf("  matched contacts (shapes whose lists agree discretely): max |dP| %.2e m, normal %.2e deg, |dSep| %.2e m; smallest margin among mismatching calls %.2e\n", maxDP, maxAng, maxDS,
                minMarginMism);
    std::sort(dPs.begin(), dPs.end());
    std::printf("  contact point |dP|: p50 %.2e, p99 %.2e, p99.9 %.2e, max %.2e m; max %.2f map-scale ulps (float spacing at the largest coordinate)\n",
                dPs.empty() ? 0.0 : dPs[dPs.size() / 2], dPs.empty() ? 0.0 : dPs[dPs.size() * 99 / 100], dPs.empty() ? 0.0 : dPs[dPs.size() * 999 / 1000], maxDP, maxUlp);
    std::printf("  near-threshold mismatches by decision: plane %zu, edge reach %zu, edge sign %zu, feature branch %zu, feature distance %zu, NearlyEqual %zu, "
                "alignment %zu\n", byKind[1], byKind[2], byKind[3], byKind[4], byKind[5], byKind[6], byKind[7]);
    return 0;
}

// M3_REFD_SR=K: on every M3_REFD_EVERY-th call (default 10), K stochastic-rounding samples of the reference
// (ref::DetectSR: a population of valid float implementations) next to FV's float stage and C's noise stage, all
// against the double reference. Per contact whose lists agree with the reference: the SR spread s = max over the
// samples of |P_sr - P_d|, and where FV's |dP| and C's |dP| fall against it. Discrete: how often an SR sample's
// lists differ from the reference, against FV's and C's.
int RefSR(const char *packPath, const char *carPath, const char *posePath, int K, int every) {
    world::WorldPack pack;
    std::string err;
    if (!world::LoadWorldPack(packPath, &pack, &err)) { std::fprintf(stderr, "pack: %s\n", err.c_str()); return 1; }
    world::WorldPackView view;
    if (!world::OpenWorldPack(pack, &view, &err, true)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    collide::CarShapes car;
    {
        std::FILE *f = std::fopen(carPath, "rb");
        if (!f || std::fread(&car, sizeof car, 1, f) != 1) { if (f) std::fclose(f); return 1; }
        std::fclose(f);
    }
    std::vector<float> cp;
    {
        std::FILE *f = std::fopen(posePath, "rb");
        if (!f) return 1;
        float b[11];
        while (std::fread(b, sizeof b, 1, f) == 1) cp.insert(cp.end(), b, b + 11);
        std::fclose(f);
    }
    const std::size_t N = cp.size() / 11;
    collide::ExactCollision fvStage, cStage;
    collide::CollideOptions copt;
    copt.narrow = &collide_c::noise::Narrow;
    if (!fvStage.Init(view, car, &err) || !cStage.Init(view, car, &err, copt)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    // same discrete list (count, shape, material, flags in order); then per contact |dP|
    auto sameList = [](const std::vector<collide::ref::ContactD> &a, const collide::ref::DetectionD &d) {
        if (a.size() != d.contacts.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i].shape != d.contacts[i].shape || a[i].material != d.contacts[i].material || a[i].flags != d.contacts[i].flags) return false;
        return true;
    };
    auto toD = [](const step::SubstepContacts &sc, std::vector<collide::ref::ContactD> *o) {
        o->clear();
        for (int c = 0; c < sc.n; ++c) {
            collide::ref::ContactD d{};
            for (int a = 0; a < 3; ++a) { d.P[a] = sc.c[c].P[a]; d.N[a] = sc.c[c].N[a]; d.Sep[a] = sc.c[c].Sep[a]; }
            d.shape = sc.c[c].shape; d.material = sc.c[c].material; d.flags = sc.c[c].flags;
            o->push_back(d);
        }
    };
    auto dist = [](const double *a, const double *b) { return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])); };
    collide::ref::DetectionD dd, ds;
    step::SubstepContacts sc;
    std::vector<collide::ref::ContactD> fvL, cL;
    std::vector<std::vector<collide::ref::ContactD>> srL(static_cast<std::size_t>(K));
    std::size_t calls = 0, srFlips = 0, fvFlips = 0, cFlips = 0, contacts = 0, fvOut = 0, cOut = 0;
    std::vector<double> fvRatio, cRatio, spreads;
    for (std::size_t i = 0; i < N; i += static_cast<std::size_t>(every)) {
        const float *pf = &cp[11 * i];
        const double pd[3] = {pf[0], pf[1], pf[2]}, qd[4] = {pf[3], pf[4], pf[5], pf[6]}, ad[4] = {pf[7], pf[8], pf[9], pf[10]};
        collide::ref::Detect(fvStage, pd, qd, ad, &dd);
        fvStage.DetectPose(pf, pf + 3, pf + 7, sc);
        toD(sc, &fvL);
        cStage.DetectPose(pf, pf + 3, pf + 7, sc);
        toD(sc, &cL);
        const tw::sr::sr ps[3] = {pf[0], pf[1], pf[2]}, qs[4] = {pf[3], pf[4], pf[5], pf[6]}, as[4] = {pf[7], pf[8], pf[9], pf[10]};
        bool srAll = true;
        for (int k = 0; k < K; ++k) {
            tw::sr::Seed(0x9E3779B97F4A7C15ull * (i + 1) + static_cast<std::uint64_t>(k) * 0x632BE59BD9B4E019ull);
            collide::ref::DetectSR(fvStage, ps, qs, as, &ds);
            srL[static_cast<std::size_t>(k)] = ds.contacts;
            if (!sameList(ds.contacts, dd)) { ++srFlips; srAll = false; }
        }
        ++calls;
        const bool fvSame = sameList(fvL, dd), cSame = sameList(cL, dd);
        fvFlips += !fvSame;
        cFlips += !cSame;
        if (!srAll) continue;   // the spread is defined on contacts every sample agrees on
        for (std::size_t c = 0; c < dd.contacts.size(); ++c) {
            double sp = 0.0;
            for (int k = 0; k < K; ++k) sp = std::max(sp, dist(srL[static_cast<std::size_t>(k)][c].P, dd.contacts[c].P));
            ++contacts;
            spreads.push_back(sp);
            const double s0 = std::max(sp, 1e-12);
            if (fvSame) { const double e = dist(fvL[c].P, dd.contacts[c].P); fvRatio.push_back(e / s0); fvOut += e > sp; }
            if (cSame) { const double e = dist(cL[c].P, dd.contacts[c].P); cRatio.push_back(e / s0); cOut += e > sp; }
        }
    }
    auto q = [](std::vector<double> &v, double f) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, static_cast<std::size_t>(f * v.size()))]; };
    std::printf("## %s: %zu calls x %d SR samples; discrete differences vs double: SR %.3f%% of samples, FV %.3f%% of calls, C %.3f%% of calls\n", posePath,
                calls, K, 100.0 * srFlips / std::max<std::size_t>(1, calls * static_cast<std::size_t>(K)), 100.0 * fvFlips / std::max<std::size_t>(1, calls),
                100.0 * cFlips / std::max<std::size_t>(1, calls));
    std::printf("  contacts %zu: SR spread of P (max of %d samples) p50 %.2e p99 %.2e max %.2e m\n", contacts, K, q(spreads, 0.5), q(spreads, 0.99), q(spreads, 1.0));
    std::printf("  |dP| / SR spread: FV p50 %.2f p99 %.2f max %.2f, outside the spread %.2f%%; C p50 %.2f p99 %.2f max %.2f, outside %.2f%%\n", q(fvRatio, 0.5),
                q(fvRatio, 0.99), q(fvRatio, 1.0), 100.0 * fvOut / std::max<std::size_t>(1, fvRatio.size()), q(cRatio, 0.5), q(cRatio, 0.99), q(cRatio, 1.0),
                100.0 * cOut / std::max<std::size_t>(1, cRatio.size()));
    return 0;
}

// M3_POSES=1: the second argument of each pair is a pose stream (twin_step_bench TWB_DUMP): the real engine's
// Detect calls. No FV reference there: the fast path is checked bit for bit against DetectShapesRef.
int PoseStream(const char *packPath, const char *posePath) {
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
    tw::bench::Pmu pmu;   // cycles of whole passes (docs/twin/harness.md "Measuring cycles")
    if (!pmu.ok()) std::fprintf(stderr, "PMU unavailable: cycle figures are 0\n");
    collide::CarShapes car;
    // the stream has no shape table: StadiumCar's wheels are shapes 4-7 (equal radii), the bodies 0-3
    for (int k = 0; k < 8; ++k)
        car.shape[k].wheel = (N && k >= 4 && std::memcmp(poses[k].radii, poses[4].radii, 12) == 0) ? k - 4 : -1;
    collide::CollideOptions opt;
    opt.checkFrames = std::getenv("M3_CHECKFRAMES") != nullptr;
    if (std::getenv("M3_PATCHES")) opt.patches = collide::kPatchMerged;
    opt.closedWheelFace = std::getenv("M3_R7") != nullptr;
    if (std::getenv("M3_LEAD")) opt.leadPoses = static_cast<float>(std::atof(std::getenv("M3_LEAD")));
    if (std::getenv("M3_MARGIN")) opt.unionMargin = static_cast<float>(std::atof(std::getenv("M3_MARGIN")));
    collide::ExactCollision stage;
    if (!stage.Init(view, car, &err, opt)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (std::getenv("M3_ABL")) {   // ablation timing: stop after the AABB filter / skip the prefilter survivors / full
        const int stops[6] = {5, 4, 3, 1, 2, 0};   // after the boxes, the memo, the shape filters, the filter, the frames and pairs; full
        double t[6] = {1e30, 1e30, 1e30, 1e30, 1e30, 1e30};
        collide::ExactCollision abl[5];
        for (int a = 0; a < 5; ++a) {
            collide::CollideOptions o = opt;
            o.profileStop = stops[a];
            if (!abl[a].Init(view, car, &err, o)) return 1;
        }
        step::SubstepContacts x;
        for (int rep = 0; rep < 5; ++rep)
            for (int a = 0; a < 6; ++a) {
                const collide::ExactCollision &st = a < 5 ? abl[a] : stage;
                const auto p0 = pmu.Read();
                for (std::size_t i = 0; i < N; ++i) st.DetectShapes(&poses[8 * i], x);
                t[a] = std::min(t[a], double(tw::bench::Pmu::Delta(p0, pmu.Read()).cycles));
            }
        std::printf("  ablation per call (PMU cycles, best of 5): boxes %.0f, + memo %.0f, + shape filters %.0f, + filter %.0f, + frames %.0f, + tests and merge (full) %.0f\n",
                    t[0] / N, t[1] / N, t[2] / N, t[3] / N, t[4] / N, t[5] / N);
    }
    step::SubstepContacts a, b;
    if (std::getenv("M3_PASSES"))
        for (int r = 0; r < std::atoi(std::getenv("M3_PASSES")); ++r)
            for (std::size_t i = 0; i < N; ++i) stage.DetectShapes(&poses[8 * i], a);
    if (std::getenv("M3_TSC")) {   // per-call rdtsc sum over one pass (as the engine bench times a call), against the pass's wall time
        const auto w0 = Clock::now();
        const std::uint64_t r0 = __rdtsc();
        std::uint64_t sum = 0;
        for (std::size_t i = 0; i < N; ++i) {
            const std::uint64_t a0 = __rdtsc();
            stage.DetectShapes(&poses[8 * i], a);
            sum += __rdtsc() - a0;
        }
        const double ns = std::chrono::duration<double, std::nano>(Clock::now() - w0).count();
        const double tscPerNs = double(__rdtsc() - r0) / ns;
        const double g = CoreGHz();
        std::printf("  rdtsc pass: %.0f cycles per call (sum of calls), %.0f (wall)\n", double(sum) / tscPerNs * g / double(N), ns * g / double(N));
    }
    if (std::getenv("M3_THRASH")) {   // data-cache thrash of KB between calls (the tick's other stages), timed: the call minus the thrash
        const std::size_t kb = static_cast<std::size_t>(std::atoi(std::getenv("M3_THRASH")));
        std::vector<std::uint64_t> buf(kb * 1024 / 8 + 8, 1);
        volatile std::uint64_t sink = 0;
        auto thrash = [&]() { std::uint64_t x = 0; for (std::size_t j = 0; j < buf.size(); j += 8) { x += buf[j]; buf[j] = x; } sink = sink + x; };
        double tCall = 1e30, tThrash = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            const auto p0 = pmu.Read();
            for (std::size_t i = 0; i < N; ++i) { thrash(); stage.DetectShapes(&poses[8 * i], a); }
            const auto p1 = pmu.Read();
            for (std::size_t i = 0; i < N; ++i) thrash();
            const auto p2 = pmu.Read();
            tCall = std::min(tCall, double(tw::bench::Pmu::Delta(p0, p1).cycles));
            tThrash = std::min(tThrash, double(tw::bench::Pmu::Delta(p1, p2).cycles));
        }
        std::printf("  with %zu KB thrashed between calls: %.0f cycles per call (thrash alone %.0f)\n", kb, (tCall - tThrash) / N, tThrash / N);
    }
    double best[2] = {1e30, 1e30}, ins[2] = {0, 0};
    for (int rep = 0; rep < 5; ++rep)
        for (int path = 0; path < 2; ++path) {
            const auto p0 = pmu.Read();
            for (std::size_t i = 0; i < N; ++i) { if (path) stage.DetectShapesRef(&poses[8 * i], a); else stage.DetectShapes(&poses[8 * i], a); }
            const auto d = tw::bench::Pmu::Delta(p0, pmu.Read());
            if (double(d.cycles) < best[path]) { best[path] = double(d.cycles); ins[path] = double(d.instructions); }
        }
    const double ghz = CoreGHz();
    collide::CollideStats st;
    std::size_t differ = 0, contacts = 0;
    for (std::size_t i = 0; i < N; ++i) {
        stage.DetectShapes(&poses[8 * i], a, &st);
        stage.DetectShapesRef(&poses[8 * i], b);
        contacts += static_cast<std::size_t>(a.n);
        bool same = a.n == b.n && a.overflow == b.overflow;
        for (int c = 0; same && c < a.n; ++c) same = std::memcmp(&a.c[c], &b.c[c], sizeof a.c[c]) == 0;
        differ += !same;
    }
    const double kk = ghz / 3.893;
    std::printf("## %s: %zu engine Detect calls, %.2f contacts/call; fast vs reference: %zu calls differ bitwise\n", posePath, N, double(contacts) / N, differ);
    std::printf("  cost per call (PMU cycles, best of 5): fast %.0f, reference %.0f; instructions %.0f / %.0f; prefilter survivors %.2f, frames %.2f per call; memo refills %.3f per call\n",
                best[0] / N, best[1] / N, ins[0] / N, ins[1] / N, double(st.triangleTests) / N, double(st.frames) / N,
                double(st.memoRefills) / std::max<double>(1.0, double(st.memoRefills + st.memoReuses)));
    std::printf("  split (stats pass, inflated by its own timers): boxes %.0f, memo %.0f, filter %.0f, frames %.0f, tests %.0f, merge+out %.0f\n",
                st.tBoxes * kk / N, st.tMemo * kk / N, st.tFilter * kk / N, st.tFrames * kk / N, st.tTests * kk / N, st.tMerge * kk / N);
    const double n = double(N);
    std::printf("  per call: memo entries %.1f; body candidates %.1f, survivors %.2f, raw %.2f; wheel survivors %.2f, raw %.2f; FV frames %.2f; "
                "Decide reject %.2f, FACE %.2f, FV form %.2f; merged %.2f\n",
                st.memoEntries / n, st.bodyCandidates / n, st.bodySurvivors / n, st.bodyRaw / n, st.wheelSurvivors / n, st.wheelRaw / n,
                st.fvFrames / n, st.decReject / n, st.decFace / n, st.decFv / n, st.merged / n);
    std::printf("  patch tests %.2f per call; survivors %.2f, pass-A decisions %.2f; R7 closed-form wheel FACE %llu of %llu wheel shape-calls (%.1f%%)\n",
                st.patchTests / n, st.triangleTests / n, (st.decFace + st.decFv + st.decReject) / n, static_cast<unsigned long long>(st.closedWheelFaces),
                static_cast<unsigned long long>(st.wheelCalls), 100.0 * st.closedWheelFaces / std::max<double>(1.0, double(st.wheelCalls)));
    std::printf("  absorbed feature pairs skipped %.2f per call; frame mismatches %llu%s\n", st.absorbedSkips / n,
                static_cast<unsigned long long>(stage.FrameMismatches()), opt.checkFrames ? "" : " (M3_CHECKFRAMES off)");
    return 0;
}

int main(int argc, char **argv) {
    if (std::getenv("M3_REFD")) {
        if (argc < 4 || (argc - 1) % 3) { std::fprintf(stderr, "usage: M3_REFD=1 tw_m3check PACK CARSHAPES CPOSES [...]\n"); return 2; }
        const int K = std::getenv("M3_REFD_SR") ? std::atoi(std::getenv("M3_REFD_SR")) : 0;
        const int every = std::getenv("M3_REFD_EVERY") ? std::atoi(std::getenv("M3_REFD_EVERY")) : 10;
        for (int a = 1; a + 2 < argc; a += 3)
            if (K > 0 ? RefSR(argv[a], argv[a + 1], argv[a + 2], K, std::max(1, every)) : RefDouble(argv[a], argv[a + 1], argv[a + 2])) return 1;
        return 0;
    }
    if (argc < 3 || (argc - 1) % 2) { std::fprintf(stderr, "usage: tw_m3check PACK REC [PACK REC ...]\n"); return 2; }
    if (std::getenv("M3_POSES")) {
        for (int a = 1; a + 1 < argc; a += 2) if (PoseStream(argv[a], argv[a + 1])) return 1;
        return 0;
    }
    const int maxExc = std::getenv("M3_EXC") ? std::atoi(std::getenv("M3_EXC")) : 0;
    for (int a = 1; a + 1 < argc; a += 2) {
        world::WorldPack pack;
        std::string err;
        if (!world::LoadWorldPack(argv[a], &pack, &err)) { std::fprintf(stderr, "pack %s: %s\n", argv[a], err.c_str()); return 1; }
        world::WorldPackView view;
        if (!world::OpenWorldPack(pack, &view, &err, true)) { std::fprintf(stderr, "open %s: %s\n", argv[a], err.c_str()); return 1; }
        Recording rec;
        if (!LoadRecording(argv[a + 1], &rec)) { std::fprintf(stderr, "cannot load %s\n", argv[a + 1]); return 1; }
        collide::CarShapes car;   // only radii/poses of recorded shapes are used by DetectShapes
        for (int k = 0; k < 8; ++k) car.shape[k].wheel = k >= 4 ? k - 4 : -1;   // StadiumCar: wheels are shapes 4-7
        collide::ExactCollision stage;
        collide::CollideOptions opt;
        opt.excludeMovingPair = std::getenv("M3_NOFLAGS") != nullptr;
        if (std::getenv("M3_PATCHES")) opt.patches = collide::kPatchMerged;
    opt.closedWheelFace = std::getenv("M3_R7") != nullptr;
            if (std::getenv("M3_LEAD")) opt.leadPoses = static_cast<float>(std::atof(std::getenv("M3_LEAD")));
        if (std::getenv("M3_MARGIN")) opt.unionMargin = static_cast<float>(std::atof(std::getenv("M3_MARGIN")));
        const bool useRef = std::getenv("M3_REF") != nullptr;
        const auto b0 = Clock::now();
        if (!stage.Init(view, car, &err, opt)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const double buildMs = std::chrono::duration<double, std::milli>(Clock::now() - b0).count();
        const std::size_t N = rec.substeps.size();
        // timed pass
        step::SubstepContacts out;
        if (std::getenv("M3_PASSES")) {   // profiling: repeat the detection passes
            collide::ShapePose sq[8];
            for (int r = 0; r < std::atoi(std::getenv("M3_PASSES")); ++r)
                for (std::size_t i = 0; i < N; ++i) { Poses(rec.substeps[i], sq); stage.DetectShapes(sq, out); }
        }
        const double ghz0 = CoreGHz();
        const auto w0 = Clock::now();
        collide::ShapePose sp[8];
        for (std::size_t i = 0; i < N; ++i) {
            Poses(rec.substeps[i], sp);
            if (useRef) stage.DetectShapesRef(sp, out); else stage.DetectShapes(sp, out);
        }
        const double ns = std::chrono::duration<double, std::nano>(Clock::now() - w0).count();
        const double ghz = 0.5 * (ghz0 + CoreGHz());
        // comparison pass
        collide::CollideStats st;
        std::size_t identical = 0, withinMetric = 0, subsWith = 0, shapeSubs = 0, shapeExc = 0, actorChecked = 0, actorEq = 0, overflow = 0, maxN = 0;
        std::size_t fvTot = 0, twTot = 0, printed = 0;
        std::size_t excClass[5] = {0, 0, 0, 0, 0};   // duplicate, internal edge, other count, values, twin more
        double maxDP = 0.0, maxAng = 0.0, maxDS = 0.0;  // matched contacts: max |dP| (m), normal angle (deg), |dSep| (m)
        for (std::size_t i = 0; i < N; ++i) {
            const tw::Substep &s = rec.substeps[i];
            Poses(s, sp);
            if (useRef) stage.DetectShapesRef(sp, out, &st); else stage.DetectShapes(sp, out, &st);
            overflow += out.overflow; maxN = std::max<std::size_t>(maxN, static_cast<std::size_t>(out.n));
            bool ident = true, metric = true;
            const bool any = s.mergedCount > 0 || out.n > 0;
            subsWith += any;
            for (int k = 0; k < 8; ++k) {
                std::vector<const FvContact *> F;
                for (std::uint32_t c = s.firstMerged; c < s.firstMerged + s.mergedCount; ++c) if (rec.merged[c].shape == k) F.push_back(&rec.merged[c]);
                std::vector<const model::RespContact *> W;
                for (int c = 0; c < out.n; ++c) if (out.c[c].shape == k) W.push_back(&out.c[c]);
                if (F.empty() && W.empty()) continue;
                ++shapeSubs; fvTot += F.size(); twTot += W.size();
                bool sIdent = F.size() == W.size();
                for (std::size_t c = 0; sIdent && c < F.size(); ++c) {
                    const FvContact &f = *F[c]; const model::RespContact &w = *W[c];
                    sIdent = Bits(f.P, w.P, 3) && Bits(f.N, w.N, 3) && Bits(f.Sep, w.Sep, 3) && f.material == w.material &&
                             (f.primary != 0) == ((w.flags & model::kRespPrimary) != 0);
                }
                // metric match (nearest point, as §6)
                bool sMetric = F.size() == W.size();
                std::vector<bool> used(W.size(), false);
                std::vector<const FvContact *> unmatched;
                for (const FvContact *f : F) {
                    int best = -1; float bc = 1e30f;
                    for (std::size_t q = 0; q < W.size(); ++q) {
                        if (used[q]) continue;
                        const float dx = W[q]->P[0] - f->P[0], dy = W[q]->P[1] - f->P[1], dz = W[q]->P[2] - f->P[2];
                        const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (d < 0.05f && d < bc) { bc = d; best = static_cast<int>(q); }
                    }
                    if (best < 0) { sMetric = false; unmatched.push_back(f); continue; }
                    used[static_cast<std::size_t>(best)] = true;
                    const model::RespContact &w = *W[static_cast<std::size_t>(best)];
                    const float dot = std::fmax(-1.0f, std::fmin(1.0f, w.N[0] * f->N[0] + w.N[1] * f->N[1] + w.N[2] * f->N[2]));
                    const float ds = std::sqrt((w.Sep[0] - f->Sep[0]) * (w.Sep[0] - f->Sep[0]) + (w.Sep[1] - f->Sep[1]) * (w.Sep[1] - f->Sep[1]) +
                                               (w.Sep[2] - f->Sep[2]) * (w.Sep[2] - f->Sep[2]));
                    maxDP = std::max(maxDP, double(bc)); maxAng = std::max(maxAng, double(std::acos(dot)) * 57.29578); maxDS = std::max(maxDS, double(ds));
                    if (bc > 1e-3f || std::acos(dot) * 57.29578f > 0.5f || ds > 1e-4f || w.material != f->material ||
                        (f->primary != 0) != ((w.flags & model::kRespPrimary) != 0))
                        sMetric = false;
                    if (f->actor != kNoActor) {
                        ++actorChecked;
                        const world::Actor &fa = view.actors[f->actor];
                        const bool trig = (w.flags & model::kRespTrigger) != 0;
                        actorEq += trig ? (fa.trigger == w.peerId) : (fa.peerId == w.peerId && Bits(fa.peerZ, w.peerZ, 3));
                    }
                }
                if (!sMetric) {
                    ++shapeExc;
                    // classes (m2-broadphase.md §6): FV's extra contact coincides with another FV contact (a per-triangle
                    // duplicate), or is a shallow feature next to a FACE (an internal-edge feature); twin extra; values
                    int cls = 3;   // values only
                    if (F.size() > W.size() || !unmatched.empty()) {
                        cls = 2;   // other count difference
                        for (const FvContact *u : unmatched) {
                            bool dup = false, internal = false;
                            for (const FvContact *g : F) {
                                if (g == u) continue;
                                const float dx = g->P[0] - u->P[0], dy = g->P[1] - u->P[1], dz = g->P[2] - u->P[2];
                                if (dx * dx + dy * dy + dz * dz < 1e-6f) dup = true;
                                if (g->primary && !u->primary) {
                                    const float sl = std::sqrt(u->Sep[0] * u->Sep[0] + u->Sep[1] * u->Sep[1] + u->Sep[2] * u->Sep[2]);
                                    if (sl < 5e-3f) internal = true;
                                }
                            }
                            if (dup) { cls = 0; break; }
                            if (internal) cls = 1;
                        }
                    } else if (W.size() > F.size()) {
                        cls = 4;   // the twin has more
                    }
                    ++excClass[cls];
                }
                ident = ident && sIdent;
                metric = metric && sMetric;
            }
            identical += ident; withinMetric += metric;
            if (!ident && static_cast<int>(printed) < maxExc) {
                ++printed;
                std::printf("  substep %zu (tick %u): %s\n", i, s.tick, metric ? "within metric, bits differ" : "OUTSIDE metric");
                for (std::uint32_t c = s.firstMerged; c < s.firstMerged + s.mergedCount; ++c) {
                    const FvContact &f = rec.merged[c];
                    std::printf("    FV   sh %u %s m%u P(%.6f %.6f %.6f) N(%.6f %.6f %.6f) Sep(%.3e %.3e %.3e)\n", f.shape, f.primary ? "FACE" : "feat", f.material, f.P[0], f.P[1], f.P[2],
                                f.N[0], f.N[1], f.N[2], f.Sep[0], f.Sep[1], f.Sep[2]);
                }
                for (int c = 0; c < out.n; ++c) {
                    const model::RespContact &w = out.c[c];
                    std::printf("    twin sh %u %s m%u P(%.6f %.6f %.6f) N(%.6f %.6f %.6f) Sep(%.3e %.3e %.3e)\n", w.shape, (w.flags & 1) ? "FACE" : "feat", w.material, w.P[0], w.P[1], w.P[2],
                                w.N[0], w.N[1], w.N[2], w.Sep[0], w.Sep[1], w.Sep[2]);
                }
            }
        }
        std::printf("## %s: %zu substeps (%u ticks), %zu tris; BVH build %.0f ms, %.1f MB\n", rec.name.c_str(), N, rec.ticks, view.tris.size(), buildMs, stage.Bytes() / 1e6);
        std::printf("  substeps bit-identical %.4f%% (%zu differ), within metric %.4f%%; (substep,shape) exceptions %zu of %zu (%.4f%%); contacts FV %zu twin %zu; "
                    "peer actor equal %zu/%zu; overflow %zu, max %zu per substep\n",
                    100.0 * identical / N, N - identical, 100.0 * withinMetric / N, shapeExc, shapeSubs, 100.0 * shapeExc / std::max<std::size_t>(1, shapeSubs), fvTot, twTot,
                    actorEq, actorChecked, overflow, maxN);
        if (opt.closedWheelFace)
            std::printf("  R7 closed-form wheel FACE: %llu of %llu wheel shape-substeps (%.1f%%)\n", static_cast<unsigned long long>(st.closedWheelFaces),
                        static_cast<unsigned long long>(st.wheelCalls), 100.0 * st.closedWheelFaces / std::max<double>(1.0, double(st.wheelCalls)));
        std::printf("  matched contacts: max |dP| %.2e m, normal %.2e deg, |dSep| %.2e m\n", maxDP, maxAng, maxDS);
        std::printf("  exception classes, %% of (substep,shape) pairs: duplicate %.4f, internal edge %.4f, other count %.4f, values %.4f, twin more %.4f\n",
                    100.0 * excClass[0] / std::max<std::size_t>(1, shapeSubs), 100.0 * excClass[1] / std::max<std::size_t>(1, shapeSubs),
                    100.0 * excClass[2] / std::max<std::size_t>(1, shapeSubs), 100.0 * excClass[3] / std::max<std::size_t>(1, shapeSubs),
                    100.0 * excClass[4] / std::max<std::size_t>(1, shapeSubs));
        std::printf("  cost %.0f cycles per substep (%.2f GHz measured) = %.0f per tick at FV's count; candidates %.1f, triangle tests %.1f, raw %.2f, merged %.2f per substep%s\n",
                    ns * ghz / N, ghz, ns * ghz / rec.ticks, double(st.candidates) / N, double(st.triangleTests) / N, double(st.rawContacts) / N, double(st.merged) / N,
                    opt.excludeMovingPair ? " [moving-pair corpus excluded]" : "");
        std::printf("  path %s; union memo refills %llu, reuses %llu (margin %.2f m)\n", useRef ? "REFERENCE (query per shape, FV form)" : "fast (union memo + fast test)",
                    static_cast<unsigned long long>(st.memoRefills), static_cast<unsigned long long>(st.memoReuses), opt.unionMargin);
        {
            const double kk = ghz / 3.893;   // TSC (3.89 GHz) -> core cycles
            std::printf("  split per substep (stats pass, core cycles): boxes %.0f, memo %.0f, filter %.0f, frames %.0f (%.2f per substep), tests %.0f, merge+out %.0f\n",
                        st.tBoxes * kk / N, st.tMemo * kk / N, st.tFilter * kk / N, st.tFrames * kk / N, double(st.frames) / N, st.tTests * kk / N, st.tMerge * kk / N);
        }
        std::printf("  moving-pair corpus (Flags): %llu candidate triangles, %llu of %llu shape queries touched it\n",
                    static_cast<unsigned long long>(st.movingPairCandidates), static_cast<unsigned long long>(st.queriesTouchingMovingPair),
                    static_cast<unsigned long long>(st.calls * 8));
        std::fflush(stdout);
    }
    return 0;
}
