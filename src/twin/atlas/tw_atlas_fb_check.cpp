// S2 check and bench (architect D): the atlas fallback on every shape of recorded engine poses, against
//   (a) the exact stage (FV's float bits, collide::ExactCollision::DetectPose): discrete equality per (call, shape);
//   (b) B's double reference (collide::ref::Detect): payload error of the fallback and of FV against the double board;
//   (c) optionally K stochastic-rounding samples (ref::DetectSR): each fallback contact within 2x the farthest member
//       (the acceptance test's band), per component group (P, N, Sep), and discrete values the population produced.
// Timing (PMU, user cycles): DetectShape over the shapes with candidates; candidates come from the stage's BVH (the
// atlas supplies them in the engine, so their cost is not S2's).
//
// usage: tw_atlas_fb_check PACK CARSHAPES CPOSES     (cposes: 11 floats per call: pos, quat xyzw, absorb x4)
// env: FB_SR=K (stochastic samples on every FB_EVERY-th call, default 10), FB_TIME=1 (timing pass), FB_MAX=N calls,
//      FB_SHOW=N (print the first N discrete mismatches)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tw_atlas_fb.h"
#include "tw_collide_ref.h"
#include "twin_pmu.h"
#include "tw_world_pack.h"
#include "tw_world.h"
#include "tw_exact_fast.h"

using namespace tw;

namespace {

struct Pc { const model::RespContact *c; };

double Dist3(const float *a, const double *b) {
    const double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
    return std::sqrt(x * x + y * y + z * z);
}
double Dist3f(const float *a, const float *b) {
    const double x = double(a[0]) - b[0], y = double(a[1]) - b[1], z = double(a[2]) - b[2];
    return std::sqrt(x * x + y * y + z * z);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s PACK CARSHAPES CPOSES\n", argv[0]); return 2; }
    world::WorldPack pack;
    std::string err;
    if (!world::LoadWorldPack(argv[1], &pack, &err)) { std::fprintf(stderr, "pack: %s\n", err.c_str()); return 1; }
    world::WorldPackView view;
    if (!world::OpenWorldPack(pack, &view, &err, true)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    collide::CarShapes car;
    {
        std::FILE *f = std::fopen(argv[2], "rb");
        if (!f || std::fread(&car, sizeof car, 1, f) != 1) { std::fprintf(stderr, "cannot read %s\n", argv[2]); if (f) std::fclose(f); return 1; }
        std::fclose(f);
    }
    std::vector<float> cp;
    {
        std::FILE *f = std::fopen(argv[3], "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
        float b[11];
        while (std::fread(b, sizeof b, 1, f) == 1) cp.insert(cp.end(), b, b + 11);
        std::fclose(f);
    }
    if (const char *recPath = std::getenv("FB_REC")) {
        // B's FV recording (twrec): every recorded substep's 8 shape poses through S2, against FV's merged contacts
        collide::ExactCollision st0;
        if (!st0.Init(view, car, &err, {})) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        atlas::Fallback f0;
        f0.Init(st0, &err);
        Recording rec;
        if (!LoadRecording(recPath, &rec)) { std::fprintf(stderr, "cannot load %s\n", recPath); return 1; }
        std::vector<std::uint32_t> cd;
        step::SubstepContacts out;
        std::size_t pairs = 0, bad = 0;
        for (std::size_t si = 0; si < rec.substeps.size(); ++si) {
            const Substep &ss = rec.substeps[si];
            collide::ShapePose sh[8];
            for (int k = 0; k < 8; ++k) {
                for (int c = 0; c < 3; ++c) for (int i = 0; i < 3; ++i) sh[k].world.r[3 * c + i] = ss.shapes[k].axis[c][i];
                for (int i = 0; i < 3; ++i) { sh[k].world.t[i] = ss.shapes[k].t[i]; sh[k].radii[i] = ss.shapes[k].r[i]; }
            }
            for (int k = 0; k < 8; ++k) {
                f0.BvhCandidates(sh[k], &cd);
                out.n = 0; out.overflow = false;
                if (!cd.empty()) f0.DetectShape(sh, k, cd.data(), cd.size(), nullptr, out, nullptr);
                std::vector<const FvContact *> F;
                for (std::uint32_t c = 0; c < ss.mergedCount; ++c) if (rec.merged[ss.firstMerged + c].shape == k) F.push_back(&rec.merged[ss.firstMerged + c]);
                if (F.empty() && out.n == 0) continue;
                ++pairs;
                bool b = static_cast<std::size_t>(out.n) != F.size();
                for (int c = 0; !b && c < out.n; ++c) {
                    const double dx = out.c[c].P[0] - F[c]->P[0], dy = out.c[c].P[1] - F[c]->P[1], dz = out.c[c].P[2] - F[c]->P[2];
                    b = ((out.c[c].flags & 1) != F[c]->primary) || std::sqrt(dx * dx + dy * dy + dz * dz) > 1e-3;
                }
                if (b) {
                    ++bad;
                    std::printf("  substep %zu tick %u shape %d: FV %zu contacts, D %d\n", si, ss.tick, k, F.size(), out.n);
                    for (auto *c : F) std::printf("    FV %s m%u P(%.5f %.5f %.5f) N(%.4f %.4f %.4f)\n", c->primary ? "FACE" : "feat", c->material, c->P[0], c->P[1], c->P[2], c->N[0], c->N[1], c->N[2]);
                    for (int c = 0; c < out.n; ++c)
                        std::printf("    D  %s m%u P(%.5f %.5f %.5f) N(%.4f %.4f %.4f)\n", (out.c[c].flags & 1) ? "FACE" : "feat", out.c[c].material, out.c[c].P[0], out.c[c].P[1],
                                    out.c[c].P[2], out.c[c].N[0], out.c[c].N[1], out.c[c].N[2]);
                    atlas::ShapeLocal sl;
                    atlas::MakeShapeLocal(sh[k].world.r, sh[k].world.t, sh[k].radii, &sl);
                    for (std::uint32_t t : cd) {
                        const atlas::FaceTri &ft = f0.Faces()[t];
                        atlas::FaceOut fo; float tie; int why;
                        const bool face = atlas::ClosedFace(ft, sl, &fo, &tie, &why);
                        if (why == atlas::kFaceBehind || why == atlas::kFaceFar) continue;
                        const world::TriInfo &ti = view.triInfo[t];
                        const world::Instance &in = view.instances[ti.instance];
                        exact::Frame fr;
                        exact::MakeFrameFv(sh[k].world.r, sh[k].world.t, sh[k].radii, in.iso, &fr);
                        const world::Model &md = view.models[in.model];
                        const world::ModelTri &mt = view.modelTris[md.firstTri + ti.modelTri];
                        exact::TriVerts tv;
                        for (int c = 0; c < 3; ++c) { const world::Vec3f &q = view.modelVerts[md.firstVert + mt.v[c]]; tv.v[c] = exact::V{q.x, q.y, q.z}; }
                        std::vector<exact::Raw> raw;
                        const int got = exact::CollideFv(fr, tv, t, &raw);
                        std::printf("    tri %u: closed why %d face %d d %.7f se (%.2e %.2e %.2e) tie %.2e valid %u rot %u | FV form: %d %s\n", t, why, int(face), fo.d, fo.se[0], fo.se[1], fo.se[2],
                                    tie, ft.valid, unsigned(in.rotation), got, got ? (raw[0].primary ? "FACE" : "feat") : "-");
                    }
                }
            }
        }
        std::printf("## %s: %zu (substep, shape) with contacts, %zu differ from FV's recorded merged contacts\n", recPath, pairs, bad);
        return 0;
    }
    std::size_t N = cp.size() / 11;
    if (std::getenv("FB_MAX")) N = std::min<std::size_t>(N, std::strtoul(std::getenv("FB_MAX"), nullptr, 10));
    collide::ExactCollision stage;
    if (!stage.Init(view, car, &err, {})) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    atlas::Fallback fb;
    if (!fb.Init(stage, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int K = std::getenv("FB_SR") ? std::atoi(std::getenv("FB_SR")) : 0;
    const int every = std::getenv("FB_EVERY") ? std::atoi(std::getenv("FB_EVERY")) : 10;
    const int show = std::getenv("FB_SHOW") ? std::atoi(std::getenv("FB_SHOW")) : 0;

    atlas::FallbackStats st;
    step::SubstepContacts fv, mine;
    collide::ref::DetectionD dd, ds;
    std::vector<std::uint32_t> cand;
    std::size_t calls = 0, shapePairs = 0, mismFv = 0, mismRef = 0, mismRefNear = 0, shown = 0;
    double maxPfv = 0, maxPmine = 0, maxNfv = 0, maxNmine = 0, maxSfv = 0, maxSmine = 0;
    double sumPfv = 0, sumPmine = 0; std::size_t nPay = 0;
    // stochastic-rounding band (acceptance form, per contact): |mine - double| <= 2 x max member distance (floor 1 ulp)
    double srNsq[3] = {0, 0, 0}, srNsum[3] = {0, 0, 0}, biasMine[3] = {0, 0, 0}, biasFv[3] = {0, 0, 0}; std::size_t srNcnt = 0, biasCnt = 0, srUnscored = 0;
    std::size_t srContacts = 0, srOutP = 0, srOutN = 0, srOutS = 0, srCalls = 0, srDiscreteOut = 0, srFvOutP = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const float *pf = &cp[11 * i];
        collide::ShapePose shapes[8];
        collide::ShapePoses(car, pf, pf + 3, pf + 7, shapes);
        stage.DetectPose(pf, pf + 3, pf + 7, fv);
        mine.n = 0; mine.overflow = false;
        const double origin[3] = {pf[0], pf[1], pf[2]};
        for (int k = 0; k < 8; ++k) {
            fb.BvhCandidates(shapes[k], &cand);
            if (cand.empty()) continue;
            fb.DetectShape(shapes, k, cand.data(), cand.size(), origin, mine, &st);
        }
        const double pd[3] = {pf[0], pf[1], pf[2]}, qd[4] = {pf[3], pf[4], pf[5], pf[6]}, ad[4] = {pf[7], pf[8], pf[9], pf[10]};
        collide::ref::Detect(stage, pd, qd, ad, &dd);
        ++calls;
        // per shape: mine vs FV (discrete), mine and FV vs double (payload)
        for (int k = 0; k < 8; ++k) {
            std::vector<const model::RespContact *> F, Mn;
            std::vector<const collide::ref::ContactD *> D;
            for (int c = 0; c < fv.n; ++c) if (fv.c[c].shape == k) F.push_back(&fv.c[c]);
            for (int c = 0; c < mine.n; ++c) if (mine.c[c].shape == k) Mn.push_back(&mine.c[c]);
            for (const auto &c : dd.contacts) if (c.shape == k) D.push_back(&c);
            if (F.empty() && Mn.empty() && D.empty()) continue;
            ++shapePairs;
            auto sameDiscrete = [](const model::RespContact *a, unsigned fl, unsigned mat) { return a->flags == fl && a->material == mat; };
            bool badFv = F.size() != Mn.size();
            for (std::size_t j = 0; !badFv && j < F.size(); ++j)
                if (!sameDiscrete(Mn[j], F[j]->flags, F[j]->material) || Dist3f(Mn[j]->P, F[j]->P) > 1e-3) badFv = true;
            bool badRef = D.size() != Mn.size();
            for (std::size_t j = 0; !badRef && j < D.size(); ++j)
                if (!sameDiscrete(Mn[j], D[j]->flags, D[j]->material) || Dist3(Mn[j]->P, D[j]->P) > 1e-3) badRef = true;
            if (badFv) ++mismFv;
            if (badRef) {
                ++mismRef;
                if (dd.perShape[k].value < 1e-3) ++mismRefNear;
            }
            if ((badFv || badRef) && shown < static_cast<std::size_t>(show)) {
                ++shown;
                std::printf("  call %zu shape %d: FV %zu, D %zu, double %zu contacts; ref shape margin %.2e kind %d\n", i, k, F.size(), Mn.size(), D.size(),
                            dd.perShape[k].value, int(dd.perShape[k].kind));
                for (auto *c : F) std::printf("    FV  %s m%u P(%.5f %.5f %.5f) N(%.4f %.4f %.4f)\n", (c->flags & 1) ? "FACE" : "feat", c->material, c->P[0], c->P[1], c->P[2], c->N[0], c->N[1], c->N[2]);
                for (auto *c : Mn) std::printf("    D   %s m%u P(%.5f %.5f %.5f) N(%.4f %.4f %.4f)\n", (c->flags & 1) ? "FACE" : "feat", c->material, c->P[0], c->P[1], c->P[2], c->N[0], c->N[1], c->N[2]);
                for (auto *c : D) std::printf("    dbl %s m%u P(%.5f %.5f %.5f) N(%.4f %.4f %.4f)\n", (c->flags & 1) ? "FACE" : "feat", c->material, c->P[0], c->P[1], c->P[2], c->N[0], c->N[1], c->N[2]);
            }
            if (!badRef && !badFv) {
                for (std::size_t j = 0; j < D.size(); ++j) {
                    const double ePm = Dist3(Mn[j]->P, D[j]->P), ePf = Dist3(F[j]->P, D[j]->P);
                    const double eNm = Dist3(Mn[j]->N, D[j]->N), eNf = Dist3(F[j]->N, D[j]->N);
                    const double eSm = Dist3(Mn[j]->Sep, D[j]->Sep), eSf = Dist3(F[j]->Sep, D[j]->Sep);
                    maxPmine = std::max(maxPmine, ePm); maxPfv = std::max(maxPfv, ePf);
                    maxNmine = std::max(maxNmine, eNm); maxNfv = std::max(maxNfv, eNf);
                    maxSmine = std::max(maxSmine, eSm); maxSfv = std::max(maxSfv, eSf);
                    sumPmine += ePm; sumPfv += ePf; ++nPay;
                }
            }
        }
        // stochastic-rounding population on every `every`-th call
        if (K > 0 && i % static_cast<std::size_t>(every) == 0) {
            ++srCalls;
            const sr::sr ps[3] = {pf[0], pf[1], pf[2]}, qs[4] = {pf[3], pf[4], pf[5], pf[6]}, as[4] = {pf[7], pf[8], pf[9], pf[10]};
            // per double contact (shape-major order) the farthest member distance; members with another discrete
            // list for that shape count as producing another discrete outcome (allowed values)
            std::vector<double> farP(dd.contacts.size(), 0.0), farN(dd.contacts.size(), 0.0), farS(dd.contacts.size(), 0.0);
            std::vector<std::vector<std::uint32_t>> altCounts(8);
            int membersMatched = 0;
            for (int s = 0; s < K; ++s) {
                sr::Seed(0x9E3779B97F4A7C15ull * (i + 1) + static_cast<std::uint64_t>(s) * 0x632BE59BD9B4E019ull);
                collide::ref::DetectSR(stage, ps, qs, as, &ds);
                for (int k = 0; k < 8; ++k) {
                    std::uint32_t cnt = 0;
                    for (const auto &c : ds.contacts) cnt += c.shape == k;
                    altCounts[k].push_back(cnt);
                }
                if (ds.contacts.size() != dd.contacts.size()) continue;
                bool same = true;
                for (std::size_t c = 0; c < dd.contacts.size() && same; ++c)
                    same = ds.contacts[c].shape == dd.contacts[c].shape && ds.contacts[c].flags == dd.contacts[c].flags && ds.contacts[c].tri == dd.contacts[c].tri;
                if (!same) continue;
                ++membersMatched;
                for (std::size_t c = 0; c < dd.contacts.size(); ++c) {
                    if (dd.contacts[c].flags & 1) {
                        for (int q = 0; q < 3; ++q) { const double e = ds.contacts[c].N[q] - dd.contacts[c].N[q]; srNsq[q] += e * e; srNsum[q] += e; }
                        ++srNcnt;
                    }
                    auto dist = [](const double *a, const double *b) {
                        const double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
                        return std::sqrt(x * x + y * y + z * z);
                    };
                    farP[c] = std::max(farP[c], dist(ds.contacts[c].P, dd.contacts[c].P));
                    farN[c] = std::max(farN[c], dist(ds.contacts[c].N, dd.contacts[c].N));
                    farS[c] = std::max(farS[c], dist(ds.contacts[c].Sep, dd.contacts[c].Sep));
                }
            }
            for (int k = 0; k < 8; ++k) {
                std::vector<const model::RespContact *> Mn, F;
                for (int c = 0; c < mine.n; ++c) if (mine.c[c].shape == k) Mn.push_back(&mine.c[c]);
                for (int c = 0; c < fv.n; ++c) if (fv.c[c].shape == k) F.push_back(&fv.c[c]);
                std::vector<std::size_t> Di;
                for (std::size_t c = 0; c < dd.contacts.size(); ++c) if (dd.contacts[c].shape == k) Di.push_back(c);
                if (Mn.size() != Di.size()) {   // allowed only if some member produced this count for the shape
                    bool allowed = false;
                    for (std::uint32_t a : altCounts[k]) allowed |= a == Mn.size();
                    if (!allowed) ++srDiscreteOut;
                    continue;
                }
                if (membersMatched == 0) { srUnscored += Di.size(); continue; }   // no member made the double's list: no continuous band
                for (std::size_t j = 0; j < Di.size(); ++j) {
                    const auto &d = dd.contacts[Di[j]];
                    ++srContacts;
                    auto ulp = [](double x) { const float f = static_cast<float>(std::fabs(x)); return double(std::nextafter(f, 1e30f) - f); };
                    const double fP = std::max(2.0 * farP[Di[j]], ulp(std::max({std::fabs(d.P[0]), std::fabs(d.P[1]), std::fabs(d.P[2])})));
                    const double fN = std::max(2.0 * farN[Di[j]], ulp(1.0));
                    const double fS = std::max(2.0 * farS[Di[j]], ulp(std::max({std::fabs(d.Sep[0]), std::fabs(d.Sep[1]), std::fabs(d.Sep[2])})));
                    if (Dist3(Mn[j]->P, d.P) > fP) {
                        ++srOutP;
                        if (std::getenv("FB_SHOWSR")) {   // the shape centre: float ShapePose vs double composition
                            const double x = pf[3], y = pf[4], z = pf[5], w = pf[6];
                            const double R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                                                    {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                                                    {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
                            const collide::CarShape &cs = car.shape[k];
                            double tl[3];
                            if (cs.wheel >= 0) { for (int q = 0; q < 3; ++q) tl[q] = car.wheelRest[cs.wheel].t[q]; tl[1] -= pf[7 + cs.wheel]; }
                            else for (int q = 0; q < 3; ++q) tl[q] = cs.bodyPose.t[q];
                            double tw[3];
                            for (int q = 0; q < 3; ++q) tw[q] = R[q][0] * tl[0] + R[q][1] * tl[1] + R[q][2] * tl[2] + pf[q];
                            {   // my closed form in double: float centre, double centre
                                const atlas::FaceTri &ft = fb.Faces()[d.tri];
                                double Md[3][3];
                                for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) Md[a][b] = double(shapes[k].world.r[3 * b + a]) * shapes[k].radii[b];
                                for (int pass = 0; pass < 2; ++pass) {
                                    const double tc[3] = {pass ? tw[0] : shapes[k].world.t[0], pass ? tw[1] : shapes[k].world.t[1], pass ? tw[2] : shapes[k].world.t[2]};
                                    const double sdd = ft.nd[0] * tc[0] + ft.nd[1] * tc[1] + ft.nd[2] * tc[2] - ft.rho;
                                    double mm[3], gg[3];
                                    for (int b = 0; b < 3; ++b) mm[b] = Md[0][b] * ft.nd[0] + Md[1][b] * ft.nd[1] + Md[2][b] * ft.nd[2];
                                    const double ss = mm[0] * mm[0] + mm[1] * mm[1] + mm[2] * mm[2];
                                    for (int a = 0; a < 3; ++a) gg[a] = Md[a][0] * mm[0] + Md[a][1] * mm[1] + Md[a][2] * mm[2];
                                    double Pd[3];
                                    for (int a = 0; a < 3; ++a) Pd[a] = tc[a] - sdd / ss * gg[a];
                                    if (pass == 0) {
                                        const world::Instance &in = view.instances[view.triInfo[d.tri].instance];
                                        double E = 0, dist = 0;
                                        for (int a = 0; a < 3; ++a) {
                                            for (int b = 0; b < 3; ++b) {
                                                double rr = 0;
                                                for (int c2 = 0; c2 < 3; ++c2) rr += double(in.iso[3 * c2 + a]) * in.iso[3 * c2 + b];
                                                E = std::max(E, std::fabs(rr - (a == b ? 1.0 : 0.0)));
                                            }
                                            dist += (tc[a] - in.iso[9 + a]) * (tc[a] - in.iso[9 + a]);
                                        }
                                        std::printf("    instance %u rotation code %u flags %u |RR^T - I| %.2e |t - t_inst| %.1f m -> shift ~%.1e\n", view.triInfo[d.tri].instance, in.rotation, in.flags, E,
                                                    std::sqrt(dist), E * std::sqrt(dist));
                                    }
                                    if (pass == 0) {
                                        const double offB = ft.nd[0] * d.P[0] + ft.nd[1] * d.P[1] + ft.nd[2] * d.P[2] - ft.rho;
                                        const double offM = ft.nd[0] * Mn[j]->P[0] + ft.nd[1] * Mn[j]->P[1] + ft.nd[2] * Mn[j]->P[2] - ft.rho;
                                        const double offF = F.size() == Di.size() ? ft.nd[0] * F[j]->P[0] + ft.nd[1] * F[j]->P[1] + ft.nd[2] * F[j]->P[2] - ft.rho : 0.0;
                                        int dupl = 0;
                                        for (const auto &c2 : dd.contacts) dupl += c2.shape == k && (c2.flags & 1);
                                        std::printf("    off-plane (tri %u plane): board %.2e mine %.2e FV %.2e; board FACEs on shape %d; board Sep.n %.3e mine Sep.n %.3e\n", d.tri, offB, offM, offF, dupl,
                                                    ft.nd[0] * d.Sep[0] + ft.nd[1] * d.Sep[1] + ft.nd[2] * d.Sep[2], ft.nd[0] * Mn[j]->Sep[0] + ft.nd[1] * Mn[j]->Sep[1] + ft.nd[2] * Mn[j]->Sep[2]);
                                    }
                                    std::printf("    closed form in double (%s centre): P - Pboard (%.2e %.2e %.2e), d %.6f tri %u\n", pass ? "double" : "float", Pd[0] - d.P[0], Pd[1] - d.P[1], Pd[2] - d.P[2],
                                                sdd / std::sqrt(ss), d.tri);
                                }
                            }
                            std::printf("    centre: float (%.6f %.6f %.6f) double (%.6f %.6f %.6f) |d| %.2e; P D-dbl (%.2e %.2e %.2e) FV-dbl (%.2e %.2e %.2e)\n",
                                        shapes[k].world.t[0], shapes[k].world.t[1], shapes[k].world.t[2], tw[0], tw[1], tw[2],
                                        std::sqrt((shapes[k].world.t[0] - tw[0]) * (shapes[k].world.t[0] - tw[0]) + (shapes[k].world.t[1] - tw[1]) * (shapes[k].world.t[1] - tw[1]) +
                                                  (shapes[k].world.t[2] - tw[2]) * (shapes[k].world.t[2] - tw[2])),
                                        Mn[j]->P[0] - d.P[0], Mn[j]->P[1] - d.P[1], Mn[j]->P[2] - d.P[2], F.size() == Di.size() ? F[j]->P[0] - d.P[0] : 0.0,
                                        F.size() == Di.size() ? F[j]->P[1] - d.P[1] : 0.0, F.size() == Di.size() ? F[j]->P[2] - d.P[2] : 0.0);
                        }
                        if (std::getenv("FB_SHOWSR"))
                            std::printf("    SR-out P: call %zu shape %d %s |dP| D %.2e FV %.2e band %.2e (far %.2e) |dSep| D %.2e FV %.2e; P(%.3f %.3f %.3f) N(%.3f %.3f %.3f) Sep(%.2e %.2e %.2e)\n", i, k,
                                        (d.flags & 1) ? "FACE" : "feat", Dist3(Mn[j]->P, d.P), F.size() == Di.size() ? Dist3(F[j]->P, d.P) : -1.0, fP, farP[Di[j]],
                                        Dist3(Mn[j]->Sep, d.Sep), F.size() == Di.size() ? Dist3(F[j]->Sep, d.Sep) : -1.0, d.P[0], d.P[1], d.P[2], d.N[0], d.N[1], d.N[2], d.Sep[0], d.Sep[1], d.Sep[2]);
                    }
                    if (Dist3(Mn[j]->N, d.N) > fN) {
                        ++srOutN;
                        if (std::getenv("FB_SHOWSR"))
                            std::printf("    SR-out N: call %zu shape %d %s |dN| D %.2e FV %.2e band %.2e (far %.2e); N(%.7f %.7f %.7f) tri %u\n", i, k, (d.flags & 1) ? "FACE" : "feat",
                                        Dist3(Mn[j]->N, d.N), F.size() == Di.size() ? Dist3(F[j]->N, d.N) : -1.0, fN, farN[Di[j]], d.N[0], d.N[1], d.N[2], d.tri);
                    }
                    if (d.flags & 1) {
                        for (int q = 0; q < 3; ++q) {
                            biasMine[q] += Mn[j]->N[q] - d.N[q];
                            if (F.size() == Di.size()) biasFv[q] += F[j]->N[q] - d.N[q];
                        }
                        ++biasCnt;
                    }
                    if (Dist3(Mn[j]->Sep, d.Sep) > fS) {
                        ++srOutS;
                        if (std::getenv("FB_SHOWSR"))
                            std::printf("    SR-out Sep: call %zu shape %d %s |dSep| D %.2e FV %.2e band %.2e (far %.2e); Sep(%.3e %.3e %.3e) |Sep| %.2e d? tri %u inst rot %u\n", i, k,
                                        (d.flags & 1) ? "FACE" : "feat", Dist3(Mn[j]->Sep, d.Sep), F.size() == Di.size() ? Dist3(F[j]->Sep, d.Sep) : -1.0, fS, farS[Di[j]],
                                        d.Sep[0], d.Sep[1], d.Sep[2], std::sqrt(d.Sep[0] * d.Sep[0] + d.Sep[1] * d.Sep[1] + d.Sep[2] * d.Sep[2]), d.tri,
                                        unsigned(view.instances[view.triInfo[d.tri].instance].rotation));
                    }
                    if (F.size() == Di.size() && Dist3(F[j]->P, d.P) > fP) ++srFvOutP;
                }
            }
        }
    }
    std::printf("## %s: %zu calls, %zu (call, shape) with contacts\n", argv[3], calls, shapePairs);
    std::printf("  discrete vs FV float: %zu mismatching (call, shape) (%.4f%%); vs double board: %zu (%zu with a ref margin < 1e-3)\n", mismFv,
                100.0 * mismFv / std::max<std::size_t>(1, shapePairs), mismRef, mismRefNear);
    std::printf("  payload vs double (max | mean P): D  P %.2e N %.2e Sep %.2e | FV P %.2e N %.2e Sep %.2e | mean P D %.2e FV %.2e (%zu contacts)\n", maxPmine,
                maxNmine, maxSmine, maxPfv, maxNfv, maxSfv, sumPmine / std::max<std::size_t>(1, nPay), sumPfv / std::max<std::size_t>(1, nPay), nPay);
    std::printf("  S2 work per call: shapes %.2f, candidates %.1f, closed FACEs %.3f, ties %.3f, outside %.3f (side rejects %.3f, proofs %.3f), FV forms %.3f "
                "(features %.3f, FACE keys %.3f), out %.3f\n",
                double(st.shapes) / calls, double(st.candidates) / calls, double(st.faces) / calls, double(st.ties) / calls, double(st.outside) / calls,
                double(st.sideRejects) / calls, double(st.proofs) / calls, double(st.fvForms) / calls, double(st.fvFeatures) / calls,
                double(st.fvFaceKeys) / calls, double(st.mergedOut) / calls);
    if (K > 0)
        std::printf("  stochastic-rounding band (%d samples, %zu calls): D contacts outside 2x the farthest member: P %zu, N %zu, Sep %zu of %zu; FV outside on P %zu; "
                    "D discrete counts no member produced %zu; contacts unscored (no member made the double's list) %zu\n",
                    K, srCalls, srOutP, srOutN, srOutS, srContacts, srFvOutP, srDiscreteOut, srUnscored);
    if (K > 0 && biasCnt && srNcnt) {
        std::printf("  FACE normal bias (mean signed error / population rms per component x y z): D");
        for (int q = 0; q < 3; ++q) std::printf(" %+.3f", biasMine[q] / biasCnt / std::sqrt(srNsq[q] / srNcnt + 1e-300));
        std::printf(" | FV");
        for (int q = 0; q < 3; ++q) std::printf(" %+.3f", biasFv[q] / biasCnt / std::sqrt(srNsq[q] / srNcnt + 1e-300));
        std::printf(" | SR members");
        for (int q = 0; q < 3; ++q) std::printf(" %+.3f", srNsum[q] / srNcnt / std::sqrt(srNsq[q] / srNcnt + 1e-300));
        std::printf("  (%zu FACE contacts, pop rms %.2e %.2e %.2e)\n", biasCnt, std::sqrt(srNsq[0] / srNcnt), std::sqrt(srNsq[1] / srNcnt), std::sqrt(srNsq[2] / srNcnt));
    }
    if (const char *dumpPath = std::getenv("FB_DUMP")) {
        // E's fallback dump: per call a u8 shape mask, then per set bit (shape order) u32 n + n u32 tri ids (FV order)
        std::FILE *f = std::fopen(dumpPath, "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", dumpPath); return 1; }
        struct Job { std::uint32_t call; int k; std::uint32_t first, n; };
        std::vector<Job> jobs;
        std::vector<std::uint32_t> ids;
        std::vector<std::uint8_t> masks;
        for (std::size_t i = 0; i < cp.size() / 11; ++i) {
            std::uint8_t m;
            if (std::fread(&m, 1, 1, f) != 1) break;
            masks.push_back(m);
            for (int k = 0; k < 8; ++k) {
                if (!(m & (1u << k))) continue;
                std::uint32_t n = 0;
                if (std::fread(&n, 4, 1, f) != 1) { std::fprintf(stderr, "short dump\n"); return 1; }
                const std::uint32_t first = static_cast<std::uint32_t>(ids.size());
                ids.resize(ids.size() + n);
                if (n && std::fread(&ids[first], 4, n, f) != n) { std::fprintf(stderr, "short dump\n"); return 1; }
                jobs.push_back({static_cast<std::uint32_t>(i), k, first, n});
            }
        }
        std::fclose(f);
        const std::size_t calls2 = masks.size();
        std::vector<collide::ShapePose> poses(calls2 * 8);
        for (std::size_t i = 0; i < calls2; ++i) {
            const float *pf = &cp[11 * i];
            collide::ShapePoses(car, pf, pf + 3, pf + 7, &poses[8 * i]);
        }
        // correctness on the dumped shapes: S2 against the exact stage per (call, shape)
        std::size_t bad = 0, cmp = 0;
        atlas::FallbackStats st2;
        for (const Job &j : jobs) {
            const float *pf = &cp[11 * j.call];
            stage.DetectPose(pf, pf + 3, pf + 7, fv);
            mine.n = 0; mine.overflow = false;
            const double origin[3] = {0, 0, 0};
            fb.DetectShape(&poses[8 * j.call], j.k, ids.data() + j.first, j.n, origin, mine, &st2, true);
            std::vector<const model::RespContact *> F;
            for (int c = 0; c < fv.n; ++c) if (fv.c[c].shape == j.k) F.push_back(&fv.c[c]);
            ++cmp;
            bool b = static_cast<std::size_t>(mine.n) != F.size();
            for (int c = 0; !b && c < mine.n; ++c)
                b = mine.c[c].flags != F[c]->flags || mine.c[c].material != F[c]->material || Dist3f(mine.c[c].P, F[c]->P) > 1e-3;
            bad += b;
        }
        tw::bench::Pmu pmu;
        double best = 1e300, bestI = 0;
        const int warmReps = std::getenv("FB_WARM") ? std::atoi(std::getenv("FB_WARM")) : 1;
        const std::uint32_t trunc = std::getenv("FB_TRUNC") ? std::uint32_t(std::atoi(std::getenv("FB_TRUNC"))) : 0xFFFFFFFFu;   // profiling: first k candidates only
        for (int rep = 0; rep < 5; ++rep) {
            const auto a = pmu.Read();
            std::uint32_t cur = 0xFFFFFFFFu;
            for (const Job &j : jobs) {
                if (j.call != cur) { cur = j.call; mine.n = 0; mine.overflow = false; }
                const double origin[3] = {0, 0, 0};
                for (int r = 0; r < warmReps; ++r) {   // FB_WARM=R: each job R times in a row (all but the first warm)
                    if (r) { mine.n = 0; mine.overflow = false; }
                    fb.DetectShape(&poses[8 * j.call], j.k, ids.data() + j.first, std::min<std::uint32_t>(j.n, trunc), origin, mine, nullptr, true);
                }
            }
            const auto b = pmu.Read();
            const auto d = tw::bench::Pmu::Delta(a, b);
            if (double(d.cycles) < best) { best = double(d.cycles); bestI = double(d.instructions); }
        }
        std::printf("## E's fallback dump %s: %zu calls, %zu fallback shapes (%.2f per call, %.1f candidates each); vs FV: %zu of %zu (call, shape) differ\n", dumpPath, calls2,
                    jobs.size(), double(jobs.size()) / calls2, double(ids.size()) / std::max<std::size_t>(1, jobs.size()), bad, cmp);
        std::printf("  S2 work per fallback shape: filter survivors %.2f, closed FACEs %.3f, ties %.3f, outside %.3f (section rejects %.3f, proofs %.3f), FV forms %.3f (features %.3f, FACE keys %.3f)\n",
                    double(st2.survivors) / cmp, double(st2.faces) / cmp, double(st2.ties) / cmp, double(st2.outside) / cmp, double(st2.sideRejects) / cmp, double(st2.proofs) / cmp,
                    double(st2.fvForms) / cmp, double(st2.fvFeatures) / cmp, double(st2.fvFaceKeys) / cmp);
        std::printf("  timing (PMU, min of 5, %d run(s) per job): %.0f cycles per call (substep), %.0f cycles and %.0f instructions per fallback shape run\n", warmReps,
                    best / calls2 / warmReps, best / std::max<std::size_t>(1, jobs.size()) / warmReps, bestI / std::max<std::size_t>(1, jobs.size()) / warmReps);
    }
    if (std::getenv("FB_TIME")) {
        // timing: candidates precomputed per (call, shape); DetectShape only
        std::vector<std::vector<std::uint32_t>> cands(N * 8);
        std::vector<collide::ShapePose> poses(N * 8);
        for (std::size_t i = 0; i < N; ++i) {
            const float *pf = &cp[11 * i];
            collide::ShapePoses(car, pf, pf + 3, pf + 7, &poses[8 * i]);
            for (int k = 0; k < 8; ++k) {
                fb.BvhCandidates(poses[8 * i + k], &cands[8 * i + k]);
                if (std::getenv("FB_PRESORT")) {   // the atlas's lists in FV order (deduplicated)
                    auto &c = cands[8 * i + k];
                    std::sort(c.begin(), c.end(), [&](std::uint32_t a, std::uint32_t b) { return view.triInfo[a].fvOrder < view.triInfo[b].fvOrder; });
                    c.erase(std::unique(c.begin(), c.end()), c.end());
                }
            }
        }
        tw::bench::Pmu pmu;
        double best = 1e300, bestI = 0;
        std::size_t shapesRun = 0;
        for (int rep = 0; rep < 5; ++rep) {
            shapesRun = 0;
            const auto a = pmu.Read();
            for (std::size_t i = 0; i < N; ++i) {
                mine.n = 0; mine.overflow = false;
                const double origin[3] = {0, 0, 0};
                for (int k = 0; k < 8; ++k) {
                    const auto &c = cands[8 * i + k];
                    if (c.empty()) continue;
                    ++shapesRun;
                    fb.DetectShape(&poses[8 * i], k, c.data(), c.size(), origin, mine, nullptr);
                }
            }
            const auto b = pmu.Read();
            const auto d = tw::bench::Pmu::Delta(a, b);
            if (double(d.cycles) < best) { best = double(d.cycles); bestI = double(d.instructions); }
        }
        std::printf("  timing (PMU, min of 5): %.0f cycles per call with all shapes through S2 (%.2f shapes with candidates per call): %.0f cycles, %.0f instructions per shape\n",
                    best / N, double(shapesRun) / N, best / std::max<std::size_t>(1, shapesRun), bestI / std::max<std::size_t>(1, shapesRun));
    }
    return 0;
}
