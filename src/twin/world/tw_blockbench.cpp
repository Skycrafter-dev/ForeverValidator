// tw_blockbench: M2 experiment 2. Block-model world with coplanar patches,
// one query per tick, all candidates tested every substep, FV's per-shape
// merge; compared with FV's merged per-shape contacts (recorded by tw_record).
// Usage: tw_blockbench REC [REC...]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <x86intrin.h>

#include "tw_accel.h"
#include "tw_blocks.h"
#include "tw_candidates.h"

using namespace tw;
using Clock = std::chrono::steady_clock;
static const double kGHz = 5.3;

// Per-tick candidate list with world-space copies of the elements (exact for
// signed-permutation instances: R*v is a permutation/negation, + T is the same
// rounding FV's world transform does).
struct CandW {
    Box3 box;
    float n[3];
    std::uint32_t first, count;
    std::uint8_t isTriangle, material, patch;
};

struct ElemIndex {  // BVH4 over the world boxes of all instance elements (one entry each)
    World boxes; Bvh4 bvh; std::vector<std::pair<std::uint32_t, std::uint32_t>> ref;
    void Build(const BlockWorld &bw) {
        boxes.tris.clear(); ref.clear();
        std::vector<std::uint8_t> seen(bw.totalInstElems, 0);
        auto add = [&](const BlockWorld::CellRef &r) {
            const std::uint32_t gid = bw.instElemBase[r.inst] + r.elem;
            if (seen[gid]) return; seen[gid] = 1;
            Tri t{}; for (int k = 0; k < 3; ++k) { t.lo[k] = r.lo[k]; t.hi[k] = r.hi[k]; }
            boxes.tris.push_back(t); ref.push_back({r.inst, r.elem});
        };
        for (const auto &r : bw.refs) add(r);
        for (const auto &r : bw.hugeRefs) add(r);
        bvh.Build(boxes);
    }
};

struct TickQuery {
    const ElemIndex *index = nullptr;
    std::vector<CandW> cands;
    std::vector<float> verts, edgeU;
    std::vector<float> bx[6];  // SoA candidate boxes: lo xyz, hi xyz
    std::vector<std::uint8_t> hit;
    std::vector<std::uint32_t> instStamp, elemStamp, elemBase;
    std::uint32_t cur = 0, ecur = 0;
    Box3 box = EmptyBox();
    std::size_t elementsScanned = 0, candTotal = 0;
    void Init(const BlockWorld &bw) {
        instStamp.assign(bw.instances.size(), 0);
        elemBase.resize(bw.models.size());
        std::size_t total = 0;
        for (std::size_t m = 0; m < bw.models.size(); ++m) { elemBase[m] = static_cast<std::uint32_t>(total); total += bw.models[m].elements.size(); }
        elemStamp.assign(total, 0);
    }
    void Run(const BlockWorld &bw, const Box3 &q) {
        cands.clear(); verts.clear(); edgeU.clear(); box = q;
        for (auto &v : bx) v.clear();
        ++ecur;
        if (elemStamp.size() != bw.totalInstElems) elemStamp.assign(bw.totalInstElems, 0);
        auto take = [&](const BlockWorld::CellRef &r) {
            ++elementsScanned;
            if (r.lo[0] > q.hi[0] || r.hi[0] < q.lo[0] || r.lo[1] > q.hi[1] || r.hi[1] < q.lo[1] || r.lo[2] > q.hi[2] || r.hi[2] < q.lo[2]) return;
            const std::uint32_t gid = bw.instElemBase[r.inst] + r.elem;
            if (elemStamp[gid] == ecur) return;
            elemStamp[gid] = ecur;
            const Instance &in = bw.instances[r.inst];
            const Model &m = bw.models[in.model];
            const Element &e = m.elements[r.elem];
            CandW cd{};
            cd.first = static_cast<std::uint32_t>(verts.size() / 3); cd.count = e.count;
            cd.isTriangle = e.isTriangle; cd.material = e.material; cd.patch = e.sourceTris > 1;
            ToWorldDir(in, e.n, cd.n);
            for (int k = 0; k < 3; ++k) { cd.box.lo[k] = r.lo[k]; cd.box.hi[k] = r.hi[k]; }
            for (std::uint32_t v = 0; v < e.count; ++v) {
                float w[3], u[3];
                ToWorldPoint(in, &m.loop[3 * (e.first + v)], w);
                ToWorldDir(in, &m.edgeU[3 * (e.first + v)], u);
                verts.insert(verts.end(), w, w + 3); edgeU.insert(edgeU.end(), u, u + 3);
            }
            cands.push_back(cd); ++candTotal;
            for (int k = 0; k < 3; ++k) { bx[k].push_back(r.lo[k]); bx[3 + k].push_back(r.hi[k]); }
        };
        const int x0 = static_cast<int>(std::floor(q.lo[0] / BlockWorld::kCellX)), x1 = static_cast<int>(std::floor(q.hi[0] / BlockWorld::kCellX));
        const int y0 = static_cast<int>(std::floor(q.lo[1] / BlockWorld::kCellY)), y1 = static_cast<int>(std::floor(q.hi[1] / BlockWorld::kCellY));
        const int z0 = static_cast<int>(std::floor(q.lo[2] / BlockWorld::kCellZ)), z1 = static_cast<int>(std::floor(q.hi[2] / BlockWorld::kCellZ));
        if (index) {
            index->bvh.Query(q, [&](std::uint32_t id) {
                BlockWorld::CellRef r; r.inst = index->ref[id].first; r.elem = index->ref[id].second;
                for (int k = 0; k < 3; ++k) { r.lo[k] = index->boxes.tris[id].lo[k]; r.hi[k] = index->boxes.tris[id].hi[k]; }
                take(r);
            });
            return;
        }
        for (int x = x0; x <= x1; ++x) for (int y = y0; y <= y1; ++y) for (int z = z0; z <= z1; ++z) {
            auto it = bw.cellIndex.find(BlockWorld::CellKey(x, y, z));
            if (it == bw.cellIndex.end()) continue;
            for (std::uint32_t k = bw.cellBegin[it->second]; k < bw.cellEnd[it->second]; ++k) take(bw.refs[k]);
        }
        const int X0 = static_cast<int>(std::floor(q.lo[0] / 256.0f)), X1 = static_cast<int>(std::floor(q.hi[0] / 256.0f));
        const int Y0 = static_cast<int>(std::floor(q.lo[1] / 128.0f)), Y1 = static_cast<int>(std::floor(q.hi[1] / 128.0f));
        const int Z0 = static_cast<int>(std::floor(q.lo[2] / 256.0f)), Z1 = static_cast<int>(std::floor(q.hi[2] / 256.0f));
        for (int x = X0; x <= X1; ++x) for (int y = Y0; y <= Y1; ++y) for (int z = Z0; z <= Z1; ++z) {
            auto it = bw.cellIndex.find(BlockWorld::CellKey(x, y, z) | (1ull << 63));
            if (it == bw.cellIndex.end()) continue;
            for (std::uint32_t k = bw.cellBegin[it->second]; k < bw.cellEnd[it->second]; ++k) take(bw.refs[k]);
        }
        for (const auto &r : bw.hugeRefs) take(r);
    }
};

struct Pipeline {
    const BlockWorld *bw;
    TickQuery tq;
    std::uint32_t tick = 0xffffffffu;
    std::size_t requeries = 0, queries = 0, pairTests = 0, elementTests = 0, fastFaces = 0, contactsOut = 0;
    std::vector<MergedContact> raw;
    std::vector<std::uint8_t> rawPatch;
    std::vector<MergedContact> merged[kShapes];
    std::vector<std::uint8_t> fromPatch[kShapes];
    bool provenance = false;
    std::uint64_t tQuery = 0, tTests = 0, tMerge = 0;

    void Step(const tw::Substep &s, const Frame &f) {
        if (tq.instStamp.empty()) tq.Init(*bw);
        const std::uint64_t c0 = __rdtsc();
        if (s.tick != tick || !Contains(tq.box, f.unionBox)) {  // one query per tick: one tick of travel + margin
            const bool re = s.tick == tick;
            tick = s.tick;
            const float w = std::sqrt(s.angVel[0] * s.angVel[0] + s.angVel[1] * s.angVel[1] + s.angVel[2] * s.angVel[2]);
            tq.Run(*bw, Inflate(f.unionBox, (f.speed + 2.6f * w) * 0.01f * 1.25f + 0.05f));
            if (re) ++requeries; else ++queries;
        }
        const std::uint64_t c1 = __rdtsc();
        tQuery += c1 - c0;
        for (int sh = 0; sh < kShapes; ++sh) {
            const std::uint64_t d0 = __rdtsc();
            raw.clear(); rawPatch.clear();
            const Box3 &sb = f.inflated[sh];
            const std::size_t nc = tq.cands.size();
            tq.hit.resize(nc + 16);
            {
                const float *l0 = tq.bx[0].data(), *l1 = tq.bx[1].data(), *l2 = tq.bx[2].data();
                const float *h0 = tq.bx[3].data(), *h1 = tq.bx[4].data(), *h2 = tq.bx[5].data();
                std::uint8_t *hm = tq.hit.data();
                for (std::size_t k = 0; k < nc; ++k)
                    hm[k] = static_cast<std::uint8_t>((l0[k] <= sb.hi[0]) & (sb.lo[0] <= h0[k]) & (l1[k] <= sb.hi[1]) & (sb.lo[1] <= h1[k]) &
                                                      (l2[k] <= sb.hi[2]) & (sb.lo[2] <= h2[k]));
            }
            for (std::size_t k = 0; k < nc; ++k) {
                if (!tq.hit[k]) continue;
                const CandW &cd = tq.cands[k];
                ++pairTests;
                Contact c; bool fast = false;
                const int kind = LoopTest(f.sm[sh], &tq.verts[3 * cd.first], &tq.edgeU[3 * cd.first], cd.count, cd.n, cd.isTriangle != 0, &c, &fast);
                fastFaces += fast;
                if (kind == kNone) continue;
                MergedContact mc; mc.c = c; mc.c.kind = kind; mc.primary = kind == kFace; mc.material = cd.material;
                raw.push_back(mc); rawPatch.push_back(cd.patch);
            }
            elementTests = pairTests;
            const std::uint64_t d1 = __rdtsc();
            MergeShape(raw, &merged[sh]);
            tTests += d1 - d0; tMerge += __rdtsc() - d1;
            contactsOut += merged[sh].size();
            if (provenance) {
                fromPatch[sh].clear();
                for (const MergedContact &m : merged[sh]) {
                    std::uint8_t p = 0;
                    for (std::size_t k = 0; k < raw.size(); ++k)
                        if (raw[k].c.P[0] == m.c.P[0] && raw[k].c.P[1] == m.c.P[1] && raw[k].c.P[2] == m.c.P[2]) { p = rawPatch[k]; break; }
                    fromPatch[sh].push_back(p);
                }
            }
        }
    }
};

static double Pct(std::vector<float> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * v.size()))];
}

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: tw_blockbench REC...\n"); return 2; }
    const int maxExceptions = 12;
    for (int a = 1; a < argc; ++a) {
        Recording rec;
        if (!LoadRecording(argv[a], &rec)) { std::fprintf(stderr, "cannot load %s\n", argv[a]); return 1; }
        const std::size_t N = rec.substeps.size();
        const double spt = static_cast<double>(N) / rec.ticks;
        // transform convention check: model vertices through the instance == compiled world triangles
        BlockWorld bwP, bwT;
        auto t0 = Clock::now();
        bwP.Build(rec.source, true, 1e-4f, 1e-4f);
        auto t1 = Clock::now();
        bwT.Build(rec.source, false, 1e-4f, 1e-4f);
        float maxDev = 0;
        {
            std::size_t wi = 0;
            for (std::uint32_t si = 0; si < rec.source.surfaces.size(); ++si) {
                const SourceSurface &s = rec.source.surfaces[si];
                const Instance &in = bwT.instances[si];
                for (std::uint32_t t = 0; t < s.triCount && wi < rec.world.tris.size(); ++t, ++wi) {
                    for (int k = 0; k < 3; ++k) {
                        const float *p = &rec.source.vertices[3 * (s.firstVertex + rec.source.tris[s.firstTri + t].v[k])];
                        float o[3]; ToWorldPoint(in, p, o);
                        for (int i = 0; i < 3; ++i) maxDev = std::fmax(maxDev, std::fabs(o[i] - rec.world.tris[wi].v[k][i]));
                    }
                }
            }
        }
        std::printf("## %s (%zu substeps, %.2f/tick)\n", rec.name.c_str(), N, spt);
        std::printf("index: %zu cell refs, %zu huge refs scanned per query\n", bwP.refs.size(), bwP.hugeRefs.size());
        std::printf("world: %zu source tris in %zu surfaces; %zu models, %zu instances (%u non-permutation); patch build %.0f ms; "
                    "elements %zu (%zu merged patches holding %zu tris) -> %.2fx fewer; block world %.1f MB vs triangle soup %.1f MB; "
                    "transform check max dev %.2e m\n",
                    bwP.sourceTris, rec.source.surfaces.size(), bwP.models.size(), bwP.instances.size(), bwP.nonPermutation,
                    std::chrono::duration<double, std::milli>(t1 - t0).count(), bwP.elements, bwP.mergedPatches, bwP.trianglesInPatches,
                    double(bwP.sourceTris) / double(bwP.elements), bwP.Bytes() / 1e6, rec.world.Bytes() / 1e6, maxDev);
        Frame fr; double chk = 0;
        auto n0 = Clock::now();
        for (std::size_t i = 0; i < N; ++i) { MakeFrame(rec.substeps[i], &fr); chk += fr.unionBox.lo[0]; }
        auto n1 = Clock::now();
        const double nullCyc = std::chrono::duration<double, std::nano>(n1 - n0).count() * kGHz / N;
        ElemIndex eiP, eiT; eiP.Build(bwP); eiT.Build(bwT);
        for (int mode = 0; mode < 4; ++mode) {
            const BlockWorld &bw = (mode & 1) == 0 ? bwP : bwT;
            const ElemIndex *eidx = mode < 2 ? &(mode == 0 ? eiP : eiT) : nullptr;
            // timed pass
            g_cntTri = g_cntPlaneReject = g_cntFast = g_cntFull = g_cntFullVerts = 0;
            Pipeline pl; pl.bw = &bw; pl.tq.index = eidx;
            auto a0 = Clock::now();
            for (std::size_t i = 0; i < N; ++i) { MakeFrame(rec.substeps[i], &fr); pl.Step(rec.substeps[i], fr); }
            auto a1 = Clock::now();
            const double cyc = std::chrono::duration<double, std::nano>(a1 - a0).count() * kGHz / N - nullCyc;
            const std::uint64_t cT = g_cntTri, cR = g_cntPlaneReject, cF = g_cntFast, cU = g_cntFull, cV = g_cntFullVerts;
            // comparison pass vs FV merged contacts
            Pipeline pc; pc.bw = &bw; pc.provenance = true; pc.tq.index = eidx;
            std::size_t shapeSubs = 0, countEq = 0, fvTotal = 0, twTotal = 0, matched = 0, matEq = 0, primEq = 0, exceptions = 0;
            std::size_t excFvExtraFeature = 0, excTwinExtra = 0, excValue = 0, excCountOther = 0, excFvExtraAtPatch = 0;
            std::vector<float> dP, dAng, dSep;
            int printed = 0;
            for (std::size_t i = 0; i < N; ++i) {
                const Substep &s = rec.substeps[i];
                MakeFrame(s, &fr); pc.Step(s, fr);
                for (int sh = 0; sh < kShapes; ++sh) {
                    std::vector<const FvContact *> F;
                    for (std::uint32_t c = s.firstMerged; c < s.firstMerged + s.mergedCount; ++c)
                        if (rec.merged[c].shape == sh) F.push_back(&rec.merged[c]);
                    const auto &W = pc.merged[sh];
                    if (F.empty() && W.empty()) continue;
                    ++shapeSubs; fvTotal += F.size(); twTotal += W.size();
                    if (F.size() == W.size()) ++countEq;
                    std::vector<bool> used(W.size(), false);
                    bool exc = F.size() != W.size(), valueExc = false;
                    for (const FvContact *f : F) {
                        int best = -1; float bd = 1e30f;
                        for (std::size_t k = 0; k < W.size(); ++k) {
                            if (used[k]) continue;
                            const float d = std::sqrt((W[k].c.P[0] - f->P[0]) * (W[k].c.P[0] - f->P[0]) + (W[k].c.P[1] - f->P[1]) * (W[k].c.P[1] - f->P[1]) +
                                                      (W[k].c.P[2] - f->P[2]) * (W[k].c.P[2] - f->P[2]));
                            const float dn = 1.0f - (W[k].c.N[0] * f->N[0] + W[k].c.N[1] * f->N[1] + W[k].c.N[2] * f->N[2]);
                            const float ds = std::fabs(std::sqrt(W[k].c.Sep[0] * W[k].c.Sep[0] + W[k].c.Sep[1] * W[k].c.Sep[1] + W[k].c.Sep[2] * W[k].c.Sep[2]) -
                                                       std::sqrt(f->Sep[0] * f->Sep[0] + f->Sep[1] * f->Sep[1] + f->Sep[2] * f->Sep[2]));
                            const float cost = d + dn + ds + (W[k].material != f->material ? 0.01f : 0.0f) + ((W[k].primary ? 1 : 0) != f->primary ? 0.01f : 0.0f);
                            if (d < 0.05f && cost < bd) { bd = cost; best = static_cast<int>(k); }
                        }
                        if (best < 0) continue;
                        used[best] = true; ++matched;
                        const MergedContact &w = W[best];
                        bd = std::sqrt((w.c.P[0] - f->P[0]) * (w.c.P[0] - f->P[0]) + (w.c.P[1] - f->P[1]) * (w.c.P[1] - f->P[1]) + (w.c.P[2] - f->P[2]) * (w.c.P[2] - f->P[2]));
                        const float dot = std::fmax(-1.0f, std::fmin(1.0f, w.c.N[0] * f->N[0] + w.c.N[1] * f->N[1] + w.c.N[2] * f->N[2]));
                        const float ang = std::acos(dot) * 57.29578f;
                        const float ds = std::sqrt((w.c.Sep[0] - f->Sep[0]) * (w.c.Sep[0] - f->Sep[0]) + (w.c.Sep[1] - f->Sep[1]) * (w.c.Sep[1] - f->Sep[1]) +
                                                   (w.c.Sep[2] - f->Sep[2]) * (w.c.Sep[2] - f->Sep[2]));
                        dP.push_back(bd); dAng.push_back(ang); dSep.push_back(ds);
                        matEq += w.material == f->material; primEq += (w.primary ? 1 : 0) == f->primary;
                        if (bd > 1e-3f || ang > 0.5f || ds > 1e-4f || w.material != f->material || (w.primary ? 1 : 0) != f->primary) valueExc = true;
                    }
                    if (exc || valueExc) {
                        ++exceptions;
                        if (exc) {
                            std::size_t fvFeat = 0, twFeat = 0;
                            for (const FvContact *f : F) fvFeat += !f->primary;
                            for (const auto &w : W) twFeat += !w.primary;
                            bool patchFace = false;
                            for (std::size_t k = 0; k < W.size(); ++k) patchFace |= W[k].primary && pc.fromPatch[sh][k];
                            if (F.size() > W.size() && fvFeat > twFeat) { ++excFvExtraFeature; excFvExtraAtPatch += patchFace; }
                            else if (W.size() > F.size()) ++excTwinExtra;
                            else ++excCountOther;
                        } else ++excValue;
                        if (printed < maxExceptions && mode == 0 && getenv("TW_EXC")) {
                            ++printed;
                            std::printf("  exception tick %u substep %zu shape %d: FV %zu, twin %zu\n", s.tick, i, sh, F.size(), W.size());
                            for (const FvContact *f : F)
                                std::printf("    FV   %s mat %2u P(%.4f %.4f %.4f) N(%.4f %.4f %.4f) |Sep| %.5f\n", f->primary ? "FACE" : "feat", f->material,
                                            f->P[0], f->P[1], f->P[2], f->N[0], f->N[1], f->N[2],
                                            std::sqrt(f->Sep[0] * f->Sep[0] + f->Sep[1] * f->Sep[1] + f->Sep[2] * f->Sep[2]));
                            for (std::size_t k = 0; k < W.size(); ++k) {
                                const auto &w = W[k];
                                std::printf("    twin %s mat %2u P(%.4f %.4f %.4f) N(%.4f %.4f %.4f) |Sep| %.5f%s\n", w.primary ? "FACE" : "feat", w.material,
                                            w.c.P[0], w.c.P[1], w.c.P[2], w.c.N[0], w.c.N[1], w.c.N[2],
                                            std::sqrt(w.c.Sep[0] * w.c.Sep[0] + w.c.Sep[1] * w.c.Sep[1] + w.c.Sep[2] * w.c.Sep[2]),
                                            pc.fromPatch[sh][k] ? " [patch]" : "");
                            }
                        }
                    }
                }
            }
            std::printf("  element tests per substep: triangle %.2f, plane reject %.2f, fast FACE %.2f, full loop %.2f (avg %.1f verts)\n",
                        double(cT) / N, double(cR) / N, double(cF) / N, double(cU) / N, double(cV) / std::max<std::uint64_t>(1, cU));
            std::printf("  rdtsc split per substep: query %.0f, tests %.0f, merge %.0f (TSC units)\n", double(pl.tQuery) / N, double(pl.tTests) / N, double(pl.tMerge) / N);
            std::printf("%s: %.0f cyc/substep, %.0f cyc/tick | queries %.3f/tick + re-queries %.4f/substep, candidates %.1f/query, "
                        "elements scanned %.1f/query, AABB pairs %.2f/substep, fast FACE %.1f%% of element tests\n",
                        mode == 0 ? "PATCHES  + BVH4 tick query" : mode == 1 ? "TRIANGLES + BVH4 tick query" : mode == 2 ? "PATCHES  + cell grid" : "TRIANGLES + cell grid", cyc, cyc * spt, double(pl.queries) / rec.ticks, double(pl.requeries) / N,
                        double(pl.tq.candTotal) / double(pl.queries + pl.requeries), double(pl.tq.elementsScanned) / double(pl.queries + pl.requeries),
                        double(pl.pairTests) / N, 100.0 * pl.fastFaces / std::max<std::size_t>(1, pl.elementTests));
            std::printf("  vs FV merged: (substep,shape) with contacts %zu, count equal %.3f%%; contacts FV %zu twin %zu matched %zu; "
                        "material eq %.3f%%, FACE/feature eq %.3f%%\n",
                        shapeSubs, 100.0 * countEq / std::max<std::size_t>(1, shapeSubs), fvTotal, twTotal, matched,
                        100.0 * matEq / std::max<std::size_t>(1, matched), 100.0 * primEq / std::max<std::size_t>(1, matched));
            std::printf("  errors |dP| m p50 %.2e p99 %.2e max %.2e; normal deg p50 %.2e p99 %.2e max %.2e; |dSep| m p50 %.2e p99 %.2e max %.2e\n",
                        Pct(dP, 0.5), Pct(dP, 0.99), Pct(dP, 1.0), Pct(dAng, 0.5), Pct(dAng, 0.99), Pct(dAng, 1.0), Pct(dSep, 0.5),
                        Pct(dSep, 0.99), Pct(dSep, 1.0));
            std::printf("  exceptions %zu (%.3f%% of (substep,shape)): FV has extra feature %zu (twin FACE on a merged patch: %zu), twin extra %zu, "
                        "other count %zu, value-only %zu\n",
                        exceptions, 100.0 * exceptions / std::max<std::size_t>(1, shapeSubs), excFvExtraFeature, excFvExtraAtPatch, excTwinExtra,
                        excCountOther, excValue);
        }
        std::fflush(stdout);
        (void)chk;
    }
    return 0;
}
