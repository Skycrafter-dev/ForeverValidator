// tw_bench: M2 collision bake-off on recorded FV substep streams (FV-independent).
// Usage: tw_bench REC [REC...]   (REC = file written by tw_record)
//
// For each recording: builds BVH4 and the loose grid, runs every candidate
// over every substep in order (timed pass), then an untimed gate pass that
// checks, for every FV narrow-phase contact, that the candidate passed every
// triangle the contact lies on (within kAttTol) to the exact test.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "tw_accel.h"
#include "tw_candidates.h"
#include "tw_exact.h"
#include "tw_world.h"

using namespace tw;
using Clock = std::chrono::steady_clock;
static const double kGHz = 5.3;
static const double kAttTol = 3e-4;

struct Attribution {
    std::vector<std::uint32_t> first, count, tris;  // per FV contact: triangles within kAttTol of P
};

static void Attribute(const Recording &rec, const Bvh4 &bvh, Attribution *a, std::size_t *empty) {
    a->first.resize(rec.contacts.size()); a->count.resize(rec.contacts.size());
    *empty = 0;
    for (std::size_t i = 0; i < rec.contacts.size(); ++i) {
        const FvContact &c = rec.contacts[i];
        const float t = static_cast<float>(kAttTol);
        Box3 q = {{c.P[0] - t, c.P[1] - t, c.P[2] - t}, {c.P[0] + t, c.P[1] + t, c.P[2] + t}};
        a->first[i] = static_cast<std::uint32_t>(a->tris.size());
        const double p[3] = {c.P[0], c.P[1], c.P[2]};
        bvh.Query(q, [&](std::uint32_t id) {
            if (PointTriDist2(p, rec.world.tris[id]) <= kAttTol * kAttTol) a->tris.push_back(id);
        });
        a->count[i] = static_cast<std::uint32_t>(a->tris.size()) - a->first[i];
        if (a->count[i] == 0) ++*empty;
    }
}

struct Result {
    const char *name;
    double broadCyc = 0, totalCyc = 0;
    Stats st;
    std::size_t misses = 0, checked = 0, contactsTwin = 0;
};

struct Required { std::vector<std::uint32_t> first, count, pairs; std::size_t unmatched = 0; };

template <class C>
static Result RunCandidate(const char *name, C &cand, const Recording &rec, double nullCyc,
                           const Required &req, std::vector<std::vector<std::uint64_t>> *contactSets) {
    Result r; r.name = name;
    std::vector<std::uint32_t> out; out.reserve(4096);
    const std::size_t N = rec.substeps.size();
    // timed pass 1: broad phase only (frames precomputed and timed separately)
    cand.Reset();
    Stats st;
    std::size_t sink = 0;
    Frame fr;
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < N; ++i) { MakeFrame(rec.substeps[i], &fr); out.clear(); cand.Run(rec.substeps[i], fr, out, st); sink += out.size(); }
    auto t1 = Clock::now();
    r.broadCyc = std::chrono::duration<double, std::nano>(t1 - t0).count() * kGHz / static_cast<double>(N) - nullCyc;
    r.st = st;
    // timed pass 2: broad + exact tests
    cand.Reset();
    Stats st2;
    Contact ct;
    std::size_t hits = 0;
    t0 = Clock::now();
    for (std::size_t i = 0; i < N; ++i) {
        MakeFrame(rec.substeps[i], &fr);
        out.clear(); cand.Run(rec.substeps[i], fr, out, st2);
        for (std::uint32_t p : out) hits += ExactTest(fr.sm[PairShape(p)], rec.world.tris[PairTri(p)], &ct) != kNone;
    }
    t1 = Clock::now();
    r.totalCyc = std::chrono::duration<double, std::nano>(t1 - t0).count() * kGHz / static_cast<double>(N) - nullCyc;
    r.contactsTwin = hits;
    // gate pass
    cand.Reset();
    Stats st3;
    std::unordered_set<std::uint32_t> set;
    if (contactSets) contactSets->assign(N, {});
    for (std::size_t i = 0; i < N; ++i) {
        MakeFrame(rec.substeps[i], &fr);
        out.clear(); cand.Run(rec.substeps[i], fr, out, st3);
        set.clear(); set.insert(out.begin(), out.end());
        const Substep &s = rec.substeps[i];
        for (std::uint32_t c = s.firstContact; c < s.firstContact + s.contactCount; ++c) {
            if (req.count[c] == 0) continue;
            ++r.checked;
            bool any = false;
            for (std::uint32_t k = 0; k < req.count[c]; ++k) any |= set.count(req.pairs[req.first[c] + k]) != 0;
            if (!any) ++r.misses;
        }
        if (contactSets) {
            auto &v = (*contactSets)[i];
            for (std::uint32_t p : out) {
                const int kind = ExactTest(fr.sm[PairShape(p)], rec.world.tris[PairTri(p)], &ct);
                if (kind != kNone) v.push_back((static_cast<std::uint64_t>(p) << 3) | static_cast<std::uint64_t>(kind));
            }
            std::sort(v.begin(), v.end());
        }
    }
    (void)sink;
    return r;
}

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: tw_bench REC...\n"); return 2; }
    std::printf("| replay | candidate | broad cyc/substep | broad+exact cyc/substep | collision cyc/tick | pairs to exact /substep | refresh /substep | full re-evals /substep | cached tris | FV-contact misses / checked |\n");
    std::printf("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for (int a = 1; a < argc; ++a) {
        Recording rec;
        if (!LoadRecording(argv[a], &rec)) { std::fprintf(stderr, "cannot load %s\n", argv[a]); return 1; }
        const std::size_t N = rec.substeps.size();
        const double spt = rec.ticks ? static_cast<double>(N) / rec.ticks : 0.0;
        auto b0 = Clock::now();
        Bvh4 bvh; bvh.Build(rec.world);
        auto b1 = Clock::now();
        LooseGrid grid; grid.Build(rec.world);
        auto b2 = Clock::now();
        // null pass: frame setup + loop overhead, subtracted from every candidate
        Frame fr; double chk = 0;
        auto f0 = Clock::now();
        for (std::size_t i = 0; i < N; ++i) { MakeFrame(rec.substeps[i], &fr); chk += fr.unionBox.lo[1]; }
        auto f1 = Clock::now();
        const double nullCyc = std::chrono::duration<double, std::nano>(f1 - f0).count() * kGHz / static_cast<double>(N);
        {   // hot-cache frame setup cost (same substep record repeatedly)
            auto h0 = Clock::now();
            for (int k = 0; k < 200000; ++k) { MakeFrame(rec.substeps[k & 63], &fr); chk += fr.unionBox.hi[0]; }
            auto h1 = Clock::now();
            std::fprintf(stderr, "  hot frame setup %.0f cyc/substep, streamed %.0f\n",
                         std::chrono::duration<double, std::nano>(h1 - h0).count() * kGHz / 200000.0, nullCyc);
        }
        // reference: exact test on all BVH pairs; match FV contacts to twin contacts (shape, point, normal)
        Attribution att; std::size_t emptyAtt = 0;
        Attribute(rec, bvh, &att, &emptyAtt);
        Required req;
        req.first.resize(rec.contacts.size()); req.count.resize(rec.contacts.size());
        std::vector<std::vector<std::uint64_t>> refSets(N);
        std::size_t fvMatched = 0, twinTotal = 0, fvFace = 0, fvWheel = 0, fvFlat = 0, fvTrigger = 0;
        std::vector<std::uint32_t> out;
        for (std::size_t i = 0; i < N; ++i) {
            MakeFrame(rec.substeps[i], &fr);
            out.clear();
            bvh.Query(fr.unionBox, [&](std::uint32_t id) {
                const Tri &t = rec.world.tris[id];
                const Box3 tb = {{t.lo[0], t.lo[1], t.lo[2]}, {t.hi[0], t.hi[1], t.hi[2]}};
                for (int k = 0; k < kShapes; ++k) if (Overlap(tb, fr.inflated[k])) out.push_back(PairCode(k, id));
            });
            std::vector<Contact> tc; std::vector<std::uint32_t> tp;
            for (std::uint32_t p : out) {
                Contact c;
                const int kind = ExactTest(fr.sm[PairShape(p)], rec.world.tris[PairTri(p)], &c);
                if (kind == kNone) continue;
                tc.push_back(c); tp.push_back(p);
                refSets[i].push_back((static_cast<std::uint64_t>(p) << 3) | static_cast<std::uint64_t>(kind));
            }
            std::sort(refSets[i].begin(), refSets[i].end());
            twinTotal += tc.size();
            std::vector<bool> used(tc.size(), false);
            const Substep &s = rec.substeps[i];
            for (std::uint32_t c = s.firstContact; c < s.firstContact + s.contactCount; ++c) {
                const FvContact &fc = rec.contacts[c];
                fvFace += fc.primary; fvWheel += fc.shape >= 4;
                if (att.count[c] && rec.world.extra[att.tris[att.first[c]]].n[1] > 0.95f) ++fvFlat;
                if (att.count[c] && (rec.world.tris[att.tris[att.first[c]]].meta & kMetaTrigger)) ++fvTrigger;
                req.first[c] = static_cast<std::uint32_t>(req.pairs.size());
                int found = -1;
                for (std::size_t k = 0; k < tc.size(); ++k) {
                    if (used[k] || PairShape(tp[k]) != fc.shape) continue;
                    const float dx = tc[k].P[0] - fc.P[0], dy = tc[k].P[1] - fc.P[1], dz = tc[k].P[2] - fc.P[2];
                    const float nd = tc[k].N[0] * fc.N[0] + tc[k].N[1] * fc.N[1] + tc[k].N[2] * fc.N[2];
                    if (dx * dx + dy * dy + dz * dz < 1e-6f && nd > 0.999f) { found = static_cast<int>(k); break; }
                }
                if (found >= 0) { used[found] = true; ++fvMatched; req.pairs.push_back(tp[found]); }
                else {  // float-noise disagreement: require any triangle the FV point lies on
                    ++req.unmatched;
                    for (std::uint32_t k = 0; k < att.count[c]; ++k) req.pairs.push_back(PairCode(fc.shape, att.tris[att.first[c] + k]));
                }
                req.count[c] = static_cast<std::uint32_t>(req.pairs.size()) - req.first[c];
            }
        }
        std::fprintf(stderr, "%s: %zu tris (%u trigger), %zu substeps (%.2f/tick), %zu FV contacts; memory: world %.1f MB, BVH4 %.1f MB (build %.0f ms), loose grid %.1f MB (build %.0f ms)\n",
                     rec.name.c_str(), rec.world.tris.size(), rec.world.triggerTris, N, spt, rec.contacts.size(),
                     rec.world.Bytes() / 1e6, bvh.Bytes() / 1e6, std::chrono::duration<double, std::milli>(b1 - b0).count(),
                     grid.Bytes() / 1e6, std::chrono::duration<double, std::milli>(b2 - b1).count());

        std::vector<std::vector<std::uint64_t>> cohSets, ancSets;
        BvhDirect cb{&rec.world, &bvh};
        GridDirect cg{&rec.world, &grid};
        TileCache ct{&rec.world, &bvh};
        std::vector<TriEdges> edges(rec.world.tris.size());
        for (std::size_t k = 0; k < edges.size(); ++k) MakeTriEdges(rec.world.tris[k], rec.world.extra[k], &edges[k]);
        Coherent cc{&rec.world, &bvh};
        Coherent cp{&rec.world, &bvh, &edges};
        Anchored ca{&rec.world, &bvh};
        std::vector<std::vector<std::uint64_t>> preSets;
        Result rs[6] = {RunCandidate("BVH4 direct", cb, rec, nullCyc, req, nullptr),
                        RunCandidate("loose grid direct", cg, rec, nullCyc, req, nullptr),
                        RunCandidate("B tile+cache", ct, rec, nullCyc, req, nullptr),
                        RunCandidate("per-pair sleep cache", cc, rec, nullCyc, req, &cohSets),
                        RunCandidate("per-pair sleep cache, precomputed edges", cp, rec, nullCyc, req, &preSets),
                        RunCandidate("anchored coherent cache", ca, rec, nullCyc, req, &ancSets)};
        for (const Result &r : rs) {
            std::printf("| %s | %s | %.0f | %.0f | %.0f | %.2f | %.3f | %.3f | %.1f | %zu / %zu |\n", rec.name.c_str(), r.name,
                        r.broadCyc, r.totalCyc, r.totalCyc * spt, r.st.pairs / N, r.st.refreshes / N, r.st.wakes / N / kShapes,
                        r.st.cacheTris / N, r.misses, r.checked);
        }
        std::size_t diffC = 0, diffA = 0;
        for (std::size_t i = 0; i < N; ++i) { diffC += (refSets[i] != cohSets[i]) + (refSets[i] != preSets[i]); diffA += refSets[i] != ancSets[i]; }
        std::size_t ticksWithLoss = 0, lostPairs = 0, allPairs = 0, ticksWithContact = 0;
        for (std::size_t i = 0; i < N;) {
            std::size_t j = i; while (j < N && rec.substeps[j].tick == rec.substeps[i].tick) ++j;
            std::unordered_set<std::uint64_t> all, last;
            for (std::size_t k = i; k < j; ++k)
                for (std::uint64_t code : refSets[k]) { all.insert(code >> 3); if (k == j - 1) last.insert(code >> 3); }
            std::size_t lost = 0; for (std::uint64_t c : all) lost += !last.count(c);
            if (!all.empty()) ++ticksWithContact;
            if (lost) ++ticksWithLoss;
            lostPairs += lost; allPairs += all.size();
            i = j;
        }
        const double nfv = static_cast<double>(rec.contacts.size());
        std::fprintf(stderr,
                     "  frame setup (subtracted) %.0f cyc/substep; FV contacts: unattributed %zu, FACE %.1f%%, wheel %.1f%%, on n.y>0.95 %.1f%%, trigger %.2f%%\n"
                     "  twin exact vs FV: %.3f%% of FV contacts matched (%zu unmatched), twin contacts %zu; sleep!=reference in %zu substeps, anchored!=reference in %zu\n"
                     "  once-per-tick detection would lose (shape,tri) contact pairs in %.1f%% of ticks with contact (%.1f%% of pairs) [chk %g]\n",
                     nullCyc, emptyAtt, 100.0 * fvFace / nfv, 100.0 * fvWheel / nfv, 100.0 * fvFlat / nfv, 100.0 * fvTrigger / nfv,
                     100.0 * fvMatched / nfv, req.unmatched, twinTotal, diffC, diffA,
                     100.0 * ticksWithLoss / std::max<std::size_t>(1, ticksWithContact), 100.0 * lostPairs / std::max<std::size_t>(1, allPairs), chk);
        std::fflush(stdout);
    }
    return 0;
}
