// Architect C: the lazy column cache (atlas_colcache.h) on real search streams (TWSRCH01, the integration
// agent's recorder: per candidate the TwinState after every tick).
//   atlas_colcache_search STREAM.bin PACK.twpack CARSHAPES [budgetMB]
// Per map: the atlas with lazy columns; every candidate tick's 8 shape centres (collide::ShapePoses) queried
// through one cache (wheels class 0, bodies class 1; a tick = BeginTick). Reported: superblocks touched (all
// candidates, and per candidate), resident bytes, builds and evictions unbounded and under the budget, build
// time mean / max and the worst build time inside one tick; then 6 threads sharing one budgeted cache, the
// candidates spread over them, every query's slot contents hashed and compared with the single-thread run.
#include "atlas_build.h"
#include "atlas_colcache.h"
#include "tw_collide.h"
#include "tw_state.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace tw;
using namespace tw::atlas;

namespace {

struct Cand {
    std::uint64_t id;
    std::vector<model::TwinState> states;   // after every tick
};

bool LoadStream(const char *path, std::vector<Cand> *out, std::string *replay) {
    std::FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    std::vector<unsigned char> b;
    unsigned char buf[65536];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
    std::fclose(f);
    if (b.size() < 20 || std::memcmp(b.data(), "TWSRCH01", 8) != 0) return false;
    std::size_t o = 8;
    auto u32 = [&](std::uint32_t *v) { std::memcpy(v, &b[o], 4); o += 4; };
    std::uint32_t ver, cands, bps;
    u32(&ver); u32(&cands); u32(&bps);
    if (bps != sizeof(model::TwinState)) return false;
    std::uint16_t ln;
    std::memcpy(&ln, &b[o], 2); o += 2;
    replay->assign(reinterpret_cast<const char *>(&b[o]), ln); o += ln;
    std::uint32_t branch;
    u32(&branch);
    for (std::uint32_t c = 0; c < cands; ++c) {
        Cand cd;
        std::memcpy(&cd.id, &b[o], 8); o += 8;
        o += bps;   // start state
        std::uint32_t ticks;
        u32(&ticks);
        cd.states.resize(ticks);
        for (std::uint32_t t = 0; t < ticks; ++t) {
            o += 8;   // control, respawns
            std::memcpy(&cd.states[t], &b[o], bps); o += bps;
        }
        out->push_back(std::move(cd));
    }
    return o == b.size();
}

inline std::uint64_t Mix(std::uint64_t h, std::uint64_t v) { h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2); return h; }

// a query's answer: hash of the slot's entries and union (what a lookup returns)
std::uint64_t SlotHash(ColumnCache &C, int r, std::uint32_t h) {
    if (h == 0xFFFFFFFFu) return 1;
    const ColumnCache::SlotRef s = C.Resolve(r, h);
    std::uint64_t x = s.n * 1315423911ull + s.unionCount;
    for (std::uint32_t i = 0; i < s.n; ++i) {
        static_assert(sizeof(ColumnEntry) <= 24, "ColumnEntry fits the hash buffer");
        std::uint64_t w[3] = {0, 0, 0};
        std::memcpy(w, &s.entries[i], sizeof(ColumnEntry));
        x = Mix(x, w[0]); x = Mix(x, w[1]); x = Mix(x, w[2]);
    }
    for (std::uint32_t i = 0; i < 2 * s.unionCount; ++i) x = Mix(x, std::uint32_t(s.unions[i]));
    return x;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: atlas_colcache_search STREAM.bin PACK.twpack CARSHAPES [budgetMB]\n"); return 2; }
    const std::size_t budget = (argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 16) << 20;
    std::vector<Cand> cands;
    std::string replay;
    if (!LoadStream(argv[1], &cands, &replay)) { std::fprintf(stderr, "bad stream %s\n", argv[1]); return 1; }
    collide::CarShapes car;
    {
        std::FILE *f = std::fopen(argv[3], "rb");
        if (!f || std::fread(&car, sizeof car, 1, f) != 1) { std::fprintf(stderr, "bad car shapes %s\n", argv[3]); return 1; }
        std::fclose(f);
    }
    world::WorldPack pack;
    std::string err;
    if (!world::LoadWorldPack(argv[2], &pack, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    world::WorldPackView view;
    if (!world::OpenWorldPack(pack, &view, &err, true)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    World w;
    w.tris.assign(view.tris.begin(), view.tris.end());
    Bvh4 bvh;
    bvh.Build(w);
    Atlas A;
    BuildOptions bo;
    bo.lazyColumns = true;
    if (!A.Build(view, bvh, bo, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    float x0, z0;
    std::uint32_t nx, nz;
    A.ColumnGrid(&x0, &z0, &nx, &nz);
    ColumnCache::Builder builder = [&A](std::uint32_t bx, std::uint32_t bz, ColumnBlock *out) { return A.BuildColumnBlock(bx, bz, out); };
    // queries: per candidate, per tick, 8 shape centres -> handles
    std::vector<std::vector<std::uint32_t>> handles(cands.size());
    std::size_t ticks = 0;
    {
        ColumnCache probe(x0, z0, nx, nz, builder, ColumnCache::Options{});
        for (std::size_t c = 0; c < cands.size(); ++c)
            for (const model::TwinState &s : cands[c].states) {
                collide::ShapePose sp[8];
                collide::ShapePoses(car, s.pos, s.quat, s.absorb, sp);
                for (int k = 0; k < 8; ++k) handles[c].push_back(probe.SlotHandle(sp[k].world.t, k >= 4 ? 0 : 1));
                ++ticks;
            }
    }
    std::size_t perCand = 0;
    std::unordered_set<std::uint32_t> all;
    for (auto &hs : handles) {
        std::unordered_set<std::uint32_t> b;
        for (std::uint32_t h : hs) if (h != 0xFFFFFFFFu) { b.insert(h >> 9); all.insert(h >> 9); }
        perCand += b.size();
    }
    // one reader, unbounded and budgeted
    std::vector<std::vector<std::uint64_t>> ref(cands.size());
    std::string name = argv[2];
    name = name.substr(name.find_last_of('/') + 1);
    for (std::size_t bud : {std::size_t(0), budget}) {
        ColumnCache::Options o;
        o.budgetBytes = bud;
        ColumnCache C(x0, z0, nx, nz, builder, o);
        const int r = C.Register();
        for (std::size_t c = 0; c < cands.size(); ++c) {
            if (bud == 0) ref[c].resize(handles[c].size());
            for (std::size_t i = 0; i < handles[c].size(); ++i) {
                if (i % 8 == 0) C.BeginTick(r);
                const std::uint64_t h = SlotHash(C, r, handles[c][i]);
                if (bud == 0) ref[c][i] = h;
            }
        }
        C.BeginTick(r);
        const ColumnCacheStats s = C.Stats();
        std::printf("%s: %zu tris, %zu candidates, %zu ticks; superblocks touched %zu (per candidate %.1f); budget %s: %llu builds, %llu evictions, "
                    "resident %.1f MB, peak %.1f MB; build %.2f ms mean, %.2f ms max; worst tick %.2f ms\n",
                    name.c_str(), view.tris.size(), cands.size(), ticks, all.size(), double(perCand) / std::max<std::size_t>(1, cands.size()),
                    bud ? (std::to_string(bud >> 20) + " MB").c_str() : "unbounded", (unsigned long long)s.builds, (unsigned long long)s.evictions,
                    s.residentBytes / 1e6, s.peakResidentBytes / 1e6, s.builds ? double(s.buildTsc) / s.builds / 3.893e6 : 0.0, s.maxBuildTsc / 3.893e6,
                    s.maxTickBuildTsc / 3.893e6);
        std::fflush(stdout);
    }
    // 6 threads, one budgeted cache, candidates round-robin: every answer against the single-thread run
    {
        ColumnCache::Options o;
        o.budgetBytes = budget;
        ColumnCache C(x0, z0, nx, nz, builder, o);
        std::atomic<std::size_t> bad{0}, qn{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 6; ++t)
            th.emplace_back([&, t] {
                const int r = C.Register();
                std::size_t b = 0, q = 0;
                for (std::size_t c = std::size_t(t); c < cands.size(); c += 6)
                    for (std::size_t i = 0; i < handles[c].size(); ++i) {
                        if (i % 8 == 0) C.BeginTick(r);
                        b += SlotHash(C, r, handles[c][i]) != ref[c][i];
                        ++q;
                    }
                C.Idle(r);
                bad += b;
                qn += q;
            });
        for (auto &x : th) x.join();
        const ColumnCacheStats s = C.Stats();
        std::printf("  6 threads, budget %zu MB: %zu of %zu answers differ; %llu builds (%llu waits), %llu evictions, peak %.1f MB\n", budget >> 20, bad.load(),
                    qn.load(), (unsigned long long)s.builds, (unsigned long long)s.raceDuplicates, (unsigned long long)s.evictions, s.peakResidentBytes / 1e6);
    }
    return 0;
}
