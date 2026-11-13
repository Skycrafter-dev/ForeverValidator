// Architect C: the packed column grid (atlas_colpack.h) against E's raw grid (atlas_build.h).
//   atlas_colpack_check PACK.twpack [REC.twrec] [PACK [REC] ...]
// Per pack: bytes raw vs packed, every slot decoded and compared bit for bit (entries with y1max, union,
// members, overlap flags); with a recording, the free test at FV's shape poses (raw union scan vs the packed
// AnyOverlap: same answers, PMU cycles per query, min of 5).
#define private public
#include "atlas_build.h"
#undef private
#include "atlas_colpack.h"
#include "tw_world.h"
#include "twin_pmu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_set>

using namespace tw;
volatile std::size_t g_sink = 0;

int main(int argc, char **argv) {
    for (int a = 1; a < argc; ++a) {
        const char *recPath = (a + 1 < argc && std::strstr(argv[a + 1], ".twrec")) ? argv[a + 1] : nullptr;
        world::WorldPack pack;
        std::string err;
        if (!world::LoadWorldPack(argv[a], &pack, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        world::WorldPackView view;
        if (!world::OpenWorldPack(pack, &view, &err, true)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        World w;
        w.tris.assign(view.tris.begin(), view.tris.end());
        Bvh4 bvh;
        bvh.Build(w);
        atlas::Atlas A;
        if (!A.Build(view, bvh, atlas::BuildOptions{}, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        atlas::RawColumns raw;
        raw.x0 = A.colX0_; raw.z0 = A.colZ0_; raw.nx = A.colNx_; raw.nz = A.colNz_;
        raw.first = A.colFirst_.data(); raw.entries = A.colEntries_.data(); raw.entMemFirst = A.entMemFirst_.data();
        raw.entMembers = A.entMembers_.data(); raw.entOverlap = A.entOverlap_.data();
        raw.plates = A.plates_.data(); raw.plateCount = A.plates_.size();
        atlas::ColumnPack P;
        P.Build(raw);
        const std::size_t rawBytes = A.colFirst_.size() * 4 + A.colEntries_.size() * sizeof(atlas::ColumnEntry) + A.colUnionFirst_.size() * 4 + A.colUnion_.size() * 4 +
                                     A.entMemFirst_.size() * 4 + A.entMembers_.size() * 4 + A.entOverlap_.size();
        // bit identity over every slot
        std::size_t bad = 0, slots = A.colFirst_.size() - 1;
        atlas::ColumnPack::SlotView sv;
        for (std::uint32_t s = 0; s < slots; ++s) {
            P.Decode(P.HandleOf(s), &sv);
            std::uint32_t n, un;
            const atlas::ColumnEntry *e = A.Entries(s, &n);
            const std::int32_t *u = A.Union(s, &un);
            bool ok = sv.n == n && sv.unionCount == un;
            for (std::uint32_t i = 0; ok && i < n; ++i) ok = std::memcmp(&sv.entries[i], &e[i], sizeof e[i]) == 0;
            for (std::uint32_t i = 0; ok && i < 2 * un; ++i) ok = std::memcmp(&sv.unionYs[i], &u[i], 4) == 0;
            for (std::uint32_t i = 0; ok && i < n; ++i) {
                std::uint32_t mc;
                bool ov;
                const std::uint32_t idx = A.colFirst_[s] + i;
                const std::uint32_t *m = A.EntryMembers(idx, &mc, &ov);
                ok = mc == sv.memFirst[i + 1] - sv.memFirst[i] && (mc == 0 || (ov == (sv.overlap[i] != 0) && std::memcmp(m, sv.members.data() + sv.memFirst[i], 4 * mc) == 0));
            }
            bad += !ok;
        }
        std::printf("%s: %zu slots, %zu entries: raw %.1f MB -> packed %.1f MB (%.1fx); %zu slots differ\n", argv[a], slots, A.colEntries_.size(), rawBytes / 1e6,
                    P.Bytes() / 1e6, double(rawBytes) / P.Bytes(), bad);
        if (recPath) {
            Recording rec;
            if (!LoadRecording(recPath, &rec)) { std::fprintf(stderr, "cannot load %s\n", recPath); return 1; }
            struct Q { std::uint32_t slot, handle; std::int32_t lo, hi; };
            std::vector<Q> qs;
            for (const Substep &st : rec.substeps)
                for (int k = 0; k < 8; ++k) {
                    const Shape &sh = st.shapes[k];
                    const float rmax = std::max(sh.r[0], std::max(sh.r[1], sh.r[2]));
                    const std::uint32_t slot = A.ColumnSlot(sh.t, k >= 4 ? 0 : 1);
                    if (slot != 0xFFFFFFFFu) qs.push_back(Q{slot, P.SlotHandle(sh.t, k >= 4 ? 0 : 1), std::int32_t(std::floor(double(sh.t[1] - rmax) * 1000.0)), std::int32_t(std::ceil(double(sh.t[1] + rmax) * 1000.0))});
                }
            {   // lazy superblocks: the 16x16-column blocks the path's shapes touch, and their raw bytes (E's layout)
                std::unordered_set<std::uint32_t> sbs;
                for (const Q &q : qs) sbs.insert(q.handle >> 9);
                std::size_t rawSb = 0;
                for (std::uint32_t sbi : sbs) {
                    const std::uint32_t sbx = sbi % ((A.colNx_ + 15) / 16), sbz = sbi / ((A.colNx_ + 15) / 16);
                    for (std::uint32_t z = sbz * 16; z < std::min(A.colNz_, sbz * 16 + 16); ++z)
                        for (std::uint32_t x = sbx * 16; x < std::min(A.colNx_, sbx * 16 + 16); ++x)
                            for (int cls = 0; cls < 2; ++cls) {
                                const std::size_t s2 = (std::size_t(z) * A.colNx_ + x) * 2 + cls;
                                const std::uint32_t ne2 = A.colFirst_[s2 + 1] - A.colFirst_[s2], nu = A.colUnionFirst_[s2 + 1] - A.colUnionFirst_[s2];
                                std::size_t nm = 0;
                                for (std::uint32_t i = A.colFirst_[s2]; i < A.colFirst_[s2 + 1]; ++i) nm += A.entMemFirst_[i + 1] - A.entMemFirst_[i];
                                rawSb += 8 + ne2 * (sizeof(atlas::ColumnEntry) + 4 + 1) + nu * 8 + nm * 4;
                            }
                }
                std::printf("  path touches %zu of %zu superblocks: raw resident %.2f MB (E's layout)\n", sbs.size(), std::size_t((A.colNx_ + 15) / 16) * ((A.colNz_ + 15) / 16),
                            rawSb / 1e6);
            }
            std::size_t diff = 0, freeN = 0;
            double best[3] = {1e30, 1e30, 1e30};
            for (int rep = 0; rep < 5; ++rep)
                for (int mode = 0; mode < 3; ++mode) {
                    tw::bench::Pmu pmu;
                    std::size_t hits = 0;
                    const auto t0 = pmu.Read();
                    if (mode == 0)
                        for (const Q &q : qs) {   // E's free test: the slot's union intervals
                            std::uint32_t un;
                            const std::int32_t *u = A.Union(q.slot, &un);
                            bool any = false;
                            for (std::uint32_t i = 0; i < un && !any; ++i) any = u[2 * i] <= q.hi && u[2 * i + 1] >= q.lo;
                            hits += any;
                        }
                    else if (mode == 1)
                        for (const Q &q : qs) hits += P.AnyOverlap(q.handle, q.lo, q.hi);
                    else
                        for (const Q &q : qs) hits += reinterpret_cast<std::uintptr_t>(P.PayloadPtr(q.handle)) & 1u;
                    const auto t1 = pmu.Read();
                    best[mode] = std::min(best[mode], double(tw::bench::Pmu::Delta(t0, t1).cycles) / std::max<std::size_t>(1, qs.size()));
                    if (rep == 0 && mode == 1) {
                        for (const Q &q : qs) {
                            std::uint32_t un;
                            const std::int32_t *u = A.Union(q.slot, &un);
                            bool any = false;
                            for (std::uint32_t i = 0; i < un && !any; ++i) any = u[2 * i] <= q.hi && u[2 * i + 1] >= q.lo;
                            diff += any != P.AnyOverlap(q.handle, q.lo, q.hi);
                            freeN += !any;
                        }
                    }
                    g_sink += hits;
                }
            std::printf("  free test at FV's poses (%s, %zu queries, %.1f%% free): raw union %.1f, packed %.1f (payload pointer only %.1f) PMU cycles per query; %zu answers differ\n",
                        recPath, qs.size(), 100.0 * freeN / std::max<std::size_t>(1, qs.size()), best[0], best[1], best[2], diff);
            ++a;
        }
        std::fflush(stdout);
    }
    return 0;
}
