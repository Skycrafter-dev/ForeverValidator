// Architect C: statistics of E's raw column grid (atlas_build.h) on a world package, for its compression.
//   atlas_colstats PACK.twpack [PACK ...]
// Reads the grid through a private-access shim (measurement only).
#define private public
#include "atlas_build.h"
#undef private
#include "tw_world.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>

using namespace tw;

namespace {
struct H3 {
    std::size_t operator()(const std::array<std::uint32_t, 3> &a) const { return (a[0] * 0x9E3779B97F4A7C15ull) ^ (std::size_t(a[1]) << 21) ^ a[2]; }
};
}  // namespace

int main(int argc, char **argv) {
    // args: PACK [REC.twrec] ...: with a recording, the decision impact of rounding the y-ranges outward
    for (int a = 1; a < argc; ++a) {
        const char *recPath = nullptr;
        if (a + 1 < argc && std::strstr(argv[a + 1], ".twrec")) recPath = argv[a + 1];
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
        const std::size_t slots = A.colFirst_.size() - 1, ne = A.colEntries_.size();
        std::size_t nonEmpty = 0, maxPer = 0;
        std::map<std::size_t, std::size_t> hist;
        for (std::size_t s = 0; s < slots; ++s) {
            const std::size_t n = A.colFirst_[s + 1] - A.colFirst_[s];
            nonEmpty += n > 0;
            maxPer = std::max(maxPer, n);
            hist[n <= 4 ? n : n <= 8 ? 8 : n <= 16 ? 16 : n <= 32 ? 32 : n <= 64 ? 64 : 1000]++;
        }
        std::unordered_set<std::array<std::uint32_t, 3>, H3> triples;
        std::unordered_set<std::uint64_t> pairs;
        std::unordered_set<std::uint32_t> ys;
        for (const atlas::ColumnEntry &e : A.colEntries_) {
            std::uint32_t y0, y1;
            std::memcpy(&y0, &e.y0, 4); std::memcpy(&y1, &e.y1, 4);
            triples.insert({e.plate, y0, y1});
            pairs.insert((std::uint64_t(y0) << 32) | y1);
            ys.insert(y0); ys.insert(y1);
        }
        // plates per column (both classes): distinct plate ids per (column) and per 8x8 column block
        std::size_t maxPlatesCol = 0, sumPlatesCol = 0, cols = slots / 2;
        for (std::size_t c = 0; c < cols; ++c) {
            std::unordered_set<std::uint32_t> pl;
            for (std::uint32_t i = A.colFirst_[2 * c]; i < A.colFirst_[2 * c + 2]; ++i) pl.insert(A.colEntries_[i].plate);
            maxPlatesCol = std::max(maxPlatesCol, pl.size());
            sumPlatesCol += pl.size();
        }
        std::size_t blkMax = 0, blkSum = 0, blocks = 0;
        for (std::uint32_t bz = 0; bz < A.colNz_; bz += 8)
            for (std::uint32_t bx = 0; bx < A.colNx_; bx += 8) {
                std::unordered_set<std::uint32_t> pl;
                std::unordered_set<std::uint64_t> pr;
                for (std::uint32_t z = bz; z < std::min(bz + 8, A.colNz_); ++z)
                    for (std::uint32_t x = bx; x < std::min(bx + 8, A.colNx_); ++x) {
                        const std::size_t c = std::size_t(z) * A.colNx_ + x;
                        for (std::uint32_t i = A.colFirst_[2 * c]; i < A.colFirst_[2 * c + 2]; ++i) pl.insert(A.colEntries_[i].plate);
                    }
                if (pl.empty()) continue;
                ++blocks;
                blkMax = std::max(blkMax, pl.size());
                blkSum += pl.size();
            }
        // repeated slots: identical entry lists (plate, y0, y1) between columns
        std::unordered_map<std::string, std::uint32_t> slotDict;
        std::size_t slotDictEntries = 0;
        for (std::size_t s = 0; s < slots; ++s) {
            const std::uint32_t f = A.colFirst_[s], l = A.colFirst_[s + 1];
            if (f == l) continue;
            std::string key(reinterpret_cast<const char *>(&A.colEntries_[f]), (l - f) * sizeof(atlas::ColumnEntry));
            if (slotDict.emplace(std::move(key), 0).second) slotDictEntries += l - f;
        }
        const std::size_t unionPairs = A.colUnion_.size() / 2;
        std::printf("%s: %u x %u columns (%zu slots, %zu non-empty), %zu entries (max %zu per slot), union %zu intervals, entry members %zu\n",
                    argv[a], A.colNx_, A.colNz_, slots, nonEmpty, ne, maxPer, unionPairs, A.entMembers_.size());
        std::printf("  raw bytes: entries %.1f MB, firsts %.1f MB, union %.1f MB (+ firsts %.1f), entry members %.1f MB (+ firsts %.1f), overlap %.1f MB\n",
                    ne * 16 / 1e6, A.colFirst_.size() * 4 / 1e6, A.colUnion_.size() * 4 / 1e6, A.colUnionFirst_.size() * 4 / 1e6,
                    A.entMembers_.size() * 4 / 1e6, A.entMemFirst_.size() * 4 / 1e6, A.entOverlap_.size() / 1e6);
        std::printf("  entries per slot:");
        for (auto &kv : hist) std::printf(" <=%zu: %.1f%%", kv.first, 100.0 * kv.second / slots);
        std::printf("\n  distinct: (plate, y0, y1) %zu, (y0, y1) %zu, y values %zu; plates per column mean %.1f max %zu, per 8x8 block mean %.1f max %zu; "
                    "distinct slot lists %zu (%zu entries)\n",
                    triples.size(), pairs.size(), ys.size(), double(sumPlatesCol) / cols, maxPlatesCol, double(blkSum) / std::max<std::size_t>(1, blocks), blkMax,
                    slotDict.size(), slotDictEntries);
        // packed estimate: 16x16-column superblocks, per-superblock plate table and y base, per slot u16 offsets,
        // per entry local plate index + y0 (int16 units of q from the superblock base, escape 4 B) + y1 - y0 (u8 units,
        // escape 2 B); identical slot payloads within a superblock shared; union derived from the entries (0 B);
        // wheel-entry members: u8 count + u16 plate-local member index (u32 when the plate has > 65535 members)
        std::vector<std::uint32_t> memberLocal(A.members_.size());
        {
            std::vector<std::uint32_t> next(A.plates_.size(), 0);
            for (std::size_t m = 0; m < A.members_.size(); ++m) memberLocal[m] = next[A.members_[m].plate]++;
        }
        for (double q : {0.0, 0.001, 0.02}) {
            std::size_t bytes = 0, escY = 0, escD = 0, shared = 0, memb = 0;
            for (std::uint32_t bz = 0; bz < A.colNz_; bz += 16)
                for (std::uint32_t bx = 0; bx < A.colNx_; bx += 16) {
                    std::unordered_map<std::uint32_t, std::uint32_t> local;
                    double ybase = 1e30;
                    std::vector<std::size_t> sl;
                    for (std::uint32_t z = bz; z < std::min(bz + 16, A.colNz_); ++z)
                        for (std::uint32_t x = bx; x < std::min(bx + 16, A.colNx_); ++x)
                            for (int cls = 0; cls < 2; ++cls) sl.push_back((std::size_t(z) * A.colNx_ + x) * 2 + cls);
                    for (std::size_t s2 : sl)
                        for (std::uint32_t i = A.colFirst_[s2]; i < A.colFirst_[s2 + 1]; ++i) {
                            local.emplace(A.colEntries_[i].plate, std::uint32_t(local.size()));
                            ybase = std::min(ybase, double(A.colEntries_[i].y0));
                        }
                    if (local.empty()) { bytes += 4; continue; }   // empty superblock: a null pointer
                    const int idxB = local.size() <= 256 ? 1 : 2;
                    bytes += 16 + 4 * local.size() + 2 * sl.size();
                    std::unordered_set<std::string> payloads;
                    for (std::size_t s2 : sl) {
                        std::string pay;
                        for (std::uint32_t i = A.colFirst_[s2]; i < A.colFirst_[s2 + 1]; ++i) {
                            const atlas::ColumnEntry &e = A.colEntries_[i];
                            const std::uint32_t li = local[e.plate];
                            pay.append(reinterpret_cast<const char *>(&li), idxB);
                            if (q == 0.0) { pay.append(reinterpret_cast<const char *>(&e.y0), 4); pay.append(reinterpret_cast<const char *>(&e.y1), 4); continue; }
                            const long long y0q = (long long)std::floor((double(e.y0) - ybase) / q), y1q = (long long)std::ceil((double(e.y1) - ybase) / q);
                            const long long d = y1q - y0q;
                            if (y0q >= -32768 && y0q < 32767) { const std::int16_t v = std::int16_t(y0q); pay.append(reinterpret_cast<const char *>(&v), 2); }
                            else { ++escY; const std::int32_t v = std::int32_t(y0q); pay.append("\x7f\xff", 2); pay.append(reinterpret_cast<const char *>(&v), 4); }
                            if (d < 255) { const std::uint8_t v = std::uint8_t(d); pay.append(reinterpret_cast<const char *>(&v), 1); }
                            else { ++escD; const std::uint32_t v = std::uint32_t(d); pay.append("\xff", 1); pay.append(reinterpret_cast<const char *>(&v), 3); }
                        }
                        if (s2 % 2 == 0)   // wheel class: members
                            for (std::uint32_t i = A.colFirst_[s2]; i < A.colFirst_[s2 + 1]; ++i) {
                                const atlas::ColumnEntry &e = A.colEntries_[i];
                                if (e.plate == atlas::kLoose) continue;
                                const std::uint32_t f = A.entMemFirst_[i], l = A.entMemFirst_[i + 1];
                                const bool wide = A.plates_[e.plate].memberCount > 65535;
                                pay.push_back(char(std::min<std::uint32_t>(255, l - f)));
                                for (std::uint32_t m = f; m < l; ++m) {
                                    const std::uint32_t ml = memberLocal[A.entMembers_[m]];
                                    pay.append(reinterpret_cast<const char *>(&ml), wide ? 4 : 2);
                                    memb += wide ? 4 : 2;
                                }
                            }
                        if (!pay.empty() && !payloads.insert(pay).second) { ++shared; continue; }
                        bytes += pay.size();
                    }
                }
            std::printf("  packed (q = %s): %.1f MB (slot payloads shared %zu, y escapes %zu, dy escapes %zu, members %.1f MB)\n",
                        q == 0.0 ? "exact floats" : q == 0.001 ? "1 mm" : "2 cm", bytes / 1e6, shared, escY, escD, memb / 1e6);
        }
        if (recPath) {   // per recorded shape: the overlapping entries of its slot, exact vs rounded outward
            Recording rec;
            if (!LoadRecording(recPath, &rec)) { std::fprintf(stderr, "cannot load %s\n", recPath); return 1; }
            for (double q : {0.001, 0.005, 0.02}) {
                std::size_t shapes = 0, freeFlip = 0, setChange = 0, outside = 0;
                for (const Substep &st : rec.substeps)
                    for (int k = 0; k < 8; ++k) {
                        const Shape &sh = st.shapes[k];
                        const int cls = k >= 4 ? 0 : 1;
                        const float rmax = std::max(sh.r[0], std::max(sh.r[1], sh.r[2]));
                        const std::uint32_t slot = A.ColumnSlot(sh.t, cls);
                        ++shapes;
                        if (slot == 0xFFFFFFFFu) { ++outside; continue; }
                        std::uint32_t n;
                        const atlas::ColumnEntry *e = A.Entries(slot, &n);
                        const float lo = sh.t[1] - rmax, hi = sh.t[1] + rmax;
                        std::vector<std::uint32_t> ex, qq;
                        for (std::uint32_t i = 0; i < n; ++i) {
                            if (e[i].y0 <= hi && e[i].y1 >= lo) ex.push_back(e[i].plate);
                            const double y0 = std::floor(double(e[i].y0) / q) * q, y1 = std::ceil(double(e[i].y1) / q) * q;
                            if (y0 <= hi && y1 >= lo) qq.push_back(e[i].plate);
                        }
                        freeFlip += ex.empty() != qq.empty();
                        setChange += ex != qq;
                    }
                std::printf("  decisions vs %s (FV poses, %zu shapes, %zu outside the grid): y rounded outward to %.0f mm: free flips %zu (%.4f%%), overlap-set changes %zu (%.4f%%)\n",
                            recPath, shapes, outside, q * 1000, freeFlip, 100.0 * freeFlip / shapes, setChange, 100.0 * setChange / shapes);
            }
            ++a;
        }
        std::fflush(stdout);
    }
    return 0;
}
