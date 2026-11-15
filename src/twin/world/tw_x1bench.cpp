// tw_x1bench: milestone X1 (design.md §11). Exact collision at FV's substep
// poses with AVX-512 packets:
//   per tick:    BVH4 query of the car's union box + one tick of travel -> SoA tiles of 16 triangles
//   per substep: shape setup (8 unit maps, SoA) -> AVX-512 tile filter (union, then 8 shapes,
//                compress to pair lists) -> 16-pair packets (permute gathers from the tile and the
//                8-shape table) -> ExactLanes<f32x16> stage 1 + 2 -> per-shape merge.
// Gates: 0 FV misses; packet contacts bit-identical to the scalar reference (ExactLanes<f32x1>)
// and to ExactTest on the BVH-direct pairs. Batch: 16 independent candidates interleaved.
// Usage: tw_x1bench REC [REC...]
#include <immintrin.h>
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "tw_accel.h"
#include "tw_blocks.h"
#include "tw_candidates.h"
#include "tw_packets.h"

using namespace tw;
using tw::math::f32x1;
using tw::math::f32x16;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kFields = 21;  // shape table: A 9 (N = Aᵀ, same values), t 3, M 9

struct alignas(64) ShapeIn {    // per-substep car-shape input in SoA (what an engine's pose stage produces)
    float axis[9][16];          // axis[j][i] at [3j + i], lanes 0-7 = shapes, 8-15 copies of shape 0
    float t[3][16];
    float r[3][16], ir[3][16];
};

struct alignas(64) ShapeTable {
    float f[kFields][16];       // lanes 0-7 = shapes
    float lo[3][16], hi[3][16]; // inflated boxes
    Box3 unionBox;
};

inline void MakeShapeIn(const Substep &s, ShapeIn *in) {
    for (int l = 0; l < 16; ++l) {
        const Shape &sh = s.shapes[l < kShapes ? l : 0];
        for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) in->axis[3 * j + i][l] = sh.axis[j][i];
        for (int i = 0; i < 3; ++i) { in->t[i][l] = sh.t[i]; in->r[i][l] = sh.r[i]; in->ir[i][l] = 1.0f / sh.r[i]; }
    }
}

// 8 unit maps at once (lanes = shapes): A = diag(1/r) axesᵀ, M = axes diag(r), inflated AABBs.
inline void SetupShapes(const ShapeIn &in, ShapeTable &T) {
    for (int j = 0; j < 3; ++j) {
        const __m512 ir = _mm512_load_ps(in.ir[j]), r = _mm512_load_ps(in.r[j]);
        for (int i = 0; i < 3; ++i) {
            const __m512 ax = _mm512_load_ps(in.axis[3 * j + i]);
            _mm512_store_ps(T.f[j * 3 + i], _mm512_mul_ps(ax, ir));       // A[j][i]
            _mm512_store_ps(T.f[12 + i * 3 + j], _mm512_mul_ps(ax, r));   // M[i][j]
        }
    }
    const __m512 eps = _mm512_set1_ps(kEps);
    for (int i = 0; i < 3; ++i) {
        const __m512 t = _mm512_load_ps(in.t[i]);
        _mm512_store_ps(T.f[9 + i], t);
        const __m512 h = _mm512_add_ps(_mm512_add_ps(_mm512_abs_ps(_mm512_load_ps(T.f[12 + i * 3])), _mm512_abs_ps(_mm512_load_ps(T.f[12 + i * 3 + 1]))),
                                       _mm512_abs_ps(_mm512_load_ps(T.f[12 + i * 3 + 2])));
        const __m512 lo = _mm512_sub_ps(_mm512_sub_ps(t, h), eps), hi = _mm512_add_ps(_mm512_add_ps(t, h), eps);
        _mm512_store_ps(T.lo[i], lo); _mm512_store_ps(T.hi[i], hi);
        T.unionBox.lo[i] = _mm512_mask_reduce_min_ps(0xff, lo);
        T.unionBox.hi[i] = _mm512_mask_reduce_max_ps(0xff, hi);
    }
}

struct alignas(64) Tile {
    float v[9][16];    // v0xyz v1xyz v2xyz
    float lo[3][16], hi[3][16];
    std::uint32_t id[16];
};

struct ContactRec {
    std::uint32_t pair;  // shape << 28 | tri
    std::int32_t kind;
    float P[3], N[3], Sep[3], K[3];
};

float g_marginTicks = 1.0f;
std::size_t g_spread = 0;

struct X1Cand {  // one simulated car: its cache and scratch
    Box3 cacheBox = EmptyBox();
    bool valid = false;
    std::int32_t localRoot = 0;
    Box3 l2Box = EmptyBox();
    bool l2Valid = false;
    std::vector<Tile> l2;
    std::size_t l2Queries = 0;
    std::vector<Tile> tiles;
    std::vector<std::uint32_t> ids;
    ShapeTable T;
    // pair lists from the filter: per tile range into (shape, lane) arrays
    std::vector<std::int32_t> pShape, pLane;
    std::vector<std::uint32_t> tileFirst, tileCount, tileOf;
    std::vector<std::uint32_t> pairs;  // stage-1 pairs this substep (for the gate)
    std::vector<ContactRec> contacts;
    std::vector<MergedContact> raw, merged[kShapes];
    std::uint64_t tBvh = 0, tQuery = 0, tSetup = 0, tFilter = 0, tPackets = 0, tMerge = 0;
    std::size_t queries = 0, stage1Packets = 0, stage2Packets = 0, pairCount = 0, cachedTris = 0;
};

float g_l2Margin = 0.0f;  // m, plus 3 ticks of travel; 0 disables the second level

void FillTiles(std::vector<Tile> &tiles, std::vector<std::uint32_t> *ids, const World &w, const Bvh4 &bvh, const Box3 &box) {
    tiles.clear();
    std::uint32_t lane = 16;
    bvh.Query(box, [&](std::uint32_t id) {
        if (lane == 16) {
            tiles.emplace_back();
            Tile &nt = tiles.back();
            for (int a = 0; a < 3; ++a) { _mm512_store_ps(nt.lo[a], _mm512_set1_ps(1e30f)); _mm512_store_ps(nt.hi[a], _mm512_set1_ps(-1e30f)); }
            for (int a = 0; a < 9; ++a) _mm512_store_ps(nt.v[a], _mm512_setzero_ps());
            lane = 0;
        }
        Tile &tl = tiles.back();
        const Tri &tr = w.tris[id];
        for (int a = 0; a < 3; ++a) { tl.v[a][lane] = tr.v[0][a]; tl.v[3 + a][lane] = tr.v[1][a]; tl.v[6 + a][lane] = tr.v[2][a]; tl.lo[a][lane] = tr.lo[a]; tl.hi[a][lane] = tr.hi[a]; }
        tl.id[lane] = id;
        if (ids) ids->push_back(id);
        ++lane;
    });
}

void Query(X1Cand &c, const World &w, const Bvh4 &bvh, const Substep &s) {
    const float wl = std::sqrt(s.angVel[0] * s.angVel[0] + s.angVel[1] * s.angVel[1] + s.angVel[2] * s.angVel[2]);
    const float sp = std::sqrt(s.vel[0] * s.vel[0] + s.vel[1] * s.vel[1] + s.vel[2] * s.vel[2]);
    c.cacheBox = Inflate(c.T.unionBox, (sp + 2.6f * wl) * 0.01f * 1.25f * g_marginTicks + 0.05f);
    c.ids.clear();
    const std::uint64_t q0 = __rdtsc();
    if (g_l2Margin <= 0.0f) {
        FillTiles(c.tiles, &c.ids, w, bvh, c.cacheBox);
    } else {
        if (!c.l2Valid || !Contains(c.l2Box, c.cacheBox)) {  // second level: big box, rare BVH query
            c.l2Box = Inflate(c.cacheBox, g_l2Margin + (sp + 2.6f * wl) * 0.03f);
            FillTiles(c.l2, nullptr, w, bvh, c.l2Box);
            c.l2Valid = true; ++c.l2Queries;
        }
        // first level: SIMD filter of the L2 tiles by the cache box, compress-store into new tiles
        c.tiles.clear();
        const __m512 qlo0 = _mm512_set1_ps(c.cacheBox.lo[0]), qlo1 = _mm512_set1_ps(c.cacheBox.lo[1]), qlo2 = _mm512_set1_ps(c.cacheBox.lo[2]);
        const __m512 qhi0 = _mm512_set1_ps(c.cacheBox.hi[0]), qhi1 = _mm512_set1_ps(c.cacheBox.hi[1]), qhi2 = _mm512_set1_ps(c.cacheBox.hi[2]);
        std::uint32_t lane = 16;
        for (const Tile &src : c.l2) {
            const __m512 lo0 = _mm512_load_ps(src.lo[0]), lo1 = _mm512_load_ps(src.lo[1]), lo2 = _mm512_load_ps(src.lo[2]);
            const __m512 hi0 = _mm512_load_ps(src.hi[0]), hi1 = _mm512_load_ps(src.hi[1]), hi2 = _mm512_load_ps(src.hi[2]);
            __mmask16 m = _mm512_cmp_ps_mask(lo0, qhi0, _CMP_LE_OQ) & _mm512_cmp_ps_mask(qlo0, hi0, _CMP_LE_OQ) &
                          _mm512_cmp_ps_mask(lo1, qhi1, _CMP_LE_OQ) & _mm512_cmp_ps_mask(qlo1, hi1, _CMP_LE_OQ) &
                          _mm512_cmp_ps_mask(lo2, qhi2, _CMP_LE_OQ) & _mm512_cmp_ps_mask(qlo2, hi2, _CMP_LE_OQ);
            while (m) {  // copy hits lane by lane into the open tile (few per L2 tile)
                const int l = __builtin_ctz(m); m &= m - 1;
                if (lane == 16) {
                    c.tiles.emplace_back();
                    Tile &nt = c.tiles.back();
                    for (int a = 0; a < 3; ++a) { _mm512_store_ps(nt.lo[a], _mm512_set1_ps(1e30f)); _mm512_store_ps(nt.hi[a], _mm512_set1_ps(-1e30f)); }
                    for (int a = 0; a < 9; ++a) _mm512_store_ps(nt.v[a], _mm512_setzero_ps());
                    lane = 0;
                }
                Tile &tl = c.tiles.back();
                for (int a = 0; a < 9; ++a) tl.v[a][lane] = src.v[a][l];
                for (int a = 0; a < 3; ++a) { tl.lo[a][lane] = src.lo[a][l]; tl.hi[a][lane] = src.hi[a][l]; }
                tl.id[lane] = src.id[l];
                c.ids.push_back(src.id[l]);
                ++lane;
            }
        }
    }
    c.tBvh += __rdtsc() - q0;
    c.valid = true;
    ++c.queries;
}

// AVX-512 tile filter: tile x union, then tile x 8 shapes, compress (shape, lane) pairs per tile.
void Filter(X1Cand &c, bool keepPairs) {
    if (keepPairs) c.pairs.clear();
    c.tileFirst.clear(); c.tileCount.clear(); c.tileOf.clear();
    if (c.pShape.size() < c.tiles.size() * 16 * kShapes + 16) { c.pShape.resize(c.tiles.size() * 16 * kShapes + 16); c.pLane.resize(c.pShape.size()); }
    const __m512 ulo0 = _mm512_set1_ps(c.T.unionBox.lo[0]), ulo1 = _mm512_set1_ps(c.T.unionBox.lo[1]), ulo2 = _mm512_set1_ps(c.T.unionBox.lo[2]);
    const __m512 uhi0 = _mm512_set1_ps(c.T.unionBox.hi[0]), uhi1 = _mm512_set1_ps(c.T.unionBox.hi[1]), uhi2 = _mm512_set1_ps(c.T.unionBox.hi[2]);
    const __m512i laneIdx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    std::uint32_t np = 0;
    c.cachedTris += c.ids.size();
    for (std::uint32_t ti = 0; ti < c.tiles.size(); ++ti) {
        const Tile &tl = c.tiles[ti];
        const __m512 lo0 = _mm512_load_ps(tl.lo[0]), lo1 = _mm512_load_ps(tl.lo[1]), lo2 = _mm512_load_ps(tl.lo[2]);
        const __m512 hi0 = _mm512_load_ps(tl.hi[0]), hi1 = _mm512_load_ps(tl.hi[1]), hi2 = _mm512_load_ps(tl.hi[2]);
        const __mmask16 mu = _mm512_cmp_ps_mask(lo0, uhi0, _CMP_LE_OQ) & _mm512_cmp_ps_mask(ulo0, hi0, _CMP_LE_OQ) &
                             _mm512_cmp_ps_mask(lo1, uhi1, _CMP_LE_OQ) & _mm512_cmp_ps_mask(ulo1, hi1, _CMP_LE_OQ) &
                             _mm512_cmp_ps_mask(lo2, uhi2, _CMP_LE_OQ) & _mm512_cmp_ps_mask(ulo2, hi2, _CMP_LE_OQ);
        if (!mu) continue;
        const std::uint32_t first = np;
        for (int s = 0; s < kShapes; ++s) {
            const __mmask16 m = mu & _mm512_cmp_ps_mask(lo0, _mm512_set1_ps(c.T.hi[0][s]), _CMP_LE_OQ) &
                                _mm512_cmp_ps_mask(_mm512_set1_ps(c.T.lo[0][s]), hi0, _CMP_LE_OQ) &
                                _mm512_cmp_ps_mask(lo1, _mm512_set1_ps(c.T.hi[1][s]), _CMP_LE_OQ) &
                                _mm512_cmp_ps_mask(_mm512_set1_ps(c.T.lo[1][s]), hi1, _CMP_LE_OQ) &
                                _mm512_cmp_ps_mask(lo2, _mm512_set1_ps(c.T.hi[2][s]), _CMP_LE_OQ) &
                                _mm512_cmp_ps_mask(_mm512_set1_ps(c.T.lo[2][s]), hi2, _CMP_LE_OQ);
            if (!m) continue;
            _mm512_mask_compressstoreu_epi32(c.pLane.data() + np, m, laneIdx);
            _mm512_storeu_si512(c.pShape.data() + np, _mm512_set1_epi32(s));
            np += static_cast<std::uint32_t>(__builtin_popcount(m));
        }
        if (np == first) continue;
        c.tileOf.push_back(ti); c.tileFirst.push_back(first); c.tileCount.push_back(np - first);
        if (keepPairs) for (std::uint32_t k = first; k < np; ++k) c.pairs.push_back(PairCode(c.pShape[k], tl.id[c.pLane[k]]));
    }
    c.pairCount += np;
}

// Stage 1 + 2 in 16-pair packets per tile; M (and N = Aᵀ) are permuted only for stage-2 packets.
void Packets(X1Cand &c) {
    c.contacts.clear();
    for (std::size_t g = 0; g < c.tileOf.size(); ++g) {
        const Tile &tl = c.tiles[c.tileOf[g]];
        const std::uint32_t first = c.tileFirst[g], np = c.tileCount[g];
        for (std::uint32_t p0 = 0; p0 < np; p0 += 16) {
            const int cnt = static_cast<int>(std::min<std::uint32_t>(16, np - p0));
            const __mmask16 live = static_cast<__mmask16>((1u << cnt) - 1u);
            const __m512i is = _mm512_maskz_loadu_epi32(live, c.pShape.data() + first + p0);
            const __m512i il = _mm512_maskz_loadu_epi32(live, c.pLane.data() + first + p0);
            f32x16 A[3][3], t[3], v[3][3];
            for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) A[j][i] = f32x16(_mm512_permutexvar_ps(is, _mm512_load_ps(c.T.f[j * 3 + i])));
            for (int i = 0; i < 3; ++i) t[i] = f32x16(_mm512_permutexvar_ps(is, _mm512_load_ps(c.T.f[9 + i])));
            for (int k = 0; k < 3; ++k) for (int a = 0; a < 3; ++a) v[k][a] = f32x16(_mm512_permutexvar_ps(il, _mm512_load_ps(tl.v[3 * k + a])));
            Stage1State<f32x16> st;
            ++c.stage1Packets;
            if (!Stage1(A, t, v, st)) continue;
            ++c.stage2Packets;
            f32x16 M[3][3], N[3][3];
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
                M[i][j] = f32x16(_mm512_permutexvar_ps(is, _mm512_load_ps(c.T.f[12 + i * 3 + j])));
                N[i][j] = A[j][i];
            }
            PairOut<f32x16> o;
            Stage2(st, t, M, N, o);
            alignas(64) std::int32_t kinds[16];
            o.kind.store(kinds);
            std::uint32_t hits = 0;
            for (int l = 0; l < cnt; ++l) hits |= static_cast<std::uint32_t>(kinds[l] != 0) << l;
            if (!hits) continue;
            alignas(64) float P[3][16], Nn[3][16], S[3][16], K[3][16];
            for (int i = 0; i < 3; ++i) { o.P[i].store(P[i]); o.Nrm[i].store(Nn[i]); o.Sep[i].store(S[i]); o.K[i].store(K[i]); }
            while (hits) {
                const int l = __builtin_ctz(hits); hits &= hits - 1;
                ContactRec r;
                r.pair = PairCode(c.pShape[first + p0 + l], tl.id[c.pLane[first + p0 + l]]);
                r.kind = kinds[l];
                for (int i = 0; i < 3; ++i) { r.P[i] = P[i][l]; r.N[i] = Nn[i][l]; r.Sep[i] = S[i][l]; r.K[i] = K[i][l]; }
                c.contacts.push_back(r);
            }
        }
    }
}

// Scalar reference: same tiles, scalar filter, ExactLanes<f32x1> per pair.
void SubstepScalar(X1Cand &c) {
    c.contacts.clear();
    for (const Tile &tl : c.tiles)
        for (int l = 0; l < 16; ++l) {
            const Box3 tb = {{tl.lo[0][l], tl.lo[1][l], tl.lo[2][l]}, {tl.hi[0][l], tl.hi[1][l], tl.hi[2][l]}};
            if (!Overlap(tb, c.T.unionBox)) continue;
            for (int s = 0; s < kShapes; ++s) {
                const Box3 sb = {{c.T.lo[0][s], c.T.lo[1][s], c.T.lo[2][s]}, {c.T.hi[0][s], c.T.hi[1][s], c.T.hi[2][s]}};
                if (!Overlap(tb, sb)) continue;
                PairIn<f32x1> q;
                for (int j = 0; j < 3; ++j)
                    for (int i = 0; i < 3; ++i) { q.A[j][i] = c.T.f[j * 3 + i][s]; q.M[i][j] = c.T.f[12 + i * 3 + j][s]; q.N[i][j] = c.T.f[j * 3 + i][s]; }
                for (int i = 0; i < 3; ++i) q.t[i] = c.T.f[9 + i][s];
                for (int k = 0; k < 3; ++k) for (int a = 0; a < 3; ++a) q.v[k][a] = tl.v[3 * k + a][l];
                PairOut<f32x1> o;
                if (!ExactLanes(q, o) || o.kind.v == 0) continue;
                ContactRec r;
                r.pair = PairCode(s, tl.id[l]); r.kind = o.kind.v;
                for (int i = 0; i < 3; ++i) { r.P[i] = o.P[i].v; r.N[i] = o.Nrm[i].v; r.Sep[i] = o.Sep[i].v; r.K[i] = o.K[i].v; }
                c.contacts.push_back(r);
            }
        }
}

void Merge(X1Cand &c) {
    // per-shape merge only where a shape has both a FACE and a feature contact (otherwise it keeps everything)
    std::uint32_t prim = 0, feat = 0;
    for (const ContactRec &r : c.contacts) { const int s = PairShape(r.pair); if (r.kind == kFace) prim |= 1u << s; else feat |= 1u << s; }
    std::uint32_t todo = prim & feat;
    while (todo) {
        const int s = __builtin_ctz(todo); todo &= todo - 1;
        c.raw.clear();
        for (const ContactRec &r : c.contacts) {
            if (PairShape(r.pair) != s) continue;
            MergedContact m;
            m.c.kind = r.kind;
            for (int i = 0; i < 3; ++i) { m.c.P[i] = r.P[i]; m.c.N[i] = r.N[i]; m.c.Sep[i] = r.Sep[i]; m.c.K[i] = r.K[i]; }
            m.primary = r.kind == kFace; m.material = 0;
            c.raw.push_back(m);
        }
        MergeShape(c.raw, &c.merged[s]);
    }
}

// ---- batch: stage 1 per candidate, survivors compressed into one dense stage-2 stream for all candidates
struct alignas(64) Survivors {
    static constexpr int kF = 34;  // X 9, n 3, d 1, t 3, M 9, N 9
    std::vector<float> f[kF];
    std::vector<std::uint32_t> pair, owner;
    std::size_t n = 0;
    void Reset(std::size_t cap) { for (auto &v : f) v.resize(cap + 16); pair.resize(cap + 16); owner.resize(cap + 16); n = 0; }
};

void Stage1Compress(X1Cand &c, std::uint32_t ownerId, Survivors &S) {
    for (std::size_t g = 0; g < c.tileOf.size(); ++g) {
        const Tile &tl = c.tiles[c.tileOf[g]];
        const std::uint32_t first = c.tileFirst[g], np = c.tileCount[g];
        for (std::uint32_t p0 = 0; p0 < np; p0 += 16) {
            const int cnt = static_cast<int>(std::min<std::uint32_t>(16, np - p0));
            const __mmask16 live = static_cast<__mmask16>((1u << cnt) - 1u);
            const __m512i is = _mm512_maskz_loadu_epi32(live, c.pShape.data() + first + p0);
            const __m512i il = _mm512_maskz_loadu_epi32(live, c.pLane.data() + first + p0);
            f32x16 A[3][3], t[3], v[3][3];
            for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) A[j][i] = f32x16(_mm512_permutexvar_ps(is, _mm512_load_ps(c.T.f[j * 3 + i])));
            for (int i = 0; i < 3; ++i) t[i] = f32x16(_mm512_permutexvar_ps(is, _mm512_load_ps(c.T.f[9 + i])));
            for (int k = 0; k < 3; ++k) for (int a = 0; a < 3; ++a) v[k][a] = f32x16(_mm512_permutexvar_ps(il, _mm512_load_ps(tl.v[3 * k + a])));
            Stage1State<f32x16> st;
            ++c.stage1Packets;
            if (!Stage1(A, t, v, st)) continue;
            const __mmask16 m = st.alive.v & live;
            if (!m) continue;
            const std::size_t at = S.n;
            int fi = 0;
            for (int k = 0; k < 3; ++k) for (int a = 0; a < 3; ++a) _mm512_mask_compressstoreu_ps(S.f[fi++].data() + at, m, st.X[k][a].v);
            for (int a = 0; a < 3; ++a) _mm512_mask_compressstoreu_ps(S.f[fi++].data() + at, m, st.n[a].v);
            _mm512_mask_compressstoreu_ps(S.f[fi++].data() + at, m, st.d.v);
            for (int a = 0; a < 3; ++a) _mm512_mask_compressstoreu_ps(S.f[fi++].data() + at, m, t[a].v);
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
                _mm512_mask_compressstoreu_ps(S.f[fi++].data() + at, m, _mm512_permutexvar_ps(is, _mm512_load_ps(c.T.f[12 + i * 3 + j])));
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) _mm512_mask_compressstoreu_ps(S.f[fi++].data() + at, m, A[j][i].v);
            std::uint32_t mm = m;
            std::size_t w = at;
            while (mm) { const int l = __builtin_ctz(mm); mm &= mm - 1; S.pair[w] = PairCode(c.pShape[first + p0 + l], tl.id[c.pLane[first + p0 + l]]); S.owner[w] = ownerId; ++w; }
            S.n = w;
        }
    }
}

void Stage2Dense(Survivors &S, std::vector<X1Cand> &cs, std::size_t *packets) {
    for (std::size_t p0 = 0; p0 < S.n; p0 += 16) {
        const int cnt = static_cast<int>(std::min<std::size_t>(16, S.n - p0));
        const __mmask16 live = static_cast<__mmask16>((1u << cnt) - 1u);
        auto L = [&](int fi) { return f32x16(_mm512_maskz_loadu_ps(live, S.f[fi].data() + p0)); };
        Stage1State<f32x16> st;
        int fi = 0;
        for (int k = 0; k < 3; ++k) for (int a = 0; a < 3; ++a) st.X[k][a] = L(fi++);
        for (int a = 0; a < 3; ++a) st.n[a] = L(fi++);
        st.d = L(fi++);
        st.alive = tw::math::m32x16(live);
        f32x16 t[3], M[3][3], N[3][3];
        for (int a = 0; a < 3; ++a) t[a] = L(fi++);
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) M[i][j] = L(fi++);
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) N[i][j] = L(fi++);
        PairOut<f32x16> o;
        Stage2(st, t, M, N, o);
        ++*packets;
        alignas(64) std::int32_t kinds[16];
        o.kind.store(kinds);
        std::uint32_t hits = 0;
        for (int l = 0; l < cnt; ++l) hits |= static_cast<std::uint32_t>(kinds[l] != 0) << l;
        if (!hits) continue;
        alignas(64) float P[3][16], Nn[3][16], Sp[3][16], K[3][16];
        for (int i = 0; i < 3; ++i) { o.P[i].store(P[i]); o.Nrm[i].store(Nn[i]); o.Sep[i].store(Sp[i]); o.K[i].store(K[i]); }
        while (hits) {
            const int l = __builtin_ctz(hits); hits &= hits - 1;
            ContactRec r;
            r.pair = S.pair[p0 + l]; r.kind = kinds[l];
            for (int i = 0; i < 3; ++i) { r.P[i] = P[i][l]; r.N[i] = Nn[i][l]; r.Sep[i] = Sp[i][l]; r.K[i] = K[i][l]; }
            cs[S.owner[p0 + l]].contacts.push_back(r);
        }
    }
}

inline void Prepare(X1Cand &c, const World &w, const Bvh4 &bvh, const Substep &s, const ShapeIn &in) {
    std::uint64_t a = __rdtsc();
    SetupShapes(in, c.T);
    std::uint64_t b = __rdtsc();
    c.tSetup += b - a;
    if (!c.valid || !Contains(c.cacheBox, c.T.unionBox)) Query(c, w, bvh, s);
    c.tQuery += __rdtsc() - b;
}

// one full substep for a candidate (timed phases)
void Step(X1Cand &c, const World &w, const Bvh4 &bvh, const Substep &s, const ShapeIn &in, bool keepPairs) {
    Prepare(c, w, bvh, s, in);
    std::uint64_t d = __rdtsc();
    Filter(c, keepPairs);
    std::uint64_t e = __rdtsc();
    Packets(c);
    std::uint64_t f = __rdtsc();
    Merge(c);
    c.tFilter += e - d; c.tPackets += f - e; c.tMerge += __rdtsc() - f;
}

bool SameBits(const ContactRec &a, const ContactRec &b) {
    return a.pair == b.pair && a.kind == b.kind && std::memcmp(a.P, b.P, sizeof a.P) == 0 && std::memcmp(a.N, b.N, sizeof a.N) == 0 &&
           std::memcmp(a.Sep, b.Sep, sizeof a.Sep) == 0 && std::memcmp(a.K, b.K, sizeof a.K) == 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: tw_x1bench REC...\n"); return 2; }
    if (const char *m = std::getenv("TW_MARGIN_TICKS")) g_marginTicks = static_cast<float>(std::atof(m));
    if (const char *m = std::getenv("TW_L2_MARGIN")) g_l2Margin = static_cast<float>(std::atof(m));
    std::printf("| replay | substeps/tick | single: setup / query / filter / packets / merge (cyc/substep) | stage 1+2 cyc/substep | all collision cyc/substep | cyc/tick | batch-16 cyc/candidate-tick | pairs, stage-1 / stage-2 packets per substep | FV misses / checked | bit-identical (L16 vs L1 vs ExactTest) |\n");
    std::printf("|---|---:|---|---:|---:|---:|---:|---|---|---|\n");
    for (int a = 1; a < argc; ++a) {
        Recording rec;
        if (!LoadRecording(argv[a], &rec)) { std::fprintf(stderr, "cannot load %s\n", argv[a]); return 1; }
        const std::size_t N = rec.substeps.size();
        const double spt = static_cast<double>(N) / rec.ticks;
        Bvh4 bvh; bvh.Build(rec.world);
        std::vector<ShapeIn> shapeIn(N);
        for (std::size_t i = 0; i < N; ++i) MakeShapeIn(rec.substeps[i], &shapeIn[i]);
        // ---- single-run timed pass (wall clock + rdtsc split)
        X1Cand c1;
        auto w0 = Clock::now(); const std::uint64_t r0 = __rdtsc();
        for (std::size_t i = 0; i < N; ++i) Step(c1, rec.world, bvh, rec.substeps[i], shapeIn[i], false);
        const std::uint64_t r1 = __rdtsc(); auto w1 = Clock::now();
        const double wallCyc = std::chrono::duration<double, std::nano>(w1 - w0).count() * 5.3;
        const double k = wallCyc / static_cast<double>(r1 - r0);  // core cycles per TSC tick
        const double cachedPerSub = double(c1.cachedTris) / N, queriesPerTick = double(c1.queries) / rec.ticks;
        const double setup = c1.tSetup * k / N, query = c1.tQuery * k / N, filt = c1.tFilter * k / N, pack = c1.tPackets * k / N,
                     merge = c1.tMerge * k / N, total = wallCyc / N;
        // ---- batch: 16 candidates at decorrelated offsets, round-robin per substep
        if (const char *sp = std::getenv("TW_SPREAD"); sp && *sp) g_spread = static_cast<std::size_t>(std::atoll(sp)); else g_spread = N / 16;
        std::vector<X1Cand> cs(16);
        Survivors S; S.Reset(4096);
        std::size_t densePackets = 0;
        std::uint64_t bP = 0, bF = 0, bS1 = 0, bS2 = 0, bM = 0;
        auto b0 = Clock::now();
        for (std::size_t i = 0; i < N; ++i) {  // phase-interleaved: each phase for all 16 candidates back to back
            std::size_t idx[16];
            for (int ci = 0; ci < 16; ++ci) idx[ci] = (i + ci * g_spread) % N;
            const std::uint64_t q0 = __rdtsc();
            for (int ci = 0; ci < 16; ++ci) Prepare(cs[ci], rec.world, bvh, rec.substeps[idx[ci]], shapeIn[idx[ci]]);
            const std::uint64_t q1 = __rdtsc();
            for (int ci = 0; ci < 16; ++ci) Filter(cs[ci], false);
            const std::uint64_t q2 = __rdtsc();
            bP += q1 - q0; bF += q2 - q1;
            std::size_t cap = 0;
            for (int ci = 0; ci < 16; ++ci) cap += cs[ci].tileCount.empty() ? 0 : cs[ci].pShape.size();
            if (S.f[0].size() < cap + 16) S.Reset(cap); else S.n = 0;
            const std::uint64_t q3 = __rdtsc();
            for (int ci = 0; ci < 16; ++ci) { cs[ci].contacts.clear(); Stage1Compress(cs[ci], static_cast<std::uint32_t>(ci), S); }
            const std::uint64_t q4 = __rdtsc();
            Stage2Dense(S, cs, &densePackets);
            const std::uint64_t q5 = __rdtsc();
            for (int ci = 0; ci < 16; ++ci) Merge(cs[ci]);
            bS1 += q4 - q3; bS2 += q5 - q4; bM += __rdtsc() - q5;
        }
        auto b1 = Clock::now();
        const double batchCyc = std::chrono::duration<double, std::nano>(b1 - b0).count() * 5.3 / (16.0 * N);
        std::size_t batch1 = 0, batch2dense = densePackets;
        {
            const double kk = k / (16.0 * N);
            std::fprintf(stderr, "  batch split per candidate-substep (cycles): setup+query %.0f, filter %.0f, stage1+compress %.0f, dense stage2 %.0f, merge %.0f\n",
                         bP * kk, bF * kk, bS1 * kk, bS2 * kk, bM * kk);
        }
        for (const auto &cc : cs) batch1 += cc.stage1Packets;
        // dense-stage-2 bit identity vs per-candidate packets (sampled: candidate 0 replayed alone over the first 2000 substeps)
        std::size_t denseDiff = 0;
        {
            std::vector<X1Cand> one(1); X1Cand ref; Survivors S1; S1.Reset(4096); std::size_t dp = 0;
            for (std::size_t i = 0; i < std::min<std::size_t>(N, 2000); ++i) {
                Prepare(one[0], rec.world, bvh, rec.substeps[i], shapeIn[i]); Filter(one[0], false);
                S1.n = 0; one[0].contacts.clear(); Stage1Compress(one[0], 0, S1); Stage2Dense(S1, one, &dp);
                Step(ref, rec.world, bvh, rec.substeps[i], shapeIn[i], false);
                auto A = one[0].contacts, B = ref.contacts;
                auto lt = [](const ContactRec &x, const ContactRec &y) { return x.pair != y.pair ? x.pair < y.pair : x.kind < y.kind; };
                std::sort(A.begin(), A.end(), lt); std::sort(B.begin(), B.end(), lt);
                bool same = A.size() == B.size();
                for (std::size_t q = 0; same && q < A.size(); ++q) same = SameBits(A[q], B[q]);
                denseDiff += !same;
            }
        }
        // ---- gates: FV misses (stage-1 pairs must include FV's contact triangles) and bit identity
        X1Cand g, gs;
        std::size_t noiseUnmatched = 0, checked = 0, misses = 0, diffL1 = 0, diffRef = 0, substepsWithContacts = 0;
        std::vector<std::uint32_t> refPairs;
        for (std::size_t i = 0; i < N; ++i) {
            const Substep &s = rec.substeps[i];
            Step(g, rec.world, bvh, s, shapeIn[i], true);
            // scalar reference on the same tiles
            gs.T = g.T; gs.tiles = g.tiles;
            SubstepScalar(gs);
            auto key = [](const ContactRec &r) { return std::make_pair(r.pair, r.kind); };
            std::vector<ContactRec> A = g.contacts, B = gs.contacts;
            auto byKey = [&](const ContactRec &x, const ContactRec &y) { return key(x) < key(y); };
            std::sort(A.begin(), A.end(), byKey); std::sort(B.begin(), B.end(), byKey);
            bool same = A.size() == B.size();
            for (std::size_t q = 0; same && q < A.size(); ++q) same = SameBits(A[q], B[q]);
            diffL1 += !same;
            // ExactTest on BVH-direct pairs at the same pose (independent reference)
            Frame fr; MakeFrame(s, &fr);
            refPairs.clear();
            bvh.Query(fr.unionBox, [&](std::uint32_t id) {
                const Tri &t = rec.world.tris[id];
                const Box3 tb = {{t.lo[0], t.lo[1], t.lo[2]}, {t.hi[0], t.hi[1], t.hi[2]}};
                for (int sh = 0; sh < kShapes; ++sh) if (Overlap(tb, fr.inflated[sh])) refPairs.push_back(PairCode(sh, id));
            });
            std::vector<ContactRec> R;
            for (std::uint32_t p : refPairs) {
                Contact ct;
                const int kind = ExactTest(fr.sm[PairShape(p)], rec.world.tris[PairTri(p)], &ct);
                if (kind == kNone) continue;
                ContactRec r; r.pair = p; r.kind = kind;
                for (int q = 0; q < 3; ++q) { r.P[q] = ct.P[q]; r.N[q] = ct.N[q]; r.Sep[q] = ct.Sep[q]; r.K[q] = ct.K[q]; }
                R.push_back(r);
            }
            std::sort(R.begin(), R.end(), byKey);
            bool sameR = R.size() == A.size();
            for (std::size_t q = 0; sameR && q < A.size(); ++q) sameR = SameBits(R[q], A[q]);
            diffRef += !sameR;
            if (!A.empty()) ++substepsWithContacts;
            // FV gate: every FV raw contact must match a packet contact (shape, point, normal) whose pair was tested
            std::unordered_set<std::uint32_t> tested(g.pairs.begin(), g.pairs.end());
            for (std::uint32_t c = s.firstContact; c < s.firstContact + s.contactCount; ++c) {
                const FvContact &fc = rec.contacts[c];
                ++checked;
                bool found = false;
                for (const ContactRec &r : A) {
                    if (PairShape(r.pair) != fc.shape) continue;
                    const float dx = r.P[0] - fc.P[0], dy = r.P[1] - fc.P[1], dz = r.P[2] - fc.P[2];
                    const float nd = r.N[0] * fc.N[0] + r.N[1] * fc.N[1] + r.N[2] * fc.N[2];
                    if (dx * dx + dy * dy + dz * dz < 1e-6f && nd > 0.999f && tested.count(r.pair)) { found = true; break; }
                }
                if (!found) {  // no matching twin contact (exact-test float noise): require any triangle the FV point lies on to be tested
                    const float tol = 3e-4f;
                    const Box3 qb = {{fc.P[0] - tol, fc.P[1] - tol, fc.P[2] - tol}, {fc.P[0] + tol, fc.P[1] + tol, fc.P[2] + tol}};
                    const double pp[3] = {fc.P[0], fc.P[1], fc.P[2]};
                    bool any = false;
                    bvh.Query(qb, [&](std::uint32_t id) {
                        if (PointTriDist2(pp, rec.world.tris[id]) <= double(tol) * tol && tested.count(PairCode(fc.shape, id))) any = true;
                    });
                    ++noiseUnmatched;
                    if (!any) ++misses;
                }
            }
        }
        std::printf("| %s | %.2f | %.0f / %.0f / %.0f / %.0f / %.0f | %.0f | %.0f | %.0f | %.0f | %.2f, %.2f / %.2f | %zu / %zu (%zu noise-unmatched) | L1 diff %zu, ExactTest diff %zu of %zu substeps | queries %.2f/tick (L2 %.3f/tick), cached %.0f tris, BVH/L2 part of query %.0f%% | batch stage-1 %.2f, dense stage-2 %.3f packets per candidate-substep; dense vs packet diff %zu/2000 |\n",
                    rec.name.c_str(), spt, setup, query, filt, pack, merge, pack, total, total * spt, batchCyc * spt,
                    double(c1.pairCount) / N, double(c1.stage1Packets) / N, double(c1.stage2Packets) / N, misses, checked, noiseUnmatched, diffL1, diffRef, N, queriesPerTick, double(c1.l2Queries) / rec.ticks, cachedPerSub, 100.0 * double(c1.tBvh) / double(std::max<std::uint64_t>(1, c1.tQuery)), double(batch1) / (16.0 * N), double(batch2dense) / (16.0 * N), denseDiff);
        std::fflush(stdout);
    }
    return 0;
}
