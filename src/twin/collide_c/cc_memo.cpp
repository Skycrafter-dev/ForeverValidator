// Architect C: candidate memo. See cc_memo.h. The box test, the prefilter (Pass / Pass16) and the
// tolerance are the force agent's (src/twin/world/tw_car_query.cpp), copied operation for operation.
#include "cc_memo.h"
#include "cc_lanes.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <x86intrin.h>

#include <xmmintrin.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#define CC_AVX512 1
#else
#define CC_AVX512 0
#endif

namespace tw::collide_c {

namespace {

template <class T> void Grow(std::vector<T> &v, std::size_t n) {
    if (v.size() < n) v.resize(std::max<std::size_t>(n, 2 * v.size()));
}

// tw_car_query.cpp BuildMemo's per-entry planes
void MakePlanes(const world::WorldPackView &view, const exact::FastTri *fast, Memo &m, std::uint32_t i) {
    const cq::Ref &rf = m.ref[i];
    const world::Instance &in = view.instances[rf.instance];
    const float *R = in.iso;
    float tmo[3];
    for (int a = 0; a < 3; ++a) tmo[a] = static_cast<float>(double(in.iso[9 + a]) - double(m.origin[a]));
    const float s = (in.flags & world::kInstMirrored) ? -1.0f : 1.0f;
    const exact::FastTri &f = fast[rf.modelTri];
    exact::FastTri &w = m.planes[i];
    for (int j = 0; j < 4; ++j) {
        const float sj = j == 0 ? s : 1.0f;
        const float nx = f.x[j], ny = f.y[j], nz = f.z[j];
        const float wx = sj * ((R[0] * nx + R[3] * ny) + R[6] * nz);
        const float wy = sj * ((R[1] * nx + R[4] * ny) + R[7] * nz);
        const float wz = sj * ((R[2] * nx + R[5] * ny) + R[8] * nz);
        w.x[j] = wx;
        w.y[j] = wy;
        w.z[j] = wz;
        w.d[j] = sj * f.d[j] + ((wx * tmo[0] + wy * tmo[1]) + wz * tmo[2]);
    }
    if (m.verts) {   // vertices rotated to world, relative to the instance translation: R v (cache-independent)
        const exact::TriVerts &tv = m.verts[rf.modelTri];
        Memo::WVerts &o = m.wv[i];
        for (int c = 0; c < 3; ++c) {
            const exact::V &p = tv.v[c];
            o.v[3 * c] = (R[0] * p.x + R[3] * p.y) + R[6] * p.z;
            o.v[3 * c + 1] = (R[1] * p.x + R[4] * p.y) + R[7] * p.z;
            o.v[3 * c + 2] = (R[2] * p.x + R[5] * p.y) + R[8] * p.z;
        }
        for (int a = 0; a < 3; ++a) o.t[a] = in.iso[9 + a];   // the instance translation t_m
    }
    m.ready[i] = 1u;
    ++m.planesMade;
}

inline bool Pass(const Memo &m, std::size_t i, const cq::ShapeFilter &f) {
    const exact::FastTri &w = m.planes[i];
    for (int j = 0; j < 4; ++j) {
        const float wx = w.x[j], wy = w.y[j], wz = w.z[j];
        const float dist = ((wx * f.c[0] + wy * f.c[1]) + wz * f.c[2]) - w.d[j];
        if ((j == 0 && dist < f.low) || dist > f.reach) return false;
        if (dist > f.lazy) {
            const float *q = f.q;
            const float a = (q[0] * wx + q[3] * wy) + q[4] * wz;
            const float b = (q[3] * wx + q[1] * wy) + q[5] * wz;
            const float c = (q[4] * wx + q[5] * wy) + q[2] * wz;
            const float h2 = (a * wx + b * wy) + c * wz;
            const float x = dist - f.tol;
            if (x > 0.0f && x * x > h2 * exact::kSupportSlack) return false;
        }
    }
    return true;
}
inline bool InBox(const Memo &m, std::size_t i, const float lo[3], const float hi[3]) {
    return m.lo[0][i] <= hi[0] && lo[0] <= m.hi[0][i] && m.lo[1][i] <= hi[1] && lo[1] <= m.hi[1][i] && m.lo[2][i] <= hi[2] && lo[2] <= m.hi[2][i];
}

#if CC_AVX512
using F = __m512;
#define CC_IN inline __attribute__((always_inline))
CC_IN F Dot3(F ax, F ay, F az, F bx, F by, F bz) { return _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(ax, bx), _mm512_mul_ps(ay, by)), _mm512_mul_ps(az, bz)); }
CC_IN void Transpose16(F r[16]) {
    F t[16];
#pragma GCC unroll 16
    for (int i = 0; i < 16; i += 2) {
        t[i] = _mm512_unpacklo_ps(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_ps(r[i], r[i + 1]);
    }
#pragma GCC unroll 16
    for (int i = 0; i < 16; i += 4) {
        r[i] = _mm512_shuffle_ps(t[i], t[i + 2], _MM_SHUFFLE(1, 0, 1, 0));
        r[i + 1] = _mm512_shuffle_ps(t[i], t[i + 2], _MM_SHUFFLE(3, 2, 3, 2));
        r[i + 2] = _mm512_shuffle_ps(t[i + 1], t[i + 3], _MM_SHUFFLE(1, 0, 1, 0));
        r[i + 3] = _mm512_shuffle_ps(t[i + 1], t[i + 3], _MM_SHUFFLE(3, 2, 3, 2));
    }
#pragma GCC unroll 16
    for (int k = 0; k < 4; ++k) {
        const F a = r[k], b = r[4 + k], c = r[8 + k], d = r[12 + k];
        const F ab0 = _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(2, 0, 2, 0));
        const F ab1 = _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(3, 1, 3, 1));
        const F cd0 = _mm512_shuffle_f32x4(c, d, _MM_SHUFFLE(2, 0, 2, 0));
        const F cd1 = _mm512_shuffle_f32x4(c, d, _MM_SHUFFLE(3, 1, 3, 1));
        t[k] = _mm512_shuffle_f32x4(ab0, cd0, _MM_SHUFFLE(2, 0, 2, 0));
        t[4 + k] = _mm512_shuffle_f32x4(ab1, cd1, _MM_SHUFFLE(2, 0, 2, 0));
        t[8 + k] = _mm512_shuffle_f32x4(ab0, cd0, _MM_SHUFFLE(3, 1, 3, 1));
        t[12 + k] = _mm512_shuffle_f32x4(ab1, cd1, _MM_SHUFFLE(3, 1, 3, 1));
    }
#pragma GCC unroll 16
    for (int i = 0; i < 16; ++i) r[i] = t[i];
}
struct Lanes { F c0, c1, c2, q[6], reach, low, tol, lazy; };
CC_IN __mmask16 Pass16(const F pl[16], const Lanes &f, __mmask16 m) {
    F dist[4];
    __mmask16 lazy[4], anyLazy = 0;
#pragma GCC unroll 4
    for (int j = 0; j < 4; ++j) {
        dist[j] = _mm512_sub_ps(Dot3(pl[j], pl[4 + j], pl[8 + j], f.c0, f.c1, f.c2), pl[12 + j]);
        if (j == 0) m &= ~_mm512_cmp_ps_mask(dist[0], f.low, _CMP_LT_OQ);
        m &= ~_mm512_cmp_ps_mask(dist[j], f.reach, _CMP_GT_OQ);
        lazy[j] = _mm512_cmp_ps_mask(dist[j], f.lazy, _CMP_GT_OQ);
        anyLazy |= lazy[j];
    }
    if (!(anyLazy & m)) return m;
    const F slack = _mm512_set1_ps(exact::kSupportSlack), zero = _mm512_setzero_ps();
#pragma GCC unroll 4
    for (int j = 0; j < 4; ++j) {
        const F w0 = pl[j], w1 = pl[4 + j], w2 = pl[8 + j];
        const F a = Dot3(f.q[0], f.q[3], f.q[4], w0, w1, w2);
        const F b = Dot3(f.q[3], f.q[1], f.q[5], w0, w1, w2);
        const F c = Dot3(f.q[4], f.q[5], f.q[2], w0, w1, w2);
        const F h2 = Dot3(a, b, c, w0, w1, w2);
        const F x = _mm512_sub_ps(dist[j], f.tol);
        m &= ~(lazy[j] & _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ) & _mm512_cmp_ps_mask(_mm512_mul_ps(x, x), _mm512_mul_ps(h2, slack), _CMP_GT_OQ));
    }
    return m;
}
// m & (a P b) with b broadcast from memory: one vcmpps with a {1to16} operand (GCC splits it into vbroadcastss +
// vcmpps). The bounds are passed as the table's own lvalues so that each is a base + constant address, not a pointer
template <int P> CC_IN __mmask16 CmpBcst(F a, const float &b, __mmask16 m) {
    __mmask16 r;
    __asm__("vcmpps %4, %2%{1to16%}, %1, %0%{%3%}" : "=Yk"(r) : "v"(a), "m"(b), "Yk"(m), "i"(P));
    return r;
}
// the box test lo <= hx, lx <= hi per axis, lo <= hx written hx >= lo (GE_OQ: the same ordered quiet predicate)
CC_IN __mmask16 BoxMask(F lx, F ly, F lz, F hx, F hy, F hz, const float &lo0, const float &lo1, const float &lo2, const float &hi0,
                        const float &hi1, const float &hi2, __mmask16 m) {
    m = CmpBcst<_CMP_LE_OQ>(lx, hi0, m);
    m = CmpBcst<_CMP_GE_OQ>(hx, lo0, m);
    m = CmpBcst<_CMP_LE_OQ>(ly, hi1, m);
    m = CmpBcst<_CMP_GE_OQ>(hy, lo1, m);
    m = CmpBcst<_CMP_LE_OQ>(lz, hi2, m);
    m = CmpBcst<_CMP_GE_OQ>(hz, lo2, m);
    return m;
}
// BoxMask with one mask write per box instead of six: a <= b exactly when VMAXPS(a, b) (a > b ? a : b) returns b,
// and a >= b when VMINPS(a, b) (a < b ? a : b) returns b; equal values of either zero sign return b, so for non-NaN
// operands the xor with b is zero exactly when the ordered compare holds. The six xors OR-ed (one vpternlogd each),
// one vptestnmd. Callers pass NaN-free operands only (NaN would return b: a false hit).
CC_IN F MaxBcst(F a, const float &b) {
    F r;
    __asm__("vmaxps %2%{1to16%}, %1, %0" : "=v"(r) : "v"(a), "m"(b));
    return r;
}
CC_IN F MinBcst(F a, const float &b) {
    F r;
    __asm__("vminps %2%{1to16%}, %1, %0" : "=v"(r) : "v"(a), "m"(b));
    return r;
}
CC_IN __m512i XorBcst(F a, const float &b) {
    __m512i r;
    __asm__("vpxord %2%{1to16%}, %1, %0" : "=v"(r) : "v"(a), "m"(b));
    return r;
}
CC_IN __m512i OrXorBcst(__m512i acc, F t, const float &b) {   // acc | (t ^ b)
    __asm__("vpternlogd $0xF6, %2%{1to16%}, %1, %0" : "+v"(acc) : "v"(t), "m"(b));
    return acc;
}
CC_IN __mmask16 BoxMaskMM(F lx, F ly, F lz, F hx, F hy, F hz, const float &lo0, const float &lo1, const float &lo2, const float &hi0,
                          const float &hi1, const float &hi2, __mmask16 m) {
    __m512i acc = XorBcst(MaxBcst(lx, hi0), hi0);
    acc = OrXorBcst(acc, MinBcst(hx, lo0), lo0);
    acc = OrXorBcst(acc, MaxBcst(ly, hi1), hi1);
    acc = OrXorBcst(acc, MinBcst(hy, lo1), lo1);
    acc = OrXorBcst(acc, MaxBcst(lz, hi2), hi2);
    acc = OrXorBcst(acc, MinBcst(hz, lo2), lo2);
    return _mm512_mask_testn_epi32_mask(m, acc, acc);
}
// AVX-512 VBMI2 byte compress / expand, zero-masked, as inline assembly: the AVX-512 variant is built for x86-64-v4,
// which lacks VBMI2, so the callers check HasVbmi2() first. Integer moves only: the bits cannot differ.
CC_IN __m512i CompressB(__mmask64 k, __m512i v) {
    __m512i r;
    __asm__("vpcompressb %1, %0%{%2%}%{z%}" : "=v"(r) : "v"(v), "Yk"(k));
    return r;
}
CC_IN __m512i ExpandB(__mmask64 k, __m512i v) {
    __m512i r;
    __asm__("vpexpandb %1, %0%{%2%}%{z%}" : "=v"(r) : "v"(v), "Yk"(k));
    return r;
}
// CPUID (once): VBMI2 usable, unless FOREVERVALIDATOR_TWIN_VBMI2=0 (the v4-only path, the same bits: A/B and tests)
bool HasVbmi2() {
    static const bool v = [] {
        const char *e = std::getenv("FOREVERVALIDATOR_TWIN_VBMI2");
        if (e != nullptr && e[0] == '0') return false;
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512vbmi2") != 0 && __builtin_cpu_supports("avx512bw") != 0;
    }();
    return v;
}
#endif

}  // namespace

void FvBvh::Build(const world::WorldPackView &view) {
    n_ = static_cast<std::uint32_t>(view.tris.size());
    levels_.clear();
    rootCount_ = 0;
    top_ = 0;
    const std::size_t cap = std::size_t(n_) + 16u;
    ref.assign(cap, cq::Ref{0u, 0u, 0u, 0u});
    std::vector<float> lo[3], hi[3];   // the current level's node boxes (level 0: the triangles, by rank)
    for (int a = 0; a < 3; ++a) { lo[a].assign(n_, 1e30f); hi[a].assign(n_, -1e30f); }
    for (std::uint32_t t = 0; t < n_; ++t) {
        const world::TriInfo &ti = view.triInfo[t];
        const std::uint32_t r = ti.fvOrder;
        const world::Instance &in = view.instances[ti.instance];
        ref[r] = cq::Ref{ti.instance, view.models[in.model].firstTri + ti.modelTri, in.model, t};
        for (int a = 0; a < 3; ++a) { lo[a][r] = view.tris[t].lo[a]; hi[a][r] = view.tris[t].hi[a]; }
    }
    auto fill = [&](Block &B, std::uint32_t first, std::uint32_t count) {   // the boxes of nodes [first, first + count)
        for (int a = 0; a < 3; ++a)
            for (std::uint32_t c = 0; c < 16; ++c) {
                B.lo[a][c] = c < count ? lo[a][first + c] : 1e30f;
                B.hi[a][c] = c < count ? hi[a][first + c] : -1e30f;
            }
    };
    if (n_ == 0u) return;
    Level L0;
    L0.n = n_;
    levels_.push_back(std::move(L0));
    // level 1: runs of up to 16 consecutive ranks, cut where the instance changes (an instance's triangles are
    // compact; consecutive instances in FV's record order need not be); above: runs of 16 nodes
    std::vector<std::uint32_t> cuts{0u};
    for (std::uint32_t r = 1; r <= n_; ++r)
        if (r == n_ || r - cuts.back() == 16u || ref[r].instance != ref[r - 1].instance) cuts.push_back(r);
    std::uint32_t cn = n_;
    while (cn > 16u) {   // the top level has at most 16 nodes
        if (levels_.size() > 1u) {
            cuts.assign(1, 0u);
            for (std::uint32_t j = 16u; j < cn; j += 16u) cuts.push_back(j);
            cuts.push_back(cn);
        }
        Level P;
        P.n = static_cast<std::uint32_t>(cuts.size() - 1u);
        P.first.assign(P.n, 0u);
        P.count.assign(P.n, 0u);
        P.block.resize(P.n);
        std::vector<float> plo[3], phi[3];
        for (int a = 0; a < 3; ++a) { plo[a].assign(P.n, 1e30f); phi[a].assign(P.n, -1e30f); }
        for (std::uint32_t j = 0; j < P.n; ++j) {
            P.first[j] = cuts[j];
            P.count[j] = static_cast<std::uint8_t>(cuts[j + 1] - cuts[j]);
            fill(P.block[j], cuts[j], cuts[j + 1] - cuts[j]);
            for (std::uint32_t i = cuts[j]; i < cuts[j + 1]; ++i)
                for (int a = 0; a < 3; ++a) {
                    plo[a][j] = std::min(plo[a][j], lo[a][i]);
                    phi[a][j] = std::max(phi[a][j], hi[a][i]);
                }
        }
        levels_.push_back(std::move(P));
        for (int a = 0; a < 3; ++a) { lo[a].swap(plo[a]); hi[a].swap(phi[a]); }
        cn = levels_.back().n;
    }
    top_ = static_cast<int>(levels_.size()) - 1;
    rootCount_ = cn;
    fill(root_, 0u, cn);
}

unsigned FvBvh::Test16(const Block &B, const Box3 &q) {
#if CC_AVX512
    __mmask16 m = _mm512_cmp_ps_mask(_mm512_load_ps(B.lo[0]), _mm512_set1_ps(q.hi[0]), _CMP_LE_OQ);
    m &= _mm512_cmp_ps_mask(_mm512_set1_ps(q.lo[0]), _mm512_load_ps(B.hi[0]), _CMP_LE_OQ);
    m &= _mm512_cmp_ps_mask(_mm512_load_ps(B.lo[1]), _mm512_set1_ps(q.hi[1]), _CMP_LE_OQ);
    m &= _mm512_cmp_ps_mask(_mm512_set1_ps(q.lo[1]), _mm512_load_ps(B.hi[1]), _CMP_LE_OQ);
    m &= _mm512_cmp_ps_mask(_mm512_load_ps(B.lo[2]), _mm512_set1_ps(q.hi[2]), _CMP_LE_OQ);
    m &= _mm512_cmp_ps_mask(_mm512_set1_ps(q.lo[2]), _mm512_load_ps(B.hi[2]), _CMP_LE_OQ);
    return m;
#else
    unsigned m = 0;
    for (std::uint32_t l = 0; l < 16; ++l) {
        const bool ov = B.lo[0][l] <= q.hi[0] && q.lo[0] <= B.hi[0][l] && B.lo[1][l] <= q.hi[1] && q.lo[1] <= B.hi[1][l] && B.lo[2][l] <= q.hi[2] &&
                        q.lo[2] <= B.hi[2][l];
        m |= static_cast<unsigned>(ov) << l;
    }
    return m;
#endif
}

#if CC_AVX512
namespace {
// MakePlanes for entries [0, n) in 16-lane passes: rows in by transposes, each lane MakePlanes' operations
// (the offset to the memo origin in double), rows out by a transpose.
void MakePlanes16(const world::WorldPackView &view, const exact::FastTri *fast, Memo &m, std::uint32_t n) {
    for (std::uint32_t i0 = 0; i0 < n; i0 += 16) {
        const std::uint32_t cnt = n - i0 >= 16 ? 16u : n - i0;
        F f[16], r[16];
        alignas(64) float sgn[16];
        for (std::uint32_t l = 0; l < 16; ++l) {
            const cq::Ref &rf = m.ref[i0 + (l < cnt ? l : 0u)];
            const world::Instance &in = view.instances[rf.instance];
            f[l] = _mm512_load_ps(&fast[rf.modelTri]);
            r[l] = _mm512_loadu_ps(in.iso);
            sgn[l] = (in.flags & world::kInstMirrored) ? -1.0f : 1.0f;
        }
        noise::Transpose16(f);
        noise::Transpose16(r);
        F tmo[3];
        for (int a = 0; a < 3; ++a) {
            const __m512d o = _mm512_set1_pd(double(m.origin[a]));
            const __m256 lo = _mm512_castps512_ps256(r[9 + a]);
            const __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(r[9 + a]), 1));
            const __m256 dl = _mm512_cvtpd_ps(_mm512_sub_pd(_mm512_cvtps_pd(lo), o));
            const __m256 dh = _mm512_cvtpd_ps(_mm512_sub_pd(_mm512_cvtps_pd(hi), o));
            tmo[a] = _mm512_insertf32x8(_mm512_castps256_ps512(dl), dh, 1);
        }
        const F s = _mm512_load_ps(sgn), one = _mm512_set1_ps(1.0f);
        F w[16];
        for (int j = 0; j < 4; ++j) {
            const F sj = j == 0 ? s : one;
            const F nx = f[j], ny = f[4 + j], nz = f[8 + j];
            const F wx = _mm512_mul_ps(sj, _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(r[0], nx), _mm512_mul_ps(r[3], ny)), _mm512_mul_ps(r[6], nz)));
            const F wy = _mm512_mul_ps(sj, _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(r[1], nx), _mm512_mul_ps(r[4], ny)), _mm512_mul_ps(r[7], nz)));
            const F wz = _mm512_mul_ps(sj, _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(r[2], nx), _mm512_mul_ps(r[5], ny)), _mm512_mul_ps(r[8], nz)));
            w[j] = wx;
            w[4 + j] = wy;
            w[8 + j] = wz;
            w[12 + j] = _mm512_add_ps(_mm512_mul_ps(sj, f[12 + j]),
                                      _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(wx, tmo[0]), _mm512_mul_ps(wy, tmo[1])), _mm512_mul_ps(wz, tmo[2])));
        }
        noise::Transpose16(w);
        for (std::uint32_t l = 0; l < cnt; ++l) {
            _mm512_store_ps(&m.planes[i0 + l], w[l]);
            m.ready[i0 + l] = 1u;
        }
        m.planesMade += cnt;
    }
}
}  // namespace
#endif

void Refill(const world::WorldPackView &view, const Bvh4 &bvh, const float *modelAbs, const Box3 &box, const float origin[3], Memo *memo,
            const FvBvh *fvb, const exact::FastTri *fast) {
    Memo &m = *memo;
    m.box = box;
    m.allReady = false;
    for (int a = 0; a < 3; ++a) m.origin[a] = origin[a];
    std::uint32_t n = 0;
    if (fvb != nullptr) {
        // ranks in FV order straight from the query; per entry only rank-ordered (contiguous) tables
        // ranks and AABBs straight from the query's blocks
        if (m.fv.size() < 64) m.fv.resize(64);
        for (int a = 0; a < 3; ++a) if (m.lo[a].size() < m.fv.size()) { m.lo[a].resize(m.fv.size()); m.hi[a].resize(m.fv.size()); }
        fvb->Query(box, [&](std::uint32_t first, const FvBvh::Block &B, unsigned mk) {
            if (n + 16u > m.fv.size()) {
                m.fv.resize(2 * m.fv.size());
                for (int a = 0; a < 3; ++a) { m.lo[a].resize(m.fv.size()); m.hi[a].resize(m.fv.size()); }
            }
#if CC_AVX512
            const __mmask16 k = static_cast<__mmask16>(mk);
            _mm512_storeu_si512(m.fv.data() + n, _mm512_maskz_compress_epi32(k, _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(first)),
                                                                                              _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))));
            for (int a = 0; a < 3; ++a) {
                _mm512_storeu_ps(m.lo[a].data() + n, _mm512_maskz_compress_ps(k, _mm512_load_ps(B.lo[a])));
                _mm512_storeu_ps(m.hi[a].data() + n, _mm512_maskz_compress_ps(k, _mm512_load_ps(B.hi[a])));
            }
            n += static_cast<std::uint32_t>(__builtin_popcount(mk));
#else
            for (; mk; mk &= mk - 1u) {
                const int lane = __builtin_ctz(mk);
                m.fv[n] = first + static_cast<std::uint32_t>(lane);
                for (int a = 0; a < 3; ++a) { m.lo[a][n] = B.lo[a][lane]; m.hi[a][n] = B.hi[a][lane]; }
                ++n;
            }
#endif
        });
        const std::size_t cap = (n + 15u) & ~std::size_t(15);
        m.n = n;
        if (m.ref.size() < cap) {
            const std::size_t c = std::max<std::size_t>(cap, m.fv.size());
            m.ref.resize(c);
            m.fv.resize(c);
            m.planes.resize(c);
            m.wv.resize(c);
            m.ready.resize(c);
            for (int a = 0; a < 3; ++a) { m.lo[a].resize(c); m.hi[a].resize(c); }
        }
        m.maxD1 = m.maxT1 = m.maxModelAbs = 0.0f;
        std::uint32_t lastInst = 0xFFFFFFFFu;
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint32_t r = m.fv[i];
            const cq::Ref rf = fvb->ref[r];
            m.ref[i] = rf;
            m.ready[i] = 0u;
            if (rf.instance != lastInst) {
                lastInst = rf.instance;
                const world::Instance &in = view.instances[rf.instance];
                const float tx = std::fabs(static_cast<float>(double(in.iso[9]) - double(origin[0])));
                const float ty = std::fabs(static_cast<float>(double(in.iso[10]) - double(origin[1])));
                const float tz = std::fabs(static_cast<float>(double(in.iso[11]) - double(origin[2])));
                m.maxD1 = std::max(m.maxD1, tx + ty + tz);
                m.maxT1 = std::max(m.maxT1, std::fabs(in.iso[9]) + std::fabs(in.iso[10]) + std::fabs(in.iso[11]));
                m.maxModelAbs = std::max(m.maxModelAbs, modelAbs[in.model]);
            }
        }
        for (std::size_t i = n; i < cap; ++i) {
            m.ref[i] = cq::Ref{0u, 0u, 0u, 0u};
            m.fv[i] = 0u;
            m.ready[i] = 1u;
            m.planes[i] = exact::FastTri{};
            for (int a = 0; a < 3; ++a) { m.lo[a][i] = 1e30f; m.hi[a][i] = -1e30f; }
        }
#if CC_AVX512
        if (fast != nullptr) {   // every entry's planes now, 16 per pass (MakePlanes' operations per lane)
            MakePlanes16(view, fast, m, n);
            m.allReady = true;
        }
#else
        (void)fast;
#endif
        return;
    } else {
        m.keys.clear();
        bvh.Query(box, [&](std::uint32_t t) { m.keys.push_back((std::uint64_t(view.triInfo[t].fvOrder) << 32) | t); });
        std::sort(m.keys.begin(), m.keys.end());   // FV detection order: every per-shape list comes out in it
        n = static_cast<std::uint32_t>(m.keys.size());
        if (m.tri.size() < n) m.tri.resize(std::max<std::size_t>(n, 2 * m.tri.size()));
        for (std::uint32_t i = 0; i < n; ++i) m.tri[i] = static_cast<std::uint32_t>(m.keys[i] & 0xFFFFFFFFu);
    }
    const std::size_t cap = (n + 15u) & ~std::size_t(15);
    m.n = n;
    Grow(m.tri, cap);
    if (m.ref.size() < cap) {
        const std::size_t c = std::max<std::size_t>(cap, m.tri.size());
        m.ref.resize(c);
        m.fv.resize(c);
        m.planes.resize(c);
        m.wv.resize(c);
        m.ready.resize(c);
        for (int a = 0; a < 3; ++a) { m.lo[a].resize(c); m.hi[a].resize(c); }
    }
    m.maxD1 = m.maxT1 = m.maxModelAbs = 0.0f;
    std::uint32_t lastInst = 0xFFFFFFFFu;
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::uint32_t t = m.tri[i];
        const Tri &tr = view.tris[t];
        const world::TriInfo &ti = view.triInfo[t];
        const world::Instance &in = view.instances[ti.instance];
        m.ref[i] = cq::Ref{ti.instance, view.models[in.model].firstTri + ti.modelTri, in.model, t};
        m.fv[i] = ti.fvOrder;
        m.ready[i] = 0u;
        for (int a = 0; a < 3; ++a) { m.lo[a][i] = tr.lo[a]; m.hi[a][i] = tr.hi[a]; }
        if (ti.instance != lastInst) {   // BVH leaves keep an instance's triangles mostly together
            lastInst = ti.instance;
            // same terms and sum order as tw_car_query.cpp: |x| + |y| + |z| of t_m - origin
            const float tx = std::fabs(static_cast<float>(double(in.iso[9]) - double(origin[0])));
            const float ty = std::fabs(static_cast<float>(double(in.iso[10]) - double(origin[1])));
            const float tz = std::fabs(static_cast<float>(double(in.iso[11]) - double(origin[2])));
            m.maxD1 = std::max(m.maxD1, tx + ty + tz);
            m.maxT1 = std::max(m.maxT1, std::fabs(in.iso[9]) + std::fabs(in.iso[10]) + std::fabs(in.iso[11]));
            m.maxModelAbs = std::max(m.maxModelAbs, modelAbs[in.model]);
        }
    }
    for (std::size_t i = n; i < cap; ++i) {
        m.tri[i] = 0u;
        m.ref[i] = cq::Ref{0u, 0u, 0u, 0u};
        m.fv[i] = 0u;
        m.ready[i] = 1u;
        m.planes[i] = exact::FastTri{};
        for (int a = 0; a < 3; ++a) { m.lo[a][i] = 1e30f; m.hi[a][i] = -1e30f; }
    }
}

void ShapeBoxes(const collide::ShapePose shapes[8], ShapeTable *st, Box3 *u) {
    __m128 ulo[3], uhi[3];
    for (int g = 0; g < 2; ++g) {
        const collide::ShapePose *p[4] = {&shapes[4 * g], &shapes[4 * g + 1], &shapes[4 * g + 2], &shapes[4 * g + 3]};
        __m128 x[12];
        for (int f = 0; f < 12; f += 4) {
            __m128 r0 = _mm_loadu_ps(p[0]->world.r + f), r1 = _mm_loadu_ps(p[1]->world.r + f), r2 = _mm_loadu_ps(p[2]->world.r + f),
                   r3 = _mm_loadu_ps(p[3]->world.r + f);
            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
            x[f] = r0; x[f + 1] = r1; x[f + 2] = r2; x[f + 3] = r3;
        }
        const __m128 rad[3] = {_mm_setr_ps(p[0]->radii[0], p[1]->radii[0], p[2]->radii[0], p[3]->radii[0]),
                               _mm_setr_ps(p[0]->radii[1], p[1]->radii[1], p[2]->radii[1], p[3]->radii[1]),
                               _mm_setr_ps(p[0]->radii[2], p[1]->radii[2], p[2]->radii[2], p[3]->radii[2])};
        const __m128 sign = _mm_set1_ps(-0.0f);
        for (int a = 0; a < 3; ++a) {
            const __m128 c = x[9 + a];
            // the ellipsoid's exact AABB: h = sqrt(Q_aa) = sqrt((r0^2 r[a]^2 + r1^2 r[3+a]^2) + r2^2 r[6+a]^2) (ShapeFilters' Q); eps = 1e-3 + 1e-6 |c|
            const __m128 h = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(rad[0], rad[0]), _mm_mul_ps(x[a], x[a])),
                                                               _mm_mul_ps(_mm_mul_ps(rad[1], rad[1]), _mm_mul_ps(x[3 + a], x[3 + a]))),
                                                    _mm_mul_ps(_mm_mul_ps(rad[2], rad[2]), _mm_mul_ps(x[6 + a], x[6 + a]))));
            const __m128 eps = _mm_add_ps(_mm_set1_ps(1e-3f), _mm_mul_ps(_mm_set1_ps(1e-6f), _mm_andnot_ps(sign, c)));
            const __m128 lo = _mm_sub_ps(_mm_sub_ps(c, h), eps), hi = _mm_add_ps(_mm_add_ps(c, h), eps);
            _mm_store_ps(&st->lo[a][4 * g], lo);
            _mm_store_ps(&st->hi[a][4 * g], hi);
            ulo[a] = g == 0 ? lo : _mm_min_ps(ulo[a], lo);
            uhi[a] = g == 0 ? hi : _mm_max_ps(uhi[a], hi);
        }
    }
    for (int a = 0; a < 3; ++a) {
        alignas(16) float l[4], h[4];
        _mm_store_ps(l, ulo[a]);
        _mm_store_ps(h, uhi[a]);
        u->lo[a] = std::min(std::min(l[0], l[1]), std::min(l[2], l[3]));
        u->hi[a] = std::max(std::max(h[0], h[1]), std::max(h[2], h[3]));
    }
}

void ShapeFilters(const Memo &memo, const collide::ShapePose shapes[8], ShapeTable *st) {
    const __m128 sign = _mm_set1_ps(-0.0f);
    auto abs = [&](__m128 v) { return _mm_andnot_ps(sign, v); };
    auto add = [](__m128 a, __m128 b) { return _mm_add_ps(a, b); };
    auto mul = [](__m128 a, __m128 b) { return _mm_mul_ps(a, b); };
    for (int g = 0; g < 2; ++g) {
        const collide::ShapePose *p[4] = {&shapes[4 * g], &shapes[4 * g + 1], &shapes[4 * g + 2], &shapes[4 * g + 3]};
        __m128 R[12];
        for (int f = 0; f < 12; f += 4) {
            __m128 r0 = _mm_loadu_ps(p[0]->world.r + f), r1 = _mm_loadu_ps(p[1]->world.r + f), r2 = _mm_loadu_ps(p[2]->world.r + f),
                   r3 = _mm_loadu_ps(p[3]->world.r + f);
            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
            R[f] = r0; R[f + 1] = r1; R[f + 2] = r2; R[f + 3] = r3;
        }
        const __m128 rad[3] = {_mm_setr_ps(p[0]->radii[0], p[1]->radii[0], p[2]->radii[0], p[3]->radii[0]),
                               _mm_setr_ps(p[0]->radii[1], p[1]->radii[1], p[2]->radii[1], p[3]->radii[1]),
                               _mm_setr_ps(p[0]->radii[2], p[1]->radii[2], p[2]->radii[2], p[3]->radii[2])};
        const __m128 r2[3] = {mul(rad[0], rad[0]), mul(rad[1], rad[1]), mul(rad[2], rad[2])};   // ShapeConst::r2
        const __m128 rmax = _mm_max_ps(rad[0], _mm_max_ps(rad[1], rad[2])), rmin = _mm_min_ps(rad[0], _mm_min_ps(rad[1], rad[2]));
        const __m128 ratio = _mm_div_ps(rmax, rmin);
        const __m128 *T = R + 9;
        __m128 c[3];
        for (int a = 0; a < 3; ++a) {
            c[a] = _mm_sub_ps(T[a], _mm_set1_ps(memo.origin[a]));
            _mm_storeu_ps(&st->t[kTc0 + a][4 * g], c[a]);
        }
        // q(a, b) = (r2[0] (R[a] R[b]) + r2[1] (R[3+a] R[3+b])) + r2[2] (R[6+a] R[6+b])
        auto q = [&](int x, int y) { return add(add(mul(r2[0], mul(R[x], R[y])), mul(r2[1], mul(R[3 + x], R[3 + y]))), mul(r2[2], mul(R[6 + x], R[6 + y]))); };
        _mm_storeu_ps(&st->t[kTq0 + 0][4 * g], q(0, 0));
        _mm_storeu_ps(&st->t[kTq0 + 1][4 * g], q(1, 1));
        _mm_storeu_ps(&st->t[kTq0 + 2][4 * g], q(2, 2));
        _mm_storeu_ps(&st->t[kTq0 + 3][4 * g], q(0, 1));
        _mm_storeu_ps(&st->t[kTq0 + 4][4 * g], q(0, 2));
        _mm_storeu_ps(&st->t[kTq0 + 5][4 * g], q(1, 2));
        const __m128 c1 = add(add(abs(c[0]), abs(c[1])), abs(c[2]));
        const __m128 t1 = add(add(abs(T[0]), abs(T[1])), abs(T[2]));
        // tol = 1e-4 rmax + 4e-6 ratio (2 (c1 + maxD1) + 3 maxModelAbs) + 1e-6 (t1 + maxT1)   (left to right)
        const __m128 inner = add(mul(_mm_set1_ps(2.0f), add(c1, _mm_set1_ps(memo.maxD1))), mul(_mm_set1_ps(3.0f), _mm_set1_ps(memo.maxModelAbs)));
        const __m128 tol = add(add(mul(_mm_set1_ps(1.0e-4f), rmax), mul(mul(_mm_set1_ps(4.0e-6f), ratio), inner)),
                               mul(_mm_set1_ps(1.0e-6f), add(t1, _mm_set1_ps(memo.maxT1))));
        _mm_storeu_ps(&st->t[kTreach][4 * g], add(rmax, tol));
        _mm_storeu_ps(&st->t[kTlow][4 * g], _mm_xor_ps(tol, sign));
        _mm_storeu_ps(&st->t[kTtol][4 * g], tol);
        _mm_storeu_ps(&st->t[kTlazy][4 * g], _mm_sub_ps(rmin, tol));
    }
}

void MakeBodyGate(const Memo &memo, const collide::ShapePose shapes[8], const ShapeTable &st, unsigned bodyShapes, BodyGate *g) {
    g->on = false;
    g->shapes = bodyShapes;
    if (!bodyShapes) return;
    const int k0 = __builtin_ctz(bodyShapes);
    const float *U = shapes[k0].world.r;   // columns: the first body shape's axes
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    float maxTol = 0.0f, maxReach = 0.0f;
    for (int k = 0; k < 8; ++k) {
        if (!(bodyShapes >> k & 1u)) continue;
        const collide::ShapePose &sp = shapes[k];
        const float c[3] = {sp.world.t[0] - memo.origin[0], sp.world.t[1] - memo.origin[1], sp.world.t[2] - memo.origin[2]};
        for (int a = 0; a < 3; ++a) {
            const float *ua = U + 3 * a;
            const float s = ua[0] * c[0] + ua[1] * c[1] + ua[2] * c[2];
            float h2 = 0.0f;   // support of ellipsoid k along u_a: sqrt(sum_j (u_a . R_j)^2 r_j^2)
            for (int j = 0; j < 3; ++j) {
                const float *rj = sp.world.r + 3 * j;
                const float d = (ua[0] * rj[0] + ua[1] * rj[1] + ua[2] * rj[2]) * sp.radii[j];
                h2 += d * d;
            }
            const float h = std::sqrt(h2) * 1.00001f + 1e-5f;
            lo[a] = std::min(lo[a], s - h);
            hi[a] = std::max(hi[a], s + h);
        }
        maxTol = std::max(maxTol, st.t[kTtol][k]);
        maxReach = std::max(maxReach, st.t[kTreach][k]);
    }
    for (int i = 0; i < 3; ++i) g->c[i] = 0.0f;
    for (int a = 0; a < 3; ++a) {
        const float mid = 0.5f * (lo[a] + hi[a]);
        g->e[a] = 0.5f * (hi[a] - lo[a]) * 1.00001f + 2e-5f;
        for (int i = 0; i < 3; ++i) { g->u[a][i] = U[3 * a + i]; g->c[i] += mid * U[3 * a + i]; }
    }
    // the box in these coordinates holds the ellipsoids up to U's non-orthonormality (~1e-7 relative):
    // absorbed with the evaluation rounding in G, above the largest body tolerance + support slack
    g->G = maxTol + 1e-4f * maxReach + 2e-4f;
    g->on = true;
}

void Filter(const world::WorldPackView &view, const exact::FastTri *fast, Memo &m, const ShapeTable &st, const Box3 &u, cq::Lists *out,
            const BodyGate &gate, Survivors *sink, NearSet *split, int phase) {
    if (sink != nullptr && !(CC_AVX512 && m.allReady && !gate.on)) {
        if (phase == 1) return;                                          // (no split on these paths: all in phase 2)
        // the list paths: their lists handed over compact,   // the list paths: their lists handed over compact,
        // shape-major (FV order within each shape)
        Filter(view, fast, m, st, u, out, gate, nullptr);
        std::size_t tot = 0;
        for (int k = 0; k < 8; ++k) tot += out->n[k];
        sink->n = 0;
        Grow(sink->e, tot + 16);
        Grow(sink->k, tot + 16);
        for (int k = 0; k < 8; ++k)
            for (std::uint32_t j = 0; j < out->n[k]; ++j) {
                sink->e[sink->n] = out->idx[k][j];
                sink->k[sink->n++] = static_cast<std::uint32_t>(k);
            }
        return;
    }
    // per-call scratch shared by every memo of the thread (stays in L1 when lanes alternate memos)
    struct FilterScratch {
        std::vector<std::uint32_t> pairEntry, pairShape;
    };
    thread_local FilterScratch fs;
    thread_local NearSet nsLocal;
    const bool splitOk = split != nullptr && sink != nullptr && m.allReady && !gate.on;
    NearSet &ns = splitOk ? *split : nsLocal;
    if (!splitOk) phase = 0;
    const unsigned bodyShapes = gate.shapes;
    const bool bodyGate = gate.on;
    (void)bodyShapes;   // the body gate exists on the AVX-512 path only
    (void)bodyGate;
    const std::size_t n = m.n;
#if defined(CC_FINE)
    std::uint64_t ft = __rdtsc();
    auto lapf = [&](int i) { const std::uint64_t t1 = __rdtsc(); m.fine[i] += t1 - ft; ft = t1; };
#else
    auto lapf = [](int) {};
#endif
#if CC_AVX512
    std::uint32_t nn = 0;
    if (phase != 2) {
        // pass 0: entries meeting the union box (every shape box is inside it), compressed with their boxes
        Grow(ns.near, n + 32);
        for (int a = 0; a < 3; ++a) { Grow(ns.lo[a], n + 32); Grow(ns.hi[a], n + 32); }
        const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        for (std::size_t b = 0; b < n; b += 16) {
            const __mmask16 valid = static_cast<__mmask16>(n - b >= 16 ? 0xFFFFu : (1u << (n - b)) - 1u);
            const F lx = _mm512_loadu_ps(&m.lo[0][b]), ly = _mm512_loadu_ps(&m.lo[1][b]), lz = _mm512_loadu_ps(&m.lo[2][b]);
            const F hx = _mm512_loadu_ps(&m.hi[0][b]), hy = _mm512_loadu_ps(&m.hi[1][b]), hz = _mm512_loadu_ps(&m.hi[2][b]);
            const __mmask16 mk = BoxMask(lx, ly, lz, hx, hy, hz, u.lo[0], u.lo[1], u.lo[2], u.hi[0], u.hi[1], u.hi[2], valid);
            if (!mk) continue;
            const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(b)), lane);
            _mm512_storeu_si512(ns.near.data() + nn, _mm512_maskz_compress_epi32(mk, idx));
            _mm512_storeu_ps(ns.lo[0].data() + nn, _mm512_maskz_compress_ps(mk, lx));
            _mm512_storeu_ps(ns.lo[1].data() + nn, _mm512_maskz_compress_ps(mk, ly));
            _mm512_storeu_ps(ns.lo[2].data() + nn, _mm512_maskz_compress_ps(mk, lz));
            _mm512_storeu_ps(ns.hi[0].data() + nn, _mm512_maskz_compress_ps(mk, hx));
            _mm512_storeu_ps(ns.hi[1].data() + nn, _mm512_maskz_compress_ps(mk, hy));
            _mm512_storeu_ps(ns.hi[2].data() + nn, _mm512_maskz_compress_ps(mk, hz));
            nn += static_cast<std::uint32_t>(__builtin_popcount(mk));
        }
        ns.nn = nn;
        m.nearEntries += nn;
        lapf(0);
        if (phase == 1) return;
    }
    nn = ns.nn;
    if (sink != nullptr) sink->n = 0;
    if (m.allReady && !bodyGate) {
        // pass 1+2 per near block: its 8 shape box masks; its planes transposed once (SoA, lane = near entry);
        // the body shapes' triangle-plane test on those lanes first (it ends most body pairs); then the
        // remaining (entry, shape) pairs in 16-lane packets, planes and shape fields by permutes; survivors
        // per shape in FV order.
        if (sink != nullptr) {
            Grow(sink->e, 8 * std::size_t(nn) + 32);
            Grow(sink->k, 8 * std::size_t(nn) + 32);
        } else {
            for (int k = 0; k < 8; ++k) {
                Grow(out->idx[k], nn + 16);
                out->n[k] = 0;
            }
        }
        F T[13];
        for (int i = 0; i < 13; ++i) T[i] = _mm512_zextps256_ps512(_mm256_load_ps(st.t[i]));   // lanes 8..15 unused (shape index < 8)
        const __m512i laneIdx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const F slack = _mm512_set1_ps(exact::kSupportSlack), zero = _mm512_setzero_ps();
        std::uint32_t np = 0;
        for (std::uint32_t b = 0; b < nn; b += 16) {
            const __mmask16 valid = static_cast<__mmask16>(nn - b >= 16 ? 0xFFFFu : (1u << (nn - b)) - 1u);
            const F lx = _mm512_loadu_ps(ns.lo[0].data() + b), ly = _mm512_loadu_ps(ns.lo[1].data() + b), lz = _mm512_loadu_ps(ns.lo[2].data() + b);
            const F hx = _mm512_loadu_ps(ns.hi[0].data() + b), hy = _mm512_loadu_ps(ns.hi[1].data() + b), hz = _mm512_loadu_ps(ns.hi[2].data() + b);
            __mmask16 mks[8], any = 0;
            for (int k = 0; k < 8; ++k) {
                mks[k] = BoxMask(lx, ly, lz, hx, hy, hz, st.lo[0][k], st.lo[1][k], st.lo[2][k], st.hi[0][k], st.hi[1][k], st.hi[2][k], valid);
                any |= mks[k];
            }
            if (!any) continue;
            const std::uint32_t first = static_cast<std::uint32_t>(__builtin_ctz(any));
            F pl[16];
#pragma GCC unroll 16
            for (std::uint32_t l = 0; l < 16; ++l) pl[l] = _mm512_load_ps(&m.planes[ns.near[b + ((any >> l) & 1u ? l : first)]]);
            Transpose16(pl);
            // the body shapes: Pass16's triangle-plane test alone (a subset of the same tests)
            for (int k = 0; k < 8; ++k) {
                if (!(bodyShapes >> k & 1u) || !mks[k]) continue;
                const F c0 = _mm512_set1_ps(st.t[kTc0][k]), c1 = _mm512_set1_ps(st.t[kTc0 + 1][k]), c2 = _mm512_set1_ps(st.t[kTc0 + 2][k]);
                const F dist = _mm512_sub_ps(Dot3(pl[0], pl[4], pl[8], c0, c1, c2), pl[12]);
                __mmask16 mk = mks[k];
                mk &= ~_mm512_cmp_ps_mask(dist, _mm512_set1_ps(st.t[kTlow][k]), _CMP_LT_OQ);
                mk &= ~_mm512_cmp_ps_mask(dist, _mm512_set1_ps(st.t[kTreach][k]), _CMP_GT_OQ);
                const __mmask16 lazy = mk & _mm512_cmp_ps_mask(dist, _mm512_set1_ps(st.t[kTlazy][k]), _CMP_GT_OQ);
                if (lazy) {
                    const F q0 = _mm512_set1_ps(st.t[kTq0][k]), q1 = _mm512_set1_ps(st.t[kTq0 + 1][k]), q2 = _mm512_set1_ps(st.t[kTq0 + 2][k]);
                    const F q3 = _mm512_set1_ps(st.t[kTq0 + 3][k]), q4 = _mm512_set1_ps(st.t[kTq0 + 4][k]), q5 = _mm512_set1_ps(st.t[kTq0 + 5][k]);
                    const F a = Dot3(q0, q3, q4, pl[0], pl[4], pl[8]);
                    const F bb = Dot3(q3, q1, q5, pl[0], pl[4], pl[8]);
                    const F c = Dot3(q4, q5, q2, pl[0], pl[4], pl[8]);
                    const F h2 = Dot3(a, bb, c, pl[0], pl[4], pl[8]);
                    const F x = _mm512_sub_ps(dist, _mm512_set1_ps(st.t[kTtol][k]));
                    mk &= ~(lazy & _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ) & _mm512_cmp_ps_mask(_mm512_mul_ps(x, x), _mm512_mul_ps(h2, slack), _CMP_GT_OQ));
                }
                mks[k] = mk;
            }
            // the block's pairs: lane (near entry) and shape
            alignas(64) std::int32_t pLane[8 * 16 + 16], pShape[8 * 16 + 16];
            std::uint32_t bp = 0;
            for (int k = 0; k < 8; ++k) {
                const __mmask16 mk = mks[k];
                if (!mk) continue;
                _mm512_storeu_si512(pLane + bp, _mm512_maskz_compress_epi32(mk, laneIdx));
                _mm512_storeu_si512(pShape + bp, _mm512_set1_epi32(k));
                bp += static_cast<std::uint32_t>(__builtin_popcount(mk));
            }
            np += bp;
            const __m512i nidx = _mm512_loadu_si512(ns.near.data() + b);
            for (std::uint32_t pb = 0; pb < bp; pb += 16) {
                const std::uint32_t cntp = bp - pb >= 16 ? 16u : bp - pb;
                const __mmask16 pv = static_cast<__mmask16>(cntp == 16 ? 0xFFFFu : (1u << cntp) - 1u);
                const __m512i J = _mm512_maskz_loadu_epi32(pv, pLane + pb), sh = _mm512_maskz_loadu_epi32(pv, pShape + pb);
                F pp[16];
#pragma GCC unroll 16
                for (int c = 0; c < 16; ++c) pp[c] = _mm512_permutexvar_ps(J, pl[c]);
                Lanes f;
                f.c0 = _mm512_permutexvar_ps(sh, T[0]);
                f.c1 = _mm512_permutexvar_ps(sh, T[1]);
                f.c2 = _mm512_permutexvar_ps(sh, T[2]);
                for (int i = 0; i < 6; ++i) f.q[i] = _mm512_permutexvar_ps(sh, T[3 + i]);
                f.reach = _mm512_permutexvar_ps(sh, T[9]);
                f.low = _mm512_permutexvar_ps(sh, T[10]);
                f.tol = _mm512_permutexvar_ps(sh, T[11]);
                f.lazy = _mm512_permutexvar_ps(sh, T[12]);
                unsigned ok = Pass16(pp, f, pv);
                if (sink != nullptr) {   // compact, branch free
                    _mm512_mask_compressstoreu_epi32(sink->e.data() + sink->n, static_cast<__mmask16>(ok), _mm512_permutexvar_epi32(J, nidx));
                    _mm512_mask_compressstoreu_epi32(sink->k.data() + sink->n, static_cast<__mmask16>(ok), sh);
                    sink->n += static_cast<std::uint32_t>(__builtin_popcount(ok));
                    continue;
                }
                if (!ok) continue;
                alignas(64) std::uint32_t ent[16], ks[16];
                _mm512_store_si512(ent, _mm512_permutexvar_epi32(J, nidx));
                _mm512_store_si512(ks, sh);
                while (ok) {
                    const unsigned l = static_cast<unsigned>(__builtin_ctz(ok));
                    ok &= ok - 1u;
                    const std::uint32_t k = ks[l];
                    out->idx[k][out->n[k]++] = ent[l];
                }
            }
        }
        m.pairs += np;
        lapf(3);
        if (sink == nullptr)
            for (int k = 0; k < 8; ++k) m.survOf[k] += out->n[k];
        lapf(4);
        return;
    }
    // pass 1: the near entries against the 8 shape boxes -> (entry, shape) pairs
    Grow(fs.pairEntry, 8 * std::size_t(nn) + 32);
    Grow(fs.pairShape, 8 * std::size_t(nn) + 32);
    std::uint32_t np = 0;
    for (std::uint32_t b = 0; b < nn; b += 16) {
        const __mmask16 valid = static_cast<__mmask16>(nn - b >= 16 ? 0xFFFFu : (1u << (nn - b)) - 1u);
        const F lx = _mm512_loadu_ps(ns.lo[0].data() + b), ly = _mm512_loadu_ps(ns.lo[1].data() + b), lz = _mm512_loadu_ps(ns.lo[2].data() + b);
        const F hx = _mm512_loadu_ps(ns.hi[0].data() + b), hy = _mm512_loadu_ps(ns.hi[1].data() + b), hz = _mm512_loadu_ps(ns.hi[2].data() + b);
        const __m512i idx = _mm512_loadu_si512(ns.near.data() + b);
        __mmask16 mks[8], bodyAny = 0;
        for (int k = 0; k < 8; ++k) {
            mks[k] = BoxMask(lx, ly, lz, hx, hy, hz, st.lo[0][k], st.lo[1][k], st.lo[2][k], st.hi[0][k], st.hi[1][k], st.hi[2][k], valid);
            if (bodyShapes >> k & 1u) bodyAny |= mks[k];
        }
        if (!m.allReady) {   // planes of entries reaching a shape box for the first time (one test per near entry, not per pair)
            __mmask16 any = 0;
            for (int k = 0; k < 8; ++k) any |= mks[k];
            unsigned todo = any;
            while (todo) {
                const std::uint32_t e = ns.near[b + static_cast<std::uint32_t>(__builtin_ctz(todo))];
                todo &= todo - 1u;
                if (!m.ready[e]) MakePlanes(view, fast, m, e);
            }
        }
        __mmask16 gated = 0;
        if (bodyGate && bodyAny) {
            const std::uint32_t cnt = nn - b >= 16 ? 16u : nn - b;
            for (std::uint32_t l = 0; l < cnt; ++l)
                if ((bodyAny >> l & 1u) && !m.ready[ns.near[b + l]]) MakePlanes(view, fast, m, ns.near[b + l]);
            F pl[16];
#pragma GCC unroll 16
            for (std::uint32_t l = 0; l < 16; ++l) pl[l] = _mm512_load_ps(&m.planes[ns.near[b + (l < cnt ? l : 0)]]);
            Transpose16(pl);
            const F cx = _mm512_set1_ps(gate.c[0]), cy = _mm512_set1_ps(gate.c[1]), cz = _mm512_set1_ps(gate.c[2]);
            const F G = _mm512_set1_ps(gate.G), nG = _mm512_set1_ps(-gate.G);
            const __m512i absMask = _mm512_set1_epi32(0x7fffffff);
            auto absf = [&](F x) { return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x), absMask)); };
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const F D = _mm512_sub_ps(Dot3(pl[j], pl[4 + j], pl[8 + j], cx, cy, cz), pl[12 + j]);
                F H = _mm512_setzero_ps();
                for (int a = 0; a < 3; ++a) {
                    const F wu = Dot3(pl[j], pl[4 + j], pl[8 + j], _mm512_set1_ps(gate.u[a][0]), _mm512_set1_ps(gate.u[a][1]), _mm512_set1_ps(gate.u[a][2]));
                    H = _mm512_add_ps(H, _mm512_mul_ps(absf(wu), _mm512_set1_ps(gate.e[a])));
                }
                gated |= _mm512_cmp_ps_mask(_mm512_sub_ps(D, H), G, _CMP_GT_OQ);                  // the whole box beyond the plane
                if (j == 0) gated |= _mm512_cmp_ps_mask(_mm512_add_ps(D, H), nG, _CMP_LT_OQ);    // the whole box behind it
            }
            gated &= bodyAny;
            m.gated += static_cast<std::uint32_t>(__builtin_popcount(gated));
        }
        for (int k = 0; k < 8; ++k) {
            const __mmask16 mk = (bodyShapes >> k & 1u) ? static_cast<__mmask16>(mks[k] & ~gated) : mks[k];
            if (!mk) continue;
            _mm512_storeu_si512(fs.pairEntry.data() + np, _mm512_maskz_compress_epi32(mk, idx));
            _mm512_storeu_si512(fs.pairShape.data() + np, _mm512_set1_epi32(k));
            np += static_cast<std::uint32_t>(__builtin_popcount(mk));
            m.pairsOf[k] += static_cast<std::uint32_t>(__builtin_popcount(mk));
        }
    }
    m.pairs += np;
    lapf(1);
    for (int k = 0; k < 8; ++k) {
        Grow(out->idx[k], np + 16);
        out->n[k] = 0;
    }
    lapf(2);
    F T[13];
    for (int i = 0; i < 13; ++i) T[i] = _mm512_zextps256_ps512(_mm256_load_ps(st.t[i]));   // lanes 8..15 unused (shape index < 8)
    for (std::uint32_t b = 0; b < np; b += 16) {
        const std::uint32_t cnt = np - b >= 16 ? 16u : np - b;
        const __mmask16 valid = static_cast<__mmask16>(cnt == 16 ? 0xFFFFu : (1u << cnt) - 1u);
        F pl[16];
#pragma GCC unroll 16
        for (std::uint32_t l = 0; l < 16; ++l) pl[l] = _mm512_load_ps(&m.planes[fs.pairEntry[b + (l < cnt ? l : 0)]]);
        Transpose16(pl);
        const __m512i sh = _mm512_maskz_loadu_epi32(valid, fs.pairShape.data() + b);
        Lanes f;
        f.c0 = _mm512_permutexvar_ps(sh, T[0]);
        f.c1 = _mm512_permutexvar_ps(sh, T[1]);
        f.c2 = _mm512_permutexvar_ps(sh, T[2]);
        for (int i = 0; i < 6; ++i) f.q[i] = _mm512_permutexvar_ps(sh, T[3 + i]);
        f.reach = _mm512_permutexvar_ps(sh, T[9]);
        f.low = _mm512_permutexvar_ps(sh, T[10]);
        f.tol = _mm512_permutexvar_ps(sh, T[11]);
        f.lazy = _mm512_permutexvar_ps(sh, T[12]);
        unsigned ok = Pass16(pl, f, valid);
        while (ok) {
            const unsigned l = static_cast<unsigned>(__builtin_ctz(ok));
            ok &= ok - 1u;
            const std::uint32_t k = fs.pairShape[b + l];
            out->idx[k][out->n[k]++] = fs.pairEntry[b + l];
        }
    }
#else
    // SSE2 (any x86-64): lanes = 4 near entries; their planes transposed once, then the 8 shapes
    for (int k = 0; k < 8; ++k) {
        Grow(out->idx[k], n + 16);
        out->n[k] = 0;
    }
    Grow(ns.near, n + 8);
    std::uint32_t nn = 0;
    for (std::size_t i = 0; i < n; ++i)   // pass 0: entries meeting the union box
        if (InBox(m, i, u.lo, u.hi)) ns.near[nn++] = static_cast<std::uint32_t>(i);
    m.nearEntries += nn;
    lapf(0);
    const __m128 slack = _mm_set1_ps(exact::kSupportSlack), zero = _mm_setzero_ps();
    auto dot3 = [](__m128 ax, __m128 ay, __m128 az, __m128 bx, __m128 by, __m128 bz) {
        return _mm_add_ps(_mm_add_ps(_mm_mul_ps(ax, bx), _mm_mul_ps(ay, by)), _mm_mul_ps(az, bz));
    };
    for (std::uint32_t b0 = 0; b0 < nn; b0 += 4) {
        const std::uint32_t cnt = nn - b0 >= 4 ? 4u : nn - b0;
        std::uint32_t e[4];
        for (std::uint32_t l = 0; l < 4; ++l) e[l] = ns.near[b0 + (l < cnt ? l : 0)];
        const int valid = (1 << cnt) - 1;
        __m128 lo[3], hi[3];
        for (int a = 0; a < 3; ++a) {
            lo[a] = _mm_setr_ps(m.lo[a][e[0]], m.lo[a][e[1]], m.lo[a][e[2]], m.lo[a][e[3]]);
            hi[a] = _mm_setr_ps(m.hi[a][e[0]], m.hi[a][e[1]], m.hi[a][e[2]], m.hi[a][e[3]]);
        }
        int mk[8], any = 0;
        for (int k = 0; k < 8; ++k) {
            __m128 c = _mm_and_ps(_mm_cmple_ps(lo[0], _mm_set1_ps(st.hi[0][k])), _mm_cmple_ps(_mm_set1_ps(st.lo[0][k]), hi[0]));
            c = _mm_and_ps(c, _mm_and_ps(_mm_cmple_ps(lo[1], _mm_set1_ps(st.hi[1][k])), _mm_cmple_ps(_mm_set1_ps(st.lo[1][k]), hi[1])));
            c = _mm_and_ps(c, _mm_and_ps(_mm_cmple_ps(lo[2], _mm_set1_ps(st.hi[2][k])), _mm_cmple_ps(_mm_set1_ps(st.lo[2][k]), hi[2])));
            mk[k] = _mm_movemask_ps(c) & valid;
            any |= mk[k];
        }
        if (!any) continue;
        for (std::uint32_t l = 0; l < cnt; ++l)
            if ((any >> l & 1) && !m.ready[e[l]]) MakePlanes(view, fast, m, e[l]);
        // pl[4 c + j]: component c (x y z d) of plane j, lanes = the 4 entries
        __m128 pl[16];
        for (int c = 0; c < 4; ++c) {
            const float *base[4];
            for (int l = 0; l < 4; ++l) base[l] = (&m.planes[e[l]].x[0]) + 4 * c;
            __m128 r0 = _mm_loadu_ps(base[0]), r1 = _mm_loadu_ps(base[1]), r2 = _mm_loadu_ps(base[2]), r3 = _mm_loadu_ps(base[3]);
            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
            pl[4 * c] = r0; pl[4 * c + 1] = r1; pl[4 * c + 2] = r2; pl[4 * c + 3] = r3;
        }
        for (int k = 0; k < 8; ++k) {
            if (!mk[k]) continue;
            m.pairs += static_cast<std::uint32_t>(__builtin_popcount(static_cast<unsigned>(mk[k])));
            m.pairsOf[k] += static_cast<std::uint32_t>(__builtin_popcount(static_cast<unsigned>(mk[k])));
            const __m128 c0 = _mm_set1_ps(st.t[kTc0][k]), c1 = _mm_set1_ps(st.t[kTc0 + 1][k]), c2 = _mm_set1_ps(st.t[kTc0 + 2][k]);
            const __m128 reach = _mm_set1_ps(st.t[kTreach][k]), low = _mm_set1_ps(st.t[kTlow][k]), tol = _mm_set1_ps(st.t[kTtol][k]),
                         lazy = _mm_set1_ps(st.t[kTlazy][k]);
            const __m128 q0 = _mm_set1_ps(st.t[kTq0][k]), q1 = _mm_set1_ps(st.t[kTq0 + 1][k]), q2 = _mm_set1_ps(st.t[kTq0 + 2][k]),
                         q3 = _mm_set1_ps(st.t[kTq0 + 3][k]), q4 = _mm_set1_ps(st.t[kTq0 + 4][k]), q5 = _mm_set1_ps(st.t[kTq0 + 5][k]);
            int ok = mk[k];
            for (int j = 0; j < 4 && ok; ++j) {   // Pass() per lane: the same values, rejections as masks
                const __m128 wx = pl[j], wy = pl[4 + j], wz = pl[8 + j];
                const __m128 dist = _mm_sub_ps(dot3(wx, wy, wz, c0, c1, c2), pl[12 + j]);
                __m128 rej = _mm_cmpgt_ps(dist, reach);
                if (j == 0) rej = _mm_or_ps(rej, _mm_cmplt_ps(dist, low));
                const __m128 isLazy = _mm_cmpgt_ps(dist, lazy);
                if (_mm_movemask_ps(isLazy) & ok) {
                    const __m128 a = dot3(q0, q3, q4, wx, wy, wz), bb = dot3(q3, q1, q5, wx, wy, wz), cc = dot3(q4, q5, q2, wx, wy, wz);
                    const __m128 h2 = dot3(a, bb, cc, wx, wy, wz);
                    const __m128 x = _mm_sub_ps(dist, tol);
                    rej = _mm_or_ps(rej, _mm_and_ps(isLazy, _mm_and_ps(_mm_cmpgt_ps(x, zero), _mm_cmpgt_ps(_mm_mul_ps(x, x), _mm_mul_ps(h2, slack)))));
                }
                ok &= ~_mm_movemask_ps(rej);
            }
            while (ok) {
                const int l = __builtin_ctz(static_cast<unsigned>(ok));
                ok &= ok - 1;
                out->idx[k][out->n[k]++] = e[l];
            }
        }
    }
#endif
    lapf(3);
    for (int k = 0; k < 8; ++k) m.survOf[k] += out->n[k];
    lapf(4);
}

bool FilterPass16(Memo *const *memos, const ShapeTable *st, const Box3 *u, std::uint32_t mask, unsigned bodyShapes, PassSurvivors *out) {
#if CC_AVX512
    // the near entries of all lanes (pass 0), lane after lane: [nb[l], nb[l] + out->nn[l])
    struct PassScratch {
        std::vector<std::uint32_t> near;
        std::vector<float> lo[3], hi[3];
    };
    thread_local PassScratch ps;
    std::size_t cap = 0;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) cap += memos[__builtin_ctz(mm)]->n + 32u;
    Grow(ps.near, cap);
    for (int a = 0; a < 3; ++a) { Grow(ps.lo[a], cap); Grow(ps.hi[a], cap); }
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    std::uint32_t nb[16], tot = 0;
    std::size_t need = out->n;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) need += 8 * std::size_t(memos[__builtin_ctz(mm)]->n) + 32;
    Grow(out->e, need);
    Grow(out->k, need);
    auto pass0 = [&](int l) {
        Memo &m = *memos[l];
        const Box3 &ul = u[l];
        const std::size_t n = m.n;
        std::uint32_t nn = 0;
        std::uint32_t *nearL = ps.near.data() + tot;
        float *lo0 = ps.lo[0].data() + tot, *lo1 = ps.lo[1].data() + tot, *lo2 = ps.lo[2].data() + tot;
        float *hi0 = ps.hi[0].data() + tot, *hi1 = ps.hi[1].data() + tot, *hi2 = ps.hi[2].data() + tot;
        // the memo's six arrays as locals (the row stores may alias the vectors: reloaded per block otherwise)
        const float *const ml0 = m.lo[0].data(), *const ml1 = m.lo[1].data(), *const ml2 = m.lo[2].data();
        const float *const mh0 = m.hi[0].data(), *const mh1 = m.hi[1].data(), *const mh2 = m.hi[2].data();
        for (std::size_t b = 0; b < n; b += 16) {
            const __mmask16 valid = static_cast<__mmask16>(n - b >= 16 ? 0xFFFFu : (1u << (n - b)) - 1u);
            const F lx = _mm512_loadu_ps(ml0 + b), ly = _mm512_loadu_ps(ml1 + b), lz = _mm512_loadu_ps(ml2 + b);
            const F hx = _mm512_loadu_ps(mh0 + b), hy = _mm512_loadu_ps(mh1 + b), hz = _mm512_loadu_ps(mh2 + b);
            const __mmask16 mk = BoxMask(lx, ly, lz, hx, hy, hz, ul.lo[0], ul.lo[1], ul.lo[2], ul.hi[0], ul.hi[1], ul.hi[2], valid);
            if (!mk) continue;
            const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(b)), lane);
            _mm512_storeu_si512(nearL + nn, _mm512_maskz_compress_epi32(mk, idx));
            _mm512_storeu_ps(lo0 + nn, _mm512_maskz_compress_ps(mk, lx));
            _mm512_storeu_ps(lo1 + nn, _mm512_maskz_compress_ps(mk, ly));
            _mm512_storeu_ps(lo2 + nn, _mm512_maskz_compress_ps(mk, lz));
            _mm512_storeu_ps(hi0 + nn, _mm512_maskz_compress_ps(mk, hx));
            _mm512_storeu_ps(hi1 + nn, _mm512_maskz_compress_ps(mk, hy));
            _mm512_storeu_ps(hi2 + nn, _mm512_maskz_compress_ps(mk, hz));
            nn += static_cast<std::uint32_t>(__builtin_popcount(mk));
        }
        // index 0 past the lane's end (its slack): phase A loads a whole block's rows without a per-row select
        _mm512_storeu_si512(nearL + nn, _mm512_setzero_si512());
        m.nearEntries += nn;
        nb[l] = static_cast<std::uint32_t>(tot);
        out->nn[l] = nn;
        tot += nn + 16u;   // whole 16-lane loads past a lane's end stay inside the buffer
    };
    // pass 1 + 2 per lane, as Filter's sink path: per near block its 8 shape box masks, its planes transposed once, the
    // body shapes' triangle-plane test first, then the remaining (entry, shape) pairs in 16-lane packets
    const __m512i laneIdx = lane;
    const bool v2 = HasVbmi2();
    const bool body4 = bodyShapes == 0x0Fu;
    const __m512i iota8 = _mm512_set_epi64(0x3F3E3D3C3B3A3938ll, 0x3736353433323130ll, 0x2F2E2D2C2B2A2928ll, 0x2726252423222120ll,
                                           0x1F1E1D1C1B1A1918ll, 0x1716151413121110ll, 0x0F0E0D0C0B0A0908ll, 0x0706050403020100ll);
    const __m512i iota8hi = _mm512_add_epi8(iota8, _mm512_set1_epi8(64));
    const F slack = _mm512_set1_ps(exact::kSupportSlack), zero = _mm512_setzero_ps();
    // A near block with pairs between its two phases: its plane rows (transposed), its pairs' byte codes 16 k + lane in pair
    // order (shape-major), their count and the block's first near entry. Two lanes' blocks at a time (a ring of two halves).
    struct alignas(64) Blk {
        F pl[16];
        std::uint8_t codes[128];
        std::uint32_t bp, b;
    };
    thread_local std::vector<Blk> blkBuf;
    std::uint32_t maxBlk = 1;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) maxBlk = std::max(maxBlk, (memos[__builtin_ctz(mm)]->n + 15u) / 16u);
    Grow(blkBuf, 2 * std::size_t(maxBlk));
    std::uint32_t bs[16], be[16];   // lane l's blocks: blkBuf[bs[l], be[l])
    // phase A per lane: per near block its 8 shape box masks, its planes transposed once, the body shapes' triangle-plane
    // test, its pairs' codes. Its inputs captured by value (one load from the closure, not closure -> object -> vector ->
    // data before a lane's first row load); nb, bs, be by reference (pass 0 writes nb after this point)
    const std::uint32_t *const nearAll = ps.near.data();
    const float *const pl0 = ps.lo[0].data(), *const pl1 = ps.lo[1].data(), *const pl2 = ps.lo[2].data();
    const float *const ph0 = ps.hi[0].data(), *const ph1 = ps.hi[1].data(), *const ph2 = ps.hi[2].data();
    Blk *const blk0 = blkBuf.data();
    auto phaseA = [=, &nb, &bs, &be](int l, std::uint32_t slot) {
        Memo &m = *memos[l];
        const ShapeTable &t = st[l];
        const std::uint32_t nl0 = nb[l], nn = out->nn[l];
        const exact::FastTri *const planes = m.planes.data();
        const std::uint32_t *nearL = nearAll + nl0;
        const float *lo0 = pl0 + nl0, *lo1 = pl1 + nl0, *lo2 = pl2 + nl0;
        const float *hi0 = ph0 + nl0, *hi1 = ph1 + nl0, *hi2 = ph2 + nl0;
        std::uint32_t np = 0, nbk = slot;
        const F t0 = _mm512_load_ps(&t.lo[0][0]), t1 = _mm512_load_ps(&t.lo[0][0] + 16), t2 = _mm512_load_ps(&t.lo[0][0] + 32);   // lo, hi: 48 floats
        const bool tabOk = (_mm512_cmp_ps_mask(t0, t0, _CMP_UNORD_Q) | _mm512_cmp_ps_mask(t1, t1, _CMP_UNORD_Q) | _mm512_cmp_ps_mask(t2, t2, _CMP_UNORD_Q)) == 0;
        for (std::uint32_t b = 0; b < nn; b += 16) {
            const __mmask16 valid = static_cast<__mmask16>(nn - b >= 16 ? 0xFFFFu : (1u << (nn - b)) - 1u);
            const F lx = _mm512_loadu_ps(lo0 + b), ly = _mm512_loadu_ps(lo1 + b), lz = _mm512_loadu_ps(lo2 + b);
            const F hx = _mm512_loadu_ps(hi0 + b), hy = _mm512_loadu_ps(hi1 + b), hz = _mm512_loadu_ps(hi2 + b);
            // the block's plane rows first, independent of its box masks: their L2 round trips overlap the 48 compares (a
            // dense pass's planes are not in L1); rows of entries meeting no shape box go unused
            // (past the lane's end pass 0 wrote index 0: those rows are row 0, masked out like the old copies of row b)
            F pl[16];
#pragma GCC unroll 16
            for (std::uint32_t q = 0; q < 16; ++q) pl[q] = _mm512_load_ps(&planes[nearL[b + q]]);
            __mmask16 mks[8], any = 0;
            // NaN-free boxes (the guard of BoxMaskMM; inf + -inf also takes the compares)
            const F sum = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(lx, ly), _mm512_add_ps(lz, hx)), _mm512_add_ps(hy, hz));
            if (tabOk && !_mm512_mask_cmp_ps_mask(valid, sum, sum, _CMP_UNORD_Q)) {
                // the wheels (4-7) in this form, the bodies with the compares: the two use different units
#pragma GCC unroll 8
                for (int k = 0; k < 8; ++k) {
                    mks[k] = k >= 4 ? BoxMaskMM(lx, ly, lz, hx, hy, hz, t.lo[0][k], t.lo[1][k], t.lo[2][k], t.hi[0][k], t.hi[1][k], t.hi[2][k], valid)
                                    : BoxMask(lx, ly, lz, hx, hy, hz, t.lo[0][k], t.lo[1][k], t.lo[2][k], t.hi[0][k], t.hi[1][k], t.hi[2][k], valid);
                    any |= mks[k];
                }
            } else {
                for (int k = 0; k < 8; ++k) {
                    mks[k] = BoxMask(lx, ly, lz, hx, hy, hz, t.lo[0][k], t.lo[1][k], t.lo[2][k], t.hi[0][k], t.hi[1][k], t.hi[2][k], valid);
                    any |= mks[k];
                }
            }
            if (!any) continue;
            Transpose16(pl);
            // the body shapes: Pass16's triangle-plane test alone (a subset of the same tests)
            if (body4) {   // body shapes 0-3 (the Stadium car): straight-line code, the masks stay in k-registers
                // one branch on the body box masks for the whole test (with no body box hit it leaves the four masks
                // empty; a lazy lane is inside its shape box): resolved as soon as the boxes are, not after the plane
                // rows, their transpose and the distances
                const bool bodyHit = (mks[0] | mks[1] | mks[2] | mks[3]) != 0;
                if (bodyHit) {
                    F dist[4];
                    __mmask16 lz[4];
#pragma GCC unroll 4
                    for (int k = 0; k < 4; ++k) {
                        const F c0 = _mm512_set1_ps(t.t[kTc0][k]), c1 = _mm512_set1_ps(t.t[kTc0 + 1][k]), c2 = _mm512_set1_ps(t.t[kTc0 + 2][k]);
                        dist[k] = _mm512_sub_ps(Dot3(pl[0], pl[4], pl[8], c0, c1, c2), pl[12]);
                        // mk & ~(dist < low) & ~(dist > reach): the negated ordered compares, unordered true
                        // (the bounds as {1to16} operands: the same predicates on the same operands)
                        mks[k] = CmpBcst<_CMP_NGT_UQ>(dist[k], t.t[kTreach][k], CmpBcst<_CMP_NLT_UQ>(dist[k], t.t[kTlow][k], mks[k]));
                        lz[k] = CmpBcst<_CMP_GT_OQ>(dist[k], t.t[kTlazy][k], mks[k]);
                    }
                    // the four lazy tests without a branch each: a block's lazy shapes come together (none in half the
                    // blocks, three or four in most others; per shape 27-40%, as unpredictable as four branches get); a
                    // shape with no lazy lane goes through with an empty mask (its mask unchanged)
#pragma GCC unroll 4
                    for (int k = 0; k < 4; ++k) {
                        const F q0 = _mm512_set1_ps(t.t[kTq0][k]), q1 = _mm512_set1_ps(t.t[kTq0 + 1][k]), q2 = _mm512_set1_ps(t.t[kTq0 + 2][k]);
                        const F q3 = _mm512_set1_ps(t.t[kTq0 + 3][k]), q4 = _mm512_set1_ps(t.t[kTq0 + 4][k]), q5 = _mm512_set1_ps(t.t[kTq0 + 5][k]);
                        const F a = Dot3(q0, q3, q4, pl[0], pl[4], pl[8]);
                        const F bb = Dot3(q3, q1, q5, pl[0], pl[4], pl[8]);
                        const F c = Dot3(q4, q5, q2, pl[0], pl[4], pl[8]);
                        const F h2 = Dot3(a, bb, c, pl[0], pl[4], pl[8]);
                        const F x = _mm512_sub_ps(dist[k], _mm512_set1_ps(t.t[kTtol][k]));
                        mks[k] &= ~_mm512_mask_cmp_ps_mask(_mm512_mask_cmp_ps_mask(lz[k], x, zero, _CMP_GT_OQ), _mm512_mul_ps(x, x), _mm512_mul_ps(h2, slack),
                                                           _CMP_GT_OQ);
                    }
                }
            } else for (int k = 0; k < 8; ++k) {
                if (!(bodyShapes >> k & 1u) || !mks[k]) continue;
                const F c0 = _mm512_set1_ps(t.t[kTc0][k]), c1 = _mm512_set1_ps(t.t[kTc0 + 1][k]), c2 = _mm512_set1_ps(t.t[kTc0 + 2][k]);
                const F dist = _mm512_sub_ps(Dot3(pl[0], pl[4], pl[8], c0, c1, c2), pl[12]);
                __mmask16 mk = mks[k];
                mk &= ~_mm512_cmp_ps_mask(dist, _mm512_set1_ps(t.t[kTlow][k]), _CMP_LT_OQ);
                mk &= ~_mm512_cmp_ps_mask(dist, _mm512_set1_ps(t.t[kTreach][k]), _CMP_GT_OQ);
                const __mmask16 lazy = mk & _mm512_cmp_ps_mask(dist, _mm512_set1_ps(t.t[kTlazy][k]), _CMP_GT_OQ);
                if (lazy) {
                    const F q0 = _mm512_set1_ps(t.t[kTq0][k]), q1 = _mm512_set1_ps(t.t[kTq0 + 1][k]), q2 = _mm512_set1_ps(t.t[kTq0 + 2][k]);
                    const F q3 = _mm512_set1_ps(t.t[kTq0 + 3][k]), q4 = _mm512_set1_ps(t.t[kTq0 + 4][k]), q5 = _mm512_set1_ps(t.t[kTq0 + 5][k]);
                    const F a = Dot3(q0, q3, q4, pl[0], pl[4], pl[8]);
                    const F bb = Dot3(q3, q1, q5, pl[0], pl[4], pl[8]);
                    const F c = Dot3(q4, q5, q2, pl[0], pl[4], pl[8]);
                    const F h2 = Dot3(a, bb, c, pl[0], pl[4], pl[8]);
                    const F x = _mm512_sub_ps(dist, _mm512_set1_ps(t.t[kTtol][k]));
                    mk &= ~(lazy & _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ) & _mm512_cmp_ps_mask(_mm512_mul_ps(x, x), _mm512_mul_ps(h2, slack), _CMP_GT_OQ));
                }
                mks[k] = mk;
            }
            // the block's pairs: lane (near entry) and shape, in pair order (shape-major), as byte codes 16 k + lane
            Blk &bk = blk0[nbk];
            std::uint32_t bp;
            if (v2) {
                // the 128 mask bits are the pairs in order: two byte compresses (shapes 0-3, 4-7) and one expand
                const __mmask64 ma = _mm512_kunpackd(_mm512_kunpackw(mks[3], mks[2]), _mm512_kunpackw(mks[1], mks[0]));
                const __mmask64 mb = _mm512_kunpackd(_mm512_kunpackw(mks[7], mks[6]), _mm512_kunpackw(mks[5], mks[4]));
                const std::uint32_t na = static_cast<std::uint32_t>(__builtin_popcountll(ma));
                bp = na + static_cast<std::uint32_t>(__builtin_popcountll(mb));
                if (bp == 0u) continue;
                const __m512i ca = CompressB(ma, iota8), cb = CompressB(mb, iota8hi);
                _mm512_store_si512(bk.codes, _mm512_or_si512(ca, ExpandB(na >= 64u ? 0ull : ~0ull << na, cb)));
                // the second half without a branch on bp > 64: with bp <= 64 phase B never reads it (shift count
                // taken mod 64: na = 0 stores cb there, unread)
                _mm512_store_si512(bk.codes + 64, CompressB(~0ull << ((64u - na) & 63u), cb));
            } else {
                bp = 0;
                for (int k = 0; k < 8; ++k) {   // 16 bytes per shape at the running count (bp <= 16 k: inside codes)
                    const __mmask16 mk = mks[k];
                    if (!mk) continue;
                    const __m512i c = _mm512_add_epi32(_mm512_maskz_compress_epi32(mk, laneIdx), _mm512_set1_epi32(16 * k));
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(bk.codes + bp), _mm512_cvtepi32_epi8(c));
                    bp += static_cast<std::uint32_t>(__builtin_popcount(mk));
                }
                if (bp == 0u) continue;
            }
#pragma GCC unroll 16
            for (int c = 0; c < 16; ++c) bk.pl[c] = pl[c];
            bk.bp = bp;
            bk.b = b;
            np += bp;
            ++nbk;
        }
        m.pairs += np;
        bs[l] = slot;
        be[l] = nbk;
    };
    // phase B per lane, a lane after its phase A: the pairs in 16-lane packets (a packet = 16 consecutive pairs of a block)
    auto phaseB = [&](int l) {
        const ShapeTable &t = st[l];
        const std::uint32_t *nearL = ps.near.data() + nb[l];
        out->s0[l] = out->n;
        std::uint32_t on = out->n;   // the output's count and arrays in registers (stores through them may alias out)
        std::uint32_t *const oe = out->e.data(), *const ok_ = out->k.data();
        // the shape fields by permutes straight from the table: a 16-float load at row i is rows i and i + 1 (t[13]: spare),
        // and a shape index < 8 picks row i
        auto T = [&](int i) { return _mm512_loadu_ps(t.t[i]); };
        for (std::uint32_t j = bs[l]; j < be[l]; ++j) {
            const Blk &bk = blkBuf[j];
            const __m512i nidx = _mm512_loadu_si512(nearL + bk.b);
            for (std::uint32_t pb = 0; pb < bk.bp; pb += 16) {
                const std::uint32_t cntp = bk.bp - pb >= 16 ? 16u : bk.bp - pb;
                const __mmask16 pv = static_cast<__mmask16>(cntp == 16 ? 0xFFFFu : (1u << cntp) - 1u);
                const __m512i J = _mm512_cvtepu8_epi32(_mm_load_si128(reinterpret_cast<const __m128i *>(bk.codes + pb)));   // lane: low 4 bits
                const __m512i sh = _mm512_srli_epi32(J, 4);
                F pp[16];
#pragma GCC unroll 16
                for (int c = 0; c < 16; ++c) pp[c] = _mm512_permutexvar_ps(J, bk.pl[c]);
                Lanes f;
                f.c0 = _mm512_permutexvar_ps(sh, T(0));
                f.c1 = _mm512_permutexvar_ps(sh, T(1));
                f.c2 = _mm512_permutexvar_ps(sh, T(2));
                for (int i = 0; i < 6; ++i) f.q[i] = _mm512_permutexvar_ps(sh, T(3 + i));
                f.reach = _mm512_permutexvar_ps(sh, T(9));
                f.low = _mm512_permutexvar_ps(sh, T(10));
                f.tol = _mm512_permutexvar_ps(sh, T(11));
                f.lazy = _mm512_permutexvar_ps(sh, T(12));
                const unsigned ok = Pass16(pp, f, pv);
                // compress in registers, then a whole-row store into the buffer's slack (need: + 32 per lane): the
                // memory form of vpcompressd is microcoded on Zen 4 / 5
                _mm512_storeu_si512(oe + on, _mm512_maskz_compress_epi32(static_cast<__mmask16>(ok), _mm512_permutexvar_epi32(J, nidx)));
                _mm512_storeu_si512(ok_ + on, _mm512_maskz_compress_epi32(static_cast<__mmask16>(ok), sh));
                on += static_cast<std::uint32_t>(__builtin_popcount(ok));
            }
        }
        out->n = on;
        out->s1[l] = on;
    };
    // lanes interleaved (independent work side by side): pass 0 two lanes ahead (its near rows are stored a lane before they
    // are read), phase A one lane ahead of phase B (a lane's masks, planes and codes overlap the previous lane's packets)
    int order[16], nl = 0;
    for (std::uint32_t mm = mask; mm != 0u; mm &= mm - 1u) order[nl++] = __builtin_ctz(mm);
    if (nl > 0) pass0(order[0]);
    if (nl > 1) pass0(order[1]);
    if (nl > 0) phaseA(order[0], 0u);
    for (int i = 0; i < nl; ++i) {
        if (i + 2 < nl) pass0(order[i + 2]);
        if (i + 1 < nl) phaseA(order[i + 1], ((i + 1) & 1) ? maxBlk : 0u);
        phaseB(order[i]);
    }
    return true;
#else
    (void)memos; (void)st; (void)u; (void)mask; (void)bodyShapes; (void)out;
    return false;
#endif
}

}  // namespace tw::collide_c
