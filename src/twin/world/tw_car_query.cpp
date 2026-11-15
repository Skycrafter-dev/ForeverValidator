// Candidate memo and world-space prefilter: see tw_car_query.h.
#include "tw_car_query.h"

#include "tw_lanes.h"
#include "tw_math.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__AVX512F__)
#include <immintrin.h>
#define TW_CQ_AVX512 1
#else
#define TW_CQ_AVX512 0
#endif

namespace tw::cq {

void BuildMemo(const world::WorldPackView &view, const Bvh4 &bvh, const exact::FastTri *fast, const float *modelAbs,
               const Box3 &box, const float origin[3], Memo *memo, const std::uint32_t *patchOf) {
    Memo &m = *memo;
    m.box = box;
    for (int a = 0; a < 3; ++a) m.origin[a] = origin[a];
    m.scratch.clear();
    bvh.Query(box, [&](std::uint32_t t) { m.scratch.push_back((std::uint64_t(view.triInfo[t].fvOrder) << 32) | t); });
    std::sort(m.scratch.begin(), m.scratch.end());
    const std::uint32_t n = static_cast<std::uint32_t>(m.scratch.size());
    const std::size_t cap = (n + 15u) & ~std::size_t(15);
    m.n = n;
    if (m.tri.size() < cap) {   // grow only (no zero fill per refill)
        const std::size_t c = std::max<std::size_t>(cap, 2 * m.tri.size());
        m.tri.resize(c);
        m.ref.resize(c);
        for (int a = 0; a < 3; ++a) { m.lo[a].resize(c); m.hi[a].resize(c); }
        m.planes.resize(c);
        m.verts.resize(c);
        m.outf.resize(c);
        m.patch.resize(c);
        m.pkey.resize(c);
    }
    m.inst.clear();
    m.maxD1 = m.maxT1 = m.maxModelAbs = 0.0f;
    m.hasKeys = patchOf != nullptr;
    std::uint32_t curInst = 0xFFFFFFFFu;
    float R[9] = {}, tmo[3] = {}, s = 1.0f;
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::uint32_t t = static_cast<std::uint32_t>(m.scratch[i] & 0xFFFFFFFFu);
        const Tri &tr = view.tris[t];
        const world::TriInfo &ti = view.triInfo[t];
        const world::Instance &in = view.instances[ti.instance];
        const std::uint32_t mt = view.models[in.model].firstTri + ti.modelTri;
        m.tri[i] = t;
        for (int a = 0; a < 3; ++a) { m.lo[a][i] = tr.lo[a]; m.hi[a][i] = tr.hi[a]; }
        if (ti.instance != curInst) {   // per instance: rotation (columns), translation - origin, mirror sign
            curInst = ti.instance;
            std::memcpy(R, in.iso, sizeof R);
            for (int a = 0; a < 3; ++a) tmo[a] = static_cast<float>(double(in.iso[9 + a]) - double(origin[a]));
            s = (in.flags & world::kInstMirrored) ? -1.0f : 1.0f;
            m.maxD1 = std::max(m.maxD1, std::fabs(tmo[0]) + std::fabs(tmo[1]) + std::fabs(tmo[2]));
            m.maxT1 = std::max(m.maxT1, std::fabs(in.iso[9]) + std::fabs(in.iso[10]) + std::fabs(in.iso[11]));
            m.maxModelAbs = std::max(m.maxModelAbs, modelAbs[in.model]);
            InstIso ii;
            std::memcpy(ii.iso, in.iso, sizeof ii.iso);
            m.inst.push_back(ii);   // entries are in FV order, which keeps an instance's triangles together; a repeat
                                    // of an earlier instance just gets its own copy
        }
        m.ref[i] = Ref{ti.instance, mt, in.model, static_cast<std::uint32_t>(m.inst.size() - 1)};
        m.patch[i] = patchOf != nullptr ? patchOf[mt] : 0xFFFFFFFFu;
        if (patchOf != nullptr)   // (instance, patch); a lone triangle gets a key of its own
            m.pkey[i] = (std::uint64_t(ti.instance) << 32) | (m.patch[i] != 0xFFFFFFFFu ? m.patch[i] : (0x80000000u | mt));
        {   // the narrow phase's and the output's per-candidate data
            const world::ModelTri &mtr = view.modelTris[mt];
            const world::Vec3f *mv = view.modelVerts.ptr + view.models[in.model].firstVert;
            for (int c = 0; c < 3; ++c) m.verts[i].v[c] = exact::V{mv[mtr.v[c]].x, mv[mtr.v[c]].y, mv[mtr.v[c]].z};
            const world::Actor &ac = view.ActorOf(tr);
            OutFields &o = m.outf[i];
            const bool trig = ti.trigger != world::kNoTrigger;
            for (int a = 0; a < 3; ++a) o.peerZ[a] = ac.peerZ[a];
            o.peerId = trig ? static_cast<std::uint32_t>(ti.trigger) : ac.peerId;
            o.material = static_cast<std::uint8_t>(tr.meta & kMetaMaterialMask);
            o.trigger = trig ? 1u : 0u;
            o.pad[0] = o.pad[1] = 0u;
        }
        // plane j: world normal R n_m (the triangle plane times the mirror sign: the world winding normal),
        // offset relative to the origin d_m + n_w . (t_m - o)
        const exact::FastTri &f = fast[mt];
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
    }
    for (std::size_t i = n; i < cap; ++i) {   // padding: empty boxes, zero planes
        m.tri[i] = 0u;
        m.ref[i] = Ref{0u, 0u, 0u, 0u};
        m.verts[i] = exact::TriVerts{};
        m.outf[i] = OutFields{};
        m.patch[i] = 0xFFFFFFFFu;
        m.pkey[i] = kNoKey;
        for (int a = 0; a < 3; ++a) { m.lo[a][i] = 1e30f; m.hi[a][i] = -1e30f; }
        m.planes[i] = exact::FastTri{};
    }
}

ShapeConst MakeShapeConst(const float radii[3]) {
    ShapeConst c;
    for (int a = 0; a < 3; ++a) c.r2[a] = radii[a] * radii[a];
    c.rmax = math::max(radii[0], math::max(radii[1], radii[2]));
    c.rmin = math::min(radii[0], math::min(radii[1], radii[2]));
    c.ratio = c.rmax / c.rmin;
    return c;
}

void MakeShapeFilter(const Memo &memo, const float ellR[9], const float ellT[3], const ShapeConst &sc, const Box3 &box, ShapeFilter *out) {
    ShapeFilter &f = *out;
    for (int a = 0; a < 3; ++a) {
        f.lo[a] = box.lo[a];
        f.hi[a] = box.hi[a];
        f.c[a] = ellT[a] - memo.origin[a];
    }
    const float *r2 = sc.r2;
    auto q = [&](int a, int b) { return (r2[0] * (ellR[a] * ellR[b]) + r2[1] * (ellR[3 + a] * ellR[3 + b])) + r2[2] * (ellR[6 + a] * ellR[6 + b]); };
    f.q[0] = q(0, 0); f.q[1] = q(1, 1); f.q[2] = q(2, 2);
    f.q[3] = q(0, 1); f.q[4] = q(0, 2); f.q[5] = q(1, 2);
    const float rmax = sc.rmax, rmin = sc.rmin;
    const float c1 = std::fabs(f.c[0]) + std::fabs(f.c[1]) + std::fabs(f.c[2]);
    const float t1 = std::fabs(ellT[0]) + std::fabs(ellT[1]) + std::fabs(ellT[2]);
#ifndef TW_CQ_TOL_SCALE
#define TW_CQ_TOL_SCALE 1.0f
#endif
    const float tol = TW_CQ_TOL_SCALE * (1.0e-4f * rmax + 4.0e-6f * sc.ratio * (2.0f * (c1 + memo.maxD1) + 3.0f * memo.maxModelAbs) +
                                         1.0e-6f * (t1 + memo.maxT1));
    f.reach = rmax + tol;
    f.low = -tol;
    f.tol = tol;
    f.lazy = rmin - tol;
}

namespace {

#if !TW_CQ_AVX512
#if TW_HAVE_AVX2
using NarrowLanes = math::f32x8;
inline unsigned NarrowBits(math::m32x8 m) { return static_cast<unsigned>(_mm256_movemask_ps(m.v)); }
#else
using NarrowLanes = math::f32x4;
inline unsigned NarrowBits(math::m32x4 m) { return static_cast<unsigned>(_mm_movemask_ps(m.v)); }
#endif
#endif

// PrefilterQ (tw_exact_fast.h) on memo entry i, world space.
inline bool Pass(const Memo &m, std::size_t i, const ShapeFilter &f) {
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
inline bool InBox(const Memo &m, std::size_t i, const ShapeFilter &f) {
    return m.lo[0][i] <= f.hi[0] && f.lo[0] <= m.hi[0][i] && m.lo[1][i] <= f.hi[1] && f.lo[1] <= m.hi[1][i] && m.lo[2][i] <= f.hi[2] &&
           f.lo[2] <= m.hi[2][i];
}

#if TW_CQ_AVX512
using F = __m512;
#define TW_CQ_INLINE inline __attribute__((always_inline))
TW_CQ_INLINE F Dot3(F ax, F ay, F az, F bx, F by, F bz) {
    return _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(ax, bx), _mm512_mul_ps(ay, by)), _mm512_mul_ps(az, bz));
}
// In-register 16 x 16 transpose (rows in, columns out).
TW_CQ_INLINE void Transpose16(F r[16]) {
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

// Pass() on 16 (candidate, shape) lanes: pl = the candidates' plane components (transposed rows),
// the shape fields per lane (permuted from the 8 shapes).
struct Lanes { F c0, c1, c2, q[6], reach, low, tol, lazy; };
TW_CQ_INLINE __mmask16 Pass16(const F pl[16], const Lanes &f, __mmask16 m) {
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
        m &= ~(lazy[j] & _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ) &
               _mm512_cmp_ps_mask(_mm512_mul_ps(x, x), _mm512_mul_ps(h2, slack), _CMP_GT_OQ));
    }
    return m;
}
#endif

}  // namespace

void Filter(Memo &memo, const ShapeFilter shapes[8], Lists *lists, bool simd, const Box3 *within) {
    const std::size_t n = memo.n;
    for (int k = 0; k < 8; ++k)
        if (lists->idx[k].size() < n + 16) lists->idx[k].resize(std::max<std::size_t>(n + 16, 2 * lists->idx[k].size()));
    const bool track = memo.hasKeys && n > 0;   // the box uniformity per shape (Lists::boxUniform)
    for (int k = 0; k < 8; ++k) { lists->boxUniform[k] = true; lists->boxKey[k] = kNoKey; }
    auto trackHit = [&](int k, std::uint32_t i) {
        const std::uint64_t key = memo.pkey[i];
        if (lists->boxKey[k] == kNoKey) lists->boxKey[k] = key;
        else if (key != lists->boxKey[k]) lists->boxUniform[k] = false;
    };
#if TW_CQ_AVX512
    if (simd) {
        // pass 1: box test, 16 candidates x 8 shapes; (candidate, shape) pairs in candidate-packet order
        if (memo.pairEntry.size() < 8 * n + 16) {
            memo.pairEntry.resize(8 * n + 16);
            memo.pairShape.resize(8 * n + 16);
        }
        std::uint32_t np = 0;
        const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        for (std::size_t b = 0; b < n; b += 16) {
            const __mmask16 valid = static_cast<__mmask16>(n - b >= 16 ? 0xFFFFu : (1u << (n - b)) - 1u);
            const F lx = _mm512_loadu_ps(&memo.lo[0][b]), ly = _mm512_loadu_ps(&memo.lo[1][b]), lz = _mm512_loadu_ps(&memo.lo[2][b]);
            const F hx = _mm512_loadu_ps(&memo.hi[0][b]), hy = _mm512_loadu_ps(&memo.hi[1][b]), hz = _mm512_loadu_ps(&memo.hi[2][b]);
            const __m512i idx = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(b)), lane);
            __mmask16 inBox = valid;
            if (within != nullptr) {   // the shapes' union first: a candidate outside it is in no shape box
                inBox &= _mm512_cmp_ps_mask(lx, _mm512_set1_ps(within->hi[0]), _CMP_LE_OQ);
                inBox &= _mm512_cmp_ps_mask(_mm512_set1_ps(within->lo[0]), hx, _CMP_LE_OQ);
                inBox &= _mm512_cmp_ps_mask(ly, _mm512_set1_ps(within->hi[1]), _CMP_LE_OQ);
                inBox &= _mm512_cmp_ps_mask(_mm512_set1_ps(within->lo[1]), hy, _CMP_LE_OQ);
                inBox &= _mm512_cmp_ps_mask(lz, _mm512_set1_ps(within->hi[2]), _CMP_LE_OQ);
                inBox &= _mm512_cmp_ps_mask(_mm512_set1_ps(within->lo[2]), hz, _CMP_LE_OQ);
                if (!inBox) continue;
            }
            for (int k = 0; k < 8; ++k) {
                const ShapeFilter &f = shapes[k];
                __mmask16 mk = inBox & _mm512_cmp_ps_mask(lx, _mm512_set1_ps(f.hi[0]), _CMP_LE_OQ);
                mk &= _mm512_cmp_ps_mask(_mm512_set1_ps(f.lo[0]), hx, _CMP_LE_OQ);
                mk &= _mm512_cmp_ps_mask(ly, _mm512_set1_ps(f.hi[1]), _CMP_LE_OQ);
                mk &= _mm512_cmp_ps_mask(_mm512_set1_ps(f.lo[1]), hy, _CMP_LE_OQ);
                mk &= _mm512_cmp_ps_mask(lz, _mm512_set1_ps(f.hi[2]), _CMP_LE_OQ);
                mk &= _mm512_cmp_ps_mask(_mm512_set1_ps(f.lo[2]), hz, _CMP_LE_OQ);
                if (!mk) continue;
                if (track && lists->boxUniform[k]) {
                    if (lists->boxKey[k] == kNoKey) lists->boxKey[k] = memo.pkey[b + static_cast<unsigned>(__builtin_ctz(mk))];
                    const __m512i want = _mm512_set1_epi64(static_cast<long long>(lists->boxKey[k]));
                    const __m512i k0 = _mm512_loadu_si512(memo.pkey.data() + b), k1 = _mm512_loadu_si512(memo.pkey.data() + b + 8);
                    if (_mm512_mask_cmpneq_epi64_mask(static_cast<__mmask8>(mk & 0xFFu), k0, want) ||
                        _mm512_mask_cmpneq_epi64_mask(static_cast<__mmask8>(mk >> 8), k1, want))
                        lists->boxUniform[k] = false;
                }
                // register compress + full store (the memory form of vpcompressd is microcoded on Zen 4/5);
                // the pair buffers have 16 entries of slack
                _mm512_storeu_si512(memo.pairEntry.data() + np, _mm512_maskz_compress_epi32(mk, idx));
                _mm512_storeu_si512(memo.pairShape.data() + np, _mm512_set1_epi32(k));
                np += static_cast<std::uint32_t>(__builtin_popcount(mk));
            }
        }
        // the 8 shapes' fields as 8-entry tables, permuted per lane by shape
        alignas(64) float tab[14][16];
        for (int k = 0; k < 8; ++k) {
            const ShapeFilter &f = shapes[k];
            tab[0][k] = f.c[0]; tab[1][k] = f.c[1]; tab[2][k] = f.c[2];
            for (int i = 0; i < 6; ++i) tab[3 + i][k] = f.q[i];
            tab[9][k] = f.reach; tab[10][k] = f.low; tab[11][k] = f.tol; tab[12][k] = f.lazy;
            tab[13][k] = 0.0f;
        }
        F T[13];
        for (int i = 0; i < 13; ++i) T[i] = _mm512_load_ps(tab[i]);
        // pass 2: the prefilter on dense packets of pairs; survivors appended per shape, in order
        for (int k = 0; k < 8; ++k) lists->n[k] = 0;
        for (std::uint32_t b = 0; b < np; b += 16) {
            const std::uint32_t cnt = np - b >= 16 ? 16u : np - b;
            const __mmask16 valid = static_cast<__mmask16>(cnt == 16 ? 0xFFFFu : (1u << cnt) - 1u);
            F pl[16];
#pragma GCC unroll 16
            for (std::uint32_t l = 0; l < 16; ++l) pl[l] = _mm512_load_ps(&memo.planes[memo.pairEntry[b + (l < cnt ? l : 0)]]);
            Transpose16(pl);
            const __m512i sh = _mm512_maskz_loadu_epi32(valid, memo.pairShape.data() + b);
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
                const std::uint32_t k = memo.pairShape[b + l];
                lists->idx[k][lists->n[k]++] = memo.pairEntry[b + l];
            }
        }
        return;
    }
#else
    if (simd) {
        // (architect C's SSE scheme, cc_memo.cpp) pass 0: the entries meeting the shapes' union box, at native
        // width; then blocks of 4 near entries: the 8 shape boxes, the 4 entries' planes transposed once, and
        // Pass() per shape with lanes = entries (the same operations per lane)
        using F = NarrowLanes;
        for (int k = 0; k < 8; ++k) lists->n[k] = 0;
        if (memo.nearIdx.size() < n + 8) memo.nearIdx.resize(n + 8);
        std::uint32_t nn = 0;
        for (std::size_t b = 0; b < n; b += F::width) {
            unsigned bits;
            if (within != nullptr) {
                const F lx = F::load(&memo.lo[0][b]), ly = F::load(&memo.lo[1][b]), lz = F::load(&memo.lo[2][b]);
                const F hx = F::load(&memo.hi[0][b]), hy = F::load(&memo.hi[1][b]), hz = F::load(&memo.hi[2][b]);
                bits = NarrowBits(math::cmp_le(lx, F(within->hi[0])) & math::cmp_le(F(within->lo[0]), hx) & math::cmp_le(ly, F(within->hi[1])) &
                                  math::cmp_le(F(within->lo[1]), hy) & math::cmp_le(lz, F(within->hi[2])) & math::cmp_le(F(within->lo[2]), hz));
            } else {
                bits = n - b >= static_cast<std::size_t>(F::width) ? (1u << F::width) - 1u : (1u << (n - b)) - 1u;
            }
            while (bits) {
                memo.nearIdx[nn++] = static_cast<std::uint32_t>(b) + static_cast<std::uint32_t>(__builtin_ctz(bits));
                bits &= bits - 1u;
            }
        }
        const __m128 slack = _mm_set1_ps(exact::kSupportSlack), zero = _mm_setzero_ps();
        auto dot3 = [](__m128 ax, __m128 ay, __m128 az, __m128 bx, __m128 by, __m128 bz) {
            return _mm_add_ps(_mm_add_ps(_mm_mul_ps(ax, bx), _mm_mul_ps(ay, by)), _mm_mul_ps(az, bz));
        };
        for (std::uint32_t b0 = 0; b0 < nn; b0 += 4) {
            const std::uint32_t cnt = nn - b0 >= 4 ? 4u : nn - b0;
            std::uint32_t e[4];
            for (std::uint32_t l = 0; l < 4; ++l) e[l] = memo.nearIdx[b0 + (l < cnt ? l : 0)];
            const int valid = (1 << cnt) - 1;
            __m128 lo[3], hi[3];
            for (int a = 0; a < 3; ++a) {
                lo[a] = _mm_setr_ps(memo.lo[a][e[0]], memo.lo[a][e[1]], memo.lo[a][e[2]], memo.lo[a][e[3]]);
                hi[a] = _mm_setr_ps(memo.hi[a][e[0]], memo.hi[a][e[1]], memo.hi[a][e[2]], memo.hi[a][e[3]]);
            }
            int mk[8], any = 0;
            for (int k = 0; k < 8; ++k) {
                const ShapeFilter &f = shapes[k];
                __m128 c = _mm_and_ps(_mm_cmple_ps(lo[0], _mm_set1_ps(f.hi[0])), _mm_cmple_ps(_mm_set1_ps(f.lo[0]), hi[0]));
                c = _mm_and_ps(c, _mm_and_ps(_mm_cmple_ps(lo[1], _mm_set1_ps(f.hi[1])), _mm_cmple_ps(_mm_set1_ps(f.lo[1]), hi[1])));
                c = _mm_and_ps(c, _mm_and_ps(_mm_cmple_ps(lo[2], _mm_set1_ps(f.hi[2])), _mm_cmple_ps(_mm_set1_ps(f.lo[2]), hi[2])));
                mk[k] = _mm_movemask_ps(c) & valid;
                any |= mk[k];
                if (track && mk[k])
                    for (int l = 0; l < 4; ++l)
                        if (mk[k] >> l & 1) trackHit(k, e[l]);
            }
            if (!any) continue;
            __m128 pl[16];   // pl[4 c + j]: component c (x y z d) of plane j, lanes = the 4 entries
            for (int c = 0; c < 4; ++c) {
                __m128 r0 = _mm_load_ps(memo.planes[e[0]].x + 4 * c), r1 = _mm_load_ps(memo.planes[e[1]].x + 4 * c);
                __m128 r2 = _mm_load_ps(memo.planes[e[2]].x + 4 * c), r3 = _mm_load_ps(memo.planes[e[3]].x + 4 * c);
                _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
                pl[4 * c] = r0; pl[4 * c + 1] = r1; pl[4 * c + 2] = r2; pl[4 * c + 3] = r3;
            }
            for (int k = 0; k < 8; ++k) {
                if (!mk[k]) continue;
                const ShapeFilter &f = shapes[k];
                const __m128 c0 = _mm_set1_ps(f.c[0]), c1 = _mm_set1_ps(f.c[1]), c2 = _mm_set1_ps(f.c[2]);
                const __m128 reach = _mm_set1_ps(f.reach), low = _mm_set1_ps(f.low), tol = _mm_set1_ps(f.tol), lazy = _mm_set1_ps(f.lazy);
                int ok = mk[k];
                for (int j = 0; j < 4 && ok; ++j) {   // Pass() per lane: the same values, rejections as masks
                    const __m128 wx = pl[j], wy = pl[4 + j], wz = pl[8 + j];
                    const __m128 dist = _mm_sub_ps(dot3(wx, wy, wz, c0, c1, c2), pl[12 + j]);
                    __m128 rej = _mm_cmpgt_ps(dist, reach);
                    if (j == 0) rej = _mm_or_ps(rej, _mm_cmplt_ps(dist, low));
                    const __m128 isLazy = _mm_cmpgt_ps(dist, lazy);
                    if (_mm_movemask_ps(isLazy) & ok) {
                        const __m128 q0 = _mm_set1_ps(f.q[0]), q1 = _mm_set1_ps(f.q[1]), q2 = _mm_set1_ps(f.q[2]);
                        const __m128 q3 = _mm_set1_ps(f.q[3]), q4 = _mm_set1_ps(f.q[4]), q5 = _mm_set1_ps(f.q[5]);
                        const __m128 qa = dot3(q0, q3, q4, wx, wy, wz), qb = dot3(q3, q1, q5, wx, wy, wz), qc = dot3(q4, q5, q2, wx, wy, wz);
                        const __m128 h2 = dot3(qa, qb, qc, wx, wy, wz);
                        const __m128 x = _mm_sub_ps(dist, tol);
                        rej = _mm_or_ps(rej, _mm_and_ps(isLazy, _mm_and_ps(_mm_cmpgt_ps(x, zero), _mm_cmpgt_ps(_mm_mul_ps(x, x), _mm_mul_ps(h2, slack)))));
                    }
                    ok &= ~_mm_movemask_ps(rej);
                }
                while (ok) {
                    const int l = __builtin_ctz(static_cast<unsigned>(ok));
                    ok &= ok - 1;
                    lists->idx[k][lists->n[k]++] = e[l];
                }
            }
        }
        return;
    }
#endif
    for (int k = 0; k < 8; ++k) lists->n[k] = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (int k = 0; k < 8; ++k)
            if (InBox(memo, i, shapes[k])) {
                if (track) trackHit(k, static_cast<std::uint32_t>(i));
                if (Pass(memo, i, shapes[k])) lists->idx[k][lists->n[k]++] = static_cast<std::uint32_t>(i);
            }
}

std::size_t Prefilter(const Memo &memo, const std::uint32_t *list, std::size_t n, const ShapeFilter &shape, std::uint32_t *survivors) {
    std::size_t c = 0;
    for (std::size_t q = 0; q < n; ++q)
        if (Pass(memo, list[q], shape)) survivors[c++] = list[q];
    return c;
}

}  // namespace tw::cq
