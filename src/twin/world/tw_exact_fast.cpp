// Fast exact test: packets (AVX-512). See tw_exact_fast.h.
//
// Lanes = pairs, in pair order. Every lane runs the scalar code's operations in the scalar code's
// order, so a packet decides exactly what Collide decides and emits the same bits.
//   Pass 1: step 1 (PrefilterQ) on 16 pairs at a time. The FastTri rows (one cache line each) and the
//           frames' prefilter blocks (Frame line 0) are loaded and transposed in registers; one frame
//           for all 16 lanes is broadcast instead. Survivors are compress-stored.
//   Pass 2: step 2 (Decide) on 16 survivors at a time: unit-space vertices in FV's form from the
//           transposed TriVerts rows and Frame line 1; FV's FACE payload (Frame lines 2-3) for the
//           kFace lanes, in the vector unit.
//   Pass 3: in pair order, the FACE contacts from pass 2 and FV's own form for the kFvForm lanes.
#include "tw_exact_fast.h"

#include <algorithm>

namespace tw::exact {

#if defined(__AVX512F__)
namespace {

#define TW_XF_INLINE inline __attribute__((always_inline))
using F = __m512;
TW_XF_INLINE F Mul(F a, F b) { return _mm512_mul_ps(a, b); }
TW_XF_INLINE F AddF(F a, F b) { return _mm512_add_ps(a, b); }
TW_XF_INLINE F SubF(F a, F b) { return _mm512_sub_ps(a, b); }
TW_XF_INLINE F Set(float s) { return _mm512_set1_ps(s); }
TW_XF_INLINE F NegF(F a) { return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a), _mm512_set1_epi32(static_cast<int>(0x80000000u)))); }
TW_XF_INLINE F AbsF(F a) { return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a), _mm512_set1_epi32(0x7fffffff))); }
TW_XF_INLINE __mmask16 Lt(F a, F b) { return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ); }
TW_XF_INLINE __mmask16 Gt(F a, F b) { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ); }
TW_XF_INLINE __mmask16 Le(F a, F b) { return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ); }
// (a.x b.x + a.y b.y) + a.z b.z, as Dot
TW_XF_INLINE F Dot3(F ax, F ay, F az, F bx, F by, F bz) { return AddF(AddF(Mul(ax, bx), Mul(ay, by)), Mul(az, bz)); }

constexpr int kFrameFloats = static_cast<int>(sizeof(Frame) / sizeof(float));
static_assert(offsetof(Frame, c) == 0 && offsetof(Frame, sgn) == 12 && offsetof(Frame, reach) == 16 && offsetof(Frame, low) == 20 &&
                  offsetof(Frame, tol) == 24 && offsetof(Frame, lazy) == 28 && offsetof(Frame, q) == 32,
              "Frame line 0 layout");
static_assert(offsetof(Frame, meshToUnit) == 64 && offsetof(Frame, c2w) == 128 && offsetof(Frame, n2w) == 176, "Frame lines 1-3 layout");
static_assert(sizeof(I) == 48, "I layout");

// In-register 16 x 16 transpose: r[i] = row i in, r[j] = column j out.
TW_XF_INLINE void Transpose16(F r[16]) {
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

// 16 floats of each lane's record (lane l at base + stride * index[l] + off) as columns: col[k] = float k
// of every lane. All lanes on one record: broadcasts.
TW_XF_INLINE void Columns(const float *base, const std::uint32_t *index, bool uniform, int strideFloats, int off, F col[16]) {
    if (uniform) {
        const float *p = base + static_cast<std::size_t>(strideFloats) * index[0] + off;
#pragma GCC unroll 16
        for (int k = 0; k < 16; ++k) col[k] = _mm512_set1_ps(p[k]);
        return;
    }
#pragma GCC unroll 16
    for (int l = 0; l < 16; ++l) col[l] = _mm512_loadu_ps(base + static_cast<std::size_t>(strideFloats) * index[l] + off);
    Transpose16(col);
}

// Step 1 (PrefilterQ) on 16 lanes (fi / ti: frame and FastTri indices). Returns the survivors.
TW_XF_INLINE __mmask16 Prefilter16(const Frame *frames, const FastTri *tris, const std::uint32_t *fi, const std::uint32_t *ti, bool uniform,
                                   __mmask16 valid) {
    F col[16], fr[16];
    Columns(reinterpret_cast<const float *>(tris), ti, false, 16, 0, col);                // x0..3 y0..3 z0..3 d0..3
    Columns(reinterpret_cast<const float *>(frames), fi, uniform, kFrameFloats, 0, fr);   // c sgn reach low tol lazy q
    const F cx = fr[0], cy = fr[1], cz = fr[2], sgn = fr[3], reach = fr[4], low = fr[5], tol = fr[6], lazyB = fr[7];
    F dist[4];
    __mmask16 out = 0, lazy[4], anyLazy = 0;
#pragma GCC unroll 4
    for (int j = 0; j < 4; ++j) {
        dist[j] = SubF(Dot3(col[j], col[4 + j], col[8 + j], cx, cy, cz), col[12 + j]);
        if (j == 0) {
            dist[0] = Mul(dist[0], sgn);
            out |= Lt(dist[0], low);
        }
        out |= Gt(dist[j], reach);
        lazy[j] = Gt(dist[j], lazyB);
        anyLazy |= lazy[j];
    }
    __mmask16 ok = valid & ~out;
    if (anyLazy & ok) {
        const F *q = fr + 8;
        const F slack = Set(kSupportSlack), zero = _mm512_setzero_ps();
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            const F w0 = col[j], w1 = col[4 + j], w2 = col[8 + j];
            const F a = AddF(AddF(Mul(q[0], w0), Mul(q[3], w1)), Mul(q[4], w2));
            const F b = AddF(AddF(Mul(q[3], w0), Mul(q[1], w1)), Mul(q[5], w2));
            const F c = AddF(AddF(Mul(q[4], w0), Mul(q[5], w1)), Mul(q[2], w2));
            const F h2 = AddF(AddF(Mul(a, w0), Mul(b, w1)), Mul(c, w2));
            const F x = SubF(dist[j], tol);
            ok &= ~(lazy[j] & Gt(x, zero) & Gt(Mul(x, x), Mul(h2, slack)));
        }
    }
    return ok;
}

struct FaceOut {   // per lane: P N Sep (world), K (unit space)
    float v[12][16];
};

// Step 2 (Decide) on 16 lanes; FV's FACE payload of the kFace lanes into fo.
TW_XF_INLINE void Decide16(const Frame *frames, const TriVerts *verts, const std::uint32_t *fi, const std::uint32_t *ti, bool uniform,
                           __mmask16 valid, __mmask16 *face, __mmask16 *fvForm, FaceOut *fo) {
    const float *fp = reinterpret_cast<const float *>(frames);
    F m[16];   // meshToUnit (Frame line 1): r.c[0] (x y z), r.c[1], r.c[2], t
    Columns(fp, fi, uniform, kFrameFloats, 16, m);
    F p[16];   // the three vertices: 9 floats of each TriVerts (masked loads: no read past the record)
#pragma GCC unroll 16
    for (int l = 0; l < 16; ++l) p[l] = _mm512_maskz_loadu_ps(0x1FF, reinterpret_cast<const float *>(&verts[ti[l]]));
    Transpose16(p);
    F vu[3][3];
#pragma GCC unroll 3
    for (int c = 0; c < 3; ++c)
#pragma GCC unroll 3
        for (int j = 0; j < 3; ++j)   // TPoint: Dot(Row(r, j), p) + t[j]; Row(r, j) = (r.c[0][j], r.c[1][j], r.c[2][j])
            vu[c][j] = AddF(Dot3(m[j], m[3 + j], m[6 + j], p[3 * c], p[3 * c + 1], p[3 * c + 2]), m[9 + j]);
    // UnitNormal
    const F e01x = SubF(vu[1][0], vu[0][0]), e01y = SubF(vu[1][1], vu[0][1]), e01z = SubF(vu[1][2], vu[0][2]);
    const F e02x = SubF(vu[2][0], vu[0][0]), e02y = SubF(vu[2][1], vu[0][1]), e02z = SubF(vu[2][2], vu[0][2]);
    const F nX = SubF(Mul(e02z, e01y), Mul(e02y, e01z));
    const F nY = SubF(Mul(e01z, e02x), Mul(e02z, e01x));
    const F nZ = SubF(Mul(e01x, e02y), Mul(e02x, e01y));
    const F lenSq = AddF(AddF(Mul(nY, nY), Mul(nX, nX)), Mul(nZ, nZ));
    // Decide, lane for lane
    const F zero = _mm512_setzero_ps(), one = Set(1.0f), k = Set(kMarginEps);
    __mmask16 open = valid & Gt(lenSq, Set(kDirEpsSq));
    __mmask16 fv = 0;
    const F P0 = NegF(Dot3(vu[0][0], vu[0][1], vu[0][2], nX, nY, nZ));
    const F d0 = Mul(k, AddF(AddF(AbsF(vu[0][0]), AbsF(vu[0][1])), AbsF(vu[0][2])));
    const F P02 = Mul(P0, P0);
    const F lo2 = Mul(Mul(d0, d0), lenSq);
    const F up = AddF(one, d0);
    open &= ~(Lt(P0, zero) & Gt(P02, lo2));
    open &= ~(Gt(P0, zero) & Gt(P02, Mul(Mul(up, up), lenSq)));
    F R2 = SubF(lenSq, P02);
    R2 = _mm512_mask_blend_ps(Gt(R2, zero), zero, R2);
    const F slack0 = AddF(Mul(Set(2.0f), d0), Mul(d0, d0));
    const F onePlusD0 = AddF(one, d0);
#pragma GCC unroll 3
    for (int e = 0; e < 3; ++e) {
        if (!open) break;
        const int nx = e == 2 ? 0 : e + 1;
        const F ax = vu[e][0], ay = vu[e][1], az = vu[e][2];
        const F Ex = SubF(vu[nx][0], ax), Ey = SubF(vu[nx][1], ay), Ez = SubF(vu[nx][2], az);
        const F el2 = Dot3(Ex, Ey, Ez, Ex, Ey, Ez);
        const F L2 = _mm512_mask_blend_ps(Gt(el2, Set(kDirEpsSq)), one, el2);
        const F mx = SubF(Mul(Ey, nZ), Mul(Ez, nY));   // Cross(E, nr)
        const F my = SubF(Mul(Ez, nX), Mul(Ex, nZ));
        const F mz = SubF(Mul(Ex, nY), Mul(Ey, nX));
        const F S = NegF(Dot3(ax, ay, az, mx, my, mz));
        const F de = Mul(k, AddF(AddF(AddF(AbsF(ax), AbsF(ay)), AbsF(az)), one));
        const F S2 = Mul(S, S);
        const F L2len = Mul(L2, lenSq);
        const __mmask16 unsure = open & Le(S2, Mul(Mul(de, de), L2len));
        fv |= unsure;
        open &= ~unsure;
        const __mmask16 pos = open & ~Lt(S, zero);
        const F inner = AddF(AddF(slack0, Mul(Mul(Set(2.0f), de), onePlusD0)), Mul(de, de));
        const F bound = AddF(Mul(L2, R2), Mul(L2len, inner));
        fv |= pos & ~Gt(S2, bound);
        open &= ~pos;
    }
    const F dn = SubF(one, d0);
    const __mmask16 sureFace = Gt(P0, zero) & Gt(P02, lo2) & Gt(dn, zero) & Lt(P02, Mul(Mul(dn, dn), lenSq));
    *face = open & sureFace;
    *fvForm = fv | (open & ~sureFace);
    if (!*face) return;
    // EmitFace, lane for lane
    const F len = _mm512_sqrt_ps(lenSq);
    const F invLen = _mm512_div_ps(one, len);
    const F n0 = Mul(nX, invLen), n1 = Mul(nY, invLen), n2 = Mul(invLen, nZ);
    const F pd = Dot3(SubF(zero, vu[0][0]), SubF(zero, vu[0][1]), SubF(zero, vu[0][2]), n0, n1, n2);
    const F mpd = NegF(pd);
    const F pr0 = AddF(zero, Mul(n0, mpd)), pr1 = AddF(zero, Mul(n1, mpd)), pr2 = AddF(zero, Mul(n2, mpd));
    const F pm1 = SubF(pd, one);
    const F s0 = Mul(n0, pm1), s1 = Mul(n1, pm1), s2 = Mul(n2, pm1);
    F w[16], w2[16];   // frame floats 32-47 and 48-63: c2w (r, t), then n2w (r, t)
    Columns(fp, fi, uniform, kFrameFloats, 32, w);
    Columns(fp, fi, uniform, kFrameFloats, 48, w2);
    auto c2w = [&](int i) { return w[i]; };                          // c2w.r.c[k] at 3k.., t at 9..11
    auto n2w = [&](int i) { return i < 4 ? w[12 + i] : w2[i - 4]; };  // n2w.r.c[k] at 3k..
    const __mmask16 fm = *face;
#pragma GCC unroll 3
    for (int j = 0; j < 3; ++j) {
        const F P = AddF(Dot3(c2w(j), c2w(3 + j), c2w(6 + j), pr0, pr1, pr2), c2w(9 + j));
        const F Sp = Dot3(c2w(j), c2w(3 + j), c2w(6 + j), s0, s1, s2);
        _mm512_mask_storeu_ps(fo->v[j], fm, P);
        _mm512_mask_storeu_ps(fo->v[6 + j], fm, Sp);
    }
    const F g0 = Dot3(n2w(0), n2w(3), n2w(6), n0, n1, n2);
    const F g1 = Dot3(n2w(1), n2w(4), n2w(7), n0, n1, n2);
    const F g2 = Dot3(n2w(2), n2w(5), n2w(8), n0, n1, n2);
    const F gl2 = Dot3(g0, g1, g2, g0, g1, g2);
    const __mmask16 gn = Lt(Set(kDirEpsSq), gl2);
    const F ig = _mm512_div_ps(one, _mm512_sqrt_ps(gl2));
    _mm512_mask_storeu_ps(fo->v[3], fm, _mm512_mask_blend_ps(gn, g0, Mul(g0, ig)));
    _mm512_mask_storeu_ps(fo->v[4], fm, _mm512_mask_blend_ps(gn, g1, Mul(g1, ig)));
    _mm512_mask_storeu_ps(fo->v[5], fm, _mm512_mask_blend_ps(gn, g2, Mul(g2, ig)));
    _mm512_mask_storeu_ps(fo->v[9], fm, n0);
    _mm512_mask_storeu_ps(fo->v[10], fm, n1);
    _mm512_mask_storeu_ps(fo->v[11], fm, n2);
}

}  // namespace
#endif

void CollidePackets(const Frame *frames, const FastTri *tris, const TriVerts *verts, const Pair *pairs, std::size_t n,
                    std::vector<Raw> *out, PacketStats *stats, int upto) {
#if defined(__AVX512F__)
    thread_local std::vector<std::uint32_t> surv;
    thread_local std::vector<std::uint8_t> kind;
    thread_local std::vector<FaceOut> faces;
    if (surv.size() < n + 16) {
        surv.resize(n + 16);
        kind.resize(n + 16);
        faces.resize(n / 16 + 2);
    }
    alignas(64) std::uint32_t fi[16], ti[16];
    std::size_t ns = 0;
    if (tris == nullptr) {   // the pairs passed step 1 already (a world-space prefilter): all go to step 2
        for (std::size_t q = 0; q < n; ++q) surv[q] = static_cast<std::uint32_t>(q);
        ns = n;
    }
    for (std::size_t b = 0; tris != nullptr && b < n; b += 16) {
        const std::size_t cnt = std::min<std::size_t>(16, n - b);
        const __mmask16 valid = static_cast<__mmask16>(cnt == 16 ? 0xFFFFu : (1u << cnt) - 1u);
        bool uniform = true;
        for (std::size_t l = 0; l < 16; ++l) {   // lanes past cnt repeat pair b
            const Pair &p = pairs[b + (l < cnt ? l : 0)];
            fi[l] = p.frame;
            ti[l] = p.modelTri;
            uniform &= p.frame == pairs[b].frame;
        }
        const __mmask16 ok = Prefilter16(frames, tris, fi, ti, uniform, valid);
        const __m512i pid = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(b)), _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
        _mm512_mask_compressstoreu_epi32(surv.data() + ns, ok, pid);
        ns += static_cast<std::size_t>(__builtin_popcount(ok));
    }
    if (upto < 2) {
        if (stats) stats->pairs += ns;
        return;
    }
    std::size_t nface = 0, nfv = 0;
    for (std::size_t b = 0; b < ns; b += 16) {
        const std::size_t cnt = std::min<std::size_t>(16, ns - b);
        const __mmask16 valid = static_cast<__mmask16>(cnt == 16 ? 0xFFFFu : (1u << cnt) - 1u);
        bool uniform = true;
        const std::uint32_t f0 = pairs[surv[b]].frame;
        for (std::size_t l = 0; l < 16; ++l) {
            const Pair &p = pairs[surv[b + (l < cnt ? l : 0)]];
            fi[l] = p.frame;
            ti[l] = p.modelTri;
            uniform &= p.frame == f0;
        }
        __mmask16 face, fvf;
        Decide16(frames, verts, fi, ti, uniform, valid, &face, &fvf, &faces[b / 16]);
        for (std::size_t l = 0; l < cnt; ++l)
            kind[b + l] = static_cast<std::uint8_t>(((face >> l) & 1u) ? kFace : ((fvf >> l) & 1u) ? kFvForm : kReject);
        nface += static_cast<std::size_t>(__builtin_popcount(face));
        nfv += static_cast<std::size_t>(__builtin_popcount(fvf));
    }
    if (upto < 3) {
        if (stats) stats->pairs += nface + nfv;
        return;
    }
    for (std::size_t s = 0; s < ns; ++s) {
        if (kind[s] == kReject) continue;
        const Pair &p = pairs[surv[s]];
        if (kind[s] == kFace) {
            const FaceOut &fo = faces[s / 16];
            const std::size_t l = s % 16;
            Raw r{V{fo.v[0][l], fo.v[1][l], fo.v[2][l]}, V{fo.v[3][l], fo.v[4][l], fo.v[5][l]}, V{fo.v[6][l], fo.v[7][l], fo.v[8][l]},
                  V{fo.v[9][l], fo.v[10][l], fo.v[11][l]}, true, p.tri};
            r.shape = static_cast<std::uint8_t>(p.shape);
            out->push_back(r);
            continue;
        }
        const Frame &f = frames[p.frame];
        V vu[3];
        UnitVerts(f, verts[p.modelTri], vu);
        const std::size_t before = out->size();
        FvCollide(f, vu, p.tri, out);
        for (std::size_t r = before; r < out->size(); ++r) (*out)[r].shape = static_cast<std::uint8_t>(p.shape);
    }
    if (stats) {
        stats->pairs += n;
        stats->prefiltered += n - ns;
        stats->sureRejected += ns - nface - nfv;
        stats->fvRuns += nfv;
        stats->faces += nface;
    }
#else
    (void)upto;
    for (std::size_t q = 0; q < n; ++q) {
        const Pair &p = pairs[q];
        const std::size_t before = out->size();
        if (tris == nullptr) {
            V vu[3];
            UnitVerts(frames[p.frame], verts[p.modelTri], vu);
            float lenSq;
            const V nr = UnitNormal(vu, &lenSq);
            const int d = Decide(vu, nr, lenSq);
            if (d == kFace) EmitFace(frames[p.frame], vu, nr, lenSq, p.tri, out);
            else if (d == kFvForm) FvCollide(frames[p.frame], vu, p.tri, out);
        } else {
            Collide(frames[p.frame], tris[p.modelTri], verts[p.modelTri], p.tri, out);
        }
        for (std::size_t r = before; r < out->size(); ++r) (*out)[r].shape = static_cast<std::uint8_t>(p.shape);
    }
    if (stats) stats->pairs += n;
#endif
}

}  // namespace tw::exact
