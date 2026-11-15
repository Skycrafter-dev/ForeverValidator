// CellGrid query (templates). Included by cc_grid.h.
#ifndef CC_GRID_INL_H
#define CC_GRID_INL_H

#include <algorithm>
#include <cmath>

#include "../math/tw_lanes.h"
#include "../math/tw_math.h"

namespace cc {

// Prepared per (instance, query): world-test broadcasts and the model SoA pointers along each world axis.
struct CellGridPrep {
    using V = tw::math::vf<16>;
    const float *lo[3], *hi[3];
    bool ng[3];
    V t[3], ql[3], qh[3];
};

template <class F>
inline void CellGrid::ScanRange(const GInst &g, std::uint32_t first, std::uint32_t count, const Dedupe *dd, const float wl[3], const float wh[3],
                                F &f) const {
    CellGridPrep pp;
    Prepare(g, wl, wh, &pp);
    ScanPrepared(g, pp, first, count, dd, f);
}

inline void CellGrid::Prepare(const GInst &g, const float wl[3], const float wh[3], CellGridPrep *pp) const {
    using V = tw::math::vf<16>;
    for (int a = 0; a < 3; ++a) {
        // world AABB of a model triangle along world axis a: (neg ? -hi[p] : lo[p]) + t, (neg ? -lo[p] : hi[p]) + t
        pp->ng[a] = (g.neg >> a) & 1u;
        pp->lo[a] = pp->ng[a] ? mhi_[g.p[a]].data() : mlo_[g.p[a]].data();
        pp->hi[a] = pp->ng[a] ? mlo_[g.p[a]].data() : mhi_[g.p[a]].data();
        pp->t[a] = V(g.t[a]);
        pp->ql[a] = V(wl[a]);
        pp->qh[a] = V(wh[a]);
    }
}

template <class F>
inline void CellGrid::ScanPrepared(const GInst &g, const CellGridPrep &pp, std::uint32_t first, std::uint32_t count, const Dedupe *dd, F &f) const {
    using V = tw::math::vf<16>;
    const V zero(0.0f);
    const std::uint32_t end = first + count;
    for (std::uint32_t b = first; b < end; b += 16) {
        CC_GRID_CNT(qBlocks);
        auto L = [&](const float *p) { return V::load(p + b); };
        // a negated float is exact, so (-hi) + t is FV's rounding of the mirrored bound
        auto ov = tw::math::cmp_le((pp.ng[0] ? zero - L(pp.lo[0]) : L(pp.lo[0])) + pp.t[0], pp.qh[0]) &
                  tw::math::cmp_ge((pp.ng[0] ? zero - L(pp.hi[0]) : L(pp.hi[0])) + pp.t[0], pp.ql[0]);
        ov = ov & tw::math::cmp_le((pp.ng[1] ? zero - L(pp.lo[1]) : L(pp.lo[1])) + pp.t[1], pp.qh[1]) &
             tw::math::cmp_ge((pp.ng[1] ? zero - L(pp.hi[1]) : L(pp.hi[1])) + pp.t[1], pp.ql[1]);
        ov = ov & tw::math::cmp_le((pp.ng[2] ? zero - L(pp.lo[2]) : L(pp.lo[2])) + pp.t[2], pp.qh[2]) &
             tw::math::cmp_ge((pp.ng[2] ? zero - L(pp.hi[2]) : L(pp.hi[2])) + pp.t[2], pp.ql[2]);
        if (!tw::math::any(ov)) continue;
        if (dd) {  // multi-registered triangle: report it only from its registration in the sub-cell holding max(local lo, query local lo)
            V cid(0.0f);
            for (int j = 0; j < 3; ++j) {
                const V fj = (tw::math::max(V::load(&mlo_[j][b]), V(dd->qlo[j])) - V(dd->org[j])) * V(dd->inv);
                cid = cid * V(dd->dim[j]) + tw::math::cvt_f32(tw::math::cvtt_i32(fj));  // exact: ids < 2^24
            }
            ov = ov & tw::math::cmp_eq(cid, V::load(&mcell_[b]));
            if (!tw::math::any(ov)) continue;
        }
#if TW_HAVE_AVX512F
        unsigned bits = ov.v;
#else
        alignas(64) float mk[16];
        tw::math::select(ov, V(1.0f), V(0.0f)).store(mk);
        unsigned bits = 0;
        for (int l = 0; l < 16; ++l) bits |= (mk[l] != 0.0f ? 1u : 0u) << l;
#endif
        if (end - b < 16) bits &= (1u << (end - b)) - 1u;
        while (bits) {
            const unsigned l = static_cast<unsigned>(__builtin_ctz(bits));
            bits &= bits - 1u;
            CC_GRID_CNT(qHits);
            f(g.firstTri + mtri_[b + l]);
        }
    }
}

template <class F>
inline void CellGrid::ScanModel(const GInst &g, const GModel &md, const Box3 &q, F &f) const {
    CellGridPrep pp;
    Prepare(g, q.lo, q.hi, &pp);
    if (md.subFirst == 0xffffffffu) {
        ScanPrepared(g, pp, md.first, md.count, nullptr, f);
        return;
    }
    if (md.nBig) ScanPrepared(g, pp, md.first, md.nBig, nullptr, f);
    // query box in mesh-local coordinates, rounded outward
    Dedupe dd;
    float lhi[3];
    for (int a = 0; a < 3; ++a) {
        const int p = g.p[a];
        float l, h;
        if ((g.neg >> a) & 1u) { l = g.t[a] - q.hi[a]; h = g.t[a] - q.lo[a]; }
        else { l = q.lo[a] - g.t[a]; h = q.hi[a] - g.t[a]; }
        const float e = 1e-4f + 1e-6f * (std::fabs(q.lo[a]) + std::fabs(q.hi[a]) + std::fabs(g.t[a]));
        dd.qlo[p] = l - e;
        lhi[p] = h + e;
    }
    int c0[3], c1[3];
    for (int a = 0; a < 3; ++a) {
        dd.org[a] = md.subOrg[a];
        dd.dim[a] = static_cast<float>(md.subDim[a]);
        const float fl = (dd.qlo[a] - md.subOrg[a]) * md.subInv, fh = (lhi[a] - md.subOrg[a]) * md.subInv;
        if (fh < 0.0f || fl >= static_cast<float>(md.subDim[a])) return;
        c0[a] = fl < 0.0f ? 0 : static_cast<int>(fl);
        c1[a] = std::min(static_cast<int>(md.subDim[a]) - 1, static_cast<int>(fh));
    }
    dd.inv = md.subInv;
    for (int x = c0[0]; x <= c1[0]; ++x)
        for (int y = c0[1]; y <= c1[1]; ++y) {  // the z run of sub-cells is one contiguous range
            const std::uint32_t row = md.subFirst + (static_cast<std::uint32_t>(x) * md.subDim[1] + static_cast<std::uint32_t>(y)) * md.subDim[2];
            const std::uint32_t a0 = subStart_[row + static_cast<std::uint32_t>(c0[2])], a1 = subStart_[row + static_cast<std::uint32_t>(c1[2]) + 1];
            if (a1 > a0) ScanPrepared(g, pp, md.first + a0, a1 - a0, &dd, f);
        }
}

template <class F>
inline void CellGrid::Query(const Box3 &q, F &&f) const {
    // huge triangles: small BVH4
    if (!skipHuge) huge_.Query(q, [&](std::uint32_t h) { f(hugeTri_[h]); });
    if (skipGrid) return;
    if (ginst_.empty()) return;
    int c0[3], c1[3];
    for (int a = 0; a < 3; ++a) {
        const float fl = (q.lo[a] - org_[a]) * inv_[a], fh = (q.hi[a] - org_[a]) * inv_[a];
        if (!(fh >= 0.0f) || !(fl < static_cast<float>(dim[a]))) return;
        c0[a] = fl < 0.0f ? 0 : static_cast<int>(fl);
        c1[a] = std::min(dim[a] - 1, static_cast<int>(fh));
    }
    ++gen_;
    if (gen_ == 0) { std::fill(stamp_.begin(), stamp_.end(), 0u); gen_ = 1; }
    for (int x = c0[0]; x <= c1[0]; ++x)
        for (int y = c0[1]; y <= c1[1]; ++y)
            for (int z = c0[2]; z <= c1[2]; ++z) {
                const std::size_t c = (static_cast<std::size_t>(x) * dim[1] + static_cast<std::size_t>(y)) * dim[2] + static_cast<std::size_t>(z);
                CC_GRID_CNT(qCells);
                for (std::uint32_t i = cellStart_[c]; i < cellStart_[c + 1]; ++i) {
                    const std::uint32_t gi = cellInst_[i];
                    if (stamp_[gi] == gen_) continue;
                    stamp_[gi] = gen_;
                    const GInst &g = ginst_[gi];
                    if (g.lo[0] > q.hi[0] || g.hi[0] < q.lo[0] || g.lo[1] > q.hi[1] || g.hi[1] < q.lo[1] || g.lo[2] > q.hi[2] || g.hi[2] < q.lo[2]) continue;
                    CC_GRID_CNT(qInstances);
                    ScanModel(g, models_[g.model], q, f);
                }
            }
}

}  // namespace cc

#endif
