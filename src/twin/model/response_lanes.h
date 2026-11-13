// Twin model: contact response, batch form over candidate lanes (M5a step 3).
//
// RespondLanes<L> runs the response of L candidates at once on TwinLanes<L>
// (tw_state.h). Lane i gives exactly the bits of RespondOrdered (response.h,
// the scalar fast kernel; RespondOrderedX is the same bits) on candidate i:
// every formula is the scalar kernel's, op for op, through lane3.h (which
// mirrors vec3.h); branches become masks; nothing is reduced across lanes.
// twin_resp_check --lanes L checks it record by record against L = 1.
//
// Per call:
//   1. pre-pass per active lane: FV sort order, trigger list (RespOut),
//      list of solid contacts (triggers and Water / GolfBall dropped: they
//      write no car state, exactly as in the scalar kernel); at L = 16 the
//      lanes whose order is P.x descending (SortLanes16) all at once, the
//      others scalar;
//   2. contact slot j = 0 .. max solid count - 1: every lane takes its j-th
//      solid contact (lanes with fewer are masked off). Contact fields are
//      gathered into lane rows; the wheel block runs if any lane has a wheel
//      contact in this slot, the body block if any lane has a body contact;
//      wheel state is written per wheel with (lane mask & wheel == w);
//   3. v, omega back to world where an impulse was applied; position
//      correction (1 cm skin) per lane.
// The branch trace (RespOut::arms) is not produced here (0).
//
// Inputs: R / Rw per lane as rotation columns c0.xyz c1.xyz c2.xyz in rows
// [9][L] (the force kernel's CarForceOutL::poseRot layout; R = the
// integrator's post-integration rotation). `active`: bit i = lane i takes
// part (inactive lanes are not written).
#ifndef TW_MODEL_RESPONSE_LANES_H
#define TW_MODEL_RESPONSE_LANES_H

#include <cstddef>
#include <cstdint>

#include "lane3.h"
#include "response.h"
#include "tw_state.h"

namespace tw::model {

// Per-tick accumulators, SoA (RespTick per lane).
template <int L>
struct alignas(64) RespTickLanes {
    std::uint32_t wheelContacts[L];
    std::uint32_t bodyContacts[L];
    float bodyNormalSum[3][L];
};

// The L = 16 gather reads a contact as 14 words: P, N, Sep, peerZ, peerId, {shape, material, flags, pad}.
static_assert(sizeof(RespContact) == 56 && offsetof(RespContact, N) == 12 && offsetof(RespContact, Sep) == 24 &&
                      offsetof(RespContact, peerZ) == 36 && offsetof(RespContact, peerId) == 48 &&
                      offsetof(RespContact, shape) == 52 && offsetof(RespContact, material) == 53 &&
                      offsetof(RespContact, flags) == 54,
              "RespContact layout");

namespace resp_lanes {

template <class F, int L>
TW_INLINE M3L<F> LoadRotRows(const float (&r)[9][L]) {
    M3L<F> m;
    m.c0 = {F::load(r[0]), F::load(r[1]), F::load(r[2])};
    m.c1 = {F::load(r[3]), F::load(r[4]), F::load(r[5])};
    m.c2 = {F::load(r[6]), F::load(r[7]), F::load(r[8])};
    return m;
}

#if TW_HAVE_AVX512F
// 16 x 16 transpose of 32-bit words (lane l's contact -> field rows).
TW_INLINE void Transpose16(__m512 *r) {
    __m512 t[16];
    for (int i = 0; i < 8; ++i) {
        t[2 * i] = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
    }
    for (int g = 0; g < 4; ++g) {
        const __m512d a0 = _mm512_castps_pd(t[4 * g]), a1 = _mm512_castps_pd(t[4 * g + 1]);
        const __m512d a2 = _mm512_castps_pd(t[4 * g + 2]), a3 = _mm512_castps_pd(t[4 * g + 3]);
        r[4 * g + 0] = _mm512_castpd_ps(_mm512_unpacklo_pd(a0, a2));
        r[4 * g + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(a0, a2));
        r[4 * g + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(a1, a3));
        r[4 * g + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(a1, a3));
    }
    for (int i = 0; i < 4; ++i) {
        t[i] = _mm512_shuffle_f32x4(r[i], r[4 + i], 0x88);
        t[4 + i] = _mm512_shuffle_f32x4(r[i], r[4 + i], 0xdd);
        t[8 + i] = _mm512_shuffle_f32x4(r[8 + i], r[12 + i], 0x88);
        t[12 + i] = _mm512_shuffle_f32x4(r[8 + i], r[12 + i], 0xdd);
    }
    for (int i = 0; i < 8; ++i) {
        r[i] = _mm512_shuffle_f32x4(t[i], t[8 + i], 0x88);
        r[8 + i] = _mm512_shuffle_f32x4(t[i], t[8 + i], 0xdd);
    }
}
#endif

template <class F> TW_INLINE V3L<F> DiagMulL(const float *d, V3L<F> a) { return {a.x * F(d[0]), a.y * F(d[1]), a.z * F(d[2])}; }

// (e - 1) / K for direction d at lever Lv, K = 1/m + (I^-1 (Lv x d)).(Lv x d)   [ImpulseFactor]
template <class F>
TW_INLINE F ImpulseFactorL(const CarForceConsts &k, F e, V3L<F> d, V3L<F> Lv, V3L<F> *aL) {
    const V3L<F> ld = crossl(Lv, d);
    *aL = DiagMulL<F>(k.inertiaInvBody, ld);
    return (e - F(1.0f)) / (F(k.invMass) + dotl(*aL, ld));
}

#if TW_HAVE_AVX512F
// The pre-pass for 16 lanes at once, for the lanes of `active` with 1..kSortLanesMax contacts whose P.x are all
// distinct and ordered and which have no trigger contact: their FV order is then P.x descending (RespSortOrder,
// whatever its path, ties only decide equal keys), so a solid contact's place is the number of solid contacts with a
// larger P.x. Writes those lanes' solid rows (solid[j][l]: lane l's j-th solid contact) and counts and raises
// *maxSolid; returns the other lanes of `active` (the scalar pre-pass: more contacts, ties, NaN, triggers, none).
constexpr int kSortLanesMax = 8;
TW_INLINE std::uint32_t SortLanes16(const RespContact *const *contacts, const int *ncontacts, std::uint32_t active,
                                    std::uint8_t (*solid)[16], std::int32_t *nsolid, int *maxSolid) {
    static_assert(sizeof(const RespContact *) == 8, "64-bit addresses");
    const __m512i n = _mm512_loadu_si512(ncontacts);
    const __mmask16 fast = _mm512_mask_cmpgt_epi32_mask(static_cast<__mmask16>(active), n, _mm512_setzero_si512()) &
                           _mm512_cmple_epi32_mask(n, _mm512_set1_epi32(kSortLanesMax));
    if (fast == 0) return active;
    const int nmax = _mm512_mask_reduce_max_epi32(fast, n);
    // the lanes' contact addresses (lanes outside `fast` are not read)
    const __m512i pa = _mm512_maskz_loadu_epi64(static_cast<__mmask8>(fast), contacts);
    const __m512i pb = _mm512_maskz_loadu_epi64(static_cast<__mmask8>(fast >> 8), contacts + 8);
    __m512 X[kSortLanesMax];
    __mmask16 S[kSortLanesMax], P[kSortLanesMax];
    __mmask16 bad = 0;
    const __m512i trigBit = _mm512_set1_epi32(static_cast<int>(kRespTrigger) << 16);
    const __m512i water = _mm512_set1_epi32(resp_detail::kMatWater), golf = _mm512_set1_epi32(resp_detail::kMatGolfBall);
    for (int i = 0; i < nmax; ++i) {
        P[i] = _mm512_mask_cmpgt_epi32_mask(fast, n, _mm512_set1_epi32(i));   // lanes with a contact i
        const __mmask8 lo = static_cast<__mmask8>(P[i]), hi = static_cast<__mmask8>(P[i] >> 8);
        const __m512i ox = _mm512_set1_epi64(static_cast<long long>(i) * static_cast<long long>(sizeof(RespContact)));
        const __m512i ow = _mm512_set1_epi64(static_cast<long long>(i) * static_cast<long long>(sizeof(RespContact)) +
                                             static_cast<long long>(offsetof(RespContact, shape)));
        const __m256 xl = _mm512_mask_i64gather_ps(_mm256_setzero_ps(), lo, _mm512_add_epi64(pa, ox), nullptr, 1);
        const __m256 xh = _mm512_mask_i64gather_ps(_mm256_setzero_ps(), hi, _mm512_add_epi64(pb, ox), nullptr, 1);
        const __m256i wl = _mm512_mask_i64gather_epi32(_mm256_setzero_si256(), lo, _mm512_add_epi64(pa, ow), nullptr, 1);
        const __m256i wh = _mm512_mask_i64gather_epi32(_mm256_setzero_si256(), hi, _mm512_add_epi64(pb, ow), nullptr, 1);
        X[i] = _mm512_insertf32x8(_mm512_castps256_ps512(xl), xh, 1);
        const __m512i w = _mm512_inserti64x4(_mm512_castsi256_si512(wl), wh, 1);   // shape | material << 8 | flags << 16
        bad |= _mm512_mask_test_epi32_mask(P[i], w, trigBit);
        const __m512i mat = _mm512_and_si512(_mm512_srli_epi32(w, 8), _mm512_set1_epi32(0xff));
        S[i] = P[i] & _mm512_cmpneq_epi32_mask(mat, water) & _mm512_cmpneq_epi32_mask(mat, golf);
    }
    __m512i rank[kSortLanesMax];
    for (int i = 0; i < nmax; ++i) rank[i] = _mm512_setzero_si512();
    const __m512i one = _mm512_set1_epi32(1);
    for (int i = 1; i < nmax; ++i)
        for (int j = 0; j < i; ++j) {
            const __mmask16 both = P[i] & P[j];
            bad |= both & ~_mm512_mask_cmp_ps_mask(both, X[i], X[j], _CMP_NEQ_OQ);   // equal or unordered
            const __mmask16 jFirst = _mm512_cmp_ps_mask(X[j], X[i], _CMP_GT_OQ);
            rank[i] = _mm512_mask_add_epi32(rank[i], S[j] & jFirst, rank[i], one);
            rank[j] = _mm512_mask_add_epi32(rank[j], S[i] & ~jFirst, rank[j], one);
        }
    const __mmask16 ok = fast & ~bad;
    if (ok == 0) return active;
    __m512i ns = _mm512_setzero_si512();
    for (int i = 0; i < nmax; ++i) ns = _mm512_mask_add_epi32(ns, S[i], ns, one);
    const int nsMax = _mm512_mask_reduce_max_epi32(ok, ns);
    for (int r = 0; r < nsMax; ++r) {
        const __m512i rr = _mm512_set1_epi32(r);
        __m512i idx = _mm512_setzero_si512();
        for (int i = 0; i < nmax; ++i)
            idx = _mm512_mask_mov_epi32(idx, S[i] & _mm512_cmpeq_epi32_mask(rank[i], rr), _mm512_set1_epi32(i));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(solid[r]), _mm512_cvtepi32_epi8(idx));
    }
    _mm512_mask_storeu_epi32(nsolid, ok, ns);
    if (nsMax > *maxSolid) *maxSolid = nsMax;
    return active & ~static_cast<std::uint32_t>(ok);
}
#endif

}  // namespace resp_lanes

template <int L>
TW_INLINE void RespondLanes(const CarForceConsts &k, TwinLanes<L> &s, const float (&Rrows)[9][L], const float (&Rwrows)[9][L],
                            std::uint32_t active, const RespContact *const *contacts, const int *ncontacts,
                            RespTickLanes<L> &tick, RespOut *out) {
    using namespace resp_detail;
    using namespace resp_lanes;
    using F = tw::math::vf<L>;
    using I = LaneI<F>;
    using M = LaneM<F>;
    static_assert(F::width == L, "lane width");

    // ---- 1. pre-pass: order, triggers, solid contacts (solid[j][l]: lane l's j-th solid contact) ----
    alignas(64) std::uint8_t solid[kRespMaxContacts][L];
    alignas(64) std::int32_t nsolid[L];
    int maxSolid = 0;
    for (int l = 0; l < L; ++l) {
        nsolid[l] = 0;
        out[l].triggerCount = 0u;
        out[l].arms = 0u;
        out[l].replacement[0] = out[l].replacement[1] = out[l].replacement[2] = 0.0f;
    }
    std::uint32_t scalarLanes = active;
#if TW_HAVE_AVX512F
    if constexpr (L == 16) scalarLanes = SortLanes16(contacts, ncontacts, active, solid, nsolid, &maxSolid);
#endif
    for (std::uint32_t m = scalarLanes; m != 0u; m &= m - 1u) {
        const int l = __builtin_ctz(m);
        int n = ncontacts[l];
        if (n <= 0) continue;
        if (n > kRespMaxContacts) {
            n = kRespMaxContacts;
            out[l].arms |= kRespArmOverflow;
        }
        const RespContact *c = contacts[l];
        std::uint8_t order[kRespMaxContacts];
        RespSortOrder(c, n, order);
        for (int oi = 0; oi < n; ++oi) {
            const RespContact &ct = c[order[oi]];
            if (ct.flags & kRespTrigger) {
                out[l].trigger[out[l].triggerCount++] = order[oi];
                continue;
            }
            if (ct.material == kMatWater || ct.material == kMatGolfBall) continue;
            solid[nsolid[l]++][l] = order[oi];
        }
        if (nsolid[l] > maxSolid) maxSolid = nsolid[l];
    }
    if (maxSolid == 0) return;   // no car contact in any lane: nothing to write (pos += 0 skipped as in L = 1)

    // ---- 2. lane state ----
    const M3L<F> R = LoadRotRows<F, L>(Rrows);
    const V3L<F> rwc2 = {F::load(Rwrows[6]), F::load(Rwrows[7]), F::load(Rwrows[8])};
    const V3L<F> x = Ld3<F>(s.pos), com = v3l<F>(k.com);
    V3L<F> v = mul_tl(R, Ld3<F>(s.vel)), w = mul_tl(R, Ld3<F>(s.angVel));
    const V3L<F> v0w = Ld3<F>(s.vel), w0w = Ld3<F>(s.angVel), v0 = v, w0 = w;   // RespondOrdered's write-back (response.h)
    F v2 = len2l(v);
    M anyImp = MaskFalse<F>();
    V3L<F> S = {F(0.0f), F(0.0f), F(0.0f)};
    M haveS = MaskFalse<F>();
    const F zero(0.0f), one(1.0f);
    const I izero(0), ione(1);
    const bool clampW = k.maxAbs >= -kEps;
    float restX[4], restY[4], restZ[4];
    for (int wh = 0; wh < 4; ++wh) {
        restX[wh] = k.wheel[wh].rest[0];
        restY[wh] = k.wheel[wh].rest[1];
        restZ[wh] = k.wheel[wh].rest[2];
    }

    alignas(64) float gP[3][L], gN[3][L], gS[3][L], gZ[3][L];
    alignas(64) std::int32_t gId[L], gMat[L], gWheel[L];
    alignas(64) float wheelTable[16];   // shape -> wheel (-1 body), padded for Lookup16
    for (int i = 0; i < 16; ++i) wheelTable[i] = static_cast<float>(k.wheelOfShape[i & 7]);
    static const RespContact kNoContact{};
    for (int j = 0; j < maxSolid; ++j) {
        std::uint32_t slotBits = 0u;
        V3L<F> N, Sep, P, PZ;
        I wi, mat, pid;
#if TW_HAVE_AVX512F
        if constexpr (L == 16) {
            // one masked 56-byte load per lane, transposed in registers (no store -> wide load round trip)
            __m512 r[16];
            for (int l = 0; l < 16; ++l) {
                const RespContact *ct = &kNoContact;
                if (j < nsolid[l]) {
                    ct = &contacts[l][solid[j][l]];
                    slotBits |= 1u << l;
                }
                r[l] = _mm512_maskz_loadu_ps(0x3FFF, reinterpret_cast<const float *>(ct));
            }
            Transpose16(r);
            P = {F(r[0]), F(r[1]), F(r[2])};
            N = {F(r[3]), F(r[4]), F(r[5])};
            Sep = {F(r[6]), F(r[7]), F(r[8])};
            PZ = {F(r[9]), F(r[10]), F(r[11])};
            pid = tw::math::as_int(F(r[12]));
            const I packed = tw::math::as_int(F(r[13]));
            mat = tw::math::shr_logical<8>(packed) & I(0xff);
            wi = select(MaskOf<F>(slotBits), tw::math::cvtt_i32(Lookup16(wheelTable, packed & I(7))), I(-1));
        } else
#endif
        {
            // gather slot j (inactive / exhausted lanes: benign zeros, masked off)
            for (int l = 0; l < L; ++l) {
                if (j < nsolid[l]) {
                    const RespContact &ct = contacts[l][solid[j][l]];
                    for (int a = 0; a < 3; ++a) {
                        gP[a][l] = ct.P[a];
                        gN[a][l] = ct.N[a];
                        gS[a][l] = ct.Sep[a];
                        gZ[a][l] = ct.peerZ[a];
                    }
                    gId[l] = static_cast<std::int32_t>(ct.peerId);
                    gMat[l] = ct.material;
                    gWheel[l] = k.wheelOfShape[ct.shape & 7u];
                    slotBits |= 1u << l;
                } else {
                    for (int a = 0; a < 3; ++a) gP[a][l] = gN[a][l] = gS[a][l] = gZ[a][l] = 0.0f;
                    gId[l] = 0;
                    gMat[l] = 0;
                    gWheel[l] = -1;
                }
            }
            N = {F::load(gN[0]), F::load(gN[1]), F::load(gN[2])};
            Sep = {F::load(gS[0]), F::load(gS[1]), F::load(gS[2])};
            P = {F::load(gP[0]), F::load(gP[1]), F::load(gP[2])};
            PZ = {F::load(gZ[0]), F::load(gZ[1]), F::load(gZ[2])};
            wi = I::load(gWheel);
            mat = I::load(gMat);
            pid = I::load(gId);
        }
        const M mj = MaskOf<F>(slotBits);
        const V3L<F> nrm = mul_tl(R, N);
        const V3L<F> p = mul_tl(R, P - x);
        const V3L<F> lp = p - com;
        const V3L<F> lnp = crossl(lp, nrm);
        StIM<F>(SF(s.airRefresh), mj, ione);
        const M isWheel = IGt(wi, I(-1));   // wi >= 0
        const M mWheel = mj & isWheel;
        const M mBody = mj & ~isWheel;
        const M metal = IEq(mat, I(static_cast<std::int32_t>(kMatMetal)));
        const F eBody = -select(metal, F(k.bodyImpulseMetal), F(k.bodyImpulseOther));
        V3L<F> rw = -Sep;
        V3L<F> D = nrm, aL = {zero, zero, zero};
        F sImp = zero;
        M mImp = MaskFalse<F>();

        if (any(mWheel)) {
            const M present = tw::math::cmp_lt(FabsL(nrm.x), F(kSinQuarterPi));
            // absorb / maxRepl / rest of each lane's wheel
            F absorbSel = zero, maxReplSel = zero, cx = zero, cy = zero, cz = zero;
            M mw[4];
            for (int wh = 0; wh < 4; ++wh) {
                mw[wh] = mWheel & IEq(wi, I(wh));
                absorbSel = select(mw[wh], Ld<F>(s.absorb[wh]), absorbSel);
                maxReplSel = select(mw[wh], Ld<F>(s.maxRepl[wh]), maxReplSel);
                cx = select(mw[wh], F(restX[wh]), cx);
                cy = select(mw[wh], F(restY[wh]), cy);
                cz = select(mw[wh], F(restZ[wh]), cz);
            }
            const F pz = dotl(rwc2, PZ);
            // suspension clamp of the vertical push-out
            const F rhoY = -dotl(R.c1, Sep);
            const M posRho = tw::math::cmp_gt(rhoY, zero);
            const F travel = absorbSel - F(k.maxAbs);
            const M atLimit = clampW ? tw::math::cmp_le(travel, rhoY) : MaskFalse<F>();
            const F cand = select(atLimit, travel, rhoY);
            const M blocked = ~posRho | atLimit;
            const M raise = mWheel & posRho & tw::math::cmp_gt(cand, maxReplSel);
            const V3L<F> rwWheel = Sel3(posRho, rw - R.c1 * cand, rw);
            for (int wh = 0; wh < 4; ++wh) {
                const M m = mw[wh];
                if (!any(m)) continue;
                StIM<F>(s.contact[wh], m, select(present, ione, izero));
                St3AtM<F>(s.normalSum, 3 * wh, m & present, Ld3At<F>(s.normalSum, 3 * wh) + nrm);
                StIM<F>(s.surface[wh], m & present, mat);
                StM<F>(s.peerAxisZ[wh], m, pz);
                StIM<F>(s.peerId[wh], m, pid);
                St3AtM<F>(s.latestPoint, 3 * wh, m, p);
                StM<F>(s.maxRepl[wh], m & raise, cand);
            }
            StIM<F>(SF(s.latContact), mWheel & ~present, ione);
            (I::load(reinterpret_cast<const std::int32_t *>(tick.wheelContacts)) + select(mWheel, ione, izero))
                    .store(reinterpret_cast<std::int32_t *>(tick.wheelContacts));
            // impulse: one lever per lane (full: p; absorbed: wheel surface point; side hit: (p.x, com.y, p.z))
            const F eWheel = -select(metal, F(k.wheelImpulseMetal), F(k.wheelImpulseOther));
            const M full = blocked | ~tw::math::cmp_gt(nrm.y, zero);
            const V3L<F> cur = {cx, cy - absorbSel, cz};
            const V3L<F> side = {lp.x, zero, lp.z};
            const V3L<F> lever = Sel3(present, Sel3(full, lp, cur - com), side);
            const F e = select(present, eWheel, eBody);
            V3L<F> aW;
            const F f = ImpulseFactorL<F>(k, e, nrm, lever, &aW);
            const F un = dotl(nrm, v) + dotl(lnp, w);
            const M neg = tw::math::cmp_lt(un, zero);
            const M fullImp = ~present | full;
            const F un2 = un - nrm.y * (nrm.y * un);
            const F sW = select(fullImp, un * f, un2 * f);
            const M impW = mWheel & neg & (fullImp | tw::math::cmp_lt(un2, zero));
            rw = Sel3(mWheel, rwWheel, rw);
            aL = aW;
            sImp = sW;
            mImp = impW;
        }
        if (any(mBody)) {
            const M roof = tw::math::cmp_lt(nrm.y, F(-0.75f));
            const V3L<F> rwBody = Sel3(roof, N * -dotl(N, Sep), -Sep);
            rw = Sel3(mBody, rwBody, rw);
            {
                const V3L<F> bn = {F::load(tick.bodyNormalSum[0]), F::load(tick.bodyNormalSum[1]), F::load(tick.bodyNormalSum[2])};
                const V3L<F> nb = Sel3(mBody, bn + nrm, bn);
                nb.x.store(tick.bodyNormalSum[0]);
                nb.y.store(tick.bodyNormalSum[1]);
                nb.z.store(tick.bodyNormalSum[2]);
                (I::load(reinterpret_cast<const std::int32_t *>(tick.bodyContacts)) + select(mBody, ione, izero))
                        .store(reinterpret_cast<std::int32_t *>(tick.bodyContacts));
            }
            StIM<F>(SF(s.bodyContact), mBody, ione);
            const V3L<F> u = v + crossl(w, lp);
            const F vn = dotl(nrm, u);
            const M approach = mBody & tw::math::cmp_lt(vn, zero);
            if (any(approach)) {
                const F mu = select(metal, F(k.tangentLimitMetal), F(k.tangentLimitOther));
                const V3L<F> nv = nrm * vn;
                V3L<F> t = u - nv;
                const F nl = tw::math::sqrt(len2l(nv));
                const F tl = tw::math::sqrt(len2l(t));
                const F tmax = nl * mu;
                t = Sel3(tw::math::cmp_gt(tl, tmax), t * (tmax / tl), t);
                V3L<F> d = -(t + nv);
                const F dl = tw::math::sqrt(len2l(d));
                const M okd = approach & tw::math::cmp_lt(F(kEps), dl);
                d = d * (one / dl);
                V3L<F> aB;
                const F fb = ImpulseFactorL<F>(k, eBody, d, lp, &aB);
                const F sB = dotl(u, d) * fb;
                D = Sel3(mBody, d, D);
                aL = Sel3(mBody, aB, aL);
                sImp = select(mBody, sB, sImp);
                mImp = mImp | okd;
            }
        }
        // ApplyLocal on the lanes with an impulse
        if (any(mImp)) {
            const V3L<F> vn2 = v + D * (sImp * F(k.invMass));
            const F a2 = len2l(vn2);
            const M guard = tw::math::cmp_lt(v2, a2) & tw::math::cmp_lt(F(k.growthLimitSq), a2 - v2);
            const V3L<F> vz = {zero, zero, zero};
            v = Sel3(mImp, Sel3(guard, vz, vn2), v);
            v2 = select(mImp, select(guard, zero, a2), v2);
            V3L<F> dw = aL * (sImp * F(k.angularImpulseScale));
            if (k.angularImpulseYScale != 1.0f) {
                V3L<F> dww = mull(R, dw);
                dww.y = F(k.angularImpulseYScale) * dww.y;
                dw = mul_tl(R, dww);
            }
            V3L<F> wn = w + dw;
            if (k.maxAngularSpeed > 0.0f) {
                const F l2 = len2l(wn);
                const M cl = tw::math::cmp_gt(l2, F(k.maxAngularSpeed * k.maxAngularSpeed));
                if (any(cl & mImp)) wn = Sel3(cl, wn * (F(k.maxAngularSpeed) / tw::math::sqrt(l2)), wn);
            }
            w = Sel3(mImp, wn, w);
            anyImp = anyImp | mImp;
        }
        // replacement merge (FoldReplacement), first contact starts the sum
        {
            const F proj = dotl(rw, S);
            const F l2 = len2l(S);
            const M c = haveS & tw::math::cmp_gt(proj, zero) & tw::math::cmp_lt(F(kLen2Eps), l2);
            const F pc = select(tw::math::cmp_lt(l2, proj), l2, proj);
            const V3L<F> Sp = Sel3(c, S - S * (pc / l2), S) + rw;
            S = Sel3(mj, Sel3(haveS, Sp, rw), S);
            haveS = haveS | mj;
        }
    }

    // ---- 3. back to world, position correction ----
    if (any(anyImp)) {
        St3M<F>(s.vel, anyImp, v0w + mull(R, v - v0));
        St3M<F>(s.angVel, anyImp, w0w + mull(R, w - w0));
    }
    const F l2 = len2l(S);
    const M ok = haveS & tw::math::cmp_lt(F(kSkin2), l2);
    if (any(ok)) {
        const F inv = one / tw::math::sqrt(l2);
        const V3L<F> delta = Sel3(ok, S - (S * inv) * F(kSkin), V3L<F>{zero, zero, zero});
        St3M<F>(s.pos, ok, x + delta);
        alignas(64) float dx[L], dy[L], dz[L];
        delta.x.store(dx);
        delta.y.store(dy);
        delta.z.store(dz);
        for (int l = 0; l < L; ++l) {
            out[l].replacement[0] = dx[l];
            out[l].replacement[1] = dy[l];
            out[l].replacement[2] = dz[l];
        }
    }
}

}  // namespace tw::model

#endif
