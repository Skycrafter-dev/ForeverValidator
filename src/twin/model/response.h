// Twin model: contact response, one substep, single lane (L = 1). M5a.
//
// Board 20 §3.5-3.7 (normative), board 10 §2.16 (state the force model reads
// next substep), docs/twin/m5a-response.md (validation against FV, cost).
// FV: CHmsZoneDynamic::ComputeCollisionResponse (collision_response.cpp)
//   -> SolveImpulse (solve_impulse.cpp:215-283) -> CSceneVehicleCar::AbsorbContact
//   (scene_vehicle_car_wheels.cpp:582-636), then CHmsDyna::DoPostCollisionDynamic
//   (dyna_core.cpp:631-716).
//
// Input: the substep's merged per-shape contacts (world frame, collision order)
// at FV's post-integration pose. Per substep:
//   1. sort by (P, N, Sep) descending, 9 keys; full tie: non-primary first, then
//      (shape, input index) [FV: qsort order; board 20 §3.5 DECIDE];
//   2. for each contact in that order, from the CURRENT v, omega (earlier
//      impulses of this substep included):
//        trigger          -> handed to the race layer (RespOut::trigger), no car effect
//        Water / GolfBall -> ignored (no state, replacement 0)
//        wheel shape      -> last-contact-wins wheel state, suspension clamp of the
//                            push-out (maxRepl), e = 0.8 impulse or side hit (e = 0.5)
//        body shape       -> roof projection of the push-out, friction-cone impulse
//                            (e = 0.5, tangent cap 0.3 / Metal 0.2)
//      every impulse: growth guard (|v|^2 grows by > 10000 -> v = 0), 0.55 angular
//      scale, |omega| <= 100;
//   3. replacement merge in the same order, 1 cm skin, pos += result (no velocity).
// Zero contacts: returns at once (FV does nothing either).
//
// Frames: R is the post-integration rotation (contact geometry, impulses),
// Rw the substep-start rotation (FV's write state; only the booster peer axis
// uses it, board 20 §3.6 DECIDE 4). Both are passed in so the substep loop can
// reuse what it already has (CarForceOut::poseRot is Rw).
#ifndef TW_MODEL_RESPONSE_H
#define TW_MODEL_RESPONSE_H

#include <cstdint>

#include "car_force_consts.h"
#include "impulse.h"
#include "tw_state.h"
#include "vec3.h"

#if defined(__SSE4_1__) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#include <smmintrin.h>
#define TW_RESP_HAVE_X 1
#else
#define TW_RESP_HAVE_X 0
#endif

namespace tw::model {

constexpr int kRespMaxContacts = 64;   // FV CUDA caps 64 replacements; observed max 26 per detection

enum : std::uint8_t {
    kRespPrimary = 1u << 0,   // FACE contact (FV sphereMergePrimary): sort tie-break only
    kRespTrigger = 1u << 1,   // race trigger (group 1): FV runs no car handler for it
};

// One merged contact as the collision stage hands it over (world frame).
struct RespContact {
    float P[3];               // contact point
    float N[3];               // impulse normal (unit, from the surface towards the car)
    float Sep[3];             // separation; the car's push-out is -Sep
    float peerZ[3];           // world +z axis of the touched block (wheel contacts: booster scale)
    std::uint32_t peerId;     // solid contact: stable block id (FV corpus id, read back by boosters);
                              // kRespTrigger contact: race trigger id (index into TwRaceMap::triggers)
    std::uint8_t shape;       // car collision tree index 0..7
    std::uint8_t material;    // surface id of the triangle (FV materialB)
    std::uint8_t flags;       // kRespPrimary | kRespTrigger
    std::uint8_t pad;
};

// Per-tick accumulators for the rules snapshot (FV AfterContacts): reset at
// every tick start; not TwinState (always zero at tick boundaries).
struct RespTick {
    std::uint32_t wheelContacts;   // wheel contacts incl. rejected side hits (not fake contacts)
    std::uint32_t bodyContacts;
    float bodyNormalSum[3];        // car frame
};

// Branch trace bits (RespOut::arms).
enum : std::uint32_t {
    kRespArmWheel = 1u << 0,        // accepted wheel contact (|n.x| < sin pi/4)
    kRespArmWheelSide = 1u << 1,    // rejected wheel contact (side hit, e = 0.5 at COM height)
    kRespArmBody = 1u << 2,
    kRespArmRoof = 1u << 3,         // body push-out projected on the normal (n.y < -0.75)
    kRespArmBlocked = 1u << 4,      // wheel push-out beyond the suspension travel: full impulse
    kRespArmAbsorbed = 1u << 5,     // vertical approach handed to the suspension
    kRespArmImpulse = 1u << 6,      // at least one impulse applied
    kRespArmTangentCap = 1u << 7,   // body friction cone clamped the tangent speed
    kRespArmGrowthGuard = 1u << 8,  // an impulse zeroed the linear velocity
    kRespArmIgnored = 1u << 9,      // Water / GolfBall contact
    kRespArmTrigger = 1u << 10,
    kRespArmProject = 1u << 11,     // replacement merge removed a forward projection
    kRespArmReplace = 1u << 12,     // position correction applied (> 1 cm)
    kRespArmOverflow = 1u << 13,    // more than kRespMaxContacts contacts: only the first ones were used
};

// Triggers: FV runs no car handler for them (pair {3,1} has no car side;
// CUDA cuda_collision_response.cuh:848-852 `continue`), so they write no car
// state. RespOut::trigger lists them in processing order (= the sort order),
// one entry per merged contact, not de-duplicated; the tick loop maps entry i
// to race::TwTriggerHit {c[trigger[i]].peerId, c[trigger[i]].shape, 0}
// (src/twin/race/tw_race.h, docs/twin/m6a-race.md).
struct RespOut {
    std::uint32_t triggerCount;
    std::uint32_t arms;
    float replacement[3];                       // position delta applied (world)
    std::uint8_t trigger[kRespMaxContacts];     // input indices of trigger contacts, in processing order
};

// Branch trace in the fast kernel (RespOut::arms): on by default (checker,
// stratification); production / bench builds may define TW_RESP_ARMS 0.
#ifndef TW_RESP_ARMS
#define TW_RESP_ARMS 1
#endif
#if TW_RESP_ARMS
#define TW_RESP_ARM(a, bit) ((a) |= (bit))
#else
#define TW_RESP_ARM(a, bit) ((void)0)
#endif

namespace resp_detail {

constexpr std::uint8_t kMatMetal = 4u, kMatWater = 13u, kMatGolfBall = 23u;
constexpr float kSinQuarterPi = 0.70710677f;     // FV CIsinQuarterPi (0x3f3504f3)
constexpr float kEps = 1.0e-5f;                  // ScalarEpsilon
constexpr float kLen2Eps = 1.0e-10f;             // dyna_core kVectorLengthEpsilonSquared
constexpr float kSkin = 0.01f;                   // dyna_core kReplacementDirectionEpsilon
constexpr float kSkin2 = kSkin * kSkin;          // float product, as FV

TW_INLINE float dotYXZ(V3 a, V3 b) { return a.y * b.y + a.x * b.x + a.z * b.z; }
TW_INLINE float len2YXZ(V3 a) { return a.y * a.y + a.x * a.x + a.z * a.z; }

// FV board 20 §3.5 comparator, as "a is processed before b". Distinct keys:
// descending lexicographic on (P, N, Sep). Full tie: non-primary first, then
// (shape, input index) so the order is total and deterministic.
TW_INLINE bool Before(const RespContact &a, int ia, const RespContact &b, int ib) {
    const float *ka[3] = {a.P, a.N, a.Sep}, *kb[3] = {b.P, b.N, b.Sep};
    for (int g = 0; g < 3; ++g)
        for (int i = 0; i < 3; ++i) {
            if (ka[g][i] > kb[g][i]) return true;
            if (ka[g][i] < kb[g][i]) return false;
        }
    const bool pa = (a.flags & kRespPrimary) != 0u, pb = (b.flags & kRespPrimary) != 0u;
    if (pa != pb) return pb;
    if (a.shape != b.shape) return a.shape < b.shape;
    return ia < ib;
}

// Order of N = 2..4 contacts by rank (fully unrolled): rank_i = number of
// contacts processed before i. P.x decides each pair without branching; a pair
// with equal P.x takes the full comparator (FV's key, then non-primary,
// shape, index: a strict total order, so the ranks are a permutation and
// equal the insertion sort's order). NaN keys return false (insertion sort).
template <int N>
TW_INLINE bool SortOrderRank(const RespContact *c, std::uint8_t *order) {
    float kx[N];
    int nan = 0;
    for (int i = 0; i < N; ++i) {
        kx[i] = c[i].P[0];
        nan |= kx[i] != kx[i];
    }
    if (nan) return false;
    int rank[N];
    for (int i = 0; i < N; ++i) rank[i] = 0;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            int jFirst = kx[j] > kx[i];
            if (__builtin_expect(kx[j] == kx[i], 0)) jFirst = Before(c[j], j, c[i], i);
            rank[i] += jFirst;
            rank[j] += 1 - jFirst;
        }
    for (int i = 0; i < N; ++i) order[rank[i]] = static_cast<std::uint8_t>(i);
    return true;
}

// Insertion sort of indices (n is 1-4 in 99.5% of contact substeps). The first
// key (P.x) decides almost always; the full comparator runs only when the two
// P.x are not ordered (equal or NaN), which gives exactly Before's order.
TW_INLINE void SortOrderInsertion(const RespContact *c, int n, std::uint8_t *order) {
    float kx[kRespMaxContacts];
    TW_DEVICE_ROLLED
    for (int i = 0; i < n; ++i) {
        const float xi = c[i].P[0];
        int j = i;
        TW_DEVICE_ROLLED
        while (j > 0) {
            const int o = order[j - 1];
            const float xo = kx[j - 1];
            const bool before = xi > xo || (!(xi < xo) && Before(c[i], i, c[o], o));
            if (!before) break;
            order[j] = order[j - 1];
            kx[j] = xo;
            --j;
        }
        order[j] = static_cast<std::uint8_t>(i);
        kx[j] = xi;
    }
}

// FV CSceneVehicleCar::ComputeAndApplyContactImpulse (local u, n, point).
TW_INLINE void Impulse(const CarForceConsts &k, const M3 &R, float e, V3 u, V3 n, V3 p, V3 &v, V3 &w,
                       std::uint32_t &arms) {
    const V3 J = ComputeImpulseLocal(k, e, u, n, p - v3(k.com));
    ApplyPointImpulseLocal(k, R, J, p, v, w);
    arms |= kRespArmImpulse;
    if (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f) arms |= kRespArmGrowthGuard;
}

// FV WheelAbsorbContact (wheels.cpp:466-525) in wheel force mode 2 (compile refuses others).
TW_INLINE void WheelContact(const CarForceConsts &k, TwinState &s, const M3 &R, const M3 &Rw, int wi,
                            const RespContact &ct, V3 n, V3 p, V3 u, V3 &rho, V3 &v, V3 &w, std::uint32_t &arms) {
    const bool present = tw::math::abs(n.x) < kSinQuarterPi;   // last contact of the wheel wins
    s.contact[wi] = present ? 1u : 0u;
    float *ns = s.normalSum + 3 * wi;
    if (present) {
        ns[0] = ns[0] + n.x;
        ns[1] = ns[1] + n.y;
        ns[2] = ns[2] + n.z;
        s.surface[wi] = ct.material;
    } else {
        s.latContact = 1u;
    }
    // peer axis in the car frame of the WRITE state (substep start); FV reads only .z
    s.peerAxisZ[wi] = dot(Rw.c2, v3(ct.peerZ));
    s.peerId[wi] = ct.peerId;
    store(s.latestPoint + 3 * wi, p);

    // ApplyFollowAbsorbReplacement (:425-464): the suspension eats the vertical push-out
    bool blocked = false;
    if (rho.y > 0.0f) {
        float cand = rho.y;
        if (k.maxAbs >= -kEps) {
            const float travel = s.absorb[wi] - k.maxAbs;
            if (travel <= rho.y) {
                cand = travel;
                blocked = true;
            }
        }
        if (cand > s.maxRepl[wi]) s.maxRepl[wi] = cand;
        rho.y = rho.y - cand;
    } else {
        blocked = true;
    }

    if (present) {   // ApplyFollowAbsorbImpulse (:369-423)
        arms |= kRespArmWheel;
        const float e = -(ct.material == kMatMetal ? k.wheelImpulseMetal : k.wheelImpulseOther);
        const float vn = dotYXZ(n, u);
        if (!(vn < 0.0f)) return;
        const float ny = n.y * vn;
        if (blocked || !(ny < 0.0f)) {
            if (blocked) arms |= kRespArmBlocked;
            Impulse(k, R, e, u, n, p, v, w, arms);
            return;
        }
        const V3 u2 = V3{u.x, u.y - ny, u.z};   // vertical normal approach goes to the spring
        arms |= kRespArmAbsorbed;
        if (dotYXZ(n, u2) < 0.0f) {
            const V3 cur = V3{k.wheel[wi].rest[0], k.wheel[wi].rest[1] - s.absorb[wi], k.wheel[wi].rest[2]};
            Impulse(k, R, e, u2, n, cur, v, w, arms);
        }
        return;
    }
    arms |= kRespArmWheelSide;   // side hit on a wheel: body coefficient, at the COM height
    const float e = -(ct.material == kMatMetal ? k.bodyImpulseMetal : k.bodyImpulseOther);
    if (dotYXZ(n, u) < 0.0f) Impulse(k, R, e, u, n, V3{p.x, k.com[1], p.z}, v, w, arms);
}

// FV AbsorbContact body branch (:616-635) + AbsorbBodyContactWithImpulse (:527-580).
TW_INLINE void BodyContact(const CarForceConsts &k, TwinState &s, RespTick &tick, const M3 &R, const RespContact &ct,
                           V3 n, V3 p, V3 u, V3 &rho, V3 &v, V3 &w, std::uint32_t &arms) {
    arms |= kRespArmBody;
    if (n.y < -0.75f) {   // roof side (GearedDrive): keep only the normal part of the push-out
        rho = n * dotYXZ(n, rho);
        arms |= kRespArmRoof;
    }
    tick.bodyNormalSum[0] = tick.bodyNormalSum[0] + n.x;
    tick.bodyNormalSum[1] = tick.bodyNormalSum[1] + n.y;
    tick.bodyNormalSum[2] = tick.bodyNormalSum[2] + n.z;
    ++tick.bodyContacts;
    s.bodyContact = 1u;
    const float vn = dotYXZ(n, u);
    if (!(vn < 0.0f)) return;
    const bool metal = ct.material == kMatMetal;
    const float e = -(metal ? k.bodyImpulseMetal : k.bodyImpulseOther);
    const float mu = metal ? k.tangentLimitMetal : k.tangentLimitOther;
    const V3 nv = n * vn;
    V3 t = u - nv;
    const float nl = tw::math::sqrt(len2YXZ(nv));
    const float tl = tw::math::sqrt(len2YXZ(t));
    const float tmax = nl * mu;
    if (tl > tmax) {
        t = t * (tmax / tl);
        arms |= kRespArmTangentCap;
    }
    V3 d = -(t + nv);
    const float dl = tw::math::sqrt(len2YXZ(d));
    if (kEps < dl) {
        d = d * (1.0f / dl);
        Impulse(k, R, e, u, d, p, v, w, arms);
    }
}

// One step of FV ComputeSynthetizedReplacement (dyna_core.cpp:640-680), folded in contact order.
TW_INLINE void FoldReplacement(V3 &S, V3 r, std::uint32_t &arms) {
    const float proj = (r.x * S.x + S.y * r.y) + S.z * r.z;
    if (proj > 0.0f) {
        const float l2 = S.z * S.z + (S.y * S.y + S.x * S.x);
        if (kLen2Eps < l2) {
            const float pc = l2 < proj ? l2 : proj;
            S = S - S * (pc / l2);
            arms |= kRespArmProject;
        }
    }
    S = S + r;
}

// ---- fast kernel (same semantics, car-frame v / omega, geometry off the chain) ----
// FV updates the world v, omega after every contact and recomputes the next
// contact's velocity from them, so the contacts of a substep form one serial
// dependency chain (measured: ~120 cycles per contact with the literal form).
// The fast kernel keeps v, omega in the car frame for the whole substep
// (R^T once before, R once after, only if an impulse was applied) and
// precomputes everything that does not depend on v, omega: levers, effective
// mass, angular response per unit impulse. Per link of the chain:
//   un = n.v + (l x n).w      (= n . (v + w x l), the normal speed at the point)
//   s  = un * (e - 1) / K     (FV's impulse magnitude, J = n s)
//   v += n s / m;  w += I^-1 (L x n) s * angularScale   (L = impulse point lever)
// then FV's growth guard (|v|^2 is frame-invariant) and the |w| clamp.
// Rounding differs from FV by a few ulp (v, w round-trip one rotation).
struct LocalBody {
    V3 v, w;      // car frame
    float v2;     // |v|^2 of the current v (FV's "before" of the growth guard)
    bool any;     // an impulse was applied (else the world v, omega stay untouched)
};

TW_INLINE V3 DiagMul(const float *d, V3 a) { return V3{a.x * d[0], a.y * d[1], a.z * d[2]}; }

// FV AddVehicleImpulse after ComputeImpulse, car frame: J = d s at lever L, dw = I^-1 (L x d) s * scale.
// FV scales the WORLD y of dw by angularImpulseYScale (1 for StadiumCar: branch never taken).
TW_INLINE void ApplyLocal(const CarForceConsts &k, const M3 &R, LocalBody &b, V3 d, V3 aL, float s, std::uint32_t &arms) {
    V3 v2 = b.v + d * (s * k.invMass);
    const float a2 = len2(v2);
    if (__builtin_expect(b.v2 < a2 && k.growthLimitSq < a2 - b.v2, 0)) {
        v2 = V3{0.0f, 0.0f, 0.0f};
        b.v2 = 0.0f;
        TW_RESP_ARM(arms, kRespArmGrowthGuard);
    } else {
        b.v2 = a2;
    }
    b.v = v2;
    V3 dw = aL * (s * k.angularImpulseScale);
    if (__builtin_expect(k.angularImpulseYScale != 1.0f, 0)) {
        V3 dww = mul(R, dw);
        dww.y = k.angularImpulseYScale * dww.y;
        dw = mul_t(R, dww);
    }
    V3 w = b.w + dw;
    if (k.maxAngularSpeed > 0.0f) {
        const float l2 = len2(w);
        if (__builtin_expect(l2 > k.maxAngularSpeed * k.maxAngularSpeed, 0)) w = w * (k.maxAngularSpeed / tw::math::sqrt(l2));
    }
    b.w = w;
    b.any = true;
    (void)arms;
    TW_RESP_ARM(arms, kRespArmImpulse);
}

// Impulse magnitude factor for direction d at lever L: (e - 1) / K, K = 1/m + (I^-1 (L x d)).(L x d).
TW_INLINE float ImpulseFactor(const CarForceConsts &k, float e, V3 d, V3 L, V3 *aL) {
    const V3 ld = cross(L, d);
    *aL = DiagMul(k.inertiaInvBody, ld);
    return (e - 1.0f) / (k.invMass + dot(*aL, ld));
}

// Wheel state writes + suspension clamp of the push-out (same as WheelContact). Returns blocked.
TW_INLINE bool WheelState(const CarForceConsts &k, TwinState &s, const M3 &Rw, int wi, const RespContact &ct, V3 n, V3 p,
                          bool present, V3 &rho) {
    s.contact[wi] = present ? 1u : 0u;
    float *ns = s.normalSum + 3 * wi;
    if (present) {
        ns[0] = ns[0] + n.x;
        ns[1] = ns[1] + n.y;
        ns[2] = ns[2] + n.z;
        s.surface[wi] = ct.material;
    } else {
        s.latContact = 1u;
    }
    s.peerAxisZ[wi] = dot(Rw.c2, v3(ct.peerZ));
    s.peerId[wi] = ct.peerId;
    store(s.latestPoint + 3 * wi, p);
    bool blocked = false;
    if (rho.y > 0.0f) {
        float cand = rho.y;
        if (k.maxAbs >= -kEps) {
            const float travel = s.absorb[wi] - k.maxAbs;
            if (travel <= rho.y) {
                cand = travel;
                blocked = true;
            }
        }
        if (cand > s.maxRepl[wi]) s.maxRepl[wi] = cand;
        rho.y = rho.y - cand;
    } else {
        blocked = true;
    }
    return blocked;
}

}  // namespace resp_detail

// Contact processing order (input indices). Exposed for the collision stage
// and the checker; Respond calls it.
TW_INLINE void RespSortOrder(const RespContact *c, int n, std::uint8_t *order) {
    if (n <= 1) {
        if (n == 1) order[0] = 0u;
        return;
    }
    if (n == 2 && resp_detail::SortOrderRank<2>(c, order)) return;
    if (n == 3 && resp_detail::SortOrderRank<3>(c, order)) return;
    if (n == 4 && resp_detail::SortOrderRank<4>(c, order)) return;
    resp_detail::SortOrderInsertion(c, n, order);
}

// Reference kernel: FV's operations one for one (world v, omega updated through
// impulse.h after every contact). Used by the checker to show the fast kernel
// below changes arithmetic only.
TW_INLINE void RespondOrderedRef(const CarForceConsts &k, TwinState &s, const M3 &R, const M3 &Rw,
                                 const RespContact *c, const std::uint8_t *order, int n, RespTick &tick, RespOut &out) {
    using namespace resp_detail;
    std::uint32_t arms = 0u;
    out.triggerCount = 0u;
    const V3 x = v3(s.pos);
    V3 v = v3(s.vel), w = v3(s.angVel);
    const V3 comW = mul(R, v3(k.com)) + x;
    V3 S = V3{0.0f, 0.0f, 0.0f};
    bool haveS = false;
    for (int oi = 0; oi < n; ++oi) {
        const int ci = order[oi];
        const RespContact &ct = c[ci];
        if (ct.flags & kRespTrigger) {
            out.trigger[out.triggerCount++] = static_cast<std::uint8_t>(ci);
            arms |= kRespArmTrigger;
            continue;
        }
        if (ct.material == kMatWater || ct.material == kMatGolfBall) {   // AbsorbContact :583-588
            arms |= kRespArmIgnored;
            continue;
        }
        // SolveImpulse: velocity at the contact point from the current state, car frame
        const V3 P = v3(ct.P);
        const V3 r = P - comW;
        const V3 vp = v + cross(w, r);
        const V3 u = mul_t(R, vp);
        const V3 n = mul_t(R, v3(ct.N));
        const V3 p = mul_t(R, P - x);
        V3 rho = mul_t(R, -v3(ct.Sep));
        s.airRefresh = 1u;
        const int wi = k.wheelOfShape[ct.shape & 7u];
        if (wi >= 0) {
            WheelContact(k, s, R, Rw, wi, ct, n, p, u, rho, v, w, arms);
            ++tick.wheelContacts;
        } else {
            BodyContact(k, s, tick, R, ct, n, p, u, rho, v, w, arms);
        }
        const V3 rw = mul(R, rho);
        if (haveS) {
            FoldReplacement(S, rw, arms);
        } else {
            S = rw;
            haveS = true;
        }
    }
    store(s.vel, v);
    store(s.angVel, w);
    // DoPostCollisionDynamic: leave 1 cm of penetration
    V3 delta = V3{0.0f, 0.0f, 0.0f};
    const float l2 = S.z * S.z + (S.y * S.y + S.x * S.x);
    if (kSkin2 < l2) {
        const float inv = 1.0f / tw::math::sqrt(l2);
        delta = S - (S * inv) * kSkin;
        s.pos[0] = s.pos[0] + delta.x;
        s.pos[1] = delta.y + s.pos[1];
        s.pos[2] = delta.z + s.pos[2];
        arms |= kRespArmReplace;
    }
    store(out.replacement, delta);
    out.arms = arms;
}

// Response with an explicit processing order (n entries of `order`): the fast kernel.
// Besides the car-frame chain (resp_detail::LocalBody), the push-out stays in
// the world frame: a wheel only needs rho.y = -R.c1 . Sep (the suspension
// clamp) and pushes -Sep - R.c1 * cand; a roof contact pushes N (N . -Sep);
// any other body contact pushes -Sep. The merge is rotation invariant, so this
// is FV's merge up to rounding.
TW_INLINE void RespondOrdered(const CarForceConsts &k, TwinState &s, const M3 &R, const M3 &Rw,
                              const RespContact *c, const std::uint8_t *order, int n, RespTick &tick, RespOut &out) {
    using namespace resp_detail;
    std::uint32_t arms = 0u;
    out.triggerCount = 0u;
    const V3 x = v3(s.pos), com = v3(k.com);
    LocalBody b;
    b.v = mul_t(R, v3(s.vel));
    b.w = mul_t(R, v3(s.angVel));
    // the velocities go back to world as v0 + R (b.v - b.v0): the round trip R R^T v is not the identity in float, and
    // FV adds its impulses to the world velocities (twin bridge, docs/twin/bridge.md: the round trip biased the speed)
    const V3 v0w = v3(s.vel), w0w = v3(s.angVel), bv0 = b.v, bw0 = b.w;
    b.v2 = len2(b.v);
    b.any = false;
    V3 S = V3{0.0f, 0.0f, 0.0f};
    bool haveS = false;
    TW_DEVICE_ROLLED
    for (int oi = 0; oi < n; ++oi) {
        const int ci = order[oi];
        const RespContact &ct = c[ci];
        if (ct.flags & kRespTrigger) {
            out.trigger[out.triggerCount++] = static_cast<std::uint8_t>(ci);
            TW_RESP_ARM(arms, kRespArmTrigger);
            continue;
        }
        if (ct.material == kMatWater || ct.material == kMatGolfBall) {
            TW_RESP_ARM(arms, kRespArmIgnored);
            continue;
        }
        // geometry (independent of v, omega)
        const V3 N = v3(ct.N), Sep = v3(ct.Sep);
        const V3 nrm = mul_t(R, N);
        const V3 p = mul_t(R, v3(ct.P) - x);
        const V3 lp = p - com;
        const V3 lnp = cross(lp, nrm);
        V3 rw;   // world push-out of this contact
        s.airRefresh = 1u;
        const int wi = k.wheelOfShape[ct.shape & 7u];
        const bool metal = ct.material == kMatMetal;
        if (wi >= 0) {
            const bool present = tw::math::abs(nrm.x) < kSinQuarterPi;
            s.contact[wi] = present ? 1u : 0u;
            float *ns = s.normalSum + 3 * wi;
            if (present) {
                ns[0] = ns[0] + nrm.x;
                ns[1] = ns[1] + nrm.y;
                ns[2] = ns[2] + nrm.z;
                s.surface[wi] = ct.material;
            } else {
                s.latContact = 1u;
            }
            s.peerAxisZ[wi] = dot(Rw.c2, v3(ct.peerZ));
            s.peerId[wi] = ct.peerId;
            store(s.latestPoint + 3 * wi, p);
            ++tick.wheelContacts;
            // suspension clamp of the vertical push-out (ApplyFollowAbsorbReplacement)
            const float rhoY = -dot(R.c1, Sep);
            bool blocked = false;
            rw = -Sep;
            if (rhoY > 0.0f) {
                float cand = rhoY;
                if (k.maxAbs >= -kEps) {
                    const float travel = s.absorb[wi] - k.maxAbs;
                    if (travel <= rhoY) {
                        cand = travel;
                        blocked = true;
                    }
                }
                if (cand > s.maxRepl[wi]) s.maxRepl[wi] = cand;
                rw = rw - R.c1 * cand;
            } else {
                blocked = true;
            }
            // impulse: lever and factor (chain-independent), then the normal speed (chain)
            V3 aL;
            float f;
            bool full = true;
            if (present) {
                TW_RESP_ARM(arms, kRespArmWheel);
                const float e = -(metal ? k.wheelImpulseMetal : k.wheelImpulseOther);
                full = blocked || !(nrm.y > 0.0f);   // FV: blocked || !(n.y * un < 0) with un < 0
                if (full) {
                    aL = DiagMul(k.inertiaInvBody, lnp);
                    f = (e - 1.0f) / (k.invMass + dot(aL, lnp));
                } else {
                    const V3 cur = V3{k.wheel[wi].rest[0], k.wheel[wi].rest[1] - s.absorb[wi], k.wheel[wi].rest[2]};
                    f = ImpulseFactor(k, e, nrm, cur - com, &aL);
                }
            } else {
                TW_RESP_ARM(arms, kRespArmWheelSide);
                const float e = -(metal ? k.bodyImpulseMetal : k.bodyImpulseOther);
                f = ImpulseFactor(k, e, nrm, V3{lp.x, 0.0f, lp.z}, &aL);   // point (p.x, com.y, p.z)
            }
            const float un = dot(nrm, b.v) + dot(lnp, b.w);
            if (un < 0.0f) {
                if (full) {
                    if (present && blocked) TW_RESP_ARM(arms, kRespArmBlocked);
                    ApplyLocal(k, R, b, nrm, aL, un * f, arms);
                } else {
                    TW_RESP_ARM(arms, kRespArmAbsorbed);   // vertical normal approach goes to the spring
                    const float un2 = un - nrm.y * (nrm.y * un);
                    if (un2 < 0.0f) ApplyLocal(k, R, b, nrm, aL, un2 * f, arms);
                }
            }
        } else {
            TW_RESP_ARM(arms, kRespArmBody);
            if (nrm.y < -0.75f) {   // roof side: keep only the normal part of the push-out
                rw = N * -dot(N, Sep);
                TW_RESP_ARM(arms, kRespArmRoof);
            } else {
                rw = -Sep;
            }
            tick.bodyNormalSum[0] = tick.bodyNormalSum[0] + nrm.x;
            tick.bodyNormalSum[1] = tick.bodyNormalSum[1] + nrm.y;
            tick.bodyNormalSum[2] = tick.bodyNormalSum[2] + nrm.z;
            ++tick.bodyContacts;
            s.bodyContact = 1u;
            const V3 u = b.v + cross(b.w, lp);   // full point velocity: the friction cone needs it
            const float vn = dotYXZ(nrm, u);
            if (vn < 0.0f) {
                const float e = -(metal ? k.bodyImpulseMetal : k.bodyImpulseOther);
                const float mu = metal ? k.tangentLimitMetal : k.tangentLimitOther;
                const V3 nv = nrm * vn;
                V3 t = u - nv;
                const float nl = tw::math::sqrt(len2YXZ(nv));
                const float tl = tw::math::sqrt(len2YXZ(t));
                const float tmax = nl * mu;
                if (tl > tmax) {
                    t = t * (tmax / tl);
                    TW_RESP_ARM(arms, kRespArmTangentCap);
                }
                V3 d = -(t + nv);
                const float dl = tw::math::sqrt(len2YXZ(d));
                if (kEps < dl) {
                    d = d * (1.0f / dl);
                    V3 aL;
                    const float f = ImpulseFactor(k, e, d, lp, &aL);
                    ApplyLocal(k, R, b, d, aL, dot(u, d) * f, arms);
                }
            }
        }
        if (haveS) {
            FoldReplacement(S, rw, arms);
        } else {
            S = rw;
            haveS = true;
        }
    }
    if (b.any) {
        store(s.vel, v0w + mul(R, b.v - bv0));
        store(s.angVel, w0w + mul(R, b.w - bw0));
    }
    V3 delta = V3{0.0f, 0.0f, 0.0f};
    const float l2 = S.z * S.z + (S.y * S.y + S.x * S.x);
    if (kSkin2 < l2) {
#if defined(__CUDA_ARCH__)
        const float inv = tw::math::dev::RsqrtAll(l2);   // the same two roundings without a branch (tw_lanes.h)
#else
        const float inv = 1.0f / tw::math::sqrt(l2);
#endif
        delta = S - (S * inv) * kSkin;
        s.pos[0] = s.pos[0] + delta.x;
        s.pos[1] = delta.y + s.pos[1];
        s.pos[2] = delta.z + s.pos[2];
        TW_RESP_ARM(arms, kRespArmReplace);
    }
    store(out.replacement, delta);
    out.arms = arms;
    (void)arms;
}

// ---------------------------------------------------------------------------
// Single-run form with xyz in one SSE register (RespondOrderedX). It is
// RespondOrdered with every 3-vector held as (x, y, z, 0) in one __m128; each
// component sees exactly the scalar kernel's operation sequence, so the bits
// are identical (twin_resp_check --identity, all records):
//   dot(a, b)   = (a.x b.x + a.y b.y) + a.z b.z   (dotYXZ, len2YXZ and FV's
//                 z + (y + x) orders are the same bits: + is commutative)
//   mul_t(R, v) = per component dot(R.ci, v), done as T0 vx + T1 vy + T2 vz
//                 with T = R^T: same products, same association
//   mul(R, v)   = (R.c0 vx + R.c1 vy) + R.c2 vz
//   cross       = a.yzx b.zxy - a.zxy b.yzx
// No FMA, no rcp/rsqrt; lane 3 is kept at +0 and never read. RespondR uses it
// when the TU has SSE4.1, the scalar RespondOrdered otherwise (same bits).

#if TW_RESP_HAVE_X
namespace resp_x {

struct X3 {
    __m128 v;
};
TW_INLINE X3 load3(const float *p) {
    const __m128 lo = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(p)));   // x y 0 0
    return X3{_mm_movelh_ps(lo, _mm_load_ss(p + 2))};                                      // x y z 0
}
TW_INLINE void store3(float *p, X3 a) {
    _mm_store_sd(reinterpret_cast<double *>(p), _mm_castps_pd(a.v));
    _mm_store_ss(p + 2, _mm_movehl_ps(a.v, a.v));
}
TW_INLINE X3 make3(float x, float y, float z) { return X3{_mm_setr_ps(x, y, z, 0.0f)}; }
TW_INLINE X3 operator+(X3 a, X3 b) { return X3{_mm_add_ps(a.v, b.v)}; }
TW_INLINE X3 operator-(X3 a, X3 b) { return X3{_mm_sub_ps(a.v, b.v)}; }
TW_INLINE X3 operator*(X3 a, X3 b) { return X3{_mm_mul_ps(a.v, b.v)}; }
TW_INLINE X3 operator*(X3 a, float s) { return X3{_mm_mul_ps(a.v, _mm_set1_ps(s))}; }
TW_INLINE X3 operator-(X3 a) { return X3{_mm_xor_ps(a.v, _mm_setr_ps(-0.0f, -0.0f, -0.0f, 0.0f))}; }
TW_INLINE float X(X3 a) { return _mm_cvtss_f32(a.v); }
TW_INLINE float Y(X3 a) { return _mm_cvtss_f32(_mm_movehdup_ps(a.v)); }
TW_INLINE float Z(X3 a) { return _mm_cvtss_f32(_mm_movehl_ps(a.v, a.v)); }
// (m.x + m.y) + m.z of m = a * b
TW_INLINE float dot(X3 a, X3 b) {
    const __m128 m = _mm_mul_ps(a.v, b.v);
    const __m128 t = _mm_add_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehl_ps(m, m)));
}
TW_INLINE float len2(X3 a) { return dot(a, a); }
TW_INLINE X3 cross(X3 a, X3 b) {
    const __m128 a1 = _mm_shuffle_ps(a.v, a.v, _MM_SHUFFLE(3, 0, 2, 1));   // y z x
    const __m128 b2 = _mm_shuffle_ps(b.v, b.v, _MM_SHUFFLE(3, 1, 0, 2));   // z x y
    const __m128 a2 = _mm_shuffle_ps(a.v, a.v, _MM_SHUFFLE(3, 1, 0, 2));
    const __m128 b1 = _mm_shuffle_ps(b.v, b.v, _MM_SHUFFLE(3, 0, 2, 1));
    return X3{_mm_sub_ps(_mm_mul_ps(a1, b2), _mm_mul_ps(a2, b1))};
}
struct XM {   // columns c0 c1 c2 and the transpose t0 t1 t2
    X3 c0, c1, c2, t0, t1, t2;
};
TW_INLINE XM loadM(const M3 &m) {
    XM r;
    r.c0 = make3(m.c0.x, m.c0.y, m.c0.z);
    r.c1 = make3(m.c1.x, m.c1.y, m.c1.z);
    r.c2 = make3(m.c2.x, m.c2.y, m.c2.z);
    r.t0 = make3(m.c0.x, m.c1.x, m.c2.x);
    r.t1 = make3(m.c0.y, m.c1.y, m.c2.y);
    r.t2 = make3(m.c0.z, m.c1.z, m.c2.z);
    return r;
}
TW_INLINE X3 bcast0(X3 a) { return X3{_mm_shuffle_ps(a.v, a.v, _MM_SHUFFLE(0, 0, 0, 0))}; }
TW_INLINE X3 bcast1(X3 a) { return X3{_mm_shuffle_ps(a.v, a.v, _MM_SHUFFLE(1, 1, 1, 1))}; }
TW_INLINE X3 bcast2(X3 a) { return X3{_mm_shuffle_ps(a.v, a.v, _MM_SHUFFLE(2, 2, 2, 2))}; }
TW_INLINE X3 lane3zero(X3 a) { return X3{_mm_blend_ps(a.v, _mm_setzero_ps(), 8)}; }
TW_INLINE X3 mul(const XM &m, X3 v) {    // R v
    return lane3zero((m.c0 * bcast0(v) + m.c1 * bcast1(v)) + m.c2 * bcast2(v));
}
TW_INLINE X3 mul_t(const XM &m, X3 v) {  // R^T v
    return lane3zero((m.t0 * bcast0(v) + m.t1 * bcast1(v)) + m.t2 * bcast2(v));
}

struct LocalBodyX {
    X3 v, w;
    float v2;
    bool any;
};

TW_INLINE void ApplyLocalX(const CarForceConsts &k, const XM &R, LocalBodyX &b, X3 d, X3 aL, float s, std::uint32_t &arms) {
    X3 v2 = b.v + d * (s * k.invMass);
    const float a2 = len2(v2);
    if (__builtin_expect(b.v2 < a2 && k.growthLimitSq < a2 - b.v2, 0)) {
        v2 = X3{_mm_setzero_ps()};
        b.v2 = 0.0f;
        TW_RESP_ARM(arms, kRespArmGrowthGuard);
    } else {
        b.v2 = a2;
    }
    b.v = v2;
    X3 dw = aL * (s * k.angularImpulseScale);
    if (__builtin_expect(k.angularImpulseYScale != 1.0f, 0)) {
        X3 dww = mul(R, dw);
        dww = make3(X(dww), k.angularImpulseYScale * Y(dww), Z(dww));
        dw = mul_t(R, dww);
    }
    X3 w = b.w + dw;
    if (k.maxAngularSpeed > 0.0f) {
        const float l2 = len2(w);
        if (__builtin_expect(l2 > k.maxAngularSpeed * k.maxAngularSpeed, 0))
            w = w * (k.maxAngularSpeed / _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(l2))));
    }
    b.w = w;
    b.any = true;
    TW_RESP_ARM(arms, kRespArmImpulse);
    (void)arms;
}

TW_INLINE float ImpulseFactorX(X3 I3, float invMass, float e, X3 d, X3 L, X3 *aL) {
    const X3 ld = cross(L, d);
    *aL = ld * I3;
    return (e - 1.0f) / (invMass + dot(*aL, ld));
}

TW_INLINE float sqrtx(float a) { return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(a))); }

}  // namespace resp_x

// Same contract and bits as RespondOrdered.
TW_INLINE void RespondOrderedX(const CarForceConsts &k, TwinState &s, const M3 &Rm, const M3 &Rw,
                               const RespContact *c, const std::uint8_t *order, int n, RespTick &tick, RespOut &out) {
    using namespace resp_detail;
    using namespace resp_x;
    std::uint32_t arms = 0u;
    out.triggerCount = 0u;
    const XM R = loadM(Rm);
    const X3 x = load3(s.pos), com = load3(k.com), I3 = load3(k.inertiaInvBody);
    const X3 rwc2 = make3(Rw.c2.x, Rw.c2.y, Rw.c2.z);
    LocalBodyX b;
    b.v = mul_t(R, load3(s.vel));
    b.w = mul_t(R, load3(s.angVel));
    const X3 v0w = load3(s.vel), w0w = load3(s.angVel), bv0 = b.v, bw0 = b.w;   // RespondOrdered's write-back
    b.v2 = len2(b.v);
    b.any = false;
    X3 S = X3{_mm_setzero_ps()};
    bool haveS = false;
    for (int oi = 0; oi < n; ++oi) {
        const int ci = order[oi];
        const RespContact &ct = c[ci];
        if (ct.flags & kRespTrigger) {
            out.trigger[out.triggerCount++] = static_cast<std::uint8_t>(ci);
            TW_RESP_ARM(arms, kRespArmTrigger);
            continue;
        }
        if (ct.material == kMatWater || ct.material == kMatGolfBall) {
            TW_RESP_ARM(arms, kRespArmIgnored);
            continue;
        }
        const X3 N = load3(ct.N), Sep = load3(ct.Sep);
        const X3 nrm = mul_t(R, N);
        const X3 p = mul_t(R, load3(ct.P) - x);
        const X3 lp = p - com;
        const X3 lnp = cross(lp, nrm);
        X3 rw;
        s.airRefresh = 1u;
        const int wi = k.wheelOfShape[ct.shape & 7u];
        const bool metal = ct.material == kMatMetal;
        const float ny = Y(nrm);
        if (wi >= 0) {
            const bool present = tw::math::abs(X(nrm)) < kSinQuarterPi;
            s.contact[wi] = present ? 1u : 0u;
            if (present) {
                float *ns = s.normalSum + 3 * wi;
                store3(ns, load3(ns) + nrm);
                s.surface[wi] = ct.material;
            } else {
                s.latContact = 1u;
            }
            s.peerAxisZ[wi] = dot(rwc2, load3(ct.peerZ));
            s.peerId[wi] = ct.peerId;
            store3(s.latestPoint + 3 * wi, p);
            ++tick.wheelContacts;
            const float rhoY = -dot(R.c1, Sep);
            bool blocked = false;
            rw = -Sep;
            if (rhoY > 0.0f) {
                float cand = rhoY;
                if (k.maxAbs >= -kEps) {
                    const float travel = s.absorb[wi] - k.maxAbs;
                    if (travel <= rhoY) {
                        cand = travel;
                        blocked = true;
                    }
                }
                if (cand > s.maxRepl[wi]) s.maxRepl[wi] = cand;
                rw = rw - R.c1 * cand;
            } else {
                blocked = true;
            }
            X3 aL;
            float f;
            bool full = true;
            if (present) {
                TW_RESP_ARM(arms, kRespArmWheel);
                const float e = -(metal ? k.wheelImpulseMetal : k.wheelImpulseOther);
                full = blocked || !(ny > 0.0f);
                if (full) {
                    aL = lnp * I3;
                    f = (e - 1.0f) / (k.invMass + dot(aL, lnp));
                } else {
                    const X3 cur = make3(k.wheel[wi].rest[0], k.wheel[wi].rest[1] - s.absorb[wi], k.wheel[wi].rest[2]);
                    f = ImpulseFactorX(I3, k.invMass, e, nrm, cur - com, &aL);
                }
            } else {
                TW_RESP_ARM(arms, kRespArmWheelSide);
                const float e = -(metal ? k.bodyImpulseMetal : k.bodyImpulseOther);
                f = ImpulseFactorX(I3, k.invMass, e, nrm, X3{_mm_blend_ps(lp.v, _mm_setzero_ps(), 2 | 8)}, &aL);
            }
            const float un = dot(nrm, b.v) + dot(lnp, b.w);
            if (un < 0.0f) {
                if (full) {
                    if (present && blocked) TW_RESP_ARM(arms, kRespArmBlocked);
                    ApplyLocalX(k, R, b, nrm, aL, un * f, arms);
                } else {
                    TW_RESP_ARM(arms, kRespArmAbsorbed);
                    const float un2 = un - ny * (ny * un);
                    if (un2 < 0.0f) ApplyLocalX(k, R, b, nrm, aL, un2 * f, arms);
                }
            }
        } else {
            TW_RESP_ARM(arms, kRespArmBody);
            if (ny < -0.75f) {
                rw = N * -dot(N, Sep);
                TW_RESP_ARM(arms, kRespArmRoof);
            } else {
                rw = -Sep;
            }
            store3(tick.bodyNormalSum, load3(tick.bodyNormalSum) + nrm);
            ++tick.bodyContacts;
            s.bodyContact = 1u;
            const X3 u = b.v + cross(b.w, lp);
            const float vn = dot(nrm, u);
            if (vn < 0.0f) {
                const float e = -(metal ? k.bodyImpulseMetal : k.bodyImpulseOther);
                const float mu = metal ? k.tangentLimitMetal : k.tangentLimitOther;
                const X3 nv = nrm * vn;
                X3 t = u - nv;
                const float nl = sqrtx(len2(nv));
                const float tl = sqrtx(len2(t));
                const float tmax = nl * mu;
                if (tl > tmax) {
                    t = t * (tmax / tl);
                    TW_RESP_ARM(arms, kRespArmTangentCap);
                }
                X3 d = -(t + nv);
                const float dl = sqrtx(len2(d));
                if (kEps < dl) {
                    d = d * (1.0f / dl);
                    X3 aL;
                    const float f = ImpulseFactorX(I3, k.invMass, e, d, lp, &aL);
                    ApplyLocalX(k, R, b, d, aL, dot(u, d) * f, arms);
                }
            }
        }
        if (haveS) {
            const float proj = dot(rw, S);
            if (proj > 0.0f) {
                const float l2 = len2(S);
                if (kLen2Eps < l2) {
                    const float pc = l2 < proj ? l2 : proj;
                    S = S - S * (pc / l2);
                    TW_RESP_ARM(arms, kRespArmProject);
                }
            }
            S = S + rw;
        } else {
            S = rw;
            haveS = true;
        }
    }
    if (b.any) {
        store3(s.vel, v0w + mul(R, b.v - bv0));
        store3(s.angVel, w0w + mul(R, b.w - bw0));
    }
    X3 delta = X3{_mm_setzero_ps()};
    const float l2 = len2(S);
    if (kSkin2 < l2) {
        const float inv = 1.0f / sqrtx(l2);
        delta = S - (S * inv) * kSkin;
        store3(s.pos, x + delta);
        TW_RESP_ARM(arms, kRespArmReplace);
    }
    store3(out.replacement, delta);
    out.arms = arms;
    (void)arms;
}

#endif  // TW_RESP_HAVE_X

#if defined(__CUDA_ARCH__)
// ---------------------------------------------------------------------------
// The device form (RespondDev): RespSortOrder and RespondOrdered with the same
// operations in the same order (the same bits), their memory accesses grouped.
// On the GPU the contacts and the state live in local memory and ptxas does not
// move a load above a store to a computed address, so the scalar loop waited on
// L1 several times a contact (the order, the contact's words, the shape's
// wheel, then each wheel row after the stores before it). Here:
//   - the order of n <= 4 contacts comes back as nibbles in a register, from one
//     path whose ties go through one copy of Before;
//   - a contact's words are loaded together, at the top of the iteration before its own;
//   - the state's wheel rows the loop reads (absorb, maxRepl, normalSum) are
//     loaded together as soon as the contact's wheel is known, before its geometry;
//   - the tick's accumulators and the state's flags are written once at the end;
//   - one ApplyLocal call site for the contact's impulse, whatever its arm;
//   - the shape -> wheel table is one 64-bit word.
namespace resp_dev {

// SortOrderRank's order for n <= kPackMax contacts on one path (the same comparisons, so the same order), as
// nibbles: contact order[k] in bits 4k..4k+3. P.x decides each pair from registers; the pairs with equal P.x (an edge
// contact found on two triangles: common) go through Before in one rolled loop, so its code (the keys' loads, the
// branches) is there once instead of once a pair and a size. bm bit 4i + j: j is processed before i; rank i =
// popcount of row i. False: a NaN key (insertion sort, as on the CPU). (Up to 8 contacts on this path: the 28
// pairs cost more than the insertion sort they replace.)
constexpr int kPackMax = 4;
TW_INLINE bool SortRankPacked(const RespContact *c, int n, std::uint32_t *ord) {
    float kx[kPackMax];
    int nan = 0;
    for (int i = 0; i < kPackMax; ++i) {
        kx[i] = i < n ? c[i].P[0] : 0.0f;
        nan |= kx[i] != kx[i];
    }
    if (nan) return false;
    std::uint32_t bm = 0u, ties = 0u;
    for (int i = 0; i < kPackMax; ++i)
        for (int j = i + 1; j < kPackMax; ++j) {
            if (j >= n) continue;
            if (kx[j] > kx[i])
                bm |= 1u << (4 * i + j);
            else if (kx[j] < kx[i])
                bm |= 1u << (4 * j + i);
            else
                ties |= 1u << (4 * i + j);
        }
    TW_DEVICE_ROLLED
    while (ties != 0u) {
        const int t = __ffs(ties) - 1;
        ties &= ties - 1u;
        const int i = t >> 2, j = t & 3;
        bm |= resp_detail::Before(c[j], j, c[i], i) ? 1u << (4 * i + j) : 1u << (4 * j + i);
    }
    std::uint32_t o = 0u;
    for (int i = 0; i < kPackMax; ++i)
        if (i < n) o |= static_cast<std::uint32_t>(i) << (4 * __popc(bm >> (4 * i) & 15u));
    *ord = o;
    return true;
}

// A contact's words as the response reads them: the three bytes (shape, material, flags) kept as their word until
// used (a struct copy splits them at once, which waits for the load).
struct Contact {
    float P[3], N[3], Sep[3], peerZ[3];
    std::uint32_t peerId, tail;   // tail: shape | material << 8 | flags << 16 (RespContact's byte order)
    TW_INLINE std::uint32_t shape() const { return tail & 0xFFu; }
    TW_INLINE std::uint32_t material() const { return tail >> 8 & 0xFFu; }
    TW_INLINE std::uint32_t flags() const { return tail >> 16 & 0xFFu; }
};
TW_INLINE Contact LoadContact(const RespContact &c) {
    static_assert(__builtin_offsetof(RespContact, shape) + 1 == __builtin_offsetof(RespContact, material) &&
                      __builtin_offsetof(RespContact, material) + 1 == __builtin_offsetof(RespContact, flags) &&
                      __builtin_offsetof(RespContact, shape) % 4 == 0,
                  "RespContact's bytes: shape, material, flags in one word");
    Contact r;
    for (int i = 0; i < 3; ++i) {
        r.P[i] = c.P[i];
        r.N[i] = c.N[i];
        r.Sep[i] = c.Sep[i];
        r.peerZ[i] = c.peerZ[i];
    }
    r.peerId = c.peerId;
    __builtin_memcpy(&r.tail, &c.shape, 4);
    return r;
}

}  // namespace resp_dev

// RespondOrdered's contract; ord: the order as nibbles (packed true) or `order` (packed false).
TW_INLINE void RespondOrderedDev(const CarForceConsts &k, TwinState &s, const M3 &R, const M3 &Rw, const RespContact *c,
                                 const std::uint8_t *order, std::uint32_t ord, bool packed, int n, RespTick &tick, RespOut &out) {
    using namespace resp_detail;
    std::uint32_t arms = 0u, triggers = 0u;
    const V3 x = v3(s.pos), com = v3(k.com);
    std::uint64_t wheelOf;   // k.wheelOfShape as one word (byte k: shape k)
    static_assert(sizeof k.wheelOfShape == sizeof wheelOf, "wheelOfShape: 8 bytes");
    __builtin_memcpy(&wheelOf, k.wheelOfShape, sizeof wheelOf);
    std::uint32_t wheelContacts = tick.wheelContacts, bodyContacts = tick.bodyContacts;
    float bns0 = tick.bodyNormalSum[0], bns1 = tick.bodyNormalSum[1], bns2 = tick.bodyNormalSum[2];
    bool airRefresh = false, latContact = false, bodyContact = false;
    LocalBody b;
    b.v = mul_t(R, v3(s.vel));
    b.w = mul_t(R, v3(s.angVel));
    const V3 v0w = v3(s.vel), w0w = v3(s.angVel), bv0 = b.v, bw0 = b.w;   // RespondOrdered's write-back
    b.v2 = len2(b.v);
    b.any = false;
    V3 S = V3{0.0f, 0.0f, 0.0f};
    bool haveS = false;
    // the next contact's words are loaded at the top of this one's iteration (before its stores): its latency hides
    // behind this contact's work
    const auto orderAt = [&](int i) { return packed ? static_cast<int>(ord >> (4 * i) & 15u) : static_cast<int>(order[i]); };
    int ciNext = orderAt(0);
    resp_dev::Contact next = resp_dev::LoadContact(c[ciNext]);
    TW_DEVICE_ROLLED
    for (int oi = 0; oi < n; ++oi) {
        const int ci = ciNext;
        const resp_dev::Contact ct = next;
        ciNext = orderAt(oi + 1 < n ? oi + 1 : oi);   // the last one: itself again (no branch)
        next = resp_dev::LoadContact(c[ciNext]);
        if (ct.flags() & kRespTrigger) {
            out.trigger[triggers++] = static_cast<std::uint8_t>(ci);
            TW_RESP_ARM(arms, kRespArmTrigger);
            continue;
        }
        if (ct.material() == kMatWater || ct.material() == kMatGolfBall) {
            TW_RESP_ARM(arms, kRespArmIgnored);
            continue;
        }
        // the wheel's rows first (wheel 0's for a body contact, unused), loaded before this contact's stores (an earlier
        // contact's stores are before them), their latency behind the geometry
        const int wi = static_cast<std::int8_t>(static_cast<std::uint8_t>(wheelOf >> (8u * (ct.shape() & 7u))));
        const int wl = wi >= 0 ? wi : 0;
        float *ns = s.normalSum + 3 * wl;
        const float absorbW = s.absorb[wl], maxReplW = s.maxRepl[wl], ns0 = ns[0], ns1 = ns[1], ns2 = ns[2];
        // geometry (independent of v, omega)
        const V3 N = v3(ct.N), Sep = v3(ct.Sep);
        const V3 nrm = mul_t(R, N);
        const V3 p = mul_t(R, v3(ct.P) - x);
        const V3 lp = p - com;
        const V3 lnp = cross(lp, nrm);
        V3 rw;   // world push-out of this contact
        bool apply = false;   // the contact's impulse (one ApplyLocal call site for every arm: one copy of its code)
        V3 ad, aaL;
        float as = 0.0f;
        airRefresh = true;
        const bool metal = ct.material() == kMatMetal;
        if (wi >= 0) {
            const bool present = tw::math::abs(nrm.x) < kSinQuarterPi;
            s.contact[wi] = present ? 1u : 0u;
            if (present) {
                ns[0] = ns0 + nrm.x;
                ns[1] = ns1 + nrm.y;
                ns[2] = ns2 + nrm.z;
                s.surface[wi] = static_cast<std::uint8_t>(ct.material());
            } else {
                latContact = true;
            }
            s.peerAxisZ[wi] = dot(Rw.c2, v3(ct.peerZ));
            s.peerId[wi] = ct.peerId;
            store(s.latestPoint + 3 * wi, p);
            ++wheelContacts;
            // suspension clamp of the vertical push-out (ApplyFollowAbsorbReplacement)
            const float rhoY = -dot(R.c1, Sep);
            bool blocked = false;
            rw = -Sep;
            if (rhoY > 0.0f) {
                float cand = rhoY;
                if (k.maxAbs >= -kEps) {
                    const float travel = absorbW - k.maxAbs;
                    if (travel <= rhoY) {
                        cand = travel;
                        blocked = true;
                    }
                }
                if (cand > maxReplW) s.maxRepl[wi] = cand;
                rw = rw - R.c1 * cand;
            } else {
                blocked = true;
            }
            // impulse: lever and factor (chain-independent), then the normal speed (chain)
            V3 aL;
            float f;
            bool full = true;
            if (present) {
                TW_RESP_ARM(arms, kRespArmWheel);
                const float e = -(metal ? k.wheelImpulseMetal : k.wheelImpulseOther);
                full = blocked || !(nrm.y > 0.0f);   // FV: blocked || !(n.y * un < 0) with un < 0
                if (full) {
                    aL = DiagMul(k.inertiaInvBody, lnp);
                    f = (e - 1.0f) / (k.invMass + dot(aL, lnp));
                } else {
                    const V3 cur = V3{k.wheel[wi].rest[0], k.wheel[wi].rest[1] - absorbW, k.wheel[wi].rest[2]};
                    f = ImpulseFactor(k, e, nrm, cur - com, &aL);
                }
            } else {
                TW_RESP_ARM(arms, kRespArmWheelSide);
                const float e = -(metal ? k.bodyImpulseMetal : k.bodyImpulseOther);
                f = ImpulseFactor(k, e, nrm, V3{lp.x, 0.0f, lp.z}, &aL);   // point (p.x, com.y, p.z)
            }
            const float un = dot(nrm, b.v) + dot(lnp, b.w);
            if (un < 0.0f) {
                if (full) {
                    if (present && blocked) TW_RESP_ARM(arms, kRespArmBlocked);
                    apply = true;
                    ad = nrm;
                    aaL = aL;
                    as = un * f;
                } else {
                    TW_RESP_ARM(arms, kRespArmAbsorbed);   // vertical normal approach goes to the spring
                    const float un2 = un - nrm.y * (nrm.y * un);
                    if (un2 < 0.0f) {
                        apply = true;
                        ad = nrm;
                        aaL = aL;
                        as = un2 * f;
                    }
                }
            }
        } else {
            TW_RESP_ARM(arms, kRespArmBody);
            if (nrm.y < -0.75f) {   // roof side: keep only the normal part of the push-out
                rw = N * -dot(N, Sep);
                TW_RESP_ARM(arms, kRespArmRoof);
            } else {
                rw = -Sep;
            }
            bns0 = bns0 + nrm.x;
            bns1 = bns1 + nrm.y;
            bns2 = bns2 + nrm.z;
            ++bodyContacts;
            bodyContact = true;
            const V3 u = b.v + cross(b.w, lp);   // full point velocity: the friction cone needs it
            const float vn = dotYXZ(nrm, u);
            if (vn < 0.0f) {
                const float e = -(metal ? k.bodyImpulseMetal : k.bodyImpulseOther);
                const float mu = metal ? k.tangentLimitMetal : k.tangentLimitOther;
                const V3 nv = nrm * vn;
                V3 t = u - nv;
                const float nl = tw::math::sqrt(len2YXZ(nv));
                const float tl = tw::math::sqrt(len2YXZ(t));
                const float tmax = nl * mu;
                if (tl > tmax) {
                    t = t * (tmax / tl);
                    TW_RESP_ARM(arms, kRespArmTangentCap);
                }
                V3 d = -(t + nv);
                const float dl = tw::math::sqrt(len2YXZ(d));
                if (kEps < dl) {
                    d = d * (1.0f / dl);
                    V3 aL;
                    const float f = ImpulseFactor(k, e, d, lp, &aL);
                    apply = true;
                    ad = d;
                    aaL = aL;
                    as = dot(u, d) * f;
                }
            }
        }
        if (apply) ApplyLocal(k, R, b, ad, aaL, as, arms);
        if (haveS) {
            FoldReplacement(S, rw, arms);
        } else {
            S = rw;
            haveS = true;
        }
    }
    out.triggerCount = triggers;
    if (airRefresh) s.airRefresh = 1u;
    if (latContact) s.latContact = 1u;
    if (bodyContact) s.bodyContact = 1u;
    tick.wheelContacts = wheelContacts;
    tick.bodyContacts = bodyContacts;
    tick.bodyNormalSum[0] = bns0;
    tick.bodyNormalSum[1] = bns1;
    tick.bodyNormalSum[2] = bns2;
    if (b.any) {
        store(s.vel, v0w + mul(R, b.v - bv0));
        store(s.angVel, w0w + mul(R, b.w - bw0));
    }
    V3 delta = V3{0.0f, 0.0f, 0.0f};
    const float l2 = S.z * S.z + (S.y * S.y + S.x * S.x);
    if (kSkin2 < l2) {
        const float inv = tw::math::dev::RsqrtAll(l2);   // 1 / sqrt(l2), the same two roundings (tw_lanes.h)
        delta = S - (S * inv) * kSkin;
        s.pos[0] = s.pos[0] + delta.x;
        s.pos[1] = delta.y + s.pos[1];
        s.pos[2] = delta.z + s.pos[2];
        TW_RESP_ARM(arms, kRespArmReplace);
    }
    store(out.replacement, delta);
    out.arms = arms;
    (void)arms;
}

// RespSortOrder + RespondOrderedDev (n in 1..kRespMaxContacts).
TW_INLINE void RespondDev(const CarForceConsts &k, TwinState &s, const M3 &R, const M3 &Rw, const RespContact *c, int n, RespTick &tick,
                          RespOut &out) {
    std::uint8_t order[kRespMaxContacts];
    std::uint32_t ord = 0u;
    const bool packed = n <= resp_dev::kPackMax && resp_dev::SortRankPacked(c, n, &ord);
    if (!packed) resp_detail::SortOrderInsertion(c, n, order);
    RespondOrderedDev(k, s, R, Rw, c, order, ord, packed, n, tick, out);
}
#endif  // __CUDA_ARCH__

// S3 response + S4 position correction for one substep. R: post-integration
// rotation (= R(s.quat)); Rw: substep-start rotation.
TW_INLINE void RespondR(const CarForceConsts &k, TwinState &s, const M3 &R, const M3 &Rw,
                        const RespContact *c, int n, RespTick &tick, RespOut &out) {
    if (n <= 0) {
        out.triggerCount = 0u;
        out.arms = 0u;
        out.replacement[0] = out.replacement[1] = out.replacement[2] = 0.0f;
        return;
    }
    // Contract: n <= kRespMaxContacts (the collision stage caps its list like FV's CUDA backend
    // and reports the overflow; FV's CPU path has no cap, observed max 59).
    const bool overflow = n > kRespMaxContacts;
    if (overflow) n = kRespMaxContacts;
#if defined(__CUDA_ARCH__)
    RespondDev(k, s, R, Rw, c, n, tick, out);   // the device form: the same bits, its loads grouped
#else
    std::uint8_t order[kRespMaxContacts];
    RespSortOrder(c, n, order);
#if TW_RESP_HAVE_X
    RespondOrderedX(k, s, R, Rw, c, order, n, tick, out);
#else
    RespondOrdered(k, s, R, Rw, c, order, n, tick, out);
#endif
#endif
    if (overflow) out.arms |= kRespArmOverflow;
}

TW_INLINE void Respond(const CarForceConsts &k, TwinState &s, const M3 &Rw, const RespContact *c,
                       int n, RespTick &tick, RespOut &out) {
    if (n <= 0) {
        RespondR(k, s, Rw, Rw, c, 0, tick, out);
        return;
    }
    RespondR(k, s, rot_from_quat(s.quat), Rw, c, n, tick, out);
}

// Non-inlined entry with the FP guard (math.md §4).
void RespondGuarded(const CarForceConsts &k, TwinState &s, const M3 &Rw, const RespContact *c,
                    int n, RespTick &tick, RespOut &out);

}  // namespace tw::model

#endif
