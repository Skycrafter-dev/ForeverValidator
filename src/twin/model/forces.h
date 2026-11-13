// Twin StadiumCar force + integration kernel: one substep, one source for every
// lane width (L = 1 on TwinState, L = 4 / 8 / 16 on TwinLanes<L>).
//
// docs/twin/m3-forces.md describes the interface; docs/twin/exits.md lists
// every early exit with its FV source lines. The order of the steps is FV's
// (board 10 §1.3-1.4 with the debate fixes, design.md §5.2):
//
//   S1  gravity * fs(groundedPrev) * m - v * lds(groundedPrev); tau = -0 * omega
//   C1  frozen: v = omega = F = tau = 0, skip C2..C20                 EXIT C1
//   C2  fake contacts (bump impulses)                                  EXIT C2 (texture-less material stops the loop)
//   C3  contact normals, suspension follow-absorb, engine, steer slew  EXIT ENG-BLOCKED
//   C4  grounded -> next fs/lds;  C5 locals;  C6 coasting + lateral window (EXIT C6)
//   C7  speed clamp;  C8 wheel-0 material;  C9 slope A, B;  C10 delta;  C11 localSpeed
//   C12 Model 6: water (EXIT SPLASH) -> donut (EXIT DONUT) | burnout phases ->
//       contact wheels (dirt EXIT DIRT-1/2) -> no ground: TimedSpin -> ExitFade (EXIT NOGROUND) | ground forces
//   C14 air control (EXIT AIR);  C17 turbo / free wheel;  C18 prevForce;  C19-C20 clears
//   S2  integrate (CarIntegrate)
//
// Lanes: every FV branch is a lane mask; an arm runs when any lane takes it
// (`if (any(m))`), and every state write is a select on its mask, so lanes
// that do not take an arm keep their bits. At L = 1 (f32x1) the masks are
// bools, the any() tests are the branches and the selects fold away. Each
// lane of an L-wide call therefore runs the same IEEE operation sequence as
// the L = 1 call on that lane's state: bit-identical (m3-forces.md §7).
//
// Forces and torques from the car are accumulated in the car frame and
// rotated to world once (FV rotates every call: [ARITH], board 10 §2.17.1).
// Velocities stay in the world frame so untouched substeps keep FV's bits.
#ifndef TW_MODEL_FORCES_H
#define TW_MODEL_FORCES_H

#include <cstdint>
#include <cstring>

#if defined(__SSE2__) && !defined(TW_CAR_NO_SSE3V) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#include <immintrin.h>
#define TW_CAR_SSE3V 1   // L = 1: 3-vectors in SSE registers where the operation order allows (same bits)
#else
#define TW_CAR_SSE3V 0
#endif

#include "car_force_consts.h"
#include "forces_io.h"
#include "impulse.h"
#include "lane3.h"
#include "tw_state.h"
#include "vec3.h"

#ifndef TW_CAR_ARMS
#define TW_CAR_ARMS 1
#endif

namespace tw::model {

// Branch trace bits (CarForceOut::arms).
enum : std::uint32_t {
    kArmFrozen = 1u << 0,       // C1
    kArmFakeImpulse = 1u << 1,  // C2 bump impulse applied
    kArmFakeReturn = 1u << 2,   // C2 texture-less material stopped the loop
    kArmEngBlocked = 1u << 3,   // engine airborne / cooldown
    kArmEngTransition = 1u << 4,
    kArmEngBurnoutHold = 1u << 5,
    kArmGearShift = 1u << 6,    // drive curve zeroed by GearShift
    kArmCoast = 1u << 7,        // C6 coasting friction
    kArmLatWindow = 1u << 8,    // C6 lateral slowdown window
    kArmSpeedClamp = 1u << 9,   // C7
    kArmGrounded = 1u << 10,
    kArmWater = 1u << 11,
    kArmSplash = 1u << 12,
    kArmDonut = 1u << 13,       // circular drift substep
    kArmDonutCancel = 1u << 14,
    kArmTimedSpin = 1u << 15,
    kArmExitFade = 1u << 16,
    kArmDirtSlow = 1u << 17,
    kArmDirtSlide = 1u << 18,
    kArmGasBrake = 1u << 19,
    kArmSideSlip = 1u << 20,    // a contact wheel exceeded its grip
    kArmYawSlip = 1u << 21,
    kArmBrakeCap = 1u << 22,
    kArmTurbo = 1u << 23,       // turbo active in the drive force
    kArmTurboEnable = 1u << 24,
    kArmFreewheel = 1u << 25,
    kArmAirMemory = 1u << 26,   // C14 memory branch (incl. air brake)
    kArmAirStrong = 1u << 27,
    kArmAirTorque = 1u << 28,   // airborne quadratic air torque
    kArmReverse = 1u << 29,
    kArmLimiter = 1u << 30,
    kArmBurnoutEnter = 1u << 31,
};

// Lane forms of the per-tick input and the force output.
template <class F> struct TickInT {
    F gas, brake, steer;
    LaneI<F> T;
};
template <class F> struct ForceOutT {
    V3L<F> force, torque, posePos;   // force world, torque car frame
    M3L<F> poseRot;
    LaneI<F> arms;
};

namespace detail {

constexpr float kEps = 1.0e-5f;
constexpr float kEps2 = 1.0e-10f;
constexpr float kGate = 0.1f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kHalfPi = 1.57079632679489661923f;
constexpr std::int32_t kMatMetal = 4, kMatDirt = 6, kMatTurbo = 7, kMatTurbo2 = 26, kMatFreeWheel = 29,
                       kMatRoulette = 30;

using tw::math::cmp_eq;
using tw::math::cmp_ge;
using tw::math::cmp_gt;
using tw::math::cmp_le;
using tw::math::cmp_lt;
using tw::math::cmp_neq;

template <class F> struct Ctx {
    using I = LaneI<F>;
    using M = LaneM<F>;
    M3L<F> R;
    V3L<F> pos;
    V3L<F> vw, ww;      // world velocity / angular velocity (state; impulses and writes go here)
    V3L<F> vl, wl;      // C5 snapshots (car frame)
    V3L<F> Fl;          // S1 force in the car frame ("currentForce")
    V3L<F> Fc, Tc;      // car forces / torques added in C6..C14 (car frame)
    F delta;            // visual steer yaw (front wheel angle)
    F cosD, sinD;
    I T;
    I arms;
    I surfaces;         // bit m set if a contacting wheel is on natural material m
    M grounded;
    M slipFlag;         // a contact wheel slipped this substep (brake cap selector)
    M live;             // active and not frozen: the lanes this call may write
#if defined(__CUDACC__)
    const std::uint8_t *fakeTex;   // GPU: the fake-contact texture (k.fakeTextureRed, or its copy in global memory)
#endif
};

template <class F> TW_INLINE void Arm(Ctx<F> &c, LaneM<F> m, std::uint32_t bit) {
#if TW_CAR_ARMS
    c.arms = c.arms | select(m, LaneI<F>(static_cast<std::int32_t>(bit)), LaneI<F>(0));
#else
    (void)c; (void)m; (void)bit;
#endif
}

// 1 / sqrt(x) (two roundings): on the GPU (f32x1) one branch-free form of the same two operations (tw_lanes.h RsqrtAll:
// the library's reciprocal and square root each branch around a slow-path call).
template <class F> TW_INLINE F InvSqrtL(F x) {
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_same<F, tw::math::f32x1>::value) return F(tw::math::dev::RsqrtAll(x.v));
#endif
    return F(1.0f) / tw::math::sqrt(x);
}
template <class F> TW_INLINE F SgnL(F v) { return select(cmp_lt(v, F(0.0f)), F(-1.0f), F(1.0f)); }   // FV SignNonNegative
template <class F> TW_INLINE LaneM<F> NzI(LaneI<F> v) { return INe(v, LaneI<F>(0)); }

// FV ClampZeroOne (scene_vehicle_car_internal.h:124-134).
template <class F> TW_INLINE F Clamp01L(F v) {
    const auto pos = ~cmp_lt(v, F(0.0f)) & cmp_neq(v, F(0.0f));
    const F c = select(cmp_eq(v, v) & cmp_lt(F(1.0f), v), F(1.0f), v);
    return select(pos, c, F(0.0f));
}

// Small-angle kernels: tw::math::sincos_small / asin_small for finite input,
// the same polynomials and operation order, without the NaN selects (and,
// at L = 1, without the integer-domain sign operations of f32x1).
template <class F> struct SinCosL {
    F sin, cos;
};
template <class F> TW_INLINE SinCosL<F> SinCosSmallL(F x) {   // |x| <= pi/4, finite
    namespace km = tw::math::k;
    const F a = FabsL(x);
    const F z = a * a;
    const F hz = z * F(0.5f);
    const F w = F(1.0f) - hz;
    const F sp = (F(km::kS3) * z + F(km::kS2)) * z + F(km::kS1);
    const F cp = (F(km::kC3) * z + F(km::kC2)) * z + F(km::kC1);
    const F sa = a + (a * z) * sp;
    return {select(cmp_lt(x, F(0.0f)), -sa, sa), w + (((F(1.0f) - w) - hz) + (z * z) * cp)};
}
template <class F> TW_INLINE F AsinSmallL(F x) {      // |x| <= 1/2, finite
    const F z = x * x;
    return x + x * (z * tw::math::detail::asin_poly(z));
}
// sin on [0, pi] and cos on [0, pi/2] by quadrant reflection of the small
// kernel; the reflected argument costs one rounding: <= 2 ulp (m3-forces.md §3).
template <class F> TW_INLINE F SinPiL(F x) {
    x = select(cmp_lt(x, F(0.0f)), F(0.0f), select(cmp_gt(x, F(kPi)), F(kPi), x));
    const auto m1 = cmp_le(x, F(0.78539816f));
    const auto m2 = ~m1 & cmp_le(x, F(2.35619449f));
    const auto m3 = ~m1 & ~m2;
    F r(0.0f);
    if (any(m1)) r = select(m1, SinCosSmallL(x).sin, r);
    if (any(m2)) r = select(m2, SinCosSmallL(x - F(kHalfPi)).cos, r);
    if (any(m3)) r = select(m3, SinCosSmallL(F(kPi) - x).sin, r);
    return r;
}
template <class F> TW_INLINE F CosHalfPiL(F x) {
    x = select(cmp_lt(x, F(0.0f)), F(0.0f), select(cmp_gt(x, F(kHalfPi)), F(kHalfPi), x));
    const auto m1 = cmp_le(x, F(0.78539816f));
    F r(0.0f);
    if (any(m1)) r = select(m1, SinCosSmallL(x).cos, r);
    if (!all(m1)) r = select(m1, r, SinCosSmallL(F(kHalfPi) - x).sin);
    return r;
}

// Curves (car_force_consts.h CompiledCurve): region count, then one formula.
template <class F> TW_INLINE F EvalCurveL(const CompiledCurve &c, F x) {
    LaneI<F> r(0);
    for (std::uint32_t j = 0; j < c.thresholds; ++j) r = r + IOf<F>(cmp_gt(x, F(c.thr[j])));
    return Lookup16(c.y0, r) + Lookup16(c.slope, r) * (x - Lookup16(c.x0, r));
}
TW_INLINE f32x1 EvalCurveL(const CompiledCurve &c, f32x1 x) { return f32x1(EvalCurve(c, x.v)); }
template <class F> TW_INLINE F EvalSpeedCurveL(const CompiledCurve &c, F speed) { return EvalCurveL(c, speed * F(3.6f)); }

template <class F> TW_INLINE V3L<F> ClampAngularL(V3L<F> w, float maxSpeed) {
    if (maxSpeed > 0.0f) {
        const F l2 = len2l(w);
        const auto m = cmp_gt(l2, F(maxSpeed * maxSpeed));
        if (any(m)) w = Sel3(m, w * (F(maxSpeed) / tw::math::sqrt(l2)), w);
    }
    return w;
}

// FV normalises each wheel's normal sum in C3 (wheels.cpp:958-984). The twin
// does not store the normalised sums (they are cleared in C19); the burnout
// paths normalise on demand, the side axis works from the raw sum.
template <class F, class S> TW_INLINE V3L<F> UnitNormalL(const S &s, int w) {
    const V3L<F> n = Ld3At<F>(s.normalSum, 3 * w);
    const F l2 = n.y * n.y + n.x * n.x + n.z * n.z;
    return Sel3(cmp_gt(l2, F(kEps2)), n * InvSqrtL(l2), n);
}

// FV GmVec3::GetAngle (gm_vector.cpp:246-267); a, b unit vectors.
template <class F> TW_INLINE F AngleL(V3L<F> a, V3L<F> b) {
    const F contracted = dotl(a, b) * F(0.99999f);   // FV: double(dot) * (1 - 1e-5), rounded once
    const F angle = tw::math::acos(contracted);
    return select(~cmp_le(angle, F(kEps)) & cmp_lt(crossl(a, b).y, F(0.0f)), -angle, angle);
}

// --- C2 fake contacts (wheels.cpp:652-711) ---------------------------------
// |fmod(x, period)| / period * 128, truncated (wheels.cpp:638-643). period is a
// power of two (compile checks), so x - period * int(x / period) is FV's exact
// fmod and the scale is exact; |x| < 2^31 period.
template <class F> TW_INLINE LaneI<F> FakeTexelL(F coordinate, F period, F invPeriod) {
    const F q = tw::math::cvt_f32(tw::math::cvtt_i32(coordinate * invPeriod));
    const F wrapped = coordinate - period * q;
    const LaneI<F> i = tw::math::cvtt_i32(FabsL(wrapped) * invPeriod * F(128.0f));
    return select(IGt(i, LaneI<F>(127)), LaneI<F>(127), i);
}

template <class F, class S> TW_INLINE void FakeContacts(const CarForceConsts &k, S &s, Ctx<F> &c) {
    using I = LaneI<F>;
    const F forward = dotl(c.R.c2, c.vw);       // read once before the loop, as FV
    auto alive = c.live;
    for (int w = 0; w < 4; ++w) {
        const auto cw = alive & NzI<F>(LdI<F>(s.contact[w]));
        if (!any(cw)) continue;
        const I surf = LdI<F>(s.surface[w]) & I(31);
        const auto uses = cmp_gt(Lookup32(k.byIdFakeUses, surf), F(0.0f));
        const auto stop = cw & ~uses;            // EXIT C2: the whole loop stops
        Arm(c, stop, kArmFakeReturn);
        alive = alive & ~stop;
        const auto m = cw & uses;
        if (!any(m)) continue;
        const V3L<F> rest = v3l<F>(k.wheel[w].rest);
        const V3L<F> p = mull(c.R, rest) + c.pos;
        const I ix = FakeTexelL(p.x, Lookup32(k.byIdFakePeriodX, surf), Lookup32(k.byIdFakeInvX, surf));
        const I iz = FakeTexelL(p.z, Lookup32(k.byIdFakePeriodZ, surf), Lookup32(k.byIdFakeInvZ, surf));
#if defined(__CUDACC__)
        const I pix = LoadBytes<F>(c.fakeTex, select(m, tw::math::shl<7>(iz) + ix, I(0)));
#else
        const I pix = LoadBytes<F>(k.fakeTextureRed, select(m, tw::math::shl<7>(iz) + ix, I(0)));
#endif
        const auto m2 = m & NzI<F>(pix);
        if (!any(m2)) continue;
        F speed = ((tw::math::cvt_f32(pix) / F(255.0f)) * forward) * Lookup32(k.byIdFakeSpeed, surf);
        const F depthMax = Lookup32(k.byIdFakeDepth, surf);
        speed = select(cmp_lt(depthMax, speed), depthMax, speed);
        // WheelAbsorbContact(n = +y, p = rest point, u = (0, -speed, 0), replacement 0):
        // contact stays, normal sum += n, latest point = rest point, replacement blocked.
        StM<F>(s.normalSum[3 * w + 1], m2, Ld<F>(s.normalSum[3 * w + 1]) + F(1.0f));
        St3AtM<F>(s.latestPoint, 3 * w, m2, rest);
        const auto m3 = m2 & cmp_lt(-speed, F(0.0f));
        if (!any(m3)) continue;
        const F e = -select(IEq(surf, I(kMatMetal)), F(k.wheelImpulseMetal), F(k.wheelImpulseOther));
        // ComputeImpulse + AddVehicleImpulse for n = +y in closed form (impulse.h has the general one)
        const F jy = ((e - F(1.0f)) * -speed) / F(k.bumpDenom[w]);
        V3L<F> v2 = c.vw + c.R.c1 * (jy * F(k.invMass));
        const F before = len2l(c.vw), after = len2l(v2);
        v2 = Sel3(cmp_lt(before, after) & cmp_lt(F(k.growthLimitSq), after - before), v3l(F(0.0f), F(0.0f), F(0.0f)), v2);
        c.vw = Sel3(m3, v2, c.vw);
        V3L<F> dww = mull(c.R, v3l<F>(k.bumpW[w]) * jy);
        dww.y = F(k.angularImpulseYScale) * dww.y;
        c.ww = Sel3(m3, ClampAngularL(c.ww + dww, k.maxAngularSpeed), c.ww);
        Arm(c, m3, kArmFakeImpulse);
    }
}

// --- C3 engine (powertrain.cpp:84-311) -------------------------------------
// `em`: lanes that run EngineIntegrate (live, not free wheel).
template <class F, class S> TW_INLINE void Engine(const CarForceConsts &k, S &s, Ctx<F> &c, F input, F h,
                                                  LaneM<F> em) {
    using I = LaneI<F>;
    const EngineConsts &e = k.engine;
    const auto active = cmp_gt(input, F(0.1f));
    F sc = Ld<F>(SF(s.shiftCooldown));
    sc = select(cmp_lt(F(0.0f), sc), sc - h, sc);
    const auto blocked = em & (~c.grounded | cmp_lt(F(0.0f), sc));
    const auto nb = em & ~blocked;
    F rpm = Ld<F>(SF(s.rpm));
    if (any(blocked)) {                                      // EXIT ENG-BLOCKED: no state change
        Arm(c, blocked, kArmEngBlocked);
        rpm = select(blocked, select(active, F(e.airRise) * h + rpm, rpm - F(e.airFall) * h), rpm);
    }
    if (any(nb)) {
        const I phase = LdI<F>(SF(s.brnPhase));
        const auto burnout = IEq(phase, I(kTimedSpin)) | IEq(phase, I(kCircularDrift));
        I gear = LdI<F>(SF(s.gear));
        I es = LdI<F>(SF(s.engineState));
        const auto hold = burnout & NzI<F>(gear);
        es = select(nb & hold, I(kBurnoutHold), select(nb & ~hold & IEq(es, I(kBurnoutHold)), I(kSteady), es));
        const F vz = Ld<F>(s.localSpeedPrev[2]);
        F target(0.0f);
        F slip = Ld<F>(SF(s.slipRpmScale));
        const auto mT = nb & (IEq(es, I(kFwdTransition)) | IEq(es, I(kRevTransition)));
        const auto mH = nb & IEq(es, I(kBurnoutHold));
        const auto mD = nb & ~mT & ~mH;
        if (any(mT)) {
            Arm(c, mT, kArmEngTransition);
            const auto fwd = IEq(es, I(kFwdTransition));
            const auto outF = cmp_lt(F(e.fwdHigh), vz) | cmp_lt(vz, F(e.fwdLow));
            const auto outR = cmp_lt(vz, F(e.revLow)) | cmp_lt(F(e.revHigh), vz);
            const auto outside = (fwd & outF) | (~fwd & outR);
            slip = select(mT, F(1.0f), slip);
            const F tT = FabsL(vz) * select(fwd, F(e.ratio[1]), F(e.ratio[0]));
            target = select(mT, tT, target);
            const auto fall = mT & (outside | ~active);
            const F next = rpm - F(e.groundFall) * h;
            const F rise = F(e.transitionRise) * h + rpm;
            const auto steady = fall & cmp_le(next, tT);
            rpm = select(fall, select(steady, tT, next), select(mT, rise, rpm));
            es = select(steady, I(kSteady), es);
        }
        if (any(mH)) {
            Arm(c, mH, kArmEngBurnoutHold);
            target = select(mH, F(e.rpmMax), target);
            slip = select(mH, F(1.15f), slip);
            const auto up = mH & cmp_lt(rpm, F(e.rpmMax));
            const auto over = mH & ~up & cmp_lt(F(e.rpmMax), rpm);   // FV adds here (clamped below)
            rpm = select(up, F(e.burnoutHoldRise) * h + rpm, select(over, F(e.airFall) * h + rpm, rpm));
        }
        if (any(mD)) {
            const auto slipAct = NzI<F>(LdI<F>(SF(s.slipActive)));
            const F slipN = select(cmp_lt(slip, F(1.15f)), (F(1.15f) - slip) * F(0.3f) * h + slip, F(1.15f));
            slip = select(mD & slipAct & active, slipN, select(mD, F(1.0f), slip));
            const auto rev = NzI<F>(LdI<F>(SF(s.reverse)));
            F tD = FabsL(vz * slip) * Lookup16(e.ratio, gear);
            tD = select((~rev & IEq(gear, I(0))) | (rev & NzI<F>(gear)), F(0.0f), tD);
            target = select(mD, tD, target);
            const auto notBelow = mD & ~cmp_lt(rpm, tD);
            const F fall = select(active, select(~slipAct | burnout, F(e.groundBrake), F(e.airFall)), F(e.groundFall));
            const F n1 = rpm - fall * h;
            const F n2 = F(e.groundRise) * h + rpm;
            es = select((notBelow & cmp_lt(n1, tD)) | (mD & ~notBelow & cmp_lt(tD, n2)), I(kSteady), es);
            rpm = select(notBelow, n1, select(mD, n2, rpm));
        }
        // UpdateTransmissionTransitions (powertrain.cpp:121-193); rpm is the pre-clamp value
        {
            const auto rev = NzI<F>(LdI<F>(SF(s.reverse)));
            const auto fwdM = nb & cmp_lt(F(e.fwdLow), vz) & cmp_lt(vz, F(e.fwdHigh)) & active & ~rev & IEq(es, I(kSteady));
            es = select(fwdM, I(kFwdTransition), es);
            const auto g0 = fwdM & IEq(gear, I(0));
            sc = select(g0, F(0.025f), sc);
            gear = select(g0, I(1), gear);
            const auto revM = nb & ~fwdM & cmp_lt(F(e.revLow), vz) & cmp_lt(vz, F(e.revHigh)) & active & rev &
                              IEq(es, I(kSteady));
            es = select(revM, I(kRevTransition), es);
            const auto gn = revM & NzI<F>(gear);
            gear = select(gn, I(0), gear);
            sc = select(gn, F(0.025f), sc);
            const auto cont = nb & (IEq(es, I(kSteady)) | IEq(es, I(kGearShift)));
            const auto A = cont & ~rev;
            const auto gz = A & IEq(gear, I(0));
            es = select(gz, I(kGearShift), es);
            const auto gzr = gz & cmp_lt(rpm, F(1000.0f));
            sc = select(gzr, F(0.025f), sc);
            gear = select(gzr, I(1), gear);
            const auto gp = A & IGt(gear, I(0));
            if (any(gp)) {
                const auto up = gp & cmp_lt(Lookup16(e.upshift, gear) * F(e.rpmMax), target) & ILt(gear, I(5));
                const auto dn = gp & ~up & cmp_lt(target, Lookup16(e.downshift, gear) * F(e.rpmMax)) & IGt(gear, I(1));
                sc = select(up | dn, F(0.025f), sc);
                gear = select(up, gear + I(1), select(dn, gear - I(1), gear));
                es = select(up | dn, I(kGearShift), es);
            }
            const auto B = cont & rev & NzI<F>(gear);
            es = select(B, I(kGearShift), es);
            const auto br = B & cmp_lt(rpm, F(1000.0f));
            gear = select(br, I(0), gear);
            sc = select(br, select(burnout, F(0.002f), F(0.025f)), sc);
        }
        StIM<F>(SF(s.engineState), nb, es);
        StIM<F>(SF(s.gear), nb, gear);
        StM<F>(SF(s.slipRpmScale), nb, slip);
    }
    // ClampEngineInput
    rpm = select(~cmp_le(rpm, F(0.0f)), select(~cmp_gt(F(e.rpmMax), rpm), F(e.rpmMax), rpm), F(0.0f));
    StM<F>(SF(s.rpm), em, rpm);
    StM<F>(SF(s.shiftCooldown), em, sc);
}

template <class F, class S> TW_INLINE void SteerSlew(const CarForceConsts &k, S &s, Ctx<F> &c, F steer, F h) {
    if (k.slewRate <= 0.0f) {
        StM<F>(SF(s.steerCur), c.live, steer);
        return;
    }
    const F cur = Ld<F>(SF(s.steerCur));
    const F dir = select(cmp_lt(cur - steer, F(0.0f)), F(1.0f), F(-1.0f));
    const F cand = F(k.slewRate) * dir * h + cur;
    const F next = select(cmp_le(steer, cur), select(cmp_gt(steer, cand), steer, cand),
                          select(cmp_lt(steer, cand), steer, cand));
    StM<F>(SF(s.steerCur), c.live, next);
}

// --- C6 coasting friction and lateral slowdown (feedback.cpp:264-353) ---------
template <class F, class S> TW_INLINE void Friction(const CarForceConsts &k, S &s, Ctx<F> &c, const TickInT<F> &in) {
    using I = LaneI<F>;
    // EXIT C6: in water (previous substep) and airborne (also skips the lateral window)
    const auto act = c.live & ~(NzI<F>(LdI<F>(SF(s.inWater))) & ~c.grounded);
    if (!any(act)) return;
    const auto fw = NzI<F>(LdI<F>(SF(s.freewheel)));
    const F pedal = select(NzI<F>(LdI<F>(SF(s.reverse))), in.brake, in.gas);
    const V3L<F> v = c.vl;
    const F l2 = len2l(v);
    const auto coast = act & (cmp_lt(pedal, F(1.0e-5f)) | fw) & cmp_lt(F(kEps2), l2);
    if (any(coast)) {
        Arm(c, coast, kArmCoast);
        V3L<F> f = v * InvSqrtL(l2) * F(-k.lowSpeedFrictionMagnitude);
        f = Sel3(~fw, v * F(-k.lowSpeedLinearDamping) + f, f);
        c.Fc = Sel3(coast, c.Fc + f, c.Fc);
    }
    I lt = LdI<F>(SF(s.latTickMs));
    const auto lc = act & NzI<F>(LdI<F>(SF(s.latContact)));
    lt = select(lc, c.T, lt);
    StIM<F>(SF(s.latTickMs), lc, lt);
    const auto win = act & ULe(lt, c.T) & ULt(c.T - lt, I(static_cast<std::int32_t>(k.lateralSlowDownWindow)));
    if (any(win)) {
        const F sp = tw::math::sqrt(len2l(v));
        const auto m = win & cmp_lt(F(kEps), sp);
        if (any(m)) {
            Arm(c, m, kArmLatWindow);
            const F scale = -EvalSpeedCurveL(k.curve[kCurveLatSlowDown], sp);
            c.Fc = Sel3(m, c.Fc + v * (F(1.0f) / sp) * scale, c.Fc);
        }
    }
}

// --- C9 slope adherence (feedback.cpp:240-262, internal.h:111-122) ------------
template <class F> TW_INLINE F SlopeBlendL(F slope, float lo, float hi) {
    const auto below = ~cmp_le(F(lo), slope);
    const auto above = ~below & ~cmp_ge(F(hi), slope);
    const auto mid = ~below & ~above;
    F r = select(below, F(0.0f), F(1.0f));
    if (any(mid)) r = select(mid, F(1.0f) - CosHalfPiL(((slope - F(lo)) / F(hi - lo)) * F(kHalfPi)), r);
    return r;
}

// --- C12 water (feedback.cpp:488-570) ------------------------------------------
// Returns the in-water lanes (forces applied); a splash is "not in water".
template <class F, class S>
TW_INLINE LaneM<F> Water(const CarForceConsts &k, const WaterZone *zone, S &s, Ctx<F> &c) {
    using M = LaneM<F>;
    if (zone == nullptr || !zone->enabled) return MaskFalse<F>();
    const M3L<F> &R = c.R;
    const V3L<F> center = mull(R, v3l<F>(k.waterBoxCenter)) + c.pos;
    const F hy = FabsL(FabsL(R.c0.y) * F(k.waterBoxHalf[0]) + FabsL(R.c1.y) * F(k.waterBoxHalf[1]) +
                       FabsL(R.c2.y) * F(k.waterBoxHalf[2]));
    const F lowerY = center.y - hy, upperY = center.y + hy;
    // CSceneVehicleWaterZone::AcceptsRegion (collision_zone.cpp:82-101) in lanes; Water = 1
    M acc;
    {
        using I = LaneI<F>;
        const I cx = tw::math::to_u32_trunc_wrap((center.x - F(zone->origin[0])) / F(zone->cellSize[0]));
        const I cy = tw::math::to_u32_trunc_wrap((center.z - F(zone->origin[1])) / F(zone->cellSize[1]));
        const M inside = ULt(cx, I(static_cast<std::int32_t>(zone->dims[0]))) & ULt(cy, I(static_cast<std::int32_t>(zone->dims[1])));
        I cell(zone->outsideValue);
        if (zone->cells != nullptr && any(inside))   // cells are padded by 3 bytes (gather overread)
            cell = select(inside, LoadBytes<F>(zone->cells, select(inside, cx + IMulC(cy, static_cast<std::int32_t>(zone->dims[0])), I(0))),
                          cell);
        const F surf(zone->surfaceHeight);
        const M outsideWater = ~inside & M(zone->outsideValue == 1u ? MaskTrue<F>() : MaskFalse<F>()) & cmp_gt(surf, lowerY);
        const M gate = cmp_gt(upperY, F(zone->secondaryCullHeight)) & cmp_gt(surf, lowerY);
        acc = outsideWater | (~outsideWater & gate & IEq(cell, I(1)));
    }
    const F depth = F(zone->surfaceHeight) - lowerY;
    const M wm = c.live & acc & cmp_gt(depth, F(0.5f));
    if (!any(wm)) return MaskFalse<F>();
    const V3L<F> vw = c.vw;                                  // FV's "localSpeed" is the world velocity
    const F h2 = vw.x * vw.x + vw.z * vw.z;
    const M cond = wm & ~NzI<F>(LdI<F>(SF(s.airRefresh))) & cmp_lt(depth, F(0.9f)) &
                   cmp_lt(F(zone->surfaceHeight) - upperY, F(0.0f)) & cmp_lt(vw.y, F(-1.0e-5f));
    const M hb = cond & cmp_gt(h2, F(k.splashHSpeed * k.splashHSpeed));
    const F uh = -tw::math::sqrt(h2) / vw.y;
    const M nanU = hb & cmp_neq(uh, uh);                     // NaN: not in water, no splash
    const M splash = (hb & ~nanU & ~cmp_lt(uh, F(0.0f))) |
                     (cond & ~hb & cmp_gt(len2l(vw), F(k.splashTotalSpeed * k.splashTotalSpeed)));
    if (any(splash)) {                                       // EXIT SPLASH: "not in water" this substep
        Arm(c, splash, kArmSplash);
        const F u = select(hb, uh, F(0.0f));
        const F cv = EvalCurveL(k.curve[kCurveSplashVertical], u);
        const F ch = EvalCurveL(k.curve[kCurveSplashHorizontal], u);
        const V3L<F> J = v3l(-ch * vw.x, -cv * vw.y, -ch * vw.z);   // world central impulse
        c.vw = Sel3(splash, c.vw + J * F(k.invMass), c.vw);
    }
    const M inW = wm & ~splash & ~nanU;
    if (!any(inW)) return inW;
    Arm(c, inW, kArmWater);
    const F sp = tw::math::sqrt(len2l(c.vl));
    const V3L<F> drag = Sel3(cmp_lt(F(kEps), sp), c.vl * (-EvalSpeedCurveL(k.curve[kCurveWaterFriction], sp)),
                             v3l(F(0.0f), F(0.0f), F(0.0f)));
    const F wlen = tw::math::sqrt(len2l(c.wl));
    const V3L<F> torque = c.wl * F(-k.waterAngLinDamp) + c.wl * (-wlen * F(k.waterAngSpeedDamp));
    const V3L<F> buoyancy = mul_tl(R, v3l(F(0.0f), F(-k.buoyancy), F(0.0f)));
    c.Fc = Sel3(inW, c.Fc + ((buoyancy + drag) - c.Fl), c.Fc);   // cancels gravity and drag
    c.Tc = Sel3(inW, c.Tc + torque, c.Tc);
    return inW;
}

// --- C12 burnouts (handling.cpp:26-460) ---------------------------------------
template <class F> TW_INLINE F FadePhaseL(LaneI<F> elapsed, float duration) {
    return (tw::math::from_u32(elapsed) * F(kPi)) / F(duration);
}

template <class F, class S> TW_INLINE F DriveFadeL(const CarForceConsts &k, const S &s, LaneI<F> T, LaneI<F> phase) {
    using I = LaneI<F>;
    const BurnoutConsts &b = k.burnout;
    const auto ts = IEq(phase, I(kTimedSpin)), ef = IEq(phase, I(kExitFade));
    F r(1.0f);
    if (any(ts))
        r = select(ts, F(b.driveFadeScale - 1.0f) * SinPiL(FadePhaseL<F>(T - LdI<F>(SF(s.brnStartMs)), b.durationF)) + F(1.0f), r);
    if (any(ef))
        r = select(ef, F(b.exitAccelFadeScale - 1.0f) * SinPiL(FadePhaseL<F>(T - LdI<F>(SF(s.brnExitStartMs)), b.exitDurationF)) +
                           F(1.0f), r);
    return r;
}

template <class F, class S> TW_INLINE F SideFadeL(const CarForceConsts &k, const S &s, LaneI<F> T, LaneI<F> phase) {
    const auto ts = IEq(phase, LaneI<F>(kTimedSpin));
    if (!any(ts)) return F(1.0f);
    const BurnoutConsts &b = k.burnout;
    return select(ts, F(b.sideForceFadeScale - 1.0f) * CosHalfPiL(FadePhaseL<F>(T - LdI<F>(SF(s.brnStartMs)), b.duration2F)) + F(1.0f),
                  F(1.0f));
}

template <class F, class S> TW_INLINE F ExitBonusL(const CarForceConsts &k, const S &s, LaneI<F> T, LaneI<F> phase) {
    using I = LaneI<F>;
    const auto ef = IEq(phase, I(kExitFade));
    if (!any(ef)) return F(0.0f);
    const I e = T - LdI<F>(SF(s.brnExitStartMs));
    I q(0);
    const auto big = ef & ~ULt(e, I(static_cast<std::int32_t>(k.burnout.exitDurationTicks)));
    if (any(big)) {                                          // integer division: FV quirk (never in practice)
        alignas(64) std::int32_t ev[F::width], qv[F::width];
        e.store(ev);
        for (int l = 0; l < F::width; ++l)
            qv[l] = static_cast<std::int32_t>(static_cast<std::uint32_t>(ev[l]) / k.burnout.exitDurationTicks);
        q = select(big, I::load(qv), I(0));
    }
    const F d = tw::math::from_u32(q) - F(1.0f);
    return select(ef, d * d * F(k.burnout.exitBonusAccelScale), F(0.0f));
}

// Returns slipSeen (TimedSpin still running) for the lanes in `m`.
template <class F, class S> TW_INLINE LaneM<F> AdvanceBurnoutPhases(const CarForceConsts &k, S &s, LaneI<F> T, LaneM<F> m) {
    using I = LaneI<F>;
    I phase = LdI<F>(SF(s.brnPhase));
    I exitStart = LdI<F>(SF(s.brnExitStartMs));
    const I start = LdI<F>(SF(s.brnStartMs));
    const auto ts = m & IEq(phase, I(kTimedSpin));
    const auto endTs = ts & (ULt(T, start) | ~ULt(T - start, I(static_cast<std::int32_t>(k.burnout.durationTicks))));
    exitStart = select(endTs, T, exitStart);
    phase = select(endTs, I(kExitFade), phase);
    const auto slipSeen = ts & ~endTs;
    const auto ef = m & IEq(phase, I(kExitFade));
    const auto endEf = ef & (ULt(T, exitStart) | ~ULt(T - exitStart, I(static_cast<std::int32_t>(k.burnout.exitDurationTicks))));
    phase = select(endEf, I(kInactive), phase);
    const auto keep = ef & ~endEf;
    if (any(keep))
        for (int w = 0; w < 4; ++w) StIM<F>(s.slipping[w], keep, I(1));
    StIM<F>(SF(s.brnPhase), m, phase);
    StIM<F>(SF(s.brnExitStartMs), m, exitStart);
    return slipSeen;
}

template <class F, class S> TW_INLINE void EnterCircularBurnout(const CarForceConsts &k, S &s, Ctx<F> &c, LaneM<F> m) {
    using I = LaneI<F>;
    I phase = select(m, I(kCircularDrift), LdI<F>(SF(s.brnPhase)));
    const F dir = SgnL(c.delta);
    StIM<F>(SF(s.brnDir), m, select(cmp_lt(dir, F(0.0f)), I(-1), I(1)));
    V3L<F> N = v3l(F(0.0f), F(0.0f), F(0.0f));
    for (int w = 0; w < 4; ++w) N = Sel3(NzI<F>(LdI<F>(s.contact[w])), N + UnitNormalL<F>(s, w), N);
    const F l2 = len2l(N);
    const auto okN = cmp_lt(F(kEps2), l2);
    N = Sel3(okN, N * InvSqrtL(l2), v3l(F(0.0f), F(1.0f), F(0.0f)));
    phase = select(m & ~okN, I(kInactive), phase);
    const V3 seed = v3(k.waterBoxCenter[0] - k.com[0], k.waterBoxCenter[1] - k.com[1],
                       k.waterBoxCenter[2] - k.com[2] + k.waterBoxHalf[2]);
    const F baseR(tw::math::sqrt(len2(seed)));
    StM<F>(SF(s.brnBaseRadius), m, baseR);
    V3L<F> t = crossl(N, c.vl) * dir;
    t = NormalizeOrL(t, t, kEps2);
    V3L<F> Nw = mull(c.R, N);
    const auto lowN = cmp_lt(Nw.y, F(0.75f));
    phase = select(m & lowN, I(kInactive), phase);
    Nw = Sel3(lowN, v3l(F(0.0f), F(1.0f), F(0.0f)), Nw);
    St3M<F>(s.brnNormal, m, Nw);
    const F signedAngle = AngleL(v3l(F(0.0f), F(0.0f), F(1.0f)), t) * SgnL(c.delta);
    const auto bad = cmp_gt(signedAngle, F(k.burnout.angleLimitPositive)) | cmp_lt(signedAngle, F(-k.burnout.angleLimitNegative));
    phase = select(m & bad, I(kInactive), phase);
    const auto good = m & ~bad;
    if (any(good)) {
        const V3L<F> tW = mull(c.R, t);
        const V3L<F> comW = mull(c.R, v3l<F>(k.com)) + c.pos;
        const F target = EvalSpeedCurveL(k.curve[kCurveBurnoutRadius], c.vl.z) + baseR;
        StM<F>(SF(s.brnTargetRadius), good, target);
        St3M<F>(s.brnCenter, good, comW + tW * target);
    }
    StIM<F>(SF(s.brnPhase), m, phase);
}

// Suspension force (0, Fy, 0) at the force point: torque (lever x f) = (-l.z Fy, 0, l.x Fy).
template <class F, class S> TW_INLINE void Suspension(const CarForceConsts &k, const S &s, Ctx<F> &c, int w, LaneM<F> m) {
    const F fy = (F(k.restAbsorb) - Ld<F>(s.absorb[w])) * F(k.springK) - F(k.damperC) * Ld<F>(s.absorbVel[w]);
    c.Fc.y = select(m, c.Fc.y + fy, c.Fc.y);
    c.Tc.x = select(m, c.Tc.x + F(-k.wheel[w].forceLever[2]) * fy, c.Tc.x);
    c.Tc.z = select(m, c.Tc.z + F(k.wheel[w].forceLever[0]) * fy, c.Tc.z);
}

// Circular drift substep (handling.cpp:304-460) for the lanes in `dn`;
// leaves brnPhase != kCircularDrift on cancel.
template <class F, class S>
TW_INLINE void DonutForces(const CarForceConsts &k, S &s, Ctx<F> &c, const TickInT<F> &in, LaneM<F> water, LaneM<F> dn) {
    using I = LaneI<F>;
    const BurnoutConsts &b = k.burnout;
    I phase = LdI<F>(SF(s.brnPhase));
    const F dirF = tw::math::cvt_f32(LdI<F>(SF(s.brnDir)));
    const auto cancel1 = dn & (cmp_lt(in.gas, F(kGate)) | cmp_lt(in.brake, F(kGate)) | cmp_lt(FabsL(c.delta), F(kEps)) |
                               cmp_neq(dirF, SgnL(c.delta)) | NzI<F>(LdI<F>(SF(s.latContact))) |
                               NzI<F>(LdI<F>(SF(s.bodyContact))) | ~c.grounded | water |
                               NzI<F>(LdI<F>(SF(s.freewheel))));
    phase = select(cancel1, I(kInactive), phase);
    auto act = dn & ~cancel1;
    if (any(act)) {
        V3L<F> N = v3l(F(0.0f), F(0.0f), F(0.0f));
        for (int w = 0; w < 4; ++w) N = N + UnitNormalL<F>(s, w);
        const V3L<F> avg = NormalizeOrL(N, v3l(F(0.0f), F(1.0f), F(0.0f)), kEps2);
        const V3L<F> Nw = mull(c.R, avg);
        const auto cancel2 = act & cmp_gt(FabsL(AngleL(Nw, Ld3<F>(s.brnNormal))), F(b.angleLimit));
        phase = select(cancel2, I(kInactive), phase);
        act = act & ~cancel2;
        if (any(act)) {
            const V3L<F> com = v3l<F>(k.com);
            const V3L<F> center = Ld3<F>(s.brnCenter);
#ifdef TW_CAR_SUBTRACT_FIRST
            const V3L<F> radial = mul_tl(c.R, center - c.pos) - com;   // subtract first: exact at map scale
#else
            // FV's InverseTransformPoint: R^T c + R^T (-p), rounded at map scale. The cancellation error
            // (~1e-4 m at 800 m) is amplified by exp() in the radius correction; FV's form is the
            // behaviour (m3-forces.md §4.2).
            const V3L<F> radial = (mul_tl(c.R, center) + mul_tl(c.R, -c.pos)) - com;
#endif
            const F r2 = len2l(radial);
            const F radius = tw::math::sqrt(r2);
            const V3L<F> rdir = Sel3(cmp_gt(r2, F(kEps2)), radial * (F(1.0f) / radius), radial);
            const V3L<F> tangent = crossl(avg, rdir);
            const F ut = dotl(c.vl, tangent);
            const F ur = dotl(c.vl, rdir);
            phase = select(act & cmp_gt(FabsL(ut), F(b.tangentSpeedMax)), I(kInactive), phase);   // forces still applied
            const F baseR = Ld<F>(SF(s.brnBaseRadius));
            const auto radOk = cmp_gt(radius, baseR) | cmp_lt(radius, F(b.radiusMin));
            phase = select(act & ~radOk, I(kInactive), phase);
            const auto corr = act & radOk;
            if (any(corr)) {
                const F ex = (radius - Ld<F>(SF(s.brnTargetRadius))) * F(b.radiusCorrectionScale) -
                             ur * F(b.radiusCorrectionSpeedScale);
                c.Fc = Sel3(corr, c.Fc + rdir * (((ut * ut) / radius) * tw::math::exp(ex)), c.Fc);
            }
            for (int w = 0; w < 4; ++w) {
                Suspension(k, s, c, w, act & NzI<F>(LdI<F>(s.contact[w])));
                StIM<F>(s.slipping[w], act, I(0));
            }
            const F latSpeed = EvalCurveL(k.curve[kCurveBurnoutLatSpeed], radius) / F(3.6f);
            c.Fc = Sel3(act, c.Fc + tangent * (F(b.lateralCorrectionScale) * (-SgnL(c.delta) * latSpeed - ut)), c.Fc);
            const F angle = AngleL(v3l(F(0.0f), F(0.0f), F(1.0f)), rdir);
            const F an = angle / F(kPi);
            const F signedAngle = dirF * an * F(kPi);
            const auto finite = cmp_eq(an - an, F(0.0f));    // false for NaN and +-inf
            const auto badA = ~finite | cmp_lt(signedAngle, F(-b.angleLimitNegative)) | cmp_gt(signedAngle, F(b.angleLimitPositive));
            phase = select(act & badA, I(kInactive), phase);
            const auto okA = act & ~badA;
            if (any(okA)) {
                const F wy = c.wl.y;
                const auto neg = cmp_le(an, F(0.0f));
                const F retNeg = select(~cmp_gt(wy, F(0.0f)), F(-b.angleReturnQuadratic) * wy * ((an + F(1.0f)) * (an + F(1.0f))),
                                        F(-b.angularDampingLinear) * wy);
                const F retPos = select(~cmp_lt(wy, F(0.0f)), F(-b.angleReturnQuadratic) * wy * ((an - F(1.0f)) * (an - F(1.0f))),
                                        F(-b.angularDampingLinear) * wy);
                const F ret = select(neg, retNeg, retPos);
                const F tw_ = ut / radius;
                const auto dm = (neg & cmp_gt(tw_, F(0.0f))) | (~neg & cmp_lt(tw_, F(0.0f)));
                const F damp = select(dm, F(-b.tangentAngularDamping) * tw_, F(0.0f));
                c.Tc.y = select(okA, c.Tc.y + (F(b.angleTorqueScale) * an + ret + damp), c.Tc.y);
            }
            c.Tc.z = select(act, c.Tc.z + EvalSpeedCurveL(k.curve[kCurveDonutRollover], FabsL(c.vl.x)) * -SgnL(c.vl.x), c.Tc.z);
            c.Tc.x = select(act, c.Tc.x + EvalSpeedCurveL(k.curve[kCurveBurnoutRollover], c.vl.z), c.Tc.x);
        }
    }
    StIM<F>(SF(s.brnPhase), dn, phase);
}

// --- C12 contact wheels (legacy_models.cpp:844-990) ------------------------
// FV calls DirtSlide once per contacting wheel w, inside the wheel loop, and each
// call pushes every wheel i that is slipping at that moment (i <= w: this
// substep's flag, i > w: the previous one). The forces of one substep do not
// depend on w, so the four calls are one sum: wheel i is pushed
// m_i = new_i (4 - i) + old_i i times, the slowdown 4 times ([ARITH] n f for
// f + ... + f). Dirt needs all four wheels in contact.
template <class F, class S>
TW_INLINE void DirtSlideAll(const CarForceConsts &k, const S &s, Ctx<F> &c, const TickInT<F> &in, const LaneI<F> *oldSlip,
                            LaneM<F> m) {
    using I = LaneI<F>;
    const V3L<F> v = c.vl;
    const auto m1 = m & ~cmp_le(v.z, F(6.0f));              // EXIT DIRT-1
    if (!any(m1)) return;
    const auto slow = m1 & cmp_gt(in.brake, F(kGate));
    if (any(slow)) {
        Arm(c, slow, kArmDirtSlow);
        c.Fc = Sel3(slow, c.Fc + v * F(-0.1f) * F(4.0f), c.Fc);
    }
    const auto m2 = m1 & ~(NzI<F>(LdI<F>(SF(s.freewheel))) | cmp_le(in.gas, F(kGate)) | ~cmp_le(in.brake, F(kGate)));   // EXIT DIRT-2
    if (!any(m2)) return;
    Arm(c, m2, kArmDirtSlide);
    const V3L<F> u = NormalizeOrL(v, v3l(F(0.0f), F(0.0f), F(1.0f)), kEps2);
    const F ax = FabsL(u.x);
    const F gx = ax * F(20.0f) + F(1.0f);
    const F db = FabsL(v.z) + F(1.0f);
    const F denom = db * db;
    const F side = (F(k.dirtSideScale) * u.x) / (F(k.dirtGateScale) * in.gas + F(1.0f));
    const F fz = F(k.dirtFwdGateScale) * in.gas * F(1.5f) * ax * gx * F(k.dirtFwdScale) / denom;
    const F rz = F(k.dirtFwdGateScale) * in.gas * ax * gx * F(k.dirtFwdScale) / denom;
    for (int i = 0; i < 4; ++i) {
        const I mi = select(NzI<F>(LdI<F>(s.slipping[i])), I(4 - i), I(0)) + select(NzI<F>(oldSlip[i]), I(i), I(0));
        const auto on = m2 & NzI<F>(mi);
        if (!any(on)) continue;
        const F mf = tw::math::cvt_f32(mi);
        const V3L<F> f = i <= 1 ? v3l(side, F(0.0f), fz) : v3l(-side, F(0.0f), rz);
        c.Fc = Sel3(on, c.Fc + f * mf, c.Fc);
        c.Tc = Sel3(on, c.Tc + crossl(v3l<F>(k.wheel[i].forceLever), f) * mf, c.Tc);
    }
}

template <class F, class S>
TW_INLINE void ContactWheel(const CarForceConsts &k, S &s, Ctx<F> &c, const TickInT<F> &in, int w, F slopeA, F sideFade,
                            F maxSideCurve, LaneM<F> cw) {
    using I = LaneI<F>;
    const I surf = LdI<F>(s.surface[w]) & I(31);
    // side axis (n.y, -n.x, 0) of the unit normal, normalised (else +x): from the raw sum,
    // |n|^2 * eps^2 is FV's threshold on the unit normal's components
    const V3L<F> n = Ld3At<F>(s.normalSum, 3 * w);
    const F s2 = n.y * n.y + n.x * n.x;
    const F l2 = s2 + n.z * n.z;
    const auto big = cmp_gt(l2, F(kEps2));
    const auto ok = (big & cmp_gt(s2, F(kEps2) * l2)) | (~big & cmp_gt(s2, F(kEps2)));
    const F inv = InvSqrtL(s2);
    V3L<F> side = v3l(select(ok, n.y * inv, F(1.0f)), select(ok, -n.x * inv, F(0.0f)), F(0.0f));
    if (k.wheel[w].front) side = v3l(c.cosD * side.x, c.cosD * side.y, -c.sinD);
    const F dmIn = k.damperModRange == 0.0f ? F(0.0f) : (Ld<F>(s.absorb[w]) - F(k.minAbs)) / F(k.damperModRange);
    const F damperMod = EvalCurveL(k.curve[kCurveDamperModulation], dmIn);
    const auto slip = NzI<F>(LdI<F>(s.slipping[w]));
    const F grip = Lookup32(k.byIdBlend[3], surf) * slopeA * maxSideCurve *
                   select(slip, F(k.slippingSideFrictionScale), F(1.0f)) *
                   select(slip & cmp_gt(in.brake, F(kGate)), F(k.lowSpeedBSlippingGripScale), F(1.0f)) * damperMod;
    F req = F(-k.lateralForceScale * 0.5f) * dotl(c.vl, side) * sideFade;
    const auto slipNow = cmp_lt(grip, FabsL(req));
    req = select(slipNow, F(1.0f - k.sideFrictionSlipBlend) * (SgnL(req) * grip) + F(k.sideFrictionSlipBlend) * req, req);
    StIM<F>(s.slipping[w], cw, IOf<F>(slipNow));
    const auto sl = cw & slipNow;
    c.slipFlag = c.slipFlag | sl;
    Arm(c, sl, kArmSideSlip);
    c.Fc = Sel3(cw, c.Fc + side * req, c.Fc);
}

// All contact wheels (legacy_models.cpp:985-989) for the lanes in `g`. The
// per-wheel load-transfer torques -(lever_w x Fp) share Fp, so their sum is
// one cross product of the summed levers (linear; [ARITH] reassociation).
template <class F, class S>
TW_INLINE void ContactWheels(const CarForceConsts &k, S &s, Ctx<F> &c, const TickInT<F> &in, F slopeA, LaneM<F> dirt,
                             F maxSideCurve, LaneM<F> g) {
    using I = LaneI<F>;
    const I phase = LdI<F>(SF(s.brnPhase));
    const F sideFade = SideFadeL<F>(k, s, c.T, phase);
    if (!(k.lateralForceScale >= 0.0f)) {   // ApplyModel6ContactWheel returns at once (LM:889-892); not StadiumCar
        for (int w = 0; w < 4; ++w) Suspension(k, s, c, w, g & NzI<F>(LdI<F>(s.contact[w])));
        return;
    }
    V3L<F> leverSum = v3l(F(0.0f), F(0.0f), F(0.0f));
    F count(0.0f);
    I oldSlip[4];
    for (int w = 0; w < 4; ++w) oldSlip[w] = LdI<F>(s.slipping[w]);
    for (int w = 0; w < 4; ++w) {
        const auto cw = g & NzI<F>(LdI<F>(s.contact[w]));
        if (!any(cw)) continue;
        Suspension(k, s, c, w, cw);
        leverSum = Sel3(cw, leverSum + Ld3At<F>(s.latestPoint, 3 * w), leverSum);
        count = select(cw, count + F(1.0f), count);
        ContactWheel(k, s, c, in, w, slopeA, sideFade, maxSideCurve, cw);
    }
    const auto dm = g & dirt;
    if (any(dm)) DirtSlideAll(k, s, c, in, oldSlip, dm);
    V3L<F> Fp = Ld3<F>(s.prevForce) * F(-k.forceDivisor);
    Fp = Sel3(cmp_lt(len2l(Fp), F(k.currentForceTorqueMin * k.currentForceTorqueMin)), v3l(F(0.0f), F(0.0f), F(0.0f)), Fp);
    const V3L<F> lt = crossl(leverSum - v3l<F>(k.com) * count, Fp);
    c.Tc.x = select(g, c.Tc.x + -lt.x * F(k.currentTorqueXScale), c.Tc.x);
    c.Tc.z = select(g, c.Tc.z + -lt.z * F(k.currentTorqueZScale), c.Tc.z);
    const auto ts = g & IEq(phase, I(kTimedSpin));
    if (any(ts)) c.Tc.x = select(ts, c.Tc.x + count * EvalSpeedCurveL(k.curve[kCurveBurnoutRollover], c.vl.z), c.Tc.x);
}

// --- C12 ground forces (legacy_models.cpp:993-1070, handling.cpp:156-662) ----
template <class F, class S>
TW_INLINE void GroundForces(const CarForceConsts &k, S &s, Ctx<F> &c, const TickInT<F> &in, LaneM<F> water,
                            LaneM<F> slipSeen, F slopeB, const F *mat, F maxSideCurve, LaneM<F> g) {
    using I = LaneI<F>;
    const V3L<F> v = c.vl;
    const I T = c.T;
    const auto fw = NzI<F>(LdI<F>(SF(s.freewheel)));
    I phase = LdI<F>(SF(s.brnPhase));
    const auto toExit = g & ~fw & cmp_gt(in.brake, F(kGate)) & cmp_lt(in.gas, F(kGate)) & IEq(phase, I(kTimedSpin));
    StIM<F>(SF(s.brnExitStartMs), toExit, T);
    phase = select(toExit, I(kExitFade), phase);
    // TryEnterForwardBurnout (handling.cpp:262-285)
    {
        const auto m0 = g & ~fw & cmp_gt(in.gas, F(kGate)) & cmp_gt(in.brake, F(kGate));
        if (any(m0)) {
            const auto tsE = m0 & cmp_lt(v.z, F(k.burnout.donutSpeedLow)) & cmp_gt(c.R.c1.y, F(0.75f));
            Arm(c, tsE, kArmBurnoutEnter);
            phase = select(tsE, I(kTimedSpin), phase);
            StIM<F>(SF(s.brnStartMs), tsE, T);
            const auto ciE = m0 & cmp_lt(v.z, F(k.burnout.donutSpeedHigh)) & cmp_gt(v.z, F(k.burnout.donutSpeedLow)) &
                             ~cmp_lt(FabsL(c.delta), F(kEps));
            StIM<F>(SF(s.brnPhase), g, phase);
            if (any(ciE)) {
                Arm(c, ciE, kArmBurnoutEnter);
                EnterCircularBurnout(k, s, c, ciE);
                phase = LdI<F>(SF(s.brnPhase));
            }
        }
    }
    StIM<F>(SF(s.brnPhase), g, phase);
    // UpdateGearDirection (handling.cpp:156-182)
    auto rev = NzI<F>(LdI<F>(SF(s.reverse)));
    {
        rev = rev | (cmp_gt(in.brake, F(kGate)) & cmp_lt(v.z, F(k.reverseGearSpeedThreshold)) & cmp_lt(FabsL(v.x), F(2.0f)));
        rev = rev & ~(cmp_gt(in.gas, F(kGate)) & (cmp_gt(v.z, F(0.0f)) | cmp_gt(FabsL(v.x), F(2.0f))));
        const auto none = cmp_lt(in.gas, F(kGate)) & cmp_lt(in.brake, F(kGate));
        const auto noneRev = ~(cmp_gt(v.z, F(0.0f)) | cmp_lt(FabsL(v.z), F(2.0f)));
        rev = (none & noneRev) | (~none & rev);
        rev = rev & ~(cmp_gt(v.z, F(0.0f)) & NzI<F>(LdI<F>(SF(s.trbType))));
        rev = rev & IEq(phase, I(kInactive));
        StIM<F>(SF(s.reverse), g, IOf<F>(rev));
        Arm(c, g & rev, kArmReverse);
    }
    // ground rollover
    const F rollIn = (v.x * v.x) / (FabsL(v.z) + F(1.0f));
    c.Tc.z = select(g, c.Tc.z + -SgnL(v.x) * EvalSpeedCurveL(k.curve[kCurveRollLatRatio], rollIn), c.Tc.z);
    // steer assist ramp
    const F speed = tw::math::sqrt(len2l(v));
    const auto r0 = cmp_lt(speed, F(0.7f));
    const auto r1 = ~r0 & cmp_gt(speed, F(k.assistFullSpeed));
    const auto rm = g & ~r0 & ~r1;
    F ramp = select(r0, F(0.0f), F(1.0f));
    if (any(rm)) ramp = select(rm, SinPiL((speed * F(k.invAssistFullSpeed)) * F(kHalfPi)), ramp);
    // yaw torque, all four wheels
    const F maxSide = maxSideCurve * mat[3];
    const F steerTorque = EvalSpeedCurveL(k.curve[kCurveSteerDriveTorque], v.z);
    F limit(0.0f), requested(0.0f), yaw(0.0f);
    // the side speed v.x + omega.y * l depends on the axle only: one request per axle
    F rAxle[2], tAxle[2];
    LaneM<F> clipAxle[2];
    for (int a = 0; a < 2; ++a) {
        const float l = a == 0 ? k.halfSpanFront : k.halfSpanRear;
        F r = F(-k.lateralForceScale * 0.5f) * (v.x + c.wl.y * F(l));
        const F ar = FabsL(r);
        clipAxle[a] = cmp_gt(ar, maxSide);
        r = select(clipAxle[a], SgnL(r) * (F(1.0f - k.driveSideFrictionSlipBlend) * maxSide + F(k.driveSideFrictionSlipBlend) * ar), r);
        rAxle[a] = ar;
        tAxle[a] = F(k.sideToDriveTorque) * r;
    }
    for (int w = 0; w < 4; ++w) {
        const int a = k.wheel[w].front ? 0 : 1;
        limit = select(clipAxle[a], limit + maxSide, limit);
        requested = select(clipAxle[a], requested + rAxle[a], requested);
        slipSeen = slipSeen | clipAxle[a];
        Arm(c, g & clipAxle[a], kArmYawSlip);
        F t = tAxle[a];
        if (a == 0) {
            F assist = ramp * Ld<F>(SF(s.steerCur)) * steerTorque;
            assist = select(rev, -assist, assist);
            assist = select(NzI<F>(LdI<F>(s.slipping[w])), assist * F(k.slippingSteerTorqueScale), assist);
            t = t - assist;
        }
        yaw = yaw + t * F(k.wheel[w].halfSpanSigned);
    }
    c.Tc.y = select(g, c.Tc.y + yaw, c.Tc.y);
    // slip memory and accel blend
    I slipLast = select(g & slipSeen, T, LdI<F>(SF(s.slipLastTickMs)));
    StIM<F>(SF(s.slipLastTickMs), g, slipLast);
    const auto mixM = IEq(T, slipLast) & cmp_gt(limit, F(kEps));
    F mix(1.0f);
    if (any(g & mixM))
        mix = select(mixM, F(1.0f) - Clamp01L((requested - limit) / limit / F(k.slipRatioScale)), F(1.0f));
    // drive force
    const F slipAcc = F(k.slippingAccelScale) * EvalSpeedCurveL(k.curve[kCurveSlippingAccel], v.z);
    F gearAcc;
    if (all(rev | ~g)) gearAcc = EvalSpeedCurveL(k.curve[kCurveRearGearAccel], v.z);
    else if (!any(rev & g)) gearAcc = EvalSpeedCurveL(k.curve[kCurveAccel], v.z);
    else gearAcc = select(rev, EvalSpeedCurveL(k.curve[kCurveRearGearAccel], v.z), EvalSpeedCurveL(k.curve[kCurveAccel], v.z));
    const auto gs = IEq(LdI<F>(SF(s.engineState)), I(kGearShift));
    Arm(c, g & gs, kArmGearShift);
    const F blend = select(gs, F(0.0f), (F(1.0f) - mix) * slipAcc + gearAcc * mix);
    const auto turbo = NzI<F>(LdI<F>(SF(s.trbType)));
    Arm(c, g & turbo, kArmTurbo);
    const F trbScale = Ld<F>(SF(s.trbScale));
    const F turboT = select(turbo, gearAcc * trbScale, F(0.0f));
    const F rearSign = select(rev, F(-1.0f), F(0.0f));
    const F cur = Ld<F>(SF(s.steerCur));
    const F slowSt = F(k.slowDownScale) * FabsL(cur) * EvalSpeedCurveL(k.curve[kCurveSteerSlowDown], v.z) *
                     select(rev, F(-1.0f), F(1.0f));
    F drive = ExitBonusL<F>(k, s, T, phase) +
              DriveFadeL<F>(k, s, T, phase) * (turboT + (in.gas * mat[1] + rearSign * mat[1] * in.brake) * blend) - slowSt;
    drive = select(water, drive * F(0.5f), drive);
    drive = select(fw, select(turbo, gearAcc * trbScale, F(0.0f)), drive);
    // opposing (brake / reverse) force
    const auto fwd = cmp_gt(v.z, F(0.0f));
    const auto bk = g & (fwd | (cmp_lt(v.z, F(0.0f)) & cmp_gt(in.gas, F(kGate))));
    F Fo(0.0f);
    if (any(bk)) {
        const auto rb = bk & ~fwd & ~fw & cmp_lt(F(k.burnout.reverseForceThreshold), -drive * F(k.forceDivisor) * v.z) &
                        cmp_gt(c.R.c1.y, F(0.75f));
        if (any(rb)) {
            Arm(c, rb, kArmBurnoutEnter);
            StIM<F>(SF(s.brnStartMs), rb, T);                 // TryEnterReverseBurnout
            StIM<F>(SF(s.brnPhase), rb, I(kTimedSpin));
        }
        F pen(1.0f);
        for (int w = 0; w < 4; ++w) pen = select(NzI<F>(LdI<F>(s.slipping[w])), pen * F(k.perSlippingWheelAccelScale), pen);
        const F FoF = (F(k.accelBase) + F(k.accelSpeedCoef) * v.z) * in.brake * pen;
        const F FoR = (F(k.accelBase) - F(k.accelSpeedCoef) * v.z) * in.gas * pen;
        Fo = select(fwd, FoF, FoR);
        const F cap = mat[2] * select(fwd, select(c.slipFlag, F(k.capSlipping), F(k.cap)),
                                      select(c.slipFlag, F(k.capSlippingReverse), F(k.capReverse)));
        const auto capM = bk & cmp_lt(cap, Fo);
        if (any(capM)) {
            Arm(c, capM, kArmBrakeCap);
            Fo = select(capM, cap, Fo);
            for (int w = 0; w < 4; ++w) StIM<F>(s.slipping[w], capM, I(1));
            slipSeen = slipSeen | capM;
        }
        Fo = select(bk, Fo, F(0.0f));
    }
    F net = drive - SgnL(v.z) * Fo;
    const auto lim1 = g & cmp_gt(v.z, F(k.engineSpeedNorm) * mat[0]);
    Arm(c, lim1, kArmLimiter);
    net = select(lim1, select(~cmp_lt(net, F(0.0f)), F(-k.speedLimitForce), net - F(k.speedLimitForce)), net);
    const auto lim2 = g & cmp_lt(v.z, F(-k.reverseSpeedNorm) * mat[0]);
    Arm(c, lim2, kArmLimiter);
    net = select(lim2, select(~cmp_gt(net, F(0.0f)), F(k.speedLimitForce), net + F(k.speedLimitForce)), net);
    c.Fc.z = select(g, c.Fc.z + net * slopeB, c.Fc.z);
    c.Fc.z = select(g, c.Fc.z + (F(-k.forceZScale) * c.Fl.z) / F(k.fsGrounded), c.Fc.z);
    StIM<F>(SF(s.slipActive), g, IOf<F>(slipSeen));
    St3M<F>(s.localSpeedPrev, g, v);
}

// --- C14 air control (feedback.cpp:355-463) ------------------------------------
template <class F, class S>
TW_INLINE void AirControl(const CarForceConsts &k, S &s, Ctx<F> &c, const TickInT<F> &in, LaneM<F> kill, LaneM<F> inWater) {
    using I = LaneI<F>;
    const auto am = c.live & ~inWater;                      // EXIT AIR
    if (!any(am)) return;
    const V3L<F> w = c.wl;
    V3L<F> tau = -w;
    I memTick = LdI<F>(SF(s.airMemTickMs));
    memTick = select(am & kill, c.T, memTick);
    StIM<F>(SF(s.airMemTickMs), am & kill, memTick);
    const auto km = am & (kill | NzI<F>(LdI<F>(SF(s.airRefresh))));
    St3M<F>(s.airMem, km, w);
    const auto wm = am & ~km & ULt(c.T - memTick, I(static_cast<std::int32_t>(k.airMemWindow)));
    if (any(wm)) {
        Arm(c, wm, kArmAirMemory);
        const F cur = in.steer;
        F my = Ld<F>(s.airMem[1]);
        const auto condA = (cmp_lt(F(1.0e-5f), cur) & cmp_lt(my, F(0.0f))) | (cmp_lt(cur, F(-1.0e-5f)) & cmp_lt(F(0.0f), my));
        const auto strongA = condA & cmp_lt(F(k.airYSwitch), FabsL(w.y));
        const auto strongB = ~condA & ((cmp_lt(F(1.0e-5f), cur) & cmp_lt(F(0.0f), my)) | (cmp_lt(cur, F(-1.0e-5f)) & cmp_lt(my, F(0.0f))));
        const auto strong = strongA | strongB;
        my = select(strongA | ~condA, w.y, my);
        const F mx = select(cmp_lt(F(1.0e-5f), in.brake) & cmp_lt(F(0.0f), Ld<F>(s.airMem[0])), F(0.0f), w.x);   // the air brake
        StM<F>(s.airMem[0], wm, mx);
        StM<F>(s.airMem[1], wm, my);
        const V3L<F> target = v3l(mx, my, w.z);
        const auto st = wm & strong;
        Arm(c, st, kArmAirStrong);
        tau = Sel3(st, tau * F(3.0f), tau);
        const auto az = wm & ~c.grounded;
        if (any(az)) tau.z = select(az, tau.z * EvalCurveL(k.curve[kCurveAirZScale], FabsL(w.z)), tau.z);
        c.ww = Sel3(wm, mull(c.R, target), c.ww);           // direct angular velocity write
    }
    const auto air = am & ~c.grounded;
    if (any(air)) {
        const F L = tw::math::sqrt(tau.z * tau.z + (tau.x * tau.x + tau.y * tau.y));
        const auto m = air & cmp_ge(L, F(1.0e-5f));
        if (any(m)) {
            Arm(c, m, kArmAirTorque);
            const F mag = (F(k.airTorqueQuad) * L) * L + L * F(k.airTorqueLin);
            c.Tc = Sel3(m, c.Tc + tau * (F(1.0f) / L) * mag, c.Tc);
        }
    }
}

// --- C17 turbo and free wheel (feedback.cpp:784-859) --------------------------
template <class F, class S>
TW_INLINE void EnableTurbo(const CarForceConsts &, S &s, LaneI<F> T, std::uint32_t duration, F scale, std::int32_t type,
                           LaneI<F> source, LaneM<F> m) {
    using I = LaneI<F>;
    const I typeOld = LdI<F>(SF(s.trbType));
    I src = select(m & INe(typeOld, I(type)), I(0), LdI<F>(SF(s.trbSourceId)));
    if (type == kTurboDirect) {
        StM<F>(SF(s.trbScale), m, scale);
    } else {
        const auto rm = m & INe(src, source);
        if (any(rm)) {
            const I d = T - LdI<F>(SF(s.trbRouletteOriginMs));
            alignas(64) std::int32_t dv[F::width], rv[F::width];
            d.store(dv);
            for (int l = 0; l < F::width; ++l) rv[l] = static_cast<std::int32_t>(static_cast<std::uint32_t>(dv[l]) % 1000u);
            const F x = tw::math::from_u32(I::load(rv)) / F(1000.0f);
            const F phase = select(cmp_lt(x, F(4.0f / 7.0f)), F(0.0f), select(cmp_lt(x, F(6.0f / 7.0f)), F(0.5f), F(1.0f)));
            StM<F>(SF(s.trbPhase), rm, phase);
            StM<F>(SF(s.trbScale), rm, (phase + F(1.0f)) * scale);
            src = select(rm, source, src);
        }
    }
    StIM<F>(SF(s.trbSourceId), m, src);
    StIM<F>(SF(s.trbEndMs), m, T + I(static_cast<std::int32_t>(duration)));
    StIM<F>(SF(s.trbType), m, I(type));
}

template <class F, class S> TW_INLINE void Turbo(const CarForceConsts &k, S &s, Ctx<F> &c) {
    using I = LaneI<F>;
    const I special((1 << kMatTurbo) | (1 << kMatTurbo2) | (1 << kMatRoulette) | (1 << kMatFreeWheel));
    const auto any4 = c.live & NzI<F>(c.surfaces & special);
    if (any(any4)) {
        struct Booster {
            std::int32_t material;
            std::uint32_t duration;
            float scale;
            std::int32_t type;
        };
        const Booster boosters[3] = {{kMatTurbo, k.turboDurationA, k.turboScaleA, kTurboDirect},
                                     {kMatTurbo2, k.turboDurationB, k.turboScaleB, kTurboDirect},
                                     {kMatRoulette, k.turboDurationA, k.turboScaleA, kTurboRoulette}};
        for (const Booster &bo : boosters) {
            // the first wheel in index order with that material gives peerZ and peerId
            auto found = MaskFalse<F>();
            F pz(0.0f);
            I pid(0);
            for (int w = 3; w >= 0; --w) {
                const auto m = NzI<F>(LdI<F>(s.contact[w])) & IEq(LdI<F>(s.surface[w]), I(bo.material));
                pz = select(m, Ld<F>(s.peerAxisZ[w]), pz);
                pid = select(m, LdI<F>(s.peerId[w]), pid);
                found = found | m;
            }
            const auto en = any4 & found;
            if (!any(en)) continue;
            Arm(c, en, kArmTurboEnable);
            EnableTurbo(k, s, c.T, bo.duration, F(bo.scale) * pz, bo.type, pid, en);
        }
        auto fwM = MaskFalse<F>();
        for (int w = 0; w < 4; ++w) fwM = fwM | (NzI<F>(LdI<F>(s.contact[w])) & IEq(LdI<F>(s.surface[w]), I(kMatFreeWheel)));
        StIM<F>(SF(s.freewheel), any4 & fwM, I(1));
    }
    const auto expire = c.live & NzI<F>(LdI<F>(SF(s.trbType))) & ULt(LdI<F>(SF(s.trbEndMs)), c.T);
    StIM<F>(SF(s.trbType), expire, I(kTurboOff));
}

}  // namespace detail

// ---------------------------------------------------------------------------
// S1 + C1..C20 for the lanes in `active`, substep-start rotation R.
template <class F, class S>
TW_INLINE void CarForcesT(const CarForceConsts &k, const WaterZone *water, S &s, const TickInT<F> &in, F h,
                          ForceOutT<F> &out, const M3L<F> &R, LaneM<F> active
#if defined(__CUDACC__)
                          , const std::uint8_t *fakeTex = nullptr
#endif
) {
    using namespace detail;
    using I = LaneI<F>;
    using M = LaneM<F>;
    Ctx<F> c;
#if defined(__CUDACC__)
    c.fakeTex = fakeTex != nullptr ? fakeTex : k.fakeTextureRed;
#endif
    c.R = R;
    c.pos = Ld3<F>(s.pos);
    c.vw = Ld3<F>(s.vel);
    c.ww = Ld3<F>(s.angVel);
    c.T = in.T;
    c.arms = I(0);
    c.slipFlag = MaskFalse<F>();
    out.posePos = c.pos;
    out.poseRot = c.R;

    // S1: gravity and air drag for the previous grounded state; zone angular damping (scale 0)
    const M gp = NzI<F>(LdI<F>(SF(s.groundedPrev)));
    const F fs = select(gp, F(k.fsGrounded), F(k.fsAirborne));
    const F lds = select(gp, F(0.0f), F(k.ldsAirborne));
    const V3L<F> Fs1 = v3l<F>(k.gravity) * (fs * F(k.mass)) + c.vw * (F(-k.zoneLinDamp) * lds);
    const V3L<F> vS1 = c.vw;

    // C1 frozen (FB:968-972): EXIT C1, skips C2..C20 (no scratch clears)
    M frozen = active & (NzI<F>(LdI<F>(SF(s.speedBlocked))) | NzI<F>(LdI<F>(SF(s.speedBlocked2))));
    if (k.frozenByWaterBox) frozen = active;
    const V3L<F> zero = v3l(F(0.0f), F(0.0f), F(0.0f));
    if (any(frozen)) {
        St3M<F>(s.vel, frozen, zero);
        St3M<F>(s.angVel, frozen, zero);
    }
    c.live = active & ~frozen;
    if (!any(c.live)) {
        out.force = zero;
        out.torque = zero;
        out.arms = select(frozen, I(static_cast<std::int32_t>(kArmFrozen)), I(0));
        return;
    }

    c.vl = c.wl = c.Fc = c.Tc = zero;
    // C2 fake contacts
    FakeContacts(k, s, c);

    // C3 IntegrateVehicle: contact normals, suspension, engine, steering
    // (contact normal normalisation: on demand, see UnitNormalL)
    const F invH = F(1.0f) / h;
    for (int w = 0; w < 4; ++w) {
        const F a = Ld<F>(s.absorb[w]);
        const F b = a - Ld<F>(s.maxRepl[w]);
        const F t = (F(k.restAbsorb) - b) * h * F(k.followCoef) + b;
        StM<F>(s.absorbVel[w], c.live, (t - a) * invH);
        StM<F>(s.absorb[w], c.live, t);
        StM<F>(s.maxRepl[w], c.live, F(0.0f));
    }
    // C4 grounded (contacts of the last collision pass); next substep's fs / lds
    {
        I cAny(0);
        c.surfaces = I(0);
        for (int w = 0; w < 4; ++w) {
            const I cw = LdI<F>(s.contact[w]);
            cAny = cAny | cw;
            c.surfaces = c.surfaces | select(NzI<F>(cw), IShlVar(I(1), LdI<F>(s.surface[w]) & I(31)), I(0));
        }
        c.grounded = NzI<F>(cAny);
    }
    Arm(c, c.live & c.grounded, kArmGrounded);
    Arm(c, c.live & cmp_gt(in.gas, F(kGate)) & cmp_gt(in.brake, F(kGate)), kArmGasBrake);
    {
        const M fw = NzI<F>(LdI<F>(SF(s.freewheel)));
        const M em = c.live & ~fw;
        if (any(em)) Engine(k, s, c, select(NzI<F>(LdI<F>(SF(s.reverse))), in.brake, in.gas), h, em);
        const M fwl = c.live & fw;
        if (any(fwl)) {
            Arm(c, fwl, kArmFreewheel);
            StM<F>(SF(s.rpm), fwl, F(0.0f));
        }
    }
    SteerSlew(k, s, c, in.steer, h);
    StIM<F>(SF(s.groundedPrev), c.live, IOf<F>(c.grounded));

    // C5 locals (after the C2 impulses)
    c.vl = mul_tl(c.R, c.vw);
    c.wl = mul_tl(c.R, c.ww);
    // R^T F_S1 = R^T g (fs m) + R^T v_S1 (-lds): the gravity part is one matrix row when g = (0, gy, 0)
    c.Fl = k.gravityYOnly ? v3l(c.R.c0.y, c.R.c1.y, c.R.c2.y) * (F(k.gravity[1]) * (fs * F(k.mass)))
                          : mul_tl(c.R, v3l<F>(k.gravity)) * (fs * F(k.mass));
    {
        const M dragM = cmp_neq(lds, F(0.0f));
        if (any(dragM)) c.Fl = Sel3(dragM, c.Fl + mul_tl(c.R, vS1) * (F(-k.zoneLinDamp) * lds), c.Fl);
    }
    c.Fc = zero;
    c.Tc = zero;
    if (k.angularDampingScale != 0.0f) c.Tc = c.wl * F(-k.zoneAngDamp * k.angularDampingScale);   // S1 zone torque (0 for StadiumCar)

    // C6 coasting friction + lateral slowdown window
    Friction(k, s, c, in);

    // C7 speed clamp (written to the body)
    {
        const float cap2 = k.linearSpeedCap * k.linearSpeedCap;
        const F sp2 = c.vl.z * c.vl.z + (c.vl.y * c.vl.y + c.vl.x * c.vl.x);
        if (cap2 > kEps2) {
            const M m = c.live & cmp_lt(F(cap2), sp2);
            if (any(m)) {
                Arm(c, m, kArmSpeedClamp);
                c.vl = Sel3(m, c.vl * (F(k.linearSpeedCap) / tw::math::sqrt(sp2)), c.vl);
                c.vw = Sel3(m, mull(c.R, c.vl), c.vw);
            }
        }
    }

    // C8 ground material: wheel 0's material for every contacting wheel
    F mat[4] = {F(0.0f), F(0.0f), F(0.0f), F(0.0f)};
    const M gl = c.live & c.grounded;
    if (any(gl)) {
        const I s0 = LdI<F>(s.surface[0]) & I(31);
        for (int i = 0; i < 4; ++i) mat[i] = select(c.grounded, Lookup32(k.byIdBlend[i], s0), F(0.0f));
    }

    // C9 slope adherence from the S1 force (read only by the ground path: skipped airborne)
    F slopeA(1.0f), slopeB(1.0f);
    if (any(gl)) {
        const F l2 = c.Fl.y * c.Fl.y + c.Fl.x * c.Fl.x + c.Fl.z * c.Fl.z;
        const M m = gl & cmp_lt(F(kEps2), l2);
        if (any(m)) {
            const F slope = FabsL(c.Fl.y / tw::math::sqrt(l2));
            slopeA = select(m, SlopeBlendL(slope, k.slope1Min, k.slope1Max), slopeA);
            slopeB = select(m, SlopeBlendL(slope, k.slope2Min, k.slope2Max), slopeB);
        }
    }

    // C10 front wheel steer angle (argument <= 1 / wheelSpeedBase = 0.5: asin_small domain)
    // (read only by the ground path: contact side axes, burnout entry, donut)
    c.delta = F(0.0f);
    c.cosD = F(1.0f);
    c.sinD = F(0.0f);
    {
        const F cur = Ld<F>(SF(s.steerCur));
        const M sm = gl & cmp_neq(cur, F(0.0f));           // straight: delta = 0 exactly
        if (any(sm)) {
            const F denom = FabsL(c.vl.z) * F(k.wheelSpeedScale) + F(k.wheelSpeedBase);
            const M dok = ~cmp_lt(denom, F(kEps));
            if (!k.steerSmall) {                            // other Model 6 cars (e.g. wheelSpeedBase 1): GmFunc::AsinSafe
                const F x = F(1.0f) / denom;
                const F a = select(dok, select(cmp_gt(x, F(1.0f - 1.0e-6f)), F(kHalfPi),
                                               select(cmp_lt(x, F(-(1.0f - 1.0e-6f))), F(-kHalfPi), tw::math::asin(x))),
                                   F(0.0f));
                c.delta = select(sm, (-cur) * a, c.delta);
                const F ad = FabsL(c.delta);
                c.cosD = select(sm, CosHalfPiL(ad), c.cosD);
                c.sinD = select(sm, select(cmp_lt(c.delta, F(0.0f)), -SinPiL(ad), SinPiL(ad)), c.sinD);
            } else {
                const F a = select(dok, AsinSmallL(F(1.0f) / denom), F(0.0f));
                c.delta = select(sm, (-cur) * a, c.delta);
                const SinCosL<F> sc = SinCosSmallL(c.delta);
                c.cosD = select(sm, sc.cos, c.cosD);
                c.sinD = select(sm, sc.sin, c.sinD);
            }
        }
    }
    // C11
    St3M<F>(s.localSpeedPrev, c.live, c.vl);

    // C12 Model 6 (legacy_models.cpp:1073-1109)
    const M inWater = Water(k, water, s, c);
    StIM<F>(SF(s.inWater), c.live, IOf<F>(inWater));
    M dirt = MaskTrue<F>();
    for (int w = 0; w < 4; ++w) dirt = dirt & NzI<F>(LdI<F>(s.contact[w])) & IEq(LdI<F>(s.surface[w]), I(kMatDirt));
    const I phase0 = LdI<F>(SF(s.brnPhase));
    const M dn = c.live & IEq(phase0, I(kCircularDrift));
    const M nd = c.live & ~dn;
    if (any(dn)) {                                           // EXIT DONUT: replaces the whole normal path
        Arm(c, dn, kArmDonut);
        DonutForces(k, s, c, in, inWater, dn);
        const M cancelled = dn & INe(LdI<F>(SF(s.brnPhase)), I(kCircularDrift));
        if (any(cancelled)) {
            Arm(c, cancelled, kArmDonutCancel);
            StIM<F>(SF(s.brnPhase), cancelled, I(kTimedSpin));
            StIM<F>(SF(s.brnStartMs), cancelled, c.T);
        }
    }
    if (any(nd)) {
        M slipSeen = AdvanceBurnoutPhases<F>(k, s, c.T, nd);
        {
            const I ph = LdI<F>(SF(s.brnPhase));
            Arm(c, nd & IEq(ph, I(kTimedSpin)), kArmTimedSpin);
            Arm(c, nd & IEq(ph, I(kExitFade)), kArmExitFade);
        }
        const M g = nd & c.grounded;
        if (any(g)) {
            const F maxSideCurve = EvalSpeedCurveL(k.curve[kCurveMaxSideFriction], c.vl.z);
            ContactWheels(k, s, c, in, slopeA, dirt, maxSideCurve, g);
            GroundForces(k, s, c, in, inWater, slipSeen, slopeB, mat, maxSideCurve, g);
        }
        const M ng = nd & ~c.grounded;                      // EXIT NOGROUND
        if (any(ng)) {
            const M ts = ng & IEq(LdI<F>(SF(s.brnPhase)), I(kTimedSpin));
            StIM<F>(SF(s.brnExitStartMs), ts, c.T);
            StIM<F>(SF(s.brnPhase), ts, I(kExitFade));
            StIM<F>(SF(s.slipActive), ng, IOf<F>(slipSeen));
        }
    }

    // C13 side-speed kill (FB:687-697): only for lateralForceScale < 0 (not StadiumCar)
    M kill = MaskFalse<F>();
    for (int w = 0; w < 4; ++w)
        if (k.wheel[w].killsLateral) kill = kill | NzI<F>(LdI<F>(s.contact[w]));
    if (k.lateralForceScale < 0.0f) {
        const M km = c.live & kill;
        if (any(km)) c.vw = Sel3(km, mull(c.R, v3l(F(0.0f), c.vl.y, c.vl.z)), c.vw);
    }
    // C14 air control; every StadiumCar wheel kills lateral speed on contact
    AirControl(k, s, c, in, kill, inWater);

    // C17 turbo / free wheel
    Turbo(k, s, c);
    Arm(c, c.live & NzI<F>(LdI<F>(SF(s.freewheel))), kArmFreewheel);

    // C18 previous force (FV representation: local total / forceDivisor)
    St3M<F>(s.prevForce, c.live, (c.Fl + c.Fc) * F(k.invForceDivisor));

    // C19 / C20 scratch clears
    for (int w = 0; w < 4; ++w) {
        StIM<F>(s.contact[w], c.live, I(0));
        StIM<F>(s.surface[w], c.live, I(0));
        St3AtM<F>(s.normalSum, 3 * w, c.live, zero);
        St3AtM<F>(s.latestPoint, 3 * w, c.live, zero);
    }
    StIM<F>(SF(s.bodyContact), c.live, I(0));
    StIM<F>(SF(s.airRefresh), c.live, I(0));
    StIM<F>(SF(s.latContact), c.live, I(0));

    St3M<F>(s.vel, c.live, c.vw);
    St3M<F>(s.angVel, c.live, c.ww);
    out.force = Sel3(c.live, Fs1 + mull(c.R, c.Fc), zero);
    out.torque = Sel3(c.live, c.Tc, zero);                   // car frame (FV world torque = R torque)
    out.arms = select(frozen, I(static_cast<std::int32_t>(kArmFrozen)), c.arms);
}

// S2 (dyna_core.cpp:229-363) for the lanes in `active`: forward Euler on
// position with the old velocity, first-order quaternion update about the
// COM, angular clamp. `post` (optional) receives R(q_new) where the
// quaternion moved, else R.
template <class F, class S>
TW_INLINE void CarIntegrateT(const CarForceConsts &k, S &s, const ForceOutT<F> &f, F h, const M3L<F> &R, LaneM<F> active,
                             M3L<F> *post = nullptr) {
    V3L<F> pos = Ld3<F>(s.pos), v = Ld3<F>(s.vel), w = Ld3<F>(s.angVel);
    pos = v * h + pos;
    v = v + f.force * F(k.invMass) * h;
    // alpha = I^-1_world tau = R D R^T (R tau_local) = R D tau_local (old rotation)
    const V3L<F> alpha = mull(R, v3l(f.torque.x * F(k.inertiaInvBody[0]), f.torque.y * F(k.inertiaInvBody[1]),
                                     f.torque.z * F(k.inertiaInvBody[2])));
    const F w2 = w.y * w.y + w.x * w.x + w.z * w.z;
    const auto qm = active & tw::math::cmp_lt(F(1.0e-10f), w2);
    M3L<F> R2 = R;
    if (any(qm)) {
        const F qx = Ld<F>(s.quat[0]), qy = Ld<F>(s.quat[1]), qz = Ld<F>(s.quat[2]), qw = Ld<F>(s.quat[3]);
        const F dw = ((-w.x * qx - w.y * qy) - qz * w.z) * F(0.5f);
        const F dx = ((qw * w.x + w.y * qz) - qy * w.z) * F(0.5f);
        const F dy = ((w.y * qw - qz * w.x) + w.z * qx) * F(0.5f);
        const F dz = ((qy * w.x - w.y * qx) + qw * w.z) * F(0.5f);
        F nq[4] = {dx * h + qx, dy * h + qy, dz * h + qz, dw * h + qw};
        const F inv = detail::InvSqrtL(((nq[0] * nq[0] + nq[3] * nq[3]) + nq[1] * nq[1]) + nq[2] * nq[2]);
        for (int i = 0; i < 4; ++i) nq[i] = nq[i] * inv;
        const M3L<F> Rn = RotFromQuatL(nq[0], nq[1], nq[2], nq[3]);
        const V3L<F> com = v3l<F>(k.com);
        pos = Sel3(qm, pos - (mull(Rn, com) - mull(R, com)), pos);
        for (int i = 0; i < 4; ++i) StM<F>(s.quat[i], qm, nq[i]);
        R2.c0 = Sel3(qm, Rn.c0, R.c0);
        R2.c1 = Sel3(qm, Rn.c1, R.c1);
        R2.c2 = Sel3(qm, Rn.c2, R.c2);
    }
    w = detail::ClampAngularL(w + alpha * h, k.maxAngularSpeed);
    St3M<F>(s.pos, active, pos);
    St3M<F>(s.vel, active, v);
    St3M<F>(s.angVel, active, w);
    if (post) *post = R2;
}

// ---- airborne lean path ------------------------------------------------------------
// CarForcesT for lanes that are live (not frozen), have no wheel contact (so not grounded, no fake
// contacts, no dirt, no side-speed kill) and are not in the donut phase: the blocks of that path in
// CarForcesT's order with the same operations, so outputs and state equal CarForcesT's bit for bit.
// Used for L8's airborne micro-steps. CarAirEligible tells which lanes qualify.
template <class F, class S> TW_INLINE LaneM<F> CarAirEligibleT(const CarForceConsts &k, const S &s) {
    using I = LaneI<F>;
    if (k.frozenByWaterBox) return MaskFalse<F>();
    I any = LdI<F>(SF(s.speedBlocked)) | LdI<F>(SF(s.speedBlocked2));
    for (int w = 0; w < 4; ++w) any = any | LdI<F>(s.contact[w]);
    return ~detail::NzI<F>(any) & INe(LdI<F>(SF(s.brnPhase)), I(kCircularDrift));
}

template <class F, class S>
TW_INLINE void CarForcesAirT(const CarForceConsts &k, const WaterZone *water, S &s, const TickInT<F> &in, F h, ForceOutT<F> &out,
                             const M3L<F> &R, LaneM<F> active) {
    using namespace detail;
    using I = LaneI<F>;
    using M = LaneM<F>;
    Ctx<F> c;
#if defined(__CUDACC__)
    c.fakeTex = k.fakeTextureRed;
#endif
    c.R = R;
    c.pos = Ld3<F>(s.pos);
    c.vw = Ld3<F>(s.vel);
    c.ww = Ld3<F>(s.angVel);
    c.T = in.T;
    c.arms = I(0);
    c.slipFlag = MaskFalse<F>();
    out.posePos = c.pos;
    out.poseRot = c.R;
    // S1
    const M gp = NzI<F>(LdI<F>(SF(s.groundedPrev)));
    const F fs = select(gp, F(k.fsGrounded), F(k.fsAirborne));
    const F lds = select(gp, F(0.0f), F(k.ldsAirborne));
    const V3L<F> Fs1 = v3l<F>(k.gravity) * (fs * F(k.mass)) + c.vw * (F(-k.zoneLinDamp) * lds);
    const V3L<F> vS1 = c.vw;
    const V3L<F> zero = v3l(F(0.0f), F(0.0f), F(0.0f));
    c.live = active;                         // not frozen (eligible)
    c.vl = c.wl = c.Fc = c.Tc = zero;
    // C2: no contact, nothing to do. C3 suspension
    const F invH = F(1.0f) / h;
    for (int w = 0; w < 4; ++w) {
        const F a = Ld<F>(s.absorb[w]);
        const F b = a - Ld<F>(s.maxRepl[w]);
        const F t = (F(k.restAbsorb) - b) * h * F(k.followCoef) + b;
        StM<F>(s.absorbVel[w], c.live, (t - a) * invH);
        StM<F>(s.absorb[w], c.live, t);
        StM<F>(s.maxRepl[w], c.live, F(0.0f));
    }
    // C4: no contact
    c.surfaces = I(0);
    c.grounded = MaskFalse<F>();
    Arm(c, c.live & cmp_gt(in.gas, F(kGate)) & cmp_gt(in.brake, F(kGate)), kArmGasBrake);
    {
        const M fw = NzI<F>(LdI<F>(SF(s.freewheel)));
        const M em = c.live & ~fw;
        if (any(em)) Engine(k, s, c, select(NzI<F>(LdI<F>(SF(s.reverse))), in.brake, in.gas), h, em);
        const M fwl = c.live & fw;
        if (any(fwl)) {
            Arm(c, fwl, kArmFreewheel);
            StM<F>(SF(s.rpm), fwl, F(0.0f));
        }
    }
    SteerSlew(k, s, c, in.steer, h);
    StIM<F>(SF(s.groundedPrev), c.live, I(0));
    // C5
    c.vl = mul_tl(c.R, c.vw);
    c.wl = mul_tl(c.R, c.ww);
    c.Fl = k.gravityYOnly ? v3l(c.R.c0.y, c.R.c1.y, c.R.c2.y) * (F(k.gravity[1]) * (fs * F(k.mass)))
                          : mul_tl(c.R, v3l<F>(k.gravity)) * (fs * F(k.mass));
    {
        const M dragM = cmp_neq(lds, F(0.0f));
        if (any(dragM)) c.Fl = Sel3(dragM, c.Fl + mul_tl(c.R, vS1) * (F(-k.zoneLinDamp) * lds), c.Fl);
    }
    c.Fc = zero;
    c.Tc = zero;
    if (k.angularDampingScale != 0.0f) c.Tc = c.wl * F(-k.zoneAngDamp * k.angularDampingScale);
    // C6
    Friction(k, s, c, in);
    // C7
    {
        const float cap2 = k.linearSpeedCap * k.linearSpeedCap;
        const F sp2 = c.vl.z * c.vl.z + (c.vl.y * c.vl.y + c.vl.x * c.vl.x);
        if (cap2 > kEps2) {
            const M m = c.live & cmp_lt(F(cap2), sp2);
            if (any(m)) {
                Arm(c, m, kArmSpeedClamp);
                c.vl = Sel3(m, c.vl * (F(k.linearSpeedCap) / tw::math::sqrt(sp2)), c.vl);
                c.vw = Sel3(m, mull(c.R, c.vl), c.vw);
            }
        }
    }
    // C8-C10: ground only
    c.delta = F(0.0f);
    c.cosD = F(1.0f);
    c.sinD = F(0.0f);
    // C11
    St3M<F>(s.localSpeedPrev, c.live, c.vl);
    // C12: water, burnout phases (not donut), EXIT NOGROUND
    const M inWater = Water(k, water, s, c);
    StIM<F>(SF(s.inWater), c.live, IOf<F>(inWater));
    {
        const M nd = c.live;
        const M slipSeen = AdvanceBurnoutPhases<F>(k, s, c.T, nd);
        const I ph = LdI<F>(SF(s.brnPhase));
        Arm(c, nd & IEq(ph, I(kTimedSpin)), kArmTimedSpin);
        Arm(c, nd & IEq(ph, I(kExitFade)), kArmExitFade);
        const M ts = nd & IEq(ph, I(kTimedSpin));
        StIM<F>(SF(s.brnExitStartMs), ts, c.T);
        StIM<F>(SF(s.brnPhase), ts, I(kExitFade));
        StIM<F>(SF(s.slipActive), nd, IOf<F>(slipSeen));
    }
    // C13: no contact, no kill. C14
    AirControl(k, s, c, in, MaskFalse<F>(), inWater);
    // C17
    Turbo(k, s, c);
    Arm(c, c.live & NzI<F>(LdI<F>(SF(s.freewheel))), kArmFreewheel);
    // C18
    St3M<F>(s.prevForce, c.live, (c.Fl + c.Fc) * F(k.invForceDivisor));
    // C19 / C20
    for (int w = 0; w < 4; ++w) {
        StIM<F>(s.contact[w], c.live, I(0));
        StIM<F>(s.surface[w], c.live, I(0));
        St3AtM<F>(s.normalSum, 3 * w, c.live, zero);
        St3AtM<F>(s.latestPoint, 3 * w, c.live, zero);
    }
    StIM<F>(SF(s.bodyContact), c.live, I(0));
    StIM<F>(SF(s.airRefresh), c.live, I(0));
    StIM<F>(SF(s.latContact), c.live, I(0));
    St3M<F>(s.vel, c.live, c.vw);
    St3M<F>(s.angVel, c.live, c.ww);
    out.force = Sel3(c.live, Fs1 + mull(c.R, c.Fc), zero);
    out.torque = Sel3(c.live, c.Tc, zero);
    out.arms = c.arms;
}

// ---- L = 1 API (TwinState) --------------------------------------------------------
namespace detail {
TW_INLINE M3L<f32x1> ToLane(const M3 &R) {
    return {v3l(f32x1(R.c0.x), f32x1(R.c0.y), f32x1(R.c0.z)), v3l(f32x1(R.c1.x), f32x1(R.c1.y), f32x1(R.c1.z)),
            v3l(f32x1(R.c2.x), f32x1(R.c2.y), f32x1(R.c2.z))};
}
TW_INLINE V3 ToV3(V3L<f32x1> a) { return V3{a.x.v, a.y.v, a.z.v}; }
}  // namespace detail

// S1 + C1..C20 with the substep-start rotation R (= R(quat); the checker can
// pass FV's stored matrix instead to separate representation noise).
TW_INLINE void CarForcesR(const CarForceConsts &k, const WaterZone *water, TwinState &s, const CarTickIn &in, float h,
                          CarForceOut &out, const M3 &R
#if defined(__CUDACC__)
                          , const std::uint8_t *fakeTex = nullptr
#endif
) {
    TickInT<f32x1> inl{f32x1(in.gas), f32x1(in.brake), f32x1(in.steer), i32x1(static_cast<std::int32_t>(in.tickMs))};
    ForceOutT<f32x1> o;
#if defined(__CUDACC__)
    CarForcesT<f32x1>(k, water, s, inl, f32x1(h), o, detail::ToLane(R), MaskTrue<f32x1>(), fakeTex);
#else
    CarForcesT<f32x1>(k, water, s, inl, f32x1(h), o, detail::ToLane(R), MaskTrue<f32x1>());
#endif
    store(out.force, detail::ToV3(o.force));
    store(out.torque, detail::ToV3(o.torque));
    store(out.posePos, detail::ToV3(o.posePos));
    store(out.poseRot, detail::ToV3(o.poseRot.c0));
    store(out.poseRot + 3, detail::ToV3(o.poseRot.c1));
    store(out.poseRot + 6, detail::ToV3(o.poseRot.c2));
    out.arms = static_cast<std::uint32_t>(o.arms.v);
}

// S1 + C1..C20. `water` may be null (no water on the map).
TW_INLINE void CarForces(const CarForceConsts &k, const WaterZone *water, TwinState &s, const CarTickIn &in, float h,
                         CarForceOut &out) {
    CarForcesR(k, water, s, in, h, out, rot_from_quat(s.quat));
}
#if defined(__CUDACC__)
// GPU: fakeTex, the fake-contact texture to read (the copy in global memory: a warp's lanes read different texels, which
// constant memory serves one address at a time).
TW_INLINE void CarForces(const CarForceConsts &k, const WaterZone *water, TwinState &s, const CarTickIn &in, float h,
                         CarForceOut &out, const std::uint8_t *fakeTex) {
    CarForcesR(k, water, s, in, h, out, rot_from_quat(s.quat), fakeTex);
}
#endif

// S2. `post` (optional): R(q_new) where the quaternion moved, else R.
#if TW_CAR_SSE3V
namespace detail {
// x y z in lanes 0-2 of an SSE register; lane 3 is never used. L3 reads 4 floats (p[3] must be readable):
// for constants only. Data the kernel has just stored as floats goes through G3 (scalar loads, so the
// stores forward; a 16-byte load over 4-byte stores stalls).
TW_INLINE __m128 L3(const float *p) { return _mm_loadu_ps(p); }
TW_INLINE __m128 G3(const float *p) { return _mm_setr_ps(p[0], p[1], p[2], 0.0f); }
TW_INLINE void S3(float *p, __m128 v) {
    _mm_storel_pi(reinterpret_cast<__m64 *>(p), v);
    _mm_store_ss(p + 2, _mm_movehl_ps(v, v));
}
TW_INLINE __m128 Bc0(__m128 v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)); }
TW_INLINE __m128 Bc1(__m128 v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)); }
TW_INLINE __m128 Bc2(__m128 v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)); }
TW_INLINE float L0(__m128 v) { return _mm_cvtss_f32(v); }
TW_INLINE float L1(__m128 v) { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1))); }
TW_INLINE float L2(__m128 v) { return _mm_cvtss_f32(_mm_movehl_ps(v, v)); }
TW_INLINE float L3s(__m128 v) { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3))); }
// mull: (c0 * b.x + c1 * b.y) + c2 * b.z per component, as mull()
TW_INLINE __m128 Mull3(__m128 c0, __m128 c1, __m128 c2, __m128 b) {
    return _mm_add_ps(_mm_add_ps(_mm_mul_ps(c0, Bc0(b)), _mm_mul_ps(c1, Bc1(b))), _mm_mul_ps(c2, Bc2(b)));
}
// the columns of an M3
TW_INLINE void Cols(const M3 &R, __m128 *c0, __m128 *c1, __m128 *c2) {
    *c0 = G3(&R.c0.x);
    *c1 = G3(&R.c1.x);
    *c2 = G3(&R.c2.x);
}
}  // namespace detail

// CarIntegrateT<f32x1> with the 3-vectors in SSE registers: every component sees the same
// operations in the same order (forward Euler, alpha = R (D tau), quaternion step, COM
// correction, angular clamp), so the result is CarIntegrateT's bits.
TW_INLINE void CarIntegrateSse(const CarForceConsts &k, TwinState &s, const CarForceOut &f, float h, const M3 &R, M3 *post) {
    using namespace detail;
    const __m128 H = _mm_set1_ps(h);
    __m128 pos = G3(s.pos), v = G3(s.vel);
    const __m128 w = G3(s.angVel);
    pos = _mm_add_ps(_mm_mul_ps(v, H), pos);                                              // v * h + pos
    v = _mm_add_ps(v, _mm_mul_ps(_mm_mul_ps(G3(f.force), _mm_set1_ps(k.invMass)), H));   // v + F * invMass * h
    __m128 c0, c1, c2;
    Cols(R, &c0, &c1, &c2);
    const __m128 alpha = Mull3(c0, c1, c2, _mm_mul_ps(G3(f.torque), L3(k.inertiaInvBody)));   // R (D tau)
    const __m128 ww = _mm_mul_ps(w, w);
    const float w2 = (L1(ww) + L0(ww)) + L2(ww);                                           // (yy + xx) + zz
    M3 R2 = R;
    if (1.0e-10f < w2) {
        const __m128 q = _mm_setr_ps(s.quat[0], s.quat[1], s.quat[2], s.quat[3]);             // x y z w
        // d = ((A + B) + C) * 0.5 per lane (x, y, z, w), the template's dx dy dz dw
        const __m128 A = _mm_xor_ps(_mm_mul_ps(_mm_shuffle_ps(q, q, _MM_SHUFFLE(0, 1, 3, 3)), _mm_shuffle_ps(w, w, _MM_SHUFFLE(0, 0, 1, 0))),
                                    _mm_setr_ps(0.0f, 0.0f, 0.0f, -0.0f));
        const __m128 B = _mm_xor_ps(_mm_mul_ps(_mm_shuffle_ps(q, q, _MM_SHUFFLE(1, 0, 2, 2)), _mm_shuffle_ps(w, w, _MM_SHUFFLE(1, 1, 0, 1))),
                                    _mm_setr_ps(0.0f, -0.0f, -0.0f, -0.0f));
        const __m128 C = _mm_xor_ps(_mm_mul_ps(_mm_shuffle_ps(q, q, _MM_SHUFFLE(2, 3, 0, 1)), Bc2(w)), _mm_setr_ps(-0.0f, 0.0f, 0.0f, -0.0f));
        const __m128 d = _mm_mul_ps(_mm_add_ps(_mm_add_ps(A, B), C), _mm_set1_ps(0.5f));
        __m128 nq = _mm_add_ps(_mm_mul_ps(d, H), q);                                          // d * h + q
        const __m128 sq = _mm_mul_ps(nq, nq);
        const float n2 = ((L0(sq) + L3s(sq)) + L1(sq)) + L2(sq);                              // ((xx + ww) + yy) + zz
        const float inv = 1.0f / tw::math::sqrt(f32x1(n2)).v;
        nq = _mm_mul_ps(nq, _mm_set1_ps(inv));
        alignas(16) float qn[4];
        _mm_store_ps(qn, nq);
        const M3L<f32x1> Rn = RotFromQuatL(f32x1(qn[0]), f32x1(qn[1]), f32x1(qn[2]), f32x1(qn[3]));
        R2 = M3{ToV3(Rn.c0), ToV3(Rn.c1), ToV3(Rn.c2)};
        __m128 n0, n1, n2v;
        Cols(R2, &n0, &n1, &n2v);
        const __m128 com = L3(k.com);
        pos = _mm_sub_ps(pos, _mm_sub_ps(Mull3(n0, n1, n2v, com), Mull3(c0, c1, c2, com)));   // pos - (Rn com - R com)
        _mm_storeu_ps(s.quat, nq);
    }
    __m128 wn = _mm_add_ps(w, _mm_mul_ps(alpha, H));                                           // w + alpha * h
    if (k.maxAngularSpeed > 0.0f) {                                                              // ClampAngularL
        const __m128 sq = _mm_mul_ps(wn, wn);
        const float l2 = (L0(sq) + L1(sq)) + L2(sq);
        if (l2 > k.maxAngularSpeed * k.maxAngularSpeed)
            wn = _mm_mul_ps(wn, _mm_set1_ps(k.maxAngularSpeed / tw::math::sqrt(f32x1(l2)).v));
    }
    S3(s.pos, pos);
    S3(s.vel, v);
    S3(s.angVel, wn);
    if (post) *post = R2;
}
#endif

TW_INLINE void CarIntegrateR(const CarForceConsts &k, TwinState &s, const CarForceOut &f, float h, const M3 &R,
                             M3 *post = nullptr) {
#if TW_CAR_SSE3V
    CarIntegrateSse(k, s, f, h, R, post);
    return;
#endif
    ForceOutT<f32x1> o;
    o.force = v3l<f32x1>(f.force);
    o.torque = v3l<f32x1>(f.torque);
    M3L<f32x1> p;
    CarIntegrateT<f32x1>(k, s, o, f32x1(h), detail::ToLane(R), MaskTrue<f32x1>(), post ? &p : nullptr);
    if (post) *post = M3{detail::ToV3(p.c0), detail::ToV3(p.c1), detail::ToV3(p.c2)};
}

TW_INLINE void CarIntegrate(const CarForceConsts &k, TwinState &s, const CarForceOut &f, float h) {
    CarIntegrateR(k, s, f, h, rot_from_quat(s.quat));
}

// One substep's forces + integration (collision S3/S4 run outside).
TW_INLINE void CarSubstep(const CarForceConsts &k, const WaterZone *water, TwinState &s, const CarTickIn &in, float h,
                          CarForceOut &out) {
    const M3 R = rot_from_quat(s.quat);
    CarForcesR(k, water, s, in, h, out, R);
    CarIntegrateR(k, s, out, h, R);
}

// Airborne micro-step (L8): CarSubstep's result, via the lean airborne path when the state
// qualifies (CarAirEligible: live, no wheel contact, not in the donut phase), else via CarSubstep.
TW_INLINE bool CarAirEligible(const CarForceConsts &k, const TwinState &s) { return CarAirEligibleT<f32x1>(k, s).v; }
// CarForcesR's result: the lean airborne path when the state qualifies.
TW_INLINE void CarForcesAirR(const CarForceConsts &k, const WaterZone *water, TwinState &s, const CarTickIn &in, float h,
                             CarForceOut &out, const M3 &R) {
    if (!CarAirEligible(k, s)) {
        CarForcesR(k, water, s, in, h, out, R);
        return;
    }
    TickInT<f32x1> inl{f32x1(in.gas), f32x1(in.brake), f32x1(in.steer), i32x1(static_cast<std::int32_t>(in.tickMs))};
    ForceOutT<f32x1> o;
    CarForcesAirT<f32x1>(k, water, s, inl, f32x1(h), o, detail::ToLane(R), MaskTrue<f32x1>());
    store(out.force, detail::ToV3(o.force));
    store(out.torque, detail::ToV3(o.torque));
    store(out.posePos, detail::ToV3(o.posePos));
    store(out.poseRot, detail::ToV3(o.poseRot.c0));
    store(out.poseRot + 3, detail::ToV3(o.poseRot.c1));
    store(out.poseRot + 6, detail::ToV3(o.poseRot.c2));
    out.arms = static_cast<std::uint32_t>(o.arms.v);
}
TW_INLINE void CarAirSubstep(const CarForceConsts &k, const WaterZone *water, TwinState &s, const CarTickIn &in, float h,
                             CarForceOut &out) {
    const M3 R = rot_from_quat(s.quat);
    CarForcesAirR(k, water, s, in, h, out, R);
    CarIntegrateR(k, s, out, h, R);
}

// Non-inlined entry; the caller holds a tw::math::FpGuard (one per API call, not per substep).
void CarSubstepNoGuard(const CarForceConsts &k, const WaterZone *water, TwinState &s, const CarTickIn &in, float h,
                       CarForceOut &out);

// Non-inlined airborne micro-step entry (L8), caller holds the FP guard (CarAirSubstep16Lanes: below).
void CarAirSubstepNoGuard(const CarForceConsts &k, const WaterZone *water, TwinState &s, const CarTickIn &in, float h,
                          CarForceOut &out);

// Non-inlined entry with the FP guard (FpGuard belongs at API entry, math.md §4).
void CarSubstepGuarded(const CarForceConsts &k, const WaterZone *water, TwinState &s, const CarTickIn &in, float h,
                       CarForceOut &out);

// Path key of a state for lane scheduling (m3-forces.md §5.4): lanes with equal keys take
// the same expensive arms (grounded block, burnout / donut arms, textured-surface bumps).
// bit 0 grounded, bits 1-2 burnout phase, bit 3 a contacting wheel on a fake-contact texture.
TW_INLINE std::uint32_t CarPathKey(const CarForceConsts &k, const TwinState &s) {
    std::uint32_t key = 0u;
    for (int w = 0; w < 4; ++w) {
        if (!s.contact[w]) continue;
        key |= 1u;
        if (k.byIdFakeUses[s.surface[w] & 31u] > 0.0f) key |= 8u;
    }
    return key | (static_cast<std::uint32_t>(s.brnPhase & 3u) << 1);
}

// ---- L-lane API (TwinLanes<L>) ----------------------------------------------------
template <int L> struct CarTickInL {
    alignas(64) float gas[L];
    alignas(64) float brake[L];
    alignas(64) float steer[L];
    alignas(64) std::uint32_t tickMs[L];
};
template <int L> struct CarForceOutL {
    alignas(64) float force[3][L];      // world
    alignas(64) float torque[3][L];     // car frame
    alignas(64) float posePos[3][L];    // substep-start pose
    alignas(64) float poseRot[9][L];    // columns basisX, basisY, basisZ
    alignas(64) std::uint32_t arms[L];
};

namespace detail {
template <int L> using LaneF = tw::math::vf<L>;
template <int L> TW_INLINE void StoreRot(float (&dst)[9][L], const M3L<LaneF<L>> &R) {
    R.c0.x.store(dst[0]); R.c0.y.store(dst[1]); R.c0.z.store(dst[2]);
    R.c1.x.store(dst[3]); R.c1.y.store(dst[4]); R.c1.z.store(dst[5]);
    R.c2.x.store(dst[6]); R.c2.y.store(dst[7]); R.c2.z.store(dst[8]);
}
template <int L> TW_INLINE M3L<LaneF<L>> LoadRot(const float (&src)[9][L]) {
    using F = LaneF<L>;
    return {v3l(F::load(src[0]), F::load(src[1]), F::load(src[2])), v3l(F::load(src[3]), F::load(src[4]), F::load(src[5])),
            v3l(F::load(src[6]), F::load(src[7]), F::load(src[8]))};
}
}  // namespace detail

// S1 + C1..C20 for the lanes whose bit is set in `active` (others untouched).
template <int L>
TW_INLINE void CarForcesL(const CarForceConsts &k, const WaterZone *water, TwinLanes<L> &s, const CarTickInL<L> &in,
                          const float (&h)[L], CarForceOutL<L> &out, std::uint32_t active = (L == 32 ? ~0u : (1u << L) - 1u)) {
    using F = detail::LaneF<L>;
    using I = LaneI<F>;
    TickInT<F> inl{F::load(in.gas), F::load(in.brake), F::load(in.steer),
                   I::load(reinterpret_cast<const std::int32_t *>(in.tickMs))};
    const M3L<F> R = RotFromQuatL(F::load(s.quat[0]), F::load(s.quat[1]), F::load(s.quat[2]), F::load(s.quat[3]));
    ForceOutT<F> o;
    CarForcesT<F>(k, water, s, inl, F::load(h), o, R, MaskOf<F>(active));
    St3(out.force, o.force);
    St3(out.torque, o.torque);
    St3(out.posePos, o.posePos);
    detail::StoreRot<L>(out.poseRot, o.poseRot);
    o.arms.store(reinterpret_cast<std::int32_t *>(out.arms));
}

// S2 for the lanes in `active`; the old rotation is out.poseRot (the force stage's).
template <int L>
TW_INLINE void CarIntegrateL(const CarForceConsts &k, TwinLanes<L> &s, const CarForceOutL<L> &f, const float (&h)[L],
                             std::uint32_t active = (L == 32 ? ~0u : (1u << L) - 1u), float (*postRot)[L] = nullptr) {
    using F = detail::LaneF<L>;
    ForceOutT<F> o;
    o.force = Ld3<F>(f.force);
    o.torque = Ld3<F>(f.torque);
    M3L<F> p;
    CarIntegrateT<F>(k, s, o, F::load(h), detail::LoadRot<L>(f.poseRot), MaskOf<F>(active), postRot ? &p : nullptr);
    if (postRot) detail::StoreRot<L>(*reinterpret_cast<float(*)[9][L]>(postRot), p);
}

template <int L>
TW_INLINE void CarSubstepL(const CarForceConsts &k, const WaterZone *water, TwinLanes<L> &s, const CarTickInL<L> &in,
                           const float (&h)[L], CarForceOutL<L> &out, std::uint32_t active = (L == 32 ? ~0u : (1u << L) - 1u)) {
    CarForcesL<L>(k, water, s, in, h, out, active);
    CarIntegrateL<L>(k, s, out, h, active);
}

// L-lane airborne micro-step: the lean path for the active lanes when all of them qualify
// (CarAirEligible), else the full path. Same bits as CarForcesL / CarSubstepL either way.
template <int L>
TW_INLINE void CarForcesAirL(const CarForceConsts &k, const WaterZone *water, TwinLanes<L> &s, const CarTickInL<L> &in,
                             const float (&h)[L], CarForceOutL<L> &out, std::uint32_t active = (L == 32 ? ~0u : (1u << L) - 1u)) {
    using F = detail::LaneF<L>;
    using I = LaneI<F>;
    const auto act = MaskOf<F>(active);
    if ((MaskBits<F>(act & ~CarAirEligibleT<F>(k, s)) & active) != 0u) {
        CarForcesL<L>(k, water, s, in, h, out, active);
        return;
    }
    TickInT<F> inl{F::load(in.gas), F::load(in.brake), F::load(in.steer), I::load(reinterpret_cast<const std::int32_t *>(in.tickMs))};
    const M3L<F> R = RotFromQuatL(F::load(s.quat[0]), F::load(s.quat[1]), F::load(s.quat[2]), F::load(s.quat[3]));
    ForceOutT<F> o;
    CarForcesAirT<F>(k, water, s, inl, F::load(h), o, R, act);
    St3(out.force, o.force);
    St3(out.torque, o.torque);
    St3(out.posePos, o.posePos);
    detail::StoreRot<L>(out.poseRot, o.poseRot);
    o.arms.store(reinterpret_cast<std::int32_t *>(out.arms));
}
template <int L>
TW_INLINE void CarAirSubstepL(const CarForceConsts &k, const WaterZone *water, TwinLanes<L> &s, const CarTickInL<L> &in,
                              const float (&h)[L], CarForceOutL<L> &out, std::uint32_t active = (L == 32 ? ~0u : (1u << L) - 1u)) {
    CarForcesAirL<L>(k, water, s, in, h, out, active);
    CarIntegrateL<L>(k, s, out, h, active);
}

// Non-inlined batch entries (forces.cpp); the caller holds the FpGuard. All lanes share `water` (one map).
void CarSubstep4(const CarForceConsts &k, const WaterZone *water, TwinLanes<4> &s, const CarTickInL<4> &in, const float (&h)[4],
                 CarForceOutL<4> &out, std::uint32_t active);
#if TW_HAVE_AVX2
void CarSubstep8(const CarForceConsts &k, const WaterZone *water, TwinLanes<8> &s, const CarTickInL<8> &in, const float (&h)[8],
                 CarForceOutL<8> &out, std::uint32_t active);
#endif
#if TW_HAVE_AVX512F
void CarSubstep16(const CarForceConsts &k, const WaterZone *water, TwinLanes<16> &s, const CarTickInL<16> &in,
                  const float (&h)[16], CarForceOutL<16> &out, std::uint32_t active);
void CarAirSubstep16Lanes(const CarForceConsts &k, const WaterZone *water, TwinLanes<16> &s, const CarTickInL<16> &in,
                          const float (&h)[16], CarForceOutL<16> &out, std::uint32_t active);   // CarAirSubstepL<16>
#endif

}  // namespace tw::model

#endif
