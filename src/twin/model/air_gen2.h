// Second-generation airborne regime (milestone M3): one whole contact-free airborne tick.
//
// design-gen2-e.md §2.2 and §3.7, debate/gen2-E-synthesis.md and gen2-D-synthesis.md (airborne row);
// docs/twin/m3-forces.md §6 describes it and its acceptance.
//
//   per substep, float, car frame: C14's air control (memory pins in FV's order, x3 when strong,
//     tau.z * C_airZ(|w.z|)), tau_air = tau (0.2 |tau| + 0.4), alpha = I^-1_body tau,
//     q <- q (1, h w / 2) unnormalised, w into the new frame by conj(dq) / |dq|^2, the w clamp;
//   per tick, double, rounded once: v and the position (per-(n, k) coefficients of the double
//     board's own split, built once per car), the COM correction (R_k - R_0) COM, rpm ramp and
//     clamp, shift cooldown, steer slew, suspension relax; q normalised once.
//
// The coefficients are the double board's recurrence applied to the basis (v0, gravity): an
// affine recurrence with tick-constant coefficients, so its closed form is exact in reals; in
// double the result is unbiased (D: in float, 1 - beta^n cancels into a bias 15-50x the
// population rms).
//
// AirTickEligible() tells whether a tick-start state qualifies; the caller adds what it alone
// knows: the tick is proved contact-free (S1's AtlasCollision::TickClear, or FV's records), it is
// not a respawn / race-start tick and not a finish-bracket tick. Otherwise the tick runs per
// substep on the twin's lean path (CarAirSubstep / CarSubstep).
//
// FV-free, header-only.
#ifndef TW_MODEL_AIR_GEN2_H
#define TW_MODEL_AIR_GEN2_H

#include <cmath>
#include <cstdint>
#include <cstring>

#include "forces.h"

namespace tw::model::gen2 {

constexpr std::uint32_t kAirMaxN = 16;   // tabulated substep counts; airborne ticks need n <= 13 (C7 cap + w clamp)

// After k of a tick's n substeps (k = 0..n): v_k = a v0 + ev, p_k = p0 + c v0 + ep (before the COM
// correction), T_k = t, suspension b0 - rest scales by s.
struct AirCoef {
    double a, c, t, s;
    double ev[3], ep[3];
    double vs, ih;   // the last step's absorb velocity factor (S_k - S_{k-1}) / h_k, and 1 / h_k
};

struct AirTables {
    double dt = 0.0;                          // the tick (double of the float dt)
    float dtF = 0.0f;
    std::uint32_t rowStart[kAirMaxN + 1] = {};   // first entry of n's row (k = 0..n)
    float hF[kAirMaxN + 1][kAirMaxN] = {};   // per n: the substep durations (float), FV's split
    double hD[kAirMaxN + 1][kAirMaxN] = {};  // per n: the double board's split
    // [lag][entry]: lag = 1 when substep 1 still uses the grounded fs / lds (10 §1.5)
    AirCoef coef[2][(kAirMaxN + 1) * (kAirMaxN + 2) / 2] = {};
    double gAir[3] = {}, gGround[3] = {};     // gravity accelerations (fs m / m), double
    double lamAir = 0.0;                      // zoneLinDamp lds_air / m
    double gMaxNorm = 0.0;                    // max(|gAir|, |gGround|)
    double invMass = 0.0, follow = 0.0, rest = 0.0;
    bool airZUnit = false;   // C_airZ is the constant 1 (StadiumCar: one key): the substep skips it
};

namespace detail {
// One row: n substeps (the double board's split: n - 1 steps of dt / n, the last the remainder by
// repeated subtraction), the first one grounded when lag.
inline void AirRow(const AirTables &t, std::uint32_t n, bool lag, const double *h, AirCoef *row) {
    AirCoef e{};
    e.a = 1.0;
    e.s = 1.0;
    row[0] = e;
    for (std::uint32_t j = 1; j <= n; ++j) {
        const double hj = h[j - 1];
        const bool grounded = lag && j == 1u;
        const double *g = grounded ? t.gGround : t.gAir;
        const double lam = grounded ? 0.0 : t.lamAir;
        // p_j = p_{j-1} + v_{j-1} h_j ; v_j = v_{j-1} + (g - lam v_{j-1}) h_j
        e.c += e.a * hj;
        for (int i = 0; i < 3; ++i) e.ep[i] += e.ev[i] * hj;
        const double beta = 1.0 - lam * hj;
        e.a *= beta;
        for (int i = 0; i < 3; ++i) e.ev[i] = e.ev[i] * beta + g[i] * hj;
        e.t += hj;
        const double sPrev = e.s;
        e.s *= 1.0 - hj * t.follow;
        e.ih = 1.0 / hj;
        e.vs = (e.s - sPrev) / hj;
        row[j] = e;
    }
}
inline void AirSplit(double dt, std::uint32_t n, double *h) {
    double remaining = dt;
    if (n > 1u) {
        const double split = dt / static_cast<double>(n);
        for (std::uint32_t i = 0; i + 1u < n; ++i) {
            h[i] = split;
            remaining = remaining - split;
        }
    }
    h[n > 0u ? n - 1u : 0u] = remaining;
}
inline void AirSplitF(float dt, std::uint32_t n, float *h) {
    float remaining = dt;
    if (n > 1u) {
        const float split = dt / static_cast<float>(n);
        for (std::uint32_t i = 0; i + 1u < n; ++i) {
            h[i] = split;
            remaining = remaining - split;
        }
    }
    h[n > 0u ? n - 1u : 0u] = remaining;
}
}  // namespace detail

inline void BuildAirTables(const CarForceConsts &k, float dt, AirTables *t) {
    *t = AirTables{};
    t->dt = static_cast<double>(dt);
    t->dtF = dt;
    t->invMass = 1.0 / static_cast<double>(k.mass);
    const double fsA = static_cast<double>(k.fsAirborne) * static_cast<double>(k.mass);
    const double fsG = static_cast<double>(k.fsGrounded) * static_cast<double>(k.mass);
    double nA = 0.0, nG = 0.0;
    for (int i = 0; i < 3; ++i) {
        t->gAir[i] = static_cast<double>(k.gravity[i]) * fsA * t->invMass;
        t->gGround[i] = static_cast<double>(k.gravity[i]) * fsG * t->invMass;
        nA += t->gAir[i] * t->gAir[i];
        nG += t->gGround[i] * t->gGround[i];
    }
    t->gMaxNorm = std::sqrt(nA > nG ? nA : nG);
    t->lamAir = static_cast<double>(k.zoneLinDamp) * static_cast<double>(k.ldsAirborne) * t->invMass;
    t->follow = static_cast<double>(k.followCoef);
    t->rest = static_cast<double>(k.restAbsorb);
    const CompiledCurve &cz = k.curve[kCurveAirZScale];
    t->airZUnit = true;
    for (std::uint32_t r = 0; r <= cz.thresholds && r <= 15u; ++r) t->airZUnit = t->airZUnit && cz.slope[r] == 0.0f && cz.y0[r] == 1.0f;
    std::uint32_t at = 0u;
    for (std::uint32_t n = 1; n <= kAirMaxN; ++n) {
        t->rowStart[n] = at;
        detail::AirSplit(t->dt, n, t->hD[n]);
        detail::AirSplitF(dt, n, t->hF[n]);
        detail::AirRow(*t, n, false, t->hD[n], &t->coef[0][at]);
        detail::AirRow(*t, n, true, t->hD[n], &t->coef[1][at]);
        at += n + 1u;
    }
}

// Why a tick is not in the regime (0 = it is).
enum AirReject : std::uint32_t {
    kAirOk = 0,
    kAirFrozen,          // C1
    kAirWheelContact,    // a wheel flag at the tick start: substep 1 is grounded
    kAirDonut,           // CircularDrift
    kAirFreewheel,       // coasting friction (-v_hat) even airborne
    kAirPedalOff,        // coasting friction (-v_hat), judged on R ? brake : gas
    kAirLateralWindow,   // lateral slowdown (-v_hat C(|v|))
    kAirWater,           // in water at the start, or the tick may reach the water
    kAirSpeedCap,        // C7 may bind inside the tick
    kAirSubsteps,        // n outside the tables
    kAirRejectCount,
};

namespace detail {
TW_INLINE double Len3d(const double *v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
}

// The tick-start checks (design-gen2-e.md §2.2 conditions, D's C7 and lag fixes). `n` = FV's
// substep count for the tick.
inline std::uint32_t AirTickEligible(const CarForceConsts &k, const AirTables &t, const WaterZone *water,
                                     const TwinState &s, const CarTickIn &in, std::uint32_t n) {
    if (k.frozenByWaterBox || s.speedBlocked || s.speedBlocked2) return kAirFrozen;
    if (s.contact[0] | s.contact[1] | s.contact[2] | s.contact[3]) return kAirWheelContact;
    if (s.brnPhase == kCircularDrift) return kAirDonut;
    if (s.freewheel) return kAirFreewheel;
    const float pedal = s.reverse ? in.brake : in.gas;
    if (pedal < 1.0e-5f) return kAirPedalOff;
    {   // Friction(): the window test with this tick's T (the lateral flag would set the tick to T)
        const std::uint32_t T = in.tickMs;
        const std::uint32_t lt = s.latContact ? T : s.latTickMs;
        if (lt <= T && T - lt < k.lateralSlowDownWindow) return kAirLateralWindow;
    }
    if (n == 0u || n > kAirMaxN || t.dtF == 0.0f) return kAirSubsteps;
    const double v0[3] = {s.vel[0], s.vel[1], s.vel[2]};
    if (k.linearSpeedCap * k.linearSpeedCap > ::tw::model::detail::kEps2) {
        // |v_k| <= |v0| + T_k |g| (drag only shrinks it); the clamp is a float compare of |v|^2 with cap^2
        const double bound = detail::Len3d(v0) + t.dt * t.gMaxNorm;
        if (!(bound < static_cast<double>(k.linearSpeedCap) * (1.0 - 1.0e-5))) return kAirSpeedCap;
    }
    if (s.inWater) return kAirWater;
    if (water != nullptr && water->enabled) {
        // In water needs surface - lowerY > 0.5 at a substep-start pose. lowerY >= COM.y - r with r the
        // water box's reach from the COM; COM.y is concave over the tick (gravity down), so its
        // minimum over the substep starts is at k = 0 or k = n - 1.
        if (!(k.gravity[1] < 0.0f) || k.gravity[0] != 0.0f || k.gravity[2] != 0.0f) return kAirWater;
        double r2 = 0.0, hb2 = 0.0;
        for (int i = 0; i < 3; ++i) {
            const double d = static_cast<double>(k.waterBoxCenter[i]) - static_cast<double>(k.com[i]);
            r2 += d * d;
            hb2 += static_cast<double>(k.waterBoxHalf[i]) * static_cast<double>(k.waterBoxHalf[i]);
        }
        const double reach = std::sqrt(r2) + std::sqrt(hb2) + 1.0e-3;
        const bool lag = s.groundedPrev != 0u;
        const AirCoef &e = t.coef[lag ? 1 : 0][t.rowStart[n] + (n - 1u)];
        const double y0 = static_cast<double>(s.pos[1]), yk = y0 + e.c * v0[1] + e.ep[1];
        const double comY = 1.2 * std::sqrt(static_cast<double>(k.com[0]) * k.com[0] + static_cast<double>(k.com[1]) * k.com[1] +
                                            static_cast<double>(k.com[2]) * k.com[2]);   // |R COM| bound (both poses)
        const double low = (y0 < yk ? y0 : yk) - comY - reach;
        if (!(static_cast<double>(water->surfaceHeight) - low < 0.5 - 1.0e-3)) return kAirWater;
    }
    return kAirOk;
}

namespace detail {

// The tick-constant discrete steps of an airborne substep (C12 burnout phases with the airborne
// TimedSpin -> ExitFade, C17 turbo expiry): FV's operations on the state. T is constant over the
// tick, so the second application reaches the fixed point.
TW_INLINE void AirDiscrete(const CarForceConsts &k, TwinState &s, std::uint32_t T) {
    if (s.brnPhase == kInactive) {   // AdvanceBurnoutPhases changes nothing; slipSeen = false
        s.slipActive = 0u;
        if (s.trbType != 0u && s.trbEndMs < T) s.trbType = kTurboOff;
        return;
    }
    using F = f32x1;
    using I = LaneI<F>;
    const I Ti(static_cast<std::int32_t>(T));
    const auto m = MaskTrue<F>();
    const auto slipSeen = ::tw::model::detail::AdvanceBurnoutPhases<F>(k, s, Ti, m);
    if (s.brnPhase == kTimedSpin) {
        s.brnExitStartMs = T;
        s.brnPhase = kExitFade;
    }
    s.slipActive = slipSeen.v ? 1u : 0u;
    if (s.trbType != 0u && s.trbEndMs < T) s.trbType = kTurboOff;
}

// The angular carry of the tick (car frame).
struct AirAng {
    float w[3];      // angular velocity, car frame
    float q[4];      // x y z w, unnormalised
    float mem[3];    // air-control memory (car frame)
};

// Rotations in double (the per-tick conversions): FV's GmMat3::Set(GmQuat) formula on q as given
// (the board's R for the stored, float-normalised tick-start quaternion) or on q / |q|. Named scalars,
// so they stay in registers (an array spilled with scalar stores and reloaded as vectors stalls on
// store forwarding).
struct M3d {
    double xx, xy, xz, yx, yy, yz, zx, zy, zz;   // row, column: R v = (xx vx + xy vy + xz vz, ...)
};
TW_INLINE M3d RotD(double x, double y, double z, double w) {
    const double x2 = x * 2.0, y2 = y * 2.0, z2 = z * 2.0;
    const double xw = w * x2, yw = w * y2, zw = w * z2;
    const double xx = x * x2, xy = x * y2, xz = x * z2;
    const double yy = y * y2, yz = y * z2, zz = z * z2;
    // columns (basis x, y, z) as vec3.h rot_from_quat
    return M3d{(1.0 - yy) - zz, xy - zw, xz + yw,
               xy + zw, (1.0 - xx) - zz, yz - xw,
               xz - yw, yz + xw, (1.0 - xx) - yy};
}
TW_INLINE void UnitD(const float *q, double *u) {
    const double a = q[0], b = q[1], c = q[2], d = q[3];
    const double inv = 1.0 / std::sqrt(((a * a + d * d) + b * b) + c * c);
    u[0] = a * inv, u[1] = b * inv, u[2] = c * inv, u[3] = d * inv;
}
TW_INLINE void MulD(const M3d &R, double x, double y, double z, double *o) {   // R v
    o[0] = (R.xx * x + R.xy * y) + R.xz * z;
    o[1] = (R.yx * x + R.yy * y) + R.yz * z;
    o[2] = (R.zx * x + R.zy * y) + R.zz * z;
}
TW_INLINE void MulTD(const M3d &R, double x, double y, double z, double *o) {  // R^T v
    o[0] = (R.xx * x + R.yx * y) + R.zx * z;
    o[1] = (R.xy * x + R.yy * y) + R.zy * z;
    o[2] = (R.xz * x + R.yz * y) + R.zz * z;
}
// R(q / |q|) without the square root: the formula on q is |q|^2 R(q / |q|) + (1 - |q|^2) I.
TW_INLINE M3d RotUnitD(const float *q) {
    const double a = q[0], b = q[1], c = q[2], d = q[3];
    M3d m = RotD(a, b, c, d);
    const double s2 = ((a * a + d * d) + b * b) + c * c;
    const double inv = 1.0 / s2, dd = s2 - 1.0;
    m.xx = (m.xx + dd) * inv, m.xy *= inv, m.xz *= inv;
    m.yx *= inv, m.yy = (m.yy + dd) * inv, m.yz *= inv;
    m.zx *= inv, m.zy *= inv, m.zz = (m.zz + dd) * inv;
    return m;
}

// The car-frame air-control pins of C14 (FV's order): updates mem, returns the pinned w and the x3 flag.
struct AirPins {
    bool steerPos, steerNeg, brakeOn;
    float ySwitch;
};
TW_INLINE bool AirPin(const AirPins &p, float *mem, float &w0, float &w1) {
    float my = mem[1];
    const bool condA = (p.steerPos && my < 0.0f) || (p.steerNeg && 0.0f < my);
    const bool strongA = condA && p.ySwitch < std::fabs(w1);
    const bool strongB = !condA && ((p.steerPos && 0.0f < my) || (p.steerNeg && my < 0.0f));
    if (strongA || !condA) my = w1;
    const float mx = (p.brakeOn && 0.0f < mem[0]) ? 0.0f : w0;
    mem[0] = mx;
    mem[1] = my;
    w0 = mx;
    w1 = my;
    return strongA || strongB;
}

// conj(dq) v dq / |dq|^2 with dq = (e, 1): v - f (e x v - e x (e x v)), f = 2 / (1 + e^2). As v plus a
// small correction (the form v (1 - e^2) / (1 + e^2) + ... rounds 1 -+ e^2 to float and, for e^2 near
// an ulp of 1, always errs the same way).
TW_INLINE void AirFrameRot(float ex, float ey, float ez, float f, float &v0, float &v1, float &v2) {
    const float cx = ey * v2 - ez * v1, cy = ez * v0 - ex * v2, cz = ex * v1 - ey * v0;   // e x v
    const float ux = ey * cz - ez * cy, uy = ez * cx - ex * cz, uz = ex * cy - ey * cx;   // e x (e x v)
    v0 = v0 - f * (cx - ux);
    v1 = v1 - f * (cy - uy);
    v2 = v2 - f * (cz - uz);
}
TW_INLINE void AirQuatStep(float *q, float hx, float hy, float hz) {   // q <- q (e, 1)
    const float qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    q[0] = qw * hx + qx + (qy * hz - qz * hy);
    q[1] = qw * hy + qy + (qz * hx - qx * hz);
    q[2] = qw * hz + qz + (qx * hy - qy * hx);
    q[3] = qw - (qx * hx + qy * hy + qz * hz);
}

}  // namespace detail

// The tick's real motion, for the caller's re-check of a whole-tick contact-free proof (E's
// AtlasCollision::TickClear inflates the tick-start shape boxes by |v0| dt + aMax dt^2 / 2 +
// |w0| dt R_car + 0.25 m): the proof covers the tick if
//   comTravel <= |v0| dt + aMax dt^2 / 2   (true whenever |g fs| <= aMax: 25 and 30 m/s^2 for StadiumCar)
//   wMax dt (R_car + |COM|) + absorbTravel <= |w0| dt R_car + 0.25
// (the car turns about its COM, R_car is measured from the car origin).
struct AirTickMotion {
    float comTravel = 0.0f;      // max_k |COM_k - COM_0| (bound: c_n |v0| + |ep_n|)
    float wMax = 0.0f;           // max over the substeps of |w| (the start included)
    float absorbTravel = 0.0f;   // max over the wheels of |a_n - a_0| (the relax is monotone)
};

// One tick of n substeps in the regime (AirTickEligible() == kAirOk and the caller's conditions).
// after (optional, n entries): the state after each substep, for the per-substep yardstick.
// motion (optional): the tick's real motion (AirTickMotion).
//
// The board's first substep rotates with R(q0) of the stored quaternion (|q0| = 1 to float precision
// only) and the later ones with R(q_k / |q_k|). Substep 1 therefore runs its velocity update in the
// world frame as the board does (w_w1 = w_w(pinned) + h R(q0) alpha, back with R(q0 / |q0|)^T), the
// others in the car frame, where the step is
//   n' = w' + Z m,   Z = R(dq)^T (h I^-1 tau_s),   m = quad s sqrt(|w|^2) + lin
// (w' the pinned w, tau_s = -s w with s = 1 or 3; R(dq)^T w' = w' because dq turns about w'), so the
// chain from w to the next w is one square root, two multiply-adds and the w' + Z m update.
inline void AirTick(const CarForceConsts &k, const AirTables &t, TwinState &s, const CarTickIn &in, std::uint32_t n,
                    TwinState *after = nullptr, AirTickMotion *motion = nullptr) {
    using namespace detail;
    const std::uint32_t T = in.tickMs;
    const bool lag = s.groundedPrev != 0u;
    const AirCoef *row = &t.coef[lag ? 1 : 0][t.rowStart[n]];
    const float *hf = t.hF[n];
    const double *hd = t.hD[n];

    // tick start: R(q0) as the board forms it (q0 as stored), the car-frame angular velocity
    const double q0d[4] = {s.quat[0], s.quat[1], s.quat[2], s.quat[3]};
    const M3d R0 = RotD(q0d[0], q0d[1], q0d[2], q0d[3]);
    const double s20 = ((q0d[0] * q0d[0] + q0d[3] * q0d[3]) + q0d[1] * q0d[1]) + q0d[2] * q0d[2];
    const double ww0[3] = {s.angVel[0], s.angVel[1], s.angVel[2]};
    // R(q0) in float for the O(delta) terms, by columns: R0f[3 j + i] = R(i, j), so (R^T v)_j = R0f[3 j ..] . v
    const float R0f[9] = {static_cast<float>(R0.xx), static_cast<float>(R0.yx), static_cast<float>(R0.zx),
                          static_cast<float>(R0.xy), static_cast<float>(R0.yy), static_cast<float>(R0.zy),
                          static_cast<float>(R0.xz), static_cast<float>(R0.yz), static_cast<float>(R0.zz)};
    const float d20f = static_cast<float>(s20 - 1.0);
    AirAng g;
    float start0[3];   // R(q0 / |q0|)^T w_w0 = (R(q0)^T w_w0 + delta w_w0) / s2
    {
        double wl[3];
        MulTD(R0, ww0[0], ww0[1], ww0[2], wl);
        for (int i = 0; i < 3; ++i) {
            g.w[i] = static_cast<float>(wl[i]);
            start0[i] = static_cast<float>((wl[i] + (s20 - 1.0) * ww0[i]) / s20);
        }
        std::memcpy(g.q, s.quat, sizeof g.q);
        std::memcpy(g.mem, s.airMem, sizeof g.mem);
    }
    const bool refresh0 = s.airRefresh != 0u;
    const bool window = T - s.airMemTickMs < k.airMemWindow;     // memTick moves only on a wheel contact
    const AirPins pins{1.0e-5f < in.steer, in.steer < -1.0e-5f, 1.0e-5f < in.brake, k.airYSwitch};
    const float quad = k.airTorqueQuad, lin = k.airTorqueLin;
    const float D0 = k.inertiaInvBody[0], D1 = k.inertiaInvBody[1], D2 = k.inertiaInvBody[2];
    const float angDamp = -k.zoneAngDamp * k.angularDampingScale;
    const float wMax = k.maxAngularSpeed, wMax2 = wMax * wMax;
    // C_airZ(|w.z|): the constant 1 for StadiumCar (one key); else its region is hoisted per tick
    const CompiledCurve &cz = k.curve[kCurveAirZScale];
    const bool zUnit = t.airZUnit;
    std::uint32_t zr = 0u;
    float zLo = INFINITY, zHi = -INFINITY;   // empty: the first use picks the region
    auto airZ = [&](float az) {
        if (!(zLo < az && az <= zHi)) {
            zr = CurveRegion(cz, az);
            zLo = zr > 0u ? cz.thr[zr - 1u] : -INFINITY;
            zHi = zr < 16u ? cz.thr[zr] : INFINITY;
        }
        return cz.y0[zr] + cz.slope[zr] * (az - cz.x0[zr]);
    };

    // the start values the closed forms need
    const double v0[3] = {s.vel[0], s.vel[1], s.vel[2]};
    const double p0[3] = {s.pos[0], s.pos[1], s.pos[2]};
    const double com[3] = {k.com[0], k.com[1], k.com[2]};
    double c0[3];
    MulD(R0, com[0], com[1], com[2], c0);
    double b0[4], a0[4];
    for (int w = 0; w < 4; ++w) {
        a0[w] = s.absorb[w];
        b0[w] = static_cast<double>(s.absorb[w]) - static_cast<double>(s.maxRepl[w]);
    }
    const double rpm0 = s.rpm, sc0 = s.shiftCooldown, cur0 = s.steerCur;
    const float steer = in.steer;
    const bool active = (s.reverse ? in.brake : in.gas) > ::tw::model::detail::kGate;
    // the discrete steps after 1 and after >= 2 substeps (only the yardstick needs both)
    TwinState disc1, disc2;
    if (after != nullptr) {
        disc1 = s;
        AirDiscrete(k, disc1, T);
        disc2 = disc1;
        AirDiscrete(k, disc2, T);
    }

    float qPrev[4];
    std::memcpy(qPrev, g.q, sizeof qPrev);

    // The state after k substeps from the carry (the angular chain at k, q_{k-1}) and the closed forms.
    auto finish = [&](std::uint32_t kk, TwinState &o) {
        const AirCoef &e = row[kk];
        const AirCoef &ep = row[kk - 1u];
        // q_k / |q_k| once (the output quaternion and R_k)
        double qu[4];
        UnitD(g.q, qu);
        const M3d Rk = RotD(qu[0], qu[1], qu[2], qu[3]);
        const M3d Rp = kk == 1u ? R0 : RotUnitD(qPrev);
        double ck[3], vp[3];
        MulD(Rk, com[0], com[1], com[2], ck);
        for (int i = 0; i < 3; ++i) {
            o.pos[i] = static_cast<float>(p0[i] + e.c * v0[i] + e.ep[i] - (ck[i] - c0[i]));
            o.vel[i] = static_cast<float>(e.a * v0[i] + e.ev[i]);
            vp[i] = ep.a * v0[i] + ep.ev[i];
        }
        {
            for (int i = 0; i < 4; ++i) o.quat[i] = static_cast<float>(qu[i]);
            double ww[3];
            MulD(Rk, g.w[0], g.w[1], g.w[2], ww);
            for (int i = 0; i < 3; ++i) o.angVel[i] = static_cast<float>(ww[i]);
        }
        std::memcpy(o.airMem, g.mem, sizeof g.mem);
        // the last substep's force-time locals: R_{k-1}^T v_{k-1}, R_{k-1}^T (g fs m - zld lds v_{k-1}) / divisor
        const bool lastGrounded = lag && kk == 1u;
        const double *ga = lastGrounded ? t.gGround : t.gAir;
        const double lam = lastGrounded ? 0.0 : t.lamAir;
        const double m = static_cast<double>(k.mass);
        double f[3], ls[3], lf[3];
        for (int i = 0; i < 3; ++i) f[i] = (ga[i] - lam * vp[i]) * m;
        MulTD(Rp, vp[0], vp[1], vp[2], ls);
        MulTD(Rp, f[0], f[1], f[2], lf);
        const double div = static_cast<double>(k.invForceDivisor);
        for (int i = 0; i < 3; ++i) {
            o.localSpeedPrev[i] = static_cast<float>(ls[i]);
            o.prevForce[i] = static_cast<float>(lf[i] * div);
        }
        // suspension: a_k = rest + S_k (b0 - rest), absorb_vel from the last step
        for (int w = 0; w < 4; ++w) {
            const double db = b0[w] - t.rest;
            const double ak = t.rest + e.s * db;
            // (a_k - a_{k-1}) / h_k: (S_k - S_{k-1}) / h_k (b0 - rest), except after 1 substep with a
            // collision push-out still pending (b0 != a0)
            const double va = (kk >= 2u || b0[w] == a0[w]) ? e.vs * db : (ak - a0[w]) * e.ih;
            o.absorb[w] = static_cast<float>(ak);
            o.absorbVel[w] = static_cast<float>(va);
            o.maxRepl[w] = 0.0f;
        }
        // engine blocked airborne: rpm ramps, clamped once; the cooldown counts down while > 0
        const EngineConsts &en = k.engine;
        double rpm = active ? rpm0 + static_cast<double>(en.airRise) * e.t : rpm0 - static_cast<double>(en.airFall) * e.t;
        rpm = !(rpm <= 0.0) ? (!(static_cast<double>(en.rpmMax) > rpm) ? static_cast<double>(en.rpmMax) : rpm) : 0.0;
        o.rpm = static_cast<float>(rpm);
        double sc = sc0;
        for (std::uint32_t j = 0; j < kk && sc > 0.0; ++j) sc -= hd[j];
        o.shiftCooldown = static_cast<float>(sc);
        // steer slew toward the input, no overshoot
        if (k.slewRate <= 0.0f) {
            o.steerCur = steer;
        } else {
            const double d = static_cast<double>(steer) - cur0, stp = static_cast<double>(k.slewRate) * e.t;
            o.steerCur = std::fabs(d) <= stp ? steer : static_cast<float>(cur0 + (d < 0.0 ? -stp : stp));
        }
        // C4 / C19 / C20
        o.groundedPrev = 0u;
        o.inWater = 0u;
        for (int w = 0; w < 4; ++w) {
            o.contact[w] = 0u;
            o.surface[w] = 0u;
        }
        std::memset(o.normalSum, 0, sizeof o.normalSum);
        std::memset(o.latestPoint, 0, sizeof o.latestPoint);
        o.bodyContact = 0u;
        o.airRefresh = 0u;
        o.latContact = 0u;
    };
    auto clamp = [&](float &n0, float &n1, float &n2) {
        if (wMax > 0.0f) {
            const float l2 = (n0 * n0 + n1 * n1) + n2 * n2;
            if (l2 > wMax2) {
                const float c = wMax / std::sqrt(l2);
                n0 *= c;
                n1 *= c;
                n2 *= c;
            }
        }
    };

    float w2Max = (g.w[0] * g.w[0] + g.w[1] * g.w[1]) + g.w[2] * g.w[2];
    // ---- substep 1: the board's world-frame update with R(q0)
    {
        const float h = hf[0];
        float w0 = g.w[0], w1 = g.w[1], w2 = g.w[2];
        float t0 = -w0, t1 = -w1, t2 = -w2;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
        if (angDamp != 0.0f) {
            d0 = w0 * angDamp;
            d1 = w1 * angDamp;
            d2 = w2 * angDamp;
        }
        bool pinned = false;
        if (refresh0) {
            g.mem[0] = w0;
            g.mem[1] = w1;
            g.mem[2] = w2;
        } else if (window) {
            pinned = true;
            if (AirPin(pins, g.mem, w0, w1)) {
                t0 *= 3.0f;
                t1 *= 3.0f;
                t2 *= 3.0f;
            }
            if (!zUnit) t2 *= airZ(std::fabs(w2));
        }
        const float L = std::sqrt(t2 * t2 + (t0 * t0 + t1 * t1));
        if (L >= 1.0e-5f) {
            const float m = quad * L + lin;   // tau_hat (quad |tau|^2 + lin |tau|) = tau (quad |tau| + lin)
            d0 += t0 * m;
            d1 += t1 * m;
            d2 += t2 * m;
        }
        // The board: w_w1 = (pinned ? R(q0) w' : w_w0) + h R(q0) alpha, then w_b = R(q0 / |q0|)^T w_w1.
        // With R(q0) = s2 R^ + (1 - s2) I (s2 = |q0|^2 = 1 + delta, R^ = R(q0 / |q0|)) that is, to
        // O(delta^2) ~ 1e-14:  B + h alpha + delta h (alpha - R(q0)^T alpha),  B = pinned ? w' + delta (w' -
        // R(q0)^T w') : R^T w_w0 (start0, double).
        const float ax = D0 * d0, ay = D1 * d1, az = D2 * d2;
        const float dl = d20f, dlh = d20f * h;
        float b0w, b1w, b2w;
        if (pinned) {
            b0w = w0 + dl * (w0 - ((R0f[0] * w0 + R0f[1] * w1) + R0f[2] * w2));
            b1w = w1 + dl * (w1 - ((R0f[3] * w0 + R0f[4] * w1) + R0f[5] * w2));
            b2w = w2 + dl * (w2 - ((R0f[6] * w0 + R0f[7] * w1) + R0f[8] * w2));
        } else {
            b0w = start0[0];
            b1w = start0[1];
            b2w = start0[2];
        }
        float n0 = (b0w + h * ax) + dlh * (ax - ((R0f[0] * ax + R0f[1] * ay) + R0f[2] * az));
        float n1 = (b1w + h * ay) + dlh * (ay - ((R0f[3] * ax + R0f[4] * ay) + R0f[5] * az));
        float n2 = (b2w + h * az) + dlh * (az - ((R0f[6] * ax + R0f[7] * ay) + R0f[8] * az));
        if (1.0e-10f < (w1 * w1 + w0 * w0) + w2 * w2) {
            const float hh = 0.5f * h;
            const float hx = hh * w0, hy = hh * w1, hz = hh * w2;
            AirQuatStep(g.q, hx, hy, hz);
            AirFrameRot(hx, hy, hz, 2.0f / (1.0f + (hx * hx + hy * hy + hz * hz)), n0, n1, n2);
        }
        clamp(n0, n1, n2);
        if (motion != nullptr) w2Max = std::fmax(w2Max, (n0 * n0 + n1 * n1) + n2 * n2);
        g.w[0] = n0;
        g.w[1] = n1;
        g.w[2] = n2;
        if (after != nullptr && n > 1u) {
            after[0] = disc1;
            finish(1u, after[0]);
        }
    }

    // ---- substeps 2..n in the car frame
    const bool fast = zUnit && angDamp == 0.0f;
    for (std::uint32_t j = 1; j < n; ++j) {
        const float h = hf[j];
        std::memcpy(qPrev, g.q, sizeof qPrev);
        const float w0 = g.w[0], w1 = g.w[1], w2 = g.w[2];
        float p0w = w0, p1w = w1;                                 // w' (pinned)
        float sc = 1.0f;
        if (window && AirPin(pins, g.mem, p0w, p1w)) sc = 3.0f;
        const float z = fast ? 1.0f : (window && !zUnit ? airZ(std::fabs(w2)) : 1.0f);
        const float w2z = fast ? w2 : w2 * z;
        // |tau| = s sqrt(w0^2 + w1^2 + (z w2)^2); m = quad |tau| + lin; h alpha = Q m, Q = -h s I^-1 (w0, w1, z w2)
        const float Lr = std::sqrt(w2z * w2z + (w0 * w0 + w1 * w1));
        const float hs = -h * sc;
        float q0 = hs * D0 * w0, q1 = hs * D1 * w1, q2 = hs * D2 * w2z;
        float pd0 = 0.0f, pd1 = 0.0f, pd2 = 0.0f;                // h I^-1 (angular damping), pre-pin w
        if (!fast && angDamp != 0.0f) {
            pd0 = h * D0 * (w0 * angDamp);
            pd1 = h * D1 * (w1 * angDamp);
            pd2 = h * D2 * (w2 * angDamp);
        }
        const float w2p = w2;
        if (1.0e-10f < (p1w * p1w + p0w * p0w) + w2p * w2p) {
            const float hh = 0.5f * h;
            const float hx = hh * p0w, hy = hh * p1w, hz = hh * w2p;
            AirQuatStep(g.q, hx, hy, hz);
            const float f = 2.0f / (1.0f + (hx * hx + hy * hy + hz * hz));
            AirFrameRot(hx, hy, hz, f, q0, q1, q2);
            if (!fast) AirFrameRot(hx, hy, hz, f, pd0, pd1, pd2);
        }
        const float L = sc * Lr;
        float n0 = p0w + pd0, n1 = p1w + pd1, n2 = w2p + pd2;
        if (L >= 1.0e-5f) {
            const float m = (quad * sc) * Lr + lin;
            n0 = n0 + q0 * m;
            n1 = n1 + q1 * m;
            n2 = n2 + q2 * m;
        }
        clamp(n0, n1, n2);
        if (motion != nullptr) w2Max = std::fmax(w2Max, (n0 * n0 + n1 * n1) + n2 * n2);
        g.w[0] = n0;
        g.w[1] = n1;
        g.w[2] = n2;
        if (after != nullptr && j + 1u < n) {
            after[j] = disc2;
            finish(j + 1u, after[j]);
        }
    }
    AirDiscrete(k, s, T);
    if (n > 1u) AirDiscrete(k, s, T);
    if (motion != nullptr) {
        const AirCoef &e = row[n];
        double c2 = 0.0;
        for (int i = 0; i < 3; ++i) c2 += e.ep[i] * e.ep[i];
        motion->comTravel = static_cast<float>(e.c * std::sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]) + std::sqrt(c2));
        motion->wMax = std::sqrt(w2Max) * (1.0f + 1.0e-6f);
        double at = 0.0;
        for (int w = 0; w < 4; ++w) at = std::fmax(at, std::fabs((e.s - 1.0) * (b0[w] - t.rest) + (b0[w] - a0[w])));
        motion->absorbTravel = static_cast<float>(at) * (1.0f + 1.0e-6f);
    }
    finish(n, s);
    if (after != nullptr) after[n - 1u] = s;
}

}  // namespace tw::model::gen2

#endif
