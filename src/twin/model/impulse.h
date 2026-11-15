// Twin model: the car's own contact impulse (board 10 §2.16.3, board 20 §3.6).
//
// Shared by the force kernel (C2 fake contacts) and the contact response
// (M5). Same semantics as FV SDynaMath::ComputeImpulse +
// CSceneVehicleCar::AddVehicleImpulse(local, point) (wheels.cpp:35-86,
// 172-222):
//   a = I^-1_body (r x n), J = n (e - 1)(n . u) / (1/m + n . (a x r))   (local)
//   v' = v + R J / m; v' = 0 if |v'|^2 > |v|^2 and the growth exceeds the limit
//   omega += R I^-1_body (r x J) * angularScale (y also * angularYScale); |omega| <= max
// The angular term is evaluated in the car frame (FV rotates r and J to world
// and uses the stored world inverse inertia, = R D R^T within 2e-6): same
// value up to float noise, and no cancellation at large |pos|.
#ifndef TW_MODEL_IMPULSE_H
#define TW_MODEL_IMPULSE_H

#include "car_force_consts.h"
#include "vec3.h"

namespace tw::model {

// e = kept fraction of the normal approach speed (-tuning value, 0.8 / 0.5).
TW_INLINE V3 ComputeImpulseLocal(const CarForceConsts &k, float e, V3 u, V3 n, V3 lever) {
    const V3 ln = cross(lever, n);
    const V3 a = V3{ln.x * k.inertiaInvBody[0], ln.y * k.inertiaInvBody[1], ln.z * k.inertiaInvBody[2]};
    const float numerator = (e - 1.0f) * dot(u, n);
    const float denom = k.invMass + dot(n, cross(a, lever));
    return n * (numerator / denom);
}

// Clamp |omega| to the configured maximum (double division in FV: one float division here).
TW_INLINE V3 ClampAngular(V3 w, float maxSpeed) {
    if (maxSpeed > 0.0f) {
        const float l2 = len2(w);
        if (l2 > maxSpeed * maxSpeed) w = w * (maxSpeed / tw::math::sqrt(l2));
    }
    return w;
}

// Point impulse J at car-local point p. v and omega are world-frame state.
TW_INLINE void ApplyPointImpulseLocal(const CarForceConsts &k, const M3 &R, V3 J, V3 p, V3 &v, V3 &w) {
    const V3 Jw = mul(R, J);
    V3 v2 = v + Jw * k.invMass;
    const float before = len2(v), after = len2(v2);
    if (before < after && k.growthLimitSq < after - before) v2 = V3{0.0f, 0.0f, 0.0f};
    v = v2;
    const V3 rj = cross(p - v3(k.com), J);
    const V3 dw = V3{rj.x * k.inertiaInvBody[0], rj.y * k.inertiaInvBody[1], rj.z * k.inertiaInvBody[2]} * k.angularImpulseScale;
    V3 dww = mul(R, dw);
    dww.y = k.angularImpulseYScale * dww.y;   // FV scales the world y component (1 for StadiumCar)
    w = ClampAngular(w + dww, k.maxAngularSpeed);
}

}  // namespace tw::model

#endif
