// Twin model: small scalar vector helpers (L = 1). Plain + - * / and sqrt,
// no FMA (build with -ffp-contract=off), no libm (docs/twin/math.md).
#ifndef TW_MODEL_VEC3_H
#define TW_MODEL_VEC3_H

#include "tw_math.h"

namespace tw::model {

struct V3 {
    float x, y, z;
};

TW_INLINE V3 v3(float x, float y, float z) { return V3{x, y, z}; }
TW_INLINE V3 v3(const float *p) { return V3{p[0], p[1], p[2]}; }
TW_INLINE void store(float *p, V3 a) { p[0] = a.x; p[1] = a.y; p[2] = a.z; }
TW_INLINE V3 operator+(V3 a, V3 b) { return V3{a.x + b.x, a.y + b.y, a.z + b.z}; }
TW_INLINE V3 operator-(V3 a, V3 b) { return V3{a.x - b.x, a.y - b.y, a.z - b.z}; }
TW_INLINE V3 operator-(V3 a) { return V3{-a.x, -a.y, -a.z}; }
TW_INLINE V3 operator*(V3 a, float s) { return V3{a.x * s, a.y * s, a.z * s}; }
TW_INLINE V3 &operator+=(V3 &a, V3 b) { a = a + b; return a; }
TW_INLINE float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
TW_INLINE float len2(V3 a) { return dot(a, a); }
TW_INLINE V3 cross(V3 a, V3 b) {
    return V3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
// FV GmMath::NormalizeOr: v / |v| if |v|^2 > minLen2, else `fallback`.
TW_INLINE V3 normalize_or(V3 a, V3 fallback, float minLen2) {
    const float l2 = len2(a);
    if (!(l2 > minLen2)) return fallback;
    return a * (1.0f / tw::math::sqrt(l2));
}

// Rotation stored as columns (FV GmMat3 basisX/Y/Z = car axes in world).
struct M3 {
    V3 c0, c1, c2;
};
TW_INLINE V3 mul(const M3 &m, V3 v) {          // R v   (local -> world)
    return V3{m.c0.x * v.x + m.c1.x * v.y + m.c2.x * v.z,
              m.c0.y * v.x + m.c1.y * v.y + m.c2.y * v.z,
              m.c0.z * v.x + m.c1.z * v.y + m.c2.z * v.z};
}
TW_INLINE V3 mul_t(const M3 &m, V3 v) {        // R^T v (world -> local)
    return V3{dot(m.c0, v), dot(m.c1, v), dot(m.c2, v)};
}
// FV GmMat3::Set(GmQuat) (gm_matrix.cpp:55-82); q = x, y, z, w.
TW_INLINE M3 rot_from_quat(const float *q) {
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    const float x2 = x * 2.0f, y2 = y * 2.0f, z2 = z * 2.0f;
    const float xw = w * x2, yw = w * y2, zw = w * z2;
    const float xx = x * x2, xy = x * y2, xz = x * z2;
    const float yy = y * y2, yz = y * z2, zz = z * z2;
    M3 m;
    m.c0 = V3{(1.0f - yy) - zz, xy + zw, xz - yw};
    m.c1 = V3{xy - zw, (1.0f - xx) - zz, yz + xw};
    m.c2 = V3{xz + yw, yz - xw, (1.0f - xx) - yy};
    return m;
}

TW_INLINE float sgn_nonneg(float v) { return v < 0.0f ? -1.0f : 1.0f; }   // FV SignNonNegative
// |v| as one andps (bitwise, exact); tw::math::abs on scalars goes through the
// integer domain (movd / and / movd).
TW_INLINE float fabsf_(float v) { return __builtin_fabsf(v); }
// FV ClampZeroOne (scene_vehicle_car_internal.h:124-134).
TW_INLINE float clamp01(float v) {
    float c = 0.0f;
    if (!(v < 0.0f) && v != 0.0f) {
        c = v;
        if (v == v && 1.0f < v) c = 1.0f;
    }
    return c;
}

}  // namespace tw::model

#endif
