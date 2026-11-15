// Body proxy: one car-frame box around the 4 body ellipsoids, a cheap first test before the body shapes
// (architect C, docs/twin/collision-c.md §8).
//
// The body shapes are rigid in the car frame, so the box is built once per car:
//   - it is the union of the 4 ellipsoids' exact car-frame AABBs, + 2 mm;
//   - hc is the box of the 4 body centres around the same centre.
// Its world pose follows the car: t = pos + R c, M = R diag(h), cM = R diag(hc).
// The patch-slab test is a certificate for all 4 bodies:
//   - out of reach: n.(t - ref) - tau - |M^T n|_1 > 0, since the box support |M^T n|_1 bounds every body's support;
//   - behind: -n.(t - ref) - |cM^T n|_1 - slack > 0. This needs every body CENTRE behind the plane (FV culls per
//     shape centre), so it uses the centre box, not the box centre. Using the box centre misses deep body FACE
//     contacts (found by the comparison with the triangle twin).
// If the box cannot certify, BodiesSlab runs the same slab test on the 4 ellipsoids (4 lanes) before any
// per-body work.
#ifndef CC_PROXY_H
#define CC_PROXY_H

#include <algorithm>

#include "../math/tw_lanes.h"
#include "../math/tw_math.h"
#include "cc_face.h"

namespace cc {
namespace proxy {

namespace m = tw::math;

struct BodyBox { float c[3], h[3], hc[3]; };  // car frame: centre, half extents, half extents of the body-centre box

// centre[k]: body ellipsoid k's centre in the car frame; axes[k][j]: its axis j in the car frame (unit); r[k]: radii.
inline BodyBox Build(const float centre[4][3], const float axes[4][3][3], const float r[4][3]) {
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 3; ++i) {
            float h2 = 0;  // exact AABB half extent of an ellipsoid along axis i: |row i of axes * diag(r)|
            for (int j = 0; j < 3; ++j) { const float a = axes[k][j][i] * r[k][j]; h2 += a * a; }
            const float h = m::sqrt(h2);
            lo[i] = std::min(lo[i], centre[k][i] - h);
            hi[i] = std::max(hi[i], centre[k][i] + h);
        }
    BodyBox b;
    for (int i = 0; i < 3; ++i) { b.c[i] = 0.5f * (lo[i] + hi[i]); b.h[i] = 0.5f * (hi[i] - lo[i]) + 2e-3f; b.hc[i] = 0.0f; }
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 3; ++i) b.hc[i] = std::max(b.hc[i], m::abs(centre[k][i] - b.c[i]) + 2e-3f);
    return b;
}

struct BoxPose { float t[3], M[3][3], cM[3][3]; };

// R: car rotation, columns basisX, basisY, basisZ (FV GmMat3 column order).
inline void WorldPose(const BodyBox &b, const float pos[3], const float R[9], BoxPose *p) {
    for (int i = 0; i < 3; ++i) {
        p->t[i] = pos[i] + ((R[i] * b.c[0] + R[3 + i] * b.c[1]) + R[6 + i] * b.c[2]);
        for (int j = 0; j < 3; ++j) { p->M[i][j] = R[3 * j + i] * b.h[j]; p->cM[i][j] = R[3 * j + i] * b.hc[j]; }
    }
}

inline float L1MtV(const float M[3][3], const float n[3]) {
    float s = 0;
    for (int j = 0; j < 3; ++j) s += m::abs((M[0][j] * n[0] + M[1][j] * n[1]) + M[2][j] * n[2]);
    return s;
}

// Patch slab: unit normal n, reference point ref, half thickness tau, behind slack = tau + alpha (|t - cen|_1 + rad).
// Returns true when no body ellipsoid can touch any member of the patch.
inline bool SlabCertified(const BoxPose &p, const float n[3], const float ref[3], float tau, float behindSlack) {
    const float sd = (n[0] * (p.t[0] - ref[0]) + n[1] * (p.t[1] - ref[1])) + n[2] * (p.t[2] - ref[2]);
    if (sd - tau - L1MtV(p.M, n) - 1e-5f > 0.0f) return true;
    return -sd - L1MtV(p.cM, n) - behindSlack - 1e-5f > 0.0f;
}

// The same slab test on the 4 body ellipsoids (x = M u + t), 4 lanes. behindSlack0 = tau + alpha rad;
// alpha and cen complete the per-centre slack.
template <class P>
inline bool BodiesSlab(const P *const *bodies, const float n[3], const float ref[3], float tau, float alpha, const float cen[3], float rad) {
    using F = m::vf<4>;
    using PL = face::Lanes<F>;
    const F tx = PL::T(bodies, 0), ty = PL::T(bodies, 1), tz = PL::T(bodies, 2);
    const F nx(n[0]), ny(n[1]), nz(n[2]);
    const F sd = (nx * (tx - F(ref[0])) + ny * (ty - F(ref[1]))) + nz * (tz - F(ref[2]));
    const F v0 = (PL::M(bodies, 0, 0) * nx + PL::M(bodies, 1, 0) * ny) + PL::M(bodies, 2, 0) * nz;
    const F v1 = (PL::M(bodies, 0, 1) * nx + PL::M(bodies, 1, 1) * ny) + PL::M(bodies, 2, 1) * nz;
    const F v2 = (PL::M(bodies, 0, 2) * nx + PL::M(bodies, 1, 2) * ny) + PL::M(bodies, 2, 2) * nz;
    const F H2 = (v0 * v0 + v1 * v1) + v2 * v2;
    const F dc = (m::abs(tx - F(cen[0])) + m::abs(ty - F(cen[1]))) + m::abs(tz - F(cen[2]));
    const F behind = (F(0.0f) - sd) - (F(tau) + F(alpha) * (dc + F(rad))) - F(1e-5f);
    const F far0 = sd - F(tau) - F(1e-5f);
    const auto ok = m::cmp_gt(behind, F(0.0f)) | (m::cmp_gt(far0, F(0.0f)) & m::cmp_gt(far0 * far0, H2 * F(1.00001f)));
    alignas(64) float v[4];
    m::select(ok, F(1.0f), F(0.0f)).store(v);
    return v[0] != 0.0f && v[1] != 0.0f && v[2] != 0.0f && v[3] != 0.0f;
}

}  // namespace proxy
}  // namespace cc

#endif
