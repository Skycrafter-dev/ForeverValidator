// Twin GPU engine: C's collision stage ("c", the noise path) as device functions, one detection per thread,
// the same contacts bit for bit (cc_stage.h). Included by the stage check (tw_gpu_collide.cu) and the tick
// kernels.
//
// Per call: the 8 shape poses (collide::ShapePoses: SetMult(local, carIso)), each shape's exact box (h =
// sqrt(Q_aa), the stage's eps) and their union; a walk of a spatial 16-ary tree over leaves of up to 16 consecutive
// ranks (cut for small boxes; tw_gpu_world.cpp) with the union box alone, on compact nodes (NodeQ: one 128-byte line,
// the children's boxes quantized outward to 7 bits a byte), its leaves' hits kept as (first rank, mask) records sorted
// by first rank, so the ranks whose box meets the union box come out increasing (the FV-ordered tree, whose leaves come
// in rank order, is the fallback when the records do not fit); then (phase 2) per such rank, the shapes whose box
// meets the triangle's exact box (closed test) and whose box midpoint is not behind its plane (BoxPlaneMask,
// conservative) make the candidate list (rank, shape mask), unless the whole union box is in front of the plane beyond
// the margin (UnionFront, conservative: no shape keeps the rank). Then
// shape by shape (the merge's order) over the candidates (a bit mask per window of 64): the plane prefilter and the
// triangle's edge planes (PlaneFilter: PlaneMayTouch, EdgeOut; two candidates' loads at a time), the noise narrow phase (the per-(shape, instance) job path's operations:
// MakeMeshToUnit once per run of one instance (with the host's 1 / rad: no division), unit-space
// vertices, UnitContact, the point back through FV's c2w factors; a rank's plane, instance, world triangle and
// mesh-local vertices in one 64-byte record, RankN) and MergeOutLean's merge as the contacts come (a primary goes out
// at once; a non-primary's rank is kept, its contact made again after the shape's primaries and tested against
// them): the raw contacts of each shape in FV order, the CPU's merge, so the same output without a raw list.
//
// The CPU stage's prefilter (world-space planes with a tolerance, cc_memo.cpp Filter) only drops pairs the
// narrow phase rejects, so running the narrow phase on every box pair the plane prefilter passes gives the same
// contacts; the identity checks (searchq --gpu-check, --gpu-detect, the recorded queue) hold the GPU to it.
//
// Forms measured and dropped (docs/twin/gpu.md, branches gpu-b, gpu-b2-exp, gpu-b3-exp): v1 (raw list, float nodes), v2
// (float nodes), the warp-cooperative form, a spatial tree over the ranks, equal-count (Morton) groups over the FV leaves,
// L1 prefetches (leaves, records, the walk's siblings), the shape boxes made again for phase 2.
#ifndef TW_GPU_COLLIDE_CUH
#define TW_GPU_COLLIDE_CUH

#include <cmath>
#include <cstring>

#include "cc_lanes.h"
#include "response.h"
#include "tw_gpu.h"
#include "tw_gpu_world.h"
#include "vec3.h"

namespace tw::gpu::dev {

namespace L = tw::exact::lanes;
namespace N = tw::collide_c::noise;
using V3 = L::V3<float>;
using I3 = L::I3<float>;

constexpr int kStack = 128;      // tree walk: at most 15 pending siblings per level
constexpr int kMaxCand = 256;    // candidate ranks per call
constexpr int kMaxNonPrim = 64;  // non-primary contacts per shape

struct DevWorld {
    CarShapesG car;
    CollisionConsts k;
    const NodeQ *qnodes;
    const RankBox *rankBox;
    const RankN *rankN;          // per rank: the plane prefilter's plane and the narrow phase's references (one sector)
    const RankE *rankE;          // per rank: the plane prefilter's edge planes
    const float *iso;
    const OutRecG *outRec;
    std::uint32_t root;
    unsigned long long *stats;   // TW_GPU_DETECT_STATS: per-warp counters of the tick's calls (DetectAt), null otherwise
    int repeat;                  // TW_GPU_DETECT_REPEAT (measurement): each tick's detection runs this many times (1)
};

struct Box {
    float lo[3], hi[3];
};

__device__ __forceinline__ I3 IsoOf(const float *p) {   // 12 floats: basis columns, translation
    return I3{L::M3<float>{{V3{p[0], p[1], p[2]}, V3{p[3], p[4], p[5]}, V3{p[6], p[7], p[8]}}}, V3{p[9], p[10], p[11]}};
}

__device__ __forceinline__ float Comp(const V3 &v, int a) { return a == 0 ? v.x : a == 1 ? v.y : v.z; }

// sqrt.rn.f32 and the reciprocal (div.rn 1 / x) without a branch: ptxas's own fast paths for them (the instructions of its
// expansion: MUFU.RSQ, two FTZ products, two FFMA; MUFU.RCP, FFMA, an FTZ negation, FFMA), whose range check would split the
// code into blocks around a call of the slow path; here the check only clears ok (+-0 is handled: sqrt(+-0) = +-0,
// 1 / +-0 = +-inf). Checked on all 2^32 inputs against sqrtf and 1.0f / x: every input with ok gives the same bits (sqrt: the
// 1,920,991,234 inputs in [2^-101, max] and +-0; reciprocal: the 4,227,858,434 with |x| normal below 2^126 and +-0). The
// caller takes the library's operations again when a lane's ok is clear.
// The fast paths alone (no range check: callers whose input is known to be in range) and the checks.
__device__ __forceinline__ float SqrtU(float x) {
    const unsigned b = __float_as_uint(x);
    float r, s, h;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    asm("mul.ftz.f32 %0, %1, %2;" : "=f"(s) : "f"(x), "f"(r));
    asm("mul.ftz.f32 %0, %1, 0f3F000000;" : "=f"(h) : "f"(r));
    const float e = __fmaf_rn(-s, s, x);
    const float y = __fmaf_rn(e, h, s);
    return (b << 1) == 0u ? x : y;
}
__device__ __forceinline__ float RcpU(float x) {
    const unsigned b = __float_as_uint(x);
    float r, t2;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    const float t = __fmaf_rn(r, x, -1.0f);
    asm("add.ftz.f32 %0, %1, 0f80000000;" : "=f"(t2) : "f"(-t));
    const float y = __fmaf_rn(r, t2, r);
    return (b << 1) == 0u ? __uint_as_float((b & 0x80000000u) | 0x7f800000u) : y;
}
__device__ __forceinline__ float SqrtF(float x, bool &ok) {
    const unsigned b = __float_as_uint(x);
    ok &= (b - 0x0d000000u <= 0x727fffffu) | ((b << 1) == 0u);
    return SqrtU(x);
}
__device__ __forceinline__ float RcpF(float x, bool &ok) {
    const unsigned b = __float_as_uint(x);
    ok &= (((b + 0x01800000u) & 0x7f800000u) > 0x01ffffffu) | ((b << 1) == 0u);
    return RcpU(x);
}
// SqrtF for a positive input: ok only in [2^-101, max] (+-0 clears it; the fast path's own result there, without the zero
// select). A subset of SqrtF's ok inputs, on which it returns SqrtF's y: the same bits as sqrtf.
__device__ __forceinline__ float SqrtP(float x, bool &ok) {
    ok &= __float_as_uint(x) - 0x0d000000u <= 0x727fffffu;
    float r, s, h;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    asm("mul.ftz.f32 %0, %1, %2;" : "=f"(s) : "f"(x), "f"(r));
    asm("mul.ftz.f32 %0, %1, 0f3F000000;" : "=f"(h) : "f"(r));
    const float e = __fmaf_rn(-s, s, x);
    return __fmaf_rn(e, h, s);
}
template <bool kFast> __device__ __forceinline__ float SqrtK(float x, bool &ok) {
    if constexpr (kFast) return SqrtF(x, ok); else return L::Sqrt(x);
}
template <bool kFast> __device__ __forceinline__ float RcpK(float x, bool &ok) {
    if constexpr (kFast) return RcpF(x, ok); else return 1.0f / x;
}
// 1 / sqrt(x) with one range check: an input in SqrtF's ok range ([2^-101, max], +-0) has its square root in
// [2^-50.5, 2^64] or +-0, inside RcpF's (|x| normal below 2^126, +-0), so the reciprocal needs no check of its own (the
// composition is also checked on all 2^32 inputs: tools/twin/search/gpu-fastop-check.cu).
template <bool kFast> __device__ __forceinline__ float RsqK(float x, bool &ok) {
    if constexpr (kFast) return RcpU(SqrtF(x, ok)); else return 1.0f / L::Sqrt(x);
}

// The stage's shape box (PrepareLane / DetectLanes16): the ellipsoid's exact AABB h = sqrt(Q_aa),
// eps = 1e-3 + 1e-6 |c|, (c - h) - eps .. (c + h) + eps
// kFast: the square root as SqrtP (one block; ok cleared out of its range: the caller makes the boxes again with the
// library's), else the library's.
template <bool kFast>
__device__ __forceinline__ Box ShapeBoxT(const CarShapesG &car, int k, const I3 &w, bool &ok) {
    Box b;
    for (int a = 0; a < 3; ++a) {
        const float w0 = Comp(w.r.c[0], a), w1 = Comp(w.r.c[1], a), w2 = Comp(w.r.c[2], a), t = Comp(w.t, a);
        const float q = (car.r2[k][0] * (w0 * w0) + car.r2[k][1] * (w1 * w1)) + car.r2[k][2] * (w2 * w2);
        float h;
        if constexpr (kFast) h = SqrtP(q, ok); else h = L::Sqrt(q);
        const float eps = 1e-3f + 1e-6f * L::Abs(t);
        b.lo[a] = (t - h) - eps;
        b.hi[a] = (t + h) + eps;
    }
    return b;
}

// The plane prefilter: false when the ellipsoid (centre c, axes a_i, radii r_i) is entirely in front of the triangle's
// plane beyond its support h = |(r_i a_i . n)|, or its centre is behind the plane. UnitContact then finds no contact
// (it needs 0 <= pd <= 1: the centre in front, within the unit sphere's reach); the 1 cm margin (and 1e-5 of the
// coordinates) covers the difference between this world-space test and the narrow phase's unit-space arithmetic.
__device__ __forceinline__ bool PlaneMayTouch(const float4 &pl, const I3 &ell, const float r[3]) {
    const V3 &c = ell.t;
    const float d = (pl.x * c.x + pl.y * c.y + pl.z * c.z) - pl.w;
    const float m = 0.01f + 1e-5f * (((fabsf(c.x) + fabsf(c.y)) + fabsf(c.z)) + fabsf(pl.w));
    if (d < -m) return false;
    const float s0 = r[0] * (ell.r.c[0].x * pl.x + ell.r.c[0].y * pl.y + ell.r.c[0].z * pl.z);
    const float s1 = r[1] * (ell.r.c[1].x * pl.x + ell.r.c[1].y * pl.y + ell.r.c[1].z * pl.z);
    const float s2 = r[2] * (ell.r.c[2].x * pl.x + ell.r.c[2].y * pl.y + ell.r.c[2].z * pl.z);
    // d - m > h without the square root: (d - m)^2 > 1.0001 h^2 (stricter than d > h + m by far less than the margin)
    const float e = d - m;
    return !((e > 0.0f) & (e * e > (s0 * s0 + s1 * s1 + s2 * s2) * 1.0001f));
}

// The plane prefilter's edge test: true when the ellipsoid lies beyond one of the triangle's edge planes (RankE: a
// half-space m.x <= off holding the whole triangle, m = q 2^-15 exactly, |m| < 1) by more than its largest radius and
// PlaneMayTouch's margin: m.c - off - (0.01 + 1e-5 (|c|_1 + |off|)) > 1.0001 rmax, m.c computed as q.(2^-15 c) (exact
// scaling) with two FMAs. The ellipsoid lies in the ball of radius rmax around its centre c, so every point x of it has
// m.x - off > 0.01 + 1e-5 (|c|_1 + |off|) less the test's own rounding (a few ulps of |c|_1 and |off|), while every point of
// the triangle has m.x <= off. UnitContact's contacts are points of the triangle (the plane's projection inside it, a
// vertex or an edge point) within distance 1 of the origin in unit space, i.e. inside the ellipsoid: none when the
// ellipsoid is beyond a half-space holding the triangle. The margin (1 cm and 1e-5 of the coordinates) covers the
// difference between this world-space test and the narrow phase's unit-space arithmetic, as PlaneMayTouch's: the world
// triangle is the mesh's vertices through the instance's iso in float (tw_world_compile.cpp), the narrow phase's the same
// vertices through MakeMeshToUnit. A degenerate edge (m = 0, off = 0) never rejects (-margin > rmax is false); NaN never
// rejects. Per shape (ShapesOut): cs = 2^-15 c, mb = 0.01 + 1e-5 |c|_1, rm = 1.0001 rmax.
__device__ __forceinline__ float Lo16(unsigned w) { return static_cast<float>(static_cast<int>(static_cast<short>(w & 0xFFFFu))); }
__device__ __forceinline__ float Hi16(unsigned w) { return static_cast<float>(static_cast<int>(w) >> 16); }
__device__ __forceinline__ bool EdgeOut(const uint4 &a, const uint4 &b, const V3 &cs, float mb, float rm) {
    const float off[3] = {__uint_as_float(a.x), __uint_as_float(a.y), __uint_as_float(a.z)};
    const float m[3][3] = {{Lo16(a.w), Hi16(a.w), Lo16(b.x)}, {Hi16(b.x), Lo16(b.y), Hi16(b.y)}, {Lo16(b.z), Hi16(b.z), Lo16(b.w)}};
    bool out = false;
    for (int e = 0; e < 3; ++e) {
        const float d = __fmaf_rn(m[e][2], cs.z, __fmaf_rn(m[e][1], cs.y, m[e][0] * cs.x)) - off[e];
        out |= d - __fmaf_rn(1e-5f, fabsf(off[e]), mb) > rm;
    }
    return out;
}

// The plane prefilter on the shape boxes (phase 2, per candidate rank, before ShapesOut): bit k clear when shape k's
// ellipsoid is behind the triangle's plane or in front of it beyond its support, by more than M = mBase + 2e-5 |w|, mBase =
// 0.02 + 2e-5 C, C the union box's largest |x| + |y| + |z| (at least any shape centre's |c|_1), so M >= 2 m, m
// PlaneMayTouch's margin 0.01 + 1e-5 (|c|_1 + |w|). Dominated by PlaneMayTouch, so it drops no pair PlaneMayTouch keeps and
// no contact. The box is lo = (t - h) - eps, hi = (t + h) + eps around the ellipsoid's centre t, so sum = n.(lo + hi) is
// 2 n.t up to a few ulps of |t|_1 + h (delta), far below M - m >= m >= 1 cm and 1e-5 of the coordinates:
// - behind: sum < 2 (w - M) gives PlaneMayTouch's d = n.t - w < -M + delta < -m: rejected (centre behind);
// - in front: sum - 2 (w + M) > sup, sup = sum_a |n.R_a| hc_a, R_a the car's axes (carIso's columns), hc_a = 2.002 x the
//   ellipsoid's half extent along R_a (host: sqrt(sum_i rad_i^2 lr_{i,a}^2), rounded up). The ellipsoid's axes are R lr_i,
//   so its support along n is s = sqrt(n'^T Q' n'), n' = R^T n, Q' = sum_i rad_i^2 lr_i lr_i^T (the car-frame matrix), and
//   n'^T Q' n' = |sum_a n'_a v_a|^2 <= (sum_a |n'_a| |v_a|)^2 with |v_a|^2 = Q'_aa = (hc_a / 2.002)^2: s <= sup / 2.002.
//   So d > M + 1.001 s - delta >= 2 m + s - delta, and PlaneMayTouch's e = d - m > s + m - delta: its computed support
//   (the shape's axes SetMult(lr, carIso), each within ~1e-6 of R lr_i; radii < 2 m: within ~1e-5 of s) times
//   sqrt(1.0001) (< s + 3e-4) stays below e by nearly m: rejected (beyond the support in front).
// A degenerate plane (0, 0, 0, 0) keeps every shape (sum 0, neither compare holds); a NaN keeps it. (The front half with the
// world box's support instead, round 6: roulette 93.5 against 88 ms, speedtech 144 against 138.)
__device__ __forceinline__ unsigned BoxPlaneMask(const float4 &pl, const Box box[8], float mBase, const I3 &carIso, const CarShapesG &car) {
    const float M = mBase + 2e-5f * fabsf(pl.w);
    const float behind2 = (pl.w - M) * 2.0f;
    const float front2 = (pl.w + M) * 2.0f;
    float aR[3];   // |n.R_a|
    for (int a = 0; a < 3; ++a) aR[a] = fabsf(__fmaf_rn(carIso.r.c[a].z, pl.z, __fmaf_rn(carIso.r.c[a].y, pl.y, carIso.r.c[a].x * pl.x)));
    unsigned keep = 0u;
    for (int k = 0; k < 8; ++k) {
        const float sum = (pl.x * (box[k].lo[0] + box[k].hi[0]) + pl.y * (box[k].lo[1] + box[k].hi[1])) + pl.z * (box[k].lo[2] + box[k].hi[2]);
        const float sup = __fmaf_rn(aR[2], car.hc[k][2], __fmaf_rn(aR[1], car.hc[k][1], aR[0] * car.hc[k][0]));
        keep |= static_cast<unsigned>(!(sum < behind2) & !(sum - front2 > sup)) << k;   // sum: 2 n.mid
    }
    return keep;
}

// The plane prefilter's "in front" half on the union box (phase 2, per rank whose box meets a shape's): true when the union
// box u lies in front of the triangle's plane beyond M (BoxPlaneMask's margin): n.mid - w - sum |n_a| hd_a > M, mid and hd
// u's midpoint and half extents (computed as (lo + hi) and (hi - lo), so the test is 2x). Then every shape is dropped for
// this rank: PlaneMayTouch rejects every pair (in front beyond the support). Exact arithmetic: shape k's box lies in u and
// is (t - h) - eps .. (t + h) + eps around the ellipsoid's centre t with h_a = sqrt(Q_aa) (Q = sum r_i^2 a_i a_i^T, the
// box's own weights), so d = n.t - w >= M + sum |n_a| h_a >= M + s, s = sqrt(n^T Q n) the support PlaneMayTouch computes
// (n^T Q n = |sum_a n_a v_a|^2 <= (sum_a |n_a| |v_a|)^2, |v_a|^2 = Q_aa). M >= 2 m (m: PlaneMayTouch's margin, as in
// BoxPlaneMask), so d - m >= s + m with m >= 1 cm + 1e-5 (|c|_1 + |w|): far above the rounding of either side (ulps of
// |c|_1 + |w| and 1e-6 of s) and of PlaneMayTouch's 1.0001 s^2 (s < 3 m). A degenerate plane (0, 0, 0, 0), NaN or inf
// never rejects (the compare is false).
__device__ __forceinline__ bool UnionFront(const float4 &pl, const Box &u, float mBase) {
    const float M = mBase + 2e-5f * fabsf(pl.w);
    const float front2 = (pl.w + M) * 2.0f;
    const float sum = (pl.x * (u.lo[0] + u.hi[0]) + pl.y * (u.lo[1] + u.hi[1])) + pl.z * (u.lo[2] + u.hi[2]);   // 2 n.mid
    const float sup = (fabsf(pl.x) * (u.hi[0] - u.lo[0]) + fabsf(pl.y) * (u.hi[1] - u.lo[1])) + fabsf(pl.z) * (u.hi[2] - u.lo[2]);   // 2 sum |n_a| hd_a
    return sum - sup > front2;
}

// geometry_helpers.cpp NearlyEqual(value, reference), per component
__device__ __forceinline__ bool NearlyEqual(float value, float reference) {
    const float tol = L::Abs(reference) * 1.0e-5f;
    return reference - tol <= value && value <= reference + tol;
}
__device__ __forceinline__ bool NearlyEqual(const V3 &a, const V3 &b) {
    return NearlyEqual(a.x, b.x) && NearlyEqual(a.y, b.y) && NearlyEqual(a.z, b.z);
}

// Per-call counters (the detection benchmark, BenchDetect).
struct DetectCount {
    unsigned nodes, childTests, leafHits, boxPairs, planePass, raws, spills;
    unsigned ucRuns, frames, second;   // UnitContact runs (first pass), frames made, non-primaries run again
    unsigned planeK[8];                // plane passes per shape
    unsigned cands;                    // candidate ranks (a shape's box meets the rank's)
    unsigned behind, front;            // plane prefilter rejects: centre behind the plane, beyond the support in front
    unsigned rankRejPairs, rankRej;    // box pairs of candidate ranks whose every pair the plane prefilter rejects; those ranks
    unsigned contacts;                 // merged contacts
    unsigned boxPlaneRej;              // box pairs BoxPlaneMask and UnionFront drop (phase 2)
    unsigned unionFront;               // ranks UnionFront drops (phase 2)
    unsigned edgeOut;                  // pairs the plane passes and the edge planes reject (PlaneFilter)
};

__device__ __forceinline__ I3 ShapePose(const CarShapesG &car, const I3 &carIso, const float absorb[4], int k) {   // ShapePoses, one shape
    I3 local{L::M3<float>{{V3{car.lr[k][0], car.lr[k][1], car.lr[k][2]}, V3{car.lr[k][3], car.lr[k][4], car.lr[k][5]},
                           V3{car.lr[k][6], car.lr[k][7], car.lr[k][8]}}},
             V3{car.lt[k][0], car.lt[k][1], car.lt[k][2]}};
    if (car.wheel[k] >= 0) local.t.y += -absorb[car.wheel[k]];
    return L::SetMult(local, carIso);
}

// The narrow phase's payload: the unit-space contact back to world (P through FV's c2w factors, S and N through the axes).
__device__ __forceinline__ void Payload(const DevWorld &w, const I3 &ell, int k, const I3 &mesh, const I3 &back, const V3 &P, const V3 &Nn,
                                        const V3 &S, V3 *oP, V3 *oN, V3 *oS) {
    const float r0 = w.car.rad[k][0], r1 = w.car.rad[k][1], r2 = w.car.rad[k][2];
    const float ir0 = w.car.irad[k][0], ir1 = w.car.irad[k][1], ir2 = w.car.irad[k][2];
    const V3 &a0 = ell.r.c[0], &a1 = ell.r.c[1], &a2 = ell.r.c[2];
    auto R = [&](float u0, float u1, float u2) {
        return V3{(a0.x * u0 + a1.x * u1) + a2.x * u2, (a0.y * u0 + a1.y * u1) + a2.y * u2, (a0.z * u0 + a1.z * u1) + a2.z * u2};
    };
    *oP = L::TPoint(mesh, L::TPoint(back, V3{r0 * P.x, r1 * P.y, r2 * P.z}));
    *oS = R(r0 * S.x, r1 * S.y, r2 * S.z);
    *oN = L::NormalizeSel<float, bool>(R(ir0 * Nn.x, ir1 * Nn.y, ir2 * Nn.z), w.k.dirEpsSq);
}

__device__ __forceinline__ void FillOutV(model::RespContact &o, const V3 &P, const V3 &Nw, const V3 &S, bool primary, const OutRecG &r, int k) {
    o.P[0] = P.x; o.P[1] = P.y; o.P[2] = P.z;
    o.N[0] = Nw.x; o.N[1] = Nw.y; o.N[2] = Nw.z;
    o.Sep[0] = S.x; o.Sep[1] = S.y; o.Sep[2] = S.z;
    o.peerZ[0] = r.peerZ[0]; o.peerZ[1] = r.peerZ[1]; o.peerZ[2] = r.peerZ[2];
    o.peerId = r.peerId;
    o.shape = static_cast<std::uint8_t>(k);
    o.material = r.material;
    o.flags = static_cast<std::uint8_t>((primary ? unsigned(model::kRespPrimary) : 0u) | r.flags);
    o.pad = 0u;
}

// A shape's meshToUnit (and the way back) for the instance of the current candidate, made again when the instance changes.
struct UnitFrame {
    std::uint32_t inst;
    I3 m2u, back;
};
// irad: CarShapesG::irad, the host's 1.0f / rad (IEEE quotients, the same bits as MakeMeshToUnit's one / rad), so the
// frame needs no division.
__device__ __forceinline__ void FrameFor(const DevWorld &w, const I3 &ell, const V3 &irad, std::uint32_t inst, UnitFrame &f) {
    if (inst == f.inst) return;
    f.inst = inst;
    f.m2u = N::MakeMeshToUnitInv<float>(ell, irad, IsoOf(w.iso + 12u * inst), &f.back);
}
// N::UnitContact<float, bool> (cc_lanes.h), the lane form as is, with warp-uniform skips of work whose results no lane of
// the calling group reads: the edges once every lane's und is clear (an edge changes nothing then: und, viol false), the
// feature's distances when no lane has a violated edge (fe false everywhere: feat false, P, N, S take the face's values).
// A per-lane exit would make the lanes that take different ways run them one after the other (lost: gpu-narrow2-exp).
// kFast: the square roots and reciprocals as SqrtF / RcpF (one block; *ok cleared for an input out of their ranges), else the
// library's.
template <bool kFast>
__device__ __forceinline__ void UnitContactT(unsigned am, const V3 vu[3], float kDirEpsSq, float kCollisionDistance, bool *face, bool *feat, V3 *P, V3 *Nn,
                                             V3 *S, V3 *Kn, bool *okp) {
    bool ok = true;
    const float zero = 0.0f, one = 1.0f, eps = kDirEpsSq, cdist = kCollisionDistance;
    float lenSq;
    const V3 nr = L::UnitNormal(vu, &lenSq);
    bool und = eps < lenSq;
    const float invLen = RsqK<kFast>(lenSq, ok);
    const V3 n = {nr.x * invLen, nr.y * invLen, invLen * nr.z};
    const V3 z3 = {zero, zero, zero};
    const float pd = L::Dot(L::Sub(z3, vu[0]), n);
    und = N::AndNot(N::AndNot(und, one < pd), pd < zero);
    const float edgeReach = SqrtK<kFast>(N::Max(one - pd * pd, zero), ok);
    const V3 proj = L::Add(z3, L::Scale(n, -pd));
    bool fe = false, cont = false, endB = false;
    V3 fpt = {zero, zero, zero};
    float minD = zero;
#pragma unroll
    for (int e = 0; e < 3; ++e) {
        const V3 &start = vu[e];
        const V3 &end = vu[e == 2 ? 0 : e + 1];
        const V3 E = L::Sub(end, start);
        const float el2 = L::Dot(E, E);
        const float s = RsqK<kFast>(el2, ok);
        const bool big = eps < el2;
        const V3 dir = {L::Select(big, E.x * s, E.x), L::Select(big, E.y * s, E.y), L::Select(big, E.z * s, E.z)};
        const V3 enrm = L::Cross(dir, n);
        const V3 rel = L::Sub(proj, start);
        const float ed = L::Dot(rel, enrm);
        und = N::AndNot(und, edgeReach < ed);
        const bool viol = und & (zero < ed);
        {   // without the lane form's branch: one block, the three edges' chains interleaved
            const float fs = L::Dot(rel, dir);
            const float fend = L::Dot(L::Sub(proj, end), dir);
            const bool cA = viol & (fs < zero);
            const bool cB = N::AndNot(N::AndNot(viol, cA), zero < fend);
            const bool cC = N::AndNot(N::AndNot(viol, cA), cB);
            const V3 pB = L::Sub(proj, L::Scale(enrm, ed));
            fpt = {L::Select(cA, start.x, L::Select(cB, pB.x, L::Select(cC, end.x, fpt.x))),
                   L::Select(cA, start.y, L::Select(cB, pB.y, L::Select(cC, end.y, fpt.y))),
                   L::Select(cA, start.z, L::Select(cB, pB.z, L::Select(cC, end.z, fpt.z)))};
            minD = L::Select(cB, cdist, L::Select(viol, eps, minD));
            cont = cont | cA;
            endB = endB | cC;
            fe = fe | viol;
            und = N::AndNot(und, viol);
        }
    }
    *face = und & (zero < pd);
    *Kn = n;
    if (!__any_sync(am, fe)) {   // no feature in the group: feat false, the face's payload (the only one read)
        *feat = false;
        *P = proj;
        *Nn = n;
        *S = L::Scale(n, pd - one);
        *okp = ok;
        return;
    }
    const V3 toC = L::Sub(z3, fpt);
    const float dSq = L::Dot(toC, toC);
    const float d = SqrtK<kFast>(dSq, ok);
    const bool okF = N::AndNot(N::AndNot(N::AndNot(fe, endB), cont & (one < dSq)), N::AndNot(fe, minD < dSq));
    const bool okB = N::AndNot(N::AndNot(endB, one < d), N::AndNot(endB, minD < d));
    *feat = okF | okB;
    // d in [2^-50.5, 2^64] or +-0 when dSq passed SqrtF's check, so its square root and the reciprocal of either need none
    float dd, inv;
    if constexpr (kFast) {
        dd = L::Select(endB, SqrtU(d), d);
        inv = RcpU(dd);
    } else {
        dd = L::Select(endB, L::Sqrt(d), d);
        inv = 1.0f / dd;
    }
    const V3 normal = L::Scale(toC, inv);
    const V3 pen = L::Scale(toC, (dd - one) * inv);
    const float along = L::Dot(pen, n);
    const bool f = *face;
    *P = {L::Select(f, proj.x, fpt.x), L::Select(f, proj.y, fpt.y), L::Select(f, proj.z, fpt.z)};
    *Nn = {L::Select(f, n.x, normal.x), L::Select(f, n.y, normal.y), L::Select(f, n.z, normal.z)};
    const float sepS = L::Select(f, pd - one, along);
    *S = L::Scale(n, sepS);
    *okp = ok;
}
// The fast form; the library's for a group in which a lane's square root or reciprocal had an input out of the fast ranges
// (a uniform branch, almost never taken).
__device__ __forceinline__ void UnitContactW(const V3 vu[3], float kDirEpsSq, float kCollisionDistance, bool *face, bool *feat, V3 *P, V3 *Nn, V3 *S,
                                             V3 *Kn) {
    const unsigned am = __activemask();
    bool ok;
    UnitContactT<true>(am, vu, kDirEpsSq, kCollisionDistance, face, feat, P, Nn, S, Kn, &ok);
    if (!__all_sync(am, ok)) UnitContactT<false>(am, vu, kDirEpsSq, kCollisionDistance, face, feat, P, Nn, S, Kn, &ok);
}

// The narrow phase on a rank's mesh-local vertices (RankN::v, FV stored order).
__device__ __forceinline__ void UnitOf(const DevWorld &w, const I3 &m2u, const float v[9], bool *face, bool *feat, V3 *P, V3 *Nn, V3 *S, V3 *Kn) {
    V3 vu[3];
    for (int c = 0; c < 3; ++c) vu[c] = L::TPoint(m2u, V3{v[3 * c], v[3 * c + 1], v[3 * c + 2]});
    N::UnitContact<float, bool>(vu, w.k.dirEpsSq, w.k.collisionDistance, face, feat, P, Nn, S, Kn);
}

// NodeQ's quantization of a query coordinate (the host's for the children: tw_gpu_world.cpp)
__device__ __forceinline__ unsigned Quant(float x, float org, float scale) {
    // clamp(floor(v), 0, 127) as one converting floor (saturating: negative and NaN give 0) and an integer min
    return min(__float2uint_rd((x - org) * scale), 127u);
}
// The children of a NodeQ whose quantized boxes meet the query box's (bit c: child c). Per byte, with 7-bit values and
// bit 7 as the guard (NodeQ): lo <= qh is bit 7 of (qh | 0x80) - lo, ql <= hi is bit 7 of (hi | 0x80) - ql (no
// borrow across bytes: each difference is 1..255), so a word of four children costs one subtraction a bound.
__device__ __forceinline__ unsigned QuantMask(const NodeQ &Q, const Box &u, std::uint32_t *first, bool *leaf) {
    const uint4 *p = reinterpret_cast<const uint4 *>(&Q);
    const uint4 h0 = p[0], h1 = p[1];
    *first = h0.w;
    *leaf = (h1.w >> 8) != 0u;
    const unsigned count = h1.w & 0xFFu;
    const float org[3] = {__uint_as_float(h0.x), __uint_as_float(h0.y), __uint_as_float(h0.z)};
    const float scale[3] = {__uint_as_float(h1.x), __uint_as_float(h1.y), __uint_as_float(h1.z)};
    unsigned ql[3], qh[3];
    for (int a = 0; a < 3; ++a) {
        ql[a] = Quant(u.lo[a], org[a], scale[a]) * 0x01010101u;                // q(lo) in each byte
        qh[a] = Quant(u.hi[a], org[a], scale[a]) * 0x01010101u + 0x80808080u;  // q(hi) | 0x80 in each byte
    }
    unsigned m4[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    for (int a = 0; a < 3; ++a) {
        const uint4 lo = p[2 + a], hi = p[5 + a];
        m4[0] &= (qh[a] - lo.x) & (hi.x - ql[a]);
        m4[1] &= (qh[a] - lo.y) & (hi.y - ql[a]);
        m4[2] &= (qh[a] - lo.z) & (hi.z - ql[a]);
        m4[3] &= (qh[a] - lo.w) & (hi.w - ql[a]);
    }
    unsigned mask = 0u;
    for (int j = 0; j < 4; ++j) mask |= (((m4[j] & 0x80808080u) * 0x00204081u) >> 28) << (4 * j);
    return mask & ((1u << count) - 1u);
}
__device__ __forceinline__ bool OverlapBox(const RankBox &b, const Box &q) {   // Overlap on a rank's box (closed)
    return b.lo[0] <= q.hi[0] && q.lo[0] <= b.hi[0] && b.lo[1] <= q.hi[1] && q.lo[1] <= b.hi[1] && b.lo[2] <= q.hi[2] && q.lo[2] <= b.hi[2];
}
// OverlapBox on the 8 shape boxes (bit k: shape k), each as one predicate chain (six ordered <= compares, then a select:
// the same results, NaN false as in C++); NVVM made 95 instructions of the 8 (a select per compare).
__device__ __forceinline__ unsigned OverlapMask8(const float4 &lo, const float4 &hi, const Box box[8]) {
    unsigned sm = 0u;
    for (int k = 0; k < 8; ++k) {
        unsigned bit;
        asm("{\n\t.reg .pred p;\n\t"
            "setp.le.f32 p, %1, %2;\n\t"
            "setp.le.and.f32 p, %3, %4, p;\n\t"
            "setp.le.and.f32 p, %5, %6, p;\n\t"
            "setp.le.and.f32 p, %7, %8, p;\n\t"
            "setp.le.and.f32 p, %9, %10, p;\n\t"
            "setp.le.and.f32 p, %11, %12, p;\n\t"
            "selp.u32 %0, %13, 0, p;\n\t}"
            : "=r"(bit)
            : "f"(lo.x), "f"(box[k].hi[0]), "f"(box[k].lo[0]), "f"(hi.x), "f"(lo.y), "f"(box[k].hi[1]), "f"(box[k].lo[1]), "f"(hi.y),
              "f"(lo.z), "f"(box[k].hi[2]), "f"(box[k].lo[2]), "f"(hi.z), "r"(1u << k));
        sm |= bit;
    }
    return sm;
}
// The plane prefilter on a window's candidates with shape k (bit j: entry c[j], the window's 64 in one mask: one copy of the
// tests in the code), kPF at a time with their loads issued together (an entry, then its rank's plane and edge planes):
// the bits that pass PlaneMayTouch and EdgeOut.
constexpr int kPF = 2;   // PlaneFilter's candidates a step (four before the edge planes: their tests run without a branch)
template <bool kCount>
__device__ __forceinline__ unsigned long long PlaneFilter(unsigned long long m, const std::uint32_t *c, const RankN *rankN, const RankE *rankE,
                                                          const I3 &ell, const float rad[3], const V3 &cs, float mb, float rm, int k,
                                                          DetectCount *cnt) {
    unsigned long long pass = 0ull;
    while (m != 0ull) {
        int j[kPF];
        bool v[kPF];
        for (int t = 0; t < kPF; ++t) {
            v[t] = m != 0ull;
            j[t] = v[t] ? __ffsll(m) - 1 : 0;
            m &= m - 1ull;
        }
        std::uint32_t e[kPF];
        for (int t = 0; t < kPF; ++t) e[t] = v[t] ? c[j[t]] : 0u;
        float4 pl[kPF];
        uint4 ea[kPF], eb[kPF];
        for (int t = 0; t < kPF; ++t) pl[t] = v[t] ? __ldg(reinterpret_cast<const float4 *>(rankN + (e[t] >> 8))) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int t = 0; t < kPF; ++t) {
            const uint4 z = make_uint4(0u, 0u, 0u, 0u);
            ea[t] = v[t] ? __ldg(reinterpret_cast<const uint4 *>(rankE + (e[t] >> 8))) : z;
            eb[t] = v[t] ? __ldg(reinterpret_cast<const uint4 *>(rankE + (e[t] >> 8)) + 1) : z;
        }
        for (int t = 0; t < kPF; ++t) {
            const bool pt = PlaneMayTouch(pl[t], ell, rad), eo = EdgeOut(ea[t], eb[t], cs, mb, rm);
            pass |= static_cast<unsigned long long>(v[t] & pt & !eo) << j[t];
            if constexpr (kCount) {
                if (v[t]) {
                    if (pt && !eo) { ++cnt->planePass; ++cnt->planeK[k]; }
                    else if (pt) ++cnt->edgeOut;
                    else {
                        const V3 &cc = ell.t;
                        const float d = (pl[t].x * cc.x + pl[t].y * cc.y + pl[t].z * cc.z) - pl[t].w;
                        const float mm = 0.01f + 1e-5f * (((fabsf(cc.x) + fabsf(cc.y)) + fabsf(cc.z)) + fabsf(pl[t].w));
                        if (d < -mm) ++cnt->behind; else ++cnt->front;
                    }
                }
            }
        }
    }
    return pass;
}
// Shape by shape over the candidates (rank << 8 | shape mask, in rank order): the plane prefilter, the narrow phase and
// the merge on the fly.
template <bool kCount>
__device__ __forceinline__ void ShapesOut(const DevWorld &w, const I3 &carIso, const float absorb[4], const std::uint32_t *cand, int nc, unsigned shapes,
                                          std::uint32_t lost, GpuContacts &out, DetectCount *cnt) {
    constexpr int kMax = model::kRespMaxContacts;
    int n = 0;
    bool overflow = false;
    V3 primK[kMax];
    std::uint32_t nonPrim[kMaxNonPrim];
    const uint4 *cand4 = reinterpret_cast<const uint4 *>(cand);   // cand: 16-byte aligned (DetectT)
#pragma unroll 1   // not 8 copies of the narrow phase: the kernel is instruction-fetch bound
    for (int k = 0; k < 8 && !overflow; ++k) {
        if (!(shapes >> k & 1u)) continue;
        const I3 ell = ShapePose(w.car, carIso, absorb, k);
        const float radA[3] = {w.car.rad[k][0], w.car.rad[k][1], w.car.rad[k][2]};
        const V3 irad{w.car.irad[k][0], w.car.irad[k][1], w.car.irad[k][2]};
        const float rm = fmaxf(radA[0], fmaxf(radA[1], radA[2])) * 1.0001f;   // EdgeOut's bound and margin base for this shape
        const float mb = 0.01f + 1e-5f * ((fabsf(ell.t.x) + fabsf(ell.t.y)) + fabsf(ell.t.z));
        const V3 cs{ell.t.x * 0x1p-15f, ell.t.y * 0x1p-15f, ell.t.z * 0x1p-15f};   // EdgeOut's centre, 2^-15 c (exact)
        const int n0 = n;
        int np = 0, nq = 0;
        bool full = false;
        UnitFrame f;
        f.inst = 0xFFFFFFFFu;
        // first the candidates (a primary goes out at once, a non-primary's rank is kept), then the kept ranks again (their
        // contacts tested against the shape's primaries): one loop, so the narrow phase has one copy in the code. The
        // candidates in windows of 64: the shape's ones as a bit mask (the window read as uint4, the loads independent),
        // then only those entries read again.
        unsigned mlo = 0u, mhi = 0u;   // the window's candidates with shape k
        int wb = -64, q = 0;           // the window's first entry
        bool second = false;
        for (;;) {
            std::uint32_t r;
            if (!second) {
                if ((mlo | mhi) == 0u) {
                    wb += 64;
                    if (wb >= nc) {
                        if (nq == 0) break;
                        second = true;
                        f.inst = 0xFFFFFFFFu;
                        continue;
                    }
                    for (int j = 0; j < 64 && wb + j < nc; j += 4) {
                        const uint4 e = cand4[(wb + j) >> 2];
                        const unsigned b = (e.x >> k & 1u) | (e.y >> k & 1u) << 1 | (e.z >> k & 1u) << 2 | (e.w >> k & 1u) << 3;
                        if (j < 32) mlo |= b << j; else mhi |= b << (j - 32);
                    }
                    const int m = nc - wb;   // entries in the window (the last uint4 may run past nc)
                    if (m < 32) mlo &= (1u << m) - 1u;
                    if (m < 64) mhi &= m <= 32 ? 0u : (1u << (m - 32)) - 1u;
                    {   // the plane prefilter first
                        const unsigned long long pm = PlaneFilter<kCount>(static_cast<unsigned long long>(mhi) << 32 | mlo, cand + wb, w.rankN, w.rankE,
                                                                          ell, radA, cs, mb, rm, k, cnt);
                        mlo = static_cast<unsigned>(pm);
                        mhi = static_cast<unsigned>(pm >> 32);
                    }
                    continue;
                }
                if (mlo != 0u) {
                    r = cand[wb + __ffs(mlo) - 1] >> 8;
                    mlo &= mlo - 1u;
                } else {
                    r = cand[wb + 32 + __ffs(mhi) - 1] >> 8;
                    mhi &= mhi - 1u;
                }
            } else {
                if (q == nq) break;
                r = nonPrim[q++];
            }
            const uint4 *rp = reinterpret_cast<const uint4 *>(w.rankN + r);   // the record past its plane: three loads issued together
            const uint4 rb = __ldg(rp + 1), rc = __ldg(rp + 2), rd = __ldg(rp + 3);
            const std::uint32_t inst = rb.x, tri = rb.y;
            const float v[9] = {__uint_as_float(rb.z), __uint_as_float(rb.w), __uint_as_float(rc.x), __uint_as_float(rc.y), __uint_as_float(rc.z),
                                __uint_as_float(rc.w), __uint_as_float(rd.x), __uint_as_float(rd.y), __uint_as_float(rd.z)};
            if constexpr (kCount) cnt->frames += inst != f.inst ? 1u : 0u;
            FrameFor(w, ell, irad, inst, f);
            bool face, feat;
            V3 P, Nn, S, Kn;
            {
                V3 vu[3];
                for (int c = 0; c < 3; ++c) vu[c] = L::TPoint(f.m2u, V3{v[3 * c], v[3 * c + 1], v[3 * c + 2]});
                if constexpr (kCount) { if (second) ++cnt->second; else ++cnt->ucRuns; }
                UnitContactW(vu, w.k.dirEpsSq, w.k.collisionDistance, &face, &feat, &P, &Nn, &S, &Kn);
            }
            if (!second) {
                if (!(face || feat)) continue;
                if constexpr (kCount) ++cnt->raws;
                if (!face) {
                    if (nq < kMaxNonPrim) nonPrim[nq++] = r; else ++lost;
                    continue;
                }
                if (n >= kMax) { full = true; break; }
            }
            V3 oP, oN, oS;
            Payload(w, ell, k, IsoOf(w.iso + 12u * inst), f.back, P, Nn, S, &oP, &oN, &oS);
            if (second) {
                int j = 0;
                for (; j < np; ++j) {
                    const model::RespContact &pc = out.c[n0 + j];
                    if (NearlyEqual(Kn, primK[j]) || w.k.normalAlignment < L::Dot(oN, V3{pc.N[0], pc.N[1], pc.N[2]})) break;
                }
                if (j < np) continue;
                if (n >= kMax) { full = true; break; }
            }
            FillOutV(out.c[n++], oP, oN, oS, !second, w.outRec[tri], k);
            if (!second) primK[np++] = Kn;
        }
        if (full) overflow = true;
    }
    out.n = n;
    out.overflow = overflow;
    out.rawLost = lost;
    if constexpr (kCount) cnt->contacts = static_cast<unsigned>(n);
}

// The walk with the union box over NodeQ nodes (children pushed last first; the leaves' records sorted by first rank, so
// the ranks come out increasing); at a leaf, each rank whose quantized box meets the union box is tested exactly against
// the 8 shape boxes, and one that meets a shape's box is a candidate (rank << 8 | shape mask); then ShapesOut.
template <bool kCount>
__device__ inline void DetectT(const DevWorld &w, const float pos[3], const float quat[4], const float absorb[4], GpuContacts &out, DetectCount *cnt) {
    const model::M3 R = model::rot_from_quat(quat);
    const I3 carIso{L::M3<float>{{V3{R.c0.x, R.c0.y, R.c0.z}, V3{R.c1.x, R.c1.y, R.c1.z}, V3{R.c2.x, R.c2.y, R.c2.z}}}, V3{pos[0], pos[1], pos[2]}};
    Box box[8], u;
    {   // the 24 square roots in one block; a lane with a radicand out of SqrtP's range (none for a car: rad^2-weighted sums of
        // a rotation's squared entries, >= the least rad^2) makes its boxes again with the library's
        bool ok = true;
        for (int k = 0; k < 8; ++k) box[k] = ShapeBoxT<true>(w.car, k, ShapePose(w.car, carIso, absorb, k), ok);
        if (!ok)
            for (int k = 0; k < 8; ++k) box[k] = ShapeBoxT<false>(w.car, k, ShapePose(w.car, carIso, absorb, k), ok);
    }
    u = box[0];
    for (int k = 1; k < 8; ++k)
        for (int a = 0; a < 3; ++a) {
            u.lo[a] = L::Select(box[k].lo[a] < u.lo[a], box[k].lo[a], u.lo[a]);
            u.hi[a] = L::Select(u.hi[a] < box[k].hi[a], box[k].hi[a], u.hi[a]);
        }
    // the union box once: without this ptxas recomputes it from the 8 boxes at every node of the walk (41 instructions a
    // node, 3% of the kernel's), keeping 3 registers fewer
    for (int a = 0; a < 3; ++a) {
        asm volatile("" : "+f"(u.lo[a]));
        asm volatile("" : "+f"(u.hi[a]));
    }
    // BoxPlaneMask's margin without |w|: twice PlaneMayTouch's for any centre in the union box (|c|_1 <= the union box's)
    const float mPlane = 0.02f + 2e-5f * ((fmaxf(fabsf(u.lo[0]), fabsf(u.hi[0])) + fmaxf(fabsf(u.lo[1]), fabsf(u.hi[1]))) +
                                          fmaxf(fabsf(u.lo[2]), fabsf(u.hi[2])));
    alignas(16) std::uint32_t cand[kMaxCand];   // read four at a time (ShapesOut)
    int nc = 0;
    unsigned shapes = 0u;
    std::uint32_t lost = 0u;
    {
        // Two phases (the shape boxes are not live in the walk: no spills at its nodes). Phase 1, the walk with the union box
        // alone: the next node in a register (an inner node's first child is visited next without the stack, its other
        // children pushed last first: the same visiting order as pushing them all); a leaf with hits is kept as a record
        // (first rank, hit mask) at the top of cand[] (record i at kMaxCand - 2 - 2i), inserted in first-rank order (the
        // spatial tree from w.root; the FV tree from node 0, whose leaves come in that order). Phase 2: each hit
        // (records in order, ranks increasing) against the 8 shape boxes and BoxPlaneMask, the candidates (rank << 8 | shape
        // mask) appended after the earlier ones (below the records: nh, the candidates and the pending hits, plus the
        // records' words never pass kMaxCand). When a leaf's record does not fit, the spatial walk starts again on the FV
        // tree; there phase 2 runs first and the walk resumes at it.
        std::uint32_t stack[kStack];
        int sp = 0;
        std::uint32_t node = w.root;
        bool more = true;   // nodes left: `node`, then the stack
        while (more) {
            int nh = nc, nl = 0;   // candidates and pending hits; records
            for (;;) {
                std::uint32_t first;
                bool leaf;
                unsigned mk = QuantMask(w.qnodes[node], u, &first, &leaf);
                if (!leaf) {
                    if constexpr (kCount) { ++cnt->nodes; cnt->childTests += 16u; }
                    if (mk != 0u) {
                        const std::uint32_t c0 = first + static_cast<std::uint32_t>(__ffs(mk) - 1);
                        for (mk &= mk - 1u; mk; mk &= ~(1u << (31 - __clz(mk)))) stack[sp++] = first + static_cast<std::uint32_t>(31 - __clz(mk));
                        node = c0;
                        continue;
                    }
                } else {
                    if (mk != 0u) {
                        if (nh + __popc(mk) + 2 * (nl + 1) > kMaxCand) {
                            // the FV tree (its nodes below the spatial tree's root, or no spatial tree): phase 2 first,
                            // then this leaf again
                            if (node < w.root || w.root == 0u) break;
                            // a spatial walk's later leaves may precede these in rank order, so phase 2 cannot run on
                            // part of them: the walk again on the FV tree, whose leaves come in rank order
                            nh = nc;
                            nl = 0;
                            sp = 0;
                            node = 0u;
                            continue;
                        }
                        int i = nl;   // the records sorted by first rank (the spatial tree visits leaves out of rank order)
                        for (; i > 0; --i) {
                            const uint2 prev = *reinterpret_cast<const uint2 *>(&cand[kMaxCand - 2 - 2 * (i - 1)]);
                            if (prev.x < first) break;
                            *reinterpret_cast<uint2 *>(&cand[kMaxCand - 2 - 2 * i]) = prev;
                        }
                        *reinterpret_cast<uint2 *>(&cand[kMaxCand - 2 - 2 * i]) = make_uint2(first, mk);
                        ++nl;
                        nh += __popc(mk);
                    }
                    if constexpr (kCount) { ++cnt->nodes; cnt->childTests += 16u; cnt->leafHits += __popc(mk); }
                }
                if (sp == 0) { more = false; break; }
                node = stack[--sp];
            }
            std::uint32_t rFirst = 0u;
            unsigned rMask = 0u;
            for (int i = 0;;) {
                if (rMask == 0u) {   // the next record
                    if (i == nl) break;
                    const uint2 rec = *reinterpret_cast<const uint2 *>(&cand[kMaxCand - 2 - 2 * i++]);
                    rFirst = rec.x;
                    rMask = rec.y;
                }
                const std::uint32_t r = rFirst + static_cast<std::uint32_t>(__ffs(rMask) - 1);
                rMask &= rMask - 1u;
                unsigned sm = 0u;
                const float4 *bp = reinterpret_cast<const float4 *>(w.rankBox + r);   // the box and the plane: three loads together
                const float4 b0 = __ldg(bp), b1 = __ldg(bp + 1);
                const float4 pl = __ldg(reinterpret_cast<const float4 *>(w.rankN + r));
                sm = OverlapMask8(b0, b1, box);
                // UnionFront (the union box in front of the plane beyond the margin: no shape keeps this rank) decided with the
                // box test, one branch for both: the plane's use before the branch keeps its load with the box's (behind
                // OverlapMask8's own branch ptxas issued it only then, a second serial load for every rank meeting a shape)
                const bool front = UnionFront(pl, u, mPlane);
                if constexpr (kCount) {
                    if (sm != 0u && front) { cnt->boxPairs += __popc(sm); cnt->boxPlaneRej += __popc(sm); ++cnt->unionFront; }
                }
                if ((sm == 0u) | front) continue;
                const unsigned pm = BoxPlaneMask(pl, box, mPlane, carIso, w.car);
                if constexpr (kCount) {
                    cnt->boxPairs += __popc(sm);
                    cnt->boxPlaneRej += __popc(sm & ~pm);
                    ++cnt->cands;
                    const RankN &rn = w.rankN[r];
                    const float4 pl = make_float4(rn.plane[0], rn.plane[1], rn.plane[2], rn.plane[3]);
                    bool any = false;
                    for (int k = 0; k < 8; ++k)
                        if (sm >> k & 1u) any |= PlaneMayTouch(pl, ShapePose(w.car, carIso, absorb, k), w.car.rad[k]);
                    if (!any) { cnt->rankRejPairs += __popc(sm); ++cnt->rankRej; }
                }
                sm &= pm;
                if (sm == 0u) continue;
                shapes |= sm;
                cand[nc++] = r << 8 | sm;
            }
            if (more && nc + 16 + 2 > kMaxCand) {   // no room for another leaf: the rest is lost (the output is not the CPU's)
                ++lost;
                break;
            }
        }
    }
    ShapesOut<kCount>(w, carIso, absorb, cand, nc, shapes, lost, out, cnt);
}

// Order-preserving keys for float min / max with __reduce_*_sync (integers); no NaN here.
__device__ __forceinline__ unsigned FKey(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float FUnkey(unsigned k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k); }

__device__ inline void Detect(const DevWorld &w, const float pos[3], const float quat[4], const float absorb[4], GpuContacts &out) {
    DetectT<false>(w, pos, quat, absorb, out, nullptr);
}

}  // namespace tw::gpu::dev

#endif
