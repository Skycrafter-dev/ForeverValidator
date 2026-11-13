// Fast exact ellipsoid-triangle test for the M3 collision stage
// (src/twin/collide/tw_collide.cpp), FV-free.
//
// Same contract as the stage's FV-form narrow phase
// (mesh_collision_queries.cpp SSphereMeshCollide::CollideTriangle in the
// surface's mesh-local frame, then TransformEllipsoidMeshCollisionsToWorldForGmSurf):
// for one (shape, instance) frame and one model triangle it appends FV's raw
// contact, or nothing. Three steps, each of which only rejects:
//
//   1. Mesh-local prefilter (PrefilterQ4, no unit-space work): the ellipsoid
//      centre c must lie in front of the triangle's plane within the support
//      radius h(n), and no further than h(u_i) outside any edge side plane.
//      Every FV contact has a triangle point inside the closed ellipsoid, so
//      this only drops pairs FV rejects; a tolerance above the float error of
//      both forms keeps it conservative.
//   2. Unit-space decisions in squared form (Decide, no sqrt, no divide):
//      degenerate triangle, pd < 0, pd > 1, and per edge in stored order
//      "first violated edge beyond reach". Each test carries a margin above
//      the rounding gap to FV's form. Result: kReject, kFace (FV's FACE for
//      sure: its payload is computed without the edge loop) or kFvForm.
//   3. FV's own form, verbatim, for kFvForm, on the unit-space vertices step 2
//      computed in FV's form: the decision and the payload are FV's bits.
//
// So the output equals FV's whenever steps 1-2 are right to reject, which the
// margins make true up to the error bounds (docs/twin/m3-exact-fast.md §2).
// FastTri / TriVerts are per model triangle (shared by all instances).
// Collide (L = 1) and CollidePackets (AVX-512 lanes = pairs) give the same bits.
#ifndef TW_EXACT_FAST_H
#define TW_EXACT_FAST_H

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tw::exact {

// ---- FV-form geometry (collide/tw_collide.cpp; FV file:function cited there) ----------
struct V { float x, y, z; };
inline float Dot(const V &a, const V &b) {
    const float xy = a.x * b.x + a.y * b.y;
    return xy + a.z * b.z;
}
inline V Add(const V &a, const V &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V Sub(const V &a, const V &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V Scale(const V &a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline V Neg(const V &a) { return {-a.x, -a.y, -a.z}; }
inline V Cross(const V &a, const V &b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float Sqrt(float v) { return std::sqrt(v); }

constexpr float kDirEpsSq = 1.0e-5f * 1.0e-5f;  // PhysicsTolerance::SurfaceDirectionLengthSquared
constexpr float kCollisionDistance = 1.0e-5f;   // PhysicsTolerance::CollisionDistance

inline V NormalizeForCollision(const V &v, float epsSq) {
    const float l2 = Dot(v, v);
    if (epsSq < l2) return Scale(v, 1.0f / Sqrt(l2));
    return v;
}

struct M { V c[3]; };   // GmMat3 columns basisX, basisY, basisZ
struct I { M r; V t; }; // GmIso4

inline M MOf(const float *r9) { return M{{V{r9[0], r9[1], r9[2]}, V{r9[3], r9[4], r9[5]}, V{r9[6], r9[7], r9[8]}}}; }
inline V Row(const M &m, int a) {
    return a == 0 ? V{m.c[0].x, m.c[1].x, m.c[2].x} : a == 1 ? V{m.c[0].y, m.c[1].y, m.c[2].y} : V{m.c[0].z, m.c[1].z, m.c[2].z};
}
inline V TDir(const M &m, const V &v) { return {Dot(Row(m, 0), v), Dot(Row(m, 1), v), Dot(Row(m, 2), v)}; }
inline V TPoint(const I &t, const V &p) { return Add(TDir(t.r, p), t.t); }
inline M Compose(const M &first, const M &second) { return M{{TDir(second, first.c[0]), TDir(second, first.c[1]), TDir(second, first.c[2])}}; }
inline I SetMult(const I &lhs, const I &rhs) { return I{Compose(lhs.r, rhs.r), TPoint(rhs, lhs.t)}; }
inline M Transpose(const M &m) { return M{{Row(m, 0), Row(m, 1), Row(m, 2)}}; }
inline I Inverse(const I &rhs) {
    const M inv = Transpose(rhs.r);
    return I{inv, TDir(inv, Neg(rhs.t))};
}
inline I MultInverse(const I &self, const I &rhs) { return SetMult(self, Inverse(rhs)); }
inline I Diagonal(const V &s) { return I{M{{V{s.x, 0.0f, 0.0f}, V{0.0f, s.y, 0.0f}, V{0.0f, 0.0f, s.z}}}, V{0.0f, 0.0f, 0.0f}}; }
inline void ScaleRows(I &t, const V &s) {
    t.r.c[0].x = s.x * t.r.c[0].x; t.r.c[1].x = t.r.c[1].x * s.x; t.r.c[2].x = s.x * t.r.c[2].x; t.t.x = t.t.x * s.x;
    t.r.c[0].y = s.y * t.r.c[0].y; t.r.c[1].y = t.r.c[1].y * s.y; t.r.c[2].y = s.y * t.r.c[2].y; t.t.y = t.t.y * s.y;
    t.r.c[0].z = s.z * t.r.c[0].z; t.r.c[1].z = t.r.c[1].z * s.z; t.r.c[2].z = s.z * t.r.c[2].z; t.t.z = t.t.z * s.z;
}

// One raw contact, world frame (P, N, Sep) and unit-space merge key K: the stage's Raw.
struct Raw {
    V P, N, Sep, K;
    bool primary;
    std::uint32_t tri;
    std::uint8_t shape = 0;   // set by CollidePackets (Pair::shape)
};

// ---- per (shape, instance) frame ----------------------------------------------------
struct alignas(64) Frame {
    // line 0: the prefilter block (step 1)
    float c[3];              // ellipsoid centre in mesh coordinates (ellipsoidToMesh translation)
    float sgn;               // sign of det(meshToUnit): -1 for a mirrored instance
    float reach;             // rmax + tol: prefilter bound above
    float low;               // -tol: prefilter bound below (plane)
    float tol;               // prefilter tolerance (m)
    float lazy;              // rmin - tol: distances below it are within every support radius
    float q[6];              // support form Q = R^T D^2 R in mesh coordinates: h(w)^2 = w^T Q w (Q00 Q11 Q22 Q01 Q02 Q12)
    float pad0[2];
    // line 1: unit map (step 2); lines 2-3: maps back to world (payload). FV form (GmCollision_Ellipsoid_Mesh)
    I meshToUnit;
    float pad1[4];
    I c2w, n2w;
    float pad2[8];
};
static_assert(sizeof(Frame) == 256, "Frame: four cache lines");

// The FV-form part (the stage's per-instance transforms); also sets c. meshIso: FV localToWorld
// (rotation columns, then translation).
inline void MakeFrameFv(const float ellR[9], const float ellT[3], const float radii[3], const float meshIso[12], Frame *f) {
    const I ell{MOf(ellR), V{ellT[0], ellT[1], ellT[2]}};
    I mesh;
    mesh.r = MOf(meshIso);
    mesh.t = V{meshIso[9], meshIso[10], meshIso[11]};
    const V rad = {radii[0], radii[1], radii[2]};
    const V invRadii = {1.0f / rad.x, 1.0f / rad.y, 1.0f / rad.z};
    const I ellipsoidToMesh = MultInverse(ell, mesh);
    const I meshToEllipsoid = Inverse(ellipsoidToMesh);
    f->meshToUnit = meshToEllipsoid;
    ScaleRows(f->meshToUnit, invRadii);
    f->c2w = SetMult(MultInverse(Diagonal(rad), meshToEllipsoid), mesh);
    f->n2w = SetMult(MultInverse(Diagonal(invRadii), meshToEllipsoid), mesh);
    f->c[0] = ellipsoidToMesh.t.x;
    f->c[1] = ellipsoidToMesh.t.y;
    f->c[2] = ellipsoidToMesh.t.z;
}

// MakeFrameFv split in two (architect B's form, tw_collide.cpp): the parts that depend only on the
// shape's rotation and radii and the instance (FvRot, shareable between shapes and lanes with the same
// bits), and the translation parts from the shape's centre (FvFrameFrom). The result is MakeFrameFv's
// bit for bit (B's checkFrames test).
struct FvRot {
    std::uint32_t instance = 0xFFFFFFFFu;
    float key[12];                       // the shape's rotation and radii (bits)
    I mesh, invMesh;
    M m2eR, invm2eR, m2uR, c2wR, n2wR;
    V invRadii;
};
inline void MakeFvRot(const float ellR[9], const float radii[3], const float meshIso[12], std::uint32_t instance, FvRot *r) {
    r->instance = instance;
    for (int i = 0; i < 9; ++i) r->key[i] = ellR[i];
    for (int i = 0; i < 3; ++i) r->key[9 + i] = radii[i];
    r->mesh.r = MOf(meshIso);
    r->mesh.t = V{meshIso[9], meshIso[10], meshIso[11]};
    const V rad = {radii[0], radii[1], radii[2]};
    r->invRadii = {1.0f / rad.x, 1.0f / rad.y, 1.0f / rad.z};
    r->invMesh = Inverse(r->mesh);
    const M e2mR = Compose(MOf(ellR), r->invMesh.r);   // ellipsoidToMesh = SetMult(ell, Inverse(mesh))
    r->m2eR = Transpose(e2mR);                         // meshToEllipsoid = Inverse(ellipsoidToMesh)
    I m2u{r->m2eR, V{0.0f, 0.0f, 0.0f}};               // meshToUnit = ScaleRows(meshToEllipsoid, 1 / radii)
    ScaleRows(m2u, r->invRadii);
    r->m2uR = m2u.r;
    r->invm2eR = Transpose(r->m2eR);                   // Inverse(meshToEllipsoid), in MultInverse(Diagonal(.), m2e)
    r->c2wR = Compose(Compose(Diagonal(rad).r, r->invm2eR), r->mesh.r);
    r->n2wR = Compose(Compose(Diagonal(r->invRadii).r, r->invm2eR), r->mesh.r);
}
inline void FvFrameFrom(const FvRot &r, const float ellT[3], Frame *f) {
    const V e2mT = TPoint(r.invMesh, V{ellT[0], ellT[1], ellT[2]});
    const V m2eT = TDir(r.m2eR, Neg(e2mT));
    const V invm2eT = TDir(r.invm2eR, Neg(m2eT));
    const V aT = Add(TDir(r.invm2eR, V{0.0f, 0.0f, 0.0f}), invm2eT);   // SetMult(Diagonal(.), Inverse(m2e)).t (Diagonal's t = 0)
    const V wT = TPoint(r.mesh, aT);                                    // SetMult(., mesh).t, the same for c2w and n2w
    f->meshToUnit.r = r.m2uR;
    f->meshToUnit.t = V{m2eT.x * r.invRadii.x, m2eT.y * r.invRadii.y, m2eT.z * r.invRadii.z};
    f->c2w = I{r.c2wR, wT};
    f->n2w = I{r.n2wR, wT};
    f->c[0] = e2mT.x;
    f->c[1] = e2mT.y;
    f->c[2] = e2mT.z;
}

// The prefilter fields. modelAbs: max |coordinate| of the model's vertices (Model lo/hi). The
// tolerance covers the float error of the prefilter in mesh coordinates and FV's own rounding of the
// unit-space plane and edge distances (docs/twin/m3-exact-fast.md §2).
inline void MakeFrameBounds(const float radii[3], float modelAbs, Frame *f) {
    // meshToUnit = D^-1 R (rows of the mesh -> ellipsoid rotation R scaled by 1/r)
    const M &a = f->meshToUnit.r;
    const float det = a.c[0].x * (a.c[1].y * a.c[2].z - a.c[2].y * a.c[1].z) - a.c[1].x * (a.c[0].y * a.c[2].z - a.c[2].y * a.c[0].z) +
                      a.c[2].x * (a.c[0].y * a.c[1].z - a.c[1].y * a.c[0].z);
    f->sgn = det < 0.0f ? -1.0f : 1.0f;
    const float r0 = radii[0], r1 = radii[1], r2 = radii[2];
    const float mx = r0 > r1 ? r0 : r1, mn = r0 < r1 ? r0 : r1;   // plain compares: no libm calls (finite radii)
    const float rmax = mx > r2 ? mx : r2, rmin = mn < r2 ? mn : r2;
    const float cabs = std::fabs(f->c[0]) + std::fabs(f->c[1]) + std::fabs(f->c[2]);
#ifndef TW_EXACT_TOL_SCALE
#define TW_EXACT_TOL_SCALE 1.0f
#endif
    const float tol = TW_EXACT_TOL_SCALE * (1.0e-4f * rmax + 4.0e-6f * (rmax / rmin) * (cabs + 3.0f * modelAbs));
    f->reach = rmax + tol;
    f->low = -tol;
    f->tol = tol;
    f->lazy = rmin - tol;
    // Q = R^T D^2 R = A^T D^4 A with A = meshToUnit (row j of A = row j of R / r_j):
    // Q_ab = sum_j r_j^4 A_ja A_jb, A_ja = a.c[a][j]
    const float w[3] = {(r0 * r0) * (r0 * r0), (r1 * r1) * (r1 * r1), (r2 * r2) * (r2 * r2)};
    auto e = [&](int col, int j) { const V &c = a.c[col]; return j == 0 ? c.x : j == 1 ? c.y : c.z; };
    auto q = [&](int x, int y) { return (w[0] * e(x, 0) * e(y, 0) + w[1] * e(x, 1) * e(y, 1)) + w[2] * e(x, 2) * e(y, 2); };
    f->q[0] = q(0, 0); f->q[1] = q(1, 1); f->q[2] = q(2, 2);
    f->q[3] = q(0, 1); f->q[4] = q(0, 2); f->q[5] = q(1, 2);
}

inline Frame MakeFrame(const float ellR[9], const float ellT[3], const float radii[3], const float meshIso[12], float modelAbs) {
    Frame f;
    MakeFrameFv(ellR, ellT, radii, meshIso, &f);
    MakeFrameBounds(radii, modelAbs, &f);
    return f;
}

// ---- per model triangle (mesh-local, double) ---------------------------------------------
// Four planes, lane j: j = 0 the triangle plane (unit front normal of (v1 - v0) x (v2 - v0), offset
// n . v0), j = 1..3 the side plane of edge j-1 (v_i -> v_i+1: outward unit normal in the triangle
// plane, offset u . v_i). Stored by component so one 16-byte vector holds a component of all four.
// A zero-area triangle or zero-length edge gets a zero plane (it never rejects).
struct alignas(64) FastTri {
    float x[4], y[4], z[4], d[4];
};
static_assert(sizeof(FastTri) == 64, "FastTri is one cache line");

struct TriVerts { V v[3]; };   // mesh-local vertices, FV stored order

inline void MakeFastTri(const V v[3], FastTri *t) {
    *t = FastTri{};
    const double p[3][3] = {{v[0].x, v[0].y, v[0].z}, {v[1].x, v[1].y, v[1].z}, {v[2].x, v[2].y, v[2].z}};
    double e1[3], e2[3];
    for (int k = 0; k < 3; ++k) { e1[k] = p[1][k] - p[0][k]; e2[k] = p[2][k] - p[0][k]; }
    double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const double nl = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (!(nl > 0.0)) return;
    for (int k = 0; k < 3; ++k) n[k] /= nl;
    auto put = [&](int j, const double w[3], const double *o) {
        t->x[j] = static_cast<float>(w[0]);
        t->y[j] = static_cast<float>(w[1]);
        t->z[j] = static_cast<float>(w[2]);
        t->d[j] = static_cast<float>(w[0] * o[0] + w[1] * o[1] + w[2] * o[2]);
    };
    put(0, n, p[0]);
    for (int i = 0; i < 3; ++i) {
        const double *a = p[i], *b = p[(i + 1) % 3];
        const double e[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        double u[3] = {e[1] * n[2] - e[2] * n[1], e[2] * n[0] - e[0] * n[2], e[0] * n[1] - e[1] * n[0]};   // e x n: outward
        const double ul = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        if (!(ul > 0.0)) continue;
        for (int k = 0; k < 3; ++k) u[k] /= ul;
        put(i + 1, u, a);
    }
}

// ---- step 1: mesh-local prefilter ------------------------------------------------------
// Signed distance of the ellipsoid centre from plane j (mesh coordinates).
inline float PlaneDist(const Frame &f, const FastTri &t, int j) {
    return ((t.x[j] * f.c[0] + t.y[j] * f.c[1]) + t.z[j] * f.c[2]) - t.d[j];
}
// Squared support radius of the ellipsoid along plane j's unit normal w: w^T Q w.
inline float Support2(const Frame &f, const FastTri &t, int j) {
    const float *q = f.q;
    const float w0 = t.x[j], w1 = t.y[j], w2 = t.z[j];
    const float a = (q[0] * w0 + q[3] * w1) + q[4] * w2;
    const float b = (q[3] * w0 + q[1] * w1) + q[5] * w2;
    const float c = (q[4] * w0 + q[5] * w1) + q[2] * w2;
    return (a * w0 + b * w1) + c * w2;
}
constexpr float kSupportSlack = 1.0001f;   // Q's float rounding (relative ~1e-6)

// rmax bounds only: front of the plane within rmax, no further than rmax outside a side plane.
inline bool Prefilter(const Frame &f, const FastTri &t) {
    const float s = f.sgn * PlaneDist(f, t, 0);
    if (s < f.low || s > f.reach) return false;
    for (int j = 1; j < 4; ++j)
        if (PlaneDist(f, t, j) > f.reach) return false;
    return true;
}

// Exact support radii: the ellipsoid within reach of the plane (0 <= sd <= h(n)) and not beyond any
// edge side plane (se_i <= h(u_i)). A support is evaluated only where rmin < distance.
inline bool PrefilterQ(const Frame &f, const FastTri &t) {
    for (int j = 0; j < 4; ++j) {
        const float dist = j == 0 ? f.sgn * PlaneDist(f, t, 0) : PlaneDist(f, t, j);
        if ((j == 0 && dist < f.low) || dist > f.reach) return false;
        if (dist > f.lazy) {
            const float x = dist - f.tol;
            if (x > 0.0f && x * x > Support2(f, t, j) * kSupportSlack) return false;
        }
    }
    return true;
}

// PrefilterQ on the four planes at once (SSE: lane j = plane j, the same operations per lane).
inline bool PrefilterQ4(const Frame &f, const FastTri &t) {
    const __m128 X = _mm_load_ps(t.x), Y = _mm_load_ps(t.y), Z = _mm_load_ps(t.z), D = _mm_load_ps(t.d);
    const __m128 cx = _mm_set1_ps(f.c[0]), cy = _mm_set1_ps(f.c[1]), cz = _mm_set1_ps(f.c[2]);
    __m128 dist = _mm_sub_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(X, cx), _mm_mul_ps(Y, cy)), _mm_mul_ps(Z, cz)), D);
    dist = _mm_mul_ps(dist, _mm_setr_ps(f.sgn, 1.0f, 1.0f, 1.0f));
    const __m128 out = _mm_or_ps(_mm_cmpgt_ps(dist, _mm_set1_ps(f.reach)), _mm_cmplt_ps(dist, _mm_setr_ps(f.low, -INFINITY, -INFINITY, -INFINITY)));
    if (_mm_movemask_ps(out)) return false;
    const __m128 lazy = _mm_cmpgt_ps(dist, _mm_set1_ps(f.lazy));
    if (!_mm_movemask_ps(lazy)) return true;
    const float *q = f.q;
    const __m128 a = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(q[0]), X), _mm_mul_ps(_mm_set1_ps(q[3]), Y)), _mm_mul_ps(_mm_set1_ps(q[4]), Z));
    const __m128 b = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(q[3]), X), _mm_mul_ps(_mm_set1_ps(q[1]), Y)), _mm_mul_ps(_mm_set1_ps(q[5]), Z));
    const __m128 c = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(q[4]), X), _mm_mul_ps(_mm_set1_ps(q[5]), Y)), _mm_mul_ps(_mm_set1_ps(q[2]), Z));
    const __m128 h2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(a, X), _mm_mul_ps(b, Y)), _mm_mul_ps(c, Z));
    const __m128 x = _mm_sub_ps(dist, _mm_set1_ps(f.tol));
    const __m128 far = _mm_and_ps(_mm_and_ps(lazy, _mm_cmpgt_ps(x, _mm_setzero_ps())),
                                  _mm_cmpgt_ps(_mm_mul_ps(x, x), _mm_mul_ps(h2, _mm_set1_ps(kSupportSlack))));
    return _mm_movemask_ps(far) == 0;
}

// ---- step 3: FV's form (stage TriangleCollide), unit space -> world ----------------------
struct FvTriangle {
    V n;
    std::vector<Raw> *out;
    std::uint32_t tri;
    static constexpr float kR = 1.0f;

    int EmitFeature(const V &point, float minDistSq, bool requireContainment) {
        const V toCenter = Sub(V{0.0f, 0.0f, 0.0f}, point);
        const float dSq = Dot(toCenter, toCenter);
        if ((requireContainment && kR * kR < dSq) || !(minDistSq < dSq)) return 0;
        const float d = Sqrt(dSq);
        const float inv = 1.0f / d;
        const V normal = {toCenter.x * inv, toCenter.y * inv, toCenter.z * inv};
        const float dmr = d - kR;
        const float ps = dmr * inv;
        const V pen = {toCenter.x * ps, toCenter.y * ps, toCenter.z * ps};
        const float along = Dot(pen, n);
        out->push_back(Raw{point, normal, Scale(n, along), n, false, tri});
        return 1;
    }
    int EmitEndpointB(const V &point, float minDistance) {
        const V toCenter = Sub(V{0.0f, 0.0f, 0.0f}, point);
        const float dSq = Dot(toCenter, toCenter);
        const float d = Sqrt(dSq);
        if (kR * kR < d || !(minDistance < d)) return 0;
        const float ed = Sqrt(d);
        const float inv = 1.0f / ed;
        const V normal = {toCenter.x * inv, toCenter.y * inv, toCenter.z * inv};
        const float edmr = ed - kR;
        const float ps = edmr * inv;
        const V pen = {toCenter.x * ps, toCenter.y * ps, toCenter.z * ps};
        const float along = Dot(pen, n);
        out->push_back(Raw{point, normal, Scale(n, along), n, false, tri});
        return 1;
    }
    int Collide(const V v[3]) {
        const V zero = {0.0f, 0.0f, 0.0f};
        const float pd = Dot(Sub(zero, v[0]), n);
        if (kR < pd || pd < 0.0f) return 0;
        const float edgeReach = Sqrt(kR * kR - pd * pd);
        const V projected = Add(zero, Scale(n, -pd));
        for (int e = 0; e < 3; ++e) {
            const int nx = e == 2 ? 0 : e + 1;
            const V start = v[e], end = v[nx];
            const V dir = NormalizeForCollision(Sub(end, start), kDirEpsSq);
            const V edgeNormal = Cross(dir, n);
            const float edgeDistance = Dot(Sub(projected, start), edgeNormal);
            if (edgeReach < edgeDistance) return 0;
            if (edgeDistance > 0.0f) {
                const float fromStart = Dot(Sub(projected, start), dir);
                if (fromStart < 0.0f) return EmitFeature(start, kDirEpsSq, true);
                const float fromEnd = Dot(Sub(projected, end), dir);
                if (!(0.0f < fromEnd)) return EmitFeature(Add(projected, Scale(edgeNormal, -edgeDistance)), kCollisionDistance, false);
                return EmitEndpointB(end, kDirEpsSq);
            }
        }
        if (pd > 0.0f) {
            out->push_back(Raw{projected, n, Scale(n, pd - kR), n, true, tri});
            return 1;
        }
        return 0;
    }
};

// Unit-space vertices in FV's form.
inline void UnitVerts(const Frame &f, const TriVerts &tv, V vu[3]) {
    for (int c = 0; c < 3; ++c) vu[c] = TPoint(f.meshToUnit, tv.v[c]);
}
// FV's raw normal of the unit-space triangle and its squared length.
inline V UnitNormal(const V vu[3], float *lenSq) {
    const V e01 = Sub(vu[1], vu[0]), e02 = Sub(vu[2], vu[0]);
    const float nX = e02.z * e01.y - e02.y * e01.z;
    const float nY = e01.z * e02.x - e02.z * e01.x;
    const float nZ = e01.x * e02.y - e02.x * e01.y;
    *lenSq = (nY * nY + nX * nX) + nZ * nZ;
    return V{nX, nY, nZ};
}

// FV's narrow phase for one pair on given unit-space vertices (the stage's loop body).
inline int FvCollide(const Frame &f, const V vu[3], std::uint32_t tri, std::vector<Raw> *out) {
    float lenSq;
    const V nr = UnitNormal(vu, &lenSq);
    if (!(lenSq > kDirEpsSq)) return 0;
    const float len = Sqrt(lenSq);
    const float invLen = 1.0f / len;
    const V n = {nr.x * invLen, nr.y * invLen, invLen * nr.z};
    const std::size_t first = out->size();
    FvTriangle tc{n, out, tri};
    if (!tc.Collide(vu)) return 0;
    for (std::size_t r = first; r < out->size(); ++r) {   // TransformEllipsoidMeshCollisionsToWorldForGmSurf
        Raw &c = (*out)[r];
        c.P = TPoint(f.c2w, c.P);
        c.N = NormalizeForCollision(TDir(f.n2w.r, c.N), kDirEpsSq);
        c.Sep = TDir(f.c2w.r, c.Sep);
    }
    return 1;
}

// ---- step 2: squared-form decisions ------------------------------------------------------
// Margin factor: k * FLT_EPSILON with k = 64 (the rounding gap to FV's form is below ~16 eps
// times the magnitudes involved, docs/twin/m3-exact-fast.md §2).
#ifndef TW_EXACT_MARGIN_K
#define TW_EXACT_MARGIN_K 64.0f
#endif
constexpr float kMarginEps = TW_EXACT_MARGIN_K * 1.1920929e-7f;

inline float Abs1(const V &a) { return (std::fabs(a.x) + std::fabs(a.y)) + std::fabs(a.z); }

enum Decision : int { kReject = 0, kFace = 1, kFvForm = 2 };

// What FV's form certainly does with the pair: no contact (kReject), FV's FACE (kFace: the plane is
// certainly in reach and no edge is violated, so FV's edge loop cannot change the result), or
// unsure (kFvForm: FV's form decides).
inline int Decide(const V vu[3], const V &nr, float lenSq) {
    if (!(lenSq > kDirEpsSq)) return kReject;                    // FV's degenerate skip (same value)
    const float P0 = -Dot(vu[0], nr);                            // pd * len
    const float d0 = kMarginEps * Abs1(vu[0]);                   // |pd - P0 / len| bound
    const float P02 = P0 * P0;
    const float lo2 = (d0 * d0) * lenSq;
    if (P0 < 0.0f && P02 > lo2) return kReject;                  // pd < 0
    const float up = 1.0f + d0;
    if (P0 > 0.0f && P02 > (up * up) * lenSq) return kReject;   // pd > 1
    float R2 = lenSq - P02;                                      // (1 - (P0/len)^2) lenSq
    R2 = R2 > 0.0f ? R2 : 0.0f;
    const float slack0 = 2.0f * d0 + d0 * d0;
    for (int e = 0; e < 3; ++e) {
        const V &a = vu[e];
        const V &b = vu[e == 2 ? 0 : e + 1];
        const V E = Sub(b, a);
        const float el2 = Dot(E, E);
        const float L2 = el2 > kDirEpsSq ? el2 : 1.0f;
        const V m = Cross(E, nr);
        const float S = -Dot(a, m);                              // edgeDistance * L * len
        const float de = kMarginEps * (Abs1(a) + 1.0f);          // |edgeDistance - S / (L len)| bound
        const float S2 = S * S;
        const float L2len = L2 * lenSq;
        if (S2 <= (de * de) * L2len) return kFvForm;             // sign of edgeDistance unsure
        if (S < 0.0f) continue;                                  // certainly not violated
        // certainly violated: reject when certainly beyond reach, else FV's form decides
        return S2 > L2 * R2 + L2len * (slack0 + 2.0f * de * (1.0f + d0) + de * de) ? kReject : kFvForm;
    }
    // no edge violated: FV's FACE when 0 < pd <= 1 is certain
    const float dn = 1.0f - d0;
    if (P0 > 0.0f && P02 > lo2 && dn > 0.0f && P02 < (dn * dn) * lenSq) return kFace;
    return kFvForm;
}
inline bool SureReject(const V vu[3], const V &nr, float lenSq) { return Decide(vu, nr, lenSq) == kReject; }

// FV's FACE contact for a pair Decide classed kFace: FV's own operations for n, pd and the payload,
// without the edge loop (which cannot change it).
inline void EmitFace(const Frame &f, const V vu[3], const V &nr, float lenSq, std::uint32_t tri, std::vector<Raw> *out) {
    const float len = Sqrt(lenSq);
    const float invLen = 1.0f / len;
    const V n = {nr.x * invLen, nr.y * invLen, invLen * nr.z};
    const V zero = {0.0f, 0.0f, 0.0f};
    const float pd = Dot(Sub(zero, vu[0]), n);
    const V projected = Add(zero, Scale(n, -pd));
    const V sep = Scale(n, pd - FvTriangle::kR);
    out->push_back(Raw{TPoint(f.c2w, projected), NormalizeForCollision(TDir(f.n2w.r, n), kDirEpsSq), TDir(f.c2w.r, sep), n, true, tri});
}

// One pair: steps 1-3. Appends FV's raw contact (world frame) and returns 1, or returns 0.
inline int Collide(const Frame &f, const FastTri &t, const TriVerts &tv, std::uint32_t tri, std::vector<Raw> *out) {
    if (!PrefilterQ4(f, t)) return 0;
    V vu[3];
    UnitVerts(f, tv, vu);
    float lenSq;
    const V nr = UnitNormal(vu, &lenSq);
    const int d = Decide(vu, nr, lenSq);
    if (d == kReject) return 0;
    if (d == kFace) {
        EmitFace(f, vu, nr, lenSq, tri, out);
        return 1;
    }
    return FvCollide(f, vu, tri, out);
}

// The stage's FV-form narrow phase for one pair (reference).
inline int CollideFv(const Frame &f, const TriVerts &tv, std::uint32_t tri, std::vector<Raw> *out) {
    V vu[3];
    UnitVerts(f, tv, vu);
    return FvCollide(f, vu, tri, out);
}

// ---- packets (tw_exact_fast.cpp, AVX-512) ------------------------------------------------
struct Pair {
    std::uint32_t frame;     // index into the caller's frame table
    std::uint32_t modelTri;  // index into the FastTri / TriVerts tables
    std::uint32_t tri;       // world triangle id (Raw::tri)
    std::uint32_t shape;     // copied to Raw::shape
};
struct PacketStats {
    std::uint64_t pairs = 0, prefiltered = 0, sureRejected = 0, fvRuns = 0, faces = 0;
};
// n pairs in FV order; frames[pair.frame]; raw contacts appended in pair order (same as calling
// Collide on each pair). Needs AVX-512F; otherwise falls back to Collide per pair.
// tris == nullptr: the pairs already passed step 1 (e.g. tw_car_query's world-space prefilter), steps 2-3 only.
// `upto` < 3 stops after step 1 or 2 (timing probes; no contacts appended).
void CollidePackets(const Frame *frames, const FastTri *tris, const TriVerts *verts, const Pair *pairs, std::size_t n,
                    std::vector<Raw> *out, PacketStats *stats = nullptr, int upto = 3);

}  // namespace tw::exact

#endif
