// M3 exact collision stage: see tw_collide.h. Every float expression below
// follows FV's source operation for operation (file:function cited), compiled
// with -ffp-contract=off, so the contacts are FV's bits.
#include "tw_collide.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

#include "tw_exact_lanes.h"
#include "tw_car_query.h"
#include "tw_math.h"
#include "tw_patches.h"
#include "vec3.h"

#include <x86intrin.h>
#include <xmmintrin.h>

namespace tw {
namespace collide {

namespace {

struct V { float x, y, z; };

// geometry_helpers.cpp SumProducts / Dot; gm_vector.cpp GmVec3::Dot (same order)
inline float Dot(const V &a, const V &b) {
    const float xy = a.x * b.x + a.y * b.y;
    return xy + a.z * b.z;
}
inline V Add(const V &a, const V &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V Sub(const V &a, const V &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V Scale(const V &a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline V Neg(const V &a) { return {-a.x, -a.y, -a.z}; }
inline V Cross(const V &a, const V &b) {  // gm_vector.cpp GmVec3::Cross
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float Sqrt(float v) { return std::sqrt(v); }  // FV CIsqrt: native sqrtss (correctly rounded)

constexpr float kDirEpsSq = 1.0e-5f * 1.0e-5f;  // PhysicsTolerance::SurfaceDirectionLengthSquared
constexpr float kCollisionDistance = 1.0e-5f;   // PhysicsTolerance::CollisionDistance
constexpr float kNormalAlignment = 0.8660254f;  // PhysicsTolerance::SphereContactNormalAlignment

// gm_vector.cpp GmVec3::NormalizeForCollision
inline V NormalizeForCollision(const V &v, float epsSq) {
    const float l2 = Dot(v, v);
    if (epsSq < l2) return Scale(v, 1.0f / Sqrt(l2));
    return v;
}

// FV GmMat3 as columns c[0..2] = basisX, basisY, basisZ.
struct M { V c[3]; };
struct I { M r; V t; };

inline M MOf(const float *r9) { return M{{V{r9[0], r9[1], r9[2]}, V{r9[3], r9[4], r9[5]}, V{r9[6], r9[7], r9[8]}}}; }
inline I IOf(const Iso &s) { return I{MOf(s.r), V{s.t[0], s.t[1], s.t[2]}}; }
inline V Row(const M &m, int a) {  // GmMat3::Row
    return a == 0 ? V{m.c[0].x, m.c[1].x, m.c[2].x} : a == 1 ? V{m.c[0].y, m.c[1].y, m.c[2].y} : V{m.c[0].z, m.c[1].z, m.c[2].z};
}
// geometry_helpers.cpp TransformDirection: (Row X . v, Row Y . v, Row Z . v)
inline V TDir(const M &m, const V &v) { return {Dot(Row(m, 0), v), Dot(Row(m, 1), v), Dot(Row(m, 2), v)}; }
inline V TPoint(const I &t, const V &p) { return Add(TDir(t.r, p), t.t); }
// geometry_helpers.cpp Compose(first, second): apply first, then second
inline M Compose(const M &first, const M &second) { return M{{TDir(second, first.c[0]), TDir(second, first.c[1]), TDir(second, first.c[2])}}; }
// gm_transform.cpp GmIso4::SetMult(lhs, rhs)
inline I SetMult(const I &lhs, const I &rhs) { return I{Compose(lhs.r, rhs.r), TPoint(rhs, lhs.t)}; }
// gm_matrix.cpp SetTranspose: basis = rows of the source
inline M Transpose(const M &m) { return M{{Row(m, 0), Row(m, 1), Row(m, 2)}}; }
// gm_transform.cpp GmIso4::SetInverse
inline I Inverse(const I &rhs) {
    const M inv = Transpose(rhs.r);
    return I{inv, TDir(inv, Neg(rhs.t))};
}
// gm_transform.cpp GmIso4::MultInverse: this * inverse(rhs)
inline I MultInverse(const I &self, const I &rhs) { return SetMult(self, Inverse(rhs)); }
// gm_transform.cpp SetNUScaleTrans (SetDiagonal)
inline I Diagonal(const V &s) { return I{M{{V{s.x, 0.0f, 0.0f}, V{0.0f, s.y, 0.0f}, V{0.0f, 0.0f, s.z}}}, V{0.0f, 0.0f, 0.0f}}; }
// GmIso4::ScaleRowsForGmSurf: row a of the rotation and translation a scaled by s[a]
inline void ScaleRows(I &t, const V &s) {
    t.r.c[0].x = s.x * t.r.c[0].x; t.r.c[1].x = t.r.c[1].x * s.x; t.r.c[2].x = s.x * t.r.c[2].x; t.t.x = t.t.x * s.x;
    t.r.c[0].y = s.y * t.r.c[0].y; t.r.c[1].y = t.r.c[1].y * s.y; t.r.c[2].y = s.y * t.r.c[2].y; t.t.y = t.t.y * s.y;
    t.r.c[0].z = s.z * t.r.c[0].z; t.r.c[1].z = t.r.c[1].z * s.z; t.r.c[2].z = s.z * t.r.c[2].z; t.t.z = t.t.z * s.z;
}

struct Raw {
    V P, N, Sep, K;
    bool primary;
    std::uint32_t tri;
};

// mesh_collision_queries.cpp SSphereMeshCollide (sphere centre 0, radius 1, unit space)
struct TriangleCollide {
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
        const float ed = Sqrt(d);  // FV: endpoint B uses a second root
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

// geometry_helpers.cpp NearlyEqual(value, reference)
inline bool NearlyEqual(float value, float reference) {
    const float tol = std::fabs(reference) * 1.0e-5f;
    return reference - tol <= value && value <= reference + tol;
}
inline bool NearlyEqual(const V &a, const V &b) { return NearlyEqual(a.x, b.x) && NearlyEqual(a.y, b.y) && NearlyEqual(a.z, b.z); }

struct Scratch {
    std::vector<std::uint32_t> cand;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> order;  // (fvOrder, tri)
    std::vector<Raw> raw;
    std::vector<int> merged;
};

}  // namespace

void ShapePoses(const CarShapes &car, const float pos[3], const float quat[4], const float absorb[4], ShapePose out[8]) {
    const model::M3 R = model::rot_from_quat(quat);   // FV GmMat3::Set(GmQuat)
    const I carIso{M{{V{R.c0.x, R.c0.y, R.c0.z}, V{R.c1.x, R.c1.y, R.c1.z}, V{R.c2.x, R.c2.y, R.c2.z}}}, V{pos[0], pos[1], pos[2]}};
    for (std::uint32_t k = 0; k < car.count && k < 8u; ++k) {
        const CarShape &sh = car.shape[k];
        I local;
        if (sh.wheel >= 0) {   // FV CopyRestSurfaceIso + OffsetCurrentSurfaceY(-absorb): translation.y += -absorb
            local = IOf(car.wheelRest[sh.wheel]);
            local.t.y += -absorb[sh.wheel];
        } else {
            local = IOf(sh.bodyPose);
        }
        const I w = SetMult(local, carIso);
        for (int c = 0; c < 3; ++c) { out[k].world.r[3 * c] = w.r.c[c].x; out[k].world.r[3 * c + 1] = w.r.c[c].y; out[k].world.r[3 * c + 2] = w.r.c[c].z; }
        out[k].world.t[0] = w.t.x; out[k].world.t[1] = w.t.y; out[k].world.t[2] = w.t.z;
        for (int a = 0; a < 3; ++a) out[k].radii[a] = sh.radii[a];
    }
}

bool ExactCollision::Init(const world::WorldPackView &view, const CarShapes &car, std::string *error, const CollideOptions &opt) {
    if (car.magic != CarShapes::kMagic || car.count != 8u) {
        if (error) *error = "collide: car shapes missing or not 8";
        return false;
    }
    view_ = &view;
    car_ = car;
    opt_ = opt;
    for (int k = 0; k < 8; ++k) {   // the shapes' constants in lanes (LaneShapes / LaneFilters)
        const CarShape &sh = car.shape[k];
        const Iso &local = sh.wheel >= 0 ? car.wheelRest[sh.wheel] : sh.bodyPose;
        for (int j = 0; j < 9; ++j) lane_.lr[j][k] = local.r[j];
        for (int a = 0; a < 3; ++a) { lane_.lt[a][k] = local.t[a]; lane_.rad[a][k] = sh.radii[a]; }
        const cq::ShapeConst c = cq::MakeShapeConst(sh.radii);
        for (int a = 0; a < 3; ++a) lane_.r2[a][k] = c.r2[a];
        lane_.rmax[k] = c.rmax;
        lane_.rmin[k] = c.rmin;
        lane_.ratio[k] = c.ratio;
        lane_.wheel[k] = sh.wheel;
    }
    static std::atomic<std::uint64_t> nextId{1};
    id_ = nextId.fetch_add(1);
    fast_.resize(view.modelTris.size());
    triVerts_.resize(view.modelTris.size());
    modelAbs_.assign(view.models.size(), 0.0f);
    for (std::size_t mi = 0; mi < view.models.size(); ++mi) {
        const world::Model &m = view.models[mi];
        float a = 0.0f;
        for (int k = 0; k < 3; ++k) a = math::max(a, math::max(std::fabs(m.lo[k]), std::fabs(m.hi[k])));
        modelAbs_[mi] = a;
        for (std::uint32_t t = 0; t < m.triCount; ++t) {
            const world::ModelTri &mt = view.modelTris[m.firstTri + t];
            exact::V vv[3];
            for (int c = 0; c < 3; ++c) { const world::Vec3f &p = view.modelVerts[m.firstVert + mt.v[c]]; vv[c] = exact::V{p.x, p.y, p.z}; }
            exact::MakeFastTri(vv, &fast_[m.firstTri + t]);
            triVerts_[m.firstTri + t] = exact::TriVerts{{vv[0], vv[1], vv[2]}};
        }
    }
    if (opt.patches != kPatchOff || opt.closedWheelFace) patch::Build(view, 1e-4f, 1e-4f, &patches_);
    World w;
    w.tris.assign(view.tris.begin(), view.tris.end());
    if (opt.excludeMovingPair)   // an empty box never overlaps a query
        for (std::size_t i = 0; i < w.tris.size(); ++i)
            if (view.instances[view.triInfo[i].instance].flags & world::kInstMovingPair)
                for (int a = 0; a < 3; ++a) { w.tris[i].lo[a] = 1e30f; w.tris[i].hi[a] = -1e30f; }
    bvh_.Build(w);
    return true;
}

const char ExactCache::kKind = 0;

ExactCache::ExactCache() : step::CollisionCache(&kKind) {}

void ExactCache::Flush() {
    memo.owner = 0;
    memo.haveLast = false;
    last.owner = 0;
}

std::size_t ExactCache::Bytes() const {
    const cq::Memo &m = memo.q;
    std::size_t b = m.tri.capacity() * sizeof(std::uint32_t) + m.ref.capacity() * sizeof(cq::Ref) + m.planes.capacity() * sizeof(exact::FastTri) +
                    m.scratch.capacity() * sizeof(std::uint64_t) + (m.pairEntry.capacity() + m.pairShape.capacity()) * sizeof(std::uint32_t);
    for (int a = 0; a < 3; ++a) b += (m.lo[a].capacity() + m.hi[a].capacity()) * sizeof(float);
    for (int k = 0; k < 8; ++k) b += memo.lists.idx[k].capacity() * sizeof(std::uint32_t);
    return b;
}

void ExactCollision::Detect(const model::TwinState &s, step::SubstepContacts &out) const {
    DetectPose(s.pos, s.quat, s.absorb, out);
}

std::unique_ptr<step::CollisionCache> ExactCollision::NewCache() const { return std::make_unique<ExactCache>(); }

void ExactCollision::DetectCached(const model::TwinState &s, step::CollisionCache *cache, step::SubstepContacts &out) const {
    DetectPose(s.pos, s.quat, s.absorb, out, cache);
}

// ShapePoses + ShapeBoxes with lane k = shape k (math::vf<8>; every lane operation is the scalar one, so
// the bits are ShapePoses' and ShapeBoxes'). wr / wt: the world rotations (columns) and centres, SoA.
void ExactCollision::LaneShapes(const float pos[3], const float quat[4], const float absorb[4], ShapePose shapes[8], Box3 q[8], Box3 *u,
                                float wr[9][8], float wt[3][8]) const {
    using F = math::vf<8>;
    const model::M3 R = model::rot_from_quat(quat);   // FV GmMat3::Set(GmQuat); columns c0 c1 c2
    const float Rc[3][3] = {{R.c0.x, R.c0.y, R.c0.z}, {R.c1.x, R.c1.y, R.c1.z}, {R.c2.x, R.c2.y, R.c2.z}};
    alignas(32) float lt[3][8];
    std::memcpy(lt, lane_.lt, sizeof lt);
    for (int k = 0; k < 8; ++k)   // FV OffsetCurrentSurfaceY(-absorb): translation.y += -absorb
        if (lane_.wheel[k] >= 0) lt[1][k] += -absorb[lane_.wheel[k]];
    // SetMult(local, carIso): rotation Compose(local.r, R) (column i = TDir(R, local column i)), translation
    // TPoint(carIso, local.t); TDir's row a . v = (R.c0[a] v.x + R.c1[a] v.y) + R.c2[a] v.z
    for (int i = 0; i < 3; ++i) {
        const F vx = F::load(lane_.lr[3 * i]), vy = F::load(lane_.lr[3 * i + 1]), vz = F::load(lane_.lr[3 * i + 2]);
        for (int a = 0; a < 3; ++a) ((F(Rc[0][a]) * vx + F(Rc[1][a]) * vy) + F(Rc[2][a]) * vz).store(wr[3 * i + a]);
    }
    {
        const F vx = F::load(lt[0]), vy = F::load(lt[1]), vz = F::load(lt[2]);
        for (int a = 0; a < 3; ++a) (((F(Rc[0][a]) * vx + F(Rc[1][a]) * vy) + F(Rc[2][a]) * vz) + F(pos[a])).store(wt[a]);
    }
    // the boxes: h = |r[a]| r0 + |r[3+a]| r1 + |r[6+a]| r2, eps = 1e-3 + 1e-6 |c|, (c - h) - eps .. (c + h) + eps
    alignas(32) float lo[3][8], hi[3][8];
    const F r0 = F::load(lane_.rad[0]), r1 = F::load(lane_.rad[1]), r2 = F::load(lane_.rad[2]);
    for (int a = 0; a < 3; ++a) {
        const F c = F::load(wt[a]);
        const F h = (math::abs(F::load(wr[a])) * r0 + math::abs(F::load(wr[3 + a])) * r1) + math::abs(F::load(wr[6 + a])) * r2;
        const F eps = F(1e-3f) + F(1e-6f) * math::abs(c);
        ((c - h) - eps).store(lo[a]);
        ((c + h) + eps).store(hi[a]);
    }
    *u = EmptyBox();
    for (int k = 0; k < 8; ++k) {
        for (int j = 0; j < 9; ++j) shapes[k].world.r[j] = wr[j][k];
        for (int a = 0; a < 3; ++a) {
            shapes[k].world.t[a] = wt[a][k];
            shapes[k].radii[a] = lane_.rad[a][k];
            q[k].lo[a] = lo[a][k];
            q[k].hi[a] = hi[a][k];
        }
        *u = Union(*u, q[k]);
    }
}

// ShapeFilters with lane k = shape k: tw_car_query MakeShapeFilter's operations per lane.
void ExactCollision::LaneFilters(const cq::Memo &memo, const float wr[9][8], const float wt[3][8], const Box3 q[8], cq::ShapeFilter sf[8]) const {
    using F = math::vf<8>;
    alignas(32) float c[3][8], qq[6][8], tolv[8], reach[8], lazy[8];
    for (int a = 0; a < 3; ++a) (F::load(wt[a]) - F(memo.origin[a])).store(c[a]);
    const F s0 = F::load(lane_.r2[0]), s1 = F::load(lane_.r2[1]), s2 = F::load(lane_.r2[2]);
    const int qa[6] = {0, 1, 2, 0, 0, 1}, qb[6] = {0, 1, 2, 1, 2, 2};
    for (int i = 0; i < 6; ++i) {
        const int a = qa[i], b = qb[i];
        ((s0 * (F::load(wr[a]) * F::load(wr[b])) + s1 * (F::load(wr[3 + a]) * F::load(wr[3 + b]))) + s2 * (F::load(wr[6 + a]) * F::load(wr[6 + b])))
                .store(qq[i]);
    }
    const F rmax = F::load(lane_.rmax), rmin = F::load(lane_.rmin), ratio = F::load(lane_.ratio);
    const F c1 = (math::abs(F::load(c[0])) + math::abs(F::load(c[1]))) + math::abs(F::load(c[2]));
    const F t1 = (math::abs(F::load(wt[0])) + math::abs(F::load(wt[1]))) + math::abs(F::load(wt[2]));
    const F tol = F(1.0f) * ((F(1.0e-4f) * rmax + (F(4.0e-6f) * ratio) * (F(2.0f) * (c1 + F(memo.maxD1)) + F(3.0f) * F(memo.maxModelAbs))) +
                             F(1.0e-6f) * (t1 + F(memo.maxT1)));
    tol.store(tolv);
    (rmax + tol).store(reach);
    (rmin - tol).store(lazy);
    for (int k = 0; k < 8; ++k) {
        cq::ShapeFilter &f = sf[k];
        for (int a = 0; a < 3; ++a) { f.lo[a] = q[k].lo[a]; f.hi[a] = q[k].hi[a]; f.c[a] = c[a][k]; }
        for (int i = 0; i < 6; ++i) f.q[i] = qq[i][k];
        f.reach = reach[k];
        f.low = -tolv[k];
        f.tol = tolv[k];
        f.lazy = lazy[k];
    }
}

void ExactCollision::DetectPose(const float pos[3], const float quat[4], const float absorb[4], step::SubstepContacts &out,
                                step::CollisionCache *cache) const {
    ExactCache *c = cache != nullptr && cache->kind == &ExactCache::kKind ? static_cast<ExactCache *>(cache) : nullptr;
    if (c != nullptr && c->flushEveryCall) c->Flush();
    std::uint32_t key[11];
    if (c != nullptr) {   // the same pose bits as the last call: the same answer
        std::memcpy(key, pos, 12);
        std::memcpy(key + 3, quat, 16);
        std::memcpy(key + 7, absorb, 16);
        ExactCache::Last &l = c->last;
        if (l.owner == id_ && std::memcmp(key, l.key, sizeof key) == 0) {
            out.n = l.out.n;
            out.overflow = l.out.overflow;
            std::memcpy(out.c, l.out.c, sizeof(out.c[0]) * static_cast<std::size_t>(l.out.n));
            ++c->lastHits;
            return;
        }
    }
    if (opt_.laneSetup) {
        // shape poses, boxes and shape filters with the 8 shapes in lanes (each lane the scalar operations)
        thread_local ExactCache threadCache;
        ExactCache &cc = c != nullptr ? *c : threadCache;
        ShapePose shapes[8];
        Box3 q[8], u;
        alignas(32) float wr[9][8], wt[3][8];
        LaneShapes(pos, quat, absorb, shapes, q, &u, wr, wt);
        const cq::Memo &memo = UpdateMemo(cc, u);
        cq::ShapeFilter sf[8];
        LaneFilters(memo, wr, wt, q, sf);
        if (opt_.checkSetup) {   // test mode: the scalar forms give the same bits
            ShapePose s2[8];
            Box3 q2[8], u2;
            cq::ShapeFilter sf2[8];
            ShapePoses(car_, pos, quat, absorb, s2);
            ShapeBoxes(s2, q2, &u2);
            ShapeFilters(memo, s2, q2, sf2);
            if (std::memcmp(shapes, s2, sizeof s2) != 0 || std::memcmp(q, q2, sizeof q2) != 0 || std::memcmp(&u, &u2, sizeof u2) != 0 ||
                std::memcmp(sf, sf2, sizeof sf2) != 0)
                ++setupMismatches_;
        }
        DetectFiltered(shapes, u, sf, cc, out);
    } else {
        ShapePose shapes[8];
        ShapePoses(car_, pos, quat, absorb, shapes);
        DetectShapes(shapes, out, nullptr, c);
    }
    if (c != nullptr) {
        ExactCache::Last &l = c->last;
        l.owner = id_;
        std::memcpy(l.key, key, sizeof key);
        l.out.n = out.n;
        l.out.overflow = out.overflow;
        std::memcpy(l.out.c, out.c, sizeof(out.c[0]) * static_cast<std::size_t>(out.n));
    }
}

void ExactCollision::DetectShapesRef(const ShapePose shapes[8], step::SubstepContacts &out, CollideStats *stats) const {
    thread_local Scratch sc;
    const world::WorldPackView &v = *view_;
    out.n = 0;
    out.overflow = false;
    if (stats) ++stats->calls;
    for (int k = 0; k < 8; ++k) {
        const ShapePose &sp = shapes[k];
        const I ell = IOf(sp.world);
        const V radii = {sp.radii[0], sp.radii[1], sp.radii[2]};
        // broad phase: world AABB of the ellipsoid, inflated for rounding
        Box3 q;
        const float ct[3] = {ell.t.x, ell.t.y, ell.t.z};
        for (int a = 0; a < 3; ++a) {
            const float *col0 = &sp.world.r[0], *col1 = &sp.world.r[3], *col2 = &sp.world.r[6];
            const float h = std::fabs(col0[a]) * radii.x + std::fabs(col1[a]) * radii.y + std::fabs(col2[a]) * radii.z;
            const float eps = 1e-3f + 1e-6f * std::fabs(ct[a]);
            q.lo[a] = ct[a] - h - eps; q.hi[a] = ct[a] + h + eps;
        }
        sc.order.clear();
        bvh_.Query(q, [&](std::uint32_t t) { sc.order.push_back({v.triInfo[t].fvOrder, t}); });
        if (stats) {
            stats->candidates += sc.order.size();
            std::size_t mp = 0;
            for (const auto &ot : sc.order) mp += (v.instances[v.triInfo[ot.second].instance].flags & world::kInstMovingPair) != 0;
            stats->movingPairCandidates += mp;
            stats->queriesTouchingMovingPair += mp > 0;
        }
        if (sc.order.empty()) continue;
        std::sort(sc.order.begin(), sc.order.end());
        // narrow phase in FV detection order, surface transforms per instance (mesh_collision_queries.cpp
        // GmCollision_Ellipsoid_Mesh)
        sc.raw.clear();
        const V invRadii = {1.0f / radii.x, 1.0f / radii.y, 1.0f / radii.z};
        std::uint32_t curInst = 0xFFFFFFFFu;
        I meshToUnit{}, c2w{}, n2w{};
        for (const auto &ot : sc.order) {
            const std::uint32_t t = ot.second;
            const world::TriInfo &ti = v.triInfo[t];
            const world::Instance &in = v.instances[ti.instance];
            if (ti.instance != curInst) {
                curInst = ti.instance;
                I meshIso;
                meshIso.r = MOf(in.iso);
                meshIso.t = V{in.iso[9], in.iso[10], in.iso[11]};
                const I ellipsoidToMesh = MultInverse(ell, meshIso);
                const I meshToEllipsoid = Inverse(ellipsoidToMesh);
                meshToUnit = meshToEllipsoid;
                ScaleRows(meshToUnit, invRadii);
                c2w = SetMult(MultInverse(Diagonal(radii), meshToEllipsoid), meshIso);
                n2w = SetMult(MultInverse(Diagonal(invRadii), meshToEllipsoid), meshIso);
            }
            const world::Model &m = v.models[in.model];
            const world::ModelTri &mt = v.modelTris[m.firstTri + ti.modelTri];
            V vu[3];
            for (int c = 0; c < 3; ++c) {
                const world::Vec3f &p = v.modelVerts[m.firstVert + mt.v[c]];
                vu[c] = TPoint(meshToUnit, V{p.x, p.y, p.z});
            }
            const V e01 = Sub(vu[1], vu[0]), e02 = Sub(vu[2], vu[0]);
            const float nX = e02.z * e01.y - e02.y * e01.z;
            const float nY = e01.z * e02.x - e02.z * e01.x;
            const float nZ = e01.x * e02.y - e02.x * e01.y;
            const float lenSq = (nY * nY + nX * nX) + nZ * nZ;
            if (stats) ++stats->triangleTests;
            if (!(lenSq > kDirEpsSq)) continue;
            const float len = Sqrt(lenSq);
            const float invLen = 1.0f / len;
            const V n = {nX * invLen, nY * invLen, invLen * nZ};
            const std::size_t first = sc.raw.size();
            TriangleCollide tc{n, &sc.raw, t};
            if (!tc.Collide(vu)) continue;
            // TransformEllipsoidMeshCollisionsToWorldForGmSurf
            for (std::size_t r = first; r < sc.raw.size(); ++r) {
                Raw &c = sc.raw[r];
                c.P = TPoint(c2w, c.P);
                c.N = NormalizeForCollision(TDir(n2w.r, c.N), kDirEpsSq);
                c.Sep = TDir(c2w.r, c.Sep);
            }
        }
        if (stats) stats->rawContacts += sc.raw.size();
        // per-shape merge (collision_buffer.cpp MergeAndAddToCollisions)
        sc.merged.clear();
        for (std::size_t r = 0; r < sc.raw.size(); ++r) if (sc.raw[r].primary) sc.merged.push_back(static_cast<int>(r));
        const std::size_t nPrim = sc.merged.size();
        for (std::size_t r = 0; r < sc.raw.size(); ++r) {
            const Raw &c = sc.raw[r];
            if (c.primary) continue;
            std::size_t j = 0;
            for (; j < nPrim; ++j) {
                const Raw &p = sc.raw[static_cast<std::size_t>(sc.merged[j])];
                if (NearlyEqual(c.K, p.K) || kNormalAlignment < Dot(c.N, p.N)) break;
            }
            if (j == nPrim) sc.merged.push_back(static_cast<int>(r));
        }
        for (int idx : sc.merged) {
            if (out.n >= model::kRespMaxContacts) { out.overflow = true; if (stats) ++stats->overflow; break; }
            const Raw &c = sc.raw[static_cast<std::size_t>(idx)];
            const Tri &tr = v.tris[c.tri];
            const world::TriInfo &ti = v.triInfo[c.tri];
            const world::Actor &a = v.ActorOf(tr);
            model::RespContact &o = out.c[out.n++];
            o.P[0] = c.P.x; o.P[1] = c.P.y; o.P[2] = c.P.z;
            o.N[0] = c.N.x; o.N[1] = c.N.y; o.N[2] = c.N.z;
            o.Sep[0] = c.Sep.x; o.Sep[1] = c.Sep.y; o.Sep[2] = c.Sep.z;
            o.peerZ[0] = a.peerZ[0]; o.peerZ[1] = a.peerZ[1]; o.peerZ[2] = a.peerZ[2];
            const bool trig = ti.trigger != world::kNoTrigger;
            o.peerId = trig ? static_cast<std::uint32_t>(ti.trigger) : a.peerId;
            o.shape = static_cast<std::uint8_t>(k);
            o.material = static_cast<std::uint8_t>(tr.meta & kMetaMaterialMask);
            o.flags = static_cast<std::uint8_t>((c.primary ? unsigned(model::kRespPrimary) : 0u) | (trig ? unsigned(model::kRespTrigger) : 0u));
            o.pad = 0u;
        }
        if (stats) stats->merged += sc.merged.size();
    }
}

std::size_t MergeShapeTo(const world::WorldPackView &v, const std::vector<exact::Raw> &raw, int k, std::vector<int> *scratch,
                         step::SubstepContacts &out) {
    return MergeShapeTo(v, raw.data(), raw.size(), k, scratch, out);
}

std::size_t MergeShapeTo(const world::WorldPackView &v, const exact::Raw *raw, std::size_t n, int k, std::vector<int> *scratch,
                         step::SubstepContacts &out) {
    // per-shape merge (collision_buffer.cpp MergeAndAddToCollisions): primaries in order, then the
    // non-primaries no primary absorbs
    std::vector<int> &merged = *scratch;
    merged.clear();
    for (std::size_t r = 0; r < n; ++r) if (raw[r].primary) merged.push_back(static_cast<int>(r));
    const std::size_t nPrim = merged.size();
    for (std::size_t r = 0; r < n; ++r) {
        const exact::Raw &c = raw[r];
        if (c.primary) continue;
        std::size_t j = 0;
        for (; j < nPrim; ++j) {
            const exact::Raw &p = raw[static_cast<std::size_t>(merged[j])];
            if (NearlyEqual(V{c.K.x, c.K.y, c.K.z}, V{p.K.x, p.K.y, p.K.z}) || kNormalAlignment < Dot(V{c.N.x, c.N.y, c.N.z}, V{p.N.x, p.N.y, p.N.z})) break;
        }
        if (j == nPrim) merged.push_back(static_cast<int>(r));
    }
    for (int idx : merged) {
        if (out.n >= model::kRespMaxContacts) { out.overflow = true; break; }
        const exact::Raw &c = raw[static_cast<std::size_t>(idx)];
        const Tri &tr = v.tris[c.tri];
        const world::TriInfo &ti = v.triInfo[c.tri];
        const world::Actor &a = v.ActorOf(tr);
        model::RespContact &o = out.c[out.n++];
        o.P[0] = c.P.x; o.P[1] = c.P.y; o.P[2] = c.P.z;
        o.N[0] = c.N.x; o.N[1] = c.N.y; o.N[2] = c.N.z;
        o.Sep[0] = c.Sep.x; o.Sep[1] = c.Sep.y; o.Sep[2] = c.Sep.z;
        o.peerZ[0] = a.peerZ[0]; o.peerZ[1] = a.peerZ[1]; o.peerZ[2] = a.peerZ[2];
        const bool trig = ti.trigger != world::kNoTrigger;
        o.peerId = trig ? static_cast<std::uint32_t>(ti.trigger) : a.peerId;
        o.shape = static_cast<std::uint8_t>(k);
        o.material = static_cast<std::uint8_t>(tr.meta & kMetaMaterialMask);
        o.flags = static_cast<std::uint8_t>((c.primary ? unsigned(model::kRespPrimary) : 0u) | (trig ? unsigned(model::kRespTrigger) : 0u));
        o.pad = 0u;
    }
    return merged.size();
}

// MergeShapeTo with the output fields from the memo (Raw::tri = the memo entry).
std::size_t MergeShapeTo(const cq::Memo &memo, const exact::Raw *raw, std::size_t n, int k, std::vector<int> *scratch, step::SubstepContacts &out) {
    std::vector<int> &merged = *scratch;
    merged.clear();
    for (std::size_t r = 0; r < n; ++r) if (raw[r].primary) merged.push_back(static_cast<int>(r));
    const std::size_t nPrim = merged.size();
    for (std::size_t r = 0; r < n; ++r) {
        const exact::Raw &c = raw[r];
        if (c.primary) continue;
        std::size_t j = 0;
        for (; j < nPrim; ++j) {
            const exact::Raw &p = raw[static_cast<std::size_t>(merged[j])];
            if (NearlyEqual(V{c.K.x, c.K.y, c.K.z}, V{p.K.x, p.K.y, p.K.z}) || kNormalAlignment < Dot(V{c.N.x, c.N.y, c.N.z}, V{p.N.x, p.N.y, p.N.z})) break;
        }
        if (j == nPrim) merged.push_back(static_cast<int>(r));
    }
    for (int idx : merged) {
        if (out.n >= model::kRespMaxContacts) { out.overflow = true; break; }
        const exact::Raw &c = raw[static_cast<std::size_t>(idx)];
        const cq::OutFields &f = memo.outf[c.tri];
        model::RespContact &o = out.c[out.n++];
        o.P[0] = c.P.x; o.P[1] = c.P.y; o.P[2] = c.P.z;
        o.N[0] = c.N.x; o.N[1] = c.N.y; o.N[2] = c.N.z;
        o.Sep[0] = c.Sep.x; o.Sep[1] = c.Sep.y; o.Sep[2] = c.Sep.z;
        o.peerZ[0] = f.peerZ[0]; o.peerZ[1] = f.peerZ[1]; o.peerZ[2] = f.peerZ[2];
        o.peerId = f.peerId;
        o.shape = static_cast<std::uint8_t>(k);
        o.material = f.material;
        o.flags = static_cast<std::uint8_t>((c.primary ? unsigned(model::kRespPrimary) : 0u) | (f.trigger ? unsigned(model::kRespTrigger) : 0u));
        o.pad = 0u;
    }
    return merged.size();
}

namespace {
namespace lanes = tw::exact::lanes;   // FV-form geometry over SSE lanes (tw_exact_lanes.h)
using lanes::B4;
using lanes::F4;
using FrameF = lanes::FrameFv<float>;      // meshToUnit, c2w, n2w: 36 floats
static_assert(sizeof(FrameF) == 36 * sizeof(float), "FrameFv<float>: 36 floats");

inline __attribute__((always_inline)) F4 G4(const float *a, const float *b, const float *c, const float *d, int f) {
    return F4{_mm_setr_ps(a[f], b[f], c[f], d[f])};
}
// 4 consecutive fields [f, f+4) of 4 lane records -> 4 field vectors (transpose in registers)
inline __attribute__((always_inline)) void G4x4(const float *a, const float *b, const float *c, const float *d, int f, F4 out[4]) {
    __m128 r0 = _mm_loadu_ps(a + f), r1 = _mm_loadu_ps(b + f), r2 = _mm_loadu_ps(c + f), r3 = _mm_loadu_ps(d + f);
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    out[0].v = r0; out[1].v = r1; out[2].v = r2; out[3].v = r3;
}
// 12 floats (an iso: rotation columns, translation) of 4 lane records at float offset f
inline __attribute__((always_inline)) lanes::I3<F4> GatherIso(const float *a, const float *b, const float *c, const float *d, int f) {
    F4 x[12];
    G4x4(a, b, c, d, f, x);
    G4x4(a, b, c, d, f + 4, x + 4);
    G4x4(a, b, c, d, f + 8, x + 8);
    lanes::I3<F4> r;
    r.r.c[0] = {x[0], x[1], x[2]};
    r.r.c[1] = {x[3], x[4], x[5]};
    r.r.c[2] = {x[6], x[7], x[8]};
    r.t = {x[9], x[10], x[11]};
    return r;
}
inline __attribute__((always_inline)) void ScatterIso(const lanes::I3<F4> &v, float *dst[4], int f) {
    const F4 x[12] = {v.r.c[0].x, v.r.c[0].y, v.r.c[0].z, v.r.c[1].x, v.r.c[1].y, v.r.c[1].z, v.r.c[2].x, v.r.c[2].y, v.r.c[2].z, v.t.x, v.t.y, v.t.z};
    for (int g = 0; g < 3; ++g) {
        __m128 r0 = x[4 * g].v, r1 = x[4 * g + 1].v, r2 = x[4 * g + 2].v, r3 = x[4 * g + 3].v;
        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
        _mm_storeu_ps(dst[0] + f + 4 * g, r0);
        _mm_storeu_ps(dst[1] + f + 4 * g, r1);
        _mm_storeu_ps(dst[2] + f + 4 * g, r2);
        _mm_storeu_ps(dst[3] + f + 4 * g, r3);
    }
}
inline exact::I ToExact(const lanes::I3<float> &a) {
    exact::I r;
    std::memcpy(&r, &a, sizeof r);
    return r;
}
// geometry_helpers.cpp NearlyEqual(value, reference), on exact::V
inline bool NearlyEqualV(const exact::V &value, const exact::V &ref) { return NearlyEqual(V{value.x, value.y, value.z}, V{ref.x, ref.y, ref.z}); }

enum : std::uint8_t { kRej = 0, kFaceC = 1, kFvViol = 2, kFvAny = 3, kPatchM = 4 };   // kPatchM: a merged patch's member
struct SurvOut {        // pass A's result for one survivor
    float x[21];        // unit-space vertices (9), FV's n (3), FACE payload P N Sep (9)
    std::uint8_t cls;
};
struct Job { std::uint32_t shape, instance, local; };   // one (shape, instance) FV frame; local: the memo's instance copy

// A merged patch as one element: FvTriangle::Collide's algorithm on the patch's convex loop in unit space, with the
// plane of its largest member (rep): FV's FACE when the projected centre is inside the loop, else FV's
// first-violated-edge feature on the loop's edges. Internal edges make no contacts. Raw::tri = tag.
__attribute__((noinline)) void PatchCollide(const FrameF &ff, const patch::Set &ps, std::uint32_t pid, const exact::TriVerts &rep, const float *meshIso,
                                            std::uint32_t tag, std::vector<exact::Raw> *out) {
    exact::Frame f;
    f.meshToUnit = ToExact(ff.meshToUnit);
    f.c2w = ToExact(ff.c2w);
    f.n2w = ToExact(ff.n2w);
    exact::V vr[3];
    exact::UnitVerts(f, rep, vr);
    float lenSq;
    const exact::V nr = exact::UnitNormal(vr, &lenSq);
    if (!(lenSq > exact::kDirEpsSq)) return;
    const float len = exact::Sqrt(lenSq);
    const float invLen = 1.0f / len;
    const exact::V n = {nr.x * invLen, nr.y * invLen, invLen * nr.z};
    const patch::Patch &pt = ps.patches[pid];
    {   // FACE fast path: FV's FACE payload on the rep's plane when its point lies inside the loop with a margin
        // (mesh-local edge planes, no unit-space loop); else the loop test below
        const exact::V zero0 = {0.0f, 0.0f, 0.0f};
        const float pd0 = exact::Dot(exact::Sub(zero0, vr[0]), n);
        if (exact::FvTriangle::kR < pd0 || pd0 < 0.0f) return;
        const exact::V projected0 = exact::Add(zero0, exact::Scale(n, -pd0));
        const exact::V Pw = exact::TPoint(f.c2w, projected0);
        const float *iso = meshIso;   // world -> mesh-local: R^T (x - t)
        const float dx = Pw.x - iso[9], dy = Pw.y - iso[10], dz = Pw.z - iso[11];
        const float pm[3] = {(iso[0] * dx + iso[1] * dy) + iso[2] * dz, (iso[3] * dx + iso[4] * dy) + iso[5] * dz, (iso[6] * dx + iso[7] * dy) + iso[8] * dz};
        bool inside = true;
        for (std::uint32_t e = 0; e < pt.loopCount && inside; ++e) {
            const float *u = &ps.edge[4 * (pt.firstLoop + e)];
            inside = ((u[0] * pm[0] + u[1] * pm[1]) + u[2] * pm[2]) - u[3] < -1e-3f;
        }
        if (inside) {
            exact::Raw r{Pw, exact::NormalizeForCollision(exact::TDir(f.n2w.r, n), exact::kDirEpsSq), exact::TDir(f.c2w.r, exact::Scale(n, pd0 - exact::FvTriangle::kR)), n,
                         true, tag};
            out->push_back(r);
            return;
        }
    }
    constexpr int kMaxLoop = 64;
    exact::V lu[kMaxLoop];
    const std::uint32_t k = std::min<std::uint32_t>(pt.loopCount, kMaxLoop);
    for (std::uint32_t i = 0; i < k; ++i) {
        const float *p = &ps.loop[3 * (pt.firstLoop + i)];
        lu[i] = exact::TPoint(f.meshToUnit, exact::V{p[0], p[1], p[2]});
    }
    const std::size_t first = out->size();
    exact::FvTriangle tc{n, out, tag};   // its Emit* helpers and constants
    const exact::V zero = {0.0f, 0.0f, 0.0f};
    const float pd = exact::Dot(exact::Sub(zero, vr[0]), n);
    if (exact::FvTriangle::kR < pd || pd < 0.0f) return;
    const float edgeReach = exact::Sqrt(exact::FvTriangle::kR * exact::FvTriangle::kR - pd * pd);
    const exact::V projected = exact::Add(zero, exact::Scale(n, -pd));
    int made = -1;
    for (std::uint32_t e = 0; e < k && made < 0; ++e) {
        const exact::V start = lu[e], end = lu[e + 1 == k ? 0 : e + 1];
        const exact::V dir = exact::NormalizeForCollision(exact::Sub(end, start), exact::kDirEpsSq);
        const exact::V edgeNormal = exact::Cross(dir, n);
        const float edgeDistance = exact::Dot(exact::Sub(projected, start), edgeNormal);
        if (edgeReach < edgeDistance) { made = 0; break; }
        if (edgeDistance > 0.0f) {
            const float fromStart = exact::Dot(exact::Sub(projected, start), dir);
            if (fromStart < 0.0f) { made = tc.EmitFeature(start, exact::kDirEpsSq, true); break; }
            const float fromEnd = exact::Dot(exact::Sub(projected, end), dir);
            if (!(0.0f < fromEnd)) { made = tc.EmitFeature(exact::Add(projected, exact::Scale(edgeNormal, -edgeDistance)), exact::kCollisionDistance, false); break; }
            made = tc.EmitEndpointB(end, exact::kDirEpsSq);
            break;
        }
    }
    if (made < 0 && pd > 0.0f) out->push_back(exact::Raw{projected, n, exact::Scale(n, pd - exact::FvTriangle::kR), n, true, tag});
    for (std::size_t r = first; r < out->size(); ++r) {   // TransformEllipsoidMeshCollisionsToWorldForGmSurf
        exact::Raw &c = (*out)[r];
        c.P = exact::TPoint(f.c2w, c.P);
        c.N = exact::NormalizeForCollision(exact::TDir(f.n2w.r, c.N), exact::kDirEpsSq);
        c.Sep = exact::TDir(f.c2w.r, c.Sep);
    }
}

// FV's form for one pair (outlined: the call's rare path), from pass A's unit-space vertices.
__attribute__((noinline)) void FvFormPair(const FrameF &ff, const float *x, std::uint32_t t, std::vector<exact::Raw> *raw) {
    const exact::V vul[3] = {{x[0], x[1], x[2]}, {x[3], x[4], x[5]}, {x[6], x[7], x[8]}};
    exact::Frame frame;
    frame.c2w = ToExact(ff.c2w);
    frame.n2w = ToExact(ff.n2w);
    exact::FvCollide(frame, vul, t, raw);
}
// Test mode (CollideOptions::checkFrames): the frames against exact::MakeFrameFv, bit for bit; the mismatch count.
__attribute__((noinline)) std::uint64_t CheckFrames(const world::WorldPackView &v, const ShapePose *shapes, const Job *jobs, const FrameF *frames, std::size_t nj) {
    std::uint64_t bad = 0;
    for (std::size_t j = 0; j < nj; ++j) {
        const ShapePose &sp = shapes[jobs[j].shape];
        exact::Frame ref;
        exact::MakeFrameFv(sp.world.r, sp.world.t, sp.radii, v.instances[jobs[j].instance].iso, &ref);
        const exact::I m2u = ToExact(frames[j].meshToUnit), c2w = ToExact(frames[j].c2w), n2w = ToExact(frames[j].n2w);
        if (std::memcmp(&ref.meshToUnit, &m2u, sizeof m2u) != 0 || std::memcmp(&ref.c2w, &c2w, sizeof c2w) != 0 ||
            std::memcmp(&ref.n2w, &n2w, sizeof n2w) != 0)
            ++bad;
    }
    return bad;
}

struct FastScratch {
    std::vector<exact::Raw> raw;          // the call's raw contacts, shape-major, FV order within a shape
    std::vector<int> merged;
    std::vector<Job> jobs;                // the call's FV frames, one per (shape, instance) with survivors
    std::vector<FrameF> frames;
    std::vector<std::uint32_t> pe, pf;    // the call's survivors, shape-major: memo entry, frame
    std::vector<SurvOut> so;
    std::vector<std::uint32_t> pp, aIdx;  // merged patches: each survivor's patch; the survivors pass A sees
};
}  // namespace

void ShapeBoxes(const ShapePose shapes[8], Box3 q[8], Box3 *u) {
    // the shapes' world AABBs (closed, inflated for rounding: the BVH leaf test) and their union
    *u = EmptyBox();
    for (int k = 0; k < 8; ++k) {
        const ShapePose &sp = shapes[k];
        for (int a = 0; a < 3; ++a) {
            const float c = sp.world.t[a];
            const float h = std::fabs(sp.world.r[a]) * sp.radii[0] + std::fabs(sp.world.r[3 + a]) * sp.radii[1] + std::fabs(sp.world.r[6 + a]) * sp.radii[2];
            const float eps = 1e-3f + 1e-6f * std::fabs(c);
            q[k].lo[a] = c - h - eps; q[k].hi[a] = c + h + eps;
        }
        *u = Union(*u, q[k]);
    }
}

void ShapeFilters(const cq::Memo &memo, const ShapePose shapes[8], const Box3 q[8], cq::ShapeFilter sf[8]) {
    for (int k = 0; k < 8; ++k)
        cq::MakeShapeFilter(memo, shapes[k].world.r, shapes[k].world.t, cq::MakeShapeConst(shapes[k].radii), q[k], &sf[k]);
}

const cq::Memo &ExactCollision::UpdateMemo(ExactCache &cc, const Box3 &u, CollideStats *stats) const {
    ExactCache::Memo &memo = cc.memo;
    float center[3];
    for (int a = 0; a < 3; ++a) center[a] = 0.5f * (u.lo[a] + u.hi[a]);
    if (memo.owner != id_ || !Contains(memo.q.box, u)) {
        ++cc.refills;
        const bool same = memo.owner == id_ && memo.haveLast;
        memo.owner = id_;
        Box3 big = u;
        if (same)   // extend along the recent motion: the next calls stay inside (only affects reuse, never the output)
            for (int a = 0; a < 3; ++a) {
                const float d = math::max(-4.0f, math::min(4.0f, (center[a] - memo.lastCenter[a]) * opt_.leadPoses));
                if (d > 0.0f) big.hi[a] += d; else big.lo[a] += d;
            }
        const Box3 box = Inflate(big, opt_.unionMargin);
        float origin[3];
        for (int a = 0; a < 3; ++a) origin[a] = 0.5f * (box.lo[a] + box.hi[a]);
        cq::BuildMemo(*view_, bvh_, fast_.data(), modelAbs_.data(), box, origin, &memo.q, patches_.patchOf.empty() ? nullptr : patches_.patchOf.data());
    } else {
        ++cc.reuses;
    }
    for (int a = 0; a < 3; ++a) memo.lastCenter[a] = center[a];
    memo.haveLast = true;
    if (stats) { stats->memoEntries += memo.q.n; stats->memoRefills = cc.refills; stats->memoReuses = cc.reuses; }
    return memo.q;
}

void ExactCollision::DetectShapes(const ShapePose shapes[8], step::SubstepContacts &out, CollideStats *stats, ExactCache *cache) const {
    thread_local ExactCache threadCache;   // callers without a per-candidate cache
    ExactCache &cc = cache != nullptr ? *cache : threadCache;
    std::uint64_t t0 = stats ? __rdtsc() : 0;
    auto lap = [&](std::uint64_t &acc) { if (stats) { const std::uint64_t t1 = __rdtsc(); acc += t1 - t0; t0 = t1; } };
    if (stats) ++stats->calls;
    Box3 q[8], u;
    ShapeBoxes(shapes, q, &u);
    if (stats) lap(stats->tBoxes);
    if (opt_.profileStop == 5) { out.n = 0; out.overflow = false; return; }
    const cq::Memo &memo = UpdateMemo(cc, u, stats);
    if (stats) lap(stats->tMemo);
    if (opt_.profileStop == 4) { out.n = 0; out.overflow = false; return; }
    cq::ShapeFilter sf[8];
    ShapeFilters(memo, shapes, q, sf);
    if (opt_.profileStop == 3) { out.n = 0; out.overflow = false; return; }
    DetectFiltered(shapes, u, sf, cc, out, stats);
}

bool ExactCollision::ClosedWheelFace(const ShapePose &sp, const cq::ShapeFilter &sf, const cq::Memo &m, const cq::Lists &lists, int k,
                                     model::RespContact *outc) const {
    const std::uint32_t *list = lists.idx[k].data();
    const std::uint32_t n = lists.n[k];
    // one coplanar group: every triangle whose AABB meets the wheel's box (a superset of the survivors that does not
    // depend on the memo, so the choice of path never depends on the cache) is in the same patch on the same
    // instance, and not a trigger
    (void)sf;
    const std::uint32_t e0 = list[0];
    const std::uint32_t pid = m.patch[e0];
    if (pid == patch::kNoPatch || m.outf[e0].trigger || !lists.boxUniform[k] || lists.boxKey[k] != m.pkey[e0]) return false;
    const patch::Patch &pt = patches_.patches[pid];
    // in the instance's mesh-local frame (a function of the pose and the instance only, never of the cache): the
    // ellipsoid x = c + M u with c = R_m^T (t - t_m), M = R_m^T axes diag(r)
    const float *iso = m.inst[m.ref[e0].local].iso;
    const float sgn = (view_->instances[m.ref[e0].instance].flags & world::kInstMirrored) ? -1.0f : 1.0f;   // FV's front side
    const float d[3] = {sp.world.t[0] - iso[9], sp.world.t[1] - iso[10], sp.world.t[2] - iso[11]};
    float c[3], M[3][3];
    for (int a = 0; a < 3; ++a) {
        c[a] = (iso[3 * a] * d[0] + iso[3 * a + 1] * d[1]) + iso[3 * a + 2] * d[2];
        for (int jj = 0; jj < 3; ++jj) {
            const float *ax = &sp.world.r[3 * jj];
            M[a][jj] = ((iso[3 * a] * ax[0] + iso[3 * a + 1] * ax[1]) + iso[3 * a + 2] * ax[2]) * sp.radii[jj];
        }
    }
    const float rmax = math::max(sp.radii[0], math::max(sp.radii[1], sp.radii[2]));
    for (std::uint32_t j = 0; j < n; ++j) {   // the member that holds the FACE point
        const exact::TriVerts &tv = m.verts[list[j]];
        const exact::V &v0 = tv.v[0], &v1 = tv.v[1], &v2 = tv.v[2];
        const float e1[3] = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z}, e2[3] = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
        float nm[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        const float nl = math::sqrt((nm[0] * nm[0] + nm[1] * nm[1]) + nm[2] * nm[2]);
        if (!(nl > 1e-12f)) return false;
        for (float &x : nm) x = x / nl;
        const float nn[3] = {sgn * nm[0], sgn * nm[1], sgn * nm[2]};
        const float sd = (nn[0] * (c[0] - v0.x) + nn[1] * (c[1] - v0.y)) + nn[2] * (c[2] - v0.z);
        float Mn[3];
        for (int jj = 0; jj < 3; ++jj) Mn[jj] = (M[0][jj] * nn[0] + M[1][jj] * nn[1]) + M[2][jj] * nn[2];
        const float H2 = (Mn[0] * Mn[0] + Mn[1] * Mn[1]) + Mn[2] * Mn[2];
        if (!(sd > 1e-5f && sd * sd <= H2 * 0.9997f)) return false;   // reached, not tangent (a noisy d takes FV's path)
        float MM[3];
        for (int a = 0; a < 3; ++a) MM[a] = (M[a][0] * Mn[0] + M[a][1] * Mn[1]) + M[a][2] * Mn[2];
        const float kk = sd / H2;
        const float P[3] = {c[0] - kk * MM[0], c[1] - kk * MM[1], c[2] - kk * MM[2]};
        float smax = -1e30f;
        const exact::V *vv[3] = {&v0, &v1, &v2};
        for (int ed = 0; ed < 3; ++ed) {   // in-plane outward edge normals of the mesh-local triangle
            const exact::V &a0 = *vv[ed], &a1 = *vv[ed == 2 ? 0 : ed + 1];
            const float ev[3] = {a1.x - a0.x, a1.y - a0.y, a1.z - a0.z};
            float u[3] = {ev[1] * nm[2] - ev[2] * nm[1], ev[2] * nm[0] - ev[0] * nm[2], ev[0] * nm[1] - ev[1] * nm[0]};
            const float ul = math::sqrt((u[0] * u[0] + u[1] * u[1]) + u[2] * u[2]);
            if (!(ul > 1e-12f)) return false;
            smax = math::max(smax, ((u[0] * (P[0] - a0.x) + u[1] * (P[1] - a0.y)) + u[2] * (P[2] - a0.z)) / ul);
        }
        if (!(smax < -2e-5f)) continue;   // not this member
        // the contact disk (the ellipsoid in the slab of half thickness tauP around the plane) clears the member's
        // edges: no other member of the patch, all within tauP of the plane, can touch (C's cc_face.h conditions)
        const float H = math::sqrt(H2);
        const float tauP = 2.0f * pt.tau + pt.alpha * rmax + 2e-7f * (32.0f + rmax) + 1e-5f;
        const float dlo = math::max(0.0f, (sd - tauP) / H);
        const float g = -smax - (tauP * rmax / H + 2e-4f);
        if (!(g >= 0.0f && g * g >= rmax * rmax * (1.0f - dlo * dlo) * 1.0002f)) return false;
        const float ks = (sd - H) / H2;
        const float S[3] = {ks * MM[0], ks * MM[1], ks * MM[2]};
        const cq::OutFields &f = m.outf[list[j]];
        model::RespContact &o = *outc;
        for (int a = 0; a < 3; ++a) {   // back to world: R_m x (+ t_m)
            o.P[a] = ((iso[a] * P[0] + iso[3 + a] * P[1]) + iso[6 + a] * P[2]) + iso[9 + a];
            o.N[a] = (iso[a] * nn[0] + iso[3 + a] * nn[1]) + iso[6 + a] * nn[2];
            o.Sep[a] = (iso[a] * S[0] + iso[3 + a] * S[1]) + iso[6 + a] * S[2];
            o.peerZ[a] = f.peerZ[a];
        }
        o.peerId = f.peerId;
        o.shape = static_cast<std::uint8_t>(k);
        o.material = f.material;
        o.flags = static_cast<std::uint8_t>(model::kRespPrimary);
        o.pad = 0u;
        return true;
    }
    return false;
}

void ExactCollision::DetectFiltered(const ShapePose shapes[8], const Box3 &u, const cq::ShapeFilter sf[8], ExactCache &cc, step::SubstepContacts &out,
                                    CollideStats *stats) const {
    ExactCache::Memo &memo = cc.memo;
    thread_local FastScratch sc;
    const world::WorldPackView &v = *view_;
    out.n = 0;
    out.overflow = false;
    std::uint64_t t0 = stats ? __rdtsc() : 0;
    auto lap = [&](std::uint64_t &acc) { if (stats) { const std::uint64_t t1 = __rdtsc(); acc += t1 - t0; t0 = t1; } };
    const bool compact = opt_.compactMemo;   // the narrow phase reads the memo's copies, not the world's tables
    // per shape: the memo's candidates in its box that pass the support prefilter, in FV order (tw_car_query.h)
    cq::Filter(memo.q, sf, &memo.lists, true, &u);
    if (stats) lap(stats->tFilter);
    if (opt_.profileStop == 1) return;
    if (opt_.narrow != nullptr) {   // another narrow phase over the same pre-narrow output (e.g. C's noise-level one)
        const PreNarrow pre{shapes, sf, &memo.q, &memo.lists, view_};
        opt_.narrow(pre, out, stats);
        return;
    }
    // R7 (CollideOptions::closedWheelFace): a wheel whose survivors are one coplanar patch on one instance, with
    // the FACE point well inside one member and the contact disk clear of its edges, gets that FACE in closed form
    // (no FV frame, no unit-space test); its survivors skip the rest
    bool closed[8] = {false, false, false, false, false, false, false, false};
    model::RespContact closedC[8];
    std::uint32_t nsk[8];
    for (int k = 0; k < 8; ++k) {
        nsk[k] = memo.lists.n[k];
        if (!opt_.closedWheelFace || car_.shape[k].wheel < 0 || nsk[k] == 0u) continue;
        if (stats) ++stats->wheelCalls;
        if (ClosedWheelFace(shapes[k], sf[k], memo.q, memo.lists, k, &closedC[k])) {
            closed[k] = true;
            nsk[k] = 0;
            if (stats) ++stats->closedWheelFaces;
        }
    }
    // the call's survivors (shape-major, FV order) and FV's frame per (shape, instance)
    std::uint32_t np = 0;
    for (int k = 0; k < 8; ++k) np += nsk[k];
    if (sc.pe.size() < np + 4) { sc.pe.resize(2 * np + 4); sc.pf.resize(2 * np + 4); sc.so.resize(2 * np + 4); sc.pp.resize(2 * np + 4); }
    sc.jobs.clear();
    for (int k = 0, p = 0; k < 8; ++k) {
        const std::uint32_t ns = nsk[k];
        if (stats && ns) { stats->triangleTests += ns; (car_.shape[k].wheel < 0 ? stats->bodySurvivors : stats->wheelSurvivors) += ns; }
        const std::uint32_t first = static_cast<std::uint32_t>(sc.jobs.size());
        const std::uint32_t *list = memo.lists.idx[k].data();
        for (std::uint32_t j = 0; j < ns; ++j, ++p) {
            const std::uint32_t e = list[j];
            const std::uint32_t inst = memo.q.ref[e].instance;
            std::uint32_t f = first;
            for (; f < sc.jobs.size(); ++f) if (sc.jobs[f].instance == inst) break;
            if (f == sc.jobs.size()) sc.jobs.push_back(Job{static_cast<std::uint32_t>(k), inst, memo.q.ref[e].local});
            sc.pe[p] = e;
            sc.pf[p] = f;
        }
    }
    const std::size_t nj = sc.jobs.size();
    if (sc.frames.size() < nj + 4) sc.frames.resize(2 * nj + 4);
    for (std::size_t j0 = 0; j0 < nj; j0 += 4) {   // FV's GmCollision_Ellipsoid_Mesh transforms, 4 frames per SSE pass, bits as FV
        const Job *jb[4];
        for (int l = 0; l < 4; ++l) jb[l] = &sc.jobs[j0 + l < nj ? j0 + l : j0];
        const ShapePose *sp[4] = {&shapes[jb[0]->shape], &shapes[jb[1]->shape], &shapes[jb[2]->shape], &shapes[jb[3]->shape]};
        const float *mi[4];
        for (int l = 0; l < 4; ++l) mi[l] = compact ? memo.q.inst[jb[l]->local].iso : v.instances[jb[l]->instance].iso;
        const lanes::I3<F4> ell = GatherIso(sp[0]->world.r, sp[1]->world.r, sp[2]->world.r, sp[3]->world.r, 0);   // Iso = r[9], t[3]
        const lanes::I3<F4> mesh = GatherIso(mi[0], mi[1], mi[2], mi[3], 0);
        const lanes::V3<F4> rad = {G4(sp[0]->radii, sp[1]->radii, sp[2]->radii, sp[3]->radii, 0), G4(sp[0]->radii, sp[1]->radii, sp[2]->radii, sp[3]->radii, 1),
                                   G4(sp[0]->radii, sp[1]->radii, sp[2]->radii, sp[3]->radii, 2)};
        const lanes::FrameFv<F4> f = lanes::MakeFrameFv(ell, rad, mesh);
        float *dst[4];
        for (int l = 0; l < 4; ++l) dst[l] = reinterpret_cast<float *>(&sc.frames[j0 + l]);
        ScatterIso(f.meshToUnit, dst, 0);
        ScatterIso(f.c2w, dst, 12);
        ScatterIso(f.n2w, dst, 24);
    }
    if (opt_.checkFrames) frameMismatches_ += CheckFrames(v, shapes, sc.jobs.data(), sc.frames.data(), nj);
    if (stats) { stats->fvFrames += nj; stats->frames += nj; lap(stats->tFrames); }
    if (opt_.profileStop == 2) return;
    // pass A, SSE lanes over the call's survivors: unit-space vertices, FV's normal, the decision
    // (tw_exact_fast Decide) and the FACE payload (EmitFace)
    // the survivors pass A sees: all, or with merged patches only the lone triangles (patch members get the
    // patch's loop test in pass B)
    const bool patches = opt_.patches == kPatchMerged;
    std::uint32_t na = np;
    if (patches) {
        if (sc.aIdx.size() < np + 4) sc.aIdx.resize(2 * np + 4);
        na = 0;
        for (std::uint32_t pi = 0; pi < np; ++pi) {
            const std::uint32_t pt = patches_.patchOf[memo.q.ref[sc.pe[pi]].modelTri];
            sc.pp[pi] = pt;
            if (pt == patch::kNoPatch) sc.aIdx[na++] = pi;
            else sc.so[pi].cls = kPatchM;
        }
    }
    for (std::uint32_t j0 = 0; j0 < na; j0 += 4) {
        std::uint32_t ix[4];
        for (int l = 0; l < 4; ++l) ix[l] = patches ? sc.aIdx[j0 + l < na ? j0 + l : j0] : (j0 + l < np ? j0 + l : j0);
        const float *fr[4], *tv[4];
        for (int l = 0; l < 4; ++l) {
            fr[l] = reinterpret_cast<const float *>(&sc.frames[sc.pf[ix[l]]]);
            tv[l] = compact ? reinterpret_cast<const float *>(&memo.q.verts[sc.pe[ix[l]]])
                            : reinterpret_cast<const float *>(&triVerts_[memo.q.ref[sc.pe[ix[l]]].modelTri]);
        }
        const lanes::I3<F4> m2u = GatherIso(fr[0], fr[1], fr[2], fr[3], 0);
        F4 pv[8];
        G4x4(tv[0], tv[1], tv[2], tv[3], 0, pv);
        G4x4(tv[0], tv[1], tv[2], tv[3], 4, pv + 4);
        const F4 p8 = G4(tv[0], tv[1], tv[2], tv[3], 8);
        const lanes::V3<F4> mv[3] = {{pv[0], pv[1], pv[2]}, {pv[3], pv[4], pv[5]}, {pv[6], pv[7], p8}};
        lanes::V3<F4> vu[3];
        for (int c = 0; c < 3; ++c) vu[c] = lanes::TPoint(m2u, mv[c]);
        F4 lenSq;
        const lanes::V3<F4> nr = lanes::UnitNormal(vu, &lenSq);
        B4 rej, face, fvb;
        lanes::Decide4(vu, nr, lenSq, exact::kDirEpsSq, exact::kMarginEps, &rej, &face, &fvb);
        const int nl = static_cast<int>(std::min<std::uint32_t>(4u, na - j0));
        const int lm = (1 << nl) - 1;
        const int rb = lanes::Bits(rej) & lm, fb = lanes::Bits(face) & lm, vb = lanes::Bits(fvb) & lm & ~rb;
        if (stats) {
            stats->decReject += static_cast<std::uint64_t>(__builtin_popcount(rb));
            stats->decFace += static_cast<std::uint64_t>(__builtin_popcount(fb));
            stats->decFv += static_cast<std::uint64_t>(nl - __builtin_popcount(rb | fb));
        }
        for (int l = 0; l < nl; ++l) sc.so[ix[l]].cls = (rb >> l & 1) ? kRej : (fb >> l & 1) ? kFaceC : (vb >> l & 1) ? kFvViol : kFvAny;
        if (rb == lm) continue;
        alignas(16) float X[21][4];   // vu (9), n (3), P N Sep (9)
        {
            const F4 len = lanes::Sqrt(lenSq);
            const F4 invLen = lanes::K(1.0f, len) / len;
            const lanes::V3<F4> n = {nr.x * invLen, nr.y * invLen, invLen * nr.z};   // FV's n (FvCollide / EmitFace)
            for (int c = 0; c < 3; ++c) {
                _mm_store_ps(X[3 * c], vu[c].x.v); _mm_store_ps(X[3 * c + 1], vu[c].y.v); _mm_store_ps(X[3 * c + 2], vu[c].z.v);
            }
            _mm_store_ps(X[9], n.x.v); _mm_store_ps(X[10], n.y.v); _mm_store_ps(X[11], n.z.v);
        }
        if (fb) {
            lanes::FrameFv<F4> f;
            f.meshToUnit = m2u;
            f.c2w = GatherIso(fr[0], fr[1], fr[2], fr[3], 12);
            f.n2w = GatherIso(fr[0], fr[1], fr[2], fr[3], 24);
            lanes::V3<F4> oP, oN, oS, oK;
            lanes::FacePayload<F4, B4>(f, vu, nr, lenSq, exact::kDirEpsSq, &oP, &oN, &oS, &oK);
            _mm_store_ps(X[12], oP.x.v); _mm_store_ps(X[13], oP.y.v); _mm_store_ps(X[14], oP.z.v);
            _mm_store_ps(X[15], oN.x.v); _mm_store_ps(X[16], oN.y.v); _mm_store_ps(X[17], oN.z.v);
            _mm_store_ps(X[18], oS.x.v); _mm_store_ps(X[19], oS.y.v); _mm_store_ps(X[20], oS.z.v);
        }
        for (int l = 0; l < nl; ++l) {
            if (rb >> l & 1) continue;
            SurvOut &o = sc.so[ix[l]];
            const int nf = (fb >> l & 1) ? 21 : 12;
            for (int f = 0; f < nf; ++f) o.x[f] = X[f][l];
        }
    }
    if (stats) lap(stats->tTests);
    // pass B per shape, FV order: the FACE payloads, FV's form for the rest; then FV's merge and the output.
    // A feature-only pair (kFvViol: FV's edge loop certainly ends at a violated edge, so it yields a feature
    // contact or none) whose K (= its n) NearlyEquals the K of a FACE of the same shape is skipped: FV's
    // merge (MergeAndAddToCollisions' first test, against any primary) drops its contact.
    for (int k = 0, p0 = 0; k < 8; ++k) {
        if (closed[k]) {
            if (out.n < model::kRespMaxContacts) out.c[out.n++] = closedC[k]; else out.overflow = true;
            continue;
        }
        const std::uint32_t ns = nsk[k];
        if (ns == 0u) continue;
        sc.raw.clear();
        int nfK = 0;
        const float *fK[16];
        for (std::uint32_t pi = p0; pi < p0 + ns && nfK < 16; ++pi)
            if (sc.so[pi].cls == kFaceC) fK[nfK++] = sc.so[pi].x + 9;
        int nDone = 0;
        std::uint64_t done[16];   // (instance, patch) already tested for this shape
        for (std::uint32_t pi = p0; pi < p0 + ns; ++pi) {
            const SurvOut &o = sc.so[pi];
            if (o.cls == kRej) continue;
            if (o.cls == kPatchM) {   // the patch's loop test once per (shape, instance, patch), at its first surviving member
                const std::uint64_t key = (std::uint64_t(memo.q.ref[sc.pe[pi]].instance) << 32) | sc.pp[pi];
                int d = 0;
                while (d < nDone && done[d] != key) ++d;
                if (d < nDone) continue;
                if (nDone < 16) done[nDone++] = key;
                PatchCollide(sc.frames[sc.pf[pi]], patches_, sc.pp[pi], triVerts_[patches_.patches[sc.pp[pi]].rep],
                             compact ? memo.q.inst[memo.q.ref[sc.pe[pi]].local].iso : v.instances[memo.q.ref[sc.pe[pi]].instance].iso,
                             compact ? sc.pe[pi] : memo.q.tri[sc.pe[pi]], &sc.raw);
                if (stats) ++stats->patchTests;
                continue;
            }
            const std::uint32_t t = compact ? sc.pe[pi] : memo.q.tri[sc.pe[pi]];   // Raw::tri: the memo entry (compact) or the world triangle
            if (o.cls == kFaceC) {
                sc.raw.push_back(exact::Raw{exact::V{o.x[12], o.x[13], o.x[14]}, exact::V{o.x[15], o.x[16], o.x[17]}, exact::V{o.x[18], o.x[19], o.x[20]},
                                            exact::V{o.x[9], o.x[10], o.x[11]}, true, t});
                continue;
            }
            if (o.cls == kFvViol) {
                const exact::V n = {o.x[9], o.x[10], o.x[11]};
                int f = 0;
                for (; f < nfK; ++f)
                    if (NearlyEqualV(n, exact::V{fK[f][0], fK[f][1], fK[f][2]})) break;
                if (f < nfK) { if (stats) ++stats->absorbedSkips; continue; }
            }
            FvFormPair(sc.frames[sc.pf[pi]], o.x, t, &sc.raw);
        }
        p0 += ns;
        if (sc.raw.empty()) continue;
        if (stats) { stats->rawContacts += sc.raw.size(); (car_.shape[k].wheel < 0 ? stats->bodyRaw : stats->wheelRaw) += sc.raw.size(); }
        const int before = out.n;
        const std::size_t merged = compact ? MergeShapeTo(memo.q, sc.raw.data(), sc.raw.size(), k, &sc.merged, out)
                                           : MergeShapeTo(v, sc.raw.data(), sc.raw.size(), k, &sc.merged, out);
        if (stats) {
            if (static_cast<std::size_t>(out.n - before) < merged) ++stats->overflow;
            stats->merged += merged;
        }
    }
    if (stats) lap(stats->tMerge);
}

}  // namespace collide
}  // namespace tw
