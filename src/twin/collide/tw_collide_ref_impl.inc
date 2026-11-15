// The body of tw_collide_ref.cpp, compiled once per scalar type T (double; tw::sr::sr). Included inside a
// namespace that defines T. Every literal is a float constructor (exact in both types); sqrt and fabs go
// through argument-dependent lookup.
struct V { T x, y, z; };
inline T Dot(const V &a, const V &b) {
    const T xy = a.x * b.x + a.y * b.y;
    return xy + a.z * b.z;
}
inline V Add(const V &a, const V &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V Sub(const V &a, const V &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V Scale(const V &a, T s) { return {a.x * s, a.y * s, a.z * s}; }
inline V Neg(const V &a) { return {-a.x, -a.y, -a.z}; }
inline V Cross(const V &a, const V &b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline T Sqrt(T v) {
    using std::sqrt;
    return sqrt(v);
}
inline T Fabs(T v) {
    using std::fabs;
    return fabs(v);
}

// FV's float constants, widened
const T kDirEpsSq = T(1.0e-5f * 1.0e-5f);   // FV's float values, exact in T
const T kCollisionDistance = T(1.0e-5f);
const T kNormalAlignment = T(0.8660254f);
const T kNearly = T(1.0e-5f);

inline V NormalizeForCollision(const V &v, T epsSq) {
    const T l2 = Dot(v, v);
    if (epsSq < l2) return Scale(v, T(1) / Sqrt(l2));
    return v;
}

struct M { V c[3]; };
struct I { M r; V t; };
inline M MOf(const float *r9) { return M{{V{r9[0], r9[1], r9[2]}, V{r9[3], r9[4], r9[5]}, V{r9[6], r9[7], r9[8]}}}; }
inline I IOf(const Iso &s) { return I{MOf(s.r), V{s.t[0], s.t[1], s.t[2]}}; }
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
inline I Diagonal(const V &s) {
    const T z = T(0.0f);
    return I{M{{V{s.x, z, z}, V{z, s.y, z}, V{z, z, s.z}}}, V{z, z, z}};
}
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

struct Ctx {
    Margin *m, *ms;   // overall, this shape's
    std::uint8_t shape;
    std::uint32_t tri;
    void Note(T margin, DecisionKind kind) {
        const double a = std::fabs(static_cast<double>(margin));
        if (a < ms->value) {
            ms->value = a; ms->kind = kind; ms->shape = shape; ms->tri = tri;
            if (a < m->value) *m = *ms;
        }
    }
};

// SSphereMeshCollide::CollideTriangle (unit sphere at the origin), every comparison's margin noted
struct TriangleCollide {
    V n;
    std::vector<Raw> *out;
    std::uint32_t tri;
    Ctx *cx;

    int EmitFeature(const V &point, T minDistSq, bool requireContainment) {
        const V toCenter = Sub(V{T(0.0f), T(0.0f), T(0.0f)}, point);
        const T dSq = Dot(toCenter, toCenter);
        if (requireContainment) cx->Note(dSq - T(1.0f) * T(1.0f), kFeatureDistance);
        cx->Note(dSq - minDistSq, kFeatureDistance);
        if ((requireContainment && T(1.0f) * T(1.0f) < dSq) || !(minDistSq < dSq)) return 0;
        const T d = Sqrt(dSq);
        const T inv = T(1) / d;
        const V normal = {toCenter.x * inv, toCenter.y * inv, toCenter.z * inv};
        const T dmr = d - T(1.0f);
        const T ps = dmr * inv;
        const V pen = {toCenter.x * ps, toCenter.y * ps, toCenter.z * ps};
        const T along = Dot(pen, n);
        out->push_back(Raw{point, normal, Scale(n, along), n, false, tri});
        return 1;
    }
    int EmitEndpointB(const V &point, T minDistance) {
        const V toCenter = Sub(V{T(0.0f), T(0.0f), T(0.0f)}, point);
        const T dSq = Dot(toCenter, toCenter);
        const T d = Sqrt(dSq);
        cx->Note(d - T(1.0f) * T(1.0f), kFeatureDistance);
        cx->Note(d - minDistance, kFeatureDistance);
        if (T(1.0f) * T(1.0f) < d || !(minDistance < d)) return 0;
        const T ed = Sqrt(d);   // FV: endpoint B uses a second root
        const T inv = T(1) / ed;
        const V normal = {toCenter.x * inv, toCenter.y * inv, toCenter.z * inv};
        const T edmr = ed - T(1.0f);
        const T ps = edmr * inv;
        const V pen = {toCenter.x * ps, toCenter.y * ps, toCenter.z * ps};
        const T along = Dot(pen, n);
        out->push_back(Raw{point, normal, Scale(n, along), n, false, tri});
        return 1;
    }
    int Collide(const V v[3]) {
        const V zero = {T(0.0f), T(0.0f), T(0.0f)};
        const T pd = Dot(Sub(zero, v[0]), n);
        cx->Note(std::min(Fabs(pd), Fabs(T(1.0f) - pd)), kPlaneReach);
        if (T(1.0f) < pd || pd < T(0)) return 0;
        const T edgeReach = Sqrt(T(1.0f) * T(1.0f) - pd * pd);
        const V projected = Add(zero, Scale(n, -pd));
        for (int e = 0; e < 3; ++e) {
            const int nx = e == 2 ? 0 : e + 1;
            const V start = v[e], end = v[nx];
            const V dir = NormalizeForCollision(Sub(end, start), kDirEpsSq);
            const V edgeNormal = Cross(dir, n);
            const T edgeDistance = Dot(Sub(projected, start), edgeNormal);
            cx->Note(edgeReach - edgeDistance, kEdgeReach);
            if (edgeReach < edgeDistance) return 0;
            cx->Note(edgeDistance, kEdgeSign);
            if (edgeDistance > T(0)) {
                const T fromStart = Dot(Sub(projected, start), dir);
                cx->Note(fromStart, kFeatureBranch);
                if (fromStart < T(0)) return EmitFeature(start, kDirEpsSq, true);
                const T fromEnd = Dot(Sub(projected, end), dir);
                cx->Note(fromEnd, kFeatureBranch);
                if (!(T(0) < fromEnd)) return EmitFeature(Add(projected, Scale(edgeNormal, -edgeDistance)), kCollisionDistance, false);
                return EmitEndpointB(end, kDirEpsSq);
            }
        }
        if (pd > T(0)) {
            out->push_back(Raw{projected, n, Scale(n, pd - T(1.0f)), n, true, tri});
            return 1;
        }
        return 0;
    }
};

// NearlyEqual(value, reference) with its margin: the distance to the band's edge (positive inside)
inline bool NearlyEqual(T value, T reference, T *margin) {
    const T tol = Fabs(reference) * kNearly;
    *margin = std::min(value - (reference - tol), (reference + tol) - value);
    return reference - tol <= value && value <= reference + tol;
}

void DetectImpl(const ExactCollision &stage, const T pos[3], const T quat[4], const T absorb[4], DetectionD *out) {
    out->contacts.clear();
    out->overflow = false;
    out->closest = Margin{};
    for (Margin &m : out->perShape) m = Margin{};
    out->candidates = 0;
    const world::WorldPackView &v = stage.View();
    const CarShapes &car = stage.Car();
    // FV GmMat3::Set(GmQuat), in double
    const T x = quat[0], y = quat[1], z = quat[2], w = quat[3];
    const T x2 = x * T(2.0f), y2 = y * T(2.0f), z2 = z * T(2.0f);
    const T xw = w * x2, yw = w * y2, zw = w * z2;
    const T xx = x * x2, xy = x * y2, xz = x * z2;
    const T yy = y * y2, yz = y * z2, zz = z * z2;
    const T one = T(1.0f);
    const I carIso{M{{V{(one - yy) - zz, xy + zw, xz - yw}, V{xy - zw, (one - xx) - zz, yz + xw}, V{xz + yw, yz - xw, (one - xx) - yy}}}, V{pos[0], pos[1], pos[2]}};
    std::vector<std::pair<std::uint32_t, std::uint32_t>> order;
    std::vector<Raw> raw;
    std::vector<int> merged;
    for (std::uint32_t k = 0; k < car.count && k < 8u; ++k) {
        const CarShape &sh = car.shape[k];
        I local;
        if (sh.wheel >= 0) {   // FV CopyRestSurfaceIso + OffsetCurrentSurfaceY(-absorb)
            local = IOf(car.wheelRest[sh.wheel]);
            local.t.y += -absorb[sh.wheel];
        } else {
            local = IOf(sh.bodyPose);
        }
        const I ell = SetMult(local, carIso);
        const V rad = {sh.radii[0], sh.radii[1], sh.radii[2]};
        const V invRadii = {T(1) / rad.x, T(1) / rad.y, T(1) / rad.z};
        // candidates: the shape's world AABB inflated by 5 cm, in FV order
        Box3 q;
        const T ax[3][3] = {{ell.r.c[0].x, ell.r.c[0].y, ell.r.c[0].z}, {ell.r.c[1].x, ell.r.c[1].y, ell.r.c[1].z}, {ell.r.c[2].x, ell.r.c[2].y, ell.r.c[2].z}};
        const T ct[3] = {ell.t.x, ell.t.y, ell.t.z};
        for (int a = 0; a < 3; ++a) {
            const T h = Fabs(ax[0][a]) * rad.x + Fabs(ax[1][a]) * rad.y + Fabs(ax[2][a]) * rad.z + T(0.05f);
            q.lo[a] = static_cast<float>(static_cast<double>(ct[a] - h));
            q.hi[a] = static_cast<float>(static_cast<double>(ct[a] + h));
        }
        order.clear();
        stage.Bvh().Query(q, [&](std::uint32_t t) { order.push_back({v.triInfo[t].fvOrder, t}); });
        std::sort(order.begin(), order.end());
        raw.clear();
        std::uint32_t curInst = 0xFFFFFFFFu;
        I meshToUnit{}, c2w{}, n2w{};
        for (const auto &ot : order) {
            const std::uint32_t t = ot.second;
            const world::TriInfo &ti = v.triInfo[t];
            if (ti.instance != curInst) {   // GmCollision_Ellipsoid_Mesh's transforms
                curInst = ti.instance;
                const world::Instance &in = v.instances[ti.instance];
                const I mesh{MOf(in.iso), V{in.iso[9], in.iso[10], in.iso[11]}};
                const I ellipsoidToMesh = MultInverse(ell, mesh);
                const I meshToEllipsoid = Inverse(ellipsoidToMesh);
                meshToUnit = meshToEllipsoid;
                ScaleRows(meshToUnit, invRadii);
                c2w = SetMult(MultInverse(Diagonal(rad), meshToEllipsoid), mesh);
                n2w = SetMult(MultInverse(Diagonal(invRadii), meshToEllipsoid), mesh);
            }
            const world::Instance &in = v.instances[ti.instance];
            const world::Model &md = v.models[in.model];
            const world::ModelTri &mt = v.modelTris[md.firstTri + ti.modelTri];
            V vu[3];
            for (int c = 0; c < 3; ++c) {
                const world::Vec3f &p = v.modelVerts[md.firstVert + mt.v[c]];
                vu[c] = TPoint(meshToUnit, V{p.x, p.y, p.z});
            }
            ++out->candidates;
            const V e01 = Sub(vu[1], vu[0]), e02 = Sub(vu[2], vu[0]);
            const T nX = e02.z * e01.y - e02.y * e01.z;
            const T nY = e01.z * e02.x - e02.z * e01.x;
            const T nZ = e01.x * e02.y - e02.x * e01.y;
            const T lenSq = (nY * nY + nX * nX) + nZ * nZ;
            if (!(lenSq > kDirEpsSq)) continue;   // FV's degenerate skip
            const T len = Sqrt(lenSq);
            const T invLen = T(1) / len;
            const V n = {nX * invLen, nY * invLen, invLen * nZ};
            Ctx cx{&out->closest, &out->perShape[k], static_cast<std::uint8_t>(k), t};
            TriangleCollide tc{n, &raw, t, &cx};
            const std::size_t first = raw.size();
            if (!tc.Collide(vu)) continue;
            for (std::size_t r = first; r < raw.size(); ++r) {   // TransformEllipsoidMeshCollisionsToWorldForGmSurf
                Raw &c = raw[r];
                c.P = TPoint(c2w, c.P);
                c.N = NormalizeForCollision(TDir(n2w.r, c.N), kDirEpsSq);
                c.Sep = TDir(c2w.r, c.Sep);
            }
        }
        // MergeAndAddToCollisions: primaries in order, then the non-primaries no primary absorbs
        merged.clear();
        for (std::size_t r = 0; r < raw.size(); ++r) if (raw[r].primary) merged.push_back(static_cast<int>(r));
        const std::size_t nPrim = merged.size();
        for (std::size_t r = 0; r < raw.size(); ++r) {
            const Raw &c = raw[r];
            if (c.primary) continue;
            Ctx cx{&out->closest, &out->perShape[k], static_cast<std::uint8_t>(k), c.tri};
            std::size_t j = 0;
            for (; j < nPrim; ++j) {
                const Raw &p = raw[static_cast<std::size_t>(merged[j])];
                T mx, my, mz;
                const bool ne = NearlyEqual(c.K.x, p.K.x, &mx) & NearlyEqual(c.K.y, p.K.y, &my) & NearlyEqual(c.K.z, p.K.z, &mz);
                cx.Note(std::min(mx, std::min(my, mz)), kMergeNearlyEqual);
                const T dot = Dot(c.N, p.N);
                cx.Note(dot - kNormalAlignment, kMergeAlignment);
                if (ne || kNormalAlignment < dot) break;
            }
            if (j == nPrim) merged.push_back(static_cast<int>(r));
        }
        for (int idx : merged) {
            if (out->contacts.size() >= static_cast<std::size_t>(model::kRespMaxContacts)) { out->overflow = true; break; }
            const Raw &c = raw[static_cast<std::size_t>(idx)];
            const Tri &tr = v.tris[c.tri];
            const world::TriInfo &ti = v.triInfo[c.tri];
            const world::Actor &a = v.ActorOf(tr);
            ContactD o{};
            o.P[0] = static_cast<double>(c.P.x); o.P[1] = static_cast<double>(c.P.y); o.P[2] = static_cast<double>(c.P.z);
            o.N[0] = static_cast<double>(c.N.x); o.N[1] = static_cast<double>(c.N.y); o.N[2] = static_cast<double>(c.N.z);
            o.Sep[0] = static_cast<double>(c.Sep.x); o.Sep[1] = static_cast<double>(c.Sep.y); o.Sep[2] = static_cast<double>(c.Sep.z);
            for (int q2 = 0; q2 < 3; ++q2) o.peerZ[q2] = a.peerZ[q2];
            const bool trig = ti.trigger != world::kNoTrigger;
            o.peerId = trig ? static_cast<std::uint32_t>(ti.trigger) : a.peerId;
            o.tri = c.tri;
            o.shape = static_cast<std::uint8_t>(k);
            o.material = static_cast<std::uint8_t>(tr.meta & kMetaMaterialMask);
            o.flags = static_cast<std::uint8_t>((c.primary ? unsigned(model::kRespPrimary) : 0u) | (trig ? unsigned(model::kRespTrigger) : 0u));
            out->contacts.push_back(o);
        }
    }
}

