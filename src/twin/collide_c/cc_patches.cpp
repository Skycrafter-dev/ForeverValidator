// World-space convex coplanar patches over the twin's triangle soup, plus a
// binary BVH over patch boxes (cold data; built once per map).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <unordered_map>

#include "cc_core.h"

namespace cc {

namespace {

struct VKey {
    std::uint32_t x, y, z;
    bool operator==(const VKey &o) const { return x == o.x && y == o.y && z == o.z; }
};
struct VKeyHash {
    std::size_t operator()(const VKey &k) const {
        std::uint64_t h = (std::uint64_t(k.x) * 0x9e3779b97f4a7c15ull) ^ (std::uint64_t(k.y) * 0xc2b2ae3d27d4eb4full) ^
                          (std::uint64_t(k.z) * 0x165667b19e3779f9ull);
        h ^= h >> 29;
        return static_cast<std::size_t>(h);
    }
};

inline void Cross(const double a[3], const double b[3], double c[3]) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}
inline double Dot(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

}  // namespace

void PatchWorld::Build(const tw::World &w, const tw::SourceScene *src, double angTol, double distTol, std::uint32_t maxMembersPerPatch) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t T = w.tris.size();
    std::vector<double> N(3 * T, 0.0), area(T, 0.0);
    std::vector<char> bad(T, 0), skip(T, 0);
    // the world's enclosing sphere (actor 1, triangles > 1 km from the map, boxes > 4 km): out of reach, left out
    for (std::size_t t = 0; t < T; ++t) {
        const tw::Tri &tr = w.tris[t];
        const float ext = std::max(tr.hi[0] - tr.lo[0], std::max(tr.hi[1] - tr.lo[1], tr.hi[2] - tr.lo[2]));
        if ((tr.meta >> tw::kMetaActorShift) == 1u && ext > 4000.0f) { skip[t] = 1; ++enclosure; }
    }
    for (std::size_t t = 0; t < T; ++t) {
        const tw::Tri &tr = w.tris[t];
        double e[3][3];
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k) e[i][k] = double(tr.v[(i + 1) % 3][k]) - double(tr.v[i][k]);
        const double e2[3] = {double(tr.v[2][0]) - tr.v[0][0], double(tr.v[2][1]) - tr.v[0][1], double(tr.v[2][2]) - tr.v[0][2]};
        double c[3];
        Cross(e[0], e2, c);
        const double l = std::sqrt(Dot(c, c));
        area[t] = 0.5 * l;
        double minEdge = 1e30;
        for (int i = 0; i < 3; ++i) minEdge = std::min(minEdge, std::sqrt(Dot(e[i], e[i])));
        // Slivers and tiny edges: FV's unit-space degeneracy thresholds (1e-10) can fire; exact test only.
        if (!(l > 1e-5) || !(minEdge > 1e-4)) { bad[t] = 1; continue; }
        for (int k = 0; k < 3; ++k) N[3 * t + k] = c[k] / l;
    }
    // Precise world geometry for the merge decisions: FV's world floats are the mesh-local vertices
    // transformed and rounded (ulp ~3e-5 m at 500 m), which puts ~1e-4..1e-3 rad of noise on the
    // normals of small triangles. Recompute vertices in double from the source scene (same triangle
    // order as the world, tw_world_compile.cpp) when it is there.
    std::vector<double> X(9 * T);
    for (std::size_t t = 0; t < T; ++t)
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k) X[9 * t + 3 * i + k] = w.tris[t].v[i][k];
    if (src && src->tris.size() == T) {
        for (const tw::SourceSurface &ss : src->surfaces)
            for (std::uint32_t t = ss.firstTri; t < ss.firstTri + ss.triCount; ++t)
                for (int i = 0; i < 3; ++i) {
                    const float *lv = &src->vertices[3 * (ss.firstVertex + src->tris[t].v[i])];
                    for (int k = 0; k < 3; ++k)
                        X[9 * t + 3 * i + k] = ((double(lv[0]) * ss.iso[k] + double(lv[1]) * ss.iso[3 + k]) + double(lv[2]) * ss.iso[6 + k]) + ss.iso[9 + k];
                }
        precise = true;
    }
    std::vector<double> NX(3 * T, 0.0);
    for (std::size_t t = 0; t < T; ++t) {
        if (bad[t]) continue;
        const double *x = &X[9 * t];
        const double e1[3] = {x[3] - x[0], x[4] - x[1], x[5] - x[2]}, e2[3] = {x[6] - x[0], x[7] - x[1], x[8] - x[2]};
        double c[3];
        Cross(e1, e2, c);
        const double l = std::sqrt(Dot(c, c));
        if (!(l > 0)) { bad[t] = 1; continue; }
        for (int k = 0; k < 3; ++k) NX[3 * t + k] = c[k] / l;
    }
    // weld vertices on the precise positions, 10 um grid (as B's block models do in mesh-local space)
    std::unordered_map<VKey, std::uint32_t, VKeyHash> vmap;
    vmap.reserve(T * 2);
    std::vector<std::uint32_t> vid(3 * T);
    std::vector<double> vpos;
    for (std::size_t t = 0; t < T; ++t)
        for (int i = 0; i < 3; ++i) {
            const double *x = &X[9 * t + 3 * i];
            const VKey key{static_cast<std::uint32_t>(std::llround(x[0] * 1e5)), static_cast<std::uint32_t>(std::llround(x[1] * 1e5)),
                           static_cast<std::uint32_t>(std::llround(x[2] * 1e5))};
            auto it = vmap.find(key);
            if (it == vmap.end()) {
                const std::uint32_t id = static_cast<std::uint32_t>(vpos.size() / 3);
                vmap.emplace(key, id);
                vpos.push_back(x[0]); vpos.push_back(x[1]); vpos.push_back(x[2]);
                vid[3 * t + i] = id;
            } else {
                vid[3 * t + i] = it->second;
            }
        }
    // directed edges (a -> b) -> chain of (triangle, edge) uses
    std::unordered_map<std::uint64_t, std::int64_t> dir;
    dir.reserve(T * 4);
    std::vector<std::int64_t> chainNext(3 * T, -1);
    auto key = [](std::uint32_t a, std::uint32_t b) { return (std::uint64_t(a) << 32) | b; };
    for (std::size_t t = 0; t < T; ++t)
        for (int i = 0; i < 3; ++i) {
            const std::uint32_t a = vid[3 * t + i], b = vid[3 * t + (i + 1) % 3];
            if (a == b) continue;
            const std::uint64_t k = key(a, b);
            auto it = dir.find(k);
            const std::int64_t me = static_cast<std::int64_t>(3 * t + i);
            if (it == dir.end()) dir.emplace(k, me);
            else { chainNext[me] = it->second; it->second = me; }
        }
    auto forEdge = [&](std::uint32_t a, std::uint32_t b, auto &&f) {  // triangles using the directed edge a -> b
        auto it = dir.find(key(a, b));
        if (it == dir.end()) return;
        for (std::int64_t e = it->second; e >= 0; e = chainNext[e]) if (f(static_cast<std::uint32_t>(e / 3))) return;
    };
    std::vector<std::uint32_t> ord(T);
    std::iota(ord.begin(), ord.end(), 0u);
    std::stable_sort(ord.begin(), ord.end(), [&](std::uint32_t a, std::uint32_t b) { return area[a] > area[b]; });
    triPatch.assign(T, 0xffffffffu);
    std::vector<std::int32_t> memberOf(T, -1);
    patches.clear(); edges.clear(); members.clear();
    std::vector<std::uint32_t> poly, mem;
    std::size_t rej[5] = {0, 0, 0, 0, 0};
    auto P = [&](std::uint32_t v) { return &vpos[3 * v]; };
    for (std::uint32_t seed : ord) {
        if (triPatch[seed] != 0xffffffffu || skip[seed]) continue;
        const std::uint32_t pid = static_cast<std::uint32_t>(patches.size());
        const tw::Tri &st = w.tris[seed];
        Patch pa{};
        pa.meta = st.meta & kMetaKeyMask;
        pa.firstMember = static_cast<std::uint32_t>(members.size());
        pa.firstEdge = static_cast<std::uint32_t>(edges.size());
        mem.clear();
        mem.push_back(seed);
        triPatch[seed] = pid;
        double np[3] = {NX[3 * seed], NX[3 * seed + 1], NX[3 * seed + 2]};
        const double refF[3] = {st.v[0][0], st.v[0][1], st.v[0][2]};
        const double *ref = P(vid[3 * seed]);  // precise position (2D basis, merge distances)
        if (bad[seed]) {
            pa.flags = kPatchExactOnly;
            ++exactOnly;
        } else {
            // in-plane 2D basis
            double b1[3] = {P(vid[3 * seed + 1])[0] - ref[0], P(vid[3 * seed + 1])[1] - ref[1], P(vid[3 * seed + 1])[2] - ref[2]};
            const double l1 = std::sqrt(Dot(b1, b1));
            for (double &x : b1) x /= l1;
            double b2[3];
            Cross(np, b1, b2);
            auto to2 = [&](std::uint32_t v, double o[2]) {
                const double d[3] = {P(v)[0] - ref[0], P(v)[1] - ref[1], P(v)[2] - ref[2]};
                o[0] = Dot(d, b1); o[1] = Dot(d, b2);
            };
            auto turn = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {  // >0: left turn (convex for CCW)
                double pa2[2], pb2[2], pc2[2];
                to2(a, pa2); to2(b, pb2); to2(c, pc2);
                const double u0 = pb2[0] - pa2[0], u1 = pb2[1] - pa2[1], v0 = pc2[0] - pb2[0], v1 = pc2[1] - pb2[1];
                const double cr = u0 * v1 - u1 * v0;
                const double sc = std::sqrt(u0 * u0 + u1 * u1) * std::sqrt(v0 * v0 + v1 * v1);
                return cr / (sc > 0 ? sc : 1.0);  // sine of the turn angle
            };
            poly.assign({vid[3 * seed], vid[3 * seed + 1], vid[3 * seed + 2]});
            bool grew = true;
            while (grew && mem.size() < maxMembersPerPatch) {
                grew = false;
                const std::size_t k = poly.size();
                for (std::size_t i = 0; i < k && !grew; ++i) {
                    const std::uint32_t a = poly[i], b = poly[(i + 1) % k];
                    std::uint32_t t2 = 0xffffffffu;
                    forEdge(b, a, [&](std::uint32_t u) {
                        if (triPatch[u] != 0xffffffffu || bad[u] || skip[u] || (w.tris[u].meta & kMetaKeyMask) != pa.meta) return false;
                        t2 = u;
                        return true;
                    });
                    if (t2 == 0xffffffffu) { ++rej[0]; continue; }
                    if ((w.tris[t2].meta & kMetaKeyMask) != pa.meta) continue;
                    const double dn[3] = {NX[3 * t2] - np[0], NX[3 * t2 + 1] - np[1], NX[3 * t2 + 2] - np[2]};
                    if (Dot(dn, dn) > angTol * angTol) { ++rej[1]; continue; }
                    std::uint32_t c = 0xffffffffu;
                    bool ok = true;
                    for (int j = 0; j < 3; ++j) {
                        const std::uint32_t v = vid[3 * t2 + j];
                        const double d[3] = {P(v)[0] - ref[0], P(v)[1] - ref[1], P(v)[2] - ref[2]};
                        if (std::fabs(Dot(d, np)) > distTol) ok = false;
                        if (v != a && v != b) { if (c != 0xffffffffu) ok = false; c = v; }
                    }
                    if (!ok || c == 0xffffffffu) { ++rej[2]; continue; }
                    if (std::find(poly.begin(), poly.end(), c) != poly.end()) { ++rej[3]; continue; }
                    const std::uint32_t prev = poly[(i + k - 1) % k], next = poly[(i + 2) % k];
                    if (turn(prev, a, c) < -1e-7 || turn(a, c, b) < 1e-7 || turn(c, b, next) < -1e-7) { ++rej[4]; continue; }
                    poly.insert(poly.begin() + static_cast<std::ptrdiff_t>(i + 1), c);
                    mem.push_back(t2);
                    triPatch[t2] = pid;
                    grew = true;
                }
            }
            // outline: drop (near-)collinear corners
            std::vector<std::uint32_t> corners;
            const std::size_t k = poly.size();
            for (std::size_t i = 0; i < k; ++i)
                if (turn(poly[(i + k - 1) % k], poly[i], poly[(i + 1) % k]) > 1e-7) corners.push_back(poly[i]);
            for (std::size_t i = 0; i < corners.size(); ++i) {
                const double *a = P(corners[i]), *b = P(corners[(i + 1) % corners.size()]);
                const double e[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
                double u[3];
                Cross(e, np, u);
                const double l = std::sqrt(Dot(u, u));
                PEdge pe;
                for (int q = 0; q < 3; ++q) { pe.u[q] = static_cast<float>(u[q] / l); pe.a[q] = static_cast<float>(a[q]); }
                edges.push_back(pe);
            }
        }
        // members
        double tau = 0, alpha = 0, cen[3] = {0, 0, 0};
        Box3 box = tw::EmptyBox();
        for (std::uint32_t t : mem) {
            memberOf[t] = static_cast<std::int32_t>(members.size());
            const tw::Tri &tr = w.tris[t];
            Member m{};
            m.tri = t; m.meta = tr.meta; m.patch = pid;
            for (int q = 0; q < 3; ++q) { m.v0[q] = tr.v[0][q]; m.n[q] = static_cast<float>(N[3 * t + q]); }
            for (int i = 0; i < 3; ++i) {
                const float *a = tr.v[i], *b = tr.v[(i + 1) % 3];
                const double e[3] = {double(b[0]) - a[0], double(b[1]) - a[1], double(b[2]) - a[2]};
                double u[3];
                Cross(e, &N[3 * t], u);
                const double l = std::sqrt(Dot(u, u));
                for (int q = 0; q < 3; ++q) { m.u[i][q] = l > 0 ? static_cast<float>(u[q] / l) : 0.0f; m.a[i][q] = a[q]; }
                m.nb[i] = -1;
            }
            members.push_back(m);
            for (int i = 0; i < 3; ++i) {
                const double d[3] = {double(tr.v[i][0]) - refF[0], double(tr.v[i][1]) - refF[1], double(tr.v[i][2]) - refF[2]};
                tau = std::max(tau, std::fabs(Dot(d, np)));
                for (int q = 0; q < 3; ++q) cen[q] += tr.v[i][q];
            }
            const double dn[3] = {N[3 * t] - np[0], N[3 * t + 1] - np[1], N[3 * t + 2] - np[2]};
            alpha = std::max(alpha, std::sqrt(Dot(dn, dn)));
            for (int q = 0; q < 3; ++q) { box.lo[q] = std::min(box.lo[q], tr.lo[q]); box.hi[q] = std::max(box.hi[q], tr.hi[q]); }
        }
        for (double &x : cen) x /= 3.0 * mem.size();
        double rad = 0;
        for (std::uint32_t t : mem)
            for (int i = 0; i < 3; ++i) {
                const double d[3] = {w.tris[t].v[i][0] - cen[0], w.tris[t].v[i][1] - cen[1], w.tris[t].v[i][2] - cen[2]};
                rad = std::max(rad, std::sqrt(Dot(d, d)));
            }
        for (int q = 0; q < 3; ++q) {
            pa.n[q] = static_cast<float>(np[q]);
            pa.ref[q] = static_cast<float>(refF[q]);
            pa.cen[q] = static_cast<float>(cen[q]);
        }
        // slack for float evaluation of n.(t - ref) far from ref and for the float-rounded normal
        pa.tau = static_cast<float>(tau + 1e-4 + 2e-7 * rad);
        pa.alpha = static_cast<float>(alpha + 1e-6);
        pa.rad = static_cast<float>(rad * (1 + 1e-6) + 1e-4);
        pa.box = tw::Inflate(box, 2e-4f);
        pa.memberCount = static_cast<std::uint32_t>(mem.size());
        pa.edgeCount = static_cast<std::uint32_t>(edges.size()) - pa.firstEdge;
        patches.push_back(pa);
        if (mem.size() > 1) ++multi;
        maxMembers = std::max(maxMembers, mem.size());
    }
    // neighbour links inside each patch
    for (Member &m : members) {
        const std::uint32_t t = m.tri;
        for (int i = 0; i < 3; ++i) {
            forEdge(vid[3 * t + (i + 1) % 3], vid[3 * t + i], [&](std::uint32_t t2) {
                if (triPatch[t2] == m.patch && memberOf[t2] >= 0 && t2 != t) { m.nb[i] = memberOf[t2]; return true; }
                return false;
            });
        }
    }
    if (std::getenv("CC_PATCHDIAG"))
        std::printf("patch growth rejections (last attempt per edge, repeated per growth round): no free neighbour %zu, normal %zu, distance/topology %zu, apex on outline %zu, convexity %zu\n",
                    rej[0], rej[1], rej[2], rej[3], rej[4]);
    BuildBvh();
    buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void PatchWorld::BuildBvh() {
    tw::World boxes;
    boxes.tris.resize(patches.size());
    for (std::size_t i = 0; i < patches.size(); ++i)
        for (int a = 0; a < 3; ++a) { boxes.tris[i].lo[a] = patches[i].box.lo[a]; boxes.tris[i].hi[a] = patches[i].box.hi[a]; }
    bvh.Build(boxes);
}

std::size_t PatchWorld::Bytes() const {
    return patches.size() * sizeof(Patch) + edges.size() * sizeof(PEdge) + members.size() * sizeof(Member) + bvh.Bytes();
}

}  // namespace cc
