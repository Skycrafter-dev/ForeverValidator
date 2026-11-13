// Coplanar patches of the world package's models: see tw_patches.h (port of tw_blocks.cpp BuildModel).
#include "tw_patches.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

namespace tw::patch {

namespace {

struct D3 { double x, y, z; };
D3 Sub(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
D3 Cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
double Dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double Len(D3 a) { return std::sqrt(Dot(a, a)); }

void BuildModel(const world::WorldPackView &view, std::uint32_t mi, float planeEps, float angleEps, Set *s) {
    const world::Model &m = view.models[mi];
    const std::uint32_t nt = m.triCount;
    const world::Vec3f *mv = view.modelVerts.ptr + m.firstVert;
    auto V = [&](std::uint32_t local) -> D3 { return {mv[local].x, mv[local].y, mv[local].z}; };
    auto T = [&](std::uint32_t t) -> const world::ModelTri & { return view.modelTris[m.firstTri + t]; };
    // canonical position ids (duplicate vertices share an id)
    std::map<std::tuple<long long, long long, long long>, std::uint32_t> posId;
    std::vector<std::uint32_t> vid(m.vertCount);
    std::vector<D3> pos;
    std::vector<std::uint32_t> posLocal;   // a source vertex of each position (exact floats)
    for (std::uint32_t i = 0; i < m.vertCount; ++i) {
        const D3 p = V(i);
        auto key = std::make_tuple(std::llround(p.x * 1e5), std::llround(p.y * 1e5), std::llround(p.z * 1e5));
        auto it = posId.find(key);
        if (it == posId.end()) {
            it = posId.emplace(key, static_cast<std::uint32_t>(pos.size())).first;
            pos.push_back(p);
            posLocal.push_back(i);
        }
        vid[i] = it->second;
    }
    std::vector<D3> nrm(nt);
    std::vector<double> off(nt), area(nt);
    std::vector<bool> ok(nt);
    for (std::uint32_t t = 0; t < nt; ++t) {
        const world::ModelTri &tr = T(t);
        const D3 a = V(tr.v[0]), b = V(tr.v[1]), c = V(tr.v[2]);
        const D3 n = Cross(Sub(b, a), Sub(c, a));
        const double l = Len(n);
        ok[t] = l > 1e-10;
        nrm[t] = ok[t] ? D3{n.x / l, n.y / l, n.z / l} : D3{0, 1, 0};
        off[t] = Dot(nrm[t], a);
        area[t] = l;
    }
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<std::uint32_t>> edges;
    for (std::uint32_t t = 0; t < nt; ++t)
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t a = vid[T(t).v[k]], b = vid[T(t).v[(k + 1) % 3]];
            edges[{std::min(a, b), std::max(a, b)}].push_back(t);
        }
    auto material = [&](std::uint32_t t) { return T(t).meta & world::kModelTriMaterialMask; };
    auto compatible = [&](std::uint32_t t, std::uint32_t u) {
        if (!ok[t] || !ok[u] || material(t) != material(u)) return false;
        if (Dot(nrm[t], nrm[u]) < 1.0 - 0.5 * double(angleEps) * double(angleEps)) return false;
        return std::fabs(off[t] - off[u]) <= planeEps;
    };
    std::vector<int> assigned(nt, -1);
    for (std::uint32_t t0 = 0; t0 < nt; ++t0) {
        if (assigned[t0] >= 0 || !ok[t0]) continue;
        const world::ModelTri &tr = T(t0);
        std::vector<std::uint32_t> loop = {vid[tr.v[0]], vid[tr.v[1]], vid[tr.v[2]]};
        std::vector<std::uint32_t> mem = {t0};
        assigned[t0] = 1;
        const D3 n = nrm[t0];
        bool grew = true;
        while (grew) {
            grew = false;
            for (std::size_t i = 0; i < loop.size() && !grew; ++i) {
                const std::uint32_t a = loop[i], b = loop[(i + 1) % loop.size()];
                auto it = edges.find({std::min(a, b), std::max(a, b)});
                if (it == edges.end()) continue;
                for (std::uint32_t u : it->second) {
                    if (assigned[u] >= 0 || !compatible(t0, u)) continue;
                    std::uint32_t c = 0xffffffffu;
                    bool orient = false;
                    for (int k = 0; k < 3; ++k)   // the neighbour runs b -> a (consistent winding)
                        if (vid[T(u).v[k]] == b && vid[T(u).v[(k + 1) % 3]] == a) { c = vid[T(u).v[(k + 2) % 3]]; orient = true; }
                    if (!orient || c == a || c == b) continue;
                    if (std::find(loop.begin(), loop.end(), c) != loop.end()) continue;   // keep the loop simple
                    const std::uint32_t pa = loop[(i + loop.size() - 1) % loop.size()], nb = loop[(i + 2) % loop.size()];
                    auto turn = [&](std::uint32_t p, std::uint32_t q, std::uint32_t r) { return Dot(Cross(Sub(pos[q], pos[p]), Sub(pos[r], pos[q])), n); };
                    const double tol = -1e-9;   // convexity after inserting c between a and b
                    if (turn(pa, a, c) < tol || turn(a, c, b) < tol || turn(c, b, nb) < tol) continue;
                    loop.insert(loop.begin() + static_cast<long>(i) + 1, c);
                    assigned[u] = 1;
                    mem.push_back(u);
                    grew = true;
                    break;
                }
            }
        }
        if (mem.size() < 2) { assigned[t0] = -2; continue; }   // alone: stays a triangle
        std::vector<std::uint32_t> simp;   // drop collinear vertices
        for (std::size_t i = 0; i < loop.size(); ++i) {
            const D3 p = pos[loop[(i + loop.size() - 1) % loop.size()]], q = pos[loop[i]], r = pos[loop[(i + 1) % loop.size()]];
            const D3 cr = Cross(Sub(q, p), Sub(r, q));
            if (std::fabs(Dot(cr, n)) > 1e-9 * Len(Sub(q, p)) * Len(Sub(r, q)) + 1e-14) simp.push_back(loop[i]);
        }
        Patch pt{};
        pt.firstLoop = static_cast<std::uint32_t>(s->loop.size() / 3);
        pt.loopCount = static_cast<std::uint32_t>(simp.size());
        for (std::uint32_t id : simp) {
            const world::Vec3f &v = mv[posLocal[id]];
            s->loop.push_back(v.x); s->loop.push_back(v.y); s->loop.push_back(v.z);
        }
        for (std::size_t i = 0; i < simp.size(); ++i) {   // edge planes in the patch plane (double, rounded once)
            const D3 a = pos[simp[i]], b = pos[simp[(i + 1) % simp.size()]];
            D3 u = Cross(Sub(b, a), n);
            const double ul = Len(u);
            if (ul > 0) u = {u.x / ul, u.y / ul, u.z / ul};
            s->edge.push_back(static_cast<float>(u.x)); s->edge.push_back(static_cast<float>(u.y)); s->edge.push_back(static_cast<float>(u.z));
            s->edge.push_back(static_cast<float>(Dot(u, a)));
        }
        pt.firstMember = static_cast<std::uint32_t>(s->members.size());
        pt.memberCount = static_cast<std::uint32_t>(mem.size());
        std::sort(mem.begin(), mem.end());
        std::uint32_t rep = mem[0];
        double tau = 0.0, alpha = 0.0;
        for (std::uint32_t u : mem) {
            s->members.push_back(m.firstTri + u);
            if (area[u] > area[rep]) rep = u;
            for (int k = 0; k < 3; ++k) tau = std::max(tau, std::fabs(Dot(n, V(T(u).v[k])) - off[t0]));
            alpha = std::max(alpha, Len(Sub(nrm[u], n)));
        }
        pt.rep = m.firstTri + rep;
        pt.tau = static_cast<float>(tau);
        pt.alpha = static_cast<float>(alpha);
        const std::uint32_t id = static_cast<std::uint32_t>(s->patches.size());
        for (std::uint32_t u : mem) s->patchOf[m.firstTri + u] = id;
        s->patches.push_back(pt);
        s->trianglesInPatches += mem.size();
    }
}

}  // namespace

void Build(const world::WorldPackView &view, float planeEps, float angleEps, Set *out) {
    *out = Set{};
    out->patchOf.assign(view.modelTris.size(), kNoPatch);
    for (std::uint32_t mi = 0; mi < view.models.size(); ++mi) BuildModel(view, mi, planeEps, angleEps, out);
}

}  // namespace tw::patch
