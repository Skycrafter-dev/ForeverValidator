#include "tw_blocks.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

namespace tw {

std::uint64_t g_cntTri = 0, g_cntPlaneReject = 0, g_cntFast = 0, g_cntFull = 0, g_cntFullVerts = 0;

namespace {

std::uint64_t HashSurface(const SourceScene &src, const SourceSurface &s) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void *p, std::size_t n) {
        const unsigned char *b = static_cast<const unsigned char *>(p);
        for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    mix(&s.vertexCount, 4); mix(&s.triCount, 4);
    mix(&src.vertices[3 * s.firstVertex], 12 * s.vertexCount);
    mix(&src.tris[s.firstTri], sizeof(SourceTri) * s.triCount);
    return h;
}

bool SameSurface(const SourceScene &src, const SourceSurface &a, const SourceSurface &b) {
    return a.vertexCount == b.vertexCount && a.triCount == b.triCount &&
           std::memcmp(&src.vertices[3 * a.firstVertex], &src.vertices[3 * b.firstVertex], 12 * a.vertexCount) == 0 &&
           std::memcmp(&src.tris[a.firstTri], &src.tris[b.firstTri], sizeof(SourceTri) * a.triCount) == 0;
}

struct D3 { double x, y, z; };
D3 Sub(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
D3 Cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
double Dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double Len(D3 a) { return std::sqrt(Dot(a, a)); }

void BuildModel(const SourceScene &src, const SourceSurface &s, bool mergePatches, float planeEps, float angleEps, Model *m,
                std::size_t *merged, std::size_t *inPatches) {
    const std::uint32_t nt = s.triCount;
    m->sourceTriCount = nt;
    auto V = [&](std::uint32_t local) -> D3 {
        const float *p = &src.vertices[3 * (s.firstVertex + local)];
        return {p[0], p[1], p[2]};
    };
    // canonical position ids (duplicate vertices share an id)
    std::map<std::tuple<long long, long long, long long>, std::uint32_t> posId;
    std::vector<std::uint32_t> vid(s.vertexCount);
    std::vector<D3> pos;
    for (std::uint32_t i = 0; i < s.vertexCount; ++i) {
        const D3 p = V(i);
        auto key = std::make_tuple(std::llround(p.x * 1e5), std::llround(p.y * 1e5), std::llround(p.z * 1e5));
        auto it = posId.find(key);
        if (it == posId.end()) { it = posId.emplace(key, static_cast<std::uint32_t>(pos.size())).first; pos.push_back(p); }
        vid[i] = it->second;
    }
    std::vector<D3> nrm(nt);
    std::vector<double> off(nt);
    std::vector<bool> ok(nt);
    for (std::uint32_t t = 0; t < nt; ++t) {
        const SourceTri &tr = src.tris[s.firstTri + t];
        const D3 a = V(tr.v[0]), b = V(tr.v[1]), c = V(tr.v[2]);
        const D3 n = Cross(Sub(b, a), Sub(c, a));
        const double l = Len(n);
        ok[t] = l > 1e-10;
        nrm[t] = ok[t] ? D3{n.x / l, n.y / l, n.z / l} : D3{0, 1, 0};
        off[t] = Dot(nrm[t], a);
    }
    // edges keyed by canonical position ids
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<std::uint32_t>> edges;
    for (std::uint32_t t = 0; t < nt; ++t) {
        const SourceTri &tr = src.tris[s.firstTri + t];
        for (int k = 0; k < 3; ++k) {
            std::uint32_t a = vid[tr.v[k]], b = vid[tr.v[(k + 1) % 3]];
            edges[{std::min(a, b), std::max(a, b)}].push_back(t);
        }
    }
    auto compatible = [&](std::uint32_t t, std::uint32_t u) {
        if (!ok[t] || !ok[u]) return false;
        if (src.tris[s.firstTri + t].material != src.tris[s.firstTri + u].material) return false;
        if (Dot(nrm[t], nrm[u]) < 1.0 - 0.5 * double(angleEps) * double(angleEps)) return false;
        return std::fabs(off[t] - off[u]) <= planeEps;
    };
    std::vector<int> assigned(nt, -1);
    for (std::uint32_t t0 = 0; t0 < nt; ++t0) {
        if (assigned[t0] >= 0) continue;
        const SourceTri &tr = src.tris[s.firstTri + t0];
        std::vector<std::uint32_t> loop = {vid[tr.v[0]], vid[tr.v[1]], vid[tr.v[2]]};
        std::uint32_t count = 1;
        assigned[t0] = static_cast<int>(m->elements.size());
        const D3 n = nrm[t0];
        if (mergePatches && ok[t0]) {
            bool grew = true;
            while (grew) {
                grew = false;
                for (std::size_t i = 0; i < loop.size() && !grew; ++i) {
                    const std::uint32_t a = loop[i], b = loop[(i + 1) % loop.size()];
                    auto it = edges.find({std::min(a, b), std::max(a, b)});
                    if (it == edges.end()) continue;
                    for (std::uint32_t u : it->second) {
                        if (assigned[u] >= 0 || !compatible(t0, u)) continue;
                        const SourceTri &tu = src.tris[s.firstTri + u];
                        std::uint32_t c = 0xffffffffu; bool orient = false;
                        for (int k = 0; k < 3; ++k) {  // neighbour must run b -> a (consistent winding)
                            if (vid[tu.v[k]] == b && vid[tu.v[(k + 1) % 3]] == a) { c = vid[tu.v[(k + 2) % 3]]; orient = true; }
                        }
                        if (!orient || c == a || c == b) continue;
                        // convexity after inserting c between a and b
                        const std::uint32_t pa = loop[(i + loop.size() - 1) % loop.size()], nb = loop[(i + 2) % loop.size()];
                        auto turn = [&](std::uint32_t p, std::uint32_t q, std::uint32_t r) {
                            return Dot(Cross(Sub(pos[q], pos[p]), Sub(pos[r], pos[q])), n);
                        };
                        const double tol = -1e-9;
                        if (turn(pa, a, c) < tol || turn(a, c, b) < tol || turn(c, b, nb) < tol) continue;
                        loop.insert(loop.begin() + static_cast<long>(i) + 1, c);
                        assigned[u] = assigned[t0];
                        ++count; grew = true; break;
                    }
                }
            }
        }
        Element e{};
        e.material = static_cast<std::uint8_t>(src.tris[s.firstTri + t0].material);
        e.sourceTris = count;
        e.first = static_cast<std::uint32_t>(m->loop.size() / 3);
        if (count == 1) {
            e.isTriangle = 1;
            for (int k = 0; k < 3; ++k) {  // stored order, exact source floats
                const float *p = &src.vertices[3 * (s.firstVertex + tr.v[k])];
                m->loop.insert(m->loop.end(), p, p + 3);
            }
            e.count = 3;
        } else {
            // drop collinear vertices
            std::vector<std::uint32_t> simp;
            for (std::size_t i = 0; i < loop.size(); ++i) {
                const D3 p = pos[loop[(i + loop.size() - 1) % loop.size()]], q = pos[loop[i]], r = pos[loop[(i + 1) % loop.size()]];
                const D3 cr = Cross(Sub(q, p), Sub(r, q));
                if (std::fabs(Dot(cr, n)) > 1e-9 * Len(Sub(q, p)) * Len(Sub(r, q)) + 1e-14) simp.push_back(loop[i]);
            }
            for (std::uint32_t id : simp) {
                // exact source float of a vertex at this position
                m->loop.push_back(static_cast<float>(pos[id].x)); m->loop.push_back(static_cast<float>(pos[id].y));
                m->loop.push_back(static_cast<float>(pos[id].z));
            }
            e.count = static_cast<std::uint16_t>(simp.size());
            ++*merged; *inPatches += count;
        }
        for (int k = 0; k < 3; ++k) e.n[k] = static_cast<float>(k == 0 ? n.x : k == 1 ? n.y : n.z);
        e.lo[0] = e.lo[1] = e.lo[2] = 1e30f; e.hi[0] = e.hi[1] = e.hi[2] = -1e30f;
        for (std::uint32_t k = 0; k < e.count; ++k) {
            const float *p = &m->loop[3 * (e.first + k)];
            for (int a = 0; a < 3; ++a) { e.lo[a] = std::fmin(e.lo[a], p[a]); e.hi[a] = std::fmax(e.hi[a], p[a]); }
            const float *q = &m->loop[3 * (e.first + (k + 1) % e.count)];
            const D3 ed = {double(q[0]) - p[0], double(q[1]) - p[1], double(q[2]) - p[2]};
            D3 u = Cross(ed, n);
            const double ul = Len(u);
            if (ul > 0) u = {u.x / ul, u.y / ul, u.z / ul};
            m->edgeU.push_back(static_cast<float>(u.x)); m->edgeU.push_back(static_cast<float>(u.y)); m->edgeU.push_back(static_cast<float>(u.z));
        }
        m->box = Union(m->box, Box3{{e.lo[0], e.lo[1], e.lo[2]}, {e.hi[0], e.hi[1], e.hi[2]}});
        m->elements.push_back(e);
    }
}

// payload of a feature or face contact in unit space -> frame of the ShapeMap
void Emit(const ShapeMap &s, int kind, const float Np[3], const float Sp[3], const float Pp[3], const float n[3], Contact *out) {
    out->kind = kind;
    float g[3];
    for (int i = 0; i < 3; ++i) {
        out->P[i] = ((s.M[i][0] * Pp[0] + s.M[i][1] * Pp[1]) + s.M[i][2] * Pp[2]) + s.t[i];
        out->Sep[i] = (s.M[i][0] * Sp[0] + s.M[i][1] * Sp[1]) + s.M[i][2] * Sp[2];
        g[i] = (s.N[i][0] * Np[0] + s.N[i][1] * Np[1]) + s.N[i][2] * Np[2];
        out->K[i] = n[i];
    }
    const float g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    const float ig = g2 > 1e-10f ? 1.0f / std::sqrt(g2) : 1.0f;
    for (int i = 0; i < 3; ++i) out->N[i] = g[i] * ig;
}

}  // namespace

bool BlockWorld::Build(const SourceScene &src, bool mergePatches, float planeEps, float angleEps) {
    models.clear(); instances.clear(); cells.clear();
    nonPermutation = 0; sourceTris = 0; elements = 0; mergedPatches = 0; trianglesInPatches = 0;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> byHash;  // hash -> surface index of model prototypes
    std::vector<std::uint32_t> modelOfProto;
    std::unordered_map<std::uint32_t, std::uint32_t> protoModel;
    for (std::uint32_t si = 0; si < src.surfaces.size(); ++si) {
        const SourceSurface &s = src.surfaces[si];
        sourceTris += s.triCount;
        const std::uint64_t h = HashSurface(src, s);
        std::uint32_t model = 0xffffffffu;
        for (std::uint32_t proto : byHash[h])
            if (SameSurface(src, src.surfaces[proto], s)) { model = protoModel[proto]; break; }
        if (model == 0xffffffffu) {
            model = static_cast<std::uint32_t>(models.size());
            models.emplace_back();
            BuildModel(src, s, mergePatches, planeEps, angleEps, &models.back(), &mergedPatches, &trianglesInPatches);
            byHash[h].push_back(si);
            protoModel[si] = model;
        }
        Instance in{};
        in.model = model;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) in.R[i][j] = s.iso[3 * j + i];
        for (int i = 0; i < 3; ++i) in.T[i] = s.iso[9 + i];
        bool perm = true;
        for (int j = 0; j < 3; ++j) {
            int ones = 0;
            for (int i = 0; i < 3; ++i) {
                const float v = in.R[i][j];
                if (v == 1.0f || v == -1.0f) ++ones; else if (v != 0.0f) perm = false;
            }
            perm = perm && ones == 1;
        }
        in.permutation = perm ? 1 : 0;
        if (!perm) ++nonPermutation;
        in.trigger = static_cast<std::uint8_t>(s.trigger);
        // world AABB from the model box
        const Box3 &mb = models[model].box;
        for (int i = 0; i < 3; ++i) {
            const float c = 0.5f * (mb.lo[0] + mb.hi[0]) * in.R[i][0] + 0.5f * (mb.lo[1] + mb.hi[1]) * in.R[i][1] +
                            0.5f * (mb.lo[2] + mb.hi[2]) * in.R[i][2] + in.T[i];
            const float h = 0.5f * (mb.hi[0] - mb.lo[0]) * std::fabs(in.R[i][0]) + 0.5f * (mb.hi[1] - mb.lo[1]) * std::fabs(in.R[i][1]) +
                            0.5f * (mb.hi[2] - mb.lo[2]) * std::fabs(in.R[i][2]);
            in.box.lo[i] = c - h * 1.000001f - 1e-4f; in.box.hi[i] = c + h * 1.000001f + 1e-4f;
        }
        const std::uint32_t id = static_cast<std::uint32_t>(instances.size());
        instances.push_back(in);
        const int x0 = static_cast<int>(std::floor(in.box.lo[0] / 32.0f)), x1 = static_cast<int>(std::floor(in.box.hi[0] / 32.0f));
        const int y0 = static_cast<int>(std::floor(in.box.lo[1] / 8.0f)), y1 = static_cast<int>(std::floor(in.box.hi[1] / 8.0f));
        const int z0 = static_cast<int>(std::floor(in.box.lo[2] / 32.0f)), z1 = static_cast<int>(std::floor(in.box.hi[2] / 32.0f));
        if (static_cast<long long>(x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) > 4096) {  // huge terrain/decoration: bucket at one key
            cells[CellKey(0x7ffff, 0x7ffff, 0x7ffff)].push_back(id);
            continue;
        }
        for (int x = x0; x <= x1; ++x) for (int y = y0; y <= y1; ++y) for (int z = z0; z <= z1; ++z) cells[CellKey(x, y, z)].push_back(id);
    }
    for (Model &m : models) { elements += m.elements.size(); m.BuildGrid(); }
    BuildCellIndex();
    return true;
}

void BlockWorld::BuildCellIndex() {
    struct KR { std::uint64_t key; CellRef r; };
    std::vector<KR> all;
    instElemBase.resize(instances.size());
    totalInstElems = 0;
    hugeRefs.clear();
    for (std::uint32_t ii = 0; ii < instances.size(); ++ii) {
        const Instance &in = instances[ii];
        const Model &m = models[in.model];
        instElemBase[ii] = totalInstElems;
        totalInstElems += static_cast<std::uint32_t>(m.elements.size());
        for (std::uint32_t ei = 0; ei < m.elements.size(); ++ei) {
            const Element &e = m.elements[ei];
            CellRef r; r.inst = ii; r.elem = ei;
            for (int i = 0; i < 3; ++i) {
                const float c = 0.5f * (e.lo[0] + e.hi[0]) * in.R[i][0] + 0.5f * (e.lo[1] + e.hi[1]) * in.R[i][1] + 0.5f * (e.lo[2] + e.hi[2]) * in.R[i][2] + in.T[i];
                const float h = 0.5f * (e.hi[0] - e.lo[0]) * std::fabs(in.R[i][0]) + 0.5f * (e.hi[1] - e.lo[1]) * std::fabs(in.R[i][1]) +
                                0.5f * (e.hi[2] - e.lo[2]) * std::fabs(in.R[i][2]);
                const float slack = 1e-4f + 2e-7f * std::fabs(c);
                r.lo[i] = c - h - slack; r.hi[i] = c + h + slack;
            }
            const int x0 = static_cast<int>(std::floor(r.lo[0] / kCellX)), x1 = static_cast<int>(std::floor(r.hi[0] / kCellX));
            const int y0 = static_cast<int>(std::floor(r.lo[1] / kCellY)), y1 = static_cast<int>(std::floor(r.hi[1] / kCellY));
            const int z0 = static_cast<int>(std::floor(r.lo[2] / kCellZ)), z1 = static_cast<int>(std::floor(r.hi[2] / kCellZ));
            if (static_cast<long long>(x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) > 64) {
                // coarse level: 256 x 128 x 256 m cells, keyed with a level tag
                const int X0 = static_cast<int>(std::floor(r.lo[0] / 256.0f)), X1 = static_cast<int>(std::floor(r.hi[0] / 256.0f));
                const int Y0 = static_cast<int>(std::floor(r.lo[1] / 128.0f)), Y1 = static_cast<int>(std::floor(r.hi[1] / 128.0f));
                const int Z0 = static_cast<int>(std::floor(r.lo[2] / 256.0f)), Z1 = static_cast<int>(std::floor(r.hi[2] / 256.0f));
                if (static_cast<long long>(X1 - X0 + 1) * (Y1 - Y0 + 1) * (Z1 - Z0 + 1) > 64) { hugeRefs.push_back(r); continue; }
                for (int x = X0; x <= X1; ++x) for (int y = Y0; y <= Y1; ++y) for (int z = Z0; z <= Z1; ++z)
                    all.push_back({CellKey(x, y, z) | (1ull << 63), r});
                continue;
            }
            for (int x = x0; x <= x1; ++x) for (int y = y0; y <= y1; ++y) for (int z = z0; z <= z1; ++z) all.push_back({CellKey(x, y, z), r});
        }
    }
    std::sort(all.begin(), all.end(), [](const KR &a, const KR &b) { return a.key < b.key; });
    refs.clear(); cellIndex.clear(); cellBegin.clear(); cellEnd.clear();
    for (std::size_t k = 0; k < all.size();) {
        std::size_t j = k;
        cellIndex[all[k].key] = static_cast<std::uint32_t>(cellBegin.size());
        cellBegin.push_back(static_cast<std::uint32_t>(refs.size()));
        while (j < all.size() && all[j].key == all[k].key) refs.push_back(all[j++].r);
        cellEnd.push_back(static_cast<std::uint32_t>(refs.size()));
        k = j;
    }
}

std::size_t BlockWorld::Bytes() const {
    std::size_t b = instances.size() * sizeof(Instance);
    for (const Model &m : models) b += m.elements.size() * sizeof(Element) + (m.loop.size() + m.edgeU.size() + m.gridStart.size() + m.gridIds.size()) * 4;
    for (const auto &kv : cells) b += 16 + kv.second.size() * 4;
    b += (refs.size() + hugeRefs.size()) * sizeof(CellRef) + cellIndex.size() * 24 + cellBegin.size() * 8;
    return b;
}

int ElementTest(const ShapeMap &m, const Model &model, const Element &e, Contact *out, bool *fast) {
    return LoopTest(m, &model.loop[3 * e.first], &model.edgeU[3 * e.first], e.count, e.n, e.isTriangle != 0, out, fast);
}

void Model::BuildGrid() {
    const float ext = std::fmax(box.hi[0] - box.lo[0], std::fmax(box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]));
    float cell = 4.0f;
    while (ext / cell > 48.0f) cell *= 2.0f;
    gridInv = 1.0f / cell;
    for (int a = 0; a < 3; ++a) {
        gridOrg[a] = box.lo[a];
        gridDim[a] = std::max(1, static_cast<int>((box.hi[a] - box.lo[a]) * gridInv) + 1);
    }
    const std::size_t ncell = static_cast<std::size_t>(gridDim[0]) * gridDim[1] * gridDim[2];
    std::vector<std::vector<std::uint32_t>> lists(ncell);
    for (std::uint32_t ei = 0; ei < elements.size(); ++ei) {
        const Element &e = elements[ei];
        int lo[3], hi[3];
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(gridDim[a] - 1, std::max(0, static_cast<int>((e.lo[a] - gridOrg[a]) * gridInv)));
            hi[a] = std::min(gridDim[a] - 1, std::max(0, static_cast<int>((e.hi[a] - gridOrg[a]) * gridInv)));
        }
        for (int x = lo[0]; x <= hi[0]; ++x) for (int y = lo[1]; y <= hi[1]; ++y) for (int z = lo[2]; z <= hi[2]; ++z)
            lists[(static_cast<std::size_t>(x) * gridDim[1] + y) * gridDim[2] + z].push_back(ei);
    }
    gridStart.assign(ncell + 1, 0);
    gridIds.clear();
    for (std::size_t k = 0; k < ncell; ++k) { gridStart[k] = static_cast<std::uint32_t>(gridIds.size()); gridIds.insert(gridIds.end(), lists[k].begin(), lists[k].end()); }
    gridStart[ncell] = static_cast<std::uint32_t>(gridIds.size());
}

int LoopTest(const ShapeMap &m, const float *L, const float *edgeU, std::uint32_t count, const float n[3], bool isTriangle,
             Contact *out, bool *fast) {
    *fast = false;
    if (isTriangle) {
        ++g_cntTri;
        Tri t{};
        for (int k = 0; k < 3; ++k) for (int a = 0; a < 3; ++a) t.v[k][a] = L[3 * k + a];
        return ExactTest(m, t, out);
    }
    // plane reach: 0 <= sd <= h(n) (squared compare first: no sqrt on rejection)
    const float sd = n[0] * (m.t[0] - L[0]) + n[1] * (m.t[1] - L[1]) + n[2] * (m.t[2] - L[2]);
    if (!(sd >= 0.0f)) { ++g_cntPlaneReject; return kNone; }
    float Mn[3];
    for (int j = 0; j < 3; ++j) Mn[j] = (m.M[0][j] * n[0] + m.M[1][j] * n[1]) + m.M[2][j] * n[2];
    const float h2 = Mn[0] * Mn[0] + Mn[1] * Mn[1] + Mn[2] * Mn[2];
    if (sd * sd > h2 * 1.000001f) { ++g_cntPlaneReject; return kNone; }
    const float h = std::sqrt(h2);
    if (sd > h) return kNone;
    const float ih = 1.0f / h;
    float np[3] = {Mn[0] * ih, Mn[1] * ih, Mn[2] * ih};  // unit-space normal (frame independent)
    if (sd > 0.0f) {
        // FACE fast path: projected point strictly inside the convex loop
        const float k = sd / h2;
        float MMn[3];
        for (int i = 0; i < 3; ++i) MMn[i] = (m.M[i][0] * Mn[0] + m.M[i][1] * Mn[1]) + m.M[i][2] * Mn[2];
        const float P[3] = {m.t[0] - k * MMn[0], m.t[1] - k * MMn[1], m.t[2] - k * MMn[2]};
        float o = -1e30f;
        for (std::uint32_t i = 0; i < count; ++i) {
            const float *v = &L[3 * i], *u = &edgeU[3 * i];
            const float s = (P[0] - v[0]) * u[0] + (P[1] - v[1]) * u[1] + (P[2] - v[2]) * u[2];
            o = s > o ? s : o;
        }
        if (o < -1e-4f) {
            *fast = true; ++g_cntFast;
            out->kind = kFace;
            const float ks = (sd - h) / h2;
            for (int i = 0; i < 3; ++i) { out->P[i] = P[i]; out->N[i] = n[i]; out->Sep[i] = ks * MMn[i]; out->K[i] = np[i]; }
            return kFace;
        }
    }
    // full unit-space test on the loop (FV's first-violated-edge rule, loop order)
    const float d = sd * ih;
    float v[64][3];
    const std::uint32_t cnt = count < 64 ? count : 64;
    ++g_cntFull; g_cntFullVerts += cnt;
    for (std::uint32_t k = 0; k < cnt; ++k) UnitPoint(m, &L[3 * k], v[k]);
    const float hh = std::sqrt(std::fmax(1.0f - d * d, 0.0f));
    const float p[3] = {-d * np[0], -d * np[1], -d * np[2]};
    for (std::uint32_t i = 0; i < cnt; ++i) {
        const float *a = v[i], *b = v[(i + 1) % cnt];
        float ed[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const float el2 = ed[0] * ed[0] + ed[1] * ed[1] + ed[2] * ed[2];
        if (el2 > 1e-10f) { const float ie = 1.0f / std::sqrt(el2); ed[0] *= ie; ed[1] *= ie; ed[2] *= ie; }
        const float mm[3] = {ed[1] * np[2] - ed[2] * np[1], ed[2] * np[0] - ed[0] * np[2], ed[0] * np[1] - ed[1] * np[0]};
        const float r[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
        const float sE = r[0] * mm[0] + r[1] * mm[1] + r[2] * mm[2];
        if (sE > hh) return kNone;
        if (sE > 0.0f) {
            int kind; float f[3];
            const float aa = r[0] * ed[0] + r[1] * ed[1] + r[2] * ed[2];
            const float bb = (p[0] - b[0]) * ed[0] + (p[1] - b[1]) * ed[1] + (p[2] - b[2]) * ed[2];
            if (aa < 0.0f) { kind = kVertexA; f[0] = a[0]; f[1] = a[1]; f[2] = a[2]; }
            else if (!(bb > 0.0f)) { kind = kEdge; f[0] = p[0] - sE * mm[0]; f[1] = p[1] - sE * mm[1]; f[2] = p[2] - sE * mm[2]; }
            else { kind = kVertexB; f[0] = b[0]; f[1] = b[1]; f[2] = b[2]; }
            const float w[3] = {-f[0], -f[1], -f[2]};
            const float w2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
            const float wl = std::sqrt(w2);
            float den = wl;
            if (kind == kVertexA) { if (!(w2 > 1e-10f) || w2 > 1.0f) return kNone; }
            else if (kind == kEdge) { if (!(w2 > 1e-5f)) return kNone; }
            else { if (!(wl > 1e-10f) || wl > 1.0f) return kNone; den = std::sqrt(wl); }
            const float idn = 1.0f / den, sep = d * (den - 1.0f) * idn;
            const float Np[3] = {w[0] * idn, w[1] * idn, w[2] * idn}, Sp[3] = {np[0] * sep, np[1] * sep, np[2] * sep};
            Emit(m, kind, Np, Sp, f, np, out);
            return kind;
        }
    }
    if (!(d > 0.0f)) return kNone;
    const float Sp[3] = {np[0] * (d - 1.0f), np[1] * (d - 1.0f), np[2] * (d - 1.0f)};
    Emit(m, kFace, np, Sp, p, np, out);
    return kFace;
}

void MergeShape(std::vector<MergedContact> &in, std::vector<MergedContact> *out) {
    out->clear();
    for (const MergedContact &c : in) if (c.primary) out->push_back(c);
    const std::size_t np = out->size();
    for (const MergedContact &f : in) {
        if (f.primary) continue;
        bool drop = false;
        for (std::size_t g = 0; g < np && !drop; ++g) {
            const Contact &p = (*out)[g].c;
            bool eq = true;
            for (int k = 0; k < 3; ++k) eq = eq && std::fabs(f.c.K[k] - p.K[k]) <= 1e-5f * std::fabs(p.K[k]);
            const float dn = f.c.N[0] * p.N[0] + f.c.N[1] * p.N[1] + f.c.N[2] * p.N[2];
            drop = eq || dn > 0.8660254f;
        }
        if (!drop) out->push_back(f);
    }
}

}  // namespace tw
