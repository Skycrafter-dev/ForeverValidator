#include "tw_voxels.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>

#include "tw_exact.h"

namespace tw {

namespace {

struct D3 { double x, y, z; };
D3 V(const float *p) { return {p[0], p[1], p[2]}; }
D3 Sub(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
double Dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
D3 Cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
D3 Scale(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
D3 Add(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

double SegDist(D3 p, D3 a, D3 b) {
    const D3 ab = Sub(b, a);
    const double l2 = Dot(ab, ab);
    double t = l2 > 0 ? Dot(Sub(p, a), ab) / l2 : 0.0;
    t = std::min(1.0, std::max(0.0, t));
    const D3 d = Sub(p, Add(a, Scale(ab, t)));
    return std::sqrt(Dot(d, d));
}

using Key3 = std::tuple<long long, long long, long long>;
Key3 Key(const float *p) { return std::make_tuple(std::llround(p[0] * 1e5), std::llround(p[1] * 1e5), std::llround(p[2] * 1e5)); }

struct GEdge { D3 a, b; std::uint32_t seg; std::int32_t owner; };

struct Group {
    D3 n{}, p0{};
    std::uint8_t material = 0;
    std::vector<std::uint16_t> members;          // element indices
    std::vector<GEdge> edges;                    // shared edges once (owner -1), boundary edges with their member index
    bool duplicates = false;                     // identical triangles in the group: never "covered" (FV emits one FACE each)
};

struct ListInfo {
    std::vector<Group> groups;  // coplanar same-material groups, largest first (groups[0] is the main group)
    std::vector<std::uint16_t> rest;  // everything outside the main group
    std::int32_t id[2] = {-1, -1};    // list ids: not covered / covered
};

struct VecHash {
    std::size_t operator()(const std::vector<std::uint16_t> &v) const {
        std::uint64_t h = 1469598103934665603ull;
        for (std::uint16_t x : v) { h ^= x; h *= 1099511628211ull; }
        return static_cast<std::size_t>(h);
    }
};

}  // namespace

bool VoxWorld::Build(const BlockWorld &bw, float voxelSize, double maxVoxels) {
    auto t0 = std::chrono::steady_clock::now();
    models.assign(bw.models.size(), VoxModel{});
    std::vector<bool> triggerModel(bw.models.size(), false);
    for (const Instance &in : bw.instances) if (in.trigger) triggerModel[in.model] = true;
    lists = items = edgesTotal = voxels = bricks = uniqueBricks = overflowGrids = coarsenedGrids = 0;
    constexpr int B = VoxGrid::kB;
    for (std::size_t mi = 0; mi < bw.models.size(); ++mi) {
        const Model &m = bw.models[mi];
        const std::size_t ne = m.elements.size();
        if (ne > 65535) { overflowGrids += VoxModel::kClasses; continue; }
        std::vector<D3> nrm(ne), v0(ne);
        std::vector<Tri> tri(ne);
        for (std::size_t e = 0; e < ne; ++e) {
            nrm[e] = {m.elements[e].n[0], m.elements[e].n[1], m.elements[e].n[2]}; v0[e] = V(&m.loop[3 * m.elements[e].first]);
            for (int k = 0; k < 3; ++k) for (int a = 0; a < 3; ++a) tri[e].v[k][a] = m.loop[3 * (m.elements[e].first + k) + a];
        }
        std::map<std::pair<Key3, Key3>, std::uint32_t> segIds;
        const double mext = std::max({m.box.hi[0] - m.box.lo[0], m.box.hi[1] - m.box.lo[1], m.box.hi[2] - m.box.lo[2]});
        for (int c = 0; c < VoxModel::kClasses; ++c) {
            if (!buildClass[c]) continue;
            VoxGrid &g = models[mi].grid[c];
            const double cap = mext > hugeExtent ? std::min(maxVoxels, hugeMaxVoxels) : maxVoxels;
            const double R = classRadius[c] + margin + 1e-3;
            const Box3 box = Inflate(m.box, static_cast<float>(R));
            const double ex = box.hi[0] - box.lo[0], ey = box.hi[1] - box.lo[1], ez = box.hi[2] - box.lo[2];
            double size0 = voxelSize;
            if (ex * ey * ez / (size0 * size0 * size0) > cap) size0 = std::cbrt(ex * ey * ez / cap);
            for (double mul = 1.0;; mul *= 1.5) {  // coarser voxels until the grid's distinct lists fit u16 ids
            g = VoxGrid{};
            const double size = size0 * mul;
            g.size = static_cast<float>(size); g.inv = static_cast<float>(1.0 / size);
            for (int a = 0; a < 3; ++a) g.org[a] = box.lo[a];
            const double ext[3] = {ex, ey, ez};
            for (int a = 0; a < 3; ++a) {
                g.bdim[a] = std::max(1, static_cast<int>(std::ceil(ext[a] / (size * B))));
                g.dim[a] = g.bdim[a] * B;
            }
            const std::size_t nb = static_cast<std::size_t>(g.bdim[0]) * g.bdim[1] * g.bdim[2];
            const double bsize = size * B, halfDiag = 0.5 * std::sqrt(3.0) * size, bHalfDiag = 0.5 * std::sqrt(3.0) * bsize;
            // 1. elements -> bricks
            std::unordered_map<std::size_t, std::vector<std::uint16_t>> brickElems;
            for (std::size_t e = 0; e < ne; ++e) {
                const Element &el = m.elements[e];
                int lo[3], hi[3];
                for (int a = 0; a < 3; ++a) {
                    lo[a] = std::max(0, static_cast<int>(std::floor((el.lo[a] - R - g.org[a]) / bsize)));
                    hi[a] = std::min(g.bdim[a] - 1, static_cast<int>(std::floor((el.hi[a] + R - g.org[a]) / bsize)));
                }
                for (int x = lo[0]; x <= hi[0]; ++x) for (int y = lo[1]; y <= hi[1]; ++y) for (int z = lo[2]; z <= hi[2]; ++z) {
                    const D3 bc = {g.org[0] + (x + 0.5) * bsize, g.org[1] + (y + 0.5) * bsize, g.org[2] + (z + 0.5) * bsize};
                    const double sdb = Dot(nrm[e], Sub(bc, v0[e]));
                    if (sdb > R + bHalfDiag + 1e-3 || (frontCull && Dot(nrm[e], nrm[e]) > 0.5 && sdb < -bHalfDiag - margin - 1e-3)) continue;
                    brickElems[(static_cast<std::size_t>(x) * g.bdim[1] + y) * g.bdim[2] + z].push_back(static_cast<std::uint16_t>(e));
                }
            }
            // 2. per non-empty brick: voxel lists, classification, dedupe
            g.brick.assign(nb, 0);
            g.cells.assign(B * B * B, 0);  // brick 0: all EMPTY
            g.lists.assign(1, VoxList{});
            g.items.clear(); g.edges.clear();
            std::unordered_map<std::vector<std::uint16_t>, ListInfo, VecHash> infos;
            std::unordered_map<std::string, std::uint32_t> brickIds;
            brickIds[std::string(reinterpret_cast<const char *>(g.cells.data()), B * B * B * 2)] = 0;
            bool overflow = false;
            std::vector<std::uint16_t> L, content(B * B * B);
            for (auto &kv : brickElems) {
                const std::size_t bi = kv.first;
                const int bx = static_cast<int>(bi / (static_cast<std::size_t>(g.bdim[1]) * g.bdim[2]));
                const int by = static_cast<int>((bi / g.bdim[2]) % g.bdim[1]), bz = static_cast<int>(bi % g.bdim[2]);
                bool any = false;
                for (int lx = 0; lx < B; ++lx) for (int ly = 0; ly < B; ++ly) for (int lz = 0; lz < B; ++lz) {
                    const int x = bx * B + lx, y = by * B + ly, z = bz * B + lz;
                    const double vlo[3] = {g.org[0] + x * size, g.org[1] + y * size, g.org[2] + z * size};
                    const D3 vc = {vlo[0] + 0.5 * size, vlo[1] + 0.5 * size, vlo[2] + 0.5 * size};
                    L.clear();
                    for (std::uint16_t e : kv.second) {
                        const Element &el = m.elements[e];
                        bool ov = true;
                        for (int a = 0; a < 3 && ov; ++a) ov = el.lo[a] - R <= vlo[a] + size && el.hi[a] + R >= vlo[a];
                        if (!ov) continue;
                        const double sdc = Dot(nrm[e], Sub(vc, v0[e]));
                        if (sdc > R + halfDiag + 1e-3) continue;
                        if (frontCull && Dot(nrm[e], nrm[e]) > 0.5 && sdc < -halfDiag - margin - 1e-3) continue;  // every centre within margin of the voxel is behind
                        if (distCull) {
                            const double pc[3] = {vc.x, vc.y, vc.z};
                            const double d2 = PointTriDist2(pc, tri[e]);
                            const double lim = R + halfDiag + 1e-3;
                            if (d2 == d2 && d2 > lim * lim) continue;
                        }
                        L.push_back(e);
                    }
                    std::uint16_t id = 0;
                    if (!L.empty()) {
                        auto it = infos.find(L);
                        if (it == infos.end()) {
                            ListInfo inf;
                            for (std::uint16_t k : L) {
                                if (!useGroups || triggerModel[mi] || Dot(nrm[k], nrm[k]) < 0.5) { inf.rest.push_back(k); continue; }
                                Group *gr = nullptr;
                                for (Group &q : inf.groups)
                                    if (q.material == m.elements[k].material && Dot(nrm[k], q.n) >= 1.0 - 1e-9 && std::fabs(Dot(q.n, Sub(v0[k], q.p0))) <= 1e-4) { gr = &q; break; }
                                if (!gr) { inf.groups.push_back(Group{}); gr = &inf.groups.back(); gr->n = nrm[k]; gr->p0 = v0[k]; gr->material = m.elements[k].material; }
                                gr->members.push_back(k);
                            }
                            std::stable_sort(inf.groups.begin(), inf.groups.end(), [](const Group &a, const Group &b) { return a.members.size() > b.members.size(); });
                            for (std::size_t gi = 1; gi < inf.groups.size(); ++gi) inf.rest.insert(inf.rest.end(), inf.groups[gi].members.begin(), inf.groups[gi].members.end());
                            if (inf.groups.size() > 1) inf.groups.resize(1);
                            if (!inf.groups.empty()) {
                                Group &q = inf.groups[0];
                                // directed edge counts: an edge is shared only if it is used in both directions
                                // (consistent winding); duplicate triangles leave their edges on the boundary
                                std::map<std::pair<Key3, Key3>, int> dir;
                                std::map<std::array<Key3, 3>, int> triSeen;
                                for (std::uint16_t k : q.members) {
                                    const float *t = &m.loop[3 * m.elements[k].first];
                                    std::array<Key3, 3> tk = {Key(t), Key(t + 3), Key(t + 6)};
                                    std::sort(tk.begin(), tk.end());
                                    if (++triSeen[tk] > 1) q.duplicates = true;
                                    for (int i = 0; i < 3; ++i) ++dir[{Key(t + 3 * i), Key(t + 3 * ((i + 1) % 3))}];
                                }
                                std::map<std::pair<Key3, Key3>, bool> emitted;
                                for (std::size_t mk = 0; mk < q.members.size(); ++mk) {
                                    const float *t = &m.loop[3 * m.elements[q.members[mk]].first];
                                    for (int i = 0; i < 3; ++i) {
                                        const float *pa = t + 3 * i, *pb = t + 3 * ((i + 1) % 3);
                                        auto a = Key(pa), b = Key(pb);
                                        const auto skey = std::make_pair(std::min(a, b), std::max(a, b));
                                        auto sit = segIds.find(skey);
                                        if (sit == segIds.end()) {
                                            VoxSeg sg{};
                                            for (int c2 = 0; c2 < 3; ++c2) { sg.a[c2] = pa[c2]; sg.b[c2] = pb[c2]; }
                                            sit = segIds.emplace(skey, static_cast<std::uint32_t>(models[mi].segs.size())).first;
                                            models[mi].segs.push_back(sg);
                                        }
                                        GEdge ge{V(pa), V(pb), sit->second, -1};
                                        const bool shared = dir.count({b, a}) != 0;
                                        if (!shared) { ge.owner = static_cast<std::int32_t>(mk); q.edges.push_back(ge); }
                                        else if (!emitted[skey]) { emitted[skey] = true; q.edges.push_back(ge); }  // shared: once
                                    }
                                }
                            }
                            it = infos.emplace(L, std::move(inf)).first;
                        }
                        ListInfo &inf = it->second;
                        int covered = 0;
                        if (!inf.groups.empty()) {  // covered: pc inside the union, every boundary edge out of reach
                            const Group &q = inf.groups[0];
                            const D3 pc = Sub(vc, Scale(q.n, Dot(q.n, Sub(vc, q.p0))));
                            bool inside = false, far = true;
                            for (std::uint16_t k : q.members) {
                                const float *t = &m.loop[3 * m.elements[k].first];
                                const D3 a = V(t), b = V(t + 3), c = V(t + 6);
                                if (Dot(Cross(Sub(b, a), Sub(pc, a)), q.n) >= -1e-9 && Dot(Cross(Sub(c, b), Sub(pc, b)), q.n) >= -1e-9 &&
                                    Dot(Cross(Sub(a, c), Sub(pc, c)), q.n) >= -1e-9) { inside = true; break; }
                            }
                            for (std::size_t e = 0; e < q.edges.size() && inside && far; ++e)
                                if (q.edges[e].owner >= 0 && SegDist(pc, q.edges[e].a, q.edges[e].b) < R + halfDiag + 1e-3) far = false;
                            covered = inside && far && !q.duplicates ? 1 : 0;
                        }
                        std::int32_t &slot = inf.id[covered];
                        if (slot < 0) {
                            if (g.lists.size() > 65535) { overflow = true; break; }
                            VoxList vl{};
                            vl.first = static_cast<std::uint32_t>(g.items.size()); vl.count = static_cast<std::uint16_t>(L.size());
                            if (!inf.groups.empty()) {
                                const Group &q = inf.groups[0];
                                const Element &e0 = m.elements[q.members[0]];
                                for (int a = 0; a < 3; ++a) { vl.n[a] = e0.n[a]; vl.p[a] = m.loop[3 * e0.first + a]; }
                                vl.memberCount = static_cast<std::uint16_t>(q.members.size()); vl.material = q.material; vl.covered = static_cast<std::uint8_t>(covered);
                                vl.edgeFirst = static_cast<std::uint32_t>(g.edges.size());
                                for (const GEdge &ge : q.edges) {
                                    if (covered && ge.owner >= 0) continue;  // boundary edges are out of reach
                                    g.edges.push_back(VoxEdge{ge.seg, ge.owner});
                                }
                                vl.edgeCount = static_cast<std::uint16_t>(g.edges.size() - vl.edgeFirst);
                                g.items.insert(g.items.end(), q.members.begin(), q.members.end());
                            }
                            g.items.insert(g.items.end(), inf.rest.begin(), inf.rest.end());
                            vl.cls = inf.groups.empty() ? kVoxGeneral : !inf.rest.empty() ? kVoxMixed : covered ? kVoxSingle : kVoxPlane;
                            slot = static_cast<std::int32_t>(g.lists.size());
                            g.lists.push_back(vl);
                        }
                        id = static_cast<std::uint16_t>(slot);
                        any = true;
                    }
                    content[(lx << 6) | (ly << 3) | lz] = id;
                }
                if (overflow) break;
                if (!any) continue;
                std::string key(reinterpret_cast<const char *>(content.data()), B * B * B * 2);
                auto it = brickIds.find(key);
                std::uint32_t bid;
                if (it == brickIds.end()) {
                    bid = static_cast<std::uint32_t>(g.cells.size() / (B * B * B));
                    g.cells.insert(g.cells.end(), content.begin(), content.end());
                    brickIds.emplace(std::move(key), bid);
                } else bid = it->second;
                g.brick[bi] = bid;
                ++g.nonEmptyBricks;
            }
            if (overflow) {
                if (mul < 30.0) continue;
                ++overflowGrids;
                g = VoxGrid{};
                break;
            }
            if (mul > 1.0) ++coarsenedGrids;
            lists += g.lists.size(); items += g.items.size(); edgesTotal += g.edges.size(); voxels += nb * B * B * B; bricks += g.nonEmptyBricks;
            uniqueBricks += g.cells.size() / (B * B * B) - 1;
            break;
            }
        }
    }
    buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

std::size_t VoxWorld::Bytes() const {
    std::size_t b = 0;
    for (const VoxModel &m : models) {
        b += m.segs.size() * sizeof(VoxSeg);
        for (const VoxGrid &g : m.grid) b += g.brick.size() * 4 + g.cells.size() * 2 + g.lists.size() * sizeof(VoxList) + g.items.size() * 2 + g.edges.size() * sizeof(VoxEdge);
    }
    return b;
}

}  // namespace tw
