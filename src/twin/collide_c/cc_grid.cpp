// CellGrid build (architect C, docs/twin/collision-c.md §9).
#include "cc_grid.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

namespace cc {

namespace {
inline int CellOf(float x, float org, float inv) {  // the query's formula (monotone in x)
    const float f = (x - org) * inv;
    return f < 0.0f ? -1 : static_cast<int>(f);
}
}  // namespace

bool CellGrid::Build(const tw::world::WorldPackView &v, const Options &opt) {
    using namespace tw::world;
    const std::size_t NI = v.instances.size();
    // ---- mesh-local triangle AABBs of every model (computed lazily per used model)
    std::vector<std::int32_t> modelSlot(v.models.size(), -1);
    std::vector<std::uint8_t> isGrid(NI, 0);
    std::vector<GInst> gtmp;
    std::vector<std::uint32_t> gsrc;
    // local AABB of model triangle (model m, triangle k)
    auto localBox = [&](const Model &md, std::uint32_t k, float lo[3], float hi[3]) {
        const ModelTri &mt = v.modelTris[md.firstTri + k];
        for (int a = 0; a < 3; ++a) { lo[a] = 1e30f; hi[a] = -1e30f; }
        for (int c = 0; c < 3; ++c) {
            const Vec3f &p = v.modelVerts[md.firstVert + mt.v[c]];
            const float q[3] = {p.x, p.y, p.z};
            for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], q[a]); hi[a] = std::max(hi[a], q[a]); }
        }
    };
    std::size_t mismatched = 0, outside = 0;
    for (std::size_t i = 0; i < NI; ++i) {
        const Instance &in = v.instances[i];
        if (in.triCount == 0) continue;
        bool grid = (in.flags & kInstPermExact) && !(in.flags & kInstMovingPair) && in.rotation != kRotGeneral;
        std::size_t ncells = 1;
        for (int a = 0; a < 3; ++a) {
            const int c0 = static_cast<int>(std::floor(in.lo[a] / opt.cell[a])), c1 = static_cast<int>(std::floor(in.hi[a] / opt.cell[a]));
            ncells *= static_cast<std::size_t>(c1 - c0 + 1);
        }
        grid = grid && ncells <= static_cast<std::size_t>(opt.maxCellRefs);
        GInst g{};
        if (grid) {
            for (int a = 0; a < 3; ++a) { g.lo[a] = in.lo[a]; g.hi[a] = in.hi[a]; g.t[a] = in.iso[9 + a]; }
            for (int a = 0; a < 3 && grid; ++a) {
                int found = -1;
                for (int j = 0; j < 3; ++j) {
                    const float e = in.iso[3 * j + a];
                    if (e == 1.0f || e == -1.0f) { found = j; if (e < 0.0f) g.neg |= static_cast<std::uint8_t>(1u << a); }
                    else if (e != 0.0f) grid = false;
                }
                if (found < 0) grid = false;
                g.p[a] = static_cast<std::uint8_t>(found < 0 ? 0 : found);
            }
            g.model = in.model;
            g.firstTri = in.firstTri;
        }
        if (grid) {  // verify: the transformed local AABBs reproduce the package's world triangle AABBs bit for bit
            const Model &md = v.models[in.model];
            for (std::uint32_t k = 0; k < md.triCount && grid; ++k) {
                float lo[3], hi[3];
                localBox(md, k, lo, hi);
                const tw::Tri &tr = v.tris[in.firstTri + k];
                for (int a = 0; a < 3; ++a) {
                    const bool ng = (g.neg >> a) & 1u;
                    const float wl = (ng ? 0.0f - hi[g.p[a]] : lo[g.p[a]]) + g.t[a];
                    const float wh = (ng ? 0.0f - lo[g.p[a]] : hi[g.p[a]]) + g.t[a];
                    if (wl != tr.lo[a] || wh != tr.hi[a]) grid = false;
                }
            }
            if (!grid) ++mismatched;
        }
        if (!grid) continue;
        isGrid[i] = 1;
        gtmp.push_back(g);
        gsrc.push_back(static_cast<std::uint32_t>(i));
    }
    // ---- region: the bounding box of the block-sized grid instances (at most 8 cells); larger grid instances that
    // reach outside it (stadium surroundings) go to the huge BVH, so the direct-indexed table covers the map only
    {
        float rlo[3] = {1e30f, 1e30f, 1e30f}, rhi[3] = {-1e30f, -1e30f, -1e30f};
        for (const GInst &g : gtmp) {
            std::size_t nc = 1;
            for (int a = 0; a < 3; ++a)
                nc *= static_cast<std::size_t>(std::floor(g.hi[a] / opt.cell[a]) - std::floor(g.lo[a] / opt.cell[a]) + 1);
            if (nc > 8) continue;
            for (int a = 0; a < 3; ++a) { rlo[a] = std::min(rlo[a], g.lo[a]); rhi[a] = std::max(rhi[a], g.hi[a]); }
        }
        std::size_t o = 0;
        for (std::size_t i = 0; i < gtmp.size(); ++i) {
            bool inside = true;
            for (int a = 0; a < 3; ++a) inside = inside && gtmp[i].lo[a] >= rlo[a] - opt.cell[a] && gtmp[i].hi[a] <= rhi[a] + opt.cell[a];
            if (!inside) { isGrid[gsrc[i]] = 0; ++outside; continue; }
            gtmp[o] = gtmp[i]; gsrc[o] = gsrc[i]; ++o;
        }
        gtmp.resize(o); gsrc.resize(o);
    }
    // ---- models used by grid instances: SoA local AABBs, sub-grids for big models
    models_.clear();
    mtri_.clear();
    mcell_.clear();
    subStart_.clear();
    for (auto &x : mlo_) x.clear();
    for (auto &x : mhi_) x.clear();
    for (GInst &g : gtmp) {
        if (modelSlot[g.model] < 0) {
            const Model &md = v.models[g.model];
            GModel gm{};
            gm.first = static_cast<std::uint32_t>(mtri_.size());
            gm.count = md.triCount;
            gm.subFirst = 0xffffffffu;
            std::vector<float> lo(3 * md.triCount), hi(3 * md.triCount);
            for (std::uint32_t k = 0; k < md.triCount; ++k) localBox(md, k, &lo[3 * k], &hi[3 * k]);
            std::vector<std::uint32_t> order(md.triCount);
            std::iota(order.begin(), order.end(), 0u);
            std::vector<float> cellIds;
            if (md.triCount >= opt.subgridMinTris) {
                float mlo[3] = {1e30f, 1e30f, 1e30f}, mhi[3] = {-1e30f, -1e30f, -1e30f};
                for (std::uint32_t k = 0; k < md.triCount; ++k)
                    for (int a = 0; a < 3; ++a) { mlo[a] = std::min(mlo[a], lo[3 * k + a]); mhi[a] = std::max(mhi[a], hi[3 * k + a]); }
                float ext = 0;
                for (int a = 0; a < 3; ++a) ext = std::max(ext, mhi[a] - mlo[a]);
                const float edge = std::max(opt.subCell, ext / 64.0f);
                gm.subInv = 1.0f / edge;
                std::size_t ncell = 1;
                for (int a = 0; a < 3; ++a) {
                    gm.subOrg[a] = mlo[a];
                    const int d = std::max(1, std::min(1024, static_cast<int>((mhi[a] - mlo[a]) * gm.subInv) + 1));
                    gm.subDim[a] = static_cast<std::uint16_t>(d);
                    ncell *= static_cast<std::size_t>(d);
                }
                // registrations: every sub-cell a triangle's local AABB overlaps (the query's CellOf formula)
                std::vector<std::vector<std::uint32_t>> reg(ncell);
                std::vector<std::uint32_t> bigList;
                for (std::uint32_t k = 0; k < md.triCount; ++k) {
                    int c0[3], c1[3];
                    std::size_t n = 1;
                    for (int a = 0; a < 3; ++a) {
                        c0[a] = std::max(0, std::min(static_cast<int>(gm.subDim[a]) - 1, CellOf(lo[3 * k + a], gm.subOrg[a], gm.subInv)));
                        c1[a] = std::max(0, std::min(static_cast<int>(gm.subDim[a]) - 1, CellOf(hi[3 * k + a], gm.subOrg[a], gm.subInv)));
                        n *= static_cast<std::size_t>(c1[a] - c0[a] + 1);
                    }
                    if (n > opt.maxReg) { bigList.push_back(k); continue; }
                    for (int x = c0[0]; x <= c1[0]; ++x)
                        for (int y = c0[1]; y <= c1[1]; ++y)
                            for (int z = c0[2]; z <= c1[2]; ++z)
                                reg[(static_cast<std::size_t>(x) * gm.subDim[1] + static_cast<std::size_t>(y)) * gm.subDim[2] + static_cast<std::size_t>(z)].push_back(k);
                }
                order = bigList;
                cellIds.assign(bigList.size(), -1.0f);
                gm.nBig = static_cast<std::uint32_t>(bigList.size());
                gm.subFirst = static_cast<std::uint32_t>(subStart_.size());
                for (std::size_t c = 0; c < ncell; ++c) {
                    subStart_.push_back(static_cast<std::uint32_t>(order.size()));
                    order.insert(order.end(), reg[c].begin(), reg[c].end());
                    cellIds.insert(cellIds.end(), reg[c].size(), static_cast<float>(c));
                }
                subStart_.push_back(static_cast<std::uint32_t>(order.size()));
                gm.count = static_cast<std::uint32_t>(order.size());
                subRegs += order.size();
                ++subgridModels;
            }
            if (cellIds.size() != order.size()) cellIds.assign(order.size(), -1.0f);
            for (std::size_t r = 0; r < order.size(); ++r) {
                const std::uint32_t k = order[r];
                for (int a = 0; a < 3; ++a) { mlo_[a].push_back(lo[3 * k + a]); mhi_[a].push_back(hi[3 * k + a]); }
                mtri_.push_back(k);
                mcell_.push_back(cellIds[r]);
            }
            // pad to 16 with empty boxes (never overlap: lo > hi)
            while (mtri_.size() % 16) {
                for (int a = 0; a < 3; ++a) { mlo_[a].push_back(1e30f); mhi_[a].push_back(-1e30f); }
                mtri_.push_back(0);
                mcell_.push_back(-1.0f);
            }
            modelSlot[g.model] = static_cast<std::int32_t>(models_.size());
            models_.push_back(gm);
        }
        g.model = static_cast<std::uint32_t>(modelSlot[g.model]);
    }
    models = models_.size();
    // ---- the cell table over the region of the grid instances
    ginst_ = gtmp;
    gridInstances = ginst_.size();
    if (!ginst_.empty()) {
        float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        for (const GInst &g : ginst_)
            for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], g.lo[a]); hi[a] = std::max(hi[a], g.hi[a]); }
        for (int a = 0; a < 3; ++a) {
            org_[a] = std::floor(lo[a] / opt.cell[a]) * opt.cell[a];
            inv_[a] = 1.0f / opt.cell[a];
            dim[a] = CellOf(hi[a], org_[a], inv_[a]) + 1;
        }
        cells = static_cast<std::size_t>(dim[0]) * dim[1] * dim[2];
        std::vector<std::uint32_t> count(cells + 1, 0);
        auto forCells = [&](const GInst &g, auto &&fn) {
            int c0[3], c1[3];
            for (int a = 0; a < 3; ++a) {
                c0[a] = std::max(0, CellOf(g.lo[a], org_[a], inv_[a]));
                c1[a] = std::min(dim[a] - 1, CellOf(g.hi[a], org_[a], inv_[a]));
            }
            for (int x = c0[0]; x <= c1[0]; ++x)
                for (int y = c0[1]; y <= c1[1]; ++y)
                    for (int z = c0[2]; z <= c1[2]; ++z) fn((static_cast<std::size_t>(x) * dim[1] + static_cast<std::size_t>(y)) * dim[2] + static_cast<std::size_t>(z));
        };
        for (const GInst &g : ginst_) forCells(g, [&](std::size_t c) { ++count[c]; });
        cellStart_.assign(cells + 1, 0);
        for (std::size_t c = 0; c < cells; ++c) cellStart_[c + 1] = cellStart_[c] + count[c];
        cellInst_.assign(cellStart_[cells], 0);
        std::vector<std::uint32_t> at(cellStart_.begin(), cellStart_.end() - 1);
        for (std::size_t i = 0; i < ginst_.size(); ++i) forCells(ginst_[i], [&](std::size_t c) { cellInst_[at[c]++] = static_cast<std::uint32_t>(i); });
        cellRefs = cellInst_.size();
    }
    stamp_.assign(ginst_.size(), 0u);
    gen_ = 0;
    // ---- huge triangles: everything not in the grid, one small BVH4
    tw::World hw;
    hugeTri_.clear();
    std::size_t why[4] = {0, 0, 0, 0}, whyTris[4] = {0, 0, 0, 0};
    for (std::size_t i = 0; i < NI; ++i) {
        const Instance &in = v.instances[i];
        if (isGrid[i] || in.triCount == 0) continue;
        ++hugeInstances;
        int r = 0;
        if (in.flags & kInstMovingPair) r = 1;
        else if (!(in.flags & kInstPermExact) || in.rotation == kRotGeneral) r = 2;
        else r = 3;  // spans more cells, reaches outside the block region (or AABB mismatch)
        ++why[r]; whyTris[r] += in.triCount;
        for (std::uint32_t k = 0; k < in.triCount; ++k) {
            hw.tris.push_back(v.tris[in.firstTri + k]);
            hugeTri_.push_back(in.firstTri + k);
        }
    }
    hugeTris = hugeTri_.size();
    if (std::getenv("CC_GRIDDIAG")) {
        std::printf("  huge instances: moving pair %zu (%zu tris), general or inexact rotation %zu (%zu tris), more than %d cells %zu (%zu tris)\n", why[1], whyTris[1],
                    why[2], whyTris[2], opt.maxCellRefs, why[3], whyTris[3]);
        for (std::size_t i = 0; i < NI; ++i) {
            const Instance &in = v.instances[i];
            if (isGrid[i] || in.triCount < 2000) continue;
            std::printf("    instance %zu: %u tris, box (%.0f %.0f %.0f)-(%.0f %.0f %.0f), flags %u rot %u\n", i, in.triCount, in.lo[0], in.lo[1], in.lo[2], in.hi[0], in.hi[1], in.hi[2],
                        in.flags, in.rotation);
        }
    }
    huge_.Build(hw);
    if (mismatched) std::printf("  cell grid: %zu permutation instances reproduce the package AABBs inexactly; they use the huge BVH\n", mismatched);
    return true;
}

std::size_t CellGrid::Bytes() const {
    std::size_t b = cellStart_.size() * 4 + cellInst_.size() * 4 + ginst_.size() * sizeof(GInst) + models_.size() * sizeof(GModel) + mtri_.size() * 8 +
                    subStart_.size() * 4 + hugeTri_.size() * 4 + huge_.Bytes() + stamp_.size() * 4;
    for (int a = 0; a < 3; ++a) b += (mlo_[a].size() + mhi_[a].size()) * 4;
    return b;
}

std::size_t CellGrid::HotBytesPerMap() const {
    return cellStart_.size() * 4 + cellInst_.size() * 4 + ginst_.size() * sizeof(GInst) + stamp_.size() * 4;
}

}  // namespace cc
