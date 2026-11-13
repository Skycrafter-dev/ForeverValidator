// Block-cell candidate index (architect C, docs/twin/collision-c.md §9): an exact, tree-free replacement for the
// BVH4 box query over the world package's triangles.
//
// Query(q, f) calls f(t) for exactly the world triangles t whose AABB (tw::Tri lo/hi, closed) overlaps the
// closed box q. That is the set Bvh4::Query reports on the same triangles; the order differs (callers sort by
// TriInfo::fvOrder anyway).
//
// - Grid instances (signed permutation with kInstPermExact, not a moving pair) are listed per Stadium block
//   cell (32 x 8 x 32 m) in one direct-indexed array over the map region:
//   an O(1) lookup, no tree. Instances are registered in every cell their AABB overlaps (at most 512).
// - A grid instance stores only its translation, its axis permutation and signs, its AABB and its model. The
//   triangles are its model's mesh-local AABBs, shared by every instance and stored SoA padded to 16. Their
//   world AABBs come out exactly: for a kInstPermExact instance FV's world vertex is s v[p] + t bit for bit, and
//   round(a + t) is monotone in a, so the world AABB is the transformed local AABB with one rounded add.
//   They are scanned 16 at a time.
// - Models with many triangles also carry a mesh-local sub-grid. Each triangle is registered in every sub-cell
//   its local AABB overlaps; one overlapping more than maxReg sub-cells goes to an always-scanned list. A query
//   scans only the sub-cells its local box overlaps, and reports a triangle only from the sub-cell holding
//   max(triangle local lo, query local lo), so it is reported exactly once.
// - Every other instance (general rotation, inexact permutation, moving pair "Flags", more than 4096 cells, or
//   reaching outside the block region, i.e. the box of the block-sized instances: stadium surroundings) goes
//   into a small BVH4 over its world triangles.
#ifndef CC_GRID_H
#define CC_GRID_H

#include <cstdint>
#include <vector>

#include "tw_accel.h"
#include "tw_world.h"
#include "tw_world_pack.h"

#ifndef CC_GRID_STATS
#define CC_GRID_STATS 0
#endif
#if CC_GRID_STATS
#define CC_GRID_CNT(x) (++(x))
#else
#define CC_GRID_CNT(x) ((void)0)
#endif

namespace cc {

using tw::Box3;
struct CellGridPrep;

class CellGrid {
public:
    struct Options {
        float cell[3] = {32.0f, 8.0f, 32.0f};
        int maxCellRefs = 4096;       // instances over more cells go to the huge-triangle BVH
        std::uint32_t subgridMinTris = 32;  // models with at least this many triangles get a sub-grid
        float subCell = 2.0f;         // sub-grid cell edge (m, mesh-local; at least the model extent / 64)
        std::uint32_t maxReg = 16;    // a triangle in more sub-cells is always scanned
    };
    bool Build(const tw::world::WorldPackView &v, const Options &opt);

    template <class F>
    void Query(const Box3 &q, F &&f) const;

    // statistics
    std::size_t Bytes() const;           // everything this index owns
    std::size_t HotBytesPerMap() const;  // cell table + instance table (per map); model data is shared across maps of one catalogue
    std::size_t gridInstances = 0, hugeInstances = 0, hugeTris = 0, cells = 0, cellRefs = 0, models = 0, subgridModels = 0, subRegs = 0;
    int dim[3] = {0, 0, 0};

    bool skipHuge = false, skipGrid = false;  // timing experiments only
    // per-query counters (not thread safe; benchmarks only)
    mutable std::uint64_t qCells = 0, qInstances = 0, qBlocks = 0, qHits = 0;

    struct GInst {                // 48 B
        float lo[3], hi[3];       // world AABB of the instance triangles
        float t[3];               // translation
        std::uint8_t p[3];        // world axis a takes mesh-local axis p[a] ...
        std::uint8_t neg;         // ... with sign -1 when bit a is set
        std::uint32_t model;      // index into models_
        std::uint32_t firstTri;   // world triangle id of model triangle 0
    };
    struct GModel {               // 48 B
        std::uint32_t first, count;   // SoA range (padded to 16)
        std::uint32_t subFirst;       // sub-grid cell table (subDim cells + 1, offsets from first), or ~0u
        std::uint32_t nBig;           // triangles larger than a sub-cell, first in the range, always scanned
        std::uint16_t subDim[3];
        std::uint16_t pad;
        float subInv;                 // 1 / sub-cell edge
        float subOrg[3];
        float subInflate[3];          // unused (kept for layout)
    };

private:
    float org_[3] = {0, 0, 0}, inv_[3] = {1, 1, 1};
    std::vector<std::uint32_t> cellStart_;   // dim0*dim1*dim2 + 1
    std::vector<std::uint32_t> cellInst_;    // grid-instance ids
    std::vector<GInst> ginst_;
    std::vector<GModel> models_;
    std::vector<float> mlo_[3], mhi_[3];     // mesh-local triangle AABBs, SoA, each model padded to 16
    std::vector<std::uint32_t> mtri_;        // model triangle index of each SoA entry (sub-grid models are bucket-sorted)
    std::vector<float> mcell_;               // sub-grid registrations: the sub-cell id (as float) the entry is registered in, -1 otherwise
    std::vector<std::uint32_t> subStart_, subTri_;  // sub-grid: per cell, model-local triangle indices
    tw::Bvh4 huge_;
    std::vector<std::uint32_t> hugeTri_;     // huge BVH id -> world triangle id
    mutable std::vector<std::uint32_t> stamp_;
    mutable std::uint32_t gen_ = 0;

    struct Dedupe { float qlo[3], org[3], dim[3]; float inv; };
    void Prepare(const GInst &g, const float wl[3], const float wh[3], struct CellGridPrep *pp) const;
    template <class F>
    void ScanPrepared(const GInst &g, const struct CellGridPrep &pp, std::uint32_t first, std::uint32_t count, const Dedupe *dd, F &f) const;
    template <class F>
    void ScanModel(const GInst &g, const GModel &md, const Box3 &q, F &f) const;
    template <class F>
    void ScanRange(const GInst &g, std::uint32_t first, std::uint32_t count, const Dedupe *dd, const float wl[3], const float wh[3], F &f) const;
};

}  // namespace cc

#include "cc_grid_inl.h"

#endif
