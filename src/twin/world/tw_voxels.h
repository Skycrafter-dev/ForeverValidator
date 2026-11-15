// M2 experiment 4: per-block-model voxel candidate lists.
//
// For each block model (shared by all instances) and each shape-radius class,
// model-local space is voxelized; each voxel stores the deduplicated list of
// model triangles that can be within reach of a shape centre anywhere in that
// voxel (conservative: triangle AABB within the voxel box inflated by the class
// bounding radius, and plane distance from the voxel centre <= radius + half
// diagonal). Lists are classified:
//   EMPTY        nothing reachable: the shape is skipped for this instance;
//   SINGLE_PLANE all reachable triangles are coplanar, same material, not a
//                trigger, and the projected voxel is inside their union, farther
//                than radius + half diagonal from every unshared edge: the
//                contact is a closed-form FACE (internal-edge features are
//                dropped by FV's merge when the contact is shallow enough;
//                deeper contacts fall back to GENERAL at run time);
//   GENERAL      exact tests on the short list.
// A SINGLE_PLANE voxel may also carry "rest" triangles (other planes within
// reach); those are tested exactly and merged with the closed-form FACE.
// List filters (conservative): centre on the front side somewhere in the voxel
// (sd(centre) >= -half diagonal) and point-triangle distance from the voxel
// centre <= radius + half diagonal.
// Storage is sparse: 8 x 8 x 8-voxel bricks, deduplicated; all-empty bricks
// cost one table entry. Query: brick table -> brick -> list id -> list.
#ifndef TW_VOXELS_H
#define TW_VOXELS_H

#include <cstdint>
#include <vector>

#include "tw_blocks.h"

namespace tw {

enum VoxClass : std::uint8_t { kVoxEmpty = 0, kVoxSingle = 1, kVoxPlane = 2, kVoxMixed = 3, kVoxGeneral = 4 };
// kVoxSingle: one coplanar group covering the voxel's reach, nothing else (the lead's SINGLE_PLANE);
// kVoxPlane: one coplanar group, not covering; kVoxMixed: a group + rest triangles; kVoxGeneral: rest only.

struct VoxSeg { float a[3], b[3]; };                       // one model edge (model-local)
struct VoxEdge { std::uint32_t seg; std::int32_t owner; };  // owner: member index of a group boundary edge, -1 for a shared edge

struct VoxList {
    std::uint32_t first = 0;        // into VoxGrid::items: the main group's members, then the rest
    std::uint16_t count = 0;        // all items
    std::uint16_t memberCount = 0;  // main coplanar group (0: none); the rest starts here
    std::uint32_t edgeFirst = 0;    // into VoxGrid::edges: the group's edges (shared ones only when covered)
    std::uint16_t edgeCount = 0;
    std::uint8_t cls = kVoxEmpty;
    std::uint8_t material = 0;      // the group's material
    std::uint8_t covered = 0;       // the group's union covers every reachable contact point of the voxel
    std::uint8_t pad[3] = {0, 0, 0};
    float n[3] = {0, 0, 0}, p[3] = {0, 0, 0};  // the group's model-local plane: through p, unit normal n
};

struct VoxGrid {           // one radius class of one model
    static constexpr int kB = 8;  // brick edge in voxels
    float org[3] = {0, 0, 0};
    float inv = 1.0f, size = 1.0f;
    int dim[3] = {0, 0, 0};        // voxels per axis (multiple of kB); 0: no grid (fallback)
    int bdim[3] = {0, 0, 0};       // bricks per axis
    std::vector<std::uint32_t> brick;  // brick table (x-major): brick id, 0 = the all-empty brick
    std::vector<std::uint16_t> cells;  // kB^3 list ids per brick, z fastest
    std::vector<VoxList> lists;        // list 0 = EMPTY
    std::vector<std::uint16_t> items;  // model element indices
    std::vector<VoxEdge> edges;
    std::size_t nonEmptyBricks = 0;
    const VoxList *Lookup(const float t[3]) const {
        int v[3];
        for (int a = 0; a < 3; ++a) {
            const float f = (t[a] - org[a]) * inv;
            if (!(f >= 0.0f)) return nullptr;
            v[a] = static_cast<int>(f);
            if (v[a] >= dim[a]) return nullptr;
        }
        const std::uint32_t b = brick[(static_cast<std::size_t>(v[0] >> 3) * bdim[1] + (v[1] >> 3)) * bdim[2] + (v[2] >> 3)];
        return &lists[cells[(static_cast<std::size_t>(b) << 9) | ((v[0] & 7) << 6) | ((v[1] & 7) << 3) | (v[2] & 7)]];
    }
};

struct VoxModel {
    static constexpr int kClasses = 5;
    VoxGrid grid[kClasses];
    std::vector<VoxSeg> segs;  // edges referenced by the grids
};

struct VoxWorld {
    // StadiumCar bounding radii: wheels, shape 3, shape 2, shape 1, shape 0 (checked at run time)
    float classRadius[VoxModel::kClasses] = {0.365f, 0.906f, 1.039f, 1.683f, 1.902f};
    bool buildClass[VoxModel::kClasses] = {true, true, true, true, true};  // skipped classes are aliased by the caller
    double hugeExtent = 128.0, hugeMaxVoxels = 2.5e5;  // models larger than hugeExtent (m) get coarser grids
    float margin = 0.0f;    // travel margin added to every class radius (tick-level lists)
    bool useGroups = true;  // coplanar groups (else every triangle is "rest": exact tests only)
    bool frontCull = true;  // drop triangles whose back side holds the whole voxel
    bool distCull = true;   // point-triangle distance filter (else AABB + plane distance only)
    std::vector<VoxModel> models;
    std::size_t lists = 0, items = 0, edgesTotal = 0, voxels = 0, bricks = 0, uniqueBricks = 0, overflowGrids = 0, coarsenedGrids = 0;
    double buildMs = 0;
    std::size_t Bytes() const;
    // voxelSize: base voxel edge (m); a grid's dense voxel count is capped at maxVoxels (coarser voxels for huge models)
    bool Build(const BlockWorld &bw, float voxelSize, double maxVoxels);
};

}  // namespace tw

#endif
