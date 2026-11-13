// Block-model world (M2 experiment 2): Stadium maps are instances of a block
// catalogue. Each distinct surface mesh (a "model") is stored once in its own
// local frame, with adjacent coplanar same-material triangles merged into
// convex planar patches; the world is a 32 x 8 x 32 m cell grid of model
// instances (90-degree yaw: exact signed-permutation transforms). A query
// collects (instance, element) candidates once per tick; every substep tests
// them against the 8 car ellipsoids and applies FV's per-shape merge.
#ifndef TW_BLOCKS_H
#define TW_BLOCKS_H

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "tw_exact.h"
#include "tw_world.h"

namespace tw {

struct Element {          // one patch (k >= 4 vertices or >= 2 source triangles) or one source triangle
    float n[3];           // model-local unit normal (front side)
    float lo[3], hi[3];   // model-local AABB
    std::uint32_t first;  // vertex loop in Model::loop (CCW about n); triangles keep FV's stored order
    std::uint16_t count;  // loop vertex count
    std::uint8_t material;
    std::uint8_t isTriangle;  // 1: single source triangle, tested with FV's triangle algorithm
    std::uint32_t sourceTris;
};

struct Model {
    std::vector<Element> elements;
    std::vector<float> loop;    // xyz per loop vertex
    std::vector<float> edgeU;   // per loop edge (v_i -> v_i+1): in-plane outward unit normal
    std::uint32_t sourceTriCount = 0;
    Box3 box = EmptyBox();
    // model-local uniform grid over the elements (query acceleration)
    float gridOrg[3] = {0, 0, 0}, gridInv = 0.25f;
    int gridDim[3] = {1, 1, 1};
    std::vector<std::uint32_t> gridStart, gridIds;
    void BuildGrid();
};

struct Instance {
    std::uint32_t model;
    float R[3][3];        // R[i][j]: row i, column j (columns = basisX, basisY, basisZ)
    float T[3];
    std::uint8_t permutation;  // R is a signed permutation (exact transforms)
    std::uint8_t trigger;
    Box3 box;             // world AABB
};

struct BlockWorld {
    std::vector<Model> models;
    std::vector<Instance> instances;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> cells;  // 32 x 8 x 32 m cells -> instances
    std::uint32_t nonPermutation = 0;
    std::size_t sourceTris = 0, elements = 0, mergedPatches = 0, trianglesInPatches = 0;
    // flattened (instance, element) refs with world AABBs per 16 x 8 x 16 m cell (query index)
    struct CellRef { float lo[3], hi[3]; std::uint32_t inst, elem; };
    std::unordered_map<std::uint64_t, std::uint32_t> cellIndex;  // key -> index into cellRanges
    std::vector<std::uint32_t> cellBegin, cellEnd;
    std::vector<CellRef> refs, hugeRefs;
    std::vector<std::uint32_t> instElemBase;
    std::uint32_t totalInstElems = 0;
    static constexpr float kCellX = 16.0f, kCellY = 8.0f, kCellZ = 16.0f;
    void BuildCellIndex();
    std::size_t Bytes() const;
    bool Build(const SourceScene &src, bool mergePatches, float planeEps, float angleEps);
    static std::uint64_t CellKey(int x, int y, int z) {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x) & 0x1fffffu) << 42) |
               (static_cast<std::uint64_t>(static_cast<std::uint32_t>(y) & 0x1fffffu) << 21) |
               static_cast<std::uint64_t>(static_cast<std::uint32_t>(z) & 0x1fffffu);
    }
};

struct Candidate {
    std::uint32_t instance, element;
    Box3 box;  // world AABB of the element
};

// Shape map expressed in a model's frame (exact for signed permutations
// except the single subtraction t - T).
inline void ToModel(const ShapeMap &w, const Instance &in, ShapeMap *m) {
    float d[3] = {w.t[0] - in.T[0], w.t[1] - in.T[1], w.t[2] - in.T[2]};
    for (int i = 0; i < 3; ++i) m->t[i] = (in.R[0][i] * d[0] + in.R[1][i] * d[1]) + in.R[2][i] * d[2];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            m->M[i][j] = (in.R[0][i] * w.M[0][j] + in.R[1][i] * w.M[1][j]) + in.R[2][i] * w.M[2][j];  // Rᵀ M
            m->N[i][j] = (in.R[0][i] * w.N[0][j] + in.R[1][i] * w.N[1][j]) + in.R[2][i] * w.N[2][j];  // Rᵀ N
            m->A[j][i] = (w.A[j][0] * in.R[0][i] + w.A[j][1] * in.R[1][i]) + w.A[j][2] * in.R[2][i];  // A R
        }
    for (int k = 0; k < 3; ++k) { m->half[k] = w.half[k]; m->box = w.box; }
}

inline void ToWorldPoint(const Instance &in, const float p[3], float o[3]) {
    for (int i = 0; i < 3; ++i) o[i] = ((in.R[i][0] * p[0] + in.R[i][1] * p[1]) + in.R[i][2] * p[2]) + in.T[i];
}
inline void ToWorldDir(const Instance &in, const float p[3], float o[3]) {
    for (int i = 0; i < 3; ++i) o[i] = (in.R[i][0] * p[0] + in.R[i][1] * p[1]) + in.R[i][2] * p[2];
}

// Element test in model space; outputs a model-space contact (K is the
// shape's unit-space normal, frame independent). Patches take the FACE fast
// path (plane distance + point-in-convex-polygon in model space) and fall
// back to FV's first-violated-edge feature rule on the polygon loop.
int ElementTest(const ShapeMap &m, const Model &model, const Element &e, Contact *out, bool *fast);
// Bench counters (not thread safe): triangle tests, plane rejects, fast FACE, full loop tests, loop vertices transformed.
extern std::uint64_t g_cntTri, g_cntPlaneReject, g_cntFast, g_cntFull, g_cntFullVerts;
// Same test on raw data (used with world-space copies made once per tick).
int LoopTest(const ShapeMap &m, const float *loop, const float *edgeU, std::uint32_t count, const float n[3], bool isTriangle,
             Contact *out, bool *fast);

// Board 20 §3.4 per-shape merge on contacts of one shape (primaries kept).
struct MergedContact { Contact c; bool primary; std::uint8_t material; };
void MergeShape(std::vector<MergedContact> &in, std::vector<MergedContact> *out);

}  // namespace tw

#endif
