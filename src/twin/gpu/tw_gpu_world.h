// Twin GPU engine: the collision world as flat arrays (host build in tw_gpu_world.cpp, device copy and
// kernels in tw_gpu_collide.cu). Plain data shared by the host compiler and nvcc: no world headers.
//
// What the device needs to give C's stage ("c", the noise path) bit for bit (cc_stage.h):
//   - the 8 shapes' local poses and radii (collide::CarShapes), for ShapePoses and the exact shape boxes;
//   - the triangles in FV detection order (rank = TriInfo::fvOrder) under FvBvh's implicit 16-ary tree
//     (level 1, the leaves: runs of up to 16 ranks cut where the instance changes; above: runs of 16 nodes), so a
//     walk visits the ranks that overlap a box in increasing order; and the walk's spatial 16-ary tree over its own
//     leaves (runs of up to 16 consecutive ranks cut for small boxes: far fewer nodes a query; the leaves' records
//     sorted back into rank order by the walk);
//   - per rank its instance, pack model triangle, world triangle and the world triangle's edge planes; per instance its iso; per pack model
//     triangle its mesh-local vertices (FV stored order); per world triangle its RespContact output fields.
#ifndef TW_GPU_WORLD_H
#define TW_GPU_WORLD_H

#include <cstdint>
#include <string>
#include <vector>

namespace tw::world {
struct WorldPackView;
}
namespace tw::collide {
struct CarShapes;
}

namespace tw::gpu {

// A node in one 128-byte line (the walk): its own box's frame and its children's boxes quantized to 7 bits in a
// byte each in it, q(x) = clamp(floor((x - org) * scale), 0, 127), the same float operations on the host (children)
// and the device (the query box). q is monotone, so a child box that meets the query box meets it quantized:
// conservative. A lo byte is q(lo) (bit 7 clear), a hi byte q(hi) | 0x80 (bit 7 set, unused children too): bit 7 is
// the guard of the walk's per-byte subtractions (one instruction a word and bound, no borrow across bytes).
struct alignas(128) NodeQ {
    float org[3];
    std::uint32_t first;     // as NodeInfo
    float scale[3];          // 127 / extent (0 for an empty extent: every child meets)
    std::uint32_t meta;      // count | leaf << 8
    std::uint32_t lo[3][4];  // per axis: children 4w .. 4w + 3 in word w's bytes (low byte first)
    std::uint32_t hi[3][4];
};
static_assert(sizeof(NodeQ) == 128, "NodeQ: one line");

struct alignas(32) RankBox {   // a rank's triangle box (the leaves' per-shape test)
    float lo[3], pad0, hi[3], pad1;
};

struct NodeInfo {
    std::uint32_t first;    // leaf node: first rank; internal node: node index of the first child
    std::uint8_t count;     // children
    std::uint8_t leaf;      // children are ranks
    std::uint16_t pad;
};

struct RankRef {
    std::uint32_t instance, modelTri, tri;   // Instances index, pack model triangle, world triangle
};

// A rank's narrow-phase record, 64 bytes (half a line; the device's copy of HostWorld::plane, ::rank and the pack model
// triangle's ::verts): the plane prefilter's plane, the instance, the world triangle and the mesh-local vertices, so a
// candidate pair needs one load (four 16-byte ones issued together) instead of a chain plane -> RankRef -> vertices.
struct alignas(64) RankN {
    float plane[4];
    std::uint32_t instance, tri;
    float v[9];
    std::uint32_t pad;
};
static_assert(sizeof(RankN) == 64, "RankN: half a line");

// A rank's edge planes (world, host-built from its world triangle): per edge e a half-space m.x <= off holding the whole
// triangle, m the edge's outward normal in the triangle's plane (unit, from double), stored as q = trunc(32767 m) in 16
// bits a component and read as q 2^-15 (exact in float, |m| < 1); off the largest m.v over the triangle's vertices with
// that m (in double, rounded up to float), so the triangle lies in m.x <= off exactly, whatever m's rounding: the plane
// prefilter's edge test (PlaneFilter, EdgeOut). A degenerate edge or plane: m = 0, off = 0 (never rejects).
struct alignas(32) RankE {
    float off[3];
    std::int16_t m[3][3];
    std::int16_t pad;
};
static_assert(sizeof(RankE) == 32, "RankE: one sector");

struct OutRecG {   // collide_c::OutRec: a world triangle's RespContact output fields
    float peerZ[3];
    std::uint32_t peerId;
    std::uint8_t material, flags, pad[2];
};

struct CarShapesG {
    float lr[8][9];     // shape -> car rotation (columns), wheel shapes: the rest surface pose
    float lt[8][3];
    float rad[8][3];
    float r2[8][3];     // rad^2 (the box's Q_aa weights)
    float irad[8][3];   // 1 / rad
    float hc[8][3];     // 2.002 x the ellipsoid's half extents along the car's axes, sqrt(sum_i rad_i^2 lr_{i,a}^2), rounded up:
                        // phase 2's front test (BoxPlaneMask)
    std::int32_t wheel[8];   // canonical wheel of a wheel shape (absorb index), -1 for a body shape
};

struct CollisionConsts {
    float dirEpsSq;          // exact::kDirEpsSq
    float collisionDistance; // exact::kCollisionDistance
    float normalAlignment;   // PhysicsTolerance::SphereContactNormalAlignment (the merge's 30 degrees)
};

// Host build (vectors), uploaded by CollisionPacks.
struct HostWorld {
    CarShapesG car{};
    CollisionConsts k{};
    std::vector<NodeInfo> nodes;     // per node (host only: the build)
    std::vector<NodeQ> qnodes;       // per node
    std::vector<RankBox> rankBox;    // per rank
    std::uint32_t root = 0;          // the walk's root: the spatial tree's, or 0 (the FV tree's root, the fallback)
    std::uint32_t stackNeed = 0;     // the spatial tree's walk: pending entries at most
    std::uint32_t ranks = 0;
    std::uint32_t depth = 0;         // levels below the FV tree's root
    std::vector<RankRef> rank;
    std::vector<float> plane;        // 4 per rank: world unit normal, n . v0 (double), for the plane prefilter; 0 for a
                                     // degenerate triangle (never rejects)
    std::vector<RankE> edge;         // per rank: the edge planes (32 bytes a rank on the device)
    std::vector<float> iso;          // 12 per instance
    std::vector<float> verts;        // 9 per pack model triangle
    std::vector<OutRecG> outRec;     // per world triangle
};

// The flat world from C's stage's world pack and the car shapes (tw_gpu_world.cpp).
bool BuildHostWorld(const world::WorldPackView &view, const collide::CarShapes &car, HostWorld *w, std::string *error);

}  // namespace tw::gpu

#endif
