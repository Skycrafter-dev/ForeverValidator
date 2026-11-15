// Twin static world and car shapes (FV-independent).
//
// The world is a flat soup of world-space triangles in FV's stored vertex
// order (board 20 §2.2), one 64-byte record each, plus a per-triangle side
// record (world unit normal, FV surface index). It is compiled from FV's flat
// scene export (tw_world_compile.cpp) and can be saved/loaded without FV.
//
// The car is 8 ellipsoids (board 20 §2.1). A Shape holds the world pose of
// one ellipsoid at detection time; ShapeMap is the per-substep derived data
// used by every query (unit-space map, support matrix, AABB).
#ifndef TW_WORLD_H
#define TW_WORLD_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace tw {

struct Box3 {
    float lo[3];
    float hi[3];
};

inline bool Overlap(const Box3 &a, const Box3 &b) {
    return a.lo[0] <= b.hi[0] && b.lo[0] <= a.hi[0] && a.lo[1] <= b.hi[1] &&
           b.lo[1] <= a.hi[1] && a.lo[2] <= b.hi[2] && b.lo[2] <= a.hi[2];
}
inline bool Contains(const Box3 &outer, const Box3 &inner) {
    return inner.lo[0] >= outer.lo[0] && inner.lo[1] >= outer.lo[1] && inner.lo[2] >= outer.lo[2] &&
           inner.hi[0] <= outer.hi[0] && inner.hi[1] <= outer.hi[1] && inner.hi[2] <= outer.hi[2];
}
inline Box3 Inflate(const Box3 &b, float m) {
    return {{b.lo[0] - m, b.lo[1] - m, b.lo[2] - m}, {b.hi[0] + m, b.hi[1] + m, b.hi[2] + m}};
}
inline Box3 Union(const Box3 &a, const Box3 &b) {
    Box3 r;
    for (int k = 0; k < 3; ++k) {
        r.lo[k] = a.lo[k] < b.lo[k] ? a.lo[k] : b.lo[k];
        r.hi[k] = a.hi[k] > b.hi[k] ? a.hi[k] : b.hi[k];
    }
    return r;
}
inline Box3 EmptyBox() { return {{1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}}; }

// meta bit layout
constexpr std::uint32_t kMetaMaterialMask = 0xffu;   // surface material id (0..31)
constexpr std::uint32_t kMetaTrigger = 1u << 8;      // checkpoint/start/finish trigger
constexpr std::uint32_t kMetaCarIgnores = 1u << 9;   // Water 13 / GolfBall 23 (board 20 §4)
constexpr int kMetaActorShift = 10;                  // actor index (22 bits)

struct Tri {  // one cache line
    float v[3][3];     // world, stored order (front = (v1-v0)x(v2-v0))
    float lo[3], hi[3];
    std::uint32_t meta;
};
static_assert(sizeof(Tri) == 64, "Tri must be 64 bytes");

struct TriExtra {
    float n[3];             // world unit normal from a double cross product (culls/sleep bounds only)
    std::uint32_t surface;  // FV flat-scene surface index
};

struct World {
    std::vector<Tri> tris;
    std::vector<TriExtra> extra;
    std::uint32_t actorCount = 0u;
    std::uint32_t surfaceCount = 0u;
    std::uint32_t triggerTris = 0u;
    std::uint32_t skippedSurfaces = 0u;  // non-mesh or non-colliding surfaces
    Box3 bounds = EmptyBox();
    std::size_t Bytes() const { return tris.size() * sizeof(Tri) + extra.size() * sizeof(TriExtra); }
};

// ---- car shapes -------------------------------------------------------------
constexpr int kShapes = 8;  // tree order: 0-3 body, 4 FL, 5 FR, 6 RL, 7 RR (board 20 §2.1)

struct Shape {
    float t[3];        // world centre
    float axis[3][3];  // world unit axes of the ellipsoid (axis[j] = image of local e_j)
    float r[3];        // radii
};

struct ShapeMap {
    float t[3];
    float A[3][3];   // unit map: u = A (x - t),  A = diag(1/r) * axesᵀ
    float M[3][3];   // inverse: x = M u + t,     M = axes * diag(r)
    float N[3][3];   // normal map back to world: N = axes * diag(1/r)
    float half[3];   // world AABB half extents (|M| * 1)
    Box3 box;
};

inline void MakeShapeMap(const Shape &s, ShapeMap *m) {
    for (int k = 0; k < 3; ++k) m->t[k] = s.t[k];
    for (int j = 0; j < 3; ++j) {
        const float ir = 1.0f / s.r[j];
        for (int i = 0; i < 3; ++i) {
            m->A[j][i] = s.axis[j][i] * ir;
            m->M[i][j] = s.axis[j][i] * s.r[j];
            m->N[i][j] = s.axis[j][i] * ir;
        }
    }
    for (int i = 0; i < 3; ++i) {
        m->half[i] = std::fabs(m->M[i][0]) + std::fabs(m->M[i][1]) + std::fabs(m->M[i][2]);
        m->box.lo[i] = m->t[i] - m->half[i];
        m->box.hi[i] = m->t[i] + m->half[i];
    }
}

// ---- recorded FV substeps ----------------------------------------------------
struct FvContact {
    std::uint8_t shape;    // tree index 0..7
    std::uint8_t primary;  // FACE (sphereMergePrimary)
    std::uint8_t material; // materialB (surface material id); TWREC002 raw contacts hold an unmapped value
    std::uint8_t pad;
    float P[3], N[3], Sep[3], K[3];  // world point, impulse normal, separation; K = unit-space merge key
    std::uint32_t actor;   // static actor = FV installation order = world-pack actor index (Tri meta);
                           // kNoActor when unknown (recordings older than TWREC003)
};
constexpr std::uint32_t kNoActor = 0xffffffffu;

struct Substep {
    std::uint32_t tick;
    std::uint32_t firstContact;   // raw narrow-phase contacts (Recording::contacts)
    std::uint32_t contactCount;
    std::uint32_t flags;          // bit0: shape poses cross-checked against an FV ellipsoid iso
    std::uint32_t firstMerged;    // FV's merged per-shape contacts (Recording::merged)
    std::uint32_t mergedCount;
    float carPos[3], carRot[9];   // rotation columns basisX, basisY, basisZ
    float vel[3], angVel[3];      // world
    Shape shapes[kShapes];
};

// FV's flat scene, kept per surface in mesh-local coordinates (block-model experiment).
struct SourceSurface {
    float iso[12];                 // localToWorld: rotation columns basisX, basisY, basisZ, then translation
    std::uint32_t firstVertex, vertexCount, firstTri, triCount;
    std::uint32_t actor, trigger;
};
struct SourceTri { std::uint32_t v[3]; std::uint32_t material; };
struct SourceScene {
    std::vector<SourceSurface> surfaces;
    std::vector<float> vertices;   // xyz, mesh-local
    std::vector<SourceTri> tris;   // vertex indices relative to the surface's firstVertex
};

struct Recording {
    std::string name;
    World world;
    SourceScene source;
    std::vector<FvContact> merged;
    std::vector<Substep> substeps;
    std::vector<FvContact> contacts;
    std::uint32_t ticks = 0u;
};

bool SaveRecording(const char *path, const Recording &rec);
bool LoadRecording(const char *path, Recording *rec);

}  // namespace tw

#endif
