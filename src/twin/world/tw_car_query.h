// Candidate memo and world-space prefilter for the M3 stage (src/twin/collide/tw_collide.cpp), FV-free.
//
// The stage keeps, per candidate, a memo of the triangles whose AABB overlaps a box around the car
// (refilled when the car's union box leaves it), in FV detection order. This file builds that memo
// and filters it per shape:
//
//   BuildMemo   one BVH query of the memo box, one sort by TriInfo::fvOrder, and per candidate its
//               AABB, its lookups, and the four prefilter planes of tw_exact_fast.h (triangle plane,
//               three edge side planes) rotated to world and offset relative to the memo origin.
//   Filter      per shape: the candidates whose AABB overlaps the shape's query box (the stage's
//               closed-box test) and that pass the support prefilter of tw_exact_fast.h
//               (PrefilterQ) in world space: no (shape, instance) frame is needed. Survivors keep FV
//               order. AVX-512: the box test on 16 candidates x 8 shapes, then the prefilter on
//               dense packets of 16 (candidate, shape) pairs; or a baseline loop. Same lists.
//
// The prefilter only drops pairs FV's CollideTriangle rejects (every FV contact has a triangle
// point inside the closed ellipsoid); its tolerance covers the float error of the world-space form
// and FV's own map-scale rounding of the composed mesh transform (docs/twin/m3-exact-fast.md §6).
#ifndef TW_CAR_QUERY_H
#define TW_CAR_QUERY_H

#include <cstdint>
#include <vector>

#include "tw_accel.h"
#include "tw_exact_fast.h"
#include "tw_world.h"
#include "tw_world_pack.h"

namespace tw::cq {

struct Ref { std::uint32_t instance, modelTri, model, local; };   // modelTri: global ModelTris index; local: Memo::inst index

// What the narrow phase and the output read per candidate, copied at refill so that a call touches only
// the candidate's memo (not the world's tables).
struct OutFields {                         // RespContact fields of the triangle's contacts
    float peerZ[3];
    std::uint32_t peerId;                  // the trigger id for a trigger triangle, else the actor's peerId
    std::uint8_t material, trigger, pad[2];
};
struct InstIso { float iso[12]; };         // an instance's localToWorld (rotation columns, translation)

struct Memo {
    Box3 box = EmptyBox();
    float origin[3] = {0.0f, 0.0f, 0.0f};
    std::uint32_t n = 0;                   // candidates, FV order (the arrays only grow: their size is a capacity)
    std::vector<std::uint32_t> tri;        // world triangle ids
    std::vector<Ref> ref;
    std::vector<float> lo[3], hi[3];       // triangle AABBs (padded to a multiple of 16, empty boxes)
    std::vector<exact::TriVerts> verts;    // per candidate: its mesh-local vertices (FV order)
    std::vector<std::uint32_t> patch;      // per candidate: its coplanar patch (tw_patches.h), when BuildMemo got a table
    std::vector<std::uint64_t> pkey;       // per candidate: instance << 32 | patch (a lone triangle: 1 << 31 | its model triangle)
    bool hasKeys = false;                  // patch and pkey filled (BuildMemo got a patch table)
    std::vector<OutFields> outf;           // per candidate: its contacts' output fields
    std::vector<InstIso> inst;             // the candidates' instances (Ref::local)
    std::vector<exact::FastTri> planes;    // the 4 prefilter planes in world (FastTri layout: x0..3 y0..3 z0..3
                                           // d0..3; unit normals, offsets relative to origin)
    float maxD1 = 0.0f;                    // max over the candidates' instances of |t_m - origin|_1
    float maxT1 = 0.0f;                    // ... of |t_m|_1
    float maxModelAbs = 0.0f;              // ... of the model's max |vertex coordinate|
    std::vector<std::uint64_t> scratch;    // (fvOrder, tri) keys
    std::vector<std::uint32_t> nearIdx;    // Filter scratch (narrow builds): entries meeting the shapes' union
    std::vector<std::uint32_t> pairEntry, pairShape;   // Filter scratch
};

// fast / modelAbs: per pack model triangle (tw::exact::MakeFastTri) and per model (max |coordinate|).
// patchOf (optional): per global ModelTris index its patch id, copied into Memo::patch.
void BuildMemo(const world::WorldPackView &view, const Bvh4 &bvh, const exact::FastTri *fast, const float *modelAbs,
               const Box3 &box, const float origin[3], Memo *memo, const std::uint32_t *patchOf = nullptr);

// One shape against a memo: query box (the stage's inflated AABB) and the support prefilter in world
// space relative to the memo origin.
struct ShapeFilter {
    float lo[3], hi[3];    // query box
    float c[3];            // ellipsoid centre - memo origin
    float q[6];            // world support form Q = R D^2 R^T (Q00 Q11 Q22 Q01 Q02 Q12)
    float reach, low, tol, lazy;
};
// Per shape, from its radii (fixed for a car): once.
struct ShapeConst {
    float r2[3], rmax, rmin, ratio;   // radii squared, max, min, rmax / rmin
};
ShapeConst MakeShapeConst(const float radii[3]);
// ellR: world axes (columns), ellT: world centre; box: the stage's query box of the shape.
void MakeShapeFilter(const Memo &memo, const float ellR[9], const float ellT[3], const ShapeConst &sc, const Box3 &box, ShapeFilter *f);

// Survivors of each shape, memo indices in FV order: idx[k][0 .. n[k]). The buffers only grow.
struct Lists {
    std::vector<std::uint32_t> idx[8];
    std::uint32_t n[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // with a patch table (Memo::pkey): per shape, whether every candidate whose AABB meets its box has the same
    // (instance, patch) key, and that key (kNoKey: no box hit). A function of the world, not of the memo.
    bool boxUniform[8];
    std::uint64_t boxKey[8];
};
constexpr std::uint64_t kNoKey = ~0ull;
// simd: the AVX-512 path, or on narrower builds the native-width box test with the four planes per pair
// in SSE (all paths give the same lists).
// within (optional): a box containing every shape's query box (their union): candidates outside it are
// skipped before the per-shape tests (same lists).
void Filter(Memo &memo, const ShapeFilter shapes[8], Lists *out, bool simd = true, const Box3 *within = nullptr);
// The prefilter alone on a given list of memo indices (no box test); returns the survivors' count.
std::size_t Prefilter(const Memo &memo, const std::uint32_t *list, std::size_t n, const ShapeFilter &shape, std::uint32_t *survivors);

}  // namespace tw::cq

#endif
