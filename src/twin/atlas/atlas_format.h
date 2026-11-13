// Contact atlas: the shared, pointer-free records (design-gen2-e.md §3.2, gen2-E-synthesis.md S1). FV-free.
//
// A plate is a plane class: every solid, non-trigger, non-duplicate world triangle whose double-precision
// unit normal and offset agree with the plate's within kPlateAngle / kPlateOffset (any material, no
// connectivity). Each plate has a frame (o, e1, e2, n) and, over its members' footprints, 8 m tiles of
// 32 x 32 cells of 0.25 m. A cell holds, per radius class, the free vertical gap [lo, hi] (along n, from the
// plate origin) that no foreign geometry within that class radius of the cell enters (CellRec). A shape whose
// height range along n lies inside the gap at every cover point is clear of every triangle that is not on
// its home plate (except back faces, which FV culls).
//
// Ownership: this header (format) and atlas_build.* (raw tiles) are E's. Architect C owns the compressed
// store (atlas_store.*): it returns these CellRec values bit for bit. Architect D owns tw_atlas_face.h (the
// closed-form FACE shared by every tier) and tw_atlas_fb.* (S2 fallback).
#ifndef TW_ATLAS_FORMAT_H
#define TW_ATLAS_FORMAT_H

#include <cstdint>

namespace tw::atlas {

constexpr std::uint32_t kAtlasVersion = 1u;

// ---- plates ---------------------------------------------------------------------------------------
constexpr double kPlateAngle = 1e-4;     // |n_member - n_plate| (unit normals, double)
constexpr double kPlateOffset = 2e-4;    // |c_member - c_plate| (m)
constexpr std::uint32_t kNoPlate = 0xFFFFFFFFu;

struct PlateRec {                        // 96 B
    double o[3];                         // origin on the plane (a member vertex, double)
    float n[3], e1[3], e2[3];            // unit normal, in-plane basis (right-handed: e1 x e2 = n)
    float slack;                         // tau + 2.5 alpha + 2e-4 (m): reach / behind margin of the plate test
    float tau, alpha;                    // member-vertex distance from the plane (m); normal spread (unit-vector distance)
    std::uint32_t firstMember, memberCount;   // Atlas::members range
    float tanProof;                      // tan(30 deg - 2 alpha - 2e-3): the T1 merge proof holds when tan(bound) < this
    std::uint32_t pad[3];
};
static_assert(sizeof(PlateRec) == 96, "PlateRec layout");

// A member triangle in its plate's frame: in-plane edge lines (inward distance = a u + b v + c, unit (a, b)),
// its own plane (double-precision normal rounded; offset relative to the plate origin).
struct MemberRec {                       // 64 B
    float ln[3][3];                      // per edge i (v_i -> v_i+1): a, b, c
    float n[3];                          // own unit normal (world)
    float c;                             // own offset: n . x - n . o_plate for x on the triangle (m)
    std::uint32_t tri;                   // world triangle (WorldPackView::tris index)
    std::uint32_t plate;
    std::uint32_t pad;
};
static_assert(sizeof(MemberRec) == 64, "MemberRec layout");

// ---- tiles ----------------------------------------------------------------------------------------
constexpr int kTileCells = 32;           // cells per tile edge
constexpr float kCellM = 0.25f;          // cell edge (m)
constexpr float kTileM = 8.0f;           // tile edge (m) = kTileCells * kCellM
constexpr int kTileBorderCells = 10;     // raw border for the dilation (2.5 m >= the largest class radius)
constexpr int kClasses = 7;              // radius classes 0..6; slot 7 is pad (always fails)
constexpr float kClassR[8] = {0.0f, 0.375f, 0.75f, 1.0f, 1.25f, 1.5f, 1.95f, 1e9f};
constexpr float kHref = 0.5f;            // the free gap is the one around this height above the plate (m)
constexpr float kBackDilation = 1.3f;    // back-face exclusion holds for centres this far outside the tile (cover offsets <= 1.27 m)
constexpr float kCentreLo = -0.5f, kCentreHi = 2.5f;   // centre heights (along n) the tile's back-face exclusion assumes
constexpr float kFieldLo = -2.5f, kFieldHi = 4.5f;     // heights (along n) whose geometry the tile records

// Canonical cell record: int16 mm along n from the plate origin, conservative rounding (lo = ceil, hi = floor),
// clamped to +-32767; slot 7 = {32767, -32767}. A shape passes class k when s_lo_mm > lo[k] and s_hi_mm < hi[k].
struct CellRec {                         // 32 B
    std::int16_t lo[8];
    std::int16_t hi[8];
};
static_assert(sizeof(CellRec) == 32, "CellRec layout");

constexpr std::int16_t kGapNone = 32767;   // lo of a cell (or class) that always fails

}  // namespace tw::atlas

#endif
