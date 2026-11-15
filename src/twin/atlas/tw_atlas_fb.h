// S2: the atlas fallback (architect D, gen2). FV-free.
//
// For one car shape that the atlas fast path (E, S1) cannot decide, the shape's merged contacts exactly as
// FV's detection and per-shape merge make them (board 20 §3.1-3.4), from a candidate triangle list:
//   - every triangle: the closed-form FACE (tw_atlas_face.h, shared with T1, so a FACE has the same bits in
//     every tier); in the near-tie band, or for a tiny triangle, FV's own triangle test decides;
//   - a triangle outside its edges but in reach (a feature is possible):
//       * the edge side-plane prefilter (the ellipsoid entirely beyond an edge's side plane: no contact);
//       * the 30-degree proof: every feature FV can emit from that plane lies in the unit-space contact disk
//         (radius sqrt(1 - d^2)); the Gram lambda_max bound on the disk's world normals, plus the angle to a
//         FACE of the same shape, below 30 degrees means FV's merge drops it (design-gen2-d.md §2.4.2);
//       * otherwise FV's triangle test (exact::CollideFv on FV's frame: FV's bits);
//   - FV's per-shape merge (primaries in FV detection order, then the non-primaries no primary absorbs).
//     Merge keys: when a feature reaches the merge, each FACE's K and N come from FV's float route too (the
//     NearlyEqual decision depends on rounding residues: gen2-D-on-E.md "Correction"); the FACE's payload
//     stays the closed form.
// Output: shape k's merged contacts appended to `out`, RespContact filled as collide::MergeShapeTo does.
#ifndef TW_ATLAS_FB_H
#define TW_ATLAS_FB_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tw_atlas_face.h"
#include "tw_collide.h"

namespace tw::atlas {

struct FallbackStats {
    std::uint64_t shapes = 0, candidates = 0, survivors = 0;   // survivors of the reject filter
    std::uint64_t faces = 0;          // closed-form FACEs emitted
    std::uint64_t ties = 0;           // triangles sent to FV's form by the near-tie band (or tiny)
    std::uint64_t outside = 0;        // in reach, foot point outside: feature possible
    std::uint64_t sideRejects = 0;    // ... rejected by the side-plane prefilter
    std::uint64_t proofs = 0;         // ... dropped by the 30-degree proof
    std::uint64_t fvForms = 0;        // FV triangle tests run (ties + unproven features + FACE keys)
    std::uint64_t fvFeatures = 0;     // FV non-primary contacts produced
    std::uint64_t fvFaceKeys = 0;     // FV runs for a FACE's merge key
    std::uint64_t mergedOut = 0;      // contacts appended
};

class Fallback {
public:
    // The stage supplies the world (WorldPackView, its BVH) and must outlive the fallback. Builds the
    // per-triangle FaceTri table (64 B per world triangle).
    bool Init(const collide::ExactCollision &stage, std::string *error);

    // Shape k (tree order 0..7) at the substep's poses. cand: world triangle ids, a superset of every
    // triangle that can contact shape k at this pose (any order, duplicates allowed). origin: unused by the
    // bits (FV frames are FV's world floats; FACEs use per-triangle anchors); kept for the interface.
    // sortedUnique: cand is already sorted by TriInfo::fvOrder and free of duplicates (the atlas's lists): no lookups.
    void DetectShape(const collide::ShapePose shapes[8], int k, const std::uint32_t *cand, std::size_t n, const double origin[3],
                     step::SubstepContacts &out, FallbackStats *st = nullptr, bool sortedUnique = false) const;

    // Test helper: shape k's candidates from the stage's BVH (the shape's world AABB inflated by 5 cm).
    void BvhCandidates(const collide::ShapePose &sp, std::vector<std::uint32_t> *cand) const;

    const std::vector<FaceTri> &Faces() const { return face_; }

    // Per world triangle, the conservative reject filter's record (one cache line, float, world coordinates).
    struct alignas(64) RejRec {
        float n[3], rho;    // plane n . x = rho
        float u[3][3];      // edge side-plane normals
        float c[3];         // edge offsets: u . x - c > 0 outside edge e
    };

private:
    const collide::ExactCollision *stage_ = nullptr;
    const world::WorldPackView *view_ = nullptr;
    std::vector<FaceTri> face_;   // per world triangle
    std::vector<RejRec> rej_;     // per world triangle
};

}  // namespace tw::atlas

#endif
