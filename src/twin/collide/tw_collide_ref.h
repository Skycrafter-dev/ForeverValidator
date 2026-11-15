// Gate 1' collision reference: FV's collision semantics in double precision (FV-free).
//
// The same operations as the exact stage (tw_collide.cpp, DetectShapesRef): FV's shape poses
// (GmMat3::Set(GmQuat), SetMult with the shape-to-car / rest surface isos, OffsetCurrentSurfaceY),
// GmCollision_Ellipsoid_Mesh's transforms, SSphereMeshCollide::CollideTriangle (FACE, EmitFeature,
// the EmitEndpointB double root), TransformEllipsoidMeshCollisionsToWorldForGmSurf and
// MergeAndAddToCollisions, operation for operation, with every value in double:
//   - geometry is the stored floats (pack model vertices, instance isos, car shape isos and radii), widened;
//   - FV's constants (SurfaceDirectionLengthSquared, CollisionDistance, the 0.8660254 alignment, the
//     NearlyEqual 1e-5) are its float values, widened;
//   - candidates: the stage's BVH4 queried with each shape's box inflated by 5 cm (a superset that no
//     float rounding can change), in FV detection order; every candidate takes FV's full test.
// Each decision FV makes on the way (plane reach, edge reach and sign, the feature branches, the
// minimum distances, the merge's NearlyEqual and alignment tests) reports its margin, so a gate can
// see which contacts sit near a threshold.
#ifndef TW_COLLIDE_REF_H
#define TW_COLLIDE_REF_H

#include <cstdint>
#include <type_traits>
#include <vector>

#include "tw_collide.h"
#include "../ref/tw_sr.h"

namespace tw::collide::ref {

struct ContactD {          // one merged contact (RespContact in double)
    double P[3], N[3], Sep[3];
    float peerZ[3];
    std::uint32_t peerId;  // the trigger id for a trigger contact
    std::uint32_t tri;     // world triangle
    std::uint8_t shape, material, flags;   // flags: model::kRespPrimary | model::kRespTrigger
};

// The decision with the smallest margin (in the units of the quantity tested: unit-space lengths for the
// triangle test, the NearlyEqual band and the cosine for the merge).
enum DecisionKind : std::uint8_t {
    kNone = 0,
    kPlaneReach,       // pd vs 0 and vs 1 (the sphere's plane distance)
    kEdgeReach,        // edgeDistance vs the in-plane reach
    kEdgeSign,         // edgeDistance vs 0 (edge violated)
    kFeatureBranch,    // fromStart / fromEnd vs 0 (vertex or edge feature)
    kFeatureDistance,  // a feature's distance vs 1 and vs its minimum
    kMergeNearlyEqual, // a non-primary's K vs a primary's K (the 1e-5 band)
    kMergeAlignment,   // a non-primary's N . a primary's N vs 0.8660254
};
struct Margin {
    double value = 1e300;
    DecisionKind kind = kNone;
    std::uint8_t shape = 0;
    std::uint32_t tri = 0;
};

struct DetectionD {
    std::vector<ContactD> contacts;   // shape-major, FV order (as the stage's SubstepContacts)
    bool overflow = false;            // more than kRespMaxContacts
    Margin closest;                   // the smallest margin of any decision taken
    Margin perShape[8];               // ... per shape
    std::uint32_t candidates = 0;     // triangle tests run
};

// Car pose in double (pos, quat x y z w, absorb per wheel); the stage supplies the world, its BVH and the car.
void Detect(const ExactCollision &stage, const double pos[3], const double quat[4], const double absorb[4], DetectionD *out);
// The same operations with the stochastic-rounding float (tw_sr.h: each op computed in double and rounded at
// random to one of the two bracketing floats): one sample of a valid float implementation. Seed() first.
// Output values are that float's, widened (exact).
void DetectSR(const ExactCollision &stage, const sr::sr pos[3], const sr::sr quat[4], const sr::sr absorb[4], DetectionD *out);
// Either, by scalar type.
template <class R>
inline void DetectT(const ExactCollision &stage, const R pos[3], const R quat[4], const R absorb[4], DetectionD *out) {
    if constexpr (std::is_same<R, double>::value) Detect(stage, pos, quat, absorb, out);
    else DetectSR(stage, pos, quat, absorb, out);
}

}  // namespace tw::collide::ref

#endif
