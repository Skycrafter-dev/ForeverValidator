// StadiumCar pack: every tuning value the twin reads, plus folded constants
// (design.md §3.3). Filled by tw_compile_car.cpp from
// src/twin/model/stadiumcar-values.json (FV's own loaders dumped it), bit-exact.
// An FV-side compile from BuildCudaHostStaticConfiguration can fill the same
// struct later (src/twin/fv/); the two are meant to be cross-checked.
#ifndef TW_CAR_PACK_H
#define TW_CAR_PACK_H

#include <cstdint>

#include "tw_curve.h"

namespace tw {
namespace model {

enum TwCurveId : std::uint32_t {
#define TW_CURVE(name, json) kCurve_##name,
#include "curve_ids.inc"
#undef TW_CURVE
    kTwCurveCount
};

struct TwCarWheel {                   // board wheel order 0 FL, 1 FR, 2 RR, 3 RL
    float restPoint[3];               // restSurfacePose.translation (car frame)
    float forceApplicationPoint[3];
    float rollingRadius;
    std::uint32_t axleFront;          // 1 front (axle 1), 0 rear
    std::uint32_t killsLateralSpeedOnContact;
    std::uint32_t surfaceId;          // collision surface id of the wheel tree
};

struct TwCarShape {                   // collision ellipsoids in tree order 0..7
    float rotation[9];                // localPose.rotation basisX, basisY, basisZ (columns)
    float translation[3];             // localPose.translation (wheels: rest point)
    float center[3];                  // localBounds.center
    float radii[3];                   // localBounds.halfExtents
    std::uint32_t role;
    std::uint32_t surfaceType;
    std::uint32_t surfaceMaterial;
    std::int32_t wheelIndex;          // board wheel index for the 4 wheel shapes, -1 for body
};

struct TwCarMaterial {                // vehicle material (by index; see materialIndexByNaturalId)
    std::uint32_t naturalId;
    float blend[4];                   // x top speed, y accel, z brake cap, w side grip
    float fakeContactPeriodX, fakeContactPeriodZ, fakeContactSpeedScale, fakeContactDepthMax;
    float feedbackSpeedDivisor, feedbackScale;
    std::uint32_t usesFakeContactTexture;
};

constexpr int kTwMaxGears = 8;
constexpr int kTwMaxMaterials = 16;
constexpr int kTwNaturalIds = 32;
constexpr int kTwFakeTextureSize = 128;

struct TwCarPack {
    // ---- every scalar of the JSON (tuning.*, vehicle.initialParameters.*, derived.dyna.*,
    //      derived.wheelSet scalars), named after the path, bit-exact ----
#define TW_CAR(member, type, path) TW_CAR_CTYPE_##type member;
#define TW_CAR_CTYPE_F32 float
#define TW_CAR_CTYPE_U32 std::uint32_t
#define TW_CAR_CTYPE_I32 std::int32_t
#define TW_CAR_CTYPE_BOOL std::uint32_t
#include "car_fields.inc"
#undef TW_CAR

    // ---- structured data ----
    std::uint32_t gearCount;
    float gearSpeedRatio[kTwMaxGears], upshiftThreshold[kTwMaxGears], downshiftThreshold[kTwMaxGears];
    float targetInputBias[kTwMaxGears], rpmDelta[kTwMaxGears];
    std::uint32_t rpmWantedCount;
    float rpmWanted[kTwMaxGears];
    TwCurve curve[kTwCurveCount];
    std::uint32_t curvePresent[kTwCurveCount];
    TwCarWheel wheel[4];
    TwCarShape shape[8];
    std::uint32_t materialCount;
    TwCarMaterial material[kTwMaxMaterials];
    std::uint32_t materialIndexByNaturalId[kTwNaturalIds];  // 31 used
    std::uint8_t fakeTextureRed[kTwFakeTextureSize * kTwFakeTextureSize];  // row-major, red channel

    // ---- folded constants (derived once; names say what they are) ----
    float invMass;                    // 1 / derived.dyna.mass
    float invInertiaDiag[3];          // diag of derived.dyna.inverseBodyInertia (0.48, 0.48, 1.2)
    float com[3];                     // derived.dyna.localCenterOfMass
    float wheelLongitudinalSpan;      // rest-point bounding box z extent (WH:857-858)
    float maxAngularSpeed;            // contactResponse.pointImpulseAngularSpeedMax (CAR:389-408)
    float gravityY;                   // -10 (replay_simulation_definition.cpp:58-60)
    float zoneLinearDamping, zoneAngularDamping;  // 1, 1 (replay_simulation_definition.cpp:53-54)
    std::uint32_t fsGroundedBits, ldsGroundedBits, fsAirborneBits, ldsAirborneBits;  // fs/lds pairs (golden codec)
};

}  // namespace model
}  // namespace tw

#endif
