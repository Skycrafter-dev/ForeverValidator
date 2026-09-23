// Generated from src/simulation/backends/cuda/cuda_static_configuration.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_STATIC_CONFIGURATION_H
#define FOREVERVALIDATOR_HIP_STATIC_CONFIGURATION_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "simulation/runtime/replay_simulation_definition.h"

namespace forevervalidator::simulation {

enum class HipStaticConfigurationBuildResult : std::uint8_t {
    Success,
    InvalidSource,
    CurveOverflow,
    TransmissionOverflow,
    MaterialOverflow,
    CollisionShapeOverflow,
    ForceFieldOverflow,
    WaterCellOverflow,
    WaterPlaneOverflow,
    AllocationFailed,
};

enum class HipHandlingSpecialization : std::uint8_t {
    Generic,
    Legacy,
    GearedDriveDry,
    GearedDriveWater,
};

struct HipStaticConfigurationBuildLimits {
    std::uint32_t maximumCurveKeys = 4096u;
    std::uint32_t maximumTransmissionValues = 64u;
    std::uint32_t maximumMaterials = 512u;
    std::uint32_t maximumMaterialRemapEntries = 65536u;
    std::uint32_t maximumCollisionShapes = 64u;
    std::uint32_t maximumForceFields = 1024u;
    std::uint32_t maximumWaterCells = 16777216u;
    std::uint32_t maximumWaterPlanes = 256u;
};

enum class HipTuningCurveId : std::uint32_t {
    LateralContactSlowDownFromSpeed,
    MaxSideFrictionFromSpeed,
    RolloverLateralFromSpeed,
    RolloverLateralCoefficientFromAngle,
    WheelVisualSteerAngleFromSpeed,
    SteeringDriveTorqueFromSpeed,
    SteerSlowDownFromSpeed,
    SuspensionDamperAbsorbModulation,
    AirControlZScale,
    RadiusSteeringRadiusFromSpeed,
    RadiusSteeringMaxFrictionFromSpeed,
    SlipResponseAccelFromSpeed,
    SlipResponseSlippingAccelFromSpeed,
    ReverseGearAccelFromSpeed,
    BurnoutRolloverLateralFromSpeedRatio,
    BurnoutRadiusFromSpeed,
    BurnoutLateralSpeedFromRadius,
    DonutRolloverFromSpeed,
    BurnoutRolloverFromSpeed,
    SplashVerticalImpulse,
    SplashHorizontalImpulse,
    WaterFrictionFromSpeed,
    SurfaceFeedback,
    VehicleFeedbackRamp1,
    VehicleFeedbackRamp0,
    VehicleDefault30To100,
    Count,
};

inline constexpr std::size_t HipTuningCurveCount =
        static_cast<std::size_t>(HipTuningCurveId::Count);
inline constexpr std::uint32_t
        HipTuningCurvePositionsNondecreasing = 1u;

struct HipTuningCurve {
    std::uint32_t firstKey = 0u;
    std::uint32_t keyCount = 0u;
    std::uint32_t interpolation = 0u;
    std::uint32_t reserved = 0u;
};

struct HipTuningCurveKey {
    float position = 0.0f;
    float value = 0.0f;
};

struct HipTransmissionArray {
    std::uint32_t firstValue = 0u;
    std::uint32_t valueCount = 0u;
};

enum class HipTransmissionArrayId : std::uint32_t {
    GearSpeedRatio,
    UpshiftThreshold,
    DownshiftThreshold,
    RpmWanted,
    TargetInputBias,
    RpmDelta,
    Count,
};

inline constexpr std::size_t HipTransmissionArrayCount =
        static_cast<std::size_t>(HipTransmissionArrayId::Count);

struct HipVehicleGearedDriveTuning {
    float forwardAccelBase = 0.0f;
    float forwardAccelSpeedCoef = 0.0f;
    float forwardAccelCapWhenSlipping = 0.0f;
    float forwardAccelCap = 0.0f;
    float speedLimitForce = 0.0f;
    float forceZScale = 0.0f;
    float sideForceToDriveTorqueScale = 0.0f;
    float slippingSteerTorqueScale = 0.0f;
    float lateralForceScale = 0.0f;
    float slippingSideFrictionScale = 0.0f;
    float sideFrictionSlipBlend = 0.0f;
    float driveSideFrictionSlipBlend = 0.0f;
    float slipRatioScale = 0.0f;
    float currentForceTorqueMin = 0.0f;
    float currentTorqueXScale = 0.0f;
    float currentTorqueZScale = 0.0f;
    float perSlippingWheelAccelScale = 0.0f;
    float lowSpeedBSlippingGripScale = 0.0f;
    float forwardAccelCapWhenSlippingReverse = 0.0f;
    float forwardAccelCapReverse = 0.0f;
    float dirtSlideSideForceScale = 0.0f;
    float dirtSlideGateScale = 0.0f;
    float dirtSlideForwardForceScale = 0.0f;
    float dirtSlideForwardGateScale = 0.0f;
    float reverseSpeedNorm = 0.0f;
    ReplayVehicleBurnoutTuning burnout{};
    ReplayVehicleEngineInputTuning input{};
    std::array<HipTransmissionArray, HipTransmissionArrayCount>
            transmissionArrays{};
};

struct HipVehicleTuning {
    float engineSpeedNorm = 0.0f;
    float lowSpeedFrictionMagnitude = 0.0f;
    float lowSpeedLinearDamping = 0.0f;
    std::uint32_t wheelForceMode = 0u;
    std::uint32_t handlingModel = 0u;
    ReplayVehicleTuningVisualSettings visual{};
    ReplayVehicleTuningSteering steering{};
    ReplayVehicleTuningSuspension suspension{};
    ReplayVehicleTuningContactResponse contactResponse{};
    ReplayVehicleTuningBodyAirResponse bodyAirResponse{};
    ReplayVehicleRadiusSteeringTuning radiusSteering{};
    ReplayVehicleSlipResponseTuning slipResponse{};
    HipVehicleGearedDriveTuning gearedDrive{};
    ReplayVehicleTuningWater water{};
    ReplayVehicleTuningTurbo turbo{};
    ReplayVehicleTuningFeedback feedback{};
    std::array<HipTuningCurve, HipTuningCurveCount> curves{};
};

struct HipVehicleCollisionShape {
    std::uint32_t surfaceType = 0u;
    GmIso4 localPose{};
    GmIso4 bodyPose{};
    GmBoxAligned localBounds{};
    GmLocalMaterialIndex localMaterial{};
    std::uint32_t surfaceMaterial = 0u;
    std::uint32_t wheelIndex = UINT32_MAX;
    std::uint32_t parentShapeIndex = UINT32_MAX;
    std::uint32_t archiveOrder = 0u;
    std::uint32_t traversalOrder = 0u;
};

enum class HipForceFieldType : std::uint32_t {
    Uniform,
    Ball,
};

struct HipForceField {
    HipForceFieldType type = HipForceFieldType::Uniform;
    GmVec3 vector{};
    float radius = 0.0f;
    float strength = 0.0f;
};

struct HipWaterGrid {
    bool present = false;
    GmVec2 cellSize{};
    GmVec2 origin{};
    GmNat2 dimensions{};
    std::uint32_t outsideOccupancy = 0u;
    std::uint32_t outsidePlaneIndex = 0u;
    float surfaceHeight = 0.0f;
    float secondaryCullHeight = 0.0f;
};

struct HipHostStaticConfiguration {
    static constexpr std::uint32_t SchemaVersion = 2u;

    std::uint32_t schemaVersion = SchemaVersion;
    std::uint64_t deterministicHash = 0u;
    VehicleInitialParameters initialParameters{};
    ReplayDynaParameters dynaParameters{};
    HipVehicleTuning tuning{};
    VehicleWheelSetDefinition wheels{};
    float zoneLinearDampingCoefficient = 1.0f;
    float zoneAngularDampingCoefficient = 1.0f;
    HipWaterGrid water{};
    std::array<HipTransmissionArray, HipTransmissionArrayCount>
            transmissionArrays{};
    std::vector<HipTuningCurveKey> curveKeys;
    std::vector<float> transmissionValues;
    std::vector<HipVehicleCollisionShape> collisionShapes;
    std::vector<VehicleMaterialDefinition> materials;
    std::vector<std::uint32_t> materialIndexByNaturalId;
    std::vector<std::uint8_t> fakeContactTextureRgb;
    std::vector<HipForceField> forceFields;
    std::vector<std::uint8_t> waterOccupancy;
    std::vector<std::uint8_t> waterPlaneIndices;
    std::vector<GmVec4> waterPlanes;

    void Clear() noexcept;
    bool Valid(const HipStaticConfigurationBuildLimits &limits = {})
            const noexcept;
};

HipStaticConfigurationBuildResult BuildHipHostStaticConfiguration(
        const ReplaySimulationDefinition &source,
        HipHostStaticConfiguration *destination,
        const HipStaticConfigurationBuildLimits &limits = {}) noexcept;

struct HipStaticConfigurationSection {
    std::uint64_t offset = 0u;
    std::uint32_t count = 0u;
    std::uint32_t stride = 0u;
};

struct HipPackedStaticConfigurationHeader {
    static constexpr std::uint32_t SchemaVersion = 2u;
    static constexpr std::uint64_t Magic = 0x4656435544414346ull;

    std::uint64_t magic = Magic;
    std::uint32_t schemaVersion = SchemaVersion;
    std::uint32_t headerSize = sizeof(HipPackedStaticConfigurationHeader);
    std::uint64_t totalSize = 0u;
    std::uint64_t deterministicHash = 0u;
    VehicleInitialParameters initialParameters{};
    ReplayDynaParameters dynaParameters{};
    HipVehicleTuning tuning{};
    VehicleWheelSetDefinition wheels{};
    float zoneLinearDampingCoefficient = 1.0f;
    float zoneAngularDampingCoefficient = 1.0f;
    HipWaterGrid water{};
    HipStaticConfigurationSection curveKeys{};
    HipStaticConfigurationSection transmissionValues{};
    HipStaticConfigurationSection collisionShapes{};
    HipStaticConfigurationSection materials{};
    HipStaticConfigurationSection materialIndexByNaturalId{};
    HipStaticConfigurationSection fakeContactTextureRgb{};
    HipStaticConfigurationSection forceFields{};
    HipStaticConfigurationSection waterOccupancy{};
    HipStaticConfigurationSection waterPlaneIndices{};
    HipStaticConfigurationSection waterPlanes{};
};

bool PackHipStaticConfiguration(
        const HipHostStaticConfiguration &source,
        std::vector<std::byte> *destination,
        HipPackedStaticConfigurationHeader *header = nullptr) noexcept;

#if defined(__HIPCC__)
namespace hip::research {

#if defined(FOREVERVALIDATOR_HIP_RESEARCH_SESSION_LTO)
extern "C" __device__ std::uint64_t
ForeverValidatorSessionConfigurationBase();
extern "C" __device__ const unsigned char *
ForeverValidatorSessionConfigurationBytes();
extern "C" __device__ const unsigned char *
ForeverValidatorSessionCollisionShapeBytes();
extern "C" __device__ const unsigned char *
ForeverValidatorSessionCurveKeyBytes();
template<std::uint32_t Index>
__device__ std::uint32_t ForeverValidatorSessionTuningWord();
template<std::uint32_t Index>
__device__ std::uint32_t ForeverValidatorSessionShapeWord(
        std::uint32_t shapeIndex);
extern __device__ __constant__
        HipPackedStaticConfigurationHeader StaticConfiguration;
extern __device__ __constant__
        std::uint64_t StaticConfigurationBase;

__device__ inline std::uint64_t SessionConfigurationBase() {
    return ForeverValidatorSessionConfigurationBase();
}
#endif
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_CONSTANT_CONFIGURATION)
__device__ __constant__
        HipPackedStaticConfigurationHeader StaticConfiguration;
__device__ __constant__
        std::uint64_t StaticConfigurationBase;
#endif
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_CONSTANT_COLLISION_SHAPES)
__device__ __constant__
        HipVehicleCollisionShape StaticCollisionShapes[8];
#endif
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_CONSTANT_CURVE_KEYS)
__device__ __constant__
        HipTuningCurveKey StaticCurveKeys[4096];
#endif

}  // namespace hip::research

namespace hip::facts {

#if defined(FOREVERVALIDATOR_HIP_RESEARCH_SESSION_LTO)
template<typename T, std::uint32_t Offset>
__device__ inline T ShapeScalar(std::uint32_t shapeIndex) {
    static_assert(sizeof(T) <= sizeof(std::uint32_t));
    constexpr std::uint32_t shift =
            (Offset & 3u) * 8u;
    const std::uint32_t bits =
            research::ForeverValidatorSessionShapeWord<
                    Offset / 4u>(shapeIndex) >> shift;
    if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
        union {
            std::uint32_t bits;
            T value;
        } exact{bits};
        return exact.value;
    } else {
        constexpr std::uint32_t mask =
                (1u << (sizeof(T) * 8u)) - 1u;
        return static_cast<T>(bits & mask);
    }
}

#define FOREVERVALIDATOR_HIP_SHAPE_FLOAT(path) \
    ShapeScalar< \
            float, \
            __builtin_offsetof(HipVehicleCollisionShape, path)>( \
            shapeIndex)

__device__ inline GmBoxAligned ShapeLocalBounds(
        std::uint32_t shapeIndex,
        const HipVehicleCollisionShape &) {
    return {
            {
                    FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                            localBounds.center.x),
                    FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                            localBounds.center.y),
                    FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                            localBounds.center.z),
            },
            {
                    FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                            localBounds.halfExtents.x),
                    FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                            localBounds.halfExtents.y),
                    FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                            localBounds.halfExtents.z),
            },
    };
}

__device__ inline GmIso4 ShapeBodyPose(
        std::uint32_t shapeIndex,
        const HipVehicleCollisionShape &) {
    return {
            {
                    {
                            FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                                    bodyPose.rotation.basisX.x),
                            FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                                    bodyPose.rotation.basisX.y),
                            FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                                    bodyPose.rotation.basisX.z),
                    },
                    {
                            FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                                    bodyPose.rotation.basisY.x),
                            FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                                    bodyPose.rotation.basisY.y),
                            FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                                    bodyPose.rotation.basisY.z),
                    },
                    {
                            FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                                    bodyPose.rotation.basisZ.x),
                            FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                                    bodyPose.rotation.basisZ.y),
                            FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                                    bodyPose.rotation.basisZ.z),
                    },
            },
            {
                    FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                            bodyPose.translation.x),
                    FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                            bodyPose.translation.y),
                    FOREVERVALIDATOR_HIP_SHAPE_FLOAT(
                            bodyPose.translation.z),
            },
    };
}

__device__ inline std::uint32_t ShapeWheelIndex(
        std::uint32_t shapeIndex,
        const HipVehicleCollisionShape &) {
    return ShapeScalar<
            std::uint32_t,
            __builtin_offsetof(
                    HipVehicleCollisionShape,
                    wheelIndex)>(shapeIndex);
}

__device__ inline std::uint32_t ShapeSurfaceMaterial(
        std::uint32_t shapeIndex,
        const HipVehicleCollisionShape &) {
    return ShapeScalar<
            std::uint32_t,
            __builtin_offsetof(
                    HipVehicleCollisionShape,
                    surfaceMaterial)>(shapeIndex);
}

#undef FOREVERVALIDATOR_HIP_SHAPE_FLOAT

template<typename T, std::uint32_t Offset>
struct TuningFact {
    static_assert(sizeof(T) <= sizeof(std::uint32_t));

    __device__ inline operator T() const {
        constexpr std::uint32_t shift =
                (Offset & 3u) * 8u;
        const std::uint32_t bits =
                research::ForeverValidatorSessionTuningWord<
                        Offset / 4u>() >> shift;
        if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
            union {
                std::uint32_t bits;
                T value;
            } exact{bits};
            return exact.value;
        } else {
            constexpr std::uint32_t mask =
                    (1u << (sizeof(T) * 8u)) - 1u;
            return static_cast<T>(bits & mask);
        }
    }
};

#define FOREVERVALIDATOR_HIP_TUNING_FACT(path, name) \
    TuningFact< \
            decltype(((HipVehicleTuning *)nullptr)->path), \
            __builtin_offsetof(HipVehicleTuning, path)> name

struct BurnoutTuningFacts {
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.angleLimit, angleLimit);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.angleLimitNegative,
            angleLimitNegative);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.angleLimitPositive,
            angleLimitPositive);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.angleReturnQuadratic,
            angleReturnQuadratic);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.angleTorqueScale,
            angleTorqueScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.angularDampingLinear,
            angularDampingLinear);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.donutSpeedHigh, donutSpeedHigh);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.donutSpeedLow, donutSpeedLow);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.driveFadeScale, driveFadeScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.durationTicks, durationTicks);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.exitAccelFadeScale,
            exitAccelFadeScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.exitBonusAccelScale,
            exitBonusAccelScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.exitDurationTicks,
            exitDurationTicks);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.lateralCorrectionScale,
            lateralCorrectionScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.radiusCorrectionScale,
            radiusCorrectionScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.radiusCorrectionSpeedScale,
            radiusCorrectionSpeedScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.radiusMin, radiusMin);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.reverseForceThreshold,
            reverseForceThreshold);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.sideForceFadeScale,
            sideForceFadeScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.tangentAngularDamping,
            tangentAngularDamping);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.burnout.tangentSpeedMax,
            tangentSpeedMax);
};

struct GearedDriveTuningFacts {
    BurnoutTuningFacts burnout;
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.currentForceTorqueMin,
            currentForceTorqueMin);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.currentTorqueXScale, currentTorqueXScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.currentTorqueZScale, currentTorqueZScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.dirtSlideForwardForceScale,
            dirtSlideForwardForceScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.dirtSlideForwardGateScale,
            dirtSlideForwardGateScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.dirtSlideGateScale, dirtSlideGateScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.dirtSlideSideForceScale,
            dirtSlideSideForceScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.driveSideFrictionSlipBlend,
            driveSideFrictionSlipBlend);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.forceZScale, forceZScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.forwardAccelBase, forwardAccelBase);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.forwardAccelCap, forwardAccelCap);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.forwardAccelCapReverse,
            forwardAccelCapReverse);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.forwardAccelCapWhenSlipping,
            forwardAccelCapWhenSlipping);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.forwardAccelCapWhenSlippingReverse,
            forwardAccelCapWhenSlippingReverse);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.forwardAccelSpeedCoef,
            forwardAccelSpeedCoef);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.lateralForceScale, lateralForceScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.lowSpeedBSlippingGripScale,
            lowSpeedBSlippingGripScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.perSlippingWheelAccelScale,
            perSlippingWheelAccelScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.reverseSpeedNorm, reverseSpeedNorm);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.sideForceToDriveTorqueScale,
            sideForceToDriveTorqueScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.sideFrictionSlipBlend,
            sideFrictionSlipBlend);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.slipRatioScale, slipRatioScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.slippingSideFrictionScale,
            slippingSideFrictionScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.slippingSteerTorqueScale,
            slippingSteerTorqueScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            gearedDrive.speedLimitForce, speedLimitForce);
};

struct BodyAirResponseTuningFacts {
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            bodyAirResponse.airControlMemoryTickWindow,
            airControlMemoryTickWindow);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            bodyAirResponse.airControlYSwitchThreshold,
            airControlYSwitchThreshold);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            bodyAirResponse.airTorqueLinearCoef,
            airTorqueLinearCoef);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            bodyAirResponse.airTorqueQuadraticCoef,
            airTorqueQuadraticCoef);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            bodyAirResponse.airborneSolidFeedback0,
            airborneSolidFeedback0);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            bodyAirResponse.airborneSolidFeedback1,
            airborneSolidFeedback1);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            bodyAirResponse.groundedSolidFeedback1,
            groundedSolidFeedback1);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            bodyAirResponse.slopeAdherence1Max,
            slopeAdherence1Max);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            bodyAirResponse.slopeAdherence1Min,
            slopeAdherence1Min);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            bodyAirResponse.slopeAdherence2Max,
            slopeAdherence2Max);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            bodyAirResponse.slopeAdherence2Min,
            slopeAdherence2Min);
};

struct ContactResponseTuningFacts {
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            contactResponse.bodyImpactFeedbackHighThreshold,
            bodyImpactFeedbackHighThreshold);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            contactResponse.bodyImpactFeedbackLowThreshold,
            bodyImpactFeedbackLowThreshold);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            contactResponse.specialContactImpulseMagnitude,
            specialContactImpulseMagnitude);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            contactResponse.specialSolidFeedbackValue,
            specialSolidFeedbackValue);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            contactResponse.wheelImpactFeedbackHighThreshold,
            wheelImpactFeedbackHighThreshold);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            contactResponse.wheelImpactFeedbackLowThreshold,
            wheelImpactFeedbackLowThreshold);
};

struct FeedbackTuningFacts {
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            feedback.forceDivisor, forceDivisor);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            feedback.surfaceBaseRate, surfaceBaseRate);
};

struct SlipResponseTuningFacts {
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            slipResponse.lateralSlowDownTickWindow,
            lateralSlowDownTickWindow);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            slipResponse.longitudinalTorqueScale,
            longitudinalTorqueScale);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            slipResponse.slippingAccelScale, slippingAccelScale);
};

struct SteeringTuningFacts {
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            steering.assistFullSpeed, assistFullSpeed);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            steering.slowDownScale, slowDownScale);
};

struct SuspensionTuningFacts {
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            suspension.damperModulationMaxAbsorb,
            damperModulationMaxAbsorb);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            suspension.damperModulationMinAbsorb,
            damperModulationMinAbsorb);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            suspension.wheelDamperCoef, wheelDamperCoef);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            suspension.wheelRestDamperAbsorb,
            wheelRestDamperAbsorb);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            suspension.wheelSpringCoef, wheelSpringCoef);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            suspension.wheelStaticSpringScale,
            wheelStaticSpringScale);
};

struct TurboTuningFacts {
    FOREVERVALIDATOR_HIP_TUNING_FACT(turbo.durationA, durationA);
    FOREVERVALIDATOR_HIP_TUNING_FACT(turbo.durationB, durationB);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            turbo.impulseScaleA, impulseScaleA);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            turbo.impulseScaleB, impulseScaleB);
};

struct VisualTuningFacts {
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            visual.wheelSpeedBase, wheelSpeedBase);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            visual.wheelSpeedScale, wheelSpeedScale);
};

struct WaterTuningFacts {
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            water.angularLinearDamping, angularLinearDamping);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            water.angularSpeedDamping, angularSpeedDamping);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            water.buoyancyForce, buoyancyForce);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            water.splashHorizontalSpeedThreshold,
            splashHorizontalSpeedThreshold);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            water.splashTotalSpeedThreshold,
            splashTotalSpeedThreshold);
};

struct SessionTuningFacts {
    BodyAirResponseTuningFacts bodyAirResponse;
    ContactResponseTuningFacts contactResponse;
    FeedbackTuningFacts feedback;
    GearedDriveTuningFacts gearedDrive;
    SlipResponseTuningFacts slipResponse;
    SteeringTuningFacts steering;
    SuspensionTuningFacts suspension;
    TurboTuningFacts turbo;
    VisualTuningFacts visual;
    WaterTuningFacts water;
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            engineSpeedNorm, engineSpeedNorm);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            handlingModel, handlingModel);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            lowSpeedFrictionMagnitude, lowSpeedFrictionMagnitude);
    FOREVERVALIDATOR_HIP_TUNING_FACT(
            lowSpeedLinearDamping, lowSpeedLinearDamping);
};

#undef FOREVERVALIDATOR_HIP_TUNING_FACT

__device__ inline SessionTuningFacts Tuning(
        const HipPackedStaticConfigurationHeader *) {
    return {};
}
#else
__device__ inline const GmBoxAligned &ShapeLocalBounds(
        std::uint32_t,
        const HipVehicleCollisionShape &shape) {
    return shape.localBounds;
}

__device__ inline const GmIso4 &ShapeBodyPose(
        std::uint32_t,
        const HipVehicleCollisionShape &shape) {
    return shape.bodyPose;
}

__device__ inline std::uint32_t ShapeWheelIndex(
        std::uint32_t,
        const HipVehicleCollisionShape &shape) {
    return shape.wheelIndex;
}

__device__ inline std::uint32_t ShapeSurfaceMaterial(
        std::uint32_t,
        const HipVehicleCollisionShape &shape) {
    return shape.surfaceMaterial;
}

__device__ inline const HipVehicleTuning &Tuning(
        const HipPackedStaticConfigurationHeader *configuration) {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_CONSTANT_CONFIGURATION)
    return research::StaticConfiguration.tuning;
#else
    return configuration->tuning;
#endif
}
#endif

__device__ inline const VehicleWheelDefinition &Wheel(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_CONSTANT_CONFIGURATION)
    return reinterpret_cast<const VehicleWheelDefinition *>(
            &research::StaticConfiguration.wheels.wheels)[index];
#else
    return reinterpret_cast<const VehicleWheelDefinition *>(
            &configuration->wheels.wheels)[index];
#endif
}

__device__ inline VehicleWheelAxle WheelAxle(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_CANONICAL_WHEEL_FACTS)
    return index < 2u
            ? VehicleWheelAxle::Front
            : VehicleWheelAxle::Rear;
#else
    return Wheel(configuration, index).axle;
#endif
}

__device__ inline bool WheelKillsLateralSpeed(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_CANONICAL_WHEEL_FACTS)
    return true;
#else
    return Wheel(
            configuration,
            index).killsLateralSpeedOnContact;
#endif
}

__device__ inline float WheelRollingRadius(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_CANONICAL_WHEEL_FACTS)
    return Wheel(configuration, 0u).rollingRadius;
#else
    return Wheel(configuration, index).rollingRadius;
#endif
}

__device__ inline std::uint32_t WheelForceMode(
        const HipPackedStaticConfigurationHeader *configuration) {
#if defined(FOREVERVALIDATOR_HIP_RESEARCH_WHEEL_FORCE_MODE)
    return FOREVERVALIDATOR_HIP_RESEARCH_WHEEL_FORCE_MODE;
#else
    return configuration->tuning.wheelForceMode;
#endif
}

}  // namespace hip::facts
#endif

}  // namespace forevervalidator::simulation

#endif
