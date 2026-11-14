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
namespace hip::facts {

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
    return configuration->tuning;
}

__device__ inline const VehicleWheelDefinition &Wheel(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
    return reinterpret_cast<const VehicleWheelDefinition *>(
            &configuration->wheels.wheels)[index];
}

__device__ inline VehicleWheelAxle WheelAxle(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
    return Wheel(configuration, index).axle;
}

__device__ inline bool WheelKillsLateralSpeed(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
    return Wheel(
            configuration,
            index).killsLateralSpeedOnContact;
}

__device__ inline float WheelRollingRadius(
        const HipPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
    return Wheel(configuration, index).rollingRadius;
}

__device__ inline std::uint32_t WheelForceMode(
        const HipPackedStaticConfigurationHeader *configuration) {
    return configuration->tuning.wheelForceMode;
}

}  // namespace hip::facts
#endif

}  // namespace forevervalidator::simulation

#endif
