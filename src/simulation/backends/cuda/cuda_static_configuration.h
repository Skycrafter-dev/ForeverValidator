#ifndef FOREVERVALIDATOR_CUDA_STATIC_CONFIGURATION_H
#define FOREVERVALIDATOR_CUDA_STATIC_CONFIGURATION_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "simulation/runtime/replay_simulation_definition.h"

namespace forevervalidator::simulation {

enum class CudaStaticConfigurationBuildResult : std::uint8_t {
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

enum class CudaHandlingSpecialization : std::uint8_t {
    Generic,
    Legacy,
    GearedDriveDry,
    GearedDriveWater,
};

struct CudaStaticConfigurationBuildLimits {
    std::uint32_t maximumCurveKeys = 4096u;
    std::uint32_t maximumTransmissionValues = 64u;
    std::uint32_t maximumMaterials = 512u;
    std::uint32_t maximumMaterialRemapEntries = 65536u;
    std::uint32_t maximumCollisionShapes = 64u;
    std::uint32_t maximumForceFields = 1024u;
    std::uint32_t maximumWaterCells = 16777216u;
    std::uint32_t maximumWaterPlanes = 256u;
};

enum class CudaTuningCurveId : std::uint32_t {
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

inline constexpr std::size_t CudaTuningCurveCount =
        static_cast<std::size_t>(CudaTuningCurveId::Count);
inline constexpr std::uint32_t
        CudaTuningCurvePositionsNondecreasing = 1u;

struct CudaTuningCurve {
    std::uint32_t firstKey = 0u;
    std::uint32_t keyCount = 0u;
    std::uint32_t interpolation = 0u;
    std::uint32_t reserved = 0u;
};

struct CudaTuningCurveKey {
    float position = 0.0f;
    float value = 0.0f;
};

struct CudaTransmissionArray {
    std::uint32_t firstValue = 0u;
    std::uint32_t valueCount = 0u;
};

enum class CudaTransmissionArrayId : std::uint32_t {
    GearSpeedRatio,
    UpshiftThreshold,
    DownshiftThreshold,
    RpmWanted,
    TargetInputBias,
    RpmDelta,
    Count,
};

inline constexpr std::size_t CudaTransmissionArrayCount =
        static_cast<std::size_t>(CudaTransmissionArrayId::Count);

struct CudaVehicleGearedDriveTuning {
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
    std::array<CudaTransmissionArray, CudaTransmissionArrayCount>
            transmissionArrays{};
};

struct CudaVehicleTuning {
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
    CudaVehicleGearedDriveTuning gearedDrive{};
    ReplayVehicleTuningWater water{};
    ReplayVehicleTuningTurbo turbo{};
    ReplayVehicleTuningFeedback feedback{};
    std::array<CudaTuningCurve, CudaTuningCurveCount> curves{};
};

struct CudaVehicleCollisionShape {
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

enum class CudaForceFieldType : std::uint32_t {
    Uniform,
    Ball,
};

struct CudaForceField {
    CudaForceFieldType type = CudaForceFieldType::Uniform;
    GmVec3 vector{};
    float radius = 0.0f;
    float strength = 0.0f;
};

struct CudaWaterGrid {
    bool present = false;
    GmVec2 cellSize{};
    GmVec2 origin{};
    GmNat2 dimensions{};
    std::uint32_t outsideOccupancy = 0u;
    std::uint32_t outsidePlaneIndex = 0u;
    float surfaceHeight = 0.0f;
    float secondaryCullHeight = 0.0f;
};

struct CudaHostStaticConfiguration {
    static constexpr std::uint32_t SchemaVersion = 2u;

    std::uint32_t schemaVersion = SchemaVersion;
    std::uint64_t deterministicHash = 0u;
    VehicleInitialParameters initialParameters{};
    ReplayDynaParameters dynaParameters{};
    CudaVehicleTuning tuning{};
    VehicleWheelSetDefinition wheels{};
    float zoneLinearDampingCoefficient = 1.0f;
    float zoneAngularDampingCoefficient = 1.0f;
    CudaWaterGrid water{};
    std::array<CudaTransmissionArray, CudaTransmissionArrayCount>
            transmissionArrays{};
    std::vector<CudaTuningCurveKey> curveKeys;
    std::vector<float> transmissionValues;
    std::vector<CudaVehicleCollisionShape> collisionShapes;
    std::vector<VehicleMaterialDefinition> materials;
    std::vector<std::uint32_t> materialIndexByNaturalId;
    std::vector<std::uint8_t> fakeContactTextureRgb;
    std::vector<CudaForceField> forceFields;
    std::vector<std::uint8_t> waterOccupancy;
    std::vector<std::uint8_t> waterPlaneIndices;
    std::vector<GmVec4> waterPlanes;

    void Clear() noexcept;
    bool Valid(const CudaStaticConfigurationBuildLimits &limits = {})
            const noexcept;
};

CudaStaticConfigurationBuildResult BuildCudaHostStaticConfiguration(
        const ReplaySimulationDefinition &source,
        CudaHostStaticConfiguration *destination,
        const CudaStaticConfigurationBuildLimits &limits = {}) noexcept;

struct CudaStaticConfigurationSection {
    std::uint64_t offset = 0u;
    std::uint32_t count = 0u;
    std::uint32_t stride = 0u;
};

struct CudaPackedStaticConfigurationHeader {
    static constexpr std::uint32_t SchemaVersion = 2u;
    static constexpr std::uint64_t Magic = 0x4656435544414346ull;

    std::uint64_t magic = Magic;
    std::uint32_t schemaVersion = SchemaVersion;
    std::uint32_t headerSize = sizeof(CudaPackedStaticConfigurationHeader);
    std::uint64_t totalSize = 0u;
    std::uint64_t deterministicHash = 0u;
    VehicleInitialParameters initialParameters{};
    ReplayDynaParameters dynaParameters{};
    CudaVehicleTuning tuning{};
    VehicleWheelSetDefinition wheels{};
    float zoneLinearDampingCoefficient = 1.0f;
    float zoneAngularDampingCoefficient = 1.0f;
    CudaWaterGrid water{};
    CudaStaticConfigurationSection curveKeys{};
    CudaStaticConfigurationSection transmissionValues{};
    CudaStaticConfigurationSection collisionShapes{};
    CudaStaticConfigurationSection materials{};
    CudaStaticConfigurationSection materialIndexByNaturalId{};
    CudaStaticConfigurationSection fakeContactTextureRgb{};
    CudaStaticConfigurationSection forceFields{};
    CudaStaticConfigurationSection waterOccupancy{};
    CudaStaticConfigurationSection waterPlaneIndices{};
    CudaStaticConfigurationSection waterPlanes{};
};

bool PackCudaStaticConfiguration(
        const CudaHostStaticConfiguration &source,
        std::vector<std::byte> *destination,
        CudaPackedStaticConfigurationHeader *header = nullptr) noexcept;

#if defined(__CUDACC__)
namespace cuda::facts {

__device__ inline const GmBoxAligned &ShapeLocalBounds(
        std::uint32_t,
        const CudaVehicleCollisionShape &shape) {
    return shape.localBounds;
}

__device__ inline const GmIso4 &ShapeBodyPose(
        std::uint32_t,
        const CudaVehicleCollisionShape &shape) {
    return shape.bodyPose;
}

__device__ inline std::uint32_t ShapeWheelIndex(
        std::uint32_t,
        const CudaVehicleCollisionShape &shape) {
    return shape.wheelIndex;
}

__device__ inline std::uint32_t ShapeSurfaceMaterial(
        std::uint32_t,
        const CudaVehicleCollisionShape &shape) {
    return shape.surfaceMaterial;
}

__device__ inline const CudaVehicleTuning &Tuning(
        const CudaPackedStaticConfigurationHeader *configuration) {
    return configuration->tuning;
}

__device__ inline const VehicleWheelDefinition &Wheel(
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
    return reinterpret_cast<const VehicleWheelDefinition *>(
            &configuration->wheels.wheels)[index];
}

__device__ inline VehicleWheelAxle WheelAxle(
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
    return Wheel(configuration, index).axle;
}

__device__ inline bool WheelKillsLateralSpeed(
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
    return Wheel(
            configuration,
            index).killsLateralSpeedOnContact;
}

__device__ inline float WheelRollingRadius(
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t index) {
    return Wheel(configuration, index).rollingRadius;
}

__device__ inline std::uint32_t WheelForceMode(
        const CudaPackedStaticConfigurationHeader *configuration) {
    return configuration->tuning.wheelForceMode;
}

}  // namespace cuda::facts
#endif

}  // namespace forevervalidator::simulation

#endif
