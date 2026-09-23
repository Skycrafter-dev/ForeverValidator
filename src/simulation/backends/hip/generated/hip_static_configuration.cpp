// Generated from src/simulation/backends/cuda/cuda_static_configuration.cpp by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_static_configuration.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <variant>

namespace forevervalidator::simulation {
namespace {

constexpr std::size_t ConfigurationAlignment = 16u;
constexpr std::uint64_t FnvOffset = 1469598103934665603ull;
constexpr std::uint64_t FnvPrime = 1099511628211ull;

std::size_t Align(std::size_t value) {
    return (value + ConfigurationAlignment - 1u) &
           ~(ConfigurationAlignment - 1u);
}

template<typename T>
void ClearPadding(T &value) {
#if defined(__has_builtin)
#if __has_builtin(__builtin_clear_padding)
    __builtin_clear_padding(&value);
#endif
#elif defined(__GNUC__) && __GNUC__ >= 12
    __builtin_clear_padding(&value);
#endif
}

template<typename T>
bool AppendSection(const std::vector<T> &source,
                   std::vector<std::byte> &destination,
                   HipStaticConfigurationSection &section) {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(!std::is_pointer_v<T>);
    if (source.size() > std::numeric_limits<std::uint32_t>::max() ||
        source.size() >
                std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return false;
    }
    const std::size_t offset = Align(destination.size());
    const std::size_t bytes = source.size() * sizeof(T);
    if (offset > std::numeric_limits<std::size_t>::max() - bytes) {
        return false;
    }
    destination.resize(offset + bytes);
    for (std::size_t index = 0u; index < source.size(); ++index) {
        T value = source[index];
        ClearPadding(value);
        std::memcpy(destination.data() + offset + index * sizeof(T),
                    &value, sizeof(value));
    }
    section.offset = offset;
    section.count = static_cast<std::uint32_t>(source.size());
    section.stride = sizeof(T);
    return true;
}

std::uint64_t HashBytes(const std::vector<std::byte> &bytes) {
    std::uint64_t value = FnvOffset;
    for (std::byte byte : bytes) {
        value ^= static_cast<std::uint8_t>(byte);
        value *= FnvPrime;
    }
    return value;
}

template<typename T>
bool Fits(const std::vector<T> &values, std::uint32_t limit) {
    return values.size() <= limit;
}

bool SliceValid(std::uint32_t first,
                std::uint32_t count,
                std::size_t total) {
    return first <= total && count <= total - first;
}

void CopyGearedDrive(const ReplayVehicleGearedDriveTuning &source,
                     HipVehicleGearedDriveTuning &target) {
    target.forwardAccelBase = source.forwardAccelBase;
    target.forwardAccelSpeedCoef = source.forwardAccelSpeedCoef;
    target.forwardAccelCapWhenSlipping =
            source.forwardAccelCapWhenSlipping;
    target.forwardAccelCap = source.forwardAccelCap;
    target.speedLimitForce = source.speedLimitForce;
    target.forceZScale = source.forceZScale;
    target.sideForceToDriveTorqueScale =
            source.sideForceToDriveTorqueScale;
    target.slippingSteerTorqueScale = source.slippingSteerTorqueScale;
    target.lateralForceScale = source.lateralForceScale;
    target.slippingSideFrictionScale =
            source.slippingSideFrictionScale;
    target.sideFrictionSlipBlend = source.sideFrictionSlipBlend;
    target.driveSideFrictionSlipBlend =
            source.driveSideFrictionSlipBlend;
    target.slipRatioScale = source.slipRatioScale;
    target.currentForceTorqueMin = source.currentForceTorqueMin;
    target.currentTorqueXScale = source.currentTorqueXScale;
    target.currentTorqueZScale = source.currentTorqueZScale;
    target.perSlippingWheelAccelScale =
            source.perSlippingWheelAccelScale;
    target.lowSpeedBSlippingGripScale =
            source.lowSpeedBSlippingGripScale;
    target.forwardAccelCapWhenSlippingReverse =
            source.forwardAccelCapWhenSlippingReverse;
    target.forwardAccelCapReverse = source.forwardAccelCapReverse;
    target.dirtSlideSideForceScale = source.dirtSlideSideForceScale;
    target.dirtSlideGateScale = source.dirtSlideGateScale;
    target.dirtSlideForwardForceScale =
            source.dirtSlideForwardForceScale;
    target.dirtSlideForwardGateScale =
            source.dirtSlideForwardGateScale;
    target.reverseSpeedNorm = source.transmission.reverseSpeedNorm;
    target.burnout = source.burnout;
    target.input = source.input;
}

void CopyTuningScalars(const ReplayVehicleTuningDefinition &source,
                       HipVehicleTuning &target) {
    target.engineSpeedNorm = source.engineSpeedNorm;
    target.lowSpeedFrictionMagnitude =
            source.lowSpeedFrictionMagnitude;
    target.lowSpeedLinearDamping = source.lowSpeedLinearDamping;
    target.wheelForceMode =
            static_cast<std::uint32_t>(source.wheelForceMode);
    target.handlingModel =
            static_cast<std::uint32_t>(source.handlingModel);
    target.visual = source.visual;
    target.steering = source.steering;
    target.suspension = source.suspension;
    target.contactResponse = source.contactResponse;
    target.bodyAirResponse = source.bodyAirResponse;
    target.radiusSteering = source.radiusSteering;
    target.slipResponse = source.slipResponse;
    CopyGearedDrive(source.gearedDrive, target.gearedDrive);
    target.water = source.water;
    target.turbo = source.turbo;
    target.feedback = source.feedback;
}

HipStaticConfigurationBuildResult AddCurve(
        const std::optional<ReplayTuningCurveDefinition> &source,
        HipTuningCurveId id,
        HipHostStaticConfiguration &target,
        const HipStaticConfigurationBuildLimits &limits) {
    HipTuningCurve &curve =
            target.tuning.curves[static_cast<std::size_t>(id)];
    curve.firstKey =
            static_cast<std::uint32_t>(target.curveKeys.size());
    curve.interpolation = source.has_value()
            ? static_cast<std::uint32_t>(source->interpolation)
            : 0u;
    curve.reserved = 0u;
    if (!source.has_value()) {
        return HipStaticConfigurationBuildResult::Success;
    }
    if (source->keys.size() >
                limits.maximumCurveKeys - target.curveKeys.size()) {
        return HipStaticConfigurationBuildResult::CurveOverflow;
    }
    curve.keyCount = static_cast<std::uint32_t>(source->keys.size());
    bool positionsNondecreasing = true;
    float previousPosition = 0.0f;
    bool hasPreviousPosition = false;
    for (const ReplayTuningCurveKey &key : source->keys) {
        if (!std::isfinite(key.position) ||
            (hasPreviousPosition &&
             previousPosition > key.position)) {
            positionsNondecreasing = false;
        }
        previousPosition = key.position;
        hasPreviousPosition = true;
        target.curveKeys.push_back({key.position, key.value});
    }
    if (positionsNondecreasing) {
        curve.reserved |=
                HipTuningCurvePositionsNondecreasing;
    }
    return HipStaticConfigurationBuildResult::Success;
}

HipStaticConfigurationBuildResult AddCurves(
        const ReplayVehicleTuningCurves &source,
        HipHostStaticConfiguration &target,
        const HipStaticConfigurationBuildLimits &limits) {
#define ADD_CURVE(field, id)                                                   \
    do {                                                                       \
        const HipStaticConfigurationBuildResult result =                      \
                AddCurve(source.field, HipTuningCurveId::id,                  \
                         target, limits);                                       \
        if (result != HipStaticConfigurationBuildResult::Success) {           \
            return result;                                                     \
        }                                                                      \
    } while (false)
    ADD_CURVE(lateralContactSlowDownFromSpeed,
              LateralContactSlowDownFromSpeed);
    ADD_CURVE(maxSideFrictionFromSpeed, MaxSideFrictionFromSpeed);
    ADD_CURVE(rolloverLateralFromSpeed, RolloverLateralFromSpeed);
    ADD_CURVE(rolloverLateralCoefficientFromAngle,
              RolloverLateralCoefficientFromAngle);
    ADD_CURVE(wheelVisualSteerAngleFromSpeed,
              WheelVisualSteerAngleFromSpeed);
    ADD_CURVE(steeringDriveTorqueFromSpeed,
              SteeringDriveTorqueFromSpeed);
    ADD_CURVE(steerSlowDownFromSpeed, SteerSlowDownFromSpeed);
    ADD_CURVE(suspensionDamperAbsorbModulation,
              SuspensionDamperAbsorbModulation);
    ADD_CURVE(airControlZScale, AirControlZScale);
    ADD_CURVE(radiusSteeringRadiusFromSpeed,
              RadiusSteeringRadiusFromSpeed);
    ADD_CURVE(radiusSteeringMaxFrictionFromSpeed,
              RadiusSteeringMaxFrictionFromSpeed);
    ADD_CURVE(slipResponseAccelFromSpeed, SlipResponseAccelFromSpeed);
    ADD_CURVE(slipResponseSlippingAccelFromSpeed,
              SlipResponseSlippingAccelFromSpeed);
    ADD_CURVE(reverseGearAccelFromSpeed, ReverseGearAccelFromSpeed);
    ADD_CURVE(burnoutRolloverLateralFromSpeedRatio,
              BurnoutRolloverLateralFromSpeedRatio);
    ADD_CURVE(burnoutRadiusFromSpeed, BurnoutRadiusFromSpeed);
    ADD_CURVE(burnoutLateralSpeedFromRadius,
              BurnoutLateralSpeedFromRadius);
    ADD_CURVE(donutRolloverFromSpeed, DonutRolloverFromSpeed);
    ADD_CURVE(burnoutRolloverFromSpeed, BurnoutRolloverFromSpeed);
    ADD_CURVE(splashVerticalImpulse, SplashVerticalImpulse);
    ADD_CURVE(splashHorizontalImpulse, SplashHorizontalImpulse);
    ADD_CURVE(waterFrictionFromSpeed, WaterFrictionFromSpeed);
    ADD_CURVE(surfaceFeedback, SurfaceFeedback);
    ADD_CURVE(vehicleFeedbackRamp1, VehicleFeedbackRamp1);
    ADD_CURVE(vehicleFeedbackRamp0, VehicleFeedbackRamp0);
    ADD_CURVE(vehicleDefault30To100, VehicleDefault30To100);
#undef ADD_CURVE
    return HipStaticConfigurationBuildResult::Success;
}

HipStaticConfigurationBuildResult AddTransmissionArray(
        const std::vector<float> &source,
        HipTransmissionArrayId id,
        HipHostStaticConfiguration &target,
        const HipStaticConfigurationBuildLimits &limits) {
    if (source.size() >
            limits.maximumTransmissionValues -
                    target.transmissionValues.size()) {
        return HipStaticConfigurationBuildResult::TransmissionOverflow;
    }
    HipTransmissionArray descriptor;
    descriptor.firstValue =
            static_cast<std::uint32_t>(target.transmissionValues.size());
    descriptor.valueCount = static_cast<std::uint32_t>(source.size());
    target.transmissionValues.insert(
            target.transmissionValues.end(), source.begin(), source.end());
    target.transmissionArrays[static_cast<std::size_t>(id)] = descriptor;
    target.tuning.gearedDrive.transmissionArrays[
            static_cast<std::size_t>(id)] = descriptor;
    return HipStaticConfigurationBuildResult::Success;
}

HipStaticConfigurationBuildResult AddTransmission(
        const ReplayVehicleTransmissionTuning &source,
        HipHostStaticConfiguration &target,
        const HipStaticConfigurationBuildLimits &limits) {
#define ADD_TRANSMISSION(field, id)                                            \
    do {                                                                       \
        const HipStaticConfigurationBuildResult result =                      \
                AddTransmissionArray(source.field,                             \
                                     HipTransmissionArrayId::id,              \
                                     target, limits);                          \
        if (result != HipStaticConfigurationBuildResult::Success) {           \
            return result;                                                     \
        }                                                                      \
    } while (false)
    ADD_TRANSMISSION(gearSpeedRatio, GearSpeedRatio);
    ADD_TRANSMISSION(upshiftThreshold, UpshiftThreshold);
    ADD_TRANSMISSION(downshiftThreshold, DownshiftThreshold);
    ADD_TRANSMISSION(rpmWanted, RpmWanted);
    ADD_TRANSMISSION(targetInputBias, TargetInputBias);
    ADD_TRANSMISSION(rpmDelta, RpmDelta);
#undef ADD_TRANSMISSION
    return HipStaticConfigurationBuildResult::Success;
}

}  // namespace

void HipHostStaticConfiguration::Clear() noexcept {
    *this = {};
}

bool HipHostStaticConfiguration::Valid(
        const HipStaticConfigurationBuildLimits &limits) const noexcept {
    if (schemaVersion != SchemaVersion ||
        !Fits(curveKeys, limits.maximumCurveKeys) ||
        !Fits(transmissionValues, limits.maximumTransmissionValues) ||
        !Fits(materials, limits.maximumMaterials) ||
        !Fits(materialIndexByNaturalId,
              limits.maximumMaterialRemapEntries) ||
        !Fits(collisionShapes, limits.maximumCollisionShapes) ||
        !Fits(forceFields, limits.maximumForceFields) ||
        !Fits(waterOccupancy, limits.maximumWaterCells) ||
        !Fits(waterPlaneIndices, limits.maximumWaterCells) ||
        !Fits(waterPlanes, limits.maximumWaterPlanes)) {
        return false;
    }
    for (const HipTuningCurve &curve : tuning.curves) {
        if (!SliceValid(curve.firstKey, curve.keyCount,
                        curveKeys.size()) ||
            curve.interpolation >
                    static_cast<std::uint32_t>(
                            ReplayTuningCurveInterpolation::Linear) ||
            (curve.reserved &
             ~HipTuningCurvePositionsNondecreasing) != 0u) {
            return false;
        }
    }
    for (std::size_t index = 0u;
         index < transmissionArrays.size(); ++index) {
        const HipTransmissionArray &array = transmissionArrays[index];
        const HipTransmissionArray &nested =
                tuning.gearedDrive.transmissionArrays[index];
        if (!SliceValid(array.firstValue, array.valueCount,
                        transmissionValues.size()) ||
            array.firstValue != nested.firstValue ||
            array.valueCount != nested.valueCount) {
            return false;
        }
    }
    for (std::size_t index = 0u; index < collisionShapes.size(); ++index) {
        const HipVehicleCollisionShape &shape = collisionShapes[index];
        if ((shape.parentShapeIndex != UINT32_MAX &&
             shape.parentShapeIndex >= index) ||
            shape.archiveOrder != index ||
            (shape.wheelIndex != UINT32_MAX &&
             shape.wheelIndex >=
                     VehicleWheelSetDefinition::OfficialWheelCount)) {
            return false;
        }
    }
    if (water.present) {
        const std::uint64_t cellCount =
                static_cast<std::uint64_t>(water.dimensions.x) *
                static_cast<std::uint64_t>(water.dimensions.y);
        if (cellCount != waterOccupancy.size() ||
            (!waterPlaneIndices.empty() &&
             waterPlaneIndices.size() != cellCount)) {
            return false;
        }
    } else if (!waterOccupancy.empty() ||
               !waterPlaneIndices.empty() || !waterPlanes.empty()) {
        return false;
    }
    if (!fakeContactTextureRgb.empty() &&
        fakeContactTextureRgb.size() !=
                VehicleFakeContactTexturePixelBytes) {
        return false;
    }
    return true;
}

HipStaticConfigurationBuildResult BuildHipHostStaticConfiguration(
        const ReplaySimulationDefinition &source,
        HipHostStaticConfiguration *destination,
        const HipStaticConfigurationBuildLimits &limits) noexcept {
    if (destination == nullptr ||
        !source.vehicle.wheels.IsComplete() ||
        !source.vehicle.collisionModel.IsComplete() ||
        !source.vehicle.materials.IsValid()) {
        return HipStaticConfigurationBuildResult::InvalidSource;
    }
    try {
        HipHostStaticConfiguration result;
        result.initialParameters = source.vehicle.initialParameters;
        result.dynaParameters = source.vehicle.dynaParameters;
        result.wheels = source.vehicle.wheels;
        result.zoneLinearDampingCoefficient =
                source.environment.zoneLinearDampingCoefficient;
        result.zoneAngularDampingCoefficient =
                source.environment.zoneAngularDampingCoefficient;
        CopyTuningScalars(source.vehicle.tuning, result.tuning);

        HipStaticConfigurationBuildResult build = AddCurves(
                source.vehicle.tuning.curves, result, limits);
        if (build != HipStaticConfigurationBuildResult::Success) {
            return build;
        }
        build = AddTransmission(
                source.vehicle.tuning.gearedDrive.transmission,
                result, limits);
        if (build != HipStaticConfigurationBuildResult::Success) {
            return build;
        }

        const auto &sourceShapes =
                source.vehicle.collisionModel.ShapesInArchiveOrder();
        if (!Fits(sourceShapes, limits.maximumCollisionShapes)) {
            return HipStaticConfigurationBuildResult::
                    CollisionShapeOverflow;
        }
        result.collisionShapes.reserve(sourceShapes.size());
        for (std::size_t index = 0u; index < sourceShapes.size(); ++index) {
            const VehicleCollisionShapeEntry &sourceShape =
                    sourceShapes[index];
            HipVehicleCollisionShape shape;
            shape.surfaceType = sourceShape.shape.surfaceType;
            shape.localPose = sourceShape.shape.localPose;
            shape.localBounds = sourceShape.shape.localBounds;
            shape.localMaterial = sourceShape.shape.localMaterial;
            shape.surfaceMaterial =
                    static_cast<std::uint32_t>(
                            sourceShape.shape.surfaceMaterial);
            shape.wheelIndex = UINT32_MAX;
            if (sourceShape.wheelRole.has_value()) {
                for (std::uint32_t wheel = 0u;
                     wheel <
                             VehicleWheelSetDefinition::
                                     OfficialWheelCount;
                     ++wheel) {
                    if (result.wheels.wheels[wheel].collisionRole ==
                        *sourceShape.wheelRole) {
                        shape.wheelIndex = wheel;
                        break;
                    }
                }
                if (shape.wheelIndex == UINT32_MAX) {
                    return HipStaticConfigurationBuildResult::
                            InvalidSource;
                }
            }
            shape.parentShapeIndex =
                    sourceShape.parentShapeIndex.has_value()
                    ? static_cast<std::uint32_t>(
                              *sourceShape.parentShapeIndex)
                    : UINT32_MAX;
            shape.archiveOrder = static_cast<std::uint32_t>(index);
            if (shape.parentShapeIndex == UINT32_MAX) {
                shape.bodyPose = shape.localPose;
            } else {
                if (shape.parentShapeIndex >=
                    result.collisionShapes.size()) {
                    return HipStaticConfigurationBuildResult::
                            InvalidSource;
                }
                shape.bodyPose.SetMult(
                        shape.localPose,
                        result.collisionShapes[
                                shape.parentShapeIndex].bodyPose);
            }
            result.collisionShapes.push_back(shape);
        }
        std::vector<std::vector<std::uint32_t>> shapeChildren(
                result.collisionShapes.size());
        for (std::uint32_t index = 0u;
             index < result.collisionShapes.size(); ++index) {
            const std::uint32_t parent =
                    result.collisionShapes[index].parentShapeIndex;
            if (parent != UINT32_MAX) {
                shapeChildren[parent].push_back(index);
            }
        }
        std::uint32_t traversalOrder = 0u;
        const auto assignTraversal =
                [&](auto &&self, std::uint32_t index) -> void {
            for (std::uint32_t child : shapeChildren[index]) {
                self(self, child);
            }
            result.collisionShapes[index].traversalOrder =
                    traversalOrder++;
        };
        for (std::uint32_t index = 0u;
             index < result.collisionShapes.size(); ++index) {
            if (result.collisionShapes[index].parentShapeIndex ==
                UINT32_MAX) {
                assignTraversal(assignTraversal, index);
            }
        }

        if (!Fits(source.vehicle.materials.materials,
                  limits.maximumMaterials)) {
            return HipStaticConfigurationBuildResult::MaterialOverflow;
        }
        if (!Fits(source.vehicle.materials.materialIndexByNaturalId,
                  limits.maximumMaterialRemapEntries)) {
            return HipStaticConfigurationBuildResult::MaterialOverflow;
        }
        result.materials = source.vehicle.materials.materials;
        result.materialIndexByNaturalId =
                source.vehicle.materials.materialIndexByNaturalId;
        if (source.vehicle.materials.fakeContactTexture.has_value()) {
            result.fakeContactTextureRgb =
                    source.vehicle.materials.fakeContactTexture->
                            rgbPixels;
        }

        if (!Fits(source.environment.forceFields,
                  limits.maximumForceFields)) {
            return HipStaticConfigurationBuildResult::ForceFieldOverflow;
        }
        result.forceFields.reserve(source.environment.forceFields.size());
        for (const ReplayForceFieldDefinition &field :
             source.environment.forceFields) {
            HipForceField flattened;
            if (const auto *uniform =
                        std::get_if<ReplayUniformForceFieldDefinition>(
                                &field)) {
                flattened.type = HipForceFieldType::Uniform;
                flattened.vector = uniform->acceleration;
            } else if (const auto *ball =
                               std::get_if<
                                       ReplayBallForceFieldDefinition>(
                                       &field)) {
                flattened.type = HipForceFieldType::Ball;
                flattened.vector = ball->center;
                flattened.radius = ball->radius;
                flattened.strength = ball->strength;
            } else {
                return HipStaticConfigurationBuildResult::InvalidSource;
            }
            result.forceFields.push_back(flattened);
        }

        if (source.environment.water.has_value()) {
            const ReplayWaterDefinition &sourceWater =
                    *source.environment.water;
            const WaterOccupancyGrid &grid =
                    sourceWater.OccupancyGrid();
            const std::uint64_t cellCount =
                    static_cast<std::uint64_t>(grid.dimensions.x) *
                    static_cast<std::uint64_t>(grid.dimensions.y);
            if (cellCount > limits.maximumWaterCells ||
                grid.cells.size() != cellCount ||
                (!grid.planeIndices.empty() &&
                 grid.planeIndices.size() != cellCount)) {
                return HipStaticConfigurationBuildResult::
                        WaterCellOverflow;
            }
            if (!Fits(grid.planes, limits.maximumWaterPlanes)) {
                return HipStaticConfigurationBuildResult::
                        WaterPlaneOverflow;
            }
            result.water.present = true;
            result.water.cellSize = grid.cellSize;
            result.water.origin = grid.origin;
            result.water.dimensions = grid.dimensions;
            result.water.outsideOccupancy =
                    static_cast<std::uint32_t>(grid.outside);
            result.water.outsidePlaneIndex = grid.outsidePlaneIndex;
            result.water.surfaceHeight = sourceWater.SurfaceHeight();
            result.water.secondaryCullHeight =
                    sourceWater.SecondaryCullHeight();
            result.waterOccupancy.reserve(grid.cells.size());
            for (WaterOccupancy occupancy : grid.cells) {
                result.waterOccupancy.push_back(
                        static_cast<std::uint8_t>(occupancy));
            }
            result.waterPlaneIndices = grid.planeIndices;
            result.waterPlanes = grid.planes;
        }

        std::vector<std::byte> canonical;
        if (!PackHipStaticConfiguration(result, &canonical)) {
            return HipStaticConfigurationBuildResult::InvalidSource;
        }
        result.deterministicHash = HashBytes(canonical);
        if (!result.Valid(limits)) {
            return HipStaticConfigurationBuildResult::InvalidSource;
        }
        *destination = std::move(result);
        return HipStaticConfigurationBuildResult::Success;
    } catch (const std::bad_alloc &) {
        return HipStaticConfigurationBuildResult::AllocationFailed;
    } catch (...) {
        return HipStaticConfigurationBuildResult::InvalidSource;
    }
}

bool PackHipStaticConfiguration(
        const HipHostStaticConfiguration &source,
        std::vector<std::byte> *destination,
        HipPackedStaticConfigurationHeader *header) noexcept {
    if (destination == nullptr || !source.Valid()) {
        return false;
    }
    try {
        HipPackedStaticConfigurationHeader packed;
        packed.deterministicHash = source.deterministicHash;
        packed.initialParameters = source.initialParameters;
        packed.dynaParameters = source.dynaParameters;
        packed.tuning = source.tuning;
        packed.wheels = source.wheels;
        packed.zoneLinearDampingCoefficient =
                source.zoneLinearDampingCoefficient;
        packed.zoneAngularDampingCoefficient =
                source.zoneAngularDampingCoefficient;
        packed.water = source.water;
        // Collision detection is traversal ordered; make its hot lookup
        // direct while retaining archive identity for diagnostics.
        std::vector<HipVehicleCollisionShape> packedCollisionShapes =
                source.collisionShapes;
        std::sort(
                packedCollisionShapes.begin(),
                packedCollisionShapes.end(),
                [](const HipVehicleCollisionShape &left,
                   const HipVehicleCollisionShape &right) {
                    return left.traversalOrder <
                            right.traversalOrder;
                });
        std::vector<std::uint32_t> packedIndexByArchiveOrder(
                packedCollisionShapes.size());
        for (std::uint32_t index = 0u;
             index < packedCollisionShapes.size(); ++index) {
            packedIndexByArchiveOrder[
                    packedCollisionShapes[index].archiveOrder] =
                    index;
        }
        for (HipVehicleCollisionShape &shape :
             packedCollisionShapes) {
            if (shape.parentShapeIndex != UINT32_MAX) {
                shape.parentShapeIndex =
                        packedIndexByArchiveOrder[
                                shape.parentShapeIndex];
            }
        }
        std::vector<std::byte> bytes(sizeof(packed), std::byte{0});
        if (!AppendSection(source.curveKeys, bytes, packed.curveKeys) ||
            !AppendSection(source.transmissionValues, bytes,
                           packed.transmissionValues) ||
            !AppendSection(packedCollisionShapes, bytes,
                           packed.collisionShapes) ||
            !AppendSection(source.materials, bytes, packed.materials) ||
            !AppendSection(source.materialIndexByNaturalId, bytes,
                           packed.materialIndexByNaturalId) ||
            !AppendSection(source.fakeContactTextureRgb, bytes,
                           packed.fakeContactTextureRgb) ||
            !AppendSection(source.forceFields, bytes,
                           packed.forceFields) ||
            !AppendSection(source.waterOccupancy, bytes,
                           packed.waterOccupancy) ||
            !AppendSection(source.waterPlaneIndices, bytes,
                           packed.waterPlaneIndices) ||
            !AppendSection(source.waterPlanes, bytes,
                           packed.waterPlanes)) {
            return false;
        }
        packed.totalSize = bytes.size();
        ClearPadding(packed);
        std::memcpy(bytes.data(), &packed, sizeof(packed));
        *destination = std::move(bytes);
        if (header != nullptr) {
            *header = packed;
        }
        return true;
    } catch (...) {
        return false;
    }
}

static_assert(std::is_trivially_copyable_v<HipVehicleTuning>);
static_assert(std::is_standard_layout_v<HipVehicleTuning>);
static_assert(std::is_trivially_copyable_v<
              HipPackedStaticConfigurationHeader>);
static_assert(std::is_standard_layout_v<
              HipPackedStaticConfigurationHeader>);

}  // namespace forevervalidator::simulation
