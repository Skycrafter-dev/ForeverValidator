#include "simulation/backends/vulkan/vulkan_scene_storage.h"

#include <array>
#include <chrono>
#include <cstring>
#include <utility>
#include <vector>

#include "engine/scene/scene_vehicle_car_tuning_types.h"
#include "simulation/backends/cuda/cuda_state_layout.h"

namespace forevervalidator::simulation {
namespace {

CudaHandlingSpecialization SelectHandlingSpecialization(
        const CudaVehicleTuning &tuning,
        bool waterPresent) {
    if (tuning.handlingModel != static_cast<std::uint32_t>(
                CSceneVehicleCarHandlingModel_GearedDrive)) {
        return CudaHandlingSpecialization::Legacy;
    }
    return waterPresent
            ? CudaHandlingSpecialization::GearedDriveWater
            : CudaHandlingSpecialization::GearedDriveDry;
}

bool CanonicalWheelFacts(const VehicleWheelSetDefinition &wheels) {
    const GmMat3 identityRotation = {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
    };
    for (std::size_t index = 0u; index < wheels.wheels.size(); ++index) {
        const VehicleWheelDefinition &wheel = wheels.wheels[index];
        const VehicleWheelAxle expectedAxle = index < 2u
                ? VehicleWheelAxle::Front
                : VehicleWheelAxle::Rear;
        if (wheel.axle != expectedAxle ||
            !wheel.killsLateralSpeedOnContact ||
            std::memcmp(&wheel.rollingRadius,
                        &wheels.wheels[0].rollingRadius,
                        sizeof(float)) != 0 ||
            std::memcmp(&wheel.restSurfacePose.rotation,
                        &identityRotation, sizeof(GmMat3)) != 0) {
            return false;
        }
    }
    return true;
}

bool CollisionShapeFacts(
        const CudaVehicleCollisionShape *shapes, std::size_t count) {
    constexpr std::uint32_t ExpectedWheelIndices[] = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
            0u, 1u, 3u, 2u};
    if (shapes == nullptr || count != std::size(ExpectedWheelIndices)) {
        return false;
    }
    for (std::size_t index = 0u; index < count; ++index) {
        if (shapes[index].parentShapeIndex != UINT32_MAX ||
            shapes[index].wheelIndex != ExpectedWheelIndices[index]) {
            return false;
        }
    }
    return true;
}

bool StaticFacts(
        const CudaVehicleTuning &tuning,
        const VehicleWheelSetDefinition &wheels,
        const CudaVehicleCollisionShape *shapes,
        std::size_t shapeCount,
        const CudaForceField *fields,
        std::size_t fieldCount) {
    return tuning.wheelForceMode ==
            CSceneVehicleCarWheelForceMode_FollowAbsorbWithImpulse
            && CanonicalWheelFacts(wheels)
            && CollisionShapeFacts(shapes, shapeCount)
            && fields != nullptr && fieldCount == 1u
            && fields[0].type == CudaForceFieldType::Uniform;
}

template<typename T, std::size_t Count>
bool ReadPackedSection(
        const std::byte *data, std::size_t size,
        const CudaStaticConfigurationSection &section,
        std::array<T, Count> *values) {
    if (section.count != Count || section.stride != sizeof(T) ||
        section.offset < sizeof(CudaPackedStaticConfigurationHeader) ||
        section.offset > size ||
        Count > (size - section.offset) / sizeof(T)) {
        return false;
    }
    std::memcpy(values->data(), data + section.offset,
                Count * sizeof(T));
    return true;
}

template<typename Source, typename Pack>
VulkanTransferMetrics PackAndUpload(
        const Source &source,
        Pack pack,
        vulkan::BufferHandle *destination) {
    VulkanTransferMetrics result;
    const auto packStart = std::chrono::steady_clock::now();
    std::vector<std::byte> bytes;
    if (!pack(source, &bytes)) {
        result.diagnostic = "Vulkan immutable data packing failed";
        return result;
    }
    const auto packEnd = std::chrono::steady_clock::now();
    result.packMilliseconds =
            std::chrono::duration<double, std::milli>(
                    packEnd - packStart).count();
    result.hostPackedBytes = bytes.size();
    if (!vulkan::UploadImmutableBuffer(
                bytes.data(), bytes.size(), destination,
                &result.uploadMilliseconds, &result.diagnostic)) {
        destination->reset();
        return result;
    }
    result.deviceBytes = vulkan::BufferSize(*destination);
    result.success = true;
    return result;
}

}  // namespace

VulkanTransferMetrics VulkanDeviceScene::Upload(
        const CudaHostScene &source) noexcept {
    Reset();
    try {
        VulkanTransferMetrics result = PackAndUpload(
                source,
                [](const CudaHostScene &value,
                   std::vector<std::byte> *bytes) {
                    return PackCudaScene(value, bytes);
                },
                &buffer_);
        if (result.success) {
            sceneHash_ = source.deterministicHash;
            result.diagnostic = "Vulkan immutable scene uploaded";
        }
        return result;
    } catch (...) {
        Reset();
        VulkanTransferMetrics result;
        result.diagnostic = "unexpected Vulkan scene upload failure";
        return result;
    }
}

VulkanTransferMetrics VulkanDeviceScene::UploadPacked(
        const std::byte *data, std::size_t size) noexcept {
    Reset();
    VulkanTransferMetrics result;
    CudaPackedSceneHeader header;
    if (data == nullptr || size < sizeof(header)) {
        result.diagnostic = "invalid packed Vulkan scene";
        return result;
    }
    std::memcpy(&header, data, sizeof(header));
    if (header.magic != CudaPackedSceneHeader::Magic ||
        header.schemaVersion != CudaPackedSceneHeader::SchemaVersion ||
        header.headerSize != sizeof(header) ||
        header.totalSize != size) {
        result.diagnostic = "packed Vulkan scene schema mismatch";
        return result;
    }
    result.hostPackedBytes = size;
    result.success = vulkan::UploadImmutableBuffer(
            data, size, &buffer_, &result.uploadMilliseconds,
            &result.diagnostic);
    if (result.success) {
        sceneHash_ = header.deterministicHash;
        result.deviceBytes = DeviceBytes();
        result.diagnostic = "packed Vulkan scene uploaded";
    }
    return result;
}

void VulkanDeviceScene::Reset() noexcept {
    buffer_.reset();
    sceneHash_ = 0u;
}

std::uint64_t VulkanDeviceScene::DeviceBytes() const noexcept {
    return vulkan::BufferSize(buffer_);
}

VulkanTransferMetrics VulkanDeviceStaticConfiguration::Upload(
        const CudaHostStaticConfiguration &source) noexcept {
    Reset();
    try {
        VulkanTransferMetrics result = PackAndUpload(
                source,
                [](const CudaHostStaticConfiguration &value,
                   std::vector<std::byte> *bytes) {
                    return PackCudaStaticConfiguration(value, bytes);
                },
                &buffer_);
        if (result.success) {
            configurationHash_ = source.deterministicHash;
            handlingSpecialization_ = SelectHandlingSpecialization(
                    source.tuning, source.water.present);
            wheels_ = source.wheels;
            steadyVelocityStaticFacts_ = StaticFacts(
                    source.tuning, source.wheels,
                    source.collisionShapes.data(),
                    source.collisionShapes.size(),
                    source.forceFields.data(), source.forceFields.size());
            result.diagnostic =
                    "Vulkan immutable simulation configuration uploaded";
        }
        return result;
    } catch (...) {
        Reset();
        VulkanTransferMetrics result;
        result.diagnostic =
                "unexpected Vulkan configuration upload failure";
        return result;
    }
}

VulkanTransferMetrics VulkanDeviceStaticConfiguration::UploadPacked(
        const std::byte *data, std::size_t size) noexcept {
    Reset();
    VulkanTransferMetrics result;
    CudaPackedStaticConfigurationHeader header;
    if (data == nullptr || size < sizeof(header)) {
        result.diagnostic = "invalid packed Vulkan configuration";
        return result;
    }
    std::memcpy(&header, data, sizeof(header));
    if (header.magic != CudaPackedStaticConfigurationHeader::Magic ||
        header.schemaVersion !=
                CudaPackedStaticConfigurationHeader::SchemaVersion ||
        header.headerSize != sizeof(header) ||
        header.totalSize != size) {
        result.diagnostic = "packed Vulkan configuration schema mismatch";
        return result;
    }
    result.hostPackedBytes = size;
    result.success = vulkan::UploadImmutableBuffer(
            data, size, &buffer_, &result.uploadMilliseconds,
            &result.diagnostic);
    if (result.success) {
        configurationHash_ = header.deterministicHash;
        handlingSpecialization_ = SelectHandlingSpecialization(
                header.tuning, header.water.present);
        wheels_ = header.wheels;
        std::array<CudaVehicleCollisionShape, 8u> shapes{};
        std::array<CudaForceField, 1u> fields{};
        steadyVelocityStaticFacts_ =
                ReadPackedSection(
                        data, size, header.collisionShapes, &shapes) &&
                ReadPackedSection(
                        data, size, header.forceFields, &fields) &&
                StaticFacts(
                        header.tuning, header.wheels,
                        shapes.data(), 8u, fields.data(), 1u);
        result.deviceBytes = DeviceBytes();
        result.diagnostic = "packed Vulkan configuration uploaded";
    }
    return result;
}

void VulkanDeviceStaticConfiguration::Reset() noexcept {
    buffer_.reset();
    configurationHash_ = 0u;
    handlingSpecialization_ = CudaHandlingSpecialization::Generic;
    wheels_ = {};
    steadyVelocityStaticFacts_ = false;
}

bool VulkanDeviceStaticConfiguration::SteadyVelocityFactsMatch(
        const CudaCandidatePhysicsState &state) const noexcept {
    if (!steadyVelocityStaticFacts_ || state.vehicle.wheels.count != 4u ||
        state.body.dynamicType != static_cast<std::uint32_t>(
                CHmsDyna::EDynamicType_FullAngularDynamics)) {
        return false;
    }
    for (std::size_t index = 0u; index < wheels_.wheels.size(); ++index) {
        const CudaWheelState &stateWheel = state.vehicle.wheels.values[index];
        const VehicleWheelDefinition &definition = wheels_.wheels[index];
        if (stateWheel.killsLateralSpeedOnContact !=
                    definition.killsLateralSpeedOnContact ||
            stateWheel.axle != static_cast<std::uint32_t>(definition.axle) ||
            std::memcmp(&stateWheel.rollingRadius, &definition.rollingRadius,
                        sizeof(float)) != 0 ||
            std::memcmp(&stateWheel.forceApplicationPoint,
                        &definition.forceApplicationPoint,
                        sizeof(GmVec3)) != 0 ||
            std::memcmp(&stateWheel.restPose, &definition.restSurfacePose,
                        sizeof(GmIso4)) != 0) {
            return false;
        }
    }
    return true;
}

std::uint64_t
VulkanDeviceStaticConfiguration::DeviceBytes() const noexcept {
    return vulkan::BufferSize(buffer_);
}

}  // namespace forevervalidator::simulation
