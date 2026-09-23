// Generated from tests/cuda_static_configuration_tests.cpp by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_static_configuration.h"
#include "simulation/backends/hip/generated/hip_static_configuration_storage.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

namespace {

GmIso4 IdentityAt(float x, float y, float z) {
    GmIso4 result{};
    result.rotation.basisX = {1.0f, 0.0f, 0.0f};
    result.rotation.basisY = {0.0f, 1.0f, 0.0f};
    result.rotation.basisZ = {0.0f, 0.0f, 1.0f};
    result.translation = {x, y, z};
    return result;
}

ReplaySimulationDefinition BuildDefinition() {
    ReplaySimulationDefinition result;
    result.environment.zoneLinearDampingCoefficient = 0.75f;
    result.environment.zoneAngularDampingCoefficient = 0.5f;
    result.environment.forceFields.push_back(
            ReplayUniformForceFieldDefinition{{0.0f, -10.0f, 0.0f}});
    result.environment.forceFields.push_back(
            ReplayBallForceFieldDefinition{
                    {10.0f, 20.0f, 30.0f}, 5.0f, -2.0f});
    std::optional<ReplayWaterDefinition> water =
            ReplayWaterDefinition::Create(
                    {4.0f, 4.0f}, {-8.0f, -8.0f}, {2u, 2u},
                    WaterOccupancy::Dry, 3.0f, -100.0f);
    water->MarkWaterCell({1u, 0u});
    result.environment.water = std::move(*water);

    result.vehicle.tuning = MakeDefaultVehicleTuningDefinition();
    result.vehicle.tuning.handlingModel =
            ReplayVehicleHandlingModel::GearedDrive;
    result.vehicle.tuning.curves.maxSideFrictionFromSpeed =
            ReplayTuningCurveDefinition{
                    ReplayTuningCurveInterpolation::Linear,
                    {{0.0f, 1.0f}, {100.0f, 0.5f}}};
    ReplayVehicleTransmissionTuning &transmission =
            result.vehicle.tuning.gearedDrive.transmission;
    transmission.gearSpeedRatio = {1.0f, 2.0f, 3.0f};
    transmission.upshiftThreshold = {0.0f, 0.5f, 0.8f};
    transmission.downshiftThreshold = {0.0f, 0.2f, 0.4f};
    transmission.rpmWanted = {0.0f, 0.6f, 0.9f};
    transmission.targetInputBias = {0.0f, 0.0f, 0.1f};
    transmission.rpmDelta = {0.0f, 0.1f, 0.0f};

    for (std::size_t index = 0u;
         index < result.vehicle.wheels.wheels.size(); ++index) {
        VehicleWheelDefinition &wheel =
                result.vehicle.wheels.wheels[index];
        wheel.surfaceId = VehicleWheelSurfaceIdForIndex(index);
        wheel.collisionRole = VehicleWheelCollisionRole(index);
        wheel.rollingRadius = 0.4f;
        wheel.restSurfacePose =
                IdentityAt(static_cast<float>(index), -0.5f, 1.0f);
    }

    VehicleCollisionShapeDefinition bodyShape;
    bodyShape.localPose = IdentityAt(0.0f, 0.0f, 0.0f);
    result.vehicle.collisionModel.AddBodyShape(bodyShape);
    for (std::size_t index = 0u; index < 4u; ++index) {
        VehicleCollisionShapeDefinition wheelShape;
        wheelShape.localPose =
                IdentityAt(static_cast<float>(index), -0.5f, 1.0f);
        result.vehicle.collisionModel.SetWheelShape(
                VehicleWheelCollisionRole(index), wheelShape, 0u);
    }

    VehicleMaterialDefinition material;
    material.naturalId = 7u;
    material.usesFakeContactTexture = true;
    result.vehicle.materials.materials.push_back(material);
    result.vehicle.materials.materialIndexByNaturalId.push_back(0u);
    VehicleFakeContactTextureDefinition texture;
    texture.rgbPixels.assign(
            VehicleFakeContactTexturePixelBytes, 0x5au);
    result.vehicle.materials.fakeContactTexture = std::move(texture);
    return result;
}

}  // namespace

int main() {
    using namespace forevervalidator::simulation;
    const ReplaySimulationDefinition source = BuildDefinition();
    HipHostStaticConfiguration first;
    HipHostStaticConfiguration second;
    if (BuildHipHostStaticConfiguration(source, &first) !=
                HipStaticConfigurationBuildResult::Success ||
        BuildHipHostStaticConfiguration(source, &second) !=
                HipStaticConfigurationBuildResult::Success ||
        !first.Valid() || first.deterministicHash == 0u ||
        first.deterministicHash != second.deterministicHash) {
        std::cerr << "static configuration build was not deterministic\n";
        return 1;
    }
    if (first.forceFields.size() != 2u ||
        first.waterOccupancy.size() != 4u ||
        first.collisionShapes.size() != 5u ||
        first.fakeContactTextureRgb.size() !=
                VehicleFakeContactTexturePixelBytes) {
        std::cerr << "static configuration omitted immutable inputs\n";
        return 1;
    }
    const HipTuningCurve &curve = first.tuning.curves[
            static_cast<std::size_t>(
                    HipTuningCurveId::MaxSideFrictionFromSpeed)];
    if (curve.keyCount != 2u ||
        (curve.reserved &
         HipTuningCurvePositionsNondecreasing) == 0u ||
        first.curveKeys[curve.firstKey + 1u].value != 0.5f) {
        std::cerr << "tuning curve ordering changed\n";
        return 1;
    }

    std::vector<std::byte> packed;
    HipPackedStaticConfigurationHeader header;
    if (!PackHipStaticConfiguration(first, &packed, &header) ||
        header.magic != HipPackedStaticConfigurationHeader::Magic ||
        header.totalSize != packed.size() ||
        header.curveKeys.offset % 16u != 0u ||
        header.collisionShapes.count != 5u) {
        std::cerr << "static configuration packing failed\n";
        return 1;
    }
    const auto *packedShapes =
            reinterpret_cast<const HipVehicleCollisionShape *>(
                    packed.data() + header.collisionShapes.offset);
    for (std::uint32_t traversal = 0u;
         traversal < header.collisionShapes.count;
         ++traversal) {
        if (packedShapes[traversal].traversalOrder != traversal) {
            std::cerr << "collision shapes were not packed in traversal order\n";
            return 1;
        }
    }
    for (std::uint32_t wheel = 0u; wheel < 4u; ++wheel) {
        if (packedShapes[wheel].wheelIndex != wheel ||
            packedShapes[wheel].parentShapeIndex != 4u) {
            std::cerr << "packed wheel collision mapping changed\n";
            return 1;
        }
    }
    if (packedShapes[4].wheelIndex != UINT32_MAX ||
        packedShapes[4].archiveOrder != 0u) {
        std::cerr << "packed body collision mapping changed\n";
        return 1;
    }

    HipStaticConfigurationBuildLimits limits;
    limits.maximumCurveKeys = 1u;
    HipHostStaticConfiguration overflow;
    if (BuildHipHostStaticConfiguration(source, &overflow, limits) !=
            HipStaticConfigurationBuildResult::CurveOverflow) {
        std::cerr << "static configuration overflow was not explicit\n";
        return 1;
    }

    HipDeviceStaticConfiguration device;
    const HipStaticConfigurationTransferMetrics transfer =
            device.Upload(first);
#if FOREVERVALIDATOR_HAS_HIP
    if (!transfer.success || !device.Ready() ||
        device.ConfigurationHash() != first.deterministicHash ||
        transfer.deviceBytes != packed.size()) {
        std::cerr << "static configuration device upload failed: "
                  << transfer.diagnostic << '\n';
        return 1;
    }
#else
    if (transfer.success || device.Ready()) {
        std::cerr << "CPU-only build reported a HIP upload\n";
        return 1;
    }
#endif
    return 0;
}
