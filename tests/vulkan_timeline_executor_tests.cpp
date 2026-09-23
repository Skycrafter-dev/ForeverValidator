#include "simulation/backends/vulkan/vulkan_compute_runtime.h"
#include "simulation/backends/vulkan/vulkan_timeline_executor.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
    using namespace forevervalidator::simulation;

    const vulkan::RuntimeDiagnostics diagnostics =
            vulkan::QueryRuntimeDiagnostics();
    if (!diagnostics.IsReady()) {
        std::cerr << diagnostics.diagnostic << '\n';
        return 1;
    }

    CudaPackedSceneHeader sceneHeader;
    sceneHeader.totalSize = sizeof(sceneHeader);
    std::vector<std::byte> sceneBytes(sizeof(sceneHeader));
    std::memcpy(sceneBytes.data(), &sceneHeader, sizeof(sceneHeader));
    VulkanDeviceScene scene;
    const VulkanTransferMetrics sceneUpload = scene.UploadPacked(
            sceneBytes.data(), sceneBytes.size());
    if (!sceneUpload.success) {
        std::cerr << sceneUpload.diagnostic << '\n';
        return 1;
    }

    CudaPackedStaticConfigurationHeader configurationHeader;
    configurationHeader.totalSize = sizeof(configurationHeader);
    std::vector<std::byte> configurationBytes(
            sizeof(configurationHeader));
    std::memcpy(configurationBytes.data(), &configurationHeader,
                sizeof(configurationHeader));
    VulkanDeviceStaticConfiguration configuration;
    const VulkanTransferMetrics configurationUpload =
            configuration.UploadPacked(configurationBytes.data(),
                                       configurationBytes.size());
    if (!configurationUpload.success) {
        std::cerr << configurationUpload.diagnostic << '\n';
        return 1;
    }

    ReplayControlTick source;
    source.periodMs = 10u;
    source.timeMs = 1234u;
    source.controls = {0.25f, 0.75f, -0.5f};
    source.observe = true;
    source.comparisonTarget = GmVec3{1.0f, 2.0f, 3.0f};
    VulkanCandidateTimelineInput candidate;
    candidate.initialState.candidateId = 42u;
    candidate.initialState.firstStep = false;
    candidate.ticks.push_back(FlattenCudaControlTick(source));

    const VulkanTimelineBatchResult executed =
            ExecuteVulkanTimelineBatch(
                    scene, configuration, {candidate});
    if (executed.status != VulkanTimelineStatus::Success ||
        executed.candidates.size() != 1u ||
        executed.candidates[0].status !=
                VulkanTimelineStatus::Success ||
        executed.candidates[0].executedTickCount != 1u ||
        executed.candidates[0].failureTick != UINT32_MAX ||
        executed.candidates[0].finalState.candidateId != 42u ||
        executed.candidates[0].finalState.world.schemePeriodMs != 10u ||
        executed.candidates[0].finalState.world.tickTimeMs != 1234u ||
        executed.candidates[0].observations.size() != 1u ||
        executed.metrics.kernelMilliseconds < 0.0) {
        std::cerr << "Vulkan timeline execution failed: "
                  << executed.diagnostic
                  << " candidate_status="
                  << (executed.candidates.empty() ? 999u :
                      static_cast<unsigned>(executed.candidates[0].status))
                  << " schema="
                  << (executed.candidates.empty() ? 999u :
                      executed.candidates[0].finalState.schemaVersion)
                  << '\n';
        return 1;
    }

    const VulkanTimelineBatchResult cancelled =
            ExecuteVulkanTimelineBatch(
                    scene, configuration, {candidate}, true);
    if (cancelled.status != VulkanTimelineStatus::Cancelled ||
        cancelled.candidates.size() != 1u ||
        cancelled.candidates[0].failureTick != 0u) {
        std::cerr << "Vulkan timeline cancellation was not deterministic\n";
        return 1;
    }

    VulkanCandidateTimelineInput wrongSchema = candidate;
    ++wrongSchema.initialState.schemaVersion;
    const VulkanTimelineBatchResult schema =
            ExecuteVulkanTimelineBatch(
                    scene, configuration, {wrongSchema});
    if (schema.status != VulkanTimelineStatus::SchemaMismatch ||
        schema.candidates.size() != 1u) {
        std::cerr << "Vulkan candidate schema mismatch was not explicit\n";
        return 1;
    }

    CudaPackedStaticConfigurationHeader factsHeader;
    factsHeader.tuning.handlingModel =
            CSceneVehicleCarHandlingModel_GearedDrive;
    factsHeader.tuning.wheelForceMode =
            CSceneVehicleCarWheelForceMode_FollowAbsorbWithImpulse;
    const GmMat3 identity = {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
    };
    for (std::size_t index = 0u;
         index < factsHeader.wheels.wheels.size(); ++index) {
        VehicleWheelDefinition &wheel = factsHeader.wheels.wheels[index];
        wheel.axle = index < 2u
                ? VehicleWheelAxle::Front
                : VehicleWheelAxle::Rear;
        wheel.killsLateralSpeedOnContact = true;
        wheel.rollingRadius = 0.5f;
        wheel.restSurfacePose.rotation = identity;
    }
    std::array<CudaVehicleCollisionShape, 8u> shapes{};
    constexpr std::uint32_t wheelIndices[] = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
            0u, 1u, 3u, 2u};
    for (std::size_t index = 0u; index < shapes.size(); ++index) {
        shapes[index].parentShapeIndex = UINT32_MAX;
        shapes[index].wheelIndex = wheelIndices[index];
    }
    const CudaForceField field{};
    factsHeader.collisionShapes.offset = sizeof(factsHeader);
    factsHeader.collisionShapes.count = shapes.size();
    factsHeader.collisionShapes.stride = sizeof(shapes[0]);
    factsHeader.forceFields.offset =
            factsHeader.collisionShapes.offset + sizeof(shapes);
    factsHeader.forceFields.count = 1u;
    factsHeader.forceFields.stride = sizeof(field);
    factsHeader.totalSize = factsHeader.forceFields.offset + sizeof(field);
    std::vector<std::byte> factsBytes(factsHeader.totalSize);
    std::memcpy(factsBytes.data(), &factsHeader, sizeof(factsHeader));
    std::memcpy(factsBytes.data() + factsHeader.collisionShapes.offset,
                shapes.data(), sizeof(shapes));
    std::memcpy(factsBytes.data() + factsHeader.forceFields.offset,
                &field, sizeof(field));
    VulkanDeviceStaticConfiguration factsConfiguration;
    if (!factsConfiguration.UploadPacked(
                factsBytes.data(), factsBytes.size()).success ||
        factsConfiguration.HandlingSpecialization() !=
                CudaHandlingSpecialization::GearedDriveDry) {
        std::cerr << "Vulkan specialization facts upload failed\n";
        return 1;
    }
    factsHeader.water.present = true;
    std::memcpy(factsBytes.data(), &factsHeader, sizeof(factsHeader));
    if (!factsConfiguration.UploadPacked(
                factsBytes.data(), factsBytes.size()).success ||
        factsConfiguration.HandlingSpecialization() !=
                CudaHandlingSpecialization::GearedDriveWater) {
        std::cerr << "Vulkan water specialization classification failed\n";
        return 1;
    }
    CudaCandidatePhysicsState factsState;
    factsState.body.dynamicType =
            CHmsDyna::EDynamicType_FullAngularDynamics;
    factsState.vehicle.wheels.count = 4u;
    for (std::size_t index = 0u;
         index < factsHeader.wheels.wheels.size(); ++index) {
        const VehicleWheelDefinition &definition =
                factsHeader.wheels.wheels[index];
        CudaWheelState &wheel = factsState.vehicle.wheels.values[index];
        wheel.killsLateralSpeedOnContact =
                definition.killsLateralSpeedOnContact;
        wheel.axle = static_cast<std::uint32_t>(definition.axle);
        wheel.rollingRadius = definition.rollingRadius;
        wheel.forceApplicationPoint = definition.forceApplicationPoint;
        wheel.restPose = definition.restSurfacePose;
    }
    if (!factsConfiguration.SteadyVelocityFactsMatch(factsState)) {
        std::cerr << "Vulkan specialization facts were not accepted\n";
        return 1;
    }
    factsState.vehicle.wheels.values[3].rollingRadius = 0.75f;
    if (factsConfiguration.SteadyVelocityFactsMatch(factsState)) {
        std::cerr << "Vulkan specialization accepted mismatched wheel facts\n";
        return 1;
    }

    std::cout << diagnostics.deviceName << " kernel_ms="
              << executed.metrics.kernelMilliseconds << '\n';
    return 0;
}
