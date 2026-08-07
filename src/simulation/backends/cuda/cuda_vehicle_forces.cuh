#ifndef FOREVERVALIDATOR_CUDA_VEHICLE_FORCES_CUH
#define FOREVERVALIDATOR_CUDA_VEHICLE_FORCES_CUH

#include "simulation/backends/cuda/cuda_dynamics.cuh"
#include "simulation/backends/cuda/cuda_state_layout.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/backends/cuda/cuda_tuning.cuh"
#include "simulation/backends/cuda/cuda_vehicle_wheels.cuh"
#include "simulation/backends/cuda/cuda_collision_response.cuh"

namespace forevervalidator::simulation::cuda::vehicle {

enum class ForceStatus : std::uint32_t {
    Success,
    UnsupportedWater,
    UnsupportedHandlingModel,
    UnsupportedCircularBurnout,
    UnsupportedDirtSlide,
    MissingMaterial,
    UnsupportedDonut,
};

namespace force_detail {

constexpr float ScalarEpsilon = 1.0e-5f;
constexpr float VectorEpsilonSquared = 1.0e-10f;
constexpr float LowSpeedGateThreshold = 0.1f;
constexpr float Pi = 3.1415927f;
constexpr float HalfPi = Pi * 0.5f;
constexpr float SafeTrigInteriorLimit = 1.0f - 1.0e-6f;
constexpr std::uint32_t FeedbackRampContactId = 6u;
constexpr std::uint32_t TurboDurationAContactId = 7u;
constexpr std::uint32_t TurboDurationBContactId = 0x1au;
constexpr std::uint32_t TurboRouletteContactId = 0x1eu;
constexpr std::uint32_t ForcedLowSpeedFrictionContactId = 0x1du;
constexpr std::uint32_t TurboRoulettePeriodMs = 1000u;

__device__ inline GmVec3 WorldToLocal(
        const CudaDynamicBodyState &body,
        const GmVec3 &world) {
    return {
            dynamics::detail::Dot(
                    body.current.rotation.basisX, world),
            dynamics::detail::Dot(
                    body.current.rotation.basisY, world),
            dynamics::detail::Dot(
                    body.current.rotation.basisZ, world),
    };
}

__device__ inline GmVec3 LocalToWorld(
        const CudaDynamicBodyState &body,
        const GmVec3 &local) {
    const GmMat3 &rotation = body.current.rotation;
    return {
            rotation.basisY.x * local.y +
                    rotation.basisX.x * local.x +
                    rotation.basisZ.x * local.z,
            rotation.basisX.y * local.x +
                    rotation.basisY.y * local.y +
                    rotation.basisZ.y * local.z,
            rotation.basisX.z * local.x +
                    rotation.basisY.z * local.y +
                    rotation.basisZ.z * local.z,
    };
}

__device__ inline GmVec3 TransformDirection(
        const GmMat3 &rotation,
        const GmVec3 &local) {
    return {
            (rotation.basisX.x * local.x +
             rotation.basisY.x * local.y) +
                    rotation.basisZ.x * local.z,
            (rotation.basisX.y * local.x +
             rotation.basisY.y * local.y) +
                    rotation.basisZ.y * local.z,
            (rotation.basisX.z * local.x +
             rotation.basisY.z * local.y) +
                    rotation.basisZ.z * local.z,
    };
}

__device__ inline GmBoxAligned WaterWorldBox(
        const CudaCandidatePhysicsState &candidate) {
    const GmBoxAligned &local =
            candidate.vehicle.water.boxLocal;
    const GmMat3 &rotation = candidate.body.write.rotation;
    GmBoxAligned world;
    world.center = TransformDirection(rotation, local.center);
    world.center.x += candidate.body.write.position.x;
    world.center.y += candidate.body.write.position.y;
    world.center.z += candidate.body.write.position.z;
    GmMat3 absoluteRotation;
    absoluteRotation.basisX = {
            fabsf(rotation.basisX.x),
            fabsf(rotation.basisX.y),
            fabsf(rotation.basisX.z),
    };
    absoluteRotation.basisY = {
            fabsf(rotation.basisY.x),
            fabsf(rotation.basisY.y),
            fabsf(rotation.basisY.z),
    };
    absoluteRotation.basisZ = {
            fabsf(rotation.basisZ.x),
            fabsf(rotation.basisZ.y),
            fabsf(rotation.basisZ.z),
    };
    world.halfExtents =
            TransformDirection(absoluteRotation, local.halfExtents);
    return world;
}

__device__ inline bool WaterAcceptsRegion(
        const CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        GmBoxAligned &world) {
    if (!configuration->water.present) return false;
    world = WaterWorldBox(candidate);
    const float halfY = fabsf(world.halfExtents.y);
    const float lowerY = world.center.y - halfY;
    const float upperY = world.center.y + halfY;
    const float xIndexFloat =
            (world.center.x - configuration->water.origin.x) /
            configuration->water.cellSize.x;
    const float zIndexFloat =
            (world.center.z - configuration->water.origin.y) /
            configuration->water.cellSize.y;
    const std::uint32_t x =
            exact::TruncateToUint32Modulo(xIndexFloat);
    const std::uint32_t z =
            exact::TruncateToUint32Modulo(zIndexFloat);
    const bool inside =
            x < configuration->water.dimensions.x &&
            z < configuration->water.dimensions.y;
    const bool declaresPlaneMap =
            configuration->waterPlaneIndices.count != 0u ||
            configuration->waterPlanes.count != 0u ||
            configuration->water.outsidePlaneIndex != 0u;
    const std::uint32_t outside = declaresPlaneMap
            ? configuration->water.outsidePlaneIndex
            : configuration->water.outsideOccupancy;
    constexpr std::uint32_t Water = 1u;
    if (!inside && outside == Water &&
        configuration->water.surfaceHeight > lowerY) {
        return true;
    }
    if (!(upperY >
                  configuration->water.secondaryCullHeight) ||
        !(configuration->water.surfaceHeight > lowerY)) {
        return false;
    }
    std::uint32_t value = outside;
    if (inside) {
        const std::uint32_t index =
                x + configuration->water.dimensions.x * z;
        const std::uint8_t *cells = declaresPlaneMap
                ? tuning::Section<std::uint8_t>(
                          configuration,
                          configuration->waterPlaneIndices)
                : tuning::Section<std::uint8_t>(
                          configuration,
                          configuration->waterOccupancy);
        value = cells[index];
    }
    return value == Water;
}

__device__ inline void AddCentralForce(
        CudaCandidatePhysicsState &candidate,
        const GmVec3 &localForce) {
    const GmVec3 world =
            LocalToWorld(candidate.body, localForce);
    candidate.body.current.force.x =
            candidate.body.current.force.x + world.x;
    candidate.body.current.force.y =
            world.y + candidate.body.current.force.y;
    candidate.body.current.force.z =
            world.z + candidate.body.current.force.z;
    candidate.vehicle.forceAccumulators.force.x =
            candidate.vehicle.forceAccumulators.force.x +
            localForce.x;
    candidate.vehicle.forceAccumulators.force.y =
            localForce.y +
            candidate.vehicle.forceAccumulators.force.y;
    candidate.vehicle.forceAccumulators.force.z =
            localForce.z +
            candidate.vehicle.forceAccumulators.force.z;
}

__device__ inline GmVec3 WorldCenterOfMass(
        const CudaCandidatePhysicsState &candidate) {
    const CHmsDyna::CHmsStateDyna &state =
            candidate.body.current;
    const GmMat3 &rotation = state.rotation;
    const GmVec3 &position = state.position;
    const GmVec3 &localCenter =
            candidate.body.parameters.localCenterOfMass;
    return {
            ((rotation.basisY.x * localCenter.y +
              rotation.basisX.x * localCenter.x) +
             rotation.basisZ.x * localCenter.z) +
                    position.x,
            ((rotation.basisY.y * localCenter.y +
              rotation.basisX.y * localCenter.x) +
             rotation.basisZ.y * localCenter.z) +
                    position.y,
            ((rotation.basisY.z * localCenter.y +
              rotation.basisX.z * localCenter.x) +
             rotation.basisZ.z * localCenter.z) +
                    position.z,
    };
}

template <bool ReuseWorldCenter = false>
__device__ inline void AddForceAtPoint(
        CudaCandidatePhysicsState &candidate,
        const GmVec3 &localForce,
        const GmVec3 &localPoint,
        const GmVec3 &sharedWorldCenter = {}) {
    const GmVec3 worldForce =
            LocalToWorld(candidate.body, localForce);
    CHmsDyna::CHmsStateDyna &state =
            candidate.body.current;
    state.force.x = worldForce.x + state.force.x;
    state.force.y = state.force.y + worldForce.y;
    state.force.z = state.force.z + worldForce.z;
    const GmMat3 &rotation = state.rotation;
    const GmVec3 &position = state.position;
    const GmVec3 worldPoint = {
            ((rotation.basisY.x * localPoint.y +
              rotation.basisX.x * localPoint.x) +
             rotation.basisZ.x * localPoint.z) +
                    position.x,
            ((rotation.basisY.y * localPoint.y +
              rotation.basisX.y * localPoint.x) +
             rotation.basisZ.y * localPoint.z) +
                    position.y,
            ((rotation.basisY.z * localPoint.y +
              rotation.basisX.z * localPoint.x) +
             rotation.basisZ.z * localPoint.z) +
                    position.z,
    };
    GmVec3 worldCenter;
    if constexpr (ReuseWorldCenter) {
        worldCenter = sharedWorldCenter;
    } else {
        worldCenter = WorldCenterOfMass(candidate);
    }
    const float rx =
            worldPoint.x - worldCenter.x;
    const float ry =
            worldPoint.y - worldCenter.y;
    const float rz =
            worldPoint.z - worldCenter.z;
    const float torqueX =
            ry * worldForce.z - rz * worldForce.y;
    const float torqueY =
            rz * worldForce.x - rx * worldForce.z;
    const float torqueZ =
            rx * worldForce.y - ry * worldForce.x;
    state.torque.x = state.torque.x + torqueX;
    state.torque.y = state.torque.y + torqueY;
    state.torque.z = state.torque.z + torqueZ;
    candidate.vehicle.forceAccumulators.force.x =
            candidate.vehicle.forceAccumulators.force.x +
            localForce.x;
    candidate.vehicle.forceAccumulators.force.y =
            localForce.y +
            candidate.vehicle.forceAccumulators.force.y;
    candidate.vehicle.forceAccumulators.force.z =
            localForce.z +
            candidate.vehicle.forceAccumulators.force.z;
}

__device__ inline void AddTorque(
        CudaCandidatePhysicsState &candidate,
        const GmVec3 &localTorque) {
    const GmVec3 world =
            LocalToWorld(candidate.body, localTorque);
    candidate.body.current.torque.x =
            candidate.body.current.torque.x + world.x;
    candidate.body.current.torque.y =
            world.y + candidate.body.current.torque.y;
    candidate.body.current.torque.z =
            world.z + candidate.body.current.torque.z;
}

__device__ inline void AddCentralImpulse(
        CudaCandidatePhysicsState &candidate,
        const GmVec3 &localImpulse) {
    const GmVec3 world =
            LocalToWorld(candidate.body, localImpulse);
    const float inverseMass =
            1.0f / candidate.body.parameters.mass;
    candidate.body.current.linearSpeed.x =
            world.x * inverseMass +
            candidate.body.current.linearSpeed.x;
    candidate.body.current.linearSpeed.y =
            candidate.body.current.linearSpeed.y +
            world.y * inverseMass;
    candidate.body.current.linearSpeed.z =
            world.z * inverseMass +
            candidate.body.current.linearSpeed.z;
    candidate.vehicle.forceAccumulators.impulse.x =
            candidate.vehicle.forceAccumulators.impulse.x +
            localImpulse.x;
    candidate.vehicle.forceAccumulators.impulse.y =
            localImpulse.y +
            candidate.vehicle.forceAccumulators.impulse.y;
    candidate.vehicle.forceAccumulators.impulse.z =
            localImpulse.z +
            candidate.vehicle.forceAccumulators.impulse.z;
}

__device__ inline int ApplyWaterForces(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        const GmVec3 &forceToSubtract) {
    GmBoxAligned worldBox;
    if (!WaterAcceptsRegion(
                candidate, configuration, worldBox)) {
        return 0;
    }
    const float halfY = fabsf(worldBox.halfExtents.y);
    const float lowerY = worldBox.center.y - halfY;
    const float upperY = worldBox.center.y + halfY;
    const float depth =
            configuration->water.surfaceHeight - lowerY;
    if (!(depth > 0.5f)) {
        return 0;
    }

    const GmVec3 linearSpeedWorld = WorldToLocal(
            candidate.body, candidate.body.current.linearSpeed);
    const GmVec3 localSpeed = TransformDirection(
            candidate.body.write.rotation, linearSpeedWorld);
    const float horizontalSpeedSq =
            localSpeed.x * localSpeed.x +
            localSpeed.z * localSpeed.z;
    float splashCurveInput = 0.0f;
    if (!candidate.vehicle.airControl.refreshMemory &&
        depth < 0.9f &&
        (configuration->water.surfaceHeight - upperY) <
                0.0f &&
        localSpeed.y < -1.0e-5f) {
        const float horizontalThreshold =
                cuda::facts::Tuning(configuration).water.
                        splashHorizontalSpeedThreshold;
        bool applySplash = false;
        if (horizontalSpeedSq >
            horizontalThreshold * horizontalThreshold) {
            const float horizontalSpeed =
                    exact::Sqrt(horizontalSpeedSq);
            splashCurveInput =
                    -horizontalSpeed / localSpeed.y;
            if (splashCurveInput != splashCurveInput) {
                return 0;
            }
            applySplash = !(splashCurveInput < 0.0f);
        } else {
            const float totalThreshold =
                    cuda::facts::Tuning(configuration).water.
                            splashTotalSpeedThreshold;
            const float totalSpeedSq =
                    linearSpeedWorld.x * linearSpeedWorld.x +
                    linearSpeedWorld.y * linearSpeedWorld.y +
                    linearSpeedWorld.z * linearSpeedWorld.z;
            applySplash =
                    totalSpeedSq >
                    totalThreshold * totalThreshold;
        }
        if (applySplash) {
            const float verticalScale = tuning::Evaluate(
                    configuration,
                    CudaTuningCurveId::SplashVerticalImpulse,
                    splashCurveInput);
            const float horizontalScale = tuning::Evaluate(
                    configuration,
                    CudaTuningCurveId::SplashHorizontalImpulse,
                    splashCurveInput);
            const GmVec3 localCurveImpulse = {
                    -horizontalScale * localSpeed.x,
                    -verticalScale * localSpeed.y,
                    -horizontalScale * localSpeed.z,
            };
            const GmVec3 impulse = {
                    dynamics::detail::Dot(
                            candidate.body.write.rotation.basisX,
                            localCurveImpulse),
                    dynamics::detail::Dot(
                            candidate.body.write.rotation.basisY,
                            localCurveImpulse),
                    dynamics::detail::Dot(
                            candidate.body.write.rotation.basisZ,
                            localCurveImpulse),
            };
            ++candidate.vehicle.vehicleEvents[
                    CSceneVehicle::EVehicleEvent_WaterSplash].value;
            candidate.vehicle.water.splashLocalSpeed =
                    localSpeed;
            candidate.vehicle.water.splashPending = true;
            AddCentralImpulse(candidate, impulse);
            return 0;
        }
    }

    GmVec3 waterDrag{};
    const float speedLen = exact::Sqrt(
            linearSpeedWorld.x * linearSpeedWorld.x +
            linearSpeedWorld.y * linearSpeedWorld.y +
            linearSpeedWorld.z * linearSpeedWorld.z);
    if (1.0e-5f < speedLen) {
        const float dragScale =
                -tuning::EvaluateSpeed(
                        configuration,
                        CudaTuningCurveId::WaterFrictionFromSpeed,
                        speedLen);
        waterDrag = {
                linearSpeedWorld.x * dragScale,
                linearSpeedWorld.y * dragScale,
                linearSpeedWorld.z * dragScale,
        };
    }

    const GmVec3 angularSpeed = WorldToLocal(
            candidate.body, candidate.body.current.angularSpeed);
    const float angularLinearScale =
            -cuda::facts::Tuning(configuration).water.angularLinearDamping;
    const float angularSpeedLen = exact::Sqrt(
            angularSpeed.x * angularSpeed.x +
            angularSpeed.y * angularSpeed.y +
            angularSpeed.z * angularSpeed.z);
    const float angularSpeedScale =
            -angularSpeedLen *
            cuda::facts::Tuning(configuration).water.angularSpeedDamping;
    const GmVec3 torque = {
            angularSpeed.x * angularLinearScale +
                    angularSpeed.x * angularSpeedScale,
            angularSpeed.y * angularLinearScale +
                    angularSpeed.y * angularSpeedScale,
            angularSpeed.z * angularLinearScale +
                    angularSpeed.z * angularSpeedScale,
    };

    const GmVec3 buoyancyLocal = {
            0.0f,
            -cuda::facts::Tuning(configuration).water.buoyancyForce,
            0.0f,
    };
    const GmVec3 buoyancy = {
            dynamics::detail::Dot(
                    candidate.body.write.rotation.basisX,
                    buoyancyLocal),
            dynamics::detail::Dot(
                    candidate.body.write.rotation.basisY,
                    buoyancyLocal),
            dynamics::detail::Dot(
                    candidate.body.write.rotation.basisZ,
                    buoyancyLocal),
    };
    const GmVec3 centralForce = {
            (buoyancy.x + waterDrag.x) -
                    forceToSubtract.x,
            (buoyancy.y + waterDrag.y) -
                    forceToSubtract.y,
            (buoyancy.z + waterDrag.z) -
                    forceToSubtract.z,
    };
    AddCentralForce(candidate, centralForce);
    AddTorque(candidate, torque);
    return 1;
}

__device__ inline void SetLocalLinearSpeed(
        CudaCandidatePhysicsState &candidate,
        const GmVec3 &localSpeed) {
    candidate.body.current.linearSpeed =
            LocalToWorld(candidate.body, localSpeed);
}

__device__ inline void SetLocalAngularSpeed(
        CudaCandidatePhysicsState &candidate,
        const GmVec3 &localSpeed) {
    candidate.body.current.angularSpeed =
            LocalToWorld(candidate.body, localSpeed);
}

__device__ inline float ClampZeroOne(float value) {
    float result = 0.0f;
    if (!(value < 0.0f) && value != 0.0f) {
        result = value;
        if (value == value && 1.0f < value) {
            result = 1.0f;
        }
    }
    return result;
}

__device__ inline float ClampSymmetric(
        float value, float magnitude) {
    const float minimum = -magnitude;
    if (value != value) return value;
    if (!(minimum < value)) return minimum;
    if (!(value < magnitude)) return magnitude;
    return value;
}

__device__ inline bool IsGroundContact(
        const CudaVehicleState &vehicle) {
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        if (vehicle.wheels.values[index].realTime.contactPresent) {
            return true;
        }
    }
    return false;
}

__device__ inline bool HasMaterialContact(
        const CudaVehicleState &vehicle,
        std::uint32_t material) {
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        const auto &wheel = vehicle.wheels.values[index].realTime;
        if (wheel.contactPresent &&
            static_cast<std::uint32_t>(wheel.contactMaterial) ==
                    material) {
            return true;
        }
    }
    return false;
}

__device__ inline const VehicleMaterialDefinition *Material(
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t naturalId) {
    const std::uint32_t *remap =
            tuning::Section<std::uint32_t>(
                    configuration,
                    configuration->materialIndexByNaturalId);
    const VehicleMaterialDefinition *materials =
            tuning::Section<VehicleMaterialDefinition>(
                    configuration, configuration->materials);
    if (naturalId >=
                configuration->materialIndexByNaturalId.count ||
        configuration->materials.count == 0u) {
        return nullptr;
    }
    const std::uint32_t index = remap[naturalId];
    return index < configuration->materials.count
            ? materials + index
            : nullptr;
}

__device__ inline void SaveAndClearFeedback(
        CudaVehicleState &vehicle,
        GmVec3 &savedForce,
        GmVec3 &savedImpulse) {
    savedImpulse = vehicle.forceAccumulators.impulse;
    savedForce = vehicle.forceAccumulators.force;
    vehicle.forceAccumulators = {};
}

__device__ inline void SetZeroDynamics(
        CudaCandidatePhysicsState &candidate) {
    candidate.body.current.linearSpeed = {};
    candidate.body.current.angularSpeed = {};
    candidate.body.current.force = {};
    candidate.body.current.torque = {};
}

__device__ inline std::uint32_t FakeContactTextureIndex(
        float coordinate,
        float period,
        std::uint32_t dimension) {
    const float wrapped = exact::Fmod(coordinate, period);
    const float scaled =
            (fabsf(wrapped) / period) *
            exact::FromUnsignedInteger(dimension);
    return exact::TruncateToUint32Modulo(scaled);
}

__device__ inline void CreateFakeContacts(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration) {
    CudaVehicleState &vehicle = candidate.vehicle;
    const GmVec3 linearSpeed = WorldToLocal(
            candidate.body, candidate.body.current.linearSpeed);
    const std::uint8_t *texture =
            tuning::Section<std::uint8_t>(
                    configuration,
                    configuration->fakeContactTextureRgb);
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        CudaWheelState &wheel = vehicle.wheels.values[index];
        if (!wheel.realTime.contactPresent) continue;
        const GmVec3 &restTranslation =
                facts::Wheel(
                        configuration,
                        index).restSurfacePose.translation;
        const VehicleMaterialDefinition *material = Material(
                configuration,
                static_cast<std::uint32_t>(
                        wheel.realTime.contactMaterial));
        if (material == nullptr ||
            !material->usesFakeContactTexture ||
            texture == nullptr ||
            configuration->fakeContactTextureRgb.count == 0u) {
            return;
        }
        const GmVec3 worldPoint = {
                (candidate.body.write.rotation.basisX.x *
                         restTranslation.x +
                 candidate.body.write.rotation.basisY.x *
                         restTranslation.y) +
                        candidate.body.write.rotation.basisZ.x *
                                restTranslation.z +
                        candidate.body.write.position.x,
                (candidate.body.write.rotation.basisX.y *
                         restTranslation.x +
                 candidate.body.write.rotation.basisY.y *
                         restTranslation.y) +
                        candidate.body.write.rotation.basisZ.y *
                                restTranslation.z +
                        candidate.body.write.position.y,
                (candidate.body.write.rotation.basisX.z *
                         restTranslation.x +
                 candidate.body.write.rotation.basisY.z *
                         restTranslation.y) +
                        candidate.body.write.rotation.basisZ.z *
                                restTranslation.z +
                        candidate.body.write.position.z,
        };
        const std::uint32_t pixelX = FakeContactTextureIndex(
                worldPoint.x, material->fakeContactPeriodX,
                VehicleFakeContactTextureWidth);
        const std::uint32_t pixelY = FakeContactTextureIndex(
                worldPoint.z, material->fakeContactPeriodZ,
                VehicleFakeContactTextureHeight);
        const std::uint8_t pixel = texture[
                (pixelY * VehicleFakeContactTextureWidth + pixelX) *
                VehicleFakeContactTextureBytesPerPixel];
        if (pixel == 0u) continue;
        const float pixelRatio =
                exact::FromUnsignedInteger(pixel) / 255.0f;
        float fakeSpeed =
                (pixelRatio * linearSpeed.z) *
                material->fakeContactSpeedScale;
        if (material->fakeContactDepthMax < fakeSpeed) {
            fakeSpeed = material->fakeContactDepthMax;
        }
        collision::response_detail::Contact contact;
        contact.localNormal = {0.0f, 1.0f, 0.0f};
        contact.localPoint = restTranslation;
        contact.localSpeed = {0.0f, -fakeSpeed, 0.0f};
        contact.peerMaterial =
                static_cast<std::uint32_t>(
                        wheel.realTime.contactMaterial);
        contact.wheelIndex = index;
        const GmVec3 peerAxis =
                wheel.realTime.peerZAxisInCarLocal;
        const CHmsCorpusId peerId =
                wheel.realTime.peerCorpusId;
        collision::response_detail::AbsorbWheel(
                candidate, configuration, contact);
        wheel.realTime.peerZAxisInCarLocal = peerAxis;
        wheel.realTime.peerCorpusId = peerId;
    }
}

template <
        CudaHandlingSpecialization Handling =
                CudaHandlingSpecialization::Generic>
__device__ inline void ApplyFrictionForces(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        const GmVec3 &speed) {
    CudaVehicleState &vehicle = candidate.vehicle;
    bool slipHandling = false;
    if constexpr (
            Handling ==
                    CudaHandlingSpecialization::GearedDriveDry ||
            Handling ==
                    CudaHandlingSpecialization::GearedDriveWater) {
        slipHandling = true;
    } else if constexpr (
            Handling == CudaHandlingSpecialization::Generic) {
        slipHandling =
                cuda::facts::Tuning(configuration).handlingModel ==
                        CSceneVehicleCarHandlingModel_SlipResponse ||
                cuda::facts::Tuning(configuration).handlingModel ==
                        CSceneVehicleCarHandlingModel_GearedDrive;
    }
    if (slipHandling &&
        vehicle.controls.noGroundFrictionGuard &&
        !IsGroundContact(vehicle)) {
        return;
    }
    const float gate =
            !vehicle.engine.useLowSpeedGateB
            ? vehicle.controls.lowSpeedGateA
            : vehicle.controls.lowSpeedGateB;
    if (gate < ScalarEpsilon ||
        vehicle.controls.forcedLowSpeedFriction) {
        GmVec3 force = speed;
        const float lengthSquared =
                (force.y * force.y + force.x * force.x) +
                force.z * force.z;
        if (VectorEpsilonSquared < lengthSquared) {
            const float inverseLength =
                    1.0f / exact::Sqrt(lengthSquared);
            force.x = inverseLength * force.x;
            force.y = force.y * inverseLength;
            force.z = inverseLength * force.z;
            const float frictionScale =
                    -cuda::facts::Tuning(configuration).
                            lowSpeedFrictionMagnitude;
            force.x = frictionScale * force.x;
            force.y = force.y * frictionScale;
            force.z = frictionScale * force.z;
            if (!vehicle.controls.forcedLowSpeedFriction) {
                const float dampingScale =
                        -cuda::facts::Tuning(configuration).
                                lowSpeedLinearDamping;
                const GmVec3 damping = {
                        speed.x * dampingScale,
                        speed.y * dampingScale,
                        dampingScale * speed.z,
                };
                force.x = damping.x + force.x;
                force.y = damping.y + force.y;
                force.z = damping.z + force.z;
            }
            AddCentralForce(candidate, force);
        }
    }
    if (!slipHandling) {
        if (vehicle.contacts.lateralSlowDownContactActive) {
            const float scale = -tuning::EvaluateSpeed(
                    configuration,
                    CudaTuningCurveId::
                            LateralContactSlowDownFromSpeed,
                    speed.z);
            AddCentralForce(candidate, {
                    speed.x * scale,
                    speed.y * scale,
                    scale * speed.z,
            });
        }
        return;
    }
    const std::uint32_t tick = candidate.world.tickTimeMs;
    if (vehicle.contacts.lateralSlowDownContactActive) {
        vehicle.contacts.lateralSlowDownLastTick = tick;
    }
    if (vehicle.contacts.lateralSlowDownLastTick <= tick) {
        const std::uint32_t elapsed =
                tick -
                vehicle.contacts.lateralSlowDownLastTick;
        if (elapsed < cuda::facts::Tuning(configuration).slipResponse.
                              lateralSlowDownTickWindow) {
            const float speedSquared =
                    (speed.y * speed.y + speed.x * speed.x) +
                    speed.z * speed.z;
            const float speedLength =
                    exact::Sqrt(speedSquared);
            if (ScalarEpsilon < speedLength) {
                const float inverseLength = 1.0f / speedLength;
                const GmVec3 unit = {
                        speed.x * inverseLength,
                        speed.y * inverseLength,
                        inverseLength * speed.z,
                };
                const float scale = -tuning::EvaluateSpeed(
                        configuration,
                        CudaTuningCurveId::
                                LateralContactSlowDownFromSpeed,
                        speedLength);
                AddCentralForce(candidate, {
                        scale * unit.x,
                        unit.y * scale,
                        scale * unit.z,
                });
            }
        }
    }
}

__device__ inline void ClampLinearSpeed(
        CudaCandidatePhysicsState &candidate,
        GmVec3 &localSpeed) {
    const float capSquared =
            candidate.vehicle.linearSpeedCap *
            candidate.vehicle.linearSpeedCap;
    const float xy =
            localSpeed.y * localSpeed.y +
            localSpeed.x * localSpeed.x;
    const float speedSquared =
            localSpeed.z * localSpeed.z + xy;
    if (capSquared < speedSquared &&
        capSquared > VectorEpsilonSquared) {
        const float scale = exact::FromDouble(
                static_cast<double>(
                        candidate.vehicle.linearSpeedCap) /
                static_cast<double>(
                        exact::Sqrt(speedSquared)));
        localSpeed.x = scale * localSpeed.x;
        localSpeed.y = localSpeed.y * scale;
        localSpeed.z = scale * localSpeed.z;
        SetLocalLinearSpeed(candidate, localSpeed);
    }
}

__device__ inline void GroundMaterial(
        const CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        VehicleMaterialBlendValues &values,
        bool &present) {
    values = {};
    present = false;
    std::uint32_t count = 0u;
    const CudaVehicleState &vehicle = candidate.vehicle;
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        if (!vehicle.wheels.values[index].realTime.contactPresent) {
            continue;
        }
        const std::uint32_t firstMaterial =
                static_cast<std::uint32_t>(
                        vehicle.wheels.values[0u].
                                realTime.contactMaterial);
        const VehicleMaterialDefinition *material =
                Material(configuration, firstMaterial);
        if (material == nullptr) continue;
        ++count;
        values.x += material->blendableValues.x;
        values.y += material->blendableValues.y;
        values.z += material->blendableValues.z;
        values.w += material->blendableValues.w;
        present = true;
    }
    if (count != 0u) {
        const float inverse =
                1.0f / exact::FromUnsignedInteger(count);
        values.x *= inverse;
        values.y *= inverse;
        values.z *= inverse;
        values.w *= inverse;
    }
}

__device__ inline float SlopeBlend(
        float value, float minimum, float maximum) {
    if (!(minimum <= value)) return 0.0f;
    if (!(maximum >= value)) return 1.0f;
    const float angle =
            ((value - minimum) / (maximum - minimum)) *
            static_cast<double>(Pi) * 0.5;
    return 1.0f - exact::Cos(angle);
}

__device__ inline void SlopeAdherence(
        const CudaPackedStaticConfigurationHeader *configuration,
        const GmVec3 &normal,
        float &first,
        float &second) {
    const float lengthSquared =
            normal.y * normal.y +
            normal.x * normal.x +
            normal.z * normal.z;
    if (!(VectorEpsilonSquared < lengthSquared)) return;
    const float slope = fabsf(exact::FromDouble(
            static_cast<double>(normal.y) /
            static_cast<double>(exact::Sqrt(lengthSquared))));
    first = SlopeBlend(
            slope,
            cuda::facts::Tuning(configuration).bodyAirResponse.
                    slopeAdherence1Min,
            cuda::facts::Tuning(configuration).bodyAirResponse.
                    slopeAdherence1Max);
    second = SlopeBlend(
            slope,
            cuda::facts::Tuning(configuration).bodyAirResponse.
                    slopeAdherence2Min,
            cuda::facts::Tuning(configuration).bodyAirResponse.
                    slopeAdherence2Max);
}

__device__ inline float VisualSteerYaw(
        const CudaVehicleState &vehicle,
        const CudaPackedStaticConfigurationHeader *configuration,
        const GmVec3 &linearSpeed) {
    const float denominator =
            fabsf(linearSpeed.z) *
                    cuda::facts::Tuning(configuration).visual.wheelSpeedScale +
            cuda::facts::Tuning(configuration).visual.wheelSpeedBase;
    float asinValue = 0.0f;
    if (!(denominator < ScalarEpsilon)) {
        const float input = 1.0f / denominator;
        if (input < -SafeTrigInteriorLimit) {
            asinValue = -HalfPi;
        } else if (SafeTrigInteriorLimit < input) {
            asinValue = HalfPi;
        } else {
            asinValue = exact::Asin(input);
        }
    }
    return -vehicle.controls.currentSteering * asinValue;
}

template <
        CudaHandlingSpecialization Handling =
                CudaHandlingSpecialization::Generic>
__device__ inline void UpdateAirControl(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        const GmVec3 &angularSpeed,
        bool groundContact,
    bool resetMemory) {
    CudaVehicleState &vehicle = candidate.vehicle;
    bool slipHandling = false;
    if constexpr (
            Handling ==
                    CudaHandlingSpecialization::GearedDriveDry ||
            Handling ==
                    CudaHandlingSpecialization::GearedDriveWater) {
        slipHandling = true;
    } else if constexpr (
            Handling == CudaHandlingSpecialization::Generic) {
        slipHandling =
                cuda::facts::Tuning(configuration).handlingModel ==
                        CSceneVehicleCarHandlingModel_SlipResponse ||
                cuda::facts::Tuning(configuration).handlingModel ==
                        CSceneVehicleCarHandlingModel_GearedDrive;
    }
    if (slipHandling &&
        vehicle.controls.noGroundFrictionGuard) {
        return;
    }
    GmVec3 torque = {
            -angularSpeed.x,
            -angularSpeed.y,
            -angularSpeed.z,
    };
    const std::uint32_t tick = candidate.world.tickTimeMs;
    if (resetMemory) vehicle.airControl.memoryTick = tick;
    if (resetMemory || vehicle.airControl.refreshMemory) {
        vehicle.airControl.memoryAngular = angularSpeed;
    } else if (
            tick - vehicle.airControl.memoryTick <
            cuda::facts::Tuning(configuration).bodyAirResponse.
                    airControlMemoryTickWindow) {
        GmVec3 target = angularSpeed;
        bool strong = false;
        if ((ScalarEpsilon <
                     vehicle.controls.steeringControl &&
             vehicle.airControl.memoryAngular.y < 0.0f) ||
            (vehicle.controls.steeringControl < -ScalarEpsilon &&
             0.0f < vehicle.airControl.memoryAngular.y)) {
            if (cuda::facts::Tuning(configuration).bodyAirResponse.
                        airControlYSwitchThreshold <
                fabsf(angularSpeed.y)) {
                strong = true;
                vehicle.airControl.memoryAngular.y =
                        angularSpeed.y;
            }
        } else {
            if (!(ScalarEpsilon <
                          vehicle.controls.steeringControl) ||
                !(0.0f <
                          vehicle.airControl.memoryAngular.y)) {
                if (vehicle.controls.steeringControl <
                            -ScalarEpsilon &&
                    vehicle.airControl.memoryAngular.y < 0.0f) {
                    strong = true;
                }
            } else {
                strong = true;
            }
            vehicle.airControl.memoryAngular.y = angularSpeed.y;
        }
        target.y = vehicle.airControl.memoryAngular.y;
        if (slipHandling) {
            if (ScalarEpsilon <
                        vehicle.controls.lowSpeedGateB &&
                0.0f <
                        vehicle.airControl.memoryAngular.x) {
                vehicle.airControl.memoryAngular.x = 0.0f;
            } else {
                vehicle.airControl.memoryAngular.x =
                        angularSpeed.x;
            }
            target.x = vehicle.airControl.memoryAngular.x;
        }
        if (strong) {
            torque.x *= 3.0f;
            torque.y *= 3.0f;
            torque.z = 3.0f * torque.z;
        }
        if (!groundContact) {
            const float scale = tuning::Evaluate(
                    configuration,
                    CudaTuningCurveId::AirControlZScale,
                    fabsf(angularSpeed.z));
            torque.z *= scale;
        }
        SetLocalAngularSpeed(candidate, target);
    }
    if (!groundContact) {
        const float xy =
                torque.x * torque.x + torque.y * torque.y;
        const float lengthSquared =
                torque.z * torque.z + xy;
        const float length = exact::Sqrt(lengthSquared);
        if (length >= ScalarEpsilon) {
            const float inverse = 1.0f / length;
            const float quadratic =
                    cuda::facts::Tuning(configuration).bodyAirResponse.
                            airTorqueQuadraticCoef *
                    length * length;
            const float linear =
                    length *
                    cuda::facts::Tuning(configuration).bodyAirResponse.
                            airTorqueLinearCoef;
            const float magnitude = quadratic + linear;
            AddTorque(candidate, {
                    magnitude * (inverse * torque.x),
                    (inverse * torque.y) * magnitude,
                    magnitude * (inverse * torque.z),
            });
        }
    }
}

__device__ inline void UpdateTurbo(CudaVehicleState &vehicle,
                                   std::uint32_t tick) {
    if (vehicle.turbo.type !=
        CSceneVehicleCar::ETurboType_Inactive) {
        if (tick > vehicle.turbo.endTick) {
            vehicle.turbo.type =
                    CSceneVehicleCar::ETurboType_Inactive;
        }
        if (vehicle.turbo.type !=
            CSceneVehicleCar::ETurboType_Inactive) {
            vehicle.turbo.progressRatio =
                    exact::FromUnsignedInteger(
                            tick - vehicle.turbo.startTick) /
                    exact::FromUnsignedInteger(
                            vehicle.turbo.endTick -
                            vehicle.turbo.startTick);
            return;
        }
    }
    vehicle.turbo.progressRatio = 0.0f;
}

__device__ inline bool GroundContact(
        const CudaVehicleState &vehicle,
        std::uint32_t material,
        GmVec3 &peerAxis,
        std::uint32_t &peerCorpusId) {
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        const auto &contact =
                vehicle.wheels.values[index].realTime;
        if (contact.contactPresent &&
            static_cast<std::uint32_t>(
                    contact.contactMaterial) == material) {
            peerAxis = contact.peerZAxisInCarLocal;
            memory::CopyBytes<sizeof(peerCorpusId)>(
                    &peerCorpusId, &contact.peerCorpusId);
            return true;
        }
    }
    return false;
}

__device__ inline void EnableTurbo(
        CudaVehicleState &vehicle,
        std::uint32_t tick,
        std::uint32_t duration,
        float impulseScale,
        CSceneVehicleCar::ETurboType type,
        std::uint32_t sourceCorpusId) {
    if (vehicle.turbo.type != type) {
        vehicle.turbo.startTick = tick;
        vehicle.turbo.sourceCorpusId = CHmsCorpusId{};
    }
    if (type == CSceneVehicleCar::ETurboType_Direct) {
        vehicle.turbo.impulseScale = impulseScale;
    } else if (type ==
                       CSceneVehicleCar::ETurboType_Roulette) {
        std::uint32_t currentSource = 0u;
        memory::CopyBytes<sizeof(currentSource)>(
                &currentSource, &vehicle.turbo.sourceCorpusId);
        if (currentSource != sourceCorpusId) {
            const std::uint32_t remainder =
                    (tick - vehicle.turbo.rouletteTickOrigin) %
                    TurboRoulettePeriodMs;
            const float phaseValue =
                    exact::FromUnsignedInteger(remainder) /
                    exact::FromUnsignedInteger(
                            TurboRoulettePeriodMs);
            const float phase =
                    phaseValue < 4.0f / 7.0f
                    ? 0.0f
                    : (phaseValue < 6.0f / 7.0f
                       ? 0.5f : 1.0f);
            vehicle.turbo.type2Phase = phase;
            vehicle.turbo.impulseScale =
                    (phase + 1.0f) * impulseScale;
            memory::CopyBytes<sizeof(sourceCorpusId)>(
                    &vehicle.turbo.sourceCorpusId,
                    &sourceCorpusId);
        }
    }
    vehicle.turbo.endTick = tick + duration;
    vehicle.turbo.type = type;
}

__device__ inline void ProcessTurboContacts(
        CudaVehicleState &vehicle,
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t tick) {
    GmVec3 peerAxis;
    std::uint32_t peerCorpusId = 0u;
    if (GroundContact(
                vehicle, TurboDurationAContactId,
                peerAxis, peerCorpusId)) {
        EnableTurbo(
                vehicle, tick,
                cuda::facts::Tuning(configuration).turbo.durationA,
                cuda::facts::Tuning(configuration).turbo.impulseScaleA *
                        peerAxis.z,
                CSceneVehicleCar::ETurboType_Direct,
                peerCorpusId);
    }
    if (GroundContact(
                vehicle, TurboDurationBContactId,
                peerAxis, peerCorpusId)) {
        EnableTurbo(
                vehicle, tick,
                cuda::facts::Tuning(configuration).turbo.durationB,
                cuda::facts::Tuning(configuration).turbo.impulseScaleB *
                        peerAxis.z,
                CSceneVehicleCar::ETurboType_Direct,
                peerCorpusId);
    }
    if (GroundContact(
                vehicle, TurboRouletteContactId,
                peerAxis, peerCorpusId)) {
        EnableTurbo(
                vehicle, tick,
                cuda::facts::Tuning(configuration).turbo.durationA,
                cuda::facts::Tuning(configuration).turbo.impulseScaleA *
                        peerAxis.z,
                CSceneVehicleCar::ETurboType_Roulette,
                peerCorpusId);
    }
    if (GroundContact(
                vehicle, ForcedLowSpeedFrictionContactId,
                peerAxis, peerCorpusId)) {
        vehicle.controls.forcedLowSpeedFriction = true;
    }
}

__device__ inline CSceneVehicleCarImpactState ImpactSeverity(
        float bucket,
        float lowThreshold,
        float highThreshold,
        CSceneVehicleCarImpactState current) {
    if (!(lowThreshold < bucket)) return current;
    if (highThreshold < bucket) {
        return current < CSceneVehicleCarImpactState_High
                ? CSceneVehicleCarImpactState_High : current;
    }
    return current == CSceneVehicleCarImpactState_None
            ? CSceneVehicleCarImpactState_Low : current;
}

__device__ inline void UpdateImpactStates(
        CudaVehicleState &vehicle,
        const CudaPackedStaticConfigurationHeader *configuration) {
    auto &contacts = vehicle.contacts;
    contacts.frontWheelImpactState = ImpactSeverity(
            contacts.frontWheelImpactBucket,
            cuda::facts::Tuning(configuration).contactResponse.
                    wheelImpactFeedbackLowThreshold,
            cuda::facts::Tuning(configuration).contactResponse.
                    wheelImpactFeedbackHighThreshold,
            contacts.frontWheelImpactState);
    contacts.rearWheelImpactState = ImpactSeverity(
            contacts.rearWheelImpactBucket,
            cuda::facts::Tuning(configuration).contactResponse.
                    wheelImpactFeedbackLowThreshold,
            cuda::facts::Tuning(configuration).contactResponse.
                    wheelImpactFeedbackHighThreshold,
            contacts.rearWheelImpactState);
    contacts.bodyImpactState = ImpactSeverity(
            contacts.bodyImpactBucket,
            cuda::facts::Tuning(configuration).contactResponse.
                    bodyImpactFeedbackLowThreshold,
            cuda::facts::Tuning(configuration).contactResponse.
                    bodyImpactFeedbackHighThreshold,
            contacts.bodyImpactState);
    if (contacts.peakRearWheelImpactState <
        contacts.rearWheelImpactState) {
        contacts.peakRearWheelImpactState =
                contacts.rearWheelImpactState;
        contacts.peakWheelImpactMaterial =
                contacts.lastWheelContactMaterial;
    }
    if (contacts.peakFrontWheelImpactState <
        contacts.frontWheelImpactState) {
        contacts.peakFrontWheelImpactState =
                contacts.frontWheelImpactState;
        contacts.peakWheelImpactMaterial =
                contacts.lastWheelContactMaterial;
    }
    if (contacts.peakBodyImpactState <
        contacts.bodyImpactState) {
        contacts.peakBodyImpactState =
                contacts.bodyImpactState;
        contacts.peakBodyImpactMaterial =
                contacts.lastBodyContactMaterial;
    }
}

__device__ inline void ApplySpecialContactResponse(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        const GmVec3 &currentForce,
        std::uint32_t tick,
        bool groundContact) {
    CudaVehicleState &vehicle = candidate.vehicle;
    if (!(vehicle.controls.specialContactResponseGate >
          ScalarEpsilon)) {
        return;
    }
    switch (vehicle.controls.specialContactResponseMode) {
    case CSceneVehicleCarSpecialContactMode_ImpulseFromForce:
        if (groundContact &&
            vehicle.contacts.
                    specialContactImpulseCooldownUntil < tick) {
            GmVec3 impulse = {
                    -currentForce.x,
                    -currentForce.y,
                    -currentForce.z,
            };
            const float lengthSquared =
                    impulse.x * impulse.x +
                    impulse.y * impulse.y +
                    impulse.z * impulse.z;
            if (lengthSquared > VectorEpsilonSquared) {
                const float inverse =
                        1.0f / exact::Sqrt(lengthSquared);
                impulse.x *= inverse;
                impulse.y *= inverse;
                impulse.z *= inverse;
            }
            const float magnitude =
                    cuda::facts::Tuning(configuration).contactResponse.
                            specialContactImpulseMagnitude;
            impulse.x *= magnitude;
            impulse.y *= magnitude;
            impulse.z *= magnitude;
            AddCentralImpulse(candidate, impulse);
            vehicle.contacts.
                    specialContactImpulseCooldownUntil =
                    tick + 100u;
        }
        break;
    case CSceneVehicleCarSpecialContactMode_SolidFeedback:
        candidate.body.physicalParameters.
                vehicleContactFeedbackScale =
                cuda::facts::Tuning(configuration).contactResponse.
                        specialSolidFeedbackValue;
        candidate.body.parameters.forceScale =
                candidate.body.physicalParameters.
                        vehicleContactFeedbackScale;
        break;
    case CSceneVehicleCarSpecialContactMode_Turbo:
        vehicle.turbo.type =
                CSceneVehicleCar::ETurboType_Direct;
        vehicle.turbo.impulseScale =
                cuda::facts::Tuning(configuration).turbo.impulseScaleA;
        break;
    default:
        break;
    }
}

__device__ inline void IntegrateSpring(
        GmSpring<float> &spring, float dt) {
    const float acceleration =
            (spring.target - spring.value) * spring.stiffness -
            spring.damping * spring.velocity;
    const float nextVelocity =
            acceleration * dt + spring.velocity;
    spring.velocity = nextVelocity;
    spring.value = dt * nextVelocity + spring.value;
}

__device__ inline void UpdateFeedbackSpring(
        CudaVehicleState &vehicle,
        GmSpring<float> &spring,
        float dt,
        float savedForce,
        float savedImpulse,
        bool invert) {
    IntegrateSpring(spring, dt);
    const float unclamped = invert
            ? -savedForce * dt - savedImpulse
            : savedForce * dt + savedImpulse;
    const float drive = ClampSymmetric(
            unclamped, vehicle.feedback.springDriveLimit);
    const float delta =
            (drive / vehicle.feedback.springDriveLimit) *
            vehicle.feedback.springVelocityLimit;
    spring.value = ClampSymmetric(
            spring.value, vehicle.feedback.springValueLimit);
    spring.velocity = ClampSymmetric(
            spring.velocity + delta,
            vehicle.feedback.springVelocityLimit);
}

__device__ inline void UpdateFeedback(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        float dt,
        const GmVec3 &linearSpeed,
        const GmVec3 &savedForce,
        const GmVec3 &savedImpulse,
        float surfaceFeedback) {
    CudaVehicleState &vehicle = candidate.vehicle;
    vehicle.gearedDrive.scaledCurrentForce =
            WorldToLocal(
                    candidate.body,
                    candidate.body.current.force);
    const float forceScale =
            1.0f / cuda::facts::Tuning(configuration).feedback.forceDivisor;
    vehicle.gearedDrive.scaledCurrentForce.x *= forceScale;
    vehicle.gearedDrive.scaledCurrentForce.y =
            forceScale *
            vehicle.gearedDrive.scaledCurrentForce.y;
    vehicle.gearedDrive.scaledCurrentForce.z *= forceScale;
    const float rate = tuning::Evaluate(
            configuration, CudaTuningCurveId::SurfaceFeedback,
            surfaceFeedback) +
            cuda::facts::Tuning(configuration).feedback.surfaceBaseRate;
    const float accumulated =
            rate * dt + vehicle.feedback.surfaceAccumulator;
    vehicle.feedback.surfaceAccumulator =
            ClampZeroOne(accumulated);
    UpdateFeedbackSpring(
            vehicle, vehicle.feedback.sideSpring, dt,
            savedForce.x, savedImpulse.x, false);
    UpdateFeedbackSpring(
            vehicle, vehicle.feedback.forwardSpring, dt,
            savedForce.z, savedImpulse.z, true);
    const float direction = HasMaterialContact(
            vehicle, FeedbackRampContactId)
            ? 1.0f
            : -1.0f;
    const float curveInput =
            fabsf(linearSpeed.z * 3.6f);
    vehicle.feedback.ramp0 = ClampZeroOne(
            vehicle.feedback.ramp0 +
            tuning::Evaluate(
                    configuration,
                    CudaTuningCurveId::VehicleFeedbackRamp0,
                    curveInput) *
                    dt * direction);
    vehicle.feedback.ramp1 = ClampZeroOne(
            vehicle.feedback.ramp1 +
            tuning::Evaluate(
                    configuration,
                    CudaTuningCurveId::VehicleFeedbackRamp1,
                    curveInput) *
                    dt * direction);
}

__device__ inline void ClearContactScratch(
        CudaVehicleState &vehicle) {
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        auto &wheel = vehicle.wheels.values[index].realTime;
        wheel.contactPresent = false;
        wheel.contactNormalSampleCount = 0u;
        wheel.latestContactPoint = {};
        wheel.accumulatedContactNormal = {};
        wheel.contactMaterial =
                EPlugSurfaceMaterialId_Concrete;
        wheel.rejectedNormalContact = false;
    }
    vehicle.contacts.frontWheelImpactBucket = 0.0f;
    vehicle.contacts.bodyContactPresent = false;
    vehicle.contacts.rearWheelImpactBucket = 0.0f;
    vehicle.airControl.refreshMemory = false;
    vehicle.contacts.bodyImpactBucket = 0.0f;
    vehicle.contacts.lateralSlowDownContactActive = false;
}

__device__ inline float SignNonNegative(float value) {
    return value < 0.0f ? -1.0f : 1.0f;
}

template <bool ReuseWorldCenter = false>
__device__ inline void WheelSuspensionForce(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        CudaWheelState &wheel,
        std::uint32_t wheelIndex,
        const GmVec3 &sharedWorldCenter = {}) {
    if (!wheel.realTime.contactPresent) return;
    float forceY = 0.0f;
    if (facts::WheelForceMode(configuration) ==
        static_cast<std::uint32_t>(
                CSceneVehicleCarWheelForceMode_DirectSpring)) {
        forceY =
                (cuda::facts::Tuning(configuration).suspension.
                         wheelRestDamperAbsorb -
                 wheel.realTime.damperAbsorb) *
                (cuda::facts::Tuning(configuration).suspension.
                         wheelSpringCoef *
                 cuda::facts::Tuning(configuration).suspension.
                         wheelStaticSpringScale);
    } else if (
            facts::WheelForceMode(configuration) ==
                    static_cast<std::uint32_t>(
                            CSceneVehicleCarWheelForceMode_FollowAbsorb) ||
            facts::WheelForceMode(configuration) ==
                    static_cast<std::uint32_t>(
                            CSceneVehicleCarWheelForceMode_FollowAbsorbWithImpulse)) {
        const float spring =
                (cuda::facts::Tuning(configuration).suspension.
                         wheelRestDamperAbsorb -
                 wheel.realTime.damperAbsorb) *
                cuda::facts::Tuning(configuration).suspension.
                        wheelSpringCoef;
        forceY =
                spring -
                cuda::facts::Tuning(configuration).suspension.
                        wheelDamperCoef *
                        wheel.realTime.damperVelocity;
    }
    AddForceAtPoint<ReuseWorldCenter>(
            candidate, {0.0f, forceY, 0.0f},
            facts::Wheel(
                    configuration,
                    wheelIndex).forceApplicationPoint,
            sharedWorldCenter);
}

__device__ inline float BurnoutPhase(
        std::uint32_t elapsed,
        std::uint32_t duration) {
    return exact::FromUnsignedInteger(elapsed) * Pi /
           exact::FromUnsignedInteger(duration);
}

__device__ inline float BurnoutFade(
        float scale, float fade) {
    return (scale - 1.0f) * fade + 1.0f;
}

__device__ inline float BurnoutDriveFade(
        const CudaVehicleState &vehicle,
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t tick) {
    if (vehicle.gearedDrive.burnoutPhase ==
        CSceneVehicleCarBurnoutPhase_TimedSpin) {
        return BurnoutFade(
                cuda::facts::Tuning(configuration).gearedDrive.burnout.
                        driveFadeScale,
                exact::Sin(BurnoutPhase(
                        tick -
                                vehicle.gearedDrive.
                                        burnoutStartTick,
                        cuda::facts::Tuning(configuration).gearedDrive.
                                burnout.durationTicks)));
    }
    if (vehicle.gearedDrive.burnoutPhase ==
        CSceneVehicleCarBurnoutPhase_ExitFade) {
        return BurnoutFade(
                cuda::facts::Tuning(configuration).gearedDrive.burnout.
                        exitAccelFadeScale,
                exact::Sin(BurnoutPhase(
                        tick -
                                vehicle.gearedDrive.
                                        burnoutExitStartTick,
                        cuda::facts::Tuning(configuration).gearedDrive.
                                burnout.exitDurationTicks)));
    }
    return 1.0f;
}

__device__ inline float BurnoutSideFade(
        const CudaVehicleState &vehicle,
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t tick) {
    if (vehicle.gearedDrive.burnoutPhase !=
        CSceneVehicleCarBurnoutPhase_TimedSpin) {
        return 1.0f;
    }
    return BurnoutFade(
            cuda::facts::Tuning(configuration).gearedDrive.burnout.
                    sideForceFadeScale,
            exact::Cos(BurnoutPhase(
                    tick -
                            vehicle.gearedDrive.
                                    burnoutStartTick,
                    cuda::facts::Tuning(configuration).gearedDrive.burnout.
                            durationTicks *
                            2u)));
}

__device__ inline float BurnoutExitAcceleration(
        const CudaVehicleState &vehicle,
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t tick) {
    if (vehicle.gearedDrive.burnoutPhase !=
        CSceneVehicleCarBurnoutPhase_ExitFade) {
        return 0.0f;
    }
    const std::uint32_t quotient =
            (tick -
             vehicle.gearedDrive.burnoutExitStartTick) /
            cuda::facts::Tuning(configuration).gearedDrive.burnout.
                    exitDurationTicks;
    const float delta =
            exact::FromUnsignedInteger(quotient) - 1.0f;
    return delta * delta *
           cuda::facts::Tuning(configuration).gearedDrive.burnout.
                   exitBonusAccelScale;
}

__device__ inline void AdvanceBurnout(
        CudaVehicleState &vehicle,
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t tick,
        int &slipSeen) {
    if (vehicle.gearedDrive.burnoutPhase ==
        CSceneVehicleCarBurnoutPhase_TimedSpin) {
        if (tick < vehicle.gearedDrive.burnoutStartTick ||
            tick - vehicle.gearedDrive.burnoutStartTick >=
                    cuda::facts::Tuning(configuration).gearedDrive.burnout.
                            durationTicks) {
            vehicle.gearedDrive.burnoutExitStartTick = tick;
            vehicle.gearedDrive.burnoutPhase =
                    CSceneVehicleCarBurnoutPhase_ExitFade;
        } else {
            slipSeen = 1;
        }
    }
    if (vehicle.gearedDrive.burnoutPhase ==
        CSceneVehicleCarBurnoutPhase_ExitFade) {
        if (tick < vehicle.gearedDrive.burnoutExitStartTick ||
            tick - vehicle.gearedDrive.burnoutExitStartTick >=
                    cuda::facts::Tuning(configuration).gearedDrive.burnout.
                            exitDurationTicks) {
            vehicle.gearedDrive.burnoutPhase =
                    CSceneVehicleCarBurnoutPhase_Inactive;
            vehicle.gearedDrive.wheelSpeedOverrideActive = false;
            return;
        }
        for (std::uint32_t index = 0u;
             index < facts::WheelCount(vehicle); ++index) {
            vehicle.wheels.values[index].
                    realTime.slipping = true;
        }
    }
}

__device__ inline bool AllWheelsMaterial(
        const CudaVehicleState &vehicle,
        std::uint32_t material) {
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        const auto &wheel = vehicle.wheels.values[index].realTime;
        if (!wheel.contactPresent ||
            static_cast<std::uint32_t>(wheel.contactMaterial) !=
                    material) {
            return false;
        }
    }
    return true;
}

__device__ inline void UpdateGearDirection(
        CudaVehicleState &vehicle,
        const GmVec3 &linearSpeed) {
    if (vehicle.gearedDrive.burnoutPhase !=
        CSceneVehicleCarBurnoutPhase_Inactive) {
        vehicle.engine.useLowSpeedGateB = false;
        return;
    }
    if (vehicle.controls.lowSpeedGateB >
                LowSpeedGateThreshold &&
        linearSpeed.z <
                vehicle.reverseGearSpeedThreshold &&
        fabsf(linearSpeed.x) < 2.0f) {
        vehicle.engine.useLowSpeedGateB = true;
    }
    if (vehicle.controls.lowSpeedGateA >
                LowSpeedGateThreshold &&
        (linearSpeed.z > 0.0f ||
         fabsf(linearSpeed.x) > 2.0f)) {
        vehicle.engine.useLowSpeedGateB = false;
    }
    if (vehicle.controls.lowSpeedGateA <
                LowSpeedGateThreshold &&
        vehicle.controls.lowSpeedGateB <
                LowSpeedGateThreshold) {
        vehicle.engine.useLowSpeedGateB =
                !(linearSpeed.z > 0.0f ||
                  fabsf(linearSpeed.z) < 2.0f);
    }
    if (linearSpeed.z > 0.0f &&
        vehicle.turbo.type !=
                CSceneVehicleCar::ETurboType_Inactive) {
        vehicle.engine.useLowSpeedGateB = false;
    }
}

__device__ inline float SteerAssistRamp(
        const CudaPackedStaticConfigurationHeader *configuration,
        const GmVec3 &linearSpeed) {
    const float speed = exact::Sqrt(
            linearSpeed.x * linearSpeed.x +
            linearSpeed.y * linearSpeed.y +
            linearSpeed.z * linearSpeed.z);
    if (speed < 0.7f) return 0.0f;
    if (speed >
        cuda::facts::Tuning(configuration).steering.assistFullSpeed) {
        return 1.0f;
    }
    return exact::Sin(
            (speed /
             cuda::facts::Tuning(configuration).steering.assistFullSpeed) *
            HalfPi);
}

__device__ inline void UpdateSlipMemory(
        CudaVehicleState &vehicle,
        std::uint32_t tick,
        int slipSeen) {
    if (slipSeen == 0) return;
    vehicle.slipMemory.lastTick = tick;
    if (!vehicle.slipMemory.active) {
        vehicle.slipMemory.startTick = tick;
    }
    vehicle.slipMemory.elapsedTicks =
            tick - vehicle.slipMemory.startTick;
}

__device__ inline float SlipAccelerationMix(
        const CudaVehicleState &vehicle,
        const CudaPackedStaticConfigurationHeader *configuration,
        std::uint32_t tick,
        float limit,
        float requested) {
    if (tick != vehicle.slipMemory.lastTick ||
        !(limit > ScalarEpsilon)) {
        return 1.0f;
    }
    const float slip =
            (requested - limit) / limit /
            cuda::facts::Tuning(configuration).gearedDrive.slipRatioScale;
    return 1.0f - ClampZeroOne(slip);
}

__device__ inline float SlippingWheelScale(
        const CudaVehicleState &vehicle,
        const CudaPackedStaticConfigurationHeader *configuration) {
    float result = 1.0f;
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        if (vehicle.wheels.values[index].realTime.slipping) {
            result *= cuda::facts::Tuning(configuration).gearedDrive.
                    perSlippingWheelAccelScale;
        }
    }
    return result;
}

__device__ inline void MarkAllWheelsSlipping(
        CudaVehicleState &vehicle) {
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        vehicle.wheels.values[index].realTime.slipping = true;
    }
}

struct Model6Result {
    float surfaceFeedback = 0.0f;
    int slipFlag = 0;
};

template <bool ReuseWheelPassInvariants>
__device__ inline ForceStatus ComputeModel3Ground(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        const GmVec3 &currentForce,
        float slopeA,
        float slopeB,
        const GmVec3 &linearSpeed,
        const GmVec3 &angularSpeed,
        float visualSteerYaw,
        bool hasMaterial,
        const VehicleMaterialBlendValues &material,
        Model6Result &result) {
    CudaVehicleState &vehicle = candidate.vehicle;
    const bool lateralHandling =
            cuda::facts::Tuning(configuration).handlingModel ==
            static_cast<std::uint32_t>(
                    CSceneVehicleCarHandlingModel_Lateral);
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        CudaWheelState &wheel = vehicle.wheels.values[index];
        WheelSuspensionForce<ReuseWheelPassInvariants>(
                candidate, configuration, wheel, index);
        if (!wheel.realTime.contactPresent ||
            !(cuda::facts::Tuning(configuration).gearedDrive.
                      lateralForceScale > 0.0f) ||
            !lateralHandling) {
            continue;
        }
        const VehicleMaterialDefinition *wheelMaterial =
                Material(
                        configuration,
                        static_cast<std::uint32_t>(
                                wheel.realTime.contactMaterial));
        if (wheelMaterial == nullptr) {
            return ForceStatus::MissingMaterial;
        }
        const float slipGrip = wheel.realTime.slipping
                ? cuda::facts::Tuning(configuration).gearedDrive.
                          slippingSideFrictionScale
                : 1.0f;
        const float maximumSide =
                wheelMaterial->blendableValues.w * slopeA *
                tuning::EvaluateSpeed(
                        configuration,
                        CudaTuningCurveId::
                                MaxSideFrictionFromSpeed,
                        linearSpeed.z) *
                slipGrip;
        GmVec3 lateral = {
                wheel.realTime.accumulatedContactNormal.y,
                -wheel.realTime.accumulatedContactNormal.x,
                0.0f,
        };
        lateral = wheel_detail::NormalizeOr(
                lateral, {1.0f, 0.0f, 0.0f},
                VectorEpsilonSquared);
        if (facts::WheelAxle(
                    configuration,
                    index) == VehicleWheelAxle::Front) {
            const exact::SinCosResult sinCos =
                    exact::SinCos(visualSteerYaw);
            lateral = {
                    sinCos.cosine * lateral.x,
                    sinCos.cosine * lateral.y,
                    -sinCos.sine +
                            sinCos.cosine * lateral.z,
            };
        }
        float sideForce =
                -cuda::facts::Tuning(configuration).gearedDrive.
                         lateralForceScale *
                0.5f *
                ((linearSpeed.y * lateral.y +
                  linearSpeed.x * lateral.x) +
                 linearSpeed.z * lateral.z);
        const float sideForceMagnitude = fabsf(sideForce);
        if (!(maximumSide < sideForceMagnitude)) {
            wheel.realTime.slipping = false;
        } else {
            const float capped =
                    SignNonNegative(sideForce) * maximumSide;
            sideForce =
                    (1.0f -
                     cuda::facts::Tuning(configuration).gearedDrive.
                             sideFrictionSlipBlend) *
                            capped +
                    cuda::facts::Tuning(configuration).gearedDrive.
                            sideFrictionSlipBlend *
                            sideForce;
            wheel.realTime.slipping = true;
        }
        if (wheel.realTime.slipping) {
            result.slipFlag = 1;
        }
        const GmVec3 lateralForce = {
                lateral.x * sideForce,
                lateral.y * sideForce,
                lateral.z * sideForce,
        };
        AddCentralForce(candidate, lateralForce);
        const float rollover =
                -tuning::EvaluateSpeed(
                        configuration,
                        CudaTuningCurveId::
                                RolloverLateralFromSpeed,
                        linearSpeed.z) *
                slopeA *
                tuning::Evaluate(
                        configuration,
                        CudaTuningCurveId::
                                RolloverLateralCoefficientFromAngle,
                        fabsf(lateral.y));
        AddTorque(candidate, {
                lateralForce.z * rollover,
                0.0f,
                -rollover * lateralForce.x,
        });
    }
    if (!hasMaterial || !lateralHandling) {
        return ForceStatus::Success;
    }

    const float speedMagnitude = exact::Sqrt(
            (linearSpeed.y * linearSpeed.y +
             linearSpeed.x * linearSpeed.x) +
            linearSpeed.z * linearSpeed.z);
    if (!vehicle.controls.forcedLowSpeedFriction) {
        if (!(speedMagnitude <
              vehicle.reverseGearSpeedThreshold)) {
            if (vehicle.controls.lowSpeedGateA >
                LowSpeedGateThreshold) {
                vehicle.engine.useLowSpeedGateB = false;
            }
        } else if (!(LowSpeedGateThreshold <
                     vehicle.controls.lowSpeedGateB)) {
            vehicle.engine.useLowSpeedGateB = false;
        } else {
            vehicle.engine.useLowSpeedGateB = true;
        }
    }
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        CudaWheelState &wheel = vehicle.wheels.values[index];
        const bool front =
                facts::WheelAxle(
                        configuration,
                        index) == VehicleWheelAxle::Front;
        const float halfTrack =
                (front
                         ? vehicle.gearedDrive.
                                   wheelLongitudinalSpan
                         : -vehicle.gearedDrive.
                                   wheelLongitudinalSpan) *
                0.5f;
        float steerRamp = 1.0f;
        if (!(cuda::facts::Tuning(configuration).steering.
                      assistFullSpeed < speedMagnitude)) {
            steerRamp = exact::Sin(
                    (speedMagnitude /
                     cuda::facts::Tuning(configuration).steering.
                             assistFullSpeed) *
                    HalfPi);
        }
        const float maximumSide =
                tuning::EvaluateSpeed(
                        configuration,
                        CudaTuningCurveId::
                                MaxSideFrictionFromSpeed,
                        linearSpeed.z) *
                material.w;
        const float wheelSideSpeed =
                linearSpeed.x +
                angularSpeed.y * halfTrack;
        float sideForce =
                -cuda::facts::Tuning(configuration).gearedDrive.
                         lateralForceScale *
                0.5f * wheelSideSpeed;
        if (maximumSide < fabsf(sideForce)) {
            const float blended =
                    (1.0f -
                     cuda::facts::Tuning(configuration).gearedDrive.
                             driveSideFrictionSlipBlend) *
                            maximumSide +
                    cuda::facts::Tuning(configuration).gearedDrive.
                            driveSideFrictionSlipBlend *
                            fabsf(sideForce);
            sideForce =
                    SignNonNegative(sideForce) * blended;
        }
        float sideTorque =
                cuda::facts::Tuning(configuration).gearedDrive.
                        sideForceToDriveTorqueScale *
                sideForce;
        if (front) {
            const float reverseSign =
                    !vehicle.engine.useLowSpeedGateB
                    ? 1.0f
                    : -1.0f;
            const float slipScale =
                    wheel.realTime.slipping
                    ? cuda::facts::Tuning(configuration).gearedDrive.
                              slippingSteerTorqueScale
                    : 1.0f;
            const float steerTorque =
                    tuning::EvaluateSpeed(
                            configuration,
                            CudaTuningCurveId::
                                    SteeringDriveTorqueFromSpeed,
                            linearSpeed.z);
            const float steerAssist =
                    reverseSign * steerRamp *
                    vehicle.controls.currentSteering *
                    steerTorque * slipScale;
            sideTorque = sideTorque - steerAssist;
        }
        AddTorque(
                candidate,
                {0.0f, sideTorque * halfTrack, 0.0f});
    }

    const float accelerationBase =
            tuning::EvaluateSpeed(
                    configuration,
                    CudaTuningCurveId::
                            SlipResponseAccelFromSpeed,
                    linearSpeed.z);
    const float sideLimit =
            tuning::EvaluateSpeed(
                    configuration,
                    CudaTuningCurveId::
                            MaxSideFrictionFromSpeed,
                    linearSpeed.z) *
            material.w;
    float sideSlowdownInput = fabsf(
            cuda::facts::Tuning(configuration).gearedDrive.
                    lateralForceScale *
            0.5f * linearSpeed.x);
    if (sideLimit < sideSlowdownInput) {
        sideSlowdownInput = sideLimit;
    }
    const float driveScale =
            material.y * vehicle.controls.lowSpeedGateA +
            (vehicle.engine.useLowSpeedGateB ? -1.0f : 0.0f) *
                    material.y *
                    vehicle.controls.lowSpeedGateB +
            (vehicle.turbo.type !=
                             CSceneVehicleCar::ETurboType_Inactive
                     ? vehicle.turbo.impulseScale
                     : 0.0f);
    float driveForce =
            (accelerationBase -
             cuda::facts::Tuning(configuration).steering.slowDownScale *
                     sideSlowdownInput *
                     fabsf(vehicle.controls.currentSteering) *
                     tuning::EvaluateSpeed(
                             configuration,
                             CudaTuningCurveId::
                                     SteerSlowDownFromSpeed,
                             linearSpeed.z)) *
            driveScale;
    if (vehicle.controls.forcedLowSpeedFriction) {
        driveForce =
                vehicle.turbo.type ==
                                CSceneVehicleCar::
                                        ETurboType_Inactive
                ? 0.0f
                : accelerationBase *
                          vehicle.turbo.impulseScale;
    }
    float opposing = 0.0f;
    if (linearSpeed.z > 0.0f) {
        opposing =
                (cuda::facts::Tuning(configuration).gearedDrive.
                         forwardAccelBase +
                 cuda::facts::Tuning(configuration).gearedDrive.
                         forwardAccelSpeedCoef *
                         linearSpeed.z) *
                vehicle.controls.lowSpeedGateB;
        const float cap =
                material.z *
                (result.slipFlag == 0
                         ? cuda::facts::Tuning(configuration).gearedDrive.
                                   forwardAccelCap
                         : cuda::facts::Tuning(configuration).gearedDrive.
                                   forwardAccelCapWhenSlipping);
        if (cap < opposing) {
            opposing = cap;
            MarkAllWheelsSlipping(vehicle);
        }
    }
    if (linearSpeed.z < 0.0f &&
        vehicle.controls.forcedLowSpeedFriction) {
        opposing =
                (cuda::facts::Tuning(configuration).gearedDrive.
                         forwardAccelBase -
                 cuda::facts::Tuning(configuration).gearedDrive.
                         forwardAccelSpeedCoef *
                         linearSpeed.z) *
                vehicle.controls.lowSpeedGateA;
        const float cap =
                material.z *
                (result.slipFlag == 0
                         ? cuda::facts::Tuning(configuration).gearedDrive.
                                   forwardAccelCap
                         : cuda::facts::Tuning(configuration).gearedDrive.
                                   forwardAccelCapWhenSlipping);
        if (cap < opposing) {
            opposing = cap;
            MarkAllWheelsSlipping(vehicle);
        }
        opposing = -opposing;
    }
    if (vehicle.controls.forcedLowSpeedFriction &&
        fabsf(linearSpeed.z) < 1.0f) {
        opposing *= fabsf(linearSpeed.z);
    }
    result.surfaceFeedback = opposing;
    float longitudinal = driveForce - opposing;
    if (cuda::facts::Tuning(configuration).engineSpeedNorm *
                material.x <
        linearSpeed.z) {
        longitudinal =
                -cuda::facts::Tuning(configuration).gearedDrive.
                         speedLimitForce;
    }
    if (linearSpeed.z <
        -(cuda::facts::Tuning(configuration).gearedDrive.
                  reverseSpeedNorm *
          material.x)) {
        longitudinal =
                cuda::facts::Tuning(configuration).gearedDrive.
                        speedLimitForce;
    }
    const float forceZ = longitudinal * slopeB;
    AddCentralForce(candidate, {0.0f, 0.0f, forceZ});
    AddTorque(candidate, {
            -forceZ *
                    cuda::facts::Tuning(configuration).slipResponse.
                            longitudinalTorqueScale,
            0.0f,
            0.0f,
    });
    AddCentralForce(candidate, {
            0.0f,
            0.0f,
            (-cuda::facts::Tuning(configuration).gearedDrive.
                     forceZScale *
             currentForce.z) /
                    cuda::facts::Tuning(configuration).bodyAirResponse.
                            groundedSolidFeedback1,
    });
    return ForceStatus::Success;
}

__device__ inline float SignedAngle(
        const GmVec3 &left,
        const GmVec3 &right) {
    const float contractedDot = exact::FromDouble(
            static_cast<double>(
                    dynamics::detail::Dot(left, right)) *
            (1.0 - static_cast<double>(ScalarEpsilon)));
    float angle = exact::Acos(contractedDot);
    if (angle <= ScalarEpsilon) {
        return angle;
    }
    const GmVec3 normalizedLeft = wheel_detail::NormalizeOr(
            left, left, VectorEpsilonSquared);
    const GmVec3 normalizedRight = wheel_detail::NormalizeOr(
            right, right, VectorEpsilonSquared);
    if (wheel_detail::Cross(
                normalizedLeft, normalizedRight).y < 0.0f) {
        angle = -angle;
    }
    return angle;
}

__device__ inline void EnterCircularBurnout(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        const GmVec3 &linearSpeed,
        float visualSteerYaw) {
    CudaVehicleState &vehicle = candidate.vehicle;
    auto &drive = vehicle.gearedDrive;
    drive.burnoutPhase =
            CSceneVehicleCarBurnoutPhase_CircularDrift;
    drive.burnoutDirection =
            SignNonNegative(visualSteerYaw);

    GmVec3 normalSum{};
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        const CudaWheelState &wheel = vehicle.wheels.values[index];
        if (!wheel.realTime.contactPresent) continue;
        normalSum.x =
                normalSum.x +
                wheel.realTime.accumulatedContactNormal.x;
        normalSum.y =
                wheel.realTime.accumulatedContactNormal.y +
                normalSum.y;
        normalSum.z =
                normalSum.z +
                wheel.realTime.accumulatedContactNormal.z;
    }
    const float normalLengthSquared =
            (normalSum.x * normalSum.x +
             normalSum.y * normalSum.y) +
            normalSum.z * normalSum.z;
    if (VectorEpsilonSquared < normalLengthSquared) {
        const float inverseLength =
                1.0f / exact::Sqrt(normalLengthSquared);
        drive.burnoutContactNormal = {
                normalSum.x * inverseLength,
                normalSum.y * inverseLength,
                normalSum.z * inverseLength,
        };
    } else {
        drive.burnoutContactNormal = {0.0f, 1.0f, 0.0f};
        drive.burnoutPhase =
                CSceneVehicleCarBurnoutPhase_Inactive;
    }

    const GmVec3 bodyCenter =
            candidate.body.physicalParameters.localCenterOfMass;
    const GmBoxAligned &waterBox = vehicle.water.boxLocal;
    float zExtra =
            waterBox.halfExtents.y * 0.0f +
            waterBox.halfExtents.x * 0.0f;
    zExtra += waterBox.halfExtents.z;
    const GmVec3 radiusSeed = {
            waterBox.center.x - bodyCenter.x,
            waterBox.center.y - bodyCenter.y,
            waterBox.center.z - bodyCenter.z + zExtra,
    };
    drive.burnoutBaseRadius = exact::Sqrt(
            (radiusSeed.x * radiusSeed.x +
             radiusSeed.y * radiusSeed.y) +
            radiusSeed.z * radiusSeed.z);

    GmVec3 tangent = wheel_detail::Cross(
            drive.burnoutContactNormal, linearSpeed);
    tangent.x *= drive.burnoutDirection;
    tangent.y *= drive.burnoutDirection;
    tangent.z *= drive.burnoutDirection;
    tangent = wheel_detail::NormalizeOr(
            tangent, tangent, VectorEpsilonSquared);

    drive.burnoutContactNormal = TransformDirection(
            drive.frameIso.rotation,
            drive.burnoutContactNormal);
    if (drive.burnoutContactNormal.y < 0.75f) {
        drive.burnoutPhase =
                CSceneVehicleCarBurnoutPhase_Inactive;
        drive.burnoutContactNormal = {0.0f, 1.0f, 0.0f};
    }

    const GmVec3 forwardAxis = {0.0f, 0.0f, 1.0f};
    const float signedAngle =
            SignedAngle(forwardAxis, tangent) *
            SignNonNegative(visualSteerYaw);
    if (signedAngle >
                cuda::facts::Tuning(configuration).gearedDrive.burnout.
                        angleLimitPositive ||
        signedAngle <
                -cuda::facts::Tuning(configuration).gearedDrive.burnout.
                        angleLimitNegative) {
        drive.burnoutPhase =
                CSceneVehicleCarBurnoutPhase_Inactive;
    } else {
        const GmVec3 tangentWorld = TransformDirection(
                drive.frameIso.rotation, tangent);
        GmVec3 bodyCenterWorld = TransformDirection(
                drive.frameIso.rotation, bodyCenter);
        bodyCenterWorld.x += drive.frameIso.translation.x;
        bodyCenterWorld.y += drive.frameIso.translation.y;
        bodyCenterWorld.z += drive.frameIso.translation.z;
        const float targetRadius = tuning::EvaluateSpeed(
                configuration,
                CudaTuningCurveId::BurnoutRadiusFromSpeed,
                linearSpeed.z) +
                drive.burnoutBaseRadius;
        drive.burnoutTargetRadius = targetRadius;
        drive.burnoutCenter = {
                bodyCenterWorld.x +
                        tangentWorld.x * targetRadius,
                bodyCenterWorld.y +
                        tangentWorld.y * targetRadius,
                bodyCenterWorld.z +
                        tangentWorld.z * targetRadius,
        };
    }
    drive.wheelSpeedOverrideActive =
            drive.burnoutPhase ==
            CSceneVehicleCarBurnoutPhase_CircularDrift;
}

__device__ inline void ApplyCircularBurnout(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        const GmVec3 &currentForce,
        const GmVec3 &linearSpeed,
        const GmVec3 &angularSpeed,
        float visualSteerYaw,
        bool hasGroundMaterial,
        int waterActive) {
    CudaVehicleState &vehicle = candidate.vehicle;
    auto &drive = vehicle.gearedDrive;
    if (vehicle.controls.lowSpeedGateA < LowSpeedGateThreshold ||
        vehicle.controls.lowSpeedGateB < LowSpeedGateThreshold ||
        fabsf(visualSteerYaw) < ScalarEpsilon ||
        drive.burnoutDirection !=
                SignNonNegative(visualSteerYaw) ||
        vehicle.contacts.lateralSlowDownContactActive ||
        vehicle.contacts.bodyContactPresent ||
        !hasGroundMaterial || waterActive != 0 ||
        vehicle.controls.forcedLowSpeedFriction) {
        drive.burnoutPhase =
                CSceneVehicleCarBurnoutPhase_Inactive;
        return;
    }

    GmVec3 normalSum{};
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        const GmVec3 &normal =
                vehicle.wheels.values[index].realTime.
                        accumulatedContactNormal;
        normalSum.x = normalSum.x + normal.x;
        normalSum.y = normal.y + normalSum.y;
        normalSum.z = normalSum.z + normal.z;
    }
    const GmVec3 averageNormal = wheel_detail::NormalizeOr(
            normalSum, {0.0f, 1.0f, 0.0f},
            VectorEpsilonSquared);
    const GmVec3 worldNormal = TransformDirection(
            drive.frameIso.rotation, averageNormal);
    const float normalDrift = fabsf(SignedAngle(
            worldNormal, drive.burnoutContactNormal));
    if (normalDrift >
        cuda::facts::Tuning(configuration).gearedDrive.burnout.angleLimit) {
        drive.burnoutPhase =
                CSceneVehicleCarBurnoutPhase_Inactive;
        return;
    }

    const GmVec3 bodyCenter =
            candidate.body.physicalParameters.localCenterOfMass;
    const GmMat3 &frameRotation = drive.frameIso.rotation;
    const GmVec3 negativeTranslation = {
            -drive.frameIso.translation.x,
            -drive.frameIso.translation.y,
            -drive.frameIso.translation.z,
    };
    const GmVec3 localCenter = {
            dynamics::detail::Dot(
                    frameRotation.basisX,
                    drive.burnoutCenter) +
                    dynamics::detail::Dot(
                            frameRotation.basisX,
                            negativeTranslation),
            dynamics::detail::Dot(
                    frameRotation.basisY,
                    drive.burnoutCenter) +
                    dynamics::detail::Dot(
                            frameRotation.basisY,
                            negativeTranslation),
            dynamics::detail::Dot(
                    frameRotation.basisZ,
                    drive.burnoutCenter) +
                    dynamics::detail::Dot(
                            frameRotation.basisZ,
                            negativeTranslation),
    };
    const GmVec3 radial = {
            localCenter.x - bodyCenter.x,
            localCenter.y - bodyCenter.y,
            localCenter.z - bodyCenter.z,
    };
    const float radiusSquared =
            (radial.y * radial.y + radial.x * radial.x) +
            radial.z * radial.z;
    const float radius = exact::Sqrt(radiusSquared);
    GmVec3 radialDirection = radial;
    if (radiusSquared > VectorEpsilonSquared) {
        const float inverseRadius = 1.0f / radius;
        radialDirection = {
                radial.x * inverseRadius,
                radial.y * inverseRadius,
                radial.z * inverseRadius,
        };
    }
    const GmVec3 tangent = wheel_detail::Cross(
            averageNormal, radialDirection);
    const float tangentSpeed =
            dynamics::detail::Dot(linearSpeed, tangent);
    const float radialSpeed =
            dynamics::detail::Dot(linearSpeed, radialDirection);
    if (fabsf(tangentSpeed) >
        cuda::facts::Tuning(configuration).gearedDrive.burnout.
                tangentSpeedMax) {
        drive.burnoutPhase =
                CSceneVehicleCarBurnoutPhase_Inactive;
    }
    const bool radiusNeedsCorrection =
            radius > drive.burnoutBaseRadius ||
            radius <
                    cuda::facts::Tuning(configuration).gearedDrive.burnout.
                            radiusMin;
    if (!radiusNeedsCorrection) {
        drive.burnoutPhase =
                CSceneVehicleCarBurnoutPhase_Inactive;
    } else {
        const float correctionExponent =
                (radius - drive.burnoutTargetRadius) *
                        cuda::facts::Tuning(configuration).gearedDrive.
                                burnout.radiusCorrectionScale -
                radialSpeed *
                        cuda::facts::Tuning(configuration).gearedDrive.
                                burnout.
                                radiusCorrectionSpeedScale;
        const float tangentSpeedSquared =
                tangentSpeed * tangentSpeed;
        const float correctionMagnitude =
                (tangentSpeedSquared / radius) *
                exact::Exp(correctionExponent);
        AddCentralForce(candidate, {
                radialDirection.x * correctionMagnitude,
                radialDirection.y * correctionMagnitude,
                radialDirection.z * correctionMagnitude,
        });
    }

    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        CudaWheelState &wheel = vehicle.wheels.values[index];
        WheelSuspensionForce(
                candidate, configuration, wheel, index);
        wheel.realTime.slipping = false;
    }

    const float lateralSpeed = exact::FromDouble(
            static_cast<double>(tuning::Evaluate(
                    configuration,
                    CudaTuningCurveId::
                            BurnoutLateralSpeedFromRadius,
                    radius)) /
            static_cast<double>(3.6f));
    const float lateralScale =
            cuda::facts::Tuning(configuration).gearedDrive.burnout.
                    lateralCorrectionScale *
            (-SignNonNegative(visualSteerYaw) * lateralSpeed -
             tangentSpeed);
    AddCentralForce(candidate, {
            tangent.x * lateralScale,
            tangent.y * lateralScale,
            tangent.z * lateralScale,
    });

    const GmVec3 forwardAxis = {0.0f, 0.0f, 1.0f};
    const float angle =
            SignedAngle(forwardAxis, radialDirection);
    const double pi = static_cast<double>(Pi);
    const float angleNorm = exact::FromDouble(
            static_cast<double>(angle) / pi);
    const double signedAngle =
            static_cast<double>(drive.burnoutDirection) *
            static_cast<double>(angleNorm) * pi;
    if (!isfinite(angleNorm) ||
        signedAngle <
                -static_cast<double>(
                        cuda::facts::Tuning(configuration).gearedDrive.
                                burnout.angleLimitNegative) ||
        signedAngle >
                static_cast<double>(
                        cuda::facts::Tuning(configuration).gearedDrive.
                                burnout.angleLimitPositive)) {
        drive.burnoutPhase =
                CSceneVehicleCarBurnoutPhase_Inactive;
    } else {
        float angleReturnTorque = 0.0f;
        const float angularY = angularSpeed.y;
        if (angleNorm <= 0.0f) {
            if (!(angularY > 0.0f)) {
                const float delta = angleNorm + 1.0f;
                const float deltaSquared = delta * delta;
                angleReturnTorque =
                        -cuda::facts::Tuning(configuration).gearedDrive.
                                 burnout.angleReturnQuadratic *
                        angularY * deltaSquared;
            } else {
                angleReturnTorque =
                        -cuda::facts::Tuning(configuration).gearedDrive.
                                 burnout.angularDampingLinear *
                        angularY;
            }
        } else if (!(angularY < 0.0f)) {
            const float delta = angleNorm - 1.0f;
            const float deltaSquared = delta * delta;
            angleReturnTorque =
                    -cuda::facts::Tuning(configuration).gearedDrive.
                             burnout.angleReturnQuadratic *
                    angularY * deltaSquared;
        } else {
            angleReturnTorque =
                    -cuda::facts::Tuning(configuration).gearedDrive.
                             burnout.angularDampingLinear *
                    angularY;
        }
        const float tangentAngularSpeed =
                tangentSpeed / radius;
        float tangentDampingTorque = 0.0f;
        if ((angleNorm <= 0.0f &&
             tangentAngularSpeed > 0.0f) ||
            (angleNorm > 0.0f &&
             tangentAngularSpeed < 0.0f)) {
            tangentDampingTorque =
                    -cuda::facts::Tuning(configuration).gearedDrive.
                             burnout.tangentAngularDamping *
                    tangentAngularSpeed;
        }
        const float circleTorqueY =
                (cuda::facts::Tuning(configuration).gearedDrive.burnout.
                         angleTorqueScale *
                 angleNorm +
                 angleReturnTorque) +
                tangentDampingTorque;
        AddTorque(candidate, {0.0f, circleTorqueY, 0.0f});
    }

    AddTorque(candidate, {
            0.0f,
            0.0f,
            tuning::EvaluateSpeed(
                    configuration,
                    CudaTuningCurveId::DonutRolloverFromSpeed,
                    fabsf(linearSpeed.x)) *
                    -SignNonNegative(linearSpeed.x),
    });
    AddTorque(candidate, {
            tuning::EvaluateSpeed(
                    configuration,
                    CudaTuningCurveId::BurnoutRolloverFromSpeed,
                    linearSpeed.z),
            0.0f,
            0.0f,
    });
}

__device__ inline void ApplyDirtSlide(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        bool dirtSurface,
        const GmVec3 &linearSpeed) {
    if (!dirtSurface || linearSpeed.z <= 6.0f) return;
    CudaVehicleState &vehicle = candidate.vehicle;
    if (vehicle.controls.lowSpeedGateB >
        LowSpeedGateThreshold) {
        AddCentralForce(candidate, {
                -0.1f * linearSpeed.x,
                -0.1f * linearSpeed.y,
                -0.1f * linearSpeed.z,
        });
    }
    const bool canApply =
            vehicle.controls.lowSpeedGateB <
                    LowSpeedGateThreshold ||
            vehicle.controls.lowSpeedGateB ==
                    LowSpeedGateThreshold;
    if (vehicle.controls.forcedLowSpeedFriction ||
        vehicle.controls.lowSpeedGateA <=
                LowSpeedGateThreshold ||
        !canApply) {
        return;
    }
    const GmVec3 unitSpeed = wheel_detail::NormalizeOr(
            linearSpeed, {0.0f, 0.0f, 1.0f},
            VectorEpsilonSquared);
    const float absoluteUnitX = fabsf(unitSpeed.x);
    const float absoluteZ = fabsf(linearSpeed.z);
    const float absoluteXGate =
            absoluteUnitX * 20.0f + 1.0f;
    const float denominatorBase = absoluteZ + 1.0f;
    const float denominator =
            denominatorBase * denominatorBase;
    const float sideDenominator =
            cuda::facts::Tuning(configuration).gearedDrive.
                    dirtSlideGateScale *
                    vehicle.controls.lowSpeedGateA +
            1.0f;
    const GmVec3 front = {
            (cuda::facts::Tuning(configuration).gearedDrive.
                     dirtSlideSideForceScale *
             unitSpeed.x) /
                    sideDenominator,
            0.0f,
            cuda::facts::Tuning(configuration).gearedDrive.
                            dirtSlideForwardGateScale *
                    vehicle.controls.lowSpeedGateA *
                    1.5f * absoluteUnitX * absoluteXGate *
                    cuda::facts::Tuning(configuration).gearedDrive.
                            dirtSlideForwardForceScale /
                    denominator,
    };
    const GmVec3 rear = {
            -front.x,
            0.0f,
            cuda::facts::Tuning(configuration).gearedDrive.
                            dirtSlideForwardGateScale *
                    vehicle.controls.lowSpeedGateA *
                    absoluteUnitX * absoluteXGate *
                    cuda::facts::Tuning(configuration).gearedDrive.
                            dirtSlideForwardForceScale /
                    denominator,
    };
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        const CudaWheelState &wheel =
                vehicle.wheels.values[index];
        if (!wheel.realTime.slipping) continue;
        if (index <= 1u) {
            AddForceAtPoint(
                    candidate, front,
                    facts::Wheel(
                            configuration,
                            index).forceApplicationPoint);
        }
        if (index == 2u || index == 3u) {
            AddForceAtPoint(
                    candidate, rear,
                    facts::Wheel(
                            configuration,
                            index).forceApplicationPoint);
        }
    }
}

__device__ inline GmVec3 GroundFeedbackForce(
        const CudaVehicleState &vehicle,
        const CudaPackedStaticConfigurationHeader *configuration) {
    GmVec3 result = {
            vehicle.gearedDrive.scaledCurrentForce.x *
                    -cuda::facts::Tuning(configuration).feedback.forceDivisor,
            vehicle.gearedDrive.scaledCurrentForce.y *
                    -cuda::facts::Tuning(configuration).feedback.forceDivisor,
            vehicle.gearedDrive.scaledCurrentForce.z *
                    -cuda::facts::Tuning(configuration).feedback.forceDivisor,
    };
    if (exact::Sqrt(dynamics::detail::Dot(result, result)) <
        cuda::facts::Tuning(configuration).gearedDrive.currentForceTorqueMin) {
        result = {};
    }
    return result;
}

template <bool ReuseWheelPassInvariants>
__device__ inline ForceStatus ComputeModel6Ground(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        float dt,
        const GmVec3 &currentForce,
        float slopeA,
        float slopeB,
        const GmVec3 &linearSpeed,
        const GmVec3 &angularSpeed,
        float visualSteerYaw,
        bool hasGroundMaterial,
        const VehicleMaterialBlendValues &material,
        int waterActive,
        Model6Result &result) {
    CudaVehicleState &vehicle = candidate.vehicle;
    const std::uint32_t tick = candidate.world.tickTimeMs;
    vehicle.gearedDrive.frameIso.rotation =
            candidate.body.write.rotation;
    vehicle.gearedDrive.frameIso.translation =
            candidate.body.write.position;
    vehicle.controls.noGroundFrictionGuard =
            waterActive != 0;
    const bool dirt = AllWheelsMaterial(
            vehicle,
            static_cast<std::uint32_t>(
                    EPlugSurfaceMaterialId_Dirt));
    if (vehicle.gearedDrive.burnoutPhase ==
        CSceneVehicleCarBurnoutPhase_CircularDrift) {
        ApplyCircularBurnout(
                candidate, configuration, currentForce,
                linearSpeed, angularSpeed, visualSteerYaw,
                hasGroundMaterial, waterActive);
        if (vehicle.gearedDrive.burnoutPhase !=
            CSceneVehicleCarBurnoutPhase_CircularDrift) {
            vehicle.gearedDrive.burnoutPhase =
                    CSceneVehicleCarBurnoutPhase_TimedSpin;
            vehicle.gearedDrive.burnoutStartTick = tick;
        }
        vehicle.gearedDrive.localSpeed = linearSpeed;
        return ForceStatus::Success;
    }
    int slipSeen = 0;
    AdvanceBurnout(vehicle, configuration, tick, slipSeen);

    const float sideFade =
            BurnoutSideFade(vehicle, configuration, tick);
    exact::SinCosResult steeringSinCos{};
    GmVec3 sharedFeedbackForce{};
    GmVec3 sharedWorldCenter{};
    float sharedMaximumSideFriction = 0.0f;
    float sharedBurnoutRollover = 0.0f;
    if constexpr (ReuseWheelPassInvariants) {
        steeringSinCos = exact::SinCos(visualSteerYaw);
        sharedFeedbackForce =
                GroundFeedbackForce(vehicle, configuration);
        sharedWorldCenter =
                WorldCenterOfMass(candidate);
        sharedMaximumSideFriction =
                tuning::EvaluateSpeed(
                        configuration,
                        CudaTuningCurveId::
                                MaxSideFrictionFromSpeed,
                        linearSpeed.z);
        if (vehicle.gearedDrive.burnoutPhase ==
            CSceneVehicleCarBurnoutPhase_TimedSpin) {
            sharedBurnoutRollover =
                    tuning::EvaluateSpeed(
                            configuration,
                            CudaTuningCurveId::
                                    BurnoutRolloverFromSpeed,
                            linearSpeed.z);
        }
    }
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        CudaWheelState &wheel = vehicle.wheels.values[index];
        WheelSuspensionForce<ReuseWheelPassInvariants>(
                candidate, configuration, wheel, index,
                sharedWorldCenter);
        if (!wheel.realTime.contactPresent ||
            !(cuda::facts::Tuning(configuration).gearedDrive.
                      lateralForceScale >= 0.0f)) {
            continue;
        }
        const VehicleMaterialDefinition *wheelMaterial =
                Material(
                        configuration,
                        static_cast<std::uint32_t>(
                                wheel.realTime.contactMaterial));
        if (wheelMaterial == nullptr) {
            return ForceStatus::MissingMaterial;
        }
        const GmVec3 lever = {
                wheel.realTime.latestContactPoint.x -
                        configuration->dynaParameters.
                                localCenterOfMass.x,
                wheel.realTime.latestContactPoint.y -
                        configuration->dynaParameters.
                                localCenterOfMass.y,
                wheel.realTime.latestContactPoint.z -
                        configuration->dynaParameters.
                                localCenterOfMass.z,
        };
        GmVec3 sideAxis = {
                wheel.realTime.accumulatedContactNormal.y,
                -wheel.realTime.accumulatedContactNormal.x,
                0.0f,
        };
        sideAxis = wheel_detail::NormalizeOr(
                sideAxis, {1.0f, 0.0f, 0.0f},
                VectorEpsilonSquared);
        if (facts::WheelAxle(
                    configuration,
                    index) == VehicleWheelAxle::Front) {
            exact::SinCosResult sinCos;
            if constexpr (ReuseWheelPassInvariants) {
                sinCos = steeringSinCos;
            } else {
                sinCos = exact::SinCos(visualSteerYaw);
            }
            const float cosine = sinCos.cosine;
            const float negativeSine = -sinCos.sine;
            sideAxis = {
                    cosine * sideAxis.x,
                    cosine * sideAxis.y,
                    negativeSine + cosine * sideAxis.z,
            };
        }
        GmVec3 feedbackForce;
        if constexpr (ReuseWheelPassInvariants) {
            feedbackForce = sharedFeedbackForce;
        } else {
            feedbackForce =
                    GroundFeedbackForce(vehicle, configuration);
        }
        GmVec3 feedbackTorque =
                wheel_detail::Cross(lever, feedbackForce);
        feedbackTorque.x =
                -feedbackTorque.x *
                cuda::facts::Tuning(configuration).gearedDrive.
                        currentTorqueXScale;
        feedbackTorque.y = 0.0f;
        feedbackTorque.z =
                -feedbackTorque.z *
                cuda::facts::Tuning(configuration).gearedDrive.
                        currentTorqueZScale;
        AddTorque(candidate, feedbackTorque);
        if (vehicle.gearedDrive.burnoutPhase ==
            CSceneVehicleCarBurnoutPhase_TimedSpin) {
            float rollover;
            if constexpr (ReuseWheelPassInvariants) {
                rollover = sharedBurnoutRollover;
            } else {
                rollover = tuning::EvaluateSpeed(
                        configuration,
                        CudaTuningCurveId::
                                BurnoutRolloverFromSpeed,
                        linearSpeed.z);
            }
            AddTorque(candidate, {
                    rollover,
                    0.0f,
                    0.0f,
            });
        }
        float normalizedDamper = 0.0f;
        const float minAbsorb =
                cuda::facts::Tuning(configuration).suspension.
                        damperModulationMinAbsorb;
        const float maxAbsorb =
                cuda::facts::Tuning(configuration).suspension.
                        damperModulationMaxAbsorb;
        if (minAbsorb != maxAbsorb) {
            normalizedDamper =
                    (wheel.realTime.damperAbsorb - minAbsorb) /
                    (maxAbsorb - minAbsorb);
        }
        const float damper = tuning::Evaluate(
                configuration,
                CudaTuningCurveId::
                        SuspensionDamperAbsorbModulation,
                normalizedDamper);
        const float slipGrip = wheel.realTime.slipping
                ? cuda::facts::Tuning(configuration).gearedDrive.
                          slippingSideFrictionScale
                : 1.0f;
        const float lowSpeedGrip =
                wheel.realTime.slipping &&
                        vehicle.controls.lowSpeedGateB >
                                LowSpeedGateThreshold
                ? cuda::facts::Tuning(configuration).gearedDrive.
                          lowSpeedBSlippingGripScale
                : 1.0f;
        float maximumSideFriction;
        if constexpr (ReuseWheelPassInvariants) {
            maximumSideFriction =
                    sharedMaximumSideFriction;
        } else {
            maximumSideFriction =
                    tuning::EvaluateSpeed(
                            configuration,
                            CudaTuningCurveId::
                                    MaxSideFrictionFromSpeed,
                            linearSpeed.z);
        }
        const float maximum =
                wheelMaterial->blendableValues.w * slopeA *
                maximumSideFriction *
                slipGrip * lowSpeedGrip * damper;
        const float sideSpeed =
                dynamics::detail::Dot(linearSpeed, sideAxis);
        float requested =
                -cuda::facts::Tuning(configuration).gearedDrive.
                         lateralForceScale *
                0.5f * sideSpeed * sideFade;
        if (!(maximum < fabsf(requested))) {
            wheel.realTime.slipping = false;
        } else {
            requested =
                    (1.0f -
                     cuda::facts::Tuning(configuration).gearedDrive.
                             sideFrictionSlipBlend) *
                            SignNonNegative(requested) * maximum +
                    cuda::facts::Tuning(configuration).gearedDrive.
                            sideFrictionSlipBlend *
                            requested;
            wheel.realTime.slipping = true;
            result.slipFlag = 1;
        }
        ApplyDirtSlide(
                candidate, configuration, dirt, linearSpeed);
        AddCentralForce(candidate, {
                sideAxis.x * requested,
                sideAxis.y * requested,
                sideAxis.z * requested,
        });
    }

    if (!hasGroundMaterial) {
        if (vehicle.gearedDrive.burnoutPhase ==
            CSceneVehicleCarBurnoutPhase_TimedSpin) {
            vehicle.gearedDrive.burnoutExitStartTick = tick;
            vehicle.gearedDrive.burnoutPhase =
                    CSceneVehicleCarBurnoutPhase_ExitFade;
        }
        vehicle.slipMemory.active = slipSeen != 0;
        vehicle.gearedDrive.localSpeed = linearSpeed;
        return ForceStatus::Success;
    }

    if (vehicle.controls.forcedLowSpeedFriction == 0u &&
        vehicle.controls.lowSpeedGateB >
                LowSpeedGateThreshold &&
        vehicle.controls.lowSpeedGateA <
                LowSpeedGateThreshold &&
        vehicle.gearedDrive.burnoutPhase ==
                CSceneVehicleCarBurnoutPhase_TimedSpin) {
        vehicle.gearedDrive.burnoutExitStartTick = tick;
        vehicle.gearedDrive.burnoutPhase =
                CSceneVehicleCarBurnoutPhase_ExitFade;
    }
    if (vehicle.controls.forcedLowSpeedFriction == 0u &&
        vehicle.controls.lowSpeedGateA >
                LowSpeedGateThreshold &&
        vehicle.controls.lowSpeedGateB >
                LowSpeedGateThreshold) {
        if (linearSpeed.z <
                    cuda::facts::Tuning(configuration).gearedDrive.burnout.
                            donutSpeedLow &&
            vehicle.gearedDrive.frameIso.rotation.basisY.y >
                    0.75f) {
            vehicle.gearedDrive.burnoutPhase =
                    CSceneVehicleCarBurnoutPhase_TimedSpin;
            vehicle.gearedDrive.wheelSpeedOverrideActive = true;
            vehicle.gearedDrive.burnoutStartTick = tick;
        } else if (
                linearSpeed.z <
                        cuda::facts::Tuning(configuration).gearedDrive.burnout.
                                donutSpeedHigh &&
                linearSpeed.z >
                        cuda::facts::Tuning(configuration).gearedDrive.burnout.
                                donutSpeedLow &&
                !(fabsf(visualSteerYaw) < ScalarEpsilon)) {
            EnterCircularBurnout(
                    candidate, configuration,
                    linearSpeed, visualSteerYaw);
        }
    }
    UpdateGearDirection(vehicle, linearSpeed);
    const float rolloverInput =
            (linearSpeed.x * linearSpeed.x) /
            (fabsf(linearSpeed.z) + 1.0f);
    AddTorque(candidate, {
            0.0f,
            0.0f,
            -SignNonNegative(linearSpeed.x) *
                    tuning::EvaluateSpeed(
                            configuration,
                            CudaTuningCurveId::
                                    BurnoutRolloverLateralFromSpeedRatio,
                            rolloverInput),
    });
    const float steerRamp =
            SteerAssistRamp(configuration, linearSpeed);
    float sharedSteeringDriveTorque = 0.0f;
    if constexpr (ReuseWheelPassInvariants) {
        sharedSteeringDriveTorque =
                tuning::EvaluateSpeed(
                        configuration,
                        CudaTuningCurveId::
                                SteeringDriveTorqueFromSpeed,
                        linearSpeed.z);
    }
    float sideLimit = 0.0f;
    float sideRequested = 0.0f;
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        CudaWheelState &wheel = vehicle.wheels.values[index];
        const bool front =
                facts::WheelAxle(
                        configuration,
                        index) == VehicleWheelAxle::Front;
        const float halfTrack =
                (front
                         ? vehicle.gearedDrive.
                                   wheelLongitudinalSpan
                         : -vehicle.gearedDrive.
                                   wheelLongitudinalSpan) *
                0.5f;
        const float wheelSpeed =
                linearSpeed.x + angularSpeed.y * halfTrack;
        float maximumSideFriction;
        if constexpr (ReuseWheelPassInvariants) {
            maximumSideFriction =
                    sharedMaximumSideFriction;
        } else {
            maximumSideFriction =
                    tuning::EvaluateSpeed(
                            configuration,
                            CudaTuningCurveId::
                                    MaxSideFrictionFromSpeed,
                            linearSpeed.z);
        }
        const float maximum =
                maximumSideFriction * material.w;
        float requested =
                -cuda::facts::Tuning(configuration).gearedDrive.
                         lateralForceScale *
                0.5f * wheelSpeed;
        bool slipped = false;
        if (fabsf(requested) > maximum) {
            const float clipped =
                    (1.0f -
                     cuda::facts::Tuning(configuration).gearedDrive.
                             driveSideFrictionSlipBlend) *
                            maximum +
                    cuda::facts::Tuning(configuration).gearedDrive.
                            driveSideFrictionSlipBlend *
                            fabsf(requested);
            sideLimit += maximum;
            sideRequested += fabsf(requested);
            requested = SignNonNegative(requested) * clipped;
            slipped = true;
        }
        float sideTorque =
                cuda::facts::Tuning(configuration).gearedDrive.
                        sideForceToDriveTorqueScale *
                requested;
        if (front) {
            float steeringDriveTorque;
            if constexpr (ReuseWheelPassInvariants) {
                steeringDriveTorque =
                        sharedSteeringDriveTorque;
            } else {
                steeringDriveTorque =
                        tuning::EvaluateSpeed(
                                configuration,
                                CudaTuningCurveId::
                                        SteeringDriveTorqueFromSpeed,
                                linearSpeed.z);
            }
            float assist =
                    steerRamp *
                    vehicle.controls.currentSteering *
                    steeringDriveTorque;
            if (vehicle.engine.useLowSpeedGateB) {
                assist = -assist;
            }
            if (wheel.realTime.slipping) {
                assist *= cuda::facts::Tuning(configuration).gearedDrive.
                        slippingSteerTorqueScale;
            }
            sideTorque -= assist;
        }
        AddTorque(
                candidate,
                {0.0f, sideTorque * halfTrack, 0.0f});
        slipSeen |= slipped;
    }
    UpdateSlipMemory(vehicle, tick, slipSeen);
    const float slipMix = SlipAccelerationMix(
            vehicle, configuration, tick,
            sideLimit, sideRequested);
    const float slippingAcceleration =
            cuda::facts::Tuning(configuration).slipResponse.
                    slippingAccelScale *
            tuning::EvaluateSpeed(
                    configuration,
                    CudaTuningCurveId::
                            SlipResponseSlippingAccelFromSpeed,
                    linearSpeed.z);
    const float gearAcceleration =
            vehicle.engine.useLowSpeedGateB
            ? tuning::EvaluateSpeed(
                      configuration,
                      CudaTuningCurveId::ReverseGearAccelFromSpeed,
                      linearSpeed.z)
            : tuning::EvaluateSpeed(
                      configuration,
                      CudaTuningCurveId::SlipResponseAccelFromSpeed,
                      linearSpeed.z, true);
    const float blended =
            vehicle.gearedDrive.engineState ==
                    CSceneVehicleCarEngineControlState_GearShift
            ? 0.0f
            : (1.0f - slipMix) * slippingAcceleration +
                      gearAcceleration * slipMix;
    const float turbo =
            vehicle.turbo.type !=
                    CSceneVehicleCar::ETurboType_Inactive
            ? gearAcceleration * vehicle.turbo.impulseScale
            : 0.0f;
    const float rearSign =
            vehicle.engine.useLowSpeedGateB ? -1.0f : 0.0f;
    const float slowdown =
            cuda::facts::Tuning(configuration).steering.slowDownScale *
            fabsf(vehicle.controls.currentSteering) *
            tuning::EvaluateSpeed(
                    configuration,
                    CudaTuningCurveId::SteerSlowDownFromSpeed,
                    linearSpeed.z, true) *
            (vehicle.engine.useLowSpeedGateB ? -1.0f : 1.0f);
    float drive =
            BurnoutExitAcceleration(
                    vehicle, configuration, tick) +
            BurnoutDriveFade(vehicle, configuration, tick) *
                    (turbo +
                     (vehicle.controls.lowSpeedGateA * material.y +
                      rearSign * material.y *
                              vehicle.controls.lowSpeedGateB) *
                             blended) -
            slowdown;
    if (waterActive != 0) {
        drive *= 0.5f;
    }
    if (vehicle.controls.forcedLowSpeedFriction) {
        drive = vehicle.turbo.type !=
                        CSceneVehicleCar::ETurboType_Inactive
                ? gearAcceleration * vehicle.turbo.impulseScale
                : 0.0f;
    }
    float opposing = 0.0f;
    bool opposingSlipped = false;
    if (linearSpeed.z > 0.0f) {
        opposing =
                (cuda::facts::Tuning(configuration).gearedDrive.
                         forwardAccelBase +
                 cuda::facts::Tuning(configuration).gearedDrive.
                         forwardAccelSpeedCoef *
                         linearSpeed.z) *
                vehicle.controls.lowSpeedGateB *
                SlippingWheelScale(vehicle, configuration);
        const float cap =
                material.z *
                (result.slipFlag
                         ? cuda::facts::Tuning(configuration).gearedDrive.
                                   forwardAccelCapWhenSlipping
                         : cuda::facts::Tuning(configuration).gearedDrive.
                                   forwardAccelCap);
        if (cap < opposing) {
            opposing = cap;
            MarkAllWheelsSlipping(vehicle);
            opposingSlipped = true;
        }
    } else if (
            linearSpeed.z < 0.0f &&
            vehicle.controls.lowSpeedGateA >
                    LowSpeedGateThreshold) {
        if (!vehicle.controls.forcedLowSpeedFriction &&
            cuda::facts::Tuning(configuration).gearedDrive.burnout.
                            reverseForceThreshold <
                    -drive *
                            cuda::facts::Tuning(configuration).feedback.
                                    forceDivisor *
                            linearSpeed.z &&
            vehicle.gearedDrive.frameIso.rotation.basisY.y >
                    0.75f) {
            vehicle.gearedDrive.burnoutStartTick = tick;
            vehicle.gearedDrive.burnoutPhase =
                    CSceneVehicleCarBurnoutPhase_TimedSpin;
            vehicle.gearedDrive.wheelSpeedOverrideActive =
                    true;
        }
        opposing =
                (cuda::facts::Tuning(configuration).gearedDrive.
                         forwardAccelBase -
                 cuda::facts::Tuning(configuration).gearedDrive.
                         forwardAccelSpeedCoef *
                         linearSpeed.z) *
                vehicle.controls.lowSpeedGateA *
                SlippingWheelScale(vehicle, configuration);
        const float cap =
                material.z *
                (result.slipFlag
                         ? cuda::facts::Tuning(configuration).gearedDrive.
                                   forwardAccelCapWhenSlippingReverse
                         : cuda::facts::Tuning(configuration).gearedDrive.
                                   forwardAccelCapReverse);
        if (cap < opposing) {
            opposing = cap;
            MarkAllWheelsSlipping(vehicle);
            opposingSlipped = true;
        }
    }
    slipSeen |= opposingSlipped;
    result.surfaceFeedback = opposing;
    float longitudinal =
            drive - SignNonNegative(linearSpeed.z) * opposing;
    if (linearSpeed.z >
        cuda::facts::Tuning(configuration).engineSpeedNorm * material.x) {
        longitudinal = !(longitudinal < 0.0f)
                ? -cuda::facts::Tuning(configuration).gearedDrive.
                          speedLimitForce
                : longitudinal -
                          cuda::facts::Tuning(configuration).gearedDrive.
                                  speedLimitForce;
    }
    if (linearSpeed.z <
        -cuda::facts::Tuning(configuration).gearedDrive.reverseSpeedNorm *
                material.x) {
        longitudinal = !(longitudinal > 0.0f)
                ? cuda::facts::Tuning(configuration).gearedDrive.
                          speedLimitForce
                : longitudinal +
                          cuda::facts::Tuning(configuration).gearedDrive.
                                  speedLimitForce;
    }
    AddCentralForce(
            candidate,
            {0.0f, 0.0f, longitudinal * slopeB});
    AddCentralForce(candidate, {
            0.0f,
            0.0f,
            (-cuda::facts::Tuning(configuration).gearedDrive.forceZScale *
             currentForce.z) /
                    cuda::facts::Tuning(configuration).bodyAirResponse.
                            groundedSolidFeedback1,
    });
    vehicle.slipMemory.active = slipSeen != 0;
    vehicle.gearedDrive.localSpeed = linearSpeed;
    return ForceStatus::Success;
}

}  // namespace force_detail

template <
        bool ReuseWheelPassInvariants = false,
        CudaHandlingSpecialization Handling =
                CudaHandlingSpecialization::Generic,
        bool CollectHotPathMetrics = false>
__device__ inline ForceStatus ComputeForcesModel6(
        CudaCandidatePhysicsState &candidate,
        const CudaPackedStaticConfigurationHeader *configuration,
        float dt,
        collision::CudaHotPathCounters *hotPathCounters = nullptr) {
    CudaVehicleState &vehicle = candidate.vehicle;
    GmVec3 savedForce;
    GmVec3 savedImpulse;
    force_detail::SaveAndClearFeedback(
            vehicle, savedForce, savedImpulse);
    if (wheel_detail::SpeedBlocked(vehicle) ||
        vehicle.water.boxLocal.halfExtents.x < 0.0f) {
        if constexpr (CollectHotPathMetrics) {
            ++hotPathCounters->zeroDynamicsForcePassCount;
        }
        force_detail::SetZeroDynamics(candidate);
        return ForceStatus::Success;
    }
    bool legacyHandling = false;
    if constexpr (Handling == CudaHandlingSpecialization::Generic) {
        legacyHandling =
                cuda::facts::Tuning(configuration).handlingModel ==
                        static_cast<std::uint32_t>(
                                CSceneVehicleCarHandlingModel_Standard) ||
                cuda::facts::Tuning(configuration).handlingModel ==
                        static_cast<std::uint32_t>(
                                CSceneVehicleCarHandlingModel_Lateral);
        if (!legacyHandling &&
            cuda::facts::Tuning(configuration).handlingModel !=
                    static_cast<std::uint32_t>(
                            CSceneVehicleCarHandlingModel_GearedDrive)) {
            return ForceStatus::UnsupportedHandlingModel;
        }
    } else if constexpr (
            Handling == CudaHandlingSpecialization::Legacy) {
        legacyHandling = true;
    }
    force_detail::CreateFakeContacts(
            candidate, configuration);
    cuda::vehicle::IntegrateVehiclePrefix<
            ReuseWheelPassInvariants,
            Handling>(
            candidate, configuration, dt);
    const bool groundContact =
            force_detail::IsGroundContact(vehicle);
    if constexpr (CollectHotPathMetrics) {
        if (groundContact) {
            ++hotPathCounters->groundForcePassCount;
        } else {
            ++hotPathCounters->airForcePassCount;
        }
    }
    candidate.body.physicalParameters.
            vehicleContactFeedbackScale =
            groundContact
            ? cuda::facts::Tuning(configuration).bodyAirResponse.
                      groundedSolidFeedback1
            : cuda::facts::Tuning(configuration).bodyAirResponse.
                      airborneSolidFeedback1;
    candidate.body.physicalParameters.linearFluidFriction =
            groundContact
            ? 0.0f
            : cuda::facts::Tuning(configuration).bodyAirResponse.
                      airborneSolidFeedback0;
    candidate.body.parameters.mass =
            candidate.body.physicalParameters.mass;
    candidate.body.parameters.bodyInertiaLike =
            candidate.body.physicalParameters.impulseInertia;
    candidate.body.parameters.linearDampingScale =
            candidate.body.physicalParameters.linearFluidFriction;
    candidate.body.parameters.angularDampingScale =
            candidate.body.physicalParameters.physicalResponseCoefA;
    candidate.body.parameters.maxStepDistance =
            candidate.body.physicalParameters.physicalResponseCoefB;
    candidate.body.parameters.forceScale =
            candidate.body.physicalParameters.
                    vehicleContactFeedbackScale;
    candidate.body.parameters.localCenterOfMass =
            candidate.body.physicalParameters.localCenterOfMass;

    GmVec3 linearSpeed = force_detail::WorldToLocal(
            candidate.body, candidate.body.current.linearSpeed);
    const GmVec3 angularSpeed = force_detail::WorldToLocal(
            candidate.body, candidate.body.current.angularSpeed);
    const GmVec3 currentForce = force_detail::WorldToLocal(
            candidate.body, candidate.body.current.force);
    vehicle.engine.lowSpeedFeedbackForce = 0.0f;
    force_detail::ApplyFrictionForces<Handling>(
            candidate, configuration, linearSpeed);
    force_detail::ClampLinearSpeed(candidate, linearSpeed);
    VehicleMaterialBlendValues material;
    bool hasMaterial = false;
    force_detail::GroundMaterial(
            candidate, configuration, material, hasMaterial);
    float slopeA = 1.0f;
    float slopeB = 1.0f;
    force_detail::SlopeAdherence(
            configuration, currentForce, slopeA, slopeB);
    const float visualSteerYaw =
            force_detail::VisualSteerYaw(
                    vehicle, configuration, linearSpeed);
    vehicle.gearedDrive.localSpeed = linearSpeed;
    force_detail::Model6Result modelResult;
    if constexpr (Handling == CudaHandlingSpecialization::Legacy) {
        const ForceStatus modelStatus =
                force_detail::ComputeModel3Ground<
                    ReuseWheelPassInvariants>(
                        candidate, configuration, currentForce,
                        slopeA, slopeB, linearSpeed, angularSpeed,
                        visualSteerYaw, hasMaterial, material,
                        modelResult);
        if (modelStatus != ForceStatus::Success) {
            return modelStatus;
        }
    } else if constexpr (
            Handling ==
                    CudaHandlingSpecialization::GearedDriveDry ||
            Handling ==
                    CudaHandlingSpecialization::GearedDriveWater) {
        int waterActive = 0;
        if constexpr (
                Handling ==
                        CudaHandlingSpecialization::
                                GearedDriveWater) {
            waterActive = force_detail::ApplyWaterForces(
                    candidate, configuration, currentForce);
            if constexpr (CollectHotPathMetrics) {
                if (waterActive != 0) {
                    ++hotPathCounters->waterForcePassCount;
                }
            }
        }
        const ForceStatus modelStatus =
                force_detail::ComputeModel6Ground<
                        ReuseWheelPassInvariants>(
                        candidate, configuration, dt, currentForce,
                        slopeA, slopeB, linearSpeed, angularSpeed,
                        visualSteerYaw, hasMaterial, material,
                        waterActive, modelResult);
        if (modelStatus != ForceStatus::Success) {
            return modelStatus;
        }
    } else if (legacyHandling) {
        const ForceStatus modelStatus =
                force_detail::ComputeModel3Ground<
                    ReuseWheelPassInvariants>(
                        candidate, configuration, currentForce,
                        slopeA, slopeB, linearSpeed, angularSpeed,
                        visualSteerYaw, hasMaterial, material,
                        modelResult);
        if (modelStatus != ForceStatus::Success) {
            return modelStatus;
        }
    } else {
        const int waterActive = force_detail::ApplyWaterForces(
                candidate, configuration, currentForce);
        if constexpr (CollectHotPathMetrics) {
            if (waterActive != 0) {
                ++hotPathCounters->waterForcePassCount;
            }
        }
        const ForceStatus modelStatus =
                force_detail::ComputeModel6Ground<
                        ReuseWheelPassInvariants>(
                        candidate, configuration, dt, currentForce,
                        slopeA, slopeB, linearSpeed, angularSpeed,
                        visualSteerYaw, hasMaterial, material,
                        waterActive, modelResult);
        if (modelStatus != ForceStatus::Success) {
            return modelStatus;
        }
    }

    bool sideKill = false;
    bool anyContact = false;
    for (std::uint32_t index = 0u;
         index < facts::WheelCount(vehicle); ++index) {
        const CudaWheelState &wheel =
                vehicle.wheels.values[index];
        if (wheel.realTime.contactPresent) {
            anyContact = true;
            sideKill |= facts::WheelKillsLateralSpeed(
                    configuration, index);
        }
    }
    if (anyContact) {
        vehicle.engine.lowSpeedFeedbackForce =
                -vehicle.controls.lowSpeedGateB *
                        vehicle.engine.
                                lowSpeedFeedbackGateScale -
                cuda::facts::Tuning(configuration).
                        lowSpeedFrictionMagnitude *
                        vehicle.engine.
                                lowSpeedFeedbackFrictionScale;
    }
    if (sideKill &&
        cuda::facts::Tuning(configuration).gearedDrive.lateralForceScale ==
                cuda::facts::Tuning(configuration).gearedDrive.
                        lateralForceScale &&
        cuda::facts::Tuning(configuration).gearedDrive.lateralForceScale <
                0.0f) {
        linearSpeed.x = 0.0f;
        force_detail::SetLocalLinearSpeed(
                candidate, linearSpeed);
    }
    force_detail::UpdateAirControl<Handling>(
            candidate, configuration, angularSpeed,
            groundContact, sideKill);
    const std::uint32_t tick = candidate.world.tickTimeMs;
    force_detail::ApplySpecialContactResponse(
            candidate, configuration, currentForce,
            tick, groundContact);
    force_detail::UpdateImpactStates(
            vehicle, configuration);
    vehicle.lastComputeForcesTick = tick;
    force_detail::ProcessTurboContacts(
            vehicle, configuration, tick);
    force_detail::UpdateTurbo(vehicle, tick);
    force_detail::UpdateFeedback(
            candidate, configuration, dt, linearSpeed,
            savedForce, savedImpulse,
            modelResult.surfaceFeedback);
    force_detail::ClearContactScratch(vehicle);
    return ForceStatus::Success;
}

}  // namespace forevervalidator::simulation::cuda::vehicle

#endif
