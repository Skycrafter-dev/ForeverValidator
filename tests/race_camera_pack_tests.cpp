#include <forevervalidator/camera.h>

#include "engine/camera/race_camera_internal.h"
#include "format/camera/race_camera_gbx_reader.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using forevervalidator::camera::RaceCameraEnvironment;
using forevervalidator::camera::RaceCameraProfile;
using forevervalidator::camera::RaceCameraQuery;
using forevervalidator::camera::RaceCameraSession;
using forevervalidator::camera::RaceCameraVehicleState;
using forevervalidator::camera::detail::CurveKey;
using forevervalidator::camera::detail::EnvironmentConfig;
using forevervalidator::camera::detail::RaceCameraEnvironmentFactory;

constexpr std::uint32_t CommonChunk = 0x0306b00bu;
constexpr std::uint32_t TargetChunk = 0x03072001u;
constexpr std::uint32_t RaceClass = 0x24085000u;
constexpr std::uint32_t Race2Class = 0x24086000u;
constexpr std::uint32_t Race2FlagsChunk = 0x24086001u;
constexpr std::uint32_t Race3Class = 0x24087000u;
constexpr std::uint32_t Race3CollisionChunk = 0x24087001u;
constexpr std::uint32_t CurveClass = 0x0501a000u;
constexpr std::uint32_t CurveXChunk = 0x05002001u;
constexpr std::uint32_t CurveIdChunk = 0x05002003u;
constexpr std::uint32_t CurveYChunk = 0x0501a001u;
constexpr std::uint32_t Facade = 0xfacade01u;

void U32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8u));
    bytes.push_back(static_cast<std::uint8_t>(value >> 16u));
    bytes.push_back(static_cast<std::uint8_t>(value >> 24u));
}

void F32(std::vector<std::uint8_t> &bytes, float value) {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    U32(bytes, bits);
}

void SetU32(std::vector<std::uint8_t> &bytes, std::size_t offset,
            std::uint32_t value) {
    bytes[offset] = static_cast<std::uint8_t>(value);
    bytes[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
    bytes[offset + 2u] = static_cast<std::uint8_t>(value >> 16u);
    bytes[offset + 3u] = static_cast<std::uint8_t>(value >> 24u);
}

void SetF32(std::vector<std::uint8_t> &bytes, std::size_t offset,
            float value) {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    SetU32(bytes, offset, bits);
}

bool ReplaceFirstU32(std::vector<std::uint8_t> &bytes,
                     std::uint32_t from,
                     std::uint32_t to) {
    for (std::size_t offset = 0u; offset + 4u <= bytes.size(); offset++) {
        std::uint32_t value = 0u;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        if (value == from) {
            SetU32(bytes, offset, to);
            return true;
        }
    }
    return false;
}

std::vector<std::uint8_t> Header(std::uint32_t classId,
                                 std::uint32_t nodeCount) {
    std::vector<std::uint8_t> bytes{
            'G', 'B', 'X', 6u, 0u, 'B', 'U', 'U', 'R'};
    U32(bytes, classId);
    U32(bytes, 0u);
    U32(bytes, nodeCount);
    U32(bytes, 0u);
    return bytes;
}

void Common(std::vector<std::uint8_t> &bytes, float height,
            float minimumDistance, float maximumDistance,
            float fieldOfView, bool ignoreTargetRotation,
            float lookFactor) {
    U32(bytes, CommonChunk);
    const std::size_t payload = bytes.size();
    bytes.resize(bytes.size() + 156u, 0u);
    SetF32(bytes, payload + 24u, 30.0f);
    SetF32(bytes, payload + 28u, height);
    SetF32(bytes, payload + 32u, minimumDistance);
    SetF32(bytes, payload + 36u, maximumDistance);
    SetF32(bytes, payload + 40u, fieldOfView);
    SetF32(bytes, payload + 44u, fieldOfView);
    SetU32(bytes, payload + 100u, ignoreTargetRotation ? 1u : 0u);

    U32(bytes, TargetChunk);
    const std::size_t target = bytes.size();
    bytes.resize(bytes.size() + 60u, 0u);
    SetF32(bytes, target + 56u, lookFactor);
}

void Smooth(std::vector<std::uint8_t> &bytes, float delta,
            std::uint32_t rise, std::uint32_t fall) {
    F32(bytes, delta);
    U32(bytes, rise);
    U32(bytes, fall);
}

void Curve(std::vector<std::uint8_t> &bytes,
           const std::vector<CurveKey> &keys,
           bool compactKeys = false,
           bool facade = true) {
    U32(bytes, CurveClass);
    if (compactKeys) {
        U32(bytes, 0x10203040u);
        U32(bytes, 0x50607080u);
    } else {
        U32(bytes, CurveXChunk);
        U32(bytes, static_cast<std::uint32_t>(keys.size()));
    }
    for (const CurveKey &key : keys) {
        F32(bytes, key.position);
    }
    U32(bytes, CurveIdChunk);
    U32(bytes, 0xffffffffu);
    U32(bytes, CurveYChunk);
    U32(bytes, static_cast<std::uint32_t>(keys.size()));
    for (const CurveKey &key : keys) {
        F32(bytes, key.value);
    }
    if (facade) {
        U32(bytes, Facade);
    }
}

std::vector<std::uint8_t> Race2Fixture() {
    std::vector<std::uint8_t> bytes = Header(Race2Class, 4u);
    Common(bytes, 6.0f, 7.0f, 8.0f, 66.0f, false, 0.42f);
    U32(bytes, Race2Class);
    for (std::uint32_t index = 0u; index < 15u; index++) {
        Smooth(bytes, static_cast<float>(index + 1u) * 0.125f,
               100u + index, 200u + index);
    }
    F32(bytes, 0.35f);
    F32(bytes, 17.0f);
    U32(bytes, 250u);
    U32(bytes, 1250u);
    U32(bytes, 1500u);
    U32(bytes, 650u);
    U32(bytes, 450u);
    U32(bytes, 175u);
    F32(bytes, 3.5f);
    F32(bytes, 22.0f);
    Curve(bytes, {{0.0f, 0.0f}, {2.0f, -0.2f}, {5.0f, -0.4f}}, true);
    Curve(bytes, {{0.0f, 0.0f}, {100.0f, 3.0f}});
    Curve(bytes, {{0.0f, 0.0f}, {100.0f, 1.5f}}, false, false);
    U32(bytes, 0xabcdef01u);
    U32(bytes, 0xabcdef02u);
    F32(bytes, 9.0f);
    F32(bytes, 4.0f);
    F32(bytes, 2.0f);
    F32(bytes, 0.8f);
    U32(bytes, Race2FlagsChunk);
    U32(bytes, 1u);
    U32(bytes, Facade);
    return bytes;
}

std::vector<std::uint8_t> RaceFixture() {
    std::vector<std::uint8_t> bytes = Header(RaceClass, 1u);
    Common(bytes, 5.5f, 2.5f, 9.0f, 70.0f, false, 0.35f);
    U32(bytes, RaceClass);
    F32(bytes, 0.25f);
    F32(bytes, 15.0f);
    F32(bytes, 90.0f);
    U32(bytes, 1u);
    F32(bytes, 4.0f);
    F32(bytes, 11.0f);
    F32(bytes, 1.5f);
    F32(bytes, 3.25f);
    U32(bytes, 1u);
    F32(bytes, 7.0f);
    F32(bytes, 12.0f);
    F32(bytes, 0.0f);
    F32(bytes, -1.0f);
    F32(bytes, 0.0f);
    U32(bytes, Facade);
    return bytes;
}

std::vector<std::uint8_t> Race3Fixture() {
    std::vector<std::uint8_t> bytes = Header(Race3Class, 4u);
    Common(bytes, 2.0f, 3.0f, 4.0f, 68.0f, false, 0.6f);
    U32(bytes, Race3Class);
    F32(bytes, 6.0f);
    F32(bytes, 1.25f);
    F32(bytes, 3.75f);
    U32(bytes, 5u);
    U32(bytes, 180u);
    U32(bytes, 1700u);
    F32(bytes, 27.0f);
    F32(bytes, 8.0f);
    Smooth(bytes, 1.0f, 2100u, 2200u);
    Smooth(bytes, 0.9f, 1900u, 1800u);
    Smooth(bytes, 0.8f, 1700u, 1600u);
    Smooth(bytes, 0.7f, 1500u, 1400u);
    F32(bytes, 2.5f);
    Smooth(bytes, 0.6f, 1300u, 1200u);
    F32(bytes, 3.5f);
    F32(bytes, 0.75f);
    Smooth(bytes, 0.5f, 1100u, 1000u);
    Smooth(bytes, -0.3f, 900u, 800u);
    F32(bytes, 0.15f);
    Smooth(bytes, 2.5f, 700u, 600u);
    F32(bytes, 0.25f);
    F32(bytes, 1.2f);
    F32(bytes, 0.65f);
    F32(bytes, 8.0f);
    F32(bytes, 4.5f);
    Smooth(bytes, 1.1f, 500u, 400u);
    Smooth(bytes, -0.6f, 510u, 410u);
    Smooth(bytes, -0.7f, 520u, 420u);
    Smooth(bytes, 9.5f, 530u, 430u);
    Smooth(bytes, -0.5f, 540u, 440u);
    Smooth(bytes, -0.4f, 550u, 450u);
    Smooth(bytes, -0.35f, 560u, 460u);
    Smooth(bytes, 0.9f, 570u, 470u);
    Curve(bytes, {{-100.0f, 1.1f}, {0.0f, 1.0f}, {300.0f, 1.8f}});
    Curve(bytes, {{0.0f, 0.0f}, {10.0f, -0.3f}}, true);
    Curve(bytes, {{0.0f, 0.0f}, {2.0f, 0.2f}, {20.0f, 0.5f}});
    U32(bytes, Race3CollisionChunk);
    F32(bytes, 1.75f);
    U32(bytes, Facade);
    return bytes;
}

bool Near(float left, float right, float tolerance = 1.0e-5f) {
    return std::fabs(left - right) <= tolerance;
}

bool Check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

bool TestDecodeAndEvaluate() {
    EnvironmentConfig config;
    config.sourceName = "synthetic-camera-pack";
    std::string error;
    const std::vector<std::uint8_t> raceBytes = RaceFixture();
    const std::vector<std::uint8_t> race2Bytes = Race2Fixture();
    const std::vector<std::uint8_t> race3Bytes = Race3Fixture();
    bool okay = Check(
            forevervalidator::camera::detail::DecodeRaceCameraProfileGbx(
                    raceBytes.data(), raceBytes.size(), &config, &error),
            "synthetic Race camera GBX did not decode");
    okay &= Check(
            forevervalidator::camera::detail::DecodeRaceCameraProfileGbx(
                    race2Bytes.data(), race2Bytes.size(), &config, &error),
            "synthetic Race2 camera GBX did not decode");
    okay &= Check(
            forevervalidator::camera::detail::DecodeRaceCameraProfileGbx(
                    race3Bytes.data(), race3Bytes.size(), &config, &error),
            "synthetic Race3 camera GBX did not decode");
    if (!okay || !config.race || !config.race2 || !config.race3) {
        return false;
    }

    okay &= Check(Near(config.race->common.height, 5.5f) &&
                            Near(config.race->common.minimumDistance, 2.5f) &&
                            Near(config.race->common.maximumDistance, 9.0f) &&
                            Near(config.race->common.baseFieldOfView, 70.0f) &&
                            Near(config.race->common.lookFactor, 0.35f) &&
                            Near(config.race->coneAperture, 0.25f) &&
                            Near(config.race->coneMinimumSpeed, 15.0f) &&
                            Near(config.race->coneMaximumSpeed, 90.0f) &&
                            config.race->useSpeedDirection &&
                            Near(config.race->cameraHeight, 4.0f) &&
                            Near(config.race->cameraDistance, 11.0f) &&
                            config.race->segmentCast &&
                            Near(config.race->segmentCastMinimumDistance, 7.0f) &&
                            Near(config.race->segmentCastLength, 12.0f),
                    "Race values were not loaded from GBX");

    okay &= Check(Near(config.race2->common.height, 6.0f) &&
                            Near(config.race2->common.minimumDistance, 7.0f) &&
                            Near(config.race2->common.maximumDistance, 8.0f) &&
                            Near(config.race2->common.baseFieldOfView, 66.0f) &&
                            Near(config.race2->common.lookFactor, 0.42f),
                    "Race2 common values were not loaded from GBX");
    okay &= Check(config.race2->rollFromInput &&
                            config.race2->flyingLookCurve.size() == 3u &&
                            config.race2->yawFromSpeedCurve.size() == 2u &&
                            config.race2->rollFromSpeedCurve.size() == 2u &&
                            Near(config.race2->distanceSpringStiffness, 9.0f) &&
                            Near(config.race2->lookSpringDamping, 0.8f),
                    "Race2 class values were not loaded from GBX");
    okay &= Check(Near(config.race3->up, 1.25f) &&
                            Near(config.race3->far, 3.75f) &&
                            Near(config.race3->collisionRadius, 1.75f) &&
                            config.race3->speedModulationCurve.size() == 3u &&
                            config.race3->flyingLookCurve.size() == 2u &&
                            config.race3->flyingDownCurve.size() == 3u,
                    "Race3 values were not loaded from GBX");

    RaceCameraEnvironment environment =
            RaceCameraEnvironmentFactory::Create(std::move(config));
    okay &= Check(environment.SourceName() == "synthetic-camera-pack" &&
                            environment.HasProfile(RaceCameraProfile::Race) &&
                            environment.HasProfile(RaceCameraProfile::Race2) &&
                            environment.HasProfile(RaceCameraProfile::Race3) &&
                            environment.ResourceCount(RaceCameraProfile::Race) ==
                                    1u &&
                            environment.ResourceName(
                                    RaceCameraProfile::Race, 0u) ==
                                    "synthetic-camera-pack",
                    "decoded environment metadata is incorrect");

    RaceCameraVehicleState vehicle;
    vehicle.targetId = 7u;
    RaceCameraQuery query;
    query.vehicle = vehicle;
    RaceCameraSession race(environment, RaceCameraProfile::Race);
    RaceCameraSession race2(environment, RaceCameraProfile::Race2);
    RaceCameraSession race3(environment, RaceCameraProfile::Race3);
    race.Reset(vehicle);
    race2.Reset(vehicle);
    race3.Reset(vehicle);
    const auto output = race.Evaluate(query);
    const auto output2 = race2.Evaluate(query);
    const auto output3 = race3.Evaluate(query);
    okay &= Check(Near(output.transform.position.y, 5.5f) &&
                            Near(output.transform.position.z, -9.0f) &&
                            Near(output.lens.fieldOfViewDegrees, 70.0f),
                    "Race session ignored decoded configuration");
    okay &= Check(Near(output2.transform.position.y, 6.0f) &&
                            Near(output2.transform.position.z, -8.0f) &&
                            Near(output2.lens.fieldOfViewDegrees, 66.0f),
                    "Race2 session ignored decoded configuration");
    okay &= Check(Near(output3.transform.position.y, 1.25f) &&
                            Near(output3.transform.position.z, -3.75f) &&
                            Near(output3.lens.fieldOfViewDegrees, 68.0f),
                    "Race3 session ignored decoded configuration");

    RaceCameraSession neutral(environment, RaceCameraProfile::Race2);
    RaceCameraSession steering(environment, RaceCameraProfile::Race2);
    RaceCameraVehicleState neutralVehicle = vehicle;
    RaceCameraVehicleState steeringVehicle = vehicle;
    neutralVehicle.signedSpeed = 120.0f;
    steeringVehicle.signedSpeed = 120.0f;
    steeringVehicle.steering = 1.0f;
    neutral.Reset(neutralVehicle);
    steering.Reset(steeringVehicle);
    forevervalidator::camera::RaceCameraOutput neutralOutput;
    forevervalidator::camera::RaceCameraOutput steeringOutput;
    for (const std::uint32_t timeMs : {0u, 100u, 300u, 600u, 1000u}) {
        neutralVehicle.timeMs = timeMs;
        steeringVehicle.timeMs = timeMs;
        RaceCameraQuery neutralQuery;
        RaceCameraQuery steeringQuery;
        neutralQuery.vehicle = neutralVehicle;
        steeringQuery.vehicle = steeringVehicle;
        neutralOutput = neutral.Evaluate(neutralQuery);
        steeringOutput = steering.Evaluate(steeringQuery);
    }
    const auto &neutralRotation = neutralOutput.transform.rotation;
    const auto &steeringRotation = steeringOutput.transform.rotation;
    const float orientationDifference =
            std::fabs(neutralRotation.w - steeringRotation.w) +
            std::fabs(neutralRotation.x - steeringRotation.x) +
            std::fabs(neutralRotation.y - steeringRotation.y) +
            std::fabs(neutralRotation.z - steeringRotation.z);
    okay &= Check(
            orientationDifference > 1.0e-4f,
            "Race2 ignored loaded steering roll/yaw configuration");
    return okay;
}

bool TestRejectsTruncatedArchive() {
    std::vector<std::uint8_t> bytes = Race2Fixture();
    bytes.resize(bytes.size() - 20u);
    EnvironmentConfig config;
    std::string error;
    return Check(
            !forevervalidator::camera::detail::DecodeRaceCameraProfileGbx(
                    bytes.data(), bytes.size(), &config, &error) &&
                    !error.empty(),
            "truncated camera GBX was accepted");
}

bool TestRejectsMissingSerializedChunks() {
    bool okay = true;

    std::vector<std::uint8_t> missingTarget = Race2Fixture();
    okay &= Check(ReplaceFirstU32(missingTarget, TargetChunk, 0u),
                    "target chunk fixture marker was not found");
    EnvironmentConfig targetConfig;
    std::string targetError;
    okay &= Check(
            !forevervalidator::camera::detail::DecodeRaceCameraProfileGbx(
                    missingTarget.data(), missingTarget.size(),
                    &targetConfig, &targetError) &&
                    !targetError.empty(),
            "camera GBX without a target chunk inherited defaults");

    std::vector<std::uint8_t> missingClass = RaceFixture();
    // Skip the root class ID in the GBX header and corrupt the class chunk.
    bool classChunkReplaced = false;
    for (std::size_t offset = 13u;
         offset + 4u <= missingClass.size();
         offset++) {
        std::uint32_t value = 0u;
        std::memcpy(&value, missingClass.data() + offset, sizeof(value));
        if (value == RaceClass) {
            SetU32(missingClass, offset, 0u);
            classChunkReplaced = true;
            break;
        }
    }
    okay &= Check(classChunkReplaced,
                    "Race class chunk fixture marker was not found");
    EnvironmentConfig classConfig;
    std::string classError;
    okay &= Check(
            !forevervalidator::camera::detail::DecodeRaceCameraProfileGbx(
                    missingClass.data(), missingClass.size(),
                    &classConfig, &classError) &&
                    !classError.empty(),
            "Race GBX without its class chunk inherited defaults");
    return okay;
}

bool TestMultipleRaceResources() {
    EnvironmentConfig config;
    config.sourceName = "multi-race-pack";
    forevervalidator::camera::detail::RaceConfig first;
    auto second = first;
    first.common.height = 4.0f;
    first.common.maximumDistance = 6.0f;
    first.common.baseFieldOfView = 71.0f;
    first.common.initialFieldOfView = 71.0f;
    second.common.height = 8.0f;
    second.common.maximumDistance = 3.0f;
    second.common.baseFieldOfView = 82.0f;
    second.common.initialFieldOfView = 82.0f;
    config.raceResources = {
            {"CameraFirst.gbx", 0u, first},
            {"CameraSecond.gbx", 1u, second},
    };
    RaceCameraEnvironment environment =
            RaceCameraEnvironmentFactory::Create(std::move(config));
    bool okay = Check(
            environment.ResourceCount(RaceCameraProfile::Race) == 2u &&
                    environment.ResourceName(RaceCameraProfile::Race, 0u) ==
                            "CameraFirst.gbx" &&
                    environment.ResourceName(RaceCameraProfile::Race, 1u) ==
                            "CameraSecond.gbx",
            "multiple Race resources were not preserved");

    RaceCameraVehicleState vehicle;
    vehicle.targetId = 9u;
    RaceCameraQuery query;
    query.vehicle = vehicle;
    RaceCameraSession defaultSession(environment, RaceCameraProfile::Race);
    RaceCameraSession firstSession(
            environment, RaceCameraProfile::Race, 0u);
    RaceCameraSession secondSession(
            environment, RaceCameraProfile::Race, 1u);
    defaultSession.Reset(vehicle);
    firstSession.Reset(vehicle);
    secondSession.Reset(vehicle);
    const auto defaultOutput = defaultSession.Evaluate(query);
    const auto firstOutput = firstSession.Evaluate(query);
    const auto secondOutput = secondSession.Evaluate(query);
    okay &= Check(
            Near(defaultOutput.transform.position.y, 8.0f) &&
                    Near(defaultOutput.transform.position.z, -3.0f) &&
                    Near(defaultOutput.lens.fieldOfViewDegrees, 82.0f) &&
                    Near(firstOutput.transform.position.y, 4.0f) &&
                    Near(firstOutput.transform.position.z, -6.0f) &&
                    Near(firstOutput.lens.fieldOfViewDegrees, 71.0f) &&
                    Near(secondOutput.transform.position.y, 8.0f),
            "indexed Race resource selection is incorrect");
    return okay;
}

} // namespace

int main() {
    bool okay = true;
    okay &= TestDecodeAndEvaluate();
    okay &= TestRejectsTruncatedArchive();
    okay &= TestRejectsMissingSerializedChunks();
    okay &= TestMultipleRaceResources();
    return okay ? 0 : 1;
}
