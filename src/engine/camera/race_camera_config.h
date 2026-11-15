#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace forevervalidator::camera::detail {

struct CurveKey {
    float position = 0.0f;
    float value = 0.0f;
};

struct SmoothRealConfig {
    float delta = 0.0f;
    std::uint32_t riseTimeMs = 1000u;
    std::uint32_t fallTimeMs = 1000u;
};

struct CameraCommonConfig {
    std::array<float, 3> followedOffset{{0.0f, 0.0f, 0.0f}};
    float maximumFollowSpeed = 30.0f;
    float height = 3.0f;
    float minimumDistance = 4.0f;
    float maximumDistance = 5.0f;
    float baseFieldOfView = 75.0f;
    float initialFieldOfView = 75.0f;
    float lensParameter1 = 30.0f;
    float lensParameter2 = 3.0f;
    float lookFactor = 1.0f;
    bool ignoreTargetRotation = false;
};

struct RaceConfig {
    CameraCommonConfig common{};
    float coneAperture = 1.0f;
    float coneMinimumSpeed = 20.0f;
    float coneMaximumSpeed = 60.0f;
    bool useSpeedDirection = false;
    float cameraHeight = 3.0f;
    float cameraDistance = 9.0f;
    float targetDistance = 0.0f;
    float cameraAlignment = 4.0f;
    bool segmentCast = true;
    float segmentCastMinimumDistance = 8.5f;
    float segmentCastLength = 10.0f;
    std::array<float, 3> segmentCastDirection{{0.0f, 0.0f, 0.0f}};
};

struct Race2Config {
    CameraCommonConfig common{};
    std::array<SmoothRealConfig, 15u> smoothers{};
    float flyingLookRate = 0.2f;
    float loadedAuxiliaryRate = 20.0f;
    std::uint32_t airborneValidationMs = 200u;
    std::uint32_t flyingCameraMoveDelayMs = 1000u;
    std::uint32_t flyingModeDelayMs = 1000u;
    std::uint32_t steeringYawTriggerMs = 300u;
    std::uint32_t steeringRollTriggerMs = 200u;
    std::uint32_t steeringInactiveResetMs = 150u;
    float orientationMinimumSpeed = 20.0f;
    float heightRate = 4.0f;
    float fieldOfViewRate = 40.0f;
    std::vector<CurveKey> flyingLookCurve;
    std::vector<CurveKey> yawFromSpeedCurve;
    std::vector<CurveKey> rollFromSpeedCurve;
    float distanceSpringStiffness = 10.0f;
    float distanceSpringDamping = 5.0f;
    float lookSpringStiffness = 1.5f;
    float lookSpringDamping = 0.7f;
    bool rollFromInput = false;
};

struct Race3Config {
    CameraCommonConfig common{};
    float directionNormalRate = 5.0f;
    float up = 2.2f;
    float far = 4.5f;
    std::uint32_t airborneValidationMs = 1u;
    std::uint32_t steeringInactiveResetMs = 150u;
    std::uint32_t burningSteerDelayMs = 2000u;
    float lowSpeedThreshold = 30.0f;
    float veryLowSpeedThreshold = 10.0f;
    SmoothRealConfig reverse{};
    SmoothRealConfig lowSpeedFlightTarget{};
    SmoothRealConfig flyingDirectionBlend{};
    SmoothRealConfig flyingUpBlend{};
    float directionFlyingRate = 3.0f;
    SmoothRealConfig flyingSpeedEffect{};
    float upNormalRate = 4.0f;
    float upFlyingRate = 1.0f;
    SmoothRealConfig flyingUpSpeedEffect{};
    SmoothRealConfig flyingLook{};
    float flyingLookRate = 0.2f;
    SmoothRealConfig flyingRadius{};
    float constantFlyingLookDownFactor = 0.3f;
    float flyingDownSpringStiffness = 1.5f;
    float flyingDownSpringDamping = 0.7f;
    float radiusSpringStiffness = 10.0f;
    float radiusSpringDamping = 5.0f;
    SmoothRealConfig gasFar{};
    SmoothRealConfig brakeFar{};
    SmoothRealConfig steerFar{};
    SmoothRealConfig turboFieldOfView{};
    SmoothRealConfig turboFar{};
    SmoothRealConfig gearFar{};
    SmoothRealConfig burningLook{};
    SmoothRealConfig burningRadius{};
    std::vector<CurveKey> speedModulationCurve;
    std::vector<CurveKey> flyingLookCurve;
    std::vector<CurveKey> flyingDownCurve;
    float collisionRadius = 2.0f;
};

struct RaceResourceConfig {
    std::string sourceName;
    std::uint32_t collectorIndex = 0xffffffffu;
    RaceConfig config;
};

struct Race2ResourceConfig {
    std::string sourceName;
    std::uint32_t collectorIndex = 0xffffffffu;
    Race2Config config;
};

struct Race3ResourceConfig {
    std::string sourceName;
    std::uint32_t collectorIndex = 0xffffffffu;
    Race3Config config;
};

struct InternalResourceConfig {
    std::string sourceName;
    std::uint32_t collectorIndex = 0xffffffffu;
    CameraCommonConfig config;
};

struct EnvironmentConfig {
    std::string sourceName;
    std::uint32_t defaultCollectorIndex = 0xffffffffu;
    std::optional<RaceConfig> race;
    std::optional<Race2Config> race2;
    std::optional<Race3Config> race3;
    std::optional<CameraCommonConfig> internal;
    std::vector<RaceResourceConfig> raceResources;
    std::vector<Race2ResourceConfig> race2Resources;
    std::vector<Race3ResourceConfig> race3Resources;
    std::vector<InternalResourceConfig> internalResources;
};

} // namespace forevervalidator::camera::detail
