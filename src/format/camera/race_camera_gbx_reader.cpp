#include "format/camera/race_camera_gbx_reader.h"

#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "format/archive/archive_binary.h"
#include "format/archive/archive_class_ids.h"
#include "format/archive/tmnf_gbx_body_reader.h"

namespace forevervalidator::camera::detail {
namespace {

using TmnfFormat::ArchiveBinary::ReadU32LE;

constexpr std::uint32_t CommonChunk = 0x0306b00bu;
constexpr std::uint32_t TargetChunk = 0x03072001u;
constexpr std::uint32_t RaceChunk = 0x24085000u;
constexpr std::uint32_t Race2Chunk = 0x24086000u;
constexpr std::uint32_t Race2FlagsChunk = 0x24086001u;
constexpr std::uint32_t Race3Chunk = 0x24087000u;
constexpr std::uint32_t Race3CollisionChunk = 0x24087001u;
constexpr std::uint32_t CurveClass = 0x0501a000u;
constexpr std::uint32_t CurveXChunk = 0x05002001u;
constexpr std::uint32_t CurveIdChunk = 0x05002003u;
constexpr std::uint32_t CurveYChunk = 0x0501a001u;
constexpr std::uint32_t Facade = 0xfacade01u;
constexpr std::size_t CommonPayloadSize = 156u;
constexpr std::size_t TargetPayloadSize = 60u;
constexpr std::size_t Race2FixedPayloadSize = 220u;
constexpr std::size_t Race3FixedPayloadSize = 260u;
constexpr std::size_t MaximumCurveKeyCount = 128u;

struct View {
    const std::uint8_t *bytes = nullptr;
    std::size_t size = 0u;

    bool Contains(std::size_t offset, std::size_t count) const {
        return bytes != nullptr && offset <= size && count <= size - offset;
    }

    std::uint32_t U32(std::size_t offset) const {
        return ReadU32LE(bytes + offset);
    }

    float F32(std::size_t offset) const {
        const std::uint32_t bits = U32(offset);
        float result = 0.0f;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
};

void SetError(std::string *error, const char *message) {
    if (error != nullptr) {
        *error = message;
    }
}

std::optional<std::size_t> FindWord(
        const View &view,
        std::size_t begin,
        std::size_t end,
        std::uint32_t value) {
    if (end > view.size) {
        end = view.size;
    }
    for (std::size_t offset = begin; offset + 4u <= end; offset++) {
        if (view.U32(offset) == value) {
            return offset;
        }
    }
    return std::nullopt;
}

bool ParseCommonConfig(
        const View &view,
        std::size_t bodyOffset,
        CameraCommonConfig *config,
        std::string *error,
        bool requireTarget = true) {
    const auto chunk = FindWord(
            view, bodyOffset, view.size, CommonChunk);
    if (!chunk || !view.Contains(*chunk + 4u, CommonPayloadSize)) {
        SetError(error, "camera GBX is missing the consolidated common chunk");
        return false;
    }
    const std::size_t payload = *chunk + 4u;
    config->followedOffset = {
            view.F32(payload), view.F32(payload + 4u),
            view.F32(payload + 8u)};
    config->maximumFollowSpeed = view.F32(payload + 24u);
    config->height = view.F32(payload + 28u);
    config->minimumDistance = view.F32(payload + 32u);
    config->maximumDistance = view.F32(payload + 36u);
    config->initialFieldOfView = view.F32(payload + 40u);
    config->baseFieldOfView = view.F32(payload + 44u);
    config->ignoreTargetRotation = view.U32(payload + 100u) != 0u;

    if (!requireTarget) return true;

    const auto target = FindWord(
            view, payload + CommonPayloadSize, view.size, TargetChunk);
    if (!target || !view.Contains(*target + 4u, TargetPayloadSize)) {
        SetError(error, "camera GBX is missing the target chunk");
        return false;
    }
    config->lookFactor = view.F32(*target + 4u + 56u);
    return true;
}

bool ParseSmooth(
        const View &view,
        std::size_t offset,
        SmoothRealConfig *out) {
    if (!view.Contains(offset, 12u) || out == nullptr) {
        return false;
    }
    out->delta = view.F32(offset);
    out->riseTimeMs = view.U32(offset + 4u);
    out->fallTimeMs = view.U32(offset + 8u);
    return out->riseTimeMs != 0u && out->fallTimeMs != 0u;
}

bool ParseCurve(
        const View &view,
        std::size_t rootOffset,
        std::size_t limit,
        std::vector<CurveKey> *out,
        std::string *error) {
    if (out == nullptr || !view.Contains(rootOffset, 4u) ||
        view.U32(rootOffset) != CurveClass || limit > view.size ||
        rootOffset >= limit) {
        SetError(error, "camera GBX contains an invalid curve root");
        return false;
    }
    const auto idChunk = FindWord(
            view, rootOffset + 4u, limit, CurveIdChunk);
    if (!idChunk || !view.Contains(*idChunk + 8u, 4u) ||
        view.U32(*idChunk + 8u) != CurveYChunk) {
        SetError(error, "camera GBX curve is missing its value chunk");
        return false;
    }
    const std::size_t yCountOffset = *idChunk + 12u;
    if (!view.Contains(yCountOffset, 4u)) {
        SetError(error, "camera GBX curve has a truncated value count");
        return false;
    }
    const std::uint32_t count = view.U32(yCountOffset);
    if (count == 0u || count > MaximumCurveKeyCount ||
        !view.Contains(yCountOffset + 4u,
                       static_cast<std::size_t>(count) * 4u)) {
        SetError(error, "camera GBX curve has an invalid value array");
        return false;
    }

    std::size_t xOffset = rootOffset + 4u;
    if (!view.Contains(xOffset, 8u)) {
        SetError(error, "camera GBX curve has truncated key metadata");
        return false;
    }
    if (view.U32(xOffset) == CurveXChunk) {
        if (!view.Contains(xOffset + 4u, 4u) ||
            view.U32(xOffset + 4u) != count) {
            SetError(error, "camera GBX curve key/value counts differ");
            return false;
        }
        xOffset += 8u;
    } else {
        // Some shipped camera curves use the old compact CFuncKeys metadata:
        // two opaque words followed by the key array. The Y count remains the
        // authoritative key count in that encoding.
        xOffset += 8u;
    }
    if (!view.Contains(xOffset, static_cast<std::size_t>(count) * 4u) ||
        xOffset + static_cast<std::size_t>(count) * 4u != *idChunk) {
        SetError(error, "camera GBX curve key array is malformed");
        return false;
    }

    out->clear();
    out->reserve(count);
    for (std::uint32_t index = 0u; index < count; index++) {
        out->push_back({view.F32(xOffset + index * 4u),
                        view.F32(yCountOffset + 4u + index * 4u)});
    }
    return true;
}

bool FindCurveRoots(
        const View &view,
        std::size_t begin,
        std::size_t end,
        std::array<std::size_t, 3u> *roots,
        std::string *error) {
    std::size_t cursor = begin;
    for (std::size_t index = 0u; index < roots->size(); index++) {
        const auto root = FindWord(view, cursor, end, CurveClass);
        if (!root) {
            SetError(error, "camera GBX is missing a serialized curve");
            return false;
        }
        (*roots)[index] = *root;
        cursor = *root + 4u;
    }
    return true;
}

bool DecodeRace(
        const View &view,
        std::size_t bodyOffset,
        EnvironmentConfig *environment,
        std::string *error) {
    RaceConfig config;
    if (!ParseCommonConfig(view, bodyOffset, &config.common, error)) {
        return false;
    }
    config.common.lensParameter1 = 30.0f;
    config.common.lensParameter2 = 3.0f;
    const auto chunk = FindWord(view, bodyOffset, view.size, RaceChunk);
    if (!chunk || !view.Contains(*chunk + 4u, 56u)) {
        SetError(error, "Race camera GBX is missing its class chunk");
        return false;
    }
    const std::size_t payload = *chunk + 4u;
    config.coneAperture = view.F32(payload);
    config.coneMinimumSpeed = view.F32(payload + 4u);
    config.coneMaximumSpeed = view.F32(payload + 8u);
    config.useSpeedDirection = view.U32(payload + 12u) != 0u;
    config.cameraHeight = view.F32(payload + 16u);
    config.cameraDistance = view.F32(payload + 20u);
    config.targetDistance = view.F32(payload + 24u);
    config.cameraAlignment = view.F32(payload + 28u);
    config.segmentCast = view.U32(payload + 32u) != 0u;
    config.segmentCastMinimumDistance = view.F32(payload + 36u);
    config.segmentCastLength = view.F32(payload + 40u);
    config.segmentCastDirection = {
            view.F32(payload + 44u), view.F32(payload + 48u),
            view.F32(payload + 52u)};
    environment->race = std::move(config);
    return true;
}

bool DecodeRace2(
        const View &view,
        std::size_t bodyOffset,
        EnvironmentConfig *environment,
        std::string *error) {
    Race2Config config;
    if (!ParseCommonConfig(view, bodyOffset, &config.common, error)) {
        return false;
    }
    config.common.lensParameter1 = 30.0f;
    config.common.lensParameter2 = 3.0f;
    const auto chunk = FindWord(view, bodyOffset, view.size, Race2Chunk);
    const auto flags = FindWord(
            view, bodyOffset, view.size, Race2FlagsChunk);
    if (!chunk || !flags || *flags <= *chunk ||
        !view.Contains(*chunk + 4u, Race2FixedPayloadSize) ||
        !view.Contains(*flags + 4u, 4u)) {
        SetError(error, "Race2 camera GBX is missing its class chunks");
        return false;
    }
    const std::size_t payload = *chunk + 4u;
    for (std::size_t index = 0u; index < config.smoothers.size(); index++) {
        if (!ParseSmooth(view, payload + index * 12u,
                         &config.smoothers[index])) {
            SetError(error, "Race2 camera GBX contains an invalid smoother");
            return false;
        }
    }
    std::size_t offset = payload + config.smoothers.size() * 12u;
    config.flyingLookRate = view.F32(offset);
    config.loadedAuxiliaryRate = view.F32(offset + 4u);
    config.orientationMinimumSpeed = config.loadedAuxiliaryRate;
    config.airborneValidationMs = view.U32(offset + 8u);
    config.flyingCameraMoveDelayMs = view.U32(offset + 12u);
    config.flyingModeDelayMs = view.U32(offset + 16u);
    // Native archive 0045db50 reads +0x2e4 before +0x2d0.
    config.steeringRollTriggerMs = view.U32(offset + 20u);
    config.steeringYawTriggerMs = view.U32(offset + 24u);
    config.steeringInactiveResetMs = view.U32(offset + 28u);
    config.heightRate = view.F32(offset + 32u);
    config.fieldOfViewRate = view.F32(offset + 36u);

    std::array<std::size_t, 3u> roots{};
    if (!FindCurveRoots(view, payload + Race2FixedPayloadSize,
                        *flags, &roots, error) ||
        !ParseCurve(view, roots[0], roots[1],
                    &config.flyingLookCurve, error) ||
        !ParseCurve(view, roots[1], roots[2],
                    &config.rollFromSpeedCurve, error) ||
        !ParseCurve(view, roots[2], *flags,
                    &config.yawFromSpeedCurve, error)) {
        return false;
    }
    if (*flags < 16u || !view.Contains(*flags - 16u, 16u)) {
        SetError(error, "Race2 camera GBX has a truncated spring tail");
        return false;
    }
    config.distanceSpringStiffness = view.F32(*flags - 16u);
    config.distanceSpringDamping = view.F32(*flags - 12u);
    config.lookSpringStiffness = view.F32(*flags - 8u);
    config.lookSpringDamping = view.F32(*flags - 4u);
    config.rollFromInput = view.U32(*flags + 4u) != 0u;
    environment->race2 = std::move(config);
    return true;
}

bool DecodeRace3(
        const View &view,
        std::size_t bodyOffset,
        EnvironmentConfig *environment,
        std::string *error) {
    Race3Config config;
    if (!ParseCommonConfig(view, bodyOffset, &config.common, error)) {
        return false;
    }
    config.common.lensParameter1 = 0.0f;
    config.common.lensParameter2 = 1.0f;
    const auto chunk = FindWord(view, bodyOffset, view.size, Race3Chunk);
    const auto collision = FindWord(
            view, bodyOffset, view.size, Race3CollisionChunk);
    if (!chunk || !collision || *collision <= *chunk ||
        !view.Contains(*chunk + 4u, Race3FixedPayloadSize) ||
        !view.Contains(*collision + 4u, 4u)) {
        SetError(error, "Race3 camera GBX is missing its class chunks");
        return false;
    }
    const std::size_t p = *chunk + 4u;
    config.directionNormalRate = view.F32(p);
    config.up = view.F32(p + 4u);
    config.far = view.F32(p + 8u);
    config.airborneValidationMs = view.U32(p + 12u);
    config.steeringInactiveResetMs = view.U32(p + 16u);
    config.burningSteerDelayMs = view.U32(p + 20u);
    config.lowSpeedThreshold = view.F32(p + 24u);
    config.veryLowSpeedThreshold = view.F32(p + 28u);
    if (!ParseSmooth(view, p + 32u, &config.reverse) ||
        !ParseSmooth(view, p + 44u, &config.lowSpeedFlightTarget) ||
        !ParseSmooth(view, p + 56u, &config.flyingDirectionBlend) ||
        !ParseSmooth(view, p + 68u, &config.flyingUpBlend) ||
        !ParseSmooth(view, p + 84u, &config.flyingSpeedEffect) ||
        !ParseSmooth(view, p + 104u, &config.flyingUpSpeedEffect) ||
        !ParseSmooth(view, p + 116u, &config.flyingLook) ||
        !ParseSmooth(view, p + 132u, &config.flyingRadius) ||
        !ParseSmooth(view, p + 164u, &config.gasFar) ||
        !ParseSmooth(view, p + 176u, &config.brakeFar) ||
        !ParseSmooth(view, p + 188u, &config.steerFar) ||
        !ParseSmooth(view, p + 200u, &config.turboFieldOfView) ||
        !ParseSmooth(view, p + 212u, &config.turboFar) ||
        !ParseSmooth(view, p + 224u, &config.gearFar) ||
        !ParseSmooth(view, p + 236u, &config.burningLook) ||
        !ParseSmooth(view, p + 248u, &config.burningRadius)) {
        SetError(error, "Race3 camera GBX contains an invalid smoother");
        return false;
    }
    config.directionFlyingRate = view.F32(p + 80u);
    config.upNormalRate = view.F32(p + 96u);
    config.upFlyingRate = view.F32(p + 100u);
    config.flyingLookRate = view.F32(p + 128u);
    config.constantFlyingLookDownFactor = view.F32(p + 144u);
    config.flyingDownSpringStiffness = view.F32(p + 148u);
    config.flyingDownSpringDamping = view.F32(p + 152u);
    config.radiusSpringStiffness = view.F32(p + 156u);
    config.radiusSpringDamping = view.F32(p + 160u);

    std::array<std::size_t, 3u> roots{};
    if (!FindCurveRoots(view, p + Race3FixedPayloadSize,
                        *collision, &roots, error) ||
        !ParseCurve(view, roots[0], roots[1],
                    &config.speedModulationCurve, error) ||
        !ParseCurve(view, roots[1], roots[2],
                    &config.flyingLookCurve, error) ||
        !ParseCurve(view, roots[2], *collision,
                    &config.flyingDownCurve, error)) {
        return false;
    }
    config.collisionRadius = view.F32(*collision + 4u);
    environment->race3 = std::move(config);
    return true;
}

} // namespace

bool DecodeRaceCameraProfileGbx(
        const std::uint8_t *bytes,
        std::size_t byteCount,
        EnvironmentConfig *environment,
        std::string *error) {
    if (environment == nullptr || bytes == nullptr ||
        byteCount > std::numeric_limits<std::uint32_t>::max()) {
        SetError(error, "invalid camera GBX input");
        return false;
    }
    std::uint32_t classId = 0u;
    std::uint32_t bodyOffset = 0u;
    if (!GbxBodyOffsetReader::TryParse(
                bytes, static_cast<std::uint32_t>(byteCount),
                &classId, &bodyOffset)) {
        SetError(error, "camera GBX header or reference table is invalid");
        return false;
    }
    const View view{bytes, byteCount};
    switch (classId) {
    case TMNF_CLASS_CGameControlCamera: {
        CameraCommonConfig config;
        if (!ParseCommonConfig(view, bodyOffset, &config, error, false)) {
            return false;
        }
        environment->internal = config;
        return true;
    }
    case TMNF_CLASS_CGameControlCameraTrackManiaRace:
        return DecodeRace(view, bodyOffset, environment, error);
    case TMNF_CLASS_CGameControlCameraTrackManiaRace2:
        return DecodeRace2(view, bodyOffset, environment, error);
    case TMNF_CLASS_CGameControlCameraTrackManiaRace3:
        return DecodeRace3(view, bodyOffset, environment, error);
    default:
        SetError(error, "GBX is not a supported TrackMania race camera");
        return false;
    }
}

} // namespace forevervalidator::camera::detail
