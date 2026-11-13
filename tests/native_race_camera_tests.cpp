#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include <forevervalidator/native.h>
#include "data/camera/native_camera_config.h"
#include "engine/camera/race_camera_internal.h"

namespace c = forevervalidator::camera;
namespace d = forevervalidator::camera::detail;

std::uint32_t ReadU32(std::istream &in) {
    std::array<unsigned char, 4> bytes{};
    if (!in.read(reinterpret_cast<char *>(bytes.data()), bytes.size()))
        throw std::runtime_error("truncated native camera fixture");
    return bytes[0] | (std::uint32_t(bytes[1]) << 8) | (std::uint32_t(bytes[2]) << 16) |
           (std::uint32_t(bytes[3]) << 24);
}

template <std::size_t N> std::array<float, N> ReadFloats(std::istream &in) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    std::array<float, N> values{};
    for (auto &value : values) {
        const auto bits = ReadU32(in);
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value))
            throw std::runtime_error("nonfinite fixture input");
    }
    return values;
}

GmIso4 Pose(const float *p) {
    GmIso4 pose;
    pose.rotation.basisX = {p[0], p[1], p[2]};
    pose.rotation.basisY = {p[3], p[4], p[5]};
    pose.rotation.basisZ = {p[6], p[7], p[8]};
    pose.translation = {p[9], p[10], p[11]};
    return pose;
}

c::RaceCameraVehicleState PublicInput(const d::VehicleFrame &v) {
    c::RaceCameraVehicleState p;
    p.targetId = v.targetId;
    p.timeMs = v.timeMs;
    p.transform = d::ToPublic(v.transform);
    p.resetTransform = d::ToPublic(*v.resetTransform);
    p.signedSpeed = v.signedSpeed;
    p.cameraSupportUp = d::ToPublic(v.cameraSupportUp);
    p.linearSpeed = d::ToPublic(v.linearSpeed);
    p.steering = v.steering;
    p.accelerate = v.accelerate;
    p.brake = v.brake;
    p.turbo = v.turbo;
    p.cameraFlightTransition = v.cameraFlightTransition;
    p.burning = v.burning;
    p.gearChanged = v.gearChanged;
    p.wheelContact = v.wheelContact;
    p.wheelHasSurface = v.wheelHasSurface;
    return p;
}

struct Errors {
    float position = 0.0f, axes = 0.0f, lens = 0.0f;
    unsigned updates = 0u, failures = 0u;
    void Observe(const d::CameraFrame &actual, const float *expected, unsigned row,
                 unsigned camera, bool publicApi) {
        const auto &m = actual.transform.rotation;
        const auto &t = actual.transform.translation;
        const std::array<float, 12> matrix{
                m.basisX.x, m.basisX.y, m.basisX.z, m.basisY.x, m.basisY.y, m.basisY.z,
                m.basisZ.x, m.basisZ.y, m.basisZ.z, t.x,        t.y,        t.z};
        float p = 0, a = 0, l = 0;
        bool encodingMismatch = false;
        for (unsigned i = 0; i < 17; ++i) {
            const float value = i < 12 ? matrix[i] : actual.lens[i - 12];
            const float error = std::fabs(value - expected[i]);
            if (!publicApi && std::memcmp(&value, &expected[i], sizeof(float)) != 0) {
                if (failures < 5)
                    std::cerr << "component=" << i << " actual=" << std::hexfloat
                              << value << " expected=" << expected[i]
                              << std::defaultfloat << '\n';
                encodingMismatch = true;
            }
            if (!std::isfinite(error))
                throw std::runtime_error("nonfinite camera output");
            auto &maximum = i < 9 ? a : i < 12 ? p : l;
            maximum = std::max(maximum, error);
        }
        position = std::max(position, p);
        axes = std::max(axes, a);
        lens = std::max(lens, l);
        ++updates;
        const float positionTolerance = publicApi ? 0.0001f : 0.0f;
        const float axisTolerance = publicApi ? 0.00002f : 0.0f;
        if (p > positionTolerance || a > axisTolerance || l != 0.0f || encodingMismatch) {
            if (failures++ < 5)
                std::cerr << "camera " << camera << (publicApi ? " public" : " raw")
                          << " row=" << row << " position=" << p << " axes=" << a
                          << " lens=" << l << " encodingMismatch=" << encodingMismatch << '\n';
        }
    }
};

int main(int argc, char **argv) {
    try {
        std::ifstream in(argc > 1 ? argv[1] : NATIVE_CAMERA_FIXTURE, std::ios::binary);
        if (ReadU32(in) != 0x46564332u || ReadU32(in) != 296u)
            throw std::runtime_error("invalid native camera fixture");
        const auto count = ReadU32(in);
        if (count > 100000u)
            throw std::runtime_error("invalid fixture count");
        auto environment =
                d::RaceCameraEnvironmentFactory::Create(NativeCameraConfig());
        if (argc > 2) {
            auto installed = forevervalidator::LoadInstalledRaceCameraEnvironment(
                    argv[2], "Stadium");
            if (!installed)
                throw std::runtime_error(installed.Error().diagnostic);
            environment = installed.Value();
            std::cout << "using installed resources: " << environment.SourceName()
                      << '\n';
        }
        const auto &config = d::RaceCameraEnvironmentFactory::Config(environment);
        CheckNativeCameraConfig(config);
        std::array<std::unique_ptr<d::Controller>, 2> controllers{
                d::CreateController(config, c::RaceCameraProfile::Race2),
                d::CreateController(config, c::RaceCameraProfile::Race3)};
        std::array<c::RaceCameraSession, 2> publicSessions{
                c::RaceCameraSession(environment, c::RaceCameraProfile::Race2),
                c::RaceCameraSession(environment, c::RaceCameraProfile::Race3)};
        std::array<Errors, 2> rawErrors{}, publicErrors{};
        std::array<unsigned, 2> resets{}, turns{}, airborne{}, reverse{};
        unsigned hits = 0;
        for (std::uint32_t row = 0; row < count; ++row) {
            const auto kind = ReadU32(in), phase = ReadU32(in), time = ReadU32(in);
            const auto p = ReadFloats<63>(in);
            const auto collisionKind = ReadU32(in);
            const auto collision = ReadFloats<7>(in);
            if (kind < 2 || kind > 3 || (phase != 1 && phase != 3) || collisionKind > 2)
                throw std::runtime_error("invalid fixture record");
            const auto index = kind - 2;
            d::VehicleFrame v;
            v.targetId = 1;
            v.timeMs = time;
            v.transform = Pose(p.data());
            v.resetTransform = Pose(p.data() + 12);
            v.signedSpeed = p[24];
            v.cameraSupportUp = {p[25], p[26], p[27]};
            v.linearSpeed = {p[28], p[29], p[30]};
            v.steering = p[31];
            v.accelerate = p[32];
            v.brake = p[33];
            v.turbo = p[34];
            v.cameraFlightTransition = p[35];
            v.burning = p[36] != 0;
            v.gearChanged = p[37] != 0;
            for (unsigned i = 0; i < 4; ++i) {
                v.wheelContact[i] = p[38 + i] != 0;
                v.wheelHasSurface[i] = p[42 + i] != 0;
            }
            const auto publicInput = PublicInput(v);
            if (phase == 3) {
                controllers[index]->Reset(v);
                publicSessions[index].Reset(publicInput);
                ++resets[index];
                continue;
            }
            if (!resets[index])
                throw std::runtime_error("missing initial reset");
            turns[index] += std::fabs(v.steering) > 0.1f;
            airborne[index] += v.AllWheelsWithoutSurface();
            reverse[index] += v.signedSpeed < -4.0f;
            hits += collisionKind == 2;
            unsigned queries = 0;
            const c::SegmentCollisionQuery query =
                    [&](const c::SegmentQuery &q) -> std::optional<c::SegmentHit> {
                if (collisionKind == 0 || queries++)
                    throw std::runtime_error("unexpected collision query at row " +
                                             std::to_string(row));
                const std::array<float, 6> coordinates{
                        q.start.x,           q.start.y,           q.start.z,
                        q.end.x - q.start.x, q.end.y - q.start.y, q.end.z - q.start.z};
                for (unsigned j = 0; j < 6; ++j) {
                    if (std::fabs(coordinates[j] - collision[j]) > 0.0001f)
                        throw std::runtime_error("collision segment mismatch at row " +
                                                 std::to_string(row));
                }
                return collisionKind == 2 ? std::optional<c::SegmentHit>{{collision[6]}}
                                          : std::nullopt;
            };
            rawErrors[index].Observe(controllers[index]->Evaluate(v, query),
                                     p.data() + 46, row, index + 1, false);
            if (queries != (collisionKind != 0))
                throw std::runtime_error("missing native collision query");
            queries = 0;
            const auto pub = publicSessions[index].Evaluate({publicInput, query});
            const d::CameraFrame converted{d::ToInternal(pub.transform),
                                           {pub.lens.fieldOfViewDegrees,
                                            pub.lens.parameter1, pub.lens.parameter2,
                                            pub.lens.nearClipDistance,
                                            pub.lens.farClipDistance}};
            publicErrors[index].Observe(converted, p.data() + 46, row, index + 1, true);
            if (queries != (collisionKind != 0))
                throw std::runtime_error("missing public collision query");
        }
        bool failed = false;
        for (unsigned i = 0; i < 2; ++i) {
            if (rawErrors[i].updates < 300 || resets[i] < 2 || turns[i] < 10 ||
                airborne[i] < 10 || reverse[i] < 10)
                throw std::runtime_error(
                        "insufficient native scenario coverage for camera " +
                        std::to_string(i + 1));
            for (const auto *errors : {&rawErrors[i], &publicErrors[i]}) {
                std::cout << "camera " << i + 1
                          << (errors == &rawErrors[i] ? " raw" : " public") << ": "
                          << errors->updates
                          << " updates, max position=" << errors->position
                          << " axes=" << errors->axes << " lens=" << errors->lens
                          << " failures=" << errors->failures << '\n';
                failed |= errors->failures != 0;
            }
            std::cout << "  resets=" << resets[i] << " turning=" << turns[i]
                      << " airborne=" << airborne[i] << " reverse=" << reverse[i]
                      << '\n';
        }
        if (hits < 10 || in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error(
                    "invalid native collision coverage or trailing data");
        std::cout << "native collision hits=" << hits << '\n';
        return failed ? 1 : 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
