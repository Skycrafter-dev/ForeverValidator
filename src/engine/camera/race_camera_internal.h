#pragma once

#include <forevervalidator/camera.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "engine/core/gm_types.h"
#include "engine/camera/race_camera_config.h"

namespace forevervalidator::camera::detail {

struct CameraFrame {
    GmIso4 transform{};
    std::array<float, 5> lens{{90.0f, 60.0f, 3.0f, -1.0f, -1.0f}};
};

struct VehicleFrame {
    std::uint32_t targetId = 0u;
    std::uint32_t timeMs = 0u;
    GmIso4 transform{};
    GmVec3 linearSpeed{};
    std::optional<GmIso4> resetTransform;
    float signedSpeed = 0.0f;
    float steering = 0.0f;
    float accelerate = 0.0f;
    float brake = 0.0f;
    float turbo = 0.0f;
    float cameraFlightTransition = 0.0f;
    bool burning = false;
    bool gearChanged = false;
    bool isVehicleCar = true;
    std::array<bool, 4> wheelContact{{true, true, true, true}};
    std::array<bool, 4> wheelHasSurface{{true, true, true, true}};
    GmVec3 cameraSupportUp{};

    bool AllWheelsWithoutContact() const noexcept;
    bool AllWheelsWithoutSurface() const noexcept;
};

class StateArchive {
  public:
    StateArchive(RaceCameraSession::State &state, bool restoring)
        : state_(state), restoring_(restoring) {}
    template <class... Values> void Fields(Values &...values) {
        (Field(values), ...);
    }
    std::size_t Size() const { return offset_; }

  private:
    template <class Value> void Field(Value &value) {
        static_assert(std::is_trivially_copyable_v<Value>);
        const auto limit = restoring_ ? state_.size : state_.bytes.size();
        if (offset_ + sizeof(value) > limit)
            throw std::invalid_argument("invalid camera controller state");
        if (restoring_)
            std::memcpy(&value, state_.bytes.data() + offset_, sizeof(value));
        else
            std::memcpy(state_.bytes.data() + offset_, &value, sizeof(value));
        offset_ += sizeof(value);
    }
    RaceCameraSession::State &state_;
    bool restoring_;
    std::size_t offset_ = 0u;
};

class Controller {
  public:
    virtual ~Controller() = default;
    virtual RaceCameraProfile Profile() const noexcept = 0;
    virtual void Reset(const VehicleFrame &vehicle) = 0;
    virtual CameraFrame
    Evaluate(const VehicleFrame &vehicle,
             const SegmentCollisionQuery &segmentCollision) = 0;
    virtual std::unique_ptr<Controller> Clone() const = 0;
    virtual void VisitState(StateArchive &archive) = 0;
};

std::unique_ptr<Controller> CreateController(
        const EnvironmentConfig &environment,
        RaceCameraProfile profile,
        std::optional<std::size_t> resourceIndex = std::nullopt);

struct RaceCameraEnvironmentFactory {
    static RaceCameraEnvironment Create(EnvironmentConfig config);
    static const EnvironmentConfig &Config(
            const RaceCameraEnvironment &environment);
};

struct RaceCameraSessionAccess {
    static CameraFrame Evaluate(RaceCameraSession &session,
                                const RaceCameraQuery &query);
};

GmVec3 ToInternal(const Vector3 &value);
Vector3 ToPublic(const GmVec3 &value);
GmQuat ToInternal(const Quaternion &value);
Quaternion ToPublic(const GmQuat &value);
GmIso4 ToInternal(const Transform &value);
Transform ToPublic(const GmIso4 &value);
VehicleFrame ToInternal(const RaceCameraVehicleState &value);
RaceCameraOutput ToPublic(const CameraFrame &value);

} // namespace forevervalidator::camera::detail
