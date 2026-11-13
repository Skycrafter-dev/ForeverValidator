#ifndef FOREVERVALIDATOR_CAMERA_H
#define FOREVERVALIDATOR_CAMERA_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>

namespace forevervalidator::camera {

namespace detail {
struct RaceCameraEnvironmentFactory;
struct RaceCameraSessionAccess;
}

struct Vector3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Quaternion {
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Transform {
    Quaternion rotation{};
    Vector3 position{};
};

// Exact public representation of the game's five-float GmLensVal. The first,
// fourth, and fifth values have stable engine semantics. The middle two are
// intentionally exposed by neutral storage names because the race controllers
// use them differently between profiles.
struct Lens {
    float fieldOfViewDegrees = 90.0f;
    float parameter1 = 60.0f;
    float parameter2 = 3.0f;
    float nearClipDistance = -1.0f;
    float farClipDistance = -1.0f;
};

enum class RaceCameraProfile : std::uint8_t {
    Race,
    Race2,
    Race3,
    Internal,
};

struct RaceCameraProfileInfo {
    RaceCameraProfile profile = RaceCameraProfile::Race;
    std::uint32_t classId = 0u;
    std::string_view className;
    std::optional<std::uint32_t> stadiumVehicleResourceIndex;
};

struct RaceCameraResourceSelection {
    RaceCameraProfile profile = RaceCameraProfile::Race;
    std::size_t resourceIndex = 0u;
    std::uint32_t collectorIndex = 0xffffffffu;
};

const RaceCameraProfileInfo &ProfileInfo(RaceCameraProfile profile) noexcept;

struct SegmentQuery {
    Vector3 start{};
    Vector3 end{};
};

struct SegmentHit {
    // Parametric hit position along start-to-end. The scene query used by the
    // game returns values in [0, 1], where 1 means no shortening.
    float fraction = 1.0f;
};

using SegmentCollisionQuery =
        std::function<std::optional<SegmentHit>(const SegmentQuery &)>;

struct RaceCameraVehicleState {
    // A stable caller-owned identity for the followed mobil. Changing it has
    // the same effect as changing CGameControlCameraTarget::TargetId in-game.
    std::uint32_t targetId = 0u;

    // Controller time. Queries are expected in chronological order. The
    // original controllers cap a single integration interval at 200 ms.
    std::uint32_t timeMs = 0u;

    Transform transform{};
    Vector3 linearSpeed{};
    float signedSpeed = 0.0f;

    float steering = 0.0f;
    float accelerate = 0.0f;
    float brake = 0.0f;
    float turbo = 0.0f;

    // Raw vehicle-camera transition source used by Race3 while airborne.
    // This maps the original vehicle field at offset 0x4cc; its gameplay
    // meaning is intentionally not guessed here.
    float cameraFlightTransition = 0.0f;

    bool burning = false;
    bool gearChanged = false;
    bool isVehicleCar = true;

    // Race2 tests the wheel contact-present field used by the simulation
    // state (CSceneVehicleCar::SSimulationWheel + 0x124).
    std::array<bool, 4> wheelContact{{true, true, true, true}};

    // Race3 instead tests the asynchronous surface snapshot's has-surface
    // field (CSceneVehicleCar::SSimulationWheel + 0x2a8), including for its
    // camera collision gate. These signals normally agree, but not on every
    // frame, so both are required for exact reproduction.
    std::array<bool, 4> wheelHasSurface{{true, true, true, true}};

    // Race and Race2 read this orientation support vector from the vehicle's
    // camera support resource. It is already negated as expected by the
    // original SetDOVandUpV call.
    Vector3 cameraSupportUp{0.0f, 1.0f, 0.0f};

    // Native reset uses the mobil's simulation pose (+0x7c), whereas ordinary
    // updates use its visual pose (+0x80). Omit when both poses are identical.
    std::optional<Transform> resetTransform;
};

struct RaceCameraQuery {
    RaceCameraVehicleState vehicle{};
    SegmentCollisionQuery segmentCollision;
};

struct RaceCameraOutput {
    Transform transform{};
    Lens lens{};
};

class RaceCameraEnvironment {
  public:
    RaceCameraEnvironment();
    ~RaceCameraEnvironment();

    RaceCameraEnvironment(const RaceCameraEnvironment &) noexcept;
    RaceCameraEnvironment &
    operator=(const RaceCameraEnvironment &) noexcept;
    RaceCameraEnvironment(RaceCameraEnvironment &&) noexcept;
    RaceCameraEnvironment &operator=(RaceCameraEnvironment &&) noexcept;

    bool HasProfile(RaceCameraProfile profile) const noexcept;
    std::string_view SourceName() const noexcept;
    // A collector can declare more than one resource with the same camera
    // class, notably the four classic Race variants.
    std::size_t ResourceCount(RaceCameraProfile profile) const noexcept;
    std::string_view ResourceName(RaceCameraProfile profile,
                                  std::size_t resourceIndex) const noexcept;
    std::optional<RaceCameraResourceSelection>
    ResourceForCollectorIndex(std::uint32_t collectorIndex) const noexcept;
    std::optional<RaceCameraResourceSelection> DefaultResource() const noexcept;

  private:
    class Impl;
    std::shared_ptr<const Impl> impl_;

    explicit RaceCameraEnvironment(std::shared_ptr<const Impl> impl);
    friend class RaceCameraSession;
    friend struct detail::RaceCameraEnvironmentFactory;
};

// Stateful, explicitly queried camera evaluator. It is not attached to the
// validator simulation loop and performs no work unless Reset or Evaluate is
// called by a consumer.
class RaceCameraSession {
  public:
    // Process-local controller state, excluding immutable camera resources.
    // Not a persistent or versioned interchange format.
    struct State {
        std::array<std::uint8_t, 1024u> bytes{};
        std::size_t size = 0u;
        RaceCameraProfile profile = RaceCameraProfile::Race;
    };
    // Uses only profiles declared by the supplied environment. When multiple
    // resources share a class, this selects the collector's final declaration.
    // Construction fails with std::invalid_argument when the profile is absent.
    RaceCameraSession(const RaceCameraEnvironment &environment,
                      RaceCameraProfile profile);
    // Selects one collector-declared resource of the requested class.
    RaceCameraSession(const RaceCameraEnvironment &environment,
                      RaceCameraProfile profile,
                      std::size_t resourceIndex);
    ~RaceCameraSession();

    RaceCameraSession(RaceCameraSession &&) noexcept;
    RaceCameraSession &operator=(RaceCameraSession &&) noexcept;

    RaceCameraSession(const RaceCameraSession &) = delete;
    RaceCameraSession &operator=(const RaceCameraSession &) = delete;

    RaceCameraProfile Profile() const noexcept;
    void Reset(const RaceCameraVehicleState &vehicle);
    RaceCameraOutput Evaluate(const RaceCameraQuery &query);
    // An independent session in this one's exact state, for example to
    // carry on later from a moment a consumer went back to.
    RaceCameraSession Clone() const;
    State CaptureState() const;
    // Restore into a session using the same environment/resource selection.
    void RestoreState(const State &state);

  private:
    class Impl;
    explicit RaceCameraSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend struct detail::RaceCameraSessionAccess;
};

} // namespace forevervalidator::camera

#endif
