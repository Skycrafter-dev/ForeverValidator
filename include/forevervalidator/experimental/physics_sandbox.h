#ifndef FOREVERVALIDATOR_EXPERIMENTAL_PHYSICS_SANDBOX_H
#define FOREVERVALIDATOR_EXPERIMENTAL_PHYSICS_SANDBOX_H

// This API is experimental. It may change without compatibility guarantees.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <forevervalidator/validation.h>

namespace forevervalidator::experimental {

enum class PhysicsSandboxErrorCode : std::uint8_t {
    InvalidSandbox,
    InvalidRequest,
    ReplayLoadingFailed,
    MapLoadingFailed,
    SimulationFailed,
    IncompatibleState,
    AllocationFailed,
    UnexpectedFailure,
};

struct PhysicsSandboxError {
    PhysicsSandboxErrorCode code = PhysicsSandboxErrorCode::UnexpectedFailure;
    ValidationError validationError{};
    std::string diagnostic;
};

template<typename T>
using PhysicsSandboxResult =
        DiscriminatedResult<T, PhysicsSandboxError>;

struct PhysicsSandboxOptions {
    SimulationBackend backend = SimulationBackend::Reference;
    std::uint32_t tickDurationMs = 10u;
    std::uint32_t prestartDurationMs = 2600u;
};

enum class PhysicsSandboxInputAction : std::uint8_t {
    Unmapped,
    Accelerate,
    Gas,
    Brake,
    Steer,
    SteerLeft,
    SteerRight,
    RaceRunning,
    FinishLine,
    Respawn,
};

enum class PhysicsSandboxInputValueKind : std::uint8_t {
    None,
    Switch,
    Analog,
};

enum class PhysicsSandboxSwitchState : std::uint8_t {
    Released,
    Pressed,
    NonCanonicalActive,
};

struct PhysicsSandboxInputValue {
    PhysicsSandboxInputValueKind kind = PhysicsSandboxInputValueKind::None;
    PhysicsSandboxSwitchState switchState =
            PhysicsSandboxSwitchState::Released;
    float analog = 0.0f;
};

struct PhysicsSandboxInputEvent {
    std::int32_t timeMs = 0;
    PhysicsSandboxInputAction action = PhysicsSandboxInputAction::Unmapped;
    PhysicsSandboxInputValue value{};
};

struct PhysicsSandboxCarState {
    float rotationX = 0.0f;
    float rotationY = 0.0f;
    float rotationZ = 0.0f;
    float rotationW = 1.0f;
    Vector3 position{};
    Vector3 linearSpeed{};
    Vector3 angularSpeed{};
    Vector3 force{};
    Vector3 torque{};
};

struct PhysicsSandboxStateView {
    std::uint64_t tick = 0u;
    std::uint64_t timeMs = 0u;
    MapEnvironment mapEnvironment = MapEnvironment::Unknown;
    VehicleModel vehicleModel = VehicleModel::Unknown;
    std::optional<PlayMode> playMode;
    PhysicsSandboxCarState car{};
    float accelerate = 0.0f;
    float brake = 0.0f;
    float steering = 0.0f;
    std::uint32_t checkpointsCollected = 0u;
    std::uint32_t checkpointsTotal = 0u;
    std::uint32_t completedLaps = 0u;
    std::uint32_t totalLaps = 1u;
    bool raceCompleted = false;
    std::optional<std::uint32_t> finishTimeMs;
    std::uint32_t respawnCount = 0u;
    std::optional<std::uint32_t> stuntsScore;
};

// An opaque in-process runtime clone. States are not serializable and are not
// compatible across ForeverValidator builds.
class PhysicsSandboxState {
public:
    PhysicsSandboxState(const PhysicsSandboxState &);
    PhysicsSandboxState &operator=(const PhysicsSandboxState &);
    PhysicsSandboxState(PhysicsSandboxState &&) noexcept;
    PhysicsSandboxState &operator=(PhysicsSandboxState &&) noexcept;
    ~PhysicsSandboxState();

    const PhysicsSandboxStateView &View() const noexcept;

private:
    struct Impl;
    explicit PhysicsSandboxState(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
    friend class PhysicsSandbox;
};

class PhysicsSandbox {
public:
    PhysicsSandbox(PhysicsSandbox &&) noexcept;
    PhysicsSandbox &operator=(PhysicsSandbox &&) noexcept;
    ~PhysicsSandbox();
    PhysicsSandbox(const PhysicsSandbox &) = delete;
    PhysicsSandbox &operator=(const PhysicsSandbox &) = delete;

    SimulationBackend Backend() const noexcept;
    PhysicsSandboxResult<PhysicsSandboxStateView> LoadReplay(
            ByteView replayBytes,
            const ReplayIdentity &identity) noexcept;
    PhysicsSandboxResult<std::vector<PhysicsSandboxInputEvent>> ReadInputs()
            const noexcept;
    PhysicsSandboxResult<std::size_t> ReplaceInputs(
            std::vector<PhysicsSandboxInputEvent> events) noexcept;
    PhysicsSandboxResult<PhysicsSandboxStateView> AdvanceTicks(
            std::uint32_t count) noexcept;
    PhysicsSandboxResult<PhysicsSandboxState> CaptureState() const noexcept;
    PhysicsSandboxResult<PhysicsSandboxStateView> RestoreState(
            const PhysicsSandboxState &state) noexcept;
    PhysicsSandboxResult<PhysicsSandboxStateView> ReadState() const noexcept;

private:
    struct Impl;
    explicit PhysicsSandbox(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend PhysicsSandboxResult<PhysicsSandbox> CreatePhysicsSandbox(
            AssetSource source,
            const PhysicsSandboxOptions &options) noexcept;
    friend std::vector<PhysicsSandboxResult<PhysicsSandboxStateView>>
            AdvancePhysicsSandboxes(
                    const std::vector<PhysicsSandbox *> &sandboxes,
                    std::uint32_t count) noexcept;
};

PhysicsSandboxResult<PhysicsSandbox> CreatePhysicsSandbox(
        AssetSource source,
        const PhysicsSandboxOptions &options = {}) noexcept;

std::vector<PhysicsSandboxResult<PhysicsSandboxStateView>>
AdvancePhysicsSandboxes(
        const std::vector<PhysicsSandbox *> &sandboxes,
        std::uint32_t count) noexcept;

}  // namespace forevervalidator::experimental

#endif
