#include "engine/camera/race_camera_internal.h"
#include "engine/camera/native_camera_math.h"

#include <algorithm>
#include <array>
#include <memory>

namespace forevervalidator::camera::detail {
namespace native {

class DrivingCamera : public Controller {
  public:
    explicit DrivingCamera(CameraCommonConfig common) : common_(common) {
        output_.transform.SetIdentity();
        output_.lens = {common.initialFieldOfView, common.initialFieldOfView, 3.0f,
                        -1.0f, -1.0f};
    }
    CameraFrame Evaluate(const VehicleFrame &v,
                         const SegmentCollisionQuery &collision) final {
        if (!initialized_ || target_ != v.targetId)
            Reset(v);
        Update(v, collision);
        return output_;
    }
    void VisitState(StateArchive &archive) override {
        archive.Fields(output_, initialized_, target_);
    }

  protected:
    void BeginReset(const VehicleFrame &v) {
        initialized_ = true;
        target_ = v.targetId;
        output_.transform = v.resetTransform.value_or(v.transform);
    }
    virtual void Update(const VehicleFrame &, const SegmentCollisionQuery &) = 0;
    CameraCommonConfig common_;
    CameraFrame output_;

  private:
    bool initialized_ = false;
    std::uint32_t target_ = 0u;
};

// World-position follower used by CGameControlCameraTrackManiaRace2.
// Update 0045e070, reset 0045d130. Binary provenance: tests/data/camera/README.md.
class Race2 final : public DrivingCamera {
  public:
    explicit Race2(const Race2Config &config)
        : DrivingCamera(config.common), cfg_(config) {
        for (std::size_t i = 0; i < smooth_.size(); ++i)
            smooth_[i] = Smooth(config.smoothers[i]);
    }
    RaceCameraProfile Profile() const noexcept final {
        return RaceCameraProfile::Race2;
    }
    std::unique_ptr<Controller> Clone() const final {
        return std::make_unique<Race2>(*this);
    }
    void VisitState(StateArchive &archive) final {
        DrivingCamera::VisitState(archive);
        archive.Fields(smooth_, distance_, look_, fov_, height_, flightLook_,
                       steerInput_, yaw_, roll_, last_, noContactStart_,
                       flightStart_, burningStart_, steerStart_, steerEnd_,
                       steerDuration_, turnStart_, mode_, noContact_, steering_,
                       turnStarted_);
    }
    void Reset(const VehicleFrame &v) final {
        BeginReset(v);
        output_.transform.translation =
                output_.transform.translation +
                Rotate(output_.transform.rotation, {0.0f, 0.0f, -15.0f});
        for (auto &smooth : smooth_)
            smooth.Reset();
        distance_ = {};
        look_ = {};
        fov_ = 0.0f;
        height_ = 0.0f;
        flightLook_ = 0.0f;
        noContact_ = false;
        steering_ = false;
        turnStarted_ = false;
        mode_ = 0;
        steerInput_ = 0.0f;
        yaw_ = 0.0f;
        roll_ = 0.0f;
        steerDuration_ = 0;
        steerStart_ = 0;
        steerEnd_ = 0;
        turnStart_ = 0;
        burningStart_ = 0;
        flightStart_ = 0;
        noContactStart_ = 0;
        last_ = v.timeMs;
    }

  private:
    float SmoothAt(std::size_t index, bool active, std::uint32_t now) {
        return smooth_[index].Step(active, now);
    }
    void Update(const VehicleFrame &v, const SegmentCollisionQuery &) final {
        const auto now = v.timeMs;
        const float dt = Seconds(Elapsed(now, last_));
        float fov = fov_, height = height_;
        if (v.isVehicleCar) {
            if (v.AllWheelsWithoutContact()) {
                if (!noContact_)
                    noContactStart_ = now;
                noContact_ = true;
            } else
                noContact_ = false;
            if (noContact_ && After(now, noContactStart_, cfg_.airborneValidationMs)) {
                if (mode_ != 2)
                    flightStart_ = now;
                mode_ = 2;
            } else if (v.burning) {
                if (mode_ != 1)
                    burningStart_ = now;
                mode_ = 1;
            } else
                mode_ = 0;
            if (cfg_.rollFromInput) {
                if (std::fabs(v.steering) > 0.1f) {
                    if (!steering_)
                        steerStart_ = now;
                    steering_ = true;
                    if (steerInput_ == 0.0f ||
                        std::signbit(steerInput_) != std::signbit(v.steering)) {
                        turnStarted_ = false;
                        steerInput_ = v.steering;
                    }
                    steerDuration_ += now - last_;
                } else {
                    if (steering_)
                        steerEnd_ = now;
                    steering_ = false;
                    if (After(now, steerEnd_, cfg_.steeringInactiveResetMs)) {
                        steerDuration_ = 0;
                        turnStarted_ = false;
                    }
                }
            }
            const bool road = mode_ == 0, burning = mode_ == 1, flying = mode_ == 2;
            const bool burningTurn = burning && steering_ &&
                                     After(now, std::max(steerStart_, burningStart_),
                                           cfg_.flyingModeDelayMs);
            const bool movingFlight =
                    flying && After(now, flightStart_, cfg_.flyingCameraMoveDelayMs);
            float desiredLook = 0.0f;
            if (movingFlight)
                desiredLook = Curve(cfg_.flyingLookCurve,
                                    -v.linearSpeed.y /
                                            (CIsqrt(v.linearSpeed.x * v.linearSpeed.x +
                                                    v.linearSpeed.z * v.linearSpeed.z) +
                                             1.0f));
            const bool turbo = v.turbo != 0.0f && (road || flying);
            const float gas = SmoothAt(
                    0, (v.accelerate > 0.1f && road) || burning || flying, now);
            fov = SmoothAt(3, turbo, now);
            const float turboDistance = SmoothAt(4, turbo, now);
            const float steer =
                    SmoothAt(2,
                             steerDuration_ != 0 && road &&
                                     v.signedSpeed >= cfg_.orientationMinimumSpeed,
                             now);
            const float gear =
                    SmoothAt(5, v.turbo == 0.0f && v.gearChanged && road, now);
            const float brake = SmoothAt(1, v.brake > 0.1f && road, now);
            const float burningDistance = SmoothAt(7, burningTurn, now);
            const float burningLook = SmoothAt(6, burningTurn, now);
            const float flightDistance = SmoothAt(8, flying, now);
            const float flightLook = SmoothAt(10, flying, now);
            height = SmoothAt(9, movingFlight, now);
            flightLook_ = Slew(flightLook_, desiredLook, cfg_.flyingLookRate, dt);
            distance_.Step(flightDistance + burningDistance +
                                   std::min(brake, std::min(gear, steer)) +
                                   turboDistance + gas,
                           cfg_.distanceSpringStiffness, cfg_.distanceSpringDamping,
                           dt);
            fov_ = Slew(fov_, fov, cfg_.fieldOfViewRate, dt);
            look_.Step(flightLook_ + flightLook + burningLook, cfg_.lookSpringStiffness,
                       cfg_.lookSpringDamping, dt);
            height_ = Slew(height_, height, cfg_.heightRate, dt);
            fov = fov_;
            height = height_;
        }
        const GmIso4 &pose = v.transform;
        const Mat tracking = TrackingAxes(Forward(pose.rotation));
        const float cameraHeight = common_.height + height;
        const Vec origin = FollowPoint(pose, common_) + WorldUp * cameraHeight;
        Vec offset = Unrotate(tracking, output_.transform.translation - origin);
        offset.y = 0.0f;
        float square = offset.z * offset.z + offset.x * offset.x;
        if (square <= NormalizeThreshold) {
            offset = {0.0f, 0.0f, -common_.minimumDistance};
            square = common_.minimumDistance * common_.minimumDistance;
        }
        const float minimum = common_.minimumDistance + distance_.position;
        const float maximum = common_.maximumDistance + distance_.position;
        if (square < minimum * minimum)
            offset = offset * (minimum / CIsqrt(square));
        else if (square > maximum * maximum)
            offset = offset * (maximum / CIsqrt(square));
        const float look = Clamp(common_.lookFactor + look_.position, 0.0f, 1.0f);
        const Vec direction =
                Rotate(tracking, {-offset.x, -offset.y - (1.0f - look) * cameraHeight,
                                  -offset.z});
        output_.transform.translation = Rotate(tracking, offset) + origin;
        output_.transform.rotation = ViewAxes(direction, v.cameraSupportUp);
        if (cfg_.rollFromInput)
            ApplyInputRotation(v);
        RotateView();
        output_.lens[0] = common_.baseFieldOfView + fov;
        output_.lens[1] = 30.0f;
        output_.lens[2] = 3.0f;
        last_ = now;
    }
    void ApplyInputRotation(const VehicleFrame &v) {
        bool yawRight = false, yawLeft = false, rollRight = false, rollLeft = false;
        if (mode_ == 0 && steerDuration_ != 0) {
            if (!turnStarted_) {
                turnStarted_ = true;
                turnStart_ = steerDuration_;
            }
            const auto duration = steerDuration_ - turnStart_;
            yawRight = (duration > cfg_.steeringYawTriggerMs || yaw_ < -0.1f) &&
                       steerInput_ > 0.1f;
            yawLeft = (duration > cfg_.steeringYawTriggerMs || yaw_ > 0.1f) &&
                      steerInput_ < -0.1f;
            rollRight = (duration > cfg_.steeringRollTriggerMs || roll_ < -0.1f) &&
                        steerInput_ > 0.1f;
            rollLeft = (duration > cfg_.steeringRollTriggerMs || roll_ > 0.1f) &&
                       steerInput_ < -0.1f;
        }
        const bool enabled = v.signedSpeed >= cfg_.orientationMinimumSpeed;
        const float yawScale =
                Curve(cfg_.yawFromSpeedCurve, v.signedSpeed) * Pi / 180.0f;
        const float rollScale =
                Curve(cfg_.rollFromSpeedCurve, v.signedSpeed) * Pi / 180.0f;
        yaw_ = Clamp(-yawScale * SmoothAt(11, yawRight && enabled, v.timeMs) +
                             yawScale * SmoothAt(12, yawLeft && enabled, v.timeMs),
                     -Pi * 0.5f, Pi * 0.5f);
        roll_ = Clamp(-rollScale * SmoothAt(13, rollRight && enabled, v.timeMs) +
                              rollScale * SmoothAt(14, rollLeft && enabled, v.timeMs),
                      -Pi * 0.5f, Pi * 0.5f);
    }
    void RotateView() {
        // The native names are counterintuitive: 0045efec rotates about DOV
        // using the first angle, then 0045f04c rotates world Y using the second.
        // Both matrix operations also run when input rotation is disabled.
        Mat &m = output_.transform.rotation;
        const Vec axis = Unit(Forward(m));
        const float s = CIsin(yaw_ * 0.5f), w = CIcos(yaw_ * 0.5f);
        const float x = axis.x * s, y = axis.y * s, z = axis.z * s;
        const Mat rotation =
                Axes({1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + z * w),
                      2.0f * (x * z - y * w)},
                     {2.0f * (x * y - z * w), 1.0f - 2.0f * (x * x + z * z),
                      2.0f * (y * z + x * w)},
                     {2.0f * (x * z + y * w), 2.0f * (y * z - x * w),
                      1.0f - 2.0f * (x * x + y * y)});
        m = Axes(Rotate(rotation, {m.basisX.x, m.basisY.x, m.basisZ.x}),
                 Rotate(rotation, Up(m)), Rotate(rotation, Forward(m)));
        const Vec xrow = m.basisX, zrow = m.basisZ;
        m.basisX = xrow * CIcos(roll_) + zrow * CIsin(roll_);
        m.basisZ = zrow * CIcos(roll_) - xrow * CIsin(roll_);
    }
    Race2Config cfg_;
    std::array<Smooth, 15> smooth_{};
    Spring distance_, look_;
    float fov_ = 0.0f, height_ = 0.0f, flightLook_ = 0.0f, steerInput_ = 0.0f,
          yaw_ = 0.0f, roll_ = 0.0f;
    std::uint32_t last_ = 0, noContactStart_ = 0, flightStart_ = 0, burningStart_ = 0;
    std::uint32_t steerStart_ = 0, steerEnd_ = 0, steerDuration_ = 0, turnStart_ = 0;
    int mode_ = 0;
    bool noContact_ = false, steering_ = false, turnStarted_ = false;
};

// Retained direction/up follower, 0045a660 / 00459d00. It is a distinct
// controller, not camera 1 with a different boom length.
class Race3 final : public DrivingCamera {
  public:
    explicit Race3(const Race3Config &c)
        : DrivingCamera(c.common), cfg_(c), reverse_(c.reverse),
          lowFlight_(c.lowSpeedFlightTarget), flightDirection_(c.flyingDirectionBlend),
          flightUp_(c.flyingUpBlend), directionRate_(c.flyingSpeedEffect),
          upRate_(c.flyingUpSpeedEffect), flightLook_(c.flyingLook),
          flightRadius_(c.flyingRadius), gas_(c.gasFar), brake_(c.brakeFar),
          steer_(c.steerFar), turboFov_(c.turboFieldOfView), turboFar_(c.turboFar),
          gear_(c.gearFar), burningLook_(c.burningLook),
          burningRadius_(c.burningRadius) {}
    RaceCameraProfile Profile() const noexcept final {
        return RaceCameraProfile::Race3;
    }
    std::unique_ptr<Controller> Clone() const final {
        return std::make_unique<Race3>(*this);
    }
    void VisitState(StateArchive &archive) final {
        DrivingCamera::VisitState(archive);
        for (Smooth *smooth : AllSmoothers()) archive.Fields(*smooth);
        archive.Fields(previousTarget_, previousUp_, direction_, groundDirection_,
                       desiredDirection_, flightTarget_, flightDirectionTarget_,
                       lowFlightTarget_, up_, groundUp_, desiredUp_, flightUpTarget_,
                       radius_, down_, flightLookValue_, last_, steerDuration_,
                       steerStart_, steerEnd_, burningStart_, airStart_,
                       previousSurfaceCount_, mode_, steering_, airborne_);
    }
    void Reset(const VehicleFrame &v) final {
        BeginReset(v);
        previousTarget_ = output_.transform.translation;
        Mat &m = output_.transform.rotation;
        if (v.isVehicleCar && v.AllWheelsWithoutSurface() && m.basisY.y < -0.5f) {
            const Vec oldX = m.basisX, oldY = m.basisY;
            m.basisX = oldX * CIcos(Pi) - oldY * CIsin(Pi);
            m.basisY = oldY * CIcos(Pi) + oldX * CIsin(Pi);
        }
        const Vec offset = Rotate(m, {0.0f, cfg_.up, -cfg_.far});
        output_.transform.translation = output_.transform.translation + offset;
        direction_ = groundDirection_ = desiredDirection_ = flightTarget_ =
                flightDirectionTarget_ = Unit(offset);
        lowFlightTarget_ = Rotate(m, {0.0f, cfg_.up + cfg_.up, -cfg_.far});
        lowFlightTarget_.y = std::fabs(lowFlightTarget_.y);
        lowFlightTarget_ = Unit(lowFlightTarget_);
        up_ = groundUp_ = desiredUp_ = Up(m);
        flightUpTarget_ = WorldUp;
        previousUp_ = WorldUp;
        radius_ = {CIsqrt(cfg_.up * cfg_.up + cfg_.far * cfg_.far), 0.0f};
        down_ = {};
        flightLookValue_ = 0.0f;
        last_ = v.timeMs;
        steerDuration_ = steerStart_ = steerEnd_ = burningStart_ = airStart_ = 0u;
        mode_ = 0;
        steering_ = false;
        airborne_ = false;
        previousSurfaceCount_ = 0u;
        for (Smooth *s : AllSmoothers())
            s->Reset();
    }

  private:
    std::array<Smooth *, 16> AllSmoothers() {
        return {&reverse_,       &lowFlight_, &flightDirection_, &flightUp_,
                &directionRate_, &upRate_,    &flightLook_,      &flightRadius_,
                &gas_,           &brake_,     &steer_,           &turboFov_,
                &turboFar_,      &gear_,      &burningLook_,     &burningRadius_};
    }
    Vec SmoothAxis(Vec current, Vec desired, float normalRate, float flyingRate,
                   Smooth &transition, bool flying, float speed, float dt,
                   std::uint32_t now) {
        const float dot = Dot(current, desired);
        const float angle = AcosClamped(dot);
        const float air = transition.Step(flying, now);
        const float denominator = (angle + angle) + 1.0f;
        const float normal = Clamp(normalRate * dt / denominator, 0.0f, 1.0f);
        const float flight = Clamp(flyingRate * dt / denominator, 0.0f, 1.0f);
        // Preserve the x87 multiplication order, including the inner product.
        const float blend =
                Curve(cfg_.speedModulationCurve, speed) * (normal * (1.0f - air)) +
                flight * air;
        float a = 1.0f - blend, b = blend;
        if (1.0f - dot > SmoothThreshold) {
            const float sine = CIsin(angle);
            a = CIsin((1.0f - blend) * angle) / sine;
            b = CIsin(blend * angle) / sine;
        }
        return current * a + desired * b;
    }
    void Collision(const VehicleFrame &v, const SegmentCollisionQuery &query) {
        if (!v.isVehicleCar || cfg_.collisionRadius <= 0.0001f)
            return;
        const auto count = static_cast<unsigned>(
                std::count(v.wheelHasSurface.begin(), v.wheelHasSurface.end(), true));
        if (count * 2 <= 4 && previousSurfaceCount_ * 2 <= 4 && query) {
            const Vec start = v.transform.translation + direction_ * radius_.position;
            const Vec displacement = {0.0f, -2.0f * cfg_.collisionRadius, 0.0f};
            const auto hit = query({ToPublic(start), ToPublic(start + displacement)});
            if (hit && hit->fraction < 1.0f) {
                const float y = (hit->fraction * displacement.y + start.y +
                                 cfg_.collisionRadius - v.transform.translation.y) /
                                radius_.position;
                if (desiredDirection_.y < y) {
                    desiredDirection_.y = Clamp(y, -1.0f, 1.0f);
                    const float factor =
                            (1.0f - desiredDirection_.y * desiredDirection_.y) /
                            (desiredDirection_.x * desiredDirection_.x +
                             desiredDirection_.z * desiredDirection_.z);
                    const float scale =
                            factor > SmoothThreshold ? CIsqrt(factor) : 0.0f;
                    desiredDirection_.x *= scale;
                    desiredDirection_.z *= scale;
                }
            }
        }
        previousSurfaceCount_ = count;
    }
    void Update(const VehicleFrame &v, const SegmentCollisionQuery &collision) final {
        previousUp_ = Up(output_.transform.rotation);
        const auto now = v.timeMs, elapsed = Elapsed(now, last_);
        const float dt = Seconds(elapsed);
        last_ = now;
        const float speed = v.signedSpeed;
        const bool lowSpeed = std::fabs(speed) < cfg_.lowSpeedThreshold;
        const bool veryLowSpeed = std::fabs(speed) < cfg_.veryLowSpeedThreshold;
        const Vec velocity = v.isVehicleCar ? v.linearSpeed : Vec{};
        const float horizontal =
                CIsqrt(velocity.x * velocity.x + velocity.z * velocity.z);
        if (v.isVehicleCar) {
            if (v.AllWheelsWithoutSurface()) {
                if (!airborne_)
                    airStart_ = now;
                airborne_ = true;
            } else
                airborne_ = false;
            if (airborne_ && After(now, airStart_, cfg_.airborneValidationMs))
                mode_ = 2;
            else if (v.burning) {
                if (mode_ != 1)
                    burningStart_ = now;
                mode_ = 1;
            } else
                mode_ = 0;
            if (std::fabs(v.steering) > 0.1f) {
                if (!steering_)
                    steerStart_ = now;
                steering_ = true;
                steerDuration_ += elapsed;
            } else {
                if (steering_)
                    steerEnd_ = now;
                steering_ = false;
                if (After(now, steerEnd_, cfg_.steeringInactiveResetMs))
                    steerDuration_ = 0;
            }
        }
        const bool road = mode_ == 0, burning = mode_ == 1, flying = mode_ == 2;
        const bool burningTurn = burning && steering_ &&
                                 After(now, std::max(steerStart_, burningStart_),
                                       cfg_.burningSteerDelayMs);
        const bool turbo = v.isVehicleCar && v.turbo != 0.0f && (road || flying);
        const bool gas = v.isVehicleCar && v.accelerate > 0.1f;
        const bool brake = v.isVehicleCar && v.brake > 0.1f;
        float far = gas_.Step(
                (((gas && speed > 2.0f) || (brake && speed < -2.0f)) && road) ||
                        burning || flying,
                now);
        const float fov = turboFov_.Step(turbo, now);
        far += turboFar_.Step(turbo, now);
        const float gear = gear_.Step(
                v.isVehicleCar && v.turbo == 0.0f && v.gearChanged && road, now);
        const float steer = steer_.Step(steerDuration_ != 0 && road && !lowSpeed, now);
        far += std::min(brake_.Step(brake && road, now), std::min(gear, steer));
        const float extraRadius =
                flightRadius_.Step(flying, now) + burningRadius_.Step(burningTurn, now);
        const float extraLook =
                flightLook_.Step(flying, now) + burningLook_.Step(burningTurn, now);
        const float slope = -velocity.y / (horizontal + 1.0f);
        flightLookValue_ = Slew(flightLookValue_,
                                flying ? Curve(cfg_.flyingLookCurve, slope) : 0.0f,
                                cfg_.flyingLookRate, dt);
        far = std::max(cfg_.far + far, 0.1f);
        radius_.Step(CIsqrt(far * far + cfg_.up * cfg_.up) + extraRadius,
                     cfg_.radiusSpringStiffness, cfg_.radiusSpringDamping, dt);
        down_.Step(Clamp(Curve(cfg_.flyingDownCurve, slope), 0.0f, 0.95f),
                   cfg_.flyingDownSpringStiffness, cfg_.flyingDownSpringDamping, dt);
        const float reverse = reverse_.Step(speed < -4.0f && road, now);
        if (!flying) {
            const float angle = reverse * Pi;
            groundDirection_ =
                    Unit(Rotate(v.transform.rotation,
                                {CIsin(angle) * far, cfg_.up, -far * CIcos(angle)}));
            groundUp_ = Up(v.transform.rotation);
        } else {
            if (!Parallel(direction_, WorldUp)) {
                const Vec horizontalDirection =
                        Unit({direction_.x, 0.0f, direction_.z});
                lowFlightTarget_ = horizontalDirection * far;
                lowFlightTarget_.y = cfg_.up + cfg_.up;
                lowFlightTarget_ = Unit(lowFlightTarget_);
            }
            Vec heading = previousTarget_ + direction_ - v.transform.translation;
            heading.y = 0.0f;
            heading = Unit(heading);
            const float pitch = Clamp(
                    cfg_.constantFlyingLookDownFactor + down_.position, 0.0f, 0.95f);
            const float horizontalWeight = CIsin((1.0f - pitch) * Pi * 0.5f);
            flightDirectionTarget_ = {heading.x * horizontalWeight,
                                      CIsin(pitch * Pi * 0.5f),
                                      heading.z * horizontalWeight};
            flightUpTarget_ = WorldUp;
        }
        flightTarget_ = Blend(
                flightDirectionTarget_, lowFlightTarget_,
                lowFlight_.Step(veryLowSpeed && v.cameraFlightTransition != 0.0f &&
                                        flying,
                                now));
        desiredDirection_ = Blend(groundDirection_, flightTarget_,
                                  flightDirection_.Step(flying, now, true));
        desiredUp_ =
                Blend(groundUp_, flightUpTarget_, flightUp_.Step(flying, now, true));
        Collision(v, collision);
        previousTarget_ = v.transform.translation;
        direction_ = SmoothAxis(direction_, desiredDirection_, cfg_.directionNormalRate,
                                cfg_.directionFlyingRate, directionRate_, flying, speed,
                                dt, now);
        up_ = SmoothAxis(up_, desiredUp_, cfg_.upNormalRate, cfg_.upFlyingRate, upRate_,
                         flying, speed, dt, now);
        output_.transform.translation =
                v.transform.translation + direction_ * radius_.position;
        const float look =
                Clamp(common_.lookFactor + (flightLookValue_ + extraLook), 0.0f, 1.0f) *
                cfg_.up;
        const Vec aim = previousTarget_ + up_ * look - output_.transform.translation;
        output_.transform.rotation =
                ViewAxes(aim, Parallel(up_, aim) ? previousUp_ : up_);
        output_.lens[0] = common_.baseFieldOfView + fov;
        output_.lens[1] = radius_.position;
        output_.lens[2] = 1.0f;
    }
    Race3Config cfg_;
    Smooth reverse_, lowFlight_, flightDirection_, flightUp_, directionRate_, upRate_;
    Smooth flightLook_, flightRadius_, gas_, brake_, steer_, turboFov_, turboFar_,
            gear_, burningLook_, burningRadius_;
    Vec previousTarget_{}, previousUp_ = WorldUp, direction_{}, groundDirection_{},
                           desiredDirection_{};
    Vec flightTarget_{}, flightDirectionTarget_{}, lowFlightTarget_{},
            up_ = WorldUp, groundUp_ = WorldUp;
    Vec desiredUp_ = WorldUp, flightUpTarget_ = WorldUp;
    Spring radius_, down_;
    float flightLookValue_ = 0.0f;
    std::uint32_t last_ = 0, steerDuration_ = 0, steerStart_ = 0, steerEnd_ = 0,
                  burningStart_ = 0, airStart_ = 0;
    unsigned previousSurfaceCount_ = 0;
    int mode_ = 0;
    bool steering_ = false, airborne_ = false;
};

class Classic final : public DrivingCamera {
  public:
    explicit Classic(const RaceConfig &c) : DrivingCamera(c.common), cfg_(c) {}
    RaceCameraProfile Profile() const noexcept final { return RaceCameraProfile::Race; }
    std::unique_ptr<Controller> Clone() const final {
        return std::make_unique<Classic>(*this);
    }
    void Reset(const VehicleFrame &v) final {
        BeginReset(v);
        output_.transform.translation =
                output_.transform.translation +
                Rotate(output_.transform.rotation, {0.0f, 0.0f, -15.0f});
    }

  private:
    void Update(const VehicleFrame &v, const SegmentCollisionQuery &) final {
        GmIso4 pose = v.transform;
        if (common_.ignoreTargetRotation)
            pose.rotation.SetIdentity();
        const Mat tracking = TrackingAxes(Forward(pose.rotation));
        const Vec origin = FollowPoint(pose, common_) + WorldUp * common_.height;
        Vec offset = Unrotate(tracking, output_.transform.translation - origin);
        offset.y = 0.0f;
        float squared = offset.x * offset.x + offset.z * offset.z;
        if (squared <= NormalizeThreshold) {
            offset = {0.0f, 0.0f, -common_.minimumDistance};
            squared = common_.minimumDistance * common_.minimumDistance;
        }
        if (squared < common_.minimumDistance * common_.minimumDistance)
            offset = offset * (common_.minimumDistance / CIsqrt(squared));
        else if (squared > common_.maximumDistance * common_.maximumDistance)
            offset = offset * (common_.maximumDistance / CIsqrt(squared));
        float cone = 1.0f;
        if (v.signedSpeed >= cfg_.coneMinimumSpeed) {
            cone = v.signedSpeed > cfg_.coneMaximumSpeed
                           ? cfg_.coneAperture
                           : (1.0f - cfg_.coneAperture) *
                                             (cfg_.coneMaximumSpeed - v.signedSpeed) /
                                             (cfg_.coneMaximumSpeed -
                                              cfg_.coneMinimumSpeed) +
                                     cfg_.coneAperture;
        }
        if (cone < 0.9999f) {
            const float angle =
                    Clamp(CIatan2(-offset.x, -offset.z), -cone * Pi, cone * Pi);
            const float radius = Length(offset);
            offset.x = -CIsin(angle) * radius;
            offset.z = -CIcos(angle) * radius;
        }
        const Vec aim = Rotate(
                tracking,
                {-offset.x, -(1.0f - common_.lookFactor) * common_.height, -offset.z});
        output_.transform.translation = Rotate(tracking, offset) + origin;
        output_.transform.rotation = ViewAxes(aim, v.cameraSupportUp);
        output_.lens[0] = common_.initialFieldOfView;
        output_.lens[1] = 30.0f;
        output_.lens[2] = 3.0f;
    }
    RaceConfig cfg_;
};

class Internal final : public DrivingCamera {
  public:
    explicit Internal(const CameraCommonConfig &c) : DrivingCamera(c) {}
    RaceCameraProfile Profile() const noexcept final {
        return RaceCameraProfile::Internal;
    }
    std::unique_ptr<Controller> Clone() const final {
        return std::make_unique<Internal>(*this);
    }
    void Reset(const VehicleFrame &v) final { BeginReset(v); }

  private:
    void Update(const VehicleFrame &v, const SegmentCollisionQuery &) final {
        output_.transform = v.transform;
        if (common_.ignoreTargetRotation)
            output_.transform.rotation.SetIdentity();
        output_.transform.translation = FollowPoint(output_.transform, common_);
    }
};

template <class Model, class Resources, class Config>
std::unique_ptr<Controller> Select(const Resources &resources,
                                   const std::optional<Config> &fallback,
                                   std::optional<std::size_t> index) {
    if (!index)
        return fallback ? std::make_unique<Model>(*fallback) : nullptr;
    return *index < resources.size() ? std::make_unique<Model>(resources[*index].config)
                                     : nullptr;
}
} // namespace native

std::unique_ptr<Controller> CreateController(const EnvironmentConfig &environment,
                                             RaceCameraProfile profile,
                                             std::optional<std::size_t> resourceIndex) {
    switch (profile) {
    case RaceCameraProfile::Race:
        return native::Select<native::Classic>(environment.raceResources,
                                               environment.race, resourceIndex);
    case RaceCameraProfile::Race2:
        return native::Select<native::Race2>(environment.race2Resources,
                                             environment.race2, resourceIndex);
    case RaceCameraProfile::Race3:
        return native::Select<native::Race3>(environment.race3Resources,
                                             environment.race3, resourceIndex);
    case RaceCameraProfile::Internal:
        return native::Select<native::Internal>(environment.internalResources,
                                                environment.internal, resourceIndex);
    }
    return nullptr;
}
} // namespace forevervalidator::camera::detail
