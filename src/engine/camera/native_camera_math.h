#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "engine/camera/race_camera_config.h"
#include "engine/core/binary32_math.h"
#include "engine/core/gm_types.h"

namespace forevervalidator::camera::detail::native {

// Native GmIso4 storage is row-major. The public adapter deliberately stores
// these rows in GmMat3's basis fields, yielding the public view quaternion.
using Vec = GmVec3;
using Mat = GmMat3;
constexpr float Pi = 3.1415927410125732f;
constexpr float NormalizeThreshold = 9.999999439624929e-11f;
constexpr float SmoothThreshold = 9.999999747378752e-6f;
constexpr Vec WorldUp{0.0f, 1.0f, 0.0f};

inline Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec operator-(Vec a) { return {-a.x, -a.y, -a.z}; }
inline Vec operator*(Vec a, float b) { return {a.x * b, a.y * b, a.z * b}; }
inline float Dot(Vec a, Vec b) { return (a.x * b.x + a.y * b.y) + a.z * b.z; }
inline Vec Cross(Vec a, Vec b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float Length(Vec a) { return CIsqrt(Dot(a, a)); }
inline Vec UnitSquared(Vec a, float squared) {
    return squared > NormalizeThreshold ? a * (1.0f / CIsqrt(squared)) : a;
}
inline Vec Unit(Vec a) { return UnitSquared(a, Dot(a, a)); }
inline float Clamp(float value, float low, float high) {
    return std::max(low, std::min(high, value));
}
// 004575b0 / 004575f0 saturate before +/-1, not merely outside the domain.
inline float AcosClamped(float value) {
    constexpr float edge = 0.9999989867210388f;
    return value < -edge ? Pi : value > edge ? 0.0f : CIacos(value);
}
inline float AsinClamped(float value) {
    constexpr float edge = 0.9999989867210388f;
    return value < -edge ? -Pi * 0.5f
                         : value > edge ? Pi * 0.5f : CIasin(value);
}
inline Vec Forward(const Mat &m) { return {m.basisX.z, m.basisY.z, m.basisZ.z}; }
inline Vec Up(const Mat &m) { return {m.basisX.y, m.basisY.y, m.basisZ.y}; }
inline Vec Rotate(const Mat &m, Vec p) {
    return {Dot(m.basisX, p), Dot(m.basisY, p), Dot(m.basisZ, p)};
}
inline Vec Unrotate(const Mat &m, Vec p) {
    return {(m.basisX.x * p.x + m.basisY.x * p.y) + m.basisZ.x * p.z,
            (m.basisX.y * p.x + m.basisY.y * p.y) + m.basisZ.y * p.z,
            (m.basisX.z * p.x + m.basisY.z * p.y) + m.basisZ.z * p.z};
}
inline Mat Axes(Vec x, Vec y, Vec z) {
    Mat m;
    m.basisX = {x.x, y.x, z.x};
    m.basisY = {x.y, y.y, z.y};
    m.basisZ = {x.z, y.z, z.z};
    return m;
}
inline Mat ViewAxes(Vec direction, Vec up) {
    const Vec right = Unit(Cross(up, direction));
    const Vec forward = Unit(direction);
    return Axes(right, Cross(forward, right), forward);
}
inline Mat TrackingAxes(Vec forward) {
    const Vec right = Unit(Cross(WorldUp, forward));
    return Axes(right, WorldUp, Cross(right, WorldUp));
}
inline bool Parallel(Vec a, Vec b) {
    return Dot(Cross(a, b), Cross(a, b)) <= SmoothThreshold;
}
inline std::uint32_t Elapsed(std::uint32_t now, std::uint32_t last) {
    return now < last ? 0u : std::min(now - last, 200u);
}
inline float Seconds(std::uint32_t ms) { return static_cast<float>(ms) * 0.001f; }
inline bool After(std::uint32_t now, std::uint32_t since, std::uint32_t delay) {
    return now >= since && now - since > delay;
}
inline float Slew(float previous, float desired, float rate, float dt) {
    const float change = desired - previous;
    return std::fabs(change) > rate * dt
                   ? previous + std::copysign(1.0f, change) * rate * dt
                   : desired;
}
inline Vec Blend(Vec a, Vec b, float blend) {
    if (blend == 0.0f)
        return a;
    if (blend == 1.0f)
        return b;
    float from = 1.0f - blend, to = blend;
    if (1.0f - Dot(a, b) > SmoothThreshold) {
        const float sine = Length(Cross(a, b));
        const float angle = AsinClamped(sine);
        from = CIsin(from * angle) / sine;
        to = CIsin(to * angle) / sine;
    }
    return a * from + b * to;
}

struct Spring {
    float position = 0.0f;
    float velocity = 0.0f;
    float Step(float target, float stiffness, float damping, float dt) {
        velocity += ((target - position) * stiffness - damping * velocity) * dt;
        position += velocity * dt;
        return position;
    }
};

// 00457790 and 00457970. Reversing a transition uses the inverse of the
// previous cosine ramp; a generic exponential interpolator is not equivalent.
class Smooth {
  public:
    Smooth() = default;
    explicit Smooth(SmoothRealConfig config) : config_(config) {}
    void Reset() {
        scale_ = 1.0f;
        value_ = 0.0f;
        startValue_ = 0.0f;
        start_ = 0u;
        rising_ = false;
    }
    float Step(bool active, std::uint32_t now, bool fullTime = false) {
        if (active != rising_) {
            if (!fullTime) {
                const float fraction = Clamp(std::fabs(value_ / config_.delta),
                                             SmoothThreshold, 0.9999899864196777f);
                scale_ = CIacos(rising_ ? 1.0f - (fraction + fraction)
                                        : (fraction + fraction) - 1.0f) /
                         Pi;
            }
            rising_ = active;
            start_ = now;
            startValue_ = value_;
        }
        const auto duration = active ? config_.riseTimeMs : config_.fallTimeMs;
        const bool unfinished = active ? std::fabs(value_) < std::fabs(config_.delta)
                                       : std::fabs(value_) > 0.0f;
        if (!unfinished || now < start_ || duration == 0u ||
            (fullTime ? scale_ == 0.0f : std::fabs(scale_) <= SmoothThreshold))
            return value_;
        const float time = static_cast<float>(now - start_) /
                           (static_cast<float>(duration) * (fullTime ? 1.0f : scale_));
        const float cosine = CIcos(Clamp(time, 0.0f, 1.0f) * Pi);
        value_ = active ? (config_.delta - startValue_) * ((1.0f - cosine) * 0.5f) +
                                  startValue_
                        : ((cosine + 1.0f) * 0.5f) * startValue_;
        return value_;
    }

  private:
    SmoothRealConfig config_;
    float scale_ = 1.0f, value_ = 0.0f, startValue_ = 0.0f;
    std::uint32_t start_ = 0u;
    bool rising_ = false;
};

inline float Curve(const std::vector<CurveKey> &keys, float input) {
    if (keys.empty())
        return 0.0f;
    if (input <= keys.front().position)
        return keys.front().value;
    for (std::size_t i = 1; i < keys.size(); ++i) {
        if (input <= keys[i].position) {
            const auto &a = keys[i - 1];
            const auto &b = keys[i];
            const float t = (input - a.position) / (b.position - a.position);
            return t * b.value + (1.0f - t) * a.value;
        }
    }
    return keys.back().value;
}

inline Vec FollowPoint(const GmIso4 &pose, const CameraCommonConfig &config) {
    return pose.translation +
           Rotate(pose.rotation, {config.followedOffset[0], config.followedOffset[1],
                                  config.followedOffset[2]});
}

} // namespace forevervalidator::camera::detail::native
