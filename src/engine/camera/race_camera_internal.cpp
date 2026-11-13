#include "engine/camera/race_camera_internal.h"

#include <algorithm>

namespace forevervalidator::camera::detail {

bool VehicleFrame::AllWheelsWithoutContact() const noexcept {
    return std::none_of(wheelContact.begin(), wheelContact.end(),
                        [](bool contact) { return contact; });
}

bool VehicleFrame::AllWheelsWithoutSurface() const noexcept {
    return std::none_of(wheelHasSurface.begin(), wheelHasSurface.end(),
                        [](bool hasSurface) { return hasSurface; });
}

GmVec3 ToInternal(const Vector3 &value) { return {value.x, value.y, value.z}; }

Vector3 ToPublic(const GmVec3 &value) { return {value.x, value.y, value.z}; }

GmQuat ToInternal(const Quaternion &value) {
    GmQuat result{value.w, value.x, value.y, value.z};
    result.Normalize();
    return result;
}

Quaternion ToPublic(const GmQuat &value) {
    return {value.w, value.x, value.y, value.z};
}

GmIso4 ToInternal(const Transform &value) {
    GmIso4 result;
    GmMat3 rotation;
    rotation.Set(ToInternal(value.rotation));
    result.Set(rotation, ToInternal(value.position));
    return result;
}

Transform ToPublic(const GmIso4 &value) {
    GmQuat rotation;
    rotation.Set(value.rotation);
    rotation.Normalize();
    return {ToPublic(rotation), ToPublic(value.translation)};
}

VehicleFrame ToInternal(const RaceCameraVehicleState &value) {
    VehicleFrame result;
    result.targetId = value.targetId;
    result.timeMs = value.timeMs;
    result.transform = ToInternal(value.transform);
    if (value.resetTransform) result.resetTransform = ToInternal(*value.resetTransform);
    result.linearSpeed = ToInternal(value.linearSpeed);
    result.signedSpeed = value.signedSpeed;
    result.steering = value.steering;
    result.accelerate = value.accelerate;
    result.brake = value.brake;
    result.turbo = value.turbo;
    result.cameraFlightTransition = value.cameraFlightTransition;
    result.burning = value.burning;
    result.gearChanged = value.gearChanged;
    result.isVehicleCar = value.isVehicleCar;
    result.wheelContact = value.wheelContact;
    result.wheelHasSurface = value.wheelHasSurface;
    result.cameraSupportUp = ToInternal(value.cameraSupportUp);
    return result;
}

RaceCameraOutput ToPublic(const CameraFrame &value) {
    RaceCameraOutput result;
    result.transform = ToPublic(value.transform);
    result.lens.fieldOfViewDegrees = value.lens[0];
    result.lens.parameter1 = value.lens[1];
    result.lens.parameter2 = value.lens[2];
    result.lens.nearClipDistance = value.lens[3];
    result.lens.farClipDistance = value.lens[4];
    return result;
}

} // namespace forevervalidator::camera::detail
