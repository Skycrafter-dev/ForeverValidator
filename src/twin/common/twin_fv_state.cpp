#include "src/twin/common/twin_fv_state.h"

#include <cstring>
#include <memory>
#include <type_traits>

#include "engine/core/binary32_math.h"
#include "validation/api/physics_sandbox_twin_test_access.h"

namespace twin_fv {
namespace {

using twin::BitsFloat;
using twin::FieldType;
using twin::FloatBits;

constexpr std::uint32_t kWheels = 4u;
constexpr std::uint32_t kCheckpointSlotWords = 8u;  // 256 slots

void PutVec(std::uint32_t *o, const GmVec3 &v) {
    o[0] = FloatBits(v.x);
    o[1] = FloatBits(v.y);
    o[2] = FloatBits(v.z);
}
void GetVec(GmVec3 &v, const std::uint32_t *i) {
    v.x = BitsFloat(i[0]);
    v.y = BitsFloat(i[1]);
    v.z = BitsFloat(i[2]);
}
void PutMat(std::uint32_t *o, const GmMat3 &m) {
    PutVec(o, m.basisX);
    PutVec(o + 3, m.basisY);
    PutVec(o + 6, m.basisZ);
}
void GetMat(GmMat3 &m, const std::uint32_t *i) {
    GetVec(m.basisX, i);
    GetVec(m.basisY, i + 3);
    GetVec(m.basisZ, i + 6);
}
void PutIso(std::uint32_t *o, const GmIso4 &iso) {
    PutMat(o, iso.rotation);
    PutVec(o + 9, iso.translation);
}
void GetIso(GmIso4 &iso, const std::uint32_t *i) {
    GetMat(iso.rotation, i);
    GetVec(iso.translation, i + 9);
}
// Quaternions are written x, y, z, w (FV stores w first).
void PutQuat(std::uint32_t *o, const GmQuat &q) {
    o[0] = FloatBits(q.x);
    o[1] = FloatBits(q.y);
    o[2] = FloatBits(q.z);
    o[3] = FloatBits(q.w);
}
void GetQuat(GmQuat &q, const std::uint32_t *i) {
    q.x = BitsFloat(i[0]);
    q.y = BitsFloat(i[1]);
    q.z = BitsFloat(i[2]);
    q.w = BitsFloat(i[3]);
}

template<typename T>
std::uint32_t ToWord(const T &value) {
    if constexpr (std::is_same_v<T, float>) {
        return FloatBits(value);
    } else if constexpr (std::is_same_v<T, bool>) {
        return value ? 1u : 0u;
    } else if constexpr (std::is_same_v<T, CHmsCorpusId>) {
        return value.Value();
    } else if constexpr (std::is_enum_v<T>) {
        return static_cast<std::uint32_t>(value);
    } else {
        return static_cast<std::uint32_t>(value);
    }
}

template<typename T>
void FromWord(T &value, std::uint32_t word) {
    if constexpr (std::is_same_v<T, float>) {
        value = BitsFloat(word);
    } else if constexpr (std::is_same_v<T, bool>) {
        value = word != 0u;
    } else if constexpr (std::is_same_v<T, CHmsCorpusId>) {
        value = CHmsCorpusId::FromValue(word);
    } else {
        value = static_cast<T>(word);
    }
}

template<typename T>
constexpr FieldType WordType() {
    if constexpr (std::is_same_v<T, float>) {
        return FieldType::F32;
    } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, std::int32_t>) {
        return FieldType::I32;
    } else {
        return FieldType::U32;
    }
}

const CSceneVehicleCar::SSimulationWheel *WheelOrNull(const Clone &c, std::uint32_t w) {
    const auto &wheels = c.runtime.vehicle.car.wheels;
    return w < wheels.size() ? &wheels[w] : nullptr;
}
CSceneVehicleCar::SSimulationWheel *WheelOrNull(Clone &c, std::uint32_t w) {
    auto &wheels = c.runtime.vehicle.car.wheels;
    return w < wheels.size() ? &wheels[w] : nullptr;
}

}  // namespace

// Scalar field: EXPR is an lvalue expression of `c`.
#define TW_SCALAR(NAME, TIER, EXPR)                                            \
    StateField{NAME,                                                           \
               WordType<std::remove_cv_t<std::remove_reference_t<decltype(     \
                       std::declval<Clone &>().EXPR)>>>(),                     \
               1u, Tier::TIER,                                                 \
               [](const Clone &c, std::uint32_t *o) { o[0] = ToWord(c.EXPR); }, \
               [](Clone &c, const std::uint32_t *i) { FromWord(c.EXPR, i[0]); }, \
               #EXPR}
#define TW_VEC(NAME, TIER, EXPR)                                               \
    StateField{NAME, FieldType::F32, 3u, Tier::TIER,                           \
               [](const Clone &c, std::uint32_t *o) { PutVec(o, c.EXPR); },    \
               [](Clone &c, const std::uint32_t *i) { GetVec(c.EXPR, i); },    \
               #EXPR}
#define TW_MAT(NAME, TIER, EXPR)                                               \
    StateField{NAME, FieldType::F32, 9u, Tier::TIER,                           \
               [](const Clone &c, std::uint32_t *o) { PutMat(o, c.EXPR); },    \
               [](Clone &c, const std::uint32_t *i) { GetMat(c.EXPR, i); },    \
               #EXPR}
#define TW_QUAT(NAME, TIER, EXPR)                                              \
    StateField{NAME, FieldType::F32, 4u, Tier::TIER,                           \
               [](const Clone &c, std::uint32_t *o) { PutQuat(o, c.EXPR); },   \
               [](Clone &c, const std::uint32_t *i) { GetQuat(c.EXPR, i); },   \
               #EXPR}
#define TW_ISO(NAME, TIER, EXPR)                                               \
    StateField{NAME, FieldType::F32, 12u, Tier::TIER,                          \
               [](const Clone &c, std::uint32_t *o) { PutIso(o, c.EXPR); },    \
               [](Clone &c, const std::uint32_t *i) { GetIso(c.EXPR, i); },    \
               #EXPR}
// Per-wheel scalar of SSimulationWheel (MEMBER relative to the wheel).
#define TW_WHEEL(NAME, TIER, MEMBER)                                           \
    StateField{NAME,                                                           \
               WordType<std::remove_cv_t<std::remove_reference_t<decltype(     \
                       std::declval<CSceneVehicleCar::SSimulationWheel &>()    \
                               .MEMBER)>>>(),                                  \
               kWheels, Tier::TIER,                                            \
               [](const Clone &c, std::uint32_t *o) {                          \
                   for (std::uint32_t w = 0; w < kWheels; ++w) {               \
                       const auto *wh = WheelOrNull(c, w);                     \
                       o[w] = wh ? ToWord(wh->MEMBER) : 0u;                    \
                   }                                                           \
               },                                                              \
               [](Clone &c, const std::uint32_t *i) {                          \
                   for (std::uint32_t w = 0; w < kWheels; ++w) {               \
                       if (auto *wh = WheelOrNull(c, w)) FromWord(wh->MEMBER, i[w]); \
                   }                                                           \
               },                                                              \
               "wheels[0..3]." #MEMBER}
#define TW_WHEEL_VEC(NAME, TIER, MEMBER)                                       \
    StateField{NAME, FieldType::F32, 3u * kWheels, Tier::TIER,                 \
               [](const Clone &c, std::uint32_t *o) {                          \
                   for (std::uint32_t w = 0; w < kWheels; ++w) {               \
                       const auto *wh = WheelOrNull(c, w);                     \
                       if (wh) PutVec(o + 3 * w, wh->MEMBER);                  \
                       else o[3 * w] = o[3 * w + 1] = o[3 * w + 2] = 0u;       \
                   }                                                           \
               },                                                              \
               [](Clone &c, const std::uint32_t *i) {                          \
                   for (std::uint32_t w = 0; w < kWheels; ++w) {               \
                       if (auto *wh = WheelOrNull(c, w)) GetVec(wh->MEMBER, i + 3 * w); \
                   }                                                           \
               },                                                              \
               "wheels[0..3]." #MEMBER}

const std::vector<StateField> &StateFields() {
    static const std::vector<StateField> fields = [] {
        std::vector<StateField> f;
        // ---- rigid body (CHmsDyna current state) ----
        f.push_back(TW_VEC("body.pos", State, runtime.body.currentState.position));
        f.push_back(TW_QUAT("body.quat", State, runtime.body.currentState.rotationQuat));
        f.push_back(TW_VEC("body.linvel", State, runtime.body.currentState.linearSpeed));
        f.push_back(TW_VEC("body.angvel", State, runtime.body.currentState.angularSpeed));
        f.push_back(TW_MAT("body.rot", Extra, runtime.body.currentState.rotation));
        f.push_back(TW_MAT("body.inv_inertia_world", Extra, runtime.body.currentState.inverseInertiaWorld));
        f.push_back(TW_VEC("body.lin_correction", Extra, runtime.body.currentState.linearCorrectionSpeed));
        f.push_back(TW_SCALAR("body.tweaked_valid", Extra, runtime.body.currentState.tweakedLinearSpeedValid));
        f.push_back(TW_VEC("body.tweaked_linvel", Extra, runtime.body.currentState.tweakedLinearSpeed));
        f.push_back(TW_VEC("body.force", Output, runtime.body.currentState.force));
        f.push_back(TW_VEC("body.torque", Output, runtime.body.currentState.torque));
        // writeState = tick-start pose after a tick (rules' writeFrame);
        // tempState = copy taken at tick start. Board: not state.
        f.push_back(TW_VEC("write.pos", Extra, runtime.body.writeState.position));
        f.push_back(TW_QUAT("write.quat", Extra, runtime.body.writeState.rotationQuat));
        f.push_back(TW_VEC("write.linvel", Extra, runtime.body.writeState.linearSpeed));
        f.push_back(TW_VEC("write.angvel", Extra, runtime.body.writeState.angularSpeed));
        f.push_back(TW_VEC("temp.pos", Extra, runtime.body.tempState.position));
        f.push_back(TW_QUAT("temp.quat", Extra, runtime.body.tempState.rotationQuat));
        f.push_back(TW_VEC("temp.linvel", Extra, runtime.body.tempState.linearSpeed));
        f.push_back(TW_VEC("temp.angvel", Extra, runtime.body.tempState.angularSpeed));
        // fs / lds selected by grounded_prev (board 10 §2.14)
        f.push_back(TW_SCALAR("dyna.force_scale", State, runtime.body.dynaParams.forceScale));
        f.push_back(TW_SCALAR("dyna.linear_damping", State, runtime.body.dynaParams.linearDampingScale));
        f.push_back(TW_SCALAR("dyna.angular_damping", Extra, runtime.body.dynaParams.angularDampingScale));
        f.push_back(TW_SCALAR("dyna.max_step", Extra, runtime.body.dynaParams.maxStepDistance));
        f.push_back(TW_SCALAR("dyna.active", Extra, runtime.body.isDynamicActive));
        f.push_back(TW_SCALAR("solid.feedback_scale", Extra, runtime.body.physicalParameters.vehicleContactFeedbackScale));
        f.push_back(TW_SCALAR("solid.fluid_friction", Extra, runtime.body.physicalParameters.linearFluidFriction));
        // ---- wheels (SRealTimeState unless noted) ----
        f.push_back(TW_WHEEL("w.absorb", State, realTimeState.damperAbsorb));
        f.push_back(TW_WHEEL("w.absorb_vel", State, realTimeState.damperVelocity));
        f.push_back(TW_WHEEL("w.max_repl_y", State, realTimeState.maxReplacementY));
        f.push_back(TW_WHEEL("w.contact", State, realTimeState.contactPresent));
        f.push_back(TW_WHEEL("w.slipping", State, realTimeState.slipping));
        f.push_back(TW_WHEEL("w.material", State, realTimeState.contactMaterial));
        f.push_back(TW_WHEEL_VEC("w.normal_sum", State, realTimeState.accumulatedContactNormal));
        f.push_back(TW_WHEEL("w.normal_count", Extra, realTimeState.contactNormalSampleCount));
        f.push_back(TW_WHEEL_VEC("w.latest_point", State, realTimeState.latestContactPoint));
        f.push_back(TW_WHEEL_VEC("w.peer_z", State, realTimeState.peerZAxisInCarLocal));
        f.push_back(TW_WHEEL("w.peer_id", State, realTimeState.peerCorpusId));
        f.push_back(TW_WHEEL("w.rejected", Extra, realTimeState.rejectedNormalContact));
        f.push_back(TW_WHEEL_VEC("w.rejected_point", Extra, realTimeState.rejectedNormalContactPoint));
        f.push_back(TW_WHEEL("w.angular_speed", Output, realTimeState.wheelAngularSpeed));
        f.push_back(TW_WHEEL("w.phys_contact", Extra, currentPhysicsState.contactPresent));
        f.push_back(TW_WHEEL("w.phys_slipping", Extra, currentPhysicsState.slipping));
        f.push_back(TW_WHEEL("w.phys_material", Extra, currentPhysicsState.contactMaterial));
        f.push_back(TW_WHEEL("w.phys_absorb", Extra, currentPhysicsState.damperAbsorb));
        f.push_back(StateField{
                "w.surface_pose", FieldType::F32, 12u * kWheels, Tier::Extra,
                [](const Clone &c, std::uint32_t *o) {
                    for (std::uint32_t w = 0; w < kWheels; ++w) {
                        const auto *wh = WheelOrNull(c, w);
                        if (wh) PutIso(o + 12 * w, wh->surfaceHandler.CurrentPose());
                        else std::memset(o + 12 * w, 0, 12 * sizeof(std::uint32_t));
                    }
                },
                [](Clone &c, const std::uint32_t *i) {
                    for (std::uint32_t w = 0; w < kWheels; ++w) {
                        if (auto *wh = WheelOrNull(c, w)) {
                            GmIso4 iso{};
                            GetIso(iso, i + 12 * w);
                            wh->surfaceHandler.SetCurrentPose(iso);
                        }
                    }
                },
                "wheels[0..3].surfaceHandler.CurrentPose()"});
        f.push_back(StateField{
                "w.surface_moved", FieldType::U32, kWheels, Tier::Extra,
                [](const Clone &c, std::uint32_t *o) {
                    const auto &v = c.runtime.vehicle.wheelSurfaces.movedByUpdateSurface;
                    for (std::uint32_t w = 0; w < kWheels; ++w) o[w] = w < v.size() && v[w] ? 1u : 0u;
                },
                [](Clone &c, const std::uint32_t *i) {
                    auto &v = c.runtime.vehicle.wheelSurfaces.movedByUpdateSurface;
                    for (std::uint32_t w = 0; w < kWheels && w < v.size(); ++w) v[w] = i[w] != 0u;
                },
                "vehicle.wheelSurfaces.movedByUpdateSurface"});
        // ---- controls / integration flags ----
        f.push_back(TW_SCALAR("ctl.gas", State, runtime.vehicle.car.controls.lowSpeedGateA));
        f.push_back(TW_SCALAR("ctl.brake", State, runtime.vehicle.car.controls.lowSpeedGateB));
        f.push_back(TW_SCALAR("ctl.steer", State, runtime.vehicle.car.controls.steeringControl));
        f.push_back(TW_SCALAR("ctl.cur_steer", State, runtime.vehicle.car.controls.currentSteering));
        f.push_back(TW_SCALAR("ctl.freewheel", State, runtime.vehicle.car.controls.forcedLowSpeedFriction));
        f.push_back(TW_SCALAR("ctl.no_ground_friction", State, runtime.vehicle.car.controls.noGroundFrictionGuard));
        f.push_back(TW_SCALAR("ctl.special_gate", Extra, runtime.vehicle.car.controls.specialContactResponseGate));
        f.push_back(TW_SCALAR("int.speed_blocked", State, runtime.vehicle.car.integration.speedBlocked));
        f.push_back(TW_SCALAR("int.speed_blocked2", State, runtime.vehicle.car.integration.speedBlockedSecondary));
        // ---- engine ----
        f.push_back(TW_SCALAR("eng.rpm", State, runtime.vehicle.car.engine.engineInputMemory));
        f.push_back(TW_SCALAR("eng.slip_rpm_scale", State, runtime.vehicle.car.engine.slipRpmScale));
        f.push_back(TW_SCALAR("eng.shift_cooldown", State, runtime.vehicle.car.engine.shiftCooldown));
        f.push_back(TW_SCALAR("eng.reverse", State, runtime.vehicle.car.engine.useLowSpeedGateB));
        f.push_back(TW_SCALAR("eng.gear", State, runtime.vehicle.car.engine.gearIndex));
        f.push_back(TW_SCALAR("eng.target_input", Extra, runtime.vehicle.car.engine.targetTransmissionInput));
        f.push_back(TW_SCALAR("eng.input_max", Extra, runtime.vehicle.car.engine.engineInputMax));
        f.push_back(TW_SCALAR("eng.low_speed_fb_force", Extra, runtime.vehicle.car.engine.lowSpeedFeedbackForce));
        // ---- geared drive (Model 6) ----
        f.push_back(TW_SCALAR("gd.engine_state", State, runtime.vehicle.car.gearedDrive.engineState));
        f.push_back(TW_SCALAR("gd.burnout_phase", State, runtime.vehicle.car.gearedDrive.burnoutPhase));
        f.push_back(TW_VEC("gd.local_speed", State, runtime.vehicle.car.gearedDrive.localSpeed));
        f.push_back(TW_VEC("gd.prev_force", State, runtime.vehicle.car.gearedDrive.scaledCurrentForce));
        f.push_back(TW_VEC("gd.burnout_center", State, runtime.vehicle.car.gearedDrive.burnoutCenter));
        f.push_back(TW_VEC("gd.burnout_normal", State, runtime.vehicle.car.gearedDrive.burnoutContactNormal));
        f.push_back(TW_SCALAR("gd.burnout_base_radius", State, runtime.vehicle.car.gearedDrive.burnoutBaseRadius));
        f.push_back(TW_SCALAR("gd.burnout_target_radius", State, runtime.vehicle.car.gearedDrive.burnoutTargetRadius));
        f.push_back(TW_SCALAR("gd.burnout_dir", State, runtime.vehicle.car.gearedDrive.burnoutDirection));
        f.push_back(TW_SCALAR("gd.burnout_start", State, runtime.vehicle.car.gearedDrive.burnoutStartTick));
        f.push_back(TW_SCALAR("gd.burnout_exit_start", State, runtime.vehicle.car.gearedDrive.burnoutExitStartTick));
        f.push_back(TW_SCALAR("gd.steer_slowdown", Extra, runtime.vehicle.car.gearedDrive.activeSteerSlowDownScale));
        f.push_back(TW_SCALAR("gd.drive_inhibited", Extra, runtime.vehicle.car.gearedDrive.wheelDriveSpeedInhibited));
        f.push_back(TW_SCALAR("gd.input_window_exceeded", Extra, runtime.vehicle.car.gearedDrive.inputWindowExceeded));
        f.push_back(TW_SCALAR("gd.wheel_span", Extra, runtime.vehicle.car.gearedDrive.wheelLongitudinalSpan));
        f.push_back(TW_SCALAR("gd.speed_override", Extra, runtime.vehicle.car.gearedDrive.wheelSpeedOverrideActive));
        f.push_back(TW_ISO("gd.frame_iso", Extra, runtime.vehicle.car.gearedDrive.frameIso));
        f.push_back(TW_SCALAR("gd.shift_dir", Output, runtime.vehicle.car.gearedDrive.shiftDirection));
        // ---- turbo ----
        f.push_back(TW_SCALAR("turbo.type", State, runtime.vehicle.car.turbo.type));
        f.push_back(TW_SCALAR("turbo.end", State, runtime.vehicle.car.turbo.endTick));
        f.push_back(TW_SCALAR("turbo.scale", State, runtime.vehicle.car.turbo.impulseScale));
        f.push_back(TW_SCALAR("turbo.roulette_origin", State, runtime.vehicle.car.turbo.rouletteTickOrigin));
        f.push_back(TW_SCALAR("turbo.source_id", State, runtime.vehicle.car.turbo.sourceCorpusId));
        f.push_back(TW_SCALAR("turbo.phase", State, runtime.vehicle.car.turbo.type2Phase));
        f.push_back(TW_SCALAR("turbo.start", Extra, runtime.vehicle.car.turbo.startTick));
        f.push_back(TW_SCALAR("turbo.progress", Output, runtime.vehicle.car.turbo.progressRatio));
        // ---- air control ----
        f.push_back(TW_SCALAR("air.refresh", State, runtime.vehicle.car.airControl.refreshMemory));
        f.push_back(TW_SCALAR("air.mem_tick", State, runtime.vehicle.car.airControl.memoryTick));
        f.push_back(TW_VEC("air.mem", State, runtime.vehicle.car.airControl.memoryAngular));
        // ---- contacts ----
        f.push_back(TW_SCALAR("ct.body_contact", State, runtime.vehicle.car.contacts.bodyContactPresent));
        f.push_back(TW_SCALAR("ct.lat_slowdown_active", State, runtime.vehicle.car.contacts.lateralSlowDownContactActive));
        f.push_back(TW_SCALAR("ct.lat_slowdown_tick", State, runtime.vehicle.car.contacts.lateralSlowDownLastTick));
        f.push_back(TW_SCALAR("ct.wheel_contacts", State, runtime.vehicle.car.contacts.wheelContactCount));
        f.push_back(TW_SCALAR("ct.body_contacts", State, runtime.vehicle.car.contacts.bodyContactCount));
        f.push_back(TW_VEC("ct.body_normal_sum", State, runtime.vehicle.car.contacts.bodyContactNormalSum));
        f.push_back(TW_VEC("ct.body_point_sum", Extra, runtime.vehicle.car.contacts.bodyContactPointSum));
        f.push_back(TW_SCALAR("ct.special_cooldown", Extra, runtime.vehicle.car.contacts.specialContactImpulseCooldownUntil));
        f.push_back(TW_SCALAR("ct.last_body_material", Output, runtime.vehicle.car.contacts.lastBodyContactMaterial));
        f.push_back(TW_SCALAR("ct.last_wheel_material", Output, runtime.vehicle.car.contacts.lastWheelContactMaterial));
        // ---- slip memory ----
        f.push_back(TW_SCALAR("slip.active", State, runtime.vehicle.car.slipMemory.active));
        f.push_back(TW_SCALAR("slip.last_tick", State, runtime.vehicle.car.slipMemory.lastTick));
        f.push_back(TW_SCALAR("slip.start_tick", Extra, runtime.vehicle.car.slipMemory.startTick));
        f.push_back(TW_SCALAR("slip.elapsed", Extra, runtime.vehicle.car.slipMemory.elapsedTicks));
        f.push_back(TW_SCALAR("slip.steer_mem_tick", Extra, runtime.vehicle.car.slipMemory.steeringMemoryTick));
        f.push_back(TW_SCALAR("slip.steer_mem_slip", Extra, runtime.vehicle.car.slipMemory.steeringMemorySlip));
        f.push_back(TW_SCALAR("veh.last_forces_tick", Extra, runtime.vehicle.car.lastComputeForcesTick));
        f.push_back(TW_SCALAR("veh.speed_cap", Extra, runtime.vehicle.car.linearSpeedCap));
        // ---- water ----
        f.push_back(TW_SCALAR("water.splash_pending", Extra, runtime.vehicle.car.vehicle.water.splashPending));
        f.push_back(TW_VEC("water.splash_speed", Extra, runtime.vehicle.car.vehicle.water.splashLocalSpeed));
        // ---- per-tick rules snapshot (AfterContacts, board 10 §3.3) ----
        f.push_back(TW_SCALAR("fr.forward_speed", Output, runtime.vehicle.car.frameHistory.physicsCurrent.forwardSpeed));
        f.push_back(TW_SCALAR("fr.side_speed", Output, runtime.vehicle.car.frameHistory.physicsCurrent.sideSpeed));
        f.push_back(TW_SCALAR("fr.has_wheel_contact", Output, runtime.vehicle.car.frameHistory.physicsCurrent.hasWheelContact));
        f.push_back(TW_SCALAR("fr.has_body_contact", Output, runtime.vehicle.car.frameHistory.physicsCurrent.hasBodyContact));
        f.push_back(TW_SCALAR("fr.body_v_angle", Output, runtime.vehicle.car.frameHistory.physicsCurrent.bodyContactVerticalAngle));
        f.push_back(TW_SCALAR("fr.body_h_angle", Output, runtime.vehicle.car.frameHistory.physicsCurrent.bodyContactHorizontalAngle));
        f.push_back(TW_SCALAR("fr.body_z_positive", Output, runtime.vehicle.car.frameHistory.physicsCurrent.bodyContactZPositive));
        f.push_back(TW_SCALAR("fr.no_ground_friction", Output, runtime.vehicle.car.frameHistory.physicsCurrent.noGroundFrictionGuard));
        f.push_back(TW_VEC("fr.local_linvel", Output, runtime.vehicle.car.frameHistory.physicsCurrent.localLinearSpeed));
        // ---- world clock ----
        f.push_back(TW_SCALAR("world.tick_ms", Race, runtime.world.tickTimeMs));
        f.push_back(TW_SCALAR("world.first_step", Extra, runtime.firstStep));
        // ---- race (board 30 §10) ----
        f.push_back(TW_SCALAR("race.cp_count", Race, race.progress.checkpointCount));
        f.push_back(TW_SCALAR("race.lap_cp_count", Race, race.progress.currentLapCheckpointCount));
        f.push_back(TW_SCALAR("race.cp_events", Race, race.progress.totalCheckpointEventCount));
        f.push_back(TW_SCALAR("race.laps", Race, race.progress.completedLapCount));
        f.push_back(TW_SCALAR("race.completed", Race, race.progress.raceCompleted));
        f.push_back(TW_SCALAR("race.finish_count", Race, race.progress.finishCount));
        f.push_back(TW_SCALAR("race.required_cps", Race, race.progress.requiredCheckpointCount));
        f.push_back(TW_SCALAR("race.required_laps", Race, race.progress.requiredLapCount));
        f.push_back(TW_SCALAR("race.freewheel_clears", Extra, race.progress.freewheelClearCount));
        f.push_back(TW_SCALAR("race.prepared_events", Extra, race.progress.preparedEventCount));
        f.push_back(TW_SCALAR("race.last_prepare_ms", Extra, race.progress.lastPrepareTimeMs));
        f.push_back(TW_SCALAR("race.prepared_event_ms", Extra, race.preparedEventTimeMs));
        f.push_back(StateField{
                "race.cp_slots", FieldType::U32, kCheckpointSlotWords, Tier::Race,
                [](const Clone &c, std::uint32_t *o) {
                    std::memset(o, 0, kCheckpointSlotWords * sizeof(std::uint32_t));
                    const auto &slots = c.race.checkpointSlotsPassed;
                    for (std::size_t s = 0; s < slots.size() && s < 32u * kCheckpointSlotWords; ++s) {
                        if (slots[s] != 0u) o[s / 32u] |= 1u << (s % 32u);
                    }
                },
                [](Clone &c, const std::uint32_t *i) {
                    auto &slots = c.race.checkpointSlotsPassed;
                    for (std::size_t s = 0; s < slots.size() && s < 32u * kCheckpointSlotWords; ++s) {
                        slots[s] = (i[s / 32u] >> (s % 32u)) & 1u;
                    }
                },
                "race.checkpointSlotsPassed (bitset, slot count = race.required_cps + 1)"});
        f.push_back(StateField{
                "race.spawn", FieldType::F32, 12u, Tier::Race,
                [](const Clone &c, std::uint32_t *o) { PutIso(o, c.race.player.Info().CurrentSpawnLocation()); },
                [](Clone &c, const std::uint32_t *i) {
                    CTrackManiaPlayer::RuntimeClone p = c.race.player.CaptureRuntimeClone();
                    GetIso(p.currentSpawnLocation, i);
                    c.race.player.RestoreRuntimeClone(p);
                },
                "race.player currentSpawnLocation"});
        f.push_back(StateField{
                "race.prev_spawn", FieldType::F32, 12u, Tier::Race,
                [](const Clone &c, std::uint32_t *o) { PutIso(o, c.race.player.CaptureRuntimeClone().previousSpawnLocation); },
                [](Clone &c, const std::uint32_t *i) {
                    CTrackManiaPlayer::RuntimeClone p = c.race.player.CaptureRuntimeClone();
                    GetIso(p.previousSpawnLocation, i);
                    c.race.player.RestoreRuntimeClone(p);
                },
                "race.player previousSpawnLocation"});
        f.push_back(StateField{
                "race.event_prepared", FieldType::U32, 1u, Tier::Extra,
                [](const Clone &c, std::uint32_t *o) { o[0] = c.race.player.CaptureRuntimeClone().eventPrepared ? 1u : 0u; },
                [](Clone &c, const std::uint32_t *i) {
                    CTrackManiaPlayer::RuntimeClone p = c.race.player.CaptureRuntimeClone();
                    p.eventPrepared = i[0] != 0u;
                    c.race.player.RestoreRuntimeClone(p);
                },
                "race.player eventPrepared"});
        f.push_back(TW_SCALAR("race.spawn_initialized", Race, race.currentSpawnLocationInitialized));
        f.push_back(StateField{
                "race.respawns", FieldType::U32, 1u, Tier::Race,
                [](const Clone &c, std::uint32_t *o) { o[0] = c.incrementalRespawnCount; },
                [](Clone &c, const std::uint32_t *i) { c.incrementalRespawnCount = i[0]; },
                "incrementalRespawnCount"});
        f.push_back(StateField{
                "race.accepted_events", FieldType::U32, 1u, Tier::Race,
                [](const Clone &c, std::uint32_t *o) { o[0] = static_cast<std::uint32_t>(c.race.acceptedCheckpointEventCount); },
                [](Clone &c, const std::uint32_t *i) { c.race.acceptedCheckpointEventCount = i[0]; },
                "race.acceptedCheckpointEventCount"});
        f.push_back(StateField{
                "race.finish_ns", FieldType::U32, 7u, Tier::Race,
                [](const Clone &c, std::uint32_t *o) {
                    const auto &ft = c.runtime.finishTime;
                    o[0] = ft.has_value() ? 1u : 0u;
                    const std::uint64_t v[3] = {ft ? ft->lowerBoundNs : 0u, ft ? ft->upperBoundNs : 0u,
                                                ft ? ft->estimatedNs : 0u};
                    for (int k = 0; k < 3; ++k) {
                        o[1 + 2 * k] = static_cast<std::uint32_t>(v[k]);
                        o[2 + 2 * k] = static_cast<std::uint32_t>(v[k] >> 32);
                    }
                },
                [](Clone &c, const std::uint32_t *i) {
                    if (i[0] == 0u) {
                        c.runtime.finishTime.reset();
                        return;
                    }
                    forevervalidator::FinishTimeEstimate ft;
                    ft.lowerBoundNs = i[1] | (static_cast<std::uint64_t>(i[2]) << 32);
                    ft.upperBoundNs = i[3] | (static_cast<std::uint64_t>(i[4]) << 32);
                    ft.estimatedNs = i[5] | (static_cast<std::uint64_t>(i[6]) << 32);
                    c.runtime.finishTime = ft;
                },
                "runtime.finishTime {present, lower lo/hi, upper lo/hi, estimated lo/hi} (absolute ns)"});
        return f;
    }();
    return fields;
}

#undef TW_SCALAR
#undef TW_VEC
#undef TW_MAT
#undef TW_QUAT
#undef TW_ISO
#undef TW_WHEEL
#undef TW_WHEEL_VEC

std::uint32_t StateWordCount() {
    static const std::uint32_t words = [] {
        std::uint32_t n = 0u;
        for (const StateField &f : StateFields()) n += f.count;
        return n;
    }();
    return words;
}

void ExtractState(const Clone &clone, std::uint32_t *out) {
    for (const StateField &f : StateFields()) {
        f.get(clone, out);
        out += f.count;
    }
}

void ApplyState(Clone &clone, const std::uint32_t *words, std::uint32_t tierMask) {
    for (const StateField &f : StateFields()) {
        if ((tierMask & TierBit(f.tier)) != 0u && f.set != nullptr) {
            f.set(clone, words);
        }
        words += f.count;
    }
}

void ApplyState(Clone &clone, const std::uint32_t *words, std::uint32_t tierMask,
                const std::vector<std::string> &with,
                const std::vector<std::string> &without) {
    auto listed = [](const std::vector<std::string> &names, const char *name) {
        for (const std::string &n : names) {
            if (n == name || (!n.empty() && n.back() == '*' && std::string(name).rfind(n.substr(0, n.size() - 1), 0) == 0)) {
                return true;
            }
        }
        return false;
    };
    for (const StateField &f : StateFields()) {
        const bool byTier = (tierMask & TierBit(f.tier)) != 0u;
        if (((byTier || listed(with, f.name)) && !listed(without, f.name)) && f.set != nullptr) {
            f.set(clone, words);
        }
        words += f.count;
    }
}

namespace {

using forevervalidator::experimental::PhysicsSandboxState;
using forevervalidator::experimental::PhysicsSandboxStateView;
using forevervalidator::experimental::twin_test::PhysicsSandboxTwinTestAccess;

bool SameBits(float a, float b) { return FloatBits(a) == FloatBits(b); }

bool Consistent(const PhysicsSandboxStateView &view, const Clone &clone) {
    const auto &s = clone.runtime.body.currentState;
    return SameBits(view.car.position.x, s.position.x) &&
           SameBits(view.car.position.y, s.position.y) &&
           SameBits(view.car.position.z, s.position.z) &&
           SameBits(view.car.rotationW, s.rotationQuat.w) &&
           view.checkpointsCollected == clone.race.progress.checkpointCount;
}

}  // namespace

const Clone *PeekClone(const PhysicsSandboxState &state, std::string *error) {
    const std::shared_ptr<const Clone> clone = PhysicsSandboxTwinTestAccess::RuntimeClone(state);
    if (clone == nullptr) {
        if (error) *error = "sandbox state has no runtime clone";
        return nullptr;
    }
    // The state keeps the clone alive; the pointer stays valid while `state` lives.
    if (!Consistent(state.View(), *clone)) {
        if (error) *error = "sandbox state view and runtime clone disagree";
        return nullptr;
    }
    return clone.get();
}

bool ReplaceClone(PhysicsSandboxState &state, const Clone &clone, std::string *error) {
    if (!PhysicsSandboxTwinTestAccess::ReplaceRuntimeClone(state, clone)) {
        if (error) *error = "cannot replace the sandbox state's runtime clone";
        return false;
    }
    return true;
}

std::uint32_t SubstepCount(const Clone &clone, float *metric) {
    const auto &s = clone.runtime.body.currentState;
    const float dt = static_cast<float>(static_cast<std::int32_t>(10)) * 0.001f;
    const float maxStep = clone.runtime.body.dynaParams.maxStepDistance;
    const float speedSum = s.linearSpeed.PhysicsStep2LinearSpeedLength() +
                           s.angularSpeed.PhysicsStep2AngularSpeedLength();
    const float scaled = (speedSum * dt) / maxStep;
    if (metric) *metric = scaled;
    std::uint32_t n = Binary32::TruncateToUint32Modulo(scaled) + 1u;
    return n > 1000u ? 1000u : n;
}

}  // namespace twin_fv
