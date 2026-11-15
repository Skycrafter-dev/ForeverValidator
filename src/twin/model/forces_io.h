// Twin StadiumCar force kernel: per-tick input, per-substep output, enums.
// The state itself is M1's TwinState (tw_state.h, state_fields.inc).
#ifndef TW_MODEL_FORCES_IO_H
#define TW_MODEL_FORCES_IO_H

#include <cstdint>

namespace tw::model {

// FV enum values (scene_vehicle_car_types.h, scene_vehicle_car.h:31-35).
enum EngineState : std::uint8_t { kSteady = 0, kGearShift = 1, kFwdTransition = 2, kRevTransition = 3, kBurnoutHold = 4 };
enum BurnoutPhase : std::uint8_t { kInactive = 0, kTimedSpin = 1, kCircularDrift = 2, kExitFade = 3 };
enum TurboType : std::uint8_t { kTurboOff = 0, kTurboDirect = 1, kTurboRoulette = 2 };

// Per-tick input (decoded once per tick; design.md §3.3 TwinControl).
struct CarTickIn {
    float gas, brake, steer;           // gas / brake are 0 or 1, steer in [-1, 1]
    std::uint32_t tickMs;              // tick time T the car sees (constant over the tick's substeps)
};

// S1..C20 output, consumed by the integrator (S2) and by collision (S3).
struct CarForceOut {
    float force[3];                    // world, total (gravity + drag + car)
    float torque[3];                   // car frame, total (FV's world torque is R * torque)
    float posePos[3];                  // substep-start pose (FV write state): S3 needs it for peerAxisZ
    float poseRot[9];                  // columns basisX, basisY, basisZ
    std::uint32_t arms;                // branch trace (kArm* bits in forces.h)
};

}  // namespace tw::model

#endif
