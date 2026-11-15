// Twin per-candidate state: the one field list (design.md §3).
//
// Every field the twin carries across ticks is one line here. tw_state.h turns
// the list into the AoS TwinState (448 B), the SoA lanes, the lane view, the
// field table, the hash and the dump; tw_state_golden.h turns it into the
// lossless import/export against the harness's golden words.
//
//   TW_S(name, type, golden, codec, life)          scalar member  s.name
//   TW_A(name, type, count, golden, codec, life)   array member   s.name[i]
//
// type   F32 U32 U16 U8 I8. The list is ordered by size class (all 4-byte
//        fields, then 2-byte, then 1-byte) so the AoS struct has no padding
//        holes; tw_state.h asserts it.
// golden harness field name (tools/twin/twin_fv_state.cpp), "" = none.
// codec  how the golden words map to the field (tw_state_golden.h):
//        DIRECT  word for word (same 32-bit pattern)
//        BOOL    u32 0/1            -> u8 0/1
//        NARROW  u32/i32 small int  -> u8/u16 (range-checked)
//        PEERZ   per wheel vec3     -> its z component (FV reads only .z:
//                                      scene_vehicle_car_feedback.cpp:791-801)
//        SIGN    f32 in {-1, 0, +1} -> i8
//        FSLDS   (dyna.force_scale, dyna.linear_damping) pair -> grounded flag
//        SLOTS   256-slot bitset    -> 64-slot bitset (high words must be 0)
//        FINLO   race.finish_ns     -> lower bound ns (u64 as 2 words)
//        FINPR   race.finish_ns     -> present flag
//        SPAWN   12-word iso        -> index into the world's spawn table
//        NONE    not in the goldens
// life   P persistent across ticks, S one substep (written by collision of
//        substep k, read by the forces of substep k+1, may cross a tick)
//
// Units and frames follow FV: world frame unless noted, quaternion x y z w,
// ticks in ms. Wheel arrays are in board wheel order 0 FL, 1 FR, 2 RR, 3 RL.
// Per-wheel vectors are wheel-major: v[3*w + k].
//
// Deliberately NOT state (tw_state_golden.h exports them from context):
//   ctl.gas/brake/steer       input echo: T1 re-applies the tick's controls
//   ct.wheel_contacts, ct.body_contacts, ct.body_normal_sum
//                             per-tick accumulators, always 0 at tick
//                             boundaries (reset in AfterContacts; checked on
//                             gate1/gate3/scenarios goldens)
//   race.required_cps/_laps   map constants (world pack)
//   w.peer_z .x .y            never read by FV physics
//   dyna.linear_damping       follows from groundedPrev + car pack
//   body.rot, body.inv_inertia_world   derived from quat (harness: within 2e-6)

// ---- 4-byte fields ----------------------------------------------------------
// rigid body (board 10 §2.17)
TW_A(pos,              F32, 3,  "body.pos",                 DIRECT, P)
TW_A(quat,             F32, 4,  "body.quat",                DIRECT, P)
TW_A(vel,              F32, 3,  "body.linvel",              DIRECT, P)
TW_A(angVel,           F32, 3,  "body.angvel",              DIRECT, P)
// FV representation: total local force / forceDivisor (multiply at use, LM:914)
TW_A(prevForce,        F32, 3,  "gd.prev_force",            DIRECT, P)
// wheels (board 10 §2.4-2.5)
TW_A(absorb,           F32, 4,  "w.absorb",                 DIRECT, P)
TW_A(absorbVel,        F32, 4,  "w.absorb_vel",             DIRECT, P)
TW_A(maxRepl,          F32, 4,  "w.max_repl_y",             DIRECT, S)
TW_A(normalSum,        F32, 12, "w.normal_sum",             DIRECT, S)
TW_A(latestPoint,      F32, 12, "w.latest_point",           DIRECT, S)
TW_A(peerAxisZ,        F32, 4,  "w.peer_z",                 PEERZ,  P)
TW_A(peerId,           U32, 4,  "w.peer_id",                DIRECT, P)
// steering, engine (board 10 §2.2-2.3)
TW_S(steerCur,         F32,     "ctl.cur_steer",            DIRECT, P)
TW_S(rpm,              F32,     "eng.rpm",                  DIRECT, P)
TW_S(slipRpmScale,     F32,     "eng.slip_rpm_scale",       DIRECT, P)
TW_S(shiftCooldown,    F32,     "eng.shift_cooldown",       DIRECT, P)
TW_A(localSpeedPrev,   F32, 3,  "gd.local_speed",           DIRECT, P)
// burnout / donut (board 10 §2.11)
TW_S(brnStartMs,       U32,     "gd.burnout_start",         DIRECT, P)
TW_S(brnExitStartMs,   U32,     "gd.burnout_exit_start",    DIRECT, P)
TW_A(brnCenter,        F32, 3,  "gd.burnout_center",        DIRECT, P)
TW_A(brnNormal,        F32, 3,  "gd.burnout_normal",        DIRECT, P)
TW_S(brnBaseRadius,    F32,     "gd.burnout_base_radius",   DIRECT, P)
TW_S(brnTargetRadius,  F32,     "gd.burnout_target_radius", DIRECT, P)
// turbo (board 10 §2.13, 30:R6.2-6.4)
TW_S(trbEndMs,         U32,     "turbo.end",                DIRECT, P)
TW_S(trbScale,         F32,     "turbo.scale",              DIRECT, P)
TW_S(trbSourceId,      U32,     "turbo.source_id",          DIRECT, P)
TW_S(trbPhase,         F32,     "turbo.phase",              DIRECT, P)
TW_S(trbRouletteOriginMs, U32,  "turbo.roulette_origin",    DIRECT, P)
// air control, contact windows, slip memory (board 10 §2.7, §2.12, §2.14)
TW_S(airMemTickMs,     U32,     "air.mem_tick",             DIRECT, P)
TW_A(airMem,           F32, 3,  "air.mem",                  DIRECT, P)
TW_S(latTickMs,        U32,     "ct.lat_slowdown_tick",     DIRECT, P)
TW_S(slipLastTickMs,   U32,     "slip.last_tick",           DIRECT, P)
// race (board 30 §10)
TW_S(tickMs,           U32,     "world.tick_ms",            DIRECT, P)
TW_A(cpSlots,          U32, 2,  "race.cp_slots",            SLOTS,  P)
TW_A(finishLowerNs,    U32, 2,  "race.finish_ns",           FINLO,  P)
TW_S(seed,             U32,     "",                         NONE,   P)

// ---- 2-byte fields ----------------------------------------------------------
TW_S(cpCount,          U16,     "race.cp_count",            NARROW, P)
TW_S(lapCpCount,       U16,     "race.lap_cp_count",        NARROW, P)
TW_S(cpEvents,         U16,     "race.cp_events",           NARROW, P)
TW_S(laps,             U16,     "race.laps",                NARROW, P)
TW_S(finishCount,      U16,     "race.finish_count",        NARROW, P)
TW_S(respawns,         U16,     "race.respawns",            NARROW, P)
TW_S(acceptedEvents,   U16,     "race.accepted_events",     NARROW, P)
TW_S(spawnIndex,       U16,     "race.spawn",               SPAWN,  P)
TW_S(prevSpawnIndex,   U16,     "race.prev_spawn",          SPAWN,  P)

// ---- 1-byte fields ----------------------------------------------------------
TW_A(contact,          U8,  4,  "w.contact",                BOOL,   S)
TW_A(slipping,         U8,  4,  "w.slipping",               BOOL,   P)
TW_A(surface,          U8,  4,  "w.material",               NARROW, S)
TW_S(gear,             U8,      "eng.gear",                 NARROW, P)
TW_S(engineState,      U8,      "gd.engine_state",          NARROW, P)
TW_S(brnPhase,         U8,      "gd.burnout_phase",         NARROW, P)
TW_S(brnDir,           I8,      "gd.burnout_dir",           SIGN,   P)
TW_S(trbType,          U8,      "turbo.type",               NARROW, P)
TW_S(reverse,          U8,      "eng.reverse",              BOOL,   P)
TW_S(freewheel,        U8,      "ctl.freewheel",            BOOL,   P)
TW_S(inWater,          U8,      "ctl.no_ground_friction",   BOOL,   P)
TW_S(speedBlocked,     U8,      "int.speed_blocked",        BOOL,   P)
TW_S(speedBlocked2,    U8,      "int.speed_blocked2",       BOOL,   P)
TW_S(groundedPrev,     U8,      "dyna.force_scale",         FSLDS,  P)
TW_S(airRefresh,       U8,      "air.refresh",              BOOL,   S)
TW_S(bodyContact,      U8,      "ct.body_contact",          BOOL,   S)
TW_S(latContact,       U8,      "ct.lat_slowdown_active",   BOOL,   S)
TW_S(slipActive,       U8,      "slip.active",              BOOL,   P)
TW_S(raceCompleted,    U8,      "race.completed",           BOOL,   P)
TW_S(spawnInitialized, U8,      "race.spawn_initialized",   BOOL,   P)
TW_S(finishPresent,    U8,      "race.finish_ns",           FINPR,  P)
