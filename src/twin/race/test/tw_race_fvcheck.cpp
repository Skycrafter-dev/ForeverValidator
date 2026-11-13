// tw_race_fvcheck: the twin race layer against FV, tick by tick, on real replays.
//
//   tw_race_fvcheck PACKS CAR_JSON [--golden F.tgold]... [--replay R.Gbx]... [--max-ticks N] [--from-golden]
//                   [--respawn-every MS]   (synthetic respawn presses; goldens are then not compared)
//
// Runs each replay through FV's Reference backend (PhysicsSandbox) and checks:
//  1. controls: TwBuildControls == FV's BuildReplayControlPlan on every tick
//     (gas, brake, steer bits, race spawn, begin race, race-start reset, respawns);
//  2. trigger rules: FV's trigger contacts are recorded per substep (hooks
//     below), fed to TwRaceTickBegin + TwRaceOnTriggerHits on the twin state
//     imported from FV's pre-tick state; the twin's race fields, events and
//     accepted journal must equal FV's after the tick (every tick);
//  3. respawn and race-start reset: the twin's TwRespawnPlace / TwVehicleReset
//     state equals FV's state captured right after FV's Respawn / race-start
//     PrepareStep (every state field);
//  4. goldens: in.* of every golden record of a run replay equals the stream;
//     records whose pre-state equals the run's are re-run from the golden
//     pre-state and compared with golden post race fields and ev.*.
// --from-golden runs the replays named by the golden files (race-event replays
// first); --replay adds others.
//
// Hooks (GNU ld --wrap, FV unmodified; CMakeLists.txt): trigger contacts at
// CHmsCorpus::RunAbsorbContactForCollisionResponse, only inside
// ReplayPhysicsWorld::Step (the finish-bracket probes rerun the response and are
// excluded); the trigger table at ReplayStaticCorpusCollection::InstallRaceTriggerHooks;
// substeps at SZone::DetectCollisionsCorpus on the car corpus; the respawn and
// race-start states at ReplayVehicleSimulation::Respawn / PrepareStep.
#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/native.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/core/gm_types.h"
#include "engine/game/game_ctn_block.h"
#include "engine/game/game_ctn_block_info.h"
#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_item.h"
#include "engine/scene/replay_static_scene_corpuses.h"
#include "engine/scene/scene_mobil.h"
#include "format/replay/replay_input_timeline.h"
#include "simulation/backends/cuda/cuda_scene_layout.h"
#include "simulation/control/replay_control_plan.h"
#include "simulation/runtime/replay_finish_time_estimator.h"
#include "simulation/runtime/replay_physics_world.h"
#include "simulation/runtime/replay_simulation_session.h"
#include "simulation/runtime/replay_vehicle_simulation.h"
#include "src/twin/common/twin_fv_state.h"
#include "src/twin/common/twin_golden.h"
#include "tw_compile_car.h"
#include "tw_controls.h"
#include "tw_race.h"
#include "tw_state_golden.h"

namespace fv = forevervalidator;
namespace fvx = forevervalidator::experimental;
using namespace tw::race;
using tw::model::TwinState;

namespace {

// ---------------------------------------------------------------------------
// Hook state
struct FvTrigger {
    const CGameCtnBlock *block;
    TwTrigger t;
    bool hasSpawn;
    GmIso4 spawn;
};
struct Hit {
    std::uint32_t substep, trigger;
};
struct VehicleCapture {
    bool valid = false;
    ReplayVehicleSimulation::RuntimeClone vehicle;
    ReplayVehicleBody::RuntimeClone body;
};

std::vector<FvTrigger> g_triggers;
// The same table from FV's flat scene export (CudaSceneActor), as M2 builds it.
struct ActorTrigger {
    TwTrigger t;
    bool hasSpawn;
    GmIso4 spawn;
};
std::vector<ActorTrigger> g_actorTriggers;
std::uint32_t g_actorCheckpointsOutsideTriggers = 0u;
std::unordered_map<const CGameCtnBlock *, std::uint32_t> g_triggerOfBlock;
std::uint32_t g_installs = 0u;
bool g_inStep = false;
bool g_recordNextStep = false;  // only the tick's own World::Step (the finish estimator reruns the tick)
std::uint32_t g_substep = 0u;
std::vector<Hit> g_hits;
std::uint32_t g_unknownTriggerHits = 0u;
VehicleCapture g_afterRespawn;
std::uint32_t g_respawnCalls = 0u;
bool g_afterRespawnLocked = false;  // set once the tick's own World::Step ran
VehicleCapture g_beforeReset, g_afterReset;
std::uint32_t g_resetTickMs = 0u;

void BuildTriggerTable(ReplayStaticCorpusCollection *c) {
    g_triggers.clear();
    g_triggerOfBlock.clear();
    for (std::uint32_t i = 0; i < c->Count(); ++i) {
        const ReplayStaticCorpus *corpus = c->CorpusAt(i);
        if (corpus == nullptr || !corpus->IsRaceTriggerMobil()) continue;
        const CSceneMobil *mobil = corpus->Item().SceneMobilOwner();
        const CGameCtnBlock *block = mobil != nullptr ? mobil->OwningBlock() : nullptr;
        const CGameCtnBlockInfo *info = block != nullptr ? block->BlockInfoRef() : nullptr;
        FvTrigger t{};
        t.block = block;
        t.t.role = info != nullptr ? static_cast<std::uint8_t>(info->RaceRole()) : static_cast<std::uint8_t>(kRoleNone);
        t.t.currentTransform = info != nullptr && info->RespawnUsesCurrentTransform() ? 1u : 0u;
        const std::optional<std::uint32_t> slot = block ? c->CheckpointSlotForBlock(block) : std::nullopt;
        t.t.slot = slot.has_value() ? static_cast<std::uint16_t>(*slot) : kNoSlot;
        const GmIso4 *spawn = block ? c->SpawnIsoForCheckpointBlock(block) : nullptr;
        t.hasSpawn = spawn != nullptr;
        if (spawn) t.spawn = *spawn;
        t.t.spawn = kNoSpawn;  // resolved against the spawn table later
        if (block) g_triggerOfBlock[block] = static_cast<std::uint32_t>(g_triggers.size());
        g_triggers.push_back(t);
    }
    ++g_installs;
}

}  // namespace

extern "C" {
bool __real__ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection(ReplaySimulationSession *,
                                                                                          StaticSceneModelCollection *);
bool __wrap__ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection(ReplaySimulationSession *self,
                                                                                          StaticSceneModelCollection *models) {
    fv::simulation::CudaHostScene scene;
    if (fv::simulation::BuildCudaHostScene(*models, &scene) == fv::simulation::CudaSceneBuildResult::Success) {
        g_actorTriggers.clear();
        g_actorCheckpointsOutsideTriggers = 0u;
        for (const fv::simulation::CudaSceneActor &a : scene.actors) {
            const bool trigger = a.purpose == static_cast<std::uint32_t>(StaticScenePurpose::CheckpointTrigger);
            if (!trigger) {
                if (a.hasCheckpoint) ++g_actorCheckpointsOutsideTriggers;
                continue;
            }
            ActorTrigger t{};
            t.t.role = a.hasCheckpoint ? static_cast<std::uint8_t>(a.checkpointRole) : static_cast<std::uint8_t>(kRoleNone);
            t.t.currentTransform = a.respawnUsesCurrentTransform ? 1u : 0u;
            t.t.slot = a.checkpointSlot == UINT32_MAX ? kNoSlot : static_cast<std::uint16_t>(a.checkpointSlot);
            t.t.spawn = kNoSpawn;
            t.hasSpawn = a.hasCheckpointSpawn;
            t.spawn = a.checkpointSpawn;
            g_actorTriggers.push_back(t);
        }
    }
    return __real__ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection(self, models);
}

int __real__ZNK10CHmsCorpus36RunAbsorbContactForCollisionResponseEP19CHmsPhysicalContact(const CHmsCorpus *,
                                                                                          CHmsPhysicalContact *);
int __wrap__ZNK10CHmsCorpus36RunAbsorbContactForCollisionResponseEP19CHmsPhysicalContact(const CHmsCorpus *self,
                                                                                          CHmsPhysicalContact *contact) {
    if (g_inStep && self != nullptr && self->Item() != nullptr) {
        const CSceneMobil *mobil = self->Item()->SceneMobilOwner();
        const CGameCtnBlock *block = mobil != nullptr ? mobil->OwningBlock() : nullptr;
        if (block != nullptr && block->BlockInfoRef() != nullptr) {
            auto it = g_triggerOfBlock.find(block);
            if (it != g_triggerOfBlock.end())
                g_hits.push_back({g_substep, it->second});
            else
                ++g_unknownTriggerHits;
        }
    }
    return __real__ZNK10CHmsCorpus36RunAbsorbContactForCollisionResponseEP19CHmsPhysicalContact(self, contact);
}

std::uint32_t __real__ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver(
    ReplayStaticCorpusCollection *, ReplayCheckpointContactObserver &);
std::uint32_t __wrap__ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver(
    ReplayStaticCorpusCollection *self, ReplayCheckpointContactObserver &observer) {
    const std::uint32_t r =
        __real__ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver(self, observer);
    BuildTriggerTable(self);
    return r;
}

void __real__ZN18ReplayPhysicsWorld4StepEv(ReplayPhysicsWorld *);
void __wrap__ZN18ReplayPhysicsWorld4StepEv(ReplayPhysicsWorld *self) {
    g_inStep = g_recordNextStep;
    g_recordNextStep = false;
    g_substep = 0u;
    __real__ZN18ReplayPhysicsWorld4StepEv(self);
    g_inStep = false;
    g_afterRespawnLocked = true;
}

void __real__ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus(
    CHmsCollisionManager::SZone *, CHmsCollisionBuffer &, CHmsCorpus *);
void __wrap__ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus(
    CHmsCollisionManager::SZone *self, CHmsCollisionBuffer &buffer, CHmsCorpus *corpus) {
    if (g_inStep && corpus != nullptr && corpus->Dynamics() != nullptr) ++g_substep;
    __real__ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus(self, buffer,
                                                                                                         corpus);
}

bool __real__ZN23ReplayVehicleSimulation7RespawnER17ReplayVehicleBody(ReplayVehicleSimulation *, ReplayVehicleBody &);
bool __wrap__ZN23ReplayVehicleSimulation7RespawnER17ReplayVehicleBody(ReplayVehicleSimulation *self,
                                                                       ReplayVehicleBody &body) {
    const bool r = __real__ZN23ReplayVehicleSimulation7RespawnER17ReplayVehicleBody(self, body);
    if (r && !g_afterRespawnLocked) {
        g_afterRespawn.vehicle = self->CaptureRuntimeClone();
        g_afterRespawn.body = body.CaptureRuntimeClone();
        g_afterRespawn.valid = true;
        ++g_respawnCalls;
    }
    return r;
}

void __real__ZN23ReplayVehicleSimulation11PrepareStepERK17ReplayControlTickR17ReplayVehicleBody(
    ReplayVehicleSimulation *, const ReplayControlTick &, ReplayVehicleBody &);
void __wrap__ZN23ReplayVehicleSimulation11PrepareStepERK17ReplayControlTickR17ReplayVehicleBody(
    ReplayVehicleSimulation *self, const ReplayControlTick &tick, ReplayVehicleBody &body) {
    const bool reset = tick.actions.resetAtRaceStart;
    if (reset) {
        g_beforeReset.vehicle = self->CaptureRuntimeClone();
        g_beforeReset.body = body.CaptureRuntimeClone();
        g_beforeReset.valid = true;
    }
    __real__ZN23ReplayVehicleSimulation11PrepareStepERK17ReplayControlTickR17ReplayVehicleBody(self, tick, body);
    if (reset) {
        g_afterReset.vehicle = self->CaptureRuntimeClone();
        g_afterReset.body = body.CaptureRuntimeClone();
        g_afterReset.valid = true;
        g_resetTickMs = tick.timeMs;
    }
}
}  // extern "C"

namespace {

// ---------------------------------------------------------------------------
struct Stats {
    std::map<std::string, std::size_t> fail;  // category -> count
    std::map<std::string, std::size_t> ok;
    std::map<std::string, std::string> firstFail;
    void Check(bool cond, const std::string &cat, const std::string &detail = {}) {
        if (cond) {
            ok[cat]++;
        } else {
            if (!fail[cat]++) firstFail[cat] = detail;
        }
    }
};
Stats g_stats;

std::uint32_t Bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
void PutIsoWords(std::uint32_t *o, const GmIso4 &iso) {
    const float v[12] = {iso.rotation.basisX.x, iso.rotation.basisX.y, iso.rotation.basisX.z,
                         iso.rotation.basisY.x, iso.rotation.basisY.y, iso.rotation.basisY.z,
                         iso.rotation.basisZ.x, iso.rotation.basisZ.y, iso.rotation.basisZ.z,
                         iso.translation.x,     iso.translation.y,     iso.translation.z};
    for (int k = 0; k < 12; ++k) o[k] = Bits(v[k]);
}

// harness state words: name -> (offset, count)
struct WordIndex {
    std::map<std::string, std::pair<std::uint32_t, std::uint32_t>> at;
    WordIndex() {
        std::uint32_t off = 0u;
        for (const auto &f : twin_fv::StateFields()) {
            at[f.name] = {off, f.count};
            off += f.count;
        }
    }
    const std::uint32_t *Ptr(const std::vector<std::uint32_t> &w, const char *name) const {
        auto it = at.find(name);
        return it == at.end() ? nullptr : w.data() + it->second.first;
    }
    std::uint32_t U(const std::vector<std::uint32_t> &w, const char *name) const {
        const std::uint32_t *p = Ptr(w, name);
        return p ? p[0] : 0xDEADBEEFu;
    }
};
const WordIndex &Words() {
    static const WordIndex w;
    return w;
}

struct RaceMapStore {
    std::vector<TwTrigger> triggers;
    std::vector<TwSpawn> spawns;
    std::vector<std::array<std::uint32_t, 12>> isos;
    TwRaceMap map;
    std::uint32_t SpawnFor(const GmIso4 &iso) {
        std::array<std::uint32_t, 12> w;
        PutIsoWords(w.data(), iso);
        for (std::size_t i = 0; i < isos.size(); ++i)
            if (isos[i] == w) return static_cast<std::uint32_t>(i);
        TwSpawn s{};
        GmQuat q;
        q.Set(iso.rotation);  // FV's matrix -> quaternion (dyna_core.cpp SetLocation)
        s.pos[0] = iso.translation.x;
        s.pos[1] = iso.translation.y;
        s.pos[2] = iso.translation.z;
        s.quat[0] = q.x;
        s.quat[1] = q.y;
        s.quat[2] = q.z;
        s.quat[3] = q.w;
        std::memcpy(s.iso, w.data(), 48);
        spawns.push_back(s);
        isos.push_back(w);
        return static_cast<std::uint32_t>(isos.size() - 1u);
    }
    void Bind(std::uint32_t cps, std::uint32_t laps) {
        map.triggers = triggers.data();
        map.triggerCount = static_cast<std::uint32_t>(triggers.size());
        map.spawns = spawns.data();
        map.spawnCount = static_cast<std::uint32_t>(spawns.size());
        map.requiredCps = cps;
        map.requiredLaps = laps;
    }
};

struct Car {
    tw::model::TwCarPack pack;
    float rest = 0.0f;
};

tw::model::TwGoldenContext Context(const Car &car, const RaceMapStore &m) {
    tw::model::TwGoldenContext ctx;
    ctx.fsGroundedBits = car.pack.fsGroundedBits;
    ctx.ldsGroundedBits = car.pack.ldsGroundedBits;
    ctx.fsAirborneBits = car.pack.fsAirborneBits;
    ctx.ldsAirborneBits = car.pack.ldsAirborneBits;
    ctx.spawns.iso = reinterpret_cast<const std::uint32_t(*)[12]>(m.isos.data());
    ctx.spawns.count = static_cast<std::uint32_t>(m.isos.size());
    ctx.requiredCps = m.map.requiredCps;
    ctx.requiredLaps = m.map.requiredLaps;
    return ctx;
}

bool ImportWords(const std::vector<std::uint32_t> &w, const tw::model::TwGoldenContext &ctx, TwinState *s,
                 std::string *why) {
    const auto e = tw::model::ImportGolden(
        [&](const char *name) -> tw::model::TwGoldenWords {
            auto it = Words().at.find(name);
            if (it == Words().at.end()) return {};
            return {w.data() + it->second.first, it->second.second};
        },
        ctx, s);
    if (e.status != tw::model::TwGoldenStatus::Ok) {
        if (why) *why = std::string(tw::model::TwGoldenStatusName(e.status)) + ":" + e.field;
        return false;
    }
    return true;
}

bool ImportRecord(const twin::File &f, const std::uint32_t *rec, const char *prefix,
                  const tw::model::TwGoldenContext &ctx, TwinState *s, std::string *why) {
    const auto e = tw::model::ImportGolden(
        [&](const char *name) -> tw::model::TwGoldenWords {
            const twin::FieldSpec *fs = f.table.Find(std::string(prefix) + name);
            if (fs == nullptr) return {};
            return {rec + fs->offset, fs->count};
        },
        ctx, s);
    if (e.status != tw::model::TwGoldenStatus::Ok) {
        if (why) *why = std::string(tw::model::TwGoldenStatusName(e.status)) + ":" + e.field;
        return false;
    }
    return true;
}

bool IsRaceField(const tw::model::TwFieldInfo &f) {
    return std::strncmp(f.golden, "race.", 5) == 0 && std::strcmp(f.name, "finishLowerNs") != 0 &&
           std::strcmp(f.name, "finishPresent") != 0;
}

// Field-by-field comparison; returns the names that differ.
std::string DiffFields(const TwinState &a, const TwinState &b, bool raceOnly, const std::set<std::string> &skip) {
    std::size_t n = 0;
    const tw::model::TwFieldInfo *fields = tw::model::TwinFields(&n);
    std::string out;
    for (std::size_t i = 0; i < n; ++i) {
        const auto &f = fields[i];
        if (raceOnly && !IsRaceField(f) && std::strcmp(f.name, "tickMs") != 0) continue;
        if (skip.count(f.name)) continue;
        const auto *pa = reinterpret_cast<const unsigned char *>(&a) + f.offset;
        const auto *pb = reinterpret_cast<const unsigned char *>(&b) + f.offset;
        if (std::memcmp(pa, pb, f.elementBytes * f.count) != 0) {
            out += out.empty() ? "" : ",";
            out += f.name;
        }
    }
    return out;
}

std::vector<std::uint32_t> Extract(const twin_fv::Clone &c) {
    std::vector<std::uint32_t> w(twin_fv::StateWordCount());
    twin_fv::ExtractState(c, w.data());
    return w;
}

bool Capture(fvx::PhysicsSandbox &sb, twin_fv::Clone *clone) {
    auto st = sb.CaptureState();
    if (!st) return false;
    fvx::PhysicsSandboxState local = std::move(st).Value();
    std::string err;
    const twin_fv::Clone *c = twin_fv::PeekClone(local, &err);
    if (!c) return false;
    *clone = *c;
    return true;
}

// ---------------------------------------------------------------------------
// FV's control plan for the sandbox inputs (validation_api.cpp BuildControlPlan, recorded mode)
bool FvPlan(const std::vector<fvx::PhysicsSandboxInputEvent> &in, std::uint32_t durationMs, ReplayControlPlan *plan) {
    std::vector<ReplayInputEvent> events;
    for (const auto &e : in) {
        ReplayInputActionValue v = ReplayInputActionValue::None();
        if (e.value.kind == fvx::PhysicsSandboxInputValueKind::Analog) v = ReplayInputActionValue::Analog(e.value.analog);
        if (e.value.kind == fvx::PhysicsSandboxInputValueKind::Switch)
            v = ReplayInputActionValue::Switch(e.value.switchState == fvx::PhysicsSandboxSwitchState::Released
                                                   ? ReplayInputSwitchState::Released
                                                   : e.value.switchState == fvx::PhysicsSandboxSwitchState::Pressed
                                                         ? ReplayInputSwitchState::Pressed
                                                         : ReplayInputSwitchState::NonCanonicalActive);
        events.push_back({static_cast<std::uint32_t>(100000 + e.timeMs),
                          static_cast<ReplayInputActionKind>(static_cast<int>(e.action)), v});
    }
    ReplayInputMetadata md;
    md.durationMs = durationMs;
    ReplayInputTimeline timeline;
    const std::vector<ReplayInputActionKind> defined = {
        ReplayInputActionKind::Accelerate, ReplayInputActionKind::Gas,         ReplayInputActionKind::Brake,
        ReplayInputActionKind::Steer,      ReplayInputActionKind::SteerLeft,   ReplayInputActionKind::SteerRight,
        ReplayInputActionKind::RaceRunning, ReplayInputActionKind::FinishLine, ReplayInputActionKind::Respawn};
    if (ReplayInputTimeline::Create(md, defined, std::move(events), &timeline) != ReplayInputTimelineCreateResult::Success)
        return false;
    ReplayControlPlanRequest request(timeline);
    request.controlTickMs = 10u;
    request.validationDurationMs = static_cast<std::int32_t>(durationMs);
    request.validationPrestartMs = 2600u;
    request.inputTimeBaseMs = 100000u;
    request.enableRaceSimulationAfterMs = 2600;
    request.establishRaceSpawnAtMs = 0;
    return BuildReplayControlPlan(request, plan) == ReplayControlPlanBuildResult::Success;
}

std::vector<TwInputEvent> ToTwin(const std::vector<fvx::PhysicsSandboxInputEvent> &in) {
    std::vector<TwInputEvent> out;
    for (const auto &e : in)
        out.push_back({e.timeMs, static_cast<TwAction>(static_cast<int>(e.action)),
                       static_cast<TwValueKind>(static_cast<int>(e.value.kind)),
                       static_cast<TwSwitch>(static_cast<int>(e.value.switchState)), e.value.analog});
    return out;
}

// ---------------------------------------------------------------------------
struct GoldenSet {
    std::string path;
    twin::File file;
    std::map<std::uint32_t, std::string> replayFile;  // index -> file name
    std::map<std::uint32_t, std::string> replayPath;
};

bool RaceRecord(const twin::File &f, const std::uint32_t *rec) {
    for (const char *n : {"ev.cp_accepted", "ev.finish", "ev.race_completed", "ev.respawned"}) {
        const twin::FieldSpec *fs = f.table.Find(n);
        if (fs && rec[fs->offset] != 0u) return true;
    }
    return false;
}

std::uint32_t Field(const twin::File &f, const std::uint32_t *rec, const char *name, std::uint32_t dflt = 0xFFFFFFFFu) {
    const twin::FieldSpec *fs = f.table.Find(name);
    return fs ? rec[fs->offset] : dflt;
}

struct GoldenHit {
    const GoldenSet *g;
    std::size_t record;
};

// ---------------------------------------------------------------------------
struct RunTotals {
    std::size_t ticks = 0, hits = 0, accepted = 0, respawns = 0, raceStartResets = 0, goldenIn = 0, goldenRace = 0,
                goldenSame = 0, goldenOther = 0, cpRespawns = 0, ctTriggers = 0, ctHits = 0, roleHits[5] = {};
};
RunTotals g_tot;

// Synthetic respawns (board 40 T6 coverage): a Respawn press every `everyMs` of
// race time (every third one pressed twice in the same tick), released 10 ms later.
std::vector<fvx::PhysicsSandboxInputEvent> WithRespawns(const std::vector<fvx::PhysicsSandboxInputEvent> &in,
                                                        std::uint32_t everyMs, std::uint32_t durationMs) {
    std::vector<fvx::PhysicsSandboxInputEvent> add;
    std::uint32_t n = 0;
    for (std::uint32_t t = everyMs; t + 20u < durationMs; t += everyMs, ++n) {
        fvx::PhysicsSandboxInputEvent e;
        e.timeMs = static_cast<std::int32_t>(t);
        e.action = fvx::PhysicsSandboxInputAction::Respawn;
        e.value.kind = fvx::PhysicsSandboxInputValueKind::Switch;
        e.value.switchState = fvx::PhysicsSandboxSwitchState::Pressed;
        add.push_back(e);
        if (n % 3u == 2u) add.push_back(e);
        e.timeMs += 10;
        e.value.switchState = fvx::PhysicsSandboxSwitchState::Released;
        add.push_back(e);
    }
    std::vector<fvx::PhysicsSandboxInputEvent> out;
    std::size_t j = 0;
    for (const auto &e : in) {
        while (j < add.size() && add[j].timeMs < e.timeMs) out.push_back(add[j++]);
        out.push_back(e);
    }
    while (j < add.size()) out.push_back(add[j++]);
    return out;
}

void RunReplay(const std::string &packs, const std::string &path, const Car &car, std::uint32_t maxTicks,
               const std::vector<GoldenHit> &golden, std::uint32_t respawnEveryMs) {
    const std::string name = path.substr(path.find_last_of('/') + 1);
    auto bytes = fv::ReadNativeReplayFile(path, fv::ReplayIdentity{path});
    if (!bytes) {
        g_stats.Check(false, "load", name + ": read");
        return;
    }
    auto source = fv::OpenInstalledPackDirectory(packs);
    if (!source) {
        g_stats.Check(false, "load", "packs");
        return;
    }
    fvx::PhysicsSandboxOptions options;
    options.backend = fv::SimulationBackend::Reference;
    auto created = fvx::CreatePhysicsSandbox(std::move(source).Value(), options);
    if (!created) {
        g_stats.Check(false, "load", name + ": sandbox");
        return;
    }
    fvx::PhysicsSandbox sb = std::move(created).Value();
    g_triggers.clear();
    g_triggerOfBlock.clear();
    g_beforeReset.valid = g_afterReset.valid = false;
    auto loaded = sb.LoadReplay({bytes.Value().data(), bytes.Value().size()}, fv::ReplayIdentity{path});
    if (!loaded) {
        g_stats.Check(false, "load", name + ": " + loaded.Error().diagnostic);
        return;
    }
    const std::uint32_t durationMs = static_cast<std::uint32_t>(loaded.Value().durationMs);

    twin_fv::Clone clone0;
    if (!Capture(sb, &clone0)) {
        g_stats.Check(false, "load", name + ": capture");
        return;
    }
    // ---- race map from FV's course: spawn 0 = race start spawn (player previous spawn)
    RaceMapStore m;
    m.SpawnFor(clone0.race.player.CaptureRuntimeClone().previousSpawnLocation);
    for (FvTrigger &t : g_triggers) {
        TwTrigger tt = t.t;
        if (t.hasSpawn) tt.spawn = static_cast<std::uint16_t>(m.SpawnFor(t.spawn));
        m.triggers.push_back(tt);
        if (tt.currentTransform) ++g_tot.ctTriggers;
    }
    m.Bind(clone0.race.progress.requiredCheckpointCount, clone0.race.progress.requiredLapCount);
    {   // M2 recipe: the trigger table from CudaSceneActor equals the one from FV's corpus collection
        bool same = g_actorTriggers.size() == g_triggers.size() && g_actorCheckpointsOutsideTriggers == 0u;
        for (std::size_t i = 0; same && i < g_triggers.size(); ++i) {
            const ActorTrigger &a = g_actorTriggers[i];
            const FvTrigger &c = g_triggers[i];
            std::array<std::uint32_t, 12> wa{}, wc{};
            if (a.hasSpawn) PutIsoWords(wa.data(), a.spawn);
            if (c.hasSpawn) PutIsoWords(wc.data(), c.spawn);
            same = a.t.role == c.t.role && a.t.currentTransform == c.t.currentTransform && a.t.slot == c.t.slot &&
                   a.hasSpawn == c.hasSpawn && wa == wc;
        }
        g_stats.Check(same, "race_map_from_actors", name + ": actors " + std::to_string(g_actorTriggers.size()) +
                                                        " collection " + std::to_string(g_triggers.size()));
    }
    g_stats.Check(TwRaceMapValid(m.map), "race_map", name);
    g_stats.Check(clone0.race.progress.installedTriggerCount <= m.map.triggerCount, "race_map", name + ": installed");
    tw::model::TwGoldenContext ctx = Context(car, m);

    if (respawnEveryMs != 0u) {
        auto original = sb.ReadInputs();
        if (!original || !sb.ReplaceInputs(WithRespawns(original.Value(), respawnEveryMs, durationMs))) {
            g_stats.Check(false, "load", name + ": replace inputs");
            return;
        }
    }
    // ---- 1. controls vs FV's plan, every tick
    auto inputs = sb.ReadInputs();
    if (!inputs) {
        g_stats.Check(false, "load", name + ": inputs");
        return;
    }
    ReplayControlPlan plan;
    const bool planOk = FvPlan(inputs.Value(), durationMs, &plan);
    g_stats.Check(planOk, "fv_plan", name);
    const std::vector<TwInputEvent> tev = ToTwin(inputs.Value());
    TwControlStream stream;
    std::string err;
    const std::uint32_t nTicks = planOk ? static_cast<std::uint32_t>(plan.ticks.size()) : 0u;
    const bool built = TwBuildControls(tev.data(), tev.size(), TwTimeline{}, 10u, nTicks, &stream, &err);
    g_stats.Check(built && stream.ticks.size() == nTicks, "controls_build", name + ": " + err);
    if (!built || stream.ticks.size() != nTicks) return;
    for (std::uint32_t i = 0; i < nTicks; ++i) {
        const ReplayControlTick &ft = plan.ticks[i];
        const TwinControl c = stream.ticks[i];
        const bool same = ft.timeMs == 10u * (i + 1u) && Bits(ft.controls.lowSpeedGateA) == Bits(TwCtlGas(c)) &&
                          Bits(ft.controls.lowSpeedGateB) == Bits(TwCtlBrake(c)) &&
                          Bits(ft.controls.steering) == Bits(TwCtlSteer(c)) &&
                          ft.actions.establishRaceSpawn == ((c & kCtlEstablishSpawn) != 0u) &&
                          ft.actions.enableRaceSimulation == ((c & kCtlBeginRace) != 0u) &&
                          ft.actions.resetAtRaceStart == ((c & kCtlRaceStartReset) != 0u) &&
                          ft.actions.respawnAtCheckpointCount == stream.Respawns(i);
        g_stats.Check(same, "controls_tick", name + " T=" + std::to_string(ft.timeMs));
    }

    // ---- 3b. race-start reset (captured during the load's prestart)
    if (g_beforeReset.valid && g_afterReset.valid) {
        twin_fv::Clone pre = clone0, post = clone0;
        pre.runtime.vehicle = g_beforeReset.vehicle;
        pre.runtime.body = g_beforeReset.body;
        post.runtime.vehicle = g_afterReset.vehicle;
        post.runtime.body = g_afterReset.body;
        TwinState a, b;
        std::string why;
        if (ImportWords(Extract(pre), ctx, &a, &why) && ImportWords(Extract(post), ctx, &b, &why)) {
            const std::uint32_t idx = g_resetTickMs / 10u - 1u;
            TwRaceEvents ev;
            TwRaceEventsClear(ev);
            TwRaceTickBegin(m.map, a, stream.ticks[idx], 0u, g_resetTickMs, car.rest, ev);
            const std::string d = DiffFields(a, b, false, {});
            g_stats.Check(d.empty() && (stream.ticks[idx] & kCtlRaceStartReset), "race_start_reset", name + ": " + d);
            ++g_tot.raceStartResets;
        } else {
            g_stats.Check(false, "race_start_reset", name + ": import " + why);
        }
    } else {
        g_stats.Check(false, "race_start_reset", name + ": not observed");
    }

    // golden records of this replay, by pre tick
    std::map<std::uint32_t, std::vector<GoldenHit>> goldenAt;
    for (const GoldenHit &gh : golden) goldenAt[Field(gh.g->file, gh.g->file.Record(gh.record), "meta.tick")].push_back(gh);

    // ---- 2. tick loop
    twin_fv::Clone pre = clone0;
    std::vector<std::uint32_t> preW = Extract(pre);
    const std::uint32_t total = std::min<std::uint32_t>(maxTicks, nTicks > 260u ? nTicks - 260u : 0u);
    for (std::uint32_t k = 0; k < total; ++k) {
        g_hits.clear();
        g_afterRespawn.valid = false;
        g_afterRespawnLocked = false;
        g_unknownTriggerHits = 0u;
        g_recordNextStep = true;
        auto adv = sb.AdvanceTicks(1u);
        g_recordNextStep = false;
        if (!adv) break;
        const fvx::PhysicsSandboxStateView &view = adv.Value();
        twin_fv::Clone post;
        if (!Capture(sb, &post)) break;
        const std::vector<std::uint32_t> postW = Extract(post);
        const std::uint32_t T = Words().U(postW, "world.tick_ms");
        const std::uint32_t idx = T / 10u - 1u;
        ++g_tot.ticks;
        g_tot.hits += g_hits.size();
        for (const Hit &h : g_hits) {
            const TwTrigger &ht = m.triggers[h.trigger];
            if (ht.role <= kRoleStartFinish) ++g_tot.roleHits[ht.role];
            if (ht.currentTransform) ++g_tot.ctHits;
        }
        g_stats.Check(g_unknownTriggerHits == 0u, "hits_identified", name + " T=" + std::to_string(T));
        if (idx >= stream.ticks.size()) break;
        const TwinControl c = stream.ticks[idx];
        const std::uint32_t nResp = stream.Respawns(idx);

        TwinState s, fvPost;
        std::string why;
        const bool okA = ImportWords(preW, ctx, &s, &why);
        const bool okB = okA && ImportWords(postW, ctx, &fvPost, &why);
        g_stats.Check(okA && okB, "import", name + " T=" + std::to_string(T) + " " + why);
        if (!okA || !okB) {
            pre = post;
            preW = postW;
            continue;
        }
        TwRaceEvents ev;
        TwRaceEventsClear(ev);
        TwRaceTickBegin(m.map, s, c, nResp, T, car.rest, ev);
        // ---- 3a. respawn state vs FV's state right after Respawn
        if (g_afterRespawn.valid || ev.respawned) {
            ++g_tot.respawns;
            if (s.spawnIndex != 0u) ++g_tot.cpRespawns;
            twin_fv::Clone r = pre;
            r.runtime.vehicle = g_afterRespawn.vehicle;
            r.runtime.body = g_afterRespawn.body;
            TwinState fr;
            std::string w2;
            if (g_afterRespawn.valid && ImportWords(Extract(r), ctx, &fr, &w2)) {
                fr.tickMs = s.tickMs;     // world tick is set in the same step (SetSimulationTime)
                fr.respawns = s.respawns;  // counter: compared with post below
                const std::string d = DiffFields(s, fr, false, {});
                g_stats.Check(d.empty(), "respawn_state", name + " T=" + std::to_string(T) + ": " + d);
            } else {
                g_stats.Check(false, "respawn_state", name + " T=" + std::to_string(T) + ": capture " + w2);
            }
        }
        // trigger contacts in FV order (substep, then contact order)
        for (const Hit &h : g_hits) TwRaceOnTriggerContact(m.map, s, h.trigger, ev);

        // compare race fields + events with FV
        const std::string d = DiffFields(s, fvPost, true, {});
        g_stats.Check(d.empty(), "race_fields", name + " T=" + std::to_string(T) + ": " + d);
        const std::uint32_t dPrepared = Words().U(postW, "race.prepared_events") - Words().U(preW, "race.prepared_events");
        const std::uint32_t dClears = Words().U(postW, "race.freewheel_clears") - Words().U(preW, "race.freewheel_clears");
        g_stats.Check(ev.contacts == dPrepared, "contacts_vs_prepared",
                      name + " T=" + std::to_string(T) + " twin " + std::to_string(ev.contacts) + " fv " +
                          std::to_string(dPrepared));
        g_stats.Check(ev.freewheelClears == dClears, "freewheel_clears", name + " T=" + std::to_string(T));
        bool journal = ev.journalCount == view.acceptedCheckpointEvents.size();
        for (std::size_t j = 0; journal && j < view.acceptedCheckpointEvents.size() && j < kJournalMax; ++j) {
            const auto &fe = view.acceptedCheckpointEvents[j];
            const TwAccepted &te = ev.journal[j];
            journal = fe.checkpointSlot == te.slot && fe.checkpointIndex == te.index && fe.lap == te.lap &&
                      fe.eventIndex == te.eventIndex && fe.finish == te.finish;
        }
        g_stats.Check(journal, "journal", name + " T=" + std::to_string(T));
        g_tot.accepted += ev.journalCount;
        g_stats.Check(ev.respawned == Words().U(postW, "race.respawns") - Words().U(preW, "race.respawns"),
                      "respawn_count", name + " T=" + std::to_string(T));
        // finish estimate appears exactly in the completion tick
        const bool fvFinishNew = Words().U(postW, "race.finish_ns") == 1u && Words().U(preW, "race.finish_ns") == 0u;
        g_stats.Check(fvFinishNew == (ev.raceCompleted != 0u), "finish_tick", name + " T=" + std::to_string(T));
        // free wheel: FV's post flag is the twin's unless the forces set it again this tick
        if (ev.freewheelClears && !s.freewheel) g_stats.Check(true, "freewheel_cleared_ticks");

        // ---- 4. goldens recorded at this pre tick
        auto git = goldenAt.find(k);
        if (git != goldenAt.end()) {
            for (const GoldenHit &gh : git->second) {
                const twin::File &f = gh.g->file;
                const std::uint32_t *rec = f.Record(gh.record);
                // in.* = stream
                const TwTickInputs in = TwCtlPhysicsInputs(c);
                const bool inputsSame = Field(f, rec, "in.tick_ms") == T && Field(f, rec, "in.gas") == Bits(in.gas) &&
                                        Field(f, rec, "in.brake") == Bits(in.brake) &&
                                        Field(f, rec, "in.steer") == Bits(in.steer) &&
                                        Field(f, rec, "in.respawn") == nResp;
                // scenario records (selected_by 253) may run scripted inputs: no input check, and
                // they are compared below only when this tick's inputs equal the replay's
                const bool scenario = Field(f, rec, "meta.selected_by") == 253u;
                if (!scenario) {
                    ++g_tot.goldenIn;
                    g_stats.Check(inputsSame, "golden_in", gh.g->path + " rec " + std::to_string(gh.record));
                }
                // same pre-state? (golden pre.* == this run's pre words, state + race tiers)
                bool same = true;
                for (const auto &kv : Words().at) {
                    const twin::FieldSpec *fs = f.table.Find("pre." + kv.first);
                    if (!fs || fs->count != kv.second.second) continue;
                    if (std::memcmp(rec + fs->offset, preW.data() + kv.second.first, 4u * fs->count) != 0) {
                        same = false;
                        break;
                    }
                }
                if (!same || !inputsSame) {
                    ++g_tot.goldenOther;
                    continue;
                }
                ++g_tot.goldenSame;
                TwinState gs, gp;
                std::string w3;
                if (!ImportRecord(f, rec, "pre.", ctx, &gs, &w3) || !ImportRecord(f, rec, "post.", ctx, &gp, &w3)) {
                    g_stats.Check(false, "golden_import", gh.g->path + ": " + w3);
                    continue;
                }
                TwRaceEvents ge;
                TwRaceEventsClear(ge);
                TwRaceTickBegin(m.map, gs, c, nResp, T, car.rest, ge);
                for (const Hit &h : g_hits) TwRaceOnTriggerContact(m.map, gs, h.trigger, ge);
                const std::string gd = DiffFields(gs, gp, true, {});
                const bool evSame = Field(f, rec, "ev.cp_accepted") == ge.cpAccepted &&
                                    Field(f, rec, "ev.cp_last_slot") == ge.cpLastSlot &&
                                    Field(f, rec, "ev.finish") == ge.finish &&
                                    Field(f, rec, "ev.race_completed") == ge.raceCompleted &&
                                    Field(f, rec, "ev.respawned") == ge.respawned;
                const bool race = RaceRecord(f, rec);
                if (race) ++g_tot.goldenRace;
                g_stats.Check(gd.empty() && evSame, race ? "golden_race_records" : "golden_other_records",
                              gh.g->path + " rec " + std::to_string(gh.record) + ": " + gd);
            }
        }
        pre = std::move(post);
        preW = postW;
    }
}


// TwRefineFinish vs FV's RefineReplayFinishTime on the same synthetic probes
// (completion iff dt >= threshold, plus a non-monotone probe).
void FinishBracketParity() {
    std::mt19937 rng(1234u);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int i = 0; i < 20000; ++i) {
        const std::uint32_t tick = 2600u + 10u * static_cast<std::uint32_t>(rng() % 20000u);
        const std::uint32_t substeps = 1u + rng() % 12u;
        const float dur = 0.01f / static_cast<float>(substeps);
        const double start = static_cast<double>(dur) * static_cast<double>(rng() % substeps);
        const float thr = static_cast<float>(u01(rng)) * dur;
        const bool wobble = (i % 7) == 0;
        auto probe = [&](float dt) { return wobble ? (dt >= thr) != (dt > 0.9f * dur && dt < 0.95f * dur) : dt >= thr; };
        std::uint64_t lo = 0, hi = 0;
        const bool tw = TwRefineFinish(TwFinishSubstep{tick, start, dur}, probe, &lo, &hi);
        const auto fvr = RefineReplayFinishTime(ReplayFinishSubstep{tick, 10u, start, dur}, ReplayFinishProbe(probe));
        const bool same = tw == fvr.has_value() && (!tw || (fvr->lowerBoundNs == lo && fvr->upperBoundNs == hi &&
                                                            fvr->estimatedNs == hi));
        g_stats.Check(same, "finish_bracket_port", "case " + std::to_string(i));
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s PACKS CAR_JSON [--golden F]... [--replay R]... [--max-ticks N] [--from-golden]\n",
                     argv[0]);
        return 2;
    }
    const std::string packs = argv[1];
    Car car;
    std::string err;
    if (!tw::model::CompileCarFromJson(argv[2], &car.pack, &err)) {
        std::fprintf(stderr, "car: %s\n", err.c_str());
        return 2;
    }
    car.rest = car.pack.suspension_wheelRestDamperAbsorb;
    std::vector<std::string> replays;
    std::vector<GoldenSet> goldens;
    goldens.reserve(8);
    std::uint32_t maxTicks = 0xFFFFFFFFu, respawnEveryMs = 0u;
    bool fromGolden = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--golden" && i + 1 < argc) {
            goldens.emplace_back();
            GoldenSet &g = goldens.back();
            g.path = argv[++i];
            if (!twin::ReadFile(g.path, &g.file, &err)) {
                std::fprintf(stderr, "golden %s: %s\n", g.path.c_str(), err.c_str());
                return 2;
            }
            std::uint32_t position = 0u;
            for (const std::string &o : twin::JsonObjects(g.file.metadata, "replays")) {
                std::uint32_t index = position++;
                const std::size_t at = o.find("\"index\":");
                if (at != std::string::npos) index = static_cast<std::uint32_t>(std::strtoul(o.c_str() + at + 8, nullptr, 10));
                g.replayFile[index] = twin::JsonString(o, "file");
                g.replayPath[index] = twin::JsonString(o, "path");
            }
        } else if (a == "--replay" && i + 1 < argc) {
            replays.push_back(argv[++i]);
        } else if (a == "--max-ticks" && i + 1 < argc) {
            maxTicks = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (a == "--respawn-every" && i + 1 < argc) {
            respawnEveryMs = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (a == "--from-golden") {
            fromGolden = true;
        }
    }
    if (fromGolden)
        for (const GoldenSet &g : goldens)
            for (const auto &kv : g.replayPath) {
                bool have = false;
                for (const std::string &r : replays) have |= r.substr(r.find_last_of('/') + 1) == g.replayFile.at(kv.first);
                if (!have && !kv.second.empty()) replays.push_back(kv.second);
            }
    FinishBracketParity();
    const auto t0 = std::chrono::steady_clock::now();
    for (const std::string &r : replays) {
        const std::string file = r.substr(r.find_last_of('/') + 1);
        std::vector<GoldenHit> gh;
        for (const GoldenSet &g : goldens) {
            const twin::FieldSpec *fr = g.file.table.Find("meta.replay");
            if (!fr) continue;
            for (std::size_t i = 0; i < g.file.RecordCount(); ++i) {
                auto it = g.replayFile.find(g.file.Record(i)[fr->offset]);
                if (it != g.replayFile.end() && it->second == file) gh.push_back({&g, i});
            }
        }
        const auto r0 = std::chrono::steady_clock::now();
        RunReplay(packs, r, car, maxTicks, respawnEveryMs ? std::vector<GoldenHit>{} : gh, respawnEveryMs);
        std::printf("%-26s %6.2f s  triggers=%zu installs=%u\n", file.c_str(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count(), g_triggers.size(),
                    g_installs);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("replays %zu, ticks %zu, trigger contacts %zu, accepted passages %zu, respawn ticks %zu (%zu at a checkpoint "
                "spawn), race-start resets %zu\n",
                replays.size(), g_tot.ticks, g_tot.hits, g_tot.accepted, g_tot.respawns, g_tot.cpRespawns,
                g_tot.raceStartResets);
    std::printf("hits by role: start %zu finish %zu checkpoint %zu none %zu start-finish %zu; current-transform "
                "triggers %zu, hits %zu\n",
                g_tot.roleHits[0], g_tot.roleHits[1], g_tot.roleHits[2], g_tot.roleHits[3], g_tot.roleHits[4],
                g_tot.ctTriggers, g_tot.ctHits);
    std::printf("golden: in.* checked %zu; records with this run's pre-state %zu (race-event %zu), other pre-state %zu\n",
                g_tot.goldenIn, g_tot.goldenSame, g_tot.goldenRace, g_tot.goldenOther);
    std::set<std::string> cats;
    for (const auto &kv : g_stats.ok) cats.insert(kv.first);
    for (const auto &kv : g_stats.fail) cats.insert(kv.first);
    std::size_t failed = 0;
    for (const std::string &cat : cats) {
        const std::size_t f = g_stats.fail.count(cat) ? g_stats.fail.at(cat) : 0u;
        const std::size_t o = g_stats.ok.count(cat) ? g_stats.ok.at(cat) : 0u;
        failed += f;
        std::printf("  %-24s %8zu ok %6zu fail%s%s\n", cat.c_str(), o, f, f ? "  first: " : "",
                    f ? g_stats.firstFail[cat].c_str() : "");
    }
    std::printf("%s (%.1f s)\n", failed ? "FAIL" : "PASS", secs);
    return failed ? 1 : 0;
}
