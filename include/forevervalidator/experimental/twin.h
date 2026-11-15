#ifndef FOREVERVALIDATOR_EXPERIMENTAL_TWIN_H
#define FOREVERVALIDATOR_EXPERIMENTAL_TWIN_H

// Twin engine API (docs/twin/design.md §8.1, docs/twin/integration.md).
//
// This API is experimental. It may change without compatibility guarantees.
// It is available when ForeverValidator is configured with
// -DFOREVERVALIDATOR_BUILD_TWIN=ON (FOREVERVALIDATOR_HAS_TWIN == 1).
//
// The twin keeps FV's observable physics but is not bit-exact with FV. It is
// bit-deterministic with itself: the same world, state and controls give the
// same bits on every supported compiler, ISA and lane width.
//
// Status of this skeleton (QueryTwinEngine() reports it at run time):
//   - TwinStep / TwinStepTicks / TwinBatchRunner: wired to the tick engine
//     (src/twin/step/), which reports TwinStatus::NotImplemented until M5.
//   - CompileTwinWorld: checks the scope (Stadium/StadiumCar) and compiles
//     the world package, car pack, water zone and race map from the
//     sandbox's FV session (src/twin/fv/twin_world_build.cpp); collision
//     (M3) is not built yet, so TwinStep reports NotImplemented.
//   - TwinControlsFromEvents: implemented (race layer, src/twin/race/).
//   - TwinImportState: implemented (src/twin/fv/import.cpp).
//   - TwinRaceStartState: NotImplemented (M6b).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/validation.h>

namespace forevervalidator::experimental::twin {

using forevervalidator::TwinCollision;
using forevervalidator::TwinPolicy;

// The per-candidate state: 448 B, 64-byte aligned, trivially copyable (a
// snapshot is a memcpy). The layout is generated from
// src/twin/model/state_fields.inc; use the accessors below or the twin's
// internal headers to read fields.
struct alignas(64) TwinState {
    std::uint8_t bytes[448];
};
static_assert(sizeof(TwinState) == 448u, "TwinState is 448 bytes");

// One tick of controls (design.md §3.3; decoders in src/twin/race/tw_race.h):
//   bits 0-17 signed steer in 1/65536, 18 gas, 19 brake, 20-27 respawns,
//   28 respawn escape, 29 race-start reset, 30 begin race, 31 establish spawn.
using TwinControl = std::uint32_t;

struct TwinControlStream {
    std::uint32_t firstTickMs = 0u;
    std::uint32_t tickMs = 10u;
    std::vector<TwinControl> ticks;  // ticks[i] is tick firstTickMs + i * tickMs
    // (tick index, respawn count) for ticks with the escape bit (>= 255 respawns)
    std::vector<std::pair<std::uint32_t, std::uint32_t>> respawnOverflow;
};

enum class TwinStatus : std::uint8_t {
    Ok,
    NotImplemented,   // this piece of the twin is not built yet
    InvalidArgument,
    Stopped,          // TwinStepTicks: the stop rule ended the run early
};

const char *TwinStatusName(TwinStatus status) noexcept;

struct TwinEvents {
    TwinStatus status = TwinStatus::Ok;
    std::uint32_t ticks = 0u;
    std::uint32_t substeps = 0u;
    std::uint32_t respawns = 0u;
    std::uint32_t checkpoints = 0u;
    bool finished = false;
    std::uint32_t finishTickMs = 0u;
    std::uint64_t finishLowerNs = 0u;  // finish bracket lower bound; orders ties at the finish tick
    std::uint32_t cpLastSlot = 0xFFFFFFFFu;  // slot of the last tick's last accepted passage (none: ~0)
    bool finishPassage = false;              // the last tick accepted a finish passage (lap end too)
    std::uint32_t wheelContacts = 0u;        // the last tick's wheel / body contact counts
    std::uint32_t bodyContacts = 0u;
};

struct TwinStopRule {
    bool atFinish = true;
};

struct TwinEngineInfo {
    bool stepImplemented = false;
    std::string modelId;   // changes whenever any golden hash changes
    std::string missing;   // what the engine still lacks
};

TwinEngineInfo QueryTwinEngine() noexcept;

struct TwinWorldInfo {
    MapEnvironment mapEnvironment = MapEnvironment::Unknown;
    VehicleModel vehicleModel = VehicleModel::Unknown;
    TwinPolicy policy = TwinPolicy::FvSubsteps;
    TwinCollision collision = TwinCollision::Noise;   // what the caller asked for
    // The collision stage this world runs: "c" (Noise), "b" (Exact), or an
    // experiment named by FOREVERVALIDATOR_TWIN_COLLIDE (tw_collision_choice.h).
    std::string collisionStage = "c";
    bool carPackReady = false;    // M4: CarForceConsts from FV's static configuration
    bool worldPackReady = false;  // M2: triangles, triggers, spawns
    std::string mapName;
    // The tick kernel variant this world runs: "baseline", "avx2" or
    // "avx512" (runtime dispatch by CPUID; FOREVERVALIDATOR_TWIN_ISA
    // overrides it). Results are the same bits on every variant.
    std::string isa = "baseline";
};

// Immutable per-map, per-car, per-policy data. Thread-safe to share.
struct TwinWorldFactory;

class TwinWorld {
public:
    ~TwinWorld();
    TwinWorld(const TwinWorld &) = delete;
    TwinWorld &operator=(const TwinWorld &) = delete;

    const TwinWorldInfo &Info() const noexcept;

    struct Impl;
    const Impl &Internal() const noexcept { return *impl_; }

private:
    explicit TwinWorld(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct TwinWorldFactory;  // src/twin/api/twin_api_internal.h
};

// ---- session creation from an FV-loaded replay or map ------------------------
// `loaded` is a sandbox after LoadReplay / LoadScenario, on any CPU backend
// (optimized_cpu recommended: FV runs the prestart, design.md §8.2).
// Errors carry ValidationFailureReason::TwinUnsupportedSimulationScope for a
// map or car the twin does not model, and for Stunts mode (out of scope).
// The defaults are the product configuration, the one that passes the
// acceptance test and is fastest (docs/twin/scoreboard.md): FvSubsteps with
// the Noise collision stage. TwinCollision::Exact is the reference stage; the
// other policies are opt-in and fail the acceptance test.
PhysicsSandboxResult<std::shared_ptr<const TwinWorld>> CompileTwinWorld(
        const PhysicsSandbox &loaded,
        TwinPolicy policy = TwinPolicy::FvSubsteps,
        TwinCollision collision = TwinCollision::Noise) noexcept;

// The race-start state (T = 2600) for `seed`, without running FV. (M6b)
PhysicsSandboxResult<TwinState> TwinRaceStartState(
        const PhysicsSandbox &loaded, std::uint32_t seed) noexcept;

// An exact FV state, e.g. the sandbox state after T = 2600 (the hand-off;
// the twin's first tick is then T = 2610), as a twin state of `world`.
// ImportGolden's refusals are errors (spawn not in the world's table, unknown
// fs/lds pair, ...).
PhysicsSandboxResult<TwinState> TwinImportState(
        const TwinWorld &world, const PhysicsSandboxState &state) noexcept;
// Same, compiling the world from `loaded` first (slow: one world compile per call).
PhysicsSandboxResult<TwinState> TwinImportState(
        const PhysicsSandbox &loaded, const PhysicsSandboxState &state) noexcept;

// FV's input state machine, run once: replay events (sandbox time base,
// 0 = race start) to one control per tick for ticks
// firstTickMs, firstTickMs + 10, ... (tickCount ticks).
PhysicsSandboxResult<TwinControlStream> TwinControlsFromEvents(
        const TwinWorld &world,
        const std::vector<PhysicsSandboxInputEvent> &events,
        std::uint32_t firstTickMs,
        std::uint32_t tickCount) noexcept;

// ---- stepping ------------------------------------------------------------------
// One 10 ms tick. On a status other than Ok the state is unchanged. On a
// respawn-escape tick (>= 255 respawns) the saturated 255 of the control is
// used: FV's respawn is idempotent apart from the counter.
TwinStatus TwinStep(const TwinWorld &world, TwinState &state, TwinControl control,
                    TwinEvents *events = nullptr) noexcept;

// Up to `count` ticks; returns the number executed (events->status says why
// it stopped early).
std::uint32_t TwinStepTicks(const TwinWorld &world, TwinState &state,
                            const TwinControl *controls, std::size_t count,
                            TwinStopRule stop = {},
                            TwinEvents *events = nullptr) noexcept;

// Ticks [first, first + count) of a control stream, with the exact respawn
// counts of escape ticks (stream.respawnOverflow).
std::uint32_t TwinStepStream(const TwinWorld &world, TwinState &state,
                             const TwinControlStream &stream, std::size_t first,
                             std::size_t count, TwinStopRule stop = {},
                             TwinEvents *events = nullptr) noexcept;

// FNV-1a 64 of the state with NaNs canonicalised (the determinism hash).
std::uint64_t TwinStateHash(const TwinState &state) noexcept;

// Reads one field by its harness (golden) name, e.g. "body.pos", as 32-bit
// words widened per element. Returns the element count, 0 if unknown.
std::uint32_t TwinReadField(const TwinState &state, const char *goldenName,
                            std::uint32_t *words, std::uint32_t capacity) noexcept;

// ---- batch -----------------------------------------------------------------------
struct TwinCandidate {
    std::uint64_t id = 0u;
    TwinState start{};
    std::vector<TwinControl> controls;
};

struct TwinCandidateResult {
    std::uint64_t id = 0u;
    TwinStatus status = TwinStatus::Ok;
    std::uint32_t ticks = 0u;
    TwinState end{};
    TwinEvents events{};
};

// What a per-tick evaluator sees of one candidate (one lane).
struct TwinLaneView {
    std::uint64_t candidateId = 0u;
    std::uint32_t tick = 0u;           // ticks executed so far
    const TwinState *state = nullptr;
    const TwinEvents *events = nullptr;
};

// Called after every tick of every candidate; return false to stop that
// candidate.
using TwinTickEvaluator = std::function<bool(const TwinLaneView &)>;

struct TwinBatchOptions {
    std::uint32_t lanes = 1u;          // 1 today; 4/8/16 with the lane scheduler (M10)
    TwinStopRule stop{};
    TwinTickEvaluator evaluator;       // optional
};

class TwinBatchRunner {
public:
    virtual ~TwinBatchRunner() = default;
    virtual std::uint32_t Lanes() const noexcept = 0;
    // Results are in candidate order. Batch == single run, bit for bit.
    virtual TwinStatus Run(const std::vector<TwinCandidate> &candidates,
                           std::vector<TwinCandidateResult> *results) noexcept = 0;
};

PhysicsSandboxResult<std::unique_ptr<TwinBatchRunner>> CreateTwinBatchRunner(
        std::shared_ptr<const TwinWorld> world,
        TwinBatchOptions options = {}) noexcept;

// ---- search (M13, docs/twin/m13-search.md) -------------------------------------
// The GPU search session API on CPU threads: `configuration` is the CUDA
// search configuration (same seeds, same candidates), `loaded` an
// optimized-CPU sandbox at the branch. Equivalent to setting
// configuration.twin and calling CreatePhysicsSandboxCudaSearchSession.
PhysicsSandboxResult<PhysicsSandboxCudaSearchSession> CreateTwinSearchSession(
        PhysicsSandbox &loaded,
        PhysicsSandboxCudaSearchConfiguration configuration,
        const PhysicsSandboxTwinSearchOptions &options = {}) noexcept;


// ---- bridge (docs/twin/bridge.md) ----------------------------------------------
// A twin result made exact: the twin winner's inputs played in FV optimized_cpu
// (pass 1) and solved tick by tick so that FV follows the twin's path (pass 2:
// steer edits, the smallest that helps, gas and brake as the twin pressed them
// but for rare one-tick timing shifts). The result is the whole input list,
// replayed through FV's own input path before it is returned (the same states
// bit for bit), and how far FV's car stayed from the twin's.
struct TwinBridgeOptions {
    std::uint32_t lookTicks = 100u;   // the solve's lookahead
    double margin = 0.02;             // an edit must beat no edit by this fraction
    double pedalMargin = 0.10;        // a pedal timing shift must beat the best steer move by this
    std::uint32_t threads = 1u;       // candidate moves scored in parallel (FV sandboxes, one a thread)
    std::function<bool()> cancelled;  // polled once a tick; true ends Run with a failure
};
struct TwinBridgeResult {
    // the whole input list (sandbox time base): the branch's own events up to
    // the branch, the followed ticks' events, then the winner's own events
    std::vector<PhysicsSandboxInputEvent> inputs;
    double maxDriftMeters = 0.0;          // largest position gap to the twin's path over the window
    double finalDriftMeters = 0.0;        // the gap at the last tick
    double passOneMaxDriftMeters = 0.0;   // the twin's inputs as they are, on FV
    bool solved = false;                  // false: pass 1 was the closer one (inputs unchanged)
    std::uint32_t ticks = 0u;
    std::uint32_t steerEdits = 0u, steerShifts = 0u, pedalShifts = 0u;
};
class TwinBridge {
public:
    virtual ~TwinBridge() = default;
    // inputs: the twin winner's whole input list; ticks: from the branch (the
    // state the bridge was created at) to the last tick to follow.
    virtual PhysicsSandboxResult<TwinBridgeResult> Run(
            const std::vector<PhysicsSandboxInputEvent> &inputs, std::uint32_t ticks,
            const TwinBridgeOptions &options = {}) noexcept = 0;
};
// `loaded`: an optimized-CPU sandbox at the branch the twin search started from
// (cloned; the twin world compiled from it once).
PhysicsSandboxResult<std::unique_ptr<TwinBridge>> CreateTwinBridge(
        const PhysicsSandbox &loaded) noexcept;

}  // namespace forevervalidator::experimental::twin

#endif
