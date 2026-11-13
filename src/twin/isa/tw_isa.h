// Twin runtime ISA dispatch (docs/twin/integration.md §6).
//
// The tick-path objects (step, collision, exact-fast test, forces) are built
// three times: at the baseline ISA (SSE2 on x86-64) and, as isolated variant
// objects, for x86-64-v3 (AVX2) and x86-64-v4 (AVX-512), with LTO and without
// FP contraction (src/twin/CMakeLists.txt). Each variant is one
// relocatable object whose every defined symbol carries a variant suffix except
// its entry (isolate_variant.sh), so no inline or template code compiled for one
// ISA can be picked up by another (the COMDAT hazard of mixed-ISA static links).
//
// A session (TwinWorld) picks one variant at creation: the best the CPU
// supports (CPUID), or FOREVERVALIDATOR_TWIN_ISA=baseline|avx2|avx512 /
// SetOverride() for tests. The variants run the same source without FMA, so
// the twin is bit-identical across widths and ISAs (determinism CI): the
// choice changes speed, never results. The ABI below passes the world's
// packs by pointer; each variant checks the layout of every shared type
// against the baseline (Layout) before it is used.
//
// FV-free.
#ifndef TW_ISA_H
#define TW_ISA_H

#include <cstddef>
#include <cstdint>

#include "tw_collide.h"
#include "tw_step.h"

namespace tw {
namespace isa {

enum class Kind : std::uint8_t { Baseline = 0, Avx2 = 1, Avx512 = 2 };
constexpr int kKinds = 3;

const char *KindName(Kind kind);

// What a variant's tick instance is built from (all owned by the caller,
// alive as long as the instance).
struct TickInputs {
    const world::WorldPackView *view = nullptr;
    const collide::CarShapes *shapes = nullptr;
    collide::CollideOptions collide{};
    // The world's collision stage by name (tw_collision_choice.h: "c" the product default,
    // "b" the reference, or an experiment); the variant builds its own copy of that stage.
    const char *collisionStage = "c";
    const model::CarForceConsts *car = nullptr;
    const model::WaterZone *water = nullptr;     // may be null
    const race::TwRaceMap *race = nullptr;
    step::Policy policy = step::Policy::FvSubsteps;
};

// ---- batch (search): candidates through this variant's batch runner (batch/batch_runner.h) ----
// One job = one candidate: a start state and its controls; the runner calls back after every tick.
struct BatchJob {
    std::uint64_t id = 0u;                        // the caller's tag
    const model::TwinState *start = nullptr;      // valid until the job's done()
    const std::uint32_t *controls = nullptr;      // TwinControl per tick (respawn counts decoded from it)
    std::uint32_t ticks = 0u;
};
// After each tick of a job: what the search's judge reads (as step::Events / the state after the tick).
struct BatchTick {
    std::uint64_t id = 0u;
    std::uint32_t tick = 0u;                      // tick index from the job's start (0 = its first tick)
    float pos[3], vel[3], quat[4];
    std::uint32_t tickMs = 0u;
    bool raceCompleted = false;
    bool finished = false;                        // sticky from the tick that completed the race (Events::finished)
    std::uint64_t finishLowerNs = 0u;             // ... and that tick's finish bracket (Events::finishLowerNs)
    const model::TwinState *state = nullptr;      // the full state (BatchPolicy::fullState only, tests)
};
struct BatchCallbacks {
    void *user = nullptr;
    bool (*next)(void *user, BatchJob *job) = nullptr;               // false: no job left
    bool (*evaluate)(void *user, const BatchTick &tick) = nullptr;   // false stops the job
    void (*done)(void *user, std::uint64_t id, std::uint32_t ticks) = nullptr;
    int laneLimit = 0;                                               // > 0: at most this many jobs in flight
};
struct BatchPolicy {
    int lanes = 16;                               // 16 or 8
    int forcesMinLanes = 3;                       // batch::Policy: wide forces + integration from this many lanes
    int respondMinLanes = 4;                      // ... wide response from this many touching lanes
    bool fullState = false;                       // BatchTick::state (an AoS copy per tick: tests only)
};
struct BatchStats {
    std::uint64_t jobs = 0u, ticks = 0u, laneSubsteps = 0u, passes = 0u;
    std::uint64_t forcesWide = 0u, forcesNarrowLanes = 0u, respondWide = 0u, respondNarrowLanes = 0u;
    std::uint64_t freeProofs = 0u, freeProven = 0u, freeSkipped = 0u, freeEscapes = 0u, freeViolations = 0u;   // batch::Policy::freeTicks
};

// sizeof / alignof of every type that crosses the boundary.
struct Layout {
    std::uint32_t abi = 0u;
    std::uint32_t words[34]{};
    bool operator==(const Layout &o) const;
};

struct TickTable {
    Kind kind;
    const char *name;
    Layout (*layout)();
    // Builds the variant's own collision stage and tick model; null on
    // failure with a static message in *diagnostic.
    void *(*create)(const TickInputs &inputs, const char **diagnostic);
    void (*destroy)(void *instance);
    // `cache`: a per-candidate collision cache from this variant's newCache
    // (one from another variant or the baseline is ignored by the stage:
    // its kind tag differs). Results never depend on it.
    step::Status (*step)(const void *instance, model::TwinState &state, step::Control control, step::Events *events,
                         step::CollisionCache *cache);
    std::uint32_t (*stepTicks)(const void *instance, model::TwinState &state, const step::Control *controls,
                               std::size_t count, step::StopRule stop, step::Events *events,
                               const std::uint32_t *respawnCounts, step::CollisionCache *cache);
    // The variant's own per-candidate cache (null if the stage has none);
    // released with destroyCache.
    step::CollisionCache *(*newCache)(const void *instance);
    void (*destroyCache)(step::CollisionCache *cache);
    // Batch runner of this variant over the instance's packs (null: the policy is not supported);
    // runBatch pulls jobs until next() returns false. The callbacks run in the caller's FP mode.
    void *(*newBatch)(const void *instance, const BatchPolicy &policy);
    void (*destroyBatch)(void *batch);
    void (*runBatch)(void *batch, const BatchCallbacks &callbacks, BatchStats *stats);
    // Hash over the ISA-sensitive kernels on fixed synthetic data (lane
    // math at every width, exact-fast test incl. AVX-512 packets): equal in
    // every variant of a correct build.
    std::uint64_t (*selftest)();
};

// The table compiled for `kind`, or null if this build has no such variant.
const TickTable *Table(Kind kind);
// Whether this CPU (and OS) can run `kind`.
bool Supported(Kind kind);
// The kind a new session uses: the override if set (and supported), else
// the best supported variant whose layout matches the baseline's.
Kind Select(const char **why = nullptr);
// Test hook: "baseline", "sse2" (= baseline), "avx2", "avx512", "auto" or
// null (back to FOREVERVALIDATOR_TWIN_ISA / CPUID).
void SetOverride(const char *name);

}  // namespace isa
}  // namespace tw

#endif
