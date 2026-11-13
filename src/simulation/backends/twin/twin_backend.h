#ifndef FOREVERVALIDATOR_TWIN_BACKEND_H
#define FOREVERVALIDATOR_TWIN_BACKEND_H

// FV side of the twin backend (docs/twin/integration.md). Compiled in every
// build. ReplaySimulationSession runs FV optimized_cpu for loading and the
// prestart, then hands the rest of the timeline to
// ExecuteTwinTimelineSegment, which is defined
//   - here (twin_backend.cpp) when FOREVERVALIDATOR_HAS_TWIN == 0: TwinUnavailable;
//   - in src/twin/fv/twin_timeline.cpp when the twin is built.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <forevervalidator/finish_time.h>
#include <forevervalidator/validation.h>

#include "engine/core/gm_types.h"
#include "simulation/backends/cuda/cuda_search_executor.h"
#include "simulation/control/replay_control_timeline.h"
#include "simulation/runtime/replay_simulation_result.h"
#include "simulation/runtime/replay_trajectory_observation.h"

struct ReplaySimulationInstanceClone;
struct ReplaySimulationDefinition;
class ReplaySimulationSession;
class StaticSceneModelCollection;

namespace forevervalidator::simulation {

struct TwinTimelineSegmentRequest {
    // Exact FV state after the last FV tick (the hand-off, T = prestart).
    const ReplaySimulationInstanceClone *start = nullptr;
    // The ticks the twin runs, in order (observe flags and targets included).
    const ReplayControlTick *ticks = nullptr;
    std::size_t tickCount = 0u;
    std::uint32_t validationSeed = 0u;
    TwinPolicy policy = TwinPolicy::FvSubsteps;
    TwinCollision collision = TwinCollision::Noise;
    // World compile inputs (src/twin/fv/twin_world_build.h): the session's
    // installed static models, its simulation definition (car tuning, water)
    // and the race start spawn the runtime started at.
    const StaticSceneModelCollection *models = nullptr;
    const ReplaySimulationDefinition *definition = nullptr;
    GmIso4 startSpawn{};
    bool haveStartSpawn = false;
};

struct TwinTimelineSegmentResult {
    ReplaySimulationRunResult result = ReplaySimulationRunResult::TwinUnavailable;
    std::vector<ReplayTrajectoryObservation> observations;  // one per observed tick
    std::optional<std::uint32_t> finishTimeMs;
    std::optional<FinishTimeEstimate> finishTime;
    std::uint32_t executedRespawnCount = 0u;
    std::string diagnostic;
};

TwinTimelineSegmentResult ExecuteTwinTimelineSegment(
        const TwinTimelineSegmentRequest &request);

// ---- search (M13, docs/twin/m13-search.md) --------------------------------
// The twin's CPU search behind PhysicsSandboxCudaSearchSession: the CUDA
// search configuration in, CudaSearchBatchExecution out, so the session,
// its Convert() and every caller are shared with CUDA, HIP and Vulkan.

struct TwinSearchOptions {
    std::uint32_t threads = 0u;     // 0: hardware concurrency (the FV delegate uses 1)
    TwinPolicy policy = TwinPolicy::FvSubsteps;
    TwinCollision collision = TwinCollision::Noise;
    bool debugDelegateToOptimizedCpu = false;
    bool debugAllowUnfinishedEngine = false;  // run the twin path before Info().stepImplemented
    bool verifyWinners = true;      // re-check each new best on FV reference
    bool gpu = false;               // the candidates' physics on the twin's GPU engine
    // Twin path: a candidate starts from the baseline's state before its first
    // changed control (same results; false: every candidate from the branch).
    bool branchAtFirstChange = true;
};

// Host-side cost of the last batch (thread-summed milliseconds).
struct TwinSearchHostMetrics {
    double generationMilliseconds = 0.0;    // candidate mutation
    double physicsMilliseconds = 0.0;       // engine ticks (twin, or the FV delegate)
    double evaluationMilliseconds = 0.0;    // per-tick controls + evaluator + candidate bookkeeping
    double selectionMilliseconds = 0.0;     // improvement count, reduction, best assembly
    double verificationMilliseconds = 0.0;  // winner re-check on FV
    double wallMilliseconds = 0.0;
    std::uint32_t threads = 0u;
    std::uint64_t simulatedTicks = 0u;
    std::uint64_t sharedPrefixTicks = 0u;   // ticks judged from the baseline's snapshots instead
    bool winnerVerified = false;            // the new best was re-simulated on FV
    bool winnerVerificationMatched = false; // and FV gave the same score
    double winnerVerifiedScore = 0.0;
    std::string verificationBackend;        // "reference" or "optimized_cpu"
};

struct TwinSearchExecutorSource {
    // The loaded sandbox's session, positioned at the branch. Read during
    // creation only (prepared scene clone, branch state capture).
    const ReplaySimulationSession *session = nullptr;
    // Starts a ClonePrepared() copy of that session the way the sandbox's
    // Restart does (race configuration, StartIncremental on tick 0), inside
    // a DeterministicExecutionScope the caller holds.
    std::function<bool(ReplaySimulationSession &)> startSession;
    // The sandbox's control plan ticks [0, branchCursor + baseline ticks).
    std::vector<ReplayControlTick> timeline;
    std::size_t branchCursor = 0u;
    std::uint32_t validationSeed = 0u;
    // The sandbox's simulation definition (twin world build: car, water).
    std::shared_ptr<const ReplaySimulationDefinition> definition;
    TwinSearchOptions options;
    std::string replayName;   // the sandbox's ReplayIdentity (the search recorder's header)
};

class TwinSearchExecutor {
public:
    virtual ~TwinSearchExecutor() = default;
    virtual CudaSearchBatchExecution EvaluateBaseline(
            const std::function<bool()> &cancellationRequested) noexcept = 0;
    virtual CudaSearchBatchExecution RunBatch(
            std::uint64_t firstCandidateId, std::uint32_t candidateCount,
            const std::function<bool()> &cancellationRequested) noexcept = 0;
    virtual bool ReserveBatchCapacity(std::uint32_t candidateCount,
                                      std::string *diagnostic) noexcept = 0;
    virtual std::uint32_t BatchCapacity() const noexcept = 0;
    virtual bool UpdateChangedSegmentCount(std::uint32_t count) noexcept = 0;
    virtual bool UpdateConditionTimes(double lastImprovementTimeSeconds,
                                      double lastRestartTimeSeconds) noexcept = 0;
    virtual const TwinSearchHostMetrics &LastMetrics() const noexcept = 0;
};

// Defined in src/twin/fv/twin_search_executor.cpp when the twin is built;
// otherwise (twin_backend.cpp) it returns null with TwinNotCompiled's text.
std::unique_ptr<TwinSearchExecutor> CreateTwinSearchExecutor(
        CudaSearchExecutorConfiguration configuration,
        TwinSearchExecutorSource source, std::string *diagnostic);

}  // namespace forevervalidator::simulation

#endif
