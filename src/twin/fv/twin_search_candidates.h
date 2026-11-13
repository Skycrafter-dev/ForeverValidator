// Twin search (M13): FV's GPU candidate generation on the host.
//
// Same seeds give the same candidates as PhysicsSandboxCudaSearchSession on
// CUDA, HIP and Vulkan: the mutation functions are FV's own CUDA source,
// extracted at build time (extract_cuda_search.py), and the per-candidate
// orchestration below is the kernel's reference ("legacy materialization")
// pipeline, which FV's differential tests hold equal to the optimized GPU
// pipelines (docs/cuda-mutation-generation-performance.md).
//
// FV-dependent (CUDA search types); compiled into forevervalidator_core when
// -DFOREVERVALIDATOR_BUILD_TWIN=ON.
#ifndef FOREVERVALIDATOR_TWIN_SEARCH_CANDIDATES_H
#define FOREVERVALIDATOR_TWIN_SEARCH_CANDIDATES_H

#include <cstdint>
#include <string>
#include <vector>

#include "simulation/backends/cuda/cuda_search_branch_state.cuh"
#include "simulation/backends/cuda/cuda_search_executor.h"
#include "simulation/backends/cuda/cuda_search_winner_selection.cuh"
#include "simulation/backends/cuda/cuda_state_layout.h"

namespace forevervalidator::simulation::twin_search {

// The session's mutation inputs, prepared once as the CUDA executor's
// Create prepares them: inputs split at the mutable boundary (the tick after
// the branch), suffix and windows relative to that boundary.
struct MutationProgram {
    std::vector<CudaSearchInputEvent> immutablePrefix;   // absolute times
    std::vector<CudaSearchInputEvent> baselineInputs;    // materialized mutable suffix, relative times
    std::vector<CudaSearchInputEvent> immutableTail;     // past the materialization horizon, relative
    std::vector<CudaSearchModifierConfiguration> modifiers;  // relative windows
    std::vector<CudaSearchMutationSegment> segments;     // relative
    std::uint32_t changedSegmentCount = 1u;
    std::vector<double> smoothWeights;
    cuda_search_detail::DeviceControlState mutableBoundaryControls{};
    std::uint32_t tickDurationMs = 10u;
    std::int64_t branchTimeMs = 0;
    std::int64_t mutableFromTimeMs = 0;
    std::uint32_t eventCapacity = 0u;                    // per candidate (materialized events)
    bool baselineInputsCanonical = false;
};

bool PrepareMutationProgram(const CudaSearchExecutorConfiguration &configuration,
                            MutationProgram *program, std::string *diagnostic);

enum class CandidateStatus : std::uint8_t { Success, Cancelled, CapacityExceeded };

struct GeneratedCandidate {
    CandidateStatus status = CandidateStatus::Success;
    bool active = false;                       // baseline, or at least one effective change
    std::uint32_t mutationCount = 0u;
    std::vector<CudaSearchInputEvent> events;  // suffix, relative times (eventCount = size)
};

// One per worker thread (owns the scratch buffers and the MT19937 state).
class CandidateGenerator {
public:
    explicit CandidateGenerator(const MutationProgram &program);
    ~CandidateGenerator();
    CandidateGenerator(const CandidateGenerator &) = delete;
    CandidateGenerator &operator=(const CandidateGenerator &) = delete;

    // Candidate `candidateId` (ignored for the baseline), exactly as
    // GenerateSearchCandidatesKernel produces it.
    void Generate(std::uint64_t candidateId, bool baseline, GeneratedCandidate *out);

private:
    struct Scratch;
    const MutationProgram &program_;
    Scratch *scratch_;
};

bool MaximizesSearchScore(CudaSearchEvaluatorKind kind, std::uint32_t optionFlags);

// FV's CUDA EvaluateState, verbatim (extracted): the evaluator kinds
// Velocity, Point, Pose, VolumeEntry, CustomVolumeEntry, StuntPoints,
// FinishTime (tick-level; the finish refinement is the caller's) and
// ConditionTime*. It reads state.body.current {position, linearSpeed,
// rotationQuat} and state.race.progress only.
cuda_search_detail::DeviceSample EvaluateSearchState(const CudaSearchEvaluatorConfiguration &evaluator,
                                                     const CudaCandidatePhysicsState &state,
                                                     const GmVec3 &previousPosition, const GmVec3 &previousVelocity,
                                                     double previousTimeMs, double currentTimeMs, std::uint32_t stuntsScore,
                                                     bool *reported);

}  // namespace forevervalidator::simulation::twin_search

#endif
