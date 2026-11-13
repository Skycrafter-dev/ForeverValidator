// Twin search (GPU engine): the per-candidate host work of a batch on the device (twin_search_device.cu).
//
// For each slot of a chunk, one CUDA thread runs what the batch path's PrepareSlot runs on the host
// (twin_search_executor.cpp): FV's CUDA mutation code (the same functions the host copy uses, extracted at build time
// with their CUDA qualifiers: extract_cuda_search.py --device; the kernel's reference pipeline, as
// CandidateGenerator::Generate), the controls (BuildTwinControls: the control state machine, ControlsFromState and
// the timeline's actions), the first tick that differs from the baseline, and the shared prefix's judge (the batch's
// baseline judge, BuildPrefix). Then the physics slots are ordered (first changed tick, then the controls from
// there; duplicates become neighbours and are folded onto the first of their run) and become the lane runner's jobs
// (tw_gpu_session.h SearchEngine::SubmitJobs); after the kernel, every slot's outcome is one DeviceSlot.
//
// FV-dependent (CUDA search types); no CUDA headers here. Compiled with -DFOREVERVALIDATOR_BUILD_TWIN_GPU=ON.
#ifndef FOREVERVALIDATOR_TWIN_SEARCH_DEVICE_H
#define FOREVERVALIDATOR_TWIN_SEARCH_DEVICE_H

#include <cstdint>
#include <memory>
#include <string>

#include "tw_gpu_session.h"
#include "twin_search_candidates.h"

namespace forevervalidator::simulation::twin_search {

// One slot's outcome (the host builds its SlotResult from it). status: SlotStatus' values (0 success, 2 capacity
// exceeded, 3 unsupported); the judge fields are valid with kSlotJudge.
struct DeviceSlot {
    double score, timeMs, detail0, detail1;
    std::uint32_t evaluationTick, eventCount, mutationCount;
    std::uint32_t job;     // kSlotPose / kSlotHost: the job whose result the slot takes
    std::uint32_t first;   // the first changed tick
    std::uint8_t status, flags, valid, preciseFinish;
};
enum : std::uint8_t {
    kSlotActive = 1u,    // the baseline or an effective change (SlotResult::active)
    kSlotJudge = 2u,     // the judge fields are the slot's best (valid, score, ...)
    kSlotPose = 4u,      // Pose: the host re-judges the job's near-best ticks
    kSlotHost = 8u,      // the device run is not the CPU's (lost contacts, Pose overflow): again on the CPU
    kSlotPhysics = 16u,  // ran on the lane runner (its own job or its run's first)
};

// The session's device side of generation: the mutation program, the timeline's actions and the baseline's
// controls, uploaded once; per job set (SearchEngine::Sets) the chunk's buffers.
class DeviceGeneration {
public:
    DeviceGeneration();
    ~DeviceGeneration();
    DeviceGeneration(const DeviceGeneration &) = delete;
    DeviceGeneration &operator=(const DeviceGeneration &) = delete;

    // actionBits[t]: tick t's TwinControl bits from the timeline's actions (ToTwinControl with no controls);
    // snap: the baseline's controls (first-change comparison, snapCount ticks at most); initial: a judge that has
    // seen no tick (TickJudge::Export after the branch); pose: the jobs' results go to the host too (Pose re-judging).
    bool Init(const MutationProgram &program, std::int64_t branchTimeMs, const std::uint32_t *actionBits, std::uint32_t ticks,
              const std::uint32_t *snap, std::uint32_t snapCount, const tw::gpu::JudgeState &initial, bool pose, std::string *error);
    // The sets in use (2..SearchEngine::kMaxSets; 2 by default): Reserve sizes those. Before the first Reserve.
    void SetSets(int sets);
    // Every set's buffers for chunks of up to `slots` (allocation synchronizes the device: before a batch's first launch).
    bool Reserve(std::size_t slots, std::string *error);

    // Chunk of `n` slots (ids firstId..) on job set `set`'s stream: generation, controls, prefix, order, dedupe, jobs.
    // prefix: the batch's baseline judge after every tick (prefixCount), prefixStop: the tick of its first verdict
    // other than Continue (UINT32_MAX: none), stopStatus: that verdict as a SlotStatus value.
    bool Generate(int set, void *stream, std::uint64_t firstId, std::uint32_t n, bool baseline, std::uint32_t changedSegmentCount,
                  const tw::gpu::JudgeState *prefix, std::uint32_t prefixCount, std::uint32_t prefixStop, std::uint8_t stopStatus,
                  std::string *error);
    // The chunk's jobs for SearchEngine::SubmitJobs (device pointers; the count is on the device).
    const tw::gpu::SearchJob *Jobs(int set) const;
    const std::uint32_t *Controls(int set) const;
    const std::uint32_t *JobCount(int set) const;
    // After the lane kernel (same stream): every slot's outcome to the host (and, for Pose, the jobs' results).
    bool Finish(int set, void *stream, const tw::gpu::SearchResult *results, bool pose, std::string *error);
    // Valid once the set's stream is done (SearchEngine::Wait).
    const DeviceSlot *Slots(int set) const;
    const tw::gpu::SearchResult *Results(int set) const;   // Pose only
    std::uint32_t PhysicsSlots(int set) const;   // slots that needed the lane runner (with their duplicates)
    std::uint32_t JobsRun(int set) const;        // jobs the lane runner ran
    float GenerationMs(int set) const;           // the chunk's generation kernels (from the first to the jobs)
    float OutcomesMs(int set) const;             // from the jobs to the outcomes copied (the lanes, the outcomes kernel, the copies)
    std::size_t SetBytes() const;                // one set's device buffers (after Reserve)
    std::uint32_t GeneratorThreads() const;

    struct Dev;

private:
    std::unique_ptr<Dev> dev_;
};

}  // namespace forevervalidator::simulation::twin_search

#endif
