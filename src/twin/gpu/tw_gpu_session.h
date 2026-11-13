// Twin GPU engine: the search session's engine (tw_gpu_session.cu). FV-free; no CUDA headers.
//
#ifndef TW_GPU_SESSION_H
#define TW_GPU_SESSION_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "tw_state.h"

namespace tw::model {
struct CarForceConsts;
struct WaterZone;
}  // namespace tw::model
namespace tw::race {
struct TwRaceMap;
}
namespace tw::world {
struct WorldPackView;
}
namespace tw::collide {
struct CarShapes;
}

namespace tw::gpu {

// The twin search session (src/twin/fv/twin_search_executor.cpp) on the device: a batch's candidates one per thread,
// each from its start state (the branch or a baseline snapshot) over its controls from its first changed tick, the
// session's per-tick judge after every tick (TickJudge::OnTick: FV's CUDA EvaluateState and the finish refinement)
// with its early stop. The judge's arithmetic is the host's (IEEE double, no FMA) except Pose's acos (device acos is
// not the host's libm): the device keeps the ticks whose Pose score is within a margin of its best, and the host
// re-judges those (exact). FV-free: the evaluator kind is CudaSearchEvaluatorKind's value.
enum class JudgeKind : std::uint32_t { Velocity = 0u, Point = 1u, Pose = 2u, VolumeEntry = 3u, StuntPoints = 4u, FinishTime = 5u };
struct JudgeConfig {                 // per batch
    std::uint32_t kind = 0u;         // JudgeKind
    std::uint32_t optionFlags = 0u;
    double values[16] = {};   // CudaSearchEvaluatorConfiguration::values
    std::int64_t branchTimeMs = 0, tickDurationMs = 10, evaluationStartTimeMs = 0, prestartDurationMs = 0;
    std::uint32_t maximize = 0u;
    std::uint32_t incumbentPrecise = 0u;    // the batch's incumbent (FinishTime pruning)
    double incumbentTimeMs = 0.0;
    std::uint64_t incumbentUpperNs = ~0ull;
    // The targets tw_gpu_judge.cuh evaluates with FV's own code (kind then = FV's CudaSearchEvaluatorKind, any of them):
    // set `targets` and the engine judges every tick there with FV's evaluator configuration and condition program
    // (SetTargets' device copies), the search loop's clock, the race's constants and, scripted, the jobs' objectives
    // (Scripted).
    std::uint32_t targets = 0u;
    std::uint32_t conditionCount = 0u;
    std::uint32_t requiredCheckpoints = 0u, requiredLaps = 0u, playMode = 0u;
    double clock[3] = {};   // last improvement, last restart, now (s)
    const void *evaluator = nullptr;   // set by Submit
    const void *condition = nullptr;   // set by Submit
    double *scripted = nullptr;        // set by Submit
};
// The tick before the judged one: its body and race counters (the targets' previous position, velocity, body.temporary
// and CheckpointProgressBefore).
struct PreviousTick {
    float position[3], velocity[3], angular[3], quat[4];
    std::uint32_t counters[4];   // lap checkpoints, checkpoint events, laps, finishes
};
constexpr std::size_t kScriptedDoubles = 32u;   // a job's objective scores then metric values (16 each)
// A candidate's judge without its identity (TickJudge: best sample, reported, evaluation index) and the previous
// tick's position.
struct JudgeState {
    double score = 0.0, timeMs = 0.0, detail0 = 0.0, detail1 = 0.0;
    std::uint32_t evaluationTick = 0u, evaluationIndex = 0u;
    std::uint8_t valid = 0u, preciseFinish = 0u, reported = 0u, pad = 0u;
    float previous[3] = {};
};
enum JudgeVerdict : std::uint8_t { kJudgeContinue = 0u, kJudgeDone = 1u, kJudgeInvalid = 2u, kJudgeUnsupported = 3u };
struct SearchJob {
    std::uint32_t start = 0u;      // index into the engine's start states
    std::uint32_t controls = 0u;   // first control in the batch's control array
    std::uint32_t ticks = 0u;      // at most this many ticks
    std::uint32_t firstTick = 0u;  // the job's first tick in the session's timeline (TickJudge's t)
    JudgeState judge;              // after the shared prefix
};
constexpr int kPoseNear = 6;
struct PoseNear {                  // a tick whose device Pose score is near the device best: the host re-judges it
    std::uint32_t t, evaluationIndex;
    float pos[3], quat[4];
};
struct SearchResult {
    JudgeState judge;
    std::uint32_t ticks = 0u;      // ticks simulated
    std::uint8_t verdict = 0u;     // JudgeVerdict of the last tick
    std::uint8_t lost = 0u;        // raw contacts lost (device buffer): rerun on the CPU
    std::uint8_t poseCount = 0u;
    std::uint8_t poseOverflow = 0u;   // more near-best Pose ticks than kPoseNear: rerun on the CPU
    PoseNear pose[kPoseNear];
};
struct SearchStats {
    double kernelMs = 0.0, wallMs = 0.0;   // the kernel; Submit to Collect's return
    std::uint64_t ticks = 0u, substeps = 0u;
    unsigned long long clk[8] = {};   // FOREVERVALIDATOR_TWIN_GPU_CLOCKS: SM cycles summed over threads: forces +
                                      // integration, collision, response, tick start, finish snapshots, substeps, the
                                      // judge, the whole thread
};
// One session's device engine: its world's packs, its start states; jobs run in batches.
class SearchEngine {
public:
    SearchEngine();
    ~SearchEngine();
    SearchEngine(const SearchEngine &) = delete;
    SearchEngine &operator=(const SearchEngine &) = delete;
    bool Init(const model::CarForceConsts &car, const model::WaterZone *water, const race::TwRaceMap &race,
              const world::WorldPackView &view, const collide::CarShapes &shapes, const model::TwinState *starts,
              std::size_t startCount, std::string *error);
    // Asynchronous job sets (Sets() in flight, each on its own stream, so a set's kernel overlaps the host's work on
    // the next and the previous set's tail): fill Jobs(set, n) and Controls(set, count) (pinned host memory, valid
    // until the set's next Jobs/Controls call), Submit, later Collect (waits; the results stay valid until the set's
    // next Submit).
    static constexpr int kMaxSets = 8;
    int Sets() const;
    // The sets in use (2..kMaxSets; 2 by default): Reserve and ReserveJobs size those. Before the first Reserve.
    bool SetSets(int sets, std::string *error);
    // Every set's buffers for up to `jobs` jobs and `controls` control words. Allocation synchronizes the device: call it
    // before the batch's first Submit, so no later Submit waits for a running set.
    bool Reserve(std::size_t jobs, std::size_t controls, std::string *error);
    SearchJob *Jobs(int set, std::size_t n, std::string *error);
    std::uint32_t *Controls(int set, std::size_t count, std::string *error);
    // after >= 0: the kernel starts once set `after`'s kernel has ended (a batch run ahead does not share the device
    // with the batch before it, only follows it without a gap)
    bool Submit(int set, const JudgeConfig &config, std::size_t n, std::size_t controlCount, std::string *error, int after = -1);
    const SearchResult *Collect(int set, SearchStats *stats, std::string *error);
    // JudgeConfig::targets: FV's evaluator configuration (bytes; its around-walls table pointer at `tablesField`, pointed at
    // the device copy of `tables`) and condition program, for every later Submit.
    std::size_t TargetBytes() const;   // sizeof the evaluator configuration the device reads
    bool SetTargets(const void *evaluator, std::size_t evaluatorBytes, std::size_t tablesField, const unsigned char *tables,
                    std::size_t tableBytes, const void *condition, std::size_t conditionBytes, std::string *error);
    // The device copy of the tables: read (the bits queries marked missed) and written back (corners filled on the host).
    bool ReadTables(unsigned char *host, std::size_t offset, std::size_t bytes, std::string *error);
    bool WriteTables(const unsigned char *host, std::size_t offset, std::size_t bytes, std::string *error);
    // Scripted targets: the jobs' objectives (kScriptedDoubles each) before the set's next Submit, after its Collect.
    double *Scripted(int set, std::size_t n, std::string *error);
    std::size_t WorldBytes() const;

    // Jobs built on the device (the search's generation on the device, twin_search_device.cu): the caller's kernels write
    // the jobs, their controls and their count on the set's stream (Stream), then SubmitJobs runs them; the results stay
    // on the device (DeviceResults) until Wait. ReserveJobs: every set's lanes and results for up to `jobs` jobs (as
    // Reserve, before the batch's first launch).
    bool ReserveJobs(std::size_t jobs, std::string *error);
    void *Stream(int set) const;
    // The stream for the work after a set's lanes (its outcomes' kernel, the copies to the host): with
    // FOREVERVALIDATOR_TWIN_SEARCH_GPU_POST=1 (measurement) a stream of the greatest priority that waits for the set's lane
    // kernel, so that work does not queue behind the blocks of batches run ahead; else Stream(set). Wait waits for both.
    void *PostStream(int set) const;
    // After the work queued on PostStream(set) for the set's batch (its outcomes): a later SubmitJobs(..., after = set) waits for
    // it instead of the set's lane kernel's end.
    void MarkPost(int set);
    std::size_t SetBytes() const;   // one set's lane rows and results (after ReserveJobs)
    bool SubmitJobs(int set, const JudgeConfig &config, const SearchJob *jobs, const std::uint32_t *controls, const std::uint32_t *count,
                    std::size_t maxJobs, std::string *error, int after = -1);
    const SearchResult *DeviceResults(int set) const;
    bool Wait(int set, SearchStats *stats, std::string *error);   // the set's stream done; its kernel's stats
    bool Ready(int set) const;                                     // the set's work is done (Wait would not wait)
    bool LanesDone(int set) const;                                 // the set's lane kernel has ended

    struct Dev;

private:
    std::unique_ptr<Dev> dev_;
};

}  // namespace tw::gpu

#endif
