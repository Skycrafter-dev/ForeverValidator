// Twin GPU engine (CUDA): host interface. FV-free; no CUDA headers.
//
// The GPU runs the twin's scalar (L = 1) path, one candidate per thread, with the same IEEE binary32
// operation sequence as the CPU twin (no FMA, correctly rounded division and square root, no
// flush-to-zero), so its states are bit-identical to the CPU's (docs/twin/gpu.md).
//
// Stage checks (RunForces, ...) run one stage of many recorded substeps on the device, for
// comparison with the CPU stage that produced them.
#ifndef TW_GPU_H
#define TW_GPU_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "forces_io.h"
#include "response.h"
#include "tw_race.h"
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

struct HostWorld;

// The device the engine uses (device 0), or why there is none.
bool DeviceInfo(std::string *info);

// Device copies of one world's car packs: CarForceConsts, WaterZone (and its cells), TwRaceMap (and
// its triggers and spawns), with the device pointers patched in.
class CarPacks {
public:
    CarPacks();
    ~CarPacks();
    CarPacks(const CarPacks &) = delete;
    CarPacks &operator=(const CarPacks &) = delete;
    bool Upload(const model::CarForceConsts &car, const model::WaterZone *water, const race::TwRaceMap &race,
                std::string *error);

    struct Dev;
    const Dev *dev() const { return dev_.get(); }

private:
    std::unique_ptr<Dev> dev_;
};

// One substep's forces and integration, as TwinStages::Forces + Integrate on the fv policy:
// CarForces, then CarIntegrateR with the force stage's pose rotation.
struct ForcesCase {
    model::TwinState s;   // the state Forces sees
    model::CarTickIn in;
    float h;
};
struct ForcesResult {
    model::TwinState s;   // after Integrate
    model::CarForceOut f;
};
bool RunForces(const CarPacks &packs, const ForcesCase *cases, std::size_t n, ForcesResult *out, std::string *error);

// One substep's response, as TwinStages::Respond: RespondR on the substep's contacts [first, first + n)
// of the batch's contact array, with the post-integration rotation R and the force stage's pose Rw.
struct RespondCase {
    model::TwinState s;
    model::M3 R, Rw;
    model::RespTick tick;
    std::uint32_t first, n;
};
struct RespondResult {
    model::TwinState s;
    model::RespTick tick;
    model::RespOut out;
};
bool RunRespond(const CarPacks &packs, const RespondCase *cases, std::size_t n, const model::RespContact *contacts,
                std::size_t contactCount, RespondResult *out, std::string *error);

// The race stages: TwinStages::TickBegin's race part (TwRaceTickBegin after clearing the events) and
// RaceHits (TwRaceOnTriggerHits on the substep's trigger hits).
struct TickBeginCase {
    model::TwinState s;
    std::uint32_t control, respawnCount;
};
struct RaceHitsCase {
    model::TwinState s;
    race::TwRaceEvents ev;
    std::uint32_t count;
    race::TwTriggerHit hits[model::kRespMaxContacts];
};
struct RaceResult {
    model::TwinState s;
    race::TwRaceEvents ev;
};
bool RunTickBegin(const CarPacks &packs, const TickBeginCase *cases, std::size_t n, RaceResult *out, std::string *error);
bool RunRaceHits(const CarPacks &packs, const RaceHitsCase *cases, std::size_t n, RaceResult *out, std::string *error);

// C's collision stage ("c") on the device (tw_gpu_collide.cu): the world's FV-ordered triangle tree, rank
// references, instance isos, mesh-local vertices and output records, built from the stage's world pack and
// car shapes.
class CollisionPacks {
public:
    CollisionPacks();
    ~CollisionPacks();
    CollisionPacks(const CollisionPacks &) = delete;
    CollisionPacks &operator=(const CollisionPacks &) = delete;
    // built: BuildHostWorld's result for this view and car, made earlier (null: built here)
    bool Upload(const world::WorldPackView &view, const collide::CarShapes &car, std::string *error,
                const HostWorld *built = nullptr);
    std::size_t Bytes() const;
    // TW_GPU_DETECT_STATS at Upload: the tick's detections' per-warp counters since then (tw_gpu_k_detect.cu DetectAt)
    static constexpr int kDetectStats = 32;   // TW_GPU_DETECT_STATS counters (tw_gpu_k_detect.cu DetectAt)
    bool DetectStats(unsigned long long out[kDetectStats], std::string *error) const;
    static void PrintDetectStats(const unsigned long long a[kDetectStats]);   // one line on stdout

    struct Dev;
    const Dev *dev() const { return dev_.get(); }

private:
    std::unique_ptr<Dev> dev_;
};

// One detection at a car pose (TwinStages::Detect: the post-integration pos, quat, absorb) and its output,
// step::SubstepContacts' fields; rawLost: raw contacts past the device's per-call buffer (the output is then
// not the CPU's).
struct DetectCase {
    float pos[3], quat[4], absorb[4];
};
struct GpuContacts {
    int n;
    bool overflow;
    std::uint32_t rawLost;
    model::RespContact c[model::kRespMaxContacts];
};
bool RunDetect(const CollisionPacks &packs, const DetectCase *cases, std::size_t n, GpuContacts *out, std::string *error);

// Detection benchmark (searchq --gpu-detect): thread i detects at cases[idx[i]] (idx kIdle: the thread idles, a lane of a
// lockstep warp whose candidate has no call there), as the tick does (DetectAt, the contacts in the thread's local
// memory, a word of them written out). `reps` timed launches (best kernel time), then one launch with per-call
// counters and the detection's SM cycles.
constexpr std::uint32_t kIdle = 0xFFFFFFFFu;
struct DetectBenchStats {
    double ms = 0.0;                 // best timed launch
    unsigned long long calls = 0, cycles = 0;   // detections, SM cycles inside them summed over threads
    unsigned long long nodes = 0;        // tree nodes visited
    unsigned long long childTests = 0;   // child boxes tested against the union box
    unsigned long long leafHits = 0;     // ranks whose box meets the union box
    unsigned long long boxPairs = 0;     // (shape, rank) box pairs
    unsigned long long planePass = 0;    // pairs past the plane prefilter (narrow phase calls)
    unsigned long long raws = 0;         // raw contacts
    unsigned long long contacts = 0;     // merged contacts
    unsigned long long warpCycles = 0;   // per warp: the slowest lane's cycles, summed over warps (x 32 = warp thread time)
};
bool BenchDetect(const CollisionPacks &packs, const DetectCase *cases, std::size_t nCases, const std::uint32_t *idx, std::size_t n, int reps,
                 DetectBenchStats *stats, std::string *error);

// The recorded-queue runner (tw_gpu_queue.cu): every candidate's ticks from its start state (TwinStages on the
// fv policy, all worlds in one launch; QueueConfig picks the schedule). Candidate i starts from start[i] and runs
// controls [firstTick, firstTick + ticks); with `after` (the recorded states, same indexing as controls)
// every state is compared and a different one is counted and replaced by the recording. One instrumented
// launch (stage clocks), then the best of `reps` launches (kernel time).
struct QueueWorld {
    const CarPacks *car;
    const CollisionPacks *collision;
};
struct QueueCand {
    std::uint32_t world, firstTick, ticks;
};
struct QueueStats {
    double ms = 0.0;
    std::uint64_t ticks = 0, substeps = 0, differ = 0, rawLost = 0;
    std::uint64_t extraSubsteps = 0;   // the lane runner's finish refinement: replayed and probe substeps
    std::uint32_t lanes = 0;           // candidates in flight (the lane runner's lanes; else the launch's threads)
    unsigned long long stageCycles[7] = {};   // SM cycles summed over threads (an instrumented launch): forces + integration,
                                              // collision, response, tick start, finish snapshots, RunSubstep calls, total
};
// The schedule (identical results either way): the lockstep lane runner (tw_gpu_lockstep.cuh, the default; its pool
// longest candidates first, TW_GPU_ORDER=world|queue to compare) or one candidate per thread running its ticks
// (RunTick, v1). The lane runner's own API (a search's jobs and per-tick hook) is device-side: docs/twin/gpu.md.
struct QueueConfig {
    bool lockstep = true;
    bool barrier = false;   // lockstep: a block's lanes start each pass together
    int block = 0;          // threads per block (0: the default)
    int lanes = 0;          // lockstep: lanes in flight (0: as many as the GPU holds at once)
    int chunk = 0;          // lockstep pool: 0, warp batches (a warp whose lanes are all free takes the next jobs); 1, a
                            // lane alone; 2..31, a batch once that many lanes are free; >= 32, a warp takes this many
};
bool RunQueue(const QueueWorld *worlds, std::size_t nWorlds, const QueueCand *cands, std::size_t nCands, const model::TwinState *start,
              const std::uint32_t *controls, const model::TwinState *after, std::size_t nTicks, int reps, QueueStats *stats,
              std::string *error);
bool RunQueue(const QueueWorld *worlds, std::size_t nWorlds, const QueueCand *cands, std::size_t nCands, const model::TwinState *start,
              const std::uint32_t *controls, const model::TwinState *after, std::size_t nTicks, int reps, const QueueConfig &config,
              QueueStats *stats, std::string *error);

}  // namespace tw::gpu

#endif
