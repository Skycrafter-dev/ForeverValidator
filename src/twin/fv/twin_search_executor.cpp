// Twin search executor (M13, docs/twin/m13-search.md).
//
// Implements FV's GPU search contract on CPU threads, behind
// PhysicsSandboxCudaSearchSession (validation_api.cpp):
//   - candidates: FV's CUDA mutation code on the host (twin_search_candidates);
//   - per tick: the CUDA kernel's control state machine (DeviceControlState
//     over the candidate's events, baseline control ticks), the step, then
//     FV's CUDA EvaluateState (extracted) on the stepped state;
//   - FinishTime: the precise finish estimate replaces the tick-level score,
//     with the CUDA refinement's incumbent pruning;
//   - winner: StrictlyBetter / BetterSample from cuda_search_winner_selection
//     in logical order (incumbent 0, then slots, then ticks), as the GPU's
//     candidate-best reduction and FinalizeSearchBatchKernel;
//   - every new best re-simulated from tick 0 on FV reference (exact), which
//     also provides the winner's captured state.
// The step is the twin (tw::step via TwinBatchRunner) or, with
// debugDelegateToOptimizedCpu, FV optimized_cpu through the same loop.
//
// Twin path, branchAtFirstChange (the default): the baseline is simulated once
// at creation without stopping, keeping its state and events after every tick.
// A candidate's ticks before its first changed control are the baseline's, so
// the judge reads them from those snapshots, and the twin starts from the
// snapshot before the first changed tick. The results are the same bits as
// running every candidate from the branch (the step depends only on the state
// and the control; the collision cache never changes results).
//
// Twin path, batches (docs/twin/search-throughput.md): a batch runs in phases on a persistent pool
// (the session's threads): (A) every slot is generated, its controls built and its shared prefix
// judged; (B) the slots that need physics are deduplicated (same first changed tick and same
// controls from there: the same trajectory) and ordered longest first; (C) each worker's 16-lane
// batch runner (the world's ISA variant, tw_isa.h runBatch) simulates them, the judge running
// after every tick of every lane; (D) each duplicate's own judge replays its original's per-tick
// views. Every slot's result is the one the serial path gives (the runner is bit-identical to
// the single step, the judge sees the same values), so batches do not change. Measurement
// switches: FOREVERVALIDATOR_TWIN_SEARCH_LANES=16 (default) | 8 | 1 (1: the serial path, one
// candidate at a time), FOREVERVALIDATOR_TWIN_SEARCH_DEDUPE=0, FOREVERVALIDATOR_TWIN_SEARCH_ORDER=slot
// (slot order instead of longest first). The recorder and FOREVERVALIDATOR_TWIN_SEARCH_CHECK use
// the serial path.
//
// Recorder (opt-in, off unless FOREVERVALIDATOR_TWIN_SEARCH_RECORD is set):
// FOREVERVALIDATOR_TWIN_SEARCH_RECORD=PATH[,n=16][,ticks=200] writes the
// first n candidates the twin runner simulates on worker 0 (the baseline's
// snapshot pass first), each up to `ticks` ticks or until the judge stops it.
// Only the first twin search session of the process records.
// File (little endian): "TWSRCH01", u32 version = 1, u32 candidates,
// u32 bytes per state (448); u16 length + the replay name; u32 the branch
// state's tick time (ms); then per candidate: u64 id, the start TwinState
// (model layout), u32 ticks, and per tick: u32 TwinControl, u32 respawn
// count (exact), the TwinState after the tick.
//
// Compiled into forevervalidator_core when -DFOREVERVALIDATOR_BUILD_TWIN=ON.
#include "simulation/backends/twin/twin_backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <optional>
#include <unordered_map>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <new>
#include <string>
#include <stdexcept>
#include <thread>
#include <type_traits>
#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

#include <forevervalidator/experimental/around_walls_table.h>
#include <forevervalidator/experimental/twin.h>

#include "simulation/backends/cuda/cuda_search_branch_state.cuh"
#include "simulation/backends/cuda/cuda_search_winner_selection.cuh"
#include "simulation/backends/cuda/cuda_state_layout.h"
#include "simulation/runtime/replay_deterministic_execution.h"
#include "simulation/runtime/replay_simulation_session.h"
#include "import.h"
#include "twin_api_internal.h"
#include "twin_search_candidates.h"
#include "twin_world_build.h"
#if FOREVERVALIDATOR_TWIN_GPU
#include "tw_gpu_session.h"
#include "twin_search_device.h"
#include "twin_world_packs.h"
#endif

namespace forevervalidator::simulation {

namespace {

using cuda_search_detail::ApplyControlEvent;
using cuda_search_detail::BetterSample;
using cuda_search_detail::ControlsFromState;
using cuda_search_detail::DeviceControlState;
using cuda_search_detail::DeviceSample;
using cuda_search_detail::InvalidCandidateSlot;
using cuda_search_detail::StrictlyBetter;
using cuda_search_detail::StrictlyDominates;
using cuda_search_detail::StuntsFromState;
using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Stage accounting inside the tick loop uses the time-stamp counter (a few
// ns per read, against ~20 ns for steady_clock), converted with a rate
// measured at session creation.
std::uint64_t Tsc() {
#if defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#else
    return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
#endif
}

double TscPerMillisecond() {
    const auto c0 = Clock::now();
    const std::uint64_t t0 = Tsc();
    while (Ms(c0, Clock::now()) < 5.0) {
    }
    const std::uint64_t t1 = Tsc();
    return static_cast<double>(t1 - t0) / Ms(c0, Clock::now());
}

enum class SlotStatus : std::uint8_t { Success, Cancelled, CapacityExceeded, Unsupported };

// What one candidate produced.
struct SlotResult {
    SlotStatus status = SlotStatus::Success;
    bool active = false;
    std::uint32_t mutationCount = 0u;
    std::vector<CudaSearchInputEvent> events;
    DeviceSample best{};
};

struct StageTimes {  // time-stamp counter ticks
    std::uint64_t generation = 0u, physics = 0u, evaluation = 0u;
    std::uint64_t ticks = 0u;
};

// Persistent worker threads of one session: threads - 1 helpers plus the calling thread, kept
// across batches (no thread creation per batch). Each task runs in an FV deterministic scope.
class WorkerPool {
public:
    explicit WorkerPool(std::uint32_t threads) {
        for (std::uint32_t t = 1u; t < threads; ++t) threads_.emplace_back([this, t] { Loop(t); });
    }
    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto &thread : threads_) thread.join();
    }
    WorkerPool(const WorkerPool &) = delete;
    WorkerPool &operator=(const WorkerPool &) = delete;

    // fn(worker) on every worker (0 = the calling thread); returns when all have returned.
    // Returns false if a helper's task threw (std::bad_alloc and the like).
    bool Run(const std::function<void(std::uint32_t)> &fn) {
        if (threads_.empty()) {
            fn(0u);
            return true;
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            task_ = &fn;
            pending_ = static_cast<std::uint32_t>(threads_.size());
            failed_ = false;
            ++generation_;
        }
        wake_.notify_all();
        bool ok = true;
        try {
            fn(0u);
        } catch (...) {
            ok = false;
        }
        std::unique_lock<std::mutex> lock(m_);
        done_.wait(lock, [&] { return pending_ == 0u; });
        task_ = nullptr;
        if (!ok) throw std::bad_alloc();
        return !failed_;
    }

private:
    void Loop(std::uint32_t t) {
        std::uint64_t seen = 0u;
        for (;;) {
            const std::function<void(std::uint32_t)> *task = nullptr;
            {
                std::unique_lock<std::mutex> lock(m_);
                wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                task = task_;
            }
            bool failed = false;
            try {
                tmnf::simulation::DeterministicExecutionScope scope;
                (*task)(t);
            } catch (...) {
                failed = true;
            }
            std::lock_guard<std::mutex> lock(m_);
            if (failed) failed_ = true;
            if (--pending_ == 0u) done_.notify_one();
        }
    }
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable wake_, done_;
    const std::function<void(std::uint32_t)> *task_ = nullptr;
    std::uint64_t generation_ = 0u;
    std::uint32_t pending_ = 0u;
    bool failed_ = false;
    bool stop_ = false;
};

// The fields of the CUDA state that EvaluateSearchState reads, refreshed
// from the stepped engine state each tick.
struct EvaluationView {
    std::unique_ptr<CudaCandidatePhysicsState> state = std::make_unique<CudaCandidatePhysicsState>();
    std::optional<FinishTimeEstimate> finish;  // absolute (incl. prestart)
    std::uint32_t stuntsScore = 0u;
};

// The engine a candidate runs on.
class CandidatePhysics {
public:
    virtual ~CandidatePhysics() = default;
    virtual bool Begin(std::string *diagnostic) = 0;           // back to the branch state
    virtual bool Step(const ReplayControlTick &tick) = 0;       // one 10 ms tick
    virtual void Read(EvaluationView *view) = 0;                // the stepped state
};

// FV optimized_cpu (the debug delegate) or reference (the verifier).
class FvPhysics final : public CandidatePhysics {
public:
    bool Init(const TwinSearchExecutorSource &source, SimulationBackend backend,
              const std::shared_ptr<const ReplaySimulationInstanceClone> &branch, std::string *diagnostic) {
        session_ = source.session->ClonePrepared(backend);
        if (!session_ || !source.startSession || !source.startSession(*session_)) {
            if (diagnostic) *diagnostic = "twin search: FV session could not be started";
            return false;
        }
        branch_ = branch;
        return true;
    }
    bool Begin(std::string *diagnostic) override {
        if (!branch_) return true;  // the verifier replays from tick 0
        ReplaySimulationInstanceClone clone = *branch_;
        if (!session_->PrepareRuntimeCloneRestore(clone)) {
            if (diagnostic) *diagnostic = "twin search: FV branch state could not be restored";
            return false;
        }
        session_->RestoreRuntimeClone(std::move(clone));
        return true;
    }
    bool Step(const ReplayControlTick &tick) override {
        tick_[0] = tick;
        return session_->AdvanceIncremental(tick_, 0u, 1u).result == ReplaySimulationRunResult::Success;
    }
    void Read(EvaluationView *view) override {
        const std::optional<ReplaySimulationStateView> state = session_->CurrentState();
        if (!state) return;
        CHmsDyna::CHmsStateDyna &frame = view->state->body.current;
        frame.position = state->frame.position;
        frame.linearSpeed = state->frame.linearSpeed;
        frame.rotationQuat = state->frame.rotationQuaternion;
        view->state->race.progress = state->race;
        view->finish = state->finishTime;
        view->stuntsScore = state->stuntsScore.value_or(0u);
    }
    ReplaySimulationSession &Session() { return *session_; }

private:
    std::unique_ptr<ReplaySimulationSession> session_;
    std::shared_ptr<const ReplaySimulationInstanceClone> branch_;
    std::vector<ReplayControlTick> tick_ = std::vector<ReplayControlTick>(1u);
};

// The twin (stub until TwinStep and the FV-to-twin state import exist).
class TwinPhysics final : public CandidatePhysics {
public:
    TwinPhysics(std::shared_ptr<const experimental::twin::TwinWorld> world, experimental::twin::TwinState start)
        : world_(std::move(world)), start_(start) {}
    bool Begin(std::string *) override {
        state_ = start_;
        return true;
    }
    bool Step(const ReplayControlTick &tick) override {
        experimental::twin::TwinEvents events;
        return experimental::twin::TwinStep(*world_, state_, ToTwinControl(tick), &events) ==
               experimental::twin::TwinStatus::Ok;
    }
    void Read(EvaluationView *view) override { ReadState(state_, nullptr, view); }
    static void ReadState(const experimental::twin::TwinState &state, const experimental::twin::TwinEvents *events,
                          EvaluationView *view) {
        // Direct field reads (offsets from the generated state layout); the
        // golden-name lookup of TwinReadField costs ~300 ns per tick here.
        using Fields = tw::model::detail::TwinStateFields;
        auto read = [&](std::size_t offset, void *out, std::size_t bytes) {
            std::memcpy(out, state.bytes + offset, bytes);
        };
        CHmsDyna::CHmsStateDyna &frame = view->state->body.current;
        float v[4];
        read(offsetof(Fields, pos), v, 12);
        frame.position = {v[0], v[1], v[2]};
        read(offsetof(Fields, vel), v, 12);
        frame.linearSpeed = {v[0], v[1], v[2]};
        read(offsetof(Fields, quat), v, 16);
        frame.rotationQuat.x = v[0];
        frame.rotationQuat.y = v[1];
        frame.rotationQuat.z = v[2];
        frame.rotationQuat.w = v[3];
        std::uint8_t completed = 0u;
        read(offsetof(Fields, raceCompleted), &completed, 1);
        view->state->race.progress.raceCompleted = completed != 0u;
        std::uint32_t tickMs = 0u;
        read(offsetof(Fields, tickMs), &tickMs, 4);
        view->state->race.progress.lastPrepareTimeMs = tickMs;  // the tick of a finish (EvaluateState's coarse time)
        view->finish.reset();
        if (events != nullptr && events->finished) {
            // FV's finish estimate is {lower, lower + 1}; the twin keeps the lower bound.
            FinishTimeEstimate finish;
            finish.lowerBoundNs = events->finishLowerNs;
            finish.upperBoundNs = events->finishLowerNs + 1u;
            finish.estimatedNs = finish.upperBoundNs;
            view->finish = finish;
        }
        ReadFull(experimental::twin::internal::ToModel(state), events != nullptr ? events->cpLastSlot : 0xFFFFFFFFu, view);
    }
    // The rest of what condition programs and the checkpoint, distance and scripted targets read: the
    // twin's fields under their FV names (tools/twin/twin_fv_state.cpp). The local velocity is the body
    // frame's (FV's goes through the corpus iso: the same for the cars the twin runs).
    static void ReadFull(const tw::model::TwinState &s, std::uint32_t cpLastSlot, EvaluationView *view) {
        CudaCandidatePhysicsState &state = *view->state;
        CHmsDyna::CHmsStateDyna &frame = state.body.current;
        frame.angularSpeed = {s.angVel[0], s.angVel[1], s.angVel[2]};
        frame.rotation.Set(frame.rotationQuat);
        state.vehicle.frameHistory.physicsCurrent.localLinearSpeed = LocalSpeed(frame);
        CudaVehicleState &vehicle = state.vehicle;
        vehicle.controls.forcedLowSpeedFriction = s.freewheel != 0u;
        vehicle.contacts.lateralSlowDownContactActive = s.latContact != 0u;
        vehicle.wheels.count = 4u;
        for (std::uint32_t i = 0u; i < 4u; ++i) {
            auto &wheel = vehicle.wheels.values[i].realTime;
            wheel.contactPresent = s.contact[i] != 0u;
            wheel.slipping = s.slipping[i] != 0u;
            wheel.contactMaterial = static_cast<decltype(wheel.contactMaterial)>(s.surface[i]);
        }
        vehicle.engine.useLowSpeedGateB = s.reverse != 0u;
        vehicle.engine.gearIndex = s.gear;
        vehicle.engine.engineInputMemory = s.rpm;
        vehicle.turbo.type = static_cast<decltype(vehicle.turbo.type)>(s.trbType);
        vehicle.turbo.type2Phase = static_cast<decltype(vehicle.turbo.type2Phase)>(s.trbPhase);
        vehicle.turbo.impulseScale = s.trbScale;
        ReplayRaceProgress &race = state.race.progress;
        race.checkpointCount = s.cpCount;
        race.currentLapCheckpointCount = s.lapCpCount;
        race.totalCheckpointEventCount = s.cpEvents;
        race.completedLapCount = s.laps;
        race.finishCount = s.finishCount;
        race.lastAcceptedBlockId = cpLastSlot;   // a slot (twin_search::SearchCheckpointEventAccepted)
        state.race.checkpointSlotsPassed.words[0] = s.cpSlots[0];
        state.race.checkpointSlotsPassed.words[1] = s.cpSlots[1];
    }
    static GmVec3 LocalSpeed(const CHmsDyna::CHmsStateDyna &frame) {
        const auto dot = [](const GmVec3 &a, const GmVec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; };
        return {dot(frame.linearSpeed, frame.rotation.basisX), dot(frame.linearSpeed, frame.rotation.basisY),
                dot(frame.linearSpeed, frame.rotation.basisZ)};
    }
    // The same fields from the batch runner's per-tick view (tw_isa.h BatchTick): the same bits.
    static void ReadTick(const tw::isa::BatchTick &t, EvaluationView *view) {
        CHmsDyna::CHmsStateDyna &frame = view->state->body.current;
        frame.position = {t.pos[0], t.pos[1], t.pos[2]};
        frame.linearSpeed = {t.vel[0], t.vel[1], t.vel[2]};
        frame.rotationQuat.x = t.quat[0];
        frame.rotationQuat.y = t.quat[1];
        frame.rotationQuat.z = t.quat[2];
        frame.rotationQuat.w = t.quat[3];
        view->state->race.progress.raceCompleted = t.raceCompleted;
        view->state->race.progress.lastPrepareTimeMs = t.tickMs;
        view->finish.reset();
        if (t.finished) {
            FinishTimeEstimate finish;
            finish.lowerBoundNs = t.finishLowerNs;
            finish.upperBoundNs = t.finishLowerNs + 1u;
            finish.estimatedNs = finish.upperBoundNs;
            view->finish = finish;
        }
        if (t.state != nullptr) ReadFull(*t.state, t.cpLastSlot, view);   // BatchPolicy::fullState
    }
    static experimental::twin::TwinControl ToTwinControl(const ReplayControlTick &tick) {
        return ToTwinControl(tick.controls, tick.actions);
    }
    static experimental::twin::TwinControl ToTwinControl(const ReplayVehicleControlState &controls,
                                                         const ReplayRaceTransitionActions &actions) {
        std::uint32_t c = static_cast<std::uint32_t>(static_cast<std::int32_t>(controls.steering * 65536.0f)) & 0x3FFFFu;
        if (controls.lowSpeedGateA > 0.0f) c |= 1u << 18;
        if (controls.lowSpeedGateB > 0.0f) c |= 1u << 19;
        const std::uint32_t respawns = std::min<std::uint32_t>(actions.respawnAtCheckpointCount, 255u);
        c |= respawns << 20;
        if (actions.respawnAtCheckpointCount >= 255u) c |= 1u << 28;
        if (actions.resetAtRaceStart) c |= 1u << 29;
        if (actions.enableRaceSimulation) c |= 1u << 30;
        if (actions.establishRaceSpawn) c |= 1u << 31;
        return c;
    }

private:
    std::shared_ptr<const experimental::twin::TwinWorld> world_;
    experimental::twin::TwinState start_;
    experimental::twin::TwinState state_{};
};

// FOREVERVALIDATOR_TWIN_SEARCH_RECORD (format at the top of this file).
class SearchRecorder {
public:
    static std::unique_ptr<SearchRecorder> FromEnvironment(const std::string &replayName, std::uint32_t branchTickMs,
                                                           std::string *diagnostic) {
        const char *value = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_RECORD");
        if (value == nullptr || *value == '\0') return nullptr;
        static std::atomic<bool> claimed{false};   // the first twin search session of the process records
        if (claimed.exchange(true)) return nullptr;
        auto r = std::unique_ptr<SearchRecorder>(new SearchRecorder());
        std::string spec(value);
        std::size_t comma = spec.find(',');
        r->path_ = spec.substr(0, comma);
        while (comma != std::string::npos) {
            const std::size_t next = spec.find(',', comma + 1u);
            const std::string item = spec.substr(comma + 1u, next == std::string::npos ? std::string::npos : next - comma - 1u);
            if (item.rfind("n=", 0) == 0) r->maxCandidates_ = static_cast<std::uint32_t>(std::strtoul(item.c_str() + 2, nullptr, 10));
            else if (item.rfind("ticks=", 0) == 0) r->maxTicks_ = static_cast<std::uint32_t>(std::strtoul(item.c_str() + 6, nullptr, 10));
            else if (!item.empty() && diagnostic) *diagnostic = "search recorder: unknown item '" + item + "' ignored";
            comma = next;
        }
        r->replay_ = replayName.substr(0, 0xFFFFu);
        r->branchTickMs_ = branchTickMs;
        return r;
    }
    ~SearchRecorder() { Write(); }

    bool Active() const { return !written_ && recorded_ < maxCandidates_; }
    void Begin(std::uint64_t id, const experimental::twin::TwinState &start) {
        open_ = Active();
        if (!open_) return;
        candidateAt_ = bytes_.size();
        Put(id);
        PutState(start);
        countAt_ = bytes_.size();
        Put(std::uint32_t{0u});
        ticks_ = 0u;
    }
    void Tick(experimental::twin::TwinControl control, std::uint32_t respawns, const experimental::twin::TwinState &after) {
        if (!open_ || ticks_ >= maxTicks_) return;
        Put(control);
        Put(respawns);
        PutState(after);
        ++ticks_;
        std::memcpy(bytes_.data() + countAt_, &ticks_, 4u);
    }
    void End() {
        if (!open_) return;
        open_ = false;
        if (++recorded_ >= maxCandidates_) Write();
    }

private:
    SearchRecorder() = default;
    template <typename T> void Put(T v) {
        const auto *p = reinterpret_cast<const std::uint8_t *>(&v);
        bytes_.insert(bytes_.end(), p, p + sizeof v);
    }
    void PutState(const experimental::twin::TwinState &s) {
        const tw::model::TwinState m = experimental::twin::internal::ToModel(s);
        const auto *p = reinterpret_cast<const std::uint8_t *>(&m);
        bytes_.insert(bytes_.end(), p, p + sizeof m);
    }
    void Write() {
        if (written_) return;
        written_ = true;
        if (open_) bytes_.resize(candidateAt_);   // drop an unfinished candidate
        open_ = false;
        if (recorded_ == 0u) return;
        std::FILE *f = std::fopen(path_.c_str(), "wb");
        if (f == nullptr) return;
        std::vector<std::uint8_t> head;
        std::swap(head, bytes_);
        Put(std::uint32_t{1u});
        Put(recorded_);
        Put(static_cast<std::uint32_t>(sizeof(tw::model::TwinState)));
        Put(static_cast<std::uint16_t>(replay_.size()));
        bytes_.insert(bytes_.end(), replay_.begin(), replay_.end());
        Put(branchTickMs_);
        std::fwrite("TWSRCH01", 1u, 8u, f);
        std::fwrite(bytes_.data(), 1u, bytes_.size(), f);
        std::fwrite(head.data(), 1u, head.size(), f);
        std::fclose(f);
    }
    std::string path_;
    std::string replay_;
    std::uint32_t branchTickMs_ = 0u;
    std::uint32_t maxCandidates_ = 16u;
    std::uint32_t maxTicks_ = 200u;
    std::uint32_t recorded_ = 0u;
    std::uint32_t ticks_ = 0u;
    std::size_t candidateAt_ = 0u, countAt_ = 0u;
    bool open_ = false;
    bool written_ = false;
    std::vector<std::uint8_t> bytes_;
};

class TwinSearchExecutorImpl final : public TwinSearchExecutor {
public:
    ~TwinSearchExecutorImpl() override {
        if (checkBranching_) {
            std::fprintf(stderr, "twin search check: %llu candidates run again from the branch, %llu different\n",
                         static_cast<unsigned long long>(checked_.load()),
                         static_cast<unsigned long long>(checkDifferent_.load()));
        }
        pool_.reset();
        workers_.clear();   // their lane runners before the world they point into
    }
    bool Create(CudaSearchExecutorConfiguration configuration, TwinSearchExecutorSource source,
                std::string *diagnostic) {
        auto fail = [&](std::string message) {
            if (diagnostic) *diagnostic = "twin search: " + message;
            return false;
        };
        config_ = std::move(configuration);
        source_ = std::move(source);
        if (source_.session == nullptr || !source_.startSession) return fail("no FV session");
        const CudaSearchEvaluatorKind kind = config_.evaluator.kind;
        // the judge's reads beyond the body and the finish (TwinPhysics::ReadFull)
        fullState_ = kind == CudaSearchEvaluatorKind::Scripted || kind == CudaSearchEvaluatorKind::CheckpointEvent ||
                     kind == CudaSearchEvaluatorKind::Distance || config_.condition.has_value();
        if (fullState_ && source_.options.debugDelegateToOptimizedCpu) {
            return fail("the FV delegate judges velocity, point, pose, volume, stunt and finish targets only");
        }
        // programs that read the candidate's iteration judge every candidate's shared prefix again
        const auto readsIteration = [](const CudaSearchConditionInstruction *instructions, std::size_t count) {
            for (std::size_t i = 0; i < count; ++i) {
                if ((instructions[i].opcode == CudaSearchConditionOpcode::Scalar ||
                     instructions[i].opcode == CudaSearchConditionOpcode::Vector) &&
                    instructions[i].value == CudaSearchConditionValue::Iterations) {
                    return true;
                }
            }
            return false;
        };
        perCandidatePrefix_ =
                (config_.condition &&
                 readsIteration(config_.condition->instructions.data(), config_.condition->instructions.size())) ||
                (kind == CudaSearchEvaluatorKind::Scripted &&
                 readsIteration(config_.evaluator.scriptedInstructions, CudaSearchMaximumScriptedInstructions));
        if (kind == CudaSearchEvaluatorKind::Distance && !config_.aroundWallsTables.empty()) {
            // around walls on the host copy of the tables (EvaluateTick fills their corners)
            aroundWalls_ = config_.aroundWallsTables.data();
            config_.evaluator.aroundWalls = aroundWalls_;
        }
        if (const std::optional<ReplaySimulationStateView> state = source_.session->CurrentState()) race_ = state->race;
        if (config_.evaluationEndTimeMs < config_.evaluationStartTimeMs ||
            config_.evaluationStartTimeMs < config_.branchTimeMs + config_.tickDurationMs ||
            config_.baselineTicks.empty()) {
            return fail("invalid evaluation window");
        }
        evaluationTickCount_ = static_cast<std::uint32_t>(
                (config_.evaluationEndTimeMs - config_.evaluationStartTimeMs) / config_.tickDurationMs + 1);
        timelineTickCount_ = static_cast<std::uint32_t>(config_.baselineTicks.size());
        if (source_.branchCursor + timelineTickCount_ > source_.timeline.size()) {
            return fail("control timeline is shorter than the search window");
        }
        if (config_.incumbent && config_.incumbent->evaluationTick >= evaluationTickCount_) {
            return fail("invalid incumbent seed");
        }
        std::string programDiagnostic;
        if (!twin_search::PrepareMutationProgram(config_, &program_, &programDiagnostic)) {
            return fail(programDiagnostic);
        }
        maximize_ = twin_search::MaximizesSearchScore(kind, config_.evaluator.optionFlags);
        tscPerMs_ = TscPerMillisecond();
        delegate_ = source_.options.debugDelegateToOptimizedCpu;
        // The branch: the sandbox session's current state.
        branch_ = source_.session->CaptureRuntimeClone();
        if (!branch_) return fail("branch state capture failed");
        if (!delegate_) {
            // (QueryTwinEngine().stepImplemented describes tw_step's own Step, not the batch runner the search
            // runs on: the search does not wait for it)
            // The session's TwinWorld and the branch state in twin form
            // (fv/twin_world_build, fv/import).
            if (!source_.definition) return fail("no simulation definition");
            std::string build;
            tw::fv::TwinWorldSource world;
            if (!tw::fv::TwinWorldSourceFromSession(*source_.session, *source_.definition, *branch_, &world, &build)) {
                return fail("world source: " + build);
            }
            world.collision = source_.options.collision;
            replayPlayMode_ = static_cast<std::uint32_t>(world.shortcutMode ? EChallengePlayMode::Shortcut
                                                                            : EChallengePlayMode::Race);
            twinWorld_ = tw::fv::BuildTwinWorld(world, source_.options.policy, tw::fv::TwinWorldLabel{}, &build);
            if (!twinWorld_) return fail("world build: " + build);
            tw::model::TwinState start;
            if (!tw::fv::ImportFvClone(*twinWorld_, *branch_, source_.validationSeed, &start, &build)) {
                return fail("branch state import: " + build);
            }
            twinBranch_ = experimental::twin::internal::FromModel(start);
            std::uint32_t branchTickMs = 0u;
            std::memcpy(&branchTickMs, twinBranch_.bytes + offsetof(tw::model::detail::TwinStateFields, tickMs), 4u);
            recorder_ = SearchRecorder::FromEnvironment(source_.replayName, branchTickMs, nullptr);
            const char *check = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_CHECK");
            checkBranching_ = check != nullptr && std::strcmp(check, "1") == 0;
            threads_ = source_.options.threads != 0u
                               ? source_.options.threads
                               : std::max(1u, std::thread::hardware_concurrency());
        } else {
            threads_ = 1u;  // FV's game random sequence is process-global: one FV thread
        }
        // Workers (delegate: FV optimized_cpu sessions at the branch).
        tmnf::simulation::DeterministicExecutionScope scope;
        if (!scope.Established() && !tmnf::simulation::DeterministicExecutionScope::IsActive()) {
            return fail("deterministic execution mode is unavailable");
        }
        for (std::uint32_t t = 0; t < threads_; ++t) {
            auto worker = std::make_unique<Worker>(program_);
            if (delegate_) {
                auto physics = std::make_unique<FvPhysics>();
                if (!physics->Init(source_, SimulationBackend::OptimizedCpu, branch_, diagnostic)) return false;
                worker->physics = std::move(physics);
            } else if (!InitTwinRunner(*worker, diagnostic)) {
                return false;
            }
            workers_.push_back(std::move(worker));
        }
        if (!delegate_) workers_[0]->recorder = recorder_.get();
        pool_ = std::make_unique<WorkerPool>(threads_);
        if (!delegate_) {
            // the lane runner needs the world's ISA variant (a baseline-only CPU keeps the serial path)
            const char *lanes = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_LANES");
            lanes_ = lanes != nullptr && *lanes != '\0' ? std::atoi(lanes) : 16;
            if (lanes_ != 16 && lanes_ != 8) lanes_ = 1;
            batchPath_ = lanes_ > 1 && twinWorld_->Internal().isa != nullptr;
            const char *dedupe = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_DEDUPE");
            // (a duplicate replays its original's ticks, which carry no full state)
            dedupe_ = !fullState_ && (dedupe == nullptr || std::strcmp(dedupe, "0") != 0);
            const char *order = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_ORDER");
            longestFirst_ = order == nullptr || std::strcmp(order, "slot") != 0;
            const char *laneMin = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_LANE_MIN");
            if (laneMin != nullptr && *laneMin != '\0') laneMin_ = std::max(1, std::atoi(laneMin));
        }
        if (!delegate_ && source_.options.branchAtFirstChange && !BuildSnapshots(diagnostic)) return false;
        if (!delegate_ && !InitGpu(diagnostic)) return false;
        if (source_.options.verifyWinners) {
            verifier_ = std::make_unique<FvPhysics>();
            if (!verifier_->Init(source_, SimulationBackend::Reference, nullptr, diagnostic)) return false;
            // Self-check: FV reference from tick 0 must reach the branch state.
            if (!VerifierToBranch(diagnostic)) return false;
            const std::shared_ptr<const ReplaySimulationInstanceClone> referenceBranch =
                    verifier_->Session().CaptureRuntimeClone();
            if (!referenceBranch || ReplaySimulationInstanceSemanticHash(*referenceBranch) !=
                                            ReplaySimulationInstanceSemanticHash(*branch_)) {
                // The sandbox's state did not come from its own timeline;
                // re-check on optimized_cpu from the captured branch instead.
                verifier_ = std::make_unique<FvPhysics>();
                if (!verifier_->Init(source_, SimulationBackend::OptimizedCpu, branch_, diagnostic)) return false;
                verifierBackend_ = "optimized_cpu";
            }
        }
        // The incumbent seed, as CudaSearchExecutor::Create.
        globalBest_ = DeviceSample{};
        globalBest_.logicalOrder = 0u;
        if (config_.incumbent) {
            const CudaSearchIncumbent &incumbent = *config_.incumbent;
            globalBest_.score = incumbent.score;
            globalBest_.timeMs = incumbent.timeMs;
            globalBest_.detail0 = incumbent.detail0;
            globalBest_.detail1 = incumbent.detail1;
            globalBest_.candidateId = incumbent.candidateId;
            globalBest_.candidateSlot = InvalidCandidateSlot;
            globalBest_.evaluationTick = incumbent.evaluationTick;
            globalBest_.valid = true;
            globalBest_.mutation = incumbent.mutation;
            globalBest_.preciseFinish = incumbent.preciseFinish;
            globalBest_.scriptedObjectiveCount = incumbent.scriptedObjectiveCount;
            for (std::uint32_t i = 0u; i < incumbent.scriptedObjectiveCount; ++i) {
                globalBest_.objectiveScores[i] = incumbent.objectiveScores[i];
                globalBest_.metricValues[i] = incumbent.metricValues[i];
            }
            globalBest_.eventCount = static_cast<std::uint32_t>(program_.baselineInputs.size());
            globalBestEvents_ = program_.baselineInputs;
            globalBestMutationCount_ = incumbent.mutationCount;
            baselineEvaluated_ = true;
        } else {
            std::memset(static_cast<void *>(&globalBest_), 0, sizeof(globalBest_));
        }
        metrics_.verificationBackend = verifierBackend_;
        return true;
    }

    CudaSearchBatchExecution EvaluateBaseline(const std::function<bool()> &cancel) noexcept override {
        if (baselineEvaluated_) {
            CudaSearchBatchExecution result;
            result.status = CudaSearchStatus::InvalidArgument;
            result.diagnostic = "twin search baseline was already evaluated";
            return result;
        }
        return Execute(0u, 1u, true, cancel);
    }

    CudaSearchBatchExecution RunBatch(std::uint64_t first, std::uint32_t count,
                                      const std::function<bool()> &cancel) noexcept override {
        if (count != 0u && first > std::numeric_limits<std::uint64_t>::max() - (count - 1u)) {
            CudaSearchBatchExecution result;
            result.status = CudaSearchStatus::InvalidArgument;
            result.firstCandidateId = first;
            result.candidateCount = count;
            result.diagnostic = "twin candidate ID range overflow";
            return result;
        }
        return Execute(first, count, false, cancel);
    }

    bool ReserveBatchCapacity(std::uint32_t count, std::string *diagnostic) noexcept override {
        if (count == 0u) {
            if (diagnostic) *diagnostic = "twin batch capacity must be positive";
            return false;
        }
        config_.maximumBatchSize = std::max(config_.maximumBatchSize, count);
        return true;
    }
    std::uint32_t BatchCapacity() const noexcept override { return config_.maximumBatchSize; }
    bool UpdateChangedSegmentCount(std::uint32_t count) noexcept override {
        if (count == 0u || count > program_.segments.size()) return false;
        program_.changedSegmentCount = count;
        config_.changedSegmentCount = count;
        ++programVersion_;
        return true;
    }
    bool UpdateConditionTimes(double lastImprovementTimeSeconds, double lastRestartTimeSeconds) noexcept override {
        if (!config_.condition || !std::isfinite(lastImprovementTimeSeconds) || !std::isfinite(lastRestartTimeSeconds)) {
            return false;
        }
        config_.condition->lastImprovementTimeSeconds = lastImprovementTimeSeconds;
        config_.condition->lastRestartTimeSeconds = lastRestartTimeSeconds;
        return true;
    }
    const TwinSearchHostMetrics &LastMetrics() const noexcept override { return metrics_; }

private:
    class TickJudge;

    struct Worker {
        explicit Worker(const twin_search::MutationProgram &program) : generator(program) {}
        twin_search::CandidateGenerator generator;
        twin_search::GeneratedCandidate candidate;
        std::unique_ptr<CandidatePhysics> physics;
        EvaluationView view;
        std::vector<ReplayControlTick> ticks;
        StageTimes times;
        // Twin mode: one batch runner per worker; its evaluator feeds `judge`.
        std::unique_ptr<experimental::twin::TwinBatchRunner> runner;
        TickJudge *judge = nullptr;
        bool unsupported = false;
        std::vector<experimental::twin::TwinControl> controls;  // every tick's control (BuildTicks)
        std::uint32_t tickOffset = 0u;                           // the runner's first tick in the timeline
        std::uint64_t sharedPrefixTicks = 0u;
        SearchRecorder *recorder = nullptr;                      // worker 0 when recording
        std::vector<experimental::twin::TwinCandidate> queue = std::vector<experimental::twin::TwinCandidate>(1u);
        std::vector<experimental::twin::TwinCandidateResult> results;
        // the batch path: this worker's lane runner (the world's ISA variant), made at first use
        const tw::isa::TickTable *batchTable = nullptr;
        void *batch = nullptr;
        std::optional<experimental::twin::internal::CollisionCachePtr> cache;   // the batch path's serial jobs
        ~Worker() {
            if (batch != nullptr) batchTable->destroyBatch(batch);
        }
    };


    // The CUDA kernel's control construction: baseline tick t with the
    // controls of the candidate's events applied from the mutable boundary.
    void BuildTicks(const std::vector<CudaSearchInputEvent> &events, std::vector<ReplayControlTick> *ticks) const {
        ticks->resize(timelineTickCount_);
        DeviceControlState control = program_.mutableBoundaryControls;
        std::size_t next = 0u;
        for (std::uint32_t t = 0u; t < timelineTickCount_; ++t) {
            const std::int64_t publicTime =
                    config_.branchTimeMs + static_cast<std::int64_t>(t + 1u) * config_.tickDurationMs;
            const std::int64_t suffixTime = publicTime - program_.mutableFromTimeMs;
            while (next < events.size() && events[next].timeMs <= suffixTime) {
                ApplyControlEvent(control, events[next], program_.mutableFromTimeMs);
                ++next;
            }
            ReplayControlTick tick = source_.timeline[source_.branchCursor + t];
            tick.controls = ControlsFromState(control);
            tick.stuntsInput = StuntsFromState(control, config_.prestartDurationMs);
            (*ticks)[t] = tick;
        }
    }

    // BuildTicks' controls straight to TwinControl (the batch path: no ReplayControlTick copies; the
    // same control state machine, ControlsFromState and the timeline's actions, so the same words).
    void BuildTwinControls(const std::vector<CudaSearchInputEvent> &events,
                           std::vector<experimental::twin::TwinControl> *out) const {
        out->resize(timelineTickCount_);
        DeviceControlState control = program_.mutableBoundaryControls;
        std::size_t next = 0u;
        for (std::uint32_t t = 0u; t < timelineTickCount_; ++t) {
            const std::int64_t publicTime =
                    config_.branchTimeMs + static_cast<std::int64_t>(t + 1u) * config_.tickDurationMs;
            const std::int64_t suffixTime = publicTime - program_.mutableFromTimeMs;
            while (next < events.size() && events[next].timeMs <= suffixTime) {
                ApplyControlEvent(control, events[next], program_.mutableFromTimeMs);
                ++next;
            }
            (*out)[t] = TwinPhysics::ToTwinControl(ControlsFromState(control),
                                                   source_.timeline[source_.branchCursor + t].actions);
        }
    }

    // The tick before the one judged, as the kernel keeps it: previousPosition, previousVelocity,
    // checkpointBefore, and the body condition programs read as body.temporary.
    struct PreviousTick {
        float position[3] = {}, velocity[3] = {}, angular[3] = {}, quat[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        twin_search::CheckpointCounters counters;
    };
    static PreviousTick Previous(const CudaCandidatePhysicsState &state) {
        const CHmsDyna::CHmsStateDyna &b = state.body.current;
        const ReplayRaceProgress &r = state.race.progress;
        PreviousTick p;
        p.position[0] = b.position.x, p.position[1] = b.position.y, p.position[2] = b.position.z;
        p.velocity[0] = b.linearSpeed.x, p.velocity[1] = b.linearSpeed.y, p.velocity[2] = b.linearSpeed.z;
        p.angular[0] = b.angularSpeed.x, p.angular[1] = b.angularSpeed.y, p.angular[2] = b.angularSpeed.z;
        p.quat[0] = b.rotationQuat.x, p.quat[1] = b.rotationQuat.y, p.quat[2] = b.rotationQuat.z, p.quat[3] = b.rotationQuat.w;
        p.counters = {r.currentLapCheckpointCount, r.totalCheckpointEventCount, r.completedLapCount, r.finishCount};
        return p;
    }

    // A judge's state without its identity (TickJudge::Export / Import).
    struct PrefixJudge {
        double score = 0.0, timeMs = 0.0, detail0 = 0.0, detail1 = 0.0;
        std::uint32_t evaluationTick = 0u, evaluationIndex = 0u;
        std::uint8_t valid = 0u, preciseFinish = 0u, reported = 0u, pad = 0u;
        float previous[3] = {};
        PreviousTick before;   // previous[] with the rest of the previous tick
        std::uint32_t objectiveCount = 0u;   // Scripted: each objective's best so far
        double objectiveScores[CudaSearchMaximumScriptedObjectives] = {};
        double metricValues[CudaSearchMaximumScriptedObjectives] = {};
    };

    // The per-tick body of SimulateSearchCandidatesKernel (and the
    // finish refinement) for one candidate, independent of the engine.
    class TickJudge {
    public:
        enum class Verdict { Continue, Done, Invalid, Unsupported };

        TickJudge(const TwinSearchExecutorImpl &e, std::uint64_t candidateId, std::uint32_t slot, bool baseline,
                  std::uint32_t eventCount, const DeviceSample &incumbent)
            : e_(e), id_(candidateId), slot_(slot), baseline_(baseline), eventCount_(eventCount),
              incumbent_(incumbent) {
            const std::uint64_t prestartNs = static_cast<std::uint64_t>(e.config_.prestartDurationMs) * 1000000u;
            incumbentUpperNs_ =
                    incumbent.preciseFinish ? prestartNs + static_cast<std::uint64_t>(incumbent.score) : ~0ull;
        }

        // The state before the first tick this judge sees (the branch's, or a prefix's last).
        void Start(const CudaCandidatePhysicsState &state) { before_ = Previous(state); }

        // Tick index t (0 = the first tick after the branch) has been stepped; the view's state is the
        // judge's to complete (race constants, body.temporary).
        Verdict OnTick(std::uint32_t t, const EvaluationView &view) {
            const PreviousTick before = before_;
            CudaCandidatePhysicsState &state = *view.state;
            ReplayRaceProgress &progress = state.race.progress;
            progress.requiredCheckpointCount = e_.race_.requiredCheckpointCount;
            progress.requiredLapCount = e_.race_.requiredLapCount;
            state.race.replayPlayMode = e_.replayPlayMode_;
            before_ = Previous(state);
            const CudaSearchExecutorConfiguration &c = e_.config_;
            const CudaSearchEvaluatorKind kind = c.evaluator.kind;
            const bool finishTime = kind == CudaSearchEvaluatorKind::FinishTime;
            const std::int64_t publicTime = c.branchTimeMs + static_cast<std::int64_t>(t + 1u) * c.tickDurationMs;
            if (publicTime < c.evaluationStartTimeMs) return Verdict::Continue;
            const std::uint64_t iteration = baseline_ ? 0u : id_ + 1u;
            if (c.condition) {
                CHmsDyna::CHmsStateDyna &previous = state.body.temporary;
                previous.position = {before.position[0], before.position[1], before.position[2]};
                previous.linearSpeed = {before.velocity[0], before.velocity[1], before.velocity[2]};
                previous.angularSpeed = {before.angular[0], before.angular[1], before.angular[2]};
                previous.rotationQuat.x = before.quat[0];
                previous.rotationQuat.y = before.quat[1];
                previous.rotationQuat.z = before.quat[2];
                previous.rotationQuat.w = before.quat[3];
                previous.rotation.Set(previous.rotationQuat);
                if (!twin_search::EvaluateSearchCondition(c.condition->instructions.data(),
                                                          static_cast<std::uint32_t>(c.condition->instructions.size()),
                                                          state, iteration, e_.clock_)) {
                    ++evaluationIndex_;
                    return progress.raceCompleted ? Verdict::Done : Verdict::Continue;
                }
            }
            if (kind == CudaSearchEvaluatorKind::Scripted) {
                if (twin_search::UpdateScriptedSearchSample(c.evaluator, state, iteration, e_.clock_,
                                                            static_cast<double>(publicTime), &best_)) {
                    Identify(&best_, evaluationIndex_);
                }
                ++evaluationIndex_;
                return progress.raceCompleted ? Verdict::Done : Verdict::Continue;
            }
            if (kind == CudaSearchEvaluatorKind::Distance &&
                twin_search::SearchDistanceTargetReached(c.evaluator, state, before.counters)) {
                DeviceSample sample{};
                sample.score = static_cast<double>(publicTime);
                sample.timeMs = static_cast<double>(publicTime);
                sample.valid = true;
                Identify(&sample, evaluationIndex_);
                if (StrictlyBetter(sample, best_, e_.maximize_)) best_ = sample;
                return Verdict::Done;
            }
            if (kind == CudaSearchEvaluatorKind::CheckpointEvent) {
                if (twin_search::SearchCheckpointEventAccepted(c.evaluator, state, before.counters)) {
                    DeviceSample sample{};
                    sample.score = static_cast<double>(publicTime);
                    sample.timeMs = static_cast<double>(publicTime);
                    sample.valid = true;
                    Identify(&sample, evaluationIndex_);
                    best_ = sample;
                    return Verdict::Done;
                }
                ++evaluationIndex_;
                return progress.raceCompleted ? Verdict::Done : Verdict::Continue;
            }
            if (finishTime && !reported_ && !progress.raceCompleted) {
                // EvaluateState returns an invalid sample here: skip its
                // 330-byte DeviceSample (exact; the index still advances).
                ++evaluationIndex_;
                if (incumbent_.preciseFinish && static_cast<double>(publicTime) >= incumbent_.timeMs) {
                    return Invalidate();
                }
                return Verdict::Continue;
            }
            DeviceSample sample = e_.EvaluateTick(
                    state, {before.position[0], before.position[1], before.position[2]},
                    {before.velocity[0], before.velocity[1], before.velocity[2]},
                    static_cast<double>(publicTime - c.tickDurationMs), static_cast<double>(publicTime),
                    view.stuntsScore, &reported_);
            Identify(&sample, evaluationIndex_);
            if (finishTime && reported_ && sample.valid) {
                // RefineSearchFinishTimesKernel: the precise finish (the
                // engine's finish estimate), with the incumbent pruning.
                const std::uint64_t prestartNs = static_cast<std::uint64_t>(c.prestartDurationMs) * 1000000u;
                const double coarseLowerMs = sample.timeMs - static_cast<double>(c.prestartDurationMs) -
                                             static_cast<double>(c.tickDurationMs);
                if (incumbent_.preciseFinish && coarseLowerMs >= incumbent_.timeMs) return Invalidate();
                if (!view.finish || view.finish->lowerBoundNs < prestartNs || view.finish->upperBoundNs < prestartNs) {
                    return Verdict::Unsupported;
                }
                if (view.finish->upperBoundNs >= incumbentUpperNs_) return Invalidate();
                sample.score = static_cast<double>(view.finish->upperBoundNs - prestartNs);
                sample.timeMs = sample.score / 1000000.0;
                sample.preciseFinish = true;
            }
            if (StrictlyBetter(sample, best_, e_.maximize_)) best_ = sample;
            ++evaluationIndex_;
            if (reported_ && (kind == CudaSearchEvaluatorKind::VolumeEntry ||
                              kind == CudaSearchEvaluatorKind::CustomVolumeEntry ||
                              kind == CudaSearchEvaluatorKind::ConditionTimeEarliest ||
                              kind == CudaSearchEvaluatorKind::ConditionTimeLatest)) {
                return Verdict::Done;
            }
            if (finishTime) {
                if (reported_) return Verdict::Done;
                if (incumbent_.preciseFinish && static_cast<double>(publicTime) >= incumbent_.timeMs) {
                    return Invalidate();
                }
            }
            return Verdict::Continue;
        }

        const DeviceSample &Best() const { return best_; }
        // Scripted on the device: the objectives (tw::gpu::kScriptedDoubles: scores, then metric values) out and back.
        void ExportObjectives(double *o) const {
            for (std::uint32_t i = 0u; i < CudaSearchMaximumScriptedObjectives; ++i) {
                o[i] = best_.objectiveScores[i];
                o[CudaSearchMaximumScriptedObjectives + i] = best_.metricValues[i];
            }
        }
        void ImportObjectives(const double *o) {
            if (!best_.valid) return;
            best_.scriptedObjectiveCount = e_.config_.evaluator.scriptedObjectiveCount;
            for (std::uint32_t i = 0u; i < CudaSearchMaximumScriptedObjectives; ++i) {
                best_.objectiveScores[i] = o[i];
                best_.metricValues[i] = o[CudaSearchMaximumScriptedObjectives + i];
            }
        }
        // Pose re-judging (the GPU engine's near-best ticks): the next OnTick's evaluation index and previous tick.
        void SetEvaluationIndex(std::uint32_t index) { evaluationIndex_ = index; }
        void SetPreviousPosition(const GmVec3 &position) {
            before_.position[0] = position.x, before_.position[1] = position.y, before_.position[2] = position.z;
        }
        // The judge without its identity (PrefixJudge, the device's tw_gpu_session.h JudgeState), and back: a valid best
        // gets this judge's identity exactly as OnTick gives it (an invalid one is DeviceSample{}, as Invalidate
        // leaves it).
        template <class J> void Export(J *j) const {
            j->score = best_.score;
            j->timeMs = best_.timeMs;
            j->detail0 = best_.detail0;
            j->detail1 = best_.detail1;
            j->evaluationTick = best_.evaluationTick;
            j->evaluationIndex = evaluationIndex_;
            j->valid = best_.valid ? 1u : 0u;
            j->preciseFinish = best_.preciseFinish ? 1u : 0u;
            j->reported = reported_ ? 1u : 0u;
            for (int a = 0; a < 3; ++a) j->previous[a] = before_.position[a];
            if constexpr (std::is_same<J, PrefixJudge>::value) {
                j->before = before_;
                j->objectiveCount = best_.scriptedObjectiveCount;
                for (std::uint32_t i = 0u; i < best_.scriptedObjectiveCount; ++i) {
                    j->objectiveScores[i] = best_.objectiveScores[i];
                    j->metricValues[i] = best_.metricValues[i];
                }
            }
        }
        template <class J> void Import(const J &j) {
            reported_ = j.reported != 0u;
            evaluationIndex_ = j.evaluationIndex;
            if constexpr (std::is_same<J, PrefixJudge>::value) {
                before_ = j.before;
            } else {
                before_ = PreviousTick{};
                for (int a = 0; a < 3; ++a) before_.position[a] = j.previous[a];
            }
            if (j.valid == 0u) {
                best_ = DeviceSample{};
                return;
            }
            DeviceSample sample;
            sample.score = j.score;
            sample.timeMs = j.timeMs;
            sample.detail0 = j.detail0;
            sample.detail1 = j.detail1;
            sample.valid = true;
            sample.preciseFinish = j.preciseFinish != 0u;
            if constexpr (std::is_same<J, PrefixJudge>::value) {
                sample.scriptedObjectiveCount = j.objectiveCount;
                for (std::uint32_t i = 0u; i < j.objectiveCount; ++i) {
                    sample.objectiveScores[i] = j.objectiveScores[i];
                    sample.metricValues[i] = j.metricValues[i];
                }
            }
            Identify(&sample, j.evaluationTick);
            best_ = sample;
        }

    private:
        void Identify(DeviceSample *sample, std::uint32_t evaluationTick) const {
            sample->candidateId = id_;
            sample->candidateSlot = slot_;
            sample->evaluationTick = evaluationTick;
            sample->eventCount = eventCount_;
            sample->logicalOrder = 1u + static_cast<std::uint64_t>(slot_) * e_.evaluationTickCount_ + evaluationTick;
            sample->mutation = !baseline_;
        }
        Verdict Invalidate() {
            best_ = DeviceSample{};
            return Verdict::Invalid;
        }
        const TwinSearchExecutorImpl &e_;
        std::uint64_t id_;
        std::uint32_t slot_;
        bool baseline_;
        std::uint32_t eventCount_;
        DeviceSample incumbent_;
        std::uint64_t incumbentUpperNs_ = ~0ull;
        DeviceSample best_{};
        bool reported_ = false;
        std::uint32_t evaluationIndex_ = 0u;
        PreviousTick before_;
    };

    // One slot of the batch path (phases in the header comment).
    struct SlotWork {
        std::vector<experimental::twin::TwinControl> controls;   // every tick's control
        std::optional<TickJudge> judge;
        std::uint32_t first = 0u;        // first changed tick: the runner starts there
        bool physics = false;            // the runner simulates it (or its original)
        bool unsupported = false;
        std::int64_t original = -1;      // a duplicate of that slot: its judge replays the original's views
        bool keepViews = false;          // has duplicates: keep the views its judge saw
        std::vector<tw::isa::BatchTick> views;
    };

    // One candidate on a tick-by-tick engine (the FV delegate, the verifier).
    SlotStatus Simulate(Worker &w, std::uint64_t candidateId, std::uint32_t slot, bool baseline,
                        std::uint32_t eventCount, const DeviceSample &incumbent,
                        const std::function<bool()> &cancel, DeviceSample *out) {
        TickJudge judge(*this, candidateId, slot, baseline, eventCount, incumbent);
        std::string diagnostic;
        const std::uint64_t t0 = Tsc();
        if (!w.physics->Begin(&diagnostic)) return SlotStatus::Unsupported;
        w.physics->Read(&w.view);
        judge.Start(*w.view.state);
        std::uint64_t mark = Tsc();
        w.times.physics += mark - t0;
        SlotStatus status = SlotStatus::Success;
        for (std::uint32_t t = 0u; t < timelineTickCount_; ++t) {
            if ((t & 63u) == 0u && cancel && cancel()) {
                status = SlotStatus::Cancelled;
                break;
            }
            const std::uint64_t s0 = Tsc();
            w.times.evaluation += s0 - mark;
            const bool stepped = w.physics->Step(w.ticks[t]);
            if (stepped) w.physics->Read(&w.view);
            mark = Tsc();
            w.times.physics += mark - s0;
            ++w.times.ticks;
            if (!stepped) {
                status = SlotStatus::Unsupported;
                break;
            }
            const TickJudge::Verdict verdict = judge.OnTick(t, w.view);
            if (verdict == TickJudge::Verdict::Continue) continue;
            if (verdict == TickJudge::Verdict::Unsupported) status = SlotStatus::Unsupported;
            break;
        }
        w.times.evaluation += Tsc() - mark;
        *out = judge.Best();
        return status;
    }

    // Twin mode: the worker's batch runner, its evaluator wired to the
    // worker's current judge (L = 1 today; M10's Runner<16> later).
    bool InitTwinRunner(Worker &w, std::string *diagnostic) {
        experimental::twin::TwinBatchOptions options;
        options.stop.atFinish = false;  // the judge decides when a candidate is done
        Worker *worker = &w;
        options.evaluator = [worker](const experimental::twin::TwinLaneView &lane) {
            const std::uint64_t e0 = Tsc();
            const std::uint32_t t = worker->tickOffset + lane.tick - 1u;
            if (worker->recorder != nullptr) {
                worker->recorder->Tick(worker->controls[t], worker->ticks[t].actions.respawnAtCheckpointCount,
                                       *lane.state);
            }
            TwinPhysics::ReadState(*lane.state, lane.events, &worker->view);
            const TickJudge::Verdict verdict = worker->judge->OnTick(t, worker->view);
            worker->unsupported = verdict == TickJudge::Verdict::Unsupported;
            worker->times.evaluation += Tsc() - e0;
            return verdict == TickJudge::Verdict::Continue;
        };
        auto runner = experimental::twin::CreateTwinBatchRunner(twinWorld_, options);
        if (!runner) {
            if (diagnostic) *diagnostic = "twin search: batch runner: " + runner.Error().diagnostic;
            return false;
        }
        w.runner = std::move(runner).Value();
        return true;
    }

    // The baseline once, without stopping: its controls and its state and
    // events after every tick (branchAtFirstChange). A failing tick ends the
    // snapshots there; a candidate that reaches it fails the same way.
    bool BuildSnapshots(std::string *diagnostic) {
        twin_search::CandidateGenerator generator(program_);
        twin_search::GeneratedCandidate baseline;
        generator.Generate(0u, true, &baseline);
        if (!baseline.active || baseline.status == twin_search::CandidateStatus::CapacityExceeded) return true;
        std::vector<ReplayControlTick> ticks;
        BuildTicks(baseline.events, &ticks);
        experimental::twin::TwinCandidate c;
        c.id = 0u;
        c.start = twinBranch_;
        c.controls.resize(ticks.size());
        for (std::size_t t = 0; t < ticks.size(); ++t) c.controls[t] = TwinPhysics::ToTwinControl(ticks[t]);
        snapStates_.clear();
        snapEvents_.clear();
        snapStates_.reserve(ticks.size());
        snapEvents_.reserve(ticks.size());
        SearchRecorder *recorder = recorder_.get();
        if (recorder) recorder->Begin(c.id, c.start);
        experimental::twin::TwinBatchOptions options;
        options.stop.atFinish = false;
        options.evaluator = [&](const experimental::twin::TwinLaneView &lane) {
            const std::uint32_t t = lane.tick - 1u;
            if (recorder) recorder->Tick(c.controls[t], ticks[t].actions.respawnAtCheckpointCount, *lane.state);
            snapStates_.push_back(*lane.state);
            snapEvents_.push_back(*lane.events);
            return true;
        };
        auto runner = experimental::twin::CreateTwinBatchRunner(twinWorld_, options);
        if (!runner) {
            if (diagnostic) *diagnostic = "twin search: snapshot runner: " + runner.Error().diagnostic;
            return false;
        }
        std::vector<experimental::twin::TwinCandidateResult> results;
        runner.Value()->Run({c}, &results);
        if (recorder) recorder->End();
        snapControls_ = std::move(c.controls);
        return true;
    }

    // One candidate on the twin. FOREVERVALIDATOR_TWIN_SEARCH_CHECK=1 (debug):
    // every candidate that started past the branch is run again from the
    // branch, and a different sample or status is counted (printed at the end).
    SlotStatus SimulateTwin(Worker &w, std::uint64_t candidateId, std::uint32_t slot, bool baseline,
                            std::uint32_t eventCount, const DeviceSample &incumbent, DeviceSample *out) {
        std::uint32_t first = 0u;
        const SlotStatus status = SimulateTwinOnce(w, candidateId, slot, baseline, eventCount, incumbent, true, out, &first);
        if (checkBranching_ && first > 0u) {
            SearchRecorder *recorder = w.recorder;
            w.recorder = nullptr;
            const std::uint64_t ticks = w.times.ticks;
            DeviceSample again{};
            std::uint32_t none = 0u;
            const SlotStatus st = SimulateTwinOnce(w, candidateId, slot, baseline, eventCount, incumbent, false, &again, &none);
            w.times.ticks = ticks;
            w.recorder = recorder;
            const bool same = st == status && again.valid == out->valid && again.score == out->score &&
                              again.timeMs == out->timeMs && again.detail0 == out->detail0 &&
                              again.detail1 == out->detail1 && again.evaluationTick == out->evaluationTick &&
                              again.preciseFinish == out->preciseFinish && again.logicalOrder == out->logicalOrder;
            ++checked_;
            if (!same) {
                ++checkDifferent_;
                std::fprintf(stderr, "twin search check: candidate %llu (first change at tick %u) differs from the "
                             "branch run (score %.9g vs %.9g, tick %u vs %u)\n",
                             static_cast<unsigned long long>(candidateId), first, out->score, again.score,
                             out->evaluationTick, again.evaluationTick);
            }
        }
        return status;
    }

    SlotStatus SimulateTwinOnce(Worker &w, std::uint64_t candidateId, std::uint32_t slot, bool baseline,
                                std::uint32_t eventCount, const DeviceSample &incumbent, bool shared, DeviceSample *out,
                                std::uint32_t *firstOut) {
        TickJudge judge(*this, candidateId, slot, baseline, eventCount, incumbent);
        const std::uint32_t tickCount = static_cast<std::uint32_t>(w.ticks.size());
        w.controls.resize(tickCount);
        for (std::uint32_t t = 0; t < tickCount; ++t) w.controls[t] = TwinPhysics::ToTwinControl(w.ticks[t]);
        TwinPhysics::ReadState(twinBranch_, nullptr, &w.view);
        judge.Start(*w.view.state);
        // The shared prefix: ticks before the first changed control, judged on the baseline's snapshots.
        std::uint32_t first = 0u;
        if (shared && !snapControls_.empty()) {
            const std::uint32_t same = static_cast<std::uint32_t>(
                    std::min<std::size_t>({w.controls.size(), snapControls_.size(), snapStates_.size()}));
            while (first < same && w.controls[first] == snapControls_[first]) ++first;
        }
        *firstOut = first;
        const std::uint64_t p0 = Tsc();
        for (std::uint32_t t = 0; t < first; ++t) {
            TwinPhysics::ReadState(snapStates_[t], &snapEvents_[t], &w.view);
            const TickJudge::Verdict verdict = judge.OnTick(t, w.view);
            if (verdict != TickJudge::Verdict::Continue) {
                w.sharedPrefixTicks += t + 1u;
                w.times.evaluation += Tsc() - p0;
                *out = judge.Best();
                return verdict == TickJudge::Verdict::Unsupported ? SlotStatus::Unsupported : SlotStatus::Success;
            }
        }
        w.sharedPrefixTicks += first;
        w.times.evaluation += Tsc() - p0;
        if (first == tickCount) {
            *out = judge.Best();
            return SlotStatus::Success;
        }
        experimental::twin::TwinCandidate &candidate = w.queue[0];
        candidate.id = candidateId;
        candidate.start = first == 0u ? twinBranch_ : snapStates_[first - 1u];
        candidate.controls.assign(w.controls.begin() + first, w.controls.end());
        w.tickOffset = first;
        w.judge = &judge;
        w.unsupported = false;
        if (w.recorder) w.recorder->Begin(candidateId, candidate.start);
        const std::uint64_t evaluationBefore = w.times.evaluation;
        const std::uint64_t t0 = Tsc();
        const experimental::twin::TwinStatus status = w.runner->Run(w.queue, &w.results);
        // The runner's time minus the evaluator's share (timed inside it).
        w.times.physics += (Tsc() - t0) - (w.times.evaluation - evaluationBefore);
        w.times.ticks += w.results.empty() ? 0u : w.results[0].ticks;
        if (w.recorder) w.recorder->End();
        w.judge = nullptr;
        w.tickOffset = 0u;
        *out = judge.Best();
        if (w.unsupported || (status != experimental::twin::TwinStatus::Ok &&
                              status != experimental::twin::TwinStatus::Stopped)) {
            return SlotStatus::Unsupported;
        }
        return SlotStatus::Success;
    }

    void RunSlot(Worker &w, std::uint64_t first, std::uint32_t slot, bool baseline, const DeviceSample &incumbent,
                 const std::function<bool()> &cancel, SlotResult *r) {
        const std::uint64_t id = first + slot;
        const std::uint64_t g0 = Tsc();
        if (cancel && cancel()) {
            r->status = SlotStatus::Cancelled;
            r->active = false;
            return;
        }
        w.generator.Generate(id, baseline, &w.candidate);
        r->mutationCount = w.candidate.mutationCount;
        r->active = w.candidate.active;
        r->events = w.candidate.events;
        if (w.candidate.status == twin_search::CandidateStatus::CapacityExceeded) {
            r->status = SlotStatus::CapacityExceeded;
            r->active = false;
        }
        const std::uint64_t g1 = Tsc();
        w.times.generation += g1 - g0;
        if (!r->active) return;
        BuildTicks(r->events, &w.ticks);
        w.times.evaluation += Tsc() - g1;
        const std::uint32_t eventCount = static_cast<std::uint32_t>(r->events.size());
        r->status = twinWorld_ ? SimulateTwin(w, id, slot, baseline, eventCount, incumbent, &r->best)
                               : Simulate(w, id, slot, baseline, eventCount, incumbent, cancel, &r->best);
    }

    // ---- the batch path ------------------------------------------------------------------------
    // The shared prefix judged once per batch: one judge over the baseline's snapshots (the batch's incumbent), its
    // state (without identity) after every tick until its first verdict other than Continue. A candidate's judge
    // after its first t ticks is that state with the candidate's identity (the samples depend on the trajectory
    // only; within a candidate the event count never decides), so PrepareSlot takes it instead of judging again.
    void BuildPrefix(bool baseline, const DeviceSample &incumbent) {
        prefix_.clear();
        prefixStop_ = std::numeric_limits<std::uint32_t>::max();
        prefixStopVerdict_ = TickJudge::Verdict::Continue;
        if (snapControls_.empty() || perCandidatePrefix_) return;
        const std::size_t n = std::min(snapControls_.size(), snapStates_.size());
        prefix_.reserve(n);
        TickJudge judge(*this, 0u, 0u, baseline, 0u, incumbent);
        EvaluationView view;
        TwinPhysics::ReadState(twinBranch_, nullptr, &view);
        judge.Start(*view.state);
        for (std::uint32_t t = 0; t < n; ++t) {
            TwinPhysics::ReadState(snapStates_[t], &snapEvents_[t], &view);
            const TickJudge::Verdict verdict = judge.OnTick(t, view);
            prefix_.emplace_back();
            judge.Export(&prefix_.back());
            if (verdict != TickJudge::Verdict::Continue) {
                prefixStop_ = t;
                prefixStopVerdict_ = verdict;
                break;
            }
        }
    }

    // (A) one slot: generation, controls, the shared prefix judged on the baseline's snapshots
    // (as RunSlot + SimulateTwinOnce up to the runner).
    void PrepareSlot(Worker &w, std::uint64_t first, std::uint32_t slot, bool baseline, const DeviceSample &incumbent,
                     const std::function<bool()> &cancel, SlotResult *r, SlotWork *sw) {
        const std::uint64_t id = first + slot;
        const std::uint64_t g0 = Tsc();
        if (cancel && cancel()) {
            r->status = SlotStatus::Cancelled;
            r->active = false;
            return;
        }
        w.generator.Generate(id, baseline, &w.candidate);
        r->mutationCount = w.candidate.mutationCount;
        r->active = w.candidate.active;
        r->events = w.candidate.events;
        if (w.candidate.status == twin_search::CandidateStatus::CapacityExceeded) {
            r->status = SlotStatus::CapacityExceeded;
            r->active = false;
        }
        const std::uint64_t g1 = Tsc();
        w.times.generation += g1 - g0;
        if (!r->active) return;
        BuildTwinControls(r->events, &sw->controls);
        const std::uint32_t eventCount = static_cast<std::uint32_t>(r->events.size());
        sw->judge.emplace(*this, id, slot, baseline, eventCount, incumbent);
        TickJudge &judge = *sw->judge;
        const std::uint32_t tickCount = static_cast<std::uint32_t>(sw->controls.size());
        TwinPhysics::ReadState(twinBranch_, nullptr, &w.view);
        judge.Start(*w.view.state);
        std::uint32_t firstChange = 0u;
        if (!snapControls_.empty()) {
            const std::uint32_t same = static_cast<std::uint32_t>(
                    std::min<std::size_t>({sw->controls.size(), snapControls_.size(), snapStates_.size()}));
            while (firstChange < same && sw->controls[firstChange] == snapControls_[firstChange]) ++firstChange;
        }
        // the shared prefix's judge: the batch's baseline judge after tick firstChange - 1 (BuildPrefix), or where
        // it stopped
        if (firstChange > 0u && perCandidatePrefix_) {
            // the candidate's own judge over the prefix (its programs read the iteration)
            for (std::uint32_t t = 0; t < firstChange; ++t) {
                TwinPhysics::ReadState(snapStates_[t], &snapEvents_[t], &w.view);
                const TickJudge::Verdict verdict = judge.OnTick(t, w.view);
                if (verdict != TickJudge::Verdict::Continue) {
                    w.sharedPrefixTicks += t + 1u;
                    w.times.evaluation += Tsc() - g1;
                    r->best = judge.Best();
                    r->status = verdict == TickJudge::Verdict::Unsupported ? SlotStatus::Unsupported : SlotStatus::Success;
                    return;
                }
            }
        } else if (firstChange > 0u) {
            if (prefixStop_ < firstChange) {
                judge.Import(prefix_[prefixStop_]);
                w.sharedPrefixTicks += prefixStop_ + 1u;
                w.times.evaluation += Tsc() - g1;
                r->best = judge.Best();
                r->status = prefixStopVerdict_ == TickJudge::Verdict::Unsupported ? SlotStatus::Unsupported : SlotStatus::Success;
                return;
            }
            judge.Import(prefix_[firstChange - 1u]);
        }
        w.sharedPrefixTicks += firstChange;
        w.times.evaluation += Tsc() - g1;
        if (firstChange == tickCount) {
            r->best = judge.Best();
            r->status = SlotStatus::Success;
            return;
        }
        sw->first = firstChange;
        sw->physics = true;
    }

    // (B) the slots that need the runner: duplicates folded onto their first occurrence, the rest
    // longest first (ties in slot order). Order and folding change no result.
    // [lo, hi): a range of `work` (the GPU path's chunks); indices stay those of `work`.
    std::vector<std::uint32_t> PhysicsJobs(std::vector<SlotWork> &work, std::uint32_t lo = 0u,
                                           std::uint32_t hi = std::numeric_limits<std::uint32_t>::max()) const {
        std::vector<std::uint32_t> jobs;
        std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> seen;
        hi = std::min<std::uint32_t>(hi, static_cast<std::uint32_t>(work.size()));
        for (std::uint32_t slot = lo; slot < hi; ++slot) {
            SlotWork &sw = work[slot];
            if (!sw.physics) continue;
            if (dedupe_) {
                std::uint64_t h = 1469598103934665603ull ^ sw.first;
                for (std::size_t t = sw.first; t < sw.controls.size(); ++t) {
                    h ^= sw.controls[t];
                    h *= 1099511628211ull;
                }
                std::vector<std::uint32_t> &bucket = seen[h];
                bool folded = false;
                for (const std::uint32_t o : bucket) {
                    const SlotWork &ow = work[o];
                    if (ow.first == sw.first && ow.controls.size() == sw.controls.size() &&
                        std::equal(sw.controls.begin() + sw.first, sw.controls.end(), ow.controls.begin() + ow.first)) {
                        sw.original = o;
                        work[o].keepViews = true;
                        folded = true;
                        break;
                    }
                }
                if (folded) continue;
                bucket.push_back(slot);
            }
            jobs.push_back(slot);
        }
        if (longestFirst_) {
            std::stable_sort(jobs.begin(), jobs.end(), [&](std::uint32_t a, std::uint32_t b) {
                return work[a].controls.size() - work[a].first > work[b].controls.size() - work[b].first;
            });
        }
        return jobs;
    }

    // (C) the jobs, dealt round robin over the workers in longest-first order; a worker whose own
    // share is done takes the next job of another's (so small batches spread over every worker).
    struct JobDeal {
        std::vector<std::vector<std::uint32_t>> lists;
        std::unique_ptr<std::atomic<std::size_t>[]> heads;
        JobDeal(const std::vector<std::uint32_t> &jobs, std::uint32_t workers)
            : lists(workers), heads(new std::atomic<std::size_t>[workers]) {
            for (std::size_t i = 0; i < jobs.size(); ++i) lists[i % workers].push_back(jobs[i]);
            for (std::uint32_t t = 0; t < workers; ++t) heads[t] = 0u;
        }
        bool Take(std::uint32_t worker, std::uint32_t *slot) {
            const std::uint32_t n = static_cast<std::uint32_t>(lists.size());
            for (std::uint32_t v = 0; v < n; ++v) {
                const std::uint32_t t = (worker + v) % n;
                if (heads[t].load(std::memory_order_relaxed) >= lists[t].size()) continue;
                const std::size_t i = heads[t]++;
                if (i < lists[t].size()) {
                    *slot = lists[t][i];
                    return true;
                }
            }
            return false;
        }
    };

    // The judge after one simulated tick of slot `sw` (the runner's evaluator, and the serial path's).
    static bool JudgeTick(SlotWork &sw, Worker &w, const tw::isa::BatchTick &tick) {
        TwinPhysics::ReadTick(tick, &w.view);
        const TickJudge::Verdict verdict = sw.judge->OnTick(sw.first + tick.tick, w.view);
        if (sw.keepViews) sw.views.push_back(tick);
        sw.unsupported = verdict == TickJudge::Verdict::Unsupported;
        return verdict == TickJudge::Verdict::Continue;
    }

    struct RunnerContext {   // work index i is slot base + i
        TwinSearchExecutorImpl *e;
        Worker *w;
        std::uint32_t worker;
        std::uint32_t base;
        std::vector<SlotResult> *slots;
        std::vector<SlotWork> *work;
        JobDeal *deal;
        const std::function<bool()> *cancel;
        std::uint64_t evaluationTsc = 0u;

        // The next job of this worker, or false; a cancelled job is marked and skipped.
        bool Take(std::uint32_t *slot) {
            while (deal->Take(worker, slot)) {
                if (*cancel && (*cancel)()) {
                    SlotResult &r = (*slots)[base + *slot];
                    r.status = SlotStatus::Cancelled;
                    r.active = false;
                    continue;
                }
                return true;
            }
            return false;
        }
        const tw::model::TwinState *Start(const SlotWork &sw) const {
            const experimental::twin::TwinState &start = sw.first == 0u ? e->twinBranch_ : e->snapStates_[sw.first - 1u];
            return reinterpret_cast<const tw::model::TwinState *>(start.bytes);
        }
        static bool Next(void *u, tw::isa::BatchJob *job) {
            RunnerContext &c = *static_cast<RunnerContext *>(u);
            std::uint32_t slot = 0u;
            if (!c.Take(&slot)) return false;
            const SlotWork &sw = (*c.work)[slot];
            job->id = slot;
            job->start = c.Start(sw);
            job->controls = sw.controls.data() + sw.first;
            job->ticks = static_cast<std::uint32_t>(sw.controls.size()) - sw.first;
            return true;
        }
        static bool Evaluate(void *u, const tw::isa::BatchTick &tick) {
            RunnerContext &c = *static_cast<RunnerContext *>(u);
            const std::uint64_t e0 = Tsc();
            const bool go = JudgeTick((*c.work)[static_cast<std::size_t>(tick.id)], *c.w, tick);
            c.evaluationTsc += Tsc() - e0;
            return go;
        }
        static void Done(void *u, std::uint64_t id, std::uint32_t ticks) {
            RunnerContext &c = *static_cast<RunnerContext *>(u);
            SlotWork &sw = (*c.work)[static_cast<std::size_t>(id)];
            SlotResult &r = (*c.slots)[c.base + static_cast<std::size_t>(id)];
            r.best = sw.judge->Best();
            r.status = sw.unsupported ? SlotStatus::Unsupported : SlotStatus::Success;
            c.w->times.ticks += ticks;
        }
    };

    // One worker's jobs: through its lane runner with at most `laneCap` in flight, or (laneCap below
    // laneMin_: the runner's per-pass bookkeeping costs more than its lanes save until most lanes
    // are busy) one at a time through the world's step with the worker's collision cache.
    void RunJobs(Worker &w, std::uint32_t worker, std::uint32_t base, std::vector<SlotResult> &slots,
                 std::vector<SlotWork> &work, JobDeal &deal, int laneCap, const std::function<bool()> &cancel) {
        const experimental::twin::TwinWorld::Impl &world = twinWorld_->Internal();
        RunnerContext c{this, &w, worker, base, &slots, &work, &deal, &cancel};
        const std::uint64_t t0 = Tsc();
        if (laneCap >= laneMin_) {
            if (w.batch == nullptr) {
                tw::isa::BatchPolicy policy;
                policy.lanes = lanes_;
                // wide forces only when most lanes take part: below ~9 of 16 the per-lane kernels
                // cost less (searchq --lane-limit, docs/twin/search-throughput.md)
                policy.forcesMinLanes = std::max(3, lanes_ / 2 + 1);
                policy.respondMinLanes = 4;
                policy.fullState = fullState_;
                w.batchTable = world.isa;
                w.batch = world.isa->newBatch(world.isaInstance.get(), policy);
                if (w.batch == nullptr) throw std::bad_alloc();
            }
            tw::isa::BatchCallbacks cb;
            cb.user = &c;
            cb.next = &RunnerContext::Next;
            cb.evaluate = &RunnerContext::Evaluate;
            cb.done = &RunnerContext::Done;
            cb.laneLimit = laneCap;
            w.batchTable->runBatch(w.batch, cb, nullptr);
        } else {
            if (!w.cache) w.cache.emplace(experimental::twin::internal::NewCollisionCache(world));
            std::uint32_t slot = 0u;
            while (c.Take(&slot)) {
                SlotWork &sw = work[slot];
                tw::model::TwinState s;
                std::memcpy(&s, c.Start(sw), sizeof s);
                tw::step::Events ev;
                tw::isa::BatchTick tick;
                tick.id = slot;
                const std::uint32_t n = static_cast<std::uint32_t>(sw.controls.size()) - sw.first;
                std::uint32_t ran = 0u;
                bool failed = false;
                for (std::uint32_t k = 0; k < n; ++k) {
                    if (experimental::twin::internal::Step(world, s, sw.controls[sw.first + k], &ev, w.cache->get()) !=
                        tw::step::Status::Ok) {
                        failed = true;
                        break;
                    }
                    ++ran;
                    tick.tick = k;
                    for (int a = 0; a < 3; ++a) tick.pos[a] = s.pos[a];
                    for (int a = 0; a < 3; ++a) tick.vel[a] = s.vel[a];
                    for (int a = 0; a < 4; ++a) tick.quat[a] = s.quat[a];
                    tick.tickMs = s.tickMs;
                    tick.raceCompleted = s.raceCompleted != 0u;
                    tick.finished = ev.finished;
                    tick.finishLowerNs = ev.finishLowerNs;
                    const std::uint64_t e0 = Tsc();
                    const bool go = JudgeTick(sw, w, tick);
                    c.evaluationTsc += Tsc() - e0;
                    if (!go) break;
                }
                SlotResult &r = slots[base + slot];
                r.best = sw.judge->Best();
                r.status = failed || sw.unsupported ? SlotStatus::Unsupported : SlotStatus::Success;
                w.times.ticks += ran;
            }
        }
        w.times.physics += (Tsc() - t0) - c.evaluationTsc;
        w.times.evaluation += c.evaluationTsc;
    }

    // (D) a duplicate: its own judge over its original's views (the same values, so the same
    // verdicts; its samples carry its own id, slot and event count).
    void JudgeDuplicate(Worker &w, std::uint32_t base, std::vector<SlotResult> &slots, std::vector<SlotWork> &work,
                        std::uint32_t slot) {
        SlotWork &sw = work[slot];
        SlotResult &r = slots[base + slot];
        const SlotWork &ow = work[static_cast<std::size_t>(sw.original)];
        const SlotResult &orig = slots[base + static_cast<std::size_t>(sw.original)];
        if (orig.status == SlotStatus::Cancelled) {
            r.status = SlotStatus::Cancelled;
            r.active = false;
            return;
        }
        const std::uint64_t e0 = Tsc();
        TickJudge::Verdict verdict = TickJudge::Verdict::Continue;
        for (const tw::isa::BatchTick &tick : ow.views) {
            TwinPhysics::ReadTick(tick, &w.view);
            verdict = sw.judge->OnTick(sw.first + tick.tick, w.view);
            if (verdict != TickJudge::Verdict::Continue) break;
        }
        r.best = sw.judge->Best();
        r.status = verdict == TickJudge::Verdict::Unsupported ? SlotStatus::Unsupported : SlotStatus::Success;
        w.times.evaluation += Tsc() - e0;
    }

    // The batch in chunks of kChunk slots (bounded memory for any batch size; duplicates are folded
    // inside a chunk).
    static constexpr std::uint32_t kChunk = 4096u;
    void RunTwinBatch(std::uint64_t first, std::uint32_t count, bool baseline, const DeviceSample &incumbent,
                      const std::function<bool()> &cancel, std::vector<SlotResult> &slots) {
        BuildPrefix(baseline, incumbent);
        std::vector<SlotWork> work;
        for (std::uint32_t base = 0u; base < count; base += kChunk) {
            const std::uint32_t n = std::min(kChunk, count - base);
            work.clear();
            work.resize(n);
            RunTwinChunk(first, base, n, baseline, incumbent, cancel, slots, work);
        }
    }

    void RunTwinChunk(std::uint64_t first, std::uint32_t base, std::uint32_t n, bool baseline, const DeviceSample &incumbent,
                      const std::function<bool()> &cancel, std::vector<SlotResult> &slots, std::vector<SlotWork> &work) {
        std::atomic<std::uint32_t> nextSlot{0u};
        const std::uint64_t a0 = Tsc();
        const bool ok = pool_->Run([&](std::uint32_t t) {
            Worker &w = *workers_[t];
            for (std::uint32_t i = nextSlot++; i < n; i = nextSlot++) {
                PrepareSlot(w, first, base + i, baseline, incumbent, cancel, &slots[base + i], &work[i]);
            }
        });
        if (!ok) throw std::bad_alloc();
        const std::uint64_t a1 = Tsc();
        const std::vector<std::uint32_t> jobs = PhysicsJobs(work);
        std::vector<std::uint64_t> workerTsc(threads_, 0u);
        if (!jobs.empty()) {
            const std::uint32_t workers = std::min<std::uint32_t>(threads_, static_cast<std::uint32_t>(jobs.size()));
            JobDeal deal(jobs, workers);
            const int perWorker = static_cast<int>((jobs.size() + workers - 1u) / workers);
            const int laneCap = std::min(lanes_, perWorker);
            if (!pool_->Run([&](std::uint32_t t) {
                    const std::uint64_t w0 = Tsc();
                    if (t < workers) RunJobs(*workers_[t], t, base, slots, work, deal, laneCap, cancel);
                    workerTsc[t] = Tsc() - w0;
                })) {
                throw std::bad_alloc();
            }
        }
        const std::uint64_t a2 = Tsc();
        if (trace_) {
            std::fprintf(stderr, "twin batch: %u slots, %zu jobs, prepare %.3f ms, physics %.3f ms (workers:", n, jobs.size(),
                         double(a1 - a0) / tscPerMs_, double(a2 - a1) / tscPerMs_);
            for (std::uint64_t v : workerTsc) std::fprintf(stderr, " %.3f", double(v) / tscPerMs_);
            std::fprintf(stderr, ")\n");
        }
        for (std::uint32_t i = 0; i < n; ++i) {
            if (work[i].original >= 0) JudgeDuplicate(*workers_[0], base, slots, work, i);
        }
    }

    // ---- the GPU engine (FOREVERVALIDATOR_TWIN_SEARCH_ENGINE=gpu, tw_gpu_session.h SearchEngine) ----------------------
    // The batch path's phases with (C) on the device: every job one thread, from its start state (the branch or
    // the baseline snapshot before its first change), the judge after every tick with its early stop. Pose's
    // near-best ticks are re-judged here (exact); a job whose device run is not the CPU's (raw contacts past the
    // device buffer, more near-best Pose ticks than the device keeps) runs again on the CPU. Duplicates take their
    // original's result with their own identity (the same trajectory and prefix give the same samples).
    bool InitGpu(std::string *diagnostic) {
        const char *engine = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_ENGINE");
        if (engine != nullptr && std::strcmp(engine, "cpu") == 0) return true;   // the CPU twin even when asked for the GPU
        if (!source_.options.gpu && (engine == nullptr || std::strcmp(engine, "gpu") != 0)) return true;
#if FOREVERVALIDATOR_TWIN_GPU
        auto fail = [&](std::string message) {
            if (diagnostic) *diagnostic = "twin search (gpu): " + message;
            return false;
        };
        const CudaSearchEvaluatorKind kind = config_.evaluator.kind;
        // the device's hand-written judge: velocity, point, pose, volume entry, stunt points, finish time; the rest with FV's
        // evaluator code (tw_gpu_judge.cuh), on the host-built jobs
        gpuTargets_ = fullState_ || (config_.evaluator.optionFlags & CudaSearchEntrySpeed) != 0u ||
                      (kind != CudaSearchEvaluatorKind::Velocity && kind != CudaSearchEvaluatorKind::Point &&
                       kind != CudaSearchEvaluatorKind::Pose && kind != CudaSearchEvaluatorKind::VolumeEntry &&
                       kind != CudaSearchEvaluatorKind::StuntPoints && kind != CudaSearchEvaluatorKind::FinishTime);
        const char *schedule = std::getenv("FOREVERVALIDATOR_TWIN_GPU_SCHEDULE");
        if (gpuTargets_ && (perCandidatePrefix_ || (schedule != nullptr && std::strcmp(schedule, "tick") == 0))) {
            // programs that read the candidate's iteration (no shared prefix), or the tick schedule (no tick events)
            std::fprintf(stderr, "twin search (gpu): this target runs on the CPU twin\n");
            return true;
        }
        const experimental::twin::TwinWorld::Impl &world = twinWorld_->Internal();
        if (!world.packs || !world.packs->useC || world.packs->collisionName != "c" || world.packs->collisionAtlas ||
            world.model.policy != tw::step::Policy::FvSubsteps || world.model.policyConfig != nullptr) {
            return fail("the GPU runs collision stage c on policy fv only");
        }
        std::vector<tw::model::TwinState> starts;
        starts.reserve(1u + snapStates_.size());
        starts.push_back(experimental::twin::internal::ToModel(twinBranch_));
        for (const experimental::twin::TwinState &s : snapStates_) starts.push_back(experimental::twin::internal::ToModel(s));
        std::string error;
        auto engine_ = std::make_unique<tw::gpu::SearchEngine>();
        const tw::fv::TwinWorldPacks &p = *world.packs;
        if (!engine_->Init(*world.model.car, world.model.water, *world.model.race, p.view, p.shapes, starts.data(), starts.size(),
                           &error)) {
            return fail(error);
        }
        gpu_ = std::move(engine_);
        if (gpuTargets_) {
            if (gpu_->TargetBytes() != sizeof(CudaSearchEvaluatorConfiguration)) {
                gpu_.reset();
                return fail("the device's evaluator configuration is not the host's");
            }
            const std::vector<CudaSearchConditionInstruction> none;
            const std::vector<CudaSearchConditionInstruction> &condition = config_.condition ? config_.condition->instructions : none;
            if (!gpu_->SetTargets(&config_.evaluator, sizeof config_.evaluator, offsetof(CudaSearchEvaluatorConfiguration, aroundWalls),
                                  aroundWalls_, aroundWalls_ != nullptr ? config_.aroundWallsTables.size() : 0u, condition.data(),
                                  sizeof(CudaSearchConditionInstruction) * condition.size(), &error)) {
                gpu_.reset();
                return fail(error);
            }
        }
        if (const char *order = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_ORDER"); order != nullptr && std::strcmp(order, "length") == 0) {
            gpuLexOrder_ = false;
        }
        if (const char *ahead = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_AHEAD"); ahead != nullptr && *ahead != '\0') {
            const unsigned long n = std::strtoul(ahead, nullptr, 10);
            gpuAhead_ = n != 0u;
            gpuDepth_ = static_cast<std::uint32_t>(std::min<unsigned long>(
                    std::max<unsigned long>(n, 1u), static_cast<unsigned long>(tw::gpu::SearchEngine::kMaxSets - 1)));
        }
        if (const char *follow = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_FOLLOW"); follow != nullptr && std::strcmp(follow, "1") == 0) {
            gpuFollow_ = true;
        } else if (follow != nullptr && std::strcmp(follow, "taken") == 0) {
            gpuFollowTaken_ = true;
        }
        if (const char *group = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_GROUP"); group != nullptr && *group != '\0') {
            gpuGroup_ = std::max(1u, std::min(16u, static_cast<std::uint32_t>(std::strtoul(group, nullptr, 10))));
        }
        if (const char *chunk = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_CHUNK"); chunk != nullptr && *chunk != '\0') {
            gpuChunk_ = static_cast<std::uint32_t>(std::strtoul(chunk, nullptr, 10));
        }
        // the per-candidate host work on the device (twin_search_device.cu; FOREVERVALIDATOR_TWIN_SEARCH_GPU_GEN=host: on the
        // session's threads, as round 1)
        const char *gen = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_GEN");
        if (!gpuTargets_ && (gen == nullptr || std::strcmp(gen, "host") != 0) &&
            (schedule == nullptr || std::strcmp(schedule, "tick") != 0)) {
            std::vector<std::uint32_t> actionBits(timelineTickCount_);
            for (std::uint32_t t = 0; t < timelineTickCount_; ++t) {
                actionBits[t] = TwinPhysics::ToTwinControl(ReplayVehicleControlState{}, source_.timeline[source_.branchCursor + t].actions);
            }
            const std::uint32_t snapCount = snapControls_.empty() ? 0u
                                                                  : static_cast<std::uint32_t>(std::min(snapControls_.size(), snapStates_.size()));
            DeviceSample none{};
            TickJudge fresh(*this, 0u, 0u, false, 0u, none);
            EvaluationView view;
            TwinPhysics::ReadState(twinBranch_, nullptr, &view);
            tw::gpu::JudgeState initial;
            fresh.Start(*view.state);
            fresh.Export(&initial);
            auto g = std::make_unique<twin_search::DeviceGeneration>();
            if (!g->Init(program_, config_.branchTimeMs, actionBits.data(), timelineTickCount_, snapControls_.data(), snapCount, initial,
                         kind == CudaSearchEvaluatorKind::Pose, &error)) {
                return fail(error);
            }
            gpuGen_ = std::move(g);
            // a job set per batch in flight: this one and those run ahead (2 at least: a batch's chunks alternate)
            const int sets = std::max(2, static_cast<int>(gpuAhead_ ? gpuDepth_ : 0u) + 1);
            if (!gpu_->SetSets(sets, &error)) return fail(error);
            gpuGen_->SetSets(sets);
            // every set's buffers now (allocation synchronizes the device and is slow: not in the first batch), for a group of
            // batches in one chunk
            const std::uint32_t group = gpuAhead_ && kind != CudaSearchEvaluatorKind::FinishTime ? gpuGroup_ : 1u;
            const std::uint64_t groupSlots = static_cast<std::uint64_t>(config_.maximumBatchSize) * group;
            const std::uint32_t most = static_cast<std::uint32_t>(std::max<std::uint64_t>(
                    1u, std::min<std::uint64_t>(gpuChunk_ == 0u ? groupSlots : std::min<std::uint64_t>(gpuChunk_, groupSlots), groupSlots)));
            if (!gpuGen_->Reserve(most, &error) || !gpu_->ReserveJobs(most, &error)) return fail(error);
            if (trace_) {
                std::fprintf(stderr, "twin gpu device generation: %u slots, capacity %u events (%zu baseline), %u ticks, %u generator threads, "
                             "%d job sets of %.1f MB (generation) + %.1f MB (lanes, results), %u groups of %u batches run ahead\n", most,
                             program_.eventCapacity, program_.baselineInputs.size(), timelineTickCount_, gpuGen_->GeneratorThreads(), sets,
                             double(gpuGen_->SetBytes()) / 1048576.0, double(gpu_->SetBytes()) / 1048576.0, gpuAhead_ ? gpuDepth_ : 0u, group);
            }
        }
        return true;
#else
        return diagnostic ? (*diagnostic = "twin search: built without the GPU engine", false) : false;
#endif
    }

#if FOREVERVALIDATOR_TWIN_GPU
    tw::gpu::JudgeConfig GpuJudgeConfig(const DeviceSample &incumbent) const {
        tw::gpu::JudgeConfig jc;
        jc.kind = static_cast<std::uint32_t>(config_.evaluator.kind);
        jc.optionFlags = config_.evaluator.optionFlags;
        for (int v = 0; v < 16; ++v) jc.values[v] = config_.evaluator.values[v];
        jc.branchTimeMs = config_.branchTimeMs;
        jc.tickDurationMs = config_.tickDurationMs;
        jc.evaluationStartTimeMs = config_.evaluationStartTimeMs;
        jc.prestartDurationMs = config_.prestartDurationMs;
        jc.maximize = maximize_ ? 1u : 0u;
        jc.incumbentPrecise = incumbent.preciseFinish ? 1u : 0u;
        jc.incumbentTimeMs = incumbent.timeMs;
        jc.incumbentUpperNs = incumbent.preciseFinish ? static_cast<std::uint64_t>(config_.prestartDurationMs) * 1000000u +
                                                                static_cast<std::uint64_t>(incumbent.score)
                                                      : ~0ull;
        if (gpuTargets_) {
            jc.targets = 1u;
            jc.conditionCount = config_.condition ? static_cast<std::uint32_t>(config_.condition->instructions.size()) : 0u;
            jc.requiredCheckpoints = race_.requiredCheckpointCount;
            jc.requiredLaps = race_.requiredLapCount;
            jc.playMode = replayPlayMode_;
            jc.clock[0] = clock_.lastImprovementSeconds;
            jc.clock[1] = clock_.lastRestartSeconds;
            jc.clock[2] = clock_.currentSeconds;
        }
        return jc;
    }

    // One batch's device work: its slots' work and results, its chunks' jobs and the job sets they run on.
    struct GpuBatch {
        std::uint64_t first = 0u;
        std::uint32_t count = 0u;
        std::uint64_t version = 0u;   // programVersion_ when prepared
        bool valid = false;
        std::vector<SlotWork> work;
        std::vector<SlotResult> slots;   // a batch run ahead (the caller's own otherwise)
        std::vector<std::vector<std::uint32_t>> jobs;
        std::vector<int> sets;
    };
    struct GpuTimes {
        std::uint64_t prepare = 0u, fill = 0u, wait = 0u, results = 0u;
        double kernelMs = 0.0;
        std::uint64_t ticks = 0u, substeps = 0u;
        std::uint64_t laneTicks = 0u;   // trace: warps x 32 lanes x their longest job's ticks
        unsigned long long clk[8] = {};
    };

    // (A) + (B) + (C) of slots [lo, hi) as chunk k of `b` on job set `set`: prepared on the session's threads, ordered,
    // the set filled and submitted.
    void GpuPrepareChunk(GpuBatch &b, std::vector<SlotResult> &slots, std::uint32_t k, std::uint32_t lo, std::uint32_t hi,
                         int set, bool baseline, const DeviceSample &incumbent, const std::function<bool()> &cancel,
                         const tw::gpu::JudgeConfig &jc, GpuTimes &t, int after = -1) {
        std::string error;
        const std::uint64_t p0 = Tsc();
        std::atomic<std::uint32_t> nextSlot{lo};
        if (!pool_->Run([&](std::uint32_t th) {
                Worker &w = *workers_[th];
                for (std::uint32_t i = nextSlot++; i < hi; i = nextSlot++) {
                    PrepareSlot(w, b.first, i, baseline, incumbent, cancel, &slots[i], &b.work[i]);
                }
            })) {
            throw std::bad_alloc();
        }
        std::vector<SlotWork> &work = b.work;
        std::vector<std::uint32_t> &jobs = b.jobs[k];
        jobs = PhysicsJobs(work, lo, hi);
        if (gpuLexOrder_) {
            // neighbours share their controls as long as possible (warp-mates on the same trajectory run the
            // same code on the same data): first changed tick, then the controls from there, lexicographic
            std::sort(jobs.begin(), jobs.end(), [&](std::uint32_t a, std::uint32_t c) {
                const SlotWork &x = work[a], &y = work[c];
                if (x.first != y.first) return x.first < y.first;
                const int d = std::memcmp(x.controls.data() + x.first, y.controls.data() + y.first,
                                          sizeof(std::uint32_t) * (x.controls.size() - x.first));
                return d != 0 ? d < 0 : a < c;
            });
        }
        const std::uint64_t p1 = Tsc();
        std::vector<std::uint32_t> offset(jobs.size() + 1u, 0u);
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            const SlotWork &sw = work[jobs[i]];
            offset[i + 1u] = offset[i] + static_cast<std::uint32_t>(sw.controls.size()) - sw.first;
        }
        tw::gpu::SearchJob *dj = gpu_->Jobs(set, jobs.size(), &error);
        std::uint32_t *dc = gpu_->Controls(set, offset.back(), &error);
        if (dj == nullptr || dc == nullptr) throw std::runtime_error(error);
        double *objectives = nullptr;   // scripted: each job's objectives after its prefix
        if (gpuTargets_ && config_.evaluator.kind == CudaSearchEvaluatorKind::Scripted &&
            (objectives = gpu_->Scripted(set, jobs.size(), &error)) == nullptr) {
            throw std::runtime_error(error);
        }
        std::atomic<std::size_t> nextJob{0u};
        if (!pool_->Run([&](std::uint32_t) {
                for (std::size_t i = nextJob++; i < jobs.size(); i = nextJob++) {
                    SlotWork &sw = work[jobs[i]];
                    tw::gpu::SearchJob &j = dj[i];
                    j.start = sw.first;   // 0: the branch, k: the snapshot after tick k - 1
                    j.controls = offset[i];
                    j.ticks = static_cast<std::uint32_t>(sw.controls.size()) - sw.first;
                    j.firstTick = sw.first;
                    sw.judge->Export(&j.judge);
                    if (objectives != nullptr) sw.judge->ExportObjectives(objectives + tw::gpu::kScriptedDoubles * i);
                    std::memcpy(dc + offset[i], sw.controls.data() + sw.first, sizeof(std::uint32_t) * j.ticks);
                }
            })) {
            throw std::bad_alloc();
        }
        if (!gpu_->Submit(set, jc, jobs.size(), offset.back(), &error, after)) throw std::runtime_error(error);
        b.sets[k] = set;
        t.prepare += p1 - p0;
        t.fill += Tsc() - p1;
    }

    // Chunk k's results into its slots (the Pose re-judging, the CPU reruns noted per thread).
    void GpuFinishChunk(GpuBatch &b, std::vector<SlotResult> &slots, std::uint32_t k,
                        std::vector<std::vector<std::uint32_t>> &reruns, GpuTimes &t) {
        std::string error;
        const std::vector<std::uint32_t> &jobs = b.jobs[k];
        const std::uint64_t f0 = Tsc();
        tw::gpu::SearchStats st;
        const tw::gpu::SearchResult *results = gpu_->Collect(b.sets[k], &st, &error);
        if (results == nullptr && !jobs.empty()) throw std::runtime_error(error);
        const std::uint64_t f1 = Tsc();
        t.kernelMs += st.kernelMs;
        for (int c = 0; c < 8; ++c) t.clk[c] += st.clk[c];
        if (trace_ && results != nullptr) {   // lane use: ticks run / warps x 32 x their longest job
            for (std::size_t w = 0; w < jobs.size(); w += 32u) {
                std::uint32_t longest = 0u;
                for (std::size_t i = w; i < std::min(jobs.size(), w + 32u); ++i) longest = std::max(longest, results[i].ticks);
                t.laneTicks += 32u * static_cast<std::uint64_t>(longest);
            }
        }
        t.ticks += st.ticks;
        t.substeps += st.substeps;
        const bool pose = config_.evaluator.kind == CudaSearchEvaluatorKind::Pose && !gpuTargets_;
        const double *objectives = gpuTargets_ && config_.evaluator.kind == CudaSearchEvaluatorKind::Scripted
                                           ? gpu_->Scripted(b.sets[k], 0u, &error)   // the set's results (Collect)
                                           : nullptr;
        std::atomic<std::size_t> nextJob{0u};
        if (!pool_->Run([&](std::uint32_t th) {
                Worker &w = *workers_[th];
                std::uint64_t ticks = 0u;
                for (std::size_t i = nextJob++; i < jobs.size(); i = nextJob++) {
                    const tw::gpu::SearchResult &r = results[i];
                    SlotWork &sw = b.work[jobs[i]];
                    SlotResult &out = slots[jobs[i]];
                    if (r.lost != 0u || r.poseOverflow != 0u) {   // not the CPU's run: again on the CPU
                        reruns[th].push_back(jobs[i]);
                        continue;
                    }
                    ticks += r.ticks;
                    if (pose) {
                        // the device's near-best ticks through the host judge, in tick order, from the prefix's best
                        for (std::uint32_t q = 0; q < r.poseCount; ++q) {
                            const tw::gpu::PoseNear &p = r.pose[q];
                            CHmsDyna::CHmsStateDyna &frame = w.view.state->body.current;
                            frame.position = {p.pos[0], p.pos[1], p.pos[2]};
                            frame.rotationQuat.x = p.quat[0];
                            frame.rotationQuat.y = p.quat[1];
                            frame.rotationQuat.z = p.quat[2];
                            frame.rotationQuat.w = p.quat[3];
                            w.view.finish.reset();
                            sw.judge->SetEvaluationIndex(p.evaluationIndex);
                            sw.judge->SetPreviousPosition(frame.position);
                            sw.judge->OnTick(p.t, w.view);
                        }
                    } else {
                        sw.judge->Import(r.judge);
                        if (objectives != nullptr) sw.judge->ImportObjectives(objectives + tw::gpu::kScriptedDoubles * i);
                    }
                    out.best = sw.judge->Best();
                    out.status = r.verdict == tw::gpu::kJudgeUnsupported ? SlotStatus::Unsupported : SlotStatus::Success;
                }
                w.times.ticks += ticks;
            })) {
            throw std::bad_alloc();
        }
        t.wait += f1 - f0;
        t.results += Tsc() - f1;
    }

    // The batch in chunks (FOREVERVALIDATOR_TWIN_SEARCH_GPU_CHUNK slots, default 65536), pipelined: while the device
    // runs chunk k (and the tail of k - 1), the session's threads prepare chunk k + 1 and take chunk k - 1's results.
    // Run ahead (default; FOREVERVALIDATOR_TWIN_SEARCH_GPU_AHEAD=0: not; one-chunk batches, evaluators whose judge does not read the
    // incumbent, i.e. not finish time): the next batch (the next ids, the same size) is prepared and submitted before
    // this one's results are taken, so the device never waits for the caller; the next RunBatch takes it if it asks
    // for exactly that batch with the same mutation program (else it is dropped). Its results are those of running
    // it then: nothing it computes depends on what this batch changes.
    void RunGpuBatch(std::uint64_t first, std::uint32_t count, bool baseline, const DeviceSample &incumbent,
                     const std::function<bool()> &cancel, std::vector<SlotResult> &slots) {
        RunGpuBatchOnce(first, count, baseline, incumbent, cancel, slots);
        // around walls: the corners the device's queries needed before they were filled, filled here, and the batch again
        // (CudaSearchExecutor's ExecuteFilled)
        while (aroundWalls_ != nullptr && FillDeviceMissedCorners()) {
            std::fill(slots.begin(), slots.end(), SlotResult{});
            RunGpuBatchOnce(first, count, baseline, incumbent, cancel, slots);
        }
    }

    // The tables' corners the device marked missed: filled on the host copy (the session's threads), the device's copy
    // updated, its marks cleared; whether there were any.
    bool FillDeviceMissedCorners() {
        namespace around_walls = forevervalidator::experimental::around_walls;
        std::string error;
        bool any = false;
        std::unique_lock<std::shared_mutex> lock(aroundWallsMutex_);
        for (std::uint32_t trigger = 0u; trigger < config_.evaluator.distanceTriggerCount; ++trigger) {
            const std::uint64_t offset = config_.evaluator.aroundWallsOffsets[trigger];
            unsigned char *bytes = aroundWalls_ + offset;
            around_walls::Header header;
            std::memcpy(&header, bytes, sizeof header);
            std::uint32_t flag = 0u;
            if (!gpu_->ReadTables(reinterpret_cast<unsigned char *>(&flag), offset + header.missed, sizeof flag, &error)) {
                throw std::runtime_error(error);
            }
            if (flag == 0u) continue;
            any = true;
            const std::size_t missedBytes = around_walls::MissedBytes(header.cornerCount);
            std::vector<std::uint32_t> words(missedBytes / 4u);
            if (!gpu_->ReadTables(reinterpret_cast<unsigned char *>(words.data()), offset + header.missed, missedBytes, &error)) {
                throw std::runtime_error(error);
            }
            const auto *corners = reinterpret_cast<const around_walls::Corner *>(bytes + header.corners);
            std::vector<std::uint64_t> missed;
            for (std::size_t word = 1u; word < words.size(); ++word) {
                for (std::uint32_t bit = 0u; words[word] != 0u && bit < 32u; ++bit) {
                    const std::uint64_t corner = (word - 1u) * 32u + bit;
                    if ((words[word] >> bit & 1u) != 0u && corners[corner].direct == around_walls::kCornerUnfilled) {
                        missed.push_back(corner);
                    }
                }
            }
            std::atomic<std::size_t> next{0u};
            if (!pool_->Run([&](std::uint32_t) {
                    for (std::size_t i; (i = next.fetch_add(1u)) < missed.size();) around_walls::FillCorner(bytes, missed[i]);
                })) {
                throw std::bad_alloc();
            }
            std::fill(words.begin(), words.end(), 0u);
            if (!gpu_->WriteTables(reinterpret_cast<const unsigned char *>(words.data()), offset + header.missed, missedBytes, &error) ||
                !gpu_->WriteTables(bytes + header.corners, offset + header.corners, header.cornerCount * sizeof(around_walls::Corner),
                                   &error)) {
                throw std::runtime_error(error);
            }
            if (!missed.empty() && config_.aroundWallsFilled) config_.aroundWallsFilled(trigger, bytes);
        }
        return any;
    }

    void RunGpuBatchOnce(std::uint64_t first, std::uint32_t count, bool baseline, const DeviceSample &incumbent,
                         const std::function<bool()> &cancel, std::vector<SlotResult> &slots) {
        if (gpuGen_ && !baseline) {
            RunGpuBatchDevice(first, count, incumbent, cancel, slots);
            return;
        }
        std::string error;
        BuildPrefix(baseline, incumbent);
        const tw::gpu::JudgeConfig jc = GpuJudgeConfig(incumbent);
        const std::uint32_t chunk = gpuChunk_ == 0u ? std::max(1u, count) : gpuChunk_;
        const std::uint32_t chunks = (count + chunk - 1u) / chunk;
        std::vector<std::vector<std::uint32_t>> reruns(threads_);
        GpuTimes times;
        const std::uint64_t a0 = Tsc();
        GpuBatch b;
        bool tookAhead = false;
        if (ahead_.valid) {
            if (ahead_.first == first && ahead_.count == count && ahead_.version == programVersion_ && !baseline) {
                b = std::move(ahead_);
                slots.swap(b.slots);
                tookAhead = true;
            } else {
                for (std::size_t k = 0; k < ahead_.jobs.size(); ++k) {   // drop it
                    tw::gpu::SearchStats st;
                    gpu_->Collect(ahead_.sets[k], &st, &error);
                }
            }
            ahead_ = GpuBatch{};
        }
        if (!tookAhead) {
            if (!gpu_->Reserve(std::min(chunk, count), static_cast<std::size_t>(std::min(chunk, count)) * timelineTickCount_, &error)) {
                throw std::runtime_error(error);
            }
            b.first = first;
            b.count = count;
            b.version = programVersion_;
            b.work.resize(count);
            b.jobs.resize(chunks);
            b.sets.resize(chunks);
        }
        const bool runAhead = gpuAhead_ && !baseline && chunks == 1u && config_.evaluator.kind != CudaSearchEvaluatorKind::FinishTime &&
                              aroundWalls_ == nullptr &&
                              first <= std::numeric_limits<std::uint64_t>::max() - 2ull * count;
        if (!tookAhead) {
            for (std::uint32_t k = 0; k < chunks; ++k) {
                GpuPrepareChunk(b, slots, k, k * chunk, std::min(count, k * chunk + chunk), nextSet_, baseline, incumbent, cancel, jc,
                                times);
                nextSet_ = (nextSet_ + 1) % gpu_->Sets();
                if (k > 0u) GpuFinishChunk(b, slots, k - 1u, reruns, times);
            }
        }
        if (runAhead) {
            GpuBatch &n = ahead_;
            n.first = first + count;
            n.count = count;
            n.version = programVersion_;
            n.work.resize(count);
            n.slots.resize(count);
            n.jobs.resize(1u);
            n.sets.resize(1u);
            // its kernel follows this batch's (no sharing of the device with it: a batch's own latency stays its own)
            GpuPrepareChunk(n, n.slots, 0u, 0u, count, nextSet_, false, incumbent, std::function<bool()>(), jc, times, b.sets[0]);
            nextSet_ = (nextSet_ + 1) % gpu_->Sets();
            n.valid = true;
        }
        if (chunks > 0u) GpuFinishChunk(b, slots, chunks - 1u, reruns, times);
        const std::uint64_t a1 = Tsc();
        std::vector<std::uint32_t> rerun;
        for (const auto &r : reruns) rerun.insert(rerun.end(), r.begin(), r.end());
        Worker &w0 = *workers_[0];
        if (!rerun.empty()) {
            JobDeal deal(rerun, 1u);
            RunJobs(w0, 0u, 0u, slots, b.work, deal, 1, cancel);
        }
        // (D) duplicates: the original's result, their own identity
        for (std::uint32_t i = 0; i < count; ++i) {
            SlotWork &sw = b.work[i];
            if (sw.original < 0) continue;
            SlotResult &r = slots[i];
            const SlotResult &orig = slots[static_cast<std::size_t>(sw.original)];
            if (orig.status == SlotStatus::Cancelled) {
                r.status = SlotStatus::Cancelled;
                r.active = false;
                continue;
            }
            r.status = orig.status;
            if (!orig.best.valid) {
                r.best = DeviceSample{};
                continue;
            }
            tw::gpu::JudgeState j;
            sw.judge->Export(&j);   // reported / index are not read below
            j.score = orig.best.score;
            j.timeMs = orig.best.timeMs;
            j.detail0 = orig.best.detail0;
            j.detail1 = orig.best.detail1;
            j.evaluationTick = orig.best.evaluationTick;
            j.valid = 1u;
            j.preciseFinish = orig.best.preciseFinish ? 1u : 0u;
            sw.judge->Import(j);
            r.best = sw.judge->Best();
        }
        const std::uint64_t a2 = Tsc();
        w0.times.physics += static_cast<std::uint64_t>(times.kernelMs * tscPerMs_);
        if (trace_) {
            std::size_t jobs = 0u;
            for (const auto &l : b.jobs) jobs += l.size();
            std::fprintf(stderr,
                         "twin gpu batch: %u slots in %u chunks%s%s, %zu jobs (%zu again on the CPU), %.3f ms: prepare %.3f, fill %.3f, "
                         "wait %.3f, results %.3f, after %.3f; kernels %.3f ms, %llu ticks, %.2f substeps/tick, lanes %.1f%% used\n",
                         count, chunks, tookAhead ? " (run ahead)" : "", runAhead ? " (next run ahead)" : "", jobs, rerun.size(),
                         double(a2 - a0) / tscPerMs_, double(times.prepare) / tscPerMs_, double(times.fill) / tscPerMs_,
                         double(times.wait) / tscPerMs_, double(times.results) / tscPerMs_, double(a2 - a1) / tscPerMs_,
                         times.kernelMs, static_cast<unsigned long long>(times.ticks),
                         times.ticks ? double(times.substeps) / double(times.ticks) : 0.0,
                         times.laneTicks ? 100.0 * double(times.ticks) / double(times.laneTicks) : 0.0);
            const unsigned long long *clk = times.clk;
            if (clk[7] != 0u) {
                const double t = double(clk[7]);
                std::fprintf(stderr, "twin gpu clocks (thread time): forces %.1f%%, collision %.1f%%, response %.1f%%, tick start %.1f%%, "
                             "finish snapshots %.1f%%, judge %.1f%%, outside the stages %.1f%%; %.0f cycles per substep\n",
                             100.0 * clk[0] / t, 100.0 * clk[1] / t, 100.0 * clk[2] / t, 100.0 * clk[3] / t, 100.0 * clk[4] / t,
                             100.0 * clk[6] / t, 100.0 * (t - clk[0] - clk[1] - clk[2] - clk[3] - clk[4] - clk[6]) / t,
                             times.substeps ? t / double(times.substeps) : 0.0);
            }
        }
    }

    // ---- the batch's per-candidate work on the device (twin_search_device.cu) ----------------------------------------
    // Each chunk on its job set's stream: generation, controls, the first change and the shared prefix's judge per slot,
    // the physics slots ordered and folded, the lane runner, every slot's outcome; the host takes the outcomes (Pose:
    // re-judges the near-best ticks; lost contacts / Pose overflow: the slot again on the CPU), and the winner's events
    // are generated again on the host (Execute). Run ahead as the host path, also for finish time: the batch run ahead
    // is taken only if its judge's incumbent is the caller's.
    // A batch's chunks, or a group of `batches` batches of `count` slots in one chunk (ids first.., one kernel: the
    // batches after the first run ahead in it; RunBatch takes them in order).
    struct DevBatch {
        std::uint64_t first = 0u;
        std::uint32_t count = 0u;      // slots a batch
        std::uint32_t batches = 1u;    // batches in the chunk (a group; 1 when the batch has several chunks)
        std::uint32_t taken = 0u;      // the group's batches returned so far
        bool waited = false;           // the group's chunk is done (Wait)
        bool overlapped = false;       // submitted while the batch before it still ran (it could fill that one's tail)
        bool firstGroup = false;       // a batch not run ahead, in a group with the next (DevTake)
        std::uint64_t version = 0u;
        tw::gpu::JudgeConfig jc{};
        bool valid = false;
        std::vector<int> sets;
        std::vector<std::uint32_t> lo, hi;   // chunk k's slots in the chunk array [lo, hi) (a group: [0, batches x count))
    };
    static bool SameIncumbent(const tw::gpu::JudgeConfig &a, const tw::gpu::JudgeConfig &b) {
        return a.incumbentPrecise == b.incumbentPrecise && a.incumbentTimeMs == b.incumbentTimeMs &&
               a.incumbentUpperNs == b.incumbentUpperNs;
    }
    struct DevTimes {
        std::uint64_t submit = 0u, wait = 0u, results = 0u;
        double kernelMs = 0.0, genMs = 0.0, outMs = 0.0;
        std::uint64_t ticks = 0u, substeps = 0u, physics = 0u, jobs = 0u;
        unsigned long long clk[8] = {};
    };

    void DevSubmit(DevBatch &b, std::uint32_t k, int set, const tw::gpu::JudgeConfig &jc, DevTimes &t, int after = -1) {
        std::string error;
        const std::uint64_t s0 = Tsc();
        const std::uint32_t lo = b.lo[k], n = b.hi[k] - b.lo[k];
        devPrefix_.resize(prefix_.size());
        for (std::size_t i = 0; i < prefix_.size(); ++i) {
            const PrefixJudge &p = prefix_[i];
            tw::gpu::JudgeState &j = devPrefix_[i];
            j.score = p.score;
            j.timeMs = p.timeMs;
            j.detail0 = p.detail0;
            j.detail1 = p.detail1;
            j.evaluationTick = p.evaluationTick;
            j.evaluationIndex = p.evaluationIndex;
            j.valid = p.valid;
            j.preciseFinish = p.preciseFinish;
            j.reported = p.reported;
            for (int a = 0; a < 3; ++a) j.previous[a] = p.previous[a];
        }
        const std::uint8_t stopStatus = prefixStopVerdict_ == TickJudge::Verdict::Unsupported ? 3u : 0u;
        void *stream = gpu_->Stream(set);
        if (!gpuGen_->Generate(set, stream, b.first + lo, n, false, program_.changedSegmentCount, devPrefix_.data(),
                               static_cast<std::uint32_t>(devPrefix_.size()), prefixStop_, stopStatus, &error) ||
            !gpu_->SubmitJobs(set, jc, gpuGen_->Jobs(set), gpuGen_->Controls(set), gpuGen_->JobCount(set), n, &error, after) ||
            !gpuGen_->Finish(set, gpu_->PostStream(set), gpu_->DeviceResults(set), config_.evaluator.kind == CudaSearchEvaluatorKind::Pose,
                             &error)) {
            throw std::runtime_error(error);
        }
        gpu_->MarkPost(set);
        b.sets[k] = set;
        t.submit += Tsc() - s0;
    }

    // Chunk k's outcomes into the batch's slots: a batch's chunk k (slots lo.. of the batch), or the group's batch `part`
    // (the chunk's slots part x count.., the batch's slots 0..). The chunk is waited for once.
    void DevFinish(DevBatch &b, std::uint32_t k, const DeviceSample &incumbent, std::vector<SlotResult> &slots,
                   std::vector<std::vector<std::uint32_t>> &reruns, DevTimes &t, std::uint32_t part = 0u) {
        std::string error;
        const std::uint64_t f0 = Tsc();
        tw::gpu::SearchStats st;
        const int set = b.sets[k];
        const bool group = b.batches > 1u;
        if (!group || !b.waited) {
            if (!gpu_->Wait(set, &st, &error)) throw std::runtime_error(error);
            b.waited = true;
            t.kernelMs += st.kernelMs;
            t.ticks += st.ticks;
            t.substeps += st.substeps;
            t.physics += gpuGen_->PhysicsSlots(set);
            t.genMs += gpuGen_->GenerationMs(set);
            t.outMs += gpuGen_->OutcomesMs(set);
            t.jobs += gpuGen_->JobsRun(set);
            for (int c = 0; c < 8; ++c) t.clk[c] += st.clk[c];
            workers_[0]->times.ticks += st.ticks;
        }
        const std::uint64_t f1 = Tsc();
        const std::uint32_t at = group ? part * b.count : 0u;   // the batch's first slot in the chunk's arrays
        const twin_search::DeviceSlot *d = gpuGen_->Slots(set) + at;
        const tw::gpu::SearchResult *results = gpuGen_->Results(set);
        const std::uint32_t lo = group ? 0u : b.lo[k], n = group ? b.count : b.hi[k] - b.lo[k];
        const std::uint64_t first = b.first + static_cast<std::uint64_t>(at);
        std::atomic<std::uint32_t> next{0u};
        constexpr std::uint32_t kStep = 256u;
        if (!pool_->Run([&](std::uint32_t th) {
                Worker &w = *workers_[th];
                for (std::uint32_t i0 = next.fetch_add(kStep); i0 < n; i0 = next.fetch_add(kStep)) {
                    for (std::uint32_t i = i0; i < std::min(n, i0 + kStep); ++i) {
                        const twin_search::DeviceSlot &x = d[i];
                        const std::uint32_t slot = lo + i;
                        SlotResult &r = slots[slot];
                        r.mutationCount = x.mutationCount;
                        r.active = (x.flags & twin_search::kSlotActive) != 0u;
                        r.status = static_cast<SlotStatus>(x.status);
                        if ((x.flags & twin_search::kSlotHost) != 0u ||
                            (gpuTestRerun_ != 0u && (x.flags & twin_search::kSlotPhysics) != 0u && slot % gpuTestRerun_ == 0u)) {
                            reruns[th].push_back(slot);
                            continue;
                        }
                        if ((x.flags & twin_search::kSlotPose) != 0u) {
                            // the device's near-best ticks through the host judge, in tick order, from the prefix's judge
                            TickJudge judge(*this, first + slot, slot, false, x.eventCount, incumbent);
                            if (x.first > 0u) judge.Import(prefix_[x.first - 1u]);
                            const tw::gpu::SearchResult &res = results[x.job];
                            for (std::uint32_t q = 0; q < res.poseCount; ++q) {
                                const tw::gpu::PoseNear &p = res.pose[q];
                                CHmsDyna::CHmsStateDyna &frame = w.view.state->body.current;
                                frame.position = {p.pos[0], p.pos[1], p.pos[2]};
                                frame.rotationQuat.x = p.quat[0];
                                frame.rotationQuat.y = p.quat[1];
                                frame.rotationQuat.z = p.quat[2];
                                frame.rotationQuat.w = p.quat[3];
                                w.view.finish.reset();
                                judge.SetEvaluationIndex(p.evaluationIndex);
                                judge.SetPreviousPosition(frame.position);
                                judge.OnTick(p.t, w.view);
                            }
                            r.best = judge.Best();
                        } else if ((x.flags & twin_search::kSlotJudge) != 0u && x.valid != 0u) {
                            // TickJudge::Import: the judge's best with this slot's identity
                            // (the batch's slots are new: r.best is DeviceSample{} already, so only its set fields are written)
                            DeviceSample &sample = r.best;
                            sample.score = x.score;
                            sample.timeMs = x.timeMs;
                            sample.detail0 = x.detail0;
                            sample.detail1 = x.detail1;
                            sample.valid = true;
                            sample.candidateId = first + slot;
                            sample.candidateSlot = slot;
                            sample.evaluationTick = x.evaluationTick;
                            sample.eventCount = x.eventCount;
                            sample.logicalOrder = 1u + static_cast<std::uint64_t>(slot) * evaluationTickCount_ + x.evaluationTick;
                            sample.mutation = true;
                            sample.preciseFinish = x.preciseFinish != 0u;
                        }
                    }
                }
            })) {
            throw std::bad_alloc();
        }
        t.wait += f1 - f0;
        t.results += Tsc() - f1;
    }

    // The batches run ahead, dropped (their sets' work waited for: the sets are free again).
    void DropAhead() {
        std::string error;
        for (const DevBatch &d : devAhead_) {
            for (int set : d.sets) {
                tw::gpu::SearchStats st;
                gpu_->Wait(set, &st, &error);
            }
        }
        devAhead_.clear();
    }

    void RunGpuBatchDevice(std::uint64_t first, std::uint32_t count, const DeviceSample &incumbent,
                           const std::function<bool()> &cancel, std::vector<SlotResult> &slots) {
        DevTake(first, count, incumbent, cancel, &slots);
        DevAhead(count, incumbent);
        DevEnd(first, incumbent, cancel, slots);
    }

    // A batch on the device path in three parts: DevTake submits it (or takes it from the batches run ahead), DevAhead
    // submits the next ones (after a taken group's readiness wait), DevEnd waits and takes the outcomes into the slots.
    // Execute makes the batch's slots between DevTake and DevAhead (DevEarly: one chunk, so DevTake needs no slots): the
    // host's first touch of them overlaps the device's work (the batch's generation, or a taken batch's lanes during the
    // readiness wait) instead of following it.
    struct DevRun {
        bool cancelled = false, tookAhead = false, ranAhead = false;
        std::uint32_t chunks = 0u, part = 0u;
        DevTimes times;
        std::uint64_t e0 = 0u, a0 = 0u;   // Execute's start (trace), DevTake's start
        Clock::time_point taken{};        // DevTake's start (DevAhead's readiness wait counts from it)
        std::vector<std::vector<std::uint32_t>> reruns;
    };
    bool DevEarly(std::uint32_t count) const {
        const std::uint32_t chunk = gpuChunk_ == 0u ? std::max(1u, count) : gpuChunk_;
        return count <= chunk;
    }
    void DevTake(std::uint64_t first, std::uint32_t count, const DeviceSample &incumbent, const std::function<bool()> &cancel,
                 std::vector<SlotResult> *slotsOrNull) {
        std::string error;
        eventsOmitted_ = true;
        const std::uint64_t a0 = Tsc();
        devRun_.taken = Clock::now();
        BuildPrefix(false, incumbent);
        const tw::gpu::JudgeConfig jc = GpuJudgeConfig(incumbent);
        const bool finishTime = config_.evaluator.kind == CudaSearchEvaluatorKind::FinishTime;
        const std::uint32_t chunk = gpuChunk_ == 0u ? std::max(1u, count) : gpuChunk_;
        const std::uint32_t chunks = (count + chunk - 1u) / chunk;
        DevRun &run = devRun_;
        run.cancelled = false;
        run.ranAhead = false;
        run.chunks = chunks;
        run.times = DevTimes{};
        run.a0 = a0;
        run.reruns.assign(threads_, {});
        std::vector<std::vector<std::uint32_t>> &reruns = run.reruns;
        DevTimes &times = run.times;
        bool tookAhead = false;
        if (!devAhead_.empty()) {
            const DevBatch &f = devAhead_.front();
            if (f.count == count && f.taken < f.batches && first == f.first + static_cast<std::uint64_t>(f.taken) * count &&
                f.version == programVersion_ && (!finishTime || SameIncumbent(f.jc, jc))) {
                tookAhead = true;
            } else {
                DropAhead();
            }
        }
        const int sets = gpu_->Sets();
        // Run ahead (one-chunk batches): groups of gpuGroup_ batches in one kernel, and up to gpuDepth_ groups after the
        // current one, each on a set of its own, so the device has the next batches' blocks queued behind this one's
        // while the caller takes this batch (DevAhead). Finish time: one batch a group, one at most ahead, only after a
        // batch that kept the incumbent (a batch run ahead under an incumbent that then changes is dropped, a whole kernel
        // lost).
        const bool oneChunk = count <= chunk;
        const std::uint32_t most = !gpuAhead_ || !oneChunk || finishTime ? 1u : std::max(1u, std::min(gpuGroup_, chunk / count));
        if (!tookAhead) {
            if (cancel && cancel()) {
                run.cancelled = true;
                run.tookAhead = false;
                return;
            }
            const std::uint32_t slotsMost = std::min(chunk, count * most);
            if (!gpuGen_->Reserve(slotsMost, &error) || !gpu_->ReserveJobs(slotsMost, &error)) throw std::runtime_error(error);
            devAhead_.emplace_front();
            DevBatch &b = devAhead_.front();
            b.first = first;
            b.count = count;
            b.batches = 1u;
            b.version = programVersion_;
            b.jc = jc;
            // A batch not run ahead (the session's first, or after a batch run ahead was dropped) in one group with the batches after
            // it (`most`, FOREVERVALIDATOR_TWIN_SEARCH_GPU_GROUP_FIRST=n: n, 0 or 1: alone): one kernel, the longest jobs of both
            // first (alone, the first batch's short jobs ran before the next batch's long ones: speedtech 16,384 x 2 178k against
            // 244k); the batches after the group run ahead once its last batch is taken (DevAhead)
            const std::uint32_t firstGroup = gpuGroupFirst_ > 1u && oneChunk && most > 1u &&
                                                             first <= std::numeric_limits<std::uint64_t>::max() - (most + 1ull) * count
                                                     ? std::min(gpuGroupFirst_, most)
                                                     : 1u;
            if (firstGroup > 1u) {
                b.firstGroup = true;
                b.batches = firstGroup;
                b.sets.assign(1u, 0);
                b.lo.assign(1u, 0u);
                b.hi.assign(1u, count * firstGroup);
                DevSubmit(b, 0u, nextSet_, jc, times);
                nextSet_ = (nextSet_ + 1) % sets;
            } else {
                b.sets.assign(chunks, 0);
                for (std::uint32_t k = 0; k < chunks; ++k) {
                    b.lo.push_back(k * chunk);
                    b.hi.push_back(std::min(count, k * chunk + chunk));
                }
                for (std::uint32_t k = 0; k < chunks; ++k) {
                    if (k >= static_cast<std::uint32_t>(sets)) DevFinish(b, k - sets, incumbent, *slotsOrNull, reruns, times);
                    DevSubmit(b, k, nextSet_, jc, times);
                    nextSet_ = (nextSet_ + 1) % sets;
                }
            }
            b.valid = true;
        }
        run.tookAhead = tookAhead;
    }

    // The readiness of a group taken (the next groups' size) and the groups run ahead after the current one.
    void DevAhead(std::uint32_t count, const DeviceSample &incumbent) {
        DevRun &run = devRun_;
        if (run.cancelled) return;
        const tw::gpu::JudgeConfig jc = GpuJudgeConfig(incumbent);
        const bool finishTime = config_.evaluator.kind == CudaSearchEvaluatorKind::FinishTime;
        const std::uint32_t chunk = gpuChunk_ == 0u ? std::max(1u, count) : gpuChunk_;
        const bool tookAhead = run.tookAhead;
        DevTimes &times = run.times;
        const int sets = gpu_->Sets();
        const bool oneChunk = count <= chunk;
        const std::uint32_t depth = !gpuAhead_ || !oneChunk ? 0u : finishTime ? (lastBatchChanged_ ? 0u : 1u) : gpuDepth_;
        const std::uint32_t most = !gpuAhead_ || !oneChunk || finishTime ? 1u : std::max(1u, std::min(gpuGroup_, chunk / count));
        auto room = [&](std::uint64_t at, std::uint32_t batches) {   // ids at.. of `batches` batches, and the next batch after them
            return at <= std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(batches + 1u) * count;
        };
        // A batch not run ahead runs alone (its latency is the caller's), and so does the batch run ahead after it. Later
        // groups: a group's first batch already done when taken (the batch run ahead filled the tail of the one before:
        // light sessions) keeps the next groups at one batch; one still running doubles them, up to `most` (the longest
        // jobs set a kernel's span: more jobs a lane, longest first, under one span).
        if (!tookAhead) {
            devGroupNext_ = 1u;
        } else if (devAhead_.front().taken == 0u) {
            const DevBatch &f = devAhead_.front();
            // done, or done within a millisecond (a batch run ahead that ends with the one before it: light sessions)
            auto done = [&] { return f.waited || (gpuReadyLanes_ ? gpu_->LanesDone(f.sets[0]) : gpu_->Ready(f.sets[0])); };
            bool ready = done();
            // (within gpuReadyUs_ of DevTake: Execute makes the batch's slots in between)
            for (const auto until = run.taken + std::chrono::microseconds(gpuReadyUs_); !ready && Clock::now() < until;) {
                std::this_thread::yield();
                ready = done();
            }
            // a group that started on an idle device (the one before it was done when it was submitted) says nothing
            if (gpuGroupAlways_ || (!ready && f.overlapped)) devGroupNext_ = std::min(2u * devGroupNext_, most);
            else if (ready) devGroupNext_ = 1u;
        }
        DevBatch &b = devAhead_.front();
        const std::uint32_t part = b.taken;
        // groups after the current one, up to `depth` (the current one keeps its set until its last batch is taken: a
        // set per group in flight, Sets() = depth + 1)
        bool ranAhead = false;
        // a first group (DevTake): its batches after the first are the ones run ahead; the batches after the group run ahead once its
        // last batch is taken (FOREVERVALIDATOR_TWIN_SEARCH_GPU_FIRST_AHEAD, measurement: =1, at once, their lanes in the group's tail:
        // offroad 16,384 x 2 2.19M against 2.62M, pressforward 16,384 x 8 1.42M against 1.56M; =follow, at once, their lanes after the
        // group's outcomes (SearchEngine::MarkPost): pressforward 16,384 x 2 1.42M against 1.53M, x 8 1.59M against 1.58M)
        const bool firstHead = b.firstGroup && part + 1u < b.batches;
        if (!(firstHead && gpuFirstAhead_ == 0)) {
            while (devAhead_.size() < 1u + depth) {
                const DevBatch &back = devAhead_.back();
                const std::uint64_t at = back.first + static_cast<std::uint64_t>(back.batches) * count;
                std::uint32_t g = std::min(devGroupNext_, most);
                while (g > 1u && !room(at, g)) --g;
                if (!room(at, g)) break;
                DevBatch n;
                n.overlapped = !(back.waited || gpu_->LanesDone(back.sets.back()));
                n.first = at;
                n.count = count;
                n.batches = g;
                n.version = programVersion_;
                n.jc = jc;
                n.sets.assign(1u, 0);
                n.lo.assign(1u, 0u);
                n.hi.assign(1u, count * g);
                // its lanes start as the batches before it leave the device's slots (FOLLOW=1: once this batch's kernel has ended)
                const bool follow = gpuFollow_ || (gpuFollowTaken_ && tookAhead) || (firstHead && gpuFirstAhead_ == 2);
                DevSubmit(n, 0u, nextSet_, jc, times, follow && devAhead_.size() == 1u ? b.sets[0] : -1);
                nextSet_ = (nextSet_ + 1) % sets;
                n.valid = true;
                devAhead_.push_back(std::move(n));
                ranAhead = true;
            }
        }
        run.tookAhead = tookAhead;
        run.ranAhead = ranAhead;
        run.part = part;
    }

    void DevEnd(std::uint64_t first, const DeviceSample &incumbent, const std::function<bool()> &cancel,
                std::vector<SlotResult> &slots) {
        DevRun &run = devRun_;
        if (run.cancelled) {
            for (SlotResult &r : slots) {
                r.status = SlotStatus::Cancelled;
                r.active = false;
            }
            return;
        }
        const std::uint32_t count = static_cast<std::uint32_t>(slots.size());
        const std::uint32_t chunks = run.chunks, part = run.part;
        const bool tookAhead = run.tookAhead, ranAhead = run.ranAhead;
        const std::uint64_t a0 = run.a0;
        const int sets = gpu_->Sets();
        std::vector<std::vector<std::uint32_t>> &reruns = run.reruns;
        DevTimes &times = run.times;
        DevBatch &b = devAhead_.front();
        if (b.batches > 1u) {
            DevFinish(b, 0u, incumbent, slots, reruns, times, part);
        } else {
            const std::uint32_t pending = std::min<std::uint32_t>(chunks, static_cast<std::uint32_t>(sets));
            for (std::uint32_t k = chunks - pending; k < chunks; ++k) DevFinish(b, k, incumbent, slots, reruns, times);
        }
        if (++b.taken == b.batches) devAhead_.pop_front();
        const bool runAhead = ranAhead || !devAhead_.empty();
        const std::uint64_t a1 = Tsc();
        std::size_t rerunCount = 0u;
        Worker &w0 = *workers_[0];
        for (const auto &list : reruns) {
            for (const std::uint32_t slot : list) {   // the CPU's run: generated again, the serial path
                ++rerunCount;
                std::vector<SlotWork> work(1u);
                PrepareSlot(w0, first, slot, false, incumbent, cancel, &slots[slot], &work[0]);
                if (work[0].physics) {
                    JobDeal deal(std::vector<std::uint32_t>{0u}, 1u);
                    RunJobs(w0, 0u, slot, slots, work, deal, 1, cancel);
                }
            }
        }
        const std::uint64_t a2 = Tsc();
        w0.times.physics += static_cast<std::uint64_t>(times.kernelMs * tscPerMs_);
        if (trace_) {
            std::fprintf(stderr,
                         "twin gpu batch (device): %u slots in %u chunks%s%s (groups of %u next), %llu physics, %llu jobs (%zu again on the CPU), %.3f ms "
                         "(%.3f before it): submit %.3f, wait %.3f, results %.3f, after %.3f; generation %.3f ms, lanes %.3f ms, jobs to outcomes %.3f ms, "
                         "%llu ticks, %.2f substeps/tick\n",
                         count, chunks, tookAhead ? " (run ahead)" : "", runAhead ? " (next run ahead)" : "", devGroupNext_,
                         static_cast<unsigned long long>(times.physics), static_cast<unsigned long long>(times.jobs), rerunCount,
                         double(a2 - a0) / tscPerMs_, run.e0 != 0u && run.e0 <= a0 ? double(a0 - run.e0) / tscPerMs_ : 0.0,
                         double(times.submit) / tscPerMs_, double(times.wait) / tscPerMs_,
                         double(times.results) / tscPerMs_, double(a2 - a1) / tscPerMs_, times.genMs, times.kernelMs, times.outMs,
                         static_cast<unsigned long long>(times.ticks), times.ticks ? double(times.substeps) / double(times.ticks) : 0.0);
        }
    }
#else
    void RunGpuBatch(std::uint64_t, std::uint32_t, bool, const DeviceSample &, const std::function<bool()> &,
                     std::vector<SlotResult> &) {}
#endif

    // Re-simulates the winner on the verifier (FV reference from tick 0, or
    // optimized_cpu from the branch) up to its evaluation tick.
    bool Verify(const SlotResult &winner, const DeviceSample &incumbent) {
        const auto v0 = Clock::now();
        Worker w(program_);
        w.physics.reset(verifier_.release());
        BuildTicks(winner.events, &w.ticks);
        bool ok = verifierBackend_ == "reference" ? VerifierToBranchOn(*static_cast<FvPhysics *>(w.physics.get()))
                                                 : true;
        DeviceSample sample{};
        if (ok) {
            // Evaluate exactly as a candidate, then compare scores.
            ok = SimulateVerifier(w, winner, incumbent, &sample);
        }
        verifier_.reset(static_cast<FvPhysics *>(w.physics.release()));
        metrics_.winnerVerified = ok;
        metrics_.winnerVerifiedScore = sample.score;
        metrics_.winnerVerificationMatched = ok && sample.valid == winner.best.valid &&
                                             sample.score == winner.best.score &&
                                             sample.evaluationTick == winner.best.evaluationTick;
        // The winner's captured state: FV at the winning tick.
        if (ok && config_.captureBestState) {
            const std::shared_ptr<const ReplaySimulationInstanceClone> clone =
                    verifier_->Session().CaptureRuntimeClone();
            globalBestStateValid_ =
                    clone && EncodeCudaCandidateState(*clone, source_.validationSeed, bestControlCursor_, 0u,
                                                      clone->randomState, &globalBestState_) ==
                                     CudaStateConversionResult::Success;
        }
        metrics_.verificationMilliseconds += Ms(v0, Clock::now());
        return ok;
    }

    bool SimulateVerifier(Worker &w, const SlotResult &winner, const DeviceSample &incumbent, DeviceSample *out) {
        // Run the candidate loop but stop right after the winning tick so the
        // verifier session holds the winning state.
        const std::uint32_t stop = winner.best.evaluationTick;
        const std::uint32_t firstEvaluationTick =
                static_cast<std::uint32_t>((config_.evaluationStartTimeMs - config_.branchTimeMs) /
                                                   config_.tickDurationMs -
                                           1);
        const std::uint32_t lastTick = std::min(timelineTickCount_, firstEvaluationTick + stop + 1u);
        const std::uint32_t saved = timelineTickCount_;
        timelineTickCount_ = lastTick;
        const SlotStatus status = Simulate(w, winner.best.candidateId, winner.best.candidateSlot, !winner.best.mutation,
                                           winner.best.eventCount, incumbent, nullptr, out);
        timelineTickCount_ = saved;
        bestControlCursor_ = source_.branchCursor + lastTick;
        return status == SlotStatus::Success;
    }

    bool VerifierToBranch(std::string *diagnostic) { return VerifierToBranchOn(*verifier_, diagnostic); }

    // FV reference: tick 0 up to the branch on the sandbox's own timeline.
    bool VerifierToBranchOn(FvPhysics &physics, std::string *diagnostic = nullptr) {
        if (!source_.startSession(physics.Session())) {
            if (diagnostic) *diagnostic = "twin search: FV reference session could not be restarted";
            return false;
        }
        if (source_.branchCursor != 0u &&
            physics.Session().AdvanceIncremental(source_.timeline, 0u, source_.branchCursor).result !=
                    ReplaySimulationRunResult::Success) {
            if (diagnostic) *diagnostic = "twin search: FV reference could not replay to the branch";
            return false;
        }
        return true;
    }

    // BetterSample (cuda_search_winner_selection.cuh) as a choice: true when it returns `right`.
    static bool BetterPicksRight(const DeviceSample &left, const DeviceSample &right, bool maximize) {
        if (left.valid != right.valid) return !left.valid;
        if (!left.valid) return false;
        if (left.score != right.score) return maximize ? !(left.score > right.score) : !(left.score < right.score);
        if (left.eventCount != right.eventCount) return !(left.eventCount < right.eventCount);
        return !(left.logicalOrder <= right.logicalOrder);
    }

    CudaSearchBatchExecution Execute(std::uint64_t first, std::uint32_t count, bool baseline,
                                     const std::function<bool()> &cancel) noexcept {
        CudaSearchBatchExecution result;
        result.firstCandidateId = first;
        result.candidateCount = count;
        try {
            if ((!baseline && !baselineEvaluated_) || count == 0u || count > config_.maximumBatchSize) {
                result.status = CudaSearchStatus::InvalidArgument;
                result.diagnostic = !baselineEvaluated_ && !baseline
                                            ? "twin baseline must be evaluated before mutation batches"
                                            : "invalid twin search batch size";
                return result;
            }
            const auto w0 = Clock::now();
#if FOREVERVALIDATOR_TWIN_GPU
            devRun_.e0 = Tsc();   // trace: Execute's start
#endif
            metrics_ = TwinSearchHostMetrics{};
            metrics_.threads = threads_;
            metrics_.verificationBackend = verifierBackend_;
            result.evaluationCurrentTimeSeconds =
                    std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            clock_.currentSeconds = result.evaluationCurrentTimeSeconds;
            clock_.lastImprovementSeconds = config_.condition ? config_.condition->lastImprovementTimeSeconds : 0.0;
            clock_.lastRestartSeconds = config_.condition ? config_.condition->lastRestartTimeSeconds : 0.0;
            DeviceSample incumbent = globalBest_;
            incumbent.logicalOrder = 0u;
            incumbent.candidateSlot = InvalidCandidateSlot;
            tmnf::simulation::DeterministicExecutionScope scope;
            if (!scope.Established() && !tmnf::simulation::DeterministicExecutionScope::IsActive()) {
                result.status = CudaSearchStatus::DeviceFailure;
                result.diagnostic = "deterministic execution mode is unavailable";
                return result;
            }
            for (auto &w : workers_) w->times = StageTimes{};
            const bool recording = workers_[0]->recorder != nullptr && workers_[0]->recorder->Active();
            eventsOmitted_ = false;
#if FOREVERVALIDATOR_TWIN_GPU
            // the device path's batch submitted before its slots are made (16,384 slots are 6 MB: the host's first touch of
            // them overlaps the device's generation instead of preceding it)
            const bool devEarly = twinWorld_ && gpu_ && gpuGen_ && !baseline && !recording && !checkBranching_ && DevEarly(count);
            if (devEarly) DevTake(first, count, incumbent, cancel, nullptr);
#else
            constexpr bool devEarly = false;
#endif
            std::vector<SlotResult> slots(count);
            if (devEarly) {
#if FOREVERVALIDATOR_TWIN_GPU
                DevAhead(count, incumbent);
                DevEnd(first, incumbent, cancel, slots);
#endif
            } else if (twinWorld_ && gpu_ && !recording && !checkBranching_) {
                RunGpuBatch(first, count, baseline, incumbent, cancel, slots);
            } else if (twinWorld_ && batchPath_ && !recording && !checkBranching_) {
                RunTwinBatch(first, count, baseline, incumbent, cancel, slots);
            } else if (threads_ == 1u) {
                for (std::uint32_t slot = 0; slot < count; ++slot) {
                    RunSlot(*workers_[0], first, slot, baseline, incumbent, cancel, &slots[slot]);
                }
            } else {
                std::atomic<std::uint32_t> next{0u};
                if (!pool_->Run([&](std::uint32_t t) {
                        for (std::uint32_t slot = next++; slot < count; slot = next++) {
                            RunSlot(*workers_[t], first, slot, baseline, incumbent, cancel, &slots[slot]);
                        }
                    })) {
                    throw std::bad_alloc();
                }
            }
            for (auto &w : workers_) {
                metrics_.generationMilliseconds += static_cast<double>(w->times.generation) / tscPerMs_;
                metrics_.physicsMilliseconds += static_cast<double>(w->times.physics) / tscPerMs_;
                metrics_.evaluationMilliseconds += static_cast<double>(w->times.evaluation) / tscPerMs_;
                metrics_.simulatedTicks += w->times.ticks;
                metrics_.sharedPrefixTicks += w->sharedPrefixTicks;
                w->sharedPrefixTicks = 0u;
            }
            const auto f0 = Clock::now();
            // FinalizeSearchBatchKernel: status, counts, improvement count.
            result.status = CudaSearchStatus::Success;
            // One pass over the slots (16,384 slots are 6 MB): the counts, the improvement walk (mutation batches) and the
            // candidate-best reduction (BetterSample over incumbent + slots; its pick and the walk by reference: no 330-byte
            // copy per slot), each in slot order as three passes would be.
            const bool scripted = config_.evaluator.kind == CudaSearchEvaluatorKind::Scripted;
            const DeviceSample *walk = &incumbent;
            const DeviceSample *pick = &incumbent;
            for (const SlotResult &s : slots) {
                if (s.status == SlotStatus::Cancelled) result.status = CudaSearchStatus::Cancelled;
                else if (s.status == SlotStatus::CapacityExceeded) result.status = CudaSearchStatus::CapacityExceeded;
                else if (s.status == SlotStatus::Unsupported)
                    result.status = CudaSearchStatus::UnsupportedPhysicsTransition;
                if (s.active) {
                    ++result.evaluatedCandidateCount;
                    result.evaluatorCalls += evaluationTickCount_;
                }
                result.totalMutationCount += s.mutationCount;
                if (scripted) {   // SelectScriptedWinnerKernel: the first to dominate the incumbent so far
                    if (StrictlyDominates(s.best, *walk)) {
                        if (!baseline) ++result.mutationImprovementCount;
                        walk = &s.best;
                    }
                    pick = walk;
                    continue;
                }
                if (!baseline && StrictlyBetter(s.best, *walk, maximize_)) {
                    ++result.mutationImprovementCount;
                    walk = &s.best;
                }
                if (BetterPicksRight(*pick, s.best, maximize_)) pick = &s.best;
            }
            const DeviceSample winner = *pick;
            if (winner.valid && winner.candidateSlot != InvalidCandidateSlot && slots[winner.candidateSlot].best.valid) {
                SlotResult &s = slots[winner.candidateSlot];
                if (eventsOmitted_ && s.events.empty()) {   // the device path keeps no events: the winner's again
                    twin_search::GeneratedCandidate again;
                    workers_[0]->generator.Generate(first + winner.candidateSlot, baseline, &again);
                    s.events = std::move(again.events);
                }
                globalBest_ = s.best;
                globalBestEvents_ = s.events;
                globalBestMutationCount_ = s.mutationCount;
                globalBestStateValid_ = false;
                result.bestChanged = true;
                metrics_.selectionMilliseconds += Ms(f0, Clock::now());
                if (verifier_) Verify(s, incumbent);
            } else {
                metrics_.selectionMilliseconds += Ms(f0, Clock::now());
            }
            const auto a0 = Clock::now();
            if (globalBest_.valid) {
                CudaSearchBest &best = result.best;
                best.valid = true;
                best.mutation = globalBest_.mutation;
                best.stateCaptured = config_.captureBestState && globalBestStateValid_;
                best.candidateId = globalBest_.candidateId;
                best.mutationCount = globalBestMutationCount_;
                best.evaluationTick = globalBest_.evaluationTick;
                best.score = globalBest_.score;
                best.timeMs = globalBest_.timeMs;
                best.detail0 = globalBest_.detail0;
                best.detail1 = globalBest_.detail1;
                best.scriptedObjectiveCount = globalBest_.scriptedObjectiveCount;
                for (std::uint32_t i = 0u; i < globalBest_.scriptedObjectiveCount; ++i) {
                    best.objectiveScores[i] = globalBest_.objectiveScores[i];
                    best.metricValues[i] = globalBest_.metricValues[i];
                }
                if (best.stateCaptured) best.state = globalBestState_;
                best.inputs = program_.immutablePrefix;
                for (CudaSearchInputEvent e : globalBestEvents_) {
                    best.inputs.push_back(cuda_search_detail::AbsoluteSuffixEvent(e, program_.mutableFromTimeMs));
                }
                for (CudaSearchInputEvent e : program_.immutableTail) {
                    best.inputs.push_back(cuda_search_detail::AbsoluteSuffixEvent(e, program_.mutableFromTimeMs));
                }
            }
            lastBatchChanged_ = result.bestChanged;
            if (baseline && result.status == CudaSearchStatus::Success) baselineEvaluated_ = true;
            if (result.status != CudaSearchStatus::Success && result.diagnostic.empty()) {
                result.diagnostic = std::string("twin search batch status: ") + CudaSearchStatusName(result.status);
            }
            metrics_.selectionMilliseconds += Ms(a0, Clock::now());
            metrics_.wallMilliseconds = Ms(w0, Clock::now());
            result.kernelMilliseconds = metrics_.wallMilliseconds;
            result.simulationKernelMilliseconds = metrics_.physicsMilliseconds;
            result.mutationKernelMilliseconds = metrics_.generationMilliseconds;
            result.winnerKernelMilliseconds = metrics_.selectionMilliseconds;
            result.winnerStateCaptureKernelMilliseconds = metrics_.verificationMilliseconds;
            return result;
        } catch (const std::bad_alloc &) {
            result.status = CudaSearchStatus::CapacityExceeded;
            result.diagnostic = "twin search host allocation failed";
            return result;
        } catch (...) {
            result.status = CudaSearchStatus::DeviceFailure;
            result.diagnostic = "unexpected twin search failure";
            return result;
        }
    }

    // EvaluateSearchState; around walls, the corners its queries need filled first (FillMissedCorners) and
    // the tick evaluated again. Queries hold the tables shared, a fill holds them alone.
    DeviceSample EvaluateTick(const CudaCandidatePhysicsState &state, const GmVec3 &previousPosition,
                              const GmVec3 &previousVelocity, double previousTimeMs, double currentTimeMs,
                              std::uint32_t stuntsScore, bool *reported) const {
        if (aroundWalls_ == nullptr) {
            return twin_search::EvaluateSearchState(config_.evaluator, state, previousPosition, previousVelocity,
                                                    previousTimeMs, currentTimeMs, stuntsScore, reported);
        }
        std::uint32_t &misses = forevervalidator::experimental::around_walls::detail::HostMisses();
        for (;;) {
            bool again = *reported;
            {
                std::shared_lock<std::shared_mutex> lock(aroundWallsMutex_);
                misses = 0u;
                const DeviceSample sample = twin_search::EvaluateSearchState(
                        config_.evaluator, state, previousPosition, previousVelocity, previousTimeMs, currentTimeMs,
                        stuntsScore, &again);
                if (misses == 0u) {
                    *reported = again;
                    return sample;
                }
            }
            FillMissedCorners();
        }
    }
    // The corners the tables' missed bits name, as CudaSearchExecutor's FillMissedAroundWallsCorners
    // (aroundWallsFilled passes each table on).
    void FillMissedCorners() const {
        namespace around_walls = forevervalidator::experimental::around_walls;
        std::unique_lock<std::shared_mutex> lock(aroundWallsMutex_);
        for (std::uint32_t trigger = 0u; trigger < config_.evaluator.distanceTriggerCount; ++trigger) {
            unsigned char *bytes = aroundWalls_ + config_.evaluator.aroundWallsOffsets[trigger];
            around_walls::Header header;
            std::memcpy(&header, bytes, sizeof header);
            auto *words = reinterpret_cast<std::uint32_t *>(bytes + header.missed);
            if (words[0] == 0u) continue;
            const auto *corners = reinterpret_cast<const around_walls::Corner *>(bytes + header.corners);
            const std::size_t wordCount = around_walls::MissedBytes(header.cornerCount) / 4u;
            bool filled = false;
            for (std::size_t word = 1u; word < wordCount; ++word) {
                for (std::uint32_t bit = 0u; words[word] != 0u && bit < 32u; ++bit) {
                    const std::uint64_t corner = (word - 1u) * 32u + bit;
                    if ((words[word] >> bit & 1u) != 0u && corners[corner].direct == around_walls::kCornerUnfilled) {
                        around_walls::FillCorner(bytes, corner);
                        filled = true;
                    }
                }
            }
            std::memset(words, 0, wordCount * 4u);
            if (filled && config_.aroundWallsFilled) config_.aroundWallsFilled(trigger, bytes);
        }
    }

    CudaSearchExecutorConfiguration config_;
    TwinSearchExecutorSource source_;
    twin_search::MutationProgram program_;
    std::shared_ptr<const ReplaySimulationInstanceClone> branch_;
    // the judge's: the race's constants, the search loop's clock (Execute), the table bytes around walls
    ReplayRaceProgress race_{};
    std::uint32_t replayPlayMode_ = 0u;
    twin_search::ProgramClock clock_;
    bool fullState_ = false;            // the judge reads more than the body (TwinPhysics::ReadFull)
    bool perCandidatePrefix_ = false;   // its programs read the iteration: no prefix judged once for all
    bool gpuTargets_ = false;           // the GPU judges with FV's evaluator code (tw_gpu_judge.cuh)
    unsigned char *aroundWalls_ = nullptr;
    mutable std::shared_mutex aroundWallsMutex_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::unique_ptr<FvPhysics> verifier_;
    // Twin mode (not delegate): the world and the imported branch state.
    std::shared_ptr<const experimental::twin::TwinWorld> twinWorld_;
    experimental::twin::TwinState twinBranch_{};
    // branchAtFirstChange: the baseline's controls, and its state and events after every tick.
    std::vector<experimental::twin::TwinControl> snapControls_;
    std::vector<experimental::twin::TwinState> snapStates_;
    std::vector<experimental::twin::TwinEvents> snapEvents_;
    std::unique_ptr<SearchRecorder> recorder_;
    bool checkBranching_ = false;                      // FOREVERVALIDATOR_TWIN_SEARCH_CHECK=1
    std::unique_ptr<WorkerPool> pool_;                 // the session's threads, across batches
    bool batchPath_ = false;                           // twin: the lane runner (header comment)
    int lanes_ = 16;
    bool dedupe_ = true;
    bool longestFirst_ = true;
    int laneMin_ = 16;   // jobs per worker below which a batch runs serially (FOREVERVALIDATOR_TWIN_SEARCH_LANE_MIN)
    bool trace_ = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_TRACE") != nullptr;   // per-batch phase times (stderr)
    std::atomic<std::uint64_t> checked_{0u}, checkDifferent_{0u};
    std::vector<PrefixJudge> prefix_;                  // BuildPrefix: the baseline's judge after every tick
    std::uint32_t prefixStop_ = std::numeric_limits<std::uint32_t>::max();
    TickJudge::Verdict prefixStopVerdict_ = TickJudge::Verdict::Continue;
    std::uint32_t gpuChunk_ = 131072u;  // FOREVERVALIDATOR_TWIN_SEARCH_GPU_CHUNK (0: the whole batch at once)
    bool gpuLexOrder_ = true;           // FOREVERVALIDATOR_TWIN_SEARCH_GPU_ORDER=length: longest first only
    bool gpuAhead_ = true;              // run the next batch ahead (FOREVERVALIDATOR_TWIN_SEARCH_GPU_AHEAD=0: not)
    std::uint32_t gpuDepth_ = 1u;       // device path: groups run ahead (FOREVERVALIDATOR_TWIN_SEARCH_GPU_AHEAD=n; finish time: 1)
    std::uint32_t gpuGroup_ = 2u;       // device path: batches a kernel (FOREVERVALIDATOR_TWIN_SEARCH_GPU_GROUP; finish time: 1)
    std::uint32_t devGroupNext_ = 1u;   // the next group's batches (doubles from 1 after a batch not run ahead)
    bool gpuReadyLanes_ = [] {          // FOREVERVALIDATOR_TWIN_SEARCH_GPU_READY=stream (measurement): the whole set done, not its lanes
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_READY");
        return v == nullptr || std::strcmp(v, "stream") != 0;
    }();
    std::uint32_t gpuReadyUs_ = [] {    // FOREVERVALIDATOR_TWIN_SEARCH_GPU_READY_US (measurement): how long a group's first batch may take to be done
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_READY_US");
        return v != nullptr ? static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10)) : 1000u;
    }();
    bool gpuGroupAlways_ = [] {         // FOREVERVALIDATOR_TWIN_SEARCH_GPU_GROUP_ALWAYS=1 (check): groups grow whatever the readiness
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_GROUP_ALWAYS");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    std::uint32_t gpuGroupFirst_ = [] { // FOREVERVALIDATOR_TWIN_SEARCH_GPU_GROUP_FIRST=n: a batch not run ahead in a group of n (default: the
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_GROUP_FIRST");   // groups' cap; 0 or 1: alone)
        return v != nullptr && *v != '\0' ? static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10)) : 16u;
    }();
    int gpuFirstAhead_ = [] {           // FOREVERVALIDATOR_TWIN_SEARCH_GPU_FIRST_AHEAD (measurement): run ahead past a first group at once
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_FIRST_AHEAD");   // (1), its lanes after the group's (follow)
        return v == nullptr ? 0 : std::strcmp(v, "1") == 0 ? 1 : std::strcmp(v, "follow") == 0 ? 2 : 0;
    }();
    bool gpuFollow_ = false;            // its lanes wait for this batch's kernel's end (FOREVERVALIDATOR_TWIN_SEARCH_GPU_FOLLOW=1)
    bool gpuFollowTaken_ = false;       // measurement (FOLLOW=taken): only the batches run ahead of a taken batch wait for its lanes' end
    int nextSet_ = 0;                   // the GPU engine's next job set
    std::uint64_t programVersion_ = 0u; // bumped when the mutation program changes (UpdateChangedSegmentCount)
    std::uint32_t gpuTestRerun_ = [] {   // FOREVERVALIDATOR_TWIN_SEARCH_GPU_TEST_RERUN=k (check): every k-th physics slot again on the CPU
        const char *v = std::getenv("FOREVERVALIDATOR_TWIN_SEARCH_GPU_TEST_RERUN");
        return v != nullptr ? static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10)) : 0u;
    }();
    bool lastBatchChanged_ = true;      // the last batch changed the best (finish time: no run ahead after it)
    bool eventsOmitted_ = false;        // this batch's slots carry no events (the device path): Execute generates the winner's
#if FOREVERVALIDATOR_TWIN_GPU
    std::unique_ptr<tw::gpu::SearchEngine> gpu_;   // FOREVERVALIDATOR_TWIN_SEARCH_ENGINE=gpu
    GpuBatch ahead_;                                // the next batch, run ahead
    std::unique_ptr<twin_search::DeviceGeneration> gpuGen_;   // the per-candidate work on the device (default)
    std::deque<DevBatch> devAhead_;                 // its next batches, run ahead (in order)
    DevRun devRun_;                                 // the batch between DevTake and DevEnd
    std::vector<tw::gpu::JudgeState> devPrefix_;
#else
    std::unique_ptr<int> gpu_;
#endif
    std::string verifierBackend_ = "reference";
    std::uint32_t threads_ = 1u;
    std::uint32_t evaluationTickCount_ = 0u;
    std::uint32_t timelineTickCount_ = 0u;
    bool maximize_ = false;
    bool delegate_ = false;
    double tscPerMs_ = 1.0;
    bool baselineEvaluated_ = false;
    DeviceSample globalBest_{};
    std::vector<CudaSearchInputEvent> globalBestEvents_;
    std::uint32_t globalBestMutationCount_ = 0u;
    CudaCandidateState globalBestState_{};
    bool globalBestStateValid_ = false;
    std::uint64_t bestControlCursor_ = 0u;
    TwinSearchHostMetrics metrics_;
};

}  // namespace

std::unique_ptr<TwinSearchExecutor> CreateTwinSearchExecutor(CudaSearchExecutorConfiguration configuration,
                                                             TwinSearchExecutorSource source,
                                                             std::string *diagnostic) {
    try {
        auto executor = std::make_unique<TwinSearchExecutorImpl>();
        if (!executor->Create(std::move(configuration), std::move(source), diagnostic)) return nullptr;
        return executor;
    } catch (const std::bad_alloc &) {
        if (diagnostic) *diagnostic = "twin search: allocation failed";
        return nullptr;
    }
}

}  // namespace forevervalidator::simulation
