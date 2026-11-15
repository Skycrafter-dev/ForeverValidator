// Twin public API (include/forevervalidator/experimental/twin.h).
//
// FV-dependent: compiled into forevervalidator_core when
// -DFOREVERVALIDATOR_BUILD_TWIN=ON (src/twin/CMakeLists.txt). The physics
// lives in the FV-free libraries; this file converts types and checks scope.
#include <forevervalidator/experimental/twin.h>

#include <cstring>
#include <new>

#include "twin_api_internal.h"
#include "twin_world_build.h"

#include "tw_controls.h"
#include "tw_race.h"
#include "tw_state.h"
#include "tw_step.h"

namespace forevervalidator::experimental::twin {

static_assert(sizeof(TwinState) == sizeof(tw::model::TwinState), "public and model TwinState sizes differ");
static_assert(alignof(TwinState) == alignof(tw::model::TwinState), "public and model TwinState alignments differ");

namespace {

PhysicsSandboxError Error(PhysicsSandboxErrorCode code, std::string diagnostic,
                          ValidationFailureReason reason = ValidationFailureReason::None) {
    PhysicsSandboxError error;
    error.code = code;
    error.diagnostic = std::move(diagnostic);
    if (reason != ValidationFailureReason::None) {
        error.validationError.category = ValidationErrorCategory::Simulation;
        error.validationError.code = reason == ValidationFailureReason::TwinUnsupportedSimulationScope
                ? ValidationErrorCode::TwinUnavailable
                : ValidationErrorCode::TwinExecutionFailed;
        error.validationError.stage = ValidationStage::SimulationStartup;
        error.validationError.reason = reason;
        error.validationError.diagnostic = error.diagnostic;
    }
    return error;
}

template <class T>
PhysicsSandboxResult<T> NotImplemented(const char *what) {
    return PhysicsSandboxResult<T>::Failure(Error(PhysicsSandboxErrorCode::SimulationFailed,
                                                  std::string("twin: ") + what + " is not implemented yet",
                                                  ValidationFailureReason::TwinNotImplemented));
}

TwinStatus FromStep(tw::step::Status status) {
    switch (status) {
    case tw::step::Status::Ok: return TwinStatus::Ok;
    case tw::step::Status::NotImplemented: return TwinStatus::NotImplemented;
    case tw::step::Status::InvalidArgument: return TwinStatus::InvalidArgument;
    case tw::step::Status::Stopped: return TwinStatus::Stopped;
    }
    return TwinStatus::InvalidArgument;
}

void FromStepEvents(const tw::step::Events &in, TwinEvents *out) {
    if (out == nullptr) return;
    out->status = FromStep(in.status);
    out->ticks = in.ticks;
    out->substeps = in.substeps;
    out->respawns = in.respawns;
    out->checkpoints = in.checkpoints;
    out->finished = in.finished;
    out->finishTickMs = in.finishTickMs;
    out->finishLowerNs = in.finishLowerNs;
    out->cpLastSlot = in.cpLastSlot;
    out->finishPassage = in.finishPassage;
    out->wheelContacts = in.wheelContacts;
    out->bodyContacts = in.bodyContacts;
}

tw::step::Events ToStepEvents(const TwinEvents *in) {
    tw::step::Events ev;
    if (in == nullptr) return ev;
    ev.ticks = in->ticks;
    ev.substeps = in->substeps;
    ev.respawns = in->respawns;
    ev.checkpoints = in->checkpoints;
    ev.finished = in->finished;
    ev.finishTickMs = in->finishTickMs;
    ev.finishLowerNs = in->finishLowerNs;
    ev.cpLastSlot = in->cpLastSlot;
    ev.finishPassage = in->finishPassage;
    ev.wheelContacts = in->wheelContacts;
    ev.bodyContacts = in->bodyContacts;
    return ev;
}

}  // namespace

// ---- internal helpers (twin_api_internal.h) ----------------------------------
namespace internal {

tw::model::TwinState ToModel(const TwinState &state) noexcept {
    tw::model::TwinState out;
    std::memcpy(&out, &state, sizeof(out));
    return out;
}

TwinState FromModel(const tw::model::TwinState &state) noexcept {
    TwinState out;
    std::memcpy(&out, &state, sizeof(out));
    return out;
}

tw::step::Policy ToStepPolicy(TwinPolicy policy) noexcept {
    switch (policy) {
    case TwinPolicy::FvSubsteps: return tw::step::Policy::FvSubsteps;
    case TwinPolicy::CappedSubsteps3: return tw::step::Policy::CappedSubsteps3;
    case TwinPolicy::ContactSubsteps: return tw::step::Policy::ContactSubsteps;
    case TwinPolicy::ContactSubstepsFull: return tw::step::Policy::ContactSubstepsFull;
    }
    return tw::step::Policy::FvSubsteps;
}

bool IsTwinScope(MapEnvironment environment, VehicleModel vehicle, std::optional<PlayMode> playMode) noexcept {
    return environment == MapEnvironment::Stadium && vehicle == VehicleModel::StadiumCar &&
           playMode != PlayMode::Stunts;
}

}  // namespace internal

// ---- TwinWorld ------------------------------------------------------------------
TwinWorld::TwinWorld(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

std::shared_ptr<const TwinWorld> TwinWorldFactory::Make(std::unique_ptr<TwinWorld::Impl> impl) {
    if (!impl) return nullptr;
    // Runtime ISA dispatch: a wider variant builds its own tick instance
    // over the same packs; on any failure the world keeps the baseline.
    impl->info.isa = "baseline";
    if (impl->packs) {
        const tw::isa::Kind kind = tw::isa::Select();
        const tw::isa::TickTable *table = tw::isa::Table(kind);
        if (kind != tw::isa::Kind::Baseline && table != nullptr) {
            tw::isa::TickInputs in;
            in.view = &impl->packs->view;
            in.shapes = &impl->packs->shapes;
            in.collisionStage = impl->packs->collisionName.c_str();
            in.car = impl->model.car;
            in.water = impl->model.water;
            in.race = impl->model.race;
            in.policy = impl->model.policy;
            const char *why = nullptr;
            void *instance = impl->model.policyConfig == nullptr ? table->create(in, &why) : nullptr;
            if (instance != nullptr) {
                impl->isa = table;
                impl->isaInstance = std::shared_ptr<void>(instance, table->destroy);
                impl->info.isa = table->name;
            }
        }
    }
    return std::shared_ptr<const TwinWorld>(new TwinWorld(std::move(impl)));
}

namespace internal {

tw::step::Status Step(const TwinWorld::Impl &world, tw::model::TwinState &state, tw::step::Control control,
                      tw::step::Events *events, tw::step::CollisionCache *cache) {
    if (world.isa != nullptr) return world.isa->step(world.isaInstance.get(), state, control, events, cache);
    return tw::step::Step(world.model, state, control, events, cache);
}

std::uint32_t StepTicks(const TwinWorld::Impl &world, tw::model::TwinState &state, const tw::step::Control *controls,
                        std::size_t count, tw::step::StopRule stop, tw::step::Events *events,
                        const std::uint32_t *respawnCounts, tw::step::CollisionCache *cache) {
    if (world.isa != nullptr) {
        return world.isa->stepTicks(world.isaInstance.get(), state, controls, count, stop, events, respawnCounts,
                                    cache);
    }
    return tw::step::StepTicks(world.model, state, controls, count, stop, events, respawnCounts, cache);
}

namespace {
void DeleteCache(tw::step::CollisionCache *cache) { delete cache; }
void NoCache(tw::step::CollisionCache *) {}
}  // namespace

CollisionCachePtr NewCollisionCache(const TwinWorld::Impl &world) {
    if (world.isa != nullptr) {
        return CollisionCachePtr(world.isa->newCache(world.isaInstance.get()), world.isa->destroyCache);
    }
    if (world.model.collision != nullptr) {
        return CollisionCachePtr(world.model.collision->NewCache().release(), DeleteCache);
    }
    return CollisionCachePtr(nullptr, NoCache);
}

}  // namespace internal
TwinWorld::~TwinWorld() = default;
const TwinWorldInfo &TwinWorld::Info() const noexcept { return impl_->info; }

const char *TwinStatusName(TwinStatus status) noexcept {
    switch (status) {
    case TwinStatus::Ok: return "ok";
    case TwinStatus::NotImplemented: return "not_implemented";
    case TwinStatus::InvalidArgument: return "invalid_argument";
    case TwinStatus::Stopped: return "stopped";
    }
    return "unknown";
}

TwinEngineInfo QueryTwinEngine() noexcept {
    TwinEngineInfo info;
    try {
        const tw::step::EngineInfo engine = tw::step::Info();
        info.stepImplemented = engine.stepImplemented;
        info.modelId = engine.modelId;
        info.missing = engine.missing;
    } catch (...) {
    }
    return info;
}

PhysicsSandboxResult<std::shared_ptr<const TwinWorld>> CompileTwinWorld(
        const PhysicsSandbox &loaded, TwinPolicy policy, TwinCollision collision) noexcept {
    using R = PhysicsSandboxResult<std::shared_ptr<const TwinWorld>>;
    try {
        auto name = loaded.ReadMapName();
        if (!name) return R::Failure(std::move(name).Error());
        auto view = loaded.ReadState();
        if (!view) return R::Failure(std::move(view).Error());
        const PhysicsSandboxStateView &v = view.Value();
        if (!internal::IsTwinScope(v.mapEnvironment, v.vehicleModel, v.playMode)) {
            const bool stunts = v.playMode == PlayMode::Stunts;
            PhysicsSandboxError error = Error(
                    PhysicsSandboxErrorCode::InvalidRequest,
                    stunts ? "twin: Stunts mode is not supported"
                           : "twin: only Stadium maps with the StadiumCar are supported",
                    ValidationFailureReason::TwinUnsupportedSimulationScope);
            error.validationError.relatedAsset = std::string(MapEnvironmentName(v.mapEnvironment)) + "/" +
                                                 VehicleModelName(v.vehicleModel) + (stunts ? "/Stunts" : "");
            return R::Failure(std::move(error));
        }
        // World package (M2), car pack + water (M4), race map and import
        // context, compiled from the sandbox's FV session (fv/twin_world_build.cpp).
        tw::fv::TwinWorldLabel label;
        label.mapEnvironment = v.mapEnvironment;
        label.vehicleModel = v.vehicleModel;
        label.mapName = name.Value();
        std::string diag;
        std::shared_ptr<const TwinWorld> world = tw::fv::BuildTwinWorldFromSandbox(loaded, policy, collision, label, &diag);
        if (!world) {
            return R::Failure(Error(PhysicsSandboxErrorCode::SimulationFailed, diag,
                                    ValidationFailureReason::TwinExecutionFailed));
        }
        return R::Success(std::move(world));
    } catch (const std::bad_alloc &) {
        return R::Failure(Error(PhysicsSandboxErrorCode::AllocationFailed, "twin: allocation failed"));
    } catch (...) {
        return R::Failure(Error(PhysicsSandboxErrorCode::UnexpectedFailure, "twin: unexpected world compile failure"));
    }
}

PhysicsSandboxResult<TwinState> TwinRaceStartState(const PhysicsSandbox &, std::uint32_t) noexcept {
    return NotImplemented<TwinState>("TwinRaceStartState (M6b)");
}

PhysicsSandboxResult<TwinState> TwinImportState(const TwinWorld &world, const PhysicsSandboxState &state) noexcept {
    using R = PhysicsSandboxResult<TwinState>;
    try {
        tw::model::TwinState s;
        std::string diag;
        if (!tw::fv::ImportSandboxState(world, state, &s, &diag)) {
            return R::Failure(Error(PhysicsSandboxErrorCode::InvalidRequest, diag,
                                    ValidationFailureReason::TwinExecutionFailed));
        }
        return R::Success(internal::FromModel(s));
    } catch (const std::bad_alloc &) {
        return R::Failure(Error(PhysicsSandboxErrorCode::AllocationFailed, "twin: allocation failed"));
    } catch (...) {
        return R::Failure(Error(PhysicsSandboxErrorCode::UnexpectedFailure, "twin: unexpected import failure"));
    }
}

PhysicsSandboxResult<TwinState> TwinImportState(const PhysicsSandbox &loaded, const PhysicsSandboxState &state) noexcept {
    auto world = CompileTwinWorld(loaded);
    if (!world) return PhysicsSandboxResult<TwinState>::Failure(std::move(world).Error());
    return TwinImportState(*world.Value(), state);
}

PhysicsSandboxResult<TwinControlStream> TwinControlsFromEvents(
        const TwinWorld &world, const std::vector<PhysicsSandboxInputEvent> &events,
        std::uint32_t firstTickMs, std::uint32_t tickCount) noexcept {
    using R = PhysicsSandboxResult<TwinControlStream>;
    try {
        std::vector<tw::race::TwInputEvent> in;
        in.reserve(events.size());
        for (const PhysicsSandboxInputEvent &e : events) {
            tw::race::TwInputEvent t;
            t.timeMs = e.timeMs;
            t.action = static_cast<tw::race::TwAction>(e.action);
            t.kind = static_cast<tw::race::TwValueKind>(e.value.kind);
            t.state = static_cast<tw::race::TwSwitch>(e.value.switchState);
            t.analog = static_cast<std::int32_t>(e.value.analog);
            in.push_back(t);
        }
        const tw::race::TwTimeline timeline = world.Internal().timeline;
        tw::race::TwControlStream stream;
        std::string error;
        if (!tw::race::TwBuildControls(in.data(), in.size(), timeline, firstTickMs, tickCount, &stream, &error)) {
            return R::Failure(Error(PhysicsSandboxErrorCode::InvalidRequest, "twin controls: " + error));
        }
        TwinControlStream out;
        out.firstTickMs = stream.firstTickMs;
        out.tickMs = timeline.tickMs;
        out.ticks = std::move(stream.ticks);
        out.respawnOverflow = std::move(stream.respawnOverflow);
        return R::Success(std::move(out));
    } catch (const std::bad_alloc &) {
        return R::Failure(Error(PhysicsSandboxErrorCode::AllocationFailed, "twin: allocation failed"));
    } catch (...) {
        return R::Failure(Error(PhysicsSandboxErrorCode::UnexpectedFailure, "twin: unexpected controls failure"));
    }
}

namespace {

TwinStatus StepWithCache(const TwinWorld &world, TwinState &state, TwinControl control, TwinEvents *events,
                         tw::step::CollisionCache *cache) noexcept {
    tw::model::TwinState s = internal::ToModel(state);
    tw::step::Events ev = ToStepEvents(events);
    const tw::step::Status status = internal::Step(world.Internal(), s, control, &ev, cache);
    if (status == tw::step::Status::Ok) state = internal::FromModel(s);
    FromStepEvents(ev, events);
    return FromStep(status);
}

}  // namespace

TwinStatus TwinStep(const TwinWorld &world, TwinState &state, TwinControl control, TwinEvents *events) noexcept {
    return StepWithCache(world, state, control, events, nullptr);
}

namespace {

std::uint32_t StepTicksImpl(const TwinWorld &world, TwinState &state, const TwinControl *controls,
                            std::size_t count, TwinStopRule stop, TwinEvents *events,
                            const std::uint32_t *respawnCounts) {
    tw::model::TwinState s = internal::ToModel(state);
    tw::step::Events ev = ToStepEvents(events);
    tw::step::StopRule rule;
    rule.atFinish = stop.atFinish;
    const internal::CollisionCachePtr cache = internal::NewCollisionCache(world.Internal());
    const std::uint32_t done =
            internal::StepTicks(world.Internal(), s, controls, count, rule, &ev, respawnCounts, cache.get());
    state = internal::FromModel(s);  // the state after the last executed tick
    FromStepEvents(ev, events);
    return done;
}

}  // namespace

std::uint32_t TwinStepTicks(const TwinWorld &world, TwinState &state, const TwinControl *controls,
                            std::size_t count, TwinStopRule stop, TwinEvents *events) noexcept {
    return StepTicksImpl(world, state, controls, count, stop, events, nullptr);
}

std::uint32_t TwinStepStream(const TwinWorld &world, TwinState &state, const TwinControlStream &stream,
                             std::size_t first, std::size_t count, TwinStopRule stop, TwinEvents *events) noexcept {
    if (first > stream.ticks.size() || count > stream.ticks.size() - first) {
        if (events != nullptr) events->status = TwinStatus::InvalidArgument;
        return 0u;
    }
    try {
        std::vector<std::uint32_t> respawns(count);
        bool escape = false;
        for (std::size_t i = 0; i < count; ++i) {
            const TwinControl c = stream.ticks[first + i];
            respawns[i] = tw::race::TwCtlRespawnField(c);
            if (c & tw::race::kCtlRespawnEscape) {
                escape = true;
                for (const auto &o : stream.respawnOverflow)
                    if (o.first == first + i) respawns[i] = o.second;
            }
        }
        return StepTicksImpl(world, state, stream.ticks.data() + first, count, stop, events,
                             escape ? respawns.data() : nullptr);
    } catch (...) {
        if (events != nullptr) events->status = TwinStatus::InvalidArgument;
        return 0u;
    }
}

std::uint64_t TwinStateHash(const TwinState &state) noexcept {
    return tw::step::CanonicalHash(internal::ToModel(state));
}

std::uint32_t TwinReadField(const TwinState &state, const char *goldenName, std::uint32_t *words,
                            std::uint32_t capacity) noexcept {
    if (goldenName == nullptr) return 0u;
    std::size_t n = 0;
    const tw::model::TwFieldInfo *fields = tw::model::TwinFields(&n);
    for (std::size_t i = 0; i < n; ++i) {
        const tw::model::TwFieldInfo &f = fields[i];
        if (std::strcmp(f.golden, goldenName) != 0 && std::strcmp(f.name, goldenName) != 0) continue;
        for (std::uint32_t k = 0; k < f.count && k < capacity && words != nullptr; ++k) {
            const std::uint8_t *p = state.bytes + f.offset + f.elementBytes * k;
            std::uint32_t w = 0u;
            switch (f.type) {
            case tw::model::TwType::F32:
            case tw::model::TwType::U32: std::memcpy(&w, p, 4); break;
            case tw::model::TwType::U16: {
                std::uint16_t v;
                std::memcpy(&v, p, 2);
                w = v;
                break;
            }
            case tw::model::TwType::U8: w = *p; break;
            case tw::model::TwType::I8: w = static_cast<std::uint32_t>(static_cast<std::int32_t>(
                                                static_cast<std::int8_t>(*p)));
                break;
            }
            words[k] = w;
        }
        return f.count;
    }
    return 0u;
}

// ---- batch -------------------------------------------------------------------------
namespace {

// L = 1 reference runner: each candidate runs through the single-run path,
// so batch == single holds by construction. M10 replaces it with the lane
// scheduler behind the same interface.
class SerialBatchRunner final : public TwinBatchRunner {
public:
    SerialBatchRunner(std::shared_ptr<const TwinWorld> world, TwinBatchOptions options)
        : world_(std::move(world)), options_(std::move(options)) {}

    std::uint32_t Lanes() const noexcept override { return 1u; }

    TwinStatus Run(const std::vector<TwinCandidate> &candidates,
                   std::vector<TwinCandidateResult> *results) noexcept override {
        if (results == nullptr) return TwinStatus::InvalidArgument;
        try {
            results->clear();
            results->reserve(candidates.size());
            TwinStatus worst = TwinStatus::Ok;
            for (const TwinCandidate &c : candidates) {
                TwinCandidateResult r;
                r.id = c.id;
                r.end = c.start;
                // One collision cache per candidate (results never depend on it).
                const internal::CollisionCachePtr cache = internal::NewCollisionCache(world_->Internal());
                for (std::size_t i = 0; i < c.controls.size(); ++i) {
                    const TwinStatus st = StepWithCache(*world_, r.end, c.controls[i], &r.events, cache.get());
                    if (st != TwinStatus::Ok) {
                        r.status = st;
                        break;
                    }
                    ++r.ticks;
                    if (options_.evaluator) {
                        TwinLaneView lane;
                        lane.candidateId = c.id;
                        lane.tick = r.ticks;
                        lane.state = &r.end;
                        lane.events = &r.events;
                        if (!options_.evaluator(lane)) {
                            r.status = TwinStatus::Stopped;
                            break;
                        }
                    }
                    if (options_.stop.atFinish && r.events.finished) {
                        if (i + 1u < c.controls.size()) r.status = TwinStatus::Stopped;
                        break;
                    }
                }
                r.events.status = r.status;
                if (r.status != TwinStatus::Ok && r.status != TwinStatus::Stopped) worst = r.status;
                results->push_back(std::move(r));
            }
            return worst;
        } catch (...) {
            return TwinStatus::InvalidArgument;
        }
    }

private:
    std::shared_ptr<const TwinWorld> world_;
    TwinBatchOptions options_;
};

}  // namespace

PhysicsSandboxResult<std::unique_ptr<TwinBatchRunner>> CreateTwinBatchRunner(
        std::shared_ptr<const TwinWorld> world, TwinBatchOptions options) noexcept {
    using R = PhysicsSandboxResult<std::unique_ptr<TwinBatchRunner>>;
    try {
        if (!world) return R::Failure(Error(PhysicsSandboxErrorCode::InvalidRequest, "twin batch: no world"));
        if (options.lanes != 1u) {
            return R::Failure(Error(PhysicsSandboxErrorCode::InvalidRequest,
                                    "twin batch: only lanes = 1 is implemented (lane scheduler is M10)",
                                    ValidationFailureReason::TwinNotImplemented));
        }
        return R::Success(std::make_unique<SerialBatchRunner>(std::move(world), std::move(options)));
    } catch (const std::bad_alloc &) {
        return R::Failure(Error(PhysicsSandboxErrorCode::AllocationFailed, "twin: allocation failed"));
    } catch (...) {
        return R::Failure(Error(PhysicsSandboxErrorCode::UnexpectedFailure, "twin: unexpected batch failure"));
    }
}

PhysicsSandboxResult<PhysicsSandboxCudaSearchSession> CreateTwinSearchSession(
        PhysicsSandbox &loaded, PhysicsSandboxCudaSearchConfiguration configuration,
        const PhysicsSandboxTwinSearchOptions &options) noexcept {
    configuration.twin = options;
    return CreatePhysicsSandboxCudaSearchSession(loaded, configuration);
}

}  // namespace forevervalidator::experimental::twin
