// Twin public API skeleton (include/forevervalidator/experimental/twin.h).
// No game assets: worlds come from TwinWorldFactory, sandboxes stay unloaded.
#include <forevervalidator/experimental/twin.h>

#include <cstring>
#include <iostream>
#include <vector>

#include "src/twin/api/twin_api_internal.h"

namespace {

int g_fail = 0;

void Check(bool ok, const char *what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_fail;
    }
}

namespace fx = forevervalidator::experimental;
namespace tx = forevervalidator::experimental::twin;

std::shared_ptr<const tx::TwinWorld> StadiumWorld() {
    auto impl = std::make_unique<tx::TwinWorld::Impl>();
    impl->info.mapEnvironment = forevervalidator::MapEnvironment::Stadium;
    impl->info.vehicleModel = forevervalidator::VehicleModel::StadiumCar;
    return tx::TwinWorldFactory::Make(std::move(impl));
}

fx::PhysicsSandboxInputEvent Switch(std::int32_t t, fx::PhysicsSandboxInputAction a, bool pressed) {
    fx::PhysicsSandboxInputEvent e;
    e.timeMs = t;
    e.action = a;
    e.value.kind = fx::PhysicsSandboxInputValueKind::Switch;
    e.value.switchState = pressed ? fx::PhysicsSandboxSwitchState::Pressed : fx::PhysicsSandboxSwitchState::Released;
    return e;
}

}  // namespace

int main() {
    const tx::TwinEngineInfo engine = tx::QueryTwinEngine();
    Check(!engine.modelId.empty(), "engine has a model id");
    std::cout << "twin engine: model " << engine.modelId << ", step "
              << (engine.stepImplemented ? "implemented" : "not implemented") << '\n';

    // Sandbox: the twin has no sandbox backend yet; unloaded sandboxes are refused.
    auto source = forevervalidator::CreateAssetSource(
            [](const forevervalidator::AssetRequest &) {
                return forevervalidator::Result<forevervalidator::AssetBytes>::Failure(
                        forevervalidator::ValidationError{});
            });
    Check(static_cast<bool>(source), "asset source");
    if (source) {
        fx::PhysicsSandboxOptions twinOptions;
        twinOptions.backend = forevervalidator::SimulationBackend::Twin;
        auto refused = fx::CreatePhysicsSandbox(std::move(source).Value(), twinOptions);
        Check(!refused && refused.Error().code == fx::PhysicsSandboxErrorCode::InvalidRequest,
              "PhysicsSandbox refuses the twin backend");
    }
    auto source2 = forevervalidator::CreateAssetSource(
            [](const forevervalidator::AssetRequest &) {
                return forevervalidator::Result<forevervalidator::AssetBytes>::Failure(
                        forevervalidator::ValidationError{});
            });
    if (source2) {
        fx::PhysicsSandboxOptions options;
        options.backend = forevervalidator::SimulationBackend::OptimizedCpu;
        auto sandbox = fx::CreatePhysicsSandbox(std::move(source2).Value(), options);
        Check(static_cast<bool>(sandbox), "optimized_cpu sandbox");
        if (sandbox) {
            Check(!tx::CompileTwinWorld(sandbox.Value()), "CompileTwinWorld refuses an unloaded sandbox");
            auto start = tx::TwinRaceStartState(sandbox.Value(), 1u);
            Check(!start && start.Error().validationError.reason ==
                                    forevervalidator::ValidationFailureReason::TwinNotImplemented,
                  "TwinRaceStartState reports not implemented");
        }
    }

    // Stepping: the stub engine reports NotImplemented and leaves the state alone.
    const auto world = StadiumWorld();
    Check(world != nullptr && world->Info().mapEnvironment == forevervalidator::MapEnvironment::Stadium,
          "factory world");
    tx::TwinState state{};
    state.bytes[0] = 0x11;
    const std::uint64_t before = tx::TwinStateHash(state);
    tx::TwinEvents events;
    const tx::TwinStatus status = tx::TwinStep(*world, state, 0u, &events);
    if (!engine.stepImplemented) {
        Check(status == tx::TwinStatus::NotImplemented, "TwinStep reports not implemented");
        Check(events.status == tx::TwinStatus::NotImplemented && events.ticks == 0u, "events say not implemented");
        Check(tx::TwinStateHash(state) == before, "state unchanged on NotImplemented");
        Check(std::strcmp(tx::TwinStatusName(status), "not_implemented") == 0, "status name");
        const std::vector<tx::TwinControl> controls(100u, 0u);
        tx::TwinEvents runEvents;
        Check(tx::TwinStepTicks(*world, state, controls.data(), controls.size(), {}, &runEvents) == 0u &&
                      runEvents.status == tx::TwinStatus::NotImplemented,
              "TwinStepTicks stops at once");
    }
    Check(tx::TwinStepTicks(*world, state, nullptr, 3u, {}, &events) == 0u &&
                  events.status == tx::TwinStatus::InvalidArgument,
          "TwinStepTicks rejects null controls");

    // Field access by golden name.
    tx::TwinState fields{};
    const float x = 12.5f;
    std::memcpy(fields.bytes, &x, 4);  // pos is the first field (state_fields.inc)
    std::uint32_t words[4] = {0u, 0u, 0u, 0u};
    Check(tx::TwinReadField(fields, "body.pos", words, 4u) == 3u, "body.pos has 3 elements");
    float back;
    std::memcpy(&back, &words[0], 4);
    Check(back == 12.5f, "body.pos.x read back");
    Check(tx::TwinReadField(fields, "no.such.field", words, 4u) == 0u, "unknown field");
    tx::TwinState nan{};
    const std::uint32_t nanA = 0x7fc00001u, nanB = 0xffc00000u;
    std::memcpy(nan.bytes, &nanA, 4);
    tx::TwinState nan2 = nan;
    std::memcpy(nan2.bytes, &nanB, 4);
    Check(tx::TwinStateHash(nan) == tx::TwinStateHash(nan2), "hash canonicalises NaN payloads");

    // Controls: FV's input state machine through the race layer.
    std::vector<fx::PhysicsSandboxInputEvent> script = {
            Switch(0, fx::PhysicsSandboxInputAction::RaceRunning, true),
            Switch(0, fx::PhysicsSandboxInputAction::Accelerate, true),
            Switch(500, fx::PhysicsSandboxInputAction::SteerLeft, true)};
    auto stream = tx::TwinControlsFromEvents(*world, script, 2600u, 100u);
    Check(static_cast<bool>(stream), "controls build");
    if (stream) {
        const tx::TwinControlStream &s = stream.Value();
        Check(s.ticks.size() == 100u && s.firstTickMs == 2600u && s.tickMs == 10u, "stream shape");
        Check((s.ticks[0] & (1u << 29)) != 0u, "race-start reset at T = 2600");
        Check((s.ticks[10] & (1u << 18)) != 0u, "gas held");
        Check((s.ticks[60] & 0x3FFFFu) != 0u, "steer left from T = 3100");
    }

    // Batch: L = 1 reference runner; results in candidate order.
    tx::TwinBatchOptions batchOptions;
    auto runner = tx::CreateTwinBatchRunner(world, batchOptions);
    Check(static_cast<bool>(runner) && runner.Value()->Lanes() == 1u, "batch runner L = 1");
    if (runner) {
        std::vector<tx::TwinCandidate> candidates(3);
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            candidates[i].id = 100u + i;
            candidates[i].controls.assign(10u, 0u);
        }
        std::vector<tx::TwinCandidateResult> results;
        const tx::TwinStatus batch = runner.Value()->Run(candidates, &results);
        Check(results.size() == 3u && results[2].id == 102u, "batch results in order");
        if (!engine.stepImplemented) {
            Check(batch == tx::TwinStatus::NotImplemented && results[0].status == tx::TwinStatus::NotImplemented,
                  "batch reports not implemented");
        }
    }
    tx::TwinBatchOptions wide;
    wide.lanes = 16u;
    auto wideRunner = tx::CreateTwinBatchRunner(world, wide);
    Check(!wideRunner && wideRunner.Error().validationError.reason ==
                                 forevervalidator::ValidationFailureReason::TwinNotImplemented,
          "lanes > 1 not implemented yet (M10)");

    std::cout << (g_fail == 0 ? "PASS" : "FAIL") << " (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
