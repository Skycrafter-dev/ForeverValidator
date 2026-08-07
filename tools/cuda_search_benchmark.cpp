#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/native.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

using forevervalidator::experimental::PhysicsSandboxError;
using forevervalidator::experimental::PhysicsSandboxCudaSearchBatch;
using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

int Fail(const std::string &message) {
    std::cerr << "cuda_search_benchmark: " << message << '\n';
    return 1;
}

std::string Diagnostic(const PhysicsSandboxError &error) {
    return error.diagnostic.empty() ? "unknown error" : error.diagnostic;
}

bool SameInput(
        const forevervalidator::experimental::PhysicsSandboxInputEvent &left,
        const forevervalidator::experimental::PhysicsSandboxInputEvent &right) {
    return left.timeMs == right.timeMs &&
            left.action == right.action &&
            left.value.kind == right.value.kind &&
            left.value.switchState == right.value.switchState &&
            left.value.analog == right.value.analog;
}

bool SameBatch(const PhysicsSandboxCudaSearchBatch &left,
               const PhysicsSandboxCudaSearchBatch &right) {
    if (left.firstCandidateId != right.firstCandidateId ||
        left.candidateCount != right.candidateCount ||
        left.evaluatedCandidateCount != right.evaluatedCandidateCount ||
        left.evaluatorCalls != right.evaluatorCalls ||
        left.qualifyingCandidateCount != right.qualifyingCandidateCount ||
        left.closestTargetDistance != right.closestTargetDistance ||
        left.totalMutationCount != right.totalMutationCount ||
        left.mutationImprovementCount != right.mutationImprovementCount ||
        left.cancelled != right.cancelled ||
        left.bestChanged != right.bestChanged ||
        left.bestValid != right.bestValid ||
        left.bestIsMutation != right.bestIsMutation ||
        left.bestCandidateId != right.bestCandidateId ||
        left.bestMutationCount != right.bestMutationCount ||
        left.bestEvaluationTick != right.bestEvaluationTick ||
        left.bestScore != right.bestScore ||
        left.bestTimeMs != right.bestTimeMs ||
        left.bestDetail0 != right.bestDetail0 ||
        left.bestDetail1 != right.bestDetail1 ||
        left.bestInputs.size() != right.bestInputs.size() ||
        left.bestSnapshot.has_value() != right.bestSnapshot.has_value() ||
        std::memcmp(
                &left.bestState.car,
                &right.bestState.car,
                sizeof(left.bestState.car)) != 0 ||
        left.bestState.tick != right.bestState.tick ||
        left.bestState.timeMs != right.bestState.timeMs ||
        left.bestState.durationMs != right.bestState.durationMs ||
        left.bestState.mapEnvironment != right.bestState.mapEnvironment ||
        left.bestState.vehicleModel != right.bestState.vehicleModel ||
        left.bestState.playMode != right.bestState.playMode ||
        left.bestState.accelerate != right.bestState.accelerate ||
        left.bestState.brake != right.bestState.brake ||
        left.bestState.steering != right.bestState.steering ||
        left.bestState.checkpointsCollected !=
                right.bestState.checkpointsCollected ||
        left.bestState.checkpointsTotal !=
                right.bestState.checkpointsTotal ||
        left.bestState.completedLaps != right.bestState.completedLaps ||
        left.bestState.totalLaps != right.bestState.totalLaps ||
        left.bestState.raceCompleted != right.bestState.raceCompleted ||
        left.bestState.finishTimeMs != right.bestState.finishTimeMs ||
        left.bestState.finishTime != right.bestState.finishTime ||
        left.bestState.respawnCount != right.bestState.respawnCount ||
        left.bestState.stuntsScore != right.bestState.stuntsScore) {
        return false;
    }
    for (std::size_t index = 0u; index < left.bestInputs.size(); ++index) {
        if (!SameInput(left.bestInputs[index], right.bestInputs[index])) {
            return false;
        }
    }
    return true;
}

constexpr std::uint64_t FnvOffset = 1469598103934665603ull;
constexpr std::uint64_t FnvPrime = 1099511628211ull;

template<typename T>
void HashValue(std::uint64_t &hash, const T &value) {
    const auto *bytes =
            reinterpret_cast<const unsigned char *>(&value);
    for (std::size_t index = 0u; index < sizeof(value); ++index) {
        hash ^= bytes[index];
        hash *= FnvPrime;
    }
}

template<typename T>
void HashOptional(std::uint64_t &hash, const std::optional<T> &value) {
    const bool present = value.has_value();
    HashValue(hash, present);
    if (present) {
        HashValue(hash, *value);
    }
}

std::uint64_t StateFingerprint(
        const forevervalidator::experimental::PhysicsSandboxStateView &view) {
    std::uint64_t hash = FnvOffset;
    HashValue(hash, view.tick);
    HashValue(hash, view.timeMs);
    HashValue(hash, view.mapEnvironment);
    HashValue(hash, view.vehicleModel);
    HashOptional(hash, view.playMode);
    HashValue(hash, view.car.rotationX);
    HashValue(hash, view.car.rotationY);
    HashValue(hash, view.car.rotationZ);
    HashValue(hash, view.car.rotationW);
    HashValue(hash, view.car.position.x);
    HashValue(hash, view.car.position.y);
    HashValue(hash, view.car.position.z);
    HashValue(hash, view.car.linearSpeed.x);
    HashValue(hash, view.car.linearSpeed.y);
    HashValue(hash, view.car.linearSpeed.z);
    HashValue(hash, view.car.angularSpeed.x);
    HashValue(hash, view.car.angularSpeed.y);
    HashValue(hash, view.car.angularSpeed.z);
    HashValue(hash, view.car.force.x);
    HashValue(hash, view.car.force.y);
    HashValue(hash, view.car.force.z);
    HashValue(hash, view.car.torque.x);
    HashValue(hash, view.car.torque.y);
    HashValue(hash, view.car.torque.z);
    HashValue(hash, view.accelerate);
    HashValue(hash, view.brake);
    HashValue(hash, view.steering);
    HashValue(hash, view.checkpointsCollected);
    HashValue(hash, view.checkpointsTotal);
    HashValue(hash, view.completedLaps);
    HashValue(hash, view.totalLaps);
    HashValue(hash, view.raceCompleted);
    HashOptional(hash, view.finishTimeMs);
    HashValue(hash, view.respawnCount);
    HashOptional(hash, view.stuntsScore);
    return hash;
}

std::uint64_t InputFingerprint(
        const std::vector<
                forevervalidator::experimental::PhysicsSandboxInputEvent>
                &inputs) {
    std::uint64_t hash = FnvOffset;
    for (const auto &input : inputs) {
        HashValue(hash, input.timeMs);
        HashValue(hash, input.action);
        HashValue(hash, input.value.kind);
        HashValue(hash, input.value.switchState);
        HashValue(hash, input.value.analog);
    }
    return hash;
}

bool IsPipeline(const std::string &value) {
    return value == "optimized" || value == "legacy" ||
           value == "differential" ||
           value == "prefix-differential";
}

bool IsEvaluator(const std::string &value) {
    return value == "velocity" || value == "point" || value == "pose" ||
           value == "volume-entry" || value == "finish-time";
}

bool IsModifier(const std::string &value) {
    return value == "random-steering" ||
           value == "existing-event" ||
           value == "existing-event-static" ||
           value == "smooth-steering" ||
           value == "input-insertion" ||
           value == "dense-insertion" ||
           value == "input-deletion" ||
           value == "mixed" ||
           value == "cancelled";
}

std::vector<
        forevervalidator::experimental::PhysicsSandboxInputEvent>
BuildSyntheticInputs(
        std::vector<
                forevervalidator::experimental::PhysicsSandboxInputEvent>
                inputs,
        std::int64_t firstTimeMs,
        std::int64_t lastTimeMs,
        std::uint32_t eventsPerSecond) {
    using namespace forevervalidator::experimental;
    if (eventsPerSecond == 0u) {
        return inputs;
    }
    const std::int64_t intervalMs = std::max<std::int64_t>(
            1, 1000 / static_cast<std::int64_t>(eventsPerSecond));
    std::uint32_t generatedIndex = 0u;
    for (std::int64_t timeMs = firstTimeMs;
         timeMs <= lastTimeMs;
         timeMs += intervalMs, ++generatedIndex) {
        PhysicsSandboxInputEvent event;
        event.timeMs = static_cast<std::int32_t>(timeMs);
        switch (generatedIndex % 3u) {
        case 0u:
            event.action = PhysicsSandboxInputAction::Steer;
            event.value.kind = PhysicsSandboxInputValueKind::Analog;
            event.value.analog =
                    static_cast<forevervalidator::AnalogInputState>(
                            -24000 +
                            static_cast<std::int32_t>(
                                    generatedIndex % 17u) *
                                    3000);
            break;
        case 1u:
            event.action = PhysicsSandboxInputAction::Accelerate;
            event.value.kind = PhysicsSandboxInputValueKind::Switch;
            event.value.switchState =
                    (generatedIndex / 3u) % 2u == 0u
                    ? PhysicsSandboxSwitchState::Pressed
                    : PhysicsSandboxSwitchState::Released;
            break;
        default:
            event.action = PhysicsSandboxInputAction::Brake;
            event.value.kind = PhysicsSandboxInputValueKind::Switch;
            event.value.switchState =
                    (generatedIndex / 3u) % 2u == 0u
                    ? PhysicsSandboxSwitchState::Released
                    : PhysicsSandboxSwitchState::Pressed;
            break;
        }
        inputs.push_back(event);
    }
    std::stable_sort(
            inputs.begin(), inputs.end(),
            [](const PhysicsSandboxInputEvent &left,
               const PhysicsSandboxInputEvent &right) {
                return left.timeMs < right.timeMs;
            });
    std::vector<PhysicsSandboxInputEvent> normalized;
    normalized.reserve(inputs.size());
    std::size_t groupBegin = 0u;
    for (const PhysicsSandboxInputEvent &event : inputs) {
        if (normalized.empty() ||
            normalized.back().timeMs != event.timeMs) {
            groupBegin = normalized.size();
        }
        const auto duplicate = std::find_if(
                normalized.begin() +
                        static_cast<std::ptrdiff_t>(groupBegin),
                normalized.end(),
                [&](const PhysicsSandboxInputEvent &candidate) {
                    return candidate.action == event.action;
                });
        if (duplicate == normalized.end()) {
            normalized.push_back(event);
        } else {
            *duplicate = event;
        }
    }
    return normalized;
}

}  // namespace

int main(int argc, char **argv) {
    using namespace forevervalidator;
    using namespace forevervalidator::experimental;

    if (argc < 6) {
        return Fail(
                "usage: PACKS REPLAY CANDIDATES TIMELINE_TICKS "
                "REPETITIONS [BRANCH_TIME_MS] "
                "[random-steering|existing-event|existing-event-static|"
                "smooth-steering|"
                "input-insertion|dense-insertion|input-deletion|mixed|"
                "cancelled] "
                "[optimized|legacy|differential|"
                "prefix-differential|"
                "velocity|point|pose|volume-entry|finish-time] "
                "[velocity|point|pose|volume-entry|finish-time] "
                "[--input-rate EVENTS_PER_SECOND] "
                "[--boundary-offset-ticks TICKS] "
                "[--existing-min COUNT] [--existing-max COUNT] "
                "[--no-winner-state] [--no-locality-sort] "
                "[--no-prefix-reuse] [--no-dedup] "
                "[--minimum-blocks 8|12|16] "
                "[--batch-capacity COUNT]");
    }
    const std::uint32_t candidateCount =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
    const std::uint32_t timelineTicks =
            static_cast<std::uint32_t>(std::stoul(argv[4]));
    const std::uint32_t repetitions =
            static_cast<std::uint32_t>(std::stoul(argv[5]));
    const std::uint64_t branchTimeMs =
            argc >= 7 ? std::stoull(argv[6]) : 5000u;
    const std::string modifier =
            argc >= 8 ? argv[7] : "random-steering";
    std::string pipeline = "optimized";
    std::string evaluatorName = "velocity";
    if (argc >= 9) {
        const std::string mode = argv[8];
        if (IsPipeline(mode)) {
            pipeline = mode;
        } else if (IsEvaluator(mode)) {
            evaluatorName = mode;
        } else {
            return Fail("unknown mutation pipeline or evaluator");
        }
    }
    if (argc >= 10) {
        if (!IsPipeline(argv[8]) || !IsEvaluator(argv[9])) {
            return Fail("explicit pipeline/evaluator pair is invalid");
        }
        pipeline = argv[8];
        evaluatorName = argv[9];
    }
    std::uint32_t inputRate = 0u;
    std::uint32_t boundaryOffsetTicks = 0u;
    std::uint32_t existingMinimumCount = 1u;
    std::uint32_t existingMaximumCount = 16u;
    bool captureBestState = true;
    bool sortCandidatesByLocality = true;
    bool reuseBaselinePrefixes = true;
    bool deduplicateLowEntropyCandidateInputs = true;
    std::uint32_t simulationMinimumBlocks = 0u;
    std::uint32_t batchCapacity = candidateCount;
    for (int argument = 10; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--no-winner-state") {
            captureBestState = false;
            continue;
        }
        if (option == "--no-locality-sort") {
            sortCandidatesByLocality = false;
            continue;
        }
        if (option == "--no-prefix-reuse") {
            reuseBaselinePrefixes = false;
            continue;
        }
        if (option == "--no-dedup") {
            deduplicateLowEntropyCandidateInputs = false;
            continue;
        }
        if (argument + 1 >= argc) {
            return Fail("benchmark option is missing a value");
        }
        if (option == "--input-rate") {
            inputRate = static_cast<std::uint32_t>(
                    std::stoul(argv[++argument]));
        } else if (option == "--boundary-offset-ticks") {
            boundaryOffsetTicks = static_cast<std::uint32_t>(
                    std::stoul(argv[++argument]));
        } else if (option == "--existing-min") {
            existingMinimumCount = static_cast<std::uint32_t>(
                    std::stoul(argv[++argument]));
        } else if (option == "--existing-max") {
            existingMaximumCount = static_cast<std::uint32_t>(
                    std::stoul(argv[++argument]));
        } else if (option == "--minimum-blocks") {
            simulationMinimumBlocks =
                    static_cast<std::uint32_t>(
                            std::stoul(argv[++argument]));
        } else if (option == "--batch-capacity") {
            batchCapacity = static_cast<std::uint32_t>(
                    std::stoul(argv[++argument]));
        } else {
            return Fail("unknown benchmark option: " + option);
        }
    }
    if (!IsModifier(modifier)) {
        return Fail("unknown modifier");
    }
    if (candidateCount == 0u || timelineTicks == 0u ||
        repetitions == 0u ||
        batchCapacity < candidateCount ||
        (simulationMinimumBlocks != 0u &&
         simulationMinimumBlocks != 16u &&
         simulationMinimumBlocks != 12u &&
         simulationMinimumBlocks != 8u) ||
        existingMinimumCount > existingMaximumCount) {
        return Fail("benchmark dimensions must be positive");
    }

    auto replay = ReadNativeReplayFile(
            argv[2], ReplayIdentity{argv[2]});
    if (!replay) {
        return Fail("could not read replay");
    }
    auto source = OpenInstalledPackDirectory(argv[1]);
    if (!source) {
        return Fail("could not open pack source");
    }
    PhysicsSandboxOptions options;
    options.backend = SimulationBackend::Cuda;
    auto sandbox = CreatePhysicsSandbox(
            std::move(source).Value(), options);
    if (!sandbox) {
        return Fail("could not create CUDA sandbox: " +
                    Diagnostic(sandbox.Error()));
    }
    auto loaded = sandbox.Value().LoadReplay(
            {replay.Value().data(), replay.Value().size()},
            ReplayIdentity{argv[2]});
    if (!loaded) {
        return Fail("could not load replay: " +
                    Diagnostic(loaded.Error()));
    }
    const std::uint32_t tickDurationMs = options.tickDurationMs;
    if (branchTimeMs < loaded.Value().timeMs ||
        (branchTimeMs - loaded.Value().timeMs) % tickDurationMs != 0u) {
        return Fail("branch time is not reachable on whole ticks");
    }
    const std::uint64_t advanceTicks =
            (branchTimeMs - loaded.Value().timeMs) / tickDurationMs;
    if (advanceTicks > std::numeric_limits<std::uint32_t>::max()) {
        return Fail("branch time is too large");
    }
    const std::int64_t firstTickTimeMs =
            static_cast<std::int64_t>(branchTimeMs + tickDurationMs);
    const std::int64_t evaluationEndTimeMs =
            firstTickTimeMs +
            static_cast<std::int64_t>(timelineTicks - 1u) *
                    tickDurationMs;
    if (boundaryOffsetTicks >= timelineTicks) {
        return Fail("mutation boundary is outside the evaluation window");
    }
    const std::int64_t modifierFromTimeMs =
            firstTickTimeMs +
            static_cast<std::int64_t>(boundaryOffsetTicks) *
                    tickDurationMs;
    std::size_t normalizedInputCount = 0u;
    if (inputRate != 0u) {
        auto currentInputs = sandbox.Value().ReadInputs();
        if (!currentInputs) {
            return Fail("could not read replay inputs: " +
                        Diagnostic(currentInputs.Error()));
        }
        auto replaced = sandbox.Value().ReplaceInputs(
                BuildSyntheticInputs(
                        std::move(currentInputs).Value(),
                        loaded.Value().timeMs,
                        evaluationEndTimeMs,
                        inputRate));
        if (!replaced) {
            return Fail("could not install synthetic dense inputs: " +
                        Diagnostic(replaced.Error()));
        }
        normalizedInputCount = replaced.Value();
    }

    PhysicsSandboxStateView branchState = loaded.Value();
    if (advanceTicks != 0u) {
        auto advanced = sandbox.Value().AdvanceTicks(
                static_cast<std::uint32_t>(advanceTicks));
        if (!advanced) {
            return Fail("could not advance to branch: " +
                        Diagnostic(advanced.Error()));
        }
        branchState = advanced.Value();
    }

    PhysicsSandboxCudaSearchConfiguration configuration;
    configuration.maximumBatchSize =
            pipeline == "differential" ? 1u : batchCapacity;
    configuration.earliestMutationTimeMs = firstTickTimeMs;
    configuration.evaluationStartTimeMs = firstTickTimeMs;
    configuration.evaluationEndTimeMs = evaluationEndTimeMs;
    const PhysicsSandboxCudaModifierWindow modifierWindow{
            modifierFromTimeMs,
            evaluationEndTimeMs,
            0x6d2b79f5u};
    if (modifier == "random-steering" || modifier == "cancelled" ||
        modifier == "mixed") {
        configuration.modifiers.push_back(
                PhysicsSandboxCudaRandomSteeringModifier{
                        modifierWindow});
    }
    if (modifier == "existing-event" ||
        modifier == "existing-event-static" ||
        modifier == "mixed") {
        PhysicsSandboxCudaExistingEventModifier existing;
        existing.window = modifierWindow;
        existing.minimumCount = existingMinimumCount;
        existing.maximumCount = existingMaximumCount;
        existing.maximumTimeShiftMs =
                modifier == "existing-event-static" ? 0 : 100;
        existing.steeringDeltaMinimum = -4096;
        existing.steeringDeltaMaximum = 4096;
        existing.toggleAccelerate = true;
        existing.toggleBrake = true;
        configuration.modifiers.push_back(existing);
    }
    if (modifier == "smooth-steering" || modifier == "mixed") {
        PhysicsSandboxCudaSmoothSteeringModifier smooth;
        smooth.window = modifierWindow;
        smooth.deformationCount = 8u;
        smooth.radiusMs = 100;
        smooth.amplitudeMinimum = -8192;
        smooth.amplitudeMaximum = 8192;
        configuration.modifiers.push_back(smooth);
    }
    if (modifier == "input-insertion" ||
        modifier == "dense-insertion" ||
        modifier == "mixed") {
        PhysicsSandboxCudaInputInsertionModifier insertion;
        insertion.window = modifierWindow;
        insertion.steering.enabled = true;
        const bool dense =
                modifier == "dense-insertion" || modifier == "mixed";
        insertion.steering.minimumCount = dense ? 16u : 1u;
        insertion.steering.maximumCount = dense ? 16u : 1u;
        insertion.steering.maximumHoldMs = dense ? 100 : 0;
        insertion.accelerate.enabled = dense;
        insertion.accelerate.minimumCount = dense ? 16u : 0u;
        insertion.accelerate.maximumCount = dense ? 16u : 0u;
        insertion.accelerate.maximumHoldMs = dense ? 100 : 0;
        insertion.brake = insertion.accelerate;
        insertion.steeringOffset = true;
        insertion.steeringOffsetMinimum = dense ? -4096 : 1;
        insertion.steeringOffsetMaximum = dense ? 4096 : 1;
        configuration.modifiers.push_back(insertion);
    }
    if (modifier == "input-deletion" || modifier == "mixed") {
        PhysicsSandboxCudaInputDeletionModifier deletion;
        deletion.window = modifierWindow;
        deletion.steering.enabled = true;
        deletion.steering.maximumCount = 16u;
        deletion.accelerate.enabled = true;
        deletion.accelerate.maximumCount = 16u;
        deletion.brake.enabled = true;
        deletion.brake.maximumCount = 16u;
        configuration.modifiers.push_back(deletion);
    }
    if (evaluatorName == "velocity") {
        configuration.evaluator = PhysicsSandboxCudaVelocityEvaluator{};
    } else if (evaluatorName == "point") {
        configuration.evaluator = PhysicsSandboxCudaPointEvaluator{
                {branchState.car.position.x,
                 branchState.car.position.y,
                 branchState.car.position.z}};
    } else if (evaluatorName == "pose") {
        PhysicsSandboxCudaPoseEvaluator evaluator;
        evaluator.targetPosition = {
                branchState.car.position.x,
                branchState.car.position.y,
                branchState.car.position.z};
        evaluator.targetRotationX = branchState.car.rotationX;
        evaluator.targetRotationY = branchState.car.rotationY;
        evaluator.targetRotationZ = branchState.car.rotationZ;
        evaluator.targetRotationW = branchState.car.rotationW;
        configuration.evaluator = evaluator;
    } else if (evaluatorName == "volume-entry") {
        constexpr double Radius = 0.01;
        configuration.evaluator =
                PhysicsSandboxCudaVolumeEntryEvaluator{
                        {branchState.car.position.x - Radius,
                         branchState.car.position.y - Radius,
                         branchState.car.position.z - Radius},
                        {branchState.car.position.x + Radius,
                         branchState.car.position.y + Radius,
                         branchState.car.position.z + Radius}};
    } else {
        configuration.evaluator = PhysicsSandboxCudaFinishTimeEvaluator{};
    }
    configuration.useLegacyMutationPipelineForTesting =
            pipeline == "legacy";
    configuration.sortCandidatesByLocality =
            sortCandidatesByLocality;
    configuration.reuseBaselinePrefixes = reuseBaselinePrefixes;
    configuration.deduplicateLowEntropyCandidateInputs =
            deduplicateLowEntropyCandidateInputs;
    configuration.simulationMinimumBlocksPerMultiprocessorForTesting =
            simulationMinimumBlocks;
    configuration.captureBestState = captureBestState;

    auto session = CreatePhysicsSandboxCudaSearchSession(
            sandbox.Value(), configuration);
    if (!session) {
        return Fail("could not create CUDA search session: " +
                    Diagnostic(session.Error()));
    }
    std::uint32_t reservedBatchCapacity = configuration.maximumBatchSize;
    std::optional<PhysicsSandboxCudaSearchSession> legacySession;
    std::optional<PhysicsSandboxCudaSearchSession> fullSimulationSession;
    if (pipeline == "differential") {
        configuration.useLegacyMutationPipelineForTesting = true;
        auto created = CreatePhysicsSandboxCudaSearchSession(
                sandbox.Value(), configuration);
        if (!created) {
            return Fail("could not create legacy CUDA search session: " +
                        Diagnostic(created.Error()));
        }
        legacySession.emplace(std::move(created).Value());
        const std::uint32_t intermediateCapacity =
                std::max(1u, candidateCount / 2u);
        auto optimizedIntermediate =
                session.Value().ReserveBatchCapacity(intermediateCapacity);
        auto legacyIntermediate =
                legacySession->ReserveBatchCapacity(intermediateCapacity);
        if (!optimizedIntermediate || !legacyIntermediate ||
            optimizedIntermediate.Value() != intermediateCapacity ||
            legacyIntermediate.Value() != intermediateCapacity) {
            return Fail(
                    "optimized and legacy intermediate capacity growth "
                    "differs");
        }
        auto optimizedCapacity =
                session.Value().ReserveBatchCapacity(candidateCount);
        auto legacyCapacity =
                legacySession->ReserveBatchCapacity(candidateCount);
        if (!optimizedCapacity || !legacyCapacity ||
            optimizedCapacity.Value() != candidateCount ||
            legacyCapacity.Value() != candidateCount) {
            return Fail(
                    "optimized and legacy CUDA capacity growth differs");
        }
        reservedBatchCapacity = optimizedCapacity.Value();
    }
    if (pipeline == "prefix-differential") {
        configuration.reuseBaselinePrefixes = false;
        configuration.sortCandidatesByLocality = false;
        configuration.deduplicateLowEntropyCandidateInputs = false;
        auto created = CreatePhysicsSandboxCudaSearchSession(
                sandbox.Value(), configuration);
        if (!created) {
            return Fail("could not create full-simulation CUDA search session: " +
                        Diagnostic(created.Error()));
        }
        fullSimulationSession.emplace(std::move(created).Value());
    }
    auto baseline = session.Value().EvaluateBaseline();
    if (!baseline) {
        return Fail("could not evaluate baseline: " +
                    Diagnostic(baseline.Error()));
    }
    if (!baseline.Value().bestValid ||
        baseline.Value().bestSnapshot.has_value() != captureBestState) {
        return Fail("CUDA baseline winner-state capture policy was not honored");
    }
    if (evaluatorName == "finish-time" &&
        (baseline.Value().metrics.baselinePrefixDeviceBytes != 0u ||
         baseline.Value().metrics.candidatePrefixDeviceBytes != 0u ||
         baseline.Value().metrics.candidateDeduplicationDeviceBytes != 0u)) {
        return Fail("FinishTime unexpectedly allocated prefix/dedup storage");
    }
    if (fullSimulationSession.has_value()) {
        auto fullBaseline = fullSimulationSession->EvaluateBaseline();
        if (!fullBaseline ||
            !SameBatch(baseline.Value(), fullBaseline.Value()) ||
            fullBaseline.Value().metrics.baselinePrefixDeviceBytes != 0u ||
            fullBaseline.Value().metrics.candidatePrefixDeviceBytes != 0u ||
            fullBaseline.Value().metrics.
                    candidateDeduplicationDeviceBytes != 0u) {
            return Fail("prefix and full-simulation CUDA baselines differ");
        }
    }
    if (legacySession.has_value()) {
        auto legacyBaseline = legacySession->EvaluateBaseline();
        if (!legacyBaseline ||
            !SameBatch(baseline.Value(), legacyBaseline.Value())) {
            return Fail("optimized and legacy CUDA baselines differ");
        }
        auto cancelled = session.Value().RunBatch(
                0u, candidateCount, true);
        auto legacyCancelled = legacySession->RunBatch(
                0u, candidateCount, true);
        if (!cancelled || !legacyCancelled ||
            !cancelled.Value().cancelled ||
            !SameBatch(cancelled.Value(), legacyCancelled.Value())) {
            if (cancelled && legacyCancelled) {
                const auto &optimized = cancelled.Value();
                const auto &legacy = legacyCancelled.Value();
                std::cerr
                        << "optimized cancellation: evaluated="
                        << optimized.evaluatedCandidateCount
                        << " evaluator_calls=" << optimized.evaluatorCalls
                        << " mutations=" << optimized.totalMutationCount
                        << " improvements="
                        << optimized.mutationImprovementCount
                        << " best_changed=" << optimized.bestChanged
                        << " best_candidate="
                        << optimized.bestCandidateId.value_or(UINT64_MAX)
                        << '\n'
                        << "legacy cancellation: evaluated="
                        << legacy.evaluatedCandidateCount
                        << " evaluator_calls=" << legacy.evaluatorCalls
                        << " mutations=" << legacy.totalMutationCount
                        << " improvements="
                        << legacy.mutationImprovementCount
                        << " best_changed=" << legacy.bestChanged
                        << " best_candidate="
                        << legacy.bestCandidateId.value_or(UINT64_MAX)
                        << '\n';
            } else {
                if (!cancelled) {
                    std::cerr << "optimized cancellation error: "
                              << Diagnostic(cancelled.Error()) << '\n';
                }
                if (!legacyCancelled) {
                    std::cerr << "legacy cancellation error: "
                              << Diagnostic(legacyCancelled.Error()) << '\n';
                }
            }
            return Fail(
                    "optimized and legacy CUDA cancellation differs");
        }
        auto boundary = session.Value().RunBatch(UINT64_MAX, 1u, false);
        auto legacyBoundary =
                legacySession->RunBatch(UINT64_MAX, 1u, false);
        if (!boundary || !legacyBoundary ||
            !SameBatch(boundary.Value(), legacyBoundary.Value())) {
            return Fail(
                    "optimized and legacy CUDA candidate-id boundary differs");
        }
    }

    std::uint64_t firstCandidateId = 0u;
    for (std::uint32_t repetition = 0u;
         repetition < repetitions; ++repetition) {
        std::uint32_t cancellationProbeCount = 0u;
        const Clock::time_point wallBegin = Clock::now();
        auto batch = modifier == "cancelled"
                ? session.Value().RunBatch(
                          firstCandidateId, candidateCount,
                          [&cancellationProbeCount] {
                              return cancellationProbeCount++ >= 30u;
                          })
                : session.Value().RunBatch(
                          firstCandidateId, candidateCount, false);
        const Clock::time_point wallEnd = Clock::now();
        if (!batch) {
            return Fail("CUDA search batch failed: " +
                        Diagnostic(batch.Error()));
        }
        if (legacySession.has_value() && modifier != "cancelled") {
            auto legacyBatch = legacySession->RunBatch(
                    firstCandidateId, candidateCount, false);
            if (!legacyBatch ||
                !SameBatch(batch.Value(), legacyBatch.Value())) {
                return Fail(
                        "optimized and legacy CUDA mutation batches differ");
            }
        }
        if (fullSimulationSession.has_value() &&
            modifier != "cancelled") {
            auto fullBatch = fullSimulationSession->RunBatch(
                    firstCandidateId, candidateCount, false);
            if (!fullBatch ||
                !SameBatch(batch.Value(), fullBatch.Value()) ||
                fullBatch.Value().metrics.baselinePrefixDeviceBytes != 0u ||
                fullBatch.Value().metrics.candidatePrefixDeviceBytes != 0u ||
                fullBatch.Value().metrics.
                        candidateDeduplicationDeviceBytes != 0u) {
                return Fail(
                        "prefix-reused and full-simulation CUDA batches differ");
            }
        }
        if ((modifier == "cancelled" && !batch.Value().cancelled) ||
            (modifier != "cancelled" &&
             (batch.Value().cancelled ||
              batch.Value().evaluatedCandidateCount == 0u))) {
            return Fail("CUDA search batch was incomplete");
        }
        if (simulationMinimumBlocks != 0u &&
            batch.Value().metrics.
                            simulationSelectedMinimumBlocksPerMultiprocessor !=
                    simulationMinimumBlocks) {
            return Fail("forced CUDA launch-bounds variant was not selected");
        }
        const double simulatedTicks =
                static_cast<double>(
                        batch.Value().evaluatedCandidateCount) *
                timelineTicks;
        const double simulationKernelMilliseconds =
                batch.Value().metrics.simulationKernelMilliseconds;
        const double normalizedPhysicsNanoseconds =
                simulatedTicks == 0.0
                ? 0.0
                : simulationKernelMilliseconds * 1.0e6 /
                          simulatedTicks;
        const double ticksPerSecond = simulationKernelMilliseconds == 0.0
                ? 0.0
                : simulatedTicks * 1000.0 /
                          simulationKernelMilliseconds;
        const std::optional<std::uint64_t> winningEvaluationTick =
                batch.Value().bestValid
                ? std::optional<std::uint64_t>(
                          batch.Value().bestEvaluationTick)
                : std::nullopt;
        const double finishRefinementTickEquivalents =
                simulationKernelMilliseconds == 0.0
                ? 0.0
                : batch.Value().metrics
                                  .finishRefinementKernelMilliseconds *
                          timelineTicks / simulationKernelMilliseconds;
        std::cout << std::fixed << std::setprecision(6)
                  << "{"
                  << "\"repetition\":" << repetition << ","
                  << "\"candidates\":" << candidateCount << ","
                  << "\"evaluated_candidates\":"
                  << batch.Value().evaluatedCandidateCount << ","
                  << "\"qualifying_candidates\":"
                  << batch.Value().qualifyingCandidateCount << ","
                  << "\"closest_target_distance\":";
        if (batch.Value().closestTargetDistance) {
            std::cout << *batch.Value().closestTargetDistance;
        } else {
            std::cout << "null";
        }
        std::cout << ","
                  << "\"calibrated_batch_size\":"
                  << reservedBatchCapacity << ","
                  << "\"baseline_input_events\":"
                  << baseline.Value().bestInputs.size() << ","
                  << "\"best_input_events\":"
                  << batch.Value().bestInputs.size() << ","
                  << "\"total_mutation_count\":"
                  << batch.Value().totalMutationCount << ","
                  << "\"timeline_ticks\":" << timelineTicks << ","
                  << "\"branch_time_ms\":" << branchTimeMs << ","
                  << "\"mutable_from_time_ms\":"
                  << firstTickTimeMs << ","
                  << "\"modifier_from_time_ms\":"
                  << modifierFromTimeMs << ","
                  << "\"input_events_per_second\":" << inputRate << ","
                  << "\"synthetic_input_start_ms\":"
                  << loaded.Value().timeMs << ","
                  << "\"normalized_input_events\":"
                  << normalizedInputCount << ","
                  << "\"batch_capacity\":" << batchCapacity << ","
                  << "\"sort_candidates_by_locality\":"
                  << (sortCandidatesByLocality ? "true" : "false") << ","
                  << "\"reuse_baseline_prefixes\":"
                  << (reuseBaselinePrefixes ? "true" : "false") << ","
                  << "\"deduplicate_low_entropy_inputs\":"
                  << (deduplicateLowEntropyCandidateInputs
                              ? "true" : "false") << ","
                  << "\"forced_minimum_blocks_per_sm\":"
                  << simulationMinimumBlocks << ","
                  << "\"existing_minimum_count\":"
                  << existingMinimumCount << ","
                  << "\"existing_maximum_count\":"
                  << existingMaximumCount << ","
                  << "\"modifier\":\"" << modifier << "\","
                  << "\"mutation_pipeline\":\"" << pipeline
                  << "\","
                  << "\"evaluator\":\"" << evaluatorName
                  << "\","
                  << "\"cancelled\":"
                  << (batch.Value().cancelled ? "true" : "false") << ","
                  << "\"best_changed\":"
                  << (batch.Value().bestChanged ? "true" : "false") << ","
                  << "\"best_is_mutation\":"
                  << (batch.Value().bestIsMutation ? "true" : "false") << ","
                  << "\"best_candidate_id\":";
        if (batch.Value().bestCandidateId.has_value()) {
            std::cout << *batch.Value().bestCandidateId;
        } else {
            std::cout << "null";
        }
        std::cout << ",\"best_evaluation_tick\":";
        if (winningEvaluationTick.has_value()) {
            std::cout << *winningEvaluationTick;
        } else {
            std::cout << "null";
        }
        std::cout << ","
                  << "\"best_score\":" << batch.Value().bestScore << ","
                  << "\"best_time_ms\":" << batch.Value().bestTimeMs
                  << ","
                  << "\"best_detail_0\":" << batch.Value().bestDetail0
                  << ","
                  << "\"best_detail_1\":" << batch.Value().bestDetail1
                  << ","
                  << "\"best_detail0\":" << batch.Value().bestDetail0
                  << ","
                  << "\"best_detail1\":" << batch.Value().bestDetail1
                  << ","
                  << "\"best_mutation_count\":"
                  << batch.Value().bestMutationCount << ","
                  << "\"mutation_improvement_count\":"
                  << batch.Value().mutationImprovementCount << ","
                  << "\"best_state_fingerprint\":"
                  << StateFingerprint(batch.Value().bestState) << ","
                  << "\"best_input_count\":"
                  << batch.Value().bestInputs.size() << ","
                  << "\"best_input_fingerprint\":"
                  << InputFingerprint(batch.Value().bestInputs) << ","
                  << "\"kernel_ms\":"
                  << batch.Value().metrics.kernelMilliseconds << ","
                  << "\"wall_ms\":"
                  << Milliseconds(wallBegin, wallEnd) << ","
                  << "\"attempts_per_second\":"
                  << (Milliseconds(wallBegin, wallEnd) == 0.0
                              ? 0.0
                              : batch.Value().evaluatedCandidateCount *
                                      1000.0 /
                                      Milliseconds(wallBegin, wallEnd))
                  << ","
                  << "\"score_initialization_kernel_ms\":"
                  << batch.Value().metrics
                             .scoreInitializationKernelMilliseconds
                  << ","
                  << "\"mutation_kernel_ms\":"
                  << batch.Value().metrics.mutationKernelMilliseconds << ","
                  << "\"simulation_kernel_ms\":"
                  << simulationKernelMilliseconds << ","
                  << "\"finish_refinement_kernel_ms\":"
                  << batch.Value().metrics
                             .finishRefinementKernelMilliseconds
                  << ","
                  << "\"finish_refinement_tick_equivalents\":"
                  << finishRefinementTickEquivalents << ","
                  << "\"winner_kernel_ms\":"
                  << batch.Value().metrics.winnerKernelMilliseconds << ","
                  << "\"winner_reduction_kernel_ms\":"
                  << batch.Value().metrics
                             .winnerReductionKernelMilliseconds
                  << ","
                  << "\"winner_state_capture_kernel_ms\":"
                  << batch.Value().metrics
                             .winnerStateCaptureKernelMilliseconds
                  << ","
                  << "\"finalization_kernel_ms\":"
                  << batch.Value().metrics.finalizationKernelMilliseconds
                  << ","
                  << "\"simulation_kernel_ns_per_tick\":"
                  << normalizedPhysicsNanoseconds << ","
                  << "\"simulation_kernel_ticks_per_second\":"
                  << ticksPerSecond << ","
                  << "\"simulation_threads_per_block\":"
                  << batch.Value().metrics.simulationThreadsPerBlock << ","
                  << "\"simulation_minimum_blocks_per_sm\":"
                  << batch.Value().metrics
                             .simulationSelectedMinimumBlocksPerMultiprocessor
                  << ","
                  << "\"simulation_registers_per_thread\":"
                  << batch.Value().metrics.simulationRegistersPerThread << ","
                  << "\"simulation_local_bytes_per_thread\":"
                  << batch.Value().metrics.simulationLocalBytesPerThread << ","
                  << "\"simulation_active_blocks_per_sm\":"
                  << batch.Value().metrics
                             .simulationActiveBlocksPerMultiprocessor
                  << ","
                  << "\"simulation_theoretical_occupancy\":"
                  << batch.Value().metrics.simulationTheoreticalOccupancy
                  << ","
                  << "\"resident_device_bytes\":"
                  << batch.Value().metrics.residentDeviceBytes << ","
                  << "\"mutation_device_bytes\":"
                  << batch.Value().metrics.mutationDeviceBytes << ","
                  << "\"candidate_input_device_bytes\":"
                  << batch.Value().metrics.candidateInputDeviceBytes << ","
                  << "\"mutation_scratch_device_bytes\":"
                  << batch.Value().metrics.mutationScratchDeviceBytes << ","
                  << "\"baseline_prefix_device_bytes\":"
                  << batch.Value().metrics.baselinePrefixDeviceBytes << ","
                  << "\"candidate_prefix_device_bytes\":"
                  << batch.Value().metrics.candidatePrefixDeviceBytes << ","
                  << "\"candidate_deduplication_device_bytes\":"
                  << batch.Value().metrics
                             .candidateDeduplicationDeviceBytes
                  << ","
                  << "\"winner_selection_device_bytes\":"
                  << batch.Value().metrics.winnerSelectionDeviceBytes << ","
                  << "\"host_to_device_bytes\":"
                  << batch.Value().metrics.hostToDeviceBytes << ","
                  << "\"device_to_host_bytes\":"
                  << batch.Value().metrics.deviceToHostBytes << ","
                  << "\"initial_host_to_device_bytes\":"
                  << baseline.Value().metrics.hostToDeviceBytes << ","
                  << "\"baseline_device_to_host_bytes\":"
                  << baseline.Value().metrics.deviceToHostBytes
                  << "}\n";
        firstCandidateId += candidateCount;
    }
    return 0;
}
