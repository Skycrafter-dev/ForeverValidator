#include "simulation/backends/cuda/cuda_scene_storage.h"
#include "simulation/backends/cuda/cuda_search_executor.h"
#include "simulation/backends/cuda/cuda_session_specialization.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace forevervalidator::simulation;

template<typename T>
class DeviceValue {
public:
    explicit DeviceValue(const T &value) {
        if (cudaMalloc(
                    reinterpret_cast<void **>(&data_),
                    sizeof(T)) == cudaSuccess &&
            cudaMemcpy(
                    data_, &value, sizeof(T),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }

    ~DeviceValue() {
        if (data_ != nullptr) cudaFree(data_);
    }

    DeviceValue(const DeviceValue &) = delete;
    DeviceValue &operator=(const DeviceValue &) = delete;
    const T *Get() const { return data_; }

private:
    T *data_ = nullptr;
};

bool SameInput(
        const CudaSearchInputEvent &left,
        const CudaSearchInputEvent &right) {
    return left.timeMs == right.timeMs &&
            left.action == right.action &&
            left.valueKind == right.valueKind &&
            left.value == right.value;
}

bool SameBest(const CudaSearchBest &left, const CudaSearchBest &right) {
    if (left.valid != right.valid ||
        left.mutation != right.mutation ||
        left.stateCaptured != right.stateCaptured ||
        left.candidateId != right.candidateId ||
        left.mutationCount != right.mutationCount ||
        left.evaluationTick != right.evaluationTick ||
        left.score != right.score ||
        left.timeMs != right.timeMs ||
        left.detail0 != right.detail0 ||
        left.detail1 != right.detail1 ||
        left.inputs.size() != right.inputs.size()) {
        return false;
    }
    for (std::size_t index = 0u; index < left.inputs.size(); ++index) {
        if (!SameInput(left.inputs[index], right.inputs[index])) {
            return false;
        }
    }
    return !left.stateCaptured ||
            std::memcmp(
                    &left.state, &right.state,
                    sizeof(left.state)) == 0;
}

bool SameSemantics(
        const CudaSearchBatchExecution &left,
        const CudaSearchBatchExecution &right) {
    return left.status == right.status &&
            left.firstCandidateId == right.firstCandidateId &&
            left.candidateCount == right.candidateCount &&
            left.evaluatedCandidateCount ==
                    right.evaluatedCandidateCount &&
            left.evaluatorCalls == right.evaluatorCalls &&
            left.qualifyingCandidateCount ==
                    right.qualifyingCandidateCount &&
            left.closestTargetDistance ==
                    right.closestTargetDistance &&
            left.totalMutationCount == right.totalMutationCount &&
            left.mutationImprovementCount ==
                    right.mutationImprovementCount &&
            left.bestChanged == right.bestChanged &&
            SameBest(left.best, right.best);
}

bool NoPrefixStorage(const CudaSearchBatchExecution &execution) {
    return execution.baselinePrefixDeviceBytes == 0u &&
            execution.candidatePrefixDeviceBytes == 0u &&
            execution.candidateDeduplicationDeviceBytes == 0u &&
            !execution.baselinePrefixReuseActive &&
            !execution.candidateDeduplicationActive &&
            execution.deduplicatedCandidateCount == 0u;
}

CudaSearchExecutorConfiguration Configuration(
        const void *deviceScene,
        const void *deviceConfiguration,
        std::uint32_t capacity = 16u);

std::unique_ptr<CudaSearchExecutor> Create(
        const CudaSearchExecutorConfiguration &configuration,
        const char *name);

bool Successful(
        const CudaSearchBatchExecution &execution,
        const char *name);

bool SameProductionStorageTuple(
        const CudaSearchBatchExecution &left,
        const CudaSearchBatchExecution &right) {
    return left.mutationDeviceBytes == right.mutationDeviceBytes &&
            left.candidateInputDeviceBytes ==
                    right.candidateInputDeviceBytes &&
            left.mutationScratchDeviceBytes ==
                    right.mutationScratchDeviceBytes &&
            left.baselinePrefixDeviceBytes ==
                    right.baselinePrefixDeviceBytes &&
            left.candidatePrefixDeviceBytes ==
                    right.candidatePrefixDeviceBytes &&
            left.candidateDeduplicationDeviceBytes ==
                    right.candidateDeduplicationDeviceBytes &&
            left.winnerSelectionDeviceBytes ==
                    right.winnerSelectionDeviceBytes &&
            left.hostToDeviceBytes == right.hostToDeviceBytes;
}

bool ValidHotPathAccounting(
        const CudaSearchBatchExecution &execution,
        std::uint32_t maximumTicks) {
    const CudaSearchHotPathMetrics &metrics = execution.hotPath;
    const std::uint64_t forcePaths =
            metrics.groundForcePassCount +
            metrics.airForcePassCount +
            metrics.physicsCallbackDisabledForcePassCount +
            metrics.zeroDynamicsForcePassCount;
    const bool firstTickBoundsValid =
            metrics.physicallySimulatedCandidateCount == 0u
            ? metrics.firstSimulationTickSum == 0u &&
                      metrics.firstSimulationTickMinimum == 0u &&
                      metrics.firstSimulationTickMaximum == 0u
            : metrics.firstSimulationTickMinimum <=
                              metrics.firstSimulationTickMaximum &&
                      metrics.firstSimulationTickMaximum < maximumTicks &&
                      metrics.firstSimulationTickSum >=
                              metrics.firstSimulationTickMinimum *
                                      metrics.
                                              physicallySimulatedCandidateCount &&
                      metrics.firstSimulationTickSum <=
                              metrics.firstSimulationTickMaximum *
                                      metrics.
                                              physicallySimulatedCandidateCount;
    return metrics.collected && metrics.complete &&
            metrics.forcedRuntimeGenericKernel &&
            metrics.physicallySimulatedCandidateCount <=
                    execution.evaluatedCandidateCount &&
            metrics.executedTickCount <=
                    metrics.physicallySimulatedCandidateCount *
                            maximumTicks &&
            metrics.completedTickCount == metrics.executedTickCount &&
            metrics.physicsSubstepCount == metrics.collisionDetectCount &&
            metrics.physicsSubstepCount == metrics.responseSortCallCount &&
            metrics.surfaceCacheEligibleCount ==
                    metrics.surfaceCacheReuseCount +
                            metrics.surfaceCacheRefreshCount &&
            metrics.surfaceCacheRefreshFailureCount <=
                    metrics.surfaceCacheRefreshCount &&
            metrics.triangleHitCount <= metrics.triangleTestCount &&
            metrics.maximumResponseSortItemCount <=
                    metrics.responseSortItemCount &&
            forcePaths == metrics.physicsSubstepCount &&
            metrics.waterForcePassCount <=
                    metrics.groundForcePassCount +
                            metrics.airForcePassCount &&
            metrics.emptyAirProbeSuccessCount +
                            metrics.emptyAirProbeBlockedCount ==
                    metrics.emptyAirProbeAttemptCount &&
            firstTickBoundsValid;
}

bool CheckEmptyAirToggleMatrix(
        const void *deviceScene,
        const void *deviceConfiguration) {
    if (CudaSearchExecutorConfiguration{}.useEmptyAirCertificate) {
        std::cerr << "empty-air certificate is not default-off\n";
        return false;
    }
    for (const std::uint32_t minimumBlocks : {16u, 12u, 8u}) {
        for (const bool collectMetrics : {false, true}) {
            CudaSearchExecutorConfiguration disabled =
                    Configuration(
                            deviceScene, deviceConfiguration, 5u);
            disabled.simulationMinimumBlocksPerMultiprocessorForTesting =
                    minimumBlocks;
            disabled.collectHotPathMetrics = collectMetrics;
            CudaSearchExecutorConfiguration enabled = disabled;
            enabled.useEmptyAirCertificate = true;

            auto disabledExecutor = Create(
                    disabled, "empty-air disabled executor");
            auto enabledExecutor = Create(
                    enabled, "empty-air enabled executor");
            if (!disabledExecutor || !enabledExecutor) return false;

            const CudaSearchBatchExecution disabledBaseline =
                    disabledExecutor->EvaluateBaseline();
            const CudaSearchBatchExecution enabledBaseline =
                    enabledExecutor->EvaluateBaseline();
            const CudaSearchBatchExecution disabledBatch =
                    disabledExecutor->RunBatch(400u, 5u, false);
            const CudaSearchBatchExecution enabledBatch =
                    enabledExecutor->RunBatch(400u, 5u, false);
            if (!Successful(
                        disabledBaseline,
                        "empty-air disabled baseline") ||
                !Successful(
                        enabledBaseline,
                        "empty-air enabled baseline") ||
                !Successful(
                        disabledBatch,
                        "empty-air disabled batch") ||
                !Successful(
                        enabledBatch,
                        "empty-air enabled batch") ||
                !SameSemantics(disabledBaseline, enabledBaseline) ||
                !SameSemantics(disabledBatch, enabledBatch) ||
                !SameProductionStorageTuple(
                        disabledBaseline, enabledBaseline) ||
                !SameProductionStorageTuple(
                        disabledBatch, enabledBatch) ||
                disabledBaseline.residentDeviceBytes !=
                        enabledBaseline.residentDeviceBytes ||
                disabledBatch.residentDeviceBytes !=
                        enabledBatch.residentDeviceBytes ||
                disabledBaseline.deviceToHostBytes !=
                        enabledBaseline.deviceToHostBytes ||
                disabledBatch.deviceToHostBytes !=
                        enabledBatch.deviceToHostBytes ||
                disabledBatch.
                                simulationSelectedMinimumBlocksPerMultiprocessor !=
                        minimumBlocks ||
                enabledBatch.
                                simulationSelectedMinimumBlocksPerMultiprocessor !=
                        minimumBlocks ||
                disabledBatch.hotPath.collected != collectMetrics ||
                enabledBatch.hotPath.collected != collectMetrics ||
                (collectMetrics &&
                 (!ValidHotPathAccounting(disabledBatch, 6u) ||
                  !ValidHotPathAccounting(enabledBatch, 6u))) ||
                (!collectMetrics &&
                 (disabledBatch.hotPath.emptyAirOpportunityCount != 0u ||
                  enabledBatch.hotPath.emptyAirOpportunityCount != 0u))) {
                std::cerr << "empty-air toggle matrix failed for "
                          << minimumBlocks << " blocks, profiling="
                          << collectMetrics << '\n';
                return false;
            }
        }
    }
    return true;
}

bool CheckHotPathMetrics(
        const void *deviceScene,
        const void *deviceConfiguration) {
    CudaSearchExecutorConfiguration referenceConfiguration =
            Configuration(deviceScene, deviceConfiguration);
    referenceConfiguration.deduplicationReplicaLimitForTesting = 1u;
    CudaSearchExecutorConfiguration profilingConfiguration =
            referenceConfiguration;
    profilingConfiguration.collectHotPathMetrics = true;

    auto reference = Create(
            referenceConfiguration, "hot-path reference executor");
    auto profiling = Create(
            profilingConfiguration, "hot-path profiling executor");
    if (!reference || !profiling) return false;

    const CudaSearchBatchExecution referenceBaseline =
            reference->EvaluateBaseline();
    const CudaSearchBatchExecution profilingBaseline =
            profiling->EvaluateBaseline();
    if (!Successful(referenceBaseline, "hot-path reference baseline") ||
        !Successful(profilingBaseline, "hot-path profiling baseline") ||
        !SameSemantics(referenceBaseline, profilingBaseline) ||
        referenceBaseline.hotPath.collected ||
        referenceBaseline.hotPath.complete ||
        referenceBaseline.hotPath.forcedRuntimeGenericKernel ||
        !ValidHotPathAccounting(profilingBaseline, 6u) ||
        profilingBaseline.hotPath.physicallySimulatedCandidateCount != 1u ||
        profilingBaseline.residentDeviceBytes <=
                referenceBaseline.residentDeviceBytes ||
        profilingBaseline.deviceToHostBytes <=
                referenceBaseline.deviceToHostBytes ||
        !SameProductionStorageTuple(
                referenceBaseline, profilingBaseline)) {
        std::cerr << "hot-path baseline parity/accounting failed\n";
        return false;
    }

    const CudaSearchBatchExecution referenceBatch =
            reference->RunBatch(0u, 16u, false);
    const CudaSearchBatchExecution profilingBatch =
            profiling->RunBatch(0u, 16u, false);
    if (!Successful(referenceBatch, "hot-path reference batch") ||
        !Successful(profilingBatch, "hot-path profiling batch") ||
        !SameSemantics(referenceBatch, profilingBatch) ||
        referenceBatch.hotPath.collected ||
        referenceBatch.hotPath.complete ||
        referenceBatch.hotPath.forcedRuntimeGenericKernel ||
        !ValidHotPathAccounting(profilingBatch, 6u) ||
        profilingBatch.hotPath.physicallySimulatedCandidateCount !=
                profilingBatch.simulatedCandidateCount ||
        profilingBatch.simulatedCandidateCount != 1u ||
        profilingBatch.deduplicatedCandidateCount != 15u ||
        profilingBatch.residentDeviceBytes <=
                referenceBatch.residentDeviceBytes ||
        profilingBatch.deviceToHostBytes <=
                referenceBatch.deviceToHostBytes ||
        !SameProductionStorageTuple(referenceBatch, profilingBatch)) {
        std::cerr << "hot-path batch parity/accounting failed\n";
        return false;
    }

    const CudaSearchBatchExecution referencePartial =
            reference->RunBatch(16u, 5u, false);
    const CudaSearchBatchExecution profilingPartial =
            profiling->RunBatch(16u, 5u, false);
    if (!Successful(referencePartial, "hot-path reference partial") ||
        !Successful(profilingPartial, "hot-path profiling partial") ||
        !SameSemantics(referencePartial, profilingPartial) ||
        !ValidHotPathAccounting(profilingPartial, 6u) ||
        profilingPartial.hotPath.physicallySimulatedCandidateCount != 1u ||
        profilingPartial.simulatedCandidateCount != 1u ||
        profilingPartial.deduplicatedCandidateCount != 4u ||
        !SameProductionStorageTuple(
                referencePartial, profilingPartial)) {
        std::cerr << "partial hot-path records retained stale slots\n";
        return false;
    }

    const CudaSearchBatchExecution cancelled =
            profiling->RunBatch(21u, 16u, true);
    if (cancelled.status != CudaSearchStatus::Cancelled ||
        !cancelled.hotPath.collected ||
        cancelled.hotPath.complete ||
        !cancelled.hotPath.forcedRuntimeGenericKernel) {
        std::cerr << "cancelled hot-path metrics were not finalized\n";
        return false;
    }
    return true;
}

CudaSearchExecutorConfiguration Configuration(
        const void *deviceScene,
        const void *deviceConfiguration,
        std::uint32_t capacity) {
    CudaSearchExecutorConfiguration result;
    result.deviceScene = deviceScene;
    result.deviceStaticConfiguration = deviceConfiguration;
    result.maximumBatchSize = capacity;
    result.tickDurationMs = 10u;
    result.branchTimeMs = 0;
    result.evaluationStartTimeMs = 10;
    result.evaluationEndTimeMs = 60;
    // Sparse insertion staging needs room for its normalized operation even
    // though the final low-entropy candidate contains one event.
    result.maximumEventCount = 4u;
    result.captureBestState = true;
    result.evaluator.kind = CudaSearchEvaluatorKind::Velocity;
    for (std::uint32_t tickIndex = 1u; tickIndex <= 6u;
         ++tickIndex) {
        CudaControlTick tick;
        tick.periodMs = 10u;
        tick.timeMs = tickIndex * 10u;
        tick.actionFlags =
                CudaControlActionSuppressVehicleForceCallbacks;
        result.baselineTicks.push_back(tick);
    }
    CudaSearchModifierConfiguration modifier;
    modifier.kind = CudaSearchModifierKind::InputInsertion;
    modifier.window.minimumTimeMs = 20;
    modifier.window.maximumTimeMs = 20;
    modifier.window.seed = 0x12345678u;
    modifier.steering.enabled = 1u;
    modifier.steering.minimumCount = 1u;
    modifier.steering.maximumCount = 1u;
    modifier.steering.maximumHoldMs = 0;
    modifier.optionFlags = 1u;
    modifier.secondaryAnalogMinimum = 1000;
    modifier.secondaryAnalogMaximum = 1000;
    result.modifiers.push_back(modifier);
    return result;
}

std::unique_ptr<CudaSearchExecutor> Create(
        const CudaSearchExecutorConfiguration &configuration,
        const char *name) {
    std::string diagnostic;
    auto executor = CudaSearchExecutor::Create(
            configuration, &diagnostic);
    if (!executor) {
        std::cerr << name << " creation failed: " << diagnostic << '\n';
    }
    return executor;
}

bool Successful(
        const CudaSearchBatchExecution &execution,
        const char *name) {
    if (execution.status == CudaSearchStatus::Success) return true;
    std::cerr << name << " failed: " << execution.diagnostic << '\n';
    return false;
}

bool CheckPrefixAndDeduplicationParity(
        const void *deviceScene,
        const void *deviceConfiguration) {
    CudaSearchExecutorConfiguration optimizedConfiguration =
            Configuration(deviceScene, deviceConfiguration);
    optimizedConfiguration.deduplicationReplicaLimitForTesting = 1u;
    CudaSearchExecutorConfiguration fullConfiguration =
            optimizedConfiguration;
    fullConfiguration.reuseBaselinePrefixes = false;
    fullConfiguration.sortCandidatesByLocality = false;
    fullConfiguration.deduplicateLowEntropyCandidateInputs = false;
    fullConfiguration.deduplicationReplicaLimitForTesting = 0u;

    auto optimized = Create(optimizedConfiguration, "optimized executor");
    auto full = Create(fullConfiguration, "full executor");
    if (!optimized || !full) return false;

    const CudaSearchBatchExecution optimizedBaseline =
            optimized->EvaluateBaseline();
    const CudaSearchBatchExecution fullBaseline =
            full->EvaluateBaseline();
    if (!Successful(optimizedBaseline, "optimized baseline") ||
        !Successful(fullBaseline, "full baseline") ||
        !SameSemantics(optimizedBaseline, fullBaseline) ||
        optimizedBaseline.baselinePrefixDeviceBytes == 0u ||
        optimizedBaseline.candidatePrefixDeviceBytes == 0u ||
        optimizedBaseline.candidateDeduplicationDeviceBytes == 0u ||
        optimizedBaseline.baselinePrefixReuseActive ||
        optimizedBaseline.candidateDeduplicationActive ||
        !NoPrefixStorage(fullBaseline)) {
        std::cerr << "optimized and full baselines or allocations differ\n";
        return false;
    }

    const CudaSearchBatchExecution optimizedFull =
            optimized->RunBatch(0u, 16u, false);
    const CudaSearchBatchExecution fullFull =
            full->RunBatch(0u, 16u, false);
    if (!Successful(optimizedFull, "optimized full batch") ||
        !Successful(fullFull, "full full batch") ||
        !SameSemantics(optimizedFull, fullFull) ||
        !optimizedFull.baselinePrefixReuseActive ||
        !optimizedFull.candidateDeduplicationActive ||
        optimizedFull.simulatedCandidateCount != 1u ||
        optimizedFull.deduplicatedCandidateCount != 15u ||
        optimizedFull.evaluatedCandidateCount != 16u ||
        optimizedFull.qualifyingCandidateCount != 16u ||
        optimizedFull.totalMutationCount != 16u ||
        fullFull.simulatedCandidateCount != 16u ||
        !NoPrefixStorage(fullFull)) {
        std::cerr << "full-batch prefix/dedup activation or parity failed: "
                  << "optimized simulated="
                  << optimizedFull.simulatedCandidateCount
                  << " deduplicated="
                  << optimizedFull.deduplicatedCandidateCount
                  << " evaluated="
                  << optimizedFull.evaluatedCandidateCount << '\n';
        return false;
    }

    // A short final batch proves capacity-sized representative storage does
    // not leak stale status/sample data into the logical suffix.
    const CudaSearchBatchExecution optimizedPartial =
            optimized->RunBatch(16u, 5u, false);
    const CudaSearchBatchExecution fullPartial =
            full->RunBatch(16u, 5u, false);
    if (!Successful(optimizedPartial, "optimized partial batch") ||
        !Successful(fullPartial, "full partial batch") ||
        !SameSemantics(optimizedPartial, fullPartial) ||
        !optimizedPartial.baselinePrefixReuseActive ||
        !optimizedPartial.candidateDeduplicationActive ||
        optimizedPartial.simulatedCandidateCount != 1u ||
        optimizedPartial.deduplicatedCandidateCount != 4u ||
        optimizedPartial.evaluatedCandidateCount != 5u ||
        optimizedPartial.qualifyingCandidateCount != 5u ||
        optimizedPartial.totalMutationCount != 5u ||
        fullPartial.simulatedCandidateCount != 5u ||
        !NoPrefixStorage(fullPartial)) {
        std::cerr << "partial-batch representative propagation failed\n";
        return false;
    }
    return true;
}

bool CheckDisabledEligibility(
        const void *deviceScene,
        const void *deviceConfiguration) {
    CudaSearchExecutorConfiguration conditionConfiguration =
            Configuration(deviceScene, deviceConfiguration, 5u);
    conditionConfiguration.condition.emplace();
    CudaSearchConditionInstruction always;
    always.opcode = CudaSearchConditionOpcode::Constant;
    always.x = 1.0;
    conditionConfiguration.condition->instructions.push_back(always);
    auto condition = Create(conditionConfiguration, "condition executor");
    if (!condition) return false;
    const CudaSearchBatchExecution conditionBaseline =
            condition->EvaluateBaseline();
    const CudaSearchBatchExecution conditionBatch =
            condition->RunBatch(0u, 5u, false);
    if (!Successful(conditionBaseline, "condition baseline") ||
        !Successful(conditionBatch, "condition batch") ||
        !NoPrefixStorage(conditionBaseline) ||
        !NoPrefixStorage(conditionBatch) ||
        conditionBatch.evaluatedCandidateCount != 5u ||
        conditionBatch.simulatedCandidateCount != 5u) {
        std::cerr << "condition mode allocated prefix/dedup storage\n";
        return false;
    }

    CudaSearchExecutorConfiguration stuntConfiguration =
            Configuration(deviceScene, deviceConfiguration, 5u);
    stuntConfiguration.branchState.stuntsEnabled = true;
    auto stunt = Create(stuntConfiguration, "stunt executor");
    if (!stunt) return false;
    const CudaSearchBatchExecution stuntBaseline =
            stunt->EvaluateBaseline();
    const CudaSearchBatchExecution stuntBatch =
            stunt->RunBatch(0u, 5u, false);
    if (!Successful(stuntBaseline, "stunt baseline") ||
        !Successful(stuntBatch, "stunt batch") ||
        !NoPrefixStorage(stuntBaseline) ||
        !NoPrefixStorage(stuntBatch) ||
        stuntBatch.evaluatedCandidateCount != 5u ||
        stuntBatch.simulatedCandidateCount != 5u) {
        std::cerr << "stunt mode allocated prefix/dedup storage\n";
        return false;
    }
    return true;
}

std::unique_ptr<CudaSearchBatchExecution> ForcedExecution(
        const void *deviceScene,
        const void *deviceConfiguration,
        std::uint32_t minimumBlocks) {
    CudaSearchExecutorConfiguration optimizedConfiguration =
            Configuration(deviceScene, deviceConfiguration, 8u);
    optimizedConfiguration.deduplicationReplicaLimitForTesting = 1u;
    optimizedConfiguration.
            simulationMinimumBlocksPerMultiprocessorForTesting =
            minimumBlocks;
    CudaSearchExecutorConfiguration fullConfiguration =
            optimizedConfiguration;
    fullConfiguration.reuseBaselinePrefixes = false;
    fullConfiguration.sortCandidatesByLocality = false;
    fullConfiguration.deduplicateLowEntropyCandidateInputs = false;
    fullConfiguration.deduplicationReplicaLimitForTesting = 0u;
    CudaSearchExecutorConfiguration profilingConfiguration =
            optimizedConfiguration;
    profilingConfiguration.collectHotPathMetrics = true;

    auto optimized = Create(
            optimizedConfiguration, "forced optimized executor");
    auto full = Create(fullConfiguration, "forced full executor");
    auto profiling = Create(
            profilingConfiguration, "forced profiling executor");
    if (!optimized || !full || !profiling) return nullptr;
    {
        const std::unique_ptr<CudaSearchBatchExecution>
                optimizedBaseline(
                        new CudaSearchBatchExecution(
                                optimized->EvaluateBaseline()));
        const std::unique_ptr<CudaSearchBatchExecution> fullBaseline(
                new CudaSearchBatchExecution(
                        full->EvaluateBaseline()));
        const std::unique_ptr<CudaSearchBatchExecution>
                profilingBaseline(
                        new CudaSearchBatchExecution(
                                profiling->EvaluateBaseline()));
        if (!Successful(
                    *optimizedBaseline, "forced optimized baseline") ||
            !Successful(*fullBaseline, "forced full baseline") ||
            !Successful(
                    *profilingBaseline, "forced profiling baseline") ||
            !SameSemantics(*optimizedBaseline, *fullBaseline) ||
            !SameSemantics(*optimizedBaseline, *profilingBaseline) ||
            optimizedBaseline->hotPath.collected ||
            !ValidHotPathAccounting(*profilingBaseline, 6u) ||
            optimizedBaseline->
                            simulationSelectedMinimumBlocksPerMultiprocessor !=
                    minimumBlocks ||
            fullBaseline->
                            simulationSelectedMinimumBlocksPerMultiprocessor !=
                    minimumBlocks ||
            profilingBaseline->
                            simulationSelectedMinimumBlocksPerMultiprocessor !=
                    minimumBlocks ||
            !SameProductionStorageTuple(
                    *optimizedBaseline, *profilingBaseline)) {
            std::cerr << "forced " << minimumBlocks
                      << "-block baseline profiling matrix failed\n";
            return nullptr;
        }
    }

    std::unique_ptr<CudaSearchBatchExecution> optimizedBatch(
            new CudaSearchBatchExecution(
                    optimized->RunBatch(100u, 5u, false)));
    const std::unique_ptr<CudaSearchBatchExecution> fullBatch(
            new CudaSearchBatchExecution(
                    full->RunBatch(100u, 5u, false)));
    std::unique_ptr<CudaSearchBatchExecution> profilingBatch(
            new CudaSearchBatchExecution(
                    profiling->RunBatch(100u, 5u, false)));
    if (!Successful(*optimizedBatch, "forced optimized batch") ||
        !Successful(*fullBatch, "forced full batch") ||
        !Successful(*profilingBatch, "forced profiling batch") ||
        !SameSemantics(*optimizedBatch, *fullBatch) ||
        !SameSemantics(*optimizedBatch, *profilingBatch) ||
        optimizedBatch->hotPath.collected ||
        !ValidHotPathAccounting(*profilingBatch, 6u) ||
        optimizedBatch->
                        simulationSelectedMinimumBlocksPerMultiprocessor !=
                minimumBlocks ||
        fullBatch->simulationSelectedMinimumBlocksPerMultiprocessor !=
                minimumBlocks ||
        profilingBatch->
                        simulationSelectedMinimumBlocksPerMultiprocessor !=
                minimumBlocks ||
        !optimizedBatch->baselinePrefixReuseActive ||
        !optimizedBatch->candidateDeduplicationActive ||
        optimizedBatch->simulatedCandidateCount != 1u ||
        optimizedBatch->deduplicatedCandidateCount != 4u ||
        profilingBatch->hotPath.physicallySimulatedCandidateCount != 1u ||
        !SameProductionStorageTuple(*optimizedBatch, *profilingBatch) ||
        !NoPrefixStorage(*fullBatch) ||
        fullBatch->simulatedCandidateCount != 5u) {
        std::cerr << "forced " << minimumBlocks
                  << "-block batch profiling matrix failed\n";
        return nullptr;
    }
    return profilingBatch;
}

bool CheckForcedVariants(
        const void *deviceScene,
        const void *deviceConfiguration) {
    const auto throughput = ForcedExecution(
            deviceScene, deviceConfiguration, 16u);
    const auto tail = ForcedExecution(
            deviceScene, deviceConfiguration, 12u);
    const auto denseTail = ForcedExecution(
            deviceScene, deviceConfiguration, 8u);
    return throughput && tail && denseTail &&
            SameSemantics(*throughput, *tail) &&
            SameSemantics(*throughput, *denseTail);
}

bool CheckSessionSpecializationLookupAndProfilingBypass(
        const CudaPackedSceneHeader &scene,
        const void *deviceScene) {
    struct SessionConfigurationFixture {
        CudaPackedStaticConfigurationHeader header;
        CudaTuningCurveKey curveKey;
        CudaVehicleCollisionShape collisionShape;
    } fixture;
    fixture.header.totalSize = sizeof(fixture);
    fixture.header.curveKeys.offset =
            offsetof(SessionConfigurationFixture, curveKey);
    fixture.header.curveKeys.count = 1u;
    fixture.header.curveKeys.stride = sizeof(fixture.curveKey);
    fixture.header.collisionShapes.offset =
            offsetof(SessionConfigurationFixture, collisionShape);
    fixture.header.collisionShapes.count = 1u;
    fixture.header.collisionShapes.stride =
            sizeof(fixture.collisionShape);
    fixture.collisionShape.localPose.SetIdentity();
    fixture.collisionShape.bodyPose.SetIdentity();
    DeviceValue<SessionConfigurationFixture> deviceConfiguration(
            fixture);
    if (deviceConfiguration.Get() == nullptr) {
        std::cerr << "session configuration fixture upload failed\n";
        return false;
    }

    auto module =
            std::make_shared<cuda::specialization::SessionModule>();
    std::string diagnostic;
    if (!module->Build(
                fixture.header,
                static_cast<std::uint64_t>(
                        reinterpret_cast<std::uintptr_t>(
                                deviceConfiguration.Get())),
                scene,
                static_cast<std::uint64_t>(
                        reinterpret_cast<std::uintptr_t>(deviceScene)),
                &diagnostic) ||
        !module->Ready() ||
        module->Kernel(16u, false) == nullptr ||
        module->Kernel(12u, false) == nullptr ||
        module->Kernel(8u, false) == nullptr ||
        module->Kernel(16u, true) == nullptr ||
        module->Kernel(12u, true) == nullptr ||
        module->Kernel(8u, true) == nullptr) {
        std::cerr << "session specialization lookup failed: "
                  << diagnostic << '\n';
        return false;
    }

    const auto sameCertificateSpecialization =
            [&](std::uint32_t minimumBlocks) {
        const char *disabledName = nullptr;
        const char *enabledName = nullptr;
        if (cuFuncGetName(
                    &disabledName,
                    module->Kernel(minimumBlocks, false)) != CUDA_SUCCESS ||
            cuFuncGetName(
                    &enabledName,
                    module->Kernel(minimumBlocks, true)) != CUDA_SUCCESS ||
            disabledName == nullptr || enabledName == nullptr) {
            std::cerr << "reading session-specialized kernel names failed\n";
            return false;
        }
        std::string normalizedEnabledName(enabledName);
        const std::string enabledSuffix =
                "ELj" + std::to_string(minimumBlocks) +
                "ELb1ELb0EE";
        const std::string disabledSuffix =
                "ELj" + std::to_string(minimumBlocks) +
                "ELb0ELb0EE";
        const std::size_t suffixPosition =
                normalizedEnabledName.find(enabledSuffix);
        if (suffixPosition == std::string::npos) {
            std::cerr << "enabled certificate kernel has an unexpected name: "
                      << enabledName << '\n';
            return false;
        }
        normalizedEnabledName.replace(
                suffixPosition,
                enabledSuffix.size(),
                disabledSuffix);
        if (normalizedEnabledName != disabledName) {
            std::cerr << "certificate changed session specialization at "
                      << minimumBlocks << " blocks per SM\ndisabled: "
                      << disabledName << "\nenabled: " << enabledName << '\n';
            return false;
        }
        return true;
    };
    if (!sameCertificateSpecialization(16u) ||
        !sameCertificateSpecialization(12u) ||
        !sameCertificateSpecialization(8u)) {
        return false;
    }

    CudaSearchExecutorConfiguration specializedConfiguration =
            Configuration(
                    deviceScene, deviceConfiguration.Get(), 8u);
    specializedConfiguration.deduplicationReplicaLimitForTesting = 1u;
    specializedConfiguration.
            simulationMinimumBlocksPerMultiprocessorForTesting = 16u;
    specializedConfiguration.sessionSpecialization = module;
    CudaSearchExecutorConfiguration profilingConfiguration =
            specializedConfiguration;
    profilingConfiguration.collectHotPathMetrics = true;
    CudaSearchExecutorConfiguration certificateConfiguration =
            specializedConfiguration;
    certificateConfiguration.useEmptyAirCertificate = true;

    auto specialized = Create(
            specializedConfiguration,
            "supplied session-specialized executor");
    auto profiling = Create(
            profilingConfiguration,
            "session-bypass profiling executor");
    auto certificate = Create(
            certificateConfiguration,
            "empty-air session-specialized executor");
    if (!specialized || !profiling || !certificate) return false;
    const CudaSearchBatchExecution specializedBaseline =
            specialized->EvaluateBaseline();
    const CudaSearchBatchExecution profilingBaseline =
            profiling->EvaluateBaseline();
    const CudaSearchBatchExecution specializedBatch =
            specialized->RunBatch(200u, 5u, false);
    const CudaSearchBatchExecution profilingBatch =
            profiling->RunBatch(200u, 5u, false);
    const CudaSearchBatchExecution certificateBaseline =
            certificate->EvaluateBaseline();
    const CudaSearchBatchExecution certificateBatch =
            certificate->RunBatch(200u, 5u, false);
    const cuda::specialization::KernelMetrics &moduleMetrics =
            module->Metrics(16u, false);
    const cuda::specialization::KernelMetrics &certificateMetrics =
            module->Metrics(16u, true);
    if (!Successful(
                specializedBaseline,
                "supplied session-specialized baseline") ||
        !Successful(
                profilingBaseline,
                "session-bypass profiling baseline") ||
        !Successful(
                specializedBatch,
                "supplied session-specialized batch") ||
        !Successful(
                profilingBatch,
                "session-bypass profiling batch") ||
        !Successful(
                certificateBaseline,
                "empty-air session-specialized baseline") ||
        !Successful(
                certificateBatch,
                "empty-air session-specialized batch") ||
        !SameSemantics(specializedBaseline, profilingBaseline) ||
        !SameSemantics(specializedBatch, profilingBatch) ||
        !SameSemantics(specializedBaseline, certificateBaseline) ||
        !SameSemantics(specializedBatch, certificateBatch) ||
        specializedBaseline.hotPath.collected ||
        specializedBatch.hotPath.collected ||
        specializedBaseline.simulationRegistersPerThread !=
                moduleMetrics.registersPerThread ||
        specializedBaseline.simulationLocalBytesPerThread !=
                moduleMetrics.localBytesPerThread ||
        specializedBaseline.
                        simulationActiveBlocksPerMultiprocessor !=
                moduleMetrics.activeBlocksPerMultiprocessor ||
        certificateBaseline.simulationRegistersPerThread !=
                certificateMetrics.registersPerThread ||
        certificateBaseline.simulationLocalBytesPerThread !=
                certificateMetrics.localBytesPerThread ||
        certificateBaseline.
                        simulationActiveBlocksPerMultiprocessor !=
                certificateMetrics.activeBlocksPerMultiprocessor ||
        !ValidHotPathAccounting(profilingBaseline, 6u) ||
        !ValidHotPathAccounting(profilingBatch, 6u) ||
        !profilingBaseline.hotPath.forcedRuntimeGenericKernel ||
        !profilingBatch.hotPath.forcedRuntimeGenericKernel) {
        std::cerr << "profiling did not bypass the supplied session "
                     "module through generic runtime AOT\n";
        return false;
    }
    return true;
}

bool CheckPostLaunchCancellationFinalization(
        const void *deviceScene,
        const void *deviceConfiguration) {
    constexpr std::uint32_t TimelineTicks = 4096u;
    constexpr std::uint32_t CandidateCount = 256u;
    CudaSearchExecutorConfiguration configuration =
            Configuration(
                    deviceScene,
                    deviceConfiguration,
                    CandidateCount);
    configuration.collectHotPathMetrics = true;
    configuration.reuseBaselinePrefixes = false;
    configuration.sortCandidatesByLocality = false;
    configuration.deduplicateLowEntropyCandidateInputs = false;
    configuration.deduplicationReplicaLimitForTesting = 0u;
    configuration.evaluationEndTimeMs =
            static_cast<std::int64_t>(TimelineTicks) * 10;
    configuration.baselineTicks.clear();
    configuration.baselineTicks.reserve(TimelineTicks);
    for (std::uint32_t tickIndex = 1u;
         tickIndex <= TimelineTicks;
         ++tickIndex) {
        CudaControlTick tick;
        tick.periodMs = 10u;
        tick.timeMs = tickIndex * 10u;
        tick.actionFlags =
                CudaControlActionSuppressVehicleForceCallbacks;
        configuration.baselineTicks.push_back(tick);
    }

    auto executor = Create(
            configuration, "post-launch cancellation executor");
    if (!executor) return false;
    const CudaSearchBatchExecution baseline =
            executor->EvaluateBaseline();
    if (!Successful(baseline, "post-launch cancellation baseline") ||
        !ValidHotPathAccounting(baseline, TimelineTicks)) {
        return false;
    }

    std::uint32_t cancellationProbeCount = 0u;
    const CudaSearchBatchExecution cancelled = executor->RunBatch(
            300u,
            CandidateCount,
            [&cancellationProbeCount] {
                ++cancellationProbeCount;
                return cancellationProbeCount >= 2u;
            });
    if (cancellationProbeCount < 2u ||
        cancelled.status != CudaSearchStatus::Cancelled ||
        !cancelled.hotPath.collected ||
        cancelled.hotPath.complete ||
        !cancelled.hotPath.forcedRuntimeGenericKernel ||
        cancelled.hotPath.physicallySimulatedCandidateCount == 0u ||
        cancelled.hotPath.physicallySimulatedCandidateCount >
                CandidateCount ||
        cancelled.hotPath.completedTickCount !=
                cancelled.hotPath.executedTickCount ||
        cancelled.hotPath.executedTickCount >
                cancelled.hotPath.physicallySimulatedCandidateCount *
                        TimelineTicks) {
        std::cerr << "post-launch cancellation metrics were not "
                     "finalized deterministically\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        static_cast<void>(cudaGetLastError());
        return 77;
    }

    CudaPackedSceneHeader scene;
    scene.totalSize = sizeof(scene);
    CudaPackedStaticConfigurationHeader configuration;
    configuration.totalSize = sizeof(configuration);
    DeviceValue<CudaPackedSceneHeader> deviceScene(scene);
    DeviceValue<CudaPackedStaticConfigurationHeader>
            deviceConfiguration(configuration);
    if (deviceScene.Get() == nullptr ||
        deviceConfiguration.Get() == nullptr) {
        std::cerr << "CUDA test fixture upload failed\n";
        return 1;
    }

    if (!CheckPrefixAndDeduplicationParity(
                deviceScene.Get(), deviceConfiguration.Get())) return 1;
    if (!CheckHotPathMetrics(
                deviceScene.Get(), deviceConfiguration.Get())) return 1;
    if (!CheckEmptyAirToggleMatrix(
                deviceScene.Get(), deviceConfiguration.Get())) return 1;
    if (!CheckDisabledEligibility(
                deviceScene.Get(), deviceConfiguration.Get())) return 1;
    if (!CheckForcedVariants(
                deviceScene.Get(), deviceConfiguration.Get())) return 1;
    if (!CheckSessionSpecializationLookupAndProfilingBypass(
                scene,
                deviceScene.Get())) return 1;
    if (!CheckPostLaunchCancellationFinalization(
                deviceScene.Get(), deviceConfiguration.Get())) return 1;
    return 0;
}
