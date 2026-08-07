#include "simulation/backends/cuda/cuda_scene_storage.h"
#include "simulation/backends/cuda/cuda_search_executor.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
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
        std::uint32_t capacity = 16u) {
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

std::optional<CudaSearchBatchExecution> ForcedExecution(
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

    auto optimized = Create(
            optimizedConfiguration, "forced optimized executor");
    auto full = Create(fullConfiguration, "forced full executor");
    if (!optimized || !full) return std::nullopt;
    const CudaSearchBatchExecution optimizedBaseline =
            optimized->EvaluateBaseline();
    const CudaSearchBatchExecution fullBaseline =
            full->EvaluateBaseline();
    CudaSearchBatchExecution optimizedBatch =
            optimized->RunBatch(100u, 5u, false);
    const CudaSearchBatchExecution fullBatch =
            full->RunBatch(100u, 5u, false);
    if (!Successful(optimizedBaseline, "forced optimized baseline") ||
        !Successful(fullBaseline, "forced full baseline") ||
        !Successful(optimizedBatch, "forced optimized batch") ||
        !Successful(fullBatch, "forced full batch") ||
        !SameSemantics(optimizedBaseline, fullBaseline) ||
        !SameSemantics(optimizedBatch, fullBatch) ||
        optimizedBaseline.
                        simulationSelectedMinimumBlocksPerMultiprocessor !=
                minimumBlocks ||
        fullBaseline.simulationSelectedMinimumBlocksPerMultiprocessor !=
                minimumBlocks ||
        optimizedBatch.
                        simulationSelectedMinimumBlocksPerMultiprocessor !=
                minimumBlocks ||
        fullBatch.simulationSelectedMinimumBlocksPerMultiprocessor !=
                minimumBlocks ||
        !optimizedBatch.baselinePrefixReuseActive ||
        !optimizedBatch.candidateDeduplicationActive ||
        optimizedBatch.simulatedCandidateCount != 1u ||
        optimizedBatch.deduplicatedCandidateCount != 4u ||
        !NoPrefixStorage(fullBatch) ||
        fullBatch.simulatedCandidateCount != 5u) {
        std::cerr << "forced " << minimumBlocks
                  << "-block optimized/full matrix failed\n";
        return std::nullopt;
    }
    return optimizedBatch;
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
                deviceScene.Get(), deviceConfiguration.Get()) ||
        !CheckDisabledEligibility(
                deviceScene.Get(), deviceConfiguration.Get()) ||
        !CheckForcedVariants(
                deviceScene.Get(), deviceConfiguration.Get())) {
        return 1;
    }
    return 0;
}
