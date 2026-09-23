#include "simulation/backends/hip/hip_state_bridge.h"

namespace forevervalidator::simulation::hip_bridge {

CudaTimelineBatchResult ToCudaResult(
        const HipTimelineBatchResult &source) {
    CudaTimelineBatchResult result;
    result.status = static_cast<CudaTimelineStatus>(source.status);
    result.metrics = BitCopy<CudaTimelineExecutionMetrics>(source.metrics);
    result.winnerCandidateIndex = source.winnerCandidateIndex;
    result.winnerCandidateId = source.winnerCandidateId;
    result.diagnostic = source.diagnostic;
    result.candidates.reserve(source.candidates.size());
    for (const HipCandidateTimelineOutput &candidate : source.candidates) {
        CudaCandidateTimelineOutput converted;
        converted.status = static_cast<CudaTimelineStatus>(candidate.status);
        converted.failureTick = candidate.failureTick;
        converted.failureDetail = candidate.failureDetail;
        converted.executedTickCount = candidate.executedTickCount;
        converted.executedRespawnCount = candidate.executedRespawnCount;
        converted.finalState = BitCopy<CudaCandidateState>(candidate.finalState);
        converted.observations.reserve(candidate.observations.size());
        for (const HipTimelineObservation &observation :
             candidate.observations) {
            converted.observations.push_back(
                    BitCopy<CudaTimelineObservation>(observation));
        }
        result.candidates.push_back(std::move(converted));
    }
    return result;
}

HipSearchExecutorConfiguration ToHipConfiguration(
        const CudaSearchExecutorConfiguration &source) {
    HipSearchExecutorConfiguration result;
    result.deviceScene = source.deviceScene;
    result.deviceStaticConfiguration = source.deviceStaticConfiguration;
    result.branchState = BitCopy<HipCandidateState>(source.branchState);
    result.baselineTicks.reserve(source.baselineTicks.size());
    for (const CudaControlTick &tick : source.baselineTicks) {
        result.baselineTicks.push_back(BitCopy<HipControlTick>(tick));
    }
    result.baselineInputs.reserve(source.baselineInputs.size());
    for (const CudaSearchInputEvent &input : source.baselineInputs) {
        result.baselineInputs.push_back(BitCopy<HipSearchInputEvent>(input));
    }
    result.modifiers.reserve(source.modifiers.size());
    for (const CudaSearchModifierConfiguration &modifier : source.modifiers) {
        result.modifiers.push_back(
                BitCopy<HipSearchModifierConfiguration>(modifier));
    }
    result.smoothWeights = source.smoothWeights;
    result.evaluator = BitCopy<HipSearchEvaluatorConfiguration>(
            source.evaluator);
    if (source.condition) {
        HipSearchConditionConfiguration condition;
        condition.lastImprovementTimeSeconds =
                source.condition->lastImprovementTimeSeconds;
        condition.lastRestartTimeSeconds =
                source.condition->lastRestartTimeSeconds;
        condition.instructions.reserve(source.condition->instructions.size());
        for (const CudaSearchConditionInstruction &instruction :
             source.condition->instructions) {
            condition.instructions.push_back(
                    BitCopy<HipSearchConditionInstruction>(instruction));
        }
        result.condition = std::move(condition);
    }
    result.maximumBatchSize = source.maximumBatchSize;
    result.tickDurationMs = source.tickDurationMs;
    result.prestartDurationMs = source.prestartDurationMs;
    result.branchTimeMs = source.branchTimeMs;
    result.evaluationStartTimeMs = source.evaluationStartTimeMs;
    result.evaluationEndTimeMs = source.evaluationEndTimeMs;
    result.maximumEventCount = source.maximumEventCount;
    result.useLegacyMutationPipelineForTesting =
            source.useLegacyMutationPipelineForTesting;
    result.captureBestState = source.captureBestState;
    if (source.incumbent) {
        result.incumbent = BitCopy<HipSearchIncumbent>(*source.incumbent);
    }
    return result;
}

CudaSearchBatchExecution ToCudaResult(
        const HipSearchBatchExecution &source) {
    CudaSearchBatchExecution result;
    result.status = static_cast<CudaSearchStatus>(source.status);
    result.firstCandidateId = source.firstCandidateId;
    result.candidateCount = source.candidateCount;
    result.evaluatedCandidateCount = source.evaluatedCandidateCount;
    result.evaluatorCalls = source.evaluatorCalls;
    result.totalMutationCount = source.totalMutationCount;
    result.mutationImprovementCount = source.mutationImprovementCount;
    result.bestChanged = source.bestChanged;
    result.best.valid = source.best.valid;
    result.best.mutation = source.best.mutation;
    result.best.stateCaptured = source.best.stateCaptured;
    result.best.candidateId = source.best.candidateId;
    result.best.mutationCount = source.best.mutationCount;
    result.best.evaluationTick = source.best.evaluationTick;
    result.best.score = source.best.score;
    result.best.timeMs = source.best.timeMs;
    result.best.detail0 = source.best.detail0;
    result.best.detail1 = source.best.detail1;
    result.best.state = BitCopy<CudaCandidateState>(source.best.state);
    result.best.inputs.reserve(source.best.inputs.size());
    for (const HipSearchInputEvent &input : source.best.inputs) {
        result.best.inputs.push_back(BitCopy<CudaSearchInputEvent>(input));
    }
    result.residentDeviceBytes = source.residentDeviceBytes;
    result.mutationDeviceBytes = source.mutationDeviceBytes;
    result.candidateInputDeviceBytes = source.candidateInputDeviceBytes;
    result.mutationScratchDeviceBytes = source.mutationScratchDeviceBytes;
    result.winnerSelectionDeviceBytes = source.winnerSelectionDeviceBytes;
    result.hostToDeviceBytes = source.hostToDeviceBytes;
    result.deviceToHostBytes = source.deviceToHostBytes;
    result.kernelMilliseconds = source.kernelMilliseconds;
    result.scoreInitializationKernelMilliseconds =
            source.scoreInitializationKernelMilliseconds;
    result.mutationKernelMilliseconds = source.mutationKernelMilliseconds;
    result.simulationKernelMilliseconds = source.simulationKernelMilliseconds;
    result.finishRefinementKernelMilliseconds =
            source.finishRefinementKernelMilliseconds;
    result.winnerKernelMilliseconds = source.winnerKernelMilliseconds;
    result.winnerReductionKernelMilliseconds =
            source.winnerReductionKernelMilliseconds;
    result.winnerStateCaptureKernelMilliseconds =
            source.winnerStateCaptureKernelMilliseconds;
    result.finalizationKernelMilliseconds =
            source.finalizationKernelMilliseconds;
    result.simulationThreadsPerBlock = source.simulationThreadsPerBlock;
    result.simulationRegistersPerThread = source.simulationRegistersPerThread;
    result.simulationLocalBytesPerThread =
            source.simulationLocalBytesPerThread;
    result.simulationActiveBlocksPerMultiprocessor =
            source.simulationActiveBlocksPerMultiprocessor;
    result.simulationTheoreticalOccupancy =
            source.simulationTheoreticalOccupancy;
    result.diagnostic = source.diagnostic;
    return result;
}

}  // namespace forevervalidator::simulation::hip_bridge
