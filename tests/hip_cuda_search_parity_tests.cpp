#include "simulation/backends/cuda/cuda_scene_storage.h"
#include "simulation/backends/cuda/cuda_search_executor.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/backends/hip/generated/hip_search_executor.h"
#include "simulation/backends/hip/hip_state_bridge.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

bool SameResult(
        const forevervalidator::simulation::CudaSearchBatchExecution &left,
        const forevervalidator::simulation::CudaSearchBatchExecution &right) {
    if (left.status != right.status ||
        left.evaluatedCandidateCount != right.evaluatedCandidateCount ||
        left.evaluatorCalls != right.evaluatorCalls ||
        left.totalMutationCount != right.totalMutationCount ||
        left.mutationImprovementCount != right.mutationImprovementCount ||
        left.bestChanged != right.bestChanged ||
        left.best.valid != right.best.valid ||
        left.best.candidateId != right.best.candidateId ||
        left.best.mutationCount != right.best.mutationCount ||
        left.best.evaluationTick != right.best.evaluationTick ||
        left.best.score != right.best.score ||
        left.best.timeMs != right.best.timeMs ||
        left.best.detail0 != right.best.detail0 ||
        left.best.detail1 != right.best.detail1 ||
        left.best.inputs.size() != right.best.inputs.size()) {
        return false;
    }
    if (left.best.stateCaptured != right.best.stateCaptured ||
        (left.best.stateCaptured &&
         std::memcmp(&left.best.state, &right.best.state,
                     sizeof(left.best.state)) != 0)) {
        return false;
    }
    for (std::size_t index = 0u; index < left.best.inputs.size(); ++index) {
        const auto &a = left.best.inputs[index];
        const auto &b = right.best.inputs[index];
        if (a.timeMs != b.timeMs || a.action != b.action ||
            a.valueKind != b.valueKind || a.value != b.value) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    using namespace forevervalidator::simulation;
    const bool benchmarkLarge =
            argc == 2 && std::strcmp(argv[1], "--benchmark-large") == 0;
    const bool customBenchmark = argc >= 2 &&
            std::strcmp(argv[1], "--benchmark") == 0 && argc >= 4;
    const bool benchmark = benchmarkLarge || customBenchmark ||
            (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0);
    std::uint32_t batchSize = benchmarkLarge ? 1024u : 256u;
    std::uint32_t tickCount = 100u;
    bool hipFirst = false;
    if (customBenchmark) {
        const auto parsePositive = [](const char *text, std::uint32_t *value) {
            const char *end = text + std::strlen(text);
            const auto result = std::from_chars(text, end, *value);
            return result.ec == std::errc{} && result.ptr == end &&
                   *value != 0u;
        };
        if (argc > 5 ||
            !parsePositive(argv[2], &batchSize) ||
            !parsePositive(argv[3], &tickCount) ||
            tickCount > static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max() / 10) ||
            (argc == 5 && std::strcmp(argv[4], "--hip-first") != 0)) {
            std::cerr << "usage: --benchmark [batch-size tick-count "
                         "[--hip-first]]\n";
            return 2;
        }
        hipFirst = argc == 5;
    } else if (argc > 2 || (argc == 2 && !benchmark)) {
        std::cerr << "usage: --benchmark [batch-size tick-count "
                     "[--hip-first]]\n";
        return 2;
    }
    const std::int32_t endTimeMs = static_cast<std::int32_t>(tickCount * 10u);
    CudaPackedSceneHeader scene;
    scene.totalSize = sizeof(scene);
    CudaPackedStaticConfigurationHeader configuration;
    configuration.totalSize = sizeof(configuration);
    void *deviceScene = nullptr;
    void *deviceConfiguration = nullptr;
    if (hipMalloc(&deviceScene, sizeof(scene)) != hipSuccess ||
        hipMalloc(&deviceConfiguration, sizeof(configuration)) != hipSuccess ||
        hipMemcpy(deviceScene, &scene, sizeof(scene),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(deviceConfiguration, &configuration,
                  sizeof(configuration), hipMemcpyHostToDevice) != hipSuccess) {
        std::cerr << "search fixture allocation failed\n";
        return 1;
    }

    CudaSearchExecutorConfiguration cudaConfig;
    cudaConfig.deviceScene = deviceScene;
    cudaConfig.deviceStaticConfiguration = deviceConfiguration;
    cudaConfig.branchState.firstStep = false;
    cudaConfig.branchState.vehicle.wheels.count = 4u;
    cudaConfig.maximumBatchSize = batchSize;
    cudaConfig.maximumEventCount = 16u;
    cudaConfig.tickDurationMs = 10u;
    cudaConfig.branchTimeMs = 0;
    cudaConfig.evaluationStartTimeMs = 10;
    cudaConfig.evaluationEndTimeMs = endTimeMs;
    cudaConfig.evaluator.kind = CudaSearchEvaluatorKind::Velocity;
    for (std::uint32_t time = 10u; time <=
         static_cast<std::uint32_t>(endTimeMs); time += 10u) {
        CudaControlTick tick;
        tick.periodMs = 10u;
        tick.timeMs = time;
        cudaConfig.baselineTicks.push_back(tick);
    }
    cudaConfig.baselineInputs.push_back({10, 4u, 2u, 0});
    cudaConfig.baselineInputs.push_back({20, 4u, 2u, 500});
    CudaSearchModifierConfiguration modifier;
    modifier.kind = CudaSearchModifierKind::RandomSteering;
    modifier.window.minimumTimeMs = 10;
    modifier.window.maximumTimeMs = endTimeMs;
    modifier.minimumCount = 1u;
    modifier.maximumCount = 1u;
    modifier.steering.enabled = 1u;
    modifier.steering.minimumCount = 1u;
    modifier.steering.maximumCount = 1u;
    modifier.analogMinimum = -1000;
    modifier.analogMaximum = 1000;
    cudaConfig.modifiers.push_back(modifier);
    CudaSearchConditionConfiguration condition;
    condition.instructions.push_back({
            CudaSearchConditionOpcode::Constant,
            CudaSearchConditionValue::Speed, 1.0, 0.0, 0.0});
    cudaConfig.condition = condition;

    std::string cudaDiagnostic;
    std::string hipDiagnostic;
    auto cuda = CudaSearchExecutor::Create(cudaConfig, &cudaDiagnostic);
    if (!cuda) {
        std::cerr << "CUDA search setup failed: " << cudaDiagnostic << '\n';
        return 1;
    }
    const auto cudaBaseline = cuda->EvaluateBaseline();
    const auto cudaBatch = cuda->RunBatch(1u, batchSize, false);
    const auto cudaCancelled = cuda->RunBatch(2048u, batchSize, true);
    const bool cudaConditionTimes = cuda->UpdateConditionTimes(2.0, 3.0);
    cuda.reset();
    CudaSearchExecutorConfiguration falseConfig = cudaConfig;
    falseConfig.condition->instructions.front().x = 0.0;
    auto cudaFalse = CudaSearchExecutor::Create(falseConfig, &cudaDiagnostic);
    if (!cudaFalse) {
        std::cerr << "CUDA false-condition setup failed: "
                  << cudaDiagnostic << '\n';
        return 1;
    }
    const auto cudaFalseBaseline = cudaFalse->EvaluateBaseline();
    const auto cudaRejected = cudaFalse->RunBatch(1u, batchSize, false);
    cudaFalse.reset();

    auto hip = HipSearchExecutor::Create(
            hip_bridge::ToHipConfiguration(cudaConfig), &hipDiagnostic);
    if (!hip) {
        std::cerr << "HIP search setup failed: " << hipDiagnostic << '\n';
        return 1;
    }
    const auto hipBaseline = hip_bridge::ToCudaResult(hip->EvaluateBaseline());
    const auto hipBatch = hip_bridge::ToCudaResult(
            hip->RunBatch(1u, batchSize, false));
    const auto hipCancelled = hip_bridge::ToCudaResult(
            hip->RunBatch(2048u, batchSize, true));
    const bool hipConditionTimes = hip->UpdateConditionTimes(2.0, 3.0);
    hip.reset();
    auto hipFalse = HipSearchExecutor::Create(
            hip_bridge::ToHipConfiguration(falseConfig), &hipDiagnostic);
    if (!hipFalse) {
        std::cerr << "HIP false-condition setup failed: "
                  << hipDiagnostic << '\n';
        return 1;
    }
    const auto hipFalseBaseline = hip_bridge::ToCudaResult(
            hipFalse->EvaluateBaseline());
    const auto hipRejected = hip_bridge::ToCudaResult(
            hipFalse->RunBatch(1u, batchSize, false));
    hipFalse.reset();

    if (!SameResult(cudaBaseline, hipBaseline) ||
        !SameResult(cudaBatch, hipBatch) ||
        cudaBatch.status != CudaSearchStatus::Success ||
        cudaBatch.evaluatedCandidateCount != batchSize ||
        cudaBatch.totalMutationCount == 0u ||
        !cudaBatch.best.valid || !cudaBatch.best.stateCaptured ||
        cudaBatch.best.inputs.size() != 2u) {
        std::cerr << "CUDA/HIP search differed: baseline "
                  << CudaSearchStatusName(cudaBaseline.status) << "/"
                  << CudaSearchStatusName(hipBaseline.status) << " batch "
                  << CudaSearchStatusName(cudaBatch.status) << "/"
                  << CudaSearchStatusName(hipBatch.status) << " diagnostics "
                  << cudaBaseline.diagnostic << " / "
                  << hipBaseline.diagnostic << " batch "
                  << cudaBatch.diagnostic << " / " << hipBatch.diagnostic
                  << '\n';
        return 1;
    }
    if (!SameResult(cudaCancelled, hipCancelled) ||
        cudaCancelled.status != CudaSearchStatus::Cancelled ||
        !cudaConditionTimes || !hipConditionTimes) {
        std::cerr << "CUDA/HIP cancellation or condition-time update differed\n";
        return 1;
    }
    if (!SameResult(cudaFalseBaseline, hipFalseBaseline)) {
        std::cerr << "CUDA/HIP false-condition setup differed\n";
        return 1;
    }
    if (!SameResult(cudaRejected, hipRejected) ||
        cudaRejected.status != CudaSearchStatus::Success ||
        cudaRejected.best.valid) {
        std::cerr << "CUDA/HIP false-condition rejection differed\n";
        return 1;
    }
    if (benchmark) {
        using Clock = std::chrono::steady_clock;
        std::vector<double> cudaMilliseconds;
        std::vector<double> hipMilliseconds;
        int cudaRepetitions = 1;
        int hipRepetitions = 1;
        const auto measureCuda = [&](int pass) {
            cuda = CudaSearchExecutor::Create(cudaConfig, &cudaDiagnostic);
            if (!cuda) {
                std::cerr << "CUDA benchmark setup failed: "
                          << cudaDiagnostic << '\n';
                return false;
            }
            const auto baseline = cuda->EvaluateBaseline();
            if (baseline.status != CudaSearchStatus::Success) {
                std::cerr << "CUDA benchmark baseline failed: "
                          << baseline.diagnostic << '\n';
                return false;
            }
            const auto start = Clock::now();
            for (int repeat = 0; repeat < cudaRepetitions; ++repeat) {
                const auto result = cuda->RunBatch(
                        4096u, batchSize, false);
                if (result.status != CudaSearchStatus::Success) {
                    std::cerr << "CUDA benchmark batch failed: "
                              << result.diagnostic << '\n';
                    return false;
                }
            }
            const double elapsed =
                    std::chrono::duration<double, std::milli>(
                            Clock::now() - start).count();
            if (pass == 0) {
                cudaRepetitions = static_cast<int>(std::clamp(
                        250.0 / elapsed, 1.0, 64.0));
            }
            if (pass >= 2) {
                cudaMilliseconds.push_back(elapsed / cudaRepetitions);
            }
            cuda.reset();
            return true;
        };
        const auto measureHip = [&](int pass) {
            hip = HipSearchExecutor::Create(
                    hip_bridge::ToHipConfiguration(cudaConfig), &hipDiagnostic);
            if (!hip) {
                std::cerr << "HIP benchmark setup failed: "
                          << hipDiagnostic << '\n';
                return false;
            }
            const auto baseline = hip->EvaluateBaseline();
            if (baseline.status != HipSearchStatus::Success) {
                std::cerr << "HIP benchmark baseline failed: "
                          << baseline.diagnostic << '\n';
                return false;
            }
            const auto start = Clock::now();
            for (int repeat = 0; repeat < hipRepetitions; ++repeat) {
                const auto result = hip->RunBatch(
                        4096u, batchSize, false);
                if (result.status != HipSearchStatus::Success) {
                    std::cerr << "HIP benchmark batch failed: "
                              << result.diagnostic << '\n';
                    return false;
                }
            }
            const double elapsed =
                    std::chrono::duration<double, std::milli>(
                            Clock::now() - start).count();
            if (pass == 0) {
                hipRepetitions = static_cast<int>(std::clamp(
                        250.0 / elapsed, 1.0, 64.0));
            }
            if (pass >= 2) {
                hipMilliseconds.push_back(elapsed / hipRepetitions);
            }
            hip.reset();
            return true;
        };
        for (int pass = 0; pass < 9; ++pass) {
            const bool hipBeforeCuda = hipFirst != ((pass & 1) != 0);
            if (!(hipBeforeCuda
                          ? (measureHip(pass) && measureCuda(pass))
                          : (measureCuda(pass) && measureHip(pass)))) {
                return 1;
            }
        }
        std::sort(cudaMilliseconds.begin(), cudaMilliseconds.end());
        std::sort(hipMilliseconds.begin(), hipMilliseconds.end());
        const double cudaMedian = cudaMilliseconds[3];
        const double hipMedian = hipMilliseconds[3];
        std::cout << "warmed " << batchSize << " x " << tickCount
                  << " tick median ms CUDA=" << cudaMedian
                  << " HIP=" << hipMedian << " HIP/CUDA throughput="
                  << cudaMedian / hipMedian << '\n';
    }
    hipFree(deviceScene);
    hipFree(deviceConfiguration);
    std::cout << "CUDA/HIP search match: status "
              << CudaSearchStatusName(cudaBatch.status) << " mutations "
              << cudaBatch.totalMutationCount << " evaluated "
              << cudaBatch.evaluatedCandidateCount << " kernel ms "
              << cudaBatch.kernelMilliseconds << "/"
              << hipBatch.kernelMilliseconds << '\n';
    return 0;
}
