#include "simulation/backends/cuda/cuda_scene_storage.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/backends/cuda/cuda_timeline_executor.h"
#include "simulation/backends/hip/generated/hip_timeline_executor.h"
#include "simulation/backends/hip/hip_state_bridge.h"

#include <hip/hip_runtime_api.h>

#include <cstring>
#include <iostream>

int main() {
    using namespace forevervalidator::simulation;
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
        std::cerr << "device fixture allocation failed\n";
        return 1;
    }

    ReplayControlTick sourceTick;
    sourceTick.periodMs = 10u;
    sourceTick.timeMs = 1234u;
    sourceTick.controls = {0.25f, 0.75f, -0.5f};
    sourceTick.actions.enableRaceSimulation = true;
    sourceTick.observe = true;
    CudaCandidateTimelineInput cudaInput;
    cudaInput.initialState.candidateId = 17u;
    cudaInput.initialState.firstStep = false;
    for (std::uint32_t index = 0u; index < 100u; ++index) {
        sourceTick.timeMs = 1234u + index * 10u;
        cudaInput.ticks.push_back(FlattenCudaControlTick(sourceTick));
    }
    HipCandidateTimelineInput hipInput;
    hipInput.initialState = hip_bridge::BitCopy<HipCandidateState>(
            cudaInput.initialState);
    for (const CudaControlTick &tick : cudaInput.ticks) {
        hipInput.ticks.push_back(hip_bridge::BitCopy<HipControlTick>(tick));
    }

    const auto cuda = ExecuteCudaTimelineBatch(
            deviceScene, deviceConfiguration, {cudaInput});
    const auto hip = ExecuteHipTimelineBatch(
            deviceScene, deviceConfiguration, {hipInput});
    if (cuda.status != CudaTimelineStatus::Success ||
        hip.status != HipTimelineStatus::Success ||
        cuda.candidates.size() != 1u || hip.candidates.size() != 1u ||
        cuda.candidates[0].status != CudaTimelineStatus::Success ||
        hip.candidates[0].status != HipTimelineStatus::Success) {
        std::cerr << "CUDA/HIP timeline execution failed: "
                  << cuda.diagnostic << " / " << hip.diagnostic << '\n';
        return 1;
    }
    const CudaCandidateState hipState = hip_bridge::BitCopy<CudaCandidateState>(
            hip.candidates[0].finalState);
    if (std::memcmp(&cuda.candidates[0].finalState, &hipState,
                    sizeof(hipState)) != 0 ||
        cuda.candidates[0].observations.size() !=
                hip.candidates[0].observations.size()) {
        std::cerr << "CUDA/HIP per-tick state differs\n";
        return 1;
    }
    for (std::size_t index = 0u;
         index < cuda.candidates[0].observations.size(); ++index) {
        const auto hipObservation =
                hip_bridge::BitCopy<CudaTimelineObservation>(
                        hip.candidates[0].observations[index]);
        if (std::memcmp(&cuda.candidates[0].observations[index],
                        &hipObservation, sizeof(hipObservation)) != 0) {
            std::cerr << "CUDA/HIP per-tick observation differs\n";
            return 1;
        }
    }
    CudaCandidateTimelineInput cudaFirst = cudaInput;
    HipCandidateTimelineInput hipFirst = hipInput;
    cudaFirst.ticks.resize(50u);
    hipFirst.ticks.resize(50u);
    const auto cudaSegment = ExecuteCudaTimelineBatch(
            deviceScene, deviceConfiguration, {cudaFirst});
    const auto hipSegment = ExecuteHipTimelineBatch(
            deviceScene, deviceConfiguration, {hipFirst});
    if (cudaSegment.status != CudaTimelineStatus::Success ||
        hipSegment.status != HipTimelineStatus::Success ||
        cudaSegment.candidates.size() != 1u ||
        hipSegment.candidates.size() != 1u) {
        std::cerr << "CUDA/HIP timeline split failed\n";
        return 1;
    }
    CudaCandidateTimelineInput cudaRestored;
    HipCandidateTimelineInput hipRestored;
    cudaRestored.initialState = cudaSegment.candidates[0].finalState;
    hipRestored.initialState = hipSegment.candidates[0].finalState;
    cudaRestored.ticks.assign(
            cudaInput.ticks.begin() + 50u, cudaInput.ticks.end());
    hipRestored.ticks.assign(
            hipInput.ticks.begin() + 50u, hipInput.ticks.end());
    const auto cudaContinuation = ExecuteCudaTimelineBatch(
            deviceScene, deviceConfiguration, {cudaRestored});
    const auto hipContinuation = ExecuteHipTimelineBatch(
            deviceScene, deviceConfiguration, {hipRestored});
    hipFree(deviceScene);
    hipFree(deviceConfiguration);
    if (cudaContinuation.status != CudaTimelineStatus::Success ||
        hipContinuation.status != HipTimelineStatus::Success ||
        cudaContinuation.candidates.size() != 1u ||
        hipContinuation.candidates.size() != 1u ||
        std::memcmp(&cuda.candidates[0].finalState,
                    &cudaContinuation.candidates[0].finalState,
                    sizeof(CudaCandidateState)) != 0 ||
        std::memcmp(&hip.candidates[0].finalState,
                    &hipContinuation.candidates[0].finalState,
                    sizeof(HipCandidateState)) != 0) {
        std::cerr << "CUDA/HIP restored timeline differs from continuous run\n";
        return 1;
    }
    std::cout << "CUDA/HIP per-tick state and observations are bit-exact\n";
    return 0;
}
