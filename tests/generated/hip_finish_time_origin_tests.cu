// Generated from tests/cuda_finish_time_origin_tests.cu by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_finish_time_origin.cuh"

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <iostream>

namespace {

__global__ void CaptureTickStart(
        std::uint32_t tickTimeMs,
        std::uint64_t *tickStartNs) {
    *tickStartNs =
            forevervalidator::simulation::hip::finish::
                    TickStartNanoseconds(tickTimeMs);
}

}  // namespace

int main() {
    constexpr std::uint32_t TickTimeMs = 29580u;
    constexpr std::uint32_t TickPeriodMs = 10u;
    constexpr std::uint64_t ExpectedTickStartNs = 29580000000u;
    constexpr std::uint64_t RepresentativeFinishNs = 29589583155u;
    static_assert(
            forevervalidator::simulation::hip::finish::
                    TickStartNanoseconds(TickTimeMs) ==
            ExpectedTickStartNs);

    std::uint64_t *deviceTickStartNs = nullptr;
    if (hipMalloc(&deviceTickStartNs, sizeof(*deviceTickStartNs)) !=
        hipSuccess) {
        std::cerr << "HIP finish-origin allocation failed\n";
        return 1;
    }
    CaptureTickStart<<<1, 1>>>(TickTimeMs, deviceTickStartNs);
    std::uint64_t tickStartNs = 0u;
    const hipError_t copyStatus = hipMemcpy(
            &tickStartNs, deviceTickStartNs,
            sizeof(tickStartNs), hipMemcpyDeviceToHost);
    const hipError_t freeStatus = hipFree(deviceTickStartNs);
    if (copyStatus != hipSuccess || freeStatus != hipSuccess) {
        std::cerr << "HIP finish-origin execution failed\n";
        return 1;
    }
    const std::uint64_t tickEndNs =
            tickStartNs +
            static_cast<std::uint64_t>(TickPeriodMs) * 1000000u;
    if (tickStartNs != ExpectedTickStartNs ||
        !(tickStartNs < RepresentativeFinishNs &&
          RepresentativeFinishNs <= tickEndNs)) {
        std::cerr << "HIP finish interval did not start at tick T\n";
        return 1;
    }
    return 0;
}
