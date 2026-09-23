// Generated from src/simulation/backends/cuda/cuda_session_specialization.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_SESSION_SPECIALIZATION_H
#define FOREVERVALIDATOR_HIP_SESSION_SPECIALIZATION_H

#include <cstddef>
#include <cstdint>
#include <string>

#include <hip/hip_runtime.h>

#include "simulation/backends/hip/generated/hip_scene_storage.h"
#include "simulation/backends/hip/generated/hip_static_configuration.h"

namespace forevervalidator::simulation::hip::specialization {

std::uint64_t SessionModuleBuildCountForTesting() noexcept;

struct KernelMetrics {
    std::uint32_t registersPerThread = 0u;
    std::uint64_t localBytesPerThread = 0u;
    std::uint32_t activeBlocksPerMultiprocessor = 0u;
};

class SessionModule {
public:
    SessionModule() = default;
    ~SessionModule();
    SessionModule(const SessionModule &) = delete;
    SessionModule &operator=(const SessionModule &) = delete;

    bool Build(
            const HipPackedStaticConfigurationHeader &configuration,
            std::uint64_t configurationBase,
            const HipPackedSceneHeader &scene,
            std::uint64_t sceneBase,
            std::string *diagnostic);
    bool Ready() const noexcept;
    hipFunction_t Kernel(
            std::uint32_t minimumBlocksPerMultiprocessor) const noexcept;
    const KernelMetrics &Metrics(
            std::uint32_t minimumBlocksPerMultiprocessor) const noexcept;

private:
    struct KernelEntry {
        hipFunction_t function = nullptr;
        KernelMetrics metrics;
    };

    void Reset() noexcept;

    hipModule_t module_ = nullptr;
    KernelEntry throughput_;
    KernelEntry tail_;
    KernelEntry denseTail_;
};

}  // namespace forevervalidator::simulation::hip::specialization

#endif
