#ifndef FOREVERVALIDATOR_CUDA_SESSION_SPECIALIZATION_H
#define FOREVERVALIDATOR_CUDA_SESSION_SPECIALIZATION_H

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda.h>

#include "simulation/backends/cuda/cuda_scene_storage.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"

namespace forevervalidator::simulation::cuda::specialization {

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
            const CudaPackedStaticConfigurationHeader &configuration,
            std::uint64_t configurationBase,
            const CudaPackedSceneHeader &scene,
            std::uint64_t sceneBase,
            std::string *diagnostic);
    bool Ready() const noexcept;
    CUfunction Kernel(
            std::uint32_t minimumBlocksPerMultiprocessor,
            bool useEmptyAirCertificate) const noexcept;
    const KernelMetrics &Metrics(
            std::uint32_t minimumBlocksPerMultiprocessor,
            bool useEmptyAirCertificate) const noexcept;

private:
    struct KernelEntry {
        CUfunction function = nullptr;
        KernelMetrics metrics;
    };

    void Reset() noexcept;

    CUmodule module_ = nullptr;
    KernelEntry throughput_[2]{};
    KernelEntry tail_[2]{};
    KernelEntry denseTail_[2]{};
};

}  // namespace forevervalidator::simulation::cuda::specialization

#endif
