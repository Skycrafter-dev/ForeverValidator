#ifndef FOREVERVALIDATOR_CUDA_SCENE_STORAGE_H
#define FOREVERVALIDATOR_CUDA_SCENE_STORAGE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "simulation/backends/cuda/cuda_scene_layout.h"

namespace forevervalidator::simulation {

struct CudaSceneSection {
    std::uint64_t offset = 0u;
    std::uint32_t count = 0u;
    std::uint32_t stride = 0u;
};

struct CudaPackedSceneHeader {
    static constexpr std::uint32_t SchemaVersion = 2u;
    static constexpr std::uint64_t Magic = 0x4656435544415343ull;

    std::uint64_t magic = Magic;
    std::uint32_t schemaVersion = SchemaVersion;
    std::uint32_t headerSize = sizeof(CudaPackedSceneHeader);
    std::uint64_t totalSize = 0u;
    std::uint64_t deterministicHash = 0u;
    CudaSceneSection actors{};
    CudaSceneSection surfaces{};
    CudaSceneSection materials{};
    CudaSceneSection vertices{};
    CudaSceneSection triangles{};
    CudaSceneSection octreeCells{};
    CudaSceneAccelerationRange accelerationGroups[5]{};
    CudaSceneSection accelerationCells{};
};

#if defined(__CUDACC__)
#define FOREVERVALIDATOR_CUDA_SCENE_HD __host__ __device__
#else
#define FOREVERVALIDATOR_CUDA_SCENE_HD
#endif

FOREVERVALIDATOR_CUDA_SCENE_HD inline bool ValidCudaSceneSection(
        const CudaSceneSection &section,
        std::uint64_t totalSize,
        std::uint32_t expectedStride) noexcept {
    if (section.count == 0u) {
        return true;
    }
    return section.stride == expectedStride &&
            section.offset >= sizeof(CudaPackedSceneHeader) &&
            section.offset <= totalSize &&
            static_cast<std::uint64_t>(section.count) <=
                    (totalSize - section.offset) / expectedStride;
}

// Packed scenes cross an untrusted host/device boundary.  Validate every
// section before a kernel derives pointers from its offset and stride.
FOREVERVALIDATOR_CUDA_SCENE_HD inline bool ValidCudaPackedSceneHeader(
        const CudaPackedSceneHeader &scene) noexcept {
    if (scene.magic != CudaPackedSceneHeader::Magic ||
        scene.schemaVersion != CudaPackedSceneHeader::SchemaVersion ||
        scene.headerSize != sizeof(CudaPackedSceneHeader) ||
        scene.totalSize < sizeof(CudaPackedSceneHeader)) {
        return false;
    }
    return ValidCudaSceneSection(
                   scene.actors, scene.totalSize,
                   sizeof(CudaSceneActor)) &&
            ValidCudaSceneSection(
                    scene.surfaces, scene.totalSize,
                    sizeof(CudaSceneSurface)) &&
            ValidCudaSceneSection(
                    scene.materials, scene.totalSize,
                    sizeof(std::uint32_t)) &&
            ValidCudaSceneSection(
                    scene.vertices, scene.totalSize,
                    sizeof(GmVec3)) &&
            ValidCudaSceneSection(
                    scene.triangles, scene.totalSize,
                    sizeof(CudaSceneTriangle)) &&
            ValidCudaSceneSection(
                    scene.octreeCells, scene.totalSize,
                    sizeof(CudaSceneOctreeCell)) &&
            ValidCudaSceneSection(
                    scene.accelerationCells, scene.totalSize,
                    sizeof(CudaSceneAccelerationCell));
}

#undef FOREVERVALIDATOR_CUDA_SCENE_HD

#if defined(__CUDACC__) && \
        defined(FOREVERVALIDATOR_CUDA_RESEARCH_CONSTANT_SCENE)
namespace cuda::research {

__device__ __constant__ CudaPackedSceneHeader StaticScene;
__device__ __constant__ std::uint64_t StaticSceneBase;

}  // namespace cuda::research
#endif

#if defined(__CUDACC__) && \
        defined(FOREVERVALIDATOR_CUDA_RESEARCH_SESSION_LTO)
namespace cuda::research {

extern "C" __device__ std::uint64_t
ForeverValidatorSessionSceneBase();
extern "C" __device__ const unsigned char *
ForeverValidatorSessionSceneBytes();
extern __device__ __constant__ CudaPackedSceneHeader StaticScene;
extern __device__ __constant__ std::uint64_t StaticSceneBase;

__device__ inline std::uint64_t SessionSceneBase() {
    return ForeverValidatorSessionSceneBase();
}

}  // namespace cuda::research
#endif

struct CudaSceneTransferMetrics {
    bool success = false;
    std::uint64_t hostPackedBytes = 0u;
    std::uint64_t deviceBytes = 0u;
    double packMilliseconds = 0.0;
    double uploadMilliseconds = 0.0;
    std::string diagnostic;
};

bool PackCudaScene(
        const CudaHostScene &source,
        std::vector<std::byte> *destination,
        CudaPackedSceneHeader *header = nullptr) noexcept;

class CudaDeviceScene {
public:
    CudaDeviceScene() = default;
    ~CudaDeviceScene();
    CudaDeviceScene(CudaDeviceScene &&other) noexcept;
    CudaDeviceScene &operator=(CudaDeviceScene &&other) noexcept;

    CudaDeviceScene(const CudaDeviceScene &) = delete;
    CudaDeviceScene &operator=(const CudaDeviceScene &) = delete;

    CudaSceneTransferMetrics Upload(
            const CudaHostScene &source) noexcept;
    void Reset() noexcept;
    bool Ready() const noexcept { return deviceData_ != nullptr; }
    std::uint64_t SceneHash() const noexcept { return sceneHash_; }
    std::uint64_t DeviceBytes() const noexcept { return deviceBytes_; }
    const void *DeviceData() const noexcept { return deviceData_; }

private:
    void *deviceData_ = nullptr;
    std::uint64_t deviceBytes_ = 0u;
    std::uint64_t sceneHash_ = 0u;
};

}  // namespace forevervalidator::simulation

#endif
