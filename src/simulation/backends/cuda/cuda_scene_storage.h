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
    static constexpr std::uint32_t SchemaVersion = 1u;
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
