// Generated from src/simulation/backends/cuda/cuda_scene_storage.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_SCENE_STORAGE_H
#define FOREVERVALIDATOR_HIP_SCENE_STORAGE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "simulation/backends/hip/generated/hip_scene_layout.h"

namespace forevervalidator::simulation {

struct HipSceneSection {
    std::uint64_t offset = 0u;
    std::uint32_t count = 0u;
    std::uint32_t stride = 0u;
};

struct HipPackedSceneHeader {
    static constexpr std::uint32_t SchemaVersion = 1u;
    static constexpr std::uint64_t Magic = 0x4656435544415343ull;

    std::uint64_t magic = Magic;
    std::uint32_t schemaVersion = SchemaVersion;
    std::uint32_t headerSize = sizeof(HipPackedSceneHeader);
    std::uint64_t totalSize = 0u;
    std::uint64_t deterministicHash = 0u;
    HipSceneSection actors{};
    HipSceneSection surfaces{};
    HipSceneSection materials{};
    HipSceneSection vertices{};
    HipSceneSection triangles{};
    HipSceneSection octreeCells{};
    HipSceneAccelerationRange accelerationGroups[5]{};
    HipSceneSection accelerationCells{};
};

#if defined(__HIPCC__) && \
        defined(FOREVERVALIDATOR_HIP_RESEARCH_CONSTANT_SCENE)
namespace hip::research {

__device__ __constant__ HipPackedSceneHeader StaticScene;
__device__ __constant__ std::uint64_t StaticSceneBase;

}  // namespace hip::research
#endif

#if defined(__HIPCC__) && \
        defined(FOREVERVALIDATOR_HIP_RESEARCH_SESSION_LTO)
namespace hip::research {

extern "C" __device__ std::uint64_t
ForeverValidatorSessionSceneBase();
extern "C" __device__ const unsigned char *
ForeverValidatorSessionSceneBytes();
extern __device__ __constant__ HipPackedSceneHeader StaticScene;
extern __device__ __constant__ std::uint64_t StaticSceneBase;

__device__ inline std::uint64_t SessionSceneBase() {
    return ForeverValidatorSessionSceneBase();
}

}  // namespace hip::research
#endif

struct HipSceneTransferMetrics {
    bool success = false;
    std::uint64_t hostPackedBytes = 0u;
    std::uint64_t deviceBytes = 0u;
    double packMilliseconds = 0.0;
    double uploadMilliseconds = 0.0;
    std::string diagnostic;
};

bool PackHipScene(
        const HipHostScene &source,
        std::vector<std::byte> *destination,
        HipPackedSceneHeader *header = nullptr) noexcept;

class HipDeviceScene {
public:
    HipDeviceScene() = default;
    ~HipDeviceScene();
    HipDeviceScene(HipDeviceScene &&other) noexcept;
    HipDeviceScene &operator=(HipDeviceScene &&other) noexcept;

    HipDeviceScene(const HipDeviceScene &) = delete;
    HipDeviceScene &operator=(const HipDeviceScene &) = delete;

    HipSceneTransferMetrics Upload(
            const HipHostScene &source) noexcept;
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
