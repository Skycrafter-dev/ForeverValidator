// Generated from src/simulation/backends/cuda/cuda_scene_storage.cpp by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_scene_storage.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <utility>

namespace forevervalidator::simulation {

#if FOREVERVALIDATOR_HAS_HIP
bool UploadHipSceneBytes(const std::byte *source,
                          std::size_t size,
                          void **destination,
                          double *milliseconds,
                          std::string *diagnostic) noexcept;
void ReleaseHipSceneBytes(void *allocation) noexcept;
#endif

namespace {

constexpr std::size_t SceneAlignment = 16u;

std::size_t Align(std::size_t value) {
    return (value + SceneAlignment - 1u) &
           ~(SceneAlignment - 1u);
}

template<typename T>
bool AppendSection(const std::vector<T> &source,
                   std::vector<std::byte> &destination,
                   HipSceneSection &section) {
    if (source.size() > std::numeric_limits<std::uint32_t>::max() ||
        source.size() >
                std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return false;
    }
    const std::size_t offset = Align(destination.size());
    const std::size_t bytes = source.size() * sizeof(T);
    if (offset > std::numeric_limits<std::size_t>::max() - bytes) {
        return false;
    }
    destination.resize(offset + bytes);
    if (bytes != 0u) {
        std::memcpy(destination.data() + offset,
                    source.data(), bytes);
    }
    section.offset = offset;
    section.count = static_cast<std::uint32_t>(source.size());
    section.stride = sizeof(T);
    return true;
}

}  // namespace

bool PackHipScene(
        const HipHostScene &source,
        std::vector<std::byte> *destination,
        HipPackedSceneHeader *header) noexcept {
    if (destination == nullptr || !source.Valid()) {
        return false;
    }
    try {
        HipPackedSceneHeader packed;
        packed.deterministicHash = source.deterministicHash;
        std::copy(source.accelerationGroups.begin(),
                  source.accelerationGroups.end(),
                  packed.accelerationGroups);
        std::vector<std::byte> bytes(sizeof(packed));
        if (!AppendSection(source.actors, bytes, packed.actors) ||
            !AppendSection(source.surfaces, bytes, packed.surfaces) ||
            !AppendSection(source.materials, bytes, packed.materials) ||
            !AppendSection(source.vertices, bytes, packed.vertices) ||
            !AppendSection(source.triangles, bytes, packed.triangles) ||
            !AppendSection(source.octreeCells, bytes,
                           packed.octreeCells) ||
            !AppendSection(source.accelerationCells, bytes,
                           packed.accelerationCells)) {
            return false;
        }
        packed.totalSize = bytes.size();
        std::memcpy(bytes.data(), &packed, sizeof(packed));
        *destination = std::move(bytes);
        if (header != nullptr) {
            *header = packed;
        }
        return true;
    } catch (...) {
        return false;
    }
}

HipDeviceScene::~HipDeviceScene() {
    Reset();
}

HipDeviceScene::HipDeviceScene(HipDeviceScene &&other) noexcept
    : deviceData_(std::exchange(other.deviceData_, nullptr)),
      deviceBytes_(std::exchange(other.deviceBytes_, 0u)),
      sceneHash_(std::exchange(other.sceneHash_, 0u)) {}

HipDeviceScene &HipDeviceScene::operator=(
        HipDeviceScene &&other) noexcept {
    if (this != &other) {
        Reset();
        deviceData_ = std::exchange(other.deviceData_, nullptr);
        deviceBytes_ = std::exchange(other.deviceBytes_, 0u);
        sceneHash_ = std::exchange(other.sceneHash_, 0u);
    }
    return *this;
}

HipSceneTransferMetrics HipDeviceScene::Upload(
        const HipHostScene &source) noexcept {
    Reset();
    HipSceneTransferMetrics result;
    const auto packStart = std::chrono::steady_clock::now();
    std::vector<std::byte> bytes;
    if (!PackHipScene(source, &bytes)) {
        result.diagnostic =
                "HIP scene packing failed or exceeded layout bounds";
        return result;
    }
    const auto packEnd = std::chrono::steady_clock::now();
    result.packMilliseconds =
            std::chrono::duration<double, std::milli>(
                    packEnd - packStart).count();
    result.hostPackedBytes = bytes.size();
#if FOREVERVALIDATOR_HAS_HIP
    if (!UploadHipSceneBytes(
                bytes.data(), bytes.size(), &deviceData_,
                &result.uploadMilliseconds, &result.diagnostic)) {
        Reset();
        return result;
    }
    deviceBytes_ = bytes.size();
    sceneHash_ = source.deterministicHash;
    result.deviceBytes = deviceBytes_;
    result.success = true;
    result.diagnostic = "HIP immutable scene uploaded";
#else
    result.diagnostic =
            "HIP scene upload unavailable in a CPU-only build";
#endif
    return result;
}

void HipDeviceScene::Reset() noexcept {
#if FOREVERVALIDATOR_HAS_HIP
    ReleaseHipSceneBytes(deviceData_);
#endif
    deviceData_ = nullptr;
    deviceBytes_ = 0u;
    sceneHash_ = 0u;
}

}  // namespace forevervalidator::simulation
