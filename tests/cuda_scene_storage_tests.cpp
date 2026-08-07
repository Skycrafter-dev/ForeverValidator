#include "simulation/backends/cuda/cuda_scene_storage.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

class Hash {
public:
    template<typename T>
    void Add(const T &value) {
        const auto *bytes =
                reinterpret_cast<const unsigned char *>(&value);
        for (std::size_t index = 0u; index < sizeof(value); ++index) {
            value_ ^= bytes[index];
            value_ *= UINT64_C(1099511628211);
        }
    }

    template<typename T>
    void Add(const std::vector<T> &values) {
        Add(values.size());
        for (const T &value : values) Add(value);
    }

    std::uint64_t Value() const { return value_; }

private:
    std::uint64_t value_ = UINT64_C(1469598103934665603);
};

std::uint64_t SceneHash(
        const forevervalidator::simulation::CudaHostScene &scene) {
    Hash hash;
    hash.Add(scene.schemaVersion);
    hash.Add(scene.actors);
    hash.Add(scene.surfaces);
    hash.Add(scene.materials);
    hash.Add(scene.vertices);
    hash.Add(scene.triangles);
    hash.Add(scene.octreeCells);
    hash.Add(scene.accelerationGroups);
    hash.Add(scene.accelerationCells);
    return hash.Value();
}

}  // namespace

int main() {
    using namespace forevervalidator::simulation;

    static_assert(CudaHostScene::SchemaVersion == 3u);
    static_assert(CudaPackedSceneHeader::SchemaVersion == 2u);

    CudaHostScene source;
    source.actors.resize(1u);
    source.surfaces.resize(1u);
    source.vertices = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}};
    source.triangles.resize(1u);
    source.triangles[0].vertexIndices[0] = 0u;
    source.triangles[0].vertexIndices[1] = 1u;
    source.triangles[0].vertexIndices[2] = 2u;
    std::memcpy(
            source.triangles[0].vertices,
            source.vertices.data(),
            sizeof(source.triangles[0].vertices));
    source.octreeCells.resize(1u);
    source.octreeCells[0].containsTriangle = 1u;
    CudaSceneSurface &surface = source.surfaces[0];
    surface.vertexCount = 3u;
    surface.triangleCount = 1u;
    surface.octreeCellCount = 1u;
    for (std::size_t index = 0u;
         index < source.accelerationGroups.size(); ++index) {
        CudaSceneAccelerationCell cell;
        cell.surfaceIndex = 0u;
        source.accelerationGroups[index].firstCell =
                static_cast<std::uint32_t>(
                        source.accelerationCells.size());
        source.accelerationGroups[index].cellCount = 1u;
        source.accelerationCells.push_back(cell);
    }
    source.deterministicHash = SceneHash(source);

    std::vector<std::byte> bytes;
    CudaPackedSceneHeader packed;
    if (!source.Valid() ||
        !PackCudaScene(source, &bytes, &packed) ||
        !ValidCudaPackedSceneHeader(packed) ||
        packed.headerSize != sizeof(CudaPackedSceneHeader) ||
        packed.totalSize != bytes.size() ||
        packed.triangles.count != 1u ||
        packed.triangles.stride != sizeof(CudaSceneTriangle)) {
        std::cerr << "CUDA scene packing did not publish its exact ABI\n";
        return 1;
    }

    CudaPackedSceneHeader corrupt = packed;
    corrupt.triangles.stride = sizeof(CudaSceneTriangle) - 1u;
    if (ValidCudaPackedSceneHeader(corrupt)) {
        std::cerr << "corrupt CUDA triangle stride was accepted\n";
        return 1;
    }
    corrupt = packed;
    --corrupt.headerSize;
    if (ValidCudaPackedSceneHeader(corrupt)) {
        std::cerr << "corrupt CUDA scene header size was accepted\n";
        return 1;
    }
    corrupt = packed;
    corrupt.triangles.offset = corrupt.totalSize;
    if (ValidCudaPackedSceneHeader(corrupt)) {
        std::cerr << "out-of-range CUDA triangle section was accepted\n";
        return 1;
    }
    return 0;
}
