#include "simulation/backends/optimized_cpu/optimized_cpu_static_surface_transform_cache.h"

#include <cstdint>
#include <cstring>
#include <typeinfo>

#include "engine/physics/geometry/plug_surface.h"

namespace {

class FingerprintHasher {
public:
    void AddBool(bool value) noexcept {
        AddByte(value ? 1u : 0u);
    }

    void AddU16(std::uint16_t value) noexcept {
        AddUnsigned(value, sizeof(value));
    }

    void AddU32(std::uint32_t value) noexcept {
        AddUnsigned(value, sizeof(value));
    }

    void AddU64(std::uint64_t value) noexcept {
        AddUnsigned(value, sizeof(value));
    }

    void AddFloat(float value) noexcept {
        std::uint32_t bits = 0u;
        static_assert(sizeof(bits) == sizeof(value),
                      "binary32 size mismatch");
        std::memcpy(&bits, &value, sizeof(bits));
        AddU32(bits);
    }

    void AddPointer(const void *value) noexcept {
        AddU64(static_cast<std::uint64_t>(
                reinterpret_cast<std::uintptr_t>(value)));
    }

    std::uint64_t Value(void) const noexcept {
        return value_;
    }

private:
    void AddByte(std::uint8_t value) noexcept {
        value_ ^= value;
        value_ *= 1099511628211ull;
    }

    void AddUnsigned(std::uint64_t value, std::size_t byteCount) noexcept {
        for (std::size_t byteIndex = 0u;
             byteIndex < byteCount;
             ++byteIndex) {
            AddByte(static_cast<std::uint8_t>(value & 0xffu));
            value >>= 8u;
        }
    }

    std::uint64_t value_ = 1469598103934665603ull;
};

void HashVector(FingerprintHasher &hash, const GmVec3 &value) noexcept {
    hash.AddFloat(value.x);
    hash.AddFloat(value.y);
    hash.AddFloat(value.z);
}

void HashBox(FingerprintHasher &hash,
             const GmBoxAligned &value) noexcept {
    HashVector(hash, value.center);
    HashVector(hash, value.halfExtents);
}

void HashTransform(FingerprintHasher &hash,
                   const GmIso4 &value) noexcept {
    HashVector(hash, value.rotation.basisX);
    HashVector(hash, value.rotation.basisY);
    HashVector(hash, value.rotation.basisZ);
    HashVector(hash, value.translation);
}

}  // namespace

std::optional<OptimizedCpuStaticSceneFingerprint>
OptimizedCpuStaticSurfaceTransformCache::
        CaptureSourceFingerprintForTesting(void) const noexcept {
    if (sourceZone_ == nullptr) {
        return std::nullopt;
    }

    FingerprintHasher identity;
    FingerprintHasher staticTreeRecords;
    FingerprintHasher staticTransforms;
    FingerprintHasher meshTriangles;
    OptimizedCpuStaticSceneFingerprint result;
    identity.AddPointer(sourceZone_);

    for (u32 groupIndex = 0u;
         groupIndex < CHmsCollisionManager_GroupCount;
         ++groupIndex) {
        const CHmsCollisionManagerSGroup *group =
                sourceZone_->GroupAtOrNull(groupIndex);
        identity.AddU32(groupIndex);
        identity.AddPointer(group);
        staticTreeRecords.AddU32(groupIndex);
        staticTreeRecords.AddBool(group != nullptr);
        if (group == nullptr) {
            continue;
        }

        ++result.groupCount;
        const auto &records = group->staticTrees.Entries();
        identity.AddPointer(records.data());
        identity.AddU64(records.size());
        staticTreeRecords.AddU64(records.size());
        result.staticTreeRecordCount += records.size();

        for (std::size_t recordIndex = 0u;
             recordIndex < records.size();
             ++recordIndex) {
            const CHmsCollisionManagerSColOctreeCell &record =
                    records[recordIndex];
            const bool containsSurface = record.ContainsSurface();
            identity.AddPointer(&record);
            staticTreeRecords.AddU64(recordIndex);
            HashBox(staticTreeRecords, record.Bounds());
            staticTreeRecords.AddU32(record.SubtreeEntryCount());
            staticTreeRecords.AddBool(containsSurface);
            if (!containsSurface) {
                continue;
            }

            ++result.staticSurfaceCount;
            const CHmsCollisionManagerSColOctreeCell::StaticSurface
                    &surface = record.SurfaceData();
            identity.AddPointer(surface.surface);
            identity.AddPointer(surface.tree);
            identity.AddPointer(surface.corpus);
            staticTransforms.AddU32(groupIndex);
            staticTransforms.AddU64(recordIndex);
            HashTransform(staticTransforms, surface.location);

            const GmSurf *geometry = surface.surface == nullptr
                    ? nullptr
                    : surface.surface->Geometry();
            identity.AddPointer(geometry);
            const bool isExactMesh = geometry != nullptr &&
                    typeid(*geometry) == typeid(GmSurfMesh);
            staticTreeRecords.AddBool(isExactMesh);
            if (!isExactMesh) {
                continue;
            }

            const GmSurfMesh &mesh =
                    static_cast<const GmSurfMesh &>(*geometry);
            const u32 vertexCount = mesh.VertexCount();
            const u32 triangleCount = mesh.TriangleCount();
            const u32 cellCount = mesh.OctreeCellCount();
            ++result.meshSurfaceCount;
            result.meshVertexCount += vertexCount;
            result.meshTriangleCount += triangleCount;
            result.meshOctreeCellCount += cellCount;

            identity.AddU32(vertexCount);
            identity.AddPointer(vertexCount == 0u
                    ? nullptr
                    : &mesh.Vertex(0u));
            identity.AddU32(triangleCount);
            identity.AddPointer(triangleCount == 0u
                    ? nullptr
                    : &mesh.Triangle(0u));
            identity.AddU32(cellCount);
            identity.AddPointer(cellCount == 0u
                    ? nullptr
                    : &mesh.OctreeCell(0u));

            meshTriangles.AddU32(groupIndex);
            meshTriangles.AddU64(recordIndex);
            meshTriangles.AddU16(mesh.material.Index());
            meshTriangles.AddU32(vertexCount);
            for (u32 vertexIndex = 0u;
                 vertexIndex < vertexCount;
                 ++vertexIndex) {
                HashVector(meshTriangles, mesh.Vertex(vertexIndex));
            }
            meshTriangles.AddU32(triangleCount);
            for (u32 triangleIndex = 0u;
                 triangleIndex < triangleCount;
                 ++triangleIndex) {
                const GmSurfMeshTriangle &triangle =
                        mesh.Triangle(triangleIndex);
                HashVector(meshTriangles, triangle.normal);
                meshTriangles.AddFloat(triangle.planeDistance);
                meshTriangles.AddU32(triangle.vertexIndex[0]);
                meshTriangles.AddU32(triangle.vertexIndex[1]);
                meshTriangles.AddU32(triangle.vertexIndex[2]);
                meshTriangles.AddU16(triangle.material.Index());
            }
            meshTriangles.AddU32(cellCount);
            for (u32 cellIndex = 0u;
                 cellIndex < cellCount;
                 ++cellIndex) {
                const GmMeshOctreeCell &cell =
                        mesh.OctreeCell(cellIndex);
                HashBox(meshTriangles, cell.Bounds());
                meshTriangles.AddU32(cell.SubtreeEntryCount());
                meshTriangles.AddBool(cell.ContainsTriangle());
                if (cell.ContainsTriangle()) {
                    meshTriangles.AddU32(cell.TriangleIndex());
                }
            }
        }
    }

    result.identityHash = identity.Value();
    result.staticTreeRecordHash = staticTreeRecords.Value();
    result.staticTransformHash = staticTransforms.Value();
    result.meshTriangleHash = meshTriangles.Value();
    return result;
}
