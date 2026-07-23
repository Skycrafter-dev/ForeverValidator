#pragma once

#include <cstdint>

// Cold diagnostic data used to prove that the map-owned collision scene stays
// sealed for the lifetime of an OptimizedCpu runtime cache.
struct OptimizedCpuStaticSceneFingerprint {
    std::uint64_t identityHash = 0u;
    std::uint64_t staticTreeRecordHash = 0u;
    std::uint64_t staticTransformHash = 0u;
    std::uint64_t meshTriangleHash = 0u;
    std::uint64_t groupCount = 0u;
    std::uint64_t staticTreeRecordCount = 0u;
    std::uint64_t staticSurfaceCount = 0u;
    std::uint64_t meshSurfaceCount = 0u;
    std::uint64_t meshVertexCount = 0u;
    std::uint64_t meshTriangleCount = 0u;
    std::uint64_t meshOctreeCellCount = 0u;

    bool operator==(
            const OptimizedCpuStaticSceneFingerprint &other) const noexcept {
        return identityHash == other.identityHash &&
               staticTreeRecordHash == other.staticTreeRecordHash &&
               staticTransformHash == other.staticTransformHash &&
               meshTriangleHash == other.meshTriangleHash &&
               groupCount == other.groupCount &&
               staticTreeRecordCount == other.staticTreeRecordCount &&
               staticSurfaceCount == other.staticSurfaceCount &&
               meshSurfaceCount == other.meshSurfaceCount &&
               meshVertexCount == other.meshVertexCount &&
               meshTriangleCount == other.meshTriangleCount &&
               meshOctreeCellCount == other.meshOctreeCellCount;
    }

    bool operator!=(
            const OptimizedCpuStaticSceneFingerprint &other) const noexcept {
        return !(*this == other);
    }
};
