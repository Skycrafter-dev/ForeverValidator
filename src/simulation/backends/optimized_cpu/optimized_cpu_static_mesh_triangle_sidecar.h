#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "engine/physics/geometry/gm_surface.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_uniform_grid.h"

struct OptimizedCpuStaticMeshTriangleSidecarTestAccess;

struct OptimizedCpuStaticMeshTriangleData {
    std::array<GmVec3, 3> vertices{};
    std::array<GmVec3, 3> edgeDirections{};
    std::array<GmVec3, 3> edgeNormals{};
    GmVec3 normal{};
    GmVec3 geometricNormal{};
    GmLocalMaterialIndex material;
};

struct OptimizedCpuStaticMeshDirectTrianglePosting {
    GmBoxAligned bounds{};
    u32 triangleIndex = 0u;
};

static_assert(std::is_standard_layout_v<
              OptimizedCpuStaticMeshDirectTrianglePosting>);
static_assert(offsetof(OptimizedCpuStaticMeshDirectTrianglePosting,
                       triangleIndex) == sizeof(GmBoxAligned));
static_assert(sizeof(OptimizedCpuStaticMeshDirectTrianglePosting) ==
              sizeof(GmBoxAligned) + sizeof(u32));
static_assert(alignof(OptimizedCpuStaticMeshDirectTrianglePosting) ==
              alignof(GmBoxAligned));

class OptimizedCpuStaticMeshTriangleSidecar {
public:
    bool TryBuild(const GmSurfMesh &mesh) noexcept;
    void Clear(void) noexcept;
    bool IsFor(const GmSurfMesh &mesh) const noexcept;

    const OptimizedCpuStaticMeshTriangleData &TriangleAt(
            u32 triangleIndex) const noexcept {
        return triangles_[triangleIndex];
    }

    bool DirectCandidateTriangleSpan(
            const GmBoxAligned &query,
            OptimizedCpuStaticUniformGrid::CandidateSpan *result) const
            noexcept {
        return triangleGrid_.DirectCandidateSpan(query, result);
    }

    const OptimizedCpuStaticMeshDirectTrianglePosting &DirectTriangleAt(
            u32 postingIndex) const noexcept {
        return directTrianglePostings_[postingIndex];
    }

    std::size_t DirectTriangleCount(void) const noexcept {
        return directTrianglePostings_.size();
    }

private:
    friend struct OptimizedCpuStaticMeshTriangleSidecarTestAccess;

    const GmSurfMesh *sourceMesh_ = nullptr;
    const GmVec3 *sourceVertices_ = nullptr;
    const GmSurfMeshTriangle *sourceTriangles_ = nullptr;
    const GmMeshOctreeCell *sourceCells_ = nullptr;
    std::size_t sourceVertexCount_ = 0u;
    std::size_t sourceTriangleCount_ = 0u;
    std::size_t sourceCellCount_ = 0u;
    std::vector<OptimizedCpuStaticMeshTriangleData> triangles_;
    std::vector<OptimizedCpuStaticMeshDirectTrianglePosting>
            directTrianglePostings_;
    OptimizedCpuStaticUniformGrid triangleGrid_;
};
