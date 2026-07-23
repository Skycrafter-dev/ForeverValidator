#include "simulation/backends/optimized_cpu/optimized_cpu_static_mesh_triangle_sidecar.h"

#include <cfenv>
#include <cmath>
#include <limits>
#include <new>
#include <utility>

#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

#include "engine/physics/geometry/physics_tolerances.h"
#include "simulation/runtime/replay_deterministic_execution.h"

struct GmSurfMeshStaticInverseOptimizedCpuAccess {
    static const std::vector<GmVec3> &Vertices(const GmSurfMesh &mesh) {
        return mesh.vertices;
    }

    static const std::vector<GmSurfMeshTriangle> &Triangles(
            const GmSurfMesh &mesh) {
        return mesh.triangles;
    }

    static const std::vector<GmMeshOctreeCell> &OctreeCells(
            const GmSurfMesh &mesh) {
        return mesh.octreeCells;
    }
};

namespace {

class FloatingEnvironmentRestore final {
public:
    FloatingEnvironmentRestore() noexcept {
        if (std::fegetenv(&environment_) != 0) {
            return;
        }
#if defined(__i386__) || defined(__x86_64__)
        mxcsr_ = _mm_getcsr();
#endif
        captured_ = true;
    }

    ~FloatingEnvironmentRestore() {
        Restore();
    }

    bool Captured(void) const noexcept {
        return captured_;
    }

    bool Restore(void) noexcept {
        if (!captured_) {
            return false;
        }
        if (restored_) {
            return true;
        }
        bool success = std::fesetenv(&environment_) == 0;
#if defined(__i386__) || defined(__x86_64__)
        _mm_setcsr(mxcsr_);
        success = _mm_getcsr() == mxcsr_ && success;
#endif
        restored_ = success;
        return success;
    }

private:
    std::fenv_t environment_{};
#if defined(__i386__) || defined(__x86_64__)
    unsigned int mxcsr_ = 0u;
#endif
    bool captured_ = false;
    bool restored_ = false;
};

bool HasValidStaticMeshTopology(
        const std::vector<GmVec3> &vertices,
        const std::vector<GmSurfMeshTriangle> &triangles,
        const std::vector<GmMeshOctreeCell> &cells) {
    if (vertices.size() > std::numeric_limits<u32>::max() ||
        triangles.size() > std::numeric_limits<u32>::max() ||
        cells.size() > std::numeric_limits<u32>::max()) {
        return false;
    }
    for (const GmSurfMeshTriangle &triangle : triangles) {
        if (triangle.vertexIndex[0] >= vertices.size() ||
            triangle.vertexIndex[1] >= vertices.size() ||
            triangle.vertexIndex[2] >= vertices.size()) {
            return false;
        }
    }
    for (std::size_t cellIndex = 0u;
         cellIndex < cells.size();
         ++cellIndex) {
        const GmMeshOctreeCell &cell = cells[cellIndex];
        const u32 subtreeEntryCount = cell.SubtreeEntryCount();
        if (subtreeEntryCount == 0u ||
            subtreeEntryCount > cells.size() - cellIndex ||
            (cell.ContainsTriangle() &&
             cell.TriangleIndex() >= triangles.size())) {
            return false;
        }
    }
    return true;
}

bool AxisContains(float parentCenter,
                  float parentHalf,
                  float childCenter,
                  float childHalf) {
    if (!std::isfinite(parentCenter) || !std::isfinite(parentHalf) ||
        !std::isfinite(childCenter) || !std::isfinite(childHalf) ||
        parentHalf < 0.0f || childHalf < 0.0f) {
        return false;
    }
    const double scale = std::fabs(static_cast<double>(parentCenter)) +
                         parentHalf +
                         std::fabs(static_cast<double>(childCenter)) +
                         childHalf + 1.0;
    const double slack = 16.0 * std::numeric_limits<float>::epsilon() * scale;
    return static_cast<double>(parentCenter) - parentHalf - slack <=
                   static_cast<double>(childCenter) - childHalf &&
           static_cast<double>(childCenter) + childHalf <=
                   static_cast<double>(parentCenter) + parentHalf + slack;
}

bool BoundsContain(const GmBoxAligned &parent,
                   const GmBoxAligned &child) {
    return AxisContains(parent.center.x,
                        parent.halfExtents.x,
                        child.center.x,
                        child.halfExtents.x) &&
           AxisContains(parent.center.y,
                        parent.halfExtents.y,
                        child.center.y,
                        child.halfExtents.y) &&
           AxisContains(parent.center.z,
                        parent.halfExtents.z,
                        child.center.z,
                        child.halfExtents.z);
}

bool HasGridCompatibleHierarchy(
        const std::vector<GmMeshOctreeCell> &cells) {
    if (cells.empty() || cells.front().SubtreeEntryCount() != cells.size()) {
        return false;
    }
    struct Parent {
        std::size_t end;
        const GmBoxAligned *bounds;
    };
    std::vector<Parent> parents;
    parents.reserve(32u);
    for (std::size_t cellIndex = 0u;
         cellIndex < cells.size();
         ++cellIndex) {
        while (!parents.empty() && cellIndex == parents.back().end) {
            parents.pop_back();
        }
        if (!parents.empty() &&
            !BoundsContain(*parents.back().bounds,
                           cells[cellIndex].Bounds())) {
            return false;
        }
        const u32 subtreeCount = cells[cellIndex].SubtreeEntryCount();
        if (cells[cellIndex].ContainsTriangle() && subtreeCount != 1u) {
            return false;
        }
        if (subtreeCount > 1u) {
            const std::size_t end = cellIndex + subtreeCount;
            if (end > cells.size() ||
                (!parents.empty() && end > parents.back().end)) {
                return false;
            }
            parents.push_back({end, &cells[cellIndex].Bounds()});
        }
    }
    while (!parents.empty() && parents.back().end == cells.size()) {
        parents.pop_back();
    }
    return parents.empty();
}

}  // namespace

bool OptimizedCpuStaticMeshTriangleSidecar::TryBuild(
        const GmSurfMesh &mesh) noexcept {
    if (!tmnf::simulation::DeterministicExecutionScope::IsActive()) {
        Clear();
        return false;
    }

    FloatingEnvironmentRestore floatingEnvironment;
    if (!floatingEnvironment.Captured()) {
        Clear();
        return false;
    }

    try {
        const std::vector<GmVec3> &vertices =
                GmSurfMeshStaticInverseOptimizedCpuAccess::Vertices(mesh);
        const std::vector<GmSurfMeshTriangle> &triangles =
                GmSurfMeshStaticInverseOptimizedCpuAccess::Triangles(mesh);
        const std::vector<GmMeshOctreeCell> &cells =
                GmSurfMeshStaticInverseOptimizedCpuAccess::OctreeCells(mesh);
        if (!HasValidStaticMeshTopology(vertices, triangles, cells)) {
            Clear();
            return false;
        }

        OptimizedCpuStaticMeshTriangleSidecar rebuilt;
        rebuilt.sourceMesh_ = &mesh;
        rebuilt.sourceVertices_ = vertices.data();
        rebuilt.sourceTriangles_ = triangles.data();
        rebuilt.sourceCells_ = cells.data();
        rebuilt.sourceVertexCount_ = vertices.size();
        rebuilt.sourceTriangleCount_ = triangles.size();
        rebuilt.sourceCellCount_ = cells.size();
        if (triangles.size() > rebuilt.triangles_.max_size()) {
            Clear();
            return false;
        }
        rebuilt.triangles_.resize(triangles.size());

        std::vector<OptimizedCpuStaticUniformGrid::Entry> gridEntries;
        gridEntries.reserve(triangles.size());
        rebuilt.directTrianglePostings_.reserve(triangles.size());
        for (std::size_t cellIndex = 0u;
             cellIndex < cells.size();
             ++cellIndex) {
            if (cells[cellIndex].ContainsTriangle()) {
                const u32 postingIndex = static_cast<u32>(
                        rebuilt.directTrianglePostings_.size());
                rebuilt.directTrianglePostings_.push_back({
                    cells[cellIndex].Bounds(),
                    cells[cellIndex].TriangleIndex(),
                });
                gridEntries.push_back({
                    postingIndex,
                    cells[cellIndex].Bounds(),
                });
            }
        }
        if (HasGridCompatibleHierarchy(cells)) {
            rebuilt.triangleGrid_.TryBuild(
                    gridEntries,
                    rebuilt.directTrianglePostings_.size(),
                    65536u);
        }

        for (std::size_t triangleIndex = 0u;
             triangleIndex < triangles.size();
             ++triangleIndex) {
            const GmSurfMeshTriangle &source = triangles[triangleIndex];
            OptimizedCpuStaticMeshTriangleData &cached =
                    rebuilt.triangles_[triangleIndex];
            cached.vertices = {
                vertices[source.vertexIndex[0]],
                vertices[source.vertexIndex[1]],
                vertices[source.vertexIndex[2]],
            };
            cached.normal = source.normal;
            cached.material = source.material;
            const GmVec3 edge01 =
                    cached.vertices[1].SubtractForCollision(
                            cached.vertices[0]);
            const GmVec3 edge02 =
                    cached.vertices[2].SubtractForCollision(
                            cached.vertices[0]);
            cached.geometricNormal = {
                edge02.z * edge01.y - edge02.y * edge01.z,
                edge01.z * edge02.x - edge02.z * edge01.x,
                edge01.x * edge02.y - edge02.x * edge01.y,
            };
            for (u32 edgeIndex = 0u; edgeIndex < 3u; ++edgeIndex) {
                const u32 nextIndex = edgeIndex == 2u
                        ? 0u
                        : edgeIndex + 1u;
                cached.edgeDirections[edgeIndex] =
                        cached.vertices[nextIndex]
                                .SubtractForCollision(
                                        cached.vertices[edgeIndex])
                                .NormalizeForCollision(
                                        PhysicsTolerance::
                                                SurfaceDirectionLengthSquared);
                cached.edgeNormals[edgeIndex] =
                        cached.edgeDirections[edgeIndex].Cross(cached.normal);
            }
        }

        *this = std::move(rebuilt);
        if (!floatingEnvironment.Restore()) {
            Clear();
            return false;
        }
        return true;
    } catch (const std::bad_alloc &) {
        Clear();
        return false;
    }
}

void OptimizedCpuStaticMeshTriangleSidecar::Clear(void) noexcept {
    sourceMesh_ = nullptr;
    sourceVertices_ = nullptr;
    sourceTriangles_ = nullptr;
    sourceCells_ = nullptr;
    sourceVertexCount_ = 0u;
    sourceTriangleCount_ = 0u;
    sourceCellCount_ = 0u;
    triangles_.clear();
    directTrianglePostings_.clear();
    triangleGrid_.Clear();
}

bool OptimizedCpuStaticMeshTriangleSidecar::IsFor(
        const GmSurfMesh &mesh) const noexcept {
    const std::vector<GmVec3> &vertices =
            GmSurfMeshStaticInverseOptimizedCpuAccess::Vertices(mesh);
    const std::vector<GmSurfMeshTriangle> &triangles =
            GmSurfMeshStaticInverseOptimizedCpuAccess::Triangles(mesh);
    const std::vector<GmMeshOctreeCell> &cells =
            GmSurfMeshStaticInverseOptimizedCpuAccess::OctreeCells(mesh);
    return sourceMesh_ == &mesh &&
           sourceVertices_ == vertices.data() &&
           sourceTriangles_ == triangles.data() &&
           sourceCells_ == cells.data() &&
           sourceVertexCount_ == vertices.size() &&
           sourceTriangleCount_ == triangles.size() &&
           sourceCellCount_ == cells.size() &&
           triangles_.size() == triangles.size();
}
