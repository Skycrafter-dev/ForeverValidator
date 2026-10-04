#pragma once

#include "simulation/backends/cuda/cuda_search_executor.h"

#include <cmath>

#if defined(__CUDACC__)
#define FV_VOLUME_HD __host__ __device__
#else
#define FV_VOLUME_HD
#endif

namespace forevervalidator::simulation::cuda::custom_volume {

struct Point {
    double x, y, normal;
};

FV_VOLUME_HD inline Point Project(const CudaSearchEvaluatorConfiguration &volume,
                                 double x, double y, double z) {
    x -= volume.values[0];
    y -= volume.values[1];
    z -= volume.values[2];
    if (volume.optionFlags == 0u) return {x, y, z};
    if (volume.optionFlags == 1u) return {x, z, y};
    return {y, z, x};
}

FV_VOLUME_HD inline bool Contains(const CudaSearchEvaluatorConfiguration &volume,
                                  const Point &point) {
    if (point.normal < 0.0 || point.normal > volume.values[3]) return false;
    bool inside = false;
    for (std::uint32_t i = 0u, previous = volume.customVolumeVertexCount - 1u;
         i < volume.customVolumeVertexCount; previous = i++) {
        const auto &a = volume.customVolumeVertices[previous];
        const auto &b = volume.customVolumeVertices[i];
        const double cross = (point.x - a[0]) * (b[1] - a[1]) -
                (point.y - a[1]) * (b[0] - a[0]);
        if (fabs(cross) <= 1e-9 &&
            point.x >= fmin(a[0], b[0]) - 1e-9 &&
            point.x <= fmax(a[0], b[0]) + 1e-9 &&
            point.y >= fmin(a[1], b[1]) - 1e-9 &&
            point.y <= fmax(a[1], b[1]) + 1e-9) return true;
        if ((a[1] > point.y) != (b[1] > point.y)) {
            const double crossing = (b[0] - a[0]) * (point.y - a[1]) /
                    (b[1] - a[1]) + a[0];
            if (point.x < crossing) inside = !inside;
        }
    }
    return inside;
}

FV_VOLUME_HD inline void AddCandidate(double *values, std::uint32_t *count,
                                      double value) {
    if (value < -1e-9 || value > 1.0 + 1e-9) return;
    values[(*count)++] = fmin(1.0, fmax(0.0, value));
}

// Match the CPU prism evaluator's sorted boundary candidates and midpoint
// probes, including concave polygons, tangencies and tolerance deduplication.
FV_VOLUME_HD inline bool SegmentEntry(
        const CudaSearchEvaluatorConfiguration &volume,
        double fromX, double fromY, double fromZ,
        double toX, double toY, double toZ, double *fraction) {
    const Point a = Project(volume, fromX, fromY, fromZ);
    const Point b = Project(volume, toX, toY, toZ);
    double candidates[CudaSearchMaximumCustomVolumeVertices + 4u]{0.0, 1.0};
    std::uint32_t count = 2u;
    const double normalDelta = b.normal - a.normal;
    if (fabs(normalDelta) > 1e-12) {
        AddCandidate(candidates, &count, -a.normal / normalDelta);
        AddCandidate(candidates, &count, (volume.values[3] - a.normal) / normalDelta);
    }
    const double pathX = b.x - a.x, pathY = b.y - a.y;
    for (std::uint32_t i = 0u; i < volume.customVolumeVertexCount; ++i) {
        const auto &start = volume.customVolumeVertices[i];
        const auto &end = volume.customVolumeVertices[(i + 1u) % volume.customVolumeVertexCount];
        const double edgeX = end[0] - start[0], edgeY = end[1] - start[1];
        const double denominator = pathX * edgeY - pathY * edgeX;
        if (fabs(denominator) <= 1e-12) continue;
        const double offsetX = start[0] - a.x, offsetY = start[1] - a.y;
        const double edgeFraction = (offsetX * pathY - offsetY * pathX) / denominator;
        if (edgeFraction >= -1e-9 && edgeFraction <= 1.0 + 1e-9) {
            AddCandidate(candidates, &count,
                         (offsetX * edgeY - offsetY * edgeX) / denominator);
        }
    }
    for (std::uint32_t i = 1u; i < count; ++i) {
        const double value = candidates[i];
        std::uint32_t j = i;
        while (j > 0u && candidates[j - 1u] > value) {
            candidates[j] = candidates[j - 1u];
            --j;
        }
        candidates[j] = value;
    }
    std::uint32_t uniqueCount = 1u;
    for (std::uint32_t i = 1u; i < count; ++i) {
        if (fabs(candidates[i] - candidates[uniqueCount - 1u]) > 1e-9)
            candidates[uniqueCount++] = candidates[i];
    }
    for (std::uint32_t i = 0u; i < uniqueCount; ++i) {
        const double candidate = candidates[i];
        // Interpolate world coordinates before projection, just like the CPU.
        if (Contains(volume, Project(volume,
                    fromX + (toX - fromX) * candidate,
                    fromY + (toY - fromY) * candidate,
                    fromZ + (toZ - fromZ) * candidate))) {
            *fraction = candidate;
            return true;
        }
        if (i + 1u == uniqueCount) continue;
        const double midpoint = (candidate + candidates[i + 1u]) * 0.5;
        if (Contains(volume, Project(volume,
                    fromX + (toX - fromX) * midpoint,
                    fromY + (toY - fromY) * midpoint,
                    fromZ + (toZ - fromZ) * midpoint))) {
            *fraction = candidate;
            return true;
        }
    }
    return false;
}

}  // namespace forevervalidator::simulation::cuda::custom_volume

#undef FV_VOLUME_HD
