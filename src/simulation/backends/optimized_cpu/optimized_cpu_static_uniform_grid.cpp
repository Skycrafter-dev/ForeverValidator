#include "simulation/backends/optimized_cpu/optimized_cpu_static_uniform_grid.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace {

constexpr u32 LooseLevelCount = 6u;

bool IsUsableBounds(const GmBoxAligned &bounds) {
    return std::isfinite(bounds.center.x) &&
           std::isfinite(bounds.center.y) &&
           std::isfinite(bounds.center.z) &&
           std::isfinite(bounds.halfExtents.x) &&
           std::isfinite(bounds.halfExtents.y) &&
           std::isfinite(bounds.halfExtents.z) &&
           bounds.halfExtents.x >= 0.0f &&
           bounds.halfExtents.y >= 0.0f &&
           bounds.halfExtents.z >= 0.0f;
}

double RoundingSlack(float center, float halfExtent) {
    return 8.0 * std::numeric_limits<float>::epsilon() *
           (std::fabs(static_cast<double>(center)) + halfExtent + 1.0);
}

double Minimum(float center, float halfExtent) {
    return static_cast<double>(center) - halfExtent -
           RoundingSlack(center, halfExtent);
}

double Maximum(float center, float halfExtent) {
    return static_cast<double>(center) + halfExtent +
           RoundingSlack(center, halfExtent);
}

u32 DimensionFor(double span, double inverseCellSize) {
    const double cells = std::ceil(span * inverseCellSize);
    if (!(cells >= 0.0) ||
        cells > static_cast<double>(std::numeric_limits<u32>::max() - 1u)) {
        return 0u;
    }
    return std::max(1u, static_cast<u32>(cells));
}

bool ProductFits(u32 x, u32 y, u32 z, u32 maximum, u32 *product) {
    const std::uint64_t count =
            static_cast<std::uint64_t>(x) * y * z;
    if (count == 0u || count > maximum) {
        return false;
    }
    *product = static_cast<u32>(count);
    return true;
}

u32 Coordinate(double value,
               double origin,
               double inverseCellSize,
               u32 dimension) {
    const double cell = std::floor((value - origin) * inverseCellSize);
    const double last = static_cast<double>(dimension - 1u);
    return static_cast<u32>(std::max(0.0, std::min(last, cell)));
}

}  // namespace

bool OptimizedCpuStaticUniformGrid::TryBuild(
        const std::vector<Entry> &entries,
        std::size_t sourceCount,
        u32 maximumCellCount) noexcept {
    Clear();
    if (entries.empty() || sourceCount == 0u || maximumCellCount == 0u ||
        sourceCount > std::numeric_limits<u32>::max()) {
        return false;
    }

    try {
        double minimumX = std::numeric_limits<double>::infinity();
        double minimumY = std::numeric_limits<double>::infinity();
        double minimumZ = std::numeric_limits<double>::infinity();
        double maximumX = -std::numeric_limits<double>::infinity();
        double maximumY = -std::numeric_limits<double>::infinity();
        double maximumZ = -std::numeric_limits<double>::infinity();
        u32 previousSourceIndex = 0u;
        bool firstEntry = true;
        for (const Entry &entry : entries) {
            if (entry.sourceIndex >= sourceCount ||
                !IsUsableBounds(entry.bounds) ||
                (!firstEntry && entry.sourceIndex <= previousSourceIndex)) {
                return false;
            }
            firstEntry = false;
            previousSourceIndex = entry.sourceIndex;
            minimumX = std::min(
                    minimumX,
                    Minimum(entry.bounds.center.x,
                            entry.bounds.halfExtents.x));
            minimumY = std::min(
                    minimumY,
                    Minimum(entry.bounds.center.y,
                            entry.bounds.halfExtents.y));
            minimumZ = std::min(
                    minimumZ,
                    Minimum(entry.bounds.center.z,
                            entry.bounds.halfExtents.z));
            maximumX = std::max(
                    maximumX,
                    Maximum(entry.bounds.center.x,
                            entry.bounds.halfExtents.x));
            maximumY = std::max(
                    maximumY,
                    Maximum(entry.bounds.center.y,
                            entry.bounds.halfExtents.y));
            maximumZ = std::max(
                    maximumZ,
                    Maximum(entry.bounds.center.z,
                            entry.bounds.halfExtents.z));
        }

        const double spanX = maximumX - minimumX;
        const double spanY = maximumY - minimumY;
        const double spanZ = maximumZ - minimumZ;
        if (!std::isfinite(spanX) || !std::isfinite(spanY) ||
            !std::isfinite(spanZ)) {
            return false;
        }

        double baseCellSize = 1.0;
        for (;;) {
            const double inverse = 1.0 / baseCellSize;
            u32 ignored = 0u;
            const u32 x = DimensionFor(spanX, inverse);
            const u32 y = DimensionFor(spanY, inverse);
            const u32 z = DimensionFor(spanZ, inverse);
            if (x != 0u && y != 0u && z != 0u &&
                ProductFits(x, y, z, maximumCellCount, &ignored)) {
                break;
            }
            baseCellSize *= 2.0;
            if (!std::isfinite(baseCellSize)) {
                return false;
            }
        }

        OptimizedCpuStaticUniformGrid rebuilt;
        rebuilt.originX_ = minimumX;
        rebuilt.originY_ = minimumY;
        rebuilt.originZ_ = minimumZ;
        rebuilt.looseLevels_.reserve(LooseLevelCount);

        struct EntryRange {
            u32 firstX;
            u32 lastX;
            u32 firstY;
            u32 lastY;
            u32 firstZ;
            u32 lastZ;
        };

        for (u32 levelIndex = 0u;
             levelIndex < LooseLevelCount;
             ++levelIndex) {
            const double cellSize = std::ldexp(baseCellSize, levelIndex);
            const double inverse = 1.0 / cellSize;
            const u32 dimensionX = DimensionFor(spanX, inverse);
            const u32 dimensionY = DimensionFor(spanY, inverse);
            const u32 dimensionZ = DimensionFor(spanZ, inverse);
            u32 cellCount = 0u;
            if (dimensionX == 0u || dimensionY == 0u ||
                dimensionZ == 0u ||
                !ProductFits(dimensionX,
                             dimensionY,
                             dimensionZ,
                             maximumCellCount,
                             &cellCount)) {
                return false;
            }

            LooseLevel level;
            level.cellSize = cellSize;
            level.inverseCellSize = inverse;
            level.dimensionX = dimensionX;
            level.dimensionY = dimensionY;
            level.dimensionZ = dimensionZ;
            level.offsets.assign(
                    static_cast<std::size_t>(cellCount) + 1u, 0u);

            std::vector<EntryRange> ranges;
            ranges.reserve(entries.size());
            const double halfCell = cellSize * 0.5;
            std::uint64_t referenceCount = 0u;
            for (const Entry &entry : entries) {
                const EntryRange range = {
                    Coordinate(Minimum(entry.bounds.center.x,
                                       entry.bounds.halfExtents.x) - halfCell,
                               minimumX, inverse, dimensionX),
                    Coordinate(Maximum(entry.bounds.center.x,
                                       entry.bounds.halfExtents.x) + halfCell,
                               minimumX, inverse, dimensionX),
                    Coordinate(Minimum(entry.bounds.center.y,
                                       entry.bounds.halfExtents.y) - halfCell,
                               minimumY, inverse, dimensionY),
                    Coordinate(Maximum(entry.bounds.center.y,
                                       entry.bounds.halfExtents.y) + halfCell,
                               minimumY, inverse, dimensionY),
                    Coordinate(Minimum(entry.bounds.center.z,
                                       entry.bounds.halfExtents.z) - halfCell,
                               minimumZ, inverse, dimensionZ),
                    Coordinate(Maximum(entry.bounds.center.z,
                                       entry.bounds.halfExtents.z) + halfCell,
                               minimumZ, inverse, dimensionZ),
                };
                ranges.push_back(range);
                referenceCount +=
                        static_cast<std::uint64_t>(
                                range.lastX - range.firstX + 1u) *
                        (range.lastY - range.firstY + 1u) *
                        (range.lastZ - range.firstZ + 1u);
                if (referenceCount > std::numeric_limits<u32>::max()) {
                    return false;
                }
                for (u32 z = range.firstZ; z <= range.lastZ; ++z) {
                    for (u32 y = range.firstY; y <= range.lastY; ++y) {
                        const u32 row =
                                (z * dimensionY + y) * dimensionX;
                        for (u32 x = range.firstX; x <= range.lastX; ++x) {
                            ++level.offsets[row + x + 1u];
                        }
                    }
                }
            }

            for (u32 index = 0u; index < cellCount; ++index) {
                level.offsets[index + 1u] += level.offsets[index];
            }
            level.entries.resize(referenceCount);
            std::vector<u32> cursors = level.offsets;
            for (std::size_t entryIndex = 0u;
                 entryIndex < entries.size();
                 ++entryIndex) {
                const EntryRange &range = ranges[entryIndex];
                for (u32 z = range.firstZ; z <= range.lastZ; ++z) {
                    for (u32 y = range.firstY; y <= range.lastY; ++y) {
                        const u32 row =
                                (z * dimensionY + y) * dimensionX;
                        for (u32 x = range.firstX; x <= range.lastX; ++x) {
                            level.entries[cursors[row + x]++] =
                                    entries[entryIndex].sourceIndex;
                        }
                    }
                }
            }
            rebuilt.looseLevels_.push_back(std::move(level));
        }

        *this = std::move(rebuilt);
        return true;
    } catch (const std::bad_alloc &) {
        Clear();
        return false;
    }
}

void OptimizedCpuStaticUniformGrid::Clear(void) noexcept {
    originX_ = 0.0;
    originY_ = 0.0;
    originZ_ = 0.0;
    looseLevels_.clear();
}

bool OptimizedCpuStaticUniformGrid::DirectCandidateSpan(
        const GmBoxAligned &query,
        CandidateSpan *result) const noexcept {
    if (result == nullptr || looseLevels_.empty() ||
        !IsUsableBounds(query)) {
        return false;
    }

    const double requiredHalfExtent = std::max({
        static_cast<double>(query.halfExtents.x) +
                RoundingSlack(query.center.x, query.halfExtents.x),
        static_cast<double>(query.halfExtents.y) +
                RoundingSlack(query.center.y, query.halfExtents.y),
        static_cast<double>(query.halfExtents.z) +
                RoundingSlack(query.center.z, query.halfExtents.z),
    });
    const LooseLevel *selected = nullptr;
    for (const LooseLevel &level : looseLevels_) {
        if (requiredHalfExtent <= level.cellSize * 0.5) {
            selected = &level;
            break;
        }
    }
    if (selected == nullptr) {
        return false;
    }

    auto queryCoordinate = [](float center,
                              double origin,
                              double inverse,
                              u32 dimension) {
        const double cell =
                (static_cast<double>(center) - origin) * inverse;
        if (cell <= 0.0) {
            return 0u;
        }
        if (cell >= static_cast<double>(dimension)) {
            return dimension - 1u;
        }
        return static_cast<u32>(cell);
    };
    const u32 x = queryCoordinate(query.center.x,
                                  originX_,
                                  selected->inverseCellSize,
                                  selected->dimensionX);
    const u32 y = queryCoordinate(query.center.y,
                                  originY_,
                                  selected->inverseCellSize,
                                  selected->dimensionY);
    const u32 z = queryCoordinate(query.center.z,
                                  originZ_,
                                  selected->inverseCellSize,
                                  selected->dimensionZ);
    const u32 cellIndex =
            (z * selected->dimensionY + y) * selected->dimensionX + x;
    const u32 first = selected->offsets[cellIndex];
    const u32 last = selected->offsets[cellIndex + 1u];
    result->size = last - first;
    result->data = result->size == 0u
            ? nullptr
            : selected->entries.data() + first;
    return true;
}
