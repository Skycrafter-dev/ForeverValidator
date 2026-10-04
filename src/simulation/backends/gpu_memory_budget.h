#ifndef FOREVERVALIDATOR_GPU_MEMORY_BUDGET_H
#define FOREVERVALIDATOR_GPU_MEMORY_BUDGET_H
#include <algorithm>
#include <cstdint>
#include <limits>

namespace forevervalidator::simulation {
inline bool GpuAllocationFitsBudget(std::uint64_t bytes, std::uint64_t available,
                                    std::uint64_t total) {
    if (total == 0 || available > total) return false;
    const auto headroom = std::max<std::uint64_t>(512ull * 1024 * 1024,
            total / 100 * 15 + (total % 100 * 15 + 99) / 100);
    const auto margin = bytes / 100 * 15 + (bytes % 100 * 15 + 99) / 100;
    return available > headroom && bytes <= available - headroom &&
            margin <= available - headroom - bytes;
}
}  // namespace forevervalidator::simulation
#endif
