#include "validation/planning/search_event_capacity.h"
#include "simulation/backends/gpu_memory_budget.h"

#include <iostream>
#include <limits>

using namespace forevervalidator::experimental;

int main() {
    std::size_t capacity = 0;
    const auto limit = kMaximumSearchInputEvents;
    bool okay = true;
    const auto check = [&](bool condition, const char *message) {
        if (!condition) std::cerr << message << '\n';
        okay &= condition;
    };
    using forevervalidator::simulation::GpuAllocationFitsBudget;
    constexpr std::uint64_t gib = 1024ull * 1024 * 1024;
    check(GpuAllocationFitsBudget(gib, 8 * gib, 8 * gib), "ordinary GPU allocation rejected");
    check(!GpuAllocationFitsBudget(gib, gib, 8 * gib), "low VRAM allocation accepted");
    check(!GpuAllocationFitsBudget(7 * gib, 8 * gib, 8 * gib), "GPU headroom consumed");
    check(!GpuAllocationFitsBudget(UINT64_MAX, UINT64_MAX, UINT64_MAX), "GPU byte arithmetic overflowed");
    check(!GpuAllocationFitsBudget(1, 0, 8 * gib), "missing GPU budget accepted");
    const std::vector<PhysicsSandboxCudaModifier> noGrowth{
        PhysicsSandboxCudaRandomSteeringModifier{}, PhysicsSandboxCudaInputDeletionModifier{}};
    for (const auto baseline : {limit - 1, limit, limit + 1, std::numeric_limits<std::size_t>::max()}) {
        check(search_limits::MaximumEventCapacity(baseline, noGrowth, 10, &capacity) ==
                      (baseline <= limit), "baseline capacity boundary failed");
    }
    capacity = limit + 1;
    check(!search_limits::AddEventCapacity(0, &capacity), "oversized capacity subtraction underflowed");
    capacity = limit;
    check(search_limits::AddEventCapacity(0, &capacity) &&
          !search_limits::AddEventCapacity(1, &capacity), "growth boundary failed");
    check(!search_limits::MaximumEventCapacity(0, {}, 0, &capacity), "zero tick was accepted");
    PhysicsSandboxCudaInputInsertionModifier insertion;
    insertion.steering.enabled = true;
    insertion.steering.maximumCount = 1;
    check(search_limits::MaximumEventCapacity(limit - 2, {insertion}, 10, &capacity) &&
                  capacity == limit, "exact-limit insertion failed");
    check(!search_limits::MaximumEventCapacity(limit - 1, {insertion}, 10, &capacity),
          "insertion above limit passed");
    insertion.steering.maximumCount = 0x80000000u;
    insertion.accelerate.enabled = true;
    insertion.accelerate.maximumCount = 0x80000000u;
    check(!search_limits::MaximumEventCapacity(0, {insertion}, 10, &capacity),
          "insertion operation sum overflowed at 32 bits");
    PhysicsSandboxCudaSmoothSteeringModifier smooth;
    smooth.radiusMs = std::numeric_limits<std::int64_t>::max();
    smooth.deformationCount = 1;
    check(!search_limits::MaximumEventCapacity(0, {smooth}, 1, &capacity),
          "smooth radius overflowed at 64 bits");
    smooth.radiusMs = 10;
    smooth.deformationCount = 1;
    insertion.steering.maximumCount = 1;
    insertion.accelerate.enabled = false;
    check(search_limits::MaximumEventCapacity(limit - 8, {smooth, insertion, insertion}, 10, &capacity) &&
                  capacity == limit, "multiple pass capacities were not accumulated");
    check(!search_limits::MaximumEventCapacity(limit - 7, {smooth, insertion, insertion}, 10, &capacity),
          "multiple pass capacity exceeded the limit");
    return okay ? 0 : 1;
}
