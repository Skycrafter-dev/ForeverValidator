#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "engine/core/gm_types.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_static_bounds_overlap.h"

namespace {

using forevervalidator::simulation::OptimizedCpuStaticBoundsOverlap;

struct TestCase {
    const char *name;
    GmBoxAligned moving;
    GmBoxAligned fixed;
};

float FloatFromBits(std::uint32_t bits) noexcept {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

GmBoxAligned Box(
        float centerX,
        float centerY,
        float centerZ,
        float extentX,
        float extentY,
        float extentZ) {
    return {{centerX, centerY, centerZ}, {extentX, extentY, extentZ}};
}

bool RunCase(const TestCase &testCase) {
    std::feclearexcept(FE_ALL_EXCEPT);
    const int reference = testCase.moving.TestInter(testCase.fixed);
    const int referenceExceptions = std::fetestexcept(FE_ALL_EXCEPT);

    std::feclearexcept(FE_ALL_EXCEPT);
    const int optimized = OptimizedCpuStaticBoundsOverlap(
            testCase.moving, testCase.fixed);
    const int optimizedExceptions = std::fetestexcept(FE_ALL_EXCEPT);

    if (reference != optimized ||
        referenceExceptions != optimizedExceptions) {
        std::fprintf(
                stderr,
                "static_bounds_overlap_differential: %s ref=%d opt=%d "
                "ref_fenv=%x opt_fenv=%x\n",
                testCase.name,
                reference,
                optimized,
                referenceExceptions,
                optimizedExceptions);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const float positiveInfinity = std::numeric_limits<float>::infinity();
    const float quietNaN = std::numeric_limits<float>::quiet_NaN();
    const float signalingNaN = FloatFromBits(0x7fa00001u);
    const float negativeZero = FloatFromBits(0x80000000u);

    const TestCase cases[] = {
        {"ordinary-overlap",
         Box(1.0f, -2.0f, 3.0f, 2.0f, 3.0f, 4.0f),
         Box(2.5f, 0.0f, 6.0f, 1.0f, 1.0f, 1.0f)},
        {"z-disjoint",
         Box(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
         Box(0.0f, 0.0f, 3.0f, 1.0f, 1.0f, 1.0f)},
        {"y-disjoint",
         Box(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
         Box(0.0f, 3.0f, 0.0f, 1.0f, 1.0f, 1.0f)},
        {"x-disjoint",
         Box(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
         Box(3.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f)},
        {"touching-boundary",
         Box(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
         Box(2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f)},
        {"signed-zero",
         Box(negativeZero, 0.0f, negativeZero,
             negativeZero, negativeZero, 0.0f),
         Box(0.0f, negativeZero, 0.0f,
             0.0f, 0.0f, negativeZero)},
        {"infinite-centers",
         Box(positiveInfinity, positiveInfinity, positiveInfinity,
             1.0f, 1.0f, 1.0f),
         Box(positiveInfinity, positiveInfinity, positiveInfinity,
             1.0f, 1.0f, 1.0f)},
        {"infinite-extents",
         Box(-positiveInfinity, 0.0f, positiveInfinity,
             positiveInfinity, positiveInfinity, positiveInfinity),
         Box(positiveInfinity, 1.0f, -positiveInfinity,
             positiveInfinity, positiveInfinity, positiveInfinity)},
        {"quiet-nan-centers",
         Box(quietNaN, quietNaN, quietNaN, 1.0f, 1.0f, 1.0f),
         Box(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f)},
        {"quiet-nan-extents",
         Box(0.0f, 0.0f, 0.0f, quietNaN, quietNaN, quietNaN),
         Box(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f)},
        {"signaling-nan",
         Box(signalingNaN, signalingNaN, signalingNaN,
             1.0f, 1.0f, 1.0f),
         Box(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f)},
        {"z-short-circuits-signaling-nan-yx",
         Box(signalingNaN, signalingNaN, 0.0f, 1.0f, 1.0f, 1.0f),
         Box(0.0f, 0.0f, 3.0f, 1.0f, 1.0f, 1.0f)},
        {"y-short-circuits-signaling-nan-x",
         Box(signalingNaN, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
         Box(0.0f, 3.0f, 0.0f, 1.0f, 1.0f, 1.0f)},
    };

    std::size_t completed = 0u;
    for (const TestCase &testCase : cases) {
        if (!RunCase(testCase)) {
            return 1;
        }
        ++completed;
    }

    std::printf(
            "static_bounds_overlap_cases=%zu bit_identical=true "
            "fenv_identical=true\n",
            completed);
    return 0;
}
