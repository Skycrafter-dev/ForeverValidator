#ifndef FOREVERVALIDATOR_OPTIMIZED_CPU_BINARY32_MATH_H
#define FOREVERVALIDATOR_OPTIMIZED_CPU_BINARY32_MATH_H

#include <cstdint>

namespace forevervalidator::simulation {

enum class OptimizedCpuBinary32MathPath : std::uint8_t {
    Reference,
    X86Sse2,
};

// Select once outside an OptimizedCpu kernel. The native path is returned only
// while the deterministic execution scope has established a compatible
// floating-point environment.
OptimizedCpuBinary32MathPath
SelectOptimizedCpuBinary32MathPathForActiveExecution() noexcept;

// Native leaves for kernels that hoist the path branch. Call them directly
// only while the selector's X86Sse2 result remains valid. On a build without
// the supported compiler/ISA implementation, these definitions call Reference.
float OptimizedCpuBinary32FromDoubleX86Sse2(double value) noexcept;
float OptimizedCpuBinary32SqrtX86Sse2(float value) noexcept;

// These names are intentionally distinct from the authoritative Binary32 and
// CI symbols. A selected Reference path always calls those original symbols;
// the native path still sends exceptional inputs through them.
float OptimizedCpuBinary32FromDouble(
        double value,
        OptimizedCpuBinary32MathPath path) noexcept;
float OptimizedCpuBinary32Sqrt(
        float value,
        OptimizedCpuBinary32MathPath path) noexcept;

}  // namespace forevervalidator::simulation

#endif
