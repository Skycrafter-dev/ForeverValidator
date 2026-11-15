// Twin GPU engine with nvcc and cl.exe as the host compiler (Windows): the GCC builtins and attributes the twin's
// sources spell, for both of nvcc's passes. Force-included there (src/twin/CMakeLists.txt); nothing elsewhere.
#ifndef TW_CUDA_MSVC_COMPAT_H
#define TW_CUDA_MSVC_COMPAT_H

#if defined(_MSC_VER) && !defined(__clang__)

#include <cmath>
#include <cstddef>
#include <cstring>
#include <intrin.h>

#define __attribute__(x)
#define __builtin_expect(e, v) (e)
#define __builtin_fabsf(x) fabsf(x)
#define __builtin_sqrtf(x) sqrtf(x)
#define __builtin_inff() HUGE_VALF
#define __builtin_memcpy memcpy
#define __builtin_prefetch(...) ((void)0)

__host__ __device__ inline int TwBuiltinCtz(unsigned x) {
#if defined(__CUDA_ARCH__)
    return __ffs(static_cast<int>(x)) - 1;
#else
    unsigned long index = 0;
    _BitScanForward(&index, x);
    return static_cast<int>(index);
#endif
}
__host__ __device__ inline int TwBuiltinPopcount(unsigned x) {
#if defined(__CUDA_ARCH__)
    return __popc(x);
#else
    return static_cast<int>(__popcnt(x));
#endif
}
#define __builtin_ctz(x) TwBuiltinCtz(static_cast<unsigned>(x))
#define __builtin_popcount(x) TwBuiltinPopcount(static_cast<unsigned>(x))

#endif

#endif
