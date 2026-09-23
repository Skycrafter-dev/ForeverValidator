#ifndef FOREVERVALIDATOR_HIP_SEARCH_ARCH_ADAPTER_H
#define FOREVERVALIDATOR_HIP_SEARCH_ARCH_ADAPTER_H

#if defined(__HIP_PLATFORM_NVIDIA__)
#define FOREVERVALIDATOR_HIP_SEARCH_LAUNCH_BOUNDS(blocks, minimum) \
    __launch_bounds__(blocks, minimum)
#else
#define FOREVERVALIDATOR_HIP_SEARCH_LAUNCH_BOUNDS(blocks, minimum) \
    __launch_bounds__(blocks)
#endif

#endif
