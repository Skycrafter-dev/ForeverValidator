#ifndef FOREVERVALIDATOR_HIP_REDUCE_ADAPTER_H
#define FOREVERVALIDATOR_HIP_REDUCE_ADAPTER_H

#if defined(__HIP_PLATFORM_NVIDIA__)
#include <cub/device/device_reduce.cuh>
namespace hipcub {
struct DeviceReduce {
    template<typename... Arguments>
    static hipError_t Reduce(Arguments &&...arguments) {
        return hipCUDAErrorTohipError(cub::DeviceReduce::Reduce(
                static_cast<Arguments &&>(arguments)...));
    }
};
}
#else
#include <hipcub/hipcub.hpp>
#endif

#endif
