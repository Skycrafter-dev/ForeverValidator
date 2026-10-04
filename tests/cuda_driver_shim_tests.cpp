#include <cuda.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <map>
#include <string>
#include <utility>

namespace {
using Request = std::pair<unsigned int, unsigned long long>;
std::map<std::string, Request> requests;
}

extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(
        const char *symbol, void **entry, unsigned int version,
        unsigned long long flags, cudaDriverEntryPointQueryResult *status) {
    requests.emplace(symbol, Request{version, flags});
    *entry = nullptr;
    *status = cudaDriverEntryPointSymbolNotFound;
    return cudaErrorInvalidValue;
}

int main() {
    bool okay = true;
    const auto missing = [&](CUresult result) { okay &= result == CUDA_ERROR_NOT_INITIALIZED; };
    missing(cuFuncGetAttribute(nullptr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, nullptr));
    missing(cuFuncGetName(nullptr, nullptr));
    const char *error = nullptr;
    okay &= cuGetErrorString(CUDA_ERROR_NOT_INITIALIZED, &error) == CUDA_SUCCESS && error != nullptr;
    missing(cuLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
    missing(cuModuleEnumerateFunctions(nullptr, 0, nullptr));
    missing(cuModuleGetFunctionCount(nullptr, nullptr));
    missing(cuModuleGetGlobal(nullptr, nullptr, nullptr, nullptr));
    missing(cuMemcpyHtoD(0, nullptr, 0));
    missing(cuModuleLoadData(nullptr, nullptr));
    missing(cuModuleUnload(nullptr));
    missing(cuOccupancyMaxActiveBlocksPerMultiprocessor(nullptr, nullptr, 1, 0));
    const std::map<std::string, Request> expected{
        {"cuFuncGetAttribute", {2020, cudaEnableDefault}},
        {"cuFuncGetName", {12030, cudaEnableDefault}},
        {"cuGetErrorString", {6000, cudaEnableDefault}},
        {"cuLaunchKernel", {4000, cudaEnableLegacyStream}},
        {"cuModuleEnumerateFunctions", {12040, cudaEnableDefault}},
        {"cuModuleGetFunctionCount", {12040, cudaEnableDefault}},
        {"cuModuleGetGlobal", {3020, cudaEnableDefault}},
        {"cuMemcpyHtoD", {3020, cudaEnableDefault}},
        {"cuModuleLoadData", {2000, cudaEnableDefault}},
        {"cuModuleUnload", {2000, cudaEnableDefault}},
        {"cuOccupancyMaxActiveBlocksPerMultiprocessor", {6050, cudaEnableDefault}}
    };
    if (!okay || requests != expected) {
        std::cerr << "Driver shim changed ABI versions, stream semantics, or missing-entry handling\n";
        return 1;
    }
    return 0;
}
