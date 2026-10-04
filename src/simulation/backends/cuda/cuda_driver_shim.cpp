#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace {

// Query the ABI matching each explicit cudaTypedefs.h type, not the toolkit
// version. Newer toolkit versions can exceed an otherwise compatible driver's
// entry-point table or select a different function signature.
template <typename Function>
Function
ResolveDriverFunction(const char *symbol,
                      unsigned int abiVersion,
                      unsigned long long flags = cudaEnableDefault) noexcept {
  void *entryPoint = nullptr;
  cudaDriverEntryPointQueryResult queryResult =
      cudaDriverEntryPointSymbolNotFound;
  const cudaError_t result = cudaGetDriverEntryPointByVersion(
      symbol, &entryPoint, abiVersion, flags, &queryResult);
  if (result != cudaSuccess || queryResult != cudaDriverEntryPointSuccess ||
      entryPoint == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<Function>(entryPoint);
}

constexpr CUresult DriverUnavailable() noexcept {
  return CUDA_ERROR_NOT_INITIALIZED;
}

} // namespace

extern "C" CUresult CUDAAPI cuFuncGetAttribute(int *value,
                                               CUfunction_attribute attribute,
                                               CUfunction function) {
  using Function = PFN_cuFuncGetAttribute_v2020;
  static Function entry = ResolveDriverFunction<Function>("cuFuncGetAttribute", 2020);
  return entry == nullptr ? DriverUnavailable()
                          : entry(value, attribute, function);
}

extern "C" CUresult CUDAAPI cuFuncGetName(const char **name,
                                          CUfunction function) {
  using Function = PFN_cuFuncGetName_v12030;
  static Function entry = ResolveDriverFunction<Function>("cuFuncGetName", 12030);
  return entry == nullptr ? DriverUnavailable() : entry(name, function);
}

extern "C" CUresult CUDAAPI cuGetErrorString(CUresult error,
                                             const char **message) {
  using Function = PFN_cuGetErrorString_v6000;
  static Function entry = ResolveDriverFunction<Function>("cuGetErrorString", 6000);
  if (entry != nullptr) {
    return entry(error, message);
  }
  if (message != nullptr) {
    *message = "CUDA driver is unavailable";
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuLaunchKernel(
    CUfunction function, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemoryBytes, CUstream stream,
    void **kernelParameters, void **extra) {
  using Function = PFN_cuLaunchKernel_v4000;
  static Function entry =
      ResolveDriverFunction<Function>("cuLaunchKernel", 4000, cudaEnableLegacyStream);
  return entry == nullptr
             ? DriverUnavailable()
             : entry(function, gridDimX, gridDimY, gridDimZ, blockDimX,
                     blockDimY, blockDimZ, sharedMemoryBytes, stream,
                     kernelParameters, extra);
}

extern "C" CUresult CUDAAPI cuModuleEnumerateFunctions(
    CUfunction *functions, unsigned int functionCount, CUmodule module) {
  using Function = PFN_cuModuleEnumerateFunctions_v12040;
  static Function entry =
      ResolveDriverFunction<Function>("cuModuleEnumerateFunctions", 12040);
  return entry == nullptr ? DriverUnavailable()
                          : entry(functions, functionCount, module);
}

extern "C" CUresult CUDAAPI
cuModuleGetFunctionCount(unsigned int *functionCount, CUmodule module) {
  using Function = PFN_cuModuleGetFunctionCount_v12040;
  static Function entry =
      ResolveDriverFunction<Function>("cuModuleGetFunctionCount", 12040);
  return entry == nullptr ? DriverUnavailable() : entry(functionCount, module);
}

extern "C" CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *devicePointer,
                                               std::size_t *bytes,
                                               CUmodule module,
                                               const char *name) {
  using Function = PFN_cuModuleGetGlobal_v3020;
  static Function entry =
      ResolveDriverFunction<Function>("cuModuleGetGlobal", 3020);
  return entry == nullptr ? DriverUnavailable()
                          : entry(devicePointer, bytes, module, name);
}

extern "C" CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr destination,
                                          const void *source,
                                          std::size_t bytes) {
  using Function = PFN_cuMemcpyHtoD_v3020;
  static Function entry = ResolveDriverFunction<Function>("cuMemcpyHtoD", 3020);
  return entry == nullptr ? DriverUnavailable()
                          : entry(destination, source, bytes);
}

extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule *module,
                                             const void *image) {
  using Function = PFN_cuModuleLoadData_v2000;
  static Function entry = ResolveDriverFunction<Function>("cuModuleLoadData", 2000);
  return entry == nullptr ? DriverUnavailable() : entry(module, image);
}

extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule module) {
  using Function = PFN_cuModuleUnload_v2000;
  static Function entry = ResolveDriverFunction<Function>("cuModuleUnload", 2000);
  return entry == nullptr ? DriverUnavailable() : entry(module);
}

extern "C" CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *blockCount, CUfunction function, int blockSize,
    std::size_t dynamicSharedMemoryBytes) {
  using Function = PFN_cuOccupancyMaxActiveBlocksPerMultiprocessor_v6050;
  static Function entry = ResolveDriverFunction<Function>(
      "cuOccupancyMaxActiveBlocksPerMultiprocessor", 6050);
  return entry == nullptr
             ? DriverUnavailable()
             : entry(blockCount, function, blockSize, dynamicSharedMemoryBytes);
}
