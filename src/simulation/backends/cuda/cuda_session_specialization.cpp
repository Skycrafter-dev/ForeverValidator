#include "simulation/backends/cuda/cuda_session_specialization.h"

#include <nvJitLink.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <vector>

#include "forevervalidator_cuda_search_lto_ir.h"

namespace forevervalidator::simulation::cuda::specialization {
namespace {

constexpr std::uint32_t SimulationBlockSize = 32u;
constexpr std::uint32_t TailMinimumBlocks = 12u;
constexpr std::uint32_t DenseTailMinimumBlocks = 8u;
std::atomic_uint64_t SessionModuleBuildCount{0u};

std::string NvrtcLog(nvrtcProgram program) {
    std::size_t size = 0u;
    if (nvrtcGetProgramLogSize(program, &size) != NVRTC_SUCCESS ||
        size == 0u) {
        return {};
    }
    std::string result(size, '\0');
    nvrtcGetProgramLog(program, result.data());
    return result;
}

std::string LinkLog(nvJitLinkHandle handle) {
    std::size_t size = 0u;
    if (nvJitLinkGetErrorLogSize(handle, &size) !=
                NVJITLINK_SUCCESS ||
        size == 0u) {
        return {};
    }
    std::string result(size, '\0');
    nvJitLinkGetErrorLog(handle, result.data());
    return result;
}

template<typename T>
void AppendByteExactFunction(
        std::ostringstream &source,
        std::string_view name,
        const T &value) {
    const auto *bytes =
            reinterpret_cast<const unsigned char *>(&value);
    source << std::dec
           << "struct alignas(" << alignof(T) << ") " << name
           << "Result { unsigned char bytes[" << sizeof(T)
           << "]; };\n"
              "extern \"C\" __device__ "
           << name << "Result " << name
           << "() {\n  " << name << "Result value;\n";
    source << std::hex << std::setfill('0');
    for (std::size_t index = 0u; index < sizeof(T); ++index) {
        source << "  value.bytes[" << std::dec << index << "] = 0x"
               << std::hex << std::setw(2)
               << static_cast<unsigned int>(bytes[index]) << "u;\n";
    }
    source << "  return value;\n}\n";
}

template<typename T>
void AppendConstantBytePointer(
        std::ostringstream &source,
        std::string_view name,
        const T &value) {
    const auto *bytes =
            reinterpret_cast<const unsigned char *>(&value);
    const std::size_t size = sizeof(T);
    source << std::dec
           << "struct alignas(" << alignof(T) << ") " << name
           << "StorageType { unsigned char bytes[" << size
           << "]; };\n"
              "__device__ __constant__ "
           << name << "StorageType " << name << "Storage = {{\n";
    source << std::hex << std::setfill('0');
    for (std::size_t index = 0u; index < size; ++index) {
        source << "0x" << std::setw(2)
               << static_cast<unsigned int>(bytes[index]) << "u,";
        if ((index & 15u) == 15u) {
            source << '\n';
        }
    }
    source << "}};\n"
              "extern \"C\" __device__ const unsigned char* "
           << name << "Bytes() { return " << name
           << "Storage.bytes; }\n";
}

void AppendConstantBytePointer(
        std::ostringstream &source,
        std::string_view name,
        const std::vector<unsigned char> &bytes,
        std::size_t alignment) {
    source << std::dec
           << "struct alignas(" << alignment << ") " << name
           << "StorageType { unsigned char bytes[" << bytes.size()
           << "]; };\n"
              "__device__ __constant__ "
           << name << "StorageType " << name << "Storage = {{\n";
    source << std::hex << std::setfill('0');
    for (std::size_t index = 0u; index < bytes.size(); ++index) {
        source << "0x" << std::setw(2)
               << static_cast<unsigned int>(bytes[index]) << "u,";
        if ((index & 15u) == 15u) {
            source << '\n';
        }
    }
    source << "}};\n"
              "extern \"C\" __device__ const unsigned char* "
           << name << "Bytes() { return " << name
           << "Storage.bytes; }\n";
}

void AppendTuningWords(
        std::ostringstream &source,
        const CudaVehicleTuning &tuning) {
    const auto *bytes =
            reinterpret_cast<const unsigned char *>(&tuning);
    const std::size_t wordCount =
            (sizeof(tuning) + sizeof(std::uint32_t) - 1u) /
            sizeof(std::uint32_t);
    source << "namespace forevervalidator::simulation::cuda::research {\n"
              "template<unsigned int Index>\n"
              "__device__ unsigned int "
              "ForeverValidatorSessionTuningWord();\n";
    source << std::hex << std::setfill('0');
    for (std::size_t index = 0u; index < wordCount; ++index) {
        std::uint32_t word = 0u;
        const std::size_t offset = index * sizeof(word);
        std::memcpy(
                &word,
                bytes + offset,
                std::min(sizeof(word), sizeof(tuning) - offset));
        source << "template<> __device__ unsigned int "
                  "ForeverValidatorSessionTuningWord<"
               << std::dec << index << ">() { return 0x"
               << std::hex << std::setw(8) << word << "u; }\n";
    }
    source << "}\n";
}

void AppendShapeWords(
        std::ostringstream &source,
        const std::vector<unsigned char> &shapes,
        std::size_t shapeStride) {
    const std::size_t shapeCount =
            shapeStride == 0u ? 0u : shapes.size() / shapeStride;
    const std::size_t wordCount =
            (sizeof(CudaVehicleCollisionShape) +
             sizeof(std::uint32_t) - 1u) /
            sizeof(std::uint32_t);
    source << "namespace forevervalidator::simulation::cuda::research {\n"
              "template<unsigned int Index>\n"
              "__device__ unsigned int "
              "ForeverValidatorSessionShapeWord("
              "unsigned int shapeIndex);\n";
    source << std::hex << std::setfill('0');
    for (std::size_t wordIndex = 0u;
         wordIndex < wordCount;
         ++wordIndex) {
        source << "template<> __device__ unsigned int "
                  "ForeverValidatorSessionShapeWord<"
               << std::dec << wordIndex
               << ">(unsigned int shapeIndex) { switch (shapeIndex) {\n";
        for (std::size_t shapeIndex = 0u;
             shapeIndex < shapeCount;
             ++shapeIndex) {
            std::uint32_t word = 0u;
            const std::size_t offset =
                    shapeIndex * shapeStride +
                    wordIndex * sizeof(word);
            const std::size_t shapeEnd =
                    (shapeIndex + 1u) * shapeStride;
            std::memcpy(
                    &word,
                    shapes.data() + offset,
                    std::min(sizeof(word), shapeEnd - offset));
            source << "case " << std::dec << shapeIndex
                   << "u: return 0x" << std::hex << std::setw(8)
                   << word << "u;\n";
        }
        source << "default: return 0u; } }\n";
    }
    source << "}\n";
}

void AppendPointerStorage(
        std::ostringstream &source,
        std::string_view name) {
    source << "extern \"C\" __device__ unsigned long long "
           << name << "Storage = 0ull;\n";
    source << "extern \"C\" __device__ unsigned long long "
           << name << "() { return " << name << "Storage; }\n";
}

bool LoadMetrics(
        CUfunction function,
        KernelMetrics *metrics,
        std::string *diagnostic) {
    int registers = 0;
    int localBytes = 0;
    int activeBlocks = 0;
    CUresult result = cuFuncGetAttribute(
            &registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
    if (result == CUDA_SUCCESS) {
        result = cuFuncGetAttribute(
                &localBytes,
                CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                function);
    }
    if (result == CUDA_SUCCESS) {
        result = cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &activeBlocks, function, SimulationBlockSize, 0u);
    }
    if (result != CUDA_SUCCESS) {
        if (diagnostic != nullptr) {
            const char *message = nullptr;
            cuGetErrorString(result, &message);
            *diagnostic =
                    "querying specialized CUDA kernel metrics: " +
                    std::string(message == nullptr ? "unknown" : message);
        }
        return false;
    }
    metrics->registersPerThread =
            static_cast<std::uint32_t>(registers);
    metrics->localBytesPerThread =
            static_cast<std::uint64_t>(localBytes);
    metrics->activeBlocksPerMultiprocessor =
            static_cast<std::uint32_t>(activeBlocks);
    return true;
}

}  // namespace

SessionModule::~SessionModule() {
    Reset();
}

void SessionModule::Reset() noexcept {
    if (module_ != nullptr) {
        cuModuleUnload(module_);
    }
    module_ = nullptr;
    throughput_ = {};
    tail_ = {};
    denseTail_ = {};
}

std::uint64_t SessionModuleBuildCountForTesting() noexcept {
    return SessionModuleBuildCount.load(std::memory_order_relaxed);
}

bool SessionModule::Build(
        const CudaPackedStaticConfigurationHeader &configuration,
        std::uint64_t configurationBase,
        const CudaPackedSceneHeader &scene,
        std::uint64_t sceneBase,
        std::string *diagnostic) {
    SessionModuleBuildCount.fetch_add(1u, std::memory_order_relaxed);
    Reset();
    const auto started = std::chrono::steady_clock::now();

    std::vector<unsigned char> collisionShapes(
            static_cast<std::size_t>(
                    configuration.collisionShapes.count) *
            configuration.collisionShapes.stride);
    std::vector<unsigned char> curveKeys(
            static_cast<std::size_t>(
                    configuration.curveKeys.count) *
            configuration.curveKeys.stride);
    const auto copySection =
            [&](std::vector<unsigned char> &destination,
                std::uint64_t offset,
                std::string_view label) {
        if (destination.empty()) {
            return true;
        }
        const cudaError_t error = cudaMemcpy(
                destination.data(),
                reinterpret_cast<const void *>(
                        configurationBase + offset),
                destination.size(),
                cudaMemcpyDeviceToHost);
        if (error != cudaSuccess && diagnostic != nullptr) {
            *diagnostic =
                    "reading exact CUDA " + std::string(label) +
                    " facts: " + cudaGetErrorString(error);
        }
        return error == cudaSuccess;
    };
    if (!copySection(
                collisionShapes,
                configuration.collisionShapes.offset,
                "collision-shape") ||
        !copySection(
                curveKeys,
                configuration.curveKeys.offset,
                "curve-key")) {
        return false;
    }
    if (std::getenv("FOREVERVALIDATOR_CUDA_DUMP_SPECIALIZATION") !=
        nullptr) {
        for (std::uint32_t index = 0u;
             index < configuration.collisionShapes.count;
             ++index) {
            const auto *shape =
                    reinterpret_cast<
                            const CudaVehicleCollisionShape *>(
                            collisionShapes.data() +
                            static_cast<std::size_t>(index) *
                                    configuration.collisionShapes.stride);
            std::fprintf(
                    stderr,
                    "shape[%u] center=(%a,%a,%a) radii=(%a,%a,%a) "
                    "wheel=%u material=%u\n",
                    index,
                    shape->localBounds.center.x,
                    shape->localBounds.center.y,
                    shape->localBounds.center.z,
                    shape->localBounds.halfExtents.x,
                    shape->localBounds.halfExtents.y,
                    shape->localBounds.halfExtents.z,
                    shape->wheelIndex,
                    shape->surfaceMaterial);
        }
    }

    std::ostringstream source;
    AppendConstantBytePointer(
            source,
            "ForeverValidatorSessionConfiguration",
            configuration);
    AppendConstantBytePointer(
            source,
            "ForeverValidatorSessionScene",
            scene);
    AppendConstantBytePointer(
            source,
            "ForeverValidatorSessionCollisionShape",
            collisionShapes,
            alignof(CudaVehicleCollisionShape));
    AppendConstantBytePointer(
            source,
            "ForeverValidatorSessionCurveKey",
            curveKeys,
            alignof(CudaTuningCurveKey));
    AppendTuningWords(source, configuration.tuning);
    AppendShapeWords(
            source,
            collisionShapes,
            configuration.collisionShapes.stride);
    AppendPointerStorage(
            source,
            "ForeverValidatorSessionConfigurationBase");
    AppendPointerStorage(
            source,
            "ForeverValidatorSessionSceneBase");
    const std::string sourceText = source.str();

    nvrtcProgram program = nullptr;
    nvrtcResult compileResult = nvrtcCreateProgram(
            &program,
            sourceText.c_str(),
            "forevervalidator_session_facts.cu",
            0,
            nullptr,
            nullptr);
    const char *compileOptions[] = {
            "--std=c++17",
            "--gpu-architecture=compute_75",
            "--device-c",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "--ftz=false",
            "-dlto",
    };
    if (compileResult == NVRTC_SUCCESS) {
        compileResult = nvrtcCompileProgram(
                program,
                static_cast<int>(std::size(compileOptions)),
                compileOptions);
    }
    if (compileResult != NVRTC_SUCCESS) {
        if (diagnostic != nullptr) {
            *diagnostic =
                    "compiling exact CUDA session facts: " +
                    std::string(nvrtcGetErrorString(compileResult)) +
                    "\n" + NvrtcLog(program);
        }
        if (program != nullptr) {
            nvrtcDestroyProgram(&program);
        }
        return false;
    }
    std::size_t factsSize = 0u;
    if (nvrtcGetLTOIRSize(program, &factsSize) != NVRTC_SUCCESS) {
        if (diagnostic != nullptr) {
            *diagnostic = "reading exact CUDA session-fact LTO size";
        }
        nvrtcDestroyProgram(&program);
        return false;
    }
    std::vector<char> facts(factsSize);
    if (nvrtcGetLTOIR(program, facts.data()) != NVRTC_SUCCESS) {
        if (diagnostic != nullptr) {
            *diagnostic = "reading exact CUDA session-fact LTO";
        }
        nvrtcDestroyProgram(&program);
        return false;
    }
    nvrtcDestroyProgram(&program);

    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        if (diagnostic != nullptr) {
            *diagnostic = "querying CUDA device for session specialization";
        }
        return false;
    }
    const std::string architecture =
            "-arch=sm_" + std::to_string(properties.major) +
            std::to_string(properties.minor);
    const char *linkOptions[] = {
            architecture.c_str(),
            "-lto",
            "-kernels-used=SimulateSearchCandidatesKernel",
            "-split-compile=0",
            "-fma=0",
            "-prec-div=1",
            "-prec-sqrt=1",
            "-ftz=0",
    };
    nvJitLinkHandle link = nullptr;
    nvJitLinkResult linkResult = nvJitLinkCreate(
            &link,
            static_cast<std::uint32_t>(std::size(linkOptions)),
            linkOptions);
    if (linkResult == NVJITLINK_SUCCESS) {
        linkResult = nvJitLinkAddData(
                link,
                NVJITLINK_INPUT_LTOIR,
                ForeverValidatorCudaSearchLtoIr,
                ForeverValidatorCudaSearchLtoIr_len,
                "forevervalidator_search.ltoir");
    }
    if (linkResult == NVJITLINK_SUCCESS) {
        linkResult = nvJitLinkAddData(
                link,
                NVJITLINK_INPUT_LTOIR,
                facts.data(),
                facts.size(),
                "forevervalidator_session_facts.ltoir");
    }
    if (linkResult == NVJITLINK_SUCCESS) {
        linkResult = nvJitLinkComplete(link);
    }
    if (linkResult != NVJITLINK_SUCCESS) {
        if (diagnostic != nullptr) {
            *diagnostic =
                    "linking exact CUDA session specialization: error " +
                    std::to_string(static_cast<int>(linkResult)) +
                    "\n" + (link == nullptr ? std::string{} : LinkLog(link));
        }
        if (link != nullptr) {
            nvJitLinkDestroy(&link);
        }
        return false;
    }

    std::size_t cubinSize = 0u;
    if (nvJitLinkGetLinkedCubinSize(link, &cubinSize) !=
                NVJITLINK_SUCCESS) {
        if (diagnostic != nullptr) {
            *diagnostic = "reading specialized CUDA cubin size";
        }
        nvJitLinkDestroy(&link);
        return false;
    }
    std::vector<char> cubin(cubinSize);
    if (nvJitLinkGetLinkedCubin(link, cubin.data()) !=
                NVJITLINK_SUCCESS) {
        if (diagnostic != nullptr) {
            *diagnostic = "reading specialized CUDA cubin";
        }
        nvJitLinkDestroy(&link);
        return false;
    }
    nvJitLinkDestroy(&link);
    if (const char *dumpPath =
                std::getenv(
                        "FOREVERVALIDATOR_CUDA_DUMP_SPECIALIZATION");
        dumpPath != nullptr && dumpPath[0] != '\0') {
        std::ofstream dump(dumpPath, std::ios::binary);
        dump.write(
                cubin.data(),
                static_cast<std::streamsize>(cubin.size()));
    }

    CUresult driverResult =
            cuModuleLoadData(&module_, cubin.data());
    if (driverResult != CUDA_SUCCESS) {
        if (diagnostic != nullptr) {
            const char *message = nullptr;
            cuGetErrorString(driverResult, &message);
            *diagnostic =
                    "loading specialized CUDA module: " +
                    std::string(message == nullptr ? "unknown" : message);
        }
        Reset();
        return false;
    }
    const auto installPointer =
            [&](const char *name, std::uint64_t value) {
        CUdeviceptr address = 0u;
        std::size_t size = 0u;
        CUresult result = cuModuleGetGlobal(
                &address, &size, module_, name);
        if (result == CUDA_SUCCESS && size != sizeof(value)) {
            result = CUDA_ERROR_INVALID_VALUE;
        }
        if (result == CUDA_SUCCESS) {
            result = cuMemcpyHtoD(address, &value, sizeof(value));
        }
        if (result != CUDA_SUCCESS && diagnostic != nullptr) {
            const char *message = nullptr;
            cuGetErrorString(result, &message);
            *diagnostic =
                    "installing specialized CUDA session pointer " +
                    std::string(name) + ": " +
                    std::string(message == nullptr ? "unknown" : message);
        }
        return result == CUDA_SUCCESS;
    };
    if (!installPointer(
                "ForeverValidatorSessionConfigurationBaseStorage",
                configurationBase) ||
        !installPointer(
                "ForeverValidatorSessionSceneBaseStorage",
                sceneBase)) {
        Reset();
        return false;
    }
    unsigned int functionCount = 0u;
    driverResult =
            cuModuleGetFunctionCount(&functionCount, module_);
    std::vector<CUfunction> functions(functionCount);
    if (driverResult == CUDA_SUCCESS) {
        driverResult = cuModuleEnumerateFunctions(
                functions.data(), functionCount, module_);
    }
    if (driverResult != CUDA_SUCCESS) {
        if (diagnostic != nullptr) {
            *diagnostic =
                    "enumerating specialized CUDA kernels";
        }
        Reset();
        return false;
    }
    for (CUfunction function : functions) {
        const char *name = nullptr;
        if (cuFuncGetName(&name, function) != CUDA_SUCCESS ||
            name == nullptr ||
            std::string_view(name).find(
                    "SimulateSearchCandidatesKernel") ==
                    std::string_view::npos) {
            continue;
        }
        KernelEntry *destination = nullptr;
        const std::string_view functionName(name);
        if (functionName.find("ELj16ELb0EE") !=
            std::string_view::npos) {
            destination = &throughput_;
        } else if (functionName.find("ELj12ELb0EE") !=
                   std::string_view::npos) {
            destination = &tail_;
        } else if (functionName.find("ELj8ELb0EE") !=
                   std::string_view::npos) {
            destination = &denseTail_;
        }
        if (destination != nullptr) {
            destination->function = function;
        }
    }
    if (throughput_.function == nullptr ||
        tail_.function == nullptr ||
        denseTail_.function == nullptr ||
        !LoadMetrics(
                throughput_.function,
                &throughput_.metrics,
                diagnostic) ||
        !LoadMetrics(
                tail_.function,
                &tail_.metrics,
                diagnostic) ||
        !LoadMetrics(
                denseTail_.function,
                &denseTail_.metrics,
                diagnostic)) {
        if (diagnostic != nullptr && diagnostic->empty()) {
            *diagnostic =
                    "specialized CUDA simulation kernels are missing";
        }
        Reset();
        return false;
    }

    const double milliseconds =
            std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
    std::fprintf(
            stderr,
            "CUDA session specialization: %.1f ms, %zu-byte cubin\n",
            milliseconds,
            cubin.size());
    return true;
}

bool SessionModule::Ready() const noexcept {
    return module_ != nullptr;
}

CUfunction SessionModule::Kernel(
        std::uint32_t minimumBlocksPerMultiprocessor) const noexcept {
    if (minimumBlocksPerMultiprocessor == DenseTailMinimumBlocks) {
        return denseTail_.function;
    }
    if (minimumBlocksPerMultiprocessor == TailMinimumBlocks) {
        return tail_.function;
    }
    return throughput_.function;
}

const KernelMetrics &SessionModule::Metrics(
        std::uint32_t minimumBlocksPerMultiprocessor) const noexcept {
    if (minimumBlocksPerMultiprocessor == DenseTailMinimumBlocks) {
        return denseTail_.metrics;
    }
    if (minimumBlocksPerMultiprocessor == TailMinimumBlocks) {
        return tail_.metrics;
    }
    return throughput_.metrics;
}

}  // namespace forevervalidator::simulation::cuda::specialization
