#include "simulation/backends/vulkan/vulkan_compute_runtime.h"

#if FOREVERVALIDATOR_HAS_VULKAN

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "forevervalidator_vulkan_timeline_spirv.h"
#include "forevervalidator_vulkan_finish_refinement_spirv.h"
#include "forevervalidator_vulkan_finish_probe_spirv.h"
#include "forevervalidator_vulkan_search_generate_spirv.h"
#include "forevervalidator_vulkan_search_initialize_spirv.h"
#include "forevervalidator_vulkan_search_prepare_spirv.h"
#include "forevervalidator_vulkan_search_physics_spirv.h"
#include "forevervalidator_vulkan_search_physics_steady_velocity_water_spirv.h"
#include "forevervalidator_vulkan_search_physics_steady_velocity_water_force_spirv.h"
#include "forevervalidator_vulkan_search_physics_steady_velocity_water_detect_spirv.h"
#include "forevervalidator_vulkan_search_physics_steady_velocity_water_respond_spirv.h"
#include "forevervalidator_vulkan_search_evaluate_spirv.h"

namespace forevervalidator::simulation::vulkan {
namespace {

constexpr std::uint32_t TimelineThreads = 32u;
constexpr std::uint32_t ComputeThreads = 32u;
constexpr std::uint32_t SteadyVelocitySearchThreads = 16u;
constexpr std::uint32_t ComputePushConstantBytes = 20u;
constexpr VkDeviceSize WorkspaceAlignment = 256u;
constexpr std::uint32_t PushConstantBytes = 84u;
constexpr std::size_t MaximumPipelineCacheBytes =
        1024ull * 1024ull * 1024ull;

constexpr std::uint32_t ComputeThreadsForKernel(ComputeKernel kernel) {
    return kernel == ComputeKernel::SearchPhysicsSteadyVelocityWater ||
                   kernel == ComputeKernel::SearchPhysicsForcePhase ||
                   kernel == ComputeKernel::SearchPhysicsDetectPhase ||
                   kernel == ComputeKernel::SearchPhysicsRespondPhase
            ? SteadyVelocitySearchThreads
            : ComputeThreads;
}

std::string VulkanFailure(const char *operation, VkResult result) {
    return std::string(operation) + " failed with VkResult " +
           std::to_string(static_cast<int>(result));
}

bool HasName(const std::vector<VkLayerProperties> &values,
             const char *name) {
    return std::any_of(values.begin(), values.end(),
                       [name](const VkLayerProperties &value) {
                           return std::strcmp(value.layerName, name) == 0;
                       });
}

bool HasName(const std::vector<VkExtensionProperties> &values,
             const char *name) {
    return std::any_of(values.begin(), values.end(),
                       [name](const VkExtensionProperties &value) {
                           return std::strcmp(value.extensionName, name) == 0;
                       });
}

class Runtime;

}  // namespace

class Buffer {
public:
    ~Buffer();
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    // Internal runtime object; fields are public only within this private
    // implementation translation unit.
    Buffer() = default;

    std::shared_ptr<Runtime> runtime_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0u;
    VkDeviceAddress address_ = 0u;
    bool coherent_ = false;
};

namespace {

struct DeviceCandidate {
    VkPhysicalDevice device = VK_NULL_HANDLE;
    std::uint32_t queueFamily = UINT32_MAX;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory{};
    std::uint32_t score = 0u;
    std::string rejection;
};

class Runtime : public std::enable_shared_from_this<Runtime> {
public:
    ~Runtime() {
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
            SavePipelineCache();
            if (queryPool_ != VK_NULL_HANDLE) {
                vkDestroyQueryPool(device_, queryPool_, nullptr);
            }
            if (pipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, pipeline_, nullptr);
            }
            if (finishRefinementPipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(
                        device_, finishRefinementPipeline_, nullptr);
            }
            if (finishProbePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, finishProbePipeline_, nullptr);
            }
            if (searchGeneratePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, searchGeneratePipeline_, nullptr);
            }
            if (searchInitializePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, searchInitializePipeline_, nullptr);
            }
            if (searchPreparePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, searchPreparePipeline_, nullptr);
            }
            if (searchPhysicsPipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, searchPhysicsPipeline_, nullptr);
            }
            if (searchPhysicsSteadyVelocityWaterPipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(
                        device_, searchPhysicsSteadyVelocityWaterPipeline_,
                        nullptr);
            }
            if (searchPhysicsForcePhasePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(
                        device_, searchPhysicsForcePhasePipeline_, nullptr);
            }
            if (searchPhysicsDetectPhasePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(
                        device_, searchPhysicsDetectPhasePipeline_, nullptr);
            }
            if (searchPhysicsRespondPhasePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(
                        device_, searchPhysicsRespondPhasePipeline_, nullptr);
            }
            if (searchEvaluatePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, searchEvaluatePipeline_, nullptr);
            }
            if (pipelineLayout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
            }
            if (pipelineCache_ != VK_NULL_HANDLE) {
                vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
            }
            if (commandPool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device_, commandPool_, nullptr);
            }
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
        }
    }

    RuntimeDiagnostics Initialize() {
        RuntimeDiagnostics result;
        result.status = RuntimeStatus::InitializationFailed;

        std::uint32_t loaderVersion = VK_API_VERSION_1_0;
        if (vkEnumerateInstanceVersion(&loaderVersion) != VK_SUCCESS ||
            loaderVersion < VK_API_VERSION_1_2) {
            result.status = RuntimeStatus::LoaderUnavailable;
            result.diagnostic = "Vulkan 1.2 loader is required";
            return diagnostics_ = result;
        }

        std::uint32_t extensionCount = 0u;
        vkEnumerateInstanceExtensionProperties(
                nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        if (extensionCount != 0u) {
            vkEnumerateInstanceExtensionProperties(
                    nullptr, &extensionCount, extensions.data());
        }
        std::vector<const char *> enabledExtensions;
        VkInstanceCreateFlags instanceFlags = 0u;
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
        if (HasName(extensions,
                    VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            enabledExtensions.push_back(
                    VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instanceFlags |=
                    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif

        std::uint32_t layerCount = 0u;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layers(layerCount);
        if (layerCount != 0u) {
            vkEnumerateInstanceLayerProperties(
                    &layerCount, layers.data());
        }
        std::vector<const char *> enabledLayers;
        const char *validation = std::getenv(
                "FOREVERVALIDATOR_VULKAN_VALIDATION");
        if (validation != nullptr && validation[0] != '\0' &&
            validation[0] != '0' &&
            HasName(layers, "VK_LAYER_KHRONOS_validation")) {
            enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
        }

        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "ForeverValidator";
        application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 2, 3);
        application.pEngineName = "ForeverValidator Vulkan Compute";
        application.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        application.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo instanceInfo{
                VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.flags = instanceFlags;
        instanceInfo.pApplicationInfo = &application;
        instanceInfo.enabledExtensionCount =
                static_cast<std::uint32_t>(enabledExtensions.size());
        instanceInfo.ppEnabledExtensionNames = enabledExtensions.data();
        instanceInfo.enabledLayerCount =
                static_cast<std::uint32_t>(enabledLayers.size());
        instanceInfo.ppEnabledLayerNames = enabledLayers.data();
        VkResult status = vkCreateInstance(
                &instanceInfo, nullptr, &instance_);
        if (status != VK_SUCCESS) {
            result.diagnostic = VulkanFailure(
                    "Vulkan instance creation", status);
            return diagnostics_ = result;
        }

        std::uint32_t deviceCount = 0u;
        status = vkEnumeratePhysicalDevices(
                instance_, &deviceCount, nullptr);
        if (status != VK_SUCCESS || deviceCount == 0u) {
            result.status = RuntimeStatus::NoDevice;
            result.diagnostic = deviceCount == 0u
                    ? "no Vulkan physical device was found"
                    : VulkanFailure("Vulkan device enumeration", status);
            return diagnostics_ = result;
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        status = vkEnumeratePhysicalDevices(
                instance_, &deviceCount, devices.data());
        if (status != VK_SUCCESS) {
            result.status = RuntimeStatus::NoDevice;
            result.diagnostic = VulkanFailure(
                    "Vulkan device enumeration", status);
            return diagnostics_ = result;
        }

        DeviceCandidate best;
        for (VkPhysicalDevice device : devices) {
            DeviceCandidate candidate = InspectDevice(device);
            if (candidate.device != VK_NULL_HANDLE &&
                (best.device == VK_NULL_HANDLE ||
                 candidate.score > best.score)) {
                best = std::move(candidate);
            } else if (!candidate.rejection.empty() &&
                       best.rejection.empty()) {
                best.rejection = std::move(candidate.rejection);
            }
        }
        if (best.device == VK_NULL_HANDLE) {
            result.status = RuntimeStatus::UnsupportedDevice;
            result.diagnostic = best.rejection.empty()
                    ? "no Vulkan device supports the required compute ABI"
                    : best.rejection;
            return diagnostics_ = result;
        }
        physicalDevice_ = best.device;
        properties_ = best.properties;
        memoryProperties_ = best.memory;
        queueFamily_ = best.queueFamily;

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = queueFamily_;
        queueInfo.queueCount = 1u;
        queueInfo.pQueuePriorities = &queuePriority;

        VkPhysicalDeviceVulkan12Features features12{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        features12.storageBuffer8BitAccess = VK_TRUE;
        features12.shaderInt8 = VK_TRUE;
        features12.scalarBlockLayout = VK_TRUE;
        features12.bufferDeviceAddress = VK_TRUE;
        VkPhysicalDeviceVulkan11Features features11{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        features11.pNext = &features12;
        features11.storageBuffer16BitAccess = VK_TRUE;
        VkPhysicalDeviceFeatures2 features{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &features11;
        features.features.shaderInt64 = VK_TRUE;
        features.features.shaderInt16 = VK_TRUE;
        features.features.shaderFloat64 = VK_TRUE;

        std::uint32_t deviceExtensionCount = 0u;
        vkEnumerateDeviceExtensionProperties(
                physicalDevice_, nullptr,
                &deviceExtensionCount, nullptr);
        std::vector<VkExtensionProperties> deviceExtensions(
                deviceExtensionCount);
        if (deviceExtensionCount != 0u) {
            vkEnumerateDeviceExtensionProperties(
                    physicalDevice_, nullptr,
                    &deviceExtensionCount, deviceExtensions.data());
        }
        std::vector<const char *> enabledDeviceExtensions;
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
        if (HasName(deviceExtensions,
                    VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
            enabledDeviceExtensions.push_back(
                    VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }
#endif
#ifdef VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME
        VkPhysicalDeviceShaderFloatControls2FeaturesKHR
                floatControls2Support{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 floatControls2SupportQuery{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        floatControls2SupportQuery.pNext = &floatControls2Support;
        vkGetPhysicalDeviceFeatures2(
                physicalDevice_, &floatControls2SupportQuery);
        VkPhysicalDeviceFloatControlsProperties floatControlsProperties{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        VkPhysicalDeviceProperties2 floatControlsPropertiesQuery{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        floatControlsPropertiesQuery.pNext = &floatControlsProperties;
        vkGetPhysicalDeviceProperties2(
                physicalDevice_, &floatControlsPropertiesQuery);
        exactSearchPhysicsSupported_ =
                HasName(deviceExtensions,
                        VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME) &&
                floatControls2Support.shaderFloatControls2 &&
                floatControlsProperties.shaderRoundingModeRTEFloat32 &&
                floatControlsProperties.shaderSignedZeroInfNanPreserveFloat32;
        VkPhysicalDeviceShaderFloatControls2FeaturesKHR
                floatControls2Features{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR};
        if (exactSearchPhysicsSupported_) {
            floatControls2Features.shaderFloatControls2 = VK_TRUE;
            floatControls2Features.pNext = features.pNext;
            features.pNext = &floatControls2Features;
            enabledDeviceExtensions.push_back(
                    VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME);
        }
#endif
        VkDeviceCreateInfo deviceInfo{
                VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.pNext = &features;
        deviceInfo.queueCreateInfoCount = 1u;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.enabledExtensionCount =
                static_cast<std::uint32_t>(
                        enabledDeviceExtensions.size());
        deviceInfo.ppEnabledExtensionNames =
                enabledDeviceExtensions.data();
        status = vkCreateDevice(
                physicalDevice_, &deviceInfo, nullptr, &device_);
        if (status != VK_SUCCESS) {
            result.diagnostic = VulkanFailure(
                    "Vulkan logical device creation", status);
            return diagnostics_ = result;
        }
        vkGetDeviceQueue(device_, queueFamily_, 0u, &queue_);

        VkCommandPoolCreateInfo poolInfo{
                VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamily_;
        status = vkCreateCommandPool(
                device_, &poolInfo, nullptr, &commandPool_);
        if (status != VK_SUCCESS) {
            result.diagnostic = VulkanFailure(
                    "Vulkan command pool creation", status);
            return diagnostics_ = result;
        }

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0u;
        pushRange.size = PushConstantBytes;
        VkPipelineLayoutCreateInfo layoutInfo{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.pushConstantRangeCount = 1u;
        layoutInfo.pPushConstantRanges = &pushRange;
        status = vkCreatePipelineLayout(
                device_, &layoutInfo, nullptr, &pipelineLayout_);
        if (status != VK_SUCCESS) {
            result.diagnostic = VulkanFailure(
                    "Vulkan pipeline layout creation", status);
            return diagnostics_ = result;
        }

        const std::vector<std::byte> cachedPipeline = LoadPipelineCache();
        VkPipelineCacheCreateInfo cacheInfo{
                VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        cacheInfo.initialDataSize = cachedPipeline.size();
        cacheInfo.pInitialData = cachedPipeline.data();
        status = vkCreatePipelineCache(
                device_, &cacheInfo, nullptr, &pipelineCache_);
        if (status != VK_SUCCESS && !cachedPipeline.empty()) {
            cacheInfo.initialDataSize = 0u;
            cacheInfo.pInitialData = nullptr;
            status = vkCreatePipelineCache(
                    device_, &cacheInfo, nullptr, &pipelineCache_);
        }
        if (status != VK_SUCCESS) {
            result.diagnostic = VulkanFailure(
                    "Vulkan pipeline cache creation", status);
            return diagnostics_ = result;
        }

        if (ForeverValidatorVulkanTimelineSpirv_len == 0u ||
            (ForeverValidatorVulkanTimelineSpirv_len & 3u) != 0u) {
            result.diagnostic = "embedded Vulkan timeline SPIR-V is invalid";
            return diagnostics_ = result;
        }
        VkShaderModuleCreateInfo shaderInfo{
                VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shaderInfo.codeSize = ForeverValidatorVulkanTimelineSpirv_len;
        shaderInfo.pCode = reinterpret_cast<const std::uint32_t *>(
                ForeverValidatorVulkanTimelineSpirv);
        VkShaderModule shader = VK_NULL_HANDLE;
        status = vkCreateShaderModule(
                device_, &shaderInfo, nullptr, &shader);
        if (status != VK_SUCCESS) {
            result.diagnostic = VulkanFailure(
                    "Vulkan shader module creation", status);
            return diagnostics_ = result;
        }
        VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipelineInfo{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = stage;
        pipelineInfo.layout = pipelineLayout_;
        status = vkCreateComputePipelines(
                device_, pipelineCache_, 1u,
                &pipelineInfo, nullptr, &pipeline_);
        vkDestroyShaderModule(device_, shader, nullptr);
        if (status != VK_SUCCESS) {
            result.diagnostic = VulkanFailure(
                    "Vulkan timeline pipeline creation", status);
            return diagnostics_ = result;
        }
        if (ForeverValidatorVulkanFinishRefinementSpirv_len == 0u ||
            (ForeverValidatorVulkanFinishRefinementSpirv_len & 3u) != 0u) {
            result.diagnostic =
                    "embedded Vulkan finish-refinement SPIR-V is invalid";
            return diagnostics_ = result;
        }
        if (ForeverValidatorVulkanFinishProbeSpirv_len == 0u ||
            (ForeverValidatorVulkanFinishProbeSpirv_len & 3u) != 0u) {
            result.diagnostic =
                    "embedded Vulkan finish-probe SPIR-V is invalid";
            return diagnostics_ = result;
        }
        VkQueryPoolCreateInfo queryInfo{
                VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount = 2u;
        status = vkCreateQueryPool(
                device_, &queryInfo, nullptr, &queryPool_);
        if (status != VK_SUCCESS) {
            result.diagnostic = VulkanFailure(
                    "Vulkan timestamp query creation", status);
            return diagnostics_ = result;
        }

        VkPhysicalDeviceSubgroupProperties subgroup{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceDriverProperties driver{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        subgroup.pNext = &driver;
        VkPhysicalDeviceProperties2 properties2{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties2.pNext = &subgroup;
        vkGetPhysicalDeviceProperties2(physicalDevice_, &properties2);

        result.status = RuntimeStatus::Ready;
        result.apiVersion = properties_.apiVersion;
        result.driverVersion = properties_.driverVersion;
        result.vendorId = properties_.vendorID;
        result.deviceId = properties_.deviceID;
        result.subgroupSize = subgroup.subgroupSize;
        result.deviceName = properties_.deviceName;
        result.driverName = driver.driverName;
        for (std::uint32_t index = 0u;
             index < memoryProperties_.memoryHeapCount; ++index) {
            if ((memoryProperties_.memoryHeaps[index].flags &
                 VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0u) {
                result.deviceLocalMemoryBytes +=
                        memoryProperties_.memoryHeaps[index].size;
            }
        }
        result.diagnostic = std::string("Vulkan compute ready on ") +
                            result.deviceName;
        return diagnostics_ = result;
    }

    const RuntimeDiagnostics &Diagnostics() const { return diagnostics_; }

    bool CreateBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags required,
                      VkMemoryPropertyFlags preferred,
                      bool deviceAddress,
                      BufferHandle *output,
                      std::string *diagnostic) {
        if (output == nullptr || size == 0u) return false;
        auto buffer = std::shared_ptr<Buffer>(new Buffer());
        buffer->runtime_ = shared_from_this();
        buffer->size_ = size;

        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult status = vkCreateBuffer(
                device_, &info, nullptr, &buffer->buffer_);
        if (status != VK_SUCCESS) {
            if (diagnostic) *diagnostic = VulkanFailure(
                    "Vulkan buffer creation", status);
            return false;
        }
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(
                device_, buffer->buffer_, &requirements);
        const std::uint32_t memoryType = FindMemoryType(
                requirements.memoryTypeBits, required, preferred);
        if (memoryType == UINT32_MAX) {
            if (diagnostic) {
                *diagnostic = "no compatible Vulkan memory type";
            }
            return false;
        }
        VkMemoryAllocateFlagsInfo flags{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = deviceAddress
                ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0u;
        VkMemoryAllocateInfo allocation{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.pNext = deviceAddress ? &flags : nullptr;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memoryType;
        status = vkAllocateMemory(
                device_, &allocation, nullptr, &buffer->memory_);
        if (status != VK_SUCCESS) {
            if (diagnostic) *diagnostic = VulkanFailure(
                    "Vulkan memory allocation", status);
            return false;
        }
        status = vkBindBufferMemory(
                device_, buffer->buffer_, buffer->memory_, 0u);
        if (status != VK_SUCCESS) {
            if (diagnostic) *diagnostic = VulkanFailure(
                    "Vulkan buffer binding", status);
            return false;
        }
        buffer->coherent_ =
                (memoryProperties_.memoryTypes[memoryType].propertyFlags &
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u;
        if (deviceAddress) {
            VkBufferDeviceAddressInfo addressInfo{
                    VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            addressInfo.buffer = buffer->buffer_;
            buffer->address_ = vkGetBufferDeviceAddress(
                    device_, &addressInfo);
            if (buffer->address_ == 0u) {
                if (diagnostic) {
                    *diagnostic = "Vulkan buffer device address is zero";
                }
                return false;
            }
        }
        *output = std::move(buffer);
        return true;
    }

    bool Map(const BufferHandle &buffer, void **data,
             std::string *diagnostic) {
        const VkResult status = vkMapMemory(
                device_, buffer->memory_, 0u,
                VK_WHOLE_SIZE, 0u, data);
        if (status != VK_SUCCESS) {
            if (diagnostic) *diagnostic = VulkanFailure(
                    "Vulkan memory mapping", status);
            return false;
        }
        return true;
    }

    bool Flush(const BufferHandle &buffer, std::string *diagnostic) {
        if (buffer->coherent_) return true;
        VkMappedMemoryRange range{
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = buffer->memory_;
        range.offset = 0u;
        range.size = VK_WHOLE_SIZE;
        const VkResult status = vkFlushMappedMemoryRanges(
                device_, 1u, &range);
        if (status != VK_SUCCESS && diagnostic) {
            *diagnostic = VulkanFailure(
                    "Vulkan mapped memory flush", status);
        }
        return status == VK_SUCCESS;
    }

    bool Invalidate(const BufferHandle &buffer,
                    std::string *diagnostic) {
        if (buffer->coherent_) return true;
        VkMappedMemoryRange range{
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = buffer->memory_;
        range.offset = 0u;
        range.size = VK_WHOLE_SIZE;
        const VkResult status = vkInvalidateMappedMemoryRanges(
                device_, 1u, &range);
        if (status != VK_SUCCESS && diagnostic) {
            *diagnostic = VulkanFailure(
                    "Vulkan mapped memory invalidation", status);
        }
        return status == VK_SUCCESS;
    }

    void Unmap(const BufferHandle &buffer) {
        vkUnmapMemory(device_, buffer->memory_);
    }

    bool Submit(const std::function<void(VkCommandBuffer)> &record,
                double *waitMilliseconds,
                std::string *diagnostic,
                const std::function<void()> &poll = {}) {
        VkCommandBufferAllocateInfo allocation{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = commandPool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1u;
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkResult status = vkAllocateCommandBuffers(
                device_, &allocation, &command);
        if (status != VK_SUCCESS) {
            if (diagnostic) *diagnostic = VulkanFailure(
                    "Vulkan command allocation", status);
            return false;
        }
        VkCommandBufferBeginInfo begin{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        status = vkBeginCommandBuffer(command, &begin);
        if (status == VK_SUCCESS) {
            record(command);
            status = vkEndCommandBuffer(command);
        }
        VkFence fence = VK_NULL_HANDLE;
        if (status == VK_SUCCESS) {
            VkFenceCreateInfo fenceInfo{
                    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            status = vkCreateFence(
                    device_, &fenceInfo, nullptr, &fence);
        }
        if (status == VK_SUCCESS) {
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1u;
            submit.pCommandBuffers = &command;
            status = vkQueueSubmit(queue_, 1u, &submit, fence);
        }
        const auto waitStart = std::chrono::steady_clock::now();
        if (status == VK_SUCCESS) {
            do {
                status = vkWaitForFences(
                        device_, 1u, &fence, VK_TRUE,
                        poll ? 0u
                             : std::numeric_limits<std::uint64_t>::max());
                if (status == VK_TIMEOUT && poll) poll();
            } while (status == VK_TIMEOUT);
        }
        const auto waitEnd = std::chrono::steady_clock::now();
        if (waitMilliseconds) {
            *waitMilliseconds =
                    std::chrono::duration<double, std::milli>(
                            waitEnd - waitStart).count();
        }
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device_, fence, nullptr);
        }
        vkFreeCommandBuffers(device_, commandPool_, 1u, &command);
        if (status != VK_SUCCESS) {
            if (diagnostic) *diagnostic = VulkanFailure(
                    "Vulkan queue submission", status);
            return false;
        }
        return true;
    }

    std::mutex &Mutex() { return mutex_; }

    std::filesystem::path PipelineCachePath() const {
        const char *xdg = std::getenv("XDG_CACHE_HOME");
        std::filesystem::path root;
        if (xdg != nullptr && xdg[0] != '\0') {
            root = xdg;
        } else {
            const char *home = std::getenv("HOME");
            if (home == nullptr || home[0] == '\0') return {};
            root = std::filesystem::path(home) / ".cache";
        }
        return root / "forevervalidator" /
               ("vulkan-pipeline-" +
                std::to_string(properties_.vendorID) + "-" +
                std::to_string(properties_.deviceID) + "-" +
                std::to_string(properties_.driverVersion) + ".bin");
    }

    std::vector<std::byte> LoadPipelineCache() const {
        std::vector<std::byte> result;
        try {
            const std::filesystem::path path = PipelineCachePath();
            if (path.empty()) return result;
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input) return result;
            const std::streamoff size = input.tellg();
            if (size <= 0 ||
                static_cast<std::uint64_t>(size) >
                        MaximumPipelineCacheBytes) {
                return result;
            }
            result.resize(static_cast<std::size_t>(size));
            input.seekg(0, std::ios::beg);
            input.read(reinterpret_cast<char *>(result.data()), size);
            if (!input) result.clear();
        } catch (...) {
            result.clear();
        }
        return result;
    }

    void SavePipelineCache() const noexcept {
        if (pipelineCache_ == VK_NULL_HANDLE) return;
        try {
            std::size_t size = 0u;
            if (vkGetPipelineCacheData(
                        device_, pipelineCache_, &size, nullptr) !=
                        VK_SUCCESS ||
                size == 0u || size > MaximumPipelineCacheBytes) {
                return;
            }
            std::vector<std::byte> data(size);
            if (vkGetPipelineCacheData(
                        device_, pipelineCache_, &size, data.data()) !=
                VK_SUCCESS) {
                return;
            }
            const std::filesystem::path path = PipelineCachePath();
            if (path.empty()) return;
            std::filesystem::create_directories(path.parent_path());
            const std::filesystem::path temporary = path.string() + ".tmp";
            {
                std::ofstream output(
                        temporary,
                        std::ios::binary | std::ios::trunc);
                if (!output) return;
                output.write(
                        reinterpret_cast<const char *>(data.data()),
                        static_cast<std::streamsize>(size));
                if (!output) return;
            }
            std::filesystem::rename(temporary, path);
        } catch (...) {
        }
    }
    VkDevice Device() const { return device_; }
    VkPipeline Pipeline() const { return pipeline_; }
    bool EnsureFinishRefinementPipelines(std::string *diagnostic) {
        const auto create = [&](const unsigned char *code,
                                std::size_t codeBytes,
                                const char *operation,
                                VkPipeline *pipeline) {
            if (*pipeline != VK_NULL_HANDLE) return true;
            VkShaderModuleCreateInfo shaderInfo{
                    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shaderInfo.codeSize = codeBytes;
            shaderInfo.pCode =
                    reinterpret_cast<const std::uint32_t *>(code);
            VkShaderModule shader = VK_NULL_HANDLE;
            VkResult status = vkCreateShaderModule(
                    device_, &shaderInfo, nullptr, &shader);
            if (status == VK_SUCCESS) {
                VkPipelineShaderStageCreateInfo stage{
                        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
                stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                stage.module = shader;
                stage.pName = "main";
                VkComputePipelineCreateInfo pipelineInfo{
                        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                pipelineInfo.stage = stage;
                pipelineInfo.layout = pipelineLayout_;
                status = vkCreateComputePipelines(
                        device_, pipelineCache_, 1u,
                        &pipelineInfo, nullptr, pipeline);
            }
            if (shader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, shader, nullptr);
            }
            if (status != VK_SUCCESS) {
                if (diagnostic) {
                    *diagnostic = VulkanFailure(operation, status);
                }
                return false;
            }
            return true;
        };
        return create(
                       ForeverValidatorVulkanFinishRefinementSpirv,
                       ForeverValidatorVulkanFinishRefinementSpirv_len,
                       "Vulkan finish-location pipeline creation",
                       &finishRefinementPipeline_) &&
               create(
                       ForeverValidatorVulkanFinishProbeSpirv,
                       ForeverValidatorVulkanFinishProbeSpirv_len,
                       "Vulkan finish-probe pipeline creation",
                       &finishProbePipeline_);
    }
    VkPipeline FinishRefinementPipeline() const {
        return finishRefinementPipeline_;
    }
    VkPipeline FinishProbePipeline() const {
        return finishProbePipeline_;
    }
    bool EnsureComputePipeline(
            ComputeKernel kernel,
            std::string *diagnostic) {
        VkPipeline *destination = nullptr;
        const unsigned char *code = nullptr;
        std::size_t codeBytes = 0u;
        const char *operation = nullptr;
        switch (kernel) {
        case ComputeKernel::SearchGenerate:
            destination = &searchGeneratePipeline_;
            code = ForeverValidatorVulkanSearchGenerateSpirv;
            codeBytes = ForeverValidatorVulkanSearchGenerateSpirv_len;
            operation = "Vulkan search-generation";
            break;
        case ComputeKernel::SearchInitialize:
            destination = &searchInitializePipeline_;
            code = ForeverValidatorVulkanSearchInitializeSpirv;
            codeBytes = ForeverValidatorVulkanSearchInitializeSpirv_len;
            operation = "Vulkan search-initialization";
            break;
        case ComputeKernel::SearchPrepare:
            destination = &searchPreparePipeline_;
            code = ForeverValidatorVulkanSearchPrepareSpirv;
            codeBytes = ForeverValidatorVulkanSearchPrepareSpirv_len;
            operation = "Vulkan search tick preparation";
            break;
        case ComputeKernel::SearchPhysics:
            destination = &searchPhysicsPipeline_;
            code = ForeverValidatorVulkanSearchPhysicsSpirv;
            codeBytes = ForeverValidatorVulkanSearchPhysicsSpirv_len;
            operation = "Vulkan search physics";
            break;
        case ComputeKernel::SearchPhysicsSteadyVelocityWater:
            destination = &searchPhysicsSteadyVelocityWaterPipeline_;
            code = ForeverValidatorVulkanSearchPhysicsSteadyVelocityWaterSpirv;
            codeBytes =
                    ForeverValidatorVulkanSearchPhysicsSteadyVelocityWaterSpirv_len;
            operation = "Vulkan water steady velocity search physics";
            break;
        case ComputeKernel::SearchPhysicsForcePhase:
            destination = &searchPhysicsForcePhasePipeline_;
            code =
                    ForeverValidatorVulkanSearchPhysicsSteadyVelocityWaterFORCESpirv;
            codeBytes =
                    ForeverValidatorVulkanSearchPhysicsSteadyVelocityWaterFORCESpirv_len;
            operation = "Vulkan split search force phase";
            break;
        case ComputeKernel::SearchPhysicsDetectPhase:
            destination = &searchPhysicsDetectPhasePipeline_;
            code =
                    ForeverValidatorVulkanSearchPhysicsSteadyVelocityWaterDETECTSpirv;
            codeBytes =
                    ForeverValidatorVulkanSearchPhysicsSteadyVelocityWaterDETECTSpirv_len;
            operation = "Vulkan split search detection phase";
            break;
        case ComputeKernel::SearchPhysicsRespondPhase:
            destination = &searchPhysicsRespondPhasePipeline_;
            code =
                    ForeverValidatorVulkanSearchPhysicsSteadyVelocityWaterRESPONDSpirv;
            codeBytes =
                    ForeverValidatorVulkanSearchPhysicsSteadyVelocityWaterRESPONDSpirv_len;
            operation = "Vulkan split search response phase";
            break;
        case ComputeKernel::Timeline:
            return pipeline_ != VK_NULL_HANDLE;
        case ComputeKernel::SearchEvaluate:
            destination = &searchEvaluatePipeline_;
            code = ForeverValidatorVulkanSearchEvaluateSpirv;
            codeBytes = ForeverValidatorVulkanSearchEvaluateSpirv_len;
            operation = "Vulkan search tick evaluation";
            break;
        }
        if (*destination != VK_NULL_HANDLE) return true;
        if (codeBytes == 0u || (codeBytes & 3u) != 0u) {
            if (diagnostic) {
                *diagnostic = std::string(operation) +
                              " SPIR-V is invalid";
            }
            return false;
        }
        VkShaderModuleCreateInfo shaderInfo{
                VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shaderInfo.codeSize = codeBytes;
        shaderInfo.pCode = reinterpret_cast<const std::uint32_t *>(code);
        VkShaderModule shader = VK_NULL_HANDLE;
        VkResult status = vkCreateShaderModule(
                device_, &shaderInfo, nullptr, &shader);
        if (status != VK_SUCCESS) {
            if (diagnostic) {
                *diagnostic = VulkanFailure(
                        (std::string(operation) +
                         " shader module creation").c_str(),
                        status);
            }
            return false;
        }
        VkPipelineShaderStageCreateInfo stage{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipelineInfo{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = stage;
        pipelineInfo.layout = pipelineLayout_;
        status = vkCreateComputePipelines(
                device_, pipelineCache_, 1u,
                &pipelineInfo, nullptr, destination);
        vkDestroyShaderModule(device_, shader, nullptr);
        if (status != VK_SUCCESS) {
            if (diagnostic) {
                *diagnostic = VulkanFailure(
                        (std::string(operation) +
                         " pipeline creation").c_str(),
                        status);
            }
            return false;
        }
        return true;
    }
    VkPipeline ComputePipeline(ComputeKernel kernel) const {
        switch (kernel) {
        case ComputeKernel::SearchGenerate:
            return searchGeneratePipeline_;
        case ComputeKernel::SearchInitialize:
            return searchInitializePipeline_;
        case ComputeKernel::SearchPrepare:
            return searchPreparePipeline_;
        case ComputeKernel::SearchPhysics:
            return searchPhysicsPipeline_;
        case ComputeKernel::SearchPhysicsSteadyVelocityWater:
            return searchPhysicsSteadyVelocityWaterPipeline_;
        case ComputeKernel::SearchPhysicsForcePhase:
            return searchPhysicsForcePhasePipeline_;
        case ComputeKernel::SearchPhysicsDetectPhase:
            return searchPhysicsDetectPhasePipeline_;
        case ComputeKernel::SearchPhysicsRespondPhase:
            return searchPhysicsRespondPhasePipeline_;
        case ComputeKernel::Timeline:
            return pipeline_;
        case ComputeKernel::SearchEvaluate:
            return searchEvaluatePipeline_;
        }
        return VK_NULL_HANDLE;
    }
    VkPipelineLayout PipelineLayout() const { return pipelineLayout_; }
    VkQueryPool QueryPool() const { return queryPool_; }
    float TimestampPeriod() const { return properties_.limits.timestampPeriod; }
    std::uint32_t TimestampBits() const { return timestampValidBits_; }
    std::uint32_t MaxDispatchX() const {
        return properties_.limits.maxComputeWorkGroupCount[0];
    }
    bool SupportsExactSearchPhysics() const {
        return exactSearchPhysicsSupported_;
    }

private:
    DeviceCandidate InspectDevice(VkPhysicalDevice device) {
        DeviceCandidate result;
        VkPhysicalDeviceVulkan12Features features12{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features features11{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        features11.pNext = &features12;
        VkPhysicalDeviceFeatures2 features{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &features11;
        vkGetPhysicalDeviceFeatures2(device, &features);
        vkGetPhysicalDeviceProperties(device, &result.properties);
        vkGetPhysicalDeviceMemoryProperties(device, &result.memory);

        if (result.properties.apiVersion < VK_API_VERSION_1_2 ||
            !features.features.shaderInt64 ||
            !features.features.shaderInt16 ||
            !features.features.shaderFloat64 ||
            !features11.storageBuffer16BitAccess ||
            !features12.storageBuffer8BitAccess ||
            !features12.shaderInt8 ||
            !features12.scalarBlockLayout ||
            !features12.bufferDeviceAddress ||
            result.properties.limits.maxPushConstantsSize <
                    PushConstantBytes ||
            result.properties.limits.maxComputeWorkGroupInvocations <
                    TimelineThreads ||
            result.properties.limits.maxComputeWorkGroupSize[0] <
                    TimelineThreads) {
            result.rejection = std::string(result.properties.deviceName) +
                    " lacks a required Vulkan 1.2 compute feature";
            return result;
        }

        std::uint32_t queueCount = 0u;
        vkGetPhysicalDeviceQueueFamilyProperties(
                device, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(
                device, &queueCount, queues.data());
        std::uint32_t fallback = UINT32_MAX;
        for (std::uint32_t index = 0u; index < queueCount; ++index) {
            if ((queues[index].queueFlags &
                 VK_QUEUE_COMPUTE_BIT) == 0u ||
                queues[index].queueCount == 0u) {
                continue;
            }
            if (fallback == UINT32_MAX) fallback = index;
            if ((queues[index].queueFlags &
                 VK_QUEUE_GRAPHICS_BIT) == 0u) {
                fallback = index;
                break;
            }
        }
        if (fallback == UINT32_MAX) {
            result.rejection = std::string(result.properties.deviceName) +
                               " has no compute queue";
            return result;
        }
        result.device = device;
        result.queueFamily = fallback;
        timestampValidBits_ = queues[fallback].timestampValidBits;
        result.score = result.properties.deviceType ==
                               VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                ? 300u
                : (result.properties.deviceType ==
                           VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
                       ? 200u : 100u);
        if ((queues[fallback].queueFlags &
             VK_QUEUE_GRAPHICS_BIT) == 0u) {
            result.score += 10u;
        }
        return result;
    }

    std::uint32_t FindMemoryType(
            std::uint32_t typeBits,
            VkMemoryPropertyFlags required,
            VkMemoryPropertyFlags preferred) const {
        std::uint32_t fallback = UINT32_MAX;
        for (std::uint32_t index = 0u;
             index < memoryProperties_.memoryTypeCount; ++index) {
            if ((typeBits & (1u << index)) == 0u) continue;
            const VkMemoryPropertyFlags flags =
                    memoryProperties_.memoryTypes[index].propertyFlags;
            if ((flags & required) != required) continue;
            if ((flags & preferred) == preferred) return index;
            if (fallback == UINT32_MAX) fallback = index;
        }
        return fallback;
    }

    RuntimeDiagnostics diagnostics_;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    std::uint32_t queueFamily_ = UINT32_MAX;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipeline finishRefinementPipeline_ = VK_NULL_HANDLE;
    VkPipeline finishProbePipeline_ = VK_NULL_HANDLE;
    VkPipeline searchGeneratePipeline_ = VK_NULL_HANDLE;
    VkPipeline searchInitializePipeline_ = VK_NULL_HANDLE;
    VkPipeline searchPreparePipeline_ = VK_NULL_HANDLE;
    VkPipeline searchPhysicsPipeline_ = VK_NULL_HANDLE;
    VkPipeline searchPhysicsSteadyVelocityWaterPipeline_ = VK_NULL_HANDLE;
    VkPipeline searchPhysicsForcePhasePipeline_ = VK_NULL_HANDLE;
    VkPipeline searchPhysicsDetectPhasePipeline_ = VK_NULL_HANDLE;
    VkPipeline searchPhysicsRespondPhasePipeline_ = VK_NULL_HANDLE;
    VkPipeline searchEvaluatePipeline_ = VK_NULL_HANDLE;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    std::uint32_t timestampValidBits_ = 0u;
    bool exactSearchPhysicsSupported_ = false;
    std::mutex mutex_;
};

std::shared_ptr<Runtime> AcquireRuntime() {
    static std::shared_ptr<Runtime> runtime = [] {
        auto value = std::make_shared<Runtime>();
        value->Initialize();
        return value;
    }();
    return runtime;
}

VkDeviceSize Align(VkDeviceSize value) {
    return (value + WorkspaceAlignment - 1u) &
           ~(WorkspaceAlignment - 1u);
}

bool AddRegion(VkDeviceSize bytes,
               VkDeviceSize *cursor,
               VkDeviceSize *offset) {
    *offset = Align(*cursor);
    const VkDeviceSize reserved = std::max<VkDeviceSize>(bytes, 8u);
    if (*offset > std::numeric_limits<VkDeviceSize>::max() - reserved) {
        return false;
    }
    *cursor = *offset + reserved;
    return true;
}

struct TimelinePushConstants {
    std::uint64_t scene;
    std::uint64_t configuration;
    std::uint64_t states;
    std::uint64_t descriptors;
    std::uint64_t ticks;
    std::uint64_t observations;
    std::uint64_t results;
    std::uint64_t scratch;
    std::uint64_t cancellation;
    std::uint32_t candidateCount;
    std::uint32_t stateStride;
    std::uint32_t fullState;
};

static_assert(offsetof(TimelinePushConstants, candidateCount) == 72u);
static_assert(offsetof(TimelinePushConstants, fullState) +
                      sizeof(std::uint32_t) == PushConstantBytes);

struct SearchPushConstants {
    std::uint64_t parameters;
    std::uint32_t tickIndex;
    std::uint32_t candidateBase;
    std::uint32_t tickCount;
};

static_assert(offsetof(SearchPushConstants, tickIndex) == 8u);
static_assert(offsetof(SearchPushConstants, tickCount) +
                      sizeof(std::uint32_t) == ComputePushConstantBytes);

}  // namespace

Buffer::~Buffer() {
    if (runtime_ != nullptr) {
        if (buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(runtime_->Device(), buffer_, nullptr);
        }
        if (memory_ != VK_NULL_HANDLE) {
            vkFreeMemory(runtime_->Device(), memory_, nullptr);
        }
    }
}

RuntimeDiagnostics QueryRuntimeDiagnostics() noexcept {
    try {
        return AcquireRuntime()->Diagnostics();
    } catch (const std::bad_alloc &) {
        RuntimeDiagnostics result;
        result.status = RuntimeStatus::InitializationFailed;
        result.diagnostic = "Vulkan runtime allocation failed";
        return result;
    } catch (...) {
        RuntimeDiagnostics result;
        result.status = RuntimeStatus::InitializationFailed;
        result.diagnostic = "unexpected Vulkan runtime initialization failure";
        return result;
    }
}

bool SupportsExactSearchPhysics() noexcept {
    try {
        const std::shared_ptr<Runtime> runtime = AcquireRuntime();
        return runtime != nullptr &&
               runtime->Diagnostics().IsReady() &&
               runtime->SupportsExactSearchPhysics();
    } catch (...) {
        return false;
    }
}

bool UploadImmutableBuffer(
        const std::byte *data,
        std::size_t size,
        BufferHandle *destination,
        double *milliseconds,
        std::string *diagnostic) noexcept {
    if (destination == nullptr || data == nullptr || size == 0u) {
        if (diagnostic) *diagnostic = "invalid Vulkan upload input";
        return false;
    }
    try {
        const auto runtime = AcquireRuntime();
        if (!runtime->Diagnostics().IsReady()) {
            if (diagnostic) *diagnostic = runtime->Diagnostics().diagnostic;
            return false;
        }
        std::lock_guard<std::mutex> lock(runtime->Mutex());
        const auto start = std::chrono::steady_clock::now();
        BufferHandle staging;
        BufferHandle device;
        if (!runtime->CreateBuffer(
                    size,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    false, &staging, diagnostic) ||
            !runtime->CreateBuffer(
                    size,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    0u, true, &device, diagnostic)) {
            return false;
        }
        void *mapped = nullptr;
        if (!runtime->Map(staging, &mapped, diagnostic)) return false;
        std::memcpy(mapped, data, size);
        if (!runtime->Flush(staging, diagnostic)) {
            runtime->Unmap(staging);
            return false;
        }
        runtime->Unmap(staging);
        double waitMilliseconds = 0.0;
        const bool submitted = runtime->Submit(
                [&](VkCommandBuffer command) {
                    VkBufferCopy copy{};
                    copy.size = size;
                    vkCmdCopyBuffer(command, staging->buffer_,
                                    device->buffer_, 1u, &copy);
                },
                &waitMilliseconds, diagnostic);
        if (!submitted) return false;
        const auto end = std::chrono::steady_clock::now();
        if (milliseconds) {
            *milliseconds =
                    std::chrono::duration<double, std::milli>(
                            end - start).count();
        }
        *destination = std::move(device);
        return true;
    } catch (...) {
        if (diagnostic) {
            *diagnostic = "unexpected Vulkan immutable upload failure";
        }
        return false;
    }
}

std::uint64_t BufferSize(const BufferHandle &buffer) noexcept {
    return buffer == nullptr ? 0u : buffer->size_;
}

bool ExecuteTimelineKernel(
        const TimelineKernelRequest &request,
        TimelineKernelMetrics *metrics,
        std::string *diagnostic) noexcept {
    if (metrics) *metrics = {};
    if (request.scene == nullptr || request.configuration == nullptr ||
        request.states == nullptr || request.descriptors == nullptr ||
        request.initialResults == nullptr || request.cancellation == nullptr ||
        request.outputStates == nullptr || request.outputResults == nullptr ||
        request.candidateCount == 0u || request.stateStride == 0u ||
        request.candidateCount > request.stateBytes / request.stateStride) {
        if (diagnostic) *diagnostic = "invalid Vulkan timeline dispatch";
        return false;
    }
    try {
        const auto runtime = AcquireRuntime();
        if (!runtime->Diagnostics().IsReady()) {
            if (diagnostic) *diagnostic = runtime->Diagnostics().diagnostic;
            return false;
        }
        const std::uint64_t groupCount =
                (static_cast<std::uint64_t>(request.candidateCount) +
                 TimelineThreads - 1u) / TimelineThreads;
        if (groupCount > runtime->MaxDispatchX()) {
            if (diagnostic) {
                *diagnostic = "Vulkan timeline dispatch exceeds device limits";
            }
            return false;
        }

        std::lock_guard<std::mutex> lock(runtime->Mutex());
        if (request.finishRefinement &&
            !runtime->EnsureFinishRefinementPipelines(diagnostic)) {
            return false;
        }
        const auto allocationStart = std::chrono::steady_clock::now();
        VkDeviceSize cursor = 0u;
        VkDeviceSize statesOffset = 0u;
        VkDeviceSize descriptorsOffset = 0u;
        VkDeviceSize ticksOffset = 0u;
        VkDeviceSize observationsOffset = 0u;
        VkDeviceSize resultsOffset = 0u;
        VkDeviceSize scratchOffset = 0u;
        VkDeviceSize cancellationOffset = 0u;
        if (!AddRegion(request.stateBytes, &cursor, &statesOffset) ||
            !AddRegion(request.descriptorBytes, &cursor,
                       &descriptorsOffset) ||
            !AddRegion(request.tickBytes, &cursor, &ticksOffset) ||
            !AddRegion(request.observationBytes, &cursor,
                       &observationsOffset) ||
            !AddRegion(request.resultBytes, &cursor, &resultsOffset) ||
            !AddRegion(request.scratchBytes, &cursor, &scratchOffset) ||
            !AddRegion(request.cancellationBytes, &cursor,
                       &cancellationOffset)) {
            if (diagnostic) *diagnostic =
                    "Vulkan timeline allocation size overflow";
            return false;
        }
        const VkDeviceSize totalSize = Align(cursor);
        BufferHandle staging;
        BufferHandle workspace;
        if (!runtime->CreateBuffer(
                    totalSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                    false, &staging, diagnostic) ||
            !runtime->CreateBuffer(
                    totalSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    0u, true, &workspace, diagnostic)) {
            return false;
        }
        const auto allocationEnd = std::chrono::steady_clock::now();
        if (metrics) {
            metrics->allocationMilliseconds =
                    std::chrono::duration<double, std::milli>(
                            allocationEnd - allocationStart).count();
            metrics->peakDeviceBytes =
                    request.scene->size_ + request.configuration->size_ +
                    workspace->size_;
        }

        const auto transferStart = std::chrono::steady_clock::now();
        void *mapped = nullptr;
        if (!runtime->Map(staging, &mapped, diagnostic)) return false;
        auto *bytes = static_cast<std::byte *>(mapped);
        std::memset(bytes, 0, static_cast<std::size_t>(totalSize));
        std::memcpy(bytes + statesOffset,
                    request.states, request.stateBytes);
        std::memcpy(bytes + descriptorsOffset,
                    request.descriptors, request.descriptorBytes);
        if (request.tickBytes != 0u) {
            std::memcpy(bytes + ticksOffset,
                        request.ticks, request.tickBytes);
        }
        std::memcpy(bytes + resultsOffset,
                    request.initialResults, request.resultBytes);
        std::memcpy(bytes + cancellationOffset,
                    request.cancellation, request.cancellationBytes);
        if (!runtime->Flush(staging, diagnostic)) {
            runtime->Unmap(staging);
            return false;
        }
        const auto uploadEnd = std::chrono::steady_clock::now();

        std::array<VkBufferCopy, 5u> uploads{};
        std::uint32_t uploadCount = 0u;
        const auto addUpload = [&](VkDeviceSize offset,
                                   VkDeviceSize size) {
            if (size == 0u) return;
            uploads[uploadCount++] = {offset, offset, size};
        };
        addUpload(statesOffset, request.stateBytes);
        addUpload(descriptorsOffset, request.descriptorBytes);
        addUpload(ticksOffset, request.tickBytes);
        addUpload(resultsOffset, request.resultBytes);
        addUpload(cancellationOffset, request.cancellationBytes);

        std::array<VkBufferCopy, 3u> downloads{};
        std::uint32_t downloadCount = 0u;
        const auto addDownload = [&](VkDeviceSize offset,
                                     VkDeviceSize size) {
            if (size == 0u) return;
            downloads[downloadCount++] = {offset, offset, size};
        };
        addDownload(statesOffset, request.stateBytes);
        addDownload(resultsOffset, request.resultBytes);
        addDownload(observationsOffset, request.observationBytes);

        TimelinePushConstants push{};
        push.scene = request.scene->address_;
        push.configuration = request.configuration->address_;
        push.states = workspace->address_ + statesOffset;
        push.descriptors = workspace->address_ + descriptorsOffset;
        push.ticks = workspace->address_ + ticksOffset;
        push.observations = workspace->address_ + observationsOffset;
        push.results = workspace->address_ + resultsOffset;
        push.scratch = workspace->address_ + scratchOffset;
        push.cancellation = workspace->address_ + cancellationOffset;
        push.candidateCount = request.candidateCount;
        push.stateStride = request.stateStride;
        push.fullState = request.fullState ? 1u : 0u;

        double waitMilliseconds = 0.0;
        const bool submitted = runtime->Submit(
                [&](VkCommandBuffer command) {
                    vkCmdFillBuffer(command, workspace->buffer_,
                                    0u, VK_WHOLE_SIZE, 0u);
                    VkMemoryBarrier zeroBarrier{
                            VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    zeroBarrier.srcAccessMask =
                            VK_ACCESS_TRANSFER_WRITE_BIT;
                    zeroBarrier.dstAccessMask =
                            VK_ACCESS_TRANSFER_WRITE_BIT;
                    vkCmdPipelineBarrier(
                            command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0u, 1u, &zeroBarrier,
                            0u, nullptr, 0u, nullptr);
                    if (uploadCount != 0u) {
                        vkCmdCopyBuffer(command, staging->buffer_,
                                        workspace->buffer_,
                                        uploadCount, uploads.data());
                    }
                    VkMemoryBarrier uploadBarrier{
                            VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    uploadBarrier.srcAccessMask =
                            VK_ACCESS_TRANSFER_WRITE_BIT;
                    uploadBarrier.dstAccessMask =
                            VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT;
                    vkCmdPipelineBarrier(
                            command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0u, 1u, &uploadBarrier,
                            0u, nullptr, 0u, nullptr);

                    vkCmdResetQueryPool(
                            command, runtime->QueryPool(), 0u, 2u);
                    vkCmdWriteTimestamp(
                            command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            runtime->QueryPool(), 0u);
                    vkCmdBindPipeline(
                            command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            request.finishRefinement
                                    ? runtime->FinishRefinementPipeline()
                                    : runtime->Pipeline());
                    vkCmdPushConstants(
                            command, runtime->PipelineLayout(),
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0u, PushConstantBytes, &push);
                    vkCmdDispatch(command,
                                  static_cast<std::uint32_t>(groupCount),
                                  1u, 1u);
                    if (request.finishRefinement) {
                        VkMemoryBarrier refinementBarrier{
                                VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        refinementBarrier.srcAccessMask =
                                VK_ACCESS_SHADER_WRITE_BIT;
                        refinementBarrier.dstAccessMask =
                                VK_ACCESS_SHADER_READ_BIT |
                                VK_ACCESS_SHADER_WRITE_BIT;
                        vkCmdPipelineBarrier(
                                command,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0u, 1u, &refinementBarrier,
                                0u, nullptr, 0u, nullptr);
                        vkCmdBindPipeline(
                                command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                runtime->FinishProbePipeline());
                        vkCmdDispatch(
                                command,
                                static_cast<std::uint32_t>(groupCount),
                                1u, 1u);
                    }
                    vkCmdWriteTimestamp(
                            command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            runtime->QueryPool(), 1u);

                    VkMemoryBarrier downloadBarrier{
                            VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    downloadBarrier.srcAccessMask =
                            VK_ACCESS_SHADER_WRITE_BIT;
                    downloadBarrier.dstAccessMask =
                            VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(
                            command,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0u, 1u, &downloadBarrier,
                            0u, nullptr, 0u, nullptr);
                    vkCmdCopyBuffer(command, workspace->buffer_,
                                    staging->buffer_, downloadCount,
                                    downloads.data());
                    VkMemoryBarrier hostBarrier{
                            VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    hostBarrier.srcAccessMask =
                            VK_ACCESS_TRANSFER_WRITE_BIT;
                    hostBarrier.dstAccessMask =
                            VK_ACCESS_HOST_READ_BIT;
                    vkCmdPipelineBarrier(
                            command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_HOST_BIT,
                            0u, 1u, &hostBarrier,
                            0u, nullptr, 0u, nullptr);
                },
                &waitMilliseconds, diagnostic);
        if (!submitted || !runtime->Invalidate(staging, diagnostic)) {
            runtime->Unmap(staging);
            return false;
        }
        const auto downloadStart = std::chrono::steady_clock::now();
        std::memcpy(request.outputStates,
                    bytes + statesOffset, request.stateBytes);
        std::memcpy(request.outputResults,
                    bytes + resultsOffset, request.resultBytes);
        if (request.observationBytes != 0u) {
            std::memcpy(request.outputObservations,
                        bytes + observationsOffset,
                        request.observationBytes);
        }
        runtime->Unmap(staging);
        const auto transferEnd = std::chrono::steady_clock::now();

        std::uint64_t timestamps[2]{};
        const VkResult timestampStatus = vkGetQueryPoolResults(
                runtime->Device(), runtime->QueryPool(), 0u, 2u,
                sizeof(timestamps), timestamps,
                sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (timestampStatus != VK_SUCCESS) {
            if (diagnostic) *diagnostic = VulkanFailure(
                    "Vulkan timestamp query", timestampStatus);
            return false;
        }
        std::uint64_t elapsedTicks = timestamps[1] - timestamps[0];
        const std::uint32_t timestampBits = runtime->TimestampBits();
        if (timestampBits != 0u && timestampBits < 64u) {
            elapsedTicks &= (std::uint64_t{1u} << timestampBits) - 1u;
        }
        if (metrics) {
            metrics->uploadBytes =
                    request.stateBytes + request.descriptorBytes +
                    request.tickBytes + request.resultBytes +
                    request.cancellationBytes;
            metrics->downloadBytes =
                    request.stateBytes + request.resultBytes +
                    request.observationBytes;
            metrics->transferMilliseconds =
                    std::chrono::duration<double, std::milli>(
                            uploadEnd - transferStart).count() +
                    std::chrono::duration<double, std::milli>(
                            transferEnd - downloadStart).count();
            metrics->kernelMilliseconds =
                    static_cast<double>(elapsedTicks) *
                    runtime->TimestampPeriod() / 1000000.0;
            metrics->synchronizationMilliseconds = waitMilliseconds;
        }
        if (diagnostic) {
            *diagnostic = "Vulkan timeline compute dispatch completed";
        }
        return true;
    } catch (const std::bad_alloc &) {
        if (diagnostic) *diagnostic =
                "Vulkan timeline host allocation failed";
        return false;
    } catch (...) {
        if (diagnostic) *diagnostic =
                "unexpected Vulkan timeline execution failure";
        return false;
    }
}

bool ExecuteComputeKernels(
        const ComputeKernelRequest &request,
        ComputeKernelMetrics *metrics,
        std::string *diagnostic) noexcept {
    if (metrics) *metrics = {};
    if (request.parameters == nullptr || request.parameterBytes == 0u ||
        request.dispatches.empty()) {
        if (diagnostic) *diagnostic = "invalid Vulkan compute dispatch";
        return false;
    }
    for (const ComputeRegion &region : request.regions) {
        if (region.bytes == 0u) {
            if (diagnostic) *diagnostic =
                    "Vulkan compute region cannot be empty";
            return false;
        }
    }
    for (const ComputeRegionAddressBinding &binding :
         request.regionBindings) {
        if (binding.regionIndex >= request.regions.size() ||
            binding.parameterOffset > request.parameterBytes ||
            request.parameterBytes - binding.parameterOffset <
                    sizeof(std::uint64_t)) {
            if (diagnostic) *diagnostic =
                    "invalid Vulkan compute region binding";
            return false;
        }
    }
    for (const ComputeExternalAddressBinding &binding :
         request.externalBindings) {
        if (binding.buffer == nullptr ||
            binding.parameterOffset > request.parameterBytes ||
            request.parameterBytes - binding.parameterOffset <
                    sizeof(std::uint64_t)) {
            if (diagnostic) *diagnostic =
                    "invalid Vulkan compute external binding";
            return false;
        }
    }
    const bool dynamicCancellation =
            static_cast<bool>(request.cancellationRequested);
    if (dynamicCancellation &&
        (request.dynamicCancellationRegion >= request.regions.size() ||
         request.regions[request.dynamicCancellationRegion].bytes <
                 sizeof(std::uint32_t))) {
        if (diagnostic) *diagnostic =
                "invalid Vulkan dynamic cancellation binding";
        return false;
    }
    const bool hasTimeline = std::any_of(
            request.dispatches.begin(), request.dispatches.end(),
            [](const ComputeDispatch &dispatch) {
                return dispatch.kernel == ComputeKernel::Timeline;
            });
    if (hasTimeline) {
        const auto validRegion = [&](std::size_t index) {
            return index < request.regions.size();
        };
        if (request.timeline.scene == nullptr ||
            request.timeline.configuration == nullptr ||
            !validRegion(request.timeline.statesRegion) ||
            !validRegion(request.timeline.descriptorsRegion) ||
            !validRegion(request.timeline.ticksRegion) ||
            !validRegion(request.timeline.observationsRegion) ||
            !validRegion(request.timeline.resultsRegion) ||
            !validRegion(request.timeline.scratchRegion) ||
            !validRegion(request.timeline.cancellationRegion) ||
            request.timeline.stateStride == 0u) {
            if (diagnostic) *diagnostic =
                    "invalid Vulkan timeline compute bindings";
            return false;
        }
    }

    try {
        const auto runtime = AcquireRuntime();
        if (!runtime->Diagnostics().IsReady()) {
            if (diagnostic) *diagnostic = runtime->Diagnostics().diagnostic;
            return false;
        }
        for (const ComputeDispatch &dispatch : request.dispatches) {
            if (dispatch.workItemCount == 0u) {
                if (diagnostic) *diagnostic =
                        "Vulkan compute dispatch cannot be empty";
                return false;
            }
            const std::uint32_t threads =
                    ComputeThreadsForKernel(dispatch.kernel);
            const std::uint64_t groups =
                    (static_cast<std::uint64_t>(dispatch.workItemCount) +
                     threads - 1u) / threads;
            if (groups > runtime->MaxDispatchX()) {
                if (diagnostic) *diagnostic =
                        "Vulkan compute dispatch exceeds device limits";
                return false;
            }
        }

        std::lock_guard<std::mutex> lock(runtime->Mutex());
        for (const ComputeDispatch &dispatch : request.dispatches) {
            if (!runtime->EnsureComputePipeline(
                        dispatch.kernel, diagnostic)) {
                return false;
            }
        }
        const auto allocationStart = std::chrono::steady_clock::now();
        VkDeviceSize cursor = 0u;
        VkDeviceSize parameterOffset = 0u;
        if (!AddRegion(
                    request.parameterBytes, &cursor, &parameterOffset)) {
            if (diagnostic) *diagnostic =
                    "Vulkan compute parameter size overflow";
            return false;
        }
        std::vector<VkDeviceSize> regionOffsets(request.regions.size());
        for (std::size_t index = 0u;
             index < request.regions.size(); ++index) {
            if (!AddRegion(
                        request.regions[index].bytes,
                        &cursor, &regionOffsets[index])) {
                if (diagnostic) *diagnostic =
                        "Vulkan compute workspace size overflow";
                return false;
            }
        }
        const VkDeviceSize totalSize = Align(cursor);
        struct ComputeWorkspaceCache {
            BufferHandle staging;
            BufferHandle workspace;
        };
        static ComputeWorkspaceCache cache;
        BufferHandle staging = cache.staging;
        BufferHandle workspace = cache.workspace;
        BufferHandle cancellationBuffer;
        if (staging == nullptr || workspace == nullptr ||
            staging->size_ < totalSize || workspace->size_ < totalSize) {
            BufferHandle grownStaging;
            BufferHandle grownWorkspace;
            if (!runtime->CreateBuffer(
                        totalSize,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                        false, &grownStaging, diagnostic) ||
                !runtime->CreateBuffer(
                        totalSize,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        0u, true, &grownWorkspace, diagnostic)) {
                return false;
            }
            cache.staging = grownStaging;
            cache.workspace = grownWorkspace;
            staging = std::move(grownStaging);
            workspace = std::move(grownWorkspace);
        }
        if (dynamicCancellation &&
            !runtime->CreateBuffer(
                    sizeof(std::uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    0u, true, &cancellationBuffer, diagnostic)) {
            return false;
        }
        const auto allocationEnd = std::chrono::steady_clock::now();

        if (metrics) {
            metrics->allocationMilliseconds =
                    std::chrono::duration<double, std::milli>(
                            allocationEnd - allocationStart).count();
            metrics->peakDeviceBytes = workspace->size_;
            for (const ComputeExternalAddressBinding &binding :
                 request.externalBindings) {
                metrics->peakDeviceBytes += binding.buffer->size_;
            }
            if (hasTimeline) {
                metrics->peakDeviceBytes += request.timeline.scene->size_ +
                        request.timeline.configuration->size_;
            }
            if (dynamicCancellation) {
                metrics->peakDeviceBytes += cancellationBuffer->size_;
            }
        }

        const auto transferStart = std::chrono::steady_clock::now();
        void *mapped = nullptr;
        if (!runtime->Map(staging, &mapped, diagnostic)) return false;
        auto *bytes = static_cast<std::byte *>(mapped);
        std::memcpy(
                bytes + parameterOffset,
                request.parameters, request.parameterBytes);
        for (const ComputeRegionAddressBinding &binding :
             request.regionBindings) {
            const std::uint64_t address =
                    dynamicCancellation &&
                            binding.regionIndex ==
                                    request.dynamicCancellationRegion
                    ? cancellationBuffer->address_
                    : workspace->address_ +
                            regionOffsets[binding.regionIndex];
            std::memcpy(
                    bytes + parameterOffset + binding.parameterOffset,
                    &address, sizeof(address));
        }
        for (const ComputeExternalAddressBinding &binding :
             request.externalBindings) {
            const std::uint64_t address = binding.buffer->address_;
            std::memcpy(
                    bytes + parameterOffset + binding.parameterOffset,
                    &address, sizeof(address));
        }
        for (std::size_t index = 0u;
             index < request.regions.size(); ++index) {
            const ComputeRegion &region = request.regions[index];
            if (region.upload != nullptr) {
                std::memcpy(
                        bytes + regionOffsets[index],
                        region.upload, region.bytes);
            }
        }
        if (!runtime->Flush(staging, diagnostic)) {
            runtime->Unmap(staging);
            return false;
        }
        const auto uploadEnd = std::chrono::steady_clock::now();

        std::vector<VkBufferCopy> uploads;
        uploads.reserve(request.regions.size() + 1u);
        uploads.push_back({parameterOffset, parameterOffset,
                           request.parameterBytes});
        for (std::size_t index = 0u;
             index < request.regions.size(); ++index) {
            const ComputeRegion &region = request.regions[index];
            if (region.upload != nullptr) {
                uploads.push_back({regionOffsets[index],
                                   regionOffsets[index],
                                   region.bytes});
            }
        }
        std::vector<VkBufferCopy> downloads;
        downloads.reserve(request.regions.size());
        for (std::size_t index = 0u;
             index < request.regions.size(); ++index) {
            const ComputeRegion &region = request.regions[index];
            if (region.download != nullptr) {
                downloads.push_back({regionOffsets[index],
                                     regionOffsets[index],
                                     region.bytes});
            }
        }

        const std::uint64_t parameterAddress =
                workspace->address_ + parameterOffset;
        void *dynamicCancellationMapped = nullptr;
        if (dynamicCancellation) {
            if (!runtime->Map(
                        cancellationBuffer, &dynamicCancellationMapped,
                        diagnostic)) {
                runtime->Unmap(staging);
                return false;
            }
            const ComputeRegion &source =
                    request.regions[request.dynamicCancellationRegion];
            const std::uint32_t initial = source.upload == nullptr
                    ? 0u
                    : *static_cast<const std::uint32_t *>(source.upload);
            *static_cast<std::uint32_t *>(dynamicCancellationMapped) =
                    initial;
        }
        const std::size_t dispatchesPerSubmission =
                request.maxDispatchesPerSubmission == 0u
                ? request.dispatches.size()
                : request.maxDispatchesPerSubmission;
        const auto pollCancellation = dynamicCancellation
                ? std::function<void()>([&] {
                      if (*static_cast<std::uint32_t *>(
                                  dynamicCancellationMapped) != 0u) {
                          return;
                      }
                      bool requested = false;
                      try {
                          requested = request.cancellationRequested();
                      } catch (...) {
                          requested = true;
                      }
                      if (requested) {
                          *static_cast<std::uint32_t *>(
                                  dynamicCancellationMapped) = 1u;
                      }
                  })
                : std::function<void()>{};
        double waitMilliseconds = 0.0;
        double kernelMilliseconds = 0.0;
        bool submitted = true;
        for (std::size_t dispatchBegin = 0u;
             dispatchBegin < request.dispatches.size();
             dispatchBegin += dispatchesPerSubmission) {
            const std::size_t dispatchEnd = dispatchBegin + std::min(
                    dispatchesPerSubmission,
                    request.dispatches.size() - dispatchBegin);
            double submissionWaitMilliseconds = 0.0;
            submitted = runtime->Submit(
                [&](VkCommandBuffer command) {
                    if (dispatchBegin == 0u) {
                        vkCmdCopyBuffer(
                                command, staging->buffer_, workspace->buffer_,
                                static_cast<std::uint32_t>(uploads.size()),
                                uploads.data());
                        VkMemoryBarrier uploadBarrier{
                                VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                VK_ACCESS_TRANSFER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT |
                                        VK_ACCESS_SHADER_WRITE_BIT};
                        vkCmdPipelineBarrier(
                                command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0u, 1u, &uploadBarrier,
                                0u, nullptr, 0u, nullptr);
                    } else {
                        VkMemoryBarrier submissionBarrier{
                                VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                VK_ACCESS_SHADER_WRITE_BIT |
                                        VK_ACCESS_HOST_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT |
                                        VK_ACCESS_SHADER_WRITE_BIT};
                        vkCmdPipelineBarrier(
                                command,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                        VK_PIPELINE_STAGE_HOST_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0u, 1u, &submissionBarrier,
                                0u, nullptr, 0u, nullptr);
                    }

                    vkCmdResetQueryPool(
                            command, runtime->QueryPool(), 0u, 2u);
                    vkCmdWriteTimestamp(
                            command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            runtime->QueryPool(), 0u);
                    for (std::size_t stageIndex = dispatchBegin;
                         stageIndex < dispatchEnd; ++stageIndex) {
                        const ComputeDispatch &dispatch =
                                request.dispatches[stageIndex];
                        vkCmdBindPipeline(
                                command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                runtime->ComputePipeline(dispatch.kernel));
                        if (dispatch.kernel == ComputeKernel::Timeline) {
                            TimelinePushConstants push{};
                            push.scene = request.timeline.scene->address_;
                            push.configuration = request.timeline.
                                    configuration->address_;
                            push.states = workspace->address_ +
                                    regionOffsets[request.timeline.statesRegion];
                            push.descriptors = workspace->address_ +
                                    regionOffsets[
                                            request.timeline.descriptorsRegion];
                            push.ticks = workspace->address_ +
                                    regionOffsets[request.timeline.ticksRegion];
                            push.observations = workspace->address_ +
                                    regionOffsets[
                                            request.timeline.observationsRegion];
                            push.results = workspace->address_ +
                                    regionOffsets[request.timeline.resultsRegion];
                            push.scratch = workspace->address_ +
                                    regionOffsets[request.timeline.scratchRegion];
                            push.cancellation = dynamicCancellation
                                    ? cancellationBuffer->address_
                                    : workspace->address_ +
                                            regionOffsets[
                                                    request.timeline.
                                                            cancellationRegion];
                            push.candidateCount = dispatch.workItemCount;
                            push.stateStride =
                                    request.timeline.stateStride;
                            push.fullState =
                                    request.timeline.fullState ? 1u : 0u;
                            vkCmdPushConstants(
                                    command, runtime->PipelineLayout(),
                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                    0u, PushConstantBytes, &push);
                        } else {
                            const SearchPushConstants push{
                                    parameterAddress,
                                    dispatch.tickIndex,
                                    dispatch.candidateBase,
                                    dispatch.tickCount};
                            vkCmdPushConstants(
                                    command, runtime->PipelineLayout(),
                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                    0u, ComputePushConstantBytes, &push);
                        }
                        const std::uint32_t threads =
                                ComputeThreadsForKernel(dispatch.kernel);
                        const std::uint32_t groups =
                                (dispatch.workItemCount +
                                 threads - 1u) / threads;
                        vkCmdDispatch(command, groups, 1u, 1u);
                        if (stageIndex + 1u < dispatchEnd) {
                            VkMemoryBarrier stageBarrier{
                                    VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                    nullptr,
                                    VK_ACCESS_SHADER_WRITE_BIT |
                                            VK_ACCESS_HOST_WRITE_BIT,
                                    VK_ACCESS_SHADER_READ_BIT |
                                            VK_ACCESS_SHADER_WRITE_BIT};
                            vkCmdPipelineBarrier(
                                    command,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                            VK_PIPELINE_STAGE_HOST_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    0u, 1u, &stageBarrier,
                                    0u, nullptr, 0u, nullptr);
                        }
                    }
                    vkCmdWriteTimestamp(
                            command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            runtime->QueryPool(), 1u);
                    if (dispatchEnd == request.dispatches.size() &&
                        !downloads.empty()) {
                        VkMemoryBarrier downloadBarrier{
                                VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_TRANSFER_READ_BIT};
                        vkCmdPipelineBarrier(
                                command,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                                0u, 1u, &downloadBarrier,
                                0u, nullptr, 0u, nullptr);
                        vkCmdCopyBuffer(
                                command, workspace->buffer_,
                                staging->buffer_,
                                static_cast<std::uint32_t>(downloads.size()),
                                downloads.data());
                        VkMemoryBarrier hostBarrier{
                                VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                VK_ACCESS_TRANSFER_WRITE_BIT,
                                VK_ACCESS_HOST_READ_BIT};
                        vkCmdPipelineBarrier(
                                command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_HOST_BIT,
                                0u, 1u, &hostBarrier,
                                0u, nullptr, 0u, nullptr);
                    }
                },
                &submissionWaitMilliseconds, diagnostic,
                pollCancellation);
            waitMilliseconds += submissionWaitMilliseconds;
            if (!submitted) {
                if (diagnostic) {
                    *diagnostic += " while executing compute dispatches " +
                            std::to_string(dispatchBegin) + " through " +
                            std::to_string(dispatchEnd - 1u) + " of " +
                            std::to_string(request.dispatches.size());
                }
                break;
            }

            std::uint64_t timestamps[2]{};
            const VkResult timestampStatus = vkGetQueryPoolResults(
                    runtime->Device(), runtime->QueryPool(), 0u, 2u,
                    sizeof(timestamps), timestamps,
                    sizeof(std::uint64_t),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (timestampStatus != VK_SUCCESS) {
                if (diagnostic) {
                    *diagnostic = VulkanFailure(
                            "Vulkan compute timestamp query",
                            timestampStatus);
                }
                submitted = false;
                break;
            }
            std::uint64_t elapsedTicks = timestamps[1] - timestamps[0];
            const std::uint32_t timestampBits = runtime->TimestampBits();
            if (timestampBits != 0u && timestampBits < 64u) {
                elapsedTicks &=
                        (std::uint64_t{1u} << timestampBits) - 1u;
            }
            kernelMilliseconds += static_cast<double>(elapsedTicks) *
                    runtime->TimestampPeriod() / 1000000.0;
        }
        if (dynamicCancellation) {
            runtime->Unmap(cancellationBuffer);
        }
        if (!submitted || !runtime->Invalidate(staging, diagnostic)) {
            runtime->Unmap(staging);
            return false;
        }
        const auto downloadStart = std::chrono::steady_clock::now();
        for (std::size_t index = 0u;
             index < request.regions.size(); ++index) {
            const ComputeRegion &region = request.regions[index];
            if (region.download != nullptr) {
                std::memcpy(
                        region.download,
                        bytes + regionOffsets[index], region.bytes);
            }
        }
        runtime->Unmap(staging);
        const auto transferEnd = std::chrono::steady_clock::now();

        if (metrics) {
            metrics->uploadBytes = request.parameterBytes;
            metrics->downloadBytes = 0u;
            for (const ComputeRegion &region : request.regions) {
                if (region.upload != nullptr) {
                    metrics->uploadBytes += region.bytes;
                }
                if (region.download != nullptr) {
                    metrics->downloadBytes += region.bytes;
                }
            }
            metrics->transferMilliseconds =
                    std::chrono::duration<double, std::milli>(
                            uploadEnd - transferStart).count() +
                    std::chrono::duration<double, std::milli>(
                            transferEnd - downloadStart).count();
            metrics->kernelMilliseconds = kernelMilliseconds;
            metrics->synchronizationMilliseconds = waitMilliseconds;
        }
        if (diagnostic) {
            *diagnostic = "Vulkan compute dispatch completed";
        }
        return true;
    } catch (const std::bad_alloc &) {
        if (diagnostic) *diagnostic =
                "Vulkan compute host allocation failed";
        return false;
    } catch (...) {
        if (diagnostic) *diagnostic =
                "unexpected Vulkan compute execution failure";
        return false;
    }
}

}  // namespace forevervalidator::simulation::vulkan

#else

namespace forevervalidator::simulation::vulkan {

RuntimeDiagnostics QueryRuntimeDiagnostics() noexcept {
    RuntimeDiagnostics result;
    result.status = RuntimeStatus::NotCompiled;
    result.diagnostic = "Vulkan support is not compiled into this build";
    return result;
}

bool SupportsExactSearchPhysics() noexcept { return false; }

bool UploadImmutableBuffer(
        const std::byte *, std::size_t, BufferHandle *, double *,
        std::string *diagnostic) noexcept {
    if (diagnostic) {
        *diagnostic = "Vulkan support is not compiled into this build";
    }
    return false;
}

std::uint64_t BufferSize(const BufferHandle &) noexcept { return 0u; }

bool ExecuteTimelineKernel(
        const TimelineKernelRequest &,
        TimelineKernelMetrics *,
        std::string *diagnostic) noexcept {
    if (diagnostic) {
        *diagnostic = "Vulkan support is not compiled into this build";
    }
    return false;
}

bool ExecuteComputeKernels(
        const ComputeKernelRequest &,
        ComputeKernelMetrics *,
        std::string *diagnostic) noexcept {
    if (diagnostic) {
        *diagnostic = "Vulkan support is not compiled into this build";
    }
    return false;
}

}  // namespace forevervalidator::simulation::vulkan

#endif
