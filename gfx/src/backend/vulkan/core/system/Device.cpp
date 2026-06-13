#include "Device.h"

#include "Adapter.h"
#include "Instance.h"
#include "Queue.h"

#include "../util/VmaAllocator.h"

#include <gfx/gfx.h>

#include <cstring>
#include <stdexcept>

namespace gfx::backend::vulkan::core {

namespace {
    uint64_t makeQueueKey(uint32_t queueFamilyIndex, uint32_t queueIndex)
    {
        return (static_cast<uint64_t>(queueFamilyIndex) << 16) | queueIndex;
    }

    bool containsString(const std::vector<std::string>& strings, const char* value)
    {
        for (const auto& str : strings) {
            if (str == value) {
                return true;
            }
        }
        return false;
    }

    // Parse the requested extension names into a DeviceExtension bitmask so all
    // later enablement checks are O(1) bit tests
    uint64_t parseEnabledExtensions(const std::vector<std::string>& enabledExtensions)
    {
        const std::pair<const char*, DeviceExtension> knownExtensions[] = {
            { extensions::SWAPCHAIN, DeviceExtension::Swapchain },
            { extensions::TIMELINE_SEMAPHORE, DeviceExtension::TimelineSemaphore },
            { extensions::MULTIVIEW, DeviceExtension::Multiview },
            { extensions::ANISOTROPIC_FILTERING, DeviceExtension::AnisotropicFiltering },
            { extensions::OCCLUSION_QUERY_PRECISE, DeviceExtension::OcclusionQueryPrecise },
            { extensions::NON_SOLID_FILL, DeviceExtension::NonSolidFill },
            { extensions::TIMESTAMP_QUERY, DeviceExtension::TimestampQuery },
            { extensions::TEXTURE_COMPRESSION_BC, DeviceExtension::TextureCompressionBC },
            { extensions::TEXTURE_COMPRESSION_ETC2, DeviceExtension::TextureCompressionETC2 },
            { extensions::TEXTURE_COMPRESSION_ASTC, DeviceExtension::TextureCompressionASTC },
        };

        uint64_t mask = 0;
        for (const auto& [name, bit] : knownExtensions) {
            if (containsString(enabledExtensions, name)) {
                mask |= static_cast<uint64_t>(bit);
            }
        }
        return mask;
    }

    bool isExtensionAvailable(const std::vector<VkExtensionProperties>& availableExtensions, const char* extension)
    {
        for (const auto& availableExt : availableExtensions) {
            if (strcmp(extension, availableExt.extensionName) == 0) {
                return true;
            }
        }
        return false;
    }
} // anonymous namespace

Device::Device(Adapter* adapter, const DeviceCreateInfo& createInfo)
    : m_adapter(adapter)
    , m_enabledExtensions(parseEnabledExtensions(createInfo.enabledExtensions))
{
    // Invariant: m_enabledExtensions records REQUESTED extensions; if one cannot
    // actually be enabled (missing device feature), this constructor throws - so
    // on any live Device a set bit means the feature IS enabled on the VkDevice.

    // Query available device features
    const auto& availableFeatures = m_adapter->getFeatures();

    // Device features
    VkPhysicalDeviceFeatures deviceFeatures{};

    // Device extensions
    std::vector<const char*> requestedExtensions;
#ifndef GFX_HEADLESS_BUILD
    if (isExtensionEnabled(DeviceExtension::Swapchain)) {
        requestedExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
#endif // GFX_HEADLESS_BUILD

    // Enable timeline semaphore extension if requested
    bool timelineSemaphoreEnabled = isExtensionEnabled(DeviceExtension::TimelineSemaphore);
    if (timelineSemaphoreEnabled) {
        requestedExtensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    }

    // Enable multiview extension if requested
    bool multiviewEnabled = isExtensionEnabled(DeviceExtension::Multiview);
    if (multiviewEnabled) {
        requestedExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
    }

    // Enable anisotropic filtering if requested
    if (isExtensionEnabled(DeviceExtension::AnisotropicFiltering)) {
        if (!availableFeatures.samplerAnisotropy) {
            throw std::runtime_error("Anisotropic filtering is not supported by this device");
        }
        deviceFeatures.samplerAnisotropy = VK_TRUE;
    }

    // Enable precise occlusion queries if requested
    if (isExtensionEnabled(DeviceExtension::OcclusionQueryPrecise)) {
        if (!availableFeatures.occlusionQueryPrecise) {
            throw std::runtime_error("Precise occlusion queries are not supported by this device");
        }
        deviceFeatures.occlusionQueryPrecise = VK_TRUE;
    }

    // Note: TIMESTAMP_QUERY needs no feature bit or device extension - it is available
    // when timestampValidBits > 0 on the graphics queue, which the Adapter already
    // verified before reporting the extension as supported.

    // Enable non-solid fill mode if requested
    if (isExtensionEnabled(DeviceExtension::NonSolidFill)) {
        if (!availableFeatures.fillModeNonSolid) {
            throw std::runtime_error("Non-solid fill mode is not supported by this device");
        }
        deviceFeatures.fillModeNonSolid = VK_TRUE;
    }

    // Enable texture compression families if requested
    if (isExtensionEnabled(DeviceExtension::TextureCompressionBC)) {
        if (!availableFeatures.textureCompressionBC) {
            throw std::runtime_error("BC texture compression is not supported by this device");
        }
        deviceFeatures.textureCompressionBC = VK_TRUE;
    }
    if (isExtensionEnabled(DeviceExtension::TextureCompressionETC2)) {
        if (!availableFeatures.textureCompressionETC2) {
            throw std::runtime_error("ETC2 texture compression is not supported by this device");
        }
        deviceFeatures.textureCompressionETC2 = VK_TRUE;
    }
    if (isExtensionEnabled(DeviceExtension::TextureCompressionASTC)) {
        if (!availableFeatures.textureCompressionASTC_LDR) {
            throw std::runtime_error("ASTC texture compression is not supported by this device");
        }
        deviceFeatures.textureCompressionASTC_LDR = VK_TRUE;
    }

    // Check if all requested extensions are available
    const auto availableExtensions = m_adapter->enumerateExtensionProperties();

    // Add native (raw Vulkan) device extensions from pNext chain
    const GfxChainHeader* header = static_cast<const GfxChainHeader*>(createInfo.pNext);
    while (header) {
        if (header->sType == GFX_STRUCTURE_TYPE_NATIVE_EXTENSIONS_DESCRIPTOR) {
            const auto* nativeDesc = reinterpret_cast<const GfxNativeExtensionsDescriptor*>(header);
            if (nativeDesc->nativeExtensions && nativeDesc->nativeExtensionCount > 0) {
                for (uint32_t i = 0; i < nativeDesc->nativeExtensionCount; ++i) {
                    if (isExtensionAvailable(availableExtensions, nativeDesc->nativeExtensions[i])) {
                        requestedExtensions.push_back(nativeDesc->nativeExtensions[i]);
                    }
                }
            }
        }
        header = static_cast<const GfxChainHeader*>(header->pNext);
    }

    constexpr const char* portabilitySubsetExtension = "VK_KHR_portability_subset";
    if (isExtensionAvailable(availableExtensions, portabilitySubsetExtension)) {
        requestedExtensions.push_back(portabilitySubsetExtension);
    }

    for (const char* requestedExt : requestedExtensions) {
        if (!isExtensionAvailable(availableExtensions, requestedExt)) {
            std::string errorMsg = "Required Vulkan device extension not available: ";
            errorMsg += requestedExt;
            throw std::runtime_error(errorMsg);
        }
    }

    // Timeline semaphore features (VK_KHR_timeline_semaphore extension for Vulkan 1.1)
    VkPhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};
    if (timelineSemaphoreEnabled) {
        timelineSemaphoreFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        timelineSemaphoreFeatures.pNext = nullptr;
        timelineSemaphoreFeatures.timelineSemaphore = VK_TRUE;
    }

    // Multiview features (VK_KHR_multiview extension for Vulkan 1.1)
    VkPhysicalDeviceMultiviewFeatures multiviewFeatures{};
    if (multiviewEnabled) {
        multiviewFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES;
        multiviewFeatures.pNext = timelineSemaphoreEnabled ? &timelineSemaphoreFeatures : nullptr;
        multiviewFeatures.multiview = VK_TRUE;
    }

    // Determine which queues to create
    std::vector<DeviceCreateInfo::QueueRequest> queueRequests;
    if (createInfo.queueRequests.empty()) {
        // Default: create one graphics queue
        queueRequests.push_back({ m_adapter->getGraphicsQueueFamily(), 0, 1.0f });
    } else {
        queueRequests = createInfo.queueRequests;
    }

    // Group queue requests by family and find max queue index per family
    std::unordered_map<uint32_t, uint32_t> maxQueueIndexPerFamily;
    for (const auto& req : queueRequests) {
        maxQueueIndexPerFamily[req.queueFamilyIndex] = std::max(maxQueueIndexPerFamily[req.queueFamilyIndex], req.queueIndex);
    }

    // Build VkDeviceQueueCreateInfo for each family
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::vector<std::vector<float>> priorityStorage; // Keep priorities alive

    for (const auto& [familyIndex, maxIndex] : maxQueueIndexPerFamily) {
        uint32_t queueCount = maxIndex + 1;
        std::vector<float> priorities(queueCount, 1.0f);

        // Set specified priorities
        for (const auto& req : queueRequests) {
            if (req.queueFamilyIndex == familyIndex) {
                priorities[req.queueIndex] = req.priority;
            }
        }

        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = familyIndex;
        queueCreateInfo.queueCount = queueCount;
        queueCreateInfo.pQueuePriorities = priorities.data();
        queueCreateInfos.push_back(queueCreateInfo);
        priorityStorage.push_back(std::move(priorities));
    }

    void* pNext = nullptr;
    if (multiviewEnabled) {
        pNext = &multiviewFeatures;
    } else if (timelineSemaphoreEnabled) {
        pNext = &timelineSemaphoreFeatures;
    }

    VkDeviceCreateInfo vkCreateInfo{};
    vkCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    vkCreateInfo.pNext = pNext;
    vkCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    vkCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    vkCreateInfo.pEnabledFeatures = &deviceFeatures;
    vkCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requestedExtensions.size());
    vkCreateInfo.ppEnabledExtensionNames = requestedExtensions.data();

    VkResult result = vkCreateDevice(m_adapter->handle(), &vkCreateInfo, nullptr, &m_device);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan device");
    }

    // Create Queue wrappers for all requested queues
    for (const auto& req : queueRequests) {
        VkQueue vkQueue = VK_NULL_HANDLE;
        vkGetDeviceQueue(m_device, req.queueFamilyIndex, req.queueIndex, &vkQueue);

        uint64_t key = makeQueueKey(req.queueFamilyIndex, req.queueIndex);
        auto queue = std::make_unique<Queue>(this, vkQueue, req.queueFamilyIndex, req.queueIndex);

        // Store default queue pointer (first one created)
        if (!m_defaultQueue) {
            m_defaultQueue = queue.get();
        }

        m_queues[key] = std::move(queue);
    }

    // Create VMA allocator
    m_allocator = std::make_unique<Allocator>(
        m_adapter->getInstance()->handle(),
        m_adapter->handle(),
        m_device);
}

Device::~Device()
{
    // Destroy queues first (they may reference device)
    m_queues.clear();
    // Destroy VMA allocator before destroying the Vulkan device
    m_allocator.reset();

    if (m_device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_device, nullptr);
    }
}

void Device::waitIdle()
{
    // vkDeviceWaitIdle requires external host synchronization on all of the device's queues.
    // Lock ordering is map mutex first, then queue mutexes - same order as queueMutex() callers,
    // so this cannot deadlock with concurrent submit/present.
    std::lock_guard<std::mutex> mapLock(m_queueMutexMapMutex);
    std::vector<std::unique_lock<std::mutex>> queueLocks;
    queueLocks.reserve(m_queueMutexes.size());
    for (auto& [queue, mutex] : m_queueMutexes) {
        queueLocks.emplace_back(mutex);
    }
    vkDeviceWaitIdle(m_device);
}

VkDevice Device::handle() const
{
    return m_device;
}

Queue* Device::getQueue()
{
    return m_defaultQueue;
}

Queue* Device::getQueueByIndex(uint32_t queueFamilyIndex, uint32_t queueIndex)
{
    uint64_t key = makeQueueKey(queueFamilyIndex, queueIndex);
    auto it = m_queues.find(key);
    return (it != m_queues.end()) ? it->second.get() : nullptr;
}

std::mutex& Device::queueMutex(VkQueue queue)
{
    // unordered_map guarantees reference stability, so the returned mutex
    // stays valid while the Device is alive
    std::lock_guard<std::mutex> lock(m_queueMutexMapMutex);
    return m_queueMutexes[queue];
}

Adapter* Device::getAdapter()
{
    return m_adapter;
}

Allocator* Device::getAllocator()
{
    return m_allocator.get();
}

const VkPhysicalDeviceProperties& Device::getProperties() const
{
    return m_adapter->getProperties();
}

bool Device::supportsShaderFormat(ShaderSourceType format) const
{
    // Vulkan backend only supports SPIR-V
    return format == ShaderSourceType::SPIRV;
}

bool Device::isExtensionEnabled(DeviceExtension extension) const
{
    return (m_enabledExtensions & static_cast<uint64_t>(extension)) != 0;
}

} // namespace gfx::backend::vulkan::core