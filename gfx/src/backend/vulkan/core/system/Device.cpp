#include "Device.h"

#include "Adapter.h"
#include "Instance.h"
#include "Queue.h"

#include "../util/Utils.h"
#include "../util/VmaAllocator.h"

#include "common/Strings.h"

#include <gfx/gfx.h>

#include <stdexcept>

namespace gfx::backend::vulkan::core {

namespace {
    uint64_t makeQueueKey(uint32_t queueFamilyIndex, uint32_t queueIndex)
    {
        return (static_cast<uint64_t>(queueFamilyIndex) << 16) | queueIndex;
    }

    // Parse the requested extension names into a DeviceExtension bitmask so all
    // later enablement checks are O(1) bit tests
    uint64_t parseEnabledExtensions(const std::vector<std::string>& enabledExtensions)
    {
        const std::pair<const char*, DeviceExtension> knownExtensions[] = {
            { extensions::SWAPCHAIN, DeviceExtension::Swapchain },
            { extensions::TIMELINE_SEMAPHORE, DeviceExtension::TimelineSemaphore },
            { extensions::MULTIVIEW, DeviceExtension::Multiview },
            { extensions::OCCLUSION_QUERY_PRECISE, DeviceExtension::OcclusionQueryPrecise },
            { extensions::NON_SOLID_FILL, DeviceExtension::NonSolidFill },
            { extensions::TIMESTAMP_QUERY, DeviceExtension::TimestampQuery },
        };

        uint64_t mask = 0;
        for (const auto& [name, bit] : knownExtensions) {
            if (gfx::common::containsString(enabledExtensions, name)) {
                mask |= static_cast<uint64_t>(bit);
            }
        }
        return mask;
    }

    bool hasExtension(uint64_t enabledExtensions, DeviceExtension extension)
    {
        return (enabledExtensions & static_cast<uint64_t>(extension)) != 0;
    }

    // A device can only enable the extensions its adapter supports
    void requireSupportedExtensions(const Adapter* adapter, const std::vector<std::string>& enabledExtensions)
    {
        const std::vector<const char*> supportedExtensions = adapter->enumerateSupportedExtensions();
        for (const std::string& extension : enabledExtensions) {
            if (!gfx::common::containsString(supportedExtensions, extension.c_str())) {
                throw std::runtime_error("Device extension not supported by this adapter: " + extension);
            }
        }
    }

    // Translates the enabled GFX extensions into VkPhysicalDeviceFeatures
    VkPhysicalDeviceFeatures buildEnabledFeatures(uint64_t enabledExtensions, const VkPhysicalDeviceFeatures& availableFeatures)
    {
        VkPhysicalDeviceFeatures deviceFeatures{};

        if (hasExtension(enabledExtensions, DeviceExtension::OcclusionQueryPrecise)) {
            deviceFeatures.occlusionQueryPrecise = VK_TRUE;
        }
        if (hasExtension(enabledExtensions, DeviceExtension::NonSolidFill)) {
            deviceFeatures.fillModeNonSolid = VK_TRUE;
        }

        // WebGPU core features and optional texture formats, enabled wherever the hardware has them
        deviceFeatures.depthBiasClamp = availableFeatures.depthBiasClamp;
        deviceFeatures.imageCubeArray = availableFeatures.imageCubeArray;
        deviceFeatures.samplerAnisotropy = availableFeatures.samplerAnisotropy;
        deviceFeatures.textureCompressionBC = availableFeatures.textureCompressionBC;
        deviceFeatures.textureCompressionETC2 = availableFeatures.textureCompressionETC2;
        deviceFeatures.textureCompressionASTC_LDR = availableFeatures.textureCompressionASTC_LDR;

        return deviceFeatures;
    }

    // Collects the Vulkan device extension names to request: extension-mapped GFX
    // extensions, native pNext passthrough extensions, and the portability subset
    std::vector<const char*> collectRequestedExtensions(uint64_t enabledExtensions, const DeviceCreateInfo& createInfo, const std::vector<VkExtensionProperties>& availableExtensions)
    {
        std::vector<const char*> requestedExtensions;
#ifndef GFX_HEADLESS_BUILD
        if (hasExtension(enabledExtensions, DeviceExtension::Swapchain)) {
            requestedExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        }
#endif // GFX_HEADLESS_BUILD
        if (hasExtension(enabledExtensions, DeviceExtension::TimelineSemaphore)) {
            requestedExtensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
        }
        if (hasExtension(enabledExtensions, DeviceExtension::Multiview)) {
            requestedExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
        }

        // Add native (raw Vulkan) device extensions from the pNext chain
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

        return requestedExtensions;
    }

    // Returns the queue requests to create: the caller's, or one default graphics queue
    std::vector<DeviceCreateInfo::QueueRequest> resolveQueueRequests(const DeviceCreateInfo& createInfo, uint32_t defaultGraphicsFamily)
    {
        if (createInfo.queueRequests.empty()) {
            return { { defaultGraphicsFamily, 0, 1.0f } };
        }
        return createInfo.queueRequests;
    }

    // Queue create-infos together with the priority arrays they point into
    struct QueueCreateInfos {
        std::vector<std::vector<float>> priorityStorage;
        std::vector<VkDeviceQueueCreateInfo> infos;
    };

    // Builds one VkDeviceQueueCreateInfo per requested family
    QueueCreateInfos buildQueueCreateInfos(const std::vector<DeviceCreateInfo::QueueRequest>& queueRequests)
    {
        // Group queue requests by family and find max queue index per family
        std::unordered_map<uint32_t, uint32_t> maxQueueIndexPerFamily;
        for (const auto& req : queueRequests) {
            maxQueueIndexPerFamily[req.queueFamilyIndex] = std::max(maxQueueIndexPerFamily[req.queueFamilyIndex], req.queueIndex);
        }

        QueueCreateInfos result;
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
            result.infos.push_back(queueCreateInfo);
            result.priorityStorage.push_back(std::move(priorities));
        }
        return result;
    }
} // anonymous namespace

Device::Device(Adapter* adapter, const DeviceCreateInfo& createInfo)
    : m_adapter(adapter)
    , m_enabledExtensions(parseEnabledExtensions(createInfo.enabledExtensions))
    , m_enabledFeatures(buildEnabledFeatures(m_enabledExtensions, m_adapter->getFeatures()))
{
    // Invariant: m_enabledExtensions records REQUESTED extensions; if the adapter does
    // not support one, this constructor throws - so on any live Device a set bit means
    // the extension IS enabled on the VkDevice.
    requireSupportedExtensions(m_adapter, createInfo.enabledExtensions);

    std::vector<const char*> requestedExtensions = collectRequestedExtensions(m_enabledExtensions, createInfo, m_adapter->enumerateExtensionProperties());

    bool timelineSemaphoreEnabled = isExtensionEnabled(DeviceExtension::TimelineSemaphore);
    bool multiviewEnabled = isExtensionEnabled(DeviceExtension::Multiview);

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

    std::vector<DeviceCreateInfo::QueueRequest> queueRequests = resolveQueueRequests(createInfo, m_adapter->getGraphicsQueueFamily());
    QueueCreateInfos queueCreateInfos = buildQueueCreateInfos(queueRequests);

    void* pNext = nullptr;
    if (multiviewEnabled) {
        pNext = &multiviewFeatures;
    } else if (timelineSemaphoreEnabled) {
        pNext = &timelineSemaphoreFeatures;
    }

    VkDeviceCreateInfo vkCreateInfo{};
    vkCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    vkCreateInfo.pNext = pNext;
    vkCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.infos.size());
    vkCreateInfo.pQueueCreateInfos = queueCreateInfos.infos.data();
    vkCreateInfo.pEnabledFeatures = &m_enabledFeatures;
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
    m_allocator = std::make_unique<Allocator>(m_adapter->getInstance()->handle(), m_adapter->handle(), m_device);
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

Queue* Device::findPresentQueue(VkSurfaceKHR surface)
{
    if (m_defaultQueue && m_adapter->supportsPresentation(m_defaultQueue->family(), surface)) {
        return m_defaultQueue;
    }
    // Otherwise fall back to any other queue created on this device
    for (auto& [key, queue] : m_queues) {
        if (m_adapter->supportsPresentation(queue->family(), surface)) {
            return queue.get();
        }
    }
    return nullptr;
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

const VkPhysicalDeviceFeatures& Device::getEnabledFeatures() const
{
    return m_enabledFeatures;
}

} // namespace gfx::backend::vulkan::core