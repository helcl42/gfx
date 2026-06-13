#ifndef GFX_VULKAN_DEVICE_H
#define GFX_VULKAN_DEVICE_H

#include "../CoreTypes.h"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace gfx::backend::vulkan::core {

class Adapter;
class Allocator;
class Queue;

class Device {
public:
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Device(Adapter* adapter, const DeviceCreateInfo& createInfo);
    ~Device();

    void waitIdle();

    VkDevice handle() const;
    Queue* getQueue();
    Queue* getQueueByIndex(uint32_t queueFamilyIndex, uint32_t queueIndex);
    Queue* findPresentQueue(VkSurfaceKHR surface);

    // Returns the mutex guarding host access to the given VkQueue.
    // Vulkan requires external synchronization for vkQueueSubmit/vkQueueWaitIdle/vkQueuePresentKHR;
    // all queue-touching paths (Queue, CommandExecutor, Swapchain::present) must hold this lock.
    std::mutex& queueMutex(VkQueue queue);
    Adapter* getAdapter();
    Allocator* getAllocator();
    const VkPhysicalDeviceProperties& getProperties() const;

    bool supportsShaderFormat(ShaderSourceType format) const;
    bool isExtensionEnabled(DeviceExtension extension) const;

    // Extension function pointer loaders
    template <typename T>
    T loadFunction(const char* name) const
    {
        return reinterpret_cast<T>(vkGetDeviceProcAddr(m_device, name));
    }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    Adapter* m_adapter = nullptr; // Non-owning pointer
    uint64_t m_enabledExtensions = 0; // Bitmask of DeviceExtension values enabled at creation
    std::unique_ptr<Allocator> m_allocator;

    // Map of (queueFamilyIndex << 16 | queueIndex) -> Queue
    std::unordered_map<uint64_t, std::unique_ptr<Queue>> m_queues;
    Queue* m_defaultQueue = nullptr; // Non-owning pointer to default queue

    // Per-VkQueue mutexes for host synchronization of queue operations
    std::mutex m_queueMutexMapMutex;
    std::unordered_map<VkQueue, std::mutex> m_queueMutexes;
};

} // namespace gfx::backend::vulkan::core

#endif // GFX_VULKAN_DEVICE_H