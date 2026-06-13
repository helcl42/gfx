#ifndef GFX_VULKAN_QUEUE_H
#define GFX_VULKAN_QUEUE_H

#include "../CoreTypes.h"

namespace gfx::backend::vulkan::core {

class Device;
class Buffer;
class Texture;

class Queue {
public:
    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    Queue(Device* device, VkQueue queue, uint32_t queueFamily, uint32_t queueIndex);
    ~Queue() = default;

    VkQueue handle() const;
    VkDevice device() const;
    VkPhysicalDevice physicalDevice() const;
    uint32_t family() const;
    uint32_t index() const;
    QueueInfo getInfo() const;

    VkResult submit(const SubmitInfo& submitInfo);
    void waitIdle();

    // Write data directly to a buffer by mapping it
    void writeBuffer(Buffer* buffer, uint64_t offset, const void* data, uint64_t size);

    // Write data directly to a texture using staging buffer
    void writeTexture(Texture* texture, const VkOffset3D& origin, uint32_t mipLevel, uint32_t arrayLayer, const void* data, uint64_t dataSize, const VkExtent3D& extent, uint32_t bytesPerRow, uint32_t rowsPerImage, VkImageAspectFlags aspectMask, VkImageLayout finalLayout);

private:
    VkQueue m_queue = VK_NULL_HANDLE;
    Device* m_device = nullptr;
    uint32_t m_queueFamily = 0;
    uint32_t m_queueIndex = 0;
};

} // namespace gfx::backend::vulkan::core

#endif // GFX_VULKAN_QUEUE_H