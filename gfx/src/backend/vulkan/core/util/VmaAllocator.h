#ifndef GFX_VULKAN_VMA_ALLOCATOR_H
#define GFX_VULKAN_VMA_ALLOCATOR_H

#include "../../common/Common.h"

// VMA configuration - must be set before including vk_mem_alloc.h
#define VMA_VULKAN_VERSION 1001000 // Vulkan 1.1
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <vk_mem_alloc.h>

namespace gfx::backend::vulkan::core {

class Allocator {
public:
    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;

    Allocator(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);
    ~Allocator();

    VmaAllocator handle() const;

    // Buffer allocation
    struct BufferAllocation {
        VkBuffer buffer;
        VmaAllocation allocation;
    };

    BufferAllocation createBuffer(const VkBufferCreateInfo& bufferInfo, VmaMemoryUsage usage, VkMemoryPropertyFlags requiredFlags = 0);
    void destroyBuffer(BufferAllocation& allocation);

    // Image allocation
    struct ImageAllocation {
        VkImage image;
        VmaAllocation allocation;
    };

    ImageAllocation createImage(const VkImageCreateInfo& imageInfo, VmaMemoryUsage usage, VkMemoryPropertyFlags requiredFlags = 0);
    void destroyImage(ImageAllocation& allocation);

    // Mapping
    void* mapMemory(VmaAllocation allocation);
    void unmapMemory(VmaAllocation allocation);
    void flushAllocation(VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size);
    void invalidateAllocation(VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size);

private:
    VmaAllocator m_allocator = VK_NULL_HANDLE;
};

} // namespace gfx::backend::vulkan::core

#endif // GFX_VULKAN_VMA_ALLOCATOR_H
