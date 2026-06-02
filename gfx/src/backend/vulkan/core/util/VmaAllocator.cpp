#include "VmaAllocator.h"

#include <stdexcept>

namespace gfx::backend::vulkan::core {

Allocator::Allocator(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device)
{
    VmaVulkanFunctions vulkanFunctions{};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.instance = instance;
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_1;
    allocatorInfo.pVulkanFunctions = &vulkanFunctions;

    VkResult result = vmaCreateAllocator(&allocatorInfo, &m_allocator);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator");
    }
}

Allocator::~Allocator()
{
    if (m_allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(m_allocator);
    }
}

VmaAllocator Allocator::handle() const
{
    return m_allocator;
}

Allocator::BufferAllocation Allocator::createBuffer(const VkBufferCreateInfo& bufferInfo, VmaMemoryUsage usage, VkMemoryPropertyFlags requiredFlags)
{
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = usage;
    allocInfo.requiredFlags = requiredFlags;

    BufferAllocation result{};
    VkResult vkResult = vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &result.buffer, &result.allocation, nullptr);
    if (vkResult != VK_SUCCESS) {
        throw std::runtime_error("VMA failed to create buffer");
    }
    return result;
}

void Allocator::destroyBuffer(BufferAllocation& allocation)
{
    if (allocation.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator, allocation.buffer, allocation.allocation);
        allocation.buffer = VK_NULL_HANDLE;
        allocation.allocation = VK_NULL_HANDLE;
    }
}

Allocator::ImageAllocation Allocator::createImage(const VkImageCreateInfo& imageInfo, VmaMemoryUsage usage, VkMemoryPropertyFlags requiredFlags)
{
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = usage;
    allocInfo.requiredFlags = requiredFlags;

    ImageAllocation result{};
    VkResult vkResult = vmaCreateImage(m_allocator, &imageInfo, &allocInfo, &result.image, &result.allocation, nullptr);
    if (vkResult != VK_SUCCESS) {
        throw std::runtime_error("VMA failed to create image");
    }
    return result;
}

void Allocator::destroyImage(ImageAllocation& allocation)
{
    if (allocation.image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, allocation.image, allocation.allocation);
        allocation.image = VK_NULL_HANDLE;
        allocation.allocation = VK_NULL_HANDLE;
    }
}

void* Allocator::mapMemory(VmaAllocation allocation)
{
    void* data = nullptr;
    VkResult result = vmaMapMemory(m_allocator, allocation, &data);
    if (result != VK_SUCCESS) {
        return nullptr;
    }
    return data;
}

void Allocator::unmapMemory(VmaAllocation allocation)
{
    vmaUnmapMemory(m_allocator, allocation);
}

void Allocator::flushAllocation(VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size)
{
    vmaFlushAllocation(m_allocator, allocation, offset, size);
}

void Allocator::invalidateAllocation(VmaAllocation allocation, VkDeviceSize offset, VkDeviceSize size)
{
    vmaInvalidateAllocation(m_allocator, allocation, offset, size);
}

} // namespace gfx::backend::vulkan::core
