#include "Buffer.h"

#include "../system/Adapter.h"
#include "../system/Device.h"
#include "../util/Utils.h"

#include <stdexcept>

namespace gfx::backend::vulkan::core {

// Helper to convert GFX memory property flags to VMA usage
static VmaMemoryUsage toVmaUsage(VkMemoryPropertyFlags memProps)
{
    if ((memProps & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        return VMA_MEMORY_USAGE_GPU_ONLY;
    }
    if ((memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (memProps & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        // Could be staging (CPU to GPU) or readback (GPU to CPU)
        if (memProps & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            return VMA_MEMORY_USAGE_CPU_TO_GPU;
        }
        if (memProps & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
            return VMA_MEMORY_USAGE_GPU_TO_CPU;
        }
        return VMA_MEMORY_USAGE_CPU_TO_GPU;
    }
    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        return VMA_MEMORY_USAGE_CPU_ONLY;
    }
    return VMA_MEMORY_USAGE_AUTO;
}

// Owning constructor - creates and manages VkBuffer via VMA
Buffer::Buffer(Device* device, const BufferCreateInfo& createInfo)
    : m_device(device)
    , m_ownsResources(true)
    , m_info(createBufferInfo(createInfo))
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = m_info.size;
    bufferInfo.usage = m_info.usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaMemoryUsage vmaUsage = toVmaUsage(createInfo.memoryProperties);
    auto alloc = m_device->getAllocator()->createBuffer(bufferInfo, vmaUsage, createInfo.memoryProperties);
    m_buffer = alloc.buffer;
    m_allocation = alloc.allocation;
}

// Non-owning constructor - wraps an existing VkBuffer
Buffer::Buffer(Device* device, VkBuffer buffer, const BufferImportInfo& importInfo)
    : m_device(device)
    , m_ownsResources(false)
    , m_buffer(buffer)
    , m_allocation(VK_NULL_HANDLE)
    , m_info(createBufferInfo(importInfo))
{
}

Buffer::~Buffer()
{
    if (m_ownsResources && m_buffer != VK_NULL_HANDLE) {
        Allocator::BufferAllocation alloc{ m_buffer, m_allocation };
        m_device->getAllocator()->destroyBuffer(alloc);
    }
}

void* Buffer::map(uint64_t offset, uint64_t size)
{
    if ((m_info.memoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
        return nullptr;
    }

    if (offset > m_info.size) {
        return nullptr;
    }

    uint64_t mapSize = size;
    if (mapSize == UINT64_MAX) {
        mapSize = m_info.size - offset;
    }

    if (offset + mapSize > m_info.size) {
        return nullptr;
    }

    void* data = m_device->getAllocator()->mapMemory(m_allocation);
    if (!data) {
        return nullptr;
    }
    return static_cast<char*>(data) + offset;
}

void Buffer::unmap()
{
    if ((m_info.memoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
        return;
    }

    m_device->getAllocator()->unmapMemory(m_allocation);
    m_asyncMappedPointer = nullptr;
    m_asyncMapped = false;
}

void Buffer::asyncMap(uint64_t offset, uint64_t size)
{
    // On Vulkan, mapping is synchronous - just do it immediately
    m_asyncMappedPointer = map(offset, size);
    m_asyncMapped = (m_asyncMappedPointer != nullptr);
}

bool Buffer::isAsyncMapped() const
{
    return m_asyncMapped;
}

void* Buffer::getAsyncMappedPointer() const
{
    return m_asyncMappedPointer;
}

bool Buffer::waitUntilAsyncMapped(uint64_t /*timeoutNs*/)
{
    // Vulkan asyncMap is synchronous — already mapped by the time this is called.
    return m_asyncMapped;
}

void Buffer::flushMappedRange(uint64_t offset, uint64_t size)
{
    // Only needed for non-coherent memory
    if ((m_info.memoryProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
        return; // Coherent memory is automatically visible
    }

    if ((m_info.memoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
        return; // Not host-visible, cannot flush
    }

    m_device->getAllocator()->flushAllocation(m_allocation, offset, size);
}

void Buffer::invalidateMappedRange(uint64_t offset, uint64_t size)
{
    // Only needed for non-coherent memory
    if ((m_info.memoryProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
        return; // Coherent memory is automatically visible
    }

    if ((m_info.memoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
        return; // Not host-visible, cannot invalidate
    }

    m_device->getAllocator()->invalidateAllocation(m_allocation, offset, size);
}

VkBuffer Buffer::handle() const
{
    return m_buffer;
}

size_t Buffer::size() const
{
    return m_info.size;
}

VkBufferUsageFlags Buffer::getUsage() const
{
    return m_info.usage;
}

const BufferInfo& Buffer::getInfo() const
{
    return m_info;
}

BufferInfo Buffer::createBufferInfo(const BufferCreateInfo& createInfo)
{
    BufferInfo info{};
    info.size = createInfo.size;
    info.usage = createInfo.usage;
    info.originalUsage = createInfo.originalUsage;
    info.memoryProperties = createInfo.memoryProperties;
    return info;
}

BufferInfo Buffer::createBufferInfo(const BufferImportInfo& importInfo)
{
    BufferInfo info{};
    info.size = importInfo.size;
    info.usage = importInfo.usage;
    info.originalUsage = importInfo.originalUsage;
    info.memoryProperties = 0; // Imported buffers don't specify memory properties
    return info;
}

} // namespace gfx::backend::vulkan::core