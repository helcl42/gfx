#include "Queue.h"

#include "Adapter.h"
#include "Device.h"

#include "../command/CommandEncoder.h"
#include "../resource/Buffer.h"
#include "../resource/Texture.h"
#include "../sync/Fence.h"
#include "../sync/Semaphore.h"
#include "../util/CommandExecutor.h"
#include "../util/Utils.h"
#include "../util/VmaAllocator.h"

#include "common/Logger.h"

#include <cstring>
#include <stdexcept>

namespace gfx::backend::vulkan::core {

Queue::Queue(Device* device, VkQueue queue, uint32_t queueFamily, uint32_t queueIndex)
    : m_queue(queue)
    , m_device(device)
    , m_queueFamily(queueFamily)
    , m_queueIndex(queueIndex)
{
}

VkQueue Queue::handle() const
{
    return m_queue;
}

VkDevice Queue::device() const
{
    return m_device->handle();
}

VkPhysicalDevice Queue::physicalDevice() const
{
    return m_device->getAdapter()->handle();
}

uint32_t Queue::family() const
{
    return m_queueFamily;
}

uint32_t Queue::index() const
{
    return m_queueIndex;
}

QueueInfo Queue::getInfo() const
{
    return { m_queueFamily, m_queueIndex };
}

VkResult Queue::submit(const SubmitInfo& submitInfo)
{
    // Convert command encoders to command buffers
    std::vector<VkCommandBuffer> commandBuffers;
    commandBuffers.reserve(submitInfo.commandEncoderCount);
    for (uint32_t i = 0; i < submitInfo.commandEncoderCount; ++i) {
        commandBuffers.push_back(submitInfo.commandEncoders[i]->handle());
    }

    // Convert wait semaphores
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<uint64_t> waitValues;
    std::vector<VkPipelineStageFlags> waitStages;
    waitSemaphores.reserve(submitInfo.waitSemaphoreCount);
    waitStages.reserve(submitInfo.waitSemaphoreCount);

    bool hasTimelineWait = false;
    for (uint32_t i = 0; i < submitInfo.waitSemaphoreCount; ++i) {
        waitSemaphores.push_back(submitInfo.waitSemaphores[i]->handle());
        waitStages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        if (submitInfo.waitSemaphores[i]->getType() == SemaphoreType::Timeline) {
            hasTimelineWait = true;
            uint64_t value = submitInfo.waitValues ? submitInfo.waitValues[i] : 0;
            waitValues.push_back(value);
        } else {
            waitValues.push_back(0);
        }
    }

    // Convert signal semaphores
    std::vector<VkSemaphore> signalSemaphores;
    std::vector<uint64_t> signalValues;
    signalSemaphores.reserve(submitInfo.signalSemaphoreCount);

    bool hasTimelineSignal = false;
    for (uint32_t i = 0; i < submitInfo.signalSemaphoreCount; ++i) {
        signalSemaphores.push_back(submitInfo.signalSemaphores[i]->handle());

        if (submitInfo.signalSemaphores[i]->getType() == SemaphoreType::Timeline) {
            hasTimelineSignal = true;
            uint64_t value = submitInfo.signalValues ? submitInfo.signalValues[i] : 0;
            signalValues.push_back(value);
        } else {
            signalValues.push_back(0);
        }
    }

    // Timeline semaphore info (if needed)
    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    if (hasTimelineWait || hasTimelineSignal) {
        timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineInfo.waitSemaphoreValueCount = static_cast<uint32_t>(waitValues.size());
        timelineInfo.pWaitSemaphoreValues = waitValues.empty() ? nullptr : waitValues.data();
        timelineInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalValues.size());
        timelineInfo.pSignalSemaphoreValues = signalValues.empty() ? nullptr : signalValues.data();
    }

    // Build submit info
    VkSubmitInfo vkSubmitInfo{};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    if (hasTimelineWait || hasTimelineSignal) {
        vkSubmitInfo.pNext = &timelineInfo;
    }
    vkSubmitInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    vkSubmitInfo.pCommandBuffers = commandBuffers.empty() ? nullptr : commandBuffers.data();
    vkSubmitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
    vkSubmitInfo.pWaitSemaphores = waitSemaphores.empty() ? nullptr : waitSemaphores.data();
    vkSubmitInfo.pWaitDstStageMask = waitStages.empty() ? nullptr : waitStages.data();
    vkSubmitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
    vkSubmitInfo.pSignalSemaphores = signalSemaphores.empty() ? nullptr : signalSemaphores.data();

    // Get fence if provided
    VkFence fence = VK_NULL_HANDLE;
    if (submitInfo.signalFence) {
        fence = submitInfo.signalFence->handle();
    }

    return vkQueueSubmit(m_queue, 1, &vkSubmitInfo, fence);
}

void Queue::waitIdle()
{
    vkQueueWaitIdle(m_queue);
}

void Queue::writeBuffer(Buffer* buffer, uint64_t offset, const void* data, uint64_t size)
{
    void* mapped = buffer->map(offset, size);
    if (mapped) {
        // Buffer is host-visible, can map directly
        memcpy(mapped, data, size);
        buffer->unmap();
    } else {
        // Buffer is not host-visible (device-local), use VMA staging buffer
        Allocator* allocator = m_device->getAllocator();

        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = size;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        auto staging = allocator->createBuffer(stagingInfo, VMA_MEMORY_USAGE_CPU_ONLY,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // Map and copy data to staging buffer
        void* stagingMapped = allocator->mapMemory(staging.allocation);
        memcpy(stagingMapped, data, size);
        allocator->unmapMemory(staging.allocation);

        // Execute copy command
        CommandExecutor executor(this);
        executor.execute([&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = offset;
            copyRegion.size = size;
            vkCmdCopyBuffer(cmd, staging.buffer, buffer->handle(), 1, &copyRegion);
        });

        // Cleanup
        allocator->destroyBuffer(staging);
    }
}

void Queue::writeTexture(Texture* texture, const VkOffset3D& origin, uint32_t mipLevel,
    uint32_t arrayLayer, const void* data, uint64_t dataSize,
    const VkExtent3D& extent, uint32_t bytesPerRow, VkImageLayout finalLayout)
{
    Allocator* allocator = m_device->getAllocator();

    // Create staging buffer via VMA
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = dataSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    auto staging = allocator->createBuffer(bufferInfo, VMA_MEMORY_USAGE_CPU_ONLY,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Copy data to staging buffer
    void* mappedData = allocator->mapMemory(staging.allocation);
    if (!mappedData) {
        gfx::common::Logger::instance().logError("Failed to map staging buffer for texture upload");
        allocator->destroyBuffer(staging);
        return;
    }
    memcpy(mappedData, data, dataSize);
    allocator->unmapMemory(staging.allocation);

    // Execute copy command
    CommandExecutor executor(this);
    // Reset tracked layout to UNDEFINED before each subresource write — the single
    // m_currentLayout may not reflect this subresource's actual layout (e.g., texture
    // array layers uploaded sequentially). UNDEFINED is always valid as oldLayout since
    // writeTexture overwrites the entire subresource contents.
    texture->setLayout(VK_IMAGE_LAYOUT_UNDEFINED);

    executor.execute([&](VkCommandBuffer cmd) {
        // Transition image to transfer dst optimal
        texture->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevel, 1, arrayLayer, 1);

        // Copy buffer to image
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = (bytesPerRow == 0) ? 0 : bytesPerRow / getVkFormatBytesPerPixel(texture->getFormat());
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = getImageAspectMask(texture->getFormat());
        region.imageSubresource.mipLevel = mipLevel;
        region.imageSubresource.baseArrayLayer = arrayLayer;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = origin;
        region.imageExtent = extent;

        vkCmdCopyBufferToImage(cmd, staging.buffer, texture->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // Transition image to final layout
        texture->transitionLayout(cmd, finalLayout, mipLevel, 1, arrayLayer, 1);
    });

    // Cleanup
    allocator->destroyBuffer(staging);
}

} // namespace gfx::backend::vulkan::core