#include "CommandEncoder.h"

#include "../render/RenderPass.h"
#include "../resource/Buffer.h"
#include "../resource/Texture.h"
#include "../system/Device.h"
#include "../system/Queue.h"
#include "../util/Utils.h"

#include <stdexcept>

namespace gfx::backend::vulkan::core {

namespace {

    VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex)
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndex;

        VkCommandPool pool = VK_NULL_HANDLE;
        VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &pool);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }
        return pool;
    }

    void destroyCommandPool(VkDevice device, VkCommandPool& pool)
    {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, pool, nullptr);
            pool = VK_NULL_HANDLE;
        }
    }

    VkCommandBuffer allocateCommandBuffer(VkDevice device, VkCommandPool pool, VkCommandBufferLevel level)
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = pool;
        allocInfo.level = level;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer buffer = VK_NULL_HANDLE;
        VkResult result = vkAllocateCommandBuffers(device, &allocInfo, &buffer);
        if (result != VK_SUCCESS) {
            destroyCommandPool(device, pool);
            throw std::runtime_error("Failed to allocate command buffer");
        }
        return buffer;
    }

} // anonymous namespace

CommandEncoder::CommandEncoder(Device* device, bool bundle)
    : m_device(device)
    , m_commandPool(createCommandPool(device->handle(), device->getQueue()->family()))
    , m_commandBuffer(allocateCommandBuffer(device->handle(), m_commandPool, bundle ? VK_COMMAND_BUFFER_LEVEL_SECONDARY : VK_COMMAND_BUFFER_LEVEL_PRIMARY))
    , m_isBundleEncoder(bundle)
{
}

CommandEncoder::~CommandEncoder()
{
    destroyCommandPool(m_device->handle(), m_commandPool);
}

VkCommandBuffer CommandEncoder::handle() const
{
    return m_commandBuffer;
}

VkDevice CommandEncoder::device() const
{
    return m_device->handle();
}

Device* CommandEncoder::getDevice() const
{
    return m_device;
}

VkPipelineLayout CommandEncoder::currentPipelineLayout() const
{
    return m_currentPipelineLayout;
}

void CommandEncoder::setCurrentPipelineLayout(VkPipelineLayout layout)
{
    m_currentPipelineLayout = layout;
}

bool CommandEncoder::isBundleEncoder() const
{
    return m_isBundleEncoder;
}

void CommandEncoder::begin()
{
    if (!m_isRecording) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
        m_isRecording = true;
    }
}

void CommandEncoder::beginBundle(RenderPass* renderPass)
{
    end();

    vkResetCommandBuffer(m_commandBuffer, 0);

    VkCommandBufferInheritanceInfo inheritance{};
    inheritance.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    inheritance.renderPass = renderPass->handle();
    inheritance.subpass = 0;
    inheritance.framebuffer = VK_NULL_HANDLE;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    beginInfo.pInheritanceInfo = &inheritance;

    vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
    m_isRecording = true;
}

void CommandEncoder::end()
{
    if (m_isRecording) {
        vkEndCommandBuffer(m_commandBuffer);
        m_isRecording = false;
    }
}

void CommandEncoder::reset()
{
    m_currentPipelineLayout = VK_NULL_HANDLE;

    // Reset the command pool (this implicitly resets all command buffers)
    vkResetCommandPool(m_device->handle(), m_commandPool, 0);

    // Mark as not recording since the command buffer was reset
    m_isRecording = false;

    // Begin recording again
    begin();
}

void CommandEncoder::pipelineBarrier(const MemoryBarrier* memoryBarriers, uint32_t memoryBarrierCount, const BufferBarrier* bufferBarriers, uint32_t bufferBarrierCount, const TextureBarrier* textureBarriers, uint32_t textureBarrierCount)
{
    std::vector<VkMemoryBarrier> memBarriers;
    memBarriers.reserve(memoryBarrierCount);

    std::vector<VkBufferMemoryBarrier> bufferMemoryBarriers;
    bufferMemoryBarriers.reserve(bufferBarrierCount);

    std::vector<VkImageMemoryBarrier> imageBarriers;
    imageBarriers.reserve(textureBarrierCount);

    // Combine pipeline stages from all barriers
    VkPipelineStageFlags srcStage = 0;
    VkPipelineStageFlags dstStage = 0;

    // Process memory barriers
    for (uint32_t i = 0; i < memoryBarrierCount; ++i) {
        const auto& barrier = memoryBarriers[i];

        VkMemoryBarrier vkBarrier{};
        vkBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        vkBarrier.srcAccessMask = barrier.srcAccessMask;
        vkBarrier.dstAccessMask = barrier.dstAccessMask;

        memBarriers.push_back(vkBarrier);

        srcStage |= barrier.srcStageMask;
        dstStage |= barrier.dstStageMask;
    }

    // Process buffer barriers
    for (uint32_t i = 0; i < bufferBarrierCount; ++i) {
        const auto& barrier = bufferBarriers[i];

        VkBufferMemoryBarrier vkBarrier{};
        vkBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        vkBarrier.buffer = barrier.buffer->handle();
        vkBarrier.offset = barrier.offset;
        vkBarrier.size = barrier.size == 0 ? VK_WHOLE_SIZE : barrier.size;
        vkBarrier.srcAccessMask = barrier.srcAccessMask;
        vkBarrier.dstAccessMask = barrier.dstAccessMask;
        vkBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        bufferMemoryBarriers.push_back(vkBarrier);

        srcStage |= barrier.srcStageMask;
        dstStage |= barrier.dstStageMask;
    }

    // Process texture barriers
    for (uint32_t i = 0; i < textureBarrierCount; ++i) {
        const auto& barrier = textureBarriers[i];

        VkImageMemoryBarrier vkBarrier{};
        vkBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        vkBarrier.image = barrier.texture->handle();
        vkBarrier.subresourceRange.aspectMask = getImageAspectMask(barrier.texture->getFormat());
        vkBarrier.subresourceRange.baseMipLevel = barrier.baseMipLevel;
        vkBarrier.subresourceRange.levelCount = barrier.mipLevelCount;
        vkBarrier.subresourceRange.baseArrayLayer = barrier.baseArrayLayer;
        vkBarrier.subresourceRange.layerCount = barrier.arrayLayerCount;

        vkBarrier.oldLayout = barrier.oldLayout;
        vkBarrier.newLayout = barrier.newLayout;
        vkBarrier.srcAccessMask = barrier.srcAccessMask;
        vkBarrier.dstAccessMask = barrier.dstAccessMask;

        vkBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        imageBarriers.push_back(vkBarrier);

        srcStage |= barrier.srcStageMask;
        dstStage |= barrier.dstStageMask;

        // Update tracked layout
        barrier.texture->setLayout(barrier.newLayout);
    }

    vkCmdPipelineBarrier(m_commandBuffer, srcStage, dstStage, 0, static_cast<uint32_t>(memBarriers.size()), memBarriers.empty() ? nullptr : memBarriers.data(), static_cast<uint32_t>(bufferMemoryBarriers.size()), bufferMemoryBarriers.empty() ? nullptr : bufferMemoryBarriers.data(), static_cast<uint32_t>(imageBarriers.size()), imageBarriers.empty() ? nullptr : imageBarriers.data());
}

void CommandEncoder::copyBufferToBuffer(Buffer* source, uint64_t sourceOffset, Buffer* destination, uint64_t destinationOffset, uint64_t size)
{
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = sourceOffset;
    copyRegion.dstOffset = destinationOffset;
    copyRegion.size = size;

    vkCmdCopyBuffer(m_commandBuffer, source->handle(), destination->handle(), 1, &copyRegion);
}

void CommandEncoder::copyBufferToTexture(Buffer* source, uint64_t sourceOffset, Texture* destination, VkOffset3D origin, VkExtent3D extent, uint32_t mipLevel, uint32_t arrayLayer, uint32_t bytesPerRow, uint32_t rowsPerImage, VkImageAspectFlags aspectMask, VkImageLayout finalLayout)
{
    // For 3D textures extent.depth is the number of depth slices; for 1D/2D textures
    // it is the number of array layers to copy starting at arrayLayer
    bool is3D = destination->getImageType() == VK_IMAGE_TYPE_3D;
    uint32_t layerCount = is3D ? 1 : (extent.depth > 0 ? extent.depth : 1);
    uint32_t texelSize = getAspectTexelSize(destination->getFormat(), aspectMask); // block size for compressed formats
    uint32_t blockWidth = 1;
    getVkFormatBlockDimensions(destination->getFormat(), &blockWidth, nullptr);

    // A full-subresource copy overwrites all prior contents, so transition from UNDEFINED — always valid,
    // and correct per array layer (the single tracked layout is wrong for individually-filled layers).
    // Partial copies must preserve untouched texels, so they keep using the tracked layout.
    const VkExtent3D baseSize = destination->getSize();
    const uint32_t mipWidth = std::max(1u, baseSize.width >> mipLevel);
    const uint32_t mipHeight = std::max(1u, baseSize.height >> mipLevel);
    const uint32_t mipDepth = std::max(1u, baseSize.depth >> mipLevel);
    const bool fullSubresource = origin.x == 0 && origin.y == 0 && origin.z == 0
        && extent.width == mipWidth && extent.height == mipHeight && (!is3D || extent.depth == mipDepth);
    if (fullSubresource) {
        destination->transitionLayout(this, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevel, 1, arrayLayer, layerCount);
    } else {
        destination->transitionLayout(this, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevel, 1, arrayLayer, layerCount);
    }

    // Copy buffer to image
    VkBufferImageCopy region{};
    region.bufferOffset = sourceOffset;
    // bufferRowLength is in texels: blocks per row (bytesPerRow / blockBytes) x texels per block
    region.bufferRowLength = (bytesPerRow == 0 || texelSize == 0) ? 0 : (bytesPerRow / texelSize) * blockWidth;
    region.bufferImageHeight = rowsPerImage;
    region.imageSubresource.aspectMask = aspectMask;
    region.imageSubresource.mipLevel = mipLevel;
    region.imageSubresource.baseArrayLayer = is3D ? 0 : arrayLayer;
    region.imageSubresource.layerCount = layerCount;
    region.imageOffset = { origin.x, origin.y, is3D ? origin.z : 0 };
    region.imageExtent = { extent.width, extent.height, is3D ? extent.depth : 1 };

    vkCmdCopyBufferToImage(m_commandBuffer, source->handle(), destination->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition image layout to final layout
    destination->transitionLayout(this, finalLayout, mipLevel, 1, arrayLayer, layerCount);
}

void CommandEncoder::copyTextureToBuffer(Texture* source, VkOffset3D origin, uint32_t mipLevel, uint32_t arrayLayer, Buffer* destination, uint64_t destinationOffset, VkExtent3D extent, uint32_t bytesPerRow, uint32_t rowsPerImage, VkImageAspectFlags aspectMask, VkImageLayout finalLayout)
{
    // For 3D textures extent.depth is the number of depth slices; for 1D/2D textures
    // it is the number of array layers to copy starting at arrayLayer
    bool is3D = source->getImageType() == VK_IMAGE_TYPE_3D;
    uint32_t layerCount = is3D ? 1 : (extent.depth > 0 ? extent.depth : 1);
    uint32_t texelSize = getAspectTexelSize(source->getFormat(), aspectMask); // block size for compressed formats
    uint32_t blockWidth = 1;
    getVkFormatBlockDimensions(source->getFormat(), &blockWidth, nullptr);

    // Transition image layout to transfer src optimal
    source->transitionLayout(this, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mipLevel, 1, arrayLayer, layerCount);

    // Copy image to buffer
    VkBufferImageCopy region{};
    region.bufferOffset = destinationOffset;
    // bufferRowLength is in texels: blocks per row (bytesPerRow / blockBytes) x texels per block
    region.bufferRowLength = (bytesPerRow == 0 || texelSize == 0) ? 0 : (bytesPerRow / texelSize) * blockWidth;
    region.bufferImageHeight = rowsPerImage;
    region.imageSubresource.aspectMask = aspectMask;
    region.imageSubresource.mipLevel = mipLevel;
    region.imageSubresource.baseArrayLayer = is3D ? 0 : arrayLayer;
    region.imageSubresource.layerCount = layerCount;
    region.imageOffset = { origin.x, origin.y, is3D ? origin.z : 0 };
    region.imageExtent = { extent.width, extent.height, is3D ? extent.depth : 1 };

    vkCmdCopyImageToBuffer(m_commandBuffer, source->handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination->handle(), 1, &region);

    // Transition image layout to final layout
    source->transitionLayout(this, finalLayout, mipLevel, 1, arrayLayer, layerCount);
}

void CommandEncoder::copyTextureToTexture(Texture* source, VkOffset3D sourceOrigin, uint32_t sourceMipLevel, uint32_t sourceArrayLayer, VkImageLayout srcFinalLayout, Texture* destination, VkOffset3D destinationOrigin, uint32_t destinationMipLevel, uint32_t destinationArrayLayer, VkImageLayout dstFinalLayout, VkExtent3D extent)
{
    VkExtent3D srcSize = source->getSize();
    bool is3DTexture = (srcSize.depth > 1);

    uint32_t layerCount = is3DTexture ? 1 : extent.depth;
    uint32_t copyDepth = is3DTexture ? extent.depth : 1;

    // Transition images to transfer layouts
    source->transitionLayout(this, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sourceMipLevel, 1, sourceArrayLayer, layerCount);
    destination->transitionLayout(this, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, destinationMipLevel, 1, destinationArrayLayer, layerCount);

    // Copy image to image
    VkImageCopy region{};
    region.srcSubresource.aspectMask = getImageAspectMask(source->getFormat());
    region.srcSubresource.mipLevel = sourceMipLevel;
    region.srcSubresource.baseArrayLayer = is3DTexture ? 0 : sourceArrayLayer;
    region.srcSubresource.layerCount = layerCount;
    region.srcOffset = { sourceOrigin.x, sourceOrigin.y, is3DTexture ? sourceOrigin.z : 0 };
    region.dstSubresource.aspectMask = getImageAspectMask(destination->getFormat());
    region.dstSubresource.mipLevel = destinationMipLevel;
    region.dstSubresource.baseArrayLayer = is3DTexture ? 0 : destinationArrayLayer;
    region.dstSubresource.layerCount = layerCount;
    region.dstOffset = { destinationOrigin.x, destinationOrigin.y, is3DTexture ? destinationOrigin.z : 0 };
    region.extent = { extent.width, extent.height, copyDepth };

    vkCmdCopyImage(m_commandBuffer, source->handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition images to final layouts
    source->transitionLayout(this, srcFinalLayout, sourceMipLevel, 1, sourceArrayLayer, layerCount);
    destination->transitionLayout(this, dstFinalLayout, destinationMipLevel, 1, destinationArrayLayer, layerCount);
}

void CommandEncoder::blitTextureToTexture(Texture* source, VkOffset3D sourceOrigin, VkExtent3D sourceExtent, uint32_t sourceMipLevel, uint32_t sourceArrayLayer, VkImageLayout srcFinalLayout, Texture* destination, VkOffset3D destinationOrigin, VkExtent3D destinationExtent, uint32_t destinationMipLevel, uint32_t destinationArrayLayer, VkImageLayout dstFinalLayout, VkFilter filter)
{
    VkExtent3D srcSize = source->getSize();
    bool is3DTexture = (srcSize.depth > 1);

    uint32_t layerCount = is3DTexture ? 1 : sourceExtent.depth;
    uint32_t srcDepth = is3DTexture ? sourceExtent.depth : 1;
    uint32_t dstDepth = is3DTexture ? destinationExtent.depth : 1;

    // Transition images to transfer layouts
    source->transitionLayout(this, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sourceMipLevel, 1, sourceArrayLayer, layerCount);
    destination->transitionLayout(this, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, destinationMipLevel, 1, destinationArrayLayer, layerCount);

    // Blit image to image with scaling and filtering
    VkImageBlit region{};
    region.srcSubresource.aspectMask = getImageAspectMask(source->getFormat());
    region.srcSubresource.mipLevel = sourceMipLevel;
    region.srcSubresource.baseArrayLayer = is3DTexture ? 0 : sourceArrayLayer;
    region.srcSubresource.layerCount = layerCount;
    region.srcOffsets[0] = { sourceOrigin.x, sourceOrigin.y, is3DTexture ? sourceOrigin.z : 0 };
    region.srcOffsets[1] = { static_cast<int32_t>(sourceOrigin.x + sourceExtent.width), static_cast<int32_t>(sourceOrigin.y + sourceExtent.height), is3DTexture ? static_cast<int32_t>(sourceOrigin.z + srcDepth) : 1 };
    region.dstSubresource.aspectMask = getImageAspectMask(destination->getFormat());
    region.dstSubresource.mipLevel = destinationMipLevel;
    region.dstSubresource.baseArrayLayer = is3DTexture ? 0 : destinationArrayLayer;
    region.dstSubresource.layerCount = layerCount;
    region.dstOffsets[0] = { destinationOrigin.x, destinationOrigin.y, is3DTexture ? destinationOrigin.z : 0 };
    region.dstOffsets[1] = { static_cast<int32_t>(destinationOrigin.x + destinationExtent.width), static_cast<int32_t>(destinationOrigin.y + destinationExtent.height), is3DTexture ? static_cast<int32_t>(destinationOrigin.z + dstDepth) : 1 };

    vkCmdBlitImage(m_commandBuffer, source->handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, filter);

    // Transition images to final layouts
    source->transitionLayout(this, srcFinalLayout, sourceMipLevel, 1, sourceArrayLayer, layerCount);
    destination->transitionLayout(this, dstFinalLayout, destinationMipLevel, 1, destinationArrayLayer, layerCount);
}

void CommandEncoder::writeTimestamp(VkQueryPool queryPool, uint32_t queryIndex)
{
    vkCmdWriteTimestamp(m_commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, queryIndex);
}

void CommandEncoder::resetQuerySet(VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    vkCmdResetQueryPool(m_commandBuffer, queryPool, firstQuery, queryCount);
}

void CommandEncoder::resolveQuerySet(VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, VkBuffer buffer, uint64_t destinationOffset)
{
    // Make the queries' results available to the copy via an explicit execution dependency rather than
    // VK_QUERY_RESULT_WAIT_BIT. Recording the copy after the query commands orders it after them but does
    // not guarantee their results are available; a waiting copy would be correct but, empirically, makes
    // MoltenVK resolve occlusion results within the render-pass scope, splitting the Metal render encoder
    // mid-pass and tripping store-action validation. This barrier runs after the render pass, so it
    // synchronizes the queries with the copy without any in-pass split.
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    vkCmdCopyQueryPoolResults(m_commandBuffer, queryPool, firstQuery, queryCount, buffer, destinationOffset, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
}

} // namespace gfx::backend::vulkan::core