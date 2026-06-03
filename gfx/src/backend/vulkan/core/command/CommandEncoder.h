#ifndef GFX_VULKAN_COMMANDENCODER_H
#define GFX_VULKAN_COMMANDENCODER_H

#include "../CoreTypes.h"

namespace gfx::backend::vulkan::core {

class Device;
class RenderPass;

class CommandEncoder {
public:
    CommandEncoder(const CommandEncoder&) = delete;
    CommandEncoder& operator=(const CommandEncoder&) = delete;

    CommandEncoder(Device* device, bool bundle = false);
    ~CommandEncoder();

    VkCommandBuffer handle() const;
    VkDevice device() const;
    Device* getDevice() const;
    VkPipelineLayout currentPipelineLayout() const;
    void setCurrentPipelineLayout(VkPipelineLayout layout);

    bool isBundleEncoder() const;

    void begin();
    void beginBundle(RenderPass* renderPass);
    void end();
    void reset();

    void pipelineBarrier(const MemoryBarrier* memoryBarriers, uint32_t memoryBarrierCount, const BufferBarrier* bufferBarriers, uint32_t bufferBarrierCount, const TextureBarrier* textureBarriers, uint32_t textureBarrierCount);

    void copyBufferToBuffer(Buffer* source, uint64_t sourceOffset, Buffer* destination, uint64_t destinationOffset, uint64_t size);
    void copyBufferToTexture(Buffer* source, uint64_t sourceOffset, Texture* destination, VkOffset3D origin, VkExtent3D extent, uint32_t mipLevel, uint32_t arrayLayer, uint32_t bytesPerRow, VkImageLayout finalLayout);
    void copyTextureToBuffer(Texture* source, VkOffset3D origin, uint32_t mipLevel, uint32_t arrayLayer, Buffer* destination, uint64_t destinationOffset, VkExtent3D extent, uint32_t bytesPerRow, VkImageLayout finalLayout);
    void copyTextureToTexture(Texture* source, VkOffset3D sourceOrigin, uint32_t sourceMipLevel, uint32_t sourceArrayLayer, VkImageLayout srcFinalLayout, Texture* destination, VkOffset3D destinationOrigin, uint32_t destinationMipLevel, uint32_t destinationArrayLayer, VkImageLayout dstFinalLayout, VkExtent3D extent);
    void blitTextureToTexture(Texture* source, VkOffset3D sourceOrigin, VkExtent3D sourceExtent, uint32_t sourceMipLevel, uint32_t sourceArrayLayer, VkImageLayout srcFinalLayout, Texture* destination, VkOffset3D destinationOrigin, VkExtent3D destinationExtent, uint32_t destinationMipLevel, uint32_t destinationArrayLayer, VkImageLayout dstFinalLayout, VkFilter filter);

    void writeTimestamp(VkQueryPool queryPool, uint32_t queryIndex);
    void resetQuerySet(VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount);
    void resolveQuerySet(VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount, VkBuffer buffer, uint64_t destinationOffset);

private:
    Device* m_device = nullptr;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    bool m_isBundleEncoder = false;
    bool m_isRecording = false;
    VkPipelineLayout m_currentPipelineLayout = VK_NULL_HANDLE;
};

} // namespace gfx::backend::vulkan::core

#endif // GFX_VULKAN_COMMANDENCODER_H