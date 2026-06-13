#ifndef GFX_WEBGPU_COMMAND_ENCODER_H
#define GFX_WEBGPU_COMMAND_ENCODER_H

#include "../CoreTypes.h"

namespace gfx::backend::webgpu::core {

class Device;
class Buffer;
class Texture;
class RenderPass;

class CommandEncoder {
public:
    // Prevent copying
    CommandEncoder(const CommandEncoder&) = delete;
    CommandEncoder& operator=(const CommandEncoder&) = delete;

    CommandEncoder(Device* device, const CommandEncoderCreateInfo& createInfo);
    CommandEncoder(Device* device);
    ~CommandEncoder();

    WGPUCommandEncoder handle() const;
    WGPUCommandBuffer commandBuffer() const;
    Device* getDevice() const;

    bool isBundleEncoder() const;

    void begin();
    void beginBundle(RenderPass* renderPass);
    void end();
    void reset();

    WGPURenderBundleEncoder getBundleEncoder() const;
    WGPURenderBundle getRenderBundle() const;

    // Copy operations
    void copyBufferToBuffer(Buffer* source, uint64_t sourceOffset, Buffer* destination, uint64_t destinationOffset, uint64_t size);
    void copyBufferToTexture(Buffer* source, uint64_t sourceOffset, Texture* destination, const WGPUOrigin3D& origin, const WGPUExtent3D& extent, uint32_t mipLevel, uint32_t arrayLayer, uint32_t bytesPerRow, uint32_t rowsPerImage, WGPUTextureAspect aspect);
    void copyTextureToBuffer(Texture* source, const WGPUOrigin3D& origin, uint32_t mipLevel, uint32_t arrayLayer, Buffer* destination, uint64_t destinationOffset, const WGPUExtent3D& extent, uint32_t bytesPerRow, uint32_t rowsPerImage, WGPUTextureAspect aspect);
    void copyTextureToTexture(Texture* source, const WGPUOrigin3D& sourceOrigin, uint32_t sourceMipLevel, uint32_t sourceArrayLayer, Texture* destination, const WGPUOrigin3D& destinationOrigin, uint32_t destinationMipLevel, uint32_t destinationArrayLayer, const WGPUExtent3D& extent);
    void blitTextureToTexture(Texture* source, const WGPUOrigin3D& sourceOrigin, const WGPUExtent3D& sourceExtent, uint32_t sourceMipLevel, uint32_t sourceArrayLayer, Texture* destination, const WGPUOrigin3D& destinationOrigin, const WGPUExtent3D& destinationExtent, uint32_t destinationMipLevel, uint32_t destinationArrayLayer, WGPUFilterMode filter);

    // Query operations
    void writeTimestamp(WGPUQuerySet querySet, uint32_t queryIndex);
    void resetQuerySet(WGPUQuerySet querySet, uint32_t firstQuery, uint32_t queryCount);
    void resolveQuerySet(WGPUQuerySet querySet, uint32_t firstQuery, uint32_t queryCount, WGPUBuffer buffer, uint64_t destinationOffset);

private:
    Device* m_device = nullptr; // Non-owning pointer
    bool m_isBundleEncoder = false;
    WGPUCommandEncoder m_encoder = nullptr;
    WGPUCommandBuffer m_commandBuffer = nullptr;
    WGPURenderBundleEncoder m_bundleEncoder = nullptr;
    WGPURenderBundle m_renderBundle = nullptr;
};

} // namespace gfx::backend::webgpu::core

#endif // GFX_WEBGPU_COMMAND_ENCODER_H