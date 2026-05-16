#include "../command/CommandEncoder.h"

#include "../render/RenderPass.h"
#include "../resource/Buffer.h"
#include "../resource/Texture.h"
#include "../system/Device.h"
#include "../util/Blit.h"
#include "../util/Utils.h"

#include <stdexcept>

namespace gfx::backend::webgpu::core {

namespace {

    WGPUCommandEncoder createEncoder(WGPUDevice device, const char* label = nullptr)
    {
        WGPUCommandEncoderDescriptor desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        if (label) {
            desc.label = toStringView(label);
        }

        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &desc);
        if (!encoder) {
            throw std::runtime_error("Failed to create WebGPU CommandEncoder");
        }
        return encoder;
    }

    void destroyEncoder(WGPUCommandEncoder& encoder)
    {
        if (encoder) {
            wgpuCommandEncoderRelease(encoder);
            encoder = nullptr;
        }
    }

    WGPURenderBundleEncoder createBundleEncoder(WGPUDevice device, const RenderPassCreateInfo& passInfo)
    {
        WGPURenderBundleEncoderDescriptor desc = WGPU_RENDER_BUNDLE_ENCODER_DESCRIPTOR_INIT;

        std::vector<WGPUTextureFormat> colorFormats;
        for (const auto& colorAtt : passInfo.colorAttachments) {
            colorFormats.push_back(colorAtt.format);
        }
        desc.colorFormats = colorFormats.data();
        desc.colorFormatCount = static_cast<uint32_t>(colorFormats.size());

        if (passInfo.depthStencilAttachment.has_value()) {
            desc.depthStencilFormat = passInfo.depthStencilAttachment->format;
        } else {
            desc.depthStencilFormat = WGPUTextureFormat_Undefined;
        }

        desc.sampleCount = passInfo.sampleCount;

        WGPURenderBundleEncoder encoder = wgpuDeviceCreateRenderBundleEncoder(device, &desc);
        if (!encoder) {
            throw std::runtime_error("Failed to create WebGPU render bundle encoder");
        }
        return encoder;
    }

    void destroyBundleEncoder(WGPURenderBundleEncoder& encoder)
    {
        if (encoder) {
            wgpuRenderBundleEncoderRelease(encoder);
            encoder = nullptr;
        }
    }

} // anonymous namespace

CommandEncoder::CommandEncoder(Device* device, const CommandEncoderCreateInfo& createInfo)
    : m_device(device)
    , m_isBundleEncoder(false)
    , m_encoder(createEncoder(device->handle(), createInfo.label))
{
}

CommandEncoder::CommandEncoder(Device* device)
    : m_device(device)
    , m_isBundleEncoder(true)
{
}

CommandEncoder::~CommandEncoder()
{
    if (m_renderBundle) {
        wgpuRenderBundleRelease(m_renderBundle);
    }
    destroyBundleEncoder(m_bundleEncoder);
    if (m_commandBuffer) {
        wgpuCommandBufferRelease(m_commandBuffer);
    }
    destroyEncoder(m_encoder);
}

WGPUCommandEncoder CommandEncoder::handle() const
{
    return m_encoder;
}

WGPUCommandBuffer CommandEncoder::commandBuffer() const
{
    return m_commandBuffer;
}

Device* CommandEncoder::getDevice() const
{
    return m_device;
}

bool CommandEncoder::isBundleEncoder() const
{
    return m_isBundleEncoder;
}

void CommandEncoder::beginBundle(RenderPass* renderPass)
{
    destroyBundleEncoder(m_bundleEncoder);
    if (m_renderBundle) {
        wgpuRenderBundleRelease(m_renderBundle);
        m_renderBundle = nullptr;
    }

    m_bundleEncoder = createBundleEncoder(m_device->handle(), renderPass->getCreateInfo());
}

void CommandEncoder::end()
{
    if (m_isBundleEncoder) {
        if (!m_bundleEncoder) {
            return;
        }
        WGPURenderBundleDescriptor bundleDesc = WGPU_RENDER_BUNDLE_DESCRIPTOR_INIT;
        m_renderBundle = wgpuRenderBundleEncoderFinish(m_bundleEncoder, &bundleDesc);
        destroyBundleEncoder(m_bundleEncoder);
    } else {
        if (!m_encoder) {
            return;
        }
        WGPUCommandBufferDescriptor cmdDesc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
        m_commandBuffer = wgpuCommandEncoderFinish(m_encoder, &cmdDesc);
        destroyEncoder(m_encoder);
    }
}

WGPURenderBundleEncoder CommandEncoder::getBundleEncoder() const
{
    return m_bundleEncoder;
}

WGPURenderBundle CommandEncoder::getRenderBundle() const
{
    return m_renderBundle;
}

void CommandEncoder::begin()
{
    if (m_encoder) {
        return; // Already have an active encoder
    }

    // Release previous command buffer if any
    if (m_commandBuffer) {
        wgpuCommandBufferRelease(m_commandBuffer);
        m_commandBuffer = nullptr;
    }

    m_encoder = createEncoder(m_device->handle());
}

void CommandEncoder::reset()
{
    // Release old state
    if (m_commandBuffer) {
        wgpuCommandBufferRelease(m_commandBuffer);
        m_commandBuffer = nullptr;
    }
    destroyEncoder(m_encoder);

    m_encoder = createEncoder(m_device->handle());
}

// Copy operations
void CommandEncoder::copyBufferToBuffer(Buffer* source, uint64_t sourceOffset, Buffer* destination, uint64_t destinationOffset, uint64_t size)
{
    wgpuCommandEncoderCopyBufferToBuffer(m_encoder, source->handle(), sourceOffset, destination->handle(), destinationOffset, size);
}

void CommandEncoder::copyBufferToTexture(Buffer* source, uint64_t sourceOffset, Texture* destination, const WGPUOrigin3D& origin, const WGPUExtent3D& extent, uint32_t mipLevel)
{
    uint32_t bytesPerRow = calculateBytesPerRow(destination->getFormat(), extent.width);

    WGPUTexelCopyBufferInfo sourceInfo = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    sourceInfo.buffer = source->handle();
    sourceInfo.layout.offset = sourceOffset;
    sourceInfo.layout.bytesPerRow = bytesPerRow;

    WGPUTexelCopyTextureInfo destInfo = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    destInfo.texture = destination->handle();
    destInfo.mipLevel = mipLevel;
    destInfo.origin = origin;

    wgpuCommandEncoderCopyBufferToTexture(m_encoder, &sourceInfo, &destInfo, &extent);
}

void CommandEncoder::copyTextureToBuffer(Texture* source, const WGPUOrigin3D& origin, uint32_t mipLevel, Buffer* destination, uint64_t destinationOffset, const WGPUExtent3D& extent)
{
    uint32_t bytesPerRow = calculateBytesPerRow(source->getFormat(), extent.width);

    WGPUTexelCopyTextureInfo sourceInfo = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    sourceInfo.texture = source->handle();
    sourceInfo.mipLevel = mipLevel;
    sourceInfo.origin = origin;

    WGPUTexelCopyBufferInfo destInfo = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    destInfo.buffer = destination->handle();
    destInfo.layout.offset = destinationOffset;
    destInfo.layout.bytesPerRow = bytesPerRow;

    wgpuCommandEncoderCopyTextureToBuffer(m_encoder, &sourceInfo, &destInfo, &extent);
}

void CommandEncoder::copyTextureToTexture(Texture* source, const WGPUOrigin3D& sourceOrigin, uint32_t sourceMipLevel, Texture* destination, const WGPUOrigin3D& destinationOrigin, uint32_t destinationMipLevel, const WGPUExtent3D& extent)
{
    // For 2D textures and arrays, depth represents layer count
    // For 3D textures, it represents actual depth
    WGPUOrigin3D srcOrigin = sourceOrigin;
    WGPUOrigin3D dstOrigin = destinationOrigin;
    if (source->getDimension() != WGPUTextureDimension_3D) {
        srcOrigin.z = 0;
        dstOrigin.z = 0;
    }

    WGPUTexelCopyTextureInfo sourceInfo = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    sourceInfo.texture = source->handle();
    sourceInfo.mipLevel = sourceMipLevel;
    sourceInfo.origin = srcOrigin;

    WGPUTexelCopyTextureInfo destInfo = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    destInfo.texture = destination->handle();
    destInfo.mipLevel = destinationMipLevel;
    destInfo.origin = dstOrigin;

    wgpuCommandEncoderCopyTextureToTexture(m_encoder, &sourceInfo, &destInfo, &extent);
}

void CommandEncoder::blitTextureToTexture(Texture* source, const WGPUOrigin3D& sourceOrigin, const WGPUExtent3D& sourceExtent, uint32_t sourceMipLevel, Texture* destination, const WGPUOrigin3D& destinationOrigin, const WGPUExtent3D& destinationExtent, uint32_t destinationMipLevel, WGPUFilterMode filter)
{
    // Get the Blit helper from the device
    Blit* blit = m_device->getBlit();
    blit->execute(m_encoder,
        source->handle(), sourceOrigin, sourceExtent, sourceMipLevel,
        destination->handle(), destinationOrigin, destinationExtent, destinationMipLevel,
        filter);
}

void CommandEncoder::writeTimestamp(WGPUQuerySet querySet, uint32_t queryIndex)
{
    wgpuCommandEncoderWriteTimestamp(m_encoder, querySet, queryIndex);
}

void CommandEncoder::resetQuerySet(WGPUQuerySet querySet, uint32_t firstQuery, uint32_t queryCount)
{
    // WebGPU handles query reset implicitly - no-op
    (void)querySet;
    (void)firstQuery;
    (void)queryCount;
}

void CommandEncoder::resolveQuerySet(WGPUQuerySet querySet, uint32_t firstQuery, uint32_t queryCount, WGPUBuffer buffer, uint64_t destinationOffset)
{
    wgpuCommandEncoderResolveQuerySet(m_encoder, querySet, firstQuery, queryCount, buffer, destinationOffset);
}

} // namespace gfx::backend::webgpu::core