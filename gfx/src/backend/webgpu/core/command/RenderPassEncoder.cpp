#include "RenderPassEncoder.h"

#include "../command/CommandEncoder.h"
#include "../render/Framebuffer.h"
#include "../render/RenderPass.h"
#include "../resource/Buffer.h"
#include "../resource/Texture.h"
#include "../resource/TextureView.h"
#include "../system/Device.h"
#include "../util/Utils.h"

#include "../../../../common/Logger.h"

#include <optional>
#include <stdexcept>
#include <vector>

namespace gfx::backend::webgpu::core {

namespace {
    // Combine render pass color ops with framebuffer views and begin-info clear values
    std::vector<WGPURenderPassColorAttachment> buildColorAttachments(
        const RenderPassCreateInfo& passInfo, const FramebufferCreateInfo& fbInfo, const RenderPassEncoderBeginInfo& beginInfo)
    {
        std::vector<WGPURenderPassColorAttachment> attachments;
        attachments.reserve(fbInfo.colorAttachmentViews.size());

        for (size_t i = 0; i < fbInfo.colorAttachmentViews.size(); ++i) {
            WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
            // Get actual WGPUTextureView handle from TextureView pointer
            attachment.view = fbInfo.colorAttachmentViews[i] ? fbInfo.colorAttachmentViews[i]->handle() : nullptr;
            attachment.loadOp = passInfo.colorAttachments[i].loadOp;
            attachment.storeOp = passInfo.colorAttachments[i].storeOp;

            // Set resolve target if provided
            if (i < fbInfo.colorResolveTargetViews.size() && fbInfo.colorResolveTargetViews[i]) {
                attachment.resolveTarget = fbInfo.colorResolveTargetViews[i]->handle();
            }

            // Set clear color from begin info
            if (i < beginInfo.colorClearValues.size()) {
                attachment.clearValue = beginInfo.colorClearValues[i];
            }

            attachments.push_back(attachment);
        }

        return attachments;
    }

    // The depth/stencil attachment is optional - present only when the framebuffer has one
    std::optional<WGPURenderPassDepthStencilAttachment> buildDepthStencilAttachment(
        const RenderPassCreateInfo& passInfo, const FramebufferCreateInfo& fbInfo, const RenderPassEncoderBeginInfo& beginInfo)
    {
        if (!fbInfo.depthStencilAttachmentView) {
            return std::nullopt;
        }

        const auto& depthStencilAtt = passInfo.depthStencilAttachment.value();

        WGPURenderPassDepthStencilAttachment wgpuDepthStencil = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
        // Get actual WGPUTextureView handle from TextureView pointer
        wgpuDepthStencil.view = fbInfo.depthStencilAttachmentView->handle();
        wgpuDepthStencil.depthLoadOp = depthStencilAtt.depthLoadOp;
        wgpuDepthStencil.depthStoreOp = depthStencilAtt.depthStoreOp;
        wgpuDepthStencil.depthClearValue = beginInfo.depthClearValue;

        // Only set stencil operations for formats that have a stencil aspect
        WGPUTextureFormat format = fbInfo.depthStencilAttachmentView->getTexture()->getFormat();
        if (hasStencil(format)) {
            wgpuDepthStencil.stencilLoadOp = depthStencilAtt.stencilLoadOp;
            wgpuDepthStencil.stencilStoreOp = depthStencilAtt.stencilStoreOp;
            wgpuDepthStencil.stencilClearValue = beginInfo.stencilClearValue;
        } else {
            wgpuDepthStencil.stencilLoadOp = WGPULoadOp_Undefined;
            wgpuDepthStencil.stencilStoreOp = WGPUStoreOp_Undefined;
            wgpuDepthStencil.stencilClearValue = 0;
        }

        return wgpuDepthStencil;
    }
} // anonymous namespace

RenderPassEncoder::RenderPassEncoder(CommandEncoder* commandEncoder, RenderPass* renderPass, Framebuffer* framebuffer, const RenderPassEncoderBeginInfo& beginInfo)
{
    // Combine render pass ops with framebuffer views
    const RenderPassCreateInfo& passInfo = renderPass->getCreateInfo();
    const FramebufferCreateInfo& fbInfo = framebuffer->getCreateInfo();

    std::vector<WGPURenderPassColorAttachment> wgpuColorAttachments = buildColorAttachments(passInfo, fbInfo, beginInfo);
    std::optional<WGPURenderPassDepthStencilAttachment> wgpuDepthStencil = buildDepthStencilAttachment(passInfo, fbInfo, beginInfo);

    // ...then assemble the descriptor in one place; the locals above outlive the create call
    WGPURenderPassDescriptor wgpuDesc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    if (!wgpuColorAttachments.empty()) {
        wgpuDesc.colorAttachments = wgpuColorAttachments.data();
        wgpuDesc.colorAttachmentCount = static_cast<uint32_t>(wgpuColorAttachments.size());
    }
    if (wgpuDepthStencil.has_value()) {
        wgpuDesc.depthStencilAttachment = &wgpuDepthStencil.value();
    }

    wgpuDesc.occlusionQuerySet = beginInfo.occlusionQuerySet;
    m_occlusionQuerySet = beginInfo.occlusionQuerySet;

    WGPUPassTimestampWrites tsWrites = WGPU_PASS_TIMESTAMP_WRITES_INIT;
    if (beginInfo.timestampQuerySet) {
        tsWrites.querySet = beginInfo.timestampQuerySet;
        tsWrites.beginningOfPassWriteIndex = 0;
        tsWrites.endOfPassWriteIndex = 1;
        wgpuDesc.timestampWrites = &tsWrites;
    }

    m_encoder = wgpuCommandEncoderBeginRenderPass(commandEncoder->handle(), &wgpuDesc);
    if (!m_encoder) {
        throw std::runtime_error("Failed to create WebGPU render pass encoder");
    }
}

RenderPassEncoder::RenderPassEncoder(CommandEncoder* bundleCommandEncoder)
    : m_bundleEncoder(bundleCommandEncoder->getBundleEncoder())
{
}

RenderPassEncoder::~RenderPassEncoder()
{
    if (m_bundleEncoder) {
        // Bundle mode: nothing to do - CommandEncoder owns the bundle encoder
        return;
    }

    if (m_encoder) {
        if (!m_ended) {
            wgpuRenderPassEncoderEnd(m_encoder);
        }
        wgpuRenderPassEncoderRelease(m_encoder);
    }
}

void RenderPassEncoder::setPipeline(WGPURenderPipeline pipeline)
{
    if (m_bundleEncoder) {
        wgpuRenderBundleEncoderSetPipeline(m_bundleEncoder, pipeline);
    } else {
        wgpuRenderPassEncoderSetPipeline(m_encoder, pipeline);
    }
}

void RenderPassEncoder::setBindGroup(uint32_t index, WGPUBindGroup bindGroup, const uint32_t* dynamicOffsets, uint32_t dynamicOffsetCount)
{
    if (m_bundleEncoder) {
        wgpuRenderBundleEncoderSetBindGroup(m_bundleEncoder, index, bindGroup, dynamicOffsetCount, dynamicOffsets);
    } else {
        wgpuRenderPassEncoderSetBindGroup(m_encoder, index, bindGroup, dynamicOffsetCount, dynamicOffsets);
    }
}

void RenderPassEncoder::setVertexBuffer(uint32_t slot, Buffer* buffer, uint64_t offset, uint64_t size)
{
    if (m_bundleEncoder) {
        wgpuRenderBundleEncoderSetVertexBuffer(m_bundleEncoder, slot, buffer->handle(), offset, size);
    } else {
        wgpuRenderPassEncoderSetVertexBuffer(m_encoder, slot, buffer->handle(), offset, size);
    }
}

void RenderPassEncoder::setIndexBuffer(Buffer* buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size)
{
    if (m_bundleEncoder) {
        wgpuRenderBundleEncoderSetIndexBuffer(m_bundleEncoder, buffer->handle(), format, offset, size);
    } else {
        wgpuRenderPassEncoderSetIndexBuffer(m_encoder, buffer->handle(), format, offset, size);
    }
}

void RenderPassEncoder::setViewport(float x, float y, float width, float height, float minDepth, float maxDepth)
{
    // Viewport is not supported in render bundles - only set on regular render pass
    if (!m_bundleEncoder) {
        wgpuRenderPassEncoderSetViewport(m_encoder, x, y, width, height, minDepth, maxDepth);
    }
}

void RenderPassEncoder::setScissorRect(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
    // Scissor rect is not supported in render bundles - only set on regular render pass
    if (!m_bundleEncoder) {
        wgpuRenderPassEncoderSetScissorRect(m_encoder, x, y, width, height);
    }
}

void RenderPassEncoder::setBlendConstant(const WGPUColor& color)
{
    // Blend constant is not supported in render bundles - only set on regular render pass
    if (!m_bundleEncoder) {
        wgpuRenderPassEncoderSetBlendConstant(m_encoder, &color);
    }
}

void RenderPassEncoder::setStencilReference(uint32_t reference)
{
    // Stencil reference is not supported in render bundles - only set on regular render pass
    if (!m_bundleEncoder) {
        wgpuRenderPassEncoderSetStencilReference(m_encoder, reference);
    }
}

void RenderPassEncoder::draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
    if (m_bundleEncoder) {
        wgpuRenderBundleEncoderDraw(m_bundleEncoder, vertexCount, instanceCount, firstVertex, firstInstance);
    } else {
        wgpuRenderPassEncoderDraw(m_encoder, vertexCount, instanceCount, firstVertex, firstInstance);
    }
}

void RenderPassEncoder::drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance)
{
    if (m_bundleEncoder) {
        wgpuRenderBundleEncoderDrawIndexed(m_bundleEncoder, indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
    } else {
        wgpuRenderPassEncoderDrawIndexed(m_encoder, indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
    }
}

void RenderPassEncoder::drawIndirect(WGPUBuffer buffer, uint64_t offset)
{
    if (m_bundleEncoder) {
        wgpuRenderBundleEncoderDrawIndirect(m_bundleEncoder, buffer, offset);
    } else {
        wgpuRenderPassEncoderDrawIndirect(m_encoder, buffer, offset);
    }
}

void RenderPassEncoder::drawIndexedIndirect(WGPUBuffer buffer, uint64_t offset)
{
    if (m_bundleEncoder) {
        wgpuRenderBundleEncoderDrawIndexedIndirect(m_bundleEncoder, buffer, offset);
    } else {
        wgpuRenderPassEncoderDrawIndexedIndirect(m_encoder, buffer, offset);
    }
}

void RenderPassEncoder::beginOcclusionQuery(WGPUQuerySet querySet, uint32_t queryIndex)
{
    if (m_bundleEncoder) {
        return; // Occlusion queries not supported in render bundles
    }
    if (!isOcclusionQuerySetCompatible(querySet)) {
        throw std::runtime_error("Occlusion query set is not compatible with render pass begin descriptor");
    }
    wgpuRenderPassEncoderBeginOcclusionQuery(m_encoder, queryIndex);
}

void RenderPassEncoder::endOcclusionQuery()
{
    if (m_bundleEncoder) {
        return; // Occlusion queries not supported in render bundles
    }
    wgpuRenderPassEncoderEndOcclusionQuery(m_encoder);
}

WGPURenderPassEncoder RenderPassEncoder::handle() const
{
    return m_encoder;
}

bool RenderPassEncoder::isBundleMode() const
{
    return m_bundleEncoder != nullptr;
}

bool RenderPassEncoder::isOcclusionQuerySetCompatible(WGPUQuerySet querySet) const
{
    return m_occlusionQuerySet != nullptr && querySet == m_occlusionQuerySet;
}

void RenderPassEncoder::executeBundles(const WGPURenderBundle* bundles, uint32_t count)
{
    wgpuRenderPassEncoderExecuteBundles(m_encoder, count, bundles);
}

} // namespace gfx::backend::webgpu::core