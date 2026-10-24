#include "RenderPassEncoder.h"

#include "CommandEncoder.h"

#include "../render/Framebuffer.h"
#include "../render/RenderPass.h"
#include "../render/RenderPipeline.h"
#include "../resource/BindGroup.h"
#include "../resource/Buffer.h"
#include "../system/Device.h"

#include <stdexcept>
#include <vector>

namespace gfx::backend::vulkan::core {

RenderPassEncoder::RenderPassEncoder(CommandEncoder* commandEncoder, RenderPass* renderPass, Framebuffer* framebuffer, const RenderPassEncoderBeginInfo& beginInfo)
    : m_commandBuffer(commandEncoder->handle())
    , m_device(commandEncoder->getDevice())
    , m_commandEncoder(commandEncoder)
    , m_isBundleMode(false)
    , m_passOcclusionQueryPool(beginInfo.occlusionQueryPool)
    , m_passTimestampQueryPool(beginInfo.timestampQueryPool)
{
    // Build clear values array
    std::vector<VkClearValue> clearValues;

    // Add color clear values with dummy values for resolve attachments
    const auto& colorHasResolve = renderPass->colorHasResolve();
    for (size_t i = 0; i < beginInfo.colorClearValues.size(); ++i) {
        // Add the color attachment clear value
        VkClearValue clearValue{};
        clearValue.color = beginInfo.colorClearValues[i];
        clearValues.push_back(clearValue);

        // If this color attachment has a resolve target, add a dummy clear value for it
        // (resolve attachments use LOAD_OP_DONT_CARE so the value doesn't matter)
        if (i < colorHasResolve.size() && colorHasResolve[i]) {
            VkClearValue dummyClear{};
            dummyClear.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
            clearValues.push_back(dummyClear);
        }
    }

    // Add depth/stencil clear value if needed
    if (renderPass->hasDepthStencil()) {
        VkClearValue depthStencilClear{};
        depthStencilClear.depthStencil.depth = beginInfo.depthClearValue;
        depthStencilClear.depthStencil.stencil = beginInfo.stencilClearValue;
        clearValues.push_back(depthStencilClear);
    }

    // Begin render pass
    VkRenderPassBeginInfo vkBeginInfo{};
    vkBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    vkBeginInfo.renderPass = renderPass->handle();
    vkBeginInfo.framebuffer = framebuffer->handle();
    vkBeginInfo.renderArea.offset = { 0, 0 };
    vkBeginInfo.renderArea.extent.width = framebuffer->width();
    vkBeginInfo.renderArea.extent.height = framebuffer->height();
    vkBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    vkBeginInfo.pClearValues = clearValues.data();

    // Always inline: executed bundles are replayed into this pass rather than run as secondary command buffers
    vkCmdBeginRenderPass(m_commandBuffer, &vkBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Match WebGPU defaults: blend constant (0,0,0,0) and stencil reference 0 at pass begin
    const float defaultBlendConstants[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    vkCmdSetBlendConstants(m_commandBuffer, defaultBlendConstants);
    vkCmdSetStencilReference(m_commandBuffer, VK_STENCIL_FACE_FRONT_AND_BACK, 0);

    // Write start timestamp if requested
    if (m_passTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(m_commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_passTimestampQueryPool, 0);
    }
}

RenderPassEncoder::RenderPassEncoder(CommandEncoder* bundleCommandEncoder)
    : m_commandBuffer(bundleCommandEncoder->handle())
    , m_device(bundleCommandEncoder->getDevice())
    , m_commandEncoder(bundleCommandEncoder)
    , m_isBundleMode(true)
{
}

RenderPassEncoder::~RenderPassEncoder()
{
    if (m_isBundleMode) {
        return; // A bundle only records commands; it has no render pass to end
    }

    // Write end timestamp before ending the pass
    if (m_passTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(m_commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_passTimestampQueryPool, 1);
    }

    vkCmdEndRenderPass(m_commandBuffer);
}

VkCommandBuffer RenderPassEncoder::handle() const
{
    return m_commandBuffer;
}

Device* RenderPassEncoder::device() const
{
    return m_device;
}

CommandEncoder* RenderPassEncoder::commandEncoder() const
{
    return m_commandEncoder;
}

bool RenderPassEncoder::isBundleMode() const
{
    return m_isBundleMode;
}

void RenderPassEncoder::setPipeline(RenderPipeline* pipeline)
{
    VkPipeline vkPipeline = pipeline->handle();
    if (m_isBundleMode) {
        m_commandEncoder->recordBundleCommand([=](VkCommandBuffer cmd) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipeline); });
    } else {
        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipeline);
    }
    m_commandEncoder->setCurrentPipelineLayout(pipeline->layout());
}

void RenderPassEncoder::setBindGroup(uint32_t index, BindGroup* bindGroup, const uint32_t* dynamicOffsets, uint32_t dynamicOffsetCount)
{
    VkPipelineLayout layout = m_commandEncoder->currentPipelineLayout();
    if (layout != VK_NULL_HANDLE) {
        VkDescriptorSet set = bindGroup->handle();
        if (m_isBundleMode) {
            // Copied: the caller's offsets array is gone by the time the bundle is replayed
            std::vector<uint32_t> offsets(dynamicOffsets, dynamicOffsets + dynamicOffsetCount);
            m_commandEncoder->recordBundleCommand([=](VkCommandBuffer cmd) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, index, 1, &set, static_cast<uint32_t>(offsets.size()), offsets.data());
            });
        } else {
            vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, index, 1, &set, dynamicOffsetCount, dynamicOffsets);
        }
    }
}

void RenderPassEncoder::setVertexBuffer(uint32_t slot, Buffer* buffer, uint64_t offset)
{
    VkBuffer vkBuf = buffer->handle();
    VkDeviceSize vkOffset = offset;
    if (m_isBundleMode) {
        m_commandEncoder->recordBundleCommand([=](VkCommandBuffer cmd) { vkCmdBindVertexBuffers(cmd, slot, 1, &vkBuf, &vkOffset); });
    } else {
        vkCmdBindVertexBuffers(m_commandBuffer, slot, 1, &vkBuf, &vkOffset);
    }
}

void RenderPassEncoder::setIndexBuffer(Buffer* buffer, VkIndexType indexType, uint64_t offset)
{
    VkBuffer vkBuf = buffer->handle();
    if (m_isBundleMode) {
        m_commandEncoder->recordBundleCommand([=](VkCommandBuffer cmd) { vkCmdBindIndexBuffer(cmd, vkBuf, offset, indexType); });
    } else {
        vkCmdBindIndexBuffer(m_commandBuffer, vkBuf, offset, indexType);
    }
}

void RenderPassEncoder::setViewport(const Viewport& viewport)
{
    if (m_isBundleMode) {
        return; // Dynamic state comes from the executing render pass, as in WebGPU
    }

    VkViewport vkViewport{};
    vkViewport.x = viewport.x;
    vkViewport.y = viewport.y;
    vkViewport.width = viewport.width;
    vkViewport.height = viewport.height;
    vkViewport.minDepth = viewport.minDepth;
    vkViewport.maxDepth = viewport.maxDepth;
    vkCmdSetViewport(m_commandBuffer, 0, 1, &vkViewport);
}

void RenderPassEncoder::setScissorRect(const ScissorRect& scissor)
{
    if (m_isBundleMode) {
        return; // Dynamic state comes from the executing render pass, as in WebGPU
    }

    VkRect2D vkScissor{};
    vkScissor.offset = { scissor.x, scissor.y };
    vkScissor.extent = { scissor.width, scissor.height };
    vkCmdSetScissor(m_commandBuffer, 0, 1, &vkScissor);
}

void RenderPassEncoder::setBlendConstant(const float color[4])
{
    if (m_isBundleMode) {
        return; // Dynamic state comes from the executing render pass, as in WebGPU
    }
    vkCmdSetBlendConstants(m_commandBuffer, color);
}

void RenderPassEncoder::setStencilReference(uint32_t reference)
{
    if (m_isBundleMode) {
        return; // Dynamic state comes from the executing render pass, as in WebGPU
    }
    vkCmdSetStencilReference(m_commandBuffer, VK_STENCIL_FACE_FRONT_AND_BACK, reference);
}

void RenderPassEncoder::draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
    if (m_isBundleMode) {
        m_commandEncoder->recordBundleCommand([=](VkCommandBuffer cmd) { vkCmdDraw(cmd, vertexCount, instanceCount, firstVertex, firstInstance); });
    } else {
        vkCmdDraw(m_commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
    }
}

void RenderPassEncoder::drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance)
{
    if (m_isBundleMode) {
        m_commandEncoder->recordBundleCommand([=](VkCommandBuffer cmd) { vkCmdDrawIndexed(cmd, indexCount, instanceCount, firstIndex, baseVertex, firstInstance); });
    } else {
        vkCmdDrawIndexed(m_commandBuffer, indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
    }
}

void RenderPassEncoder::drawIndirect(Buffer* buffer, uint64_t offset)
{
    VkBuffer vkBuf = buffer->handle();
    if (m_isBundleMode) {
        m_commandEncoder->recordBundleCommand([=](VkCommandBuffer cmd) { vkCmdDrawIndirect(cmd, vkBuf, offset, 1, 0); });
    } else {
        vkCmdDrawIndirect(m_commandBuffer, vkBuf, offset, 1, 0);
    }
}

void RenderPassEncoder::drawIndexedIndirect(Buffer* buffer, uint64_t offset)
{
    VkBuffer vkBuf = buffer->handle();
    if (m_isBundleMode) {
        m_commandEncoder->recordBundleCommand([=](VkCommandBuffer cmd) { vkCmdDrawIndexedIndirect(cmd, vkBuf, offset, 1, 0); });
    } else {
        vkCmdDrawIndexedIndirect(m_commandBuffer, vkBuf, offset, 1, 0);
    }
}

void RenderPassEncoder::beginOcclusionQuery(VkQueryPool queryPool, uint32_t queryIndex, VkQueryControlFlags flags)
{
    if (m_isBundleMode) {
        return; // Occlusion queries not supported in render bundles
    }
    if (!isOcclusionQueryPoolCompatible(queryPool)) {
        throw std::runtime_error("Occlusion query pool is not compatible with render pass begin descriptor");
    }

    m_activeQueryPool = queryPool;
    m_activeQueryIndex = queryIndex;
    vkCmdBeginQuery(m_commandBuffer, queryPool, queryIndex, flags);
}

void RenderPassEncoder::endOcclusionQuery()
{
    if (m_isBundleMode) {
        return; // Occlusion queries not supported in render bundles
    }
    if (m_activeQueryPool != VK_NULL_HANDLE) {
        vkCmdEndQuery(m_commandBuffer, m_activeQueryPool, m_activeQueryIndex);
        m_activeQueryPool = VK_NULL_HANDLE;
        m_activeQueryIndex = 0;
    }
}

bool RenderPassEncoder::isOcclusionQueryPoolCompatible(VkQueryPool queryPool) const
{
    return m_passOcclusionQueryPool != VK_NULL_HANDLE && queryPool == m_passOcclusionQueryPool;
}

void RenderPassEncoder::executeBundles(const CommandEncoder* const* bundles, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        bundles[i]->replayBundle(m_commandBuffer);
    }
}

} // namespace gfx::backend::vulkan::core
