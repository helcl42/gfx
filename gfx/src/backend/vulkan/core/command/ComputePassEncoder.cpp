#include "ComputePassEncoder.h"

#include "CommandEncoder.h"

#include "../compute/ComputePipeline.h"
#include "../resource/BindGroup.h"
#include "../resource/Buffer.h"
#include "../system/Device.h"

namespace gfx::backend::vulkan::core {

ComputePassEncoder::ComputePassEncoder(CommandEncoder* commandEncoder, const ComputePassEncoderCreateInfo& createInfo)
    : m_commandBuffer(commandEncoder->handle())
    , m_device(commandEncoder->getDevice())
    , m_commandEncoder(commandEncoder)
{
    (void)createInfo; // Label unused for now
}

VkCommandBuffer ComputePassEncoder::handle() const
{
    return m_commandBuffer;
}

Device* ComputePassEncoder::device() const
{
    return m_device;
}

CommandEncoder* ComputePassEncoder::commandEncoder() const
{
    return m_commandEncoder;
}

void ComputePassEncoder::setPipeline(ComputePipeline* pipeline)
{
    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle());
    m_pipeline = pipeline;

    // Groups set before a pipeline could take them apply to this one; already bound groups stay as they are
    for (uint32_t index = 0; index < m_pendingBindGroups.size(); ++index) {
        if (applyBindGroup(index, m_pendingBindGroups[index])) {
            m_pendingBindGroups[index] = {};
        }
    }
}

void ComputePassEncoder::setBindGroup(uint32_t index, BindGroup* bindGroup, const uint32_t* dynamicOffsets, uint32_t dynamicOffsetCount)
{
    BindGroupBinding binding{ bindGroup->handle(), std::vector<uint32_t>(dynamicOffsets, dynamicOffsets + dynamicOffsetCount) };
    if (index >= m_pendingBindGroups.size()) {
        m_pendingBindGroups.resize(index + 1);
    }
    // Bound now if the current pipeline takes it, otherwise once a pipeline that does is set
    m_pendingBindGroups[index] = applyBindGroup(index, binding) ? BindGroupBinding{} : std::move(binding);
}

bool ComputePassEncoder::applyBindGroup(uint32_t index, const BindGroupBinding& binding)
{
    if (!m_pipeline || binding.set == VK_NULL_HANDLE || index >= m_pipeline->bindGroupLayoutCount()) {
        return false;
    }
    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->layout(), index, 1, &binding.set, static_cast<uint32_t>(binding.dynamicOffsets.size()), binding.dynamicOffsets.data());
    return true;
}

void ComputePassEncoder::dispatchWorkgroups(uint32_t workgroupCountX, uint32_t workgroupCountY, uint32_t workgroupCountZ)
{
    vkCmdDispatch(m_commandBuffer, workgroupCountX, workgroupCountY, workgroupCountZ);
}

void ComputePassEncoder::dispatchIndirect(Buffer* buffer, uint64_t offset)
{
    vkCmdDispatchIndirect(m_commandBuffer, buffer->handle(), offset);
}

} // namespace gfx::backend::vulkan::core