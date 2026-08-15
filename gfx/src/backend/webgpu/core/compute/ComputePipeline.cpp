#include "ComputePipeline.h"

#include "../system/Device.h"

#include <stdexcept>

namespace gfx::backend::webgpu::core {

namespace {
    std::vector<WGPUConstantEntry> makeConstantEntries(const std::vector<ConstantEntry>& constants)
    {
        std::vector<WGPUConstantEntry> entries;
        entries.reserve(constants.size());
        for (const auto& constant : constants) {
            WGPUConstantEntry entry = WGPU_CONSTANT_ENTRY_INIT;
            entry.key = { constant.key.c_str(), WGPU_STRLEN };
            entry.value = constant.value;
            entries.push_back(entry);
        }
        return entries;
    }
} // namespace

ComputePipeline::ComputePipeline(Device* device, const ComputePipelineCreateInfo& createInfo)
{
    // Create pipeline layout if bind group layouts are provided
    WGPUPipelineLayout pipelineLayout = nullptr;
    if (!createInfo.bindGroupLayouts.empty()) {
        WGPUPipelineLayoutDescriptor layoutDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        layoutDesc.bindGroupLayouts = createInfo.bindGroupLayouts.data();
        layoutDesc.bindGroupLayoutCount = static_cast<uint32_t>(createInfo.bindGroupLayouts.size());
        pipelineLayout = wgpuDeviceCreatePipelineLayout(device->handle(), &layoutDesc);
    }

    // Borrowed by the descriptor below - must stay alive across the create call
    const std::vector<WGPUConstantEntry> constants = makeConstantEntries(createInfo.constants);

    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.layout = pipelineLayout;
    desc.compute.module = createInfo.module;
    desc.compute.entryPoint = { createInfo.entryPoint, WGPU_STRLEN };
    desc.compute.constants = constants.empty() ? nullptr : constants.data();
    desc.compute.constantCount = constants.size();

    m_pipeline = wgpuDeviceCreateComputePipeline(device->handle(), &desc);

    // Release the pipeline layout if we created one (pipeline holds its own reference)
    if (pipelineLayout) {
        wgpuPipelineLayoutRelease(pipelineLayout);
    }

    if (!m_pipeline) {
        throw std::runtime_error("Failed to create WebGPU ComputePipeline");
    }
}

ComputePipeline::~ComputePipeline()
{
    if (m_pipeline) {
        wgpuComputePipelineRelease(m_pipeline);
    }
}

WGPUComputePipeline ComputePipeline::handle() const
{
    return m_pipeline;
}

} // namespace gfx::backend::webgpu::core