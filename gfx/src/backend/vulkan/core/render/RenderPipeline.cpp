#include "RenderPipeline.h"

#include "../system/Device.h"

#include <stdexcept>

namespace gfx::backend::vulkan::core {

namespace {
    VkSpecializationInfo makeSpecializationInfo(const SpecializationConstants& constants)
    {
        VkSpecializationInfo info{};
        info.mapEntryCount = static_cast<uint32_t>(constants.entries.size());
        info.pMapEntries = constants.entries.data();
        info.dataSize = constants.data.size();
        info.pData = constants.data.data();
        return info;
    }

    VkPipelineShaderStageCreateInfo makeShaderStage(VkShaderStageFlagBits stage, VkShaderModule module, const char* entryPoint, const VkSpecializationInfo* specialization)
    {
        VkPipelineShaderStageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage = stage;
        info.module = module;
        info.pName = entryPoint;
        info.pSpecializationInfo = specialization;
        return info;
    }

    // Vertex input create-info together with the arrays it points into
    struct VertexInputState {
        std::vector<VkVertexInputBindingDescription> bindings;
        std::vector<VkVertexInputAttributeDescription> attributes;
        VkPipelineVertexInputStateCreateInfo info{};
    };

    VertexInputState buildVertexInputState(const VertexState& vertex)
    {
        VertexInputState state;
        for (size_t i = 0; i < vertex.buffers.size(); ++i) {
            const auto& bufferLayout = vertex.buffers[i];

            VkVertexInputBindingDescription binding{};
            binding.binding = static_cast<uint32_t>(i);
            binding.stride = static_cast<uint32_t>(bufferLayout.arrayStride);
            binding.inputRate = bufferLayout.inputRate;
            state.bindings.push_back(binding);

            state.attributes.insert(state.attributes.end(), bufferLayout.attributes.begin(), bufferLayout.attributes.end());
        }

        state.info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        state.info.vertexBindingDescriptionCount = static_cast<uint32_t>(state.bindings.size());
        state.info.pVertexBindingDescriptions = state.bindings.data();
        state.info.vertexAttributeDescriptionCount = static_cast<uint32_t>(state.attributes.size());
        state.info.pVertexAttributeDescriptions = state.attributes.data();
        return state;
    }

    // Color blend create-info together with the attachment array it points into
    struct ColorBlendState {
        std::vector<VkPipelineColorBlendAttachmentState> attachments;
        VkPipelineColorBlendStateCreateInfo info{};
    };

    ColorBlendState buildColorBlendState(const FragmentState& fragment)
    {
        ColorBlendState state;
        if (!fragment.targets.empty()) {
            for (const auto& target : fragment.targets) {
                state.attachments.push_back(target.blendState);
            }
        } else {
            VkPipelineColorBlendAttachmentState blendAttachment{};
            blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            blendAttachment.blendEnable = VK_FALSE;
            state.attachments.push_back(blendAttachment);
        }

        state.info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        state.info.attachmentCount = static_cast<uint32_t>(state.attachments.size());
        state.info.pAttachments = state.attachments.data();
        return state;
    }

    VkPipelineRasterizationStateCreateInfo makeRasterizerState(const PrimitiveState& primitive)
    {
        VkPipelineRasterizationStateCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        info.polygonMode = primitive.polygonMode;
        info.lineWidth = 1.0f;
        info.cullMode = primitive.cullMode;
        info.frontFace = primitive.frontFace;
        return info;
    }

    VkPipelineDepthStencilStateCreateInfo makeDepthStencilState(const DepthStencilState& depthStencilState)
    {
        VkPipelineDepthStencilStateCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        info.depthTestEnable = VK_TRUE;
        info.depthWriteEnable = depthStencilState.depthWriteEnabled ? VK_TRUE : VK_FALSE;
        info.depthCompareOp = depthStencilState.depthCompareOp;
        info.depthBoundsTestEnable = VK_FALSE;
        info.stencilTestEnable = VK_FALSE;
        return info;
    }
} // anonymous namespace

RenderPipeline::RenderPipeline(Device* device, const RenderPipelineCreateInfo& createInfo)
    : m_device(device)
{
    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(createInfo.bindGroupLayouts.size());
    pipelineLayoutInfo.pSetLayouts = createInfo.bindGroupLayouts.data();

    VkResult result = vkCreatePipelineLayout(m_device->handle(), &pipelineLayoutInfo, nullptr, &m_pipelineLayout);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    // Borrowed by the stages below until vkCreateGraphicsPipelines consumes them
    const VkSpecializationInfo vertexSpecialization = makeSpecializationInfo(createInfo.vertex.constants);
    const VkSpecializationInfo fragmentSpecialization = makeSpecializationInfo(createInfo.fragment.constants);

    // Shader stages (vertex always present, fragment optional)
    VkPipelineShaderStageCreateInfo shaderStages[2] = {
        makeShaderStage(VK_SHADER_STAGE_VERTEX_BIT, createInfo.vertex.module, createInfo.vertex.entryPoint,
            createInfo.vertex.constants.entries.empty() ? nullptr : &vertexSpecialization),
        {}
    };
    uint32_t stageCount = 1;
    if (createInfo.fragment.module != VK_NULL_HANDLE) {
        shaderStages[1] = makeShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, createInfo.fragment.module, createInfo.fragment.entryPoint,
            createInfo.fragment.constants.entries.empty() ? nullptr : &fragmentSpecialization);
        stageCount = 2;
    }

    VertexInputState vertexInput = buildVertexInputState(createInfo.vertex);
    ColorBlendState colorBlend = buildColorBlendState(createInfo.fragment);
    VkPipelineRasterizationStateCreateInfo rasterizer = makeRasterizerState(createInfo.primitive);

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = createInfo.primitive.topology;

    // Viewport and scissor are dynamic state; the pointers are ignored, only counts matter
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = createInfo.sampleCount;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_REFERENCE };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 4;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    if (createInfo.depthStencil.has_value()) {
        depthStencil = makeDepthStencilState(createInfo.depthStencil.value());
    }

    // Create graphics pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = stageCount;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInput.info;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlend.info;
    pipelineInfo.pDynamicState = &dynamicState;
    if (createInfo.depthStencil.has_value()) {
        pipelineInfo.pDepthStencilState = &depthStencil;
    }
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.renderPass = createInfo.renderPass;
    pipelineInfo.subpass = 0;

    result = vkCreateGraphicsPipelines(m_device->handle(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);

    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device->handle(), m_pipelineLayout, nullptr);
        throw std::runtime_error("Failed to create graphics pipeline");
    }
}

RenderPipeline::~RenderPipeline()
{
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device->handle(), m_pipeline, nullptr);
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device->handle(), m_pipelineLayout, nullptr);
    }
}

VkPipeline RenderPipeline::handle() const
{
    return m_pipeline;
}

VkPipelineLayout RenderPipeline::layout() const
{
    return m_pipelineLayout;
}

} // namespace gfx::backend::vulkan::core