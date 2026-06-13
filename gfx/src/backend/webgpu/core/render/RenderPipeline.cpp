#include "RenderPipeline.h"

#include "../system/Device.h"

#include <optional>
#include <stdexcept>

namespace gfx::backend::webgpu::core {

namespace {
    // Vertex state together with the attribute/buffer arrays it points into
    struct VertexStateData {
        std::vector<std::vector<WGPUVertexAttribute>> allAttributes;
        std::vector<WGPUVertexBufferLayout> buffers;
        WGPUVertexState state = WGPU_VERTEX_STATE_INIT;
    };

    VertexStateData buildVertexState(const VertexState& vertex)
    {
        VertexStateData data;
        data.state.module = vertex.module;
        data.state.entryPoint = { vertex.entryPoint, WGPU_STRLEN };

        if (vertex.buffers.empty()) {
            return data;
        }

        data.buffers.reserve(vertex.buffers.size());
        data.allAttributes.reserve(vertex.buffers.size());

        for (const auto& buffer : vertex.buffers) {
            std::vector<WGPUVertexAttribute> attributes;
            attributes.reserve(buffer.attributes.size());

            for (const auto& attr : buffer.attributes) {
                WGPUVertexAttribute wgpuAttr = WGPU_VERTEX_ATTRIBUTE_INIT;
                wgpuAttr.format = attr.format;
                wgpuAttr.offset = attr.offset;
                wgpuAttr.shaderLocation = attr.shaderLocation;
                attributes.push_back(wgpuAttr);
            }

            data.allAttributes.push_back(std::move(attributes));

            WGPUVertexBufferLayout wgpuBuffer = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
            wgpuBuffer.arrayStride = buffer.arrayStride;
            wgpuBuffer.stepMode = buffer.stepMode;
            wgpuBuffer.attributes = data.allAttributes.back().data();
            wgpuBuffer.attributeCount = static_cast<uint32_t>(data.allAttributes.back().size());
            data.buffers.push_back(wgpuBuffer);
        }

        data.state.buffers = data.buffers.data();
        data.state.bufferCount = static_cast<uint32_t>(data.buffers.size());
        return data;
    }

    // Fragment state together with the target/blend arrays it points into
    struct FragmentStateData {
        std::vector<WGPUBlendState> blendStates;
        std::vector<WGPUColorTargetState> colorTargets;
        WGPUFragmentState state = WGPU_FRAGMENT_STATE_INIT;
    };

    // The fragment stage is optional - nullopt in, nullopt out
    std::optional<FragmentStateData> buildFragmentState(const std::optional<FragmentState>& fragment)
    {
        if (!fragment.has_value()) {
            return std::nullopt;
        }

        FragmentStateData data;
        data.state.module = fragment->module;
        data.state.entryPoint = { fragment->entryPoint, WGPU_STRLEN };

        if (fragment->targets.empty()) {
            return data;
        }

        data.colorTargets.reserve(fragment->targets.size());
        // Reserve so the WGPUColorTargetState::blend pointers stay valid as we append
        data.blendStates.reserve(fragment->targets.size());

        for (const auto& target : fragment->targets) {
            WGPUColorTargetState wgpuTarget = WGPU_COLOR_TARGET_STATE_INIT;
            wgpuTarget.format = target.format;
            wgpuTarget.writeMask = target.writeMask;

            if (target.blend.has_value()) {
                WGPUBlendState blend = WGPU_BLEND_STATE_INIT;
                blend.color.operation = target.blend->color.operation;
                blend.color.srcFactor = target.blend->color.srcFactor;
                blend.color.dstFactor = target.blend->color.dstFactor;
                blend.alpha.operation = target.blend->alpha.operation;
                blend.alpha.srcFactor = target.blend->alpha.srcFactor;
                blend.alpha.dstFactor = target.blend->alpha.dstFactor;
                data.blendStates.push_back(blend);
                wgpuTarget.blend = &data.blendStates.back();
            }

            data.colorTargets.push_back(wgpuTarget);
        }

        data.state.targets = data.colorTargets.data();
        data.state.targetCount = static_cast<uint32_t>(data.colorTargets.size());
        return data;
    }

    WGPUDepthStencilState makeDepthStencilState(const DepthStencilState& depthStencil)
    {
        WGPUDepthStencilState state = WGPU_DEPTH_STENCIL_STATE_INIT;
        state.format = depthStencil.format;
        state.depthWriteEnabled = depthStencil.depthWriteEnabled ? WGPUOptionalBool_True : WGPUOptionalBool_False;
        state.depthCompare = depthStencil.depthCompare;

        state.stencilFront.compare = depthStencil.stencilFront.compare;
        state.stencilFront.failOp = depthStencil.stencilFront.failOp;
        state.stencilFront.depthFailOp = depthStencil.stencilFront.depthFailOp;
        state.stencilFront.passOp = depthStencil.stencilFront.passOp;

        state.stencilBack.compare = depthStencil.stencilBack.compare;
        state.stencilBack.failOp = depthStencil.stencilBack.failOp;
        state.stencilBack.depthFailOp = depthStencil.stencilBack.depthFailOp;
        state.stencilBack.passOp = depthStencil.stencilBack.passOp;

        state.stencilReadMask = depthStencil.stencilReadMask;
        state.stencilWriteMask = depthStencil.stencilWriteMask;
        state.depthBias = depthStencil.depthBias;
        state.depthBiasSlopeScale = depthStencil.depthBiasSlopeScale;
        state.depthBiasClamp = depthStencil.depthBiasClamp;
        return state;
    }

    WGPUPrimitiveState makePrimitiveState(const PrimitiveState& primitive)
    {
        WGPUPrimitiveState state = WGPU_PRIMITIVE_STATE_INIT;
        state.topology = primitive.topology;
        state.frontFace = primitive.frontFace;
        state.cullMode = primitive.cullMode;
        state.stripIndexFormat = primitive.stripIndexFormat;
        return state;
    }

    WGPUMultisampleState makeMultisampleState(uint32_t sampleCount)
    {
        WGPUMultisampleState state = WGPU_MULTISAMPLE_STATE_INIT;
        // WebGPU requires sampleCount >= 1, clamp to valid range
        state.count = std::max<uint32_t>(1, sampleCount);
        return state;
    }

    // Returns nullptr when no bind group layouts are provided (WebGPU then derives the layout)
    WGPUPipelineLayout createPipelineLayout(WGPUDevice device, const std::vector<WGPUBindGroupLayout>& bindGroupLayouts)
    {
        if (bindGroupLayouts.empty()) {
            return nullptr;
        }
        WGPUPipelineLayoutDescriptor layoutDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        layoutDesc.bindGroupLayouts = bindGroupLayouts.data();
        layoutDesc.bindGroupLayoutCount = static_cast<uint32_t>(bindGroupLayouts.size());
        return wgpuDeviceCreatePipelineLayout(device, &layoutDesc);
    }
} // anonymous namespace

RenderPipeline::RenderPipeline(Device* device, const RenderPipelineCreateInfo& createInfo)
{
    WGPUPipelineLayout pipelineLayout = createPipelineLayout(device->handle(), createInfo.bindGroupLayouts);
    
    VertexStateData vertex = buildVertexState(createInfo.vertex);
    std::optional<FragmentStateData> fragment = buildFragmentState(createInfo.fragment);
    std::optional<WGPUDepthStencilState> depthStencil;
    if (createInfo.depthStencil.has_value()) {
        depthStencil = makeDepthStencilState(createInfo.depthStencil.value());
    }

    // ...then assemble the descriptor in one place
    WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    desc.layout = pipelineLayout; // nullptr = WebGPU derives the layout
    desc.vertex = vertex.state;
    desc.fragment = fragment.has_value() ? &fragment->state : nullptr;
    desc.primitive = makePrimitiveState(createInfo.primitive);
    desc.depthStencil = depthStencil.has_value() ? &depthStencil.value() : nullptr;
    desc.multisample = makeMultisampleState(createInfo.sampleCount);

    m_pipeline = wgpuDeviceCreateRenderPipeline(device->handle(), &desc);

    // Release the pipeline layout if we created one (pipeline holds its own reference)
    if (pipelineLayout) {
        wgpuPipelineLayoutRelease(pipelineLayout);
    }

    if (!m_pipeline) {
        throw std::runtime_error("Failed to create WebGPU RenderPipeline");
    }
}

RenderPipeline::~RenderPipeline()
{
    if (m_pipeline) {
        wgpuRenderPipelineRelease(m_pipeline);
    }
}

WGPURenderPipeline RenderPipeline::handle() const
{
    return m_pipeline;
}

} // namespace gfx::backend::webgpu::core