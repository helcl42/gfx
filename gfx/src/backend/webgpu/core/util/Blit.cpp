#include "Blit.h"

#include "Utils.h"
#include "common/Logger.h"

namespace gfx::backend::webgpu::core {

namespace {
    // Matches SourceRegion in the shader (padded to its 8-byte alignment)
    struct SourceRegion {
        float uvMinX, uvMinY;
        float uvMaxX, uvMaxY;
        float depth;
        float padding;
    };
} // anonymous namespace

Blit::Blit(WGPUDevice device)
    : m_device(device)
{
    // Draws the destination region, sampling the matching source region
    const char* shaderCode = R"(
            struct SourceRegion {
                uvMin: vec2f,
                uvMax: vec2f,
                depth: f32, // Normalized depth of the sampled slice of a 3D source
            }

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) texCoord: vec2f,
            }

            @group(0) @binding(0) var srcTexture: texture_2d<f32>;
            @group(0) @binding(1) var srcSampler: sampler;
            @group(0) @binding(2) var<uniform> sourceRegion: SourceRegion;
            @group(0) @binding(3) var srcTexture3D: texture_3d<f32>;

            @vertex
            fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                var output: VertexOutput;
                let x = f32((vertexIndex & 1u) << 1u) - 1.0;
                let y = 1.0 - f32((vertexIndex & 2u));
                output.position = vec4f(x, y, 0.0, 1.0);
                // Map vertex coordinates [0,1] to source region
                let uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
                output.texCoord = mix(sourceRegion.uvMin, sourceRegion.uvMax, uv);
                return output;
            }

            @fragment
            fn fs_main(input: VertexOutput) -> @location(0) vec4f {
                return textureSample(srcTexture, srcSampler, input.texCoord);
            }

            @fragment
            fn fs_main3D(input: VertexOutput) -> @location(0) vec4f {
                return textureSample(srcTexture3D, srcSampler, vec3f(input.texCoord, sourceRegion.depth));
            }
        )";

    WGPUShaderSourceWGSL wgslSource = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSource.code = toStringView(shaderCode);

    WGPUShaderModuleDescriptor shaderDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    shaderDesc.nextInChain = &wgslSource.chain;
    m_shaderModule = wgpuDeviceCreateShaderModule(m_device, &shaderDesc);

    initDimension(m_blit2D, WGPUTextureViewDimension_2D);
    initDimension(m_blit3D, WGPUTextureViewDimension_3D);
}

Blit::~Blit()
{
    destroyDimension(m_blit2D);
    destroyDimension(m_blit3D);
    for (auto& pair : m_samplers) {
        wgpuSamplerRelease(pair.second);
    }
    if (m_shaderModule) {
        wgpuShaderModuleRelease(m_shaderModule);
    }
}

void Blit::initDimension(DimensionPipelines& blit, WGPUTextureViewDimension viewDimension)
{
    WGPUBindGroupLayoutEntry bgLayoutEntries[3] = { WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT };
    bgLayoutEntries[0].binding = blit.textureBinding;
    bgLayoutEntries[0].visibility = WGPUShaderStage_Fragment;
    bgLayoutEntries[0].texture.sampleType = WGPUTextureSampleType_Float;
    bgLayoutEntries[0].texture.viewDimension = viewDimension;

    bgLayoutEntries[1].binding = 1;
    bgLayoutEntries[1].visibility = WGPUShaderStage_Fragment;
    bgLayoutEntries[1].sampler.type = WGPUSamplerBindingType_Filtering;

    bgLayoutEntries[2].binding = 2;
    bgLayoutEntries[2].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    bgLayoutEntries[2].buffer.type = WGPUBufferBindingType_Uniform;
    bgLayoutEntries[2].buffer.minBindingSize = sizeof(SourceRegion);

    WGPUBindGroupLayoutDescriptor bgLayoutDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgLayoutDesc.entryCount = 3;
    bgLayoutDesc.entries = bgLayoutEntries;
    blit.bindGroupLayout = wgpuDeviceCreateBindGroupLayout(m_device, &bgLayoutDesc);

    WGPUPipelineLayoutDescriptor pipelineLayoutDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &blit.bindGroupLayout;
    blit.pipelineLayout = wgpuDeviceCreatePipelineLayout(m_device, &pipelineLayoutDesc);
}

void Blit::destroyDimension(DimensionPipelines& blit)
{
    for (auto& pair : blit.pipelines) {
        wgpuRenderPipelineRelease(pair.second);
    }
    blit.pipelines.clear();
    if (blit.pipelineLayout) {
        wgpuPipelineLayoutRelease(blit.pipelineLayout);
        blit.pipelineLayout = nullptr;
    }
    if (blit.bindGroupLayout) {
        wgpuBindGroupLayoutRelease(blit.bindGroupLayout);
        blit.bindGroupLayout = nullptr;
    }
}

void Blit::execute(WGPUCommandEncoder commandEncoder, WGPUTexture srcTexture, const WGPUOrigin3D& srcOrigin, const WGPUExtent3D& srcExtent, uint32_t srcMipLevel, WGPUTexture dstTexture, const WGPUOrigin3D& dstOrigin, const WGPUExtent3D& dstExtent, uint32_t dstMipLevel, WGPUFilterMode filterMode)
{
    const WGPUTextureDimension textureDimension = wgpuTextureGetDimension(srcTexture);
    if (textureDimension == WGPUTextureDimension_1D) {
        gfx::common::Logger::instance().logWarning("[WebGPU Blit] 1D textures are not supported for blitting");
        return;
    }
    // A 3D texture is sampled whole and drawn one destination slice at a time; for 2D, origin z selects the array layer
    const bool is3D = textureDimension == WGPUTextureDimension_3D;
    DimensionPipelines& blit = is3D ? m_blit3D : m_blit2D;
    const WGPUTextureViewDimension viewDimension = is3D ? WGPUTextureViewDimension_3D : WGPUTextureViewDimension_2D;

    WGPUTextureViewDescriptor srcViewDesc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    srcViewDesc.format = wgpuTextureGetFormat(srcTexture);
    srcViewDesc.dimension = viewDimension;
    srcViewDesc.baseMipLevel = srcMipLevel;
    srcViewDesc.mipLevelCount = 1;
    srcViewDesc.baseArrayLayer = is3D ? 0 : srcOrigin.z;
    srcViewDesc.arrayLayerCount = 1;
    WGPUTextureView srcView = wgpuTextureCreateView(srcTexture, &srcViewDesc);

    const WGPUTextureFormat dstFormat = wgpuTextureGetFormat(dstTexture);
    WGPUTextureViewDescriptor dstViewDesc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    dstViewDesc.format = dstFormat;
    dstViewDesc.dimension = viewDimension;
    dstViewDesc.baseMipLevel = dstMipLevel;
    dstViewDesc.mipLevelCount = 1;
    dstViewDesc.baseArrayLayer = is3D ? 0 : dstOrigin.z;
    dstViewDesc.arrayLayerCount = 1;
    WGPUTextureView dstView = wgpuTextureCreateView(dstTexture, &dstViewDesc);

    // Source region in normalized coordinates of the source mip level
    const float srcWidth = static_cast<float>(std::max(1u, wgpuTextureGetWidth(srcTexture) >> srcMipLevel));
    const float srcHeight = static_cast<float>(std::max(1u, wgpuTextureGetHeight(srcTexture) >> srcMipLevel));
    const float srcDepth = is3D ? static_cast<float>(std::max(1u, wgpuTextureGetDepthOrArrayLayers(srcTexture) >> srcMipLevel)) : 1.0f;
    SourceRegion region{};
    region.uvMinX = static_cast<float>(srcOrigin.x) / srcWidth;
    region.uvMinY = static_cast<float>(srcOrigin.y) / srcHeight;
    region.uvMaxX = static_cast<float>(srcOrigin.x + srcExtent.width) / srcWidth;
    region.uvMaxY = static_cast<float>(srcOrigin.y + srcExtent.height) / srcHeight;

    WGPURenderPipeline pipeline = getOrCreatePipeline(blit, dstFormat);
    WGPUSampler sampler = getOrCreateSampler(filterMode);
    WGPUQueue queue = wgpuDeviceGetQueue(m_device);

    const uint32_t sliceCount = is3D ? dstExtent.depthOrArrayLayers : 1;
    for (uint32_t slice = 0; slice < sliceCount; ++slice) {
        if (is3D) {
            // Sample the source depth at the center of this destination slice
            const float sliceCenter = (static_cast<float>(slice) + 0.5f) * static_cast<float>(srcExtent.depthOrArrayLayers) / static_cast<float>(dstExtent.depthOrArrayLayers);
            region.depth = (static_cast<float>(srcOrigin.z) + sliceCenter) / srcDepth;
        }

        // Each slice needs its own buffer: queue writes all land before the commands run
        WGPUBufferDescriptor uniformBufferDesc = WGPU_BUFFER_DESCRIPTOR_INIT;
        uniformBufferDesc.size = sizeof(SourceRegion);
        uniformBufferDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        WGPUBuffer uniformBuffer = wgpuDeviceCreateBuffer(m_device, &uniformBufferDesc);
        wgpuQueueWriteBuffer(queue, uniformBuffer, 0, &region, sizeof(SourceRegion));

        WGPUBindGroupEntry bgEntries[3] = { WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT };
        bgEntries[0].binding = blit.textureBinding;
        bgEntries[0].textureView = srcView;
        bgEntries[1].binding = 1;
        bgEntries[1].sampler = sampler;
        bgEntries[2].binding = 2;
        bgEntries[2].buffer = uniformBuffer;
        bgEntries[2].size = sizeof(SourceRegion);

        WGPUBindGroupDescriptor bgDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bgDesc.layout = blit.bindGroupLayout;
        bgDesc.entryCount = 3;
        bgDesc.entries = bgEntries;
        WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(m_device, &bgDesc);

        WGPURenderPassColorAttachment colorAttachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        colorAttachment.view = dstView;
        if (is3D) {
            colorAttachment.depthSlice = dstOrigin.z + slice;
        }
        colorAttachment.loadOp = WGPULoadOp_Load;
        colorAttachment.storeOp = WGPUStoreOp_Store;

        WGPURenderPassDescriptor renderPassDesc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        renderPassDesc.colorAttachmentCount = 1;
        renderPassDesc.colorAttachments = &colorAttachment;

        WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(commandEncoder, &renderPassDesc);
        wgpuRenderPassEncoderSetPipeline(renderPass, pipeline);
        wgpuRenderPassEncoderSetBindGroup(renderPass, 0, bindGroup, 0, nullptr);

        // Viewport and scissor confine drawing to the destination region
        wgpuRenderPassEncoderSetViewport(renderPass, static_cast<float>(dstOrigin.x), static_cast<float>(dstOrigin.y), static_cast<float>(dstExtent.width), static_cast<float>(dstExtent.height), 0.0f, 1.0f);
        wgpuRenderPassEncoderSetScissorRect(renderPass, dstOrigin.x, dstOrigin.y, dstExtent.width, dstExtent.height);

        wgpuRenderPassEncoderDraw(renderPass, 4, 1, 0, 0);
        wgpuRenderPassEncoderEnd(renderPass);

        wgpuRenderPassEncoderRelease(renderPass);
        wgpuBindGroupRelease(bindGroup);
        wgpuBufferRelease(uniformBuffer);
    }

    wgpuQueueRelease(queue);
    wgpuTextureViewRelease(dstView);
    wgpuTextureViewRelease(srcView);
}

WGPURenderPipeline Blit::getOrCreatePipeline(DimensionPipelines& blit, WGPUTextureFormat format)
{
    // Check cache
    auto it = blit.pipelines.find(format);
    if (it != blit.pipelines.end()) {
        return it->second;
    }

    // Create new pipeline for this format
    WGPUColorTargetState colorTarget = WGPU_COLOR_TARGET_STATE_INIT;
    colorTarget.format = format;
    colorTarget.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragmentState = WGPU_FRAGMENT_STATE_INIT;
    fragmentState.module = m_shaderModule;
    fragmentState.entryPoint = toStringView(blit.fragmentEntryPoint);
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;

    WGPURenderPipelineDescriptor pipelineDesc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    pipelineDesc.layout = blit.pipelineLayout;
    pipelineDesc.vertex.module = m_shaderModule;
    pipelineDesc.vertex.entryPoint = toStringView("vs_main");
    pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleStrip;
    pipelineDesc.fragment = &fragmentState;
    pipelineDesc.multisample.count = 1;

    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(m_device, &pipelineDesc);
    blit.pipelines[format] = pipeline;
    return pipeline;
}

WGPUSampler Blit::getOrCreateSampler(WGPUFilterMode filterMode)
{
    // Check cache
    auto it = m_samplers.find(filterMode);
    if (it != m_samplers.end()) {
        return it->second;
    }

    // Create new sampler for this filter mode
    WGPUSamplerDescriptor samplerDesc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    samplerDesc.addressModeU = WGPUAddressMode_ClampToEdge;
    samplerDesc.addressModeV = WGPUAddressMode_ClampToEdge;
    samplerDesc.addressModeW = WGPUAddressMode_ClampToEdge;
    samplerDesc.magFilter = filterMode;
    samplerDesc.minFilter = filterMode;
    samplerDesc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    samplerDesc.maxAnisotropy = 1;

    WGPUSampler sampler = wgpuDeviceCreateSampler(m_device, &samplerDesc);
    m_samplers[filterMode] = sampler;
    return sampler;
}

} // namespace gfx::backend::webgpu::core
