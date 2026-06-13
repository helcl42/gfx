#include "Blit.h"

#include "Utils.h"
#include "common/Logger.h"

namespace gfx::backend::webgpu::core {

Blit::Blit(WGPUDevice device)
    : m_device(device)
{
    initBlit2D();
    initBlit3D();
}

void Blit::initBlit2D()
{
    // Create shader module for 2D textures with source region support
    const char* shader2DCode = R"(
            struct SourceRegion {
                uvMin: vec2f,
                uvMax: vec2f,
            }
            
            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) texCoord: vec2f,
            }
            
            @group(0) @binding(0) var srcTexture: texture_2d<f32>;
            @group(0) @binding(1) var srcSampler: sampler;
            @group(0) @binding(2) var<uniform> sourceRegion: SourceRegion;
            
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
        )";

    WGPUShaderSourceWGSL wgslSource = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSource.code = toStringView(shader2DCode);

    WGPUShaderModuleDescriptor shaderDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    shaderDesc.nextInChain = &wgslSource.chain;
    m_shaderModule = wgpuDeviceCreateShaderModule(m_device, &shaderDesc);

    // Create bind group layout with uniform buffer for source region
    WGPUBindGroupLayoutEntry bgLayoutEntries[3] = { WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT };
    bgLayoutEntries[0].binding = 0;
    bgLayoutEntries[0].visibility = WGPUShaderStage_Fragment;
    bgLayoutEntries[0].texture.sampleType = WGPUTextureSampleType_Float;
    bgLayoutEntries[0].texture.viewDimension = WGPUTextureViewDimension_2D;

    bgLayoutEntries[1].binding = 1;
    bgLayoutEntries[1].visibility = WGPUShaderStage_Fragment;
    bgLayoutEntries[1].sampler.type = WGPUSamplerBindingType_Filtering;

    bgLayoutEntries[2].binding = 2;
    bgLayoutEntries[2].visibility = WGPUShaderStage_Vertex;
    bgLayoutEntries[2].buffer.type = WGPUBufferBindingType_Uniform;
    bgLayoutEntries[2].buffer.minBindingSize = 16; // vec2f + vec2f = 16 bytes

    WGPUBindGroupLayoutDescriptor bgLayoutDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgLayoutDesc.entryCount = 3;
    bgLayoutDesc.entries = bgLayoutEntries;
    m_bindGroupLayout = wgpuDeviceCreateBindGroupLayout(m_device, &bgLayoutDesc);

    // Create pipeline layout
    WGPUPipelineLayoutDescriptor pipelineLayoutDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &m_bindGroupLayout;
    m_pipelineLayout = wgpuDeviceCreatePipelineLayout(m_device, &pipelineLayoutDesc);
}

void Blit::initBlit3D()
{
    // Create compute shader for 3D texture mipmap downsampling
    const char* shader3DCode = R"(
            @group(0) @binding(0) var srcTexture: texture_3d<f32>;
            @group(0) @binding(1) var dstTexture: texture_storage_3d<rgba8unorm, write>;

            @compute @workgroup_size(4, 4, 4)
            fn main(@builtin(global_invocation_id) id: vec3u) {
                let dstSize = textureDimensions(dstTexture);
                if (id.x >= dstSize.x || id.y >= dstSize.y || id.z >= dstSize.z) {
                    return;
                }

                // Sample 8 texels from the source mip level (2x2x2 box filter)
                let srcCoord = id * 2u;
                let s000 = textureLoad(srcTexture, vec3i(srcCoord + vec3u(0u, 0u, 0u)), 0);
                let s100 = textureLoad(srcTexture, vec3i(srcCoord + vec3u(1u, 0u, 0u)), 0);
                let s010 = textureLoad(srcTexture, vec3i(srcCoord + vec3u(0u, 1u, 0u)), 0);
                let s110 = textureLoad(srcTexture, vec3i(srcCoord + vec3u(1u, 1u, 0u)), 0);
                let s001 = textureLoad(srcTexture, vec3i(srcCoord + vec3u(0u, 0u, 1u)), 0);
                let s101 = textureLoad(srcTexture, vec3i(srcCoord + vec3u(1u, 0u, 1u)), 0);
                let s011 = textureLoad(srcTexture, vec3i(srcCoord + vec3u(0u, 1u, 1u)), 0);
                let s111 = textureLoad(srcTexture, vec3i(srcCoord + vec3u(1u, 1u, 1u)), 0);

                let result = (s000 + s100 + s010 + s110 + s001 + s101 + s011 + s111) * 0.125;
                textureStore(dstTexture, vec3i(id), result);
            }
        )";

    WGPUShaderSourceWGSL wgslSource = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSource.code = toStringView(shader3DCode);

    WGPUShaderModuleDescriptor shaderDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    shaderDesc.nextInChain = &wgslSource.chain;
    m_shaderModule3D = wgpuDeviceCreateShaderModule(m_device, &shaderDesc);

    // Bind group layout: binding 0 = texture_3d, binding 1 = storage texture 3D write
    WGPUBindGroupLayoutEntry bgLayoutEntries3D[2] = { WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT };
    bgLayoutEntries3D[0].binding = 0;
    bgLayoutEntries3D[0].visibility = WGPUShaderStage_Compute;
    bgLayoutEntries3D[0].texture.sampleType = WGPUTextureSampleType_Float;
    bgLayoutEntries3D[0].texture.viewDimension = WGPUTextureViewDimension_3D;

    bgLayoutEntries3D[1].binding = 1;
    bgLayoutEntries3D[1].visibility = WGPUShaderStage_Compute;
    bgLayoutEntries3D[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    bgLayoutEntries3D[1].storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
    bgLayoutEntries3D[1].storageTexture.viewDimension = WGPUTextureViewDimension_3D;

    WGPUBindGroupLayoutDescriptor bgLayoutDesc3D = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgLayoutDesc3D.entryCount = 2;
    bgLayoutDesc3D.entries = bgLayoutEntries3D;
    m_bindGroupLayout3D = wgpuDeviceCreateBindGroupLayout(m_device, &bgLayoutDesc3D);

    WGPUPipelineLayoutDescriptor pipelineLayoutDesc3D = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipelineLayoutDesc3D.bindGroupLayoutCount = 1;
    pipelineLayoutDesc3D.bindGroupLayouts = &m_bindGroupLayout3D;
    m_pipelineLayout3D = wgpuDeviceCreatePipelineLayout(m_device, &pipelineLayoutDesc3D);

    WGPUComputePipelineDescriptor computeDesc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    computeDesc.layout = m_pipelineLayout3D;
    computeDesc.compute.module = m_shaderModule3D;
    computeDesc.compute.entryPoint = toStringView("main");
    m_pipeline3D = wgpuDeviceCreateComputePipeline(m_device, &computeDesc);
}

Blit::~Blit()
{
    destroyBlit2D();
    destroyBlit3D();
}

void Blit::destroyBlit2D()
{
    for (auto& pair : m_pipelines) {
        wgpuRenderPipelineRelease(pair.second);
    }
    m_pipelines.clear();
    for (auto& pair : m_samplers) {
        wgpuSamplerRelease(pair.second);
    }
    m_samplers.clear();
    if (m_pipelineLayout) {
        wgpuPipelineLayoutRelease(m_pipelineLayout);
        m_pipelineLayout = nullptr;
    }
    if (m_bindGroupLayout) {
        wgpuBindGroupLayoutRelease(m_bindGroupLayout);
        m_bindGroupLayout = nullptr;
    }
    if (m_shaderModule) {
        wgpuShaderModuleRelease(m_shaderModule);
        m_shaderModule = nullptr;
    }
}

void Blit::destroyBlit3D()
{
    if (m_pipeline3D) {
        wgpuComputePipelineRelease(m_pipeline3D);
        m_pipeline3D = nullptr;
    }
    if (m_pipelineLayout3D) {
        wgpuPipelineLayoutRelease(m_pipelineLayout3D);
        m_pipelineLayout3D = nullptr;
    }
    if (m_bindGroupLayout3D) {
        wgpuBindGroupLayoutRelease(m_bindGroupLayout3D);
        m_bindGroupLayout3D = nullptr;
    }
    if (m_shaderModule3D) {
        wgpuShaderModuleRelease(m_shaderModule3D);
        m_shaderModule3D = nullptr;
    }
}

void Blit::execute(WGPUCommandEncoder commandEncoder, WGPUTexture srcTexture, const WGPUOrigin3D& srcOrigin, const WGPUExtent3D& srcExtent, uint32_t srcMipLevel, WGPUTexture dstTexture, const WGPUOrigin3D& dstOrigin, const WGPUExtent3D& dstExtent, uint32_t dstMipLevel, WGPUFilterMode filterMode)
{
    WGPUTextureDimension srcDimension = wgpuTextureGetDimension(srcTexture);

    if (srcDimension == WGPUTextureDimension_3D) {
        execute3D(commandEncoder, srcTexture, srcMipLevel, dstTexture, dstMipLevel);
    } else if (srcDimension == WGPUTextureDimension_2D) {
        execute2D(commandEncoder, srcTexture, srcOrigin, srcExtent, srcMipLevel, dstTexture, dstOrigin, dstExtent, dstMipLevel, filterMode);
    } else {
        gfx::common::Logger::instance().logWarning("[WebGPU Blit] 1D textures are not supported for blitting");
    }
}

void Blit::execute3D(WGPUCommandEncoder commandEncoder, WGPUTexture srcTexture, uint32_t srcMipLevel, WGPUTexture dstTexture, uint32_t dstMipLevel)
{
    // Create 3D texture views for src and dst mip levels
    WGPUTextureViewDescriptor srcViewDesc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    srcViewDesc.format = wgpuTextureGetFormat(srcTexture);
    srcViewDesc.dimension = WGPUTextureViewDimension_3D;
    srcViewDesc.baseMipLevel = srcMipLevel;
    srcViewDesc.mipLevelCount = 1;
    srcViewDesc.baseArrayLayer = 0;
    srcViewDesc.arrayLayerCount = 1;
    WGPUTextureView srcView = wgpuTextureCreateView(srcTexture, &srcViewDesc);

    WGPUTextureViewDescriptor dstViewDesc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    dstViewDesc.format = wgpuTextureGetFormat(dstTexture);
    dstViewDesc.dimension = WGPUTextureViewDimension_3D;
    dstViewDesc.baseMipLevel = dstMipLevel;
    dstViewDesc.mipLevelCount = 1;
    dstViewDesc.baseArrayLayer = 0;
    dstViewDesc.arrayLayerCount = 1;
    WGPUTextureView dstView = wgpuTextureCreateView(dstTexture, &dstViewDesc);

    // Create bind group
    WGPUBindGroupEntry bgEntries[2] = { WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT };
    bgEntries[0].binding = 0;
    bgEntries[0].textureView = srcView;
    bgEntries[1].binding = 1;
    bgEntries[1].textureView = dstView;

    WGPUBindGroupDescriptor bgDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bgDesc.layout = m_bindGroupLayout3D;
    bgDesc.entryCount = 2;
    bgDesc.entries = bgEntries;
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(m_device, &bgDesc);

    // Calculate destination mip dimensions
    uint32_t dstWidth = std::max(1u, wgpuTextureGetWidth(dstTexture) >> dstMipLevel);
    uint32_t dstHeight = std::max(1u, wgpuTextureGetHeight(dstTexture) >> dstMipLevel);
    uint32_t dstDepth = std::max(1u, wgpuTextureGetDepthOrArrayLayers(dstTexture) >> dstMipLevel);

    // Dispatch compute shader
    WGPUComputePassDescriptor passDesc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder computePass = wgpuCommandEncoderBeginComputePass(commandEncoder, &passDesc);
    wgpuComputePassEncoderSetPipeline(computePass, m_pipeline3D);
    wgpuComputePassEncoderSetBindGroup(computePass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(computePass, (dstWidth + 3) / 4, (dstHeight + 3) / 4, (dstDepth + 3) / 4);
    wgpuComputePassEncoderEnd(computePass);

    // Cleanup
    wgpuComputePassEncoderRelease(computePass);
    wgpuBindGroupRelease(bindGroup);
    wgpuTextureViewRelease(dstView);
    wgpuTextureViewRelease(srcView);
}

void Blit::execute2D(WGPUCommandEncoder commandEncoder, WGPUTexture srcTexture, const WGPUOrigin3D& srcOrigin, const WGPUExtent3D& srcExtent, uint32_t srcMipLevel, WGPUTexture dstTexture, const WGPUOrigin3D& dstOrigin, const WGPUExtent3D& dstExtent, uint32_t dstMipLevel, WGPUFilterMode filterMode)
{
    // Get or create sampler with requested filter mode
    WGPUSampler sampler = getOrCreateSampler(filterMode);

    // Create texture view for source
    WGPUTextureViewDescriptor srcViewDesc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    srcViewDesc.format = wgpuTextureGetFormat(srcTexture);
    srcViewDesc.dimension = WGPUTextureViewDimension_2D;
    srcViewDesc.baseMipLevel = srcMipLevel;
    srcViewDesc.mipLevelCount = 1;
    srcViewDesc.baseArrayLayer = srcOrigin.z;
    srcViewDesc.arrayLayerCount = 1;
    WGPUTextureView srcView = wgpuTextureCreateView(srcTexture, &srcViewDesc);

    // Calculate source texture size at mip level
    uint32_t srcTexWidth = std::max(1u, wgpuTextureGetWidth(srcTexture) >> srcMipLevel);
    uint32_t srcTexHeight = std::max(1u, wgpuTextureGetHeight(srcTexture) >> srcMipLevel);

    // Calculate UV coordinates for source region
    struct SourceRegionData {
        float uvMinX, uvMinY;
        float uvMaxX, uvMaxY;
    } regionData;
    regionData.uvMinX = static_cast<float>(srcOrigin.x) / static_cast<float>(srcTexWidth);
    regionData.uvMinY = static_cast<float>(srcOrigin.y) / static_cast<float>(srcTexHeight);
    regionData.uvMaxX = static_cast<float>(srcOrigin.x + srcExtent.width) / static_cast<float>(srcTexWidth);
    regionData.uvMaxY = static_cast<float>(srcOrigin.y + srcExtent.height) / static_cast<float>(srcTexHeight);

    // Create uniform buffer for source region
    WGPUBufferDescriptor uniformBufferDesc = WGPU_BUFFER_DESCRIPTOR_INIT;
    uniformBufferDesc.size = sizeof(SourceRegionData);
    uniformBufferDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    WGPUBuffer uniformBuffer = wgpuDeviceCreateBuffer(m_device, &uniformBufferDesc);
    wgpuQueueWriteBuffer(wgpuDeviceGetQueue(m_device), uniformBuffer, 0, &regionData, sizeof(SourceRegionData));

    // Create bind group
    WGPUBindGroupEntry bgEntries[3] = { WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT };
    bgEntries[0].binding = 0;
    bgEntries[0].textureView = srcView;

    bgEntries[1].binding = 1;
    bgEntries[1].sampler = sampler;

    bgEntries[2].binding = 2;
    bgEntries[2].buffer = uniformBuffer;
    bgEntries[2].size = sizeof(SourceRegionData);

    WGPUBindGroupDescriptor bgDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bgDesc.layout = m_bindGroupLayout;
    bgDesc.entryCount = 3;
    bgDesc.entries = bgEntries;
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(m_device, &bgDesc);

    // Get or create render pipeline for this format
    WGPUTextureFormat dstFormat = wgpuTextureGetFormat(dstTexture);
    WGPURenderPipeline pipeline = getOrCreatePipeline(dstFormat);

    // Create texture view for destination
    WGPUTextureViewDescriptor dstViewDesc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    dstViewDesc.format = dstFormat;
    dstViewDesc.dimension = WGPUTextureViewDimension_2D;
    dstViewDesc.baseMipLevel = dstMipLevel;
    dstViewDesc.mipLevelCount = 1;
    dstViewDesc.baseArrayLayer = dstOrigin.z;
    dstViewDesc.arrayLayerCount = 1;
    WGPUTextureView dstView = wgpuTextureCreateView(dstTexture, &dstViewDesc);

    // Create render pass
    WGPURenderPassColorAttachment colorAttachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    colorAttachment.view = dstView;
    colorAttachment.loadOp = WGPULoadOp_Load;
    colorAttachment.storeOp = WGPUStoreOp_Store;

    WGPURenderPassDescriptor renderPassDesc = {};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;

    WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(commandEncoder, &renderPassDesc);
    wgpuRenderPassEncoderSetPipeline(renderPass, pipeline);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 0, bindGroup, 0, nullptr);

    // Set viewport to destination region for proper scaling
    wgpuRenderPassEncoderSetViewport(renderPass, static_cast<float>(dstOrigin.x), static_cast<float>(dstOrigin.y), static_cast<float>(dstExtent.width), static_cast<float>(dstExtent.height), 0.0f, 1.0f);

    // Set scissor to destination region
    wgpuRenderPassEncoderSetScissorRect(renderPass, dstOrigin.x, dstOrigin.y, dstExtent.width, dstExtent.height);

    wgpuRenderPassEncoderDraw(renderPass, 4, 1, 0, 0);
    wgpuRenderPassEncoderEnd(renderPass);

    // Cleanup temporary resources
    wgpuRenderPassEncoderRelease(renderPass);
    wgpuTextureViewRelease(dstView);
    wgpuBindGroupRelease(bindGroup);
    wgpuBufferRelease(uniformBuffer);
    wgpuTextureViewRelease(srcView);
    // Note: sampler is cached, don't release it here
}

WGPURenderPipeline Blit::getOrCreatePipeline(WGPUTextureFormat format)
{
    // Check cache
    auto it = m_pipelines.find(format);
    if (it != m_pipelines.end()) {
        return it->second;
    }

    // Create new pipeline for this format
    WGPUColorTargetState colorTarget = WGPU_COLOR_TARGET_STATE_INIT;
    colorTarget.format = format;
    colorTarget.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fragmentState = WGPU_FRAGMENT_STATE_INIT;
    fragmentState.module = m_shaderModule;
    fragmentState.entryPoint = toStringView("fs_main");
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;

    WGPURenderPipelineDescriptor pipelineDesc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    pipelineDesc.layout = m_pipelineLayout;
    pipelineDesc.vertex.module = m_shaderModule;
    pipelineDesc.vertex.entryPoint = toStringView("vs_main");
    pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleStrip;
    pipelineDesc.fragment = &fragmentState;
    pipelineDesc.multisample.count = 1;

    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(m_device, &pipelineDesc);
    m_pipelines[format] = pipeline;
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