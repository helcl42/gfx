#ifndef GFX_WEBGPU_BLIT_H
#define GFX_WEBGPU_BLIT_H

#include "../../common/Common.h"

#include <unordered_map>

namespace gfx::backend::webgpu::core {

class Blit {
public:
    // Prevent copying
    Blit(const Blit&) = delete;
    Blit& operator=(const Blit&) = delete;

    Blit(WGPUDevice device);
    ~Blit();

    void execute(WGPUCommandEncoder commandEncoder, WGPUTexture srcTexture, const WGPUOrigin3D& srcOrigin, const WGPUExtent3D& srcExtent, uint32_t srcMipLevel, WGPUTexture dstTexture, const WGPUOrigin3D& dstOrigin, const WGPUExtent3D& dstExtent, uint32_t dstMipLevel, WGPUFilterMode filterMode);

private:
    // 2D and 3D sources are sampled through their own binding by their own fragment entry point
    struct DimensionPipelines {
        uint32_t textureBinding;
        const char* fragmentEntryPoint;
        WGPUBindGroupLayout bindGroupLayout = nullptr;
        WGPUPipelineLayout pipelineLayout = nullptr;
        std::unordered_map<WGPUTextureFormat, WGPURenderPipeline> pipelines = {};
    };

    void initDimension(DimensionPipelines& blit, WGPUTextureViewDimension viewDimension);
    void destroyDimension(DimensionPipelines& blit);
    WGPURenderPipeline getOrCreatePipeline(DimensionPipelines& blit, WGPUTextureFormat format);
    WGPUSampler getOrCreateSampler(WGPUFilterMode filterMode);

    WGPUDevice m_device = nullptr;
    WGPUShaderModule m_shaderModule = nullptr;
    DimensionPipelines m_blit2D{ 0, "fs_main" };
    DimensionPipelines m_blit3D{ 3, "fs_main3D" };
    std::unordered_map<WGPUFilterMode, WGPUSampler> m_samplers;
};

} // namespace gfx::backend::webgpu::core

#endif // GFX_WEBGPU_BLIT_H
