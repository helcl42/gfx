#ifndef GFX_TEST_COMMON_H
#define GFX_TEST_COMMON_H

#include <gfx/gfx.h>

#include <gtest/gtest.h>

#include <vector>

inline std::vector<GfxBackend> getActiveBackends()
{
    return {
#if defined(GFX_ENABLE_VULKAN)
        GFX_BACKEND_VULKAN,
#endif
#if defined(GFX_ENABLE_WEBGPU)
        GFX_BACKEND_WEBGPU,
#endif
    };
}

inline const char* convertTestParamToString(const testing::TestParamInfo<GfxBackend>& info)
{
    switch (info.param) {
    case GFX_BACKEND_VULKAN:
        return "Vulkan";
    case GFX_BACKEND_WEBGPU:
        return "WebGPU";
    default:
        return "Unknown";
    }
}

// Vulkan guarantees one of the two combined formats; Apple GPUs only have DEPTH32_FLOAT_STENCIL8
inline GfxFormat supportedDepthStencilFormat(GfxDevice device)
{
    bool supported = false;
    gfxDeviceSupportsFormat(device, GFX_FORMAT_DEPTH24_PLUS_STENCIL8, &supported);
    return supported ? GFX_FORMAT_DEPTH24_PLUS_STENCIL8 : GFX_FORMAT_DEPTH32_FLOAT_STENCIL8;
}

#endif // !GFX_TEST_COMMON_H
