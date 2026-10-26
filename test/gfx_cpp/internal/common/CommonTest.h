#ifndef GFX_CPP_TEST_INTERNAL_H
#define GFX_CPP_TEST_INTERNAL_H

#include <gfx/gfx.h>

#include <gfx_cpp/gfx.hpp>

#include <gtest/gtest.h>

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

// Vulkan guarantees one of the two combined formats; Apple GPUs only have Depth32FloatStencil8
inline gfx::Format supportedDepthStencilFormat(GfxDevice device)
{
    bool supported = false;
    gfxDeviceSupportsFormat(device, GFX_FORMAT_DEPTH24_PLUS_STENCIL8, &supported);
    return supported ? gfx::Format::Depth24PlusStencil8 : gfx::Format::Depth32FloatStencil8;
}

#endif // !GFX_CPP_TEST_INTERNAL_H
