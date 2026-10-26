#ifndef GFX_CPP_TEST_COMMON_H
#define GFX_CPP_TEST_COMMON_H

#include <gfx_cpp/gfx.hpp>

#include <gtest/gtest.h>

#include <vector>

inline std::vector<gfx::Backend> getActiveBackends()
{
    return {
#if defined(GFX_ENABLE_VULKAN)
        gfx::Backend::Vulkan,
#endif
#if defined(GFX_ENABLE_WEBGPU)
        gfx::Backend::WebGPU,
#endif
    };
}

inline const char* convertTestParamToString(const testing::TestParamInfo<gfx::Backend>& info)
{
    switch (info.param) {
    case gfx::Backend::Vulkan:
        return "Vulkan";
    case gfx::Backend::WebGPU:
        return "WebGPU";
    default:
        return "Unknown";
    }
}

// Vulkan guarantees one of the two combined formats; Apple GPUs only have Depth32FloatStencil8
inline gfx::Format supportedDepthStencilFormat(const gfx::Device& device)
{
    return device.supportsFormat(gfx::Format::Depth24PlusStencil8) ? gfx::Format::Depth24PlusStencil8 : gfx::Format::Depth32FloatStencil8;
}

#endif // !GFX_CPP_COMMON_TEST_H
