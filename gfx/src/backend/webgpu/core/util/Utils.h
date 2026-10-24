#ifndef GFX_WEBGPU_UTILS_H
#define GFX_WEBGPU_UTILS_H

#include "../../common/Common.h"

namespace gfx::backend::webgpu::core {

WGPUStringView toStringView(const char* str);

bool hasStencil(WGPUTextureFormat format);

// Get bytes per pixel for a texture format
uint32_t getFormatBytesPerPixel(WGPUTextureFormat format);

// Get the size in bytes of one texel in buffer memory for a given aspect of a format
uint32_t getAspectTexelSize(WGPUTextureFormat format, WGPUTextureAspect aspect);

// Block-compressed format utilities
bool isCompressedWGPUFormat(WGPUTextureFormat format);
// Block dimensions in texels (1x1 for uncompressed formats)
void getWGPUFormatBlockDimensions(WGPUTextureFormat format, uint32_t* outWidth, uint32_t* outHeight);

// Tightly packed bytes per row of an aspect (block-aware: blocks-per-row * bytes-per-block)
uint32_t getPackedBytesPerRow(WGPUTextureFormat format, WGPUTextureAspect aspect, uint32_t width);

} // namespace gfx::backend::webgpu::core

#endif // GFX_WEBGPU_UTILS_H