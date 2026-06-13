#ifndef GFX_VULKAN_CORE_UTILS_H
#define GFX_VULKAN_CORE_UTILS_H

#include "../../common/Common.h"

namespace gfx::backend::vulkan::core {

// ============================================================================
// Vulkan Format and Image Utilities
// ============================================================================

// Get the appropriate image aspect mask for a given format
VkImageAspectFlags getImageAspectMask(VkFormat format);

// Get the size in bytes of one texel in buffer memory for a given aspect of a format
// (stencil data is 1 byte/texel, depth data is 2 or 4 bytes/texel depending on format)
uint32_t getAspectTexelSize(VkFormat format, VkImageAspectFlags aspectMask);

// Get the appropriate access flags for a given image layout
VkAccessFlags getVkAccessFlagsForLayout(VkImageLayout layout);

// Check if format has depth component
bool isDepthFormat(VkFormat format);

// Check if format has stencil component
bool hasStencilComponent(VkFormat format);

// Find suitable memory type index for given requirements and properties
// Returns UINT32_MAX if no suitable memory type is found
uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties& memProperties, uint32_t memoryTypeBits, VkMemoryPropertyFlags requiredProperties);

// ============================================================================
// Vulkan Error Handling
// ============================================================================

// Convert VkResult to human-readable string
const char* vkResultToString(VkResult result);

// Get bytes per pixel for a given Vulkan format (uncompressed formats only)
uint32_t getVkFormatBytesPerPixel(VkFormat format);

// Block-compressed format utilities
bool isCompressedVkFormat(VkFormat format);
// Bytes per block (equals bytes-per-pixel for uncompressed formats)
uint32_t getVkFormatBlockSize(VkFormat format);
// Block dimensions in texels (1x1 for uncompressed formats)
void getVkFormatBlockDimensions(VkFormat format, uint32_t* outWidth, uint32_t* outHeight);

} // namespace gfx::backend::vulkan::core

#endif // GFX_VULKAN_CORE_UTILS_H
