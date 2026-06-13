#include "util/Utils.h"

#include <cstdio>

#ifdef GFX_HAS_COCOA
#include <objc/message.h>
#include <objc/runtime.h>
#endif

namespace gfx::util {

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

uint64_t alignDown(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    return value & ~(alignment - 1);
}

uint32_t getFormatBytesPerPixel(GfxFormat format)
{
    switch (format) {
    // 1 byte
    case GFX_FORMAT_R8_UNORM:
    case GFX_FORMAT_R8_SINT:
    case GFX_FORMAT_R8_UINT:
    case GFX_FORMAT_STENCIL8:
        return 1;
    // 2 bytes
    case GFX_FORMAT_R8G8_UNORM:
    case GFX_FORMAT_R8G8_SINT:
    case GFX_FORMAT_R8G8_UINT:
    case GFX_FORMAT_R16_FLOAT:
    case GFX_FORMAT_R16_SINT:
    case GFX_FORMAT_R16_UINT:
    case GFX_FORMAT_DEPTH16_UNORM:
        return 2;
    // 4 bytes
    case GFX_FORMAT_R8G8B8A8_UNORM:
    case GFX_FORMAT_R8G8B8A8_UNORM_SRGB:
    case GFX_FORMAT_R8G8B8A8_SINT:
    case GFX_FORMAT_R8G8B8A8_UINT:
    case GFX_FORMAT_B8G8R8A8_UNORM:
    case GFX_FORMAT_B8G8R8A8_UNORM_SRGB:
    case GFX_FORMAT_R16G16_FLOAT:
    case GFX_FORMAT_R16G16_SINT:
    case GFX_FORMAT_R16G16_UINT:
    case GFX_FORMAT_R32_FLOAT:
    case GFX_FORMAT_R32_SINT:
    case GFX_FORMAT_R32_UINT:
    case GFX_FORMAT_DEPTH24_PLUS:
    case GFX_FORMAT_DEPTH32_FLOAT:
    case GFX_FORMAT_DEPTH24_PLUS_STENCIL8:
        return 4;
    // 8 bytes
    case GFX_FORMAT_R16G16B16A16_FLOAT:
    case GFX_FORMAT_R16G16B16A16_SINT:
    case GFX_FORMAT_R16G16B16A16_UINT:
    case GFX_FORMAT_R32G32_FLOAT:
    case GFX_FORMAT_R32G32_SINT:
    case GFX_FORMAT_R32G32_UINT:
    case GFX_FORMAT_DEPTH32_FLOAT_STENCIL8:
        return 8;
    // 12 bytes
    case GFX_FORMAT_R32G32B32_FLOAT:
        return 12;
    // 16 bytes
    case GFX_FORMAT_R32G32B32A32_FLOAT:
    case GFX_FORMAT_R32G32B32A32_SINT:
    case GFX_FORMAT_R32G32B32A32_UINT:
        return 16;
    case GFX_FORMAT_UNDEFINED:
    default:
        // Block-compressed formats have no per-pixel size - use getFormatBlockSize
        return 0;
    }
}

uint32_t getFormatBlockSize(GfxFormat format)
{
    switch (format) {
    // 8-byte blocks
    case GFX_FORMAT_BC1_RGBA_UNORM:
    case GFX_FORMAT_BC1_RGBA_UNORM_SRGB:
    case GFX_FORMAT_BC4_R_UNORM:
    case GFX_FORMAT_BC4_R_SNORM:
    case GFX_FORMAT_ETC2_RGB8_UNORM:
    case GFX_FORMAT_ETC2_RGB8_UNORM_SRGB:
    case GFX_FORMAT_ETC2_RGB8A1_UNORM:
    case GFX_FORMAT_ETC2_RGB8A1_UNORM_SRGB:
    case GFX_FORMAT_EAC_R11_UNORM:
    case GFX_FORMAT_EAC_R11_SNORM:
        return 8;
    // 16-byte blocks
    case GFX_FORMAT_BC2_RGBA_UNORM:
    case GFX_FORMAT_BC2_RGBA_UNORM_SRGB:
    case GFX_FORMAT_BC3_RGBA_UNORM:
    case GFX_FORMAT_BC3_RGBA_UNORM_SRGB:
    case GFX_FORMAT_BC5_RG_UNORM:
    case GFX_FORMAT_BC5_RG_SNORM:
    case GFX_FORMAT_BC6H_RGB_UFLOAT:
    case GFX_FORMAT_BC6H_RGB_SFLOAT:
    case GFX_FORMAT_BC7_RGBA_UNORM:
    case GFX_FORMAT_BC7_RGBA_UNORM_SRGB:
    case GFX_FORMAT_ETC2_RGBA8_UNORM:
    case GFX_FORMAT_ETC2_RGBA8_UNORM_SRGB:
    case GFX_FORMAT_EAC_RG11_UNORM:
    case GFX_FORMAT_EAC_RG11_SNORM:
    case GFX_FORMAT_ASTC_4X4_UNORM:
    case GFX_FORMAT_ASTC_4X4_UNORM_SRGB:
    case GFX_FORMAT_ASTC_5X4_UNORM:
    case GFX_FORMAT_ASTC_5X4_UNORM_SRGB:
    case GFX_FORMAT_ASTC_5X5_UNORM:
    case GFX_FORMAT_ASTC_5X5_UNORM_SRGB:
    case GFX_FORMAT_ASTC_6X5_UNORM:
    case GFX_FORMAT_ASTC_6X5_UNORM_SRGB:
    case GFX_FORMAT_ASTC_6X6_UNORM:
    case GFX_FORMAT_ASTC_6X6_UNORM_SRGB:
    case GFX_FORMAT_ASTC_8X5_UNORM:
    case GFX_FORMAT_ASTC_8X5_UNORM_SRGB:
    case GFX_FORMAT_ASTC_8X6_UNORM:
    case GFX_FORMAT_ASTC_8X6_UNORM_SRGB:
    case GFX_FORMAT_ASTC_8X8_UNORM:
    case GFX_FORMAT_ASTC_8X8_UNORM_SRGB:
    case GFX_FORMAT_ASTC_10X5_UNORM:
    case GFX_FORMAT_ASTC_10X5_UNORM_SRGB:
    case GFX_FORMAT_ASTC_10X6_UNORM:
    case GFX_FORMAT_ASTC_10X6_UNORM_SRGB:
    case GFX_FORMAT_ASTC_10X8_UNORM:
    case GFX_FORMAT_ASTC_10X8_UNORM_SRGB:
    case GFX_FORMAT_ASTC_10X10_UNORM:
    case GFX_FORMAT_ASTC_10X10_UNORM_SRGB:
    case GFX_FORMAT_ASTC_12X10_UNORM:
    case GFX_FORMAT_ASTC_12X10_UNORM_SRGB:
    case GFX_FORMAT_ASTC_12X12_UNORM:
    case GFX_FORMAT_ASTC_12X12_UNORM_SRGB:
        return 16;
    // Uncompressed formats: block = one texel
    default:
        return getFormatBytesPerPixel(format);
    }
}

FormatCompressionFamily getFormatCompressionFamily(GfxFormat format)
{
    if (format >= GFX_FORMAT_BC1_RGBA_UNORM && format <= GFX_FORMAT_BC7_RGBA_UNORM_SRGB) {
        return FormatCompressionFamily::BC;
    }
    if (format >= GFX_FORMAT_ETC2_RGB8_UNORM && format <= GFX_FORMAT_EAC_RG11_SNORM) {
        return FormatCompressionFamily::ETC2;
    }
    if (format >= GFX_FORMAT_ASTC_4X4_UNORM && format <= GFX_FORMAT_ASTC_12X12_UNORM_SRGB) {
        return FormatCompressionFamily::ASTC;
    }
    return FormatCompressionFamily::None;
}

bool isCompressedFormat(GfxFormat format)
{
    return getFormatCompressionFamily(format) != FormatCompressionFamily::None;
}

void getFormatBlockDimensions(GfxFormat format, uint32_t* outWidth, uint32_t* outHeight)
{
    uint32_t width = 1;
    uint32_t height = 1;
    switch (format) {
    // 4x4 block families
    case GFX_FORMAT_BC1_RGBA_UNORM:
    case GFX_FORMAT_BC1_RGBA_UNORM_SRGB:
    case GFX_FORMAT_BC2_RGBA_UNORM:
    case GFX_FORMAT_BC2_RGBA_UNORM_SRGB:
    case GFX_FORMAT_BC3_RGBA_UNORM:
    case GFX_FORMAT_BC3_RGBA_UNORM_SRGB:
    case GFX_FORMAT_BC4_R_UNORM:
    case GFX_FORMAT_BC4_R_SNORM:
    case GFX_FORMAT_BC5_RG_UNORM:
    case GFX_FORMAT_BC5_RG_SNORM:
    case GFX_FORMAT_BC6H_RGB_UFLOAT:
    case GFX_FORMAT_BC6H_RGB_SFLOAT:
    case GFX_FORMAT_BC7_RGBA_UNORM:
    case GFX_FORMAT_BC7_RGBA_UNORM_SRGB:
    case GFX_FORMAT_ETC2_RGB8_UNORM:
    case GFX_FORMAT_ETC2_RGB8_UNORM_SRGB:
    case GFX_FORMAT_ETC2_RGB8A1_UNORM:
    case GFX_FORMAT_ETC2_RGB8A1_UNORM_SRGB:
    case GFX_FORMAT_ETC2_RGBA8_UNORM:
    case GFX_FORMAT_ETC2_RGBA8_UNORM_SRGB:
    case GFX_FORMAT_EAC_R11_UNORM:
    case GFX_FORMAT_EAC_R11_SNORM:
    case GFX_FORMAT_EAC_RG11_UNORM:
    case GFX_FORMAT_EAC_RG11_SNORM:
    case GFX_FORMAT_ASTC_4X4_UNORM:
    case GFX_FORMAT_ASTC_4X4_UNORM_SRGB:
        width = 4;
        height = 4;
        break;
    case GFX_FORMAT_ASTC_5X4_UNORM:
    case GFX_FORMAT_ASTC_5X4_UNORM_SRGB:
        width = 5;
        height = 4;
        break;
    case GFX_FORMAT_ASTC_5X5_UNORM:
    case GFX_FORMAT_ASTC_5X5_UNORM_SRGB:
        width = 5;
        height = 5;
        break;
    case GFX_FORMAT_ASTC_6X5_UNORM:
    case GFX_FORMAT_ASTC_6X5_UNORM_SRGB:
        width = 6;
        height = 5;
        break;
    case GFX_FORMAT_ASTC_6X6_UNORM:
    case GFX_FORMAT_ASTC_6X6_UNORM_SRGB:
        width = 6;
        height = 6;
        break;
    case GFX_FORMAT_ASTC_8X5_UNORM:
    case GFX_FORMAT_ASTC_8X5_UNORM_SRGB:
        width = 8;
        height = 5;
        break;
    case GFX_FORMAT_ASTC_8X6_UNORM:
    case GFX_FORMAT_ASTC_8X6_UNORM_SRGB:
        width = 8;
        height = 6;
        break;
    case GFX_FORMAT_ASTC_8X8_UNORM:
    case GFX_FORMAT_ASTC_8X8_UNORM_SRGB:
        width = 8;
        height = 8;
        break;
    case GFX_FORMAT_ASTC_10X5_UNORM:
    case GFX_FORMAT_ASTC_10X5_UNORM_SRGB:
        width = 10;
        height = 5;
        break;
    case GFX_FORMAT_ASTC_10X6_UNORM:
    case GFX_FORMAT_ASTC_10X6_UNORM_SRGB:
        width = 10;
        height = 6;
        break;
    case GFX_FORMAT_ASTC_10X8_UNORM:
    case GFX_FORMAT_ASTC_10X8_UNORM_SRGB:
        width = 10;
        height = 8;
        break;
    case GFX_FORMAT_ASTC_10X10_UNORM:
    case GFX_FORMAT_ASTC_10X10_UNORM_SRGB:
        width = 10;
        height = 10;
        break;
    case GFX_FORMAT_ASTC_12X10_UNORM:
    case GFX_FORMAT_ASTC_12X10_UNORM_SRGB:
        width = 12;
        height = 10;
        break;
    case GFX_FORMAT_ASTC_12X12_UNORM:
    case GFX_FORMAT_ASTC_12X12_UNORM_SRGB:
        width = 12;
        height = 12;
        break;
    default:
        break; // Uncompressed: 1x1
    }
    if (outWidth) {
        *outWidth = width;
    }
    if (outHeight) {
        *outHeight = height;
    }
}

const char* resultToString(GfxResult result)
{
    switch (result) {
    case GFX_RESULT_SUCCESS:
        return "GFX_RESULT_SUCCESS";
    case GFX_RESULT_TIMEOUT:
        return "GFX_RESULT_TIMEOUT";
    case GFX_RESULT_NOT_READY:
        return "GFX_RESULT_NOT_READY";
    case GFX_RESULT_ERROR_INVALID_ARGUMENT:
        return "GFX_RESULT_ERROR_INVALID_ARGUMENT";
    case GFX_RESULT_ERROR_NOT_FOUND:
        return "GFX_RESULT_ERROR_NOT_FOUND";
    case GFX_RESULT_ERROR_OUT_OF_MEMORY:
        return "GFX_RESULT_ERROR_OUT_OF_MEMORY";
    case GFX_RESULT_ERROR_DEVICE_LOST:
        return "GFX_RESULT_ERROR_DEVICE_LOST";
    case GFX_RESULT_ERROR_SURFACE_LOST:
        return "GFX_RESULT_ERROR_SURFACE_LOST";
    case GFX_RESULT_ERROR_OUT_OF_DATE:
        return "GFX_RESULT_ERROR_OUT_OF_DATE";
    case GFX_RESULT_ERROR_BACKEND_NOT_LOADED:
        return "GFX_RESULT_ERROR_BACKEND_NOT_LOADED";
    case GFX_RESULT_ERROR_FEATURE_NOT_SUPPORTED:
        return "GFX_RESULT_ERROR_FEATURE_NOT_SUPPORTED";
    case GFX_RESULT_ERROR_UNKNOWN:
        return "GFX_RESULT_ERROR_UNKNOWN";
    default:
        return "GFX_RESULT_UNKNOWN";
    }
}

void* getMetalLayerFromCocoaWindow(void* cocoaWindow)
{
#if defined(GFX_HAS_COCOA)
    // Cast the input parameter directly
    id nsWindow = (id)cocoaWindow;

    if (!nsWindow) {
        return nullptr;
    }

    // 1. Get contentView: [nsWindow contentView]
    auto getContentView = (id (*)(id, SEL))objc_msgSend;
    id nsView = getContentView(nsWindow, sel_getUid("contentView"));

    if (!nsView) {
        return nullptr;
    }

    // 2. Ensure it's layer-backed: [nsView setWantsLayer:YES]
    auto setWantsLayer = (void (*)(id, SEL, bool))objc_msgSend;
    setWantsLayer(nsView, sel_getUid("setWantsLayer:"), true);

    // 3. Get the existing layer (GLFW already creates a CAMetalLayer for Vulkan windows)
    // [nsView layer]
    auto getLayer = (id (*)(id, SEL))objc_msgSend;
    id existingLayer = getLayer(nsView, sel_getUid("layer"));

    // If an existing layer exists and it's a CAMetalLayer, use it
    if (existingLayer) {
        // Check if it's a CAMetalLayer
        Class caMetalLayerClass = objc_getClass("CAMetalLayer");
        if (!caMetalLayerClass) {
            return nullptr;
        }

        auto isKindOfClass = (bool (*)(id, SEL, Class))objc_msgSend;
        bool isMetalLayer = isKindOfClass(existingLayer, sel_getUid("isKindOfClass:"), caMetalLayerClass);

        if (isMetalLayer) {
            // Get the backing scale factor and set contentsScale for retina support
            auto getBackingScaleFactor = (double (*)(id, SEL))objc_msgSend;
            double scaleFactor = getBackingScaleFactor(nsWindow, sel_getUid("backingScaleFactor"));

            // [existingLayer setContentsScale:scaleFactor]
            auto setContentsScale = (void (*)(id, SEL, double))objc_msgSend;
            setContentsScale(existingLayer, sel_getUid("setContentsScale:"), scaleFactor);

            return (void*)existingLayer;
        }
    }

    // If no existing CAMetalLayer, create a new one
    Class caMetalLayerClass = objc_getClass("CAMetalLayer");
    if (!caMetalLayerClass) {
        return nullptr;
    }

    // Create new CAMetalLayer: [CAMetalLayer layer]
    auto layerMethod = (id (*)(id, SEL))objc_msgSend;
    id metalLayer = layerMethod((id)caMetalLayerClass, sel_getUid("layer"));

    if (!metalLayer) {
        return nullptr;
    }

    // 4. Set this as the view's layer: [nsView setLayer:metalLayer]
    auto setLayer = (void (*)(id, SEL, id))objc_msgSend;
    setLayer(nsView, sel_getUid("setLayer:"), metalLayer);

    // 5. Get the backing scale factor and set contentsScale for retina support
    // [nsWindow backingScaleFactor]
    auto getBackingScaleFactor = (double (*)(id, SEL))objc_msgSend;
    double scaleFactor = getBackingScaleFactor(nsWindow, sel_getUid("backingScaleFactor"));

    // [metalLayer setContentsScale:scaleFactor]
    auto setContentsScale = (void (*)(id, SEL, double))objc_msgSend;
    setContentsScale(metalLayer, sel_getUid("setContentsScale:"), scaleFactor);

    return (void*)metalLayer;
#else
    // Deriving a CAMetalLayer from a window is only supported for Cocoa (NSWindow).
    // On iOS/UIKit, pass a CAMetalLayer directly via gfxPlatformWindowHandleFromMetalLayer.
    (void)cocoaWindow;
    return nullptr;
#endif
}

} // namespace gfx::util
