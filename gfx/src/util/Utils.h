#ifndef GFX_UTIL_UTILS_H
#define GFX_UTIL_UTILS_H

#include <gfx/gfx.h>

namespace gfx::util {

// Alignment utilities
uint64_t alignUp(uint64_t value, uint64_t alignment);
uint64_t alignDown(uint64_t value, uint64_t alignment);

// Format utilities
uint32_t getFormatBytesPerPixel(GfxFormat format);
uint32_t getFormatBlockSize(GfxFormat format);
void getFormatBlockDimensions(GfxFormat format, uint32_t* outWidth, uint32_t* outHeight);

// Compressed format families (each is gated behind a device extension)
enum class FormatCompressionFamily {
    None, // Uncompressed format
    BC,
    ETC2,
    ASTC,
};
FormatCompressionFamily getFormatCompressionFamily(GfxFormat format);
bool isCompressedFormat(GfxFormat format);

// Result to string conversion
const char* resultToString(GfxResult result);

// Returns the CAMetalLayer* backing the NSWindow's content view, creating one if needed.
// Cocoa (macOS) only - returns nullptr on other platforms and for a null window.
void* getMetalLayerFromCocoaWindow(void* cocoaWindow);

} // namespace gfx::util

#endif // GFX_UTIL_UTILS_H