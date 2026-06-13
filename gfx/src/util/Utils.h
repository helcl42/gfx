#ifndef GFX_UTIL_UTILS_H
#define GFX_UTIL_UTILS_H

#include <gfx/gfx.h>

namespace gfx::util {

// Alignment utilities
uint64_t alignUp(uint64_t value, uint64_t alignment);
uint64_t alignDown(uint64_t value, uint64_t alignment);

// Format utilities
uint32_t getFormatBytesPerPixel(GfxFormat format);

// Result to string conversion
const char* resultToString(GfxResult result);

// Returns the CAMetalLayer* backing the NSWindow's content view, creating one if needed.
// Cocoa (macOS) only - returns nullptr on other platforms and for a null window.
void* getMetalLayerFromCocoaWindow(void* cocoaWindow);

} // namespace gfx::util

#endif // GFX_UTIL_UTILS_H