#ifndef GFX_COMMON_EXTENSIONS_H
#define GFX_COMMON_EXTENSIONS_H

#include <cstdint>
#include <string>
#include <vector>

namespace gfx::common {

// Whether every requested extension is among the supported ones
bool areExtensionsSupported(const std::vector<const char*>& supportedExtensions, const char* const* extensions, uint32_t extensionCount);
// Throws if a requested extension is not among the supported ones
void requireSupportedExtensions(const std::vector<const char*>& supportedExtensions, const std::vector<std::string>& extensions);

} // namespace gfx::common

#endif // GFX_COMMON_EXTENSIONS_H
