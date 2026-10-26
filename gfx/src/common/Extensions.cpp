#include "Extensions.h"

#include "Strings.h"

#include <stdexcept>

namespace gfx::common {

bool areExtensionsSupported(const std::vector<const char*>& supportedExtensions, const char* const* extensions, uint32_t extensionCount)
{
    for (uint32_t i = 0; i < extensionCount; ++i) {
        if (!containsString(supportedExtensions, extensions[i])) {
            return false;
        }
    }
    return true;
}

void requireSupportedExtensions(const std::vector<const char*>& supportedExtensions, const std::vector<std::string>& extensions)
{
    for (const std::string& extension : extensions) {
        if (!containsString(supportedExtensions, extension.c_str())) {
            throw std::runtime_error("Extension not supported: " + extension);
        }
    }
}

} // namespace gfx::common
