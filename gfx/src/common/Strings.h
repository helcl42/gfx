#ifndef GFX_COMMON_STRINGS_H
#define GFX_COMMON_STRINGS_H

#include <string>
#include <vector>

namespace gfx::common {

bool containsString(const std::vector<std::string>& strings, const char* value);
bool containsString(const std::vector<const char*>& strings, const char* value);

} // namespace gfx::common

#endif // GFX_COMMON_STRINGS_H
