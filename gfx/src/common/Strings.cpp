#include "Strings.h"

#include <cstring>

namespace gfx::common {

bool containsString(const std::vector<std::string>& strings, const char* value)
{
    for (const auto& str : strings) {
        if (str == value) {
            return true;
        }
    }
    return false;
}

bool containsString(const std::vector<const char*>& strings, const char* value)
{
    for (const char* str : strings) {
        if (strcmp(str, value) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace gfx::common
