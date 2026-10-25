#include "CommonTest.h"

#include <string_view>

namespace {

// The library reports validation errors only through its log, so fail the running test on them
void failOnValidationError(GfxLogLevel level, const char* message, void*)
{
    const std::string_view text(message);
    if (level == GFX_LOG_LEVEL_ERROR && (text.starts_with("Vulkan [Error|Validation]") || text.starts_with("[WebGPU Uncaptured Error]"))) {
        ADD_FAILURE() << text;
    }
}

const bool validationEnforced = (gfxSetLogCallback(failOnValidationError, nullptr), true);

} // namespace
