#include "CommonTest.h"

#include <string>

namespace {

// The library reports validation errors only through its log, so fail the running test on them
void failOnValidationError(gfx::LogLevel level, const std::string& message)
{
    if (level == gfx::LogLevel::Error && (message.starts_with("Vulkan [Error|Validation]") || message.starts_with("[WebGPU Uncaptured Error]"))) {
        ADD_FAILURE() << message;
    }
}

const bool validationEnforced = (gfx::setLogCallback(failOnValidationError), true);

} // namespace
