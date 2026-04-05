#ifndef GFX_WEBGPU_SURFACE_H
#define GFX_WEBGPU_SURFACE_H

#include "../CoreTypes.h"

#include <optional>

namespace gfx::backend::webgpu::core {

class Surface {
public:
    // Prevent copying
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    Surface(WGPUInstance instance, const SurfaceCreateInfo& createInfo);
    ~Surface();

    WGPUSurface handle() const;

    const WGPUSurfaceCapabilities& getCapabilities(WGPUAdapter adapter) const;

    SurfaceInfo getInfo() const;

private:
    WGPUSurface m_surface = nullptr;

    mutable std::optional<WGPUSurfaceCapabilities> m_capabilities{};
};

} // namespace gfx::backend::webgpu::core

#endif // GFX_WEBGPU_SURFACE_H