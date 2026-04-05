#ifndef GFX_VULKAN_SURFACE_H
#define GFX_VULKAN_SURFACE_H

#include "../CoreTypes.h"

namespace gfx::backend::vulkan::core {

class Adapter;
class Instance;

class Surface {
public:
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    Surface(Instance* instance, const SurfaceCreateInfo& createInfo);
    ~Surface();

    VkInstance instance() const;
    VkSurfaceKHR handle() const;
    Instance* getInstance() const;

    std::vector<VkSurfaceFormatKHR> getSupportedFormats(Adapter* adapter) const;
    std::vector<VkPresentModeKHR> getSupportedPresentModes(Adapter* adapter) const;

    VkSurfaceCapabilitiesKHR getCapabilities(Adapter* adapter) const;

private:
    Instance* m_instance = nullptr;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
};

} // namespace gfx::backend::vulkan::core

#endif // GFX_VULKAN_SURFACE_H