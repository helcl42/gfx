#ifndef GFX_VULKAN_BINDGROUPLAYOUT_H
#define GFX_VULKAN_BINDGROUPLAYOUT_H

#include "../CoreTypes.h"

#include <vector>

namespace gfx::backend::vulkan::core {

class Device;

class BindGroupLayout {
public:
    BindGroupLayout(const BindGroupLayout&) = delete;
    BindGroupLayout& operator=(const BindGroupLayout&) = delete;

    BindGroupLayout(Device* device, const BindGroupLayoutCreateInfo& createInfo);
    ~BindGroupLayout();

    VkDescriptorSetLayout handle() const;
    VkDescriptorType getBindingType(uint32_t binding) const;
    const std::vector<BindGroupLayoutEntry>& getEntries() const;

private:
    VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
    Device* m_device = nullptr;
    std::vector<BindGroupLayoutEntry> m_entries;
};

} // namespace gfx::backend::vulkan::core

#endif // GFX_VULKAN_BINDGROUPLAYOUT_H