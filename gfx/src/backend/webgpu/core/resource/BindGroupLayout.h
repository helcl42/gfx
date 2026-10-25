#ifndef GFX_WEBGPU_BIND_GROUP_LAYOUT_H
#define GFX_WEBGPU_BIND_GROUP_LAYOUT_H

#include "../CoreTypes.h"

namespace gfx::backend::webgpu::core {

class Device;

class BindGroupLayout {
public:
    // Prevent copying
    BindGroupLayout(const BindGroupLayout&) = delete;
    BindGroupLayout& operator=(const BindGroupLayout&) = delete;

    BindGroupLayout(Device* device, const BindGroupLayoutCreateInfo& createInfo);
    ~BindGroupLayout();

    WGPUBindGroupLayout handle() const;
    const std::vector<BindGroupLayoutEntry>& getEntries() const;

private:
    WGPUBindGroupLayout m_layout = nullptr;
    std::vector<BindGroupLayoutEntry> m_entries;
};

} // namespace gfx::backend::webgpu::core

#endif // GFX_WEBGPU_BIND_GROUP_LAYOUT_H