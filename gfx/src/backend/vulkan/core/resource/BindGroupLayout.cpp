#include "BindGroupLayout.h"

#include "../system/Device.h"

#include <stdexcept>

namespace gfx::backend::vulkan::core {

BindGroupLayout::BindGroupLayout(Device* device, const BindGroupLayoutCreateInfo& createInfo)
    : m_device(device)
    , m_entries(createInfo.entries)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    for (const auto& entry : createInfo.entries) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = entry.binding;
        binding.descriptorCount = entry.descriptorCount;
        binding.descriptorType = entry.descriptorType;
        binding.stageFlags = entry.stageFlags;

        bindings.push_back(binding);
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VkResult result = vkCreateDescriptorSetLayout(m_device->handle(), &layoutInfo, nullptr, &m_layout);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
}

BindGroupLayout::~BindGroupLayout()
{
    if (m_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device->handle(), m_layout, nullptr);
    }
}

VkDescriptorSetLayout BindGroupLayout::handle() const
{
    return m_layout;
}

VkDescriptorType BindGroupLayout::getBindingType(uint32_t binding) const
{
    for (const BindGroupLayoutEntry& entry : m_entries) {
        if (entry.binding == binding) {
            return entry.descriptorType;
        }
    }
    return VK_DESCRIPTOR_TYPE_MAX_ENUM; // Invalid
}

const std::vector<BindGroupLayoutEntry>& BindGroupLayout::getEntries() const
{
    return m_entries;
}

} // namespace gfx::backend::vulkan::core