#include "BindGroup.h"

#include "../system/Device.h"

#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace gfx::backend::vulkan::core {

namespace {
    // Creates a descriptor pool sized exactly for the given entries (one set)
    VkDescriptorPool createExactSizePool(VkDevice device, const std::vector<BindGroupEntry>& entries)
    {
        std::unordered_map<VkDescriptorType, uint32_t> descriptorCounts;
        for (const auto& entry : entries) {
            ++descriptorCounts[entry.descriptorType];
        }

        std::vector<VkDescriptorPoolSize> poolSizes;
        poolSizes.reserve(descriptorCounts.size());
        for (const auto& [type, count] : descriptorCounts) {
            poolSizes.push_back({ type, count });
        }

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 1; // Each BindGroup only allocates one descriptor set

        VkDescriptorPool pool = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }
        return pool;
    }

    VkWriteDescriptorSet makeWrite(VkDescriptorSet set, const BindGroupEntry& entry)
    {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = entry.binding;
        write.dstArrayElement = entry.arrayElement;
        write.descriptorType = entry.descriptorType;
        write.descriptorCount = 1;
        return write;
    }
} // anonymous namespace

BindGroup::BindGroup(Device* device, const BindGroupCreateInfo& createInfo)
    : m_device(device)
{
    m_pool = createExactSizePool(m_device->handle(), createInfo.entries);

    // Allocate descriptor set
    VkDescriptorSetLayout setLayout = createInfo.layout;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &setLayout;

    VkResult result = vkAllocateDescriptorSets(m_device->handle(), &allocInfo, &m_descriptorSet);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    // Build the descriptor writes; the info arrays are reserved up front so the
    // pointers stored in each write stay valid until vkUpdateDescriptorSets
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    bufferInfos.reserve(createInfo.entries.size());
    imageInfos.reserve(createInfo.entries.size());
    descriptorWrites.reserve(createInfo.entries.size());

    for (const auto& entry : createInfo.entries) {
        VkWriteDescriptorSet write = makeWrite(m_descriptorSet, entry);

        switch (entry.descriptorType) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            bufferInfos.push_back({ entry.buffer, entry.bufferOffset, entry.bufferSize });
            write.pBufferInfo = &bufferInfos.back();
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            imageInfos.push_back({ entry.sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED });
            write.pImageInfo = &imageInfos.back();
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            imageInfos.push_back({ VK_NULL_HANDLE, entry.imageView, entry.imageLayout });
            write.pImageInfo = &imageInfos.back();
            break;
        default:
            continue; // Unknown descriptor type - skip
        }

        descriptorWrites.push_back(write);
    }

    if (!descriptorWrites.empty()) {
        vkUpdateDescriptorSets(m_device->handle(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

BindGroup::~BindGroup()
{
    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device->handle(), m_pool, nullptr);
    }
}

VkDescriptorSet BindGroup::handle() const
{
    return m_descriptorSet;
}

} // namespace gfx::backend::vulkan::core