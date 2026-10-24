#include "Framebuffer.h"

#include "../resource/TextureView.h"
#include "../system/Device.h"

#include <stdexcept>
#include <vector>

namespace gfx::backend::vulkan::core {

Framebuffer::Framebuffer(Device* device, const FramebufferCreateInfo& createInfo)
    : m_device(device)
    , m_attachments(createInfo.attachments)
    , m_width(createInfo.width)
    , m_height(createInfo.height)
{
    std::vector<VkImageView> imageViews;
    imageViews.reserve(m_attachments.size());
    for (const TextureView* view : m_attachments) {
        imageViews.push_back(view->handle());
    }

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = createInfo.renderPass;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(imageViews.size());
    framebufferInfo.pAttachments = imageViews.data();
    framebufferInfo.width = createInfo.width;
    framebufferInfo.height = createInfo.height;
    framebufferInfo.layers = 1;

    VkResult result = vkCreateFramebuffer(m_device->handle(), &framebufferInfo, nullptr, &m_framebuffer);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create framebuffer");
    }
}

Framebuffer::~Framebuffer()
{
    if (m_framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(m_device->handle(), m_framebuffer, nullptr);
    }
}

VkFramebuffer Framebuffer::handle() const
{
    return m_framebuffer;
}

uint32_t Framebuffer::width() const
{
    return m_width;
}

uint32_t Framebuffer::height() const
{
    return m_height;
}

const std::vector<TextureView*>& Framebuffer::attachments() const
{
    return m_attachments;
}

} // namespace gfx::backend::vulkan::core