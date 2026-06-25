#ifndef GFX_VULKAN_TEXTURE_H
#define GFX_VULKAN_TEXTURE_H

#include "../CoreTypes.h"
#include "../util/VmaAllocator.h"

namespace gfx::backend::vulkan::core {

class CommandEncoder;
class Device;

class Texture {
public:
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Device* device, const TextureCreateInfo& createInfo);
    Texture(Device* device, VkImage image, const TextureCreateInfo& createInfo);
    Texture(Device* device, VkImage image, const TextureImportInfo& importInfo);
    ~Texture();

    VkImage handle() const;
    VkDevice device() const;
    VkImageType getImageType() const;
    VkExtent3D getSize() const;
    uint32_t getArrayLayers() const;
    VkFormat getFormat() const;
    uint32_t getMipLevelCount() const;
    VkSampleCountFlagBits getSampleCount() const;
    VkImageUsageFlags getUsage() const;
    const TextureInfo& getInfo() const;

    VkImageLayout getLayout() const;
    void setLayout(VkImageLayout layout);

    // Transition a subresource range, using the single tracked whole-image layout as the old layout.
    void transitionLayout(CommandEncoder* encoder, VkImageLayout newLayout, uint32_t baseMipLevel, uint32_t levelCount, uint32_t baseArrayLayer, uint32_t layerCount);
    void transitionLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout, uint32_t baseMipLevel, uint32_t levelCount, uint32_t baseArrayLayer, uint32_t layerCount);

    // Transition with an EXPLICIT old layout (the CommandEncoder* overload updates the tracker, the
    // VkCommandBuffer overload doesn't — caller owns it).
    void transitionLayout(CommandEncoder* encoder, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t baseMipLevel, uint32_t levelCount, uint32_t baseArrayLayer, uint32_t layerCount);
    void transitionLayout(VkCommandBuffer commandBuffer, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t baseMipLevel, uint32_t levelCount, uint32_t baseArrayLayer, uint32_t layerCount);
    void generateMipmaps(CommandEncoder* encoder);
    void generateMipmapsRange(CommandEncoder* encoder, uint32_t baseMipLevel, uint32_t levelCount);

private:

    static TextureInfo createTextureInfo(const TextureCreateInfo& info);
    static TextureInfo createTextureInfo(const TextureImportInfo& info);

    Device* m_device = nullptr;
    bool m_ownsResources = true;
    TextureInfo m_info{};
    VkImage m_image = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VkImageLayout m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};

} // namespace gfx::backend::vulkan::core

#endif // GFX_VULKAN_TEXTURE_H