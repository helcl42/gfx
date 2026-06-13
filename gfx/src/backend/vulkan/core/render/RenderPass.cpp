#include "RenderPass.h"

#include "../system/Device.h"

#include <stdexcept>

namespace gfx::backend::vulkan::core {

namespace {
    VkAttachmentDescription makeColorAttachmentDescription(const RenderPassColorAttachmentTarget& target, VkSampleCountFlagBits samples)
    {
        VkAttachmentDescription desc{};
        desc.format = target.format;
        desc.samples = samples;
        desc.loadOp = target.loadOp;
        desc.storeOp = target.storeOp;
        desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        desc.initialLayout = (target.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD)
            ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            : VK_IMAGE_LAYOUT_UNDEFINED;
        desc.finalLayout = target.finalLayout;
        return desc;
    }

    VkAttachmentDescription makeDepthStencilAttachmentDescription(const RenderPassDepthStencilAttachmentTarget& target)
    {
        bool loadDepth = (target.depthLoadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
        bool loadStencil = (target.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_LOAD);

        VkAttachmentDescription desc{};
        desc.format = target.format;
        desc.samples = target.sampleCount;
        desc.loadOp = target.depthLoadOp;
        desc.storeOp = target.depthStoreOp;
        desc.stencilLoadOp = target.stencilLoadOp;
        desc.stencilStoreOp = target.stencilStoreOp;
        desc.initialLayout = (loadDepth || loadStencil)
            ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            : VK_IMAGE_LAYOUT_UNDEFINED;
        desc.finalLayout = target.finalLayout;
        return desc;
    }

    // Flattened attachment descriptions and subpass references for one render pass
    struct AttachmentLayout {
        std::vector<VkAttachmentDescription> attachments;
        std::vector<VkAttachmentReference> colorRefs;
        std::vector<VkAttachmentReference> resolveRefs;
        std::vector<bool> colorHasResolve;
        VkAttachmentReference depthRef{};
        bool hasDepth = false;
    };

    AttachmentLayout buildAttachmentLayout(const RenderPassCreateInfo& createInfo)
    {
        AttachmentLayout layout;
        uint32_t attachmentIndex = 0;

        for (const auto& colorAttachment : createInfo.colorAttachments) {
            const RenderPassColorAttachmentTarget& target = colorAttachment.target;
            bool isMSAA = (target.sampleCount > VK_SAMPLE_COUNT_1_BIT);

            layout.attachments.push_back(makeColorAttachmentDescription(target, target.sampleCount));
            layout.colorRefs.push_back({ attachmentIndex++, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

            if (colorAttachment.resolveTarget.has_value()) {
                // Resolve targets are always single-sampled
                layout.attachments.push_back(makeColorAttachmentDescription(colorAttachment.resolveTarget.value(), VK_SAMPLE_COUNT_1_BIT));
                layout.resolveRefs.push_back({ attachmentIndex++, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
                layout.colorHasResolve.push_back(true);
            } else if (isMSAA) {
                // MSAA without resolve needs an unused reference
                layout.resolveRefs.push_back({ VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED });
                layout.colorHasResolve.push_back(false);
            } else {
                layout.colorHasResolve.push_back(false);
            }
        }

        // NOTE - handle stencil-only attachments? We would need Vulkan 1.2 for that
        if (createInfo.depthStencilAttachment.has_value()) {
            layout.attachments.push_back(makeDepthStencilAttachmentDescription(createInfo.depthStencilAttachment->target));
            layout.depthRef = { attachmentIndex++, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
            layout.hasDepth = true;
        }

        return layout;
    }

    // External->subpass dependency covering the stages the pass actually uses
    VkSubpassDependency makeExternalDependency(bool hasColor, bool hasDepth)
    {
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        if (hasColor) {
            dependency.srcStageMask |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.dstStageMask |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.srcAccessMask |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependency.dstAccessMask |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        }
        if (hasDepth) {
            dependency.srcStageMask |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            dependency.dstStageMask |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            dependency.srcAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            dependency.dstAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        }
        return dependency;
    }
} // anonymous namespace

RenderPass::RenderPass(Device* device, const RenderPassCreateInfo& createInfo)
    : m_device(device)
{
    m_colorAttachmentCount = static_cast<uint32_t>(createInfo.colorAttachments.size());
    m_hasDepthStencil = createInfo.depthStencilAttachment.has_value();

    AttachmentLayout layout = buildAttachmentLayout(createInfo);
    m_colorHasResolve = std::move(layout.colorHasResolve);

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(layout.colorRefs.size());
    subpass.pColorAttachments = layout.colorRefs.empty() ? nullptr : layout.colorRefs.data();
    subpass.pResolveAttachments = layout.resolveRefs.empty() ? nullptr : layout.resolveRefs.data();
    subpass.pDepthStencilAttachment = layout.hasDepth ? &layout.depthRef : nullptr;

    VkSubpassDependency dependency = makeExternalDependency(!layout.colorRefs.empty(), layout.hasDepth);

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(layout.attachments.size());
    renderPassInfo.pAttachments = layout.attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    // Add multiview support if requested
    VkRenderPassMultiviewCreateInfo multiviewInfo{};
    if (createInfo.viewMask.has_value()) {
        multiviewInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
        multiviewInfo.subpassCount = 1;
        multiviewInfo.pViewMasks = &createInfo.viewMask.value();
        multiviewInfo.correlationMaskCount = static_cast<uint32_t>(createInfo.correlationMasks.size());
        multiviewInfo.pCorrelationMasks = createInfo.correlationMasks.empty() ? nullptr : createInfo.correlationMasks.data();

        renderPassInfo.pNext = &multiviewInfo;
    }

    VkResult result = vkCreateRenderPass(m_device->handle(), &renderPassInfo, nullptr, &m_renderPass);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render pass");
    }
}

RenderPass::~RenderPass()
{
    if (m_renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(m_device->handle(), m_renderPass, nullptr);
    }
}

VkRenderPass RenderPass::handle() const
{
    return m_renderPass;
}

uint32_t RenderPass::colorAttachmentCount() const
{
    return m_colorAttachmentCount;
}

bool RenderPass::hasDepthStencil() const
{
    return m_hasDepthStencil;
}

const std::vector<bool>& RenderPass::colorHasResolve() const
{
    return m_colorHasResolve;
}

} // namespace gfx::backend::vulkan::core