#include "RenderPass.h"

#include "../system/Device.h"
#include "../util/Utils.h"

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
        VkAttachmentReference depthResolveRef{};
        bool depthHasResolve = false;
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

            if (createInfo.depthStencilAttachment->resolveTarget.has_value()) {
                layout.attachments.push_back(makeDepthStencilAttachmentDescription(createInfo.depthStencilAttachment->resolveTarget.value()));
                layout.depthResolveRef = { attachmentIndex++, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
                layout.depthHasResolve = true;
            }
        }

        return layout;
    }

    // Stages and writes through which the pass uses its attachments
    struct AttachmentScope {
        VkPipelineStageFlags stages = 0;
        VkAccessFlags writes = 0;
    };

    AttachmentScope makeAttachmentScope(bool hasColor, bool hasDepth)
    {
        AttachmentScope scope;
        if (hasColor) {
            scope.stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            scope.writes |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        }
        if (hasDepth) {
            scope.stages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            scope.writes |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        }
        return scope;
    }

    // External->subpass dependency: earlier attachment writes finish before the pass writes them
    VkSubpassDependency makeIncomingDependency(const AttachmentScope& scope)
    {
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = scope.stages;
        dependency.dstStageMask = scope.stages;
        dependency.srcAccessMask = scope.writes;
        dependency.dstAccessMask = scope.writes;
        dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        return dependency;
    }

    // Subpass->external dependency: the pass's writes and final layout transitions finish before the uses its final layouts name
    VkSubpassDependency makeOutgoingDependency(const AttachmentScope& scope, const std::vector<VkAttachmentDescription>& attachments)
    {
        VkSubpassDependency dependency{};
        dependency.srcSubpass = 0;
        dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
        dependency.srcStageMask = scope.stages;
        dependency.srcAccessMask = scope.writes;
        for (const auto& attachment : attachments) {
            dependency.dstStageMask |= getVkPipelineStagesForLayout(attachment.finalLayout);
            dependency.dstAccessMask |= getVkAccessFlagsForLayout(attachment.finalLayout);
        }
        return dependency;
    }

    VkRenderPass createRenderPass(Device* device, const AttachmentLayout& layout, const std::vector<VkSubpassDependency>& dependencies, const RenderPassCreateInfo& createInfo)
    {
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = static_cast<uint32_t>(layout.colorRefs.size());
        subpass.pColorAttachments = layout.colorRefs.empty() ? nullptr : layout.colorRefs.data();
        subpass.pResolveAttachments = layout.resolveRefs.empty() ? nullptr : layout.resolveRefs.data();
        subpass.pDepthStencilAttachment = layout.hasDepth ? &layout.depthRef : nullptr;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(layout.attachments.size());
        renderPassInfo.pAttachments = layout.attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();

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

        VkRenderPass renderPass = VK_NULL_HANDLE;
        if (vkCreateRenderPass(device->handle(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass");
        }
        return renderPass;
    }

    VkAttachmentDescription2 makeAttachmentDescription2(const VkAttachmentDescription& description)
    {
        VkAttachmentDescription2 description2{};
        description2.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        description2.flags = description.flags;
        description2.format = description.format;
        description2.samples = description.samples;
        description2.loadOp = description.loadOp;
        description2.storeOp = description.storeOp;
        description2.stencilLoadOp = description.stencilLoadOp;
        description2.stencilStoreOp = description.stencilStoreOp;
        description2.initialLayout = description.initialLayout;
        description2.finalLayout = description.finalLayout;
        return description2;
    }

    VkAttachmentReference2 makeAttachmentReference2(const VkAttachmentReference& reference)
    {
        VkAttachmentReference2 reference2{};
        reference2.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        reference2.attachment = reference.attachment;
        reference2.layout = reference.layout;
        return reference2;
    }

    VkSubpassDependency2 makeSubpassDependency2(const VkSubpassDependency& dependency)
    {
        VkSubpassDependency2 dependency2{};
        dependency2.sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
        dependency2.srcSubpass = dependency.srcSubpass;
        dependency2.dstSubpass = dependency.dstSubpass;
        dependency2.srcStageMask = dependency.srcStageMask;
        dependency2.dstStageMask = dependency.dstStageMask;
        dependency2.srcAccessMask = dependency.srcAccessMask;
        dependency2.dstAccessMask = dependency.dstAccessMask;
        dependency2.dependencyFlags = dependency.dependencyFlags;
        return dependency2;
    }

    // Creates the pass through render pass 2, which can also resolve the depth/stencil attachment (from sample 0)
    VkRenderPass createRenderPass2(Device* device, const AttachmentLayout& layout, const std::vector<VkSubpassDependency>& dependencies, const RenderPassCreateInfo& createInfo)
    {
        std::vector<VkAttachmentDescription2> attachments;
        for (const auto& attachment : layout.attachments) {
            attachments.push_back(makeAttachmentDescription2(attachment));
        }
        std::vector<VkAttachmentReference2> colorRefs;
        for (const auto& ref : layout.colorRefs) {
            colorRefs.push_back(makeAttachmentReference2(ref));
        }
        std::vector<VkAttachmentReference2> resolveRefs;
        for (const auto& ref : layout.resolveRefs) {
            resolveRefs.push_back(makeAttachmentReference2(ref));
        }
        const VkAttachmentReference2 depthRef = makeAttachmentReference2(layout.depthRef);
        const VkAttachmentReference2 depthResolveRef = makeAttachmentReference2(layout.depthResolveRef);

        VkSubpassDescriptionDepthStencilResolve depthStencilResolve{};
        if (layout.depthHasResolve) {
            const VkFormat format = layout.attachments[layout.depthResolveRef.attachment].format;
            depthStencilResolve.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE;
            depthStencilResolve.depthResolveMode = isDepthFormat(format) ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT : VK_RESOLVE_MODE_NONE;
            depthStencilResolve.stencilResolveMode = hasStencilComponent(format) ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT : VK_RESOLVE_MODE_NONE;
            depthStencilResolve.pDepthStencilResolveAttachment = &depthResolveRef;
        }

        VkSubpassDescription2 subpass{};
        subpass.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
        subpass.pNext = layout.depthHasResolve ? &depthStencilResolve : nullptr;
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.viewMask = createInfo.viewMask.value_or(0);
        subpass.colorAttachmentCount = static_cast<uint32_t>(colorRefs.size());
        subpass.pColorAttachments = colorRefs.empty() ? nullptr : colorRefs.data();
        subpass.pResolveAttachments = resolveRefs.empty() ? nullptr : resolveRefs.data();
        subpass.pDepthStencilAttachment = layout.hasDepth ? &depthRef : nullptr;

        std::vector<VkSubpassDependency2> dependencies2;
        for (const auto& dependency : dependencies) {
            dependencies2.push_back(makeSubpassDependency2(dependency));
        }

        VkRenderPassCreateInfo2 renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies2.size());
        renderPassInfo.pDependencies = dependencies2.data();
        renderPassInfo.correlatedViewMaskCount = static_cast<uint32_t>(createInfo.correlationMasks.size());
        renderPassInfo.pCorrelatedViewMasks = createInfo.correlationMasks.empty() ? nullptr : createInfo.correlationMasks.data();

        auto pfnCreateRenderPass2 = device->loadFunction<PFN_vkCreateRenderPass2KHR>("vkCreateRenderPass2KHR");
        VkRenderPass renderPass = VK_NULL_HANDLE;
        if (!pfnCreateRenderPass2 || pfnCreateRenderPass2(device->handle(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass");
        }
        return renderPass;
    }
} // anonymous namespace

RenderPass::RenderPass(Device* device, const RenderPassCreateInfo& createInfo)
    : m_device(device)
{
    m_colorAttachmentCount = static_cast<uint32_t>(createInfo.colorAttachments.size());
    m_hasDepthStencil = createInfo.depthStencilAttachment.has_value();

    AttachmentLayout layout = buildAttachmentLayout(createInfo);
    m_colorHasResolve = std::move(layout.colorHasResolve);
    m_depthStencilHasResolve = layout.depthHasResolve;
    for (const auto& attachment : layout.attachments) {
        m_attachmentInitialLayouts.push_back(attachment.initialLayout);
        m_attachmentFinalLayouts.push_back(attachment.finalLayout);
    }

    // Depth/stencil resolves are synchronized as color attachment writes
    const AttachmentScope scope = makeAttachmentScope(!layout.colorRefs.empty() || layout.depthHasResolve, layout.hasDepth);
    const std::vector<VkSubpassDependency> dependencies = { makeIncomingDependency(scope), makeOutgoingDependency(scope, layout.attachments) };

    if (layout.depthHasResolve) {
        m_renderPass = createRenderPass2(m_device, layout, dependencies, createInfo);
    } else {
        m_renderPass = createRenderPass(m_device, layout, dependencies, createInfo);
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

bool RenderPass::depthStencilHasResolve() const
{
    return m_depthStencilHasResolve;
}

const std::vector<bool>& RenderPass::colorHasResolve() const
{
    return m_colorHasResolve;
}

const std::vector<VkImageLayout>& RenderPass::attachmentInitialLayouts() const
{
    return m_attachmentInitialLayouts;
}

const std::vector<VkImageLayout>& RenderPass::attachmentFinalLayouts() const
{
    return m_attachmentFinalLayouts;
}

} // namespace gfx::backend::vulkan::core