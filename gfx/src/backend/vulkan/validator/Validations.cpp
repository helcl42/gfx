#include "Validations.h"

#include "backend/vulkan/converter/Conversions.h"
#include "backend/vulkan/core/command/CommandEncoder.h"
#include "backend/vulkan/core/command/RenderPassEncoder.h"
#include "backend/vulkan/core/query/QuerySet.h"
#include "backend/vulkan/core/resource/BindGroupLayout.h"
#include "backend/vulkan/core/resource/Buffer.h"
#include "backend/vulkan/core/resource/Texture.h"
#include "backend/vulkan/core/system/Adapter.h"
#include "backend/vulkan/core/system/Device.h"
#include "backend/vulkan/core/system/Queue.h"
#include "backend/vulkan/core/util/Utils.h"
#include "common/Strings.h"
#include "util/Utils.h"

#include <cstdint>

namespace gfx::backend::vulkan::validator {

namespace {

    // ============================================================================
    // Internal descriptor validation functions
    // ============================================================================

    GfxResult validateInstanceDescriptor(const GfxInstanceDescriptor* descriptor)
    {
        // Descriptor is optional
        if (!descriptor) {
            return GFX_RESULT_SUCCESS;
        }

        // All fields are optional - no specific validation needed
        // applicationName, applicationVersion, enabledFeatures are all optional
        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateAdapterDescriptor(const GfxAdapterDescriptor* descriptor)
    {
        // Descriptor is optional
        if (!descriptor) {
            return GFX_RESULT_SUCCESS;
        }

        // With UNDEFINED preference the adapterIndex is used; UINT32_MAX can never be
        // a valid index (and aliases the internal preference sentinel), so reject it
        if (descriptor->preference == GFX_ADAPTER_PREFERENCE_UNDEFINED && descriptor->adapterIndex == UINT32_MAX) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateSwapchainDescriptor(const GfxSwapchainDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate surface
        if (!descriptor->surface) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate dimensions
        if (descriptor->extent.width == 0 || descriptor->extent.height == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate image count
        if (descriptor->imageCount == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate format
        if (descriptor->format == GFX_FORMAT_UNDEFINED) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate usage flags
        if (descriptor->usage == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate present mode
        if (descriptor->presentMode < GFX_PRESENT_MODE_IMMEDIATE || descriptor->presentMode > GFX_PRESENT_MODE_MAILBOX) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateDeviceDescriptor(const GfxDeviceDescriptor* descriptor)
    {
        // Descriptor is optional - NULL means use defaults
        if (!descriptor) {
            return GFX_RESULT_SUCCESS;
        }

        // Validate queueRequests and queueRequestCount consistency
        if (descriptor->queueRequests != nullptr && descriptor->queueRequestCount == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (descriptor->queueRequests == nullptr && descriptor->queueRequestCount != 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate individual queue requests
        if (descriptor->queueRequests != nullptr) {
            for (uint32_t i = 0; i < descriptor->queueRequestCount; ++i) {
                const GfxQueueRequest& request = descriptor->queueRequests[i];

                // Validate priority (0.0 to 1.0)
                if (request.priority < 0.0f || request.priority > 1.0f) {
                    return GFX_RESULT_ERROR_INVALID_ARGUMENT;
                }
            }
        }

        // Validate enabledExtensions and enabledExtensionCount consistency
        if (descriptor->enabledExtensions != nullptr && descriptor->enabledExtensionCount == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (descriptor->enabledExtensions == nullptr && descriptor->enabledExtensionCount != 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateBufferDescriptor(const GfxBufferDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate size
        if (descriptor->size == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate usage flags
        if (descriptor->usage == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate memory properties - must be specified
        if (descriptor->memoryProperties == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate memory properties - check for invalid flag combinations
        {
            constexpr uint32_t validFlags = GFX_MEMORY_PROPERTY_DEVICE_LOCAL | GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT | GFX_MEMORY_PROPERTY_HOST_CACHED;

            if (descriptor->memoryProperties & ~validFlags) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }

            // HOST_COHERENT requires HOST_VISIBLE
            if ((descriptor->memoryProperties & GFX_MEMORY_PROPERTY_HOST_COHERENT) && !(descriptor->memoryProperties & GFX_MEMORY_PROPERTY_HOST_VISIBLE)) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }

            // HOST_CACHED requires HOST_VISIBLE
            if ((descriptor->memoryProperties & GFX_MEMORY_PROPERTY_HOST_CACHED) && !(descriptor->memoryProperties & GFX_MEMORY_PROPERTY_HOST_VISIBLE)) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }
        }

        // Validate consistency between map usage and memory properties
        {
            const bool hasMapUsage = (descriptor->usage & (GFX_BUFFER_USAGE_MAP_READ | GFX_BUFFER_USAGE_MAP_WRITE)) != 0;
            const bool hasHostVisible = (descriptor->memoryProperties & GFX_MEMORY_PROPERTY_HOST_VISIBLE) != 0;

            // MapRead or MapWrite requires HostVisible
            if (hasMapUsage && !hasHostVisible) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }

            // HostVisible requires MapRead or MapWrite (or is intended for staging)
            // Note: We don't enforce this strictly as HostVisible staging buffers
            // without explicit map usage are valid use cases
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateTextureDescriptor(const GfxTextureDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate dimensions based on texture type
        switch (descriptor->type) {
        case GFX_TEXTURE_TYPE_1D:
            if (descriptor->size.width == 0) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }
            break;
        case GFX_TEXTURE_TYPE_2D:
            if (descriptor->size.width == 0 || descriptor->size.height == 0) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }
            break;
        case GFX_TEXTURE_TYPE_3D:
            if (descriptor->size.width == 0 || descriptor->size.height == 0 || descriptor->size.depth == 0) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }
            break;
        default:
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate format
        if (descriptor->format == GFX_FORMAT_UNDEFINED) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate usage flags
        if (descriptor->usage == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate mip levels
        if (descriptor->mipLevelCount == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate array layers
        if (descriptor->arrayLayerCount == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateBufferImportDescriptor(const GfxBufferImportDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate native handle
        if (!descriptor->nativeHandle) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate size
        if (descriptor->size == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate usage flags
        if (descriptor->usage == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateTextureImportDescriptor(const GfxTextureImportDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate native handle
        if (!descriptor->nativeHandle) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate dimensions based on texture type
        switch (descriptor->type) {
        case GFX_TEXTURE_TYPE_1D:
            if (descriptor->size.width == 0) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }
            break;
        case GFX_TEXTURE_TYPE_2D:
            if (descriptor->size.width == 0 || descriptor->size.height == 0) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }
            break;
        case GFX_TEXTURE_TYPE_3D:
            if (descriptor->size.width == 0 || descriptor->size.height == 0 || descriptor->size.depth == 0) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }
            break;
        default:
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate usage flags
        if (descriptor->usage == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate mip levels
        if (descriptor->mipLevelCount == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate array layers
        if (descriptor->arrayLayerCount == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateSamplerDescriptor(const GfxSamplerDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate filter modes
        if (descriptor->magFilter < GFX_FILTER_MODE_NEAREST || descriptor->magFilter > GFX_FILTER_MODE_LINEAR) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        if (descriptor->minFilter < GFX_FILTER_MODE_NEAREST || descriptor->minFilter > GFX_FILTER_MODE_LINEAR) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        if (descriptor->mipmapFilter < GFX_FILTER_MODE_NEAREST || descriptor->mipmapFilter > GFX_FILTER_MODE_LINEAR) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate address modes
        if (descriptor->addressModeU < GFX_ADDRESS_MODE_REPEAT || descriptor->addressModeU > GFX_ADDRESS_MODE_CLAMP_TO_EDGE) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        if (descriptor->addressModeV < GFX_ADDRESS_MODE_REPEAT || descriptor->addressModeV > GFX_ADDRESS_MODE_CLAMP_TO_EDGE) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        if (descriptor->addressModeW < GFX_ADDRESS_MODE_REPEAT || descriptor->addressModeW > GFX_ADDRESS_MODE_CLAMP_TO_EDGE) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // WebGPU rule: anisotropic filtering requires linear filtering
        if (descriptor->maxAnisotropy > 1 && (descriptor->magFilter != GFX_FILTER_MODE_LINEAR || descriptor->minFilter != GFX_FILTER_MODE_LINEAR || descriptor->mipmapFilter != GFX_FILTER_MODE_LINEAR)) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateShaderDescriptor(const GfxShaderDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate code pointer and size
        if (!descriptor->code || descriptor->codeSize == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // For SPIR-V, code size should be multiple of 4 bytes (for binary formats)
        if (descriptor->sourceType == GFX_SHADER_SOURCE_SPIRV && descriptor->codeSize % 4 != 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateTextureViewDescriptor(const GfxTextureViewDescriptor* descriptor)
    {
        // Descriptor is required
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate format
        if (descriptor->format == GFX_FORMAT_UNDEFINED) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate mip level count
        if (descriptor->mipLevelCount == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate array layer count
        if (descriptor->arrayLayerCount == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateBindGroupLayoutDescriptor(const GfxBindGroupLayoutDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate entries if provided
        if (descriptor->entryCount > 0 && !descriptor->entries) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    bool hasBindGroupEntry(const GfxBindGroupDescriptor* descriptor, uint32_t binding, uint32_t arrayElement)
    {
        for (uint32_t i = 0; i < descriptor->entryCount; ++i) {
            if (descriptor->entries[i].binding == binding && descriptor->entries[i].arrayElement == arrayElement) {
                return true;
            }
        }
        return false;
    }

    GfxResult validateBindGroupDescriptor(const GfxBindGroupDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate layout
        if (!descriptor->layout) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate entries if provided
        if (descriptor->entryCount > 0 && !descriptor->entries) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateConstants(const GfxConstantEntry* constants, uint32_t constantCount)
    {
        if (constantCount == 0) {
            return GFX_RESULT_SUCCESS;
        }

        if (!constants) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        for (uint32_t i = 0; i < constantCount; ++i) {
            switch (constants[i].type) {
            case GFX_CONSTANT_TYPE_BOOL:
            case GFX_CONSTANT_TYPE_I32:
            case GFX_CONSTANT_TYPE_U32:
            case GFX_CONSTANT_TYPE_F32:
                break;
            default:
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }

            // Duplicate ids would make the applied value depend on entry order
            for (uint32_t j = i + 1; j < constantCount; ++j) {
                if (constants[j].id == constants[i].id) {
                    return GFX_RESULT_ERROR_INVALID_ARGUMENT;
                }
            }
        }

        return GFX_RESULT_SUCCESS;
    }

    // Optional formats require their device extension to be enabled
    GfxResult validateFormatEnabled(const core::Device* dev, GfxFormat format)
    {
        switch (gfx::util::getFormatCompressionFamily(format)) {
        case gfx::util::FormatCompressionFamily::BC:
            if (!dev->isExtensionEnabled(core::DeviceExtension::TextureCompressionBC)) {
                return GFX_RESULT_ERROR_FEATURE_NOT_SUPPORTED;
            }
            break;
        case gfx::util::FormatCompressionFamily::ETC2:
            if (!dev->isExtensionEnabled(core::DeviceExtension::TextureCompressionETC2)) {
                return GFX_RESULT_ERROR_FEATURE_NOT_SUPPORTED;
            }
            break;
        case gfx::util::FormatCompressionFamily::ASTC:
            if (!dev->isExtensionEnabled(core::DeviceExtension::TextureCompressionASTC)) {
                return GFX_RESULT_ERROR_FEATURE_NOT_SUPPORTED;
            }
            break;
        case gfx::util::FormatCompressionFamily::None:
            break;
        }
        if (format == GFX_FORMAT_DEPTH32_FLOAT_STENCIL8 && !dev->isExtensionEnabled(core::DeviceExtension::Depth32FloatStencil8)) {
            return GFX_RESULT_ERROR_FEATURE_NOT_SUPPORTED;
        }
        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateRenderPipelineDescriptor(const GfxRenderPipelineDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate vertex state
        if (!descriptor->vertex) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate render pass
        if (!descriptor->renderPass) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // A pipeline must write a color target or depth-stencil, as in WebGPU
        if ((!descriptor->fragment || descriptor->fragment->targetCount == 0) && !descriptor->depthStencil) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (descriptor->fragment && descriptor->fragment->targetCount > 0 && !descriptor->fragment->targets) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        GfxResult constantsResult = validateConstants(descriptor->vertex->constants, descriptor->vertex->constantCount);
        if (constantsResult != GFX_RESULT_SUCCESS) {
            return constantsResult;
        }

        if (descriptor->fragment) {
            constantsResult = validateConstants(descriptor->fragment->constants, descriptor->fragment->constantCount);
            if (constantsResult != GFX_RESULT_SUCCESS) {
                return constantsResult;
            }
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateComputePipelineDescriptor(const GfxComputePipelineDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate compute state
        if (!descriptor->compute || !descriptor->compute->module) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return validateConstants(descriptor->compute->constants, descriptor->compute->constantCount);
    }

    GfxResult validateRenderPassDescriptor(const GfxRenderPassDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate color attachments if provided
        if (descriptor->colorAttachmentCount > 0 && !descriptor->colorAttachments) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Depth/stencil resolve is not supported
        if (descriptor->depthStencilAttachment && descriptor->depthStencilAttachment->resolveTarget) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // A texture cannot be left in the UNDEFINED layout
        for (uint32_t i = 0; i < descriptor->colorAttachmentCount; ++i) {
            const GfxRenderPassColorAttachment& attachment = descriptor->colorAttachments[i];
            if (attachment.target.finalLayout == GFX_TEXTURE_LAYOUT_UNDEFINED || (attachment.resolveTarget && attachment.resolveTarget->finalLayout == GFX_TEXTURE_LAYOUT_UNDEFINED)) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }
        }
        if (descriptor->depthStencilAttachment && descriptor->depthStencilAttachment->target.finalLayout == GFX_TEXTURE_LAYOUT_UNDEFINED) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateFramebufferDescriptor(const GfxFramebufferDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate render pass
        if (!descriptor->renderPass) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate dimensions
        if (descriptor->extent.width == 0 || descriptor->extent.height == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate color attachments if provided
        if (descriptor->colorAttachmentCount > 0 && !descriptor->colorAttachments) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Depth/stencil resolve is not supported
        if (descriptor->depthStencilAttachment.resolveTarget) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateFenceDescriptor(const GfxFenceDescriptor* descriptor)
    {
        // Descriptor is optional
        if (!descriptor) {
            return GFX_RESULT_SUCCESS;
        }

        // No specific validation needed - signaled flag is any bool value
        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateSemaphoreDescriptor(const GfxSemaphoreDescriptor* descriptor)
    {
        // Descriptor is optional
        if (!descriptor) {
            return GFX_RESULT_SUCCESS;
        }

        // No specific validation needed - type and initialValue are both valid
        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateRenderPassBeginDescriptor(const GfxRenderPassBeginDescriptor* descriptor, bool isBundle)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        if (!descriptor->renderPass) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Framebuffer is required for regular render passes, but not for bundle recording
        if (!isBundle && !descriptor->framebuffer) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate color clear values if provided
        if (descriptor->colorClearValueCount > 0 && !descriptor->colorClearValues) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateComputePassBeginDescriptor(const GfxComputePassBeginDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // No specific validation needed - label is optional
        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateCopyBufferToBufferDescriptor(const GfxCopyBufferToBufferDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate source and destination buffers
        if (!descriptor->source || !descriptor->destination) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate size
        if (descriptor->size == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    // Common aspect/layout checks for buffer<->texture copies
    GfxResult validateCopyAspectAndRowPitch(GfxTexture texture, GfxTextureAspect aspect, uint32_t bytesPerRow)
    {
        const auto* tex = converter::toNative<core::Texture>(texture);
        VkFormat format = tex->getFormat();
        VkImageAspectFlags aspectMask = converter::gfxTextureAspectToVkAspectMask(aspect, format);

        // Buffer<->texture copies of combined depth-stencil formats must select a single aspect
        if (aspectMask == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // bytesPerRow must be a whole number of texels (Vulkan's bufferRowLength is in texels)
        uint32_t texelSize = core::getAspectTexelSize(format, aspectMask);
        if (bytesPerRow != 0 && texelSize != 0 && bytesPerRow % texelSize != 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    // WebGPU rule: the buffer row pitch of a buffer<->texture copy must be a multiple of 256 bytes;
    // bytesPerRow = 0 (tightly packed) qualifies when the copy is a single row or the packed row is aligned
    GfxResult validateBufferCopyRowPitch(GfxTexture texture, GfxTextureAspect aspect, const GfxExtent3D& extent, uint32_t bytesPerRow)
    {
        if (bytesPerRow % 256 != 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (bytesPerRow != 0) {
            return GFX_RESULT_SUCCESS;
        }

        const auto* tex = converter::toNative<core::Texture>(texture);
        VkFormat format = tex->getFormat();
        uint32_t blockWidth = 1;
        uint32_t blockHeight = 1;
        core::getVkFormatBlockDimensions(format, &blockWidth, &blockHeight);
        const uint32_t rowCount = (extent.height + blockHeight - 1) / blockHeight;
        const uint32_t packedBytesPerRow = ((extent.width + blockWidth - 1) / blockWidth) * core::getAspectTexelSize(format, converter::gfxTextureAspectToVkAspectMask(aspect, format));
        if ((rowCount > 1 || extent.depth > 1) && packedBytesPerRow % 256 != 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        return GFX_RESULT_SUCCESS;
    }

    GfxResult validateCopyBufferToTextureDescriptor(const GfxCopyBufferToTextureDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate source and destination
        if (!descriptor->source || !descriptor->destination) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (descriptor->finalLayout == GFX_TEXTURE_LAYOUT_UNDEFINED) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate extent
        if (descriptor->extent.width == 0 || descriptor->extent.height == 0 || descriptor->extent.depth == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        GfxResult aspectResult = validateCopyAspectAndRowPitch(descriptor->destination, descriptor->aspect, descriptor->bytesPerRow);
        if (aspectResult != GFX_RESULT_SUCCESS) {
            return aspectResult;
        }
        return validateBufferCopyRowPitch(descriptor->destination, descriptor->aspect, descriptor->extent, descriptor->bytesPerRow);
    }

    GfxResult validateCopyTextureToBufferDescriptor(const GfxCopyTextureToBufferDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate source and destination
        if (!descriptor->source || !descriptor->destination) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (descriptor->finalLayout == GFX_TEXTURE_LAYOUT_UNDEFINED) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        GfxResult aspectResult = validateCopyAspectAndRowPitch(descriptor->source, descriptor->aspect, descriptor->bytesPerRow);
        if (aspectResult != GFX_RESULT_SUCCESS) {
            return aspectResult;
        }

        // Validate extent
        if (descriptor->extent.width == 0 || descriptor->extent.height == 0 || descriptor->extent.depth == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return validateBufferCopyRowPitch(descriptor->source, descriptor->aspect, descriptor->extent, descriptor->bytesPerRow);
    }

    GfxResult validateCopyTextureToTextureDescriptor(const GfxCopyTextureToTextureDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate source and destination
        if (!descriptor->source || !descriptor->destination) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (descriptor->sourceFinalLayout == GFX_TEXTURE_LAYOUT_UNDEFINED || descriptor->destinationFinalLayout == GFX_TEXTURE_LAYOUT_UNDEFINED) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate extent
        if (descriptor->extent.width == 0 || descriptor->extent.height == 0 || descriptor->extent.depth == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    // Blits and mipmaps use transfer commands on Vulkan but render on WebGPU, so they need the usage of both
    GfxTextureInfo textureInfo(GfxTexture texture)
    {
        return converter::vkTextureInfoToGfxTextureInfo(converter::toNative<core::Texture>(texture)->getInfo());
    }

    bool hasBlitSourceUsage(const GfxTextureInfo& info)
    {
        const GfxTextureUsageFlags required = GFX_TEXTURE_USAGE_COPY_SRC | GFX_TEXTURE_USAGE_TEXTURE_BINDING;
        return (info.usage & required) == required;
    }

    bool hasBlitDestinationUsage(const GfxTextureInfo& info)
    {
        const GfxTextureUsageFlags required = GFX_TEXTURE_USAGE_COPY_DST | GFX_TEXTURE_USAGE_RENDER_ATTACHMENT;
        return (info.usage & required) == required;
    }

    GfxResult validateBlitTextureToTextureDescriptor(const GfxBlitTextureToTextureDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate source and destination
        if (!descriptor->source || !descriptor->destination) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (descriptor->sourceFinalLayout == GFX_TEXTURE_LAYOUT_UNDEFINED || descriptor->destinationFinalLayout == GFX_TEXTURE_LAYOUT_UNDEFINED) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        if (!hasBlitSourceUsage(textureInfo(descriptor->source)) || !hasBlitDestinationUsage(textureInfo(descriptor->destination))) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate extents
        if (descriptor->sourceExtent.width == 0 || descriptor->sourceExtent.height == 0 || descriptor->sourceExtent.depth == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        if (descriptor->destinationExtent.width == 0 || descriptor->destinationExtent.height == 0 || descriptor->destinationExtent.depth == 0) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

    GfxResult validatePipelineBarrierDescriptor(const GfxPipelineBarrierDescriptor* descriptor)
    {
        if (!descriptor) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate memory barriers
        if (descriptor->memoryBarrierCount > 0 && !descriptor->memoryBarriers) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate buffer barriers
        if (descriptor->bufferBarrierCount > 0 && !descriptor->bufferBarriers) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        // Validate texture barriers
        if (descriptor->textureBarrierCount > 0 && !descriptor->textureBarriers) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }

        return GFX_RESULT_SUCCESS;
    }

} // anonymous namespace

// ============================================================================
// Combined validation functions (parameters + descriptors)
// ============================================================================

GfxResult validateCreateInstance(const GfxInstanceDescriptor* descriptor, GfxInstance* outInstance)
{
    if (!outInstance) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateInstanceDescriptor(descriptor);
}

GfxResult validateInstanceRequestAdapter(GfxInstance instance, const GfxAdapterDescriptor* descriptor, GfxAdapter* outAdapter)
{
    if (!instance || !outAdapter) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateAdapterDescriptor(descriptor);
}

GfxResult validateInstanceEnumerateAdapters(GfxInstance instance, uint32_t* adapterCount)
{
    if (!instance || !adapterCount) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateInstanceGetNativeHandle(GfxInstance instance, void** outHandle)
{
    if (!instance || !outHandle) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateEnumerateInstanceExtensions(uint32_t* extensionCount)
{
    if (!extensionCount) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateAdapterCreateDevice(GfxAdapter adapter, const GfxDeviceDescriptor* descriptor, GfxDevice* outDevice)
{
    if (!adapter || !outDevice) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }

    GfxResult descriptorResult = validateDeviceDescriptor(descriptor);
    if (descriptorResult != GFX_RESULT_SUCCESS) {
        return descriptorResult;
    }

    // Validate each requested queue family/index against the physical device's actual queue families.
    // Without this, an out-of-range family index would flow straight into vkCreateDevice.
    if (descriptor != nullptr && descriptor->queueRequests != nullptr) {
        const auto queueFamilies = converter::toNative<core::Adapter>(adapter)->getQueueFamilyProperties();
        for (uint32_t i = 0; i < descriptor->queueRequestCount; ++i) {
            const GfxQueueRequest& request = descriptor->queueRequests[i];
            if (request.queueFamilyIndex >= queueFamilies.size()) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }
            if (request.queueIndex >= queueFamilies[request.queueFamilyIndex].queueCount) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }
        }
    }

    // A device can only enable the extensions its adapter supports
    if (descriptor != nullptr && descriptor->enabledExtensions != nullptr) {
        const std::vector<const char*> supportedExtensions = converter::toNative<core::Adapter>(adapter)->enumerateSupportedExtensions();
        for (uint32_t i = 0; i < descriptor->enabledExtensionCount; ++i) {
            if (!gfx::common::containsString(supportedExtensions, descriptor->enabledExtensions[i])) {
                return GFX_RESULT_ERROR_FEATURE_NOT_SUPPORTED;
            }
        }
    }

    return GFX_RESULT_SUCCESS;
}

GfxResult validateAdapterGetNativeHandle(GfxAdapter adapter, void** outHandle)
{
    if (!adapter || !outHandle) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateAdapterGetInfo(GfxAdapter adapter, GfxAdapterInfo* outInfo)
{
    if (!adapter || !outInfo) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateAdapterGetLimits(GfxAdapter adapter, GfxDeviceLimits* outLimits)
{
    if (!adapter || !outLimits) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateAdapterEnumerateQueueFamilies(GfxAdapter adapter, uint32_t* queueFamilyCount)
{
    if (!adapter || !queueFamilyCount) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateAdapterGetQueueFamilySurfaceSupport(GfxAdapter adapter, GfxSurface surface, bool* outSupported)
{
    if (!adapter || !surface || !outSupported) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateAdapterEnumerateExtensions(GfxAdapter adapter, uint32_t* extensionCount)
{
    if (!adapter || !extensionCount) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceGetNativeHandle(GfxDevice device, void** outHandle)
{
    if (!device || !outHandle) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceSupportsShaderFormat(GfxDevice device, bool* outSupported)
{
    if (!device || !outSupported) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceGetQueue(GfxDevice device, GfxQueue* outQueue)
{
    if (!device || !outQueue) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceGetQueueByIndex(GfxDevice device, GfxQueue* outQueue)
{
    if (!device || !outQueue) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateInstanceCreateSurface(GfxInstance instance, const GfxSurfaceDescriptor* descriptor, GfxSurface* outSurface)
{
    if (!instance || !descriptor || !outSurface) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceCreateSwapchain(GfxDevice device, const GfxSwapchainDescriptor* descriptor, GfxSwapchain* outSwapchain)
{
    if (!device || !descriptor || !outSwapchain) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateSwapchainDescriptor(descriptor);
}

GfxResult validateDeviceCreateBuffer(GfxDevice device, const GfxBufferDescriptor* descriptor, GfxBuffer* outBuffer)
{
    if (!device || !descriptor || !outBuffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateBufferDescriptor(descriptor);
}

GfxResult validateDeviceImportBuffer(GfxDevice device, const GfxBufferImportDescriptor* descriptor, GfxBuffer* outBuffer)
{
    if (!device || !outBuffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateBufferImportDescriptor(descriptor);
}

GfxResult validateDeviceCreateTexture(GfxDevice device, const GfxTextureDescriptor* descriptor, GfxTexture* outTexture)
{
    if (!device || !descriptor || !outTexture) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }

    const auto* dev = converter::toNative<core::Device>(device);
    GfxResult formatResult = validateFormatEnabled(dev, descriptor->format);
    if (formatResult != GFX_RESULT_SUCCESS) {
        return formatResult;
    }

    return validateTextureDescriptor(descriptor);
}

GfxResult validateDeviceImportTexture(GfxDevice device, const GfxTextureImportDescriptor* descriptor, GfxTexture* outTexture)
{
    if (!device || !outTexture) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateTextureImportDescriptor(descriptor);
}

GfxResult validateDeviceCreateSampler(GfxDevice device, const GfxSamplerDescriptor* descriptor, GfxSampler* outSampler)
{
    if (!device || !descriptor || !outSampler) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateSamplerDescriptor(descriptor);
}

GfxResult validateDeviceCreateShader(GfxDevice device, const GfxShaderDescriptor* descriptor, GfxShader* outShader)
{
    if (!device || !descriptor || !outShader) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateShaderDescriptor(descriptor);
}

GfxResult validateDeviceCreateBindGroupLayout(GfxDevice device, const GfxBindGroupLayoutDescriptor* descriptor, GfxBindGroupLayout* outLayout)
{
    if (!device || !descriptor || !outLayout) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateBindGroupLayoutDescriptor(descriptor);
}

GfxResult validateDeviceCreateBindGroup(GfxDevice device, const GfxBindGroupDescriptor* descriptor, GfxBindGroup* outBindGroup)
{
    if (!device || !descriptor || !outBindGroup) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }

    GfxResult descriptorResult = validateBindGroupDescriptor(descriptor);
    if (descriptorResult != GFX_RESULT_SUCCESS) {
        return descriptorResult;
    }

    // WebGPU rule: every layout binding, and each element of an array binding, needs an entry
    for (const core::BindGroupLayoutEntry& layoutEntry : converter::toNative<core::BindGroupLayout>(descriptor->layout)->getEntries()) {
        for (uint32_t arrayElement = 0; arrayElement < layoutEntry.descriptorCount; ++arrayElement) {
            if (!hasBindGroupEntry(descriptor, layoutEntry.binding, arrayElement)) {
                return GFX_RESULT_ERROR_INVALID_ARGUMENT;
            }
        }
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceCreateRenderPipeline(GfxDevice device, const GfxRenderPipelineDescriptor* descriptor, GfxRenderPipeline* outPipeline)
{
    if (!device || !descriptor || !outPipeline) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }

    const auto* dev = converter::toNative<core::Device>(device);

    // Non-solid fill modes require GFX_DEVICE_EXTENSION_NON_SOLID_FILL to be enabled
    if (descriptor->primitive && descriptor->primitive->polygonMode != GFX_POLYGON_MODE_FILL) {
        if (!dev->isExtensionEnabled(core::DeviceExtension::NonSolidFill)) {
            return GFX_RESULT_ERROR_FEATURE_NOT_SUPPORTED;
        }
    }

    GfxResult descriptorResult = validateRenderPipelineDescriptor(descriptor);
    if (descriptorResult != GFX_RESULT_SUCCESS) {
        return descriptorResult;
    }

    if (descriptor->fragment) {
        for (uint32_t i = 0; i < descriptor->fragment->targetCount; ++i) {
            GfxResult formatResult = validateFormatEnabled(dev, descriptor->fragment->targets[i].format);
            if (formatResult != GFX_RESULT_SUCCESS) {
                return formatResult;
            }
        }
    }
    if (descriptor->depthStencil) {
        return validateFormatEnabled(dev, descriptor->depthStencil->format);
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceCreateComputePipeline(GfxDevice device, const GfxComputePipelineDescriptor* descriptor, GfxComputePipeline* outPipeline)
{
    if (!device || !descriptor || !outPipeline) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateComputePipelineDescriptor(descriptor);
}

GfxResult validateDeviceCreateRenderPass(GfxDevice device, const GfxRenderPassDescriptor* descriptor, GfxRenderPass* outRenderPass)
{
    if (!device || !descriptor || !outRenderPass) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }

    GfxResult descriptorResult = validateRenderPassDescriptor(descriptor);
    if (descriptorResult != GFX_RESULT_SUCCESS) {
        return descriptorResult;
    }

    const auto* dev = converter::toNative<core::Device>(device);
    for (uint32_t i = 0; i < descriptor->colorAttachmentCount; ++i) {
        GfxResult formatResult = validateFormatEnabled(dev, descriptor->colorAttachments[i].target.format);
        if (formatResult != GFX_RESULT_SUCCESS) {
            return formatResult;
        }
    }
    if (descriptor->depthStencilAttachment) {
        return validateFormatEnabled(dev, descriptor->depthStencilAttachment->target.format);
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceCreateFramebuffer(GfxDevice device, const GfxFramebufferDescriptor* descriptor, GfxFramebuffer* outFramebuffer)
{
    if (!device || !descriptor || !outFramebuffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateFramebufferDescriptor(descriptor);
}

GfxResult validateDeviceCreateCommandEncoder(GfxDevice device, const GfxCommandEncoderDescriptor* descriptor, GfxCommandEncoder* outEncoder)
{
    if (!device || !descriptor || !outEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (descriptor->queue && converter::toNative<core::Queue>(descriptor->queue)->device() != converter::toNative<core::Device>(device)->handle()) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceCreateRenderBundleCommandEncoder(GfxDevice device, const GfxRenderBundleEncoderDescriptor* descriptor, GfxCommandEncoder* outEncoder)
{
    if (!device || !descriptor || !outEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceCreateFence(GfxDevice device, const GfxFenceDescriptor* descriptor, GfxFence* outFence)
{
    if (!device || !outFence) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateFenceDescriptor(descriptor);
}

GfxResult validateDeviceCreateSemaphore(GfxDevice device, const GfxSemaphoreDescriptor* descriptor, GfxSemaphore* outSemaphore)
{
    if (!device || !outSemaphore) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateSemaphoreDescriptor(descriptor);
}

GfxResult validateDeviceCreateQuerySet(GfxDevice device, const GfxQuerySetDescriptor* descriptor, GfxQuerySet* outQuerySet)
{
    if (!device || !descriptor || !outQuerySet) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (descriptor->count == 0) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (descriptor->type == GFX_QUERY_TYPE_TIMESTAMP) {
        const auto* dev = converter::toNative<core::Device>(device);
        if (!dev->isExtensionEnabled(core::DeviceExtension::TimestampQuery)) {
            return GFX_RESULT_ERROR_FEATURE_NOT_SUPPORTED;
        }
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceGetLimits(GfxDevice device, GfxDeviceLimits* outLimits)
{
    if (!device || !outLimits) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSurfaceGetInfo(GfxSurface surface, GfxSurfaceInfo* outInfo)
{
    if (!surface || !outInfo) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSurfaceEnumerateSupportedFormats(GfxSurface surface, uint32_t* formatCount)
{
    if (!surface || !formatCount) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSurfaceEnumerateSupportedPresentModes(GfxSurface surface, uint32_t* presentModeCount)
{
    if (!surface || !presentModeCount) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSwapchainGetInfo(GfxSwapchain swapchain, GfxSwapchainInfo* outInfo)
{
    if (!swapchain || !outInfo) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSwapchainAcquireNextImage(GfxSwapchain swapchain, uint32_t* outImageIndex)
{
    if (!swapchain || !outImageIndex) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSwapchainGetTextureView(GfxSwapchain swapchain, GfxTextureView* outView)
{
    if (!swapchain || !outView) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSwapchainGetCurrentTextureView(GfxSwapchain swapchain, GfxTextureView* outView)
{
    if (!swapchain || !outView) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSwapchainPresent(GfxSwapchain swapchain, const GfxPresentDescriptor* presentDescriptor)
{
    if (!swapchain) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    // presentDescriptor is optional, but if provided, validate its sType
    if (presentDescriptor && presentDescriptor->sType != GFX_STRUCTURE_TYPE_PRESENT_DESCRIPTOR) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateBufferGetInfo(GfxBuffer buffer, GfxBufferInfo* outInfo)
{
    if (!buffer || !outInfo) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateBufferGetNativeHandle(GfxBuffer buffer, void** outHandle)
{
    if (!buffer || !outHandle) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateBufferMap(GfxBuffer buffer, void** outMappedPointer)
{
    if (!buffer || !outMappedPointer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateTextureGetInfo(GfxTexture texture, GfxTextureInfo* outInfo)
{
    if (!texture || !outInfo) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateTextureGetNativeHandle(GfxTexture texture, void** outHandle)
{
    if (!texture || !outHandle) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateTextureGetLayout(GfxTexture texture, GfxTextureLayout* outLayout)
{
    if (!texture || !outLayout) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateTextureCreateView(GfxTexture texture, const GfxTextureViewDescriptor* descriptor, GfxTextureView* outView)
{
    if (!texture || !outView) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }

    GfxResult descriptorResult = validateTextureViewDescriptor(descriptor);
    if (descriptorResult != GFX_RESULT_SUCCESS) {
        return descriptorResult;
    }

    const VkFormat format = converter::toNative<core::Texture>(texture)->getFormat();

    // WebGPU rule: a view keeps its texture's format, as textures declare no other view formats
    if (converter::gfxFormatToVkFormat(descriptor->format) != format) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }

    // A single-aspect view needs that aspect in the texture's format
    if ((descriptor->aspect == GFX_TEXTURE_ASPECT_DEPTH_ONLY && !core::isDepthFormat(format)) || (descriptor->aspect == GFX_TEXTURE_ASPECT_STENCIL_ONLY && !core::hasStencilComponent(format))) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }

    return GFX_RESULT_SUCCESS;
}

GfxResult validateQueueGetInfo(GfxQueue queue, GfxQueueInfo* outInfo)
{
    if (!queue || !outInfo) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateQueueGetNativeHandle(GfxQueue queue, void** outHandle)
{
    if (!queue || !outHandle) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateQueueSubmit(GfxQueue queue, const GfxSubmitDescriptor* submitDescriptor)
{
    if (!queue || !submitDescriptor) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (submitDescriptor->waitSemaphoreCount > 0 && (!submitDescriptor->waitSemaphores || !submitDescriptor->waitStages)) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if ((submitDescriptor->commandEncoderCount > 0 && !submitDescriptor->commandEncoders) || (submitDescriptor->signalSemaphoreCount > 0 && !submitDescriptor->signalSemaphores)) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    const uint32_t queueFamily = converter::toNative<core::Queue>(queue)->family();
    for (uint32_t i = 0; i < submitDescriptor->commandEncoderCount; ++i) {
        GfxCommandEncoder encoder = submitDescriptor->commandEncoders[i];
        if (!encoder) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
        // A render bundle is executed inside a pass, never submitted
        const auto* nativeEncoder = converter::toNative<core::CommandEncoder>(encoder);
        if (nativeEncoder->isBundleEncoder() || nativeEncoder->queueFamily() != queueFamily) {
            return GFX_RESULT_ERROR_INVALID_ARGUMENT;
        }
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateQueueWriteBuffer(GfxQueue queue, GfxBuffer buffer, const void* data)
{
    if (!queue || !buffer || !data) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (converter::toNative<core::Buffer>(buffer)->isMapped()) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateQueueWriteTexture(GfxQueue queue, const GfxWriteTextureDescriptor* descriptor, const void* data)
{
    if (!queue || !descriptor || !descriptor->texture || !data) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (descriptor->finalLayout == GFX_TEXTURE_LAYOUT_UNDEFINED) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (descriptor->extent.width == 0 || descriptor->extent.height == 0 || descriptor->extent.depth == 0) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateCopyAspectAndRowPitch(descriptor->texture, descriptor->aspect, descriptor->bytesPerRow);
}

GfxResult validateCommandEncoderBeginRenderPass(GfxCommandEncoder commandEncoder, const GfxRenderPassBeginDescriptor* beginDescriptor, GfxRenderPassEncoder* outRenderPass)
{
    if (!commandEncoder || !outRenderPass) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    bool isBundle = converter::toNative<core::CommandEncoder>(commandEncoder)->isBundleEncoder();
    return validateRenderPassBeginDescriptor(beginDescriptor, isBundle);
}

GfxResult validateCommandEncoderBeginComputePass(GfxCommandEncoder commandEncoder, const GfxComputePassBeginDescriptor* beginDescriptor, GfxComputePassEncoder* outComputePass)
{
    if (!commandEncoder || !outComputePass) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateComputePassBeginDescriptor(beginDescriptor);
}

GfxResult validateCommandEncoderCopyBufferToBuffer(GfxCommandEncoder commandEncoder, const GfxCopyBufferToBufferDescriptor* descriptor)
{
    if (!commandEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateCopyBufferToBufferDescriptor(descriptor);
}

GfxResult validateCommandEncoderCopyBufferToTexture(GfxCommandEncoder commandEncoder, const GfxCopyBufferToTextureDescriptor* descriptor)
{
    if (!commandEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateCopyBufferToTextureDescriptor(descriptor);
}

GfxResult validateCommandEncoderCopyTextureToBuffer(GfxCommandEncoder commandEncoder, const GfxCopyTextureToBufferDescriptor* descriptor)
{
    if (!commandEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateCopyTextureToBufferDescriptor(descriptor);
}

GfxResult validateCommandEncoderCopyTextureToTexture(GfxCommandEncoder commandEncoder, const GfxCopyTextureToTextureDescriptor* descriptor)
{
    if (!commandEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateCopyTextureToTextureDescriptor(descriptor);
}

GfxResult validateCommandEncoderBlitTextureToTexture(GfxCommandEncoder commandEncoder, const GfxBlitTextureToTextureDescriptor* descriptor)
{
    if (!commandEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validateBlitTextureToTextureDescriptor(descriptor);
}

GfxResult validateCommandEncoderPipelineBarrier(GfxCommandEncoder commandEncoder, const GfxPipelineBarrierDescriptor* descriptor)
{
    if (!commandEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return validatePipelineBarrierDescriptor(descriptor);
}

GfxResult validateCommandEncoderGenerateMipmaps(GfxCommandEncoder commandEncoder, GfxTexture texture)
{
    if (!commandEncoder || !texture) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    // Each level is blitted from the one above, so the texture is both blit source and destination
    const GfxTextureInfo info = textureInfo(texture);
    if (!hasBlitSourceUsage(info) || !hasBlitDestinationUsage(info)) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateCommandEncoderGenerateMipmapsRange(GfxCommandEncoder commandEncoder, GfxTexture texture)
{
    if (!commandEncoder || !texture) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    // Each level is blitted from the one above, so the texture is both blit source and destination
    const GfxTextureInfo info = textureInfo(texture);
    if (!hasBlitSourceUsage(info) || !hasBlitDestinationUsage(info)) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateCommandEncoderWriteTimestamp(GfxCommandEncoder commandEncoder, GfxQuerySet querySet)
{
    if (!commandEncoder || !querySet) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateCommandEncoderResetQuerySet(GfxCommandEncoder commandEncoder, GfxQuerySet querySet)
{
    if (!commandEncoder || !querySet) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateCommandEncoderResolveQuerySet(GfxCommandEncoder commandEncoder, GfxQuerySet querySet, GfxBuffer destinationBuffer, uint64_t destinationOffset)
{
    if (!commandEncoder || !querySet || !destinationBuffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    // WebGPU rule: query results resolve to a 256-byte aligned buffer offset
    if (destinationOffset % 256 != 0) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    const auto* buf = converter::toNative<core::Buffer>(destinationBuffer);
    if (!(buf->getInfo().originalUsage & GFX_BUFFER_USAGE_QUERY_RESOLVE)) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderSetPipeline(GfxRenderPassEncoder renderPassEncoder, GfxRenderPipeline pipeline)
{
    if (!renderPassEncoder || !pipeline) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderSetBindGroup(GfxRenderPassEncoder renderPassEncoder, GfxBindGroup bindGroup)
{
    if (!renderPassEncoder || !bindGroup) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderSetVertexBuffer(GfxRenderPassEncoder renderPassEncoder, GfxBuffer buffer)
{
    if (!renderPassEncoder || !buffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderSetIndexBuffer(GfxRenderPassEncoder renderPassEncoder, GfxBuffer buffer)
{
    if (!renderPassEncoder || !buffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderSetViewport(GfxRenderPassEncoder renderPassEncoder, const GfxViewport* viewport)
{
    if (!renderPassEncoder || !viewport) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderSetScissorRect(GfxRenderPassEncoder renderPassEncoder, const GfxScissorRect* scissor)
{
    if (!renderPassEncoder || !scissor) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderSetBlendConstant(GfxRenderPassEncoder renderPassEncoder, const GfxColor* color)
{
    if (!renderPassEncoder || !color) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderSetStencilReference(GfxRenderPassEncoder renderPassEncoder)
{
    if (!renderPassEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderDrawIndirect(GfxRenderPassEncoder renderPassEncoder, GfxBuffer indirectBuffer)
{
    if (!renderPassEncoder || !indirectBuffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderDrawIndexedIndirect(GfxRenderPassEncoder renderPassEncoder, GfxBuffer indirectBuffer)
{
    if (!renderPassEncoder || !indirectBuffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderBeginOcclusionQuery(GfxRenderPassEncoder renderPassEncoder, GfxQuerySet querySet)
{
    if (!renderPassEncoder || !querySet) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    const auto* encoder = converter::toNative<core::RenderPassEncoder>(renderPassEncoder);
    const auto* query = converter::toNative<core::QuerySet>(querySet);
    if (!encoder->isOcclusionQueryPoolCompatible(query->handle())) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderEndOcclusionQuery(GfxRenderPassEncoder renderPassEncoder)
{
    if (!renderPassEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateComputePassEncoderSetPipeline(GfxComputePassEncoder computePassEncoder, GfxComputePipeline pipeline)
{
    if (!computePassEncoder || !pipeline) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateComputePassEncoderSetBindGroup(GfxComputePassEncoder computePassEncoder, GfxBindGroup bindGroup)
{
    if (!computePassEncoder || !bindGroup) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateComputePassEncoderDispatchIndirect(GfxComputePassEncoder computePassEncoder, GfxBuffer indirectBuffer)
{
    if (!computePassEncoder || !indirectBuffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateFenceGetStatus(GfxFence fence, bool* isSignaled)
{
    if (!fence || !isSignaled) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSemaphoreGetType(GfxSemaphore semaphore, GfxSemaphoreType* outType)
{
    if (!semaphore || !outType) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSemaphoreGetValue(GfxSemaphore semaphore, uint64_t* outValue)
{
    if (!semaphore || !outValue) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

// ============================================================================
// Simple validation functions (destroy, wait, etc.)
// ============================================================================

GfxResult validateInstanceDestroy(GfxInstance instance)
{
    if (!instance) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateAdapterDestroy(GfxAdapter adapter)
{
    if (!adapter) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceDestroy(GfxDevice device)
{
    if (!device) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateDeviceWaitIdle(GfxDevice device)
{
    if (!device) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSurfaceDestroy(GfxSurface surface)
{
    if (!surface) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSwapchainDestroy(GfxSwapchain swapchain)
{
    if (!swapchain) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateBufferDestroy(GfxBuffer buffer)
{
    if (!buffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateBufferUnmap(GfxBuffer buffer)
{
    if (!buffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateBufferFlushMappedRange(GfxBuffer buffer)
{
    if (!buffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateBufferInvalidateMappedRange(GfxBuffer buffer)
{
    if (!buffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateTextureDestroy(GfxTexture texture)
{
    if (!texture) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateTextureViewDestroy(GfxTextureView textureView)
{
    if (!textureView) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSamplerDestroy(GfxSampler sampler)
{
    if (!sampler) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateShaderDestroy(GfxShader shader)
{
    if (!shader) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateBindGroupLayoutDestroy(GfxBindGroupLayout bindGroupLayout)
{
    if (!bindGroupLayout) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateBindGroupDestroy(GfxBindGroup bindGroup)
{
    if (!bindGroup) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPipelineDestroy(GfxRenderPipeline renderPipeline)
{
    if (!renderPipeline) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateComputePipelineDestroy(GfxComputePipeline computePipeline)
{
    if (!computePipeline) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassDestroy(GfxRenderPass renderPass)
{
    if (!renderPass) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateFramebufferDestroy(GfxFramebuffer framebuffer)
{
    if (!framebuffer) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateQuerySetDestroy(GfxQuerySet querySet)
{
    if (!querySet) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateQueueWaitIdle(GfxQueue queue)
{
    if (!queue) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateCommandEncoderDestroy(GfxCommandEncoder commandEncoder)
{
    if (!commandEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateCommandEncoderEnd(GfxCommandEncoder commandEncoder)
{
    if (!commandEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateCommandEncoderBegin(GfxCommandEncoder commandEncoder)
{
    if (!commandEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderDraw(GfxRenderPassEncoder renderPassEncoder)
{
    if (!renderPassEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderDrawIndexed(GfxRenderPassEncoder renderPassEncoder)
{
    if (!renderPassEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderEnd(GfxRenderPassEncoder renderPassEncoder)
{
    if (!renderPassEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateRenderPassEncoderExecuteBundles(GfxRenderPassEncoder renderPassEncoder, const GfxCommandEncoder* bundleEncoders, uint32_t bundleCount)
{
    if (!renderPassEncoder || !bundleEncoders || bundleCount == 0) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateComputePassEncoderDispatch(GfxComputePassEncoder computePassEncoder)
{
    if (!computePassEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateComputePassEncoderEnd(GfxComputePassEncoder computePassEncoder)
{
    if (!computePassEncoder) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateFenceDestroy(GfxFence fence)
{
    if (!fence) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateFenceWait(GfxFence fence)
{
    if (!fence) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateFenceReset(GfxFence fence)
{
    if (!fence) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSemaphoreDestroy(GfxSemaphore semaphore)
{
    if (!semaphore) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSemaphoreSignal(GfxSemaphore semaphore)
{
    if (!semaphore) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

GfxResult validateSemaphoreWait(GfxSemaphore semaphore)
{
    if (!semaphore) {
        return GFX_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return GFX_RESULT_SUCCESS;
}

} // namespace gfx::backend::vulkan::validator
