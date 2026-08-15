#include "Conversions.h"

#include "../core/command/CommandEncoder.h"
#include "../core/command/ComputePassEncoder.h"
#include "../core/command/RenderPassEncoder.h"
#include "../core/compute/ComputePipeline.h"
#include "../core/presentation/Surface.h"
#include "../core/presentation/Swapchain.h"
#include "../core/query/QuerySet.h"
#include "../core/render/Framebuffer.h"
#include "../core/render/RenderPass.h"
#include "../core/render/RenderPipeline.h"
#include "../core/resource/BindGroup.h"
#include "../core/resource/BindGroupLayout.h"
#include "../core/resource/Buffer.h"
#include "../core/resource/Sampler.h"
#include "../core/resource/Shader.h"
#include "../core/resource/Texture.h"
#include "../core/resource/TextureView.h"
#include "../core/sync/Fence.h"
#include "../core/sync/Semaphore.h"
#include "../core/system/Adapter.h"
#include "../core/system/Device.h"
#include "../core/system/Instance.h"
#include "../core/system/Queue.h"

#include <cstring>
#include <vector>

namespace gfx::backend::webgpu::converter {

// ============================================================================
// Extension Name Mapping
// ============================================================================

const char* instanceExtensionNameToGfx(const char* internalName)
{
    if (std::strcmp(internalName, core::extensions::SURFACE) == 0) {
        return GFX_INSTANCE_EXTENSION_SURFACE;
    }
    if (std::strcmp(internalName, core::extensions::DEBUG) == 0) {
        return GFX_INSTANCE_EXTENSION_DEBUG;
    }
    // Unknown extension - return as-is
    return internalName;
}

const char* deviceExtensionNameToGfx(const char* internalName)
{
    if (std::strcmp(internalName, core::extensions::SWAPCHAIN) == 0) {
        return GFX_DEVICE_EXTENSION_SWAPCHAIN;
    }
    if (std::strcmp(internalName, core::extensions::TIMELINE_SEMAPHORE) == 0) {
        return GFX_DEVICE_EXTENSION_TIMELINE_SEMAPHORE;
    }
    if (std::strcmp(internalName, core::extensions::TIMESTAMP_QUERY) == 0) {
        return GFX_DEVICE_EXTENSION_TIMESTAMP_QUERY;
    }
    // Unknown extension - return as-is
    return internalName;
}

using namespace core;

// ============================================================================
// Device Limits Conversion
// ============================================================================

GfxDeviceLimits wgpuLimitsToGfxDeviceLimits(const WGPULimits& limits)
{
    GfxDeviceLimits gfxLimits{};
    gfxLimits.minUniformBufferOffsetAlignment = limits.minUniformBufferOffsetAlignment;
    gfxLimits.minStorageBufferOffsetAlignment = limits.minStorageBufferOffsetAlignment;
    gfxLimits.maxUniformBufferBindingSize = static_cast<uint32_t>(limits.maxUniformBufferBindingSize);
    gfxLimits.maxStorageBufferBindingSize = static_cast<uint32_t>(limits.maxStorageBufferBindingSize);
    gfxLimits.maxBufferSize = limits.maxBufferSize;
    gfxLimits.maxTextureDimension1D = limits.maxTextureDimension1D;
    gfxLimits.maxTextureDimension2D = limits.maxTextureDimension2D;
    gfxLimits.maxTextureDimension3D = limits.maxTextureDimension3D;
    gfxLimits.maxTextureArrayLayers = limits.maxTextureArrayLayers;
    gfxLimits.maxBindGroups = limits.maxBindGroups;
    gfxLimits.maxColorAttachments = limits.maxColorAttachments;
    gfxLimits.maxVertexAttributes = limits.maxVertexAttributes;
    gfxLimits.maxVertexBuffers = limits.maxVertexBuffers;
    gfxLimits.maxVertexBufferArrayStride = limits.maxVertexBufferArrayStride;
    gfxLimits.maxSamplerAnisotropy = 16; // Fixed by the WebGPU specification
    gfxLimits.maxComputeWorkgroupSizeX = limits.maxComputeWorkgroupSizeX;
    gfxLimits.maxComputeWorkgroupSizeY = limits.maxComputeWorkgroupSizeY;
    gfxLimits.maxComputeWorkgroupSizeZ = limits.maxComputeWorkgroupSizeZ;
    gfxLimits.maxComputeInvocationsPerWorkgroup = limits.maxComputeInvocationsPerWorkgroup;
    gfxLimits.maxComputeWorkgroupsPerDimension = limits.maxComputeWorkgroupsPerDimension;
    gfxLimits.maxComputeWorkgroupStorageSize = limits.maxComputeWorkgroupStorageSize;
    gfxLimits.timestampPeriod = 1.0f; // WebGPU timestamp query values are already in nanoseconds
    return gfxLimits;
}

// ============================================================================
// Type Conversion Functions
// ============================================================================

core::SemaphoreType gfxSemaphoreTypeToWebGPUSemaphoreType(GfxSemaphoreType gfxType)
{
    switch (gfxType) {
    case GFX_SEMAPHORE_TYPE_BINARY:
        return core::SemaphoreType::Binary;
    case GFX_SEMAPHORE_TYPE_TIMELINE:
        return core::SemaphoreType::Timeline;
    default:
        return core::SemaphoreType::Binary;
    }
}

core::ShaderSourceType gfxShaderSourceTypeToWebGPUShaderSourceType(GfxShaderSourceType type)
{
    switch (type) {
    case GFX_SHADER_SOURCE_WGSL:
        return core::ShaderSourceType::WGSL;
    case GFX_SHADER_SOURCE_SPIRV:
        return core::ShaderSourceType::SPIRV;
    default:
        return core::ShaderSourceType::WGSL; // WebGPU defaults to WGSL
    }
}

WGPUQueryType gfxQueryTypeToWebGPUQueryType(GfxQueryType type)
{
    switch (type) {
    case GFX_QUERY_TYPE_OCCLUSION:
        return WGPUQueryType_Occlusion;
    case GFX_QUERY_TYPE_TIMESTAMP:
        return WGPUQueryType_Timestamp;
    default:
        return WGPUQueryType_Occlusion;
    }
}

// ============================================================================
// Adapter Type Conversion
// ============================================================================

GfxAdapterType wgpuAdapterTypeToGfxAdapterType(WGPUAdapterType adapterType)
{
    switch (adapterType) {
    case WGPUAdapterType_DiscreteGPU:
        return GFX_ADAPTER_TYPE_DISCRETE_GPU;
    case WGPUAdapterType_IntegratedGPU:
        return GFX_ADAPTER_TYPE_INTEGRATED_GPU;
    case WGPUAdapterType_CPU:
        return GFX_ADAPTER_TYPE_CPU;
    case WGPUAdapterType_Unknown:
    default:
        return GFX_ADAPTER_TYPE_UNKNOWN;
    }
}

// ============================================================================
// Adapter Info Conversion
// ============================================================================

GfxAdapterInfo wgpuAdapterToGfxAdapterInfo(const core::AdapterInfo& info)
{
    GfxAdapterInfo adapterInfo{};
    adapterInfo.name = info.name.c_str();
    adapterInfo.driverDescription = info.driverDescription.c_str();
    adapterInfo.vendorID = info.vendorID;
    adapterInfo.deviceID = info.deviceID;
    adapterInfo.backend = GFX_BACKEND_WEBGPU;
    adapterInfo.adapterType = wgpuAdapterTypeToGfxAdapterType(info.adapterType);
    return adapterInfo;
}

// ============================================================================
// Queue Family Conversion
// ============================================================================

GfxQueueFamilyProperties wgpuQueueFamilyPropertiesToGfx(const core::QueueFamilyProperties& props)
{
    // Build flags based on capabilities
    uint32_t flags = 0;
    if (props.supportsGraphics) {
        flags |= GFX_QUEUE_FLAG_GRAPHICS;
    }
    if (props.supportsCompute) {
        flags |= GFX_QUEUE_FLAG_COMPUTE;
    }
    if (props.supportsTransfer) {
        flags |= GFX_QUEUE_FLAG_TRANSFER;
    }

    GfxQueueFamilyProperties gfxProps = {};
    gfxProps.flags = static_cast<GfxQueueFlags>(flags);
    gfxProps.queueCount = props.queueCount;
    return gfxProps;
}

// ============================================================================
// CreateInfo Conversion Functions - GfxDescriptor to Internal CreateInfo
// ============================================================================

core::AdapterCreateInfo gfxDescriptorToWebGPUAdapterCreateInfo(const GfxAdapterDescriptor* descriptor)
{
    core::AdapterCreateInfo createInfo{};

    if (descriptor) {
        // UNDEFINED preference selects by index; otherwise the preference is used
        // (UINT32_MAX = preference-based selection internally)
        if (descriptor->preference == GFX_ADAPTER_PREFERENCE_UNDEFINED) {
            createInfo.adapterIndex = descriptor->adapterIndex;
            createInfo.powerPreference = WGPUPowerPreference_Undefined;
            createInfo.forceFallbackAdapter = false;
        } else {
            // Fall back to preference-based selection
            createInfo.adapterIndex = UINT32_MAX;
            switch (descriptor->preference) {
            case GFX_ADAPTER_PREFERENCE_LOW_POWER:
                createInfo.powerPreference = WGPUPowerPreference_LowPower;
                createInfo.forceFallbackAdapter = false;
                break;
            case GFX_ADAPTER_PREFERENCE_HIGH_PERFORMANCE:
                createInfo.powerPreference = WGPUPowerPreference_HighPerformance;
                createInfo.forceFallbackAdapter = false;
                break;
            case GFX_ADAPTER_PREFERENCE_SOFTWARE:
                createInfo.powerPreference = WGPUPowerPreference_Undefined;
                createInfo.forceFallbackAdapter = true;
                break;
            default:
                createInfo.powerPreference = WGPUPowerPreference_Undefined;
                createInfo.forceFallbackAdapter = false;
                break;
            }
        }
    } else {
        createInfo.powerPreference = WGPUPowerPreference_Undefined;
        createInfo.forceFallbackAdapter = false;
    }

    return createInfo;
}

core::InstanceCreateInfo gfxDescriptorToWebGPUInstanceCreateInfo(const GfxInstanceDescriptor* descriptor)
{
    core::InstanceCreateInfo createInfo{};

    if (descriptor) {
        createInfo.applicationName = descriptor->applicationName ? descriptor->applicationName : "Gfx Application";
        createInfo.applicationVersion = descriptor->applicationVersion;

        // Convert enabled extensions from const char** to std::vector<std::string>
        if (descriptor->enabledExtensions && descriptor->enabledExtensionCount > 0) {
            createInfo.enabledExtensions.reserve(descriptor->enabledExtensionCount);
            for (uint32_t i = 0; i < descriptor->enabledExtensionCount; ++i) {
                createInfo.enabledExtensions.push_back(descriptor->enabledExtensions[i]);
            }
        }
    } else {
        createInfo.applicationName = "Gfx Application";
        createInfo.applicationVersion = 1;
    }

    return createInfo;
}

core::DeviceCreateInfo gfxDescriptorToWebGPUDeviceCreateInfo(const GfxDeviceDescriptor* descriptor)
{
    core::DeviceCreateInfo createInfo{};

    if (descriptor) {
        // Convert enabled extensions from const char** to std::vector<std::string>
        if (descriptor->enabledExtensions && descriptor->enabledExtensionCount > 0) {
            createInfo.enabledExtensions.reserve(descriptor->enabledExtensionCount);
            for (uint32_t i = 0; i < descriptor->enabledExtensionCount; ++i) {
                createInfo.enabledExtensions.push_back(descriptor->enabledExtensions[i]);
            }
        }
    }

    return createInfo;
}

core::BufferCreateInfo gfxDescriptorToWebGPUBufferCreateInfo(const GfxBufferDescriptor* descriptor)
{
    core::BufferCreateInfo createInfo{};
    createInfo.size = descriptor->size;
    createInfo.usage = gfxBufferUsageToWGPU(descriptor->usage);
    createInfo.memoryProperties = descriptor->memoryProperties;
    return createInfo;
}

core::BufferImportInfo gfxExternalDescriptorToWebGPUBufferImportInfo(const GfxBufferImportDescriptor* descriptor)
{
    core::BufferImportInfo importInfo{};
    importInfo.size = descriptor->size;
    importInfo.usage = gfxBufferUsageToWGPU(descriptor->usage);
    importInfo.memoryProperties = 0; // External buffers have unknown/unspecified memory properties
    return importInfo;
}

core::TextureCreateInfo gfxDescriptorToWebGPUTextureCreateInfo(const GfxTextureDescriptor* descriptor)
{
    core::TextureCreateInfo createInfo{};
    createInfo.format = gfxFormatToWGPUFormat(descriptor->format);
    createInfo.size.width = descriptor->size.width;
    createInfo.size.height = descriptor->size.height;
    // For 3D textures, use depth; for 1D/2D textures, use arrayLayerCount
    createInfo.size.depthOrArrayLayers = (descriptor->type == GFX_TEXTURE_TYPE_3D)
        ? descriptor->size.depth
        : (descriptor->arrayLayerCount > 0 ? descriptor->arrayLayerCount : 1);
    createInfo.usage = gfxTextureUsageToWGPU(descriptor->usage);
    createInfo.sampleCount = descriptor->sampleCount;
    createInfo.mipLevelCount = descriptor->mipLevelCount;
    createInfo.dimension = gfxTextureTypeToWGPUTextureDimension(descriptor->type);
    createInfo.arrayLayers = descriptor->arrayLayerCount > 0 ? descriptor->arrayLayerCount : 1;
    return createInfo;
}

core::TextureImportInfo gfxExternalDescriptorToWebGPUTextureImportInfo(const GfxTextureImportDescriptor* descriptor)
{
    core::TextureImportInfo importInfo{};
    importInfo.format = gfxFormatToWGPUFormat(descriptor->format);
    importInfo.size.width = descriptor->size.width;
    importInfo.size.height = descriptor->size.height;
    // For 3D textures, use depth; for 1D/2D textures, use arrayLayerCount
    importInfo.size.depthOrArrayLayers = (descriptor->type == GFX_TEXTURE_TYPE_3D)
        ? descriptor->size.depth
        : (descriptor->arrayLayerCount > 0 ? descriptor->arrayLayerCount : 1);
    importInfo.usage = gfxTextureUsageToWGPU(descriptor->usage);
    importInfo.sampleCount = descriptor->sampleCount;
    importInfo.mipLevelCount = descriptor->mipLevelCount;
    importInfo.dimension = gfxTextureTypeToWGPUTextureDimension(descriptor->type);
    importInfo.arrayLayers = descriptor->arrayLayerCount > 0 ? descriptor->arrayLayerCount : 1;
    return importInfo;
}

core::TextureViewCreateInfo gfxDescriptorToWebGPUTextureViewCreateInfo(const GfxTextureViewDescriptor* descriptor)
{
    core::TextureViewCreateInfo createInfo{};
    createInfo.viewDimension = gfxTextureViewTypeToWGPU(descriptor->viewType);
    createInfo.format = gfxFormatToWGPUFormat(descriptor->format);
    createInfo.baseMipLevel = descriptor->baseMipLevel;
    createInfo.mipLevelCount = descriptor->mipLevelCount;
    createInfo.baseArrayLayer = descriptor->baseArrayLayer;
    createInfo.arrayLayerCount = descriptor->arrayLayerCount;
    return createInfo;
}

core::ShaderSourceType gfxShaderSourceTypeToWebGPU(GfxShaderSourceType sourceType)
{
    switch (sourceType) {
    case GFX_SHADER_SOURCE_SPIRV:
        return core::ShaderSourceType::SPIRV;
    case GFX_SHADER_SOURCE_WGSL:
    default:
        return core::ShaderSourceType::WGSL;
    }
}

core::ShaderCreateInfo gfxDescriptorToWebGPUShaderCreateInfo(const GfxShaderDescriptor* descriptor)
{
    core::ShaderCreateInfo createInfo{};
    createInfo.sourceType = gfxShaderSourceTypeToWebGPU(descriptor->sourceType);
    createInfo.code = descriptor->code;
    createInfo.codeSize = descriptor->codeSize;
    createInfo.entryPoint = descriptor->entryPoint;
    return createInfo;
}

core::SamplerCreateInfo gfxDescriptorToWebGPUSamplerCreateInfo(const GfxSamplerDescriptor* descriptor)
{
    core::SamplerCreateInfo createInfo{};
    createInfo.addressModeU = gfxAddressModeToWGPU(descriptor->addressModeU);
    createInfo.addressModeV = gfxAddressModeToWGPU(descriptor->addressModeV);
    createInfo.addressModeW = gfxAddressModeToWGPU(descriptor->addressModeW);
    createInfo.magFilter = gfxFilterModeToWGPU(descriptor->magFilter);
    createInfo.minFilter = gfxFilterModeToWGPU(descriptor->minFilter);
    createInfo.mipmapFilter = gfxMipmapFilterModeToWGPU(descriptor->mipmapFilter);
    createInfo.lodMinClamp = descriptor->lodMinClamp;
    createInfo.lodMaxClamp = descriptor->lodMaxClamp;
    createInfo.maxAnisotropy = descriptor->maxAnisotropy;
    createInfo.compareFunction = gfxCompareFunctionToWGPU(descriptor->compare);
    return createInfo;
}

core::SemaphoreCreateInfo gfxDescriptorToWebGPUSemaphoreCreateInfo(const GfxSemaphoreDescriptor* descriptor)
{
    core::SemaphoreCreateInfo createInfo{};

    if (descriptor) {
        createInfo.type = gfxSemaphoreTypeToWebGPUSemaphoreType(descriptor->type);
        createInfo.initialValue = descriptor->initialValue;
    } else {
        createInfo.type = core::SemaphoreType::Binary;
        createInfo.initialValue = 0;
    }

    return createInfo;
}

core::FenceCreateInfo gfxDescriptorToWebGPUFenceCreateInfo(const GfxFenceDescriptor* descriptor)
{
    core::FenceCreateInfo createInfo{};

    if (descriptor) {
        createInfo.signaled = descriptor->signaled;
    } else {
        createInfo.signaled = false;
    }

    return createInfo;
}

core::PlatformWindowHandle gfxWindowHandleToWebGPUPlatformWindowHandle(const GfxPlatformWindowHandle& gfxHandle)
{
    core::PlatformWindowHandle handle{};

    switch (gfxHandle.windowingSystem) {
    case GFX_WINDOWING_SYSTEM_XCB:
        handle.platform = core::PlatformWindowHandle::Platform::Xcb;
        handle.handle.xcb.connection = gfxHandle.handle.xcb.connection;
        handle.handle.xcb.window = gfxHandle.handle.xcb.window;
        break;
    case GFX_WINDOWING_SYSTEM_XLIB:
        handle.platform = core::PlatformWindowHandle::Platform::Xlib;
        handle.handle.xlib.display = gfxHandle.handle.xlib.display;
        handle.handle.xlib.window = gfxHandle.handle.xlib.window;
        break;
    case GFX_WINDOWING_SYSTEM_WAYLAND:
        handle.platform = core::PlatformWindowHandle::Platform::Wayland;
        handle.handle.wayland.display = gfxHandle.handle.wayland.display;
        handle.handle.wayland.surface = gfxHandle.handle.wayland.surface;
        break;
    case GFX_WINDOWING_SYSTEM_WIN32:
        handle.platform = core::PlatformWindowHandle::Platform::Win32;
        handle.handle.win32.hinstance = gfxHandle.handle.win32.hinstance;
        handle.handle.win32.hwnd = gfxHandle.handle.win32.hwnd;
        break;
    case GFX_WINDOWING_SYSTEM_METAL:
        handle.platform = core::PlatformWindowHandle::Platform::Metal;
        handle.handle.metal.layer = gfxHandle.handle.metal.layer;
        break;
    case GFX_WINDOWING_SYSTEM_EMSCRIPTEN:
        handle.platform = core::PlatformWindowHandle::Platform::Emscripten;
        handle.handle.emscripten.canvasSelector = gfxHandle.handle.emscripten.canvasSelector;
        break;
    case GFX_WINDOWING_SYSTEM_ANDROID:
        handle.platform = core::PlatformWindowHandle::Platform::Android;
        handle.handle.android.window = gfxHandle.handle.android.window;
        break;
    default:
        handle.platform = core::PlatformWindowHandle::Platform::Unknown;
        break;
    }

    return handle;
}

core::SurfaceCreateInfo gfxDescriptorToWebGPUSurfaceCreateInfo(const GfxSurfaceDescriptor* descriptor)
{
    core::SurfaceCreateInfo createInfo{};
    if (descriptor) {
        createInfo.windowHandle = gfxWindowHandleToWebGPUPlatformWindowHandle(descriptor->windowHandle);
    }
    return createInfo;
}

core::SwapchainCreateInfo gfxDescriptorToWebGPUSwapchainCreateInfo(const GfxSwapchainDescriptor* descriptor)
{
    core::SwapchainCreateInfo createInfo{};
    createInfo.width = descriptor->extent.width;
    createInfo.height = descriptor->extent.height;
    createInfo.format = gfxFormatToWGPUFormat(descriptor->format);
    createInfo.usage = gfxTextureUsageToWGPU(descriptor->usage);
    createInfo.presentMode = gfxPresentModeToWGPU(descriptor->presentMode);
    createInfo.imageCount = descriptor->imageCount;
    return createInfo;
}

namespace {
    WGPUShaderStage gfxShaderStageToWGPU(GfxShaderStageFlags visibility)
    {
        WGPUShaderStage flags = WGPUShaderStage_None;
        if (visibility & GFX_SHADER_STAGE_VERTEX) {
            flags |= WGPUShaderStage_Vertex;
        }
        if (visibility & GFX_SHADER_STAGE_FRAGMENT) {
            flags |= WGPUShaderStage_Fragment;
        }
        if (visibility & GFX_SHADER_STAGE_COMPUTE) {
            flags |= WGPUShaderStage_Compute;
        }
        return flags;
    }

    core::BindGroupLayoutEntry convertBindGroupLayoutEntry(const GfxBindGroupLayoutEntry& entry)
    {
        core::BindGroupLayoutEntry layoutEntry{};
        layoutEntry.binding = entry.binding;
        layoutEntry.count = entry.count > 0 ? entry.count : 1;
        layoutEntry.visibility = gfxShaderStageToWGPU(entry.visibility);

        // All binding-type fields default to Undefined; set only the active one
        layoutEntry.bufferType = WGPUBufferBindingType_Undefined;
        layoutEntry.bufferHasDynamicOffset = WGPU_FALSE;
        layoutEntry.bufferMinBindingSize = 0;
        layoutEntry.samplerType = WGPUSamplerBindingType_Undefined;
        layoutEntry.textureSampleType = WGPUTextureSampleType_Undefined;
        layoutEntry.textureViewDimension = WGPUTextureViewDimension_Undefined;
        layoutEntry.textureMultisampled = WGPU_FALSE;
        layoutEntry.storageTextureAccess = WGPUStorageTextureAccess_Undefined;
        layoutEntry.storageTextureFormat = WGPUTextureFormat_Undefined;
        layoutEntry.storageTextureViewDimension = WGPUTextureViewDimension_Undefined;

        switch (entry.type) {
        case GFX_BINDING_TYPE_UNIFORM_BUFFER:
            layoutEntry.bufferType = WGPUBufferBindingType_Uniform;
            layoutEntry.bufferHasDynamicOffset = entry.uniformBuffer.hasDynamicOffset ? WGPU_TRUE : WGPU_FALSE;
            layoutEntry.bufferMinBindingSize = entry.uniformBuffer.minBindingSize;
            break;
        case GFX_BINDING_TYPE_STORAGE_BUFFER:
            layoutEntry.bufferType = gfxStorageBufferAccessToWGPU(entry.storageBuffer.access);
            layoutEntry.bufferHasDynamicOffset = entry.storageBuffer.hasDynamicOffset ? WGPU_TRUE : WGPU_FALSE;
            layoutEntry.bufferMinBindingSize = entry.storageBuffer.minBindingSize;
            break;
        case GFX_BINDING_TYPE_SAMPLER:
            layoutEntry.samplerType = gfxSamplerBindingTypeToWGPU(entry.sampler.type);
            break;
        case GFX_BINDING_TYPE_TEXTURE:
            layoutEntry.textureSampleType = gfxTextureSampleTypeToWGPU(entry.texture.sampleType);
            layoutEntry.textureViewDimension = gfxTextureViewTypeToWGPU(entry.texture.viewDimension);
            layoutEntry.textureMultisampled = entry.texture.multisampled ? WGPU_TRUE : WGPU_FALSE;
            break;
        case GFX_BINDING_TYPE_STORAGE_TEXTURE:
            layoutEntry.storageTextureAccess = gfxStorageTextureAccessToWGPU(entry.storageTexture.access);
            layoutEntry.storageTextureFormat = gfxFormatToWGPUFormat(entry.storageTexture.format);
            layoutEntry.storageTextureViewDimension = gfxTextureViewTypeToWGPU(entry.storageTexture.viewDimension);
            break;
        default:
            // Unknown type - leave as Undefined
            break;
        }
        return layoutEntry;
    }
} // anonymous namespace

core::BindGroupLayoutCreateInfo gfxDescriptorToWebGPUBindGroupLayoutCreateInfo(const GfxBindGroupLayoutDescriptor* descriptor)
{
    core::BindGroupLayoutCreateInfo createInfo{};
    for (uint32_t i = 0; i < descriptor->entryCount; ++i) {
        createInfo.entries.push_back(convertBindGroupLayoutEntry(descriptor->entries[i]));
    }
    return createInfo;
}

core::BindGroupCreateInfo gfxDescriptorToWebGPUBindGroupCreateInfo(const GfxBindGroupDescriptor* descriptor, WGPUBindGroupLayout layout)
{
    core::BindGroupCreateInfo createInfo{};
    createInfo.layout = layout;

    if (descriptor->entryCount > 0 && descriptor->entries) {
        createInfo.entries.reserve(descriptor->entryCount);

        for (uint32_t i = 0; i < descriptor->entryCount; ++i) {
            const auto& entry = descriptor->entries[i];
            core::BindGroupEntry bindEntry{};

            bindEntry.binding = entry.binding;
            bindEntry.arrayElement = entry.arrayElement;

            switch (entry.type) {
            case GFX_BIND_GROUP_ENTRY_TYPE_BUFFER: {
                auto* buffer = toNative<core::Buffer>(entry.resource.buffer.buffer);
                bindEntry.buffer = buffer->handle();
                bindEntry.bufferOffset = entry.resource.buffer.offset;
                bindEntry.bufferSize = entry.resource.buffer.size;
                break;
            }
            case GFX_BIND_GROUP_ENTRY_TYPE_SAMPLER: {
                auto* sampler = toNative<core::Sampler>(entry.resource.sampler);
                bindEntry.sampler = sampler->handle();
                break;
            }
            case GFX_BIND_GROUP_ENTRY_TYPE_TEXTURE_VIEW: {
                auto* textureView = toNative<core::TextureView>(entry.resource.textureView);
                bindEntry.textureView = textureView->handle();
                break;
            }
            default:
                // Unknown type - skip
                break;
            }

            createInfo.entries.push_back(bindEntry);
        }
    }

    return createInfo;
}

namespace {
    std::vector<core::ConstantEntry> convertConstants(const GfxConstantEntry* constants, uint32_t constantCount)
    {
        std::vector<core::ConstantEntry> result;
        if (!constants || constantCount == 0) {
            return result;
        }

        result.reserve(constantCount);
        for (uint32_t i = 0; i < constantCount; ++i) {
            core::ConstantEntry entry{};
            entry.key = std::to_string(constants[i].id);

            switch (constants[i].type) {
            case GFX_CONSTANT_TYPE_BOOL:
                entry.value = constants[i].value.b ? 1.0 : 0.0;
                break;
            case GFX_CONSTANT_TYPE_I32:
                entry.value = static_cast<double>(constants[i].value.i32);
                break;
            case GFX_CONSTANT_TYPE_U32:
                entry.value = static_cast<double>(constants[i].value.u32);
                break;
            case GFX_CONSTANT_TYPE_F32:
                entry.value = static_cast<double>(constants[i].value.f32);
                break;
            }

            result.push_back(std::move(entry));
        }
        return result;
    }

    core::VertexState convertVertexState(const GfxVertexState& vertex)
    {
        core::VertexState vkVertex{};
        auto* vertexShader = toNative<Shader>(vertex.module);
        vkVertex.module = vertexShader->handle();
        vkVertex.entryPoint = vertex.entryPoint;
        vkVertex.constants = convertConstants(vertex.constants, vertex.constantCount);

        vkVertex.buffers.reserve(vertex.bufferCount);
        for (uint32_t i = 0; i < vertex.bufferCount; ++i) {
            const auto& buffer = vertex.buffers[i];
            VertexBufferLayout vbLayout{};
            vbLayout.arrayStride = buffer.arrayStride;
            vbLayout.stepMode = gfxVertexStepModeToWGPU(buffer.stepMode);

            vbLayout.attributes.reserve(buffer.attributeCount);
            for (uint32_t j = 0; j < buffer.attributeCount; ++j) {
                const auto& attr = buffer.attributes[j];
                VertexAttribute vbAttr{};
                vbAttr.format = gfxFormatToWGPUVertexFormat(attr.format);
                vbAttr.offset = attr.offset;
                vbAttr.shaderLocation = attr.shaderLocation;
                vbLayout.attributes.push_back(vbAttr);
            }

            vkVertex.buffers.push_back(std::move(vbLayout));
        }
        return vkVertex;
    }

    // Color targets take their format from the render pass; writeMask/blend come
    // from the fragment descriptor when an entry is provided for that index
    ColorTargetState convertColorTarget(WGPUTextureFormat format, const GfxColorTargetState* target)
    {
        ColorTargetState colorTarget{};
        colorTarget.format = format;

        if (!target) {
            colorTarget.writeMask = GFX_COLOR_WRITE_MASK_ALL; // Default when not specified
            return colorTarget;
        }

        colorTarget.writeMask = target->writeMask;
        if (target->blend) {
            BlendState blend{};
            blend.color.operation = gfxBlendOperationToWGPU(target->blend->color.operation);
            blend.color.srcFactor = gfxBlendFactorToWGPU(target->blend->color.srcFactor);
            blend.color.dstFactor = gfxBlendFactorToWGPU(target->blend->color.dstFactor);
            blend.alpha.operation = gfxBlendOperationToWGPU(target->blend->alpha.operation);
            blend.alpha.srcFactor = gfxBlendFactorToWGPU(target->blend->alpha.srcFactor);
            blend.alpha.dstFactor = gfxBlendFactorToWGPU(target->blend->alpha.dstFactor);
            colorTarget.blend = blend;
        }
        return colorTarget;
    }

    FragmentState convertFragmentState(const GfxFragmentState& fragment, const RenderPass& renderPass)
    {
        FragmentState fragState{};
        auto* fragmentShader = toNative<Shader>(fragment.module);
        fragState.module = fragmentShader->handle();
        fragState.entryPoint = fragment.entryPoint;
        fragState.constants = convertConstants(fragment.constants, fragment.constantCount);

        // One target per render pass color attachment; the format always comes from
        // the render pass, blend/writeMask from the matching fragment target if present
        const auto& rpInfo = renderPass.getCreateInfo();
        fragState.targets.reserve(rpInfo.colorAttachments.size());
        for (uint32_t i = 0; i < rpInfo.colorAttachments.size(); ++i) {
            const GfxColorTargetState* target = (fragment.targetCount > i) ? &fragment.targets[i] : nullptr;
            fragState.targets.push_back(convertColorTarget(rpInfo.colorAttachments[i].format, target));
        }
        return fragState;
    }

    DepthStencilState convertDepthStencilState(const GfxDepthStencilState& depthStencil)
    {
        DepthStencilState dsState{};
        dsState.format = gfxFormatToWGPUFormat(depthStencil.format);
        dsState.depthWriteEnabled = depthStencil.depthWriteEnabled;
        dsState.depthCompare = gfxCompareFunctionToWGPU(depthStencil.depthCompare);

        dsState.stencilFront.compare = gfxCompareFunctionToWGPU(depthStencil.stencilFront.compare);
        dsState.stencilFront.failOp = gfxStencilOperationToWGPU(depthStencil.stencilFront.failOp);
        dsState.stencilFront.depthFailOp = gfxStencilOperationToWGPU(depthStencil.stencilFront.depthFailOp);
        dsState.stencilFront.passOp = gfxStencilOperationToWGPU(depthStencil.stencilFront.passOp);

        dsState.stencilBack.compare = gfxCompareFunctionToWGPU(depthStencil.stencilBack.compare);
        dsState.stencilBack.failOp = gfxStencilOperationToWGPU(depthStencil.stencilBack.failOp);
        dsState.stencilBack.depthFailOp = gfxStencilOperationToWGPU(depthStencil.stencilBack.depthFailOp);
        dsState.stencilBack.passOp = gfxStencilOperationToWGPU(depthStencil.stencilBack.passOp);

        dsState.stencilReadMask = depthStencil.stencilReadMask;
        dsState.stencilWriteMask = depthStencil.stencilWriteMask;
        dsState.depthBias = depthStencil.depthBias;
        dsState.depthBiasSlopeScale = depthStencil.depthBiasSlopeScale;
        dsState.depthBiasClamp = depthStencil.depthBiasClamp;
        return dsState;
    }
} // anonymous namespace

core::RenderPipelineCreateInfo gfxDescriptorToWebGPURenderPipelineCreateInfo(const GfxRenderPipelineDescriptor* descriptor)
{
    core::RenderPipelineCreateInfo createInfo{};

    if (descriptor->bindGroupLayoutCount > 0 && descriptor->bindGroupLayouts) {
        createInfo.bindGroupLayouts.reserve(descriptor->bindGroupLayoutCount);
        for (uint32_t i = 0; i < descriptor->bindGroupLayoutCount; ++i) {
            auto* layout = toNative<BindGroupLayout>(descriptor->bindGroupLayouts[i]);
            createInfo.bindGroupLayouts.push_back(layout->handle());
        }
    }

    createInfo.vertex = convertVertexState(*descriptor->vertex);
    if (descriptor->fragment) {
        // RenderPass is mandatory - color target formats come from it
        auto* renderPass = toNative<RenderPass>(descriptor->renderPass);
        createInfo.fragment = convertFragmentState(*descriptor->fragment, *renderPass);
    }

    createInfo.primitive.topology = gfxPrimitiveTopologyToWGPU(descriptor->primitive->topology);
    createInfo.primitive.frontFace = gfxFrontFaceToWGPU(descriptor->primitive->frontFace);
    createInfo.primitive.cullMode = gfxCullModeToWGPU(descriptor->primitive->cullMode);
    createInfo.primitive.stripIndexFormat = gfxIndexFormatToWGPU(descriptor->primitive->stripIndexFormat);

    if (descriptor->depthStencil) {
        createInfo.depthStencil = convertDepthStencilState(*descriptor->depthStencil);
    }

    createInfo.sampleCount = descriptor->sampleCount;

    return createInfo;
}

core::ComputePipelineCreateInfo gfxDescriptorToWebGPUComputePipelineCreateInfo(const GfxComputePipelineDescriptor* descriptor)
{
    core::ComputePipelineCreateInfo createInfo{};

    // Extract bind group layouts
    if (descriptor->bindGroupLayoutCount > 0 && descriptor->bindGroupLayouts) {
        createInfo.bindGroupLayouts.reserve(descriptor->bindGroupLayoutCount);
        for (uint32_t i = 0; i < descriptor->bindGroupLayoutCount; ++i) {
            auto* layout = toNative<BindGroupLayout>(descriptor->bindGroupLayouts[i]);
            createInfo.bindGroupLayouts.push_back(layout->handle());
        }
    }

    // Extract shader module
    auto* shader = toNative<Shader>(descriptor->compute);
    createInfo.module = shader->handle();
    createInfo.entryPoint = descriptor->entryPoint;
    createInfo.constants = convertConstants(descriptor->constants, descriptor->constantCount);

    return createInfo;
}

core::CommandEncoderCreateInfo gfxDescriptorToWebGPUCommandEncoderCreateInfo(const GfxCommandEncoderDescriptor* descriptor)
{
    core::CommandEncoderCreateInfo createInfo;
    createInfo.label = descriptor->label;
    return createInfo;
}

core::SubmitInfo gfxDescriptorToWebGPUSubmitInfo(const GfxSubmitDescriptor* descriptor)
{
    core::SubmitInfo submitInfo{};
    // Note: Array pointer conversions use reinterpret_cast as toNative<> is for individual objects
    submitInfo.commandEncoders = reinterpret_cast<CommandEncoder* const*>(descriptor->commandEncoders);
    submitInfo.commandEncoderCount = descriptor->commandEncoderCount;
    submitInfo.signalFence = toNative<Fence>(descriptor->signalFence);
    submitInfo.waitSemaphores = reinterpret_cast<Semaphore* const*>(descriptor->waitSemaphores);
    submitInfo.waitValues = descriptor->waitValues;
    submitInfo.waitSemaphoreCount = descriptor->waitSemaphoreCount;
    submitInfo.signalSemaphores = reinterpret_cast<Semaphore* const*>(descriptor->signalSemaphores);
    submitInfo.signalValues = descriptor->signalValues;
    submitInfo.signalSemaphoreCount = descriptor->signalSemaphoreCount;
    return submitInfo;
}

core::QuerySetCreateInfo gfxDescriptorToWebGPUQuerySetCreateInfo(const GfxQuerySetDescriptor* descriptor)
{
    core::QuerySetCreateInfo createInfo{};

    if (descriptor) {
        createInfo.label = descriptor->label;
        createInfo.type = gfxQueryTypeToWebGPUQueryType(descriptor->type);
        createInfo.count = descriptor->count;
    } else {
        createInfo.type = WGPUQueryType_Occlusion;
        createInfo.count = 1;
    }

    return createInfo;
}

// ============================================================================
// Reverse Conversions - Internal to Gfx API types
// ============================================================================

GfxBufferUsageFlags webgpuBufferUsageToGfxBufferUsage(WGPUBufferUsage usage)
{
    uint32_t gfxUsage = GFX_BUFFER_USAGE_NONE;
    if (usage & WGPUBufferUsage_MapRead) {
        gfxUsage |= GFX_BUFFER_USAGE_MAP_READ;
    }
    if (usage & WGPUBufferUsage_MapWrite) {
        gfxUsage |= GFX_BUFFER_USAGE_MAP_WRITE;
    }
    if (usage & WGPUBufferUsage_CopySrc) {
        gfxUsage |= GFX_BUFFER_USAGE_COPY_SRC;
    }
    if (usage & WGPUBufferUsage_CopyDst) {
        gfxUsage |= GFX_BUFFER_USAGE_COPY_DST;
    }
    if (usage & WGPUBufferUsage_Index) {
        gfxUsage |= GFX_BUFFER_USAGE_INDEX;
    }
    if (usage & WGPUBufferUsage_Vertex) {
        gfxUsage |= GFX_BUFFER_USAGE_VERTEX;
    }
    if (usage & WGPUBufferUsage_Uniform) {
        gfxUsage |= GFX_BUFFER_USAGE_UNIFORM;
    }
    if (usage & WGPUBufferUsage_Storage) {
        gfxUsage |= GFX_BUFFER_USAGE_STORAGE;
    }
    if (usage & WGPUBufferUsage_Indirect) {
        gfxUsage |= GFX_BUFFER_USAGE_INDIRECT;
    }
    if (usage & WGPUBufferUsage_QueryResolve) {
        gfxUsage |= GFX_BUFFER_USAGE_QUERY_RESOLVE;
    }
    return gfxUsage;
}

GfxSemaphoreType webgpuSemaphoreTypeToGfxSemaphoreType(core::SemaphoreType type)
{
    switch (type) {
    case core::SemaphoreType::Binary:
        return GFX_SEMAPHORE_TYPE_BINARY;
    case core::SemaphoreType::Timeline:
        return GFX_SEMAPHORE_TYPE_TIMELINE;
    default:
        return GFX_SEMAPHORE_TYPE_BINARY;
    }
}

GfxTextureInfo wgpuTextureInfoToGfxTextureInfo(const core::TextureInfo& info)
{
    GfxTextureInfo gfxInfo{};
    gfxInfo.type = wgpuTextureDimensionToGfxTextureType(info.dimension);
    gfxInfo.size = wgpuExtent3DToGfxExtent3D(info.size);
    gfxInfo.arrayLayerCount = info.arrayLayers;
    gfxInfo.mipLevelCount = info.mipLevels;
    gfxInfo.sampleCount = wgpuSampleCountToGfxSampleCount(info.sampleCount);
    gfxInfo.format = wgpuFormatToGfxFormat(info.format);
    gfxInfo.usage = wgpuTextureUsageToGfxTextureUsage(info.usage);
    return gfxInfo;
}

GfxSurfaceInfo wgpuSurfaceInfoToGfxSurfaceInfo(const core::SurfaceInfo& surfaceInfo)
{
    GfxSurfaceInfo info{};
    info.minImageCount = surfaceInfo.minImageCount;
    info.maxImageCount = surfaceInfo.maxImageCount;
    info.minExtent = { surfaceInfo.minWidth, surfaceInfo.minHeight };
    info.maxExtent = { surfaceInfo.maxWidth, surfaceInfo.maxHeight };
    return info;
}

GfxSwapchainInfo wgpuSwapchainInfoToGfxSwapchainInfo(const core::SwapchainInfo& info)
{
    GfxSwapchainInfo gfxInfo{};
    gfxInfo.extent = { info.width, info.height };
    gfxInfo.format = wgpuFormatToGfxFormat(info.format);
    gfxInfo.imageCount = info.imageCount;
    gfxInfo.presentMode = wgpuPresentModeToGfxPresentMode(info.presentMode);
    return gfxInfo;
}

GfxBufferInfo wgpuBufferToGfxBufferInfo(const core::BufferInfo& info)
{
    GfxBufferInfo gfxInfo{};
    gfxInfo.size = info.size;
    gfxInfo.usage = webgpuBufferUsageToGfxBufferUsage(info.usage);
    gfxInfo.memoryProperties = info.memoryProperties;
    return gfxInfo;
}

GfxQueueInfo wgpuQueueInfoToGfxQueueInfo(const core::QueueInfo& info)
{
    GfxQueueInfo gfxInfo{};
    gfxInfo.queueFamilyIndex = info.queueFamilyIndex;
    gfxInfo.queueIndex = info.queueIndex;
    return gfxInfo;
}

// ============================================================================
// Conversion Function Implementations
// ============================================================================

WGPUStringView gfxStringView(const char* str)
{
    if (!str) {
        return WGPUStringView{ nullptr, WGPU_STRLEN };
    }
    return WGPUStringView{ str, WGPU_STRLEN };
}

WGPUTextureFormat gfxFormatToWGPUFormat(GfxFormat format)
{
    switch (format) {
    case GFX_FORMAT_R8_UNORM:
        return WGPUTextureFormat_R8Unorm;
    case GFX_FORMAT_R8G8_UNORM:
        return WGPUTextureFormat_RG8Unorm;
    case GFX_FORMAT_R8G8B8A8_UNORM:
        return WGPUTextureFormat_RGBA8Unorm;
    case GFX_FORMAT_R8G8B8A8_UNORM_SRGB:
        return WGPUTextureFormat_RGBA8UnormSrgb;
    case GFX_FORMAT_B8G8R8A8_UNORM:
        return WGPUTextureFormat_BGRA8Unorm;
    case GFX_FORMAT_B8G8R8A8_UNORM_SRGB:
        return WGPUTextureFormat_BGRA8UnormSrgb;
    case GFX_FORMAT_R16_FLOAT:
        return WGPUTextureFormat_R16Float;
    case GFX_FORMAT_R16G16_FLOAT:
        return WGPUTextureFormat_RG16Float;
    case GFX_FORMAT_R16G16B16A16_FLOAT:
        return WGPUTextureFormat_RGBA16Float;
    case GFX_FORMAT_R32_FLOAT:
        return WGPUTextureFormat_R32Float;
    case GFX_FORMAT_R32G32_FLOAT:
        return WGPUTextureFormat_RG32Float;
    case GFX_FORMAT_R32G32B32A32_FLOAT:
        return WGPUTextureFormat_RGBA32Float;
    case GFX_FORMAT_DEPTH16_UNORM:
        return WGPUTextureFormat_Depth16Unorm;
    case GFX_FORMAT_DEPTH24_PLUS:
        return WGPUTextureFormat_Depth24Plus;
    case GFX_FORMAT_DEPTH32_FLOAT:
        return WGPUTextureFormat_Depth32Float;
    case GFX_FORMAT_STENCIL8:
        return WGPUTextureFormat_Stencil8;
    case GFX_FORMAT_DEPTH24_PLUS_STENCIL8:
        return WGPUTextureFormat_Depth24PlusStencil8;
    case GFX_FORMAT_DEPTH32_FLOAT_STENCIL8:
        return WGPUTextureFormat_Depth32FloatStencil8;
    case GFX_FORMAT_R8_SINT:
        return WGPUTextureFormat_R8Sint;
    case GFX_FORMAT_R8_UINT:
        return WGPUTextureFormat_R8Uint;
    case GFX_FORMAT_R8G8_SINT:
        return WGPUTextureFormat_RG8Sint;
    case GFX_FORMAT_R8G8_UINT:
        return WGPUTextureFormat_RG8Uint;
    case GFX_FORMAT_R8G8B8A8_SINT:
        return WGPUTextureFormat_RGBA8Sint;
    case GFX_FORMAT_R8G8B8A8_UINT:
        return WGPUTextureFormat_RGBA8Uint;
    case GFX_FORMAT_R16_SINT:
        return WGPUTextureFormat_R16Sint;
    case GFX_FORMAT_R16_UINT:
        return WGPUTextureFormat_R16Uint;
    case GFX_FORMAT_R16G16_SINT:
        return WGPUTextureFormat_RG16Sint;
    case GFX_FORMAT_R16G16_UINT:
        return WGPUTextureFormat_RG16Uint;
    case GFX_FORMAT_R16G16B16A16_SINT:
        return WGPUTextureFormat_RGBA16Sint;
    case GFX_FORMAT_R16G16B16A16_UINT:
        return WGPUTextureFormat_RGBA16Uint;
    case GFX_FORMAT_R32_SINT:
        return WGPUTextureFormat_R32Sint;
    case GFX_FORMAT_R32_UINT:
        return WGPUTextureFormat_R32Uint;
    case GFX_FORMAT_R32G32_SINT:
        return WGPUTextureFormat_RG32Sint;
    case GFX_FORMAT_R32G32_UINT:
        return WGPUTextureFormat_RG32Uint;
    case GFX_FORMAT_R32G32B32A32_SINT:
        return WGPUTextureFormat_RGBA32Sint;
    case GFX_FORMAT_R32G32B32A32_UINT:
        return WGPUTextureFormat_RGBA32Uint;
    // BC
    case GFX_FORMAT_BC1_RGBA_UNORM:
        return WGPUTextureFormat_BC1RGBAUnorm;
    case GFX_FORMAT_BC1_RGBA_UNORM_SRGB:
        return WGPUTextureFormat_BC1RGBAUnormSrgb;
    case GFX_FORMAT_BC2_RGBA_UNORM:
        return WGPUTextureFormat_BC2RGBAUnorm;
    case GFX_FORMAT_BC2_RGBA_UNORM_SRGB:
        return WGPUTextureFormat_BC2RGBAUnormSrgb;
    case GFX_FORMAT_BC3_RGBA_UNORM:
        return WGPUTextureFormat_BC3RGBAUnorm;
    case GFX_FORMAT_BC3_RGBA_UNORM_SRGB:
        return WGPUTextureFormat_BC3RGBAUnormSrgb;
    case GFX_FORMAT_BC4_R_UNORM:
        return WGPUTextureFormat_BC4RUnorm;
    case GFX_FORMAT_BC4_R_SNORM:
        return WGPUTextureFormat_BC4RSnorm;
    case GFX_FORMAT_BC5_RG_UNORM:
        return WGPUTextureFormat_BC5RGUnorm;
    case GFX_FORMAT_BC5_RG_SNORM:
        return WGPUTextureFormat_BC5RGSnorm;
    case GFX_FORMAT_BC6H_RGB_UFLOAT:
        return WGPUTextureFormat_BC6HRGBUfloat;
    case GFX_FORMAT_BC6H_RGB_SFLOAT:
        return WGPUTextureFormat_BC6HRGBFloat;
    case GFX_FORMAT_BC7_RGBA_UNORM:
        return WGPUTextureFormat_BC7RGBAUnorm;
    case GFX_FORMAT_BC7_RGBA_UNORM_SRGB:
        return WGPUTextureFormat_BC7RGBAUnormSrgb;
    // ETC2 / EAC
    case GFX_FORMAT_ETC2_RGB8_UNORM:
        return WGPUTextureFormat_ETC2RGB8Unorm;
    case GFX_FORMAT_ETC2_RGB8_UNORM_SRGB:
        return WGPUTextureFormat_ETC2RGB8UnormSrgb;
    case GFX_FORMAT_ETC2_RGB8A1_UNORM:
        return WGPUTextureFormat_ETC2RGB8A1Unorm;
    case GFX_FORMAT_ETC2_RGB8A1_UNORM_SRGB:
        return WGPUTextureFormat_ETC2RGB8A1UnormSrgb;
    case GFX_FORMAT_ETC2_RGBA8_UNORM:
        return WGPUTextureFormat_ETC2RGBA8Unorm;
    case GFX_FORMAT_ETC2_RGBA8_UNORM_SRGB:
        return WGPUTextureFormat_ETC2RGBA8UnormSrgb;
    case GFX_FORMAT_EAC_R11_UNORM:
        return WGPUTextureFormat_EACR11Unorm;
    case GFX_FORMAT_EAC_R11_SNORM:
        return WGPUTextureFormat_EACR11Snorm;
    case GFX_FORMAT_EAC_RG11_UNORM:
        return WGPUTextureFormat_EACRG11Unorm;
    case GFX_FORMAT_EAC_RG11_SNORM:
        return WGPUTextureFormat_EACRG11Snorm;
    // ASTC
    case GFX_FORMAT_ASTC_4X4_UNORM:
        return WGPUTextureFormat_ASTC4x4Unorm;
    case GFX_FORMAT_ASTC_4X4_UNORM_SRGB:
        return WGPUTextureFormat_ASTC4x4UnormSrgb;
    case GFX_FORMAT_ASTC_5X4_UNORM:
        return WGPUTextureFormat_ASTC5x4Unorm;
    case GFX_FORMAT_ASTC_5X4_UNORM_SRGB:
        return WGPUTextureFormat_ASTC5x4UnormSrgb;
    case GFX_FORMAT_ASTC_5X5_UNORM:
        return WGPUTextureFormat_ASTC5x5Unorm;
    case GFX_FORMAT_ASTC_5X5_UNORM_SRGB:
        return WGPUTextureFormat_ASTC5x5UnormSrgb;
    case GFX_FORMAT_ASTC_6X5_UNORM:
        return WGPUTextureFormat_ASTC6x5Unorm;
    case GFX_FORMAT_ASTC_6X5_UNORM_SRGB:
        return WGPUTextureFormat_ASTC6x5UnormSrgb;
    case GFX_FORMAT_ASTC_6X6_UNORM:
        return WGPUTextureFormat_ASTC6x6Unorm;
    case GFX_FORMAT_ASTC_6X6_UNORM_SRGB:
        return WGPUTextureFormat_ASTC6x6UnormSrgb;
    case GFX_FORMAT_ASTC_8X5_UNORM:
        return WGPUTextureFormat_ASTC8x5Unorm;
    case GFX_FORMAT_ASTC_8X5_UNORM_SRGB:
        return WGPUTextureFormat_ASTC8x5UnormSrgb;
    case GFX_FORMAT_ASTC_8X6_UNORM:
        return WGPUTextureFormat_ASTC8x6Unorm;
    case GFX_FORMAT_ASTC_8X6_UNORM_SRGB:
        return WGPUTextureFormat_ASTC8x6UnormSrgb;
    case GFX_FORMAT_ASTC_8X8_UNORM:
        return WGPUTextureFormat_ASTC8x8Unorm;
    case GFX_FORMAT_ASTC_8X8_UNORM_SRGB:
        return WGPUTextureFormat_ASTC8x8UnormSrgb;
    case GFX_FORMAT_ASTC_10X5_UNORM:
        return WGPUTextureFormat_ASTC10x5Unorm;
    case GFX_FORMAT_ASTC_10X5_UNORM_SRGB:
        return WGPUTextureFormat_ASTC10x5UnormSrgb;
    case GFX_FORMAT_ASTC_10X6_UNORM:
        return WGPUTextureFormat_ASTC10x6Unorm;
    case GFX_FORMAT_ASTC_10X6_UNORM_SRGB:
        return WGPUTextureFormat_ASTC10x6UnormSrgb;
    case GFX_FORMAT_ASTC_10X8_UNORM:
        return WGPUTextureFormat_ASTC10x8Unorm;
    case GFX_FORMAT_ASTC_10X8_UNORM_SRGB:
        return WGPUTextureFormat_ASTC10x8UnormSrgb;
    case GFX_FORMAT_ASTC_10X10_UNORM:
        return WGPUTextureFormat_ASTC10x10Unorm;
    case GFX_FORMAT_ASTC_10X10_UNORM_SRGB:
        return WGPUTextureFormat_ASTC10x10UnormSrgb;
    case GFX_FORMAT_ASTC_12X10_UNORM:
        return WGPUTextureFormat_ASTC12x10Unorm;
    case GFX_FORMAT_ASTC_12X10_UNORM_SRGB:
        return WGPUTextureFormat_ASTC12x10UnormSrgb;
    case GFX_FORMAT_ASTC_12X12_UNORM:
        return WGPUTextureFormat_ASTC12x12Unorm;
    case GFX_FORMAT_ASTC_12X12_UNORM_SRGB:
        return WGPUTextureFormat_ASTC12x12UnormSrgb;
    default:
        return WGPUTextureFormat_Undefined;
    }
}

GfxFormat wgpuFormatToGfxFormat(WGPUTextureFormat format)
{
    switch (format) {
    case WGPUTextureFormat_R8Unorm:
        return GFX_FORMAT_R8_UNORM;
    case WGPUTextureFormat_RG8Unorm:
        return GFX_FORMAT_R8G8_UNORM;
    case WGPUTextureFormat_RGBA8Unorm:
        return GFX_FORMAT_R8G8B8A8_UNORM;
    case WGPUTextureFormat_RGBA8UnormSrgb:
        return GFX_FORMAT_R8G8B8A8_UNORM_SRGB;
    case WGPUTextureFormat_BGRA8Unorm:
        return GFX_FORMAT_B8G8R8A8_UNORM;
    case WGPUTextureFormat_BGRA8UnormSrgb:
        return GFX_FORMAT_B8G8R8A8_UNORM_SRGB;
    case WGPUTextureFormat_R16Float:
        return GFX_FORMAT_R16_FLOAT;
    case WGPUTextureFormat_RG16Float:
        return GFX_FORMAT_R16G16_FLOAT;
    case WGPUTextureFormat_RGBA16Float:
        return GFX_FORMAT_R16G16B16A16_FLOAT;
    case WGPUTextureFormat_R32Float:
        return GFX_FORMAT_R32_FLOAT;
    case WGPUTextureFormat_RG32Float:
        return GFX_FORMAT_R32G32_FLOAT;
    case WGPUTextureFormat_RGBA32Float:
        return GFX_FORMAT_R32G32B32A32_FLOAT;
    case WGPUTextureFormat_Depth16Unorm:
        return GFX_FORMAT_DEPTH16_UNORM;
    case WGPUTextureFormat_Depth24Plus:
        return GFX_FORMAT_DEPTH24_PLUS;
    case WGPUTextureFormat_Depth32Float:
        return GFX_FORMAT_DEPTH32_FLOAT;
    case WGPUTextureFormat_Stencil8:
        return GFX_FORMAT_STENCIL8;
    case WGPUTextureFormat_Depth24PlusStencil8:
        return GFX_FORMAT_DEPTH24_PLUS_STENCIL8;
    case WGPUTextureFormat_Depth32FloatStencil8:
        return GFX_FORMAT_DEPTH32_FLOAT_STENCIL8;
    case WGPUTextureFormat_R8Sint:
        return GFX_FORMAT_R8_SINT;
    case WGPUTextureFormat_R8Uint:
        return GFX_FORMAT_R8_UINT;
    case WGPUTextureFormat_RG8Sint:
        return GFX_FORMAT_R8G8_SINT;
    case WGPUTextureFormat_RG8Uint:
        return GFX_FORMAT_R8G8_UINT;
    case WGPUTextureFormat_RGBA8Sint:
        return GFX_FORMAT_R8G8B8A8_SINT;
    case WGPUTextureFormat_RGBA8Uint:
        return GFX_FORMAT_R8G8B8A8_UINT;
    case WGPUTextureFormat_R16Sint:
        return GFX_FORMAT_R16_SINT;
    case WGPUTextureFormat_R16Uint:
        return GFX_FORMAT_R16_UINT;
    case WGPUTextureFormat_RG16Sint:
        return GFX_FORMAT_R16G16_SINT;
    case WGPUTextureFormat_RG16Uint:
        return GFX_FORMAT_R16G16_UINT;
    case WGPUTextureFormat_RGBA16Sint:
        return GFX_FORMAT_R16G16B16A16_SINT;
    case WGPUTextureFormat_RGBA16Uint:
        return GFX_FORMAT_R16G16B16A16_UINT;
    case WGPUTextureFormat_R32Sint:
        return GFX_FORMAT_R32_SINT;
    case WGPUTextureFormat_R32Uint:
        return GFX_FORMAT_R32_UINT;
    case WGPUTextureFormat_RG32Sint:
        return GFX_FORMAT_R32G32_SINT;
    case WGPUTextureFormat_RG32Uint:
        return GFX_FORMAT_R32G32_UINT;
    case WGPUTextureFormat_RGBA32Sint:
        return GFX_FORMAT_R32G32B32A32_SINT;
    case WGPUTextureFormat_RGBA32Uint:
        return GFX_FORMAT_R32G32B32A32_UINT;
    // BC
    case WGPUTextureFormat_BC1RGBAUnorm:
        return GFX_FORMAT_BC1_RGBA_UNORM;
    case WGPUTextureFormat_BC1RGBAUnormSrgb:
        return GFX_FORMAT_BC1_RGBA_UNORM_SRGB;
    case WGPUTextureFormat_BC2RGBAUnorm:
        return GFX_FORMAT_BC2_RGBA_UNORM;
    case WGPUTextureFormat_BC2RGBAUnormSrgb:
        return GFX_FORMAT_BC2_RGBA_UNORM_SRGB;
    case WGPUTextureFormat_BC3RGBAUnorm:
        return GFX_FORMAT_BC3_RGBA_UNORM;
    case WGPUTextureFormat_BC3RGBAUnormSrgb:
        return GFX_FORMAT_BC3_RGBA_UNORM_SRGB;
    case WGPUTextureFormat_BC4RUnorm:
        return GFX_FORMAT_BC4_R_UNORM;
    case WGPUTextureFormat_BC4RSnorm:
        return GFX_FORMAT_BC4_R_SNORM;
    case WGPUTextureFormat_BC5RGUnorm:
        return GFX_FORMAT_BC5_RG_UNORM;
    case WGPUTextureFormat_BC5RGSnorm:
        return GFX_FORMAT_BC5_RG_SNORM;
    case WGPUTextureFormat_BC6HRGBUfloat:
        return GFX_FORMAT_BC6H_RGB_UFLOAT;
    case WGPUTextureFormat_BC6HRGBFloat:
        return GFX_FORMAT_BC6H_RGB_SFLOAT;
    case WGPUTextureFormat_BC7RGBAUnorm:
        return GFX_FORMAT_BC7_RGBA_UNORM;
    case WGPUTextureFormat_BC7RGBAUnormSrgb:
        return GFX_FORMAT_BC7_RGBA_UNORM_SRGB;
    // ETC2 / EAC
    case WGPUTextureFormat_ETC2RGB8Unorm:
        return GFX_FORMAT_ETC2_RGB8_UNORM;
    case WGPUTextureFormat_ETC2RGB8UnormSrgb:
        return GFX_FORMAT_ETC2_RGB8_UNORM_SRGB;
    case WGPUTextureFormat_ETC2RGB8A1Unorm:
        return GFX_FORMAT_ETC2_RGB8A1_UNORM;
    case WGPUTextureFormat_ETC2RGB8A1UnormSrgb:
        return GFX_FORMAT_ETC2_RGB8A1_UNORM_SRGB;
    case WGPUTextureFormat_ETC2RGBA8Unorm:
        return GFX_FORMAT_ETC2_RGBA8_UNORM;
    case WGPUTextureFormat_ETC2RGBA8UnormSrgb:
        return GFX_FORMAT_ETC2_RGBA8_UNORM_SRGB;
    case WGPUTextureFormat_EACR11Unorm:
        return GFX_FORMAT_EAC_R11_UNORM;
    case WGPUTextureFormat_EACR11Snorm:
        return GFX_FORMAT_EAC_R11_SNORM;
    case WGPUTextureFormat_EACRG11Unorm:
        return GFX_FORMAT_EAC_RG11_UNORM;
    case WGPUTextureFormat_EACRG11Snorm:
        return GFX_FORMAT_EAC_RG11_SNORM;
    // ASTC
    case WGPUTextureFormat_ASTC4x4Unorm:
        return GFX_FORMAT_ASTC_4X4_UNORM;
    case WGPUTextureFormat_ASTC4x4UnormSrgb:
        return GFX_FORMAT_ASTC_4X4_UNORM_SRGB;
    case WGPUTextureFormat_ASTC5x4Unorm:
        return GFX_FORMAT_ASTC_5X4_UNORM;
    case WGPUTextureFormat_ASTC5x4UnormSrgb:
        return GFX_FORMAT_ASTC_5X4_UNORM_SRGB;
    case WGPUTextureFormat_ASTC5x5Unorm:
        return GFX_FORMAT_ASTC_5X5_UNORM;
    case WGPUTextureFormat_ASTC5x5UnormSrgb:
        return GFX_FORMAT_ASTC_5X5_UNORM_SRGB;
    case WGPUTextureFormat_ASTC6x5Unorm:
        return GFX_FORMAT_ASTC_6X5_UNORM;
    case WGPUTextureFormat_ASTC6x5UnormSrgb:
        return GFX_FORMAT_ASTC_6X5_UNORM_SRGB;
    case WGPUTextureFormat_ASTC6x6Unorm:
        return GFX_FORMAT_ASTC_6X6_UNORM;
    case WGPUTextureFormat_ASTC6x6UnormSrgb:
        return GFX_FORMAT_ASTC_6X6_UNORM_SRGB;
    case WGPUTextureFormat_ASTC8x5Unorm:
        return GFX_FORMAT_ASTC_8X5_UNORM;
    case WGPUTextureFormat_ASTC8x5UnormSrgb:
        return GFX_FORMAT_ASTC_8X5_UNORM_SRGB;
    case WGPUTextureFormat_ASTC8x6Unorm:
        return GFX_FORMAT_ASTC_8X6_UNORM;
    case WGPUTextureFormat_ASTC8x6UnormSrgb:
        return GFX_FORMAT_ASTC_8X6_UNORM_SRGB;
    case WGPUTextureFormat_ASTC8x8Unorm:
        return GFX_FORMAT_ASTC_8X8_UNORM;
    case WGPUTextureFormat_ASTC8x8UnormSrgb:
        return GFX_FORMAT_ASTC_8X8_UNORM_SRGB;
    case WGPUTextureFormat_ASTC10x5Unorm:
        return GFX_FORMAT_ASTC_10X5_UNORM;
    case WGPUTextureFormat_ASTC10x5UnormSrgb:
        return GFX_FORMAT_ASTC_10X5_UNORM_SRGB;
    case WGPUTextureFormat_ASTC10x6Unorm:
        return GFX_FORMAT_ASTC_10X6_UNORM;
    case WGPUTextureFormat_ASTC10x6UnormSrgb:
        return GFX_FORMAT_ASTC_10X6_UNORM_SRGB;
    case WGPUTextureFormat_ASTC10x8Unorm:
        return GFX_FORMAT_ASTC_10X8_UNORM;
    case WGPUTextureFormat_ASTC10x8UnormSrgb:
        return GFX_FORMAT_ASTC_10X8_UNORM_SRGB;
    case WGPUTextureFormat_ASTC10x10Unorm:
        return GFX_FORMAT_ASTC_10X10_UNORM;
    case WGPUTextureFormat_ASTC10x10UnormSrgb:
        return GFX_FORMAT_ASTC_10X10_UNORM_SRGB;
    case WGPUTextureFormat_ASTC12x10Unorm:
        return GFX_FORMAT_ASTC_12X10_UNORM;
    case WGPUTextureFormat_ASTC12x10UnormSrgb:
        return GFX_FORMAT_ASTC_12X10_UNORM_SRGB;
    case WGPUTextureFormat_ASTC12x12Unorm:
        return GFX_FORMAT_ASTC_12X12_UNORM;
    case WGPUTextureFormat_ASTC12x12UnormSrgb:
        return GFX_FORMAT_ASTC_12X12_UNORM_SRGB;
    default:
        return GFX_FORMAT_UNDEFINED;
    }
}

GfxPresentMode wgpuPresentModeToGfxPresentMode(WGPUPresentMode mode)
{
    switch (mode) {
    case WGPUPresentMode_Immediate:
        return GFX_PRESENT_MODE_IMMEDIATE;
    case WGPUPresentMode_Mailbox:
        return GFX_PRESENT_MODE_MAILBOX;
    case WGPUPresentMode_Fifo:
        return GFX_PRESENT_MODE_FIFO;
    case WGPUPresentMode_FifoRelaxed:
        return GFX_PRESENT_MODE_FIFO_RELAXED;
    default:
        return GFX_PRESENT_MODE_FIFO;
    }
}

GfxSampleCount wgpuSampleCountToGfxSampleCount(uint32_t sampleCount)
{
    switch (sampleCount) {
    case 1:
        return GFX_SAMPLE_COUNT_1;
    case 2:
        return GFX_SAMPLE_COUNT_2;
    case 4:
        return GFX_SAMPLE_COUNT_4;
    case 8:
        return GFX_SAMPLE_COUNT_8;
    case 16:
        return GFX_SAMPLE_COUNT_16;
    case 32:
        return GFX_SAMPLE_COUNT_32;
    case 64:
        return GFX_SAMPLE_COUNT_64;
    default:
        return GFX_SAMPLE_COUNT_1;
    }
}

WGPUPresentMode gfxPresentModeToWGPU(GfxPresentMode mode)
{
    switch (mode) {
    case GFX_PRESENT_MODE_IMMEDIATE:
        return WGPUPresentMode_Immediate;
    case GFX_PRESENT_MODE_FIFO:
        return WGPUPresentMode_Fifo;
    case GFX_PRESENT_MODE_FIFO_RELAXED:
        return WGPUPresentMode_FifoRelaxed;
    case GFX_PRESENT_MODE_MAILBOX:
        return WGPUPresentMode_Mailbox;
    default:
        return WGPUPresentMode_Fifo;
    }
}

bool formatHasStencil(GfxFormat format)
{
    switch (format) {
    case GFX_FORMAT_STENCIL8:
    case GFX_FORMAT_DEPTH24_PLUS_STENCIL8:
    case GFX_FORMAT_DEPTH32_FLOAT_STENCIL8:
        return true;
    default:
        return false;
    }
}

WGPULoadOp gfxLoadOpToWGPULoadOp(GfxLoadOp loadOp)
{
    switch (loadOp) {
    case GFX_LOAD_OP_LOAD:
        return WGPULoadOp_Load;
    case GFX_LOAD_OP_CLEAR:
        return WGPULoadOp_Clear;
    case GFX_LOAD_OP_DONT_CARE:
        return WGPULoadOp_Undefined;
    default:
        return WGPULoadOp_Undefined;
    }
}

WGPUStoreOp gfxStoreOpToWGPUStoreOp(GfxStoreOp storeOp)
{
    switch (storeOp) {
    case GFX_STORE_OP_STORE:
        return WGPUStoreOp_Store;
    case GFX_STORE_OP_DONT_CARE:
        return WGPUStoreOp_Discard;
    default:
        return WGPUStoreOp_Undefined;
    }
}

WGPUBufferUsage gfxBufferUsageToWGPU(GfxBufferUsageFlags usage)
{
    WGPUBufferUsage wgpuUsage = WGPUBufferUsage_None;

    // WebGPU constraint: MapWrite can only combine with CopySrc, MapRead can only combine with CopyDst.
    // If incompatible usages are present, strip MAP flags and add CopyDst for queue-write fallback.
    const bool hasMapWrite = (usage & GFX_BUFFER_USAGE_MAP_WRITE) != 0;
    const bool hasMapRead = (usage & GFX_BUFFER_USAGE_MAP_READ) != 0;
    const GfxBufferUsageFlags nonMapNonCopy = usage & ~(GfxBufferUsageFlags)(GFX_BUFFER_USAGE_MAP_READ | GFX_BUFFER_USAGE_MAP_WRITE | GFX_BUFFER_USAGE_COPY_SRC | GFX_BUFFER_USAGE_COPY_DST);

    if ((hasMapWrite || hasMapRead) && nonMapNonCopy != 0) {
        // Incompatible: strip MAP flags, add CopyDst so data can be uploaded via queue write
        wgpuUsage |= WGPUBufferUsage_CopyDst;
    } else {
        if (hasMapRead) {
            wgpuUsage |= WGPUBufferUsage_MapRead;
        }
        if (hasMapWrite) {
            wgpuUsage |= WGPUBufferUsage_MapWrite;
        }
    }

    if (usage & GFX_BUFFER_USAGE_COPY_SRC) {
        wgpuUsage |= WGPUBufferUsage_CopySrc;
    }
    if (usage & GFX_BUFFER_USAGE_COPY_DST) {
        wgpuUsage |= WGPUBufferUsage_CopyDst;
    }
    if (usage & GFX_BUFFER_USAGE_INDEX) {
        wgpuUsage |= WGPUBufferUsage_Index;
    }
    if (usage & GFX_BUFFER_USAGE_VERTEX) {
        wgpuUsage |= WGPUBufferUsage_Vertex;
    }
    if (usage & GFX_BUFFER_USAGE_UNIFORM) {
        wgpuUsage |= WGPUBufferUsage_Uniform;
    }
    if (usage & GFX_BUFFER_USAGE_STORAGE) {
        wgpuUsage |= WGPUBufferUsage_Storage;
    }
    if (usage & GFX_BUFFER_USAGE_INDIRECT) {
        wgpuUsage |= WGPUBufferUsage_Indirect;
    }
    if (usage & GFX_BUFFER_USAGE_QUERY_RESOLVE) {
        wgpuUsage |= WGPUBufferUsage_QueryResolve;
    }
    return wgpuUsage;
}

WGPUTextureUsage gfxTextureUsageToWGPU(GfxTextureUsageFlags usage)
{
    WGPUTextureUsage wgpuUsage = WGPUTextureUsage_None;
    if (usage & GFX_TEXTURE_USAGE_COPY_SRC) {
        wgpuUsage |= WGPUTextureUsage_CopySrc;
    }
    if (usage & GFX_TEXTURE_USAGE_COPY_DST) {
        wgpuUsage |= WGPUTextureUsage_CopyDst;
    }
    if (usage & GFX_TEXTURE_USAGE_TEXTURE_BINDING) {
        wgpuUsage |= WGPUTextureUsage_TextureBinding;
    }
    if (usage & GFX_TEXTURE_USAGE_STORAGE_BINDING) {
        wgpuUsage |= WGPUTextureUsage_StorageBinding;
    }
    if (usage & GFX_TEXTURE_USAGE_RENDER_ATTACHMENT) {
        wgpuUsage |= WGPUTextureUsage_RenderAttachment;
    }
    return wgpuUsage;
}

GfxTextureUsageFlags wgpuTextureUsageToGfxTextureUsage(WGPUTextureUsage usage)
{
    uint32_t gfxUsage = 0;
    if (usage & WGPUTextureUsage_CopySrc) {
        gfxUsage |= GFX_TEXTURE_USAGE_COPY_SRC;
    }
    if (usage & WGPUTextureUsage_CopyDst) {
        gfxUsage |= GFX_TEXTURE_USAGE_COPY_DST;
    }
    if (usage & WGPUTextureUsage_TextureBinding) {
        gfxUsage |= GFX_TEXTURE_USAGE_TEXTURE_BINDING;
    }
    if (usage & WGPUTextureUsage_StorageBinding) {
        gfxUsage |= GFX_TEXTURE_USAGE_STORAGE_BINDING;
    }
    if (usage & WGPUTextureUsage_RenderAttachment) {
        gfxUsage |= GFX_TEXTURE_USAGE_RENDER_ATTACHMENT;
    }
    return gfxUsage;
}

WGPUAddressMode gfxAddressModeToWGPU(GfxAddressMode mode)
{
    switch (mode) {
    case GFX_ADDRESS_MODE_REPEAT:
        return WGPUAddressMode_Repeat;
    case GFX_ADDRESS_MODE_MIRROR_REPEAT:
        return WGPUAddressMode_MirrorRepeat;
    case GFX_ADDRESS_MODE_CLAMP_TO_EDGE:
        return WGPUAddressMode_ClampToEdge;
    default:
        return WGPUAddressMode_Undefined;
    }
}

WGPUFilterMode gfxFilterModeToWGPU(GfxFilterMode mode)
{
    return (mode == GFX_FILTER_MODE_LINEAR) ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
}

WGPUMipmapFilterMode gfxMipmapFilterModeToWGPU(GfxFilterMode mode)
{
    return (mode == GFX_FILTER_MODE_LINEAR) ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
}

WGPUPrimitiveTopology gfxPrimitiveTopologyToWGPU(GfxPrimitiveTopology topology)
{
    switch (topology) {
    case GFX_PRIMITIVE_TOPOLOGY_POINT_LIST:
        return WGPUPrimitiveTopology_PointList;
    case GFX_PRIMITIVE_TOPOLOGY_LINE_LIST:
        return WGPUPrimitiveTopology_LineList;
    case GFX_PRIMITIVE_TOPOLOGY_LINE_STRIP:
        return WGPUPrimitiveTopology_LineStrip;
    case GFX_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
        return WGPUPrimitiveTopology_TriangleList;
    case GFX_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
        return WGPUPrimitiveTopology_TriangleStrip;
    default:
        return WGPUPrimitiveTopology_Undefined;
    }
}

WGPUFrontFace gfxFrontFaceToWGPU(GfxFrontFace frontFace)
{
    return (frontFace == GFX_FRONT_FACE_COUNTER_CLOCKWISE) ? WGPUFrontFace_CCW : WGPUFrontFace_CW;
}

WGPUCullMode gfxCullModeToWGPU(GfxCullMode cullMode)
{
    switch (cullMode) {
    case GFX_CULL_MODE_NONE:
        return WGPUCullMode_None;
    case GFX_CULL_MODE_FRONT:
        return WGPUCullMode_Front;
    case GFX_CULL_MODE_BACK:
        return WGPUCullMode_Back;
    default:
        return WGPUCullMode_Undefined;
    }
}

WGPUIndexFormat gfxIndexFormatToWGPU(GfxIndexFormat format)
{
    switch (format) {
    case GFX_INDEX_FORMAT_UINT16:
        return WGPUIndexFormat_Uint16;
    case GFX_INDEX_FORMAT_UINT32:
        return WGPUIndexFormat_Uint32;
    case GFX_INDEX_FORMAT_UNDEFINED:
    default:
        return WGPUIndexFormat_Undefined;
    }
}

WGPUVertexStepMode gfxVertexStepModeToWGPU(GfxVertexStepMode mode)
{
    switch (mode) {
    case GFX_VERTEX_STEP_MODE_INSTANCE:
        return WGPUVertexStepMode_Instance;
    case GFX_VERTEX_STEP_MODE_VERTEX:
    default:
        return WGPUVertexStepMode_Vertex;
    }
}

WGPUBlendOperation gfxBlendOperationToWGPU(GfxBlendOperation operation)
{
    switch (operation) {
    case GFX_BLEND_OPERATION_ADD:
        return WGPUBlendOperation_Add;
    case GFX_BLEND_OPERATION_SUBTRACT:
        return WGPUBlendOperation_Subtract;
    case GFX_BLEND_OPERATION_REVERSE_SUBTRACT:
        return WGPUBlendOperation_ReverseSubtract;
    case GFX_BLEND_OPERATION_MIN:
        return WGPUBlendOperation_Min;
    case GFX_BLEND_OPERATION_MAX:
        return WGPUBlendOperation_Max;
    default:
        return WGPUBlendOperation_Undefined;
    }
}

WGPUBlendFactor gfxBlendFactorToWGPU(GfxBlendFactor factor)
{
    switch (factor) {
    case GFX_BLEND_FACTOR_ZERO:
        return WGPUBlendFactor_Zero;
    case GFX_BLEND_FACTOR_ONE:
        return WGPUBlendFactor_One;
    case GFX_BLEND_FACTOR_SRC:
        return WGPUBlendFactor_Src;
    case GFX_BLEND_FACTOR_ONE_MINUS_SRC:
        return WGPUBlendFactor_OneMinusSrc;
    case GFX_BLEND_FACTOR_SRC_ALPHA:
        return WGPUBlendFactor_SrcAlpha;
    case GFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
        return WGPUBlendFactor_OneMinusSrcAlpha;
    case GFX_BLEND_FACTOR_DST:
        return WGPUBlendFactor_Dst;
    case GFX_BLEND_FACTOR_ONE_MINUS_DST:
        return WGPUBlendFactor_OneMinusDst;
    case GFX_BLEND_FACTOR_DST_ALPHA:
        return WGPUBlendFactor_DstAlpha;
    case GFX_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
        return WGPUBlendFactor_OneMinusDstAlpha;
    case GFX_BLEND_FACTOR_SRC_ALPHA_SATURATED:
        return WGPUBlendFactor_SrcAlphaSaturated;
    case GFX_BLEND_FACTOR_CONSTANT:
        return WGPUBlendFactor_Constant;
    case GFX_BLEND_FACTOR_ONE_MINUS_CONSTANT:
        return WGPUBlendFactor_OneMinusConstant;
    default:
        return WGPUBlendFactor_Zero;
    }
}

WGPUCompareFunction gfxCompareFunctionToWGPU(GfxCompareFunction func)
{
    switch (func) {
    case GFX_COMPARE_FUNCTION_NEVER:
        return WGPUCompareFunction_Never;
    case GFX_COMPARE_FUNCTION_LESS:
        return WGPUCompareFunction_Less;
    case GFX_COMPARE_FUNCTION_EQUAL:
        return WGPUCompareFunction_Equal;
    case GFX_COMPARE_FUNCTION_LESS_EQUAL:
        return WGPUCompareFunction_LessEqual;
    case GFX_COMPARE_FUNCTION_GREATER:
        return WGPUCompareFunction_Greater;
    case GFX_COMPARE_FUNCTION_NOT_EQUAL:
        return WGPUCompareFunction_NotEqual;
    case GFX_COMPARE_FUNCTION_GREATER_EQUAL:
        return WGPUCompareFunction_GreaterEqual;
    case GFX_COMPARE_FUNCTION_ALWAYS:
        return WGPUCompareFunction_Always;
    default:
        return WGPUCompareFunction_Undefined;
    }
}

WGPUStencilOperation gfxStencilOperationToWGPU(GfxStencilOperation op)
{
    switch (op) {
    case GFX_STENCIL_OPERATION_KEEP:
        return WGPUStencilOperation_Keep;
    case GFX_STENCIL_OPERATION_ZERO:
        return WGPUStencilOperation_Zero;
    case GFX_STENCIL_OPERATION_REPLACE:
        return WGPUStencilOperation_Replace;
    case GFX_STENCIL_OPERATION_INVERT:
        return WGPUStencilOperation_Invert;
    case GFX_STENCIL_OPERATION_INCREMENT_CLAMP:
        return WGPUStencilOperation_IncrementClamp;
    case GFX_STENCIL_OPERATION_DECREMENT_CLAMP:
        return WGPUStencilOperation_DecrementClamp;
    case GFX_STENCIL_OPERATION_INCREMENT_WRAP:
        return WGPUStencilOperation_IncrementWrap;
    case GFX_STENCIL_OPERATION_DECREMENT_WRAP:
        return WGPUStencilOperation_DecrementWrap;
    default:
        return WGPUStencilOperation_Undefined;
    }
}

WGPUStorageTextureAccess gfxStorageTextureAccessToWGPU(GfxStorageTextureAccess access)
{
    switch (access) {
    case GFX_STORAGE_TEXTURE_ACCESS_WRITE_ONLY:
        return WGPUStorageTextureAccess_WriteOnly;
    case GFX_STORAGE_TEXTURE_ACCESS_READ_ONLY:
        return WGPUStorageTextureAccess_ReadOnly;
    case GFX_STORAGE_TEXTURE_ACCESS_READ_WRITE:
        return WGPUStorageTextureAccess_ReadWrite;
    default:
        return WGPUStorageTextureAccess_WriteOnly;
    }
}

WGPUBufferBindingType gfxStorageBufferAccessToWGPU(GfxStorageBufferAccess access)
{
    switch (access) {
    case GFX_STORAGE_BUFFER_ACCESS_READ_WRITE:
        return WGPUBufferBindingType_Storage;
    case GFX_STORAGE_BUFFER_ACCESS_READ_ONLY:
        return WGPUBufferBindingType_ReadOnlyStorage;
    default:
        return WGPUBufferBindingType_ReadOnlyStorage;
    }
}

WGPUSamplerBindingType gfxSamplerBindingTypeToWGPU(GfxSamplerBindingType type)
{
    switch (type) {
    case GFX_SAMPLER_BINDING_TYPE_FILTERING:
        return WGPUSamplerBindingType_Filtering;
    case GFX_SAMPLER_BINDING_TYPE_NON_FILTERING:
        return WGPUSamplerBindingType_NonFiltering;
    case GFX_SAMPLER_BINDING_TYPE_COMPARISON:
        return WGPUSamplerBindingType_Comparison;
    default:
        return WGPUSamplerBindingType_Filtering;
    }
}

WGPUTextureSampleType gfxTextureSampleTypeToWGPU(GfxTextureSampleType sampleType)
{
    switch (sampleType) {
    case GFX_TEXTURE_SAMPLE_TYPE_FLOAT:
        return WGPUTextureSampleType_Float;
    case GFX_TEXTURE_SAMPLE_TYPE_UNFILTERABLE_FLOAT:
        return WGPUTextureSampleType_UnfilterableFloat;
    case GFX_TEXTURE_SAMPLE_TYPE_DEPTH:
        return WGPUTextureSampleType_Depth;
    case GFX_TEXTURE_SAMPLE_TYPE_SINT:
        return WGPUTextureSampleType_Sint;
    case GFX_TEXTURE_SAMPLE_TYPE_UINT:
        return WGPUTextureSampleType_Uint;
    default:
        return WGPUTextureSampleType_Undefined;
    }
}

WGPUVertexFormat gfxFormatToWGPUVertexFormat(GfxFormat format)
{
    switch (format) {
    case GFX_FORMAT_R32_FLOAT:
        return WGPUVertexFormat_Float32;
    case GFX_FORMAT_R32G32_FLOAT:
        return WGPUVertexFormat_Float32x2;
    case GFX_FORMAT_R32G32B32_FLOAT:
        return WGPUVertexFormat_Float32x3;
    case GFX_FORMAT_R32G32B32A32_FLOAT:
        return WGPUVertexFormat_Float32x4;
    case GFX_FORMAT_R16G16_FLOAT:
        return WGPUVertexFormat_Float16x2;
    case GFX_FORMAT_R16G16B16A16_FLOAT:
        return WGPUVertexFormat_Float16x4;
    case GFX_FORMAT_R8G8B8A8_UNORM:
        return WGPUVertexFormat_Unorm8x4;
    case GFX_FORMAT_R8G8B8A8_UNORM_SRGB:
        return WGPUVertexFormat_Unorm8x4;
    case GFX_FORMAT_R8_UINT:
        return WGPUVertexFormat_Uint8;
    case GFX_FORMAT_R8G8_UINT:
        return WGPUVertexFormat_Uint8x2;
    case GFX_FORMAT_R8G8B8A8_UINT:
        return WGPUVertexFormat_Uint8x4;
    case GFX_FORMAT_R8_SINT:
        return WGPUVertexFormat_Sint8;
    case GFX_FORMAT_R8G8_SINT:
        return WGPUVertexFormat_Sint8x2;
    case GFX_FORMAT_R8G8B8A8_SINT:
        return WGPUVertexFormat_Sint8x4;
    case GFX_FORMAT_R16_UINT:
        return WGPUVertexFormat_Uint16;
    case GFX_FORMAT_R16G16_UINT:
        return WGPUVertexFormat_Uint16x2;
    case GFX_FORMAT_R16G16B16A16_UINT:
        return WGPUVertexFormat_Uint16x4;
    case GFX_FORMAT_R16_SINT:
        return WGPUVertexFormat_Sint16;
    case GFX_FORMAT_R16G16_SINT:
        return WGPUVertexFormat_Sint16x2;
    case GFX_FORMAT_R16G16B16A16_SINT:
        return WGPUVertexFormat_Sint16x4;
    case GFX_FORMAT_R32_UINT:
        return WGPUVertexFormat_Uint32;
    case GFX_FORMAT_R32G32_UINT:
        return WGPUVertexFormat_Uint32x2;
    case GFX_FORMAT_R32G32B32A32_UINT:
        return WGPUVertexFormat_Uint32x4;
    case GFX_FORMAT_R32_SINT:
        return WGPUVertexFormat_Sint32;
    case GFX_FORMAT_R32G32_SINT:
        return WGPUVertexFormat_Sint32x2;
    case GFX_FORMAT_R32G32B32A32_SINT:
        return WGPUVertexFormat_Sint32x4;
    default:
        return static_cast<WGPUVertexFormat>(0);
    }
}

WGPUTextureDimension gfxTextureTypeToWGPUTextureDimension(GfxTextureType type)
{
    switch (type) {
    case GFX_TEXTURE_TYPE_1D:
        return WGPUTextureDimension_1D;
    case GFX_TEXTURE_TYPE_2D:
        return WGPUTextureDimension_2D;
    case GFX_TEXTURE_TYPE_CUBE:
        return WGPUTextureDimension_2D; // Cube maps are 2D arrays in WebGPU
    case GFX_TEXTURE_TYPE_3D:
        return WGPUTextureDimension_3D;
    default:
        return WGPUTextureDimension_2D;
    }
}

GfxTextureType wgpuTextureDimensionToGfxTextureType(WGPUTextureDimension dimension)
{
    switch (dimension) {
    case WGPUTextureDimension_1D:
        return GFX_TEXTURE_TYPE_1D;
    case WGPUTextureDimension_2D:
        return GFX_TEXTURE_TYPE_2D; // Note: Can't distinguish CUBE from this alone
    case WGPUTextureDimension_3D:
        return GFX_TEXTURE_TYPE_3D;
    default:
        return GFX_TEXTURE_TYPE_2D;
    }
}

WGPUTextureViewDimension gfxTextureViewTypeToWGPU(GfxTextureViewType type)
{
    switch (type) {
    case GFX_TEXTURE_VIEW_TYPE_1D:
        return WGPUTextureViewDimension_1D;
    case GFX_TEXTURE_VIEW_TYPE_2D:
        return WGPUTextureViewDimension_2D;
    case GFX_TEXTURE_VIEW_TYPE_3D:
        return WGPUTextureViewDimension_3D;
    case GFX_TEXTURE_VIEW_TYPE_CUBE:
        return WGPUTextureViewDimension_Cube;
    case GFX_TEXTURE_VIEW_TYPE_1D_ARRAY:
        return WGPUTextureViewDimension_1D; // WebGPU doesn't have 1D arrays
    case GFX_TEXTURE_VIEW_TYPE_2D_ARRAY:
        return WGPUTextureViewDimension_2DArray;
    case GFX_TEXTURE_VIEW_TYPE_CUBE_ARRAY:
        return WGPUTextureViewDimension_CubeArray;
    default:
        return WGPUTextureViewDimension_Undefined;
    }
}

WGPUTextureAspect gfxTextureAspectToWGPUTextureAspect(GfxTextureAspect aspect)
{
    switch (aspect) {
    case GFX_TEXTURE_ASPECT_DEPTH_ONLY:
        return WGPUTextureAspect_DepthOnly;
    case GFX_TEXTURE_ASPECT_STENCIL_ONLY:
        return WGPUTextureAspect_StencilOnly;
    case GFX_TEXTURE_ASPECT_ALL:
    default:
        return WGPUTextureAspect_All;
    }
}

WGPUOrigin3D gfxOrigin3DToWGPUOrigin3D(const GfxOrigin3D* origin)
{
    if (!origin) {
        return { 0, 0, 0 };
    }
    return { static_cast<uint32_t>(origin->x), static_cast<uint32_t>(origin->y), static_cast<uint32_t>(origin->z) };
}

WGPUExtent3D gfxExtent3DToWGPUExtent3D(const GfxExtent3D* extent)
{
    if (!extent) {
        return { 0, 0, 0 };
    }
    return { extent->width, extent->height, extent->depth };
}

GfxExtent3D wgpuExtent3DToGfxExtent3D(const WGPUExtent3D& extent)
{
    return { extent.width, extent.height, extent.depthOrArrayLayers };
}

namespace {
    // WebGPU render passes store ops/formats only; views come from the framebuffer at begin time
    RenderPassColorAttachment convertColorAttachment(const GfxRenderPassColorAttachmentTarget& target)
    {
        RenderPassColorAttachment attachment{};
        attachment.format = gfxFormatToWGPUFormat(target.format);
        attachment.loadOp = gfxLoadOpToWGPULoadOp(target.ops.loadOp);
        attachment.storeOp = gfxStoreOpToWGPUStoreOp(target.ops.storeOp);
        return attachment;
    }

    RenderPassDepthStencilAttachment convertDepthStencilAttachment(const GfxRenderPassDepthStencilAttachmentTarget& target)
    {
        RenderPassDepthStencilAttachment attachment{};
        attachment.format = gfxFormatToWGPUFormat(target.format);
        attachment.depthLoadOp = gfxLoadOpToWGPULoadOp(target.depthOps.loadOp);
        attachment.depthStoreOp = gfxStoreOpToWGPUStoreOp(target.depthOps.storeOp);
        attachment.stencilLoadOp = gfxLoadOpToWGPULoadOp(target.stencilOps.loadOp);
        attachment.stencilStoreOp = gfxStoreOpToWGPUStoreOp(target.stencilOps.storeOp);
        return attachment;
    }
} // anonymous namespace

core::RenderPassCreateInfo gfxRenderPassDescriptorToRenderPassCreateInfo(const GfxRenderPassDescriptor* descriptor)
{
    core::RenderPassCreateInfo createInfo{};

    for (uint32_t i = 0; i < descriptor->colorAttachmentCount; ++i) {
        createInfo.colorAttachments.push_back(convertColorAttachment(descriptor->colorAttachments[i].target));
    }

    // Store sample count from first color attachment (all must match)
    if (descriptor->colorAttachmentCount > 0) {
        createInfo.sampleCount = static_cast<uint32_t>(descriptor->colorAttachments[0].target.sampleCount);
    } else if (descriptor->depthStencilAttachment) {
        createInfo.sampleCount = static_cast<uint32_t>(descriptor->depthStencilAttachment->target.sampleCount);
    }

    if (descriptor->depthStencilAttachment) {
        createInfo.depthStencilAttachment = convertDepthStencilAttachment(descriptor->depthStencilAttachment->target);
    }

    return createInfo;
}

core::FramebufferCreateInfo gfxFramebufferDescriptorToFramebufferCreateInfo(const GfxFramebufferDescriptor* descriptor)
{
    core::FramebufferCreateInfo createInfo{};

    // Convert color attachment views and resolve targets - store pointers
    for (uint32_t i = 0; i < descriptor->colorAttachmentCount; ++i) {
        const GfxFramebufferAttachment& colorAtt = descriptor->colorAttachments[i];

        auto* view = toNative<TextureView>(colorAtt.view);
        createInfo.colorAttachmentViews.push_back(view); // Store pointer

        // Add resolve target if provided
        if (colorAtt.resolveTarget) {
            auto* resolveView = toNative<TextureView>(colorAtt.resolveTarget);
            createInfo.colorResolveTargetViews.push_back(resolveView); // Store pointer
        } else {
            createInfo.colorResolveTargetViews.push_back(nullptr);
        }
    }

    // Convert depth/stencil attachment view if present
    if (descriptor->depthStencilAttachment.view) {
        auto* view = toNative<TextureView>(descriptor->depthStencilAttachment.view);
        createInfo.depthStencilAttachmentView = view; // Store pointer

        // Convert depth/stencil resolve target if present
        if (descriptor->depthStencilAttachment.resolveTarget) {
            auto* resolveView = toNative<TextureView>(descriptor->depthStencilAttachment.resolveTarget);
            createInfo.depthStencilResolveTargetView = resolveView; // Store pointer
        }
    }

    createInfo.width = descriptor->extent.width;
    createInfo.height = descriptor->extent.height;

    return createInfo;
}

core::RenderPassEncoderBeginInfo gfxRenderPassBeginDescriptorToBeginInfo(const GfxRenderPassBeginDescriptor* descriptor)
{
    core::RenderPassEncoderBeginInfo beginInfo{};

    // Convert color clear values
    for (uint32_t i = 0; i < descriptor->colorClearValueCount; ++i) {
        const GfxColor& color = descriptor->colorClearValues[i];
        beginInfo.colorClearValues.push_back({ color.r, color.g, color.b, color.a });
    }

    beginInfo.depthClearValue = descriptor->depthClearValue;
    beginInfo.stencilClearValue = descriptor->stencilClearValue;

    beginInfo.occlusionQuerySet = nullptr;
    if (descriptor->occlusionQuerySet) {
        auto* querySet = toNative<core::QuerySet>(descriptor->occlusionQuerySet);
        beginInfo.occlusionQuerySet = querySet->handle();
    }

    beginInfo.timestampQuerySet = nullptr;
    if (descriptor->timestampQuerySet) {
        auto* querySet = toNative<core::QuerySet>(descriptor->timestampQuerySet);
        beginInfo.timestampQuerySet = querySet->handle();
    }

    return beginInfo;
}

core::ComputePassEncoderCreateInfo gfxComputePassBeginDescriptorToCreateInfo(const GfxComputePassBeginDescriptor* descriptor)
{
    core::ComputePassEncoderCreateInfo createInfo{};
    createInfo.label = descriptor->label;
    return createInfo;
}

} // namespace gfx::backend::webgpu::converter
