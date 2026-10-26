#include "../system/Device.h"

#include "Adapter.h"
#include "Instance.h"
#include "Queue.h"

#include "../util/Blit.h"
#include "../util/Utils.h"

#include "common/Logger.h"
#include "common/Strings.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gfx::backend::webgpu::core {

namespace {
    uint64_t parseEnabledExtensions(const std::vector<std::string>& enabledExtensions)
    {
        const std::pair<const char*, DeviceExtension> knownExtensions[] = {
            { extensions::TIMELINE_SEMAPHORE, DeviceExtension::TimelineSemaphore },
            { extensions::TIMESTAMP_QUERY, DeviceExtension::TimestampQuery },
        };

        uint64_t mask = 0;
        for (const auto& [name, bit] : knownExtensions) {
            if (gfx::common::containsString(enabledExtensions, name)) {
                mask |= static_cast<uint64_t>(bit);
            }
        }
        return mask;
    }

    bool hasExtension(uint64_t enabledExtensions, DeviceExtension extension)
    {
        return (enabledExtensions & static_cast<uint64_t>(extension)) != 0;
    }

    // A device can only enable the extensions its adapter supports
    void requireSupportedExtensions(const Adapter* adapter, const std::vector<std::string>& enabledExtensions)
    {
        const std::vector<const char*> supportedExtensions = adapter->enumerateSupportedExtensions();
        for (const std::string& extension : enabledExtensions) {
            if (!gfx::common::containsString(supportedExtensions, extension.c_str())) {
                throw std::runtime_error("Device extension not supported by this adapter: " + extension);
            }
        }
    }

    // The WGPU features of the requested GFX extensions, plus the optional texture formats the adapter has
    std::vector<WGPUFeatureName> collectRequiredFeatures(WGPUAdapter adapter, uint64_t enabledExtensions)
    {
        std::vector<WGPUFeatureName> requiredFeatures;
        if (hasExtension(enabledExtensions, DeviceExtension::TimestampQuery)) {
            requiredFeatures.push_back(WGPUFeatureName_TimestampQuery);
        }

        const WGPUFeatureName formatFeatures[] = {
            WGPUFeatureName_TextureCompressionBC,
            WGPUFeatureName_TextureCompressionETC2,
            WGPUFeatureName_TextureCompressionASTC,
            WGPUFeatureName_Depth32FloatStencil8,
        };
        for (WGPUFeatureName feature : formatFeatures) {
            if (wgpuAdapterHasFeature(adapter, feature)) {
                requiredFeatures.push_back(feature);
            }
        }
        return requiredFeatures;
    }

    // Issues wgpuAdapterRequestDevice and synchronously waits for the result, throwing on failure
    WGPUDevice requestDeviceSync(Adapter* adapter, const WGPUDeviceDescriptor& wgpuDesc)
    {
        WGPUDevice device = nullptr;

        WGPURequestDeviceCallbackInfo callbackInfo = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
        callbackInfo.mode = WGPUCallbackMode_WaitAnyOnly;
        callbackInfo.callback = [](WGPURequestDeviceStatus status, WGPUDevice deviceResult, WGPUStringView message, void* userdata1, void*) {
            auto* devicePtr = static_cast<WGPUDevice*>(userdata1);
            if (status == WGPURequestDeviceStatus_Success && deviceResult) {
                *devicePtr = deviceResult;
            } else if (message.data) {
                gfx::common::Logger::instance().logError("Error: Failed to request device: {}",
                    std::string_view(message.data, message.length));
            }
        };
        callbackInfo.userdata1 = &device;

        WGPUFuture future = wgpuAdapterRequestDevice(adapter->handle(), &wgpuDesc, callbackInfo);
        if (!waitForFuture(adapter->getInstance()->handle(), future, UINT64_MAX) || !device) {
            throw std::runtime_error("Failed to request device");
        }
        return device;
    }
} // anonymous namespace

Device::Device(Adapter* adapter, const DeviceCreateInfo& createInfo)
    : m_device(nullptr)
    , m_adapter(adapter)
    , m_enabledExtensions(parseEnabledExtensions(createInfo.enabledExtensions))
{
    if (!adapter) {
        throw std::runtime_error("Invalid adapter for device creation");
    }
    requireSupportedExtensions(adapter, createInfo.enabledExtensions);

    WGPUUncapturedErrorCallbackInfo errorCallbackInfo = WGPU_UNCAPTURED_ERROR_CALLBACK_INFO_INIT;
    errorCallbackInfo.callback = [](WGPUDevice const*, WGPUErrorType type, WGPUStringView message, void*, void*) {
        gfx::common::Logger::instance().logError("[WebGPU Uncaptured Error] Type: {}, Message: {}", static_cast<int>(type), std::string_view(message.data, message.length));
    };

    WGPUDeviceLostCallbackInfo deviceLostCallbackInfo = WGPU_DEVICE_LOST_CALLBACK_INFO_INIT;
    deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    deviceLostCallbackInfo.callback = [](WGPUDevice const*, WGPUDeviceLostReason reason, WGPUStringView message, void*, void*) {
        gfx::common::Logger::instance().logError("[WebGPU Device Lost] Reason: {}, Message: {}", static_cast<int>(reason), std::string_view(message.data, message.length));
    };

    WGPUDeviceDescriptor wgpuDesc = WGPU_DEVICE_DESCRIPTOR_INIT;
    wgpuDesc.uncapturedErrorCallbackInfo = errorCallbackInfo;
    wgpuDesc.deviceLostCallbackInfo = deviceLostCallbackInfo;

    // Request the adapter's full limits instead of the conservative WebGPU defaults,
    // so device limits match adapter limits - the same semantics as Vulkan, where a
    // logical device always has the physical device's capabilities
    WGPULimits adapterLimits = adapter->getLimits();
    wgpuDesc.requiredLimits = &adapterLimits;

    std::vector<WGPUFeatureName> requiredFeatures = collectRequiredFeatures(adapter->handle(), m_enabledExtensions);

#ifndef __EMSCRIPTEN__
    // Enable Dawn toggles to allow SPIR-V at device level (Dawn-specific)
    static const char* enabledToggles[] = { "allow_unsafe_apis" };
    static const char* disabledToggles[] = { "disallow_spirv" };
    WGPUDawnTogglesDescriptor deviceTogglesDesc = {};
    deviceTogglesDesc.chain.sType = WGPUSType_DawnTogglesDescriptor;
    deviceTogglesDesc.enabledToggleCount = 1;
    deviceTogglesDesc.enabledToggles = enabledToggles;
    deviceTogglesDesc.disabledToggleCount = 1;
    deviceTogglesDesc.disabledToggles = disabledToggles;

    gfx::common::Logger::instance().logDebug("WebGPU Device: Enabling Dawn toggles - allow_unsafe_apis (enabled), disallow_spirv (disabled)");

    wgpuDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&deviceTogglesDesc);

    // Dawn native is NOT thread-safe by default; enable implicit device synchronization
    // so queue operations are internally synchronized as the API threading contract promises
    if (wgpuAdapterHasFeature(adapter->handle(), WGPUFeatureName_ImplicitDeviceSynchronization)) {
        requiredFeatures.push_back(WGPUFeatureName_ImplicitDeviceSynchronization);
    } else {
        gfx::common::Logger::instance().logWarning(
            "WebGPU adapter does not support implicit device synchronization - "
            "concurrent queue operations from multiple threads are NOT safe on this device");
    }
#endif

    if (!requiredFeatures.empty()) {
        wgpuDesc.requiredFeatures = requiredFeatures.data();
        wgpuDesc.requiredFeatureCount = requiredFeatures.size();
    }

    m_device = requestDeviceSync(adapter, wgpuDesc);

    // Create queue
    WGPUQueue wgpuQueue = wgpuDeviceGetQueue(m_device);
    if (!wgpuQueue) {
        throw std::runtime_error("Failed to get default queue from WGPUDevice");
    }
    m_queue = std::make_unique<Queue>(wgpuQueue, this);

    // Create blit helper
    m_blit = std::make_unique<Blit>(m_device);
}

Device::~Device()
{
    if (m_device) {
        // Release queue first
        m_queue.reset();
        // Destroy device to ensure proper cleanup of internal resources
        wgpuDeviceDestroy(m_device);
        wgpuDeviceRelease(m_device);
    }
}

WGPUDevice Device::handle() const
{
    return m_device;
}

Queue* Device::getQueue()
{
    return m_queue.get();
}

Adapter* Device::getAdapter()
{
    return m_adapter;
}

WGPULimits Device::getLimits() const
{
    WGPULimits limits = WGPU_LIMITS_INIT;
    WGPUStatus status = wgpuDeviceGetLimits(m_device, &limits);
    if (status != WGPUStatus_Success) {
        throw std::runtime_error("Failed to get device limits");
    }
    return limits;
}

void Device::waitIdle() const
{
    m_queue->waitIdle();
}

Blit* Device::getBlit()
{
    return m_blit.get();
}

bool Device::supportsShaderFormat(ShaderSourceType format) const
{
#ifdef __EMSCRIPTEN__
    // Emscripten (WebAssembly) only supports WGSL, SPIR-V doesn't work
    return format == ShaderSourceType::WGSL;
#else
    // Native WebGPU (Dawn) supports both SPIR-V and WGSL
    return format == ShaderSourceType::SPIRV || format == ShaderSourceType::WGSL;
#endif
}

bool Device::isExtensionEnabled(DeviceExtension extension) const
{
    return hasExtension(m_enabledExtensions, extension);
}

} // namespace gfx::backend::webgpu::core