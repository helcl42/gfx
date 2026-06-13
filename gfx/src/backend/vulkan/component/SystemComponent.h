#ifndef GFX_BACKEND_VULKAN_SYSTEM_COMPONENT_H
#define GFX_BACKEND_VULKAN_SYSTEM_COMPONENT_H

#include <gfx/gfx.h>

namespace gfx::backend::vulkan::component {

class SystemComponent {
public:
    // Instance functions
    GfxResult createInstance(const GfxInstanceDescriptor* descriptor, GfxInstance* outInstance) const;
    GfxResult instanceDestroy(GfxInstance instance) const;
    GfxResult instanceGetNativeHandle(GfxInstance instance, void** outHandle) const;
    GfxResult instanceRequestAdapter(GfxInstance instance, const GfxAdapterDescriptor* descriptor, GfxAdapter* outAdapter) const;
    GfxResult instanceEnumerateAdapters(GfxInstance instance, uint32_t* adapterCount, GfxAdapter* adapters) const;
    GfxResult enumerateInstanceExtensions(uint32_t* extensionCount, const char** extensionNames) const;

    // Adapter functions
    GfxResult adapterCreateDevice(GfxAdapter adapter, const GfxDeviceDescriptor* descriptor, GfxDevice* outDevice) const;
    GfxResult adapterGetNativeHandle(GfxAdapter adapter, void** outHandle) const;
    GfxResult adapterGetInfo(GfxAdapter adapter, GfxAdapterInfo* outInfo) const;
    GfxResult adapterGetLimits(GfxAdapter adapter, GfxDeviceLimits* outLimits) const;
    GfxResult adapterEnumerateQueueFamilies(GfxAdapter adapter, uint32_t* queueFamilyCount, GfxQueueFamilyProperties* queueFamilies) const;
    GfxResult adapterGetQueueFamilySurfaceSupport(GfxAdapter adapter, uint32_t queueFamilyIndex, GfxSurface surface, bool* outSupported) const;
    GfxResult adapterEnumerateExtensions(GfxAdapter adapter, uint32_t* extensionCount, const char** extensionNames) const;

    // Device functions
    GfxResult deviceDestroy(GfxDevice device) const;
    GfxResult deviceGetNativeHandle(GfxDevice device, void** outHandle) const;
    GfxResult deviceGetQueue(GfxDevice device, GfxQueue* outQueue) const;
    GfxResult deviceGetQueueByIndex(GfxDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, GfxQueue* outQueue) const;
    GfxResult deviceWaitIdle(GfxDevice device) const;
    GfxResult deviceGetLimits(GfxDevice device, GfxDeviceLimits* outLimits) const;
    GfxResult deviceSupportsShaderFormat(GfxDevice device, GfxShaderSourceType format, bool* outSupported) const;

    // Queue functions
    GfxResult queueSubmit(GfxQueue queue, const GfxSubmitDescriptor* submitDescriptor) const;
    GfxResult queueGetInfo(GfxQueue queue, GfxQueueInfo* outInfo) const;
    GfxResult queueGetNativeHandle(GfxQueue queue, void** outHandle) const;
    GfxResult queueWriteBuffer(GfxQueue queue, GfxBuffer buffer, uint64_t offset, const void* data, uint64_t size) const;
    GfxResult queueWriteTexture(GfxQueue queue, const GfxWriteTextureDescriptor* descriptor, const void* data, uint64_t dataSize) const;
    GfxResult queueWaitIdle(GfxQueue queue) const;
};

} // namespace gfx::backend::vulkan::component

#endif // GFX_BACKEND_VULKAN_SYSTEM_COMPONENT_H
