#include "../resource/Buffer.h"

#include "../system/Adapter.h"
#include "../system/Device.h"
#include "../system/Instance.h"

#include <stdexcept>

namespace gfx::backend::webgpu::core {

Buffer::Buffer(Device* device, const BufferCreateInfo& createInfo)
    : m_device(device)
    , m_ownsResources(true)
    , m_buffer(nullptr)
    , m_info(createBufferInfo(createInfo))
{
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.size = m_info.size;
    desc.usage = m_info.usage;
    desc.mappedAtCreation = false;

    m_buffer = wgpuDeviceCreateBuffer(m_device->handle(), &desc);
    if (!m_buffer) {
        throw std::runtime_error("Failed to create WebGPU buffer");
    }
}

Buffer::Buffer(Device* device, WGPUBuffer buffer, const BufferImportInfo& importInfo)
    : m_device(device)
    , m_ownsResources(false)
    , m_buffer(buffer)
    , m_info(createBufferInfo(importInfo))
{
}

Buffer::~Buffer()
{
    if (m_ownsResources && m_buffer) {
        wgpuBufferRelease(m_buffer);
    }
}

WGPUBuffer Buffer::handle() const
{
    return m_buffer;
}

uint64_t Buffer::getSize() const
{
    return m_info.size;
}

WGPUBufferUsage Buffer::getUsage() const
{
    return m_info.usage;
}

const BufferInfo& Buffer::getInfo() const
{
    return m_info;
}

Device* Buffer::getDevice() const
{
    return m_device;
}

Buffer::MapStatus Buffer::map(uint64_t offset, uint64_t size, void** outPointer, uint64_t timeoutNs)
{
    *outPointer = nullptr;

    if (m_mapped) {
        *outPointer = m_mappedPointer;
        return MapStatus::Ready;
    }

    WGPUMapMode mapMode = WGPUMapMode_None;
    if (m_info.usage & WGPUBufferUsage_MapRead) {
        mapMode |= WGPUMapMode_Read;
    }
    if (m_info.usage & WGPUBufferUsage_MapWrite) {
        mapMode |= WGPUMapMode_Write;
    }
    if (mapMode == WGPUMapMode_None || offset > m_info.size) {
        return MapStatus::Failed;
    }

    uint64_t mapSize = size;
    if (mapSize == UINT64_MAX) {
        mapSize = m_info.size - offset;
    }
    if (offset + mapSize > m_info.size) {
        return MapStatus::Failed;
    }

    if (!m_asyncMapPending) {
        m_asyncCallbackData = {};
        m_asyncCallbackData.offset = offset;
        m_asyncCallbackData.size = mapSize;

        WGPUBufferMapCallbackInfo callbackInfo = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        callbackInfo.mode = WGPUCallbackMode_WaitAnyOnly;
        callbackInfo.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
            auto* data = static_cast<AsyncMapCallbackData*>(userdata1);
            data->status = status;
            data->completed = true;
        };
        callbackInfo.userdata1 = &m_asyncCallbackData;

        m_asyncMapFuture = wgpuBufferMapAsync(m_buffer, mapMode, offset, mapSize, callbackInfo);
        m_asyncMapPending = true;
    }

    WGPUFutureWaitInfo waitInfo = WGPU_FUTURE_WAIT_INFO_INIT;
    waitInfo.future = m_asyncMapFuture;
    wgpuInstanceWaitAny(m_device->getAdapter()->getInstance()->handle(), 1, &waitInfo, timeoutNs);
    if (!m_asyncCallbackData.completed) {
        return MapStatus::Pending;
    }
    m_asyncMapPending = false;
    if (m_asyncCallbackData.status != WGPUMapAsyncStatus_Success) {
        return MapStatus::Failed;
    }

    if (mapMode & WGPUMapMode_Read) {
        m_mappedPointer = const_cast<void*>(wgpuBufferGetConstMappedRange(m_buffer, m_asyncCallbackData.offset, m_asyncCallbackData.size));
    } else {
        m_mappedPointer = wgpuBufferGetMappedRange(m_buffer, m_asyncCallbackData.offset, m_asyncCallbackData.size);
    }
    if (!m_mappedPointer) {
        return MapStatus::Failed;
    }
    m_mapped = true;
    *outPointer = m_mappedPointer;
    return MapStatus::Ready;
}

void Buffer::unmap()
{
    wgpuBufferUnmap(m_buffer);
    m_asyncMapPending = false;
    m_mapped = false;
    m_mappedPointer = nullptr;
    m_asyncCallbackData = {};
}

void Buffer::flushMappedRange(uint64_t offset, uint64_t size)
{
    // WebGPU memory is always coherent - no-op
    (void)offset;
    (void)size;
}

void Buffer::invalidateMappedRange(uint64_t offset, uint64_t size)
{
    // WebGPU memory is always coherent - no-op
    (void)offset;
    (void)size;
}

BufferInfo Buffer::createBufferInfo(const BufferCreateInfo& createInfo)
{
    BufferInfo info{};
    info.size = createInfo.size;
    info.usage = createInfo.usage;
    info.memoryProperties = createInfo.memoryProperties;
    return info;
}

BufferInfo Buffer::createBufferInfo(const BufferImportInfo& importInfo)
{
    BufferInfo info{};
    info.size = importInfo.size;
    info.usage = importInfo.usage;
    info.memoryProperties = importInfo.memoryProperties;
    return info;
}

} // namespace gfx::backend::webgpu::core