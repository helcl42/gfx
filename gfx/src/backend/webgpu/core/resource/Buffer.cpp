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

// Map buffer for CPU access
// Returns mapped pointer on success, nullptr on failure
void* Buffer::map(uint64_t offset, uint64_t size)
{
    // Determine map mode based on original GFX usage flags (not WebGPU usage)
    // WebGPU usage is derived from GFX usage and has MapRead/MapWrite
    WGPUMapMode mapMode = WGPUMapMode_None;
    if (m_info.usage & WGPUBufferUsage_MapRead) {
        mapMode |= WGPUMapMode_Read;
    }
    if (m_info.usage & WGPUBufferUsage_MapWrite) {
        mapMode |= WGPUMapMode_Write;
    }

    if (mapMode == WGPUMapMode_None) {
        return nullptr;
    }

    if (offset > m_info.size) {
        return nullptr;
    }

    // If size is UINT64_MAX, map the entire buffer from offset
    uint64_t mapSize = size;
    if (mapSize == UINT64_MAX) {
        mapSize = m_info.size - offset;
    }

    if (offset + mapSize > m_info.size) {
        return nullptr;
    }

    // Set up async mapping with synchronous wait
    struct MapCallbackData {
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
        bool completed = false;
    };
    MapCallbackData callbackData;

    WGPUBufferMapCallbackInfo callbackInfo = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    callbackInfo.mode = WGPUCallbackMode_WaitAnyOnly;
    callbackInfo.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*) {
        auto* data = static_cast<MapCallbackData*>(userdata1);
        data->status = status;
        data->completed = true;
    };
    callbackInfo.userdata1 = &callbackData;

    WGPUFuture future = wgpuBufferMapAsync(m_buffer, mapMode, offset, mapSize, callbackInfo);

    // Properly wait for the mapping to complete
    WGPUFutureWaitInfo waitInfo = WGPU_FUTURE_WAIT_INFO_INIT;
    waitInfo.future = future;
    wgpuInstanceWaitAny(m_device->getAdapter()->getInstance()->handle(), 1, &waitInfo, UINT64_MAX);

    if (!callbackData.completed || callbackData.status != WGPUMapAsyncStatus_Success) {
        return nullptr;
    }

    // Get the mapped range. Read-only buffers require wgpuBufferGetConstMappedRange;
    // wgpuBufferGetMappedRange returns null for MapRead buffers.
    void* mappedData;
    if (mapMode & WGPUMapMode_Read) {
        mappedData = const_cast<void*>(wgpuBufferGetConstMappedRange(m_buffer, offset, mapSize));
    } else {
        mappedData = wgpuBufferGetMappedRange(m_buffer, offset, mapSize);
    }

    if (!mappedData) {
        wgpuBufferUnmap(m_buffer);
        return nullptr;
    }

    return mappedData;
}

void Buffer::unmap()
{
    wgpuBufferUnmap(m_buffer);
    m_asyncMapPending = false;
    m_asyncMapped = false;
    m_asyncCallbackData = {};
}

void Buffer::asyncMap(uint64_t offset, uint64_t size)
{
    WGPUMapMode mapMode = WGPUMapMode_None;
    if (m_info.usage & WGPUBufferUsage_MapRead) {
        mapMode |= WGPUMapMode_Read;
    }
    if (m_info.usage & WGPUBufferUsage_MapWrite) {
        mapMode |= WGPUMapMode_Write;
    }

    if (mapMode == WGPUMapMode_None) {
        return;
    }

    uint64_t mapSize = size;
    if (mapSize == UINT64_MAX) {
        mapSize = m_info.size - offset;
    }

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
    m_asyncMapped = false;
}

bool Buffer::isAsyncMapped() const
{
    if (m_asyncMapped) {
        return true;
    }
    if (!m_asyncMapPending) {
        return false;
    }

    // Non-blocking poll with timeout=0
    WGPUFutureWaitInfo waitInfo = WGPU_FUTURE_WAIT_INFO_INIT;
    waitInfo.future = m_asyncMapFuture;
    wgpuInstanceWaitAny(m_device->getAdapter()->getInstance()->handle(), 1, &waitInfo, 0);

    if (waitInfo.completed && m_asyncCallbackData.completed && m_asyncCallbackData.status == WGPUMapAsyncStatus_Success) {
        m_asyncMapped = true;
        m_asyncMapPending = false;
        return true;
    }
    return false;
}

void* Buffer::getAsyncMappedPointer()
{
    if (!m_asyncMapped) {
        return nullptr;
    }

    WGPUMapMode mapMode = WGPUMapMode_None;
    if (m_info.usage & WGPUBufferUsage_MapRead) {
        mapMode |= WGPUMapMode_Read;
    }
    if (m_info.usage & WGPUBufferUsage_MapWrite) {
        mapMode |= WGPUMapMode_Write;
    }

    if (mapMode & WGPUMapMode_Read) {
        return const_cast<void*>(wgpuBufferGetConstMappedRange(m_buffer, m_asyncCallbackData.offset, m_asyncCallbackData.size));
    } else {
        return wgpuBufferGetMappedRange(m_buffer, m_asyncCallbackData.offset, m_asyncCallbackData.size);
    }
}

bool Buffer::waitUntilAsyncMapped(uint64_t timeoutNs)
{
    if (m_asyncMapped) {
        return true;
    }
    if (!m_asyncMapPending) {
        return false;
    }

    WGPUFutureWaitInfo waitInfo = WGPU_FUTURE_WAIT_INFO_INIT;
    waitInfo.future = m_asyncMapFuture;
    wgpuInstanceWaitAny(m_device->getAdapter()->getInstance()->handle(), 1, &waitInfo, timeoutNs);

    if (waitInfo.completed && m_asyncCallbackData.completed && m_asyncCallbackData.status == WGPUMapAsyncStatus_Success) {
        m_asyncMapped = true;
        m_asyncMapPending = false;
        return true;
    }
    return false;
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