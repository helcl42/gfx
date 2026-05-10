#include "Buffer.h"

#include "../../converter/Conversions.h"

#include <stdexcept>

namespace gfx {

BufferImpl::BufferImpl(GfxBuffer h)
    : m_handle(h)
{
    GfxResult result = gfxBufferGetInfo(m_handle, &m_info);
    if (result != GFX_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to get buffer info");
    }
}

BufferImpl::~BufferImpl()
{
    if (m_handle) {
        gfxBufferDestroy(m_handle);
    }
}

GfxBuffer BufferImpl::getHandle() const
{
    return m_handle;
}

BufferInfo BufferImpl::getInfo() const
{
    return cBufferInfoToCppBufferInfo(m_info);
}

void* BufferImpl::getNativeHandle() const
{
    void* handle = nullptr;
    GfxResult result = gfxBufferGetNativeHandle(m_handle, &handle);
    if (result != GFX_RESULT_SUCCESS) {
        return nullptr;
    }
    return handle;
}

void* BufferImpl::map(uint64_t offset, uint64_t size)
{
    void* mappedPointer = nullptr;
    GfxResult result = gfxBufferMap(m_handle, offset, size, &mappedPointer);
    if (result != GFX_RESULT_SUCCESS) {
        return nullptr;
    }
    return mappedPointer;
}

void BufferImpl::unmap()
{
    gfxBufferUnmap(m_handle);
}

void BufferImpl::flushMappedRange(uint64_t offset, uint64_t size)
{
    gfxBufferFlushMappedRange(m_handle, offset, size);
}

void BufferImpl::invalidateMappedRange(uint64_t offset, uint64_t size)
{
    gfxBufferInvalidateMappedRange(m_handle, offset, size);
}

void BufferImpl::asyncMap(uint64_t offset, uint64_t size)
{
    gfxBufferAsyncMap(m_handle, offset, size);
}

bool BufferImpl::isAsyncMapped() const
{
    bool mapped = false;
    gfxBufferIsAsyncMapped(m_handle, &mapped);
    return mapped;
}

void* BufferImpl::getAsyncMappedPointer() const
{
    void* ptr = nullptr;
    gfxBufferGetAsyncMappedPointer(m_handle, &ptr);
    return ptr;
}

bool BufferImpl::waitAsyncMapped(uint64_t timeoutNs)
{
    return gfxBufferWaitAsyncMapped(m_handle, timeoutNs) == GFX_RESULT_SUCCESS;
}

} // namespace gfx
