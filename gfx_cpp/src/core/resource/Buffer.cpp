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
    if (gfxBufferMap(m_handle, offset, size, &mappedPointer) != GFX_RESULT_SUCCESS) {
        return nullptr;
    }
    return mappedPointer;
}

Result BufferImpl::mapAsync(void*& outPointer, uint64_t offset, uint64_t size)
{
    outPointer = nullptr;
    return cResultToCppResult(gfxBufferMapAsync(m_handle, offset, size, &outPointer));
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

} // namespace gfx
