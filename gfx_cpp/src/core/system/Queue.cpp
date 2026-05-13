#include "Queue.h"

#include "../command/CommandEncoder.h"
#include "../resource/Buffer.h"
#include "../resource/Texture.h"
#include "../sync/Fence.h"
#include "../sync/Semaphore.h"

#include "../../converter/Conversions.h"

#include <stdexcept>
#include <vector>

namespace gfx {

QueueImpl::QueueImpl(GfxQueue h)
    : m_handle(h)
{
}

QueueInfo QueueImpl::getInfo() const
{
    GfxQueueInfo cInfo = {};
    gfxQueueGetInfo(m_handle, &cInfo);
    QueueInfo info;
    info.queueFamilyIndex = cInfo.queueFamilyIndex;
    info.queueIndex = cInfo.queueIndex;
    return info;
}

void* QueueImpl::getNativeHandle() const
{
    void* handle = nullptr;
    GfxResult result = gfxQueueGetNativeHandle(m_handle, &handle);
    if (result != GFX_RESULT_SUCCESS) {
        return nullptr;
    }
    return handle;
}

Result QueueImpl::submit(const SubmitDescriptor& submitDescriptor)
{
    std::vector<GfxCommandEncoder> cEncoders;
    std::vector<GfxSemaphore> cWaitSems;
    std::vector<GfxSemaphore> cSignalSems;

    GfxSubmitDescriptor cDescriptor = {};
    convertSubmitDescriptor(submitDescriptor, cDescriptor, cEncoders, cWaitSems, cSignalSems);

    return cResultToCppResult(gfxQueueSubmit(m_handle, &cDescriptor));
}

void QueueImpl::writeBuffer(std::shared_ptr<Buffer> buffer, uint64_t offset, const void* data, uint64_t size)
{
    auto impl = std::dynamic_pointer_cast<BufferImpl>(buffer);
    if (!impl) {
        throw std::runtime_error("Invalid buffer type");
    }
    gfxQueueWriteBuffer(m_handle, impl->getHandle(), offset, data, size);
}

void QueueImpl::writeTexture(std::shared_ptr<Texture> texture, const Origin3D& origin, uint32_t mipLevel, uint32_t arrayLayer, const void* data, uint64_t dataSize, const Extent3D& extent, TextureLayout finalLayout)
{
    auto impl = std::dynamic_pointer_cast<TextureImpl>(texture);
    if (!impl) {
        throw std::runtime_error("Invalid texture type");
    }
    GfxOrigin3D cOrigin = cppOrigin3DToCOrigin3D(origin);
    GfxExtent3D cExtent = cppExtent3DToCExtent3D(extent);
    GfxTextureLayout cFinalLayout = cppLayoutToCLayout(finalLayout);
    gfxQueueWriteTexture(m_handle, impl->getHandle(), &cOrigin, &cExtent, mipLevel, arrayLayer, data, dataSize, cFinalLayout);
}

void QueueImpl::waitIdle()
{
    gfxQueueWaitIdle(m_handle);
}

} // namespace gfx
