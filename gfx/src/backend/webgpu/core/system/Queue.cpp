#include "Queue.h"

#include "../command/CommandEncoder.h"
#include "../resource/Buffer.h"
#include "../resource/Texture.h"
#include "../sync/Fence.h"
#include "../sync/Semaphore.h"
#include "../system/Adapter.h"
#include "../system/Device.h"
#include "../system/Instance.h"
#include "../util/Utils.h"

namespace gfx::backend::webgpu::core {

Queue::Queue(WGPUQueue queue, Device* device)
    : m_queue(queue)
    , m_device(device)
{
    // Don't add ref - emdawnwebgpu doesn't provide wgpuQueueAddRef
    // The queue is owned by the device and automatically destroyed with it
}

Queue::~Queue()
{
    if (m_queue) {
        wgpuQueueRelease(m_queue);
    }
}

WGPUQueue Queue::handle() const
{
    return m_queue;
}

Device* Queue::getDevice() const
{
    return m_device;
}

QueueInfo Queue::getInfo() const
{
    return { 0, 0 };
}

bool Queue::submit(const SubmitInfo& submitInfo)
{
    // Wait semaphores are implicit: the single queue runs submissions in order
    for (uint32_t i = 0; i < submitInfo.commandEncoderCount; ++i) {
        WGPUCommandBuffer cmdBuffer = submitInfo.commandEncoders[i]->commandBuffer();
        if (!cmdBuffer) {
            return false;
        }
        wgpuQueueSubmit(m_queue, 1, &cmdBuffer);
    }

    // Signaled once this work completes, so submit does not block
    WGPUInstance instance = m_device->getAdapter()->getInstance()->handle();
    if (submitInfo.signalFence) {
        submitInfo.signalFence->signalOnCompletion(m_queue, instance);
    }
    for (uint32_t i = 0; i < submitInfo.signalSemaphoreCount; ++i) {
        const uint64_t value = submitInfo.signalValues ? submitInfo.signalValues[i] : 0;
        submitInfo.signalSemaphores[i]->signalOnCompletion(m_queue, instance, value);
    }

    return true;
}

void Queue::writeBuffer(Buffer* buffer, uint64_t offset, const void* data, uint64_t size)
{
    wgpuQueueWriteBuffer(m_queue, buffer->handle(), offset, data, size);
}

void Queue::writeTexture(Texture* texture, uint32_t mipLevel, uint32_t arrayLayer, const WGPUOrigin3D& origin, const void* data, uint64_t dataSize, const WGPUExtent3D& extent, uint32_t bytesPerRow, uint32_t rowsPerImage, WGPUTextureAspect aspect)
{
    // For WriteTexture (CPU→texture), bytesPerRow doesn't require 256-byte alignment
    // (unlike buffer-to-texture copies). Use tight packing if bytesPerRow is 0.
    uint32_t blockHeight = 1;
    getWGPUFormatBlockDimensions(texture->getFormat(), nullptr, &blockHeight);
    uint32_t effectiveBytesPerRow = (bytesPerRow == 0) ? getPackedBytesPerRow(texture->getFormat(), aspect, extent.width) : bytesPerRow;

    WGPUTexelCopyTextureInfo dest = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    dest.texture = texture->handle();
    dest.mipLevel = mipLevel;
    dest.origin = origin;
    // For 1D/2D textures origin.z selects the array layer; for 3D textures it is the depth slice
    if (texture->getDimension() != WGPUTextureDimension_3D) {
        dest.origin.z = arrayLayer;
    }
    dest.aspect = aspect;

    WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    layout.bytesPerRow = effectiveBytesPerRow;
    layout.rowsPerImage = (rowsPerImage == 0) ? (extent.height + blockHeight - 1) / blockHeight : rowsPerImage; // rows of blocks for compressed formats

    wgpuQueueWriteTexture(m_queue, &dest, data, dataSize, &layout, &extent);
}

bool Queue::waitIdle()
{
#ifdef __EMSCRIPTEN__
    // The web cannot block on the GPU; queue ordering covers the callers
    return true;
#else
    return waitForFuture(m_device->getAdapter()->getInstance()->handle(), queueWorkDoneFuture(m_queue), UINT64_MAX);
#endif // __EMSCRIPTEN__
}

} // namespace gfx::backend::webgpu::core