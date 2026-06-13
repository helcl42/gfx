#ifndef GFX_CPP_QUEUE_H
#define GFX_CPP_QUEUE_H

#include <gfx_cpp/gfx.hpp>

#include <gfx/gfx.h>

#include <memory>

namespace gfx {

class QueueImpl : public Queue {
public:
    explicit QueueImpl(GfxQueue h);
    // Queue is owned by device, do not destroy
    ~QueueImpl() override = default;

    QueueInfo getInfo() const override;
    void* getNativeHandle() const override;
    Result submit(const SubmitDescriptor& submitDescriptor) override;
    void writeBuffer(std::shared_ptr<Buffer> buffer, uint64_t offset, const void* data, uint64_t size) override;
    void writeTexture(const WriteTextureDescriptor& descriptor, const void* data, uint64_t dataSize) override;
    void waitIdle() override;

    GfxQueue getHandle() const { return m_handle; }

private:
    GfxQueue m_handle;
};

} // namespace gfx

#endif // GFX_CPP_QUEUE_H
