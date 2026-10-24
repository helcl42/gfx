#ifndef GFX_WEBGPU_FENCE_H
#define GFX_WEBGPU_FENCE_H

#include "../CoreTypes.h"

namespace gfx::backend::webgpu::core {

class Fence {
public:
    // Prevent copying
    Fence(const Fence&) = delete;
    Fence& operator=(const Fence&) = delete;

    Fence(bool signaled);
    ~Fence() = default;

    bool isSignaled() const;
    void signal();
    void reset();
    bool wait(uint64_t timeoutNs) const;

    // Signals once the work submitted to the queue completes
    void signalOnCompletion(WGPUQueue queue, WGPUInstance instance);

private:
    bool m_signaled = false;
    WGPUInstance m_instance = nullptr;
    WGPUFuture m_pendingWork = {};
};

} // namespace gfx::backend::webgpu::core

#endif // GFX_WEBGPU_FENCE_H
