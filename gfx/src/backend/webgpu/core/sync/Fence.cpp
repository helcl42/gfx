#include "../sync/Fence.h"

#include "../util/Utils.h"

namespace gfx::backend::webgpu::core {

Fence::Fence(bool signaled)
    : m_signaled(signaled)
{
}

bool Fence::isSignaled() const
{
    return wait(0);
}

void Fence::signal()
{
    m_signaled = true;
}

void Fence::reset()
{
    m_signaled = false;
    m_pendingWork = {};
}

bool Fence::wait(uint64_t timeoutNs) const
{
    return m_signaled || (m_pendingWork.id != 0 && waitForFuture(m_instance, m_pendingWork, timeoutNs));
}

void Fence::signalOnCompletion(WGPUQueue queue, WGPUInstance instance)
{
#ifdef __EMSCRIPTEN__
    // The web cannot block on the GPU, so signal at submit
    (void)queue;
    (void)instance;
    signal();
#else
    m_instance = instance;
    m_pendingWork = queueWorkDoneFuture(queue);
#endif
}

} // namespace gfx::backend::webgpu::core
