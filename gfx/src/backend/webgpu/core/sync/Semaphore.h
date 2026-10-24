#ifndef GFX_WEBGPU_SEMAPHORE_H
#define GFX_WEBGPU_SEMAPHORE_H

#include "../CoreTypes.h"

#include <condition_variable>
#include <mutex>
#include <vector>

namespace gfx::backend::webgpu::core {

class Semaphore {
public:
    // Prevent copying
    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;

    Semaphore(SemaphoreType type, uint64_t value);
    ~Semaphore() = default;

    SemaphoreType getType() const;
    uint64_t getValue() const;
    void signal(uint64_t value = 0);
    bool wait(uint64_t value = 0, uint64_t timeoutNs = UINT64_MAX) const;

    // Signals value once the work submitted to the queue completes
    void signalOnCompletion(WGPUQueue queue, WGPUInstance instance, uint64_t value);

private:
    struct PendingSignal {
        WGPUFuture future;
        uint64_t value; // Semaphore value once the work completes
    };

    uint64_t signaledValue(uint64_t current, uint64_t value) const;
    size_t finishedSignalCount() const;
    void completeFinishedWork();

    SemaphoreType m_type = SemaphoreType::Binary;
    uint64_t m_value = 0;
    WGPUInstance m_instance = nullptr;
    std::vector<PendingSignal> m_pendingSignals; // Submission order, which is completion order
    mutable std::mutex m_mutex;
    mutable std::condition_variable m_signaled;
};

} // namespace gfx::backend::webgpu::core

#endif // GFX_WEBGPU_SEMAPHORE_H
