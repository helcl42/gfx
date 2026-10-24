#include "Semaphore.h"

#include "../util/Utils.h"

#include <algorithm>
#include <chrono>

namespace gfx::backend::webgpu::core {

namespace {
    // UINT64_MAX stays infinite
    uint64_t remainingNs(std::chrono::steady_clock::time_point start, uint64_t timeoutNs)
    {
        if (timeoutNs == UINT64_MAX) {
            return UINT64_MAX;
        }
        const auto elapsedNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        return timeoutNs - std::min(elapsedNs, timeoutNs);
    }
} // anonymous namespace

Semaphore::Semaphore(SemaphoreType type, uint64_t value)
    : m_type(type)
    , m_value(value)
{
}

SemaphoreType Semaphore::getType() const
{
    return m_type;
}

uint64_t Semaphore::getValue() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    const size_t finished = finishedSignalCount();
    return finished > 0 ? m_pendingSignals[finished - 1].value : m_value;
}

void Semaphore::signal(uint64_t value)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    completeFinishedWork();
    m_value = signaledValue(m_value, value);
    m_signaled.notify_all();
}

bool Semaphore::wait(uint64_t value, uint64_t timeoutNs) const
{
    const auto start = std::chrono::steady_clock::now();
    const uint64_t target = (m_type == SemaphoreType::Timeline) ? value : 1;
    const auto reachesTarget = [target](const PendingSignal& pending) { return pending.value >= target; };

    std::unique_lock<std::mutex> lock(m_mutex);
    // Wait for a host signal or a submission that reaches the target
    const auto signalable = [&] { return m_value >= target || std::any_of(m_pendingSignals.begin(), m_pendingSignals.end(), reachesTarget); };
    if (timeoutNs == UINT64_MAX) {
        m_signaled.wait(lock, signalable);
    } else if (!m_signaled.wait_for(lock, std::chrono::nanoseconds(timeoutNs), signalable)) {
        return false;
    }
    if (m_value >= target) {
        return true;
    }

    // Submissions complete in order, so the earliest one reaching the target suffices
    const WGPUFuture future = std::find_if(m_pendingSignals.begin(), m_pendingSignals.end(), reachesTarget)->future;
    // Unlock so other threads can signal and submit meanwhile
    const WGPUInstance instance = m_instance;
    lock.unlock();
    return waitForFuture(instance, future, remainingNs(start, timeoutNs));
}

void Semaphore::signalOnCompletion(WGPUQueue queue, WGPUInstance instance, uint64_t value)
{
#ifdef __EMSCRIPTEN__
    // The web cannot block on the GPU, so signal at submit
    (void)queue;
    (void)instance;
    signal(value);
#else
    std::lock_guard<std::mutex> lock(m_mutex);
    completeFinishedWork();
    const uint64_t current = m_pendingSignals.empty() ? m_value : m_pendingSignals.back().value;
    m_instance = instance;
    m_pendingSignals.push_back({ queueWorkDoneFuture(queue), signaledValue(current, value) });
    m_signaled.notify_all();
#endif
}

uint64_t Semaphore::signaledValue(uint64_t current, uint64_t value) const
{
    if (m_type == SemaphoreType::Binary) {
        return 1;
    }
    // A timeline value of 0 increments
    return value > 0 ? value : current + 1;
}

size_t Semaphore::finishedSignalCount() const
{
    size_t count = 0;
    while (count < m_pendingSignals.size() && waitForFuture(m_instance, m_pendingSignals[count].future, 0)) {
        ++count;
    }
    return count;
}

void Semaphore::completeFinishedWork()
{
    const size_t finished = finishedSignalCount();
    if (finished > 0) {
        m_value = m_pendingSignals[finished - 1].value;
        m_pendingSignals.erase(m_pendingSignals.begin(), m_pendingSignals.begin() + finished);
    }
}

} // namespace gfx::backend::webgpu::core
