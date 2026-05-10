#ifndef GFX_WEBGPU_BUFFER_H
#define GFX_WEBGPU_BUFFER_H

#include "../CoreTypes.h"

namespace gfx::backend::webgpu::core {

class Device;

class Buffer {
public:
    // Prevent copying
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Owning constructor - creates and manages WGPUBuffer
    Buffer(Device* device, const BufferCreateInfo& createInfo);
    // Non-owning constructor for imported buffers
    Buffer(Device* device, WGPUBuffer buffer, const BufferImportInfo& importInfo);
    ~Buffer();

    WGPUBuffer handle() const;
    uint64_t getSize() const;
    WGPUBufferUsage getUsage() const;
    const BufferInfo& getInfo() const;
    Device* getDevice() const;

    // Map buffer for CPU access
    // Returns mapped pointer on success, nullptr on failure
    void* map(uint64_t offset, uint64_t size);
    void unmap();

    // Async mapping - non-blocking
    void asyncMap(uint64_t offset, uint64_t size);
    bool isAsyncMapped() const;
    void* getAsyncMappedPointer();
    // Blocking wait until async map completes or timeout (nanoseconds) elapses.
    // Returns true if mapped successfully.
    bool waitUntilAsyncMapped(uint64_t timeoutNs = UINT64_MAX);

    // Memory synchronization (no-ops on WebGPU - memory is always coherent)
    void flushMappedRange(uint64_t offset, uint64_t size);
    void invalidateMappedRange(uint64_t offset, uint64_t size);

private:
    static BufferInfo createBufferInfo(const BufferCreateInfo& createInfo);
    static BufferInfo createBufferInfo(const BufferImportInfo& importInfo);

    struct AsyncMapCallbackData {
        WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
        bool completed = false;
        uint64_t offset = 0;
        uint64_t size = 0;
    };

private:
    Device* m_device = nullptr; // Non-owning pointer for device operations
    bool m_ownsResources = true;
    WGPUBuffer m_buffer = nullptr;
    BufferInfo m_info{};

    // Async map state
    WGPUFuture m_asyncMapFuture{};
    mutable AsyncMapCallbackData m_asyncCallbackData{};
    mutable bool m_asyncMapPending = false;
    mutable bool m_asyncMapped = false;
};

} // namespace gfx::backend::webgpu::core

#endif // GFX_WEBGPU_BUFFER_H