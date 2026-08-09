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

    enum class MapStatus {
        Ready,
        Pending,
        Failed
    };

    // Waits up to timeoutNs for the mapping; 0 polls (Pending if not landed yet).
    MapStatus map(uint64_t offset, uint64_t size, void** outPointer, uint64_t timeoutNs = 0);
    void unmap();

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

    // Map state
    WGPUFuture m_asyncMapFuture{};
    AsyncMapCallbackData m_asyncCallbackData{};
    bool m_asyncMapPending = false;
    bool m_mapped = false;
    void* m_mappedPointer = nullptr;
};

} // namespace gfx::backend::webgpu::core

#endif // GFX_WEBGPU_BUFFER_H