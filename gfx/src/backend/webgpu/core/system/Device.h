#ifndef GFX_WEGPU_DEVICE_H
#define GFX_WEGPU_DEVICE_H

#include "../CoreTypes.h"

#include <memory>

namespace gfx::backend::webgpu::core {

class Adapter;
class Queue;
class Blit;

class Device {
public:
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Device(Adapter* adapter, const DeviceCreateInfo& createInfo);
    ~Device();

    WGPUDevice handle() const;
    Queue* getQueue();
    Adapter* getAdapter();

    WGPULimits getLimits() const;

    void waitIdle() const;

    bool supportsShaderFormat(ShaderSourceType format) const;
    bool isExtensionEnabled(DeviceExtension extension) const;

    Blit* getBlit();

private:
    WGPUDevice m_device = nullptr;
    Adapter* m_adapter = nullptr; // Non-owning pointer
    uint64_t m_enabledExtensions = 0;
    std::unique_ptr<Queue> m_queue;
    std::unique_ptr<Blit> m_blit;
};

} // namespace gfx::backend::webgpu::core

#endif // GFX_WEGPU_DEVICE_H