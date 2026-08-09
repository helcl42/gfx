#include <backend/webgpu/core/resource/Buffer.h>
#include <backend/webgpu/core/system/Device.h>
#include <backend/webgpu/core/system/Instance.h>

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

namespace {

class WebGPUBufferTest : public testing::Test {
protected:
    void SetUp() override
    {
        try {
            gfx::backend::webgpu::core::InstanceCreateInfo instInfo{};
            instance = std::make_unique<gfx::backend::webgpu::core::Instance>(instInfo);

            gfx::backend::webgpu::core::AdapterCreateInfo adapterInfo{};
            adapterInfo.adapterIndex = 0;
            adapter = instance->requestAdapter(adapterInfo);

            gfx::backend::webgpu::core::DeviceCreateInfo deviceInfo{};
            device = std::make_unique<gfx::backend::webgpu::core::Device>(adapter, deviceInfo);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "WebGPU not available: " << e.what();
        }
    }

    std::unique_ptr<gfx::backend::webgpu::core::Instance> instance;
    gfx::backend::webgpu::core::Adapter* adapter = nullptr;
    std::unique_ptr<gfx::backend::webgpu::core::Device> device;
};

TEST_F(WebGPUBufferTest, CreateBuffer_WithBasicSettings)
{
    gfx::backend::webgpu::core::BufferCreateInfo createInfo{};
    createInfo.size = 1024;
    createInfo.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    auto buffer = std::make_unique<gfx::backend::webgpu::core::Buffer>(device.get(), createInfo);

    EXPECT_NE(buffer->handle(), nullptr);
    EXPECT_EQ(buffer->getSize(), 1024);
}

TEST_F(WebGPUBufferTest, GetDevice_ReturnsCorrectDevice)
{
    gfx::backend::webgpu::core::BufferCreateInfo createInfo{};
    createInfo.size = 256;
    createInfo.usage = WGPUBufferUsage_CopyDst;

    auto buffer = std::make_unique<gfx::backend::webgpu::core::Buffer>(device.get(), createInfo);

    EXPECT_EQ(buffer->getDevice(), device.get());
}

TEST_F(WebGPUBufferTest, Map_UnmapBuffer)
{
    gfx::backend::webgpu::core::BufferCreateInfo createInfo{};
    createInfo.size = 1024;
    createInfo.usage = WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc;

    auto buffer = std::make_unique<gfx::backend::webgpu::core::Buffer>(device.get(), createInfo);

    void* mapped = nullptr;
    auto status = buffer->map(0, 1024, &mapped, UINT64_MAX);
    if (status == gfx::backend::webgpu::core::Buffer::MapStatus::Ready) {
        // Write some data
        std::vector<uint32_t> data(256, 42);
        std::memcpy(mapped, data.data(), data.size() * sizeof(uint32_t));

        buffer->unmap();
    }
}

TEST_F(WebGPUBufferTest, MultipleBuffers_CanCoexist)
{
    gfx::backend::webgpu::core::BufferCreateInfo createInfo{};
    createInfo.size = 512;
    createInfo.usage = WGPUBufferUsage_CopyDst;

    auto buffer1 = std::make_unique<gfx::backend::webgpu::core::Buffer>(device.get(), createInfo);
    auto buffer2 = std::make_unique<gfx::backend::webgpu::core::Buffer>(device.get(), createInfo);

    EXPECT_NE(buffer1->handle(), nullptr);
    EXPECT_NE(buffer2->handle(), nullptr);
    EXPECT_NE(buffer1->handle(), buffer2->handle());
}

// ============================================================================
// Non-blocking Map Tests
// ============================================================================

TEST_F(WebGPUBufferTest, Map_PollsUntilReady_ThenWrites)
{
    using MapStatus = gfx::backend::webgpu::core::Buffer::MapStatus;

    gfx::backend::webgpu::core::BufferCreateInfo createInfo{};
    createInfo.size = 256;
    createInfo.usage = WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc;

    auto buffer = std::make_unique<gfx::backend::webgpu::core::Buffer>(device.get(), createInfo);

    void* ptr = nullptr;
    MapStatus status = buffer->map(0, 256, &ptr);
    ASSERT_NE(status, MapStatus::Failed);

    constexpr int maxAttempts = 100000;
    for (int i = 0; status == MapStatus::Pending && i < maxAttempts; ++i) {
        status = buffer->map(0, 256, &ptr);
    }
    ASSERT_EQ(status, MapStatus::Ready) << "buffer did not become mapped within timeout";
    ASSERT_NE(ptr, nullptr);

    std::memset(ptr, 0xCD, 256);

    buffer->unmap();
}

} // anonymous namespace
