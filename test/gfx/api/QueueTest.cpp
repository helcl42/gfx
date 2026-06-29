#include "CommonTest.h"

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

// C API tests compiled with C++ for GoogleTest compatibility

// ===========================================================================
// Parameterized Tests - Run on both Vulkan and WebGPU backends
// ===========================================================================

namespace {

class GfxQueueTest : public testing::TestWithParam<GfxBackend> {
protected:
    void SetUp() override
    {
        backend = GetParam();

        if (gfxLoadBackend(backend) != GFX_RESULT_SUCCESS) {
            GTEST_SKIP() << "Backend not available";
        }

        const char* extensions[] = { GFX_INSTANCE_EXTENSION_DEBUG };
        GfxInstanceDescriptor instDesc = {};
        instDesc.sType = GFX_STRUCTURE_TYPE_INSTANCE_DESCRIPTOR;
        instDesc.pNext = nullptr;
        instDesc.backend = backend;
        instDesc.enabledExtensions = extensions;
        instDesc.enabledExtensionCount = 1;

        if (gfxCreateInstance(&instDesc, &instance) != GFX_RESULT_SUCCESS) {
            gfxUnloadBackend(backend);
            GTEST_SKIP() << "Failed to create instance";
        }

        GfxAdapterDescriptor adapterDesc = {};
        adapterDesc.sType = GFX_STRUCTURE_TYPE_ADAPTER_DESCRIPTOR;
        adapterDesc.pNext = nullptr;

        if (gfxInstanceRequestAdapter(instance, &adapterDesc, &adapter) != GFX_RESULT_SUCCESS) {
            gfxInstanceDestroy(instance);
            gfxUnloadBackend(backend);
            GTEST_SKIP() << "Failed to request adapter";
        }

        GfxDeviceDescriptor deviceDesc = {};
        deviceDesc.sType = GFX_STRUCTURE_TYPE_DEVICE_DESCRIPTOR;
        deviceDesc.pNext = nullptr;
        deviceDesc.label = "Test Device";

        if (gfxAdapterCreateDevice(adapter, &deviceDesc, &device) != GFX_RESULT_SUCCESS) {
            gfxInstanceDestroy(instance);
            gfxUnloadBackend(backend);
            GTEST_SKIP() << "Failed to create device";
        }
    }

    void TearDown() override
    {
        if (device) {
            gfxDeviceDestroy(device);
        }
        if (instance) {
            gfxInstanceDestroy(instance);
        }
        gfxUnloadBackend(backend);
    }

    GfxBackend backend = GFX_BACKEND_VULKAN;
    GfxInstance instance = nullptr;
    GfxAdapter adapter = nullptr;
    GfxDevice device = nullptr;
};

// ===========================================================================
// Queue Tests
// ===========================================================================

// Test: Get default queue with NULL device
TEST_P(GfxQueueTest, GetQueueWithNullDevice)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(nullptr, &queue);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Get default queue with NULL output
TEST_P(GfxQueueTest, GetQueueWithNullOutput)
{
    GfxResult result = gfxDeviceGetQueue(device, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Get default queue
TEST_P(GfxQueueTest, GetDefaultQueue)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
    EXPECT_NE(queue, nullptr);
}

// Test: Get queue by index with NULL device
TEST_P(GfxQueueTest, GetQueueByIndexWithNullDevice)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueueByIndex(nullptr, 0, 0, &queue);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Get queue by index with NULL output
TEST_P(GfxQueueTest, GetQueueByIndexWithNullOutput)
{
    GfxResult result = gfxDeviceGetQueueByIndex(device, 0, 0, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Get queue by index
TEST_P(GfxQueueTest, GetQueueByIndex)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueueByIndex(device, 0, 0, &queue);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
    EXPECT_NE(queue, nullptr);
}

// Test: Queue submit with NULL queue
TEST_P(GfxQueueTest, SubmitWithNullQueue)
{
    GfxSubmitDescriptor submitDesc = {};
    GfxResult result = gfxQueueSubmit(nullptr, &submitDesc);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Queue submit with NULL descriptor
TEST_P(GfxQueueTest, SubmitWithNullDescriptor)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    result = gfxQueueSubmit(queue, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Queue submit with empty descriptor
TEST_P(GfxQueueTest, SubmitWithEmptyDescriptor)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    GfxSubmitDescriptor submitDesc = {};
    submitDesc.commandEncoders = nullptr;
    submitDesc.commandEncoderCount = 0;

    result = gfxQueueSubmit(queue, &submitDesc);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
}

// Test: Vulkan requires a waitStages entry per wait semaphore. The validator must reject a
// submit that has wait semaphores but a NULL waitStages array (rejected before any GPU wait,
// so no deadlock). WebGPU ignores waitStages, so this requirement is Vulkan-only.
TEST_P(GfxQueueTest, SubmitWithWaitSemaphoresRequiresWaitStages)
{
    if (GetParam() != GFX_BACKEND_VULKAN) {
        GTEST_SKIP() << "waitStages is a Vulkan-only requirement; WebGPU ignores it";
    }

    GfxQueue queue = nullptr;
    ASSERT_EQ(gfxDeviceGetQueue(device, &queue), GFX_RESULT_SUCCESS);

    GfxSemaphoreDescriptor semDesc = {};
    semDesc.sType = GFX_STRUCTURE_TYPE_SEMAPHORE_DESCRIPTOR;
    semDesc.type = GFX_SEMAPHORE_TYPE_BINARY;
    GfxSemaphore semaphore = nullptr;
    ASSERT_EQ(gfxDeviceCreateSemaphore(device, &semDesc, &semaphore), GFX_RESULT_SUCCESS);

    GfxSubmitDescriptor submitDesc = {};
    submitDesc.waitSemaphores = &semaphore;
    submitDesc.waitSemaphoreCount = 1;
    submitDesc.waitStages = nullptr; // missing -> must be rejected

    EXPECT_EQ(gfxQueueSubmit(queue, &submitDesc), GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxSemaphoreDestroy(semaphore);
}

// Test: Concurrent queue operations from multiple threads on the SAME queue.
// The API guarantees internal synchronization for queue operations (see THREADING MODEL).
TEST_P(GfxQueueTest, ConcurrentQueueOperationsAreThreadSafe)
{
    GfxQueue queue = nullptr;
    ASSERT_EQ(gfxDeviceGetQueue(device, &queue), GFX_RESULT_SUCCESS);

    // A texture for gfxQueueWriteTexture (exercises the internal staging-submit path)
    GfxTextureDescriptor textureDesc = {};
    textureDesc.type = GFX_TEXTURE_TYPE_2D;
    textureDesc.size = { 16, 16, 1 };
    textureDesc.arrayLayerCount = 1;
    textureDesc.mipLevelCount = 1;
    textureDesc.sampleCount = GFX_SAMPLE_COUNT_1;
    textureDesc.format = GFX_FORMAT_R8G8B8A8_UNORM;
    textureDesc.usage = GFX_FLAGS(GFX_TEXTURE_USAGE_COPY_DST | GFX_TEXTURE_USAGE_TEXTURE_BINDING);

    GfxTexture texture = nullptr;
    ASSERT_EQ(gfxDeviceCreateTexture(device, &textureDesc, &texture), GFX_RESULT_SUCCESS);

    constexpr int kThreads = 4;
    constexpr int kIterations = 25;
    std::atomic<int> failures{ 0 };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<uint8_t> pixels(16 * 16 * 4, static_cast<uint8_t>(t));
            for (int i = 0; i < kIterations; ++i) {
                if (t % 2 == 0) {
                    // Empty submit from this thread
                    GfxSubmitDescriptor submitDesc = {};
                    if (gfxQueueSubmit(queue, &submitDesc) != GFX_RESULT_SUCCESS) {
                        ++failures;
                    }
                } else {
                    // Texture write from this thread (internally submits + waits)
                    GfxWriteTextureDescriptor writeDesc = {};
                    writeDesc.texture = texture;
                    writeDesc.extent = { 16, 16, 1 };
                    writeDesc.finalLayout = GFX_TEXTURE_LAYOUT_SHADER_READ_ONLY;
                    if (gfxQueueWriteTexture(queue, &writeDesc, pixels.data(), pixels.size()) != GFX_RESULT_SUCCESS) {
                        ++failures;
                    }
                }
            }
        });
    }

    // Main thread concurrently waits for the queue to go idle a few times
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(gfxQueueWaitIdle(queue), GFX_RESULT_SUCCESS);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(failures.load(), 0);

    gfxQueueWaitIdle(queue);
    gfxTextureDestroy(texture);
}

// Test: Queue write buffer with NULL queue
TEST_P(GfxQueueTest, WriteBufferWithNullQueue)
{
    uint32_t data = 42;
    GfxResult result = gfxQueueWriteBuffer(nullptr, nullptr, 0, &data, sizeof(data));
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Queue write buffer with NULL buffer
TEST_P(GfxQueueTest, WriteBufferWithNullBuffer)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    uint32_t data = 42;
    result = gfxQueueWriteBuffer(queue, nullptr, 0, &data, sizeof(data));
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Queue write buffer with NULL data
TEST_P(GfxQueueTest, WriteBufferWithNullData)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    // Create a small buffer
    GfxBufferDescriptor bufferDesc = {};
    bufferDesc.label = "Test Buffer";
    bufferDesc.size = 256;
    bufferDesc.usage = GFX_BUFFER_USAGE_COPY_DST;
    bufferDesc.memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL;

    GfxBuffer buffer = nullptr;
    result = gfxDeviceCreateBuffer(device, &bufferDesc, &buffer);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    result = gfxQueueWriteBuffer(queue, buffer, 0, nullptr, 256);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxBufferDestroy(buffer);
}

// Test: Queue write buffer
TEST_P(GfxQueueTest, WriteBuffer)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    // Create a buffer
    GfxBufferDescriptor bufferDesc = {};
    bufferDesc.label = "Test Buffer";
    bufferDesc.size = 256;
    bufferDesc.usage = GFX_BUFFER_USAGE_COPY_DST;
    bufferDesc.memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL;

    GfxBuffer buffer = nullptr;
    result = gfxDeviceCreateBuffer(device, &bufferDesc, &buffer);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    // Write data to buffer
    uint32_t data[64];
    for (int i = 0; i < 64; i++) {
        data[i] = i;
    }

    result = gfxQueueWriteBuffer(queue, buffer, 0, data, sizeof(data));
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);

    gfxBufferDestroy(buffer);
}

// Test: Queue wait idle with NULL queue
TEST_P(GfxQueueTest, WaitIdleWithNullQueue)
{
    GfxResult result = gfxQueueWaitIdle(nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Queue wait idle
TEST_P(GfxQueueTest, WaitIdle)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    result = gfxQueueWaitIdle(queue);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
}

// Test: Queue write buffer with offset
TEST_P(GfxQueueTest, WriteBufferWithOffset)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    // Create a buffer
    GfxBufferDescriptor bufferDesc = {};
    bufferDesc.label = "Test Buffer";
    bufferDesc.size = 256;
    bufferDesc.usage = GFX_BUFFER_USAGE_COPY_DST;
    bufferDesc.memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL;

    GfxBuffer buffer = nullptr;
    result = gfxDeviceCreateBuffer(device, &bufferDesc, &buffer);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    // Write data at offset 64
    uint32_t data[16];
    for (int i = 0; i < 16; i++) {
        data[i] = i + 100;
    }

    result = gfxQueueWriteBuffer(queue, buffer, 64, data, sizeof(data));
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);

    gfxBufferDestroy(buffer);
}

// Test: Get queue info
TEST_P(GfxQueueTest, GetQueueInfo)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);
    ASSERT_NE(queue, nullptr);

    GfxQueueInfo info = {};
    result = gfxQueueGetInfo(queue, &info);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
}

// Test: Get queue info with null queue
TEST_P(GfxQueueTest, GetQueueInfoNullQueue)
{
    GfxQueueInfo info = {};
    GfxResult result = gfxQueueGetInfo(nullptr, &info);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Get queue info with null output
TEST_P(GfxQueueTest, GetQueueInfoNullOutput)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    result = gfxQueueGetInfo(queue, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Get queue native handle
TEST_P(GfxQueueTest, GetNativeHandle)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);
    ASSERT_NE(queue, nullptr);

    void* handle = nullptr;
    result = gfxQueueGetNativeHandle(queue, &handle);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
    EXPECT_NE(handle, nullptr);
}

// Test: Get queue native handle with null queue
TEST_P(GfxQueueTest, GetNativeHandleNullQueue)
{
    void* handle = nullptr;
    GfxResult result = gfxQueueGetNativeHandle(nullptr, &handle);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Get queue native handle with null output
TEST_P(GfxQueueTest, GetNativeHandleNullOutput)
{
    GfxQueue queue = nullptr;
    GfxResult result = gfxDeviceGetQueue(device, &queue);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    result = gfxQueueGetNativeHandle(queue, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// ===========================================================================
// Test Instantiation
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    GfxQueueTest,
    testing::ValuesIn(getActiveBackends()),
    convertTestParamToString);

} // namespace
