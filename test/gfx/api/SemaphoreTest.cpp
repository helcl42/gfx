#include "CommonTest.h"

#include <cstring>
#include <thread>

// C API tests compiled with C++ for GoogleTest compatibility

// ===========================================================================
// Parameterized Tests - Run on both Vulkan and WebGPU backends
// ===========================================================================

namespace {

class GfxSemaphoreTest : public testing::TestWithParam<GfxBackend> {
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

        // Check if timeline semaphore extension is supported
        uint32_t extensionCount = 0;
        gfxAdapterEnumerateExtensions(adapter, &extensionCount, nullptr);
        std::vector<const char*> supportedExtensions(extensionCount);
        gfxAdapterEnumerateExtensions(adapter, &extensionCount, supportedExtensions.data());

        bool timelineSemaphoreSupported = false;
        for (uint32_t i = 0; i < extensionCount; i++) {
            if (strcmp(supportedExtensions[i], GFX_DEVICE_EXTENSION_TIMELINE_SEMAPHORE) == 0) {
                timelineSemaphoreSupported = true;
                break;
            }
        }

        // Enable timeline semaphore extension if supported
        const char* deviceExtensions[] = { GFX_DEVICE_EXTENSION_TIMELINE_SEMAPHORE };
        if (timelineSemaphoreSupported) {
            deviceDesc.enabledExtensions = deviceExtensions;
            deviceDesc.enabledExtensionCount = 1;
        }

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

// NULL parameter validation tests
TEST_P(GfxSemaphoreTest, CreateWithNullDevice)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_BINARY;
    GfxSemaphore semaphore = nullptr;
    GfxResult result = gfxDeviceCreateSemaphore(nullptr, &semaphoreDesc, &semaphore);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSemaphoreTest, CreateWithNullDescriptor)
{
    GfxSemaphore semaphore = nullptr;
    GfxResult result = gfxDeviceCreateSemaphore(device, nullptr, &semaphore);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSemaphoreTest, CreateWithNullOutput)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_BINARY;
    GfxResult result = gfxDeviceCreateSemaphore(device, &semaphoreDesc, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSemaphoreTest, DestroyWithNullSemaphore)
{
    GfxResult result = gfxSemaphoreDestroy(nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSemaphoreTest, GetTypeWithNullSemaphore)
{
    GfxSemaphoreType type;
    GfxResult result = gfxSemaphoreGetType(nullptr, &type);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSemaphoreTest, GetTypeWithNullOutput)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_BINARY;
    GfxSemaphore semaphore = nullptr;
    ASSERT_EQ(gfxDeviceCreateSemaphore(device, &semaphoreDesc, &semaphore), GFX_RESULT_SUCCESS);

    GfxResult result = gfxSemaphoreGetType(semaphore, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxSemaphoreDestroy(semaphore);
}

TEST_P(GfxSemaphoreTest, SignalWithNullSemaphore)
{
    GfxResult result = gfxSemaphoreSignal(nullptr, 1);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSemaphoreTest, WaitWithNullSemaphore)
{
    GfxResult result = gfxSemaphoreWait(nullptr, 1, 0);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSemaphoreTest, GetValueWithNullSemaphore)
{
    uint64_t value;
    GfxResult result = gfxSemaphoreGetValue(nullptr, &value);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSemaphoreTest, GetValueWithNullOutput)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_TIMELINE;
    semaphoreDesc.initialValue = 0;
    GfxSemaphore semaphore = nullptr;
    GfxResult result = gfxDeviceCreateSemaphore(device, &semaphoreDesc, &semaphore);
    if (result != GFX_RESULT_SUCCESS) {
        GTEST_SKIP() << "Timeline semaphores not supported";
    }

    result = gfxSemaphoreGetValue(semaphore, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxSemaphoreDestroy(semaphore);
}

// Functional tests
TEST_P(GfxSemaphoreTest, CreateAndDestroyBinary)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.label = "Test Binary Semaphore";
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_BINARY;

    GfxSemaphore semaphore = nullptr;
    ASSERT_EQ(gfxDeviceCreateSemaphore(device, &semaphoreDesc, &semaphore), GFX_RESULT_SUCCESS);
    EXPECT_NE(semaphore, nullptr);

    EXPECT_EQ(gfxSemaphoreDestroy(semaphore), GFX_RESULT_SUCCESS);
}

TEST_P(GfxSemaphoreTest, CreateAndDestroyTimeline)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.label = "Test Timeline Semaphore";
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_TIMELINE;
    semaphoreDesc.initialValue = 0;

    GfxSemaphore semaphore = nullptr;
    GfxResult result = gfxDeviceCreateSemaphore(device, &semaphoreDesc, &semaphore);
    if (result != GFX_RESULT_SUCCESS) {
        GTEST_SKIP() << "Timeline semaphores not supported";
    }
    EXPECT_NE(semaphore, nullptr);

    EXPECT_EQ(gfxSemaphoreDestroy(semaphore), GFX_RESULT_SUCCESS);
}

TEST_P(GfxSemaphoreTest, GetTypeBinary)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_BINARY;

    GfxSemaphore semaphore = nullptr;
    ASSERT_EQ(gfxDeviceCreateSemaphore(device, &semaphoreDesc, &semaphore), GFX_RESULT_SUCCESS);

    GfxSemaphoreType type;
    ASSERT_EQ(gfxSemaphoreGetType(semaphore, &type), GFX_RESULT_SUCCESS);
    EXPECT_EQ(type, GFX_SEMAPHORE_TYPE_BINARY);

    gfxSemaphoreDestroy(semaphore);
}

TEST_P(GfxSemaphoreTest, GetTypeTimeline)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_TIMELINE;
    semaphoreDesc.initialValue = 0;

    GfxSemaphore semaphore = nullptr;
    GfxResult result = gfxDeviceCreateSemaphore(device, &semaphoreDesc, &semaphore);
    if (result != GFX_RESULT_SUCCESS) {
        GTEST_SKIP() << "Timeline semaphores not supported";
    }

    GfxSemaphoreType type;
    ASSERT_EQ(gfxSemaphoreGetType(semaphore, &type), GFX_RESULT_SUCCESS);
    EXPECT_EQ(type, GFX_SEMAPHORE_TYPE_TIMELINE);

    gfxSemaphoreDestroy(semaphore);
}

TEST_P(GfxSemaphoreTest, TimelineInitialValue)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_TIMELINE;
    semaphoreDesc.initialValue = 42;

    GfxSemaphore semaphore = nullptr;
    GfxResult result = gfxDeviceCreateSemaphore(device, &semaphoreDesc, &semaphore);
    if (result != GFX_RESULT_SUCCESS) {
        GTEST_SKIP() << "Timeline semaphores not supported";
    }

    uint64_t value;
    ASSERT_EQ(gfxSemaphoreGetValue(semaphore, &value), GFX_RESULT_SUCCESS);
    EXPECT_EQ(value, 42);

    gfxSemaphoreDestroy(semaphore);
}

TEST_P(GfxSemaphoreTest, TimelineSignal)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_TIMELINE;
    semaphoreDesc.initialValue = 0;

    GfxSemaphore semaphore = nullptr;
    GfxResult result = gfxDeviceCreateSemaphore(device, &semaphoreDesc, &semaphore);
    if (result != GFX_RESULT_SUCCESS) {
        GTEST_SKIP() << "Timeline semaphores not supported";
    }

    ASSERT_EQ(gfxSemaphoreSignal(semaphore, 10), GFX_RESULT_SUCCESS);

    uint64_t value;
    ASSERT_EQ(gfxSemaphoreGetValue(semaphore, &value), GFX_RESULT_SUCCESS);
    EXPECT_EQ(value, 10);

    gfxSemaphoreDestroy(semaphore);
}

TEST_P(GfxSemaphoreTest, TimelineWait)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_TIMELINE;
    semaphoreDesc.initialValue = 5;

    GfxSemaphore semaphore = nullptr;
    GfxResult result = gfxDeviceCreateSemaphore(device, &semaphoreDesc, &semaphore);
    if (result != GFX_RESULT_SUCCESS) {
        GTEST_SKIP() << "Timeline semaphores not supported";
    }

    // Should succeed immediately since value is already 5
    EXPECT_EQ(gfxSemaphoreWait(semaphore, 5, 0), GFX_RESULT_SUCCESS);

    gfxSemaphoreDestroy(semaphore);
}

// ===========================================================================
// Queue Signal Tests
// ===========================================================================

// Submits an empty encoder that signals semaphore to value
static void submitSignal(GfxQueue queue, GfxCommandEncoder encoder, GfxSemaphore semaphore, uint64_t value)
{
    ASSERT_EQ(gfxCommandEncoderEnd(encoder), GFX_RESULT_SUCCESS);

    GfxSubmitDescriptor submitDesc = {};
    submitDesc.commandEncoders = &encoder;
    submitDesc.commandEncoderCount = 1;
    submitDesc.signalSemaphores = &semaphore;
    submitDesc.signalValues = &value;
    submitDesc.signalSemaphoreCount = 1;
    ASSERT_EQ(gfxQueueSubmit(queue, &submitDesc), GFX_RESULT_SUCCESS);
}

// A host wait observes the value signaled by a submit
TEST_P(GfxSemaphoreTest, TimelineSignaledBySubmit)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_TIMELINE;
    GfxSemaphore semaphore = nullptr;
    if (gfxDeviceCreateSemaphore(device, &semaphoreDesc, &semaphore) != GFX_RESULT_SUCCESS) {
        GTEST_SKIP() << "Timeline semaphores not supported";
    }

    GfxQueue queue = nullptr;
    ASSERT_EQ(gfxDeviceGetQueue(device, &queue), GFX_RESULT_SUCCESS);
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor encoderDesc = {};
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &encoderDesc, &encoder), GFX_RESULT_SUCCESS);

    ASSERT_NO_FATAL_FAILURE(submitSignal(queue, encoder, semaphore, 1));

    EXPECT_EQ(gfxSemaphoreWait(semaphore, 1, GFX_TIMEOUT_INFINITE), GFX_RESULT_SUCCESS);
    uint64_t value = 0;
    EXPECT_EQ(gfxSemaphoreGetValue(semaphore, &value), GFX_RESULT_SUCCESS);
    EXPECT_EQ(value, 1u);

    gfxQueueWaitIdle(queue);
    gfxCommandEncoderDestroy(encoder);
    gfxSemaphoreDestroy(semaphore);
}

// A wait started before the signaling submit still completes
TEST_P(GfxSemaphoreTest, TimelineWaitBeforeSubmit)
{
    GfxSemaphoreDescriptor semaphoreDesc = {};
    semaphoreDesc.type = GFX_SEMAPHORE_TYPE_TIMELINE;
    GfxSemaphore semaphore = nullptr;
    if (gfxDeviceCreateSemaphore(device, &semaphoreDesc, &semaphore) != GFX_RESULT_SUCCESS) {
        GTEST_SKIP() << "Timeline semaphores not supported";
    }

    GfxQueue queue = nullptr;
    ASSERT_EQ(gfxDeviceGetQueue(device, &queue), GFX_RESULT_SUCCESS);
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor encoderDesc = {};
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &encoderDesc, &encoder), GFX_RESULT_SUCCESS);

    constexpr uint64_t kTimeoutNs = 5'000'000'000ull; // Fail rather than hang
    GfxResult waitResult = GFX_RESULT_ERROR_UNKNOWN;
    std::thread consumer([&]() { waitResult = gfxSemaphoreWait(semaphore, 1, kTimeoutNs); });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    submitSignal(queue, encoder, semaphore, 1);
    consumer.join();

    EXPECT_EQ(waitResult, GFX_RESULT_SUCCESS);

    gfxQueueWaitIdle(queue);
    gfxCommandEncoderDestroy(encoder);
    gfxSemaphoreDestroy(semaphore);
}

// ===========================================================================
// Test Instantiation
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    GfxSemaphoreTest,
    testing::ValuesIn(getActiveBackends()),
    convertTestParamToString);

} // namespace
