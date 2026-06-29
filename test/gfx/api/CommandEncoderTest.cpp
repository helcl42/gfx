#include "CommonTest.h"

#include <cstring>

// C API tests compiled with C++ for GoogleTest compatibility

// ===========================================================================
// Parameterized Tests - Run on both Vulkan and WebGPU backends
// ===========================================================================

namespace {

class GfxCommandEncoderTest : public testing::TestWithParam<GfxBackend> {
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

// NULL parameter validation tests
TEST_P(GfxCommandEncoderTest, CreateCommandEncoderWithNullDevice)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";

    GfxResult result = gfxDeviceCreateCommandEncoder(nullptr, &desc, &encoder);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, CreateCommandEncoderWithNullDescriptor)
{
    GfxCommandEncoder encoder = nullptr;

    GfxResult result = gfxDeviceCreateCommandEncoder(device, nullptr, &encoder);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, CreateCommandEncoderWithNullOutput)
{
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";

    GfxResult result = gfxDeviceCreateCommandEncoder(device, &desc, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Basic functionality tests
TEST_P(GfxCommandEncoderTest, CreateCommandEncoder)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";

    GfxResult result = gfxDeviceCreateCommandEncoder(device, &desc, &encoder);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
    EXPECT_NE(encoder, nullptr);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, CreateCommandEncoderWithoutLabel)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = nullptr;

    GfxResult result = gfxDeviceCreateCommandEncoder(device, &desc, &encoder);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
    EXPECT_NE(encoder, nullptr);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, DestroyNullCommandEncoder)
{
    GfxResult result = gfxCommandEncoderDestroy(nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, DestroyCommandEncoder)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";

    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxResult result = gfxCommandEncoderDestroy(encoder);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
}

// Copy operations tests
TEST_P(GfxCommandEncoderTest, CopyBufferToBufferWithNullEncoder)
{
    GfxCopyBufferToBufferDescriptor copyDesc = {};

    GfxResult result = gfxCommandEncoderCopyBufferToBuffer(nullptr, &copyDesc);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, CopyBufferToBufferWithNullDescriptor)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxResult result = gfxCommandEncoderCopyBufferToBuffer(encoder, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, CopyBufferToBuffer)
{
    // Create buffers
    GfxBufferDescriptor srcBufferDesc = {};
    srcBufferDesc.label = "source_buffer";
    srcBufferDesc.size = 256;
    srcBufferDesc.usage = GFX_BUFFER_USAGE_COPY_SRC;
    srcBufferDesc.memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL;
    GfxBuffer srcBuffer = nullptr;
    ASSERT_EQ(gfxDeviceCreateBuffer(device, &srcBufferDesc, &srcBuffer), GFX_RESULT_SUCCESS);

    GfxBufferDescriptor dstBufferDesc = {};
    dstBufferDesc.label = "destination_buffer";
    dstBufferDesc.size = 256;
    dstBufferDesc.usage = GFX_BUFFER_USAGE_COPY_DST;
    dstBufferDesc.memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL;
    GfxBuffer dstBuffer = nullptr;
    ASSERT_EQ(gfxDeviceCreateBuffer(device, &dstBufferDesc, &dstBuffer), GFX_RESULT_SUCCESS);

    // Create command encoder
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor encoderDesc = {};
    encoderDesc.label = "copy_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &encoderDesc, &encoder), GFX_RESULT_SUCCESS);

    // Set up copy operation
    GfxCopyBufferToBufferDescriptor copyDesc = {};
    copyDesc.source = srcBuffer;
    copyDesc.sourceOffset = 0;
    copyDesc.destination = dstBuffer;
    copyDesc.destinationOffset = 0;
    copyDesc.size = 256;

    // Test that copy operation succeeds
    GfxResult result = gfxCommandEncoderCopyBufferToBuffer(encoder, &copyDesc);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);

    gfxCommandEncoderDestroy(encoder);
    gfxBufferDestroy(srcBuffer);
    gfxBufferDestroy(dstBuffer);
}

// GFX_WHOLE_SIZE must copy the rest of the source from sourceOffset (not a zero-length copy).
TEST_P(GfxCommandEncoderTest, CopyBufferToBufferWholeSizeCopiesRest)
{
    GfxQueue queue = nullptr;
    ASSERT_EQ(gfxDeviceGetQueue(device, &queue), GFX_RESULT_SUCCESS);

    constexpr uint64_t kSize = 256;
    constexpr uint64_t kOffset = 64; // copy [kOffset, kSize) via GFX_WHOLE_SIZE -> kSize - kOffset bytes

    GfxBufferDescriptor srcDesc = {};
    srcDesc.size = kSize;
    srcDesc.usage = GFX_FLAGS(GFX_BUFFER_USAGE_MAP_WRITE | GFX_BUFFER_USAGE_COPY_SRC);
    srcDesc.memoryProperties = GFX_FLAGS(GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT);
    GfxBuffer src = nullptr;
    ASSERT_EQ(gfxDeviceCreateBuffer(device, &srcDesc, &src), GFX_RESULT_SUCCESS);

    GfxBufferDescriptor dstDesc = {};
    dstDesc.size = kSize;
    dstDesc.usage = GFX_FLAGS(GFX_BUFFER_USAGE_MAP_READ | GFX_BUFFER_USAGE_COPY_DST);
    dstDesc.memoryProperties = GFX_FLAGS(GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT);
    GfxBuffer dst = nullptr;
    ASSERT_EQ(gfxDeviceCreateBuffer(device, &dstDesc, &dst), GFX_RESULT_SUCCESS);

    // Seed the source with a known per-byte pattern.
    void* srcPtr = nullptr;
    ASSERT_EQ(gfxBufferMap(src, 0, kSize, &srcPtr), GFX_RESULT_SUCCESS);
    auto* srcBytes = static_cast<uint8_t*>(srcPtr);
    for (uint64_t i = 0; i < kSize; ++i) {
        srcBytes[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);
    }
    ASSERT_EQ(gfxBufferUnmap(src), GFX_RESULT_SUCCESS);

    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor encoderDesc = {};
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &encoderDesc, &encoder), GFX_RESULT_SUCCESS);

    GfxCopyBufferToBufferDescriptor copyDesc = {};
    copyDesc.source = src;
    copyDesc.sourceOffset = kOffset;
    copyDesc.destination = dst;
    copyDesc.destinationOffset = 0;
    copyDesc.size = GFX_WHOLE_SIZE; // resolves to kSize - kOffset
    ASSERT_EQ(gfxCommandEncoderCopyBufferToBuffer(encoder, &copyDesc), GFX_RESULT_SUCCESS);
    ASSERT_EQ(gfxCommandEncoderEnd(encoder), GFX_RESULT_SUCCESS);

    GfxFenceDescriptor fenceDesc = {};
    GfxFence fence = nullptr;
    ASSERT_EQ(gfxDeviceCreateFence(device, &fenceDesc, &fence), GFX_RESULT_SUCCESS);

    GfxSubmitDescriptor submitDesc = {};
    submitDesc.commandEncoders = &encoder;
    submitDesc.commandEncoderCount = 1;
    submitDesc.signalFence = fence;
    ASSERT_EQ(gfxQueueSubmit(queue, &submitDesc), GFX_RESULT_SUCCESS);
    ASSERT_EQ(gfxFenceWait(fence, GFX_TIMEOUT_INFINITE), GFX_RESULT_SUCCESS);

    // The whole tail of the source must have landed at the start of the destination.
    void* dstPtr = nullptr;
    ASSERT_EQ(gfxBufferMap(dst, 0, kSize, &dstPtr), GFX_RESULT_SUCCESS);
    auto* dstBytes = static_cast<uint8_t*>(dstPtr);
    for (uint64_t i = 0; i < kSize - kOffset; ++i) {
        EXPECT_EQ(dstBytes[i], static_cast<uint8_t>(((i + kOffset) * 7 + 3) & 0xFF)) << "mismatch at byte " << i;
    }
    ASSERT_EQ(gfxBufferUnmap(dst), GFX_RESULT_SUCCESS);

    gfxFenceDestroy(fence);
    gfxCommandEncoderDestroy(encoder);
    gfxBufferDestroy(src);
    gfxBufferDestroy(dst);
}

// Lifecycle contract: a fresh encoder records without Begin; after submit + fence wait
// Begin resets it for reuse
TEST_P(GfxCommandEncoderTest, EncoderLifecycleRecordSubmitReuse)
{
    GfxQueue queue = nullptr;
    ASSERT_EQ(gfxDeviceGetQueue(device, &queue), GFX_RESULT_SUCCESS);

    GfxBufferDescriptor srcBufferDesc = {};
    srcBufferDesc.size = 256;
    srcBufferDesc.usage = GFX_BUFFER_USAGE_COPY_SRC;
    srcBufferDesc.memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL;
    GfxBuffer srcBuffer = nullptr;
    ASSERT_EQ(gfxDeviceCreateBuffer(device, &srcBufferDesc, &srcBuffer), GFX_RESULT_SUCCESS);

    GfxBufferDescriptor dstBufferDesc = {};
    dstBufferDesc.size = 256;
    dstBufferDesc.usage = GFX_BUFFER_USAGE_COPY_DST;
    dstBufferDesc.memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL;
    GfxBuffer dstBuffer = nullptr;
    ASSERT_EQ(gfxDeviceCreateBuffer(device, &dstBufferDesc, &dstBuffer), GFX_RESULT_SUCCESS);

    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor encoderDesc = {};
    encoderDesc.label = "lifecycle_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &encoderDesc, &encoder), GFX_RESULT_SUCCESS);

    GfxFenceDescriptor fenceDesc = {};
    GfxFence fence = nullptr;
    ASSERT_EQ(gfxDeviceCreateFence(device, &fenceDesc, &fence), GFX_RESULT_SUCCESS);

    GfxCopyBufferToBufferDescriptor copyDesc = {};
    copyDesc.source = srcBuffer;
    copyDesc.destination = dstBuffer;
    copyDesc.size = 256;

    for (int i = 0; i < 3; ++i) {
        if (i > 0) {
            // Reuse: reset the encoder once the previous submission completed
            ASSERT_EQ(gfxCommandEncoderBegin(encoder), GFX_RESULT_SUCCESS);
        }
        // A fresh (or re-begun) encoder records without any further setup
        ASSERT_EQ(gfxCommandEncoderCopyBufferToBuffer(encoder, &copyDesc), GFX_RESULT_SUCCESS);
        ASSERT_EQ(gfxCommandEncoderEnd(encoder), GFX_RESULT_SUCCESS);

        GfxSubmitDescriptor submitDesc = {};
        submitDesc.commandEncoders = &encoder;
        submitDesc.commandEncoderCount = 1;
        submitDesc.signalFence = fence;
        ASSERT_EQ(gfxQueueSubmit(queue, &submitDesc), GFX_RESULT_SUCCESS);

        ASSERT_EQ(gfxFenceWait(fence, GFX_TIMEOUT_INFINITE), GFX_RESULT_SUCCESS);
        ASSERT_EQ(gfxFenceReset(fence), GFX_RESULT_SUCCESS);
    }

    gfxFenceDestroy(fence);
    gfxCommandEncoderDestroy(encoder);
    gfxBufferDestroy(srcBuffer);
    gfxBufferDestroy(dstBuffer);
}

TEST_P(GfxCommandEncoderTest, CopyBufferToTextureWithNullEncoder)
{
    GfxCopyBufferToTextureDescriptor copyDesc = {};

    GfxResult result = gfxCommandEncoderCopyBufferToTexture(nullptr, &copyDesc);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, CopyBufferToTextureWithNullDescriptor)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxResult result = gfxCommandEncoderCopyBufferToTexture(encoder, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, CopyTextureToBufferWithNullEncoder)
{
    GfxCopyTextureToBufferDescriptor copyDesc = {};

    GfxResult result = gfxCommandEncoderCopyTextureToBuffer(nullptr, &copyDesc);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, CopyTextureToBufferWithNullDescriptor)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxResult result = gfxCommandEncoderCopyTextureToBuffer(encoder, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, CopyTextureToTextureWithNullEncoder)
{
    GfxCopyTextureToTextureDescriptor copyDesc = {};

    GfxResult result = gfxCommandEncoderCopyTextureToTexture(nullptr, &copyDesc);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, CopyTextureToTextureWithNullDescriptor)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxResult result = gfxCommandEncoderCopyTextureToTexture(encoder, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, BlitTextureToTextureWithNullEncoder)
{
    GfxBlitTextureToTextureDescriptor blitDesc = {};

    GfxResult result = gfxCommandEncoderBlitTextureToTexture(nullptr, &blitDesc);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, BlitTextureToTextureWithNullDescriptor)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxResult result = gfxCommandEncoderBlitTextureToTexture(encoder, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

// Pass encoder tests
TEST_P(GfxCommandEncoderTest, BeginRenderPassWithNullEncoder)
{
    GfxRenderPassBeginDescriptor beginDesc = {};
    GfxRenderPassEncoder passEncoder = nullptr;

    GfxResult result = gfxCommandEncoderBeginRenderPass(nullptr, &beginDesc, &passEncoder);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, BeginRenderPassWithNullDescriptor)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxRenderPassEncoder passEncoder = nullptr;
    GfxResult result = gfxCommandEncoderBeginRenderPass(encoder, nullptr, &passEncoder);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, BeginRenderPassWithNullOutput)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxRenderPassBeginDescriptor beginDesc = {};
    GfxResult result = gfxCommandEncoderBeginRenderPass(encoder, &beginDesc, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, BeginComputePassWithNullEncoder)
{
    GfxComputePassBeginDescriptor beginDesc = {};
    GfxComputePassEncoder passEncoder = nullptr;

    GfxResult result = gfxCommandEncoderBeginComputePass(nullptr, &beginDesc, &passEncoder);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, BeginComputePassWithNullDescriptor)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxComputePassEncoder passEncoder = nullptr;
    GfxResult result = gfxCommandEncoderBeginComputePass(encoder, nullptr, &passEncoder);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, BeginComputePassWithNullOutput)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxComputePassBeginDescriptor beginDesc = {};
    GfxResult result = gfxCommandEncoderBeginComputePass(encoder, &beginDesc, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, GenerateMipmapsWithNullEncoder)
{
    GfxResult result = gfxCommandEncoderGenerateMipmaps(nullptr, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, GenerateMipmapsWithNullTexture)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxResult result = gfxCommandEncoderGenerateMipmaps(encoder, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, GenerateMipmapsRangeWithNullEncoder)
{
    GfxResult result = gfxCommandEncoderGenerateMipmapsRange(nullptr, nullptr, 0, 1);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, GenerateMipmapsRangeWithNullTexture)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxResult result = gfxCommandEncoderGenerateMipmapsRange(encoder, nullptr, 0, 1);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, PipelineBarrierWithNullEncoder)
{
    GfxPipelineBarrierDescriptor barrierDesc = {};

    GfxResult result = gfxCommandEncoderPipelineBarrier(nullptr, &barrierDesc);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, PipelineBarrierWithNullDescriptor)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxResult result = gfxCommandEncoderPipelineBarrier(encoder, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(encoder);
}

TEST_P(GfxCommandEncoderTest, PipelineBarrierEmpty)
{
    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor desc = {};
    desc.label = "test_encoder";
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &desc, &encoder), GFX_RESULT_SUCCESS);

    GfxPipelineBarrierDescriptor barrierDesc = {};
    barrierDesc.memoryBarriers = nullptr;
    barrierDesc.memoryBarrierCount = 0;
    barrierDesc.bufferBarriers = nullptr;
    barrierDesc.bufferBarrierCount = 0;
    barrierDesc.textureBarriers = nullptr;
    barrierDesc.textureBarrierCount = 0;

    GfxResult result = gfxCommandEncoderPipelineBarrier(encoder, &barrierDesc);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);

    gfxCommandEncoderDestroy(encoder);
}

// ===========================================================================
// Render Bundle Command Encoder Tests
// ===========================================================================

TEST_P(GfxCommandEncoderTest, CreateRenderBundleCommandEncoderWithNullDevice)
{
    GfxCommandEncoder encoder = nullptr;
    GfxRenderBundleEncoderDescriptor desc = {};
    desc.sType = GFX_STRUCTURE_TYPE_RENDER_BUNDLE_ENCODER_DESCRIPTOR;
    desc.label = "test_bundle_encoder";

    GfxResult result = gfxDeviceCreateRenderBundleCommandEncoder(nullptr, &desc, &encoder);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, CreateRenderBundleCommandEncoderWithNullDescriptor)
{
    GfxCommandEncoder encoder = nullptr;

    GfxResult result = gfxDeviceCreateRenderBundleCommandEncoder(device, nullptr, &encoder);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxCommandEncoderTest, CreateRenderBundleCommandEncoderWithNullOutput)
{
    GfxRenderBundleEncoderDescriptor desc = {};
    desc.sType = GFX_STRUCTURE_TYPE_RENDER_BUNDLE_ENCODER_DESCRIPTOR;
    desc.label = "test_bundle_encoder";

    GfxResult result = gfxDeviceCreateRenderBundleCommandEncoder(device, &desc, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// ===========================================================================
// Test Instantiation
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    GfxCommandEncoderTest,
    testing::ValuesIn(getActiveBackends()),
    convertTestParamToString);

} // namespace
