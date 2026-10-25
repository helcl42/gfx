#include "CommonTest.h"

#include <cstring>
#include <vector>

// C API tests compiled with C++ for GoogleTest compatibility

// ===========================================================================
// Parameterized Tests - Run on both Vulkan and WebGPU backends
// ===========================================================================

namespace {

class GfxRenderPassEncoderTest : public testing::TestWithParam<GfxBackend> {
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
TEST_P(GfxRenderPassEncoderTest, SetPipelineWithNullEncoder)
{
    GfxResult result = gfxRenderPassEncoderSetPipeline(nullptr, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, SetBindGroupWithNullEncoder)
{
    GfxResult result = gfxRenderPassEncoderSetBindGroup(nullptr, 0, nullptr, nullptr, 0);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, SetVertexBufferWithNullEncoder)
{
    GfxResult result = gfxRenderPassEncoderSetVertexBuffer(nullptr, 0, nullptr, 0, 0);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, SetIndexBufferWithNullEncoder)
{
    GfxResult result = gfxRenderPassEncoderSetIndexBuffer(nullptr, nullptr, GFX_INDEX_FORMAT_UINT16, 0, 0);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, SetViewportWithNullEncoder)
{
    GfxViewport viewport = {};
    GfxResult result = gfxRenderPassEncoderSetViewport(nullptr, &viewport);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, SetViewportWithNullViewport)
{
    // Create a command encoder first
    GfxCommandEncoder cmdEncoder = nullptr;
    GfxCommandEncoderDescriptor cmdDesc = {};
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &cmdDesc, &cmdEncoder), GFX_RESULT_SUCCESS);

    // We need a render pass to get a render pass encoder, but for this test
    // we're just testing NULL validation, so we can skip creating one
    // and just test with nullptr
    GfxResult result = gfxRenderPassEncoderSetViewport(nullptr, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxCommandEncoderDestroy(cmdEncoder);
}

TEST_P(GfxRenderPassEncoderTest, SetScissorRectWithNullEncoder)
{
    GfxScissorRect scissor = {};
    GfxResult result = gfxRenderPassEncoderSetScissorRect(nullptr, &scissor);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, SetScissorRectWithNullScissor)
{
    GfxResult result = gfxRenderPassEncoderSetScissorRect(nullptr, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, SetBlendConstantWithNullEncoder)
{
    GfxColor color = { 1.0f, 0.5f, 0.25f, 1.0f };
    GfxResult result = gfxRenderPassEncoderSetBlendConstant(nullptr, &color);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, SetBlendConstantWithNullColor)
{
    GfxResult result = gfxRenderPassEncoderSetBlendConstant(nullptr, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, SetStencilReferenceWithNullEncoder)
{
    GfxResult result = gfxRenderPassEncoderSetStencilReference(nullptr, 0x42);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, DrawWithNullEncoder)
{
    GfxResult result = gfxRenderPassEncoderDraw(nullptr, 3, 1, 0, 0);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, DrawIndexedWithNullEncoder)
{
    GfxResult result = gfxRenderPassEncoderDrawIndexed(nullptr, 3, 1, 0, 0, 0);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, DrawIndirectWithNullEncoder)
{
    GfxResult result = gfxRenderPassEncoderDrawIndirect(nullptr, nullptr, 0);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, DrawIndexedIndirectWithNullEncoder)
{
    GfxResult result = gfxRenderPassEncoderDrawIndexedIndirect(nullptr, nullptr, 0);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxRenderPassEncoderTest, EndWithNullEncoder)
{
    GfxResult result = gfxRenderPassEncoderEnd(nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// ===========================================================================
// ExecuteBundles NULL parameter validation tests
// ===========================================================================

TEST_P(GfxRenderPassEncoderTest, ExecuteBundlesWithNullEncoder)
{
    GfxCommandEncoder bundle = nullptr;
    GfxResult result = gfxRenderPassEncoderExecuteBundles(nullptr, &bundle, 1);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// ===========================================================================
// Attachment layout tests
// ===========================================================================

// A copy after the pass must start from the pass's final layout, or the cleared contents are lost
TEST_P(GfxRenderPassEncoderTest, CopyAfterRenderPassKeepsContents)
{
    GfxQueue queue = nullptr;
    ASSERT_EQ(gfxDeviceGetQueue(device, &queue), GFX_RESULT_SUCCESS);

    constexpr uint32_t kSize = 64;
    constexpr uint32_t kBytesPerRow = kSize * 4; // 256, as WebGPU requires for texture-to-buffer copies

    GfxTextureDescriptor textureDesc = {};
    textureDesc.type = GFX_TEXTURE_TYPE_2D;
    textureDesc.size = { kSize, kSize, 1 };
    textureDesc.arrayLayerCount = 1;
    textureDesc.mipLevelCount = 1;
    textureDesc.sampleCount = GFX_SAMPLE_COUNT_1;
    textureDesc.format = GFX_FORMAT_R8G8B8A8_UNORM;
    textureDesc.usage = GFX_FLAGS(GFX_TEXTURE_USAGE_RENDER_ATTACHMENT | GFX_TEXTURE_USAGE_COPY_SRC | GFX_TEXTURE_USAGE_TEXTURE_BINDING);
    GfxTexture texture = nullptr;
    ASSERT_EQ(gfxDeviceCreateTexture(device, &textureDesc, &texture), GFX_RESULT_SUCCESS);

    GfxTextureViewDescriptor viewDesc = {};
    viewDesc.viewType = GFX_TEXTURE_VIEW_TYPE_2D;
    viewDesc.format = GFX_FORMAT_R8G8B8A8_UNORM;
    viewDesc.mipLevelCount = 1;
    viewDesc.arrayLayerCount = 1;
    GfxTextureView view = nullptr;
    ASSERT_EQ(gfxTextureCreateView(texture, &viewDesc, &view), GFX_RESULT_SUCCESS);

    GfxRenderPassColorAttachment colorAttachment = {};
    colorAttachment.target.format = GFX_FORMAT_R8G8B8A8_UNORM;
    colorAttachment.target.sampleCount = GFX_SAMPLE_COUNT_1;
    colorAttachment.target.ops.loadOp = GFX_LOAD_OP_CLEAR;
    colorAttachment.target.ops.storeOp = GFX_STORE_OP_STORE;
    colorAttachment.target.finalLayout = GFX_TEXTURE_LAYOUT_SHADER_READ_ONLY;

    GfxRenderPassDescriptor renderPassDesc = {};
    renderPassDesc.colorAttachments = &colorAttachment;
    renderPassDesc.colorAttachmentCount = 1;
    GfxRenderPass renderPass = nullptr;
    ASSERT_EQ(gfxDeviceCreateRenderPass(device, &renderPassDesc, &renderPass), GFX_RESULT_SUCCESS);

    GfxFramebufferAttachment framebufferAttachment = {};
    framebufferAttachment.view = view;
    GfxFramebufferDescriptor framebufferDesc = {};
    framebufferDesc.renderPass = renderPass;
    framebufferDesc.colorAttachments = &framebufferAttachment;
    framebufferDesc.colorAttachmentCount = 1;
    framebufferDesc.extent = { kSize, kSize };
    GfxFramebuffer framebuffer = nullptr;
    ASSERT_EQ(gfxDeviceCreateFramebuffer(device, &framebufferDesc, &framebuffer), GFX_RESULT_SUCCESS);

    GfxBufferDescriptor readbackDesc = {};
    readbackDesc.size = kBytesPerRow * kSize;
    readbackDesc.usage = GFX_FLAGS(GFX_BUFFER_USAGE_MAP_READ | GFX_BUFFER_USAGE_COPY_DST);
    readbackDesc.memoryProperties = GFX_FLAGS(GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT);
    GfxBuffer readback = nullptr;
    ASSERT_EQ(gfxDeviceCreateBuffer(device, &readbackDesc, &readback), GFX_RESULT_SUCCESS);

    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor encoderDesc = {};
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &encoderDesc, &encoder), GFX_RESULT_SUCCESS);

    const GfxColor clearColor = { 0.0f, 1.0f, 0.0f, 1.0f };
    GfxRenderPassBeginDescriptor beginDesc = {};
    beginDesc.renderPass = renderPass;
    beginDesc.framebuffer = framebuffer;
    beginDesc.colorClearValues = &clearColor;
    beginDesc.colorClearValueCount = 1;
    GfxRenderPassEncoder pass = nullptr;
    ASSERT_EQ(gfxCommandEncoderBeginRenderPass(encoder, &beginDesc, &pass), GFX_RESULT_SUCCESS);
    ASSERT_EQ(gfxRenderPassEncoderEnd(pass), GFX_RESULT_SUCCESS);

    GfxCopyTextureToBufferDescriptor copyDesc = {};
    copyDesc.source = texture;
    copyDesc.destination = readback;
    copyDesc.bytesPerRow = kBytesPerRow;
    copyDesc.extent = { kSize, kSize, 1 };
    copyDesc.finalLayout = GFX_TEXTURE_LAYOUT_SHADER_READ_ONLY;
    ASSERT_EQ(gfxCommandEncoderCopyTextureToBuffer(encoder, &copyDesc), GFX_RESULT_SUCCESS);
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

    void* mapped = nullptr;
    ASSERT_EQ(gfxBufferMap(readback, 0, readbackDesc.size, &mapped), GFX_RESULT_SUCCESS);
    const auto* bytes = static_cast<const uint8_t*>(mapped);
    const uint8_t green[4] = { 0x00, 0xFF, 0x00, 0xFF };
    EXPECT_EQ(std::memcmp(bytes, green, sizeof(green)), 0);
    EXPECT_EQ(std::memcmp(bytes + (kSize - 1) * kBytesPerRow + (kSize - 1) * 4, green, sizeof(green)), 0);
    ASSERT_EQ(gfxBufferUnmap(readback), GFX_RESULT_SUCCESS);

    gfxFenceDestroy(fence);
    gfxCommandEncoderDestroy(encoder);
    gfxBufferDestroy(readback);
    gfxFramebufferDestroy(framebuffer);
    gfxRenderPassDestroy(renderPass);
    gfxTextureViewDestroy(view);
    gfxTextureDestroy(texture);
}

// A loaded attachment keeps its contents whatever the texture was last used for
TEST_P(GfxRenderPassEncoderTest, LoadKeepsContentsAfterOtherUse)
{
    GfxQueue queue = nullptr;
    ASSERT_EQ(gfxDeviceGetQueue(device, &queue), GFX_RESULT_SUCCESS);

    constexpr uint32_t kSize = 64;
    constexpr uint32_t kBytesPerRow = kSize * 4;

    GfxTextureDescriptor textureDesc = {};
    textureDesc.type = GFX_TEXTURE_TYPE_2D;
    textureDesc.size = { kSize, kSize, 1 };
    textureDesc.arrayLayerCount = 1;
    textureDesc.mipLevelCount = 1;
    textureDesc.sampleCount = GFX_SAMPLE_COUNT_1;
    textureDesc.format = GFX_FORMAT_R8G8B8A8_UNORM;
    textureDesc.usage = GFX_FLAGS(GFX_TEXTURE_USAGE_RENDER_ATTACHMENT | GFX_TEXTURE_USAGE_COPY_SRC | GFX_TEXTURE_USAGE_COPY_DST | GFX_TEXTURE_USAGE_TEXTURE_BINDING);
    GfxTexture texture = nullptr;
    ASSERT_EQ(gfxDeviceCreateTexture(device, &textureDesc, &texture), GFX_RESULT_SUCCESS);

    // Fill with green and leave the texture ready for sampling, not for rendering
    const std::vector<uint8_t> green = [&] {
        std::vector<uint8_t> pixels(kSize * kSize * 4);
        for (size_t i = 0; i < pixels.size(); i += 4) {
            pixels[i + 1] = 0xFF;
            pixels[i + 3] = 0xFF;
        }
        return pixels;
    }();
    GfxWriteTextureDescriptor writeDesc = {};
    writeDesc.texture = texture;
    writeDesc.extent = { kSize, kSize, 1 };
    writeDesc.finalLayout = GFX_TEXTURE_LAYOUT_SHADER_READ_ONLY;
    ASSERT_EQ(gfxQueueWriteTexture(queue, &writeDesc, green.data(), green.size()), GFX_RESULT_SUCCESS);

    GfxTextureViewDescriptor viewDesc = {};
    viewDesc.viewType = GFX_TEXTURE_VIEW_TYPE_2D;
    viewDesc.format = GFX_FORMAT_R8G8B8A8_UNORM;
    viewDesc.mipLevelCount = 1;
    viewDesc.arrayLayerCount = 1;
    GfxTextureView view = nullptr;
    ASSERT_EQ(gfxTextureCreateView(texture, &viewDesc, &view), GFX_RESULT_SUCCESS);

    GfxRenderPassColorAttachment colorAttachment = {};
    colorAttachment.target.format = GFX_FORMAT_R8G8B8A8_UNORM;
    colorAttachment.target.sampleCount = GFX_SAMPLE_COUNT_1;
    colorAttachment.target.ops.loadOp = GFX_LOAD_OP_LOAD;
    colorAttachment.target.ops.storeOp = GFX_STORE_OP_STORE;
    colorAttachment.target.finalLayout = GFX_TEXTURE_LAYOUT_TRANSFER_SRC;

    GfxRenderPassDescriptor renderPassDesc = {};
    renderPassDesc.colorAttachments = &colorAttachment;
    renderPassDesc.colorAttachmentCount = 1;
    GfxRenderPass renderPass = nullptr;
    ASSERT_EQ(gfxDeviceCreateRenderPass(device, &renderPassDesc, &renderPass), GFX_RESULT_SUCCESS);

    GfxFramebufferAttachment framebufferAttachment = {};
    framebufferAttachment.view = view;
    GfxFramebufferDescriptor framebufferDesc = {};
    framebufferDesc.renderPass = renderPass;
    framebufferDesc.colorAttachments = &framebufferAttachment;
    framebufferDesc.colorAttachmentCount = 1;
    framebufferDesc.extent = { kSize, kSize };
    GfxFramebuffer framebuffer = nullptr;
    ASSERT_EQ(gfxDeviceCreateFramebuffer(device, &framebufferDesc, &framebuffer), GFX_RESULT_SUCCESS);

    GfxBufferDescriptor readbackDesc = {};
    readbackDesc.size = kBytesPerRow * kSize;
    readbackDesc.usage = GFX_FLAGS(GFX_BUFFER_USAGE_MAP_READ | GFX_BUFFER_USAGE_COPY_DST);
    readbackDesc.memoryProperties = GFX_FLAGS(GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT);
    GfxBuffer readback = nullptr;
    ASSERT_EQ(gfxDeviceCreateBuffer(device, &readbackDesc, &readback), GFX_RESULT_SUCCESS);

    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor encoderDesc = {};
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &encoderDesc, &encoder), GFX_RESULT_SUCCESS);

    GfxRenderPassBeginDescriptor beginDesc = {};
    beginDesc.renderPass = renderPass;
    beginDesc.framebuffer = framebuffer;
    GfxRenderPassEncoder pass = nullptr;
    ASSERT_EQ(gfxCommandEncoderBeginRenderPass(encoder, &beginDesc, &pass), GFX_RESULT_SUCCESS);
    ASSERT_EQ(gfxRenderPassEncoderEnd(pass), GFX_RESULT_SUCCESS);

    GfxCopyTextureToBufferDescriptor copyDesc = {};
    copyDesc.source = texture;
    copyDesc.destination = readback;
    copyDesc.bytesPerRow = kBytesPerRow;
    copyDesc.extent = { kSize, kSize, 1 };
    copyDesc.finalLayout = GFX_TEXTURE_LAYOUT_TRANSFER_SRC;
    ASSERT_EQ(gfxCommandEncoderCopyTextureToBuffer(encoder, &copyDesc), GFX_RESULT_SUCCESS);
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

    void* mapped = nullptr;
    ASSERT_EQ(gfxBufferMap(readback, 0, readbackDesc.size, &mapped), GFX_RESULT_SUCCESS);
    const auto* bytes = static_cast<const uint8_t*>(mapped);
    EXPECT_EQ(std::memcmp(bytes, green.data(), 4), 0);
    EXPECT_EQ(std::memcmp(bytes + (kSize - 1) * kBytesPerRow + (kSize - 1) * 4, green.data(), 4), 0);
    ASSERT_EQ(gfxBufferUnmap(readback), GFX_RESULT_SUCCESS);

    gfxFenceDestroy(fence);
    gfxCommandEncoderDestroy(encoder);
    gfxBufferDestroy(readback);
    gfxFramebufferDestroy(framebuffer);
    gfxRenderPassDestroy(renderPass);
    gfxTextureViewDestroy(view);
    gfxTextureDestroy(texture);
}

// A cleared color attachment without a clear value clears to zero, as in WebGPU
TEST_P(GfxRenderPassEncoderTest, ClearWithoutClearValueClearsToZero)
{
    GfxQueue queue = nullptr;
    ASSERT_EQ(gfxDeviceGetQueue(device, &queue), GFX_RESULT_SUCCESS);

    constexpr uint32_t kSize = 64;
    constexpr uint32_t kBytesPerRow = kSize * 4;

    GfxTextureDescriptor textureDesc = {};
    textureDesc.type = GFX_TEXTURE_TYPE_2D;
    textureDesc.size = { kSize, kSize, 1 };
    textureDesc.arrayLayerCount = 1;
    textureDesc.mipLevelCount = 1;
    textureDesc.sampleCount = GFX_SAMPLE_COUNT_1;
    textureDesc.format = GFX_FORMAT_R8G8B8A8_UNORM;
    textureDesc.usage = GFX_FLAGS(GFX_TEXTURE_USAGE_RENDER_ATTACHMENT | GFX_TEXTURE_USAGE_COPY_SRC);
    GfxTexture texture = nullptr;
    ASSERT_EQ(gfxDeviceCreateTexture(device, &textureDesc, &texture), GFX_RESULT_SUCCESS);

    GfxTextureViewDescriptor viewDesc = {};
    viewDesc.viewType = GFX_TEXTURE_VIEW_TYPE_2D;
    viewDesc.format = GFX_FORMAT_R8G8B8A8_UNORM;
    viewDesc.mipLevelCount = 1;
    viewDesc.arrayLayerCount = 1;
    GfxTextureView view = nullptr;
    ASSERT_EQ(gfxTextureCreateView(texture, &viewDesc, &view), GFX_RESULT_SUCCESS);

    GfxRenderPassColorAttachment colorAttachment = {};
    colorAttachment.target.format = GFX_FORMAT_R8G8B8A8_UNORM;
    colorAttachment.target.sampleCount = GFX_SAMPLE_COUNT_1;
    colorAttachment.target.ops.loadOp = GFX_LOAD_OP_CLEAR;
    colorAttachment.target.ops.storeOp = GFX_STORE_OP_STORE;
    colorAttachment.target.finalLayout = GFX_TEXTURE_LAYOUT_TRANSFER_SRC;

    GfxRenderPassDescriptor renderPassDesc = {};
    renderPassDesc.colorAttachments = &colorAttachment;
    renderPassDesc.colorAttachmentCount = 1;
    GfxRenderPass renderPass = nullptr;
    ASSERT_EQ(gfxDeviceCreateRenderPass(device, &renderPassDesc, &renderPass), GFX_RESULT_SUCCESS);

    GfxFramebufferAttachment framebufferAttachment = {};
    framebufferAttachment.view = view;
    GfxFramebufferDescriptor framebufferDesc = {};
    framebufferDesc.renderPass = renderPass;
    framebufferDesc.colorAttachments = &framebufferAttachment;
    framebufferDesc.colorAttachmentCount = 1;
    framebufferDesc.extent = { kSize, kSize };
    GfxFramebuffer framebuffer = nullptr;
    ASSERT_EQ(gfxDeviceCreateFramebuffer(device, &framebufferDesc, &framebuffer), GFX_RESULT_SUCCESS);

    GfxBufferDescriptor readbackDesc = {};
    readbackDesc.size = kBytesPerRow * kSize;
    readbackDesc.usage = GFX_FLAGS(GFX_BUFFER_USAGE_MAP_READ | GFX_BUFFER_USAGE_COPY_DST);
    readbackDesc.memoryProperties = GFX_FLAGS(GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT);
    GfxBuffer readback = nullptr;
    ASSERT_EQ(gfxDeviceCreateBuffer(device, &readbackDesc, &readback), GFX_RESULT_SUCCESS);

    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor encoderDesc = {};
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &encoderDesc, &encoder), GFX_RESULT_SUCCESS);

    // Fill with green first, so a zero result cannot come from the texture's initial contents
    const GfxColor green = { 0.0f, 1.0f, 0.0f, 1.0f };
    GfxRenderPassBeginDescriptor beginDesc = {};
    beginDesc.renderPass = renderPass;
    beginDesc.framebuffer = framebuffer;
    beginDesc.colorClearValues = &green;
    beginDesc.colorClearValueCount = 1;
    GfxRenderPassEncoder pass = nullptr;
    ASSERT_EQ(gfxCommandEncoderBeginRenderPass(encoder, &beginDesc, &pass), GFX_RESULT_SUCCESS);
    ASSERT_EQ(gfxRenderPassEncoderEnd(pass), GFX_RESULT_SUCCESS);

    beginDesc.colorClearValues = nullptr;
    beginDesc.colorClearValueCount = 0;
    ASSERT_EQ(gfxCommandEncoderBeginRenderPass(encoder, &beginDesc, &pass), GFX_RESULT_SUCCESS);
    ASSERT_EQ(gfxRenderPassEncoderEnd(pass), GFX_RESULT_SUCCESS);

    GfxCopyTextureToBufferDescriptor copyDesc = {};
    copyDesc.source = texture;
    copyDesc.destination = readback;
    copyDesc.bytesPerRow = kBytesPerRow;
    copyDesc.extent = { kSize, kSize, 1 };
    copyDesc.finalLayout = GFX_TEXTURE_LAYOUT_TRANSFER_SRC;
    ASSERT_EQ(gfxCommandEncoderCopyTextureToBuffer(encoder, &copyDesc), GFX_RESULT_SUCCESS);
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

    void* mapped = nullptr;
    ASSERT_EQ(gfxBufferMap(readback, 0, readbackDesc.size, &mapped), GFX_RESULT_SUCCESS);
    const auto* bytes = static_cast<const uint8_t*>(mapped);
    const uint8_t zero[4] = {};
    EXPECT_EQ(std::memcmp(bytes, zero, sizeof(zero)), 0);
    EXPECT_EQ(std::memcmp(bytes + (kSize - 1) * kBytesPerRow + (kSize - 1) * 4, zero, sizeof(zero)), 0);
    ASSERT_EQ(gfxBufferUnmap(readback), GFX_RESULT_SUCCESS);

    gfxFenceDestroy(fence);
    gfxCommandEncoderDestroy(encoder);
    gfxBufferDestroy(readback);
    gfxFramebufferDestroy(framebuffer);
    gfxRenderPassDestroy(renderPass);
    gfxTextureViewDestroy(view);
    gfxTextureDestroy(texture);
}

// ===========================================================================
// Test Instantiation
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    GfxRenderPassEncoderTest,
    testing::ValuesIn(getActiveBackends()),
    convertTestParamToString);

} // namespace
