#include "CommonTest.h"

#include <cstring>

// C API tests compiled with C++ for GoogleTest compatibility

// ===========================================================================
// Parameterized Tests - Run on both Vulkan and WebGPU backends
// ===========================================================================

namespace {

class GfxComputePipelineTest : public testing::TestWithParam<GfxBackend> {
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

// Simple WGSL compute shader - fills output with red
static const char* wgslComputeShader = R"(
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let index = globalId.x;
    output[index] = 1.0;
}
)";

// Simple SPIR-V compute shader binary - minimal shader for testing
// Does nothing, just tests pipeline creation
static const uint32_t spirvComputeShader[] = {
    0x07230203,
    0x00010000,
    0x0008000b,
    0x0000000b,
    0x00000000,
    0x00020011,
    0x00000001,
    0x0006000b,
    0x00000001,
    0x4c534c47,
    0x6474732e,
    0x3035342e,
    0x00000000,
    0x0003000e,
    0x00000000,
    0x00000001,
    0x0005000f,
    0x00000005,
    0x00000004,
    0x6e69616d,
    0x00000000,
    0x00060010,
    0x00000004,
    0x00000011,
    0x00000040,
    0x00000001,
    0x00000001,
    0x00030003,
    0x00000002,
    0x000001c2,
    0x00040005,
    0x00000004,
    0x6e69616d,
    0x00000000,
    0x00040047,
    0x0000000a,
    0x0000000b,
    0x00000019,
    0x00020013,
    0x00000002,
    0x00030021,
    0x00000003,
    0x00000002,
    0x00040015,
    0x00000006,
    0x00000020,
    0x00000000,
    0x00040017,
    0x00000007,
    0x00000006,
    0x00000003,
    0x0004002b,
    0x00000006,
    0x00000008,
    0x00000040,
    0x0004002b,
    0x00000006,
    0x00000009,
    0x00000001,
    0x0006002c,
    0x00000007,
    0x0000000a,
    0x00000008,
    0x00000009,
    0x00000009,
    0x00050036,
    0x00000002,
    0x00000004,
    0x00000000,
    0x00000003,
    0x000200f8,
    0x00000005,
    0x000100fd,
    0x00010038,
};

// Compute shader with pipeline constants; defaults are 7, 1.5 and false.
static const char* wgslConstantsComputeShader = R"(
@id(0) override valueU: u32 = 7u;
@id(1) override valueF: f32 = 1.5;
@id(2) override flag: bool = false;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main() {
    data[0] = valueU;
    data[1] = u32(valueF * 2.0);
    data[2] = select(0u, 1u, flag);
}
)";

// Slang-compiled from the same source; carries SpecId-decorated OpSpecConstants.
static const uint32_t spirvConstantsComputeShader[] = {
    0x07230203,
    0x00010300,
    0x00280000,
    0x0000002b,
    0x00000000,
    0x00020011,
    0x00000001,
    0x0003000e,
    0x00000000,
    0x00000001,
    0x0005000f,
    0x00000005,
    0x00000002,
    0x6e69616d,
    0x00000000,
    0x00060010,
    0x00000002,
    0x00000011,
    0x00000001,
    0x00000001,
    0x00000001,
    0x00030003,
    0x0000000b,
    0x00000001,
    0x00070005,
    0x0000000f,
    0x74535752,
    0x74637572,
    0x64657275,
    0x66667542,
    0x00007265,
    0x00060006,
    0x0000000f,
    0x00000000,
    0x656d5f5f,
    0x7265626d,
    0x00000030,
    0x00040005,
    0x00000012,
    0x61746164,
    0x00000000,
    0x00040005,
    0x00000013,
    0x756c6176,
    0x00005565,
    0x00040005,
    0x00000019,
    0x756c6176,
    0x00004665,
    0x00040005,
    0x00000020,
    0x67616c66,
    0x00000000,
    0x00040005,
    0x00000002,
    0x6e69616d,
    0x00000000,
    0x00040047,
    0x00000010,
    0x00000006,
    0x00000004,
    0x00030047,
    0x0000000f,
    0x00000003,
    0x00050048,
    0x0000000f,
    0x00000000,
    0x00000023,
    0x00000000,
    0x00040047,
    0x00000012,
    0x00000021,
    0x00000000,
    0x00040047,
    0x00000012,
    0x00000022,
    0x00000000,
    0x00040047,
    0x00000013,
    0x00000001,
    0x00000000,
    0x00040047,
    0x00000019,
    0x00000001,
    0x00000001,
    0x00040047,
    0x00000020,
    0x00000001,
    0x00000002,
    0x00020013,
    0x00000001,
    0x00030021,
    0x00000003,
    0x00000001,
    0x00040015,
    0x00000005,
    0x00000020,
    0x00000000,
    0x00040020,
    0x00000006,
    0x00000007,
    0x00000005,
    0x00040015,
    0x0000000b,
    0x00000020,
    0x00000001,
    0x0004002b,
    0x0000000b,
    0x0000000c,
    0x00000000,
    0x00040020,
    0x0000000d,
    0x00000002,
    0x00000005,
    0x0003001d,
    0x00000010,
    0x00000005,
    0x0003001e,
    0x0000000f,
    0x00000010,
    0x00040020,
    0x00000011,
    0x00000002,
    0x0000000f,
    0x00040032,
    0x00000005,
    0x00000013,
    0x00000007,
    0x0004002b,
    0x0000000b,
    0x00000015,
    0x00000001,
    0x00030016,
    0x00000017,
    0x00000020,
    0x00040032,
    0x00000017,
    0x00000019,
    0x3fc00000,
    0x0004002b,
    0x00000017,
    0x0000001a,
    0x40000000,
    0x0004002b,
    0x0000000b,
    0x0000001d,
    0x00000002,
    0x00020014,
    0x0000001f,
    0x00030031,
    0x0000001f,
    0x00000020,
    0x0004002b,
    0x00000005,
    0x00000022,
    0x00000001,
    0x0004002b,
    0x00000005,
    0x00000025,
    0x00000000,
    0x0004003b,
    0x00000011,
    0x00000012,
    0x00000002,
    0x00050036,
    0x00000001,
    0x00000002,
    0x00000000,
    0x00000003,
    0x000200f8,
    0x00000004,
    0x0004003b,
    0x00000006,
    0x00000007,
    0x00000007,
    0x00060041,
    0x0000000d,
    0x0000000e,
    0x00000012,
    0x0000000c,
    0x0000000c,
    0x0003003e,
    0x0000000e,
    0x00000013,
    0x00060041,
    0x0000000d,
    0x00000016,
    0x00000012,
    0x0000000c,
    0x00000015,
    0x00050085,
    0x00000017,
    0x00000018,
    0x00000019,
    0x0000001a,
    0x0004006d,
    0x00000005,
    0x0000001b,
    0x00000018,
    0x0003003e,
    0x00000016,
    0x0000001b,
    0x00060041,
    0x0000000d,
    0x0000001e,
    0x00000012,
    0x0000000c,
    0x0000001d,
    0x000300f7,
    0x0000000a,
    0x00000000,
    0x000400fa,
    0x00000020,
    0x00000008,
    0x00000009,
    0x000200f8,
    0x00000008,
    0x0003003e,
    0x00000007,
    0x00000022,
    0x000200f9,
    0x0000000a,
    0x000200f8,
    0x00000009,
    0x0003003e,
    0x00000007,
    0x00000025,
    0x000200f9,
    0x0000000a,
    0x000200f8,
    0x0000000a,
    0x0004003d,
    0x00000005,
    0x00000028,
    0x00000007,
    0x0003003e,
    0x0000001e,
    0x00000028,
    0x000100fd,
    0x00010038,
};

// ===========================================================================
// ComputePipeline Tests
// ===========================================================================

// Test: Create ComputePipeline with NULL device
TEST_P(GfxComputePipelineTest, CreateComputePipelineWithNullDevice)
{
    GfxShaderDescriptor shaderDesc = {};
    shaderDesc.label = "Test Compute Shader";
    if (backend == GFX_BACKEND_VULKAN) {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_SPIRV;
        shaderDesc.code = spirvComputeShader;
        shaderDesc.codeSize = sizeof(spirvComputeShader);
    } else {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_WGSL;
        shaderDesc.code = wgslComputeShader;
        shaderDesc.codeSize = strlen(wgslComputeShader) + 1;
    }
    shaderDesc.entryPoint = "main";

    GfxShader computeShader = nullptr;
    GfxResult result = gfxDeviceCreateShader(device, &shaderDesc, &computeShader);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    GfxComputeState computeState = {};
    computeState.module = computeShader;
    computeState.entryPoint = "main";

    GfxComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.label = "Test Compute Pipeline";
    pipelineDesc.compute = &computeState;

    GfxComputePipeline pipeline = nullptr;
    result = gfxDeviceCreateComputePipeline(nullptr, &pipelineDesc, &pipeline);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxShaderDestroy(computeShader);
}

// Test: Create ComputePipeline with NULL descriptor
TEST_P(GfxComputePipelineTest, CreateComputePipelineWithNullDescriptor)
{
    GfxComputePipeline pipeline = nullptr;
    GfxResult result = gfxDeviceCreateComputePipeline(device, nullptr, &pipeline);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Test: Create ComputePipeline with NULL output
TEST_P(GfxComputePipelineTest, CreateComputePipelineWithNullOutput)
{
    GfxShaderDescriptor shaderDesc = {};
    shaderDesc.label = "Test Compute Shader";
    if (backend == GFX_BACKEND_VULKAN) {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_SPIRV;
        shaderDesc.code = spirvComputeShader;
        shaderDesc.codeSize = sizeof(spirvComputeShader);
    } else {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_WGSL;
        shaderDesc.code = wgslComputeShader;
        shaderDesc.codeSize = strlen(wgslComputeShader) + 1;
    }
    shaderDesc.entryPoint = "main";

    GfxShader computeShader = nullptr;
    GfxResult result = gfxDeviceCreateShader(device, &shaderDesc, &computeShader);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    GfxComputeState computeState = {};
    computeState.module = computeShader;
    computeState.entryPoint = "main";

    GfxComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.label = "Test Compute Pipeline";
    pipelineDesc.compute = &computeState;

    result = gfxDeviceCreateComputePipeline(device, &pipelineDesc, nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxShaderDestroy(computeShader);
}

// Test: Create basic ComputePipeline
TEST_P(GfxComputePipelineTest, CreateBasicComputePipeline)
{
    GfxShaderDescriptor shaderDesc = {};
    shaderDesc.label = "Test Compute Shader";
    if (backend == GFX_BACKEND_VULKAN) {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_SPIRV;
        shaderDesc.code = spirvComputeShader;
        shaderDesc.codeSize = sizeof(spirvComputeShader);
    } else {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_WGSL;
        shaderDesc.code = wgslComputeShader;
        shaderDesc.codeSize = strlen(wgslComputeShader) + 1;
    }
    shaderDesc.entryPoint = "main";

    GfxShader computeShader = nullptr;
    GfxResult result = gfxDeviceCreateShader(device, &shaderDesc, &computeShader);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);
    ASSERT_NE(computeShader, nullptr);

    GfxComputeState computeState = {};
    computeState.module = computeShader;
    computeState.entryPoint = nullptr; // Let it use shader's entry point

    GfxComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.label = "Basic Compute Pipeline";
    pipelineDesc.bindGroupLayouts = nullptr;
    pipelineDesc.bindGroupLayoutCount = 0;
    pipelineDesc.compute = &computeState;

    GfxComputePipeline pipeline = nullptr;
    result = gfxDeviceCreateComputePipeline(device, &pipelineDesc, &pipeline);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
    EXPECT_NE(pipeline, nullptr);

    gfxComputePipelineDestroy(pipeline);
    gfxShaderDestroy(computeShader);
}

// Test: Create ComputePipeline with bind group layouts
TEST_P(GfxComputePipelineTest, CreateComputePipelineWithBindGroupLayouts)
{
    // Create a storage buffer bind group layout
    GfxBindGroupLayoutEntry entry = {};
    entry.binding = 0;
    entry.visibility = GFX_SHADER_STAGE_COMPUTE;
    entry.type = GFX_BINDING_TYPE_UNIFORM_BUFFER;
    entry.uniformBuffer.hasDynamicOffset = false;
    entry.uniformBuffer.minBindingSize = 0;

    GfxBindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.label = "Compute Bind Group Layout";
    layoutDesc.entries = &entry;
    layoutDesc.entryCount = 1;

    GfxBindGroupLayout bindGroupLayout = nullptr;
    GfxResult result = gfxDeviceCreateBindGroupLayout(device, &layoutDesc, &bindGroupLayout);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);
    ASSERT_NE(bindGroupLayout, nullptr);

    // Create compute shader
    GfxShaderDescriptor shaderDesc = {};
    shaderDesc.label = "Test Compute Shader";
    if (backend == GFX_BACKEND_VULKAN) {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_SPIRV;
        shaderDesc.code = spirvComputeShader;
        shaderDesc.codeSize = sizeof(spirvComputeShader);
    } else {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_WGSL;
        shaderDesc.code = wgslComputeShader;
        shaderDesc.codeSize = strlen(wgslComputeShader) + 1;
    }
    shaderDesc.entryPoint = "main";

    GfxShader computeShader = nullptr;
    result = gfxDeviceCreateShader(device, &shaderDesc, &computeShader);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);
    ASSERT_NE(computeShader, nullptr);

    // Create compute pipeline with bind group layout
    GfxComputeState computeState = {};
    computeState.module = computeShader;
    computeState.entryPoint = "main";

    GfxComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.label = "Compute Pipeline with Bind Groups";
    pipelineDesc.bindGroupLayouts = &bindGroupLayout;
    pipelineDesc.bindGroupLayoutCount = 1;
    pipelineDesc.compute = &computeState;

    GfxComputePipeline pipeline = nullptr;
    result = gfxDeviceCreateComputePipeline(device, &pipelineDesc, &pipeline);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
    EXPECT_NE(pipeline, nullptr);

    gfxComputePipelineDestroy(pipeline);
    gfxShaderDestroy(computeShader);
    gfxBindGroupLayoutDestroy(bindGroupLayout);
}

// Test: Create ComputePipeline with multiple bind group layouts
TEST_P(GfxComputePipelineTest, CreateComputePipelineWithMultipleBindGroupLayouts)
{
    // Create first bind group layout (storage buffer)
    GfxBindGroupLayoutEntry entry1 = {};
    entry1.binding = 0;
    entry1.visibility = GFX_SHADER_STAGE_COMPUTE;
    entry1.type = GFX_BINDING_TYPE_UNIFORM_BUFFER;
    entry1.uniformBuffer.hasDynamicOffset = false;
    entry1.uniformBuffer.minBindingSize = 0;

    GfxBindGroupLayoutDescriptor layoutDesc1 = {};
    layoutDesc1.label = "Storage Buffer Layout";
    layoutDesc1.entries = &entry1;
    layoutDesc1.entryCount = 1;

    GfxBindGroupLayout bindGroupLayout1 = nullptr;
    GfxResult result = gfxDeviceCreateBindGroupLayout(device, &layoutDesc1, &bindGroupLayout1);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    // Create second bind group layout (uniform buffer)
    GfxBindGroupLayoutEntry entry2 = {};
    entry2.binding = 0;
    entry2.visibility = GFX_SHADER_STAGE_COMPUTE;
    entry2.type = GFX_BINDING_TYPE_UNIFORM_BUFFER;
    entry2.uniformBuffer.hasDynamicOffset = false;
    entry2.uniformBuffer.minBindingSize = 0;

    GfxBindGroupLayoutDescriptor layoutDesc2 = {};
    layoutDesc2.label = "Uniform Buffer Layout";
    layoutDesc2.entries = &entry2;
    layoutDesc2.entryCount = 1;

    GfxBindGroupLayout bindGroupLayout2 = nullptr;
    result = gfxDeviceCreateBindGroupLayout(device, &layoutDesc2, &bindGroupLayout2);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    // Create compute shader
    GfxShaderDescriptor shaderDesc = {};
    shaderDesc.label = "Test Compute Shader";
    if (backend == GFX_BACKEND_VULKAN) {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_SPIRV;
        shaderDesc.code = spirvComputeShader;
        shaderDesc.codeSize = sizeof(spirvComputeShader);
    } else {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_WGSL;
        shaderDesc.code = wgslComputeShader;
        shaderDesc.codeSize = strlen(wgslComputeShader) + 1;
    }
    shaderDesc.entryPoint = "main";

    GfxShader computeShader = nullptr;
    result = gfxDeviceCreateShader(device, &shaderDesc, &computeShader);
    ASSERT_EQ(result, GFX_RESULT_SUCCESS);

    // Create pipeline with multiple bind group layouts
    GfxBindGroupLayout layouts[] = { bindGroupLayout1, bindGroupLayout2 };

    GfxComputeState computeState = {};
    computeState.module = computeShader;
    computeState.entryPoint = "main";

    GfxComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.label = "Compute Pipeline with Multiple Bind Groups";
    pipelineDesc.bindGroupLayouts = layouts;
    pipelineDesc.bindGroupLayoutCount = 2;
    pipelineDesc.compute = &computeState;

    GfxComputePipeline pipeline = nullptr;
    result = gfxDeviceCreateComputePipeline(device, &pipelineDesc, &pipeline);
    EXPECT_EQ(result, GFX_RESULT_SUCCESS);
    EXPECT_NE(pipeline, nullptr);

    gfxComputePipelineDestroy(pipeline);
    gfxShaderDestroy(computeShader);
    gfxBindGroupLayoutDestroy(bindGroupLayout2);
    gfxBindGroupLayoutDestroy(bindGroupLayout1);
}

// Test: Destroy NULL ComputePipeline
TEST_P(GfxComputePipelineTest, DestroyNullComputePipeline)
{
    GfxResult result = gfxComputePipelineDestroy(nullptr);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// ===========================================================================
// Pipeline Constant Tests
// ===========================================================================

// Every expected value differs from the shader default, so dropped constants fail here.
TEST_P(GfxComputePipelineTest, ComputePipelineConstantsReachShader)
{
    GfxQueue queue = nullptr;
    ASSERT_EQ(gfxDeviceGetQueue(device, &queue), GFX_RESULT_SUCCESS);

    GfxShaderDescriptor shaderDesc = {};
    shaderDesc.label = "Constants Compute Shader";
    if (backend == GFX_BACKEND_VULKAN) {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_SPIRV;
        shaderDesc.code = spirvConstantsComputeShader;
        shaderDesc.codeSize = sizeof(spirvConstantsComputeShader);
    } else {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_WGSL;
        shaderDesc.code = wgslConstantsComputeShader;
        shaderDesc.codeSize = strlen(wgslConstantsComputeShader) + 1;
    }
    shaderDesc.entryPoint = "main";

    GfxShader computeShader = nullptr;
    ASSERT_EQ(gfxDeviceCreateShader(device, &shaderDesc, &computeShader), GFX_RESULT_SUCCESS);

    GfxBindGroupLayoutEntry layoutEntry = {};
    layoutEntry.binding = 0;
    layoutEntry.visibility = GFX_SHADER_STAGE_COMPUTE;
    layoutEntry.type = GFX_BINDING_TYPE_STORAGE_BUFFER;
    layoutEntry.storageBuffer.access = GFX_STORAGE_BUFFER_ACCESS_READ_WRITE;

    GfxBindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entries = &layoutEntry;
    layoutDesc.entryCount = 1;

    GfxBindGroupLayout layout = nullptr;
    ASSERT_EQ(gfxDeviceCreateBindGroupLayout(device, &layoutDesc, &layout), GFX_RESULT_SUCCESS);

    constexpr uint64_t kSize = 256;

    GfxBufferDescriptor storageDesc = {};
    storageDesc.size = kSize;
    storageDesc.usage = GFX_FLAGS(GFX_BUFFER_USAGE_STORAGE | GFX_BUFFER_USAGE_COPY_SRC);
    storageDesc.memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL;
    GfxBuffer storage = nullptr;
    ASSERT_EQ(gfxDeviceCreateBuffer(device, &storageDesc, &storage), GFX_RESULT_SUCCESS);

    GfxBufferDescriptor readbackDesc = {};
    readbackDesc.size = kSize;
    readbackDesc.usage = GFX_FLAGS(GFX_BUFFER_USAGE_MAP_READ | GFX_BUFFER_USAGE_COPY_DST);
    readbackDesc.memoryProperties = GFX_FLAGS(GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT);
    GfxBuffer readback = nullptr;
    ASSERT_EQ(gfxDeviceCreateBuffer(device, &readbackDesc, &readback), GFX_RESULT_SUCCESS);

    GfxBindGroupEntry bindEntry = {};
    bindEntry.binding = 0;
    bindEntry.type = GFX_BIND_GROUP_ENTRY_TYPE_BUFFER;
    bindEntry.resource.buffer.buffer = storage;
    bindEntry.resource.buffer.offset = 0;
    bindEntry.resource.buffer.size = GFX_WHOLE_SIZE;

    GfxBindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = layout;
    bindGroupDesc.entries = &bindEntry;
    bindGroupDesc.entryCount = 1;

    GfxBindGroup bindGroup = nullptr;
    ASSERT_EQ(gfxDeviceCreateBindGroup(device, &bindGroupDesc, &bindGroup), GFX_RESULT_SUCCESS);

    GfxConstantEntry constants[3] = {};
    constants[0].id = 0;
    constants[0].type = GFX_CONSTANT_TYPE_U32;
    constants[0].value.u32 = 11;
    constants[1].id = 1;
    constants[1].type = GFX_CONSTANT_TYPE_F32;
    constants[1].value.f32 = 2.5f;
    constants[2].id = 2;
    constants[2].type = GFX_CONSTANT_TYPE_BOOL;
    constants[2].value.b = true;

    GfxComputeState computeState = {};
    computeState.module = computeShader;
    computeState.entryPoint = "main";
    computeState.constants = constants;
    computeState.constantCount = 3;

    GfxComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.label = "Constants Compute Pipeline";
    pipelineDesc.bindGroupLayouts = &layout;
    pipelineDesc.bindGroupLayoutCount = 1;
    pipelineDesc.compute = &computeState;

    GfxComputePipeline pipeline = nullptr;
    ASSERT_EQ(gfxDeviceCreateComputePipeline(device, &pipelineDesc, &pipeline), GFX_RESULT_SUCCESS);

    GfxCommandEncoder encoder = nullptr;
    GfxCommandEncoderDescriptor encoderDesc = {};
    ASSERT_EQ(gfxDeviceCreateCommandEncoder(device, &encoderDesc, &encoder), GFX_RESULT_SUCCESS);

    GfxComputePassEncoder pass = nullptr;
    GfxComputePassBeginDescriptor passDesc = {};
    ASSERT_EQ(gfxCommandEncoderBeginComputePass(encoder, &passDesc, &pass), GFX_RESULT_SUCCESS);
    ASSERT_EQ(gfxComputePassEncoderSetPipeline(pass, pipeline), GFX_RESULT_SUCCESS);
    ASSERT_EQ(gfxComputePassEncoderSetBindGroup(pass, 0, bindGroup, nullptr, 0), GFX_RESULT_SUCCESS);
    ASSERT_EQ(gfxComputePassEncoderDispatch(pass, 1, 1, 1), GFX_RESULT_SUCCESS);
    ASSERT_EQ(gfxComputePassEncoderEnd(pass), GFX_RESULT_SUCCESS);

    GfxCopyBufferToBufferDescriptor copyDesc = {};
    copyDesc.source = storage;
    copyDesc.sourceOffset = 0;
    copyDesc.destination = readback;
    copyDesc.destinationOffset = 0;
    copyDesc.size = kSize;
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

    void* mapped = nullptr;
    ASSERT_EQ(gfxBufferMap(readback, 0, kSize, &mapped), GFX_RESULT_SUCCESS);
    const auto* values = static_cast<const uint32_t*>(mapped);
    EXPECT_EQ(values[0], 11u); // default 7
    EXPECT_EQ(values[1], 5u); // 2.5 * 2, default would be 3
    EXPECT_EQ(values[2], 1u); // default false
    ASSERT_EQ(gfxBufferUnmap(readback), GFX_RESULT_SUCCESS);

    gfxFenceDestroy(fence);
    gfxCommandEncoderDestroy(encoder);
    gfxComputePipelineDestroy(pipeline);
    gfxBindGroupDestroy(bindGroup);
    gfxBufferDestroy(readback);
    gfxBufferDestroy(storage);
    gfxBindGroupLayoutDestroy(layout);
    gfxShaderDestroy(computeShader);
}

// Test: A non-zero constant count with no array is rejected
TEST_P(GfxComputePipelineTest, CreateComputePipelineWithNullConstantsArray)
{
    GfxShaderDescriptor shaderDesc = {};
    if (backend == GFX_BACKEND_VULKAN) {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_SPIRV;
        shaderDesc.code = spirvComputeShader;
        shaderDesc.codeSize = sizeof(spirvComputeShader);
    } else {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_WGSL;
        shaderDesc.code = wgslComputeShader;
        shaderDesc.codeSize = strlen(wgslComputeShader) + 1;
    }
    shaderDesc.entryPoint = "main";

    GfxShader computeShader = nullptr;
    ASSERT_EQ(gfxDeviceCreateShader(device, &shaderDesc, &computeShader), GFX_RESULT_SUCCESS);

    GfxComputeState computeState = {};
    computeState.module = computeShader;
    computeState.entryPoint = "main";
    computeState.constants = nullptr;
    computeState.constantCount = 1;

    GfxComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.compute = &computeState;

    GfxComputePipeline pipeline = nullptr;
    EXPECT_EQ(gfxDeviceCreateComputePipeline(device, &pipelineDesc, &pipeline), GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxShaderDestroy(computeShader);
}

// Test: Two entries for the same id would make the applied value order-dependent
TEST_P(GfxComputePipelineTest, CreateComputePipelineWithDuplicateConstantIds)
{
    GfxShaderDescriptor shaderDesc = {};
    if (backend == GFX_BACKEND_VULKAN) {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_SPIRV;
        shaderDesc.code = spirvComputeShader;
        shaderDesc.codeSize = sizeof(spirvComputeShader);
    } else {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_WGSL;
        shaderDesc.code = wgslComputeShader;
        shaderDesc.codeSize = strlen(wgslComputeShader) + 1;
    }
    shaderDesc.entryPoint = "main";

    GfxShader computeShader = nullptr;
    ASSERT_EQ(gfxDeviceCreateShader(device, &shaderDesc, &computeShader), GFX_RESULT_SUCCESS);

    GfxConstantEntry constants[2] = {};
    constants[0].id = 3;
    constants[0].type = GFX_CONSTANT_TYPE_U32;
    constants[0].value.u32 = 1;
    constants[1].id = 3;
    constants[1].type = GFX_CONSTANT_TYPE_U32;
    constants[1].value.u32 = 2;

    GfxComputeState computeState = {};
    computeState.module = computeShader;
    computeState.entryPoint = "main";
    computeState.constants = constants;
    computeState.constantCount = 2;

    GfxComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.compute = &computeState;

    GfxComputePipeline pipeline = nullptr;
    EXPECT_EQ(gfxDeviceCreateComputePipeline(device, &pipelineDesc, &pipeline), GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxShaderDestroy(computeShader);
}

// Test: An out-of-range constant type is rejected
TEST_P(GfxComputePipelineTest, CreateComputePipelineWithInvalidConstantType)
{
    GfxShaderDescriptor shaderDesc = {};
    if (backend == GFX_BACKEND_VULKAN) {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_SPIRV;
        shaderDesc.code = spirvComputeShader;
        shaderDesc.codeSize = sizeof(spirvComputeShader);
    } else {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_WGSL;
        shaderDesc.code = wgslComputeShader;
        shaderDesc.codeSize = strlen(wgslComputeShader) + 1;
    }
    shaderDesc.entryPoint = "main";

    GfxShader computeShader = nullptr;
    ASSERT_EQ(gfxDeviceCreateShader(device, &shaderDesc, &computeShader), GFX_RESULT_SUCCESS);

    GfxConstantEntry constant = {};
    constant.id = 0;
    constant.type = static_cast<GfxConstantType>(0x1234);
    constant.value.u32 = 1;

    GfxComputeState computeState = {};
    computeState.module = computeShader;
    computeState.entryPoint = "main";
    computeState.constants = &constant;
    computeState.constantCount = 1;

    GfxComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.compute = &computeState;

    GfxComputePipeline pipeline = nullptr;
    EXPECT_EQ(gfxDeviceCreateComputePipeline(device, &pipelineDesc, &pipeline), GFX_RESULT_ERROR_INVALID_ARGUMENT);

    gfxShaderDestroy(computeShader);
}

// Test: A zero constant count leaves the shader's defaults in place
TEST_P(GfxComputePipelineTest, CreateComputePipelineWithoutConstants)
{
    GfxShaderDescriptor shaderDesc = {};
    if (backend == GFX_BACKEND_VULKAN) {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_SPIRV;
        shaderDesc.code = spirvConstantsComputeShader;
        shaderDesc.codeSize = sizeof(spirvConstantsComputeShader);
    } else {
        shaderDesc.sourceType = GFX_SHADER_SOURCE_WGSL;
        shaderDesc.code = wgslConstantsComputeShader;
        shaderDesc.codeSize = strlen(wgslConstantsComputeShader) + 1;
    }
    shaderDesc.entryPoint = "main";

    GfxShader computeShader = nullptr;
    ASSERT_EQ(gfxDeviceCreateShader(device, &shaderDesc, &computeShader), GFX_RESULT_SUCCESS);

    GfxBindGroupLayoutEntry layoutEntry = {};
    layoutEntry.binding = 0;
    layoutEntry.visibility = GFX_SHADER_STAGE_COMPUTE;
    layoutEntry.type = GFX_BINDING_TYPE_STORAGE_BUFFER;
    layoutEntry.storageBuffer.access = GFX_STORAGE_BUFFER_ACCESS_READ_WRITE;

    GfxBindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entries = &layoutEntry;
    layoutDesc.entryCount = 1;

    GfxBindGroupLayout layout = nullptr;
    ASSERT_EQ(gfxDeviceCreateBindGroupLayout(device, &layoutDesc, &layout), GFX_RESULT_SUCCESS);

    GfxComputeState computeState = {};
    computeState.module = computeShader;
    computeState.entryPoint = "main";
    computeState.constants = nullptr;
    computeState.constantCount = 0;

    GfxComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.bindGroupLayouts = &layout;
    pipelineDesc.bindGroupLayoutCount = 1;
    pipelineDesc.compute = &computeState;

    GfxComputePipeline pipeline = nullptr;
    EXPECT_EQ(gfxDeviceCreateComputePipeline(device, &pipelineDesc, &pipeline), GFX_RESULT_SUCCESS);
    EXPECT_NE(pipeline, nullptr);

    gfxComputePipelineDestroy(pipeline);
    gfxBindGroupLayoutDestroy(layout);
    gfxShaderDestroy(computeShader);
}

// ===========================================================================
// Test Instantiation
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    GfxComputePipelineTest,
    testing::ValuesIn(getActiveBackends()),
    convertTestParamToString);

} // namespace
