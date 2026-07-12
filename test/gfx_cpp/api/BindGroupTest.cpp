#include "CommonTest.h"

// ===========================================================================
// Parameterized Tests - Run on both Vulkan and WebGPU backends
// ===========================================================================

namespace {

class GfxCppBindGroupTest : public testing::TestWithParam<gfx::Backend> {
protected:
    void SetUp() override
    {
        backend = GetParam();

        try {
            gfx::InstanceDescriptor instDesc{
                .backend = backend,
                .enabledExtensions = { gfx::INSTANCE_EXTENSION_DEBUG }
            };
            instance = gfx::createInstance(instDesc);

            gfx::AdapterDescriptor adapterDesc{
            };
            adapter = instance->requestAdapter(adapterDesc);

            gfx::DeviceDescriptor deviceDesc{
                .label = "Test Device"
            };
            device = adapter->createDevice(deviceDesc);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Failed to set up: " << e.what();
        }
    }

    gfx::Backend backend;
    std::shared_ptr<gfx::Instance> instance;
    std::shared_ptr<gfx::Adapter> adapter;
    std::shared_ptr<gfx::Device> device;
};

// ===========================================================================
// BindGroup Tests
// ===========================================================================

TEST_P(GfxCppBindGroupTest, CreateBindGroupWithUniformBuffer)
{
    ASSERT_NE(device, nullptr);

    // Create bind group layout
    gfx::BindGroupLayoutEntry layoutEntry{
        .binding = 0,
        .visibility = gfx::ShaderStage::Vertex | gfx::ShaderStage::Fragment,
        .resource = gfx::BindGroupLayoutEntry::UniformBufferBinding{
            .hasDynamicOffset = false,
            .minBindingSize = 0 }
    };

    gfx::BindGroupLayoutDescriptor layoutDesc{
        .entries = { layoutEntry }
    };

    auto layout = device->createBindGroupLayout(layoutDesc);
    ASSERT_NE(layout, nullptr);

    // Create uniform buffer
    gfx::BufferDescriptor bufferDesc{
        .size = 256,
        .usage = gfx::BufferUsage::Uniform,
        .memoryProperties = gfx::MemoryProperty::HostVisible | gfx::MemoryProperty::HostCoherent
    };

    auto buffer = device->createBuffer(bufferDesc);
    ASSERT_NE(buffer, nullptr);

    // Create bind group
    gfx::BindGroupEntry entry{
        .binding = 0,
        .resource = buffer,
        .offset = 0,
        .size = 256
    };

    gfx::BindGroupDescriptor bindGroupDesc{
        .layout = layout,
        .entries = { entry }
    };

    auto bindGroup = device->createBindGroup(bindGroupDesc);
    EXPECT_NE(bindGroup, nullptr);
}

TEST_P(GfxCppBindGroupTest, CreateBindGroupWithSampler)
{
    ASSERT_NE(device, nullptr);

    // Create bind group layout
    gfx::BindGroupLayoutEntry layoutEntry{
        .binding = 0,
        .visibility = gfx::ShaderStage::Fragment,
        .resource = gfx::BindGroupLayoutEntry::SamplerBinding{
            .type = gfx::SamplerBindingType::Filtering }
    };

    gfx::BindGroupLayoutDescriptor layoutDesc{
        .entries = { layoutEntry }
    };

    auto layout = device->createBindGroupLayout(layoutDesc);
    ASSERT_NE(layout, nullptr);

    // Create sampler
    gfx::SamplerDescriptor samplerDesc{
        .addressModeU = gfx::AddressMode::Repeat,
        .addressModeV = gfx::AddressMode::Repeat,
        .addressModeW = gfx::AddressMode::Repeat,
        .magFilter = gfx::FilterMode::Linear,
        .minFilter = gfx::FilterMode::Linear,
        .mipmapFilter = gfx::FilterMode::Linear
    };

    auto sampler = device->createSampler(samplerDesc);
    ASSERT_NE(sampler, nullptr);

    // Create bind group
    gfx::BindGroupEntry entry{
        .binding = 0,
        .resource = sampler
    };

    gfx::BindGroupDescriptor bindGroupDesc{
        .layout = layout,
        .entries = { entry }
    };

    auto bindGroup = device->createBindGroup(bindGroupDesc);
    EXPECT_NE(bindGroup, nullptr);
}

TEST_P(GfxCppBindGroupTest, CreateBindGroupWithTextureView)
{
    ASSERT_NE(device, nullptr);

    // Create bind group layout
    gfx::BindGroupLayoutEntry layoutEntry{
        .binding = 0,
        .visibility = gfx::ShaderStage::Fragment,
        .resource = gfx::BindGroupLayoutEntry::TextureBinding{
            .multisampled = false,
            .viewDimension = gfx::TextureViewType::View2D }
    };

    gfx::BindGroupLayoutDescriptor layoutDesc{
        .entries = { layoutEntry }
    };

    auto layout = device->createBindGroupLayout(layoutDesc);
    ASSERT_NE(layout, nullptr);

    // Create texture
    gfx::TextureDescriptor textureDesc{
        .type = gfx::TextureType::Texture2D,
        .size = { 256, 256, 1 },
        .arrayLayerCount = 1,
        .mipLevelCount = 1,
        .sampleCount = gfx::SampleCount::Count1,
        .format = gfx::Format::R8G8B8A8Unorm,
        .usage = gfx::TextureUsage::TextureBinding
    };

    auto texture = device->createTexture(textureDesc);
    ASSERT_NE(texture, nullptr);

    // Create texture view
    gfx::TextureViewDescriptor viewDesc{
        .viewType = gfx::TextureViewType::View2D,
        .format = gfx::Format::R8G8B8A8Unorm,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };

    auto textureView = texture->createView(viewDesc);
    ASSERT_NE(textureView, nullptr);

    // Create bind group
    gfx::BindGroupEntry entry{
        .binding = 0,
        .resource = textureView
    };

    gfx::BindGroupDescriptor bindGroupDesc{
        .layout = layout,
        .entries = { entry }
    };

    auto bindGroup = device->createBindGroup(bindGroupDesc);
    EXPECT_NE(bindGroup, nullptr);
}

TEST_P(GfxCppBindGroupTest, CreateBindGroupWithTextureArrayBinding)
{
    ASSERT_NE(device, nullptr);

    // Layout with a texture array binding of 2 elements
    gfx::BindGroupLayoutEntry layoutEntry{
        .binding = 0,
        .visibility = gfx::ShaderStage::Fragment,
        .count = 2,
        .resource = gfx::BindGroupLayoutEntry::TextureBinding{
            .multisampled = false,
            .viewDimension = gfx::TextureViewType::View2D }
    };

    gfx::BindGroupLayoutDescriptor layoutDesc{
        .entries = { layoutEntry }
    };

    std::shared_ptr<gfx::BindGroupLayout> layout;
    try {
        layout = device->createBindGroupLayout(layoutDesc);
    } catch (const std::exception&) {
        layout = nullptr;
    }
    if (!layout) {
        // Binding arrays may be unsupported by the backend (e.g. WebGPU without the feature)
        GTEST_SKIP() << "Binding arrays not supported by this backend";
    }

    gfx::TextureDescriptor textureDesc{
        .type = gfx::TextureType::Texture2D,
        .size = { 64, 64, 1 },
        .arrayLayerCount = 1,
        .mipLevelCount = 1,
        .sampleCount = gfx::SampleCount::Count1,
        .format = gfx::Format::R8G8B8A8Unorm,
        .usage = gfx::TextureUsage::TextureBinding
    };

    gfx::TextureViewDescriptor viewDesc{
        .viewType = gfx::TextureViewType::View2D,
        .format = gfx::Format::R8G8B8A8Unorm,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };

    auto texture0 = device->createTexture(textureDesc);
    auto texture1 = device->createTexture(textureDesc);
    ASSERT_NE(texture0, nullptr);
    ASSERT_NE(texture1, nullptr);

    auto view0 = texture0->createView(viewDesc);
    auto view1 = texture1->createView(viewDesc);
    ASSERT_NE(view0, nullptr);
    ASSERT_NE(view1, nullptr);

    // One entry per array element, same binding
    gfx::BindGroupDescriptor bindGroupDesc{
        .layout = layout,
        .entries = {
            gfx::BindGroupEntry{ .binding = 0, .arrayElement = 0, .resource = view0 },
            gfx::BindGroupEntry{ .binding = 0, .arrayElement = 1, .resource = view1 } }
    };

    auto bindGroup = device->createBindGroup(bindGroupDesc);
    EXPECT_NE(bindGroup, nullptr);
}

TEST_P(GfxCppBindGroupTest, CreateBindGroupWithStorageBuffer)
{
    ASSERT_NE(device, nullptr);

    // Create bind group layout
    gfx::BindGroupLayoutEntry layoutEntry{
        .binding = 0,
        .visibility = gfx::ShaderStage::Compute,
        .resource = gfx::BindGroupLayoutEntry::UniformBufferBinding{
            .hasDynamicOffset = false,
            .minBindingSize = 1024 }
    };

    gfx::BindGroupLayoutDescriptor layoutDesc{
        .entries = { layoutEntry }
    };

    auto layout = device->createBindGroupLayout(layoutDesc);
    ASSERT_NE(layout, nullptr);

    // Create storage buffer
    gfx::BufferDescriptor bufferDesc{
        .size = 1024,
        .usage = gfx::BufferUsage::Storage,
        .memoryProperties = gfx::MemoryProperty::HostVisible | gfx::MemoryProperty::HostCoherent
    };

    auto buffer = device->createBuffer(bufferDesc);
    ASSERT_NE(buffer, nullptr);

    // Create bind group
    gfx::BindGroupEntry entry{
        .binding = 0,
        .resource = buffer,
        .offset = 0,
        .size = 256
    };

    gfx::BindGroupDescriptor bindGroupDesc{
        .layout = layout,
        .entries = { entry }
    };

    auto bindGroup = device->createBindGroup(bindGroupDesc);
    EXPECT_NE(bindGroup, nullptr);
}

TEST_P(GfxCppBindGroupTest, CreateBindGroupWithMultipleEntries)
{
    ASSERT_NE(device, nullptr);

    // Create bind group layout
    std::vector<gfx::BindGroupLayoutEntry> layoutEntries;

    // Binding 0: Uniform buffer
    layoutEntries.push_back(gfx::BindGroupLayoutEntry{
        .binding = 0,
        .visibility = gfx::ShaderStage::Vertex,
        .resource = gfx::BindGroupLayoutEntry::UniformBufferBinding{
            .hasDynamicOffset = false,
            .minBindingSize = 0 } });

    // Binding 1: Sampler
    layoutEntries.push_back(gfx::BindGroupLayoutEntry{
        .binding = 1,
        .visibility = gfx::ShaderStage::Fragment,
        .resource = gfx::BindGroupLayoutEntry::SamplerBinding{
            .type = gfx::SamplerBindingType::Filtering } });

    // Binding 2: Texture
    layoutEntries.push_back(gfx::BindGroupLayoutEntry{
        .binding = 2,
        .visibility = gfx::ShaderStage::Fragment,
        .resource = gfx::BindGroupLayoutEntry::TextureBinding{
            .multisampled = false,
            .viewDimension = gfx::TextureViewType::View2D } });

    gfx::BindGroupLayoutDescriptor layoutDesc{
        .entries = layoutEntries
    };

    auto layout = device->createBindGroupLayout(layoutDesc);
    ASSERT_NE(layout, nullptr);

    // Create resources
    gfx::BufferDescriptor bufferDesc{
        .size = 256,
        .usage = gfx::BufferUsage::Uniform,
        .memoryProperties = gfx::MemoryProperty::HostVisible | gfx::MemoryProperty::HostCoherent
    };
    auto buffer = device->createBuffer(bufferDesc);
    ASSERT_NE(buffer, nullptr);

    gfx::SamplerDescriptor samplerDesc{
        .addressModeU = gfx::AddressMode::Repeat,
        .addressModeV = gfx::AddressMode::Repeat,
        .addressModeW = gfx::AddressMode::Repeat,
        .magFilter = gfx::FilterMode::Linear,
        .minFilter = gfx::FilterMode::Linear,
        .mipmapFilter = gfx::FilterMode::Linear
    };
    auto sampler = device->createSampler(samplerDesc);
    ASSERT_NE(sampler, nullptr);

    gfx::TextureDescriptor textureDesc{
        .type = gfx::TextureType::Texture2D,
        .size = { 256, 256, 1 },
        .arrayLayerCount = 1,
        .mipLevelCount = 1,
        .sampleCount = gfx::SampleCount::Count1,
        .format = gfx::Format::R8G8B8A8Unorm,
        .usage = gfx::TextureUsage::TextureBinding
    };
    auto texture = device->createTexture(textureDesc);
    ASSERT_NE(texture, nullptr);

    gfx::TextureViewDescriptor viewDesc{
        .viewType = gfx::TextureViewType::View2D,
        .format = gfx::Format::R8G8B8A8Unorm,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };
    auto textureView = texture->createView(viewDesc);
    ASSERT_NE(textureView, nullptr);

    // Create bind group entries
    std::vector<gfx::BindGroupEntry> entries;

    entries.push_back(gfx::BindGroupEntry{
        .binding = 0,
        .resource = buffer,
        .offset = 0,
        .size = 256 });

    entries.push_back(gfx::BindGroupEntry{
        .binding = 1,
        .resource = sampler });

    entries.push_back(gfx::BindGroupEntry{
        .binding = 2,
        .resource = textureView });

    gfx::BindGroupDescriptor bindGroupDesc{
        .layout = layout,
        .entries = entries
    };

    auto bindGroup = device->createBindGroup(bindGroupDesc);
    EXPECT_NE(bindGroup, nullptr);
}

TEST_P(GfxCppBindGroupTest, CreateBindGroupWithBufferOffset)
{
    ASSERT_NE(device, nullptr);

    // Create bind group layout
    gfx::BindGroupLayoutEntry layoutEntry{
        .binding = 0,
        .visibility = gfx::ShaderStage::Vertex,
        .resource = gfx::BindGroupLayoutEntry::UniformBufferBinding{
            .hasDynamicOffset = false,
            .minBindingSize = 256 }
    };

    gfx::BindGroupLayoutDescriptor layoutDesc{
        .entries = { layoutEntry }
    };

    auto layout = device->createBindGroupLayout(layoutDesc);
    ASSERT_NE(layout, nullptr);

    // Create buffer
    gfx::BufferDescriptor bufferDesc{
        .size = 512,
        .usage = gfx::BufferUsage::Uniform,
        .memoryProperties = gfx::MemoryProperty::HostVisible | gfx::MemoryProperty::HostCoherent
    };

    auto buffer = device->createBuffer(bufferDesc);
    ASSERT_NE(buffer, nullptr);

    // Create bind group with offset
    gfx::BindGroupEntry entry{
        .binding = 0,
        .resource = buffer,
        .offset = 256,
        .size = 256
    };

    gfx::BindGroupDescriptor bindGroupDesc{
        .layout = layout,
        .entries = { entry }
    };

    auto bindGroup = device->createBindGroup(bindGroupDesc);
    EXPECT_NE(bindGroup, nullptr);
}

TEST_P(GfxCppBindGroupTest, CreateMultipleBindGroupsWithSameLayout)
{
    ASSERT_NE(device, nullptr);

    // Create bind group layout
    gfx::BindGroupLayoutEntry layoutEntry{
        .binding = 0,
        .visibility = gfx::ShaderStage::Vertex,
        .resource = gfx::BindGroupLayoutEntry::UniformBufferBinding{
            .hasDynamicOffset = false,
            .minBindingSize = 0 }
    };

    gfx::BindGroupLayoutDescriptor layoutDesc{
        .entries = { layoutEntry }
    };

    auto layout = device->createBindGroupLayout(layoutDesc);
    ASSERT_NE(layout, nullptr);

    // Create two buffers
    gfx::BufferDescriptor bufferDesc{
        .size = 256,
        .usage = gfx::BufferUsage::Uniform,
        .memoryProperties = gfx::MemoryProperty::HostVisible | gfx::MemoryProperty::HostCoherent
    };

    auto buffer1 = device->createBuffer(bufferDesc);
    ASSERT_NE(buffer1, nullptr);

    auto buffer2 = device->createBuffer(bufferDesc);
    ASSERT_NE(buffer2, nullptr);

    // Create first bind group
    gfx::BindGroupEntry entry1{
        .binding = 0,
        .resource = buffer1,
        .offset = 0,
        .size = 256
    };

    gfx::BindGroupDescriptor bindGroupDesc1{
        .layout = layout,
        .entries = { entry1 }
    };

    auto bindGroup1 = device->createBindGroup(bindGroupDesc1);
    EXPECT_NE(bindGroup1, nullptr);

    // Create second bind group
    gfx::BindGroupEntry entry2{
        .binding = 0,
        .resource = buffer2,
        .offset = 0,
        .size = 256
    };

    gfx::BindGroupDescriptor bindGroupDesc2{
        .layout = layout,
        .entries = { entry2 }
    };

    auto bindGroup2 = device->createBindGroup(bindGroupDesc2);
    EXPECT_NE(bindGroup2, nullptr);
}

// ===========================================================================
// Test Instantiation
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    GfxCppBindGroupTest,
    testing::ValuesIn(getActiveBackends()),
    convertTestParamToString);

} // namespace
