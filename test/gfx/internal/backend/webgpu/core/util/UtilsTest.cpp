#include <backend/webgpu/core/system/Adapter.h>
#include <backend/webgpu/core/system/Instance.h>
#include <backend/webgpu/core/util/Utils.h>

#include <gtest/gtest.h>

// Test WebGPU core utility functions
// These tests verify the internal utility implementations

namespace {

// ============================================================================
// Test Fixture
// ============================================================================

class WebGPUUtilsTest : public testing::Test {
protected:
    void SetUp() override
    {
        try {
            gfx::backend::webgpu::core::InstanceCreateInfo instInfo{};
            instance = std::make_unique<gfx::backend::webgpu::core::Instance>(instInfo);

            gfx::backend::webgpu::core::AdapterCreateInfo adapterInfo{};
            adapterInfo.adapterIndex = 0;
            adapter = instance->requestAdapter(adapterInfo);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Failed to set up WebGPU: " << e.what();
        }
    }

    std::unique_ptr<gfx::backend::webgpu::core::Instance> instance;
    gfx::backend::webgpu::core::Adapter* adapter = nullptr;
};

// ============================================================================
// Stencil Format Tests
// ============================================================================

TEST_F(WebGPUUtilsTest, HasDepth_Depth24PlusStencil8_ReturnsTrue)
{
    bool result = gfx::backend::webgpu::core::hasDepth(WGPUTextureFormat_Depth24PlusStencil8);
    EXPECT_TRUE(result);
}

TEST_F(WebGPUUtilsTest, HasDepth_Depth32Float_ReturnsTrue)
{
    bool result = gfx::backend::webgpu::core::hasDepth(WGPUTextureFormat_Depth32Float);
    EXPECT_TRUE(result);
}

TEST_F(WebGPUUtilsTest, HasDepth_Stencil8_ReturnsFalse)
{
    bool result = gfx::backend::webgpu::core::hasDepth(WGPUTextureFormat_Stencil8);
    EXPECT_FALSE(result);
}

TEST_F(WebGPUUtilsTest, HasDepth_ColorFormat_ReturnsFalse)
{
    bool result = gfx::backend::webgpu::core::hasDepth(WGPUTextureFormat_RGBA8Unorm);
    EXPECT_FALSE(result);
}

TEST_F(WebGPUUtilsTest, HasStencil_Depth24PlusStencil8_ReturnsTrue)
{
    bool result = gfx::backend::webgpu::core::hasStencil(WGPUTextureFormat_Depth24PlusStencil8);
    EXPECT_TRUE(result);
}

TEST_F(WebGPUUtilsTest, HasStencil_Depth32FloatStencil8_ReturnsTrue)
{
    bool result = gfx::backend::webgpu::core::hasStencil(WGPUTextureFormat_Depth32FloatStencil8);
    EXPECT_TRUE(result);
}

TEST_F(WebGPUUtilsTest, HasStencil_Stencil8_ReturnsTrue)
{
    bool result = gfx::backend::webgpu::core::hasStencil(WGPUTextureFormat_Stencil8);
    EXPECT_TRUE(result);
}

TEST_F(WebGPUUtilsTest, HasStencil_Depth32Float_ReturnsFalse)
{
    bool result = gfx::backend::webgpu::core::hasStencil(WGPUTextureFormat_Depth32Float);
    EXPECT_FALSE(result);
}

TEST_F(WebGPUUtilsTest, HasStencil_Depth16Unorm_ReturnsFalse)
{
    bool result = gfx::backend::webgpu::core::hasStencil(WGPUTextureFormat_Depth16Unorm);
    EXPECT_FALSE(result);
}

TEST_F(WebGPUUtilsTest, HasStencil_Depth24Plus_ReturnsFalse)
{
    bool result = gfx::backend::webgpu::core::hasStencil(WGPUTextureFormat_Depth24Plus);
    EXPECT_FALSE(result);
}

TEST_F(WebGPUUtilsTest, HasStencil_ColorFormat_ReturnsFalse)
{
    bool result = gfx::backend::webgpu::core::hasStencil(WGPUTextureFormat_RGBA8Unorm);
    EXPECT_FALSE(result);
}

TEST_F(WebGPUUtilsTest, HasStencil_UndefinedFormat_ReturnsFalse)
{
    bool result = gfx::backend::webgpu::core::hasStencil(WGPUTextureFormat_Undefined);
    EXPECT_FALSE(result);
}

// ============================================================================
// Format Bytes Per Pixel Tests
// ============================================================================

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_R8Formats_Returns1)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_R8Unorm), 1u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_R8Snorm), 1u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_R8Uint), 1u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_R8Sint), 1u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_R16Formats_Returns2)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_R16Uint), 2u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_R16Sint), 2u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_R16Float), 2u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_RG8Formats_Returns2)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RG8Unorm), 2u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RG8Snorm), 2u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RG8Uint), 2u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RG8Sint), 2u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_R32Formats_Returns4)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_R32Float), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_R32Uint), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_R32Sint), 4u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_RGBA8Formats_Returns4)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGBA8Unorm), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGBA8UnormSrgb), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGBA8Snorm), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGBA8Uint), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGBA8Sint), 4u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_BGRA8Formats_Returns4)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BGRA8Unorm), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BGRA8UnormSrgb), 4u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_SpecialRGBFormats_Returns4)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGB10A2Unorm), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RG11B10Ufloat), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGB9E5Ufloat), 4u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_RG32Formats_Returns8)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RG32Float), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RG32Uint), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RG32Sint), 8u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_RGBA16Formats_Returns8)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGBA16Uint), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGBA16Sint), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGBA16Float), 8u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_RGBA32Formats_Returns16)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGBA32Float), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGBA32Uint), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_RGBA32Sint), 16u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_DepthFormats_ReturnsCorrectSize)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_Depth16Unorm), 2u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_Depth24Plus), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_Depth24PlusStencil8), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_Depth32Float), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_Depth32FloatStencil8), 8u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_Stencil8_Returns1)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_Stencil8), 1u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_BC1Formats_Returns8)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC1RGBAUnorm), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC1RGBAUnormSrgb), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC4RUnorm), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC4RSnorm), 8u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_BC2BC3Formats_Returns16)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC2RGBAUnorm), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC2RGBAUnormSrgb), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC3RGBAUnorm), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC3RGBAUnormSrgb), 16u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_BC5BC6BC7Formats_Returns16)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC5RGUnorm), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC5RGSnorm), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC6HRGBUfloat), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC6HRGBFloat), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC7RGBAUnorm), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_BC7RGBAUnormSrgb), 16u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_ETC2Formats_ReturnsCorrectSize)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ETC2RGB8Unorm), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ETC2RGB8UnormSrgb), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ETC2RGB8A1Unorm), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ETC2RGB8A1UnormSrgb), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ETC2RGBA8Unorm), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ETC2RGBA8UnormSrgb), 16u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_EACFormats_ReturnsCorrectSize)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_EACR11Unorm), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_EACR11Snorm), 8u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_EACRG11Unorm), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_EACRG11Snorm), 16u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_ASTCFormats_Returns16)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ASTC4x4Unorm), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ASTC4x4UnormSrgb), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ASTC8x8Unorm), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ASTC8x8UnormSrgb), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ASTC12x12Unorm), 16u);
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_ASTC12x12UnormSrgb), 16u);
}

TEST_F(WebGPUUtilsTest, GetFormatBytesPerPixel_UndefinedFormat_Returns0)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getFormatBytesPerPixel(WGPUTextureFormat_Undefined), 0u);
}

// ============================================================================
// Packed Bytes Per Row Tests
// ============================================================================

TEST_F(WebGPUUtilsTest, GetPackedBytesPerRow_UncompressedFormats_IsWidthTimesTexelSize)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getPackedBytesPerRow(WGPUTextureFormat_RGBA8Unorm, WGPUTextureAspect_All, 1), 4u);
    EXPECT_EQ(gfx::backend::webgpu::core::getPackedBytesPerRow(WGPUTextureFormat_RGBA8Unorm, WGPUTextureAspect_All, 65), 260u);
    EXPECT_EQ(gfx::backend::webgpu::core::getPackedBytesPerRow(WGPUTextureFormat_R8Unorm, WGPUTextureAspect_All, 257), 257u);
    EXPECT_EQ(gfx::backend::webgpu::core::getPackedBytesPerRow(WGPUTextureFormat_RGBA32Float, WGPUTextureAspect_All, 17), 272u);
}

TEST_F(WebGPUUtilsTest, GetPackedBytesPerRow_CompressedFormat_CountsBlocks)
{
    // BC1 uses 4x4 blocks of 8 bytes, so 10 texels need 3 blocks
    EXPECT_EQ(gfx::backend::webgpu::core::getPackedBytesPerRow(WGPUTextureFormat_BC1RGBAUnorm, WGPUTextureAspect_All, 10), 24u);
}

TEST_F(WebGPUUtilsTest, GetPackedBytesPerRow_ZeroWidth_Returns0)
{
    EXPECT_EQ(gfx::backend::webgpu::core::getPackedBytesPerRow(WGPUTextureFormat_RGBA8Unorm, WGPUTextureAspect_All, 0), 0u);
}

// ============================================================================
// String View Tests
// ============================================================================

TEST_F(WebGPUUtilsTest, ToStringView_ValidString_ReturnsCorrectView)
{
    const char* str = "test";
    WGPUStringView view = gfx::backend::webgpu::core::toStringView(str);

    EXPECT_EQ(view.data, str);
    EXPECT_EQ(view.length, WGPU_STRLEN);
}

TEST_F(WebGPUUtilsTest, ToStringView_NullString_ReturnsNullView)
{
    WGPUStringView view = gfx::backend::webgpu::core::toStringView(nullptr);

    EXPECT_EQ(view.data, nullptr);
    EXPECT_EQ(view.length, WGPU_STRLEN);
}

TEST_F(WebGPUUtilsTest, ToStringView_EmptyString_ReturnsValidView)
{
    const char* str = "";
    WGPUStringView view = gfx::backend::webgpu::core::toStringView(str);

    EXPECT_EQ(view.data, str);
    EXPECT_EQ(view.length, WGPU_STRLEN);
}

} // anonymous namespace
