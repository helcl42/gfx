#include "CommonTest.h"

// C API tests compiled with C++ for GoogleTest compatibility

// ===========================================================================
// Parameterized Tests - Run on both Vulkan and WebGPU backends
// ===========================================================================

namespace {

class GfxSurfaceTest : public testing::TestWithParam<GfxBackend> {
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
        adapterDesc.preference = GFX_ADAPTER_PREFERENCE_HIGH_PERFORMANCE;

        if (gfxInstanceRequestAdapter(instance, &adapterDesc, &adapter) != GFX_RESULT_SUCCESS) {
            gfxInstanceDestroy(instance);
            gfxUnloadBackend(backend);
            GTEST_SKIP() << "Failed to get adapter";
        }

        GfxDeviceDescriptor deviceDesc = {};
        deviceDesc.sType = GFX_STRUCTURE_TYPE_DEVICE_DESCRIPTOR;
        deviceDesc.pNext = nullptr;

        if (gfxAdapterCreateDevice(adapter, &deviceDesc, &device) != GFX_RESULT_SUCCESS) {
            gfxInstanceDestroy(instance);
            gfxUnloadBackend(backend);
            GTEST_SKIP() << "Failed to create device";
        }
    }

    void TearDown() override
    {
        if (surface) {
            gfxSurfaceDestroy(surface);
        }
        if (device) {
            gfxDeviceDestroy(device);
        }
        if (instance) {
            gfxInstanceDestroy(instance);
        }
        gfxUnloadBackend(backend);
    }

    GfxBackend backend;
    GfxInstance instance = NULL;
    GfxAdapter adapter = NULL;
    GfxDevice device = NULL;
    GfxSurface surface = NULL;
};

TEST_P(GfxSurfaceTest, CreateSurfaceInvalidArguments)
{
    GfxSurfaceDescriptor desc = {};
    desc.label = "TestSurface";
    desc.windowHandle.windowingSystem = GFX_WINDOWING_SYSTEM_XLIB;
    desc.windowHandle.handle.xlib.display = NULL; // Invalid display
    desc.windowHandle.handle.xlib.window = 0; // Invalid window

    // NULL instance
    GfxResult result = gfxInstanceCreateSurface(NULL, &desc, &surface);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    // NULL descriptor
    result = gfxInstanceCreateSurface(instance, NULL, &surface);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    // NULL output pointer
    result = gfxInstanceCreateSurface(instance, &desc, NULL);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSurfaceTest, DestroyNullSurface)
{
    GfxResult result = gfxSurfaceDestroy(NULL);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSurfaceTest, EnumerateSupportedFormatsInvalidArguments)
{
    // Create dummy pointers (won't be valid, but testing argument validation)
    GfxSurface dummySurface = (GfxSurface)0x1;
    GfxAdapter dummyAdapter = (GfxAdapter)0x1;
    uint32_t formatCount = 0;

    // NULL surface
    GfxResult result = gfxSurfaceEnumerateSupportedFormats(NULL, dummyAdapter, &formatCount, NULL);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    // NULL adapter
    result = gfxSurfaceEnumerateSupportedFormats(dummySurface, NULL, &formatCount, NULL);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    // NULL count pointer
    result = gfxSurfaceEnumerateSupportedFormats(dummySurface, dummyAdapter, NULL, NULL);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSurfaceTest, EnumerateSupportedPresentModesInvalidArguments)
{
    // Create dummy pointers (won't be valid, but testing argument validation)
    GfxSurface dummySurface = (GfxSurface)0x1;
    GfxAdapter dummyAdapter = (GfxAdapter)0x1;
    uint32_t presentModeCount = 0;

    // NULL surface
    GfxResult result = gfxSurfaceEnumerateSupportedPresentModes(NULL, dummyAdapter, &presentModeCount, NULL);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    // NULL adapter
    result = gfxSurfaceEnumerateSupportedPresentModes(dummySurface, NULL, &presentModeCount, NULL);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    // NULL count pointer
    result = gfxSurfaceEnumerateSupportedPresentModes(dummySurface, dummyAdapter, NULL, NULL);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSurfaceTest, GetQueueFamilySurfaceSupportInvalidArguments)
{
    // Create a dummy surface pointer (won't be valid, but testing argument validation)
    GfxSurface dummySurface = (GfxSurface)0x1;
    bool supported = false;

    // NULL adapter
    GfxResult result = gfxAdapterGetQueueFamilySurfaceSupport(NULL, 0, dummySurface, &supported);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    // NULL surface
    result = gfxAdapterGetQueueFamilySurfaceSupport(adapter, 0, NULL, &supported);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    // NULL output pointer
    result = gfxAdapterGetQueueFamilySurfaceSupport(adapter, 0, dummySurface, NULL);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(GfxSurfaceTest, GetSurfaceInfoInvalidArguments)
{
    // Create dummy pointers (won't be valid, but testing argument validation)
    GfxSurface dummySurface = (GfxSurface)0x1;
    GfxAdapter dummyAdapter = (GfxAdapter)0x1;
    GfxSurfaceInfo info = {};

    // NULL surface
    GfxResult result = gfxSurfaceGetInfo(NULL, dummyAdapter, &info);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    // NULL adapter
    result = gfxSurfaceGetInfo(dummySurface, NULL, &info);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);

    // NULL output pointer
    result = gfxSurfaceGetInfo(dummySurface, dummyAdapter, NULL);
    EXPECT_EQ(result, GFX_RESULT_ERROR_INVALID_ARGUMENT);
}

// Note: Creating actual surfaces requires real window handles from X11/Wayland/etc.
// These tests verify API contracts and argument validation without requiring a display server

// ===========================================================================
// Test Instantiation
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    GfxSurfaceTest,
    testing::ValuesIn(getActiveBackends()),
    convertTestParamToString);

} // namespace
