#include "../../common/CommonTest.h"

#include <core/system/Instance.h>

namespace gfx {

class InstanceImplTest : public ::testing::TestWithParam<GfxBackend> {
protected:
    void SetUp() override
    {
        backend = GetParam();

        ASSERT_EQ(gfxLoadBackend(backend), GFX_RESULT_SUCCESS);

        const char* instanceExtensions[] = { GFX_INSTANCE_EXTENSION_DEBUG };
        GfxInstanceDescriptor instanceDesc{
            .backend = backend,
            .applicationName = "InstanceImplTest",
            .enabledExtensions = instanceExtensions,
            .enabledExtensionCount = 1
        };
        ASSERT_EQ(gfxCreateInstance(&instanceDesc, &instance), GFX_RESULT_SUCCESS);
    }

    void TearDown() override
    {
        if (instance) {
            gfxInstanceDestroy(instance);
        }
        gfxUnloadBackend(backend);
    }

    GfxBackend backend;
    GfxInstance instance = nullptr;
};

TEST_P(InstanceImplTest, CreateWrapper)
{
    InstanceImpl wrapper(instance);
    // Wrapper created successfully
}

TEST_P(InstanceImplTest, RequestAdapter)
{
    InstanceImpl wrapper(instance);

    AdapterDescriptor desc{};
    auto adapter = wrapper.requestAdapter(desc);

    EXPECT_NE(adapter, nullptr);
}

TEST_P(InstanceImplTest, RequestAdapterWithPreference)
{
    InstanceImpl wrapper(instance);

    AdapterDescriptor desc{
        .preference = AdapterPreference::HighPerformance
    };
    auto adapter = wrapper.requestAdapter(desc);

    EXPECT_NE(adapter, nullptr);
}

TEST_P(InstanceImplTest, EnumerateAdapters)
{
    InstanceImpl wrapper(instance);

    auto adapters = wrapper.enumerateAdapters();

    // Should have at least one adapter
    EXPECT_GT(adapters.size(), 0u);
}

TEST_P(InstanceImplTest, GetNativeHandle)
{
    InstanceImpl wrapper(instance);

    void* handle = wrapper.getNativeHandle();
    EXPECT_NE(handle, nullptr);
}

// ===========================================================================
// Test Instantiation
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    InstanceImplTest,
    testing::ValuesIn(getActiveBackends()),
    convertTestParamToString);

} // namespace gfx
