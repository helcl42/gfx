#include "../../common/CommonTest.h"

#include <core/query/QuerySet.h>
#include <core/system/Device.h>

#include <cstring>
#include <vector>

namespace gfx {

class QuerySetImplTest : public ::testing::TestWithParam<GfxBackend> {
protected:
    void SetUp() override
    {
        backend = GetParam();

        ASSERT_EQ(gfxLoadBackend(backend), GFX_RESULT_SUCCESS);

        GfxInstanceDescriptor instanceDesc{
            .backend = backend,
            .applicationName = "QuerySetImplTest"
        };
        ASSERT_EQ(gfxCreateInstance(&instanceDesc, &instance), GFX_RESULT_SUCCESS);

        GfxAdapterDescriptor adapterDesc{
            .sType = GFX_STRUCTURE_TYPE_ADAPTER_DESCRIPTOR,
            .pNext = nullptr,
        };
        ASSERT_EQ(gfxInstanceRequestAdapter(instance, &adapterDesc, &adapter), GFX_RESULT_SUCCESS);

        // Probe for timestamp query extension
        uint32_t extCount = 0;
        gfxAdapterEnumerateExtensions(adapter, &extCount, nullptr);
        std::vector<const char*> adapterExts(extCount);
        gfxAdapterEnumerateExtensions(adapter, &extCount, adapterExts.data());
        std::vector<const char*> deviceExts;
        for (const char* ext : adapterExts) {
            if (std::strcmp(ext, GFX_DEVICE_EXTENSION_TIMESTAMP_QUERY) == 0) {
                timestampQuerySupported = true;
                deviceExts.push_back(GFX_DEVICE_EXTENSION_TIMESTAMP_QUERY);
            }
        }

        GfxDeviceDescriptor deviceDesc{
            .sType = GFX_STRUCTURE_TYPE_DEVICE_DESCRIPTOR,
            .pNext = nullptr,
            .label = nullptr,
            .queueRequests = nullptr,
            .queueRequestCount = 0,
            .enabledExtensions = deviceExts.empty() ? nullptr : deviceExts.data(),
            .enabledExtensionCount = static_cast<uint32_t>(deviceExts.size())
        };
        ASSERT_EQ(gfxAdapterCreateDevice(adapter, &deviceDesc, &device), GFX_RESULT_SUCCESS);
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

    GfxBackend backend;
    GfxInstance instance = nullptr;
    GfxAdapter adapter = nullptr;
    GfxDevice device = nullptr;
    bool timestampQuerySupported = false;
};

TEST_P(QuerySetImplTest, CreateOcclusionQuerySet)
{
    DeviceImpl deviceWrapper(device);

    QuerySetDescriptor desc{
        .type = QueryType::Occlusion,
        .count = 4
    };

    auto querySet = deviceWrapper.createQuerySet(desc);
    EXPECT_NE(querySet, nullptr);
}

TEST_P(QuerySetImplTest, CreateTimestampQuerySet)
{
    if (!timestampQuerySupported) {
        GTEST_SKIP() << "GFX_DEVICE_EXTENSION_TIMESTAMP_QUERY not supported";
    }

    DeviceImpl deviceWrapper(device);

    QuerySetDescriptor desc{
        .type = QueryType::Timestamp,
        .count = 2
    };

    auto querySet = deviceWrapper.createQuerySet(desc);
    EXPECT_NE(querySet, nullptr);
}

TEST_P(QuerySetImplTest, MultipleQuerySets_IndependentHandles)
{
    DeviceImpl deviceWrapper(device);

    QuerySetDescriptor desc{
        .type = QueryType::Occlusion,
        .count = 4
    };

    auto querySet1 = deviceWrapper.createQuerySet(desc);
    auto querySet2 = deviceWrapper.createQuerySet(desc);

    EXPECT_NE(querySet1, nullptr);
    EXPECT_NE(querySet2, nullptr);
    EXPECT_NE(querySet1, querySet2);
}

// ===========================================================================
// Test Instantiation
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    QuerySetImplTest,
    testing::ValuesIn(getActiveBackends()),
    convertTestParamToString);

} // namespace gfx
