#ifndef GFX_VULKAN_CORE_TYPES_H
#define GFX_VULKAN_CORE_TYPES_H

#include "../common/Common.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gfx::backend::vulkan::core {

// Forward declarations for SubmitInfo
class CommandEncoder;
class Fence;
class Semaphore;
struct RenderPassEncoderCreateInfo;

// Forward declarations for barriers
class Buffer;
class Texture;

// ============================================================================
// Internal Extension Names
// ============================================================================

namespace extensions {
    constexpr const char* SURFACE = "gfx_surface";
    constexpr const char* DEBUG = "gfx_debug";
    constexpr const char* SWAPCHAIN = "gfx_swapchain";
    constexpr const char* TIMELINE_SEMAPHORE = "gfx_timeline_semaphore";
    constexpr const char* MULTIVIEW = "gfx_multiview";
    constexpr const char* ANISOTROPIC_FILTERING = "gfx_anisotropic_filtering";
    constexpr const char* OCCLUSION_QUERY_PRECISE = "gfx_occlusion_query_precise";
    constexpr const char* NON_SOLID_FILL = "gfx_non_solid_fill";
    constexpr const char* TIMESTAMP_QUERY = "gfx_timestamp_query";
    constexpr const char* TEXTURE_COMPRESSION_BC = "gfx_texture_compression_bc";
    constexpr const char* TEXTURE_COMPRESSION_ETC2 = "gfx_texture_compression_etc2";
    constexpr const char* TEXTURE_COMPRESSION_ASTC = "gfx_texture_compression_astc";
} // namespace extensions

// Bitmask of the known GFX device extensions, parsed once at device creation
// so enablement checks are O(1) instead of string comparisons
enum class DeviceExtension : uint64_t {
    Swapchain = 1ull << 0,
    TimelineSemaphore = 1ull << 1,
    Multiview = 1ull << 2,
    AnisotropicFiltering = 1ull << 3,
    OcclusionQueryPrecise = 1ull << 4,
    NonSolidFill = 1ull << 5,
    TimestampQuery = 1ull << 6,
    TextureCompressionBC = 1ull << 7,
    TextureCompressionETC2 = 1ull << 8,
    TextureCompressionASTC = 1ull << 9,
};

// ============================================================================
// Internal Type Definitions
// ============================================================================

enum class SemaphoreType {
    Binary,
    Timeline
};

enum class ShaderSourceType {
    WGSL = 0,
    SPIRV = 1
};

enum class DeviceTypePreference {
    HighPerformance, // Prefer discrete GPU
    LowPower, // Prefer integrated GPU
    SoftwareRenderer // Force CPU-based software renderer
};

// ============================================================================
// Internal CreateInfo structs - pure Vulkan types, no GFX dependencies
// ============================================================================
struct BufferCreateInfo {
    size_t size;
    VkBufferUsageFlags usage;
    uint32_t originalUsage; // Original usage flags as uint32_t (underlying type, not GfxBufferUsageFlags)
    VkMemoryPropertyFlags memoryProperties;
};

struct BufferImportInfo {
    size_t size;
    VkBufferUsageFlags usage;
    uint32_t originalUsage; // Original usage flags as uint32_t (underlying type, not GfxBufferUsageFlags)
    VkMemoryPropertyFlags memoryProperties;
};

struct BufferInfo {
    uint64_t size;
    VkBufferUsageFlags usage;
    uint32_t originalUsage; // Original usage flags as uint32_t (underlying type, not GfxBufferUsageFlags)
    VkMemoryPropertyFlags memoryProperties;
};

struct TextureCreateInfo {
    VkFormat format;
    VkExtent3D size;
    VkImageUsageFlags usage;
    VkSampleCountFlagBits sampleCount;
    uint32_t mipLevelCount;
    VkImageType imageType;
    uint32_t arrayLayers;
    VkImageCreateFlags flags; // For cube maps, etc.
};

struct TextureImportInfo {
    VkFormat format;
    VkExtent3D size;
    VkImageUsageFlags usage;
    VkSampleCountFlagBits sampleCount;
    uint32_t mipLevelCount;
    VkImageType imageType;
    uint32_t arrayLayers;
    VkImageCreateFlags flags; // For cube maps, etc.
};

struct TextureInfo {
    VkImageType imageType;
    VkExtent3D size;
    uint32_t arrayLayers;
    VkFormat format;
    uint32_t mipLevelCount;
    VkSampleCountFlagBits sampleCount;
    VkImageUsageFlags usage;
};

struct SwapchainInfo {
    uint32_t width;
    uint32_t height;
    VkFormat format;
    uint32_t imageCount;
    VkPresentModeKHR presentMode;
};

struct QueueInfo {
    uint32_t queueFamilyIndex;
    uint32_t queueIndex;
};

struct TextureViewCreateInfo {
    VkImageViewType viewType;
    VkFormat format; // VK_FORMAT_UNDEFINED means use texture's format
    uint32_t baseMipLevel;
    uint32_t mipLevelCount;
    uint32_t baseArrayLayer;
    uint32_t arrayLayerCount;
};

struct ShaderCreateInfo {
    const void* code;
    size_t codeSize;
    const char* entryPoint; // nullptr means "main"
};

struct SemaphoreCreateInfo {
    SemaphoreType type;
    uint64_t initialValue;
};

struct FenceCreateInfo {
    bool signaled; // true = create in signaled state
};

struct QuerySetCreateInfo {
    const char* label = nullptr;
    VkQueryType type = VK_QUERY_TYPE_OCCLUSION;
    uint32_t count = 0;
    bool precise = false; // Only meaningful when type == VK_QUERY_TYPE_OCCLUSION
};

struct MemoryBarrier {
    VkPipelineStageFlags srcStageMask;
    VkPipelineStageFlags dstStageMask;
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;
};

struct BufferBarrier {
    class Buffer* buffer;
    VkPipelineStageFlags srcStageMask;
    VkPipelineStageFlags dstStageMask;
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;
    uint64_t offset;
    uint64_t size; // 0 means whole buffer
};

struct TextureBarrier {
    class Texture* texture;
    VkPipelineStageFlags srcStageMask;
    VkPipelineStageFlags dstStageMask;
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;
    VkImageLayout oldLayout;
    VkImageLayout newLayout;
    uint32_t baseMipLevel;
    uint32_t mipLevelCount;
    uint32_t baseArrayLayer;
    uint32_t arrayLayerCount;
};

struct Viewport {
    float x;
    float y;
    float width;
    float height;
    float minDepth;
    float maxDepth;
};

struct ScissorRect {
    int32_t x;
    int32_t y;
    uint32_t width;
    uint32_t height;
};

struct SamplerCreateInfo {
    VkSamplerAddressMode addressModeU;
    VkSamplerAddressMode addressModeV;
    VkSamplerAddressMode addressModeW;
    VkFilter magFilter;
    VkFilter minFilter;
    VkSamplerMipmapMode mipmapMode;
    float lodMinClamp;
    float lodMaxClamp;
    uint32_t maxAnisotropy;
    VkCompareOp compareOp; // VK_COMPARE_OP_MAX_ENUM means no compare
};

// Forward declaration for complex types
struct BindGroupLayoutEntry {
    uint32_t binding;
    VkDescriptorType descriptorType;
    VkShaderStageFlags stageFlags;
    uint32_t descriptorCount = 1; // Number of descriptors in the array
};

struct BindGroupLayoutCreateInfo {
    std::vector<BindGroupLayoutEntry> entries;
};

struct BindGroupEntry {
    uint32_t binding;
    uint32_t arrayElement = 0; // Index within the binding array (dstArrayElement)
    VkDescriptorType descriptorType;
    // Union-like storage for different resource types
    VkBuffer buffer;
    VkDeviceSize bufferOffset;
    VkDeviceSize bufferSize;
    VkSampler sampler;
    VkImageView imageView;
    VkImageLayout imageLayout;
};

struct BindGroupCreateInfo {
    VkDescriptorSetLayout layout; // From BindGroupLayout
    std::vector<BindGroupEntry> entries;
};

struct InstanceCreateInfo {
    const char* applicationName = "Gfx Application";
    uint32_t applicationVersion = 1;
    std::vector<std::string> enabledExtensions;
    const void* pNext = nullptr;
};

struct AdapterCreateInfo {
    uint32_t adapterIndex = UINT32_MAX; // Adapter index (UINT32_MAX = use preference)
    DeviceTypePreference devicePreference = DeviceTypePreference::HighPerformance; // Only used when adapterIndex is UINT32_MAX
};

struct DeviceCreateInfo {
    struct QueueRequest {
        uint32_t queueFamilyIndex;
        uint32_t queueIndex;
        float priority;
    };

    std::vector<std::string> enabledExtensions;
    std::vector<QueueRequest> queueRequests;
    const void* pNext = nullptr;
};

struct PlatformWindowHandle {
    // Platform-specific window handles (Vulkan native)
    enum class Platform {
        Unknown,
        Xlib,
        Xcb,
        Wayland,
        Win32,
        Metal,
        Android,
        Emscripten
    } platform;

    union {
        struct {
            void* display; // Display*
            unsigned long window; // Window
        } xlib;
        struct {
            void* connection; // xcb_connection_t*
            uint32_t window; // xcb_window_t
        } xcb;
        struct {
            void* display; // wl_display*
            void* surface; // wl_surface*
        } wayland;
        struct {
            void* hinstance; // HINSTANCE
            void* hwnd; // HWND
        } win32;
        struct {
            void* layer; // CAMetalLayer*
        } metal;
        struct {
            void* window; // ANativeWindow*
        } android;
        struct {
            const char* canvasSelector; // CSS selector for canvas element (e.g., "#canvas")
        } emscripten;
    } handle;
};

struct SurfaceCreateInfo {
    PlatformWindowHandle windowHandle;
};

struct SwapchainCreateInfo {
    uint32_t width;
    uint32_t height;
    VkFormat format;
    VkColorSpaceKHR colorSpace;
    VkPresentModeKHR presentMode;
    uint32_t imageCount;
};

// Pipeline CreateInfo structs - these are complex
struct VertexBufferLayout {
    uint64_t arrayStride;
    VkVertexInputRate inputRate;
    std::vector<VkVertexInputAttributeDescription> attributes;
};

struct VertexState {
    VkShaderModule module;
    const char* entryPoint;
    std::vector<VertexBufferLayout> buffers;
};

struct ColorTargetState {
    VkFormat format;
    VkColorComponentFlags writeMask;
    VkPipelineColorBlendAttachmentState blendState;
};

struct FragmentState {
    VkShaderModule module;
    const char* entryPoint;
    std::vector<ColorTargetState> targets;
};

struct PrimitiveState {
    VkPrimitiveTopology topology;
    VkPolygonMode polygonMode;
    VkCullModeFlags cullMode;
    VkFrontFace frontFace;
};

struct DepthStencilState {
    VkFormat format;
    bool depthWriteEnabled;
    VkCompareOp depthCompareOp;
};

struct RenderPipelineCreateInfo {
    VkRenderPass renderPass = VK_NULL_HANDLE; // Render pass this pipeline will be used with
    std::vector<VkDescriptorSetLayout> bindGroupLayouts;
    VertexState vertex;
    FragmentState fragment;
    PrimitiveState primitive;
    std::optional<DepthStencilState> depthStencil;
    VkSampleCountFlagBits sampleCount;
};

struct ComputePipelineCreateInfo {
    std::vector<VkDescriptorSetLayout> bindGroupLayouts;
    VkShaderModule module;
    const char* entryPoint;
};

// Color attachment target for render pass (main or resolve)
struct RenderPassColorAttachmentTarget {
    VkFormat format;
    VkSampleCountFlagBits sampleCount;
    VkAttachmentLoadOp loadOp;
    VkAttachmentStoreOp storeOp;
    VkImageLayout finalLayout;
};

// Color attachment with optional resolve target
struct RenderPassColorAttachment {
    RenderPassColorAttachmentTarget target;
    std::optional<RenderPassColorAttachmentTarget> resolveTarget;
};

// Depth/stencil attachment target for render pass (main or resolve)
struct RenderPassDepthStencilAttachmentTarget {
    VkFormat format;
    VkSampleCountFlagBits sampleCount;
    VkAttachmentLoadOp depthLoadOp;
    VkAttachmentStoreOp depthStoreOp;
    VkAttachmentLoadOp stencilLoadOp;
    VkAttachmentStoreOp stencilStoreOp;
    VkImageLayout finalLayout;
};

// Depth/stencil attachment with optional resolve target
struct RenderPassDepthStencilAttachment {
    RenderPassDepthStencilAttachmentTarget target;
    std::optional<RenderPassDepthStencilAttachmentTarget> resolveTarget;
};

struct RenderPassCreateInfo {
    std::vector<RenderPassColorAttachment> colorAttachments;
    std::optional<RenderPassDepthStencilAttachment> depthStencilAttachment;

    // Multiview extension (optional)
    std::optional<uint32_t> viewMask;
    std::vector<uint32_t> correlationMasks;
};

struct FramebufferCreateInfo {
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkImageView> attachments; // Interleaved: [color0, resolve0, color1, resolve1, ..., depth, depthResolve]
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t colorAttachmentCount = 0; // Number of color attachments (not including resolves)
    bool hasDepthResolve = false;
};

struct RenderPassEncoderBeginInfo {
    std::vector<VkClearColorValue> colorClearValues;
    float depthClearValue;
    uint32_t stencilClearValue;
    VkQueryPool occlusionQueryPool = VK_NULL_HANDLE;
    VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
    bool bundleExecution = false;
};

struct ComputePassEncoderCreateInfo {
    const char* label;
};

struct SubmitInfo {
    CommandEncoder* const* commandEncoders;
    uint32_t commandEncoderCount;
    Fence* signalFence;
    Semaphore* const* waitSemaphores;
    const uint64_t* waitValues;
    std::vector<VkPipelineStageFlags> waitStages; // One per wait semaphore (owned: converted from gfx stages)
    uint32_t waitSemaphoreCount;
    Semaphore* const* signalSemaphores;
    const uint64_t* signalValues;
    uint32_t signalSemaphoreCount;
};

} // namespace gfx::backend::vulkan::core

#endif // GFX_VULKAN_CREATEINFO_H