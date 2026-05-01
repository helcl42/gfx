#include <gfx/gfx.h>

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Platform-specific includes and macros
#if defined(__ANDROID__)
#include <android/log.h>
#include <android_native_app_glue.h>
#include <time.h>

// Android logging macros
#define LOG_TAG "GFX_COMPUTE"
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOG_WARN(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOG_DEBUG(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#else
// Desktop/Web logging macros
#define LOG_INFO(...)                  \
    do {                               \
        printf("[INFO] " __VA_ARGS__); \
        printf("\n");                  \
    } while (0)
#define LOG_ERROR(...)                           \
    do {                                         \
        fprintf(stderr, "[ERROR] " __VA_ARGS__); \
        fprintf(stderr, "\n");                   \
    } while (0)
#define LOG_WARN(...)                           \
    do {                                        \
        fprintf(stderr, "[WARN] " __VA_ARGS__); \
        fprintf(stderr, "\n");                  \
    } while (0)
#define LOG_DEBUG(...)                  \
    do {                                \
        printf("[DEBUG] " __VA_ARGS__); \
        printf("\n");                   \
    } while (0)

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#else
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <GLFW/glfw3native.h>
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define COMPUTE_TEXTURE_WIDTH 512
#define COMPUTE_TEXTURE_HEIGHT 512
#define COLOR_FORMAT GFX_FORMAT_B8G8R8A8_UNORM_SRGB

// Log callback function
static void logCallback(GfxLogLevel level, const char* message, void* userData)
{
    (void)userData;
    switch (level) {
    case GFX_LOG_LEVEL_ERROR:
        LOG_ERROR("%s", message);
        break;
    case GFX_LOG_LEVEL_WARNING:
        LOG_WARN("%s", message);
        break;
    case GFX_LOG_LEVEL_INFO:
        LOG_INFO("%s", message);
        break;
    case GFX_LOG_LEVEL_DEBUG:
        LOG_DEBUG("%s", message);
        break;
    default:
        LOG_INFO("[UNKNOWN] %s", message);
        break;
    }
}

typedef struct {
    float time;
    float padding[3]; // Pad to 16 bytes for WebGPU alignment
} ComputeUniformData;

typedef struct {
    float postProcessStrength;
    float padding[3]; // Pad to 16 bytes for WebGPU alignment
} RenderUniformData;

// Application settings/configuration
typedef struct {
    GfxBackend backend; // Graphics backend selection
    bool vsync; // VSync enabled (FIFO) or disabled (IMMEDIATE)
} Settings;

typedef struct {
    GfxCommandEncoder commandEncoder;
    GfxSemaphore imageAvailableSemaphore;
    GfxFence inFlightFence;
    GfxBindGroup computeBindGroup;
    GfxBuffer computeUniformBuffer;
    GfxBindGroup renderBindGroup;
    GfxBuffer renderUniformBuffer;
} PerFrameResources;

typedef struct {
#if defined(__ANDROID__)
    struct android_app* androidApp;
    bool animating;
#elif TARGET_OS_IOS
    void* metalLayer;
#else
    GLFWwindow* window;
#endif

    GfxInstance instance;
    GfxAdapter adapter;
    GfxAdapterInfo adapterInfo; // Cached adapter info
    GfxDevice device;
    GfxQueue queue;
    GfxSurface surface;
    GfxSwapchain swapchain;
    GfxSwapchainInfo swapchainInfo;

    // Compute resources
    GfxTexture computeTexture;
    GfxTextureView computeTextureView;
    GfxShader computeShader;
    GfxComputePipeline computePipeline;
    GfxBindGroupLayout computeBindGroupLayout;

    // Render resources (fullscreen quad)
    GfxShader vertexShader;
    GfxShader fragmentShader;
    GfxRenderPass renderPass;
    GfxRenderPipeline renderPipeline;
    GfxBindGroupLayout renderBindGroupLayout;
    GfxSampler sampler;

    // Per-frame resources (one per swapchain image)
    PerFrameResources* frameResources;
    GfxFramebuffer* framebuffers;
    GfxSemaphore* renderFinishedSemaphores;
    uint32_t currentFrame;

    uint32_t windowWidth;
    uint32_t windowHeight;
    uint32_t previousWidth;
    uint32_t previousHeight;

    // State
    float elapsedTime;
    float lastFrameTime;

    // FPS tracking
    uint32_t fpsFrameCount;
    float fpsTimeAccumulator;
    float fpsFrameTimeMin;
    float fpsFrameTimeMax;

    // Application settings
    Settings settings;
} ComputeApp;

// Private function declarations
static bool createWindow(ComputeApp* app, uint32_t width, uint32_t height);
static void destroyWindow(ComputeApp* app);
static bool createGraphics(ComputeApp* app);
static void destroyGraphics(ComputeApp* app);
static bool createPerFrameResources(ComputeApp* app);
static void destroyPerFrameResources(ComputeApp* app);
static bool createSizeDependentResources(ComputeApp* app, uint32_t width, uint32_t height);
static void destroySizeDependentResources(ComputeApp* app);
static bool createRenderPass(ComputeApp* app);
static void destroyRenderPass(ComputeApp* app);
static bool createSwapchain(ComputeApp* app, uint32_t width, uint32_t height);
static void destroySwapchain(ComputeApp* app);
static bool createFramebuffers(ComputeApp* app);
static void destroyFramebuffers(ComputeApp* app);

static bool createComputeTexture(ComputeApp* app);
static void destroyComputeTexture(ComputeApp* app);
static bool createComputeShaders(ComputeApp* app);
static void destroyComputeShaders(ComputeApp* app);
static bool createComputeBindGroupLayout(ComputeApp* app);
static void destroyComputeBindGroupLayout(ComputeApp* app);
static bool createComputePipeline(ComputeApp* app);
static void destroyComputePipeline(ComputeApp* app);
static bool transitionComputeTexture(ComputeApp* app);
static bool createComputeResources(ComputeApp* app);
static void destroyComputeResources(ComputeApp* app);
static bool createSampler(ComputeApp* app);
static void destroySampler(ComputeApp* app);
static bool createRenderShaders(ComputeApp* app);
static void destroyRenderShaders(ComputeApp* app);
static bool createRenderBindGroupLayout(ComputeApp* app);
static void destroyRenderBindGroupLayout(ComputeApp* app);
static bool createRenderPipeline(ComputeApp* app);
static void destroyRenderPipeline(ComputeApp* app);
static bool createRenderResources(ComputeApp* app);
static void destroyRenderResources(ComputeApp* app);

static GfxPlatformWindowHandle getPlatformWindowHandle(ComputeApp* app);
static float getCurrentTime(void);
static void* loadBinaryFile(ComputeApp* app, const char* filepath, size_t* outSize);
static void* loadTextFile(ComputeApp* app, const char* filepath, size_t* outSize);

// The public functions called from main
static bool parseArguments(int argc, char** argv, Settings* settings);
static bool init(ComputeApp* app);
static void cleanup(ComputeApp* app);
static void update(ComputeApp* app, float deltaTime);
static void render(ComputeApp* app);
static bool handleResize(ComputeApp* app, uint32_t width, uint32_t height);
static void updateFPS(ComputeApp* app, float deltaTime);

#if defined(__ANDROID__)
// Android-specific callbacks
static void handleAppCommand(struct android_app* app, int32_t cmd);
static int32_t handleInput(struct android_app* app, AInputEvent* event);
#elif TARGET_OS_IOS
// iOS doesn't use GLFW callbacks - UIKit handles events
#else
// GLFW callbacks
static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    ComputeApp* app = (ComputeApp*)glfwGetWindowUserPointer(window);
    app->windowWidth = (uint32_t)width;
    app->windowHeight = (uint32_t)height;
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

static void errorCallback(int error, const char* description)
{
    LOG_ERROR("GLFW Error %d: %s", error, description);
}
#endif // __ANDROID__ / TARGET_OS_IOS / desktop

static bool createWindow(ComputeApp* app, uint32_t width, uint32_t height)
{
#if defined(__ANDROID__) || TARGET_OS_IOS
    // On mobile platforms, window/layer is managed by the system
    app->windowWidth = width;
    app->windowHeight = height;
    LOG_INFO("Mobile window placeholder created");
    return true;
#else
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        LOG_ERROR("Failed to initialize GLFW");
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    const char* backendName = (app->settings.backend == GFX_BACKEND_VULKAN) ? "Vulkan" : "WebGPU";
    char windowTitle[128];
    snprintf(windowTitle, sizeof(windowTitle), "Compute Example - %s", backendName);

    app->window = glfwCreateWindow(width, height, windowTitle, NULL, NULL);
    if (!app->window) {
        LOG_ERROR("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }

    app->windowWidth = width;
    app->windowHeight = height;

    glfwSetWindowUserPointer(app->window, app);
    glfwSetFramebufferSizeCallback(app->window, framebufferResizeCallback);
    glfwSetKeyCallback(app->window, keyCallback);

    return true;
#endif
}

static void destroyWindow(ComputeApp* app)
{
#if defined(__ANDROID__) || TARGET_OS_IOS
    // Mobile window/layer is managed by the system
    (void)app;
#else
    if (app->window) {
        glfwDestroyWindow(app->window);
        app->window = NULL;
    }
    glfwTerminate();
#endif
}

static bool createGraphics(ComputeApp* app)
{
    // Set up logging callback
    gfxSetLogCallback(logCallback, NULL);

    const char* backendName = (app->settings.backend == GFX_BACKEND_VULKAN) ? "Vulkan" : "WebGPU";
    LOG_INFO("Loading graphics backend (%s)...", backendName);
    if (gfxLoadBackend(app->settings.backend) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to load any graphics backend");
        return false;
    }

    // Create graphics instance
    const char* instanceExtensions[] = { GFX_INSTANCE_EXTENSION_SURFACE, GFX_INSTANCE_EXTENSION_DEBUG };
    GfxInstanceDescriptor instanceDesc = {
        .sType = GFX_STRUCTURE_TYPE_INSTANCE_DESCRIPTOR,
        .pNext = NULL,
        .backend = app->settings.backend,
        .applicationName = "Compute Example (C)",
        .applicationVersion = 1,
        .enabledExtensions = instanceExtensions,
        .enabledExtensionCount = ARRAY_SIZE(instanceExtensions)
    };

    if (gfxCreateInstance(&instanceDesc, &app->instance) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create graphics instance");
        return false;
    }

    // Get adapter
    GfxAdapterDescriptor adapterDesc = {
        .sType = GFX_STRUCTURE_TYPE_ADAPTER_DESCRIPTOR,
        .pNext = NULL,
        .adapterIndex = UINT32_MAX,
        .preference = GFX_ADAPTER_PREFERENCE_HIGH_PERFORMANCE
    };

    if (gfxInstanceRequestAdapter(app->instance, &adapterDesc, &app->adapter) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to request adapter");
        return false;
    }

    // Query and store adapter info
    gfxAdapterGetInfo(app->adapter, &app->adapterInfo);
    LOG_INFO("Using adapter: %s", app->adapterInfo.name);
    LOG_INFO("  Backend: %s", app->adapterInfo.backend == GFX_BACKEND_VULKAN ? "Vulkan" : "WebGPU");
    LOG_INFO("  Vendor ID: 0x%04X, Device ID: 0x%04X", app->adapterInfo.vendorID, app->adapterInfo.deviceID);

    // Create device
    const char* deviceExtensions[] = { GFX_DEVICE_EXTENSION_SWAPCHAIN };
    GfxDeviceDescriptor deviceDesc = {
        .sType = GFX_STRUCTURE_TYPE_DEVICE_DESCRIPTOR,
        .pNext = NULL,
        .label = NULL,
        .queueRequests = NULL,
        .queueRequestCount = 0,
        .enabledExtensions = deviceExtensions,
        .enabledExtensionCount = ARRAY_SIZE(deviceExtensions)
    };

    if (gfxAdapterCreateDevice(app->adapter, &deviceDesc, &app->device) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create device");
        return false;
    }

    if (gfxDeviceGetQueue(app->device, &app->queue) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get device queue");
        return false;
    }

    GfxPlatformWindowHandle windowHandle = getPlatformWindowHandle(app);
    GfxSurfaceDescriptor surfaceDesc = {
        .sType = GFX_STRUCTURE_TYPE_SURFACE_DESCRIPTOR,
        .pNext = NULL,
        .label = "Main Surface",
        .windowHandle = windowHandle
    };

    if (gfxInstanceCreateSurface(app->instance, &surfaceDesc, &app->surface) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create surface");
        return false;
    }

    return true;
}

static void destroyGraphics(ComputeApp* app)
{
    if (app->surface) {
        gfxSurfaceDestroy(app->surface);
        app->surface = NULL;
    }
    if (app->device) {
        gfxDeviceDestroy(app->device);
        app->device = NULL;
    }
    if (app->instance) {
        gfxInstanceDestroy(app->instance);
        app->instance = NULL;
    }

    gfxUnloadBackend(app->settings.backend);
}

static bool createPerFrameResources(ComputeApp* app)
{
    // Allocate per-frame resources array
    app->frameResources = (PerFrameResources*)calloc(app->swapchainInfo.imageCount, sizeof(PerFrameResources));
    if (!app->frameResources) {
        LOG_ERROR("Failed to allocate per-frame resources array");
        return false;
    }

    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        GfxSemaphoreDescriptor semaphoreDesc = {
            .sType = GFX_STRUCTURE_TYPE_SEMAPHORE_DESCRIPTOR,
            .pNext = NULL,
            .type = GFX_SEMAPHORE_TYPE_BINARY
        };

        if (gfxDeviceCreateSemaphore(app->device, &semaphoreDesc, &app->frameResources[i].imageAvailableSemaphore) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create image available semaphore");
            return false;
        }

        GfxFenceDescriptor fenceDesc = {
            .sType = GFX_STRUCTURE_TYPE_FENCE_DESCRIPTOR,
            .pNext = NULL,
            .signaled = true
        };

        if (gfxDeviceCreateFence(app->device, &fenceDesc, &app->frameResources[i].inFlightFence) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create fence");
            return false;
        }

        // Create command encoder for this frame
        char label[64];
        snprintf(label, sizeof(label), "Command Encoder %u", i);
        GfxCommandEncoderDescriptor encoderDesc = {
            .sType = GFX_STRUCTURE_TYPE_COMMAND_ENCODER_DESCRIPTOR,
            .pNext = NULL,
            .label = label
        };

        if (gfxDeviceCreateCommandEncoder(app->device, &encoderDesc, &app->frameResources[i].commandEncoder) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create command encoder %u", i);
            return false;
        }

        // Create compute uniform buffer for this frame
        GfxBufferDescriptor computeUniformBufferDesc = {
            .sType = GFX_STRUCTURE_TYPE_BUFFER_DESCRIPTOR,
            .pNext = NULL,
            .size = sizeof(ComputeUniformData),
            .usage = GFX_BUFFER_USAGE_UNIFORM | GFX_BUFFER_USAGE_COPY_DST,
            .memoryProperties = GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT
        };

        if (gfxDeviceCreateBuffer(app->device, &computeUniformBufferDesc, &app->frameResources[i].computeUniformBuffer) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create compute uniform buffer %u", i);
            return false;
        }

        // Create render uniform buffer for this frame
        GfxBufferDescriptor renderUniformBufferDesc = {
            .sType = GFX_STRUCTURE_TYPE_BUFFER_DESCRIPTOR,
            .pNext = NULL,
            .size = sizeof(RenderUniformData),
            .usage = GFX_BUFFER_USAGE_UNIFORM | GFX_BUFFER_USAGE_COPY_DST,
            .memoryProperties = GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT
        };

        if (gfxDeviceCreateBuffer(app->device, &renderUniformBufferDesc, &app->frameResources[i].renderUniformBuffer) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create render uniform buffer %u", i);
            return false;
        }

        // Create compute bind group for this frame
        GfxBindGroupEntry computeEntries[2] = {
            {
                .binding = 0,
                .type = GFX_BIND_GROUP_ENTRY_TYPE_TEXTURE_VIEW,
                .resource = {
                    .textureView = app->computeTextureView,
                },
            },
            {
                .binding = 1,
                .type = GFX_BIND_GROUP_ENTRY_TYPE_BUFFER,
                .resource = {
                    .buffer = {
                        .buffer = app->frameResources[i].computeUniformBuffer,
                        .offset = 0,
                        .size = sizeof(ComputeUniformData),
                    },
                },
            },
        };

        GfxBindGroupDescriptor computeBindGroupDesc = {
            .sType = GFX_STRUCTURE_TYPE_BIND_GROUP_DESCRIPTOR,
            .pNext = NULL,
            .layout = app->computeBindGroupLayout,
            .entryCount = ARRAY_SIZE(computeEntries),
            .entries = computeEntries
        };

        if (gfxDeviceCreateBindGroup(app->device, &computeBindGroupDesc, &app->frameResources[i].computeBindGroup) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create compute bind group %u", i);
            return false;
        }

        // Create render bind group for this frame
        GfxBindGroupEntry renderEntries[3] = {
            {
                .binding = 0,
                .type = GFX_BIND_GROUP_ENTRY_TYPE_SAMPLER,
                .resource = {
                    .sampler = app->sampler,
                },
            },
            {
                .binding = 1,
                .type = GFX_BIND_GROUP_ENTRY_TYPE_TEXTURE_VIEW,
                .resource = {
                    .textureView = app->computeTextureView,
                },
            },
            {
                .binding = 2,
                .type = GFX_BIND_GROUP_ENTRY_TYPE_BUFFER,
                .resource = {
                    .buffer = {
                        .buffer = app->frameResources[i].renderUniformBuffer,
                        .offset = 0,
                        .size = sizeof(RenderUniformData),
                    },
                },
            }
        };

        GfxBindGroupDescriptor renderBindGroupDesc = {
            .sType = GFX_STRUCTURE_TYPE_BIND_GROUP_DESCRIPTOR,
            .pNext = NULL,
            .layout = app->renderBindGroupLayout,
            .entryCount = ARRAY_SIZE(renderEntries),
            .entries = renderEntries
        };

        if (gfxDeviceCreateBindGroup(app->device, &renderBindGroupDesc, &app->frameResources[i].renderBindGroup) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create render bind group %u", i);
            return false;
        }
    }

    return true;
}

static void destroyPerFrameResources(ComputeApp* app)
{
    if (app->frameResources) {
        for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
            PerFrameResources* frame = &app->frameResources[i];

            // Wait for fence before destroying resources
            if (frame->inFlightFence) {
                gfxFenceWait(frame->inFlightFence, GFX_TIMEOUT_INFINITE);
            }

            // Destroy in reverse order of creation
            // 8. Destroy bind groups (created last)
            if (frame->renderBindGroup) {
                gfxBindGroupDestroy(frame->renderBindGroup);
                frame->renderBindGroup = NULL;
            }
            if (frame->computeBindGroup) {
                gfxBindGroupDestroy(frame->computeBindGroup);
                frame->computeBindGroup = NULL;
            }

            // 7-6. Destroy uniform buffers
            if (frame->renderUniformBuffer) {
                gfxBufferDestroy(frame->renderUniformBuffer);
                frame->renderUniformBuffer = NULL;
            }
            if (frame->computeUniformBuffer) {
                gfxBufferDestroy(frame->computeUniformBuffer);
                frame->computeUniformBuffer = NULL;
            }

            // 5. Destroy command encoder
            if (frame->commandEncoder) {
                gfxCommandEncoderDestroy(frame->commandEncoder);
                frame->commandEncoder = NULL;
            }

            // 4-1. Destroy synchronization objects
            if (frame->inFlightFence) {
                gfxFenceDestroy(frame->inFlightFence);
                frame->inFlightFence = NULL;
            }
            if (frame->imageAvailableSemaphore) {
                gfxSemaphoreDestroy(frame->imageAvailableSemaphore);
                frame->imageAvailableSemaphore = NULL;
            }
        }
        free(app->frameResources);
        app->frameResources = NULL;
    }
}

// Helper function to recreate size-dependent resources
static bool createSizeDependentResources(ComputeApp* app, uint32_t width, uint32_t height)
{
    // Create swapchain with new dimensions
    // Compute texture stays at fixed resolution and is sampled with linear filtering
    if (!createSwapchain(app, width, height)) {
        return false;
    }

    if (!createRenderPass(app)) {
        return false;
    }

    // Recreate framebuffers with new swapchain images
    if (!createFramebuffers(app)) {
        return false;
    }

    return true;
}

static void destroySizeDependentResources(ComputeApp* app)
{
    destroyFramebuffers(app);
    destroyRenderPass(app);
    destroySwapchain(app);
}

static bool createRenderPass(ComputeApp* app)
{
    // Define color attachment target
    GfxRenderPassColorAttachmentTarget colorTarget = {
        .format = app->swapchainInfo.format,
        .sampleCount = GFX_SAMPLE_COUNT_1,
        .ops = {
            .loadOp = GFX_LOAD_OP_CLEAR,
            .storeOp = GFX_STORE_OP_STORE },
        .finalLayout = GFX_TEXTURE_LAYOUT_PRESENT_SRC
    };

    GfxRenderPassColorAttachment colorAttachment = {
        .target = colorTarget,
        .resolveTarget = NULL
    };

    GfxRenderPassDescriptor renderPassDesc = {
        .sType = GFX_STRUCTURE_TYPE_RENDER_PASS_DESCRIPTOR,
        .pNext = NULL,
        .label = "Fullscreen Render Pass",
        .colorAttachments = &colorAttachment,
        .colorAttachmentCount = 1,
        .depthStencilAttachment = NULL
    };

    if (gfxDeviceCreateRenderPass(app->device, &renderPassDesc, &app->renderPass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create render pass");
        return false;
    }

    return true;
}

static void destroyRenderPass(ComputeApp* app)
{
    if (app->renderPass) {
        gfxRenderPassDestroy(app->renderPass);
        app->renderPass = NULL;
    }
}

static bool createSwapchain(ComputeApp* app, uint32_t width, uint32_t height)
{
    // Query surface capabilities to determine frame count
    GfxSurfaceInfo surfaceInfo;
    if (gfxSurfaceGetInfo(app->surface, app->adapter, &surfaceInfo) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get surface info");
        return false;
    }

    LOG_INFO("Surface Info:");
    LOG_INFO("  Image Count: min %u, max %u", surfaceInfo.minImageCount, surfaceInfo.maxImageCount);
    LOG_INFO("  Extent: min (%u, %u), max (%u, %u)",
        surfaceInfo.minExtent.width, surfaceInfo.minExtent.height,
        surfaceInfo.maxExtent.width, surfaceInfo.maxExtent.height);

    GfxSwapchainDescriptor swapchainDesc = {
        .sType = GFX_STRUCTURE_TYPE_SWAPCHAIN_DESCRIPTOR,
        .pNext = NULL,
        .surface = app->surface,
        .extent.width = width,
        .extent.height = height,
        .format = COLOR_FORMAT,
        .usage = GFX_TEXTURE_USAGE_RENDER_ATTACHMENT,
        .presentMode = app->settings.vsync ? GFX_PRESENT_MODE_FIFO : GFX_PRESENT_MODE_IMMEDIATE,
        .imageCount = surfaceInfo.minImageCount
    };

    if (gfxDeviceCreateSwapchain(app->device, &swapchainDesc, &app->swapchain) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create swapchain");
        return false;
    }

    gfxSwapchainGetInfo(app->swapchain, &app->swapchainInfo);

    // Create render finished semaphores (one per swapchain image)
    app->renderFinishedSemaphores = (GfxSemaphore*)malloc(app->swapchainInfo.imageCount * sizeof(GfxSemaphore));
    if (!app->renderFinishedSemaphores) {
        LOG_ERROR("Failed to allocate renderFinishedSemaphores");
        return false;
    }
    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        app->renderFinishedSemaphores[i] = NULL;
    }

    GfxSemaphoreDescriptor semaphoreDesc = {
        .sType = GFX_STRUCTURE_TYPE_SEMAPHORE_DESCRIPTOR,
        .pNext = NULL,
        .type = GFX_SEMAPHORE_TYPE_BINARY
    };

    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        if (gfxDeviceCreateSemaphore(app->device, &semaphoreDesc, &app->renderFinishedSemaphores[i]) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create render finished semaphore %u", i);
            return false;
        }
    }

    return true;
}

static void destroySwapchain(ComputeApp* app)
{
    // Clean up render finished semaphores
    if (app->renderFinishedSemaphores) {
        for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
            if (app->renderFinishedSemaphores[i]) {
                gfxSemaphoreDestroy(app->renderFinishedSemaphores[i]);
            }
        }
        free(app->renderFinishedSemaphores);
        app->renderFinishedSemaphores = NULL;
    }

    if (app->swapchain) {
        gfxSwapchainDestroy(app->swapchain);
        app->swapchain = NULL;
    }
}

static bool createFramebuffers(ComputeApp* app)
{
    // Allocate framebuffers array
    app->framebuffers = (GfxFramebuffer*)calloc(app->swapchainInfo.imageCount, sizeof(GfxFramebuffer));
    if (!app->framebuffers) {
        LOG_ERROR("Failed to allocate framebuffers array");
        return false;
    }

    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        GfxTextureView backbuffer = NULL;
        GfxResult result = gfxSwapchainGetTextureView(app->swapchain, i, &backbuffer);
        if (result != GFX_RESULT_SUCCESS || !backbuffer) {
            LOG_ERROR("Failed to get swapchain image view %u", i);
            return false;
        }

        GfxFramebufferAttachment fbColorAttachment = {
            .view = backbuffer,
            .resolveTarget = NULL
        };

        GfxFramebufferAttachment fbDepthAttachment = {
            .view = NULL,
            .resolveTarget = NULL
        };

        char label[64];
        snprintf(label, sizeof(label), "Framebuffer %u", i);

        GfxFramebufferDescriptor fbDesc = {
            .sType = GFX_STRUCTURE_TYPE_FRAMEBUFFER_DESCRIPTOR,
            .pNext = NULL,
            .label = label,
            .renderPass = app->renderPass,
            .colorAttachments = &fbColorAttachment,
            .colorAttachmentCount = 1,
            .depthStencilAttachment = fbDepthAttachment,
            .extent.width = app->swapchainInfo.extent.width,
            .extent.height = app->swapchainInfo.extent.height
        };

        if (gfxDeviceCreateFramebuffer(app->device, &fbDesc, &app->framebuffers[i]) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create framebuffer %u", i);
            return false;
        }
    }

    return true;
}

static void destroyFramebuffers(ComputeApp* app)
{
    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        if (app->framebuffers[i]) {
            gfxFramebufferDestroy(app->framebuffers[i]);
            app->framebuffers[i] = NULL;
        }
    }

    // Free the framebuffers array
    if (app->framebuffers) {
        free(app->framebuffers);
        app->framebuffers = NULL;
    }
}

static bool createComputeTexture(ComputeApp* app)
{
    GfxTextureDescriptor textureDesc = {
        .sType = GFX_STRUCTURE_TYPE_TEXTURE_DESCRIPTOR,
        .pNext = NULL,
        .type = GFX_TEXTURE_TYPE_2D,
        .size = (GfxExtent3D){ COMPUTE_TEXTURE_WIDTH, COMPUTE_TEXTURE_HEIGHT, 1 },
        .format = GFX_FORMAT_R8G8B8A8_UNORM,
        .usage = GFX_TEXTURE_USAGE_STORAGE_BINDING | GFX_TEXTURE_USAGE_TEXTURE_BINDING,
        .arrayLayerCount = 1,
        .mipLevelCount = 1,
        .sampleCount = GFX_SAMPLE_COUNT_1
    };

    if (gfxDeviceCreateTexture(app->device, &textureDesc, &app->computeTexture) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create compute texture");
        return false;
    }

    LOG_INFO("Created compute texture: %dx%d", COMPUTE_TEXTURE_WIDTH, COMPUTE_TEXTURE_HEIGHT);

    GfxTextureViewDescriptor viewDesc = {
        .sType = GFX_STRUCTURE_TYPE_TEXTURE_VIEW_DESCRIPTOR,
        .pNext = NULL,
        .format = GFX_FORMAT_R8G8B8A8_UNORM,
        .viewType = GFX_TEXTURE_VIEW_TYPE_2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };

    if (gfxTextureCreateView(app->computeTexture, &viewDesc, &app->computeTextureView) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create compute texture view");
        return false;
    }

    return true;
}

static void destroyComputeTexture(ComputeApp* app)
{
    if (app->computeTextureView) {
        gfxTextureViewDestroy(app->computeTextureView);
        app->computeTextureView = NULL;
    }
    if (app->computeTexture) {
        gfxTextureDestroy(app->computeTexture);
        app->computeTexture = NULL;
    }
}

static bool createComputeShaders(ComputeApp* app)
{
    void* computeCode = NULL;
    size_t computeSize = 0;
    GfxShaderSourceType sourceType = GFX_SHADER_SOURCE_SPIRV;

    // Try shader formats in order of preference
    const struct {
        GfxShaderSourceType format;
        const char* path;
    } shaderFormats[] = {
        { GFX_SHADER_SOURCE_SPIRV, "shaders/generate.comp.spv" },
        { GFX_SHADER_SOURCE_WGSL, "shaders/generate.comp.wgsl" }
    };

    for (size_t i = 0; i < ARRAY_SIZE(shaderFormats); ++i) {
        bool formatSupported = false;
        if (gfxDeviceSupportsShaderFormat(app->device, shaderFormats[i].format, &formatSupported) != GFX_RESULT_SUCCESS || !formatSupported) {
            continue;
        }

        LOG_INFO("Loading compute shader: %s", shaderFormats[i].path);

        if (shaderFormats[i].format == GFX_SHADER_SOURCE_SPIRV) {
            computeCode = loadBinaryFile(app, shaderFormats[i].path, &computeSize);
        } else {
            computeCode = loadTextFile(app, shaderFormats[i].path, &computeSize);
        }

        if (computeCode) {
            sourceType = shaderFormats[i].format;
            LOG_INFO("Successfully loaded compute shader (%zu bytes)", computeSize);
            break;
        }

        // Failed to load this format, try next
        free(computeCode);
        computeCode = NULL;
    }

    if (!computeCode) {
        LOG_ERROR("Error: No supported shader format found or failed to load compute shader");
        return false;
    }

    GfxShaderDescriptor computeShaderDesc = {
        .sType = GFX_STRUCTURE_TYPE_SHADER_DESCRIPTOR,
        .pNext = NULL,
        .sourceType = sourceType,
        .code = computeCode,
        .codeSize = computeSize,
        .entryPoint = "main"
    };

    if (gfxDeviceCreateShader(app->device, &computeShaderDesc, &app->computeShader) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create compute shader");
        free(computeCode);
        return false;
    }
    free(computeCode);
    return true;
}

static void destroyComputeShaders(ComputeApp* app)
{
    if (app->computeShader) {
        gfxShaderDestroy(app->computeShader);
        app->computeShader = NULL;
    }
}

static bool createComputeBindGroupLayout(ComputeApp* app)
{
    GfxBindGroupLayoutEntry computeLayoutEntries[2] = {
        {
            .binding = 0,
            .visibility = GFX_SHADER_STAGE_COMPUTE,
            .type = GFX_BINDING_TYPE_STORAGE_TEXTURE,
            .count = 1,
            .storageTexture = {
                .format = GFX_FORMAT_R8G8B8A8_UNORM,
                .viewDimension = GFX_TEXTURE_VIEW_TYPE_2D,
                .access = GFX_STORAGE_TEXTURE_ACCESS_WRITE_ONLY,
            },
        },
        {
            .binding = 1,
            .visibility = GFX_SHADER_STAGE_COMPUTE,
            .type = GFX_BINDING_TYPE_BUFFER,
            .count = 1,
            .buffer = {
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(ComputeUniformData),
            },
        }
    };

    GfxBindGroupLayoutDescriptor computeLayoutDesc = {
        .sType = GFX_STRUCTURE_TYPE_BIND_GROUP_LAYOUT_DESCRIPTOR,
        .pNext = NULL,
        .entryCount = ARRAY_SIZE(computeLayoutEntries),
        .entries = computeLayoutEntries
    };

    if (gfxDeviceCreateBindGroupLayout(app->device, &computeLayoutDesc, &app->computeBindGroupLayout) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create compute bind group layout");
        return false;
    }

    // Note: Bind groups are created in createPerFrameResources() after all resources are ready
    return true;
}

static void destroyComputeBindGroupLayout(ComputeApp* app)
{
    // Compute bind groups are now destroyed in destroyPerFrameResources
    // Only destroy the layout here
    if (app->computeBindGroupLayout) {
        gfxBindGroupLayoutDestroy(app->computeBindGroupLayout);
        app->computeBindGroupLayout = NULL;
    }
}

static bool createComputePipeline(ComputeApp* app)
{
    GfxComputePipelineDescriptor computePipelineDesc = {
        .sType = GFX_STRUCTURE_TYPE_COMPUTE_PIPELINE_DESCRIPTOR,
        .pNext = NULL,
        .compute = app->computeShader,
        .entryPoint = "main",
        .bindGroupLayouts = &app->computeBindGroupLayout,
        .bindGroupLayoutCount = 1
    };

    if (gfxDeviceCreateComputePipeline(app->device, &computePipelineDesc, &app->computePipeline) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create compute pipeline");
        return false;
    }
    return true;
}

static void destroyComputePipeline(ComputeApp* app)
{
    if (app->computePipeline) {
        gfxComputePipelineDestroy(app->computePipeline);
        app->computePipeline = NULL;
    }
}

static bool transitionComputeTexture(ComputeApp* app)
{
    // Transition compute texture to SHADER_READ_ONLY layout initially
    // This way we don't need special handling for the first frame
    GfxCommandEncoder initEncoder = NULL;
    GfxCommandEncoderDescriptor initEncoderDesc = {
        .sType = GFX_STRUCTURE_TYPE_COMMAND_ENCODER_DESCRIPTOR,
        .pNext = NULL,
        .label = "Init Layout Transition"
    };

    if (gfxDeviceCreateCommandEncoder(app->device, &initEncoderDesc, &initEncoder) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create command encoder for initial layout transition");
        return false;
    }

    if (gfxCommandEncoderBegin(initEncoder) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to begin initialization command encoder");
        gfxCommandEncoderDestroy(initEncoder);
        return false;
    }

    GfxTextureBarrier initBarrier = {
        .texture = app->computeTexture,
        .oldLayout = GFX_TEXTURE_LAYOUT_UNDEFINED,
        .newLayout = GFX_TEXTURE_LAYOUT_SHADER_READ_ONLY,
        .srcStageMask = GFX_PIPELINE_STAGE_TOP_OF_PIPE,
        .dstStageMask = GFX_PIPELINE_STAGE_FRAGMENT_SHADER,
        .srcAccessMask = 0,
        .dstAccessMask = GFX_ACCESS_SHADER_READ,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };
    GfxPipelineBarrierDescriptor barrierDesc = {
        .sType = GFX_STRUCTURE_TYPE_PIPELINE_BARRIER_DESCRIPTOR,
        .pNext = NULL,
        .memoryBarriers = NULL,
        .memoryBarrierCount = 0,
        .bufferBarriers = NULL,
        .bufferBarrierCount = 0,
        .textureBarriers = &initBarrier,
        .textureBarrierCount = 1
    };

    if (gfxCommandEncoderPipelineBarrier(initEncoder, &barrierDesc) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to record initialization barrier");
        gfxCommandEncoderDestroy(initEncoder);
        return false;
    }

    if (gfxCommandEncoderEnd(initEncoder) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to end initialization command encoder");
        gfxCommandEncoderDestroy(initEncoder);
        return false;
    }

    GfxSubmitDescriptor submitDescriptor = {
        .sType = GFX_STRUCTURE_TYPE_SUBMIT_DESCRIPTOR,
        .pNext = NULL,
        .commandEncoderCount = 1,
        .commandEncoders = &initEncoder
    };

    if (gfxQueueSubmit(app->queue, &submitDescriptor) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to submit initialization commands");
        gfxCommandEncoderDestroy(initEncoder);
        return false;
    }
    gfxDeviceWaitIdle(app->device);

    gfxCommandEncoderDestroy(initEncoder);
    return true;
}

static bool createComputeResources(ComputeApp* app)
{
    if (!createComputeTexture(app)) {
        return false;
    }

    if (!createComputeShaders(app)) {
        return false;
    }

    if (!createComputeBindGroupLayout(app)) {
        return false;
    }

    if (!createComputePipeline(app)) {
        return false;
    }

    if (!transitionComputeTexture(app)) {
        return false;
    }

    LOG_INFO("Compute resources created successfully");
    return true;
}

static void destroyComputeResources(ComputeApp* app)
{
    destroyComputePipeline(app);
    destroyComputeBindGroupLayout(app);
    destroyComputeShaders(app);
    destroyComputeTexture(app);
}

static bool createSampler(ComputeApp* app)
{
    GfxSamplerDescriptor samplerDesc = {
        .sType = GFX_STRUCTURE_TYPE_SAMPLER_DESCRIPTOR,
        .pNext = NULL,
        .magFilter = GFX_FILTER_MODE_LINEAR,
        .minFilter = GFX_FILTER_MODE_LINEAR,
        .addressModeU = GFX_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = GFX_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxAnisotropy = 1
    };

    if (gfxDeviceCreateSampler(app->device, &samplerDesc, &app->sampler) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create sampler");
        return false;
    }

    return true;
}

static void destroySampler(ComputeApp* app)
{
    if (app->sampler) {
        gfxSamplerDestroy(app->sampler);
        app->sampler = NULL;
    }
}

static bool createRenderShaders(ComputeApp* app)
{
    GfxShaderSourceType sourceType = GFX_SHADER_SOURCE_SPIRV;
    void* vertexCode = NULL;
    void* fragmentCode = NULL;
    size_t vertexSize = 0;
    size_t fragmentSize = 0;

    // Try shader formats in order of preference
    const struct {
        GfxShaderSourceType format;
        const char* vertexPath;
        const char* fragmentPath;
    } shaderFormats[] = {
        { GFX_SHADER_SOURCE_SPIRV, "shaders/fullscreen.vert.spv", "shaders/postprocess.frag.spv" },
        { GFX_SHADER_SOURCE_WGSL, "shaders/fullscreen.vert.wgsl", "shaders/postprocess.frag.wgsl" }
    };

    for (size_t i = 0; i < ARRAY_SIZE(shaderFormats); ++i) {
        bool formatSupported = false;
        if (gfxDeviceSupportsShaderFormat(app->device, shaderFormats[i].format, &formatSupported) != GFX_RESULT_SUCCESS || !formatSupported) {
            continue;
        }

        LOG_INFO("Loading render shaders: %s, %s", shaderFormats[i].vertexPath, shaderFormats[i].fragmentPath);

        if (shaderFormats[i].format == GFX_SHADER_SOURCE_SPIRV) {
            vertexCode = loadBinaryFile(app, shaderFormats[i].vertexPath, &vertexSize);
            fragmentCode = loadBinaryFile(app, shaderFormats[i].fragmentPath, &fragmentSize);
        } else {
            vertexCode = loadTextFile(app, shaderFormats[i].vertexPath, &vertexSize);
            fragmentCode = loadTextFile(app, shaderFormats[i].fragmentPath, &fragmentSize);
        }

        if (vertexCode && fragmentCode) {
            sourceType = shaderFormats[i].format;
            LOG_INFO("Successfully loaded render shaders (vertex: %zu bytes, fragment: %zu bytes)",
                vertexSize, fragmentSize);
            break;
        }

        // Failed to load this format, try next
        free(vertexCode);
        free(fragmentCode);
        vertexCode = NULL;
        fragmentCode = NULL;
    }

    if (!vertexCode || !fragmentCode) {
        LOG_ERROR("Error: No supported shader format found or failed to load render shaders");
        free(vertexCode);
        free(fragmentCode);
        return false;
    }

    GfxShaderDescriptor vertexShaderDesc = {
        .sType = GFX_STRUCTURE_TYPE_SHADER_DESCRIPTOR,
        .pNext = NULL,
        .sourceType = sourceType,
        .code = vertexCode,
        .codeSize = vertexSize,
        .entryPoint = "main"
    };

    if (gfxDeviceCreateShader(app->device, &vertexShaderDesc, &app->vertexShader) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create vertex shader");
        free(vertexCode);
        free(fragmentCode);
        return false;
    }
    free(vertexCode);

    GfxShaderDescriptor fragmentShaderDesc = {
        .sType = GFX_STRUCTURE_TYPE_SHADER_DESCRIPTOR,
        .pNext = NULL,
        .sourceType = sourceType,
        .code = fragmentCode,
        .codeSize = fragmentSize,
        .entryPoint = "main"
    };

    if (gfxDeviceCreateShader(app->device, &fragmentShaderDesc, &app->fragmentShader) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create fragment shader");
        free(fragmentCode);
        return false;
    }
    free(fragmentCode);
    return true;
}

static void destroyRenderShaders(ComputeApp* app)
{
    if (app->fragmentShader) {
        gfxShaderDestroy(app->fragmentShader);
        app->fragmentShader = NULL;
    }
    if (app->vertexShader) {
        gfxShaderDestroy(app->vertexShader);
        app->vertexShader = NULL;
    }
}

static bool createRenderBindGroupLayout(ComputeApp* app)
{
    // Create render bind group layout
    GfxBindGroupLayoutEntry renderLayoutEntries[3] = {
        {
            .binding = 0,
            .visibility = GFX_SHADER_STAGE_FRAGMENT,
            .type = GFX_BINDING_TYPE_SAMPLER,
            .count = 1,
            .sampler = {
                .type = GFX_SAMPLER_BINDING_TYPE_FILTERING,
            },
        },
        {
            .binding = 1,
            .visibility = GFX_SHADER_STAGE_FRAGMENT,
            .type = GFX_BINDING_TYPE_TEXTURE,
            .count = 1,
            .texture = {
                .sampleType = GFX_TEXTURE_SAMPLE_TYPE_FLOAT,
                .viewDimension = GFX_TEXTURE_VIEW_TYPE_2D,
                .multisampled = false,
            },
        },
        {
            .binding = 2,
            .visibility = GFX_SHADER_STAGE_FRAGMENT,
            .type = GFX_BINDING_TYPE_BUFFER,
            .count = 1,
            .buffer = {
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(RenderUniformData),
            },
        },
    };

    GfxBindGroupLayoutDescriptor renderLayoutDesc = {
        .sType = GFX_STRUCTURE_TYPE_BIND_GROUP_LAYOUT_DESCRIPTOR,
        .pNext = NULL,
        .entryCount = ARRAY_SIZE(renderLayoutEntries),
        .entries = renderLayoutEntries
    };

    if (gfxDeviceCreateBindGroupLayout(app->device, &renderLayoutDesc, &app->renderBindGroupLayout) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create render bind group layout");
        return false;
    }

    // Note: Bind groups are created in createPerFrameResources() after all resources are ready
    return true;
}

static void destroyRenderBindGroupLayout(ComputeApp* app)
{
    // Render bind groups are now destroyed in destroyPerFrameResources
    // Only destroy the layout here
    if (app->renderBindGroupLayout) {
        gfxBindGroupLayoutDestroy(app->renderBindGroupLayout);
        app->renderBindGroupLayout = NULL;
    }
}

static bool createRenderPipeline(ComputeApp* app)
{
    GfxVertexState vertexState = {
        .module = app->vertexShader,
        .entryPoint = "main",
        .bufferCount = 0
    };

    GfxColorTargetState colorTarget = {
        .format = app->swapchainInfo.format,
        .writeMask = GFX_COLOR_WRITE_MASK_ALL
    };

    GfxFragmentState fragmentState = {
        .module = app->fragmentShader,
        .entryPoint = "main",
        .targetCount = 1,
        .targets = &colorTarget
    };

    GfxPrimitiveState primitiveState = {
        .topology = GFX_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .frontFace = GFX_FRONT_FACE_COUNTER_CLOCKWISE,
        .cullMode = GFX_CULL_MODE_NONE,
        .polygonMode = GFX_POLYGON_MODE_FILL
    };

    GfxRenderPipelineDescriptor pipelineDesc = {
        .sType = GFX_STRUCTURE_TYPE_RENDER_PIPELINE_DESCRIPTOR,
        .pNext = NULL,
        .renderPass = app->renderPass,
        .vertex = &vertexState,
        .fragment = &fragmentState,
        .primitive = &primitiveState,
        .depthStencil = NULL,
        .sampleCount = GFX_SAMPLE_COUNT_1,
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &app->renderBindGroupLayout
    };

    if (gfxDeviceCreateRenderPipeline(app->device, &pipelineDesc, &app->renderPipeline) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create render pipeline");
        return false;
    }

    return true;
}

static void destroyRenderPipeline(ComputeApp* app)
{
    if (app->renderPipeline) {
        gfxRenderPipelineDestroy(app->renderPipeline);
        app->renderPipeline = NULL;
    }
}

static bool createRenderResources(ComputeApp* app)
{
    if (!createRenderShaders(app)) {
        return false;
    }

    if (!createSampler(app)) {
        return false;
    }

    if (!createRenderBindGroupLayout(app)) {
        return false;
    }

    if (!createRenderPipeline(app)) {
        return false;
    }

    LOG_INFO("Render resources created successfully");
    return true;
}

static void destroyRenderResources(ComputeApp* app)
{
    destroyRenderPipeline(app);
    destroyRenderBindGroupLayout(app);
    destroySampler(app);
    destroyRenderShaders(app);
}

static GfxPlatformWindowHandle getPlatformWindowHandle(ComputeApp* app)
{
    GfxPlatformWindowHandle handle = { 0 };
#if defined(__ANDROID__)
    handle = gfxPlatformWindowHandleFromAndroid(app->androidApp->window);
#elif defined(__EMSCRIPTEN__)
    handle = gfxPlatformWindowHandleFromEmscripten("#canvas");
#elif TARGET_OS_IOS
    handle = gfxPlatformWindowHandleFromMetal(app->metalLayer);
#elif defined(_WIN32)
    handle = gfxPlatformWindowHandleFromWin32(GetModuleHandle(NULL), glfwGetWin32Window(app->window));
#elif defined(__linux__)
    // handle = gfxPlatformWindowHandleFromXlib(glfwGetX11Display(), glfwGetX11Window(app->window));
    handle = gfxPlatformWindowHandleFromWayland(glfwGetWaylandDisplay(), glfwGetWaylandWindow(app->window));
#elif defined(__APPLE__)
    handle = gfxPlatformWindowHandleFromMetal(glfwGetCocoaWindow(app->window));
#endif
    return handle;
}

static float getCurrentTime(void)
{
#if defined(__ANDROID__) || TARGET_OS_IOS
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1000000000.0f;
#elif defined(__EMSCRIPTEN__)
    return (float)emscripten_get_now() / 1000.0f;
#else
    return (float)glfwGetTime();
#endif
}

// Helper function to load binary files (SPIR-V shaders)
static void* loadBinaryFile(ComputeApp* app, const char* filepath, size_t* outSize)
{
#if defined(__ANDROID__)
    if (!app || !app->androidApp || !app->androidApp->activity || !app->androidApp->activity->assetManager) {
        LOG_ERROR("AssetManager not available");
        return NULL;
    }

    AAsset* asset = AAssetManager_open(app->androidApp->activity->assetManager, filepath, AASSET_MODE_BUFFER);
    if (!asset) {
        LOG_ERROR("Failed to open asset: %s", filepath);
        return NULL;
    }

    size_t size = AAsset_getLength(asset);
    void* buffer = malloc(size);
    if (!buffer) {
        AAsset_close(asset);
        return NULL;
    }

    int bytesRead = AAsset_read(asset, buffer, size);
    AAsset_close(asset);

    if (bytesRead < 0 || (size_t)bytesRead != size) {
        free(buffer);
        return NULL;
    }

    if (outSize) {
        *outSize = size;
    }

    return buffer;
#else
    (void)app; // Unused on desktop
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filepath);
        return NULL;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (fileSize <= 0) {
        LOG_ERROR("Invalid file size for: %s", filepath);
        fclose(file);
        return NULL;
    }

    // Allocate buffer
    void* buffer = malloc(fileSize);
    if (!buffer) {
        LOG_ERROR("Failed to allocate memory for file: %s", filepath);
        fclose(file);
        return NULL;
    }

    // Read file
    size_t bytesRead = fread(buffer, 1, fileSize, file);
    fclose(file);

    if (bytesRead != (size_t)fileSize) {
        LOG_ERROR("Failed to read complete file: %s", filepath);
        free(buffer);
        return NULL;
    }

    *outSize = fileSize;
    return buffer;
#endif
}

// Helper function to load text files (WGSL shaders)
static void* loadTextFile(ComputeApp* app, const char* filepath, size_t* outSize)
{
#if defined(__ANDROID__)
    if (!app || !app->androidApp || !app->androidApp->activity || !app->androidApp->activity->assetManager) {
        LOG_ERROR("AssetManager not available for text file");
        return NULL;
    }

    AAsset* asset = AAssetManager_open(app->androidApp->activity->assetManager, filepath, AASSET_MODE_BUFFER);
    if (!asset) {
        LOG_ERROR("Failed to open asset: %s", filepath);
        return NULL;
    }

    size_t size = AAsset_getLength(asset);
    char* buffer = (char*)malloc(size + 1);
    if (!buffer) {
        AAsset_close(asset);
        return NULL;
    }

    int bytesRead = AAsset_read(asset, buffer, size);
    AAsset_close(asset);

    if (bytesRead < 0 || (size_t)bytesRead != size) {
        free(buffer);
        return NULL;
    }

    buffer[size] = '\0';

    if (outSize) {
        *outSize = size + 1;
    }

    return buffer;
#else
    (void)app; // Unused on desktop
    FILE* file = fopen(filepath, "r");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filepath);
        return NULL;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (fileSize <= 0) {
        LOG_ERROR("Invalid file size for: %s", filepath);
        fclose(file);
        return NULL;
    }

    // Allocate buffer with extra byte for null terminator
    char* buffer = (char*)malloc(fileSize + 1);
    if (!buffer) {
        LOG_ERROR("Failed to allocate memory for file: %s", filepath);
        fclose(file);
        return NULL;
    }

    // Read file
    size_t bytesRead = fread(buffer, 1, fileSize, file);
    fclose(file);

    if (bytesRead != (size_t)fileSize) {
        fprintf(stderr, "Failed to read complete file: %s\n", filepath);
        free(buffer);
        return NULL;
    }

    // Null-terminate for text files
    buffer[fileSize] = '\0';

    // Return size including null terminator for shader code
    *outSize = fileSize + 1;
    return buffer;
#endif
}

static bool init(ComputeApp* app)
{
    // Initialize in order of dependencies

    // 1. Create window
    if (!createWindow(app, WINDOW_WIDTH, WINDOW_HEIGHT)) {
        LOG_ERROR("Failed to create window");
        return false;
    }

    // 2. Create graphics context (instance, adapter, device, surface)
    if (!createGraphics(app)) {
        LOG_ERROR("Failed to create graphics");
        return false;
    }

    // 3. Create size-dependent resources (swapchain, framebuffers, render pass)
    if (!createSizeDependentResources(app, app->windowWidth, app->windowHeight)) {
        LOG_ERROR("Failed to create size-dependent resources");
        return false;
    }

    // 4. Create compute resources (textures, shaders, layouts, pipelines)
    if (!createComputeResources(app)) {
        LOG_ERROR("Failed to create compute resources");
        return false;
    }

    // 5. Create render resources (shaders, sampler, layouts, pipelines)
    if (!createRenderResources(app)) {
        LOG_ERROR("Failed to create render resources");
        return false;
    }

    // 6. Create per-frame resources (semaphores, fences, encoders, buffers, bind groups)
    if (!createPerFrameResources(app)) {
        LOG_ERROR("Failed to create per-frame resources");
        return false;
    }

    // Initialize loop state
    app->currentFrame = 0;
    app->previousWidth = app->windowWidth;
    app->previousHeight = app->windowHeight;
    app->elapsedTime = 0.0f;
    app->lastFrameTime = getCurrentTime();

    // Initialize FPS tracking
    app->fpsFrameCount = 0;
    app->fpsTimeAccumulator = 0.0f;
    app->fpsFrameTimeMin = FLT_MAX;
    app->fpsFrameTimeMax = 0.0f;

    LOG_INFO("Application initialized successfully!");
    return true;
}

static void cleanup(ComputeApp* app)
{
    // Wait for device to finish all GPU work before cleanup
    if (app->device) {
        gfxDeviceWaitIdle(app->device);
    }

    // 6. Destroy per-frame resources (bind groups, buffers, semaphores, fences, encoders)
    destroyPerFrameResources(app);

    // 5. Destroy render resources (pipelines, shaders, samplers)
    destroyRenderResources(app);

    // 4. Destroy compute resources (pipelines, shaders, textures)
    destroyComputeResources(app);

    // 3. Destroy size-dependent resources (swapchain, framebuffers, render pass)
    destroySizeDependentResources(app);

    // 2. Destroy graphics context (surface, device, instance)
    destroyGraphics(app);

    // 1. Destroy window
    destroyWindow(app);
}

static void update(ComputeApp* app, float deltaTime)
{
    updateFPS(app, deltaTime);
    app->elapsedTime += deltaTime;
}

static void render(ComputeApp* app)
{
    uint32_t frameIndex = app->currentFrame;
    PerFrameResources* frame = &app->frameResources[frameIndex];

    // Wait for previous frame
    gfxFenceWait(frame->inFlightFence, GFX_TIMEOUT_INFINITE);
    gfxFenceReset(frame->inFlightFence);

    // Acquire swapchain image
    uint32_t imageIndex = 0;
    GfxResult result = gfxSwapchainAcquireNextImage(
        app->swapchain,
        UINT64_MAX,
        frame->imageAvailableSemaphore,
        NULL,
        &imageIndex);

    if (result != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to acquire swapchain image");
        return;
    }

    // Validate framebuffer array and image index
    if (!app->framebuffers || imageIndex >= app->swapchainInfo.imageCount) {
        LOG_ERROR("Invalid framebuffer state: framebuffers=%p, imageIndex=%u, swapchain images=%u",
            (void*)app->framebuffers, imageIndex, app->swapchainInfo.imageCount);
        return;
    }

    if (!app->framebuffers[imageIndex]) {
        LOG_ERROR("Framebuffer at index %u is NULL", imageIndex);
        return;
    }

    // Update compute uniforms for current frame
    ComputeUniformData computeUniforms = {
        .time = app->elapsedTime
    };

    gfxQueueWriteBuffer(app->queue, frame->computeUniformBuffer, 0, &computeUniforms, sizeof(computeUniforms));

    // Update render uniforms for current frame
    RenderUniformData renderUniforms = {
        .postProcessStrength = 0.5f + 0.5f * sinf(app->elapsedTime * 0.5f) // Animate strength
    };
    gfxQueueWriteBuffer(app->queue, frame->renderUniformBuffer, 0, &renderUniforms, sizeof(renderUniforms));

    // Begin command encoder for reuse
    GfxCommandEncoder encoder = frame->commandEncoder;
    gfxCommandEncoderBegin(encoder);

    // Transition compute texture to GENERAL layout for compute shader write
    GfxTextureBarrier readToWriteBarrier = {
        .texture = app->computeTexture,
        .oldLayout = GFX_TEXTURE_LAYOUT_SHADER_READ_ONLY,
        .newLayout = GFX_TEXTURE_LAYOUT_GENERAL,
        .srcStageMask = GFX_PIPELINE_STAGE_FRAGMENT_SHADER,
        .dstStageMask = GFX_PIPELINE_STAGE_COMPUTE_SHADER,
        .srcAccessMask = GFX_ACCESS_SHADER_READ,
        .dstAccessMask = GFX_ACCESS_SHADER_WRITE,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };
    GfxPipelineBarrierDescriptor readToWriteBarrierDesc = {
        .sType = GFX_STRUCTURE_TYPE_PIPELINE_BARRIER_DESCRIPTOR,
        .pNext = NULL,
        .memoryBarriers = NULL,
        .memoryBarrierCount = 0,
        .bufferBarriers = NULL,
        .bufferBarrierCount = 0,
        .textureBarriers = &readToWriteBarrier,
        .textureBarrierCount = 1
    };

    gfxCommandEncoderPipelineBarrier(encoder, &readToWriteBarrierDesc);

    // --- COMPUTE PASS: Generate pattern ---
    GfxComputePassBeginDescriptor computePassDesc = {
        .label = "Generate Pattern"
    };
    GfxComputePassEncoder computePass = NULL;
    if (gfxCommandEncoderBeginComputePass(encoder, &computePassDesc, &computePass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to begin compute pass");
        return;
    }

    gfxComputePassEncoderSetPipeline(computePass, app->computePipeline);
    gfxComputePassEncoderSetBindGroup(computePass, 0, frame->computeBindGroup, NULL, 0);

    // Dispatch compute (16x16 local size, so divide by 16)
    // Uses fixed compute texture resolution, sampler will upscale/downscale to window
    uint32_t workGroupsX = (COMPUTE_TEXTURE_WIDTH + 15) / 16;
    uint32_t workGroupsY = (COMPUTE_TEXTURE_HEIGHT + 15) / 16;
    gfxComputePassEncoderDispatch(computePass, workGroupsX, workGroupsY, 1);

    if (gfxComputePassEncoderEnd(computePass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to end compute pass");
        return;
    }

    // Transition compute texture for shader read
    GfxTextureBarrier computeToReadBarrier = {
        .texture = app->computeTexture,
        .oldLayout = GFX_TEXTURE_LAYOUT_GENERAL,
        .newLayout = GFX_TEXTURE_LAYOUT_SHADER_READ_ONLY,
        .srcStageMask = GFX_PIPELINE_STAGE_COMPUTE_SHADER,
        .dstStageMask = GFX_PIPELINE_STAGE_FRAGMENT_SHADER,
        .srcAccessMask = GFX_ACCESS_SHADER_WRITE,
        .dstAccessMask = GFX_ACCESS_SHADER_READ,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };
    GfxPipelineBarrierDescriptor computeToReadBarrierDesc = {
        .sType = GFX_STRUCTURE_TYPE_PIPELINE_BARRIER_DESCRIPTOR,
        .pNext = NULL,
        .memoryBarriers = NULL,
        .memoryBarrierCount = 0,
        .bufferBarriers = NULL,
        .bufferBarrierCount = 0,
        .textureBarriers = &computeToReadBarrier,
        .textureBarrierCount = 1
    };

    gfxCommandEncoderPipelineBarrier(encoder, &computeToReadBarrierDesc);

    // --- RENDER PASS: Post-process and display ---
    GfxColor clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };

    GfxRenderPassBeginDescriptor renderPassBeginDesc = {
        .label = "Fullscreen Render Pass",
        .renderPass = app->renderPass,
        .framebuffer = app->framebuffers[imageIndex],
        .colorClearValues = &clearColor,
        .colorClearValueCount = 1,
        .depthClearValue = 0.0f,
        .stencilClearValue = 0
    };

    GfxRenderPassEncoder renderPass = NULL;
    if (gfxCommandEncoderBeginRenderPass(encoder, &renderPassBeginDesc, &renderPass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to begin render pass");
        return;
    }

    gfxRenderPassEncoderSetPipeline(renderPass, app->renderPipeline);
    gfxRenderPassEncoderSetBindGroup(renderPass, 0, frame->renderBindGroup, NULL, 0);

    // Set viewport and scissor to match the actual swapchain extent
    GfxViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)app->swapchainInfo.extent.width,
        .height = (float)app->swapchainInfo.extent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };
    gfxRenderPassEncoderSetViewport(renderPass, &viewport);

    GfxScissorRect scissor = {
        .origin = { 0, 0 },
        .extent = { app->swapchainInfo.extent.width, app->swapchainInfo.extent.height }
    };
    gfxRenderPassEncoderSetScissorRect(renderPass, &scissor);

    // Draw fullscreen quad (6 vertices, no buffers needed)
    gfxRenderPassEncoderDraw(renderPass, 6, 1, 0, 0);

    if (gfxRenderPassEncoderEnd(renderPass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to end render pass");
        return;
    }

    if (gfxCommandEncoderEnd(encoder) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to end command encoder");
        return;
    }

    // Submit
    GfxSubmitDescriptor submitDescriptor = {
        .sType = GFX_STRUCTURE_TYPE_SUBMIT_DESCRIPTOR,
        .pNext = NULL,
        .commandEncoderCount = 1,
        .commandEncoders = &encoder,
        .waitSemaphoreCount = 1,
        .waitSemaphores = &frame->imageAvailableSemaphore,
        .signalSemaphoreCount = 1,
        .signalSemaphores = &app->renderFinishedSemaphores[imageIndex],
        .signalFence = frame->inFlightFence
    };

    if (gfxQueueSubmit(app->queue, &submitDescriptor) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to submit command buffer");
        return;
    }

    // Present
    GfxPresentDescriptor presentDescriptor = {
        .sType = GFX_STRUCTURE_TYPE_PRESENT_DESCRIPTOR,
        .pNext = NULL,
        .waitSemaphoreCount = 1,
        .waitSemaphores = &app->renderFinishedSemaphores[imageIndex]
    };

    result = gfxSwapchainPresent(app->swapchain, &presentDescriptor);
    if (result != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to present swapchain image");
        return;
    }

    app->currentFrame = (app->currentFrame + 1) % app->swapchainInfo.imageCount;
}

// Handle window resize by recreating size-dependent resources
static bool handleResize(ComputeApp* app, uint32_t width, uint32_t height)
{
    LOG_INFO("Resizing to %ux%u", width, height);

    app->windowWidth = width;
    app->windowHeight = height;

    // Wait for all in-flight frames to complete
    gfxDeviceWaitIdle(app->device);

    // Destroy per-frame resources first (they depend on swapchain image count)
    destroyPerFrameResources(app);

    // Recreate size-dependent resources (including swapchain)
    destroySizeDependentResources(app);
    if (!createSizeDependentResources(app, width, height)) {
        LOG_ERROR("Failed to recreate size-dependent resources after resize");
        return false;
    }

    // Recreate per-frame resources with new swapchain image count
    if (!createPerFrameResources(app)) {
        LOG_ERROR("Failed to recreate per-frame resources after resize");
        return false;
    }

    // Reset frame index to prevent out-of-bounds access
    app->currentFrame = 0;

    app->previousWidth = app->windowWidth;
    app->previousHeight = app->windowHeight;

    LOG_INFO("Successfully recreated resources for new size");
    return true;
}

// Track FPS and log statistics every second
static void updateFPS(ComputeApp* app, float deltaTime)
{
    app->fpsFrameCount++;
    app->fpsTimeAccumulator += deltaTime;

    if (deltaTime < app->fpsFrameTimeMin) {
        app->fpsFrameTimeMin = deltaTime;
    }
    if (deltaTime > app->fpsFrameTimeMax) {
        app->fpsFrameTimeMax = deltaTime;
    }

    // Log FPS every second
    if (app->fpsTimeAccumulator >= 1.0f) {
        float avgFPS = (float)app->fpsFrameCount / app->fpsTimeAccumulator;
        float avgFrameTime = (app->fpsTimeAccumulator / (float)app->fpsFrameCount) * 1000.0f;
        float minFPS = 1.0f / app->fpsFrameTimeMax;
        float maxFPS = 1.0f / app->fpsFrameTimeMin;
        LOG_INFO("FPS - Avg: %.1f, Min: %.1f, Max: %.1f | Frame Time - Avg: %.2f ms, Min: %.2f ms, Max: %.2f ms",
            avgFPS, minFPS, maxFPS,
            avgFrameTime, app->fpsFrameTimeMin * 1000.0f, app->fpsFrameTimeMax * 1000.0f);

        // Reset for next second
        app->fpsFrameCount = 0;
        app->fpsTimeAccumulator = 0.0f;
        app->fpsFrameTimeMin = FLT_MAX;
        app->fpsFrameTimeMax = 0.0f;
    }
}

#ifdef __ANDROID__
// ============================================================================
// Android Platform Implementation
// ============================================================================

// Android-specific event handlers
static void handleAppCommand(struct android_app* state, int32_t cmd)
{
    ComputeApp* app = (ComputeApp*)state->userData;

    switch (cmd) {
    case APP_CMD_INIT_WINDOW:
        if (state->window != NULL) {
            app->windowWidth = ANativeWindow_getWidth(state->window);
            app->windowHeight = ANativeWindow_getHeight(state->window);
            LOG_INFO("Window initialized: %dx%d", app->windowWidth, app->windowHeight);

            if (!app->instance) {
                // First time init
                if (init(app)) {
                    app->animating = true;
                    LOG_INFO("Application initialized successfully!");
                } else {
                    LOG_ERROR("Failed to initialize graphics");
                }
            }
        }
        break;

    case APP_CMD_TERM_WINDOW:
        LOG_INFO("Window terminating");
        app->animating = false;
        cleanup(app);
        break;

    case APP_CMD_GAINED_FOCUS:
        LOG_INFO("Gained focus");
        if (app->instance) {
            app->animating = true;
        }
        break;

    case APP_CMD_LOST_FOCUS:
        LOG_INFO("Lost focus");
        app->animating = false;
        break;

    case APP_CMD_PAUSE:
        LOG_INFO("Paused");
        app->animating = false;
        break;

    case APP_CMD_RESUME:
        LOG_INFO("Resumed");
        if (app->instance) {
            app->animating = true;
        }
        break;

    case APP_CMD_WINDOW_RESIZED:
        if (state->window != NULL && app->instance) {
            uint32_t newWidth = ANativeWindow_getWidth(state->window);
            uint32_t newHeight = ANativeWindow_getHeight(state->window);
            if (!handleResize(app, newWidth, newHeight)) {
                LOG_ERROR("Failed to handle window resize");
            }
        }
        break;
    }
}

static int32_t handleInput(struct android_app* state, AInputEvent* event)
{
    (void)state;
    (void)event;
    return 0;
}

// Returns false if loop should exit
static bool mainLoopIteration(ComputeApp* app, struct android_app* state)
{
    int events;
    struct android_poll_source* source;

    // Poll all events
    while (ALooper_pollOnce(app->animating ? 0 : -1, NULL, &events, (void**)&source) >= 0) {
        if (source != NULL) {
            source->process(state, source);
        }

        if (state->destroyRequested != 0) {
            LOG_INFO("Destroy requested");
            cleanup(app);
            return false;
        }
    }

    if (app->animating && app->instance) {
        float currentTime = getCurrentTime();
        float deltaTime = currentTime - app->lastFrameTime;
        app->lastFrameTime = currentTime;

        update(app, deltaTime);
        render(app);
    }

    return true;
}

void android_main(struct android_app* state)
{
    ComputeApp app = { 0 };
    app.androidApp = state;
    app.settings.backend = GFX_BACKEND_VULKAN;
    app.settings.vsync = true;
    app.animating = false;

    state->userData = &app;
    state->onAppCmd = handleAppCommand;
    state->onInputEvent = handleInput;

    LOG_INFO("=== GFX Compute Example (Android) ===");

    // Main loop
    while (mainLoopIteration(&app, state)) {
        // Loop continues until mainLoopIteration returns false
    }
}

#elif TARGET_OS_IOS
// ============================================================================
// iOS Platform Implementation
// ============================================================================

// C bridge functions called from Objective-C ViewController

void* computeAppCreate(void* metalLayer, int width, int height)
{
    if (!metalLayer) {
        LOG_ERROR("Metal layer is NULL");
        return NULL;
    }

    LOG_INFO("Creating compute app with size %dx%d", width, height);

    ComputeApp* app = (ComputeApp*)calloc(1, sizeof(ComputeApp));
    if (!app) {
        LOG_ERROR("Failed to allocate compute app");
        return NULL;
    }

    app->metalLayer = metalLayer;

    // Set up app settings
    app->settings.backend = GFX_BACKEND_VULKAN; // iOS uses Metal backend
    app->settings.vsync = true;
    app->windowWidth = width;
    app->windowHeight = height;

    LOG_INFO("=== GFX Compute Example (iOS) ===");

    // Initialize the compute app
    if (!init(app)) {
        LOG_ERROR("Failed to initialize compute app");
        free(app);
        return NULL;
    }

    LOG_INFO("Compute app created successfully");
    return app;
}

void computeAppDestroy(void* appPtr)
{
    if (!appPtr) {
        return;
    }

    ComputeApp* app = (ComputeApp*)appPtr;
    cleanup(app);
    free(app);
    LOG_INFO("Compute app destroyed");
}

void computeAppResize(void* appPtr, int width, int height)
{
    if (!appPtr) {
        return;
    }

    ComputeApp* app = (ComputeApp*)appPtr;
    if (!handleResize(app, (uint32_t)width, (uint32_t)height)) {
        LOG_ERROR("Failed to handle resize");
    }
}

void computeAppUpdate(void* appPtr, float deltaTime)
{
    if (!appPtr) {
        return;
    }

    ComputeApp* app = (ComputeApp*)appPtr;
    update(app, deltaTime);
}

void computeAppRender(void* appPtr)
{
    if (!appPtr) {
        return;
    }

    ComputeApp* app = (ComputeApp*)appPtr;
    render(app);
}

#else
// ============================================================================
// Desktop/Web Platform Implementation
// ============================================================================

// Returns false if loop should exit
static bool mainLoopIteration(ComputeApp* app)
{
    if (!app || glfwWindowShouldClose(app->window)) {
        return false;
    }

    glfwPollEvents();

    // Handle framebuffer resize
    if (app->previousWidth != app->windowWidth || app->previousHeight != app->windowHeight) {
        if (!handleResize(app, app->windowWidth, app->windowHeight)) {
            return false;
        }
        return true; // Skip rendering this frame
    }

    // Calculate delta time
    float currentTime = getCurrentTime();
    float deltaTime = currentTime - app->lastFrameTime;
    app->lastFrameTime = currentTime;

    update(app, deltaTime);
    render(app);

    return true;
}

#if defined(__EMSCRIPTEN__)
static void emscriptenMainLoop(void* userData)
{
    ComputeApp* app = (ComputeApp*)userData;
    if (!mainLoopIteration(app)) {
        emscripten_cancel_main_loop();
        cleanup(app);
    }
}
#endif

static bool parseBackend(const char* backendStr, GfxBackend* outBackend)
{
    if (strcmp(backendStr, "vulkan") == 0) {
        *outBackend = GFX_BACKEND_VULKAN;
        return true;
    } else if (strcmp(backendStr, "webgpu") == 0) {
        *outBackend = GFX_BACKEND_WEBGPU;
        return true;
    } else {
        LOG_ERROR("Unknown backend: %s", backendStr);
        LOG_ERROR("Valid values: vulkan, webgpu");
        return false;
    }
}

static bool parseVsync(const char* vsyncStr, bool* outVsync)
{
    int vsync = atoi(vsyncStr);
    if (vsync == 0) {
        *outVsync = false;
        return true;
    } else if (vsync == 1) {
        *outVsync = true;
        return true;
    } else {
        LOG_ERROR("Invalid vsync value: %s", vsyncStr);
        LOG_ERROR("Valid values: 0 (off), 1 (on)");
        return false;
    }
}

static void printHelp(const char* programName)
{
    LOG_INFO("Usage: %s [options]", programName);
    LOG_INFO("Options:");
    LOG_INFO("  --backend [vulkan|webgpu]   Select graphics backend");
    LOG_INFO("  --vsync [0|1]               VSync: 0=off, 1=on");
    LOG_INFO("  --help                      Show this help message");
}

static bool parseArguments(int argc, char** argv, Settings* settings)
{
#if defined(__EMSCRIPTEN__)
    settings->backend = GFX_BACKEND_WEBGPU;
#else
    settings->backend = GFX_BACKEND_VULKAN;
#endif
    settings->vsync = true; // VSync on by default

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            i++;
            if (!parseBackend(argv[i], &settings->backend)) {
                return false;
            }
        } else if (strcmp(argv[i], "--vsync") == 0 && i + 1 < argc) {
            i++;
            if (!parseVsync(argv[i], &settings->vsync)) {
                return false;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            printHelp(argv[0]);
            return false;
        } else {
            LOG_ERROR("Unknown argument: %s", argv[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    LOG_INFO("=== Compute & Postprocess Example (C) ===");
    LOG_INFO("");

    ComputeApp app = { 0 }; // Initialize all members to NULL/0

    // Parse command line arguments
    if (!parseArguments(argc, argv, &app.settings)) {
        printHelp(argv[0]);
        return 0;
    }

    // Initialize all resources and state
    if (!init(&app)) {
        cleanup(&app);
        return -1;
    }

    LOG_INFO("Press ESC to exit");
    LOG_INFO("");

    // Run main loop (platform-specific)
#if defined(__EMSCRIPTEN__)
    // Note: emscripten_set_main_loop_arg returns immediately and never blocks
    // Cleanup happens in emscriptenMainLoop when the loop exits
    // Execution continues in the browser event loop
    emscripten_set_main_loop_arg(emscriptenMainLoop, &app, 0, 1);
#else
    while (mainLoopIteration(&app)) {
        // Loop continues until mainLoopIteration returns false
    }

    LOG_INFO("");
    LOG_INFO("Cleaning up resources...");
    cleanup(&app);
    LOG_INFO("Example completed successfully!");
#endif

    return 0;
}

#endif // __ANDROID__ / TARGET_OS_IOS
