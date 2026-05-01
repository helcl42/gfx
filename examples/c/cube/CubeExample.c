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
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, "GFX_CUBE", __VA_ARGS__)
#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, "GFX_CUBE", __VA_ARGS__)
#define LOG_WARN(...) __android_log_print(ANDROID_LOG_WARN, "GFX_CUBE", __VA_ARGS__)
#define LOG_DEBUG(...) __android_log_print(ANDROID_LOG_DEBUG, "GFX_CUBE", __VA_ARGS__)
#else
// Desktop/Web logging macros (map to printf)
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
#if defined(_WIN32)
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

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define CUBE_COUNT 3
#define COLOR_FORMAT GFX_FORMAT_B8G8R8A8_UNORM_SRGB
#define DEPTH_FORMAT GFX_FORMAT_DEPTH32_FLOAT

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

// Math types for improved API clarity and type safety
typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    float m[16]; // Column-major 4x4 matrix
} Mat4;

// Vertex structure for cube
typedef struct {
    Vec3 position;
    Vec3 color;
} Vertex;

// Uniform buffer structure for transformations
typedef struct {
    Mat4 model; // Model matrix
    Mat4 view; // View matrix
    Mat4 projection; // Projection matrix
} UniformData;

// Application settings/configuration
typedef struct {
    GfxBackend backend; // Graphics backend selection
    GfxSampleCount msaaSampleCount; // MSAA sample count
    bool vsync; // VSync enabled (FIFO) or disabled (IMMEDIATE)
} Settings;

// Per-frame-in-flight resources
typedef struct {
    GfxCommandEncoder commandEncoder;
    GfxSemaphore imageAvailableSemaphore;
    GfxFence inFlightFence;
    GfxBindGroup uniformBindGroups[CUBE_COUNT]; // One per cube
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
    GfxSurfaceInfo surfaceInfo;
    GfxSwapchain swapchain;
    GfxSwapchainInfo swapchainInfo;

    GfxBuffer vertexBuffer;
    GfxBuffer indexBuffer;
    GfxBufferInfo vertexBufferInfo;
    GfxBufferInfo indexBufferInfo;
    GfxShader vertexShader;
    GfxShader fragmentShader;
    GfxRenderPass renderPass;
    GfxRenderPipeline renderPipeline;
    GfxBindGroupLayout uniformBindGroupLayout;
    GfxBindGroupLayout textureBindGroupLayout;
    GfxBindGroup textureBindGroup;

    // Depth buffer
    GfxTexture depthTexture;
    GfxTextureView depthTextureView;

    // MSAA color buffer
    GfxTexture msaaColorTexture;
    GfxTextureView msaaColorTextureView;

    // Per-frame resources (one per swapchain image)
    PerFrameResources* frameResources;
    GfxFramebuffer* framebuffers;
    GfxSemaphore* renderFinishedSemaphores;
    uint32_t currentFrame;

    uint32_t windowWidth;
    uint32_t windowHeight;
    uint32_t previousWidth;
    uint32_t previousHeight;

    // Shared resources (not per-frame)
    GfxBuffer sharedUniformBuffer;
    size_t uniformAlignedSize; // Aligned size per frame
    GfxTexture cubeTexture;
    GfxTextureView cubeTextureView;
    GfxSampler cubeSampler;

    // Async texture upload resources
    GfxBuffer textureStagingBuffer;
    GfxCommandEncoder textureUploadEncoder;
    GfxFence textureUploadFence;
    bool textureUploadComplete;

    // Loop state
    float elapsedTime;
    float lastFrameTime;
    float rotationAngleX;
    float rotationAngleY;

    // FPS tracking
    uint32_t fpsFrameCount;
    float fpsTimeAccumulator;
    float fpsFrameTimeMin;
    float fpsFrameTimeMax;

    // Application settings
    Settings settings;
} CubeApp;

// Private function declarations
static bool createWindow(CubeApp* app, uint32_t width, uint32_t height);
static void destroyWindow(CubeApp* app);
static bool createGraphics(CubeApp* app);
static void destroyGraphics(CubeApp* app);
static bool createPerFrameResources(CubeApp* app);
static void destroyPerFrameResources(CubeApp* app);
static bool createSizeDependentResources(CubeApp* app, uint32_t width, uint32_t height);
static void destroySizeDependentResources(CubeApp* app);
static bool createRenderPass(CubeApp* app);
static void destroyRenderPass(CubeApp* app);
static bool createSwapchain(CubeApp* app, uint32_t width, uint32_t height);
static void destroySwapchain(CubeApp* app);
static bool createRenderTargetTextures(CubeApp* app, uint32_t width, uint32_t height);
static void destroyRenderTargetTextures(CubeApp* app);
static bool createFrameBuffers(CubeApp* app, uint32_t width, uint32_t height);
static void destroyFrameBuffers(CubeApp* app);
static bool createGeometry(CubeApp* app);
static void destroyGeometry(CubeApp* app);
static bool createUniformBuffer(CubeApp* app);
static void destroyUniformBuffer(CubeApp* app);
static bool createBindGroup(CubeApp* app);
static void destroyBindGroup(CubeApp* app);
static bool createShaders(CubeApp* app);
static void destroyShaders(CubeApp* app);
static bool loadTexture(CubeApp* app);
static void unloadTexture(CubeApp* app);
static bool createRenderingResources(CubeApp* app);
static void destroyRenderingResources(CubeApp* app);
static bool createRenderPipeline(CubeApp* app);
static void destroyRenderPipeline(CubeApp* app);

static void updateCube(CubeApp* app, int cubeIndex);
static GfxPlatformWindowHandle getPlatformWindowHandle(CubeApp* app);
static float getCurrentTime(void);
static void* loadBinaryFile(CubeApp* app, const char* filepath, size_t* outSize);
static void* loadTextFile(CubeApp* app, const char* filepath, size_t* outSize);

// Matrix/Vector math function declarations
static void matrixIdentity(Mat4* matrix);
static void matrixMultiply(Mat4* result, const Mat4* a, const Mat4* b);
static void matrixRotateX(Mat4* matrix, float angle);
static void matrixRotateY(Mat4* matrix, float angle);
static void matrixRotateZ(Mat4* matrix, float angle);
static void matrixPerspective(Mat4* matrix, float fov, float aspect, float nearPlane, float farPlane, GfxBackend backend);
static void matrixLookAt(Mat4* matrix, const Vec3* eye, const Vec3* center, const Vec3* up);
static bool vectorNormalize(Vec3* v);

// The public functions called from main/android_main
static bool init(CubeApp* app);
static void cleanup(CubeApp* app);
static void update(CubeApp* app, float deltaTime);
static void render(CubeApp* app);

#if defined(__ANDROID__)
// Android-specific callbacks
static void handleAppCommand(struct android_app* app, int32_t cmd);
static int32_t handleInput(struct android_app* app, AInputEvent* event);
#elif TARGET_OS_IOS
// iOS doesn't use GLFW callbacks - UIKit handles events
#else
// GLFW callbacks
static void errorCallback(int error, const char* description)
{
    LOG_ERROR("GLFW Error %d: %s", error, description);
}

static void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    CubeApp* app = (CubeApp*)glfwGetWindowUserPointer(window);
    if (app) {
        app->windowWidth = (uint32_t)width;
        app->windowHeight = (uint32_t)height;
    }
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}
#endif // __ANDROID__ / TARGET_OS_IOS / desktop

static bool createWindow(CubeApp* app, uint32_t width, uint32_t height)
{
#if defined(__ANDROID__) || TARGET_OS_IOS
    // On mobile platforms, window/layer is managed by the system
    // On Android, window is managed by the system
    app->windowWidth = width;
    app->windowHeight = height;
    return true;
#else
    // Desktop: GLFW
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        LOG_ERROR("Failed to initialize GLFW");
        return false;
    }

    // Don't create OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    const char* backendName = (app->settings.backend == GFX_BACKEND_VULKAN) ? "Vulkan" : "WebGPU";
    char windowTitle[128];
    snprintf(windowTitle, sizeof(windowTitle), "Cube Example - %s", backendName);

    app->window = glfwCreateWindow(width, height, windowTitle, NULL, NULL);

    if (!app->window) {
        LOG_ERROR("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }

    app->windowWidth = width;
    app->windowHeight = height;

    glfwSetWindowUserPointer(app->window, app);
    glfwSetFramebufferSizeCallback(app->window, framebufferSizeCallback);
    glfwSetKeyCallback(app->window, keyCallback);

    return true;
#endif
}

static void destroyWindow(CubeApp* app)
{
#if defined(__ANDROID__) || TARGET_OS_IOS
    (void)app;
#else
    if (app->window) {
        glfwDestroyWindow(app->window);
        app->window = NULL;
    }
    glfwTerminate();
#endif
}

static bool createGraphics(CubeApp* app)
{
    gfxSetLogCallback(logCallback, NULL);

    // Load the graphics backend BEFORE creating an instance
    // This is now decoupled - you load the backend API once at startup
    const char* backendName = (app->settings.backend == GFX_BACKEND_VULKAN) ? "Vulkan" : "WebGPU";
    LOG_INFO("Loading graphics backend (%s)...", backendName);
    if (gfxLoadBackend(app->settings.backend) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to load graphics backend");
        return false;
    }
    LOG_INFO("Graphics backend loaded successfully!");

    // Create graphics instance
    const char* instanceExtensions[] = { GFX_INSTANCE_EXTENSION_SURFACE, GFX_INSTANCE_EXTENSION_DEBUG };
    GfxInstanceDescriptor instanceDesc = {
        .sType = GFX_STRUCTURE_TYPE_INSTANCE_DESCRIPTOR,
        .pNext = NULL,
        .backend = app->settings.backend,
        .applicationName = "Cube Example (C)",
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
        .adapterIndex = UINT32_MAX, // Use preference-based selection
        .preference = GFX_ADAPTER_PREFERENCE_HIGH_PERFORMANCE
    };

    if (gfxInstanceRequestAdapter(app->instance, &adapterDesc, &app->adapter) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get graphics adapter");
        return false;
    }

    // Query and store adapter info
    gfxAdapterGetInfo(app->adapter, &app->adapterInfo);
    LOG_INFO("Using adapter: %s", app->adapterInfo.name);
    LOG_INFO("  Vendor ID: 0x%04X, Device ID: 0x%04X", app->adapterInfo.vendorID, app->adapterInfo.deviceID);
    LOG_INFO("  Type: %s",
        app->adapterInfo.adapterType == GFX_ADAPTER_TYPE_DISCRETE_GPU ? "Discrete GPU" : app->adapterInfo.adapterType == GFX_ADAPTER_TYPE_INTEGRATED_GPU ? "Integrated GPU"
            : app->adapterInfo.adapterType == GFX_ADAPTER_TYPE_CPU                                                                                       ? "CPU"
                                                                                                                                                         : "Unknown");
    LOG_INFO("  Backend: %s", app->adapterInfo.backend == GFX_BACKEND_VULKAN ? "Vulkan" : "WebGPU");

    // Create device
    const char* deviceExtensions[] = {
        GFX_DEVICE_EXTENSION_SWAPCHAIN,
        GFX_DEVICE_EXTENSION_ANISOTROPIC_FILTERING
    };
    GfxDeviceDescriptor deviceDesc = {
        .sType = GFX_STRUCTURE_TYPE_DEVICE_DESCRIPTOR,
        .pNext = NULL,
        .label = "Main Device",
        .enabledExtensions = deviceExtensions,
        .enabledExtensionCount = ARRAY_SIZE(deviceExtensions)
    };

    if (gfxAdapterCreateDevice(app->adapter, &deviceDesc, &app->device) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create device");
        return false;
    }

    // Query and print device limits
    GfxDeviceLimits limits;
    if (gfxDeviceGetLimits(app->device, &limits) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get device limits");
        return false;
    }
    LOG_INFO("Device Limits:");
    LOG_INFO("  Min Uniform Buffer Offset Alignment: %u bytes", limits.minUniformBufferOffsetAlignment);
    LOG_INFO("  Min Storage Buffer Offset Alignment: %u bytes", limits.minStorageBufferOffsetAlignment);
    LOG_INFO("  Max Uniform Buffer Binding Size: %u bytes", limits.maxUniformBufferBindingSize);
    LOG_INFO("  Max Storage Buffer Binding Size: %u bytes", limits.maxStorageBufferBindingSize);
    LOG_INFO("  Max Buffer Size: %llu bytes", (unsigned long long)limits.maxBufferSize);
    LOG_INFO("  Max Texture Dimension 1D: %u", limits.maxTextureDimension1D);
    LOG_INFO("  Max Texture Dimension 2D: %u", limits.maxTextureDimension2D);
    LOG_INFO("  Max Texture Dimension 3D: %u", limits.maxTextureDimension3D);
    LOG_INFO("  Max Texture Array Layers: %u", limits.maxTextureArrayLayers);

    if (gfxDeviceGetQueue(app->device, &app->queue) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get device queue");
        return false;
    }

    // Create surface
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

    // Query surface capabilities and info
    if (gfxSurfaceGetInfo(app->surface, app->adapter, &app->surfaceInfo) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get surface info");
        return false;
    }
    LOG_INFO("Surface Info:");
    LOG_INFO("  Image Count: min %u, max %u", app->surfaceInfo.minImageCount, app->surfaceInfo.maxImageCount);
    LOG_INFO("  Extent: min (%u, %u), max (%u, %u)",
        app->surfaceInfo.minExtent.width, app->surfaceInfo.minExtent.height,
        app->surfaceInfo.maxExtent.width, app->surfaceInfo.maxExtent.height);

    return true;
}

static void destroyGraphics(CubeApp* app)
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

    // Unload the backend API after destroying all instances
    LOG_INFO("Unloading graphics backend...");
    gfxUnloadBackend(app->settings.backend);
}

static bool createPerFrameResources(CubeApp* app)
{
    // Allocate per-frame resources array
    app->frameResources = (PerFrameResources*)calloc(app->swapchainInfo.imageCount, sizeof(PerFrameResources));
    if (!app->frameResources) {
        LOG_ERROR("Failed to allocate per-frame resources array");
        return false;
    }

    // Create synchronization objects and command encoders for each frame in flight
    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        char label[64];
        PerFrameResources* frame = &app->frameResources[i];

        // Create semaphores
        snprintf(label, sizeof(label), "Image Available Semaphore %u", i);
        GfxSemaphoreDescriptor semaphoreDesc = {
            .sType = GFX_STRUCTURE_TYPE_SEMAPHORE_DESCRIPTOR,
            .pNext = NULL,
            .label = label,
            .type = GFX_SEMAPHORE_TYPE_BINARY,
            .initialValue = 0
        };

        if (gfxDeviceCreateSemaphore(app->device, &semaphoreDesc, &frame->imageAvailableSemaphore) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create image available semaphore %u", i);
            return false;
        }

        // Create fence
        snprintf(label, sizeof(label), "In Flight Fence %u", i);
        GfxFenceDescriptor fenceDesc = {
            .sType = GFX_STRUCTURE_TYPE_FENCE_DESCRIPTOR,
            .pNext = NULL,
            .label = label,
            .signaled = true // Start signaled so first frame doesn't wait
        };

        if (gfxDeviceCreateFence(app->device, &fenceDesc, &frame->inFlightFence) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create in flight fence %u", i);
            return false;
        }

        // Create command encoder
        snprintf(label, sizeof(label), "Command Encoder %u", i);
        GfxCommandEncoderDescriptor encoderDesc = {
            .sType = GFX_STRUCTURE_TYPE_COMMAND_ENCODER_DESCRIPTOR,
            .pNext = NULL,
            .label = label
        };
        if (gfxDeviceCreateCommandEncoder(app->device, &encoderDesc, &frame->commandEncoder) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create command encoder %u", i);
            return false;
        }

        // Create bind groups for each cube in this frame
        for (int cubeIdx = 0; cubeIdx < CUBE_COUNT; ++cubeIdx) {
            snprintf(label, sizeof(label), "Uniform Bind Group (Frame %u, Cube %d)", i, cubeIdx);

            size_t offset = (i * CUBE_COUNT + cubeIdx) * app->uniformAlignedSize;

            GfxBindGroupEntry entry = {
                .binding = 0,
                .type = GFX_BIND_GROUP_ENTRY_TYPE_BUFFER,
                .resource = {
                    .buffer = {
                        .buffer = app->sharedUniformBuffer,
                        .offset = offset,
                        .size = sizeof(UniformData) } }
            };

            GfxBindGroupDescriptor bindGroupDesc = {
                .sType = GFX_STRUCTURE_TYPE_BIND_GROUP_DESCRIPTOR,
                .pNext = NULL,
                .label = label,
                .layout = app->uniformBindGroupLayout,
                .entries = &entry,
                .entryCount = 1
            };

            if (gfxDeviceCreateBindGroup(app->device, &bindGroupDesc, &frame->uniformBindGroups[cubeIdx]) != GFX_RESULT_SUCCESS) {
                LOG_ERROR("Failed to create bind group for frame %u, cube %d", i, cubeIdx);
                return false;
            }
        }
    }

    app->currentFrame = 0;
    return true;
}

static void destroyPerFrameResources(CubeApp* app)
{
    if (!app->frameResources) {
        return;
    }

    // Wait for all in-flight frames to complete before destroying
    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        PerFrameResources* frame = &app->frameResources[i];
        if (frame->inFlightFence) {
            gfxFenceWait(frame->inFlightFence, GFX_TIMEOUT_INFINITE);
        }
    }

    // Destroy per-frame resources
    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        PerFrameResources* frame = &app->frameResources[i];

        // Destroy bind groups
        for (int cubeIdx = 0; cubeIdx < CUBE_COUNT; ++cubeIdx) {
            if (frame->uniformBindGroups[cubeIdx]) {
                gfxBindGroupDestroy(frame->uniformBindGroups[cubeIdx]);
                frame->uniformBindGroups[cubeIdx] = NULL;
            }
        }

        // Destroy synchronization objects
        if (frame->imageAvailableSemaphore) {
            gfxSemaphoreDestroy(frame->imageAvailableSemaphore);
            frame->imageAvailableSemaphore = NULL;
        }
        if (frame->inFlightFence) {
            gfxFenceDestroy(frame->inFlightFence);
            frame->inFlightFence = NULL;
        }

        // Destroy command encoder
        if (frame->commandEncoder) {
            gfxCommandEncoderDestroy(frame->commandEncoder);
            frame->commandEncoder = NULL;
        }
    }

    // Free the per-frame resources array
    if (app->frameResources) {
        free(app->frameResources);
        app->frameResources = NULL;
    }
}

static bool createSizeDependentResources(CubeApp* app, uint32_t width, uint32_t height)
{
    if (!createSwapchain(app, width, height)) {
        return false;
    }

    uint32_t actualWidth = app->swapchainInfo.extent.width;
    uint32_t actualHeight = app->swapchainInfo.extent.height;

    if (!createRenderTargetTextures(app, actualWidth, actualHeight)) {
        return false;
    }

    // Create render pass AFTER swapchain so we have the format
    if (!createRenderPass(app)) {
        return false;
    }

    if (!createFrameBuffers(app, actualWidth, actualHeight)) {
        return false;
    }

    return true;
}

static void destroySizeDependentResources(CubeApp* app)
{
    // Destroy framebuffers
    destroyFrameBuffers(app);

    // Destroy render pass (depends on swapchain format)
    destroyRenderPass(app);

    // Destroy render target textures (depth and MSAA)
    destroyRenderTargetTextures(app);

    // Destroy swapchain
    destroySwapchain(app);
}

static bool createRenderPass(CubeApp* app)
{
    // Create render pass (persistent, reusable across frames)
    // Define color attachment target with MSAA
    GfxRenderPassColorAttachmentTarget colorTarget = {
        .format = app->swapchainInfo.format,
        .sampleCount = app->settings.msaaSampleCount,
        .ops = {
            .loadOp = GFX_LOAD_OP_CLEAR,
            .storeOp = (app->settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? GFX_STORE_OP_DONT_CARE : GFX_STORE_OP_STORE },
        .finalLayout = (app->settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? GFX_TEXTURE_LAYOUT_COLOR_ATTACHMENT : GFX_TEXTURE_LAYOUT_PRESENT_SRC
    };

    // Define resolve target (for MSAA -> non-MSAA resolve)
    GfxRenderPassColorAttachmentTarget resolveTarget = {
        .format = app->swapchainInfo.format,
        .sampleCount = GFX_SAMPLE_COUNT_1,
        .ops = {
            .loadOp = GFX_LOAD_OP_DONT_CARE,
            .storeOp = GFX_STORE_OP_STORE },
        .finalLayout = GFX_TEXTURE_LAYOUT_PRESENT_SRC
    };

    // Bundle color attachment with optional resolve
    GfxRenderPassColorAttachment colorAttachment = {
        .target = colorTarget,
        .resolveTarget = (app->settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? &resolveTarget : NULL
    };

    // Define depth/stencil attachment target
    GfxRenderPassDepthStencilAttachmentTarget depthTarget = {
        .format = DEPTH_FORMAT,
        .sampleCount = app->settings.msaaSampleCount,
        .depthOps = {
            .loadOp = GFX_LOAD_OP_CLEAR,
            .storeOp = GFX_STORE_OP_DONT_CARE },
        .stencilOps = { .loadOp = GFX_LOAD_OP_DONT_CARE, .storeOp = GFX_STORE_OP_DONT_CARE },
        .finalLayout = GFX_TEXTURE_LAYOUT_DEPTH_STENCIL_ATTACHMENT
    };

    GfxRenderPassDepthStencilAttachment depthAttachment = {
        .target = depthTarget,
        .resolveTarget = NULL
    };

    GfxRenderPassDescriptor renderPassDesc = {
        .sType = GFX_STRUCTURE_TYPE_RENDER_PASS_DESCRIPTOR,
        .pNext = NULL,
        .label = "Main Render Pass",
        .colorAttachments = &colorAttachment,
        .colorAttachmentCount = 1,
        .depthStencilAttachment = &depthAttachment
    };

    if (gfxDeviceCreateRenderPass(app->device, &renderPassDesc, &app->renderPass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create render pass");
        return false;
    }

    return true;
}

static void destroyRenderPass(CubeApp* app)
{
    if (app->renderPass) {
        gfxRenderPassDestroy(app->renderPass);
        app->renderPass = NULL;
    }
}

static bool createSwapchain(CubeApp* app, uint32_t width, uint32_t height)
{
    LOG_INFO("Creating swapchain: requested %ux%u", width, height);

    GfxSwapchainDescriptor swapchainDesc = {
        .sType = GFX_STRUCTURE_TYPE_SWAPCHAIN_DESCRIPTOR,
        .pNext = NULL,
        .label = "Main Swapchain",
        .surface = app->surface,
        .extent.width = width,
        .extent.height = height,
        .format = COLOR_FORMAT,
        .usage = GFX_TEXTURE_USAGE_RENDER_ATTACHMENT,
        .presentMode = app->settings.vsync ? GFX_PRESENT_MODE_FIFO : GFX_PRESENT_MODE_IMMEDIATE,
        .imageCount = app->surfaceInfo.minImageCount
    };

    if (gfxDeviceCreateSwapchain(app->device, &swapchainDesc, &app->swapchain) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create swapchain");
        return false;
    }

    // Query the actual swapchain format (may differ from requested format on web)
    GfxResult result = gfxSwapchainGetInfo(app->swapchain, &app->swapchainInfo);
    if (result != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get swapchain info");
        return false;
    }
    LOG_INFO("Swapchain created: actual %ux%u, format %d (requested %d), images=%u",
        app->swapchainInfo.extent.width,
        app->swapchainInfo.extent.height,
        app->swapchainInfo.format,
        COLOR_FORMAT,
        app->swapchainInfo.imageCount);

    // Create per-swapchain-image render finished semaphores (to avoid semaphore reuse issues)
    app->renderFinishedSemaphores = (GfxSemaphore*)calloc(app->swapchainInfo.imageCount, sizeof(GfxSemaphore));
    if (!app->renderFinishedSemaphores) {
        LOG_ERROR("Failed to allocate render finished semaphores array");
        return false;
    }

    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        char label[64];

        snprintf(label, sizeof(label), "Render Finished Semaphore (Image %u)", i);
        GfxSemaphoreDescriptor semaphoreDesc = {
            .sType = GFX_STRUCTURE_TYPE_SEMAPHORE_DESCRIPTOR,
            .pNext = NULL,
            .type = GFX_SEMAPHORE_TYPE_BINARY,
            .label = label,
        };

        if (gfxDeviceCreateSemaphore(app->device, &semaphoreDesc, &app->renderFinishedSemaphores[i]) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create render finished semaphore for image %u", i);
            return false;
        }
    }

    return true;
}

static void destroySwapchain(CubeApp* app)
{
    // Destroy per-swapchain-image semaphores
    if (app->renderFinishedSemaphores) {
        for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
            if (app->renderFinishedSemaphores[i]) {
                gfxSemaphoreDestroy(app->renderFinishedSemaphores[i]);
                app->renderFinishedSemaphores[i] = NULL;
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

static bool createRenderTargetTextures(CubeApp* app, uint32_t width, uint32_t height)
{
    // Create depth texture (MSAA must match color attachment)
    GfxTextureDescriptor depthTextureDesc = {
        .sType = GFX_STRUCTURE_TYPE_TEXTURE_DESCRIPTOR,
        .pNext = NULL,
        .label = "Depth Buffer",
        .type = GFX_TEXTURE_TYPE_2D,
        .size = (GfxExtent3D){ .width = width, .height = height, .depth = 1 },
        .arrayLayerCount = 1,
        .mipLevelCount = 1,
        .sampleCount = app->settings.msaaSampleCount,
        .format = DEPTH_FORMAT,
        .usage = GFX_TEXTURE_USAGE_RENDER_ATTACHMENT
    };

    if (gfxDeviceCreateTexture(app->device, &depthTextureDesc, &app->depthTexture) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create depth texture");
        return false;
    }

    // Create depth texture view
    GfxTextureViewDescriptor depthViewDesc = {
        .sType = GFX_STRUCTURE_TYPE_TEXTURE_VIEW_DESCRIPTOR,
        .pNext = NULL,
        .label = "Depth Buffer View",
        .viewType = GFX_TEXTURE_VIEW_TYPE_2D,
        .format = DEPTH_FORMAT,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };

    if (gfxTextureCreateView(app->depthTexture, &depthViewDesc, &app->depthTextureView) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create depth texture view");
        return false;
    }

    // Create MSAA color texture (is unused if MSAA sample count == 1)
    GfxTextureDescriptor msaaColorTextureDesc = {
        .sType = GFX_STRUCTURE_TYPE_TEXTURE_DESCRIPTOR,
        .pNext = NULL,
        .label = "MSAA Color Buffer",
        .type = GFX_TEXTURE_TYPE_2D,
        .size = (GfxExtent3D){ .width = width, .height = height, .depth = 1 },
        .arrayLayerCount = 1,
        .mipLevelCount = 1,
        .sampleCount = app->settings.msaaSampleCount,
        .format = app->swapchainInfo.format,
        .usage = GFX_TEXTURE_USAGE_RENDER_ATTACHMENT
    };

    if (gfxDeviceCreateTexture(app->device, &msaaColorTextureDesc, &app->msaaColorTexture) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create MSAA color texture");
        return false;
    }

    // Create MSAA color texture view (is unused if MSAA sample count == 1)
    GfxTextureViewDescriptor msaaColorViewDesc = {
        .sType = GFX_STRUCTURE_TYPE_TEXTURE_VIEW_DESCRIPTOR,
        .pNext = NULL,
        .label = "MSAA Color Buffer View",
        .viewType = GFX_TEXTURE_VIEW_TYPE_2D,
        .format = app->swapchainInfo.format,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };

    if (gfxTextureCreateView(app->msaaColorTexture, &msaaColorViewDesc, &app->msaaColorTextureView) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create MSAA color texture view");
        return false;
    }

    return true;
}

static void destroyRenderTargetTextures(CubeApp* app)
{
    // Destroy MSAA color texture
    if (app->msaaColorTextureView) {
        gfxTextureViewDestroy(app->msaaColorTextureView);
        app->msaaColorTextureView = NULL;
    }
    if (app->msaaColorTexture) {
        gfxTextureDestroy(app->msaaColorTexture);
        app->msaaColorTexture = NULL;
    }

    // Destroy depth texture
    if (app->depthTextureView) {
        gfxTextureViewDestroy(app->depthTextureView);
        app->depthTextureView = NULL;
    }
    if (app->depthTexture) {
        gfxTextureDestroy(app->depthTexture);
        app->depthTexture = NULL;
    }
}

static bool createFrameBuffers(CubeApp* app, uint32_t width, uint32_t height)
{
    // Allocate framebuffers array
    app->framebuffers = (GfxFramebuffer*)calloc(app->swapchainInfo.imageCount, sizeof(GfxFramebuffer));
    if (!app->framebuffers) {
        LOG_ERROR("Failed to allocate framebuffers array");
        return false;
    }

    // Create framebuffers (one per swapchain image)
    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        GfxTextureView backbuffer = NULL;
        GfxResult result = gfxSwapchainGetTextureView(app->swapchain, i, &backbuffer);
        if (result != GFX_RESULT_SUCCESS || !backbuffer) {
            LOG_ERROR("Failed to get swapchain image view %u", i);
            return false;
        }

        // Bundle color view with resolve target
        GfxFramebufferAttachment fbColorAttachment = {
            .view = (app->settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? app->msaaColorTextureView : backbuffer,
            .resolveTarget = (app->settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? backbuffer : NULL
        };

        // Depth/stencil attachment (no resolve)
        GfxFramebufferAttachment fbDepthAttachment = {
            .view = app->depthTextureView,
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
            .extent.width = width,
            .extent.height = height
        };

        if (gfxDeviceCreateFramebuffer(app->device, &fbDesc, &app->framebuffers[i]) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create framebuffer %u", i);
            return false;
        }
    }

    return true;
}

static void destroyFrameBuffers(CubeApp* app)
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

static bool createGeometry(CubeApp* app)
{
    // Create cube vertices (24 vertices - 4 per face for proper UV mapping)
    Vertex vertices[] = {
        // Front face (Z+)
        { { -1.0f, -1.0f, 1.0f }, { 0.0f, 1.0f } }, // 0
        { { 1.0f, -1.0f, 1.0f }, { 1.0f, 1.0f } }, // 1
        { { 1.0f, 1.0f, 1.0f }, { 1.0f, 0.0f } }, // 2
        { { -1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f } }, // 3

        // Back face (Z-)
        { { 1.0f, -1.0f, -1.0f }, { 0.0f, 1.0f } }, // 4
        { { -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f } }, // 5
        { { -1.0f, 1.0f, -1.0f }, { 1.0f, 0.0f } }, // 6
        { { 1.0f, 1.0f, -1.0f }, { 0.0f, 0.0f } }, // 7

        // Left face (X-)
        { { -1.0f, -1.0f, -1.0f }, { 0.0f, 1.0f } }, // 8
        { { -1.0f, -1.0f, 1.0f }, { 1.0f, 1.0f } }, // 9
        { { -1.0f, 1.0f, 1.0f }, { 1.0f, 0.0f } }, // 10
        { { -1.0f, 1.0f, -1.0f }, { 0.0f, 0.0f } }, // 11

        // Right face (X+)
        { { 1.0f, -1.0f, 1.0f }, { 0.0f, 1.0f } }, // 12
        { { 1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f } }, // 13
        { { 1.0f, 1.0f, -1.0f }, { 1.0f, 0.0f } }, // 14
        { { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f } }, // 15

        // Top face (Y+)
        { { -1.0f, 1.0f, 1.0f }, { 0.0f, 1.0f } }, // 16
        { { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f } }, // 17
        { { 1.0f, 1.0f, -1.0f }, { 1.0f, 0.0f } }, // 18
        { { -1.0f, 1.0f, -1.0f }, { 0.0f, 0.0f } }, // 19

        // Bottom face (Y-)
        { { -1.0f, -1.0f, -1.0f }, { 0.0f, 1.0f } }, // 20
        { { 1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f } }, // 21
        { { 1.0f, -1.0f, 1.0f }, { 1.0f, 0.0f } }, // 22
        { { -1.0f, -1.0f, 1.0f }, { 0.0f, 0.0f } } // 23
    };

    // Create cube indices (36 indices for 12 triangles)
    uint16_t indices[] = {
        // Front face (Z+)
        0, 1, 2, 2, 3, 0,
        // Back face (Z-)
        4, 5, 6, 6, 7, 4,
        // Left face (X-)
        8, 9, 10, 10, 11, 8,
        // Right face (X+)
        12, 13, 14, 14, 15, 12,
        // Top face (Y+)
        16, 17, 18, 18, 19, 16,
        // Bottom face (Y-)
        20, 21, 22, 22, 23, 20
    };
    GfxBufferDescriptor vertexBufferDesc = {
        .label = "Cube Vertices",
        .size = sizeof(vertices),
        .usage = GFX_BUFFER_USAGE_VERTEX | GFX_BUFFER_USAGE_COPY_DST,
        .memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL
    };

    if (gfxDeviceCreateBuffer(app->device, &vertexBufferDesc, &app->vertexBuffer) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create vertex buffer");
        return false;
    }
    gfxBufferGetInfo(app->vertexBuffer, &app->vertexBufferInfo);

    // Create index buffer
    GfxBufferDescriptor indexBufferDesc = {
        .sType = GFX_STRUCTURE_TYPE_BUFFER_DESCRIPTOR,
        .pNext = NULL,
        .label = "Cube Indices",
        .size = sizeof(indices),
        .usage = GFX_BUFFER_USAGE_INDEX | GFX_BUFFER_USAGE_COPY_DST,
        .memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL
    };

    if (gfxDeviceCreateBuffer(app->device, &indexBufferDesc, &app->indexBuffer) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create index buffer");
        return false;
    }
    gfxBufferGetInfo(app->indexBuffer, &app->indexBufferInfo);

    // Upload vertex and index data
    if (gfxQueueWriteBuffer(app->queue, app->vertexBuffer, 0, vertices, sizeof(vertices)) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to write vertex data to buffer");
        return false;
    }
    if (gfxQueueWriteBuffer(app->queue, app->indexBuffer, 0, indices, sizeof(indices)) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to write index data to buffer");
        return false;
    }

    return true;
}

static void destroyGeometry(CubeApp* app)
{
    if (app->indexBuffer) {
        gfxBufferDestroy(app->indexBuffer);
        app->indexBuffer = NULL;
    }
    if (app->vertexBuffer) {
        gfxBufferDestroy(app->vertexBuffer);
        app->vertexBuffer = NULL;
    }
}

static bool createUniformBuffer(CubeApp* app)
{
    // Create single large uniform buffer for all frames with proper alignment
    GfxDeviceLimits limits;
    if (gfxDeviceGetLimits(app->device, &limits) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get device limits");
        return false;
    }

    size_t uniformSize = sizeof(UniformData);
    app->uniformAlignedSize = gfxAlignUp(uniformSize, limits.minUniformBufferOffsetAlignment);
    // Need space for CUBE_COUNT cubes per swapchain image
    size_t totalBufferSize = app->uniformAlignedSize * app->swapchainInfo.imageCount * CUBE_COUNT;

    GfxBufferDescriptor uniformBufferDesc = {
        .sType = GFX_STRUCTURE_TYPE_BUFFER_DESCRIPTOR,
        .pNext = NULL,
        .label = "Shared Transform Uniforms",
        .size = totalBufferSize,
        .usage = GFX_BUFFER_USAGE_UNIFORM | GFX_BUFFER_USAGE_COPY_DST,
        .memoryProperties = GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT
    };

    if (gfxDeviceCreateBuffer(app->device, &uniformBufferDesc, &app->sharedUniformBuffer) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create shared uniform buffer");
        return false;
    }

    return true;
}

static void destroyUniformBuffer(CubeApp* app)
{
    if (app->sharedUniformBuffer) {
        gfxBufferDestroy(app->sharedUniformBuffer);
        app->sharedUniformBuffer = NULL;
    }
}

static bool createBindGroup(CubeApp* app)
{
    // Create bind group layout for uniforms
    GfxBindGroupLayoutEntry uniformLayoutEntry = {
        .binding = 0,
        .visibility = GFX_SHADER_STAGE_VERTEX,
        .type = GFX_BINDING_TYPE_BUFFER,
        .count = 1,
        .buffer = {
            .hasDynamicOffset = false,
            .minBindingSize = sizeof(UniformData),
        },
    };

    GfxBindGroupLayoutDescriptor uniformLayoutDesc = {
        .sType = GFX_STRUCTURE_TYPE_BIND_GROUP_LAYOUT_DESCRIPTOR,
        .pNext = NULL,
        .label = "Uniform Bind Group Layout",
        .entries = &uniformLayoutEntry,
        .entryCount = 1
    };

    if (gfxDeviceCreateBindGroupLayout(app->device, &uniformLayoutDesc, &app->uniformBindGroupLayout) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create uniform bind group layout");
        return false;
    }

    // Note: Bind groups are created in createPerFrameResources() after all resources are ready

    // Create texture bind group layout
    GfxBindGroupLayoutEntry textureLayoutEntries[] = {
        {
            .binding = 0,
            .visibility = GFX_SHADER_STAGE_FRAGMENT,
            .type = GFX_BINDING_TYPE_TEXTURE,
            .count = 1,
            .texture = {
                .sampleType = GFX_TEXTURE_SAMPLE_TYPE_FLOAT,
                .viewDimension = GFX_TEXTURE_VIEW_TYPE_2D,
            },
        },
        {
            .binding = 1,
            .visibility = GFX_SHADER_STAGE_FRAGMENT,
            .type = GFX_BINDING_TYPE_SAMPLER,
            .count = 1,
            .sampler = {
                .type = GFX_SAMPLER_BINDING_TYPE_FILTERING,
            },
        }
    };

    GfxBindGroupLayoutDescriptor textureLayoutDesc = {
        .label = "Texture Bind Group Layout",
        .entries = textureLayoutEntries,
        .entryCount = ARRAY_SIZE(textureLayoutEntries)
    };

    if (gfxDeviceCreateBindGroupLayout(app->device, &textureLayoutDesc, &app->textureBindGroupLayout) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create texture bind group layout");
        return false;
    }

    // Create texture bind group
    GfxBindGroupEntry textureEntries[] = {
        { .binding = 0,
            .type = GFX_BIND_GROUP_ENTRY_TYPE_TEXTURE_VIEW,
            .resource = { .textureView = app->cubeTextureView } },
        { .binding = 1,
            .type = GFX_BIND_GROUP_ENTRY_TYPE_SAMPLER,
            .resource = { .sampler = app->cubeSampler } }
    };

    GfxBindGroupDescriptor textureBindGroupDesc = {
        .sType = GFX_STRUCTURE_TYPE_BIND_GROUP_DESCRIPTOR,
        .pNext = NULL,
        .label = "Texture Bind Group",
        .layout = app->textureBindGroupLayout,
        .entries = textureEntries,
        .entryCount = ARRAY_SIZE(textureEntries)
    };

    if (gfxDeviceCreateBindGroup(app->device, &textureBindGroupDesc, &app->textureBindGroup) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create texture bind group");
        return false;
    }

    return true;
}

static void destroyBindGroup(CubeApp* app)
{
    // Destroy texture bind groups and layouts
    if (app->textureBindGroup) {
        gfxBindGroupDestroy(app->textureBindGroup);
        app->textureBindGroup = NULL;
    }
    if (app->textureBindGroupLayout) {
        gfxBindGroupLayoutDestroy(app->textureBindGroupLayout);
        app->textureBindGroupLayout = NULL;
    }

    // Destroy uniform bind group layout
    if (app->uniformBindGroupLayout) {
        gfxBindGroupLayoutDestroy(app->uniformBindGroupLayout);
        app->uniformBindGroupLayout = NULL;
    }
}

static bool createShaders(CubeApp* app)
{
    // Load shaders from files (works for both native and web)
    size_t vertexShaderSize, fragmentShaderSize;
    void* vertexShaderCode = NULL;
    void* fragmentShaderCode = NULL;
    GfxShaderSourceType sourceType = GFX_SHADER_SOURCE_SPIRV;

    // Try shader formats in order of preference
    const struct {
        GfxShaderSourceType format;
        const char* vertexPath;
        const char* fragmentPath;
    } shaderFormats[] = {
        { GFX_SHADER_SOURCE_SPIRV, "shaders/cube_textured.vert.spv", "shaders/cube_textured.frag.spv" },
        { GFX_SHADER_SOURCE_WGSL, "shaders/cube_textured.vert.wgsl", "shaders/cube_textured.frag.wgsl" }
    };

    for (size_t i = 0; i < ARRAY_SIZE(shaderFormats); ++i) {
        bool formatSupported = false;
        if (gfxDeviceSupportsShaderFormat(app->device, shaderFormats[i].format, &formatSupported) != GFX_RESULT_SUCCESS || !formatSupported) {
            continue;
        }

        LOG_INFO("Loading shaders: %s, %s", shaderFormats[i].vertexPath, shaderFormats[i].fragmentPath);

        if (shaderFormats[i].format == GFX_SHADER_SOURCE_SPIRV) {
            vertexShaderCode = loadBinaryFile(app, shaderFormats[i].vertexPath, &vertexShaderSize);
            fragmentShaderCode = loadBinaryFile(app, shaderFormats[i].fragmentPath, &fragmentShaderSize);
        } else {
            vertexShaderCode = loadTextFile(app, shaderFormats[i].vertexPath, &vertexShaderSize);
            fragmentShaderCode = loadTextFile(app, shaderFormats[i].fragmentPath, &fragmentShaderSize);
        }

        if (vertexShaderCode && fragmentShaderCode) {
            sourceType = shaderFormats[i].format;
            LOG_INFO("Successfully loaded shaders (vertex: %zu bytes, fragment: %zu bytes)",
                vertexShaderSize, fragmentShaderSize);
            break;
        }

        // Failed to load this format, clean up and try next
        free(vertexShaderCode);
        free(fragmentShaderCode);
        vertexShaderCode = NULL;
        fragmentShaderCode = NULL;
    }

    if (!vertexShaderCode || !fragmentShaderCode) {
        LOG_ERROR("Error: No supported shader format found or failed to load shaders");
        return false;
    }

    // Create vertex shader
    GfxShaderDescriptor vertexShaderDesc = {
        .sType = GFX_STRUCTURE_TYPE_SHADER_DESCRIPTOR,
        .pNext = NULL,
        .label = "Cube Vertex Shader",
        .sourceType = sourceType,
        .code = vertexShaderCode,
        .codeSize = vertexShaderSize,
        .entryPoint = "main"
    };

    if (gfxDeviceCreateShader(app->device, &vertexShaderDesc, &app->vertexShader) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create vertex shader");
        free(vertexShaderCode);
        free(fragmentShaderCode);
        return false;
    }

    // Create fragment shader
    GfxShaderDescriptor fragmentShaderDesc = {
        .sType = GFX_STRUCTURE_TYPE_SHADER_DESCRIPTOR,
        .pNext = NULL,
        .label = "Cube Fragment Shader",
        .sourceType = sourceType,
        .code = fragmentShaderCode,
        .codeSize = fragmentShaderSize,
        .entryPoint = "main"
    };

    if (gfxDeviceCreateShader(app->device, &fragmentShaderDesc, &app->fragmentShader) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create fragment shader");
        free(vertexShaderCode);
        free(fragmentShaderCode);
        return false;
    }

    free(vertexShaderCode);
    free(fragmentShaderCode);
    return true;
}

static void destroyShaders(CubeApp* app)
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

static bool loadTexture(CubeApp* app)
{
    const char* texturePath = "textures/vulkan.png";

    // Load image with stb_image
    int width, height, channels;
    unsigned char* pixels = NULL;
    stbi_set_flip_vertically_on_load(1);

#if defined(__ANDROID__)
    // On Android, load from assets and use stbi_load_from_memory
    LOG_INFO("Loading texture from assets: %s", texturePath);
    size_t fileSize = 0;
    void* fileData = loadBinaryFile(app, texturePath, &fileSize);
    if (!fileData) {
        LOG_ERROR("Failed to load texture file from assets: %s", texturePath);
        return false;
    }

    pixels = stbi_load_from_memory((const stbi_uc*)fileData, (int)fileSize, &width, &height, &channels, STBI_rgb_alpha);
    free(fileData); // Free the asset data after loading

    if (!pixels) {
        LOG_ERROR("Failed to decode texture: %s", texturePath);
        return false;
    }
    LOG_INFO("Texture loaded: %dx%d, %d channels", width, height, channels);
#else
    // On desktop, use stbi_load which uses fopen
    pixels = stbi_load(texturePath, &width, &height, &channels, STBI_rgb_alpha);

    if (!pixels) {
        LOG_ERROR("Failed to load texture: %s", texturePath);
        return false;
    }
    LOG_INFO("Texture loaded: %dx%d, %d channels", width, height, channels);
#endif

    // Calculate mip levels (log2(max(width, height)) + 1)
    uint32_t maxDim = width > height ? width : height;
    uint32_t mipLevels = 1;
    while (maxDim > 1) {
        maxDim >>= 1;
        mipLevels++;
    }

    LOG_INFO("Creating texture with %u mip levels (%dx%d) - ASYNC UPLOAD", mipLevels, width, height);

    // Create texture with mipmaps
    GfxTextureDescriptor textureDesc = {
        .sType = GFX_STRUCTURE_TYPE_TEXTURE_DESCRIPTOR,
        .pNext = NULL,
        .label = "Cube Texture",
        .type = GFX_TEXTURE_TYPE_2D,
        .size = (GfxExtent3D){ (uint32_t)width, (uint32_t)height, 1 },
        .arrayLayerCount = 1,
        .mipLevelCount = mipLevels,
        .sampleCount = GFX_SAMPLE_COUNT_1,
        .format = GFX_FORMAT_R8G8B8A8_UNORM_SRGB,
        .usage = GFX_TEXTURE_USAGE_TEXTURE_BINDING | GFX_TEXTURE_USAGE_COPY_SRC | GFX_TEXTURE_USAGE_COPY_DST | GFX_TEXTURE_USAGE_RENDER_ATTACHMENT
    };

    if (gfxDeviceCreateTexture(app->device, &textureDesc, &app->cubeTexture) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create texture");
        stbi_image_free(pixels);
        return false;
    }

    // Create staging buffer for async upload
    uint64_t dataSize = (uint64_t)width * height * 4;
    GfxBufferDescriptor stagingDesc = {
        .sType = GFX_STRUCTURE_TYPE_BUFFER_DESCRIPTOR,
        .pNext = NULL,
        .label = "Texture Staging Buffer",
        .size = dataSize,
        .usage = GFX_BUFFER_USAGE_MAP_WRITE | GFX_BUFFER_USAGE_COPY_SRC,
        .memoryProperties = GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT
    };

    if (gfxDeviceCreateBuffer(app->device, &stagingDesc, &app->textureStagingBuffer) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create staging buffer");
        stbi_image_free(pixels);
        return false;
    }

    // Map and copy texture data to staging buffer
    void* mappedData;
    if (gfxBufferMap(app->textureStagingBuffer, 0, GFX_WHOLE_SIZE, &mappedData) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to map staging buffer");
        stbi_image_free(pixels);
        return false;
    }

    memcpy(mappedData, pixels, dataSize);
    gfxBufferUnmap(app->textureStagingBuffer);
    stbi_image_free(pixels);

    // Create command encoder for async upload
    GfxCommandEncoderDescriptor encoderDesc = {
        .sType = GFX_STRUCTURE_TYPE_COMMAND_ENCODER_DESCRIPTOR,
        .pNext = NULL,
        .label = "Texture Upload Encoder"
    };

    if (gfxDeviceCreateCommandEncoder(app->device, &encoderDesc, &app->textureUploadEncoder) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create command encoder for texture upload");
        return false;
    }

    if (gfxCommandEncoderBegin(app->textureUploadEncoder) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to begin texture upload command encoder");
        return false;
    }

    // Copy from staging buffer to texture
    GfxCopyBufferToTextureDescriptor copyDesc = {
        .source = app->textureStagingBuffer,
        .sourceOffset = 0,
        .destination = app->cubeTexture,
        .origin = { 0, 0, 0 },
        .extent = { (uint32_t)width, (uint32_t)height, 1 },
        .mipLevel = 0,
        .finalLayout = GFX_TEXTURE_LAYOUT_SHADER_READ_ONLY
    };

    if (gfxCommandEncoderCopyBufferToTexture(app->textureUploadEncoder, &copyDesc) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to copy buffer to texture");
        return false;
    }

    // Generate mipmaps in the same command buffer
    if (gfxCommandEncoderGenerateMipmaps(app->textureUploadEncoder, app->cubeTexture) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to generate mipmaps");
        return false;
    }

    if (gfxCommandEncoderEnd(app->textureUploadEncoder) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to end texture upload command encoder");
        return false;
    }

    // Create fence for tracking upload completion
    GfxFenceDescriptor fenceDesc = {
        .sType = GFX_STRUCTURE_TYPE_FENCE_DESCRIPTOR,
        .pNext = NULL,
        .label = "Texture Upload Fence",
        .signaled = false
    };

    if (gfxDeviceCreateFence(app->device, &fenceDesc, &app->textureUploadFence) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create upload fence");
        gfxCommandEncoderDestroy(app->textureUploadEncoder);
        return false;
    }

    // Submit upload commands with fence
    GfxSubmitDescriptor submitDesc = {
        .commandEncoders = &app->textureUploadEncoder,
        .commandEncoderCount = 1,
        .signalFence = app->textureUploadFence
    };

    if (gfxQueueSubmit(app->queue, &submitDesc) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to submit texture upload commands");
        return false;
    }

    // Texture upload is now happening asynchronously
    // The render loop will check app->textureUploadFence status
    // and clean up resources when complete
    LOG_INFO("Texture upload submitted asynchronously...");

    // Create texture view with all mip levels
    GfxTextureViewDescriptor viewDesc = {
        .sType = GFX_STRUCTURE_TYPE_TEXTURE_VIEW_DESCRIPTOR,
        .pNext = NULL,
        .label = "Cube Texture View",
        .viewType = GFX_TEXTURE_VIEW_TYPE_2D,
        .format = GFX_FORMAT_R8G8B8A8_UNORM_SRGB,
        .baseMipLevel = 0,
        .mipLevelCount = mipLevels,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };

    if (gfxTextureCreateView(app->cubeTexture, &viewDesc, &app->cubeTextureView) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create texture view");
        return false;
    }

    // Create sampler with mipmap filtering
    GfxSamplerDescriptor samplerDesc = {
        .sType = GFX_STRUCTURE_TYPE_SAMPLER_DESCRIPTOR,
        .pNext = NULL,
        .label = "Cube Sampler",
        .magFilter = GFX_FILTER_MODE_LINEAR,
        .minFilter = GFX_FILTER_MODE_LINEAR,
        .mipmapFilter = GFX_FILTER_MODE_LINEAR,
        .addressModeU = GFX_ADDRESS_MODE_REPEAT,
        .addressModeV = GFX_ADDRESS_MODE_REPEAT,
        .addressModeW = GFX_ADDRESS_MODE_REPEAT,
        .maxAnisotropy = 1
    };

    if (gfxDeviceCreateSampler(app->device, &samplerDesc, &app->cubeSampler) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create sampler");
        return false;
    }

    return true;
}

static void unloadTexture(CubeApp* app)
{
    // If async texture upload hasn't completed yet, wait for it
    if (!app->textureUploadComplete && app->textureUploadFence) {
        LOG_INFO("Waiting for texture upload to complete before cleanup...");
        gfxFenceWait(app->textureUploadFence, GFX_TIMEOUT_INFINITE);

        if (app->textureStagingBuffer) {
            gfxBufferDestroy(app->textureStagingBuffer);
            app->textureStagingBuffer = NULL;
        }
        if (app->textureUploadEncoder) {
            gfxCommandEncoderDestroy(app->textureUploadEncoder);
            app->textureUploadEncoder = NULL;
        }
        app->textureUploadComplete = true;
    }

    // Clean up async texture upload fence
    if (app->textureUploadFence) {
        gfxFenceDestroy(app->textureUploadFence);
        app->textureUploadFence = NULL;
    }

    if (app->cubeSampler) {
        gfxSamplerDestroy(app->cubeSampler);
        app->cubeSampler = NULL;
    }
    if (app->cubeTextureView) {
        gfxTextureViewDestroy(app->cubeTextureView);
        app->cubeTextureView = NULL;
    }
    if (app->cubeTexture) {
        gfxTextureDestroy(app->cubeTexture);
        app->cubeTexture = NULL;
    }
}

static bool createRenderingResources(CubeApp* app)
{
    if (!createGeometry(app)) {
        return false;
    }

    if (!loadTexture(app)) {
        return false;
    }

    if (!createUniformBuffer(app)) {
        return false;
    }

    if (!createBindGroup(app)) {
        return false;
    }

    if (!createShaders(app)) {
        return false;
    }
    return true;
}

static void destroyRenderingResources(CubeApp* app)
{
    // Destroy render pipeline
    destroyRenderPipeline(app);

    // Destroy shaders
    destroyShaders(app);

    // Unload texture
    unloadTexture(app);

    // Destroy bind groups and layouts
    destroyBindGroup(app);

    // Destroy uniform buffer
    destroyUniformBuffer(app);

    // Destroy geometry buffers
    destroyGeometry(app);
}

static bool createRenderPipeline(CubeApp* app)
{
    // Define vertex attributes
    GfxVertexAttribute attributes[] = {
        { .format = GFX_FORMAT_R32G32B32_FLOAT,
            .offset = offsetof(Vertex, position),
            .shaderLocation = 0 },
        { .format = GFX_FORMAT_R32G32B32_FLOAT,
            .offset = offsetof(Vertex, color),
            .shaderLocation = 1 }
    };

    // Define vertex buffer layout
    GfxVertexBufferLayout vertexBufferLayout = {
        .arrayStride = sizeof(Vertex),
        .attributes = attributes,
        .attributeCount = ARRAY_SIZE(attributes),
        .stepMode = GFX_VERTEX_STEP_MODE_VERTEX
    };

    // Vertex state
    GfxVertexState vertexState = {
        .module = app->vertexShader,
        .entryPoint = "main",
        .buffers = &vertexBufferLayout,
        .bufferCount = 1
    };

    // Color target state
    // Note: Always 1 target even with MSAA - resolve is handled by render pass, not fragment shader
    // layout(location = 0) out vec4 outColor;
    // Use actual swapchain format (may differ from requested format on web)
    GfxColorTargetState colorTarget = {
        .format = app->swapchainInfo.format,
        .blend = NULL,
        .writeMask = GFX_COLOR_WRITE_MASK_ALL
    };

    // Fragment state
    GfxFragmentState fragmentState = {
        .module = app->fragmentShader,
        .entryPoint = "main",
        .targets = &colorTarget,
        .targetCount = 1
    };

    // Primitive state
    GfxPrimitiveState primitiveState = {
        .topology = GFX_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .stripIndexFormat = GFX_INDEX_FORMAT_UNDEFINED,
        .frontFace = GFX_FRONT_FACE_COUNTER_CLOCKWISE,
        .cullMode = GFX_CULL_MODE_BACK, // Enable back-face culling
        .polygonMode = GFX_POLYGON_MODE_FILL
    };

    // Depth/stencil state - enable depth testing
    GfxDepthStencilState depthStencilState = {
        .format = DEPTH_FORMAT,
        .depthWriteEnabled = true,
        .depthCompare = GFX_COMPARE_FUNCTION_LESS,
        .stencilFront = {
            .compare = GFX_COMPARE_FUNCTION_ALWAYS,
            .failOp = GFX_STENCIL_OPERATION_KEEP,
            .depthFailOp = GFX_STENCIL_OPERATION_KEEP,
            .passOp = GFX_STENCIL_OPERATION_KEEP },
        .stencilBack = { .compare = GFX_COMPARE_FUNCTION_ALWAYS, .failOp = GFX_STENCIL_OPERATION_KEEP, .depthFailOp = GFX_STENCIL_OPERATION_KEEP, .passOp = GFX_STENCIL_OPERATION_KEEP },
        .stencilReadMask = 0xFF,
        .stencilWriteMask = 0xFF,
        .depthBias = 0,
        .depthBiasSlopeScale = 0.0f,
        .depthBiasClamp = 0.0f
    };

    // Create render pipeline
    // Create array with the bind group layout pointer
    GfxBindGroupLayout bindGroupLayouts[] = { app->uniformBindGroupLayout, app->textureBindGroupLayout };

    GfxRenderPipelineDescriptor pipelineDesc = {
        .sType = GFX_STRUCTURE_TYPE_RENDER_PIPELINE_DESCRIPTOR,
        .pNext = NULL,
        .label = "Cube Render Pipeline",
        .vertex = &vertexState,
        .fragment = &fragmentState,
        .primitive = &primitiveState,
        .depthStencil = &depthStencilState,
        .sampleCount = app->settings.msaaSampleCount,
        .renderPass = app->renderPass,
        .bindGroupLayouts = bindGroupLayouts,
        .bindGroupLayoutCount = ARRAY_SIZE(bindGroupLayouts)
    };

    if (gfxDeviceCreateRenderPipeline(app->device, &pipelineDesc, &app->renderPipeline) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create render pipeline");
        return false;
    }

    return true;
}

static void destroyRenderPipeline(CubeApp* app)
{
    if (app->renderPipeline) {
        gfxRenderPipelineDestroy(app->renderPipeline);
        app->renderPipeline = NULL;
    }
}

static void updateCube(CubeApp* app, int cubeIndex)
{
    UniformData uniforms = { 0 }; // Initialize to zero!

    // Create rotation matrices (combine X and Y rotations)
    // Each cube rotates slightly differently
    Mat4 rotX, rotY, tempModel;
    matrixRotateX(&rotX, (app->rotationAngleX + cubeIndex * 30.0f) * M_PI / 180.0f);
    matrixRotateY(&rotY, (app->rotationAngleY + cubeIndex * 45.0f) * M_PI / 180.0f);
    matrixMultiply(&tempModel, &rotY, &rotX);

    // Position cubes side by side: left (-3, 0, 0), center (0, 0, 0), right (3, 0, 0)
    Mat4 translation;
    matrixIdentity(&translation);
    translation.m[12] = (cubeIndex - 1) * 3.0f; // x offset: -3, 0, 3

    // Apply translation after rotation: model = rotation * translation
    // This rotates in place, then translates to world position
    matrixMultiply(&uniforms.model, &tempModel, &translation);

    // Create view matrix (camera positioned at 0, 0, 10 looking at origin)
    Vec3 eye = { 0.0f, 0.0f, 10.0f }; // pulled back to see all 3 cubes
    Vec3 center = { 0.0f, 0.0f, 0.0f }; // look at point
    Vec3 up = { 0.0f, 1.0f, 0.0f }; // up vector
    matrixLookAt(&uniforms.view, &eye, &center, &up);

    // Create perspective projection matrix
    float aspect = (float)app->swapchainInfo.extent.width / (float)app->swapchainInfo.extent.height;
    matrixPerspective(&uniforms.projection,
        45.0f * M_PI / 180.0f, // 45 degree FOV
        aspect,
        0.1f, // near plane
        100.0f, // far plane
        app->adapterInfo.backend); // Adjust for clip space differences

    // Upload uniform data to buffer at aligned offset
    // Formula: (frame * CUBE_COUNT + cube) * alignedSize
    size_t offset = (app->currentFrame * CUBE_COUNT + cubeIndex) * app->uniformAlignedSize;
    if (gfxQueueWriteBuffer(app->queue, app->sharedUniformBuffer, offset, &uniforms, sizeof(uniforms)) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to write uniform data for cube %d", cubeIndex);
    }
}

static GfxPlatformWindowHandle getPlatformWindowHandle(CubeApp* app)
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
// On Android, uses AAssetManager; on desktop, uses fopen
static void* loadBinaryFile(CubeApp* app, const char* filepath, size_t* outSize)
{
#if defined(__ANDROID__)
    if (!app || !app->androidApp || !app->androidApp->activity || !app->androidApp->activity->assetManager) {
        LOG_ERROR("AssetManager not available (app=%p, androidApp=%p, activity=%p)",
            (void*)app,
            app ? (void*)app->androidApp : NULL,
            (app && app->androidApp) ? (void*)app->androidApp->activity : NULL);
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
    (void)app; // Unused on desktop/iOS
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
// On Android, uses AAssetManager; on desktop, uses fopen
static void* loadTextFile(CubeApp* app, const char* filepath, size_t* outSize)
{
#if defined(__ANDROID__)
    if (!app || !app->androidApp || !app->androidApp->activity || !app->androidApp->activity->assetManager) {
        LOG_ERROR("AssetManager not available for text file (app=%p, androidApp=%p, activity=%p)",
            (void*)app,
            app ? (void*)app->androidApp : NULL,
            (app && app->androidApp) ? (void*)app->androidApp->activity : NULL);
        return NULL;
    }

    AAsset* asset = AAssetManager_open(app->androidApp->activity->assetManager, filepath, AASSET_MODE_BUFFER);
    if (!asset) {
        LOG_ERROR("Failed to open asset: %s", filepath);
        return NULL;
    }

    size_t size = AAsset_getLength(asset);
    // Allocate +1 for null terminator
    void* buffer = malloc(size + 1);
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

    // Null-terminate for text files
    ((char*)buffer)[size] = '\0';

    if (outSize) {
        *outSize = size;
    }

    return buffer;
#else
    (void)app; // Unused on desktop/iOS
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
        LOG_ERROR("Failed to read complete file: %s", filepath);
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

// ============================================================================
// Math Functions
// ============================================================================

// Matrix math utility functions
static void matrixIdentity(Mat4* matrix)
{
    memset(matrix->m, 0, 16 * sizeof(float));
    matrix->m[0] = matrix->m[5] = matrix->m[10] = matrix->m[15] = 1.0f;
}

static void matrixMultiply(Mat4* result, const Mat4* a, const Mat4* b)
{
    float temp[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            temp[i * 4 + j] = 0;
            for (int k = 0; k < 4; k++) {
                temp[i * 4 + j] += a->m[i * 4 + k] * b->m[k * 4 + j];
            }
        }
    }
    memcpy(result->m, temp, sizeof(float) * 16);
}

static void matrixRotateX(Mat4* matrix, float angle)
{
    float c = cosf(angle);
    float s = sinf(angle);

    matrixIdentity(matrix);
    matrix->m[5] = c;
    matrix->m[6] = -s;
    matrix->m[9] = s;
    matrix->m[10] = c;
}

static void matrixRotateY(Mat4* matrix, float angle)
{
    float c = cosf(angle);
    float s = sinf(angle);

    matrixIdentity(matrix);
    matrix->m[0] = c;
    matrix->m[2] = s;
    matrix->m[8] = -s;
    matrix->m[10] = c;
}

static void matrixRotateZ(Mat4* matrix, float angle)
{
    float c = cosf(angle);
    float s = sinf(angle);

    matrixIdentity(matrix);
    matrix->m[0] = c;
    matrix->m[1] = -s;
    matrix->m[4] = s;
    matrix->m[5] = c;
}

static void matrixPerspective(Mat4* matrix, float fov, float aspect, float nearPlane, float farPlane, GfxBackend backend)
{
    memset(matrix->m, 0, 16 * sizeof(float));

    float f = 1.0f / tanf(fov / 2.0f);

    matrix->m[0] = f / aspect;
    if (backend == GFX_BACKEND_VULKAN) {
        matrix->m[5] = -f; // Invert Y for Vulkan
    } else {
        matrix->m[5] = f;
    }
    matrix->m[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
    matrix->m[11] = -1.0f;
    matrix->m[14] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
}

static void matrixLookAt(Mat4* matrix, const Vec3* eye, const Vec3* center, const Vec3* up)
{
    // Calculate forward vector
    Vec3 forward = { center->x - eye->x, center->y - eye->y, center->z - eye->z };

    // Normalize forward vector
    if (!vectorNormalize(&forward)) {
        matrixIdentity(matrix);
        return;
    }

    // Calculate right vector (forward cross up)
    Vec3 right = {
        forward.y * up->z - forward.z * up->y,
        forward.z * up->x - forward.x * up->z,
        forward.x * up->y - forward.y * up->x
    };

    // Normalize right vector (check if forward and up are parallel)
    if (!vectorNormalize(&right)) {
        matrixIdentity(matrix);
        return;
    }

    // Calculate up vector (right cross forward)
    Vec3 upCorrect = {
        right.y * forward.z - right.z * forward.y,
        right.z * forward.x - right.x * forward.z,
        right.x * forward.y - right.y * forward.x
    };

    // Build view matrix
    matrix->m[0] = right.x;
    matrix->m[1] = upCorrect.x;
    matrix->m[2] = -forward.x;
    matrix->m[3] = 0.0f;

    matrix->m[4] = right.y;
    matrix->m[5] = upCorrect.y;
    matrix->m[6] = -forward.y;
    matrix->m[7] = 0.0f;

    matrix->m[8] = right.z;
    matrix->m[9] = upCorrect.z;
    matrix->m[10] = -forward.z;
    matrix->m[11] = 0.0f;

    matrix->m[12] = -(right.x * eye->x + right.y * eye->y + right.z * eye->z);
    matrix->m[13] = -(upCorrect.x * eye->x + upCorrect.y * eye->y + upCorrect.z * eye->z);
    matrix->m[14] = forward.x * eye->x + forward.y * eye->y + forward.z * eye->z;
    matrix->m[15] = 1.0f;
}

// Normalize a 3D vector in place. Returns false if vector is too small to normalize.
static bool vectorNormalize(Vec3* v)
{
    const float epsilon = 1e-6f;
    float len = sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);

    if (len < epsilon) {
        return false;
    }

    v->x /= len;
    v->y /= len;
    v->z /= len;
    return true;
}

static bool init(CubeApp* app)
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

    // 4. Create rendering resources (textures, buffers, layouts)
    if (!createRenderingResources(app)) {
        LOG_ERROR("Failed to create rendering resources");
        return false;
    }

    // 5. Create per-frame resources (depends on uniform buffer and layouts)
    if (!createPerFrameResources(app)) {
        LOG_ERROR("Failed to create per-frame resources");
        return false;
    }

    // 6. Create render pipeline (depends on render pass and resources)
    if (!createRenderPipeline(app)) {
        LOG_ERROR("Failed to create render pipeline");
        return false;
    }

    // Initialize loop state
    app->currentFrame = 0;
    app->previousWidth = app->windowWidth;
    app->previousHeight = app->windowHeight;
    app->elapsedTime = 0.0f;
    app->lastFrameTime = getCurrentTime();

    // Initialize animation state
    app->rotationAngleX = 0.0f;
    app->rotationAngleY = 0.0f;

    // Initialize FPS tracking
    app->fpsFrameCount = 0;
    app->fpsTimeAccumulator = 0.0f;
    app->fpsFrameTimeMin = FLT_MAX;
    app->fpsFrameTimeMax = 0.0f;

    LOG_INFO("Application initialized successfully!");
    return true;
}

static void cleanup(CubeApp* app)
{
    // Wait for device to finish all GPU work before cleanup
    if (app->device) {
        gfxDeviceWaitIdle(app->device);
    }

    // 5. Destroy render pipeline (depends on render pass and resources)
    destroyRenderPipeline(app);

    // 4. Destroy per-frame resources (depends on uniform buffer and layouts)
    destroyPerFrameResources(app);

    // 3. Destroy rendering resources (textures, buffers, layouts)
    destroyRenderingResources(app);

    // 2. Destroy size-dependent resources (swapchain, framebuffers, render pass)
    destroySizeDependentResources(app);

    // 1. Destroy graphics context (surface, device, instance)
    destroyGraphics(app);

    // 0. Destroy window
    destroyWindow(app);
}

// Handle window resize - common logic for all platforms
static bool handleResize(CubeApp* app, uint32_t width, uint32_t height)
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
static void updateFPS(CubeApp* app, float deltaTime)
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

static void update(CubeApp* app, float deltaTime)
{
    updateFPS(app, deltaTime);
    app->elapsedTime += deltaTime;

    // Update rotation angles (both X and Y axes)
    app->rotationAngleX += 45.0f * deltaTime; // 45 degrees per second around X
    app->rotationAngleY += 30.0f * deltaTime; // 30 degrees per second around Y
    if (app->rotationAngleX >= 360.0f) {
        app->rotationAngleX -= 360.0f;
    }
    if (app->rotationAngleY >= 360.0f) {
        app->rotationAngleY -= 360.0f;
    }

    // Update uniforms for each cube
    for (int i = 0; i < CUBE_COUNT; ++i) {
        updateCube(app, i);
    }
}

static void render(CubeApp* app)
{
    // Check if async texture upload is complete
    if (!app->textureUploadComplete) {
        bool isReady;
        gfxFenceGetStatus(app->textureUploadFence, &isReady);
        if (isReady) {
            LOG_INFO("Texture upload completed asynchronously!");
            app->textureUploadComplete = true;

            // Clean up resources now that upload is done
            gfxBufferDestroy(app->textureStagingBuffer);
            app->textureStagingBuffer = NULL;

            // Safe to destroy encoder now that GPU is done with it
            gfxCommandEncoderDestroy(app->textureUploadEncoder);
            app->textureUploadEncoder = NULL;
        } else {
            // Texture not ready yet - render clear color only
            LOG_INFO("Waiting for async texture upload...");
        }
    }

    PerFrameResources* frame = &app->frameResources[app->currentFrame];

    // Wait for previous frame to finish
    gfxFenceWait(frame->inFlightFence, GFX_TIMEOUT_INFINITE);
    gfxFenceReset(frame->inFlightFence);

    // Acquire next swapchain image
    uint32_t imageIndex;
    GfxResult result = gfxSwapchainAcquireNextImage(app->swapchain, GFX_TIMEOUT_INFINITE,
        frame->imageAvailableSemaphore, NULL, &imageIndex);
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

    GfxCommandEncoder encoder = frame->commandEncoder;
    if (gfxCommandEncoderBegin(encoder) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to begin command encoder");
        return;
    }

    // Begin render pass using pre-created render pass and framebuffer
    const GfxColor clearColor = { 0.1f, 0.2f, 0.3f, 1.0f };

    GfxRenderPassBeginDescriptor beginDesc = {
        .label = "Main Render Pass",
        .renderPass = app->renderPass,
        .framebuffer = app->framebuffers[imageIndex],
        .colorClearValues = &clearColor,
        .colorClearValueCount = 1,
        .depthClearValue = 1.0f,
        .stencilClearValue = 0
    };

    GfxRenderPassEncoder renderPass;
    if (gfxCommandEncoderBeginRenderPass(encoder, &beginDesc, &renderPass) == GFX_RESULT_SUCCESS) {

        // Set pipeline
        gfxRenderPassEncoderSetPipeline(renderPass, app->renderPipeline);

        // Set viewport and scissor to fill the entire render target
        GfxViewport viewport = {
            .x = 0.0f,
            .y = 0.0f,
            .width = (float)app->swapchainInfo.extent.width,
            .height = (float)app->swapchainInfo.extent.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f
        };
        GfxScissorRect scissor = {
            .origin = { .x = 0, .y = 0 },
            .extent = { .width = app->swapchainInfo.extent.width, .height = app->swapchainInfo.extent.height }
        };
        gfxRenderPassEncoderSetViewport(renderPass, &viewport);
        gfxRenderPassEncoderSetScissorRect(renderPass, &scissor);

        // Only draw if texture is loaded
        if (app->textureUploadComplete) {
            // Set vertex buffer
            gfxRenderPassEncoderSetVertexBuffer(renderPass, 0, app->vertexBuffer, 0, app->vertexBufferInfo.size);

            // Set index buffer
            gfxRenderPassEncoderSetIndexBuffer(renderPass, app->indexBuffer, GFX_INDEX_FORMAT_UINT16, 0, app->indexBufferInfo.size);

            // Calculate index count from buffer size
            uint32_t indexCount = app->indexBufferInfo.size / sizeof(uint16_t);

            // Bind texture (shared by all cubes)
            gfxRenderPassEncoderSetBindGroup(renderPass, 1, app->textureBindGroup, NULL, 0);

            // Draw CUBE_COUNT cubes at different positions
            for (int i = 0; i < CUBE_COUNT; ++i) {
                // Bind the specific cube's bind group (no dynamic offsets)
                gfxRenderPassEncoderSetBindGroup(renderPass, 0, frame->uniformBindGroups[i], NULL, 0);

                // Draw indexed
                gfxRenderPassEncoderDrawIndexed(renderPass, indexCount, 1, 0, 0, 0);
            }
        }

        // End render pass
        if (gfxRenderPassEncoderEnd(renderPass) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to end render pass");
            return;
        }
    }

    // Finish command encoding
    if (gfxCommandEncoderEnd(encoder) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to end command encoder");
        return;
    }

    // Submit commands with synchronization
    GfxSubmitDescriptor submitDescriptor = {
        .commandEncoders = &encoder,
        .commandEncoderCount = 1,
        .waitSemaphores = &frame->imageAvailableSemaphore,
        .waitSemaphoreCount = 1,
        .signalSemaphores = &app->renderFinishedSemaphores[imageIndex],
        .signalSemaphoreCount = 1,
        .signalFence = frame->inFlightFence
    };

    if (gfxQueueSubmit(app->queue, &submitDescriptor) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to submit command buffer to queue");
        return;
    }

    // Present with synchronization
    GfxPresentDescriptor presentDescriptor = {
        .sType = GFX_STRUCTURE_TYPE_PRESENT_DESCRIPTOR,
        .pNext = NULL,
        .waitSemaphores = &app->renderFinishedSemaphores[imageIndex],
        .waitSemaphoreCount = 1
    };

    if (gfxSwapchainPresent(app->swapchain, &presentDescriptor) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to present swapchain image");
        return;
    }

    // Move to next frame
    app->currentFrame = (app->currentFrame + 1) % app->swapchainInfo.imageCount;
}

#ifdef __ANDROID__
// ============================================================================
// Android Platform Implementation
// ============================================================================

// Android-specific event handlers
static void handleAppCommand(struct android_app* state, int32_t cmd)
{
    CubeApp* app = (CubeApp*)state->userData;

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
        } else {
            LOG_WARN("Gained focus but app not initialized yet");
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
        } else {
            LOG_WARN("Resumed but app not initialized yet");
        }
        break;

    case APP_CMD_WINDOW_RESIZED:
        if (state->window != NULL && app->instance) {
            uint32_t newWidth = ANativeWindow_getWidth(state->window);
            uint32_t newHeight = ANativeWindow_getHeight(state->window);
            handleResize(app, newWidth, newHeight);
        }
        break;
    }
}

static int32_t handleInput(struct android_app* state, AInputEvent* event)
{
    CubeApp* app = (CubeApp*)state->userData;
    (void)app; // Unused for now

    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
        // Handle touch input if needed
        return 1;
    }

    return 0;
}

// Returns false if loop should exit
static bool mainLoopIteration(CubeApp* app, struct android_app* state)
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
    CubeApp app = { 0 };
    app.androidApp = state;
    app.settings.backend = GFX_BACKEND_VULKAN;
    app.settings.msaaSampleCount = GFX_SAMPLE_COUNT_4;
    app.settings.vsync = true;
    app.animating = false;

    state->userData = &app;
    state->onAppCmd = handleAppCommand;
    state->onInputEvent = handleInput;

    LOG_INFO("=== GFX Cube Example (Android) ===");

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

void* cubeAppCreate(void* metalLayer, int width, int height)
{
    if (!metalLayer) {
        LOG_ERROR("Metal layer is NULL");
        return NULL;
    }

    LOG_INFO("Creating cube app with size %dx%d", width, height);

    CubeApp* app = (CubeApp*)calloc(1, sizeof(CubeApp));
    if (!app) {
        LOG_ERROR("Failed to allocate cube app");
        return NULL;
    }

    app->metalLayer = metalLayer;

    // Set up app settings
    app->settings.backend = GFX_BACKEND_VULKAN; // iOS uses Metal backend
    app->settings.msaaSampleCount = GFX_SAMPLE_COUNT_4;
    app->settings.vsync = true;
    app->windowWidth = width;
    app->windowHeight = height;

    LOG_INFO("=== GFX Cube Example (iOS) ===");

    // Initialize the cube app
    if (!init(app)) {
        LOG_ERROR("Failed to initialize cube app");
        free(app);
        return NULL;
    }

    LOG_INFO("Cube app initialized successfully");

    return app;
}

void cubeAppUpdate(void* appPtr, float deltaTime)
{
    if (!appPtr) {
        return;
    }

    CubeApp* app = (CubeApp*)appPtr;
    update(app, deltaTime);
}

void cubeAppRender(void* appPtr)
{
    if (!appPtr) {
        return;
    }

    CubeApp* app = (CubeApp*)appPtr;
    render(app);
}

void cubeAppResize(void* appPtr, int width, int height)
{
    if (!appPtr) {
        return;
    }

    CubeApp* app = (CubeApp*)appPtr;
    handleResize(app, (uint32_t)width, (uint32_t)height);
}

void cubeAppDestroy(void* appPtr)
{
    if (!appPtr) {
        return;
    }

    CubeApp* app = (CubeApp*)appPtr;

    LOG_INFO("Destroying cube app");
    cleanup(app);
    free(app);
    LOG_INFO("Cube app destroyed");
}

#else
// ============================================================================
// Desktop/Web Platform Implementation
// ============================================================================

// Returns false if loop should exit
static bool mainLoopIteration(CubeApp* app)
{
    // Desktop/web event loop
    if (glfwWindowShouldClose(app->window)) {
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
    CubeApp* app = (CubeApp*)userData;
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

static bool parseMsaa(const char* msaaStr, GfxSampleCount* outSampleCount)
{
    int samples = atoi(msaaStr);
    switch (samples) {
    case 1:
        *outSampleCount = GFX_SAMPLE_COUNT_1;
        break;
    case 2:
        *outSampleCount = GFX_SAMPLE_COUNT_2;
        break;
    case 4:
        *outSampleCount = GFX_SAMPLE_COUNT_4;
        break;
    case 8:
        *outSampleCount = GFX_SAMPLE_COUNT_8;
        break;
    case 16:
        *outSampleCount = GFX_SAMPLE_COUNT_16;
        break;
    case 32:
        *outSampleCount = GFX_SAMPLE_COUNT_32;
        break;
    case 64:
        *outSampleCount = GFX_SAMPLE_COUNT_64;
        break;
    default:
        LOG_ERROR("Invalid MSAA sample count: %d", samples);
        LOG_ERROR("Valid values: 1, 2, 4, 8, 16, 32, 64");
        return false;
    }
    return true;
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
    LOG_INFO("  --msaa [1|2|4|8]            Select MSAA sample count");
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
    settings->msaaSampleCount = GFX_SAMPLE_COUNT_4;
    settings->vsync = true; // VSync on by default

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            i++;
            if (!parseBackend(argv[i], &settings->backend)) {
                return false;
            }
        } else if (strcmp(argv[i], "--msaa") == 0 && i + 1 < argc) {
            i++;
            if (!parseMsaa(argv[i], &settings->msaaSampleCount)) {
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
    LOG_INFO("=== Cube Example with Unified Graphics API (C) ===");
    LOG_INFO("");

    CubeApp app = { 0 }; // Initialize all members to NULL/0

    // Parse command line arguments
    if (!parseArguments(argc, argv, &app.settings)) {
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

#endif // __ANDROID__
