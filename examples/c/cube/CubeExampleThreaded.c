#include <gfx/gfx.h>

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(__EMSCRIPTEN__)
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

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define CUBE_COUNT 12
#define COLOR_FORMAT GFX_FORMAT_B8G8R8A8_UNORM_SRGB
#define DEPTH_FORMAT GFX_FORMAT_DEPTH32_FLOAT

#if defined(__EMSCRIPTEN__) || defined(_WIN32) || defined(__APPLE__) || defined(__ANDROID__)
#define USE_THREADING 0
#else
#include <pthread.h>
#define USE_THREADING 1
#endif

// Logging macros
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
    GfxCommandEncoder clearEncoder;
    GfxCommandEncoder cubeEncoders[CUBE_COUNT]; // One per cube for parallel recording
    GfxCommandEncoder resolveEncoder;
    GfxCommandEncoder transitionEncoder; // For COLOR_ATTACHMENT->PRESENT_SRC transition (MSAA=1 only)
    GfxSemaphore imageAvailableSemaphore;
    GfxSemaphore clearFinishedSemaphore;
    GfxFence inFlightFence;
    GfxBindGroup uniformBindGroups[CUBE_COUNT]; // One per cube
} PerFrameResources;

// Forward declarations
typedef struct CubeApp CubeApp;

#if USE_THREADING
// Thread work data - one per cube
typedef struct {
    CubeApp* app;
    int cubeIndex;
    pthread_barrier_t* barrier;
} CubeThreadData;
#endif

typedef struct CubeApp {
    GLFWwindow* window;

    GfxInstance instance;
    GfxAdapter adapter;
    GfxAdapterInfo adapterInfo; // Cached adapter info
    GfxDevice device;
    GfxQueue queue;
    GfxSurface surface;
    GfxSurfaceInfo surfaceInfo; // Surface capabilities
    GfxSwapchain swapchain;
    GfxSwapchainInfo swapchainInfo;

    GfxBuffer vertexBuffer;
    GfxBuffer indexBuffer;
    GfxBufferInfo vertexBufferInfo;
    GfxBufferInfo indexBufferInfo;
    GfxShader vertexShader;
    GfxShader fragmentShader;
    GfxRenderPass clearRenderPass; // For clear pass (UNDEFINED->COLOR_ATTACHMENT)
    GfxRenderPass renderPass; // For cube passes (LOAD from COLOR_ATTACHMENT)
    GfxRenderPass transitionRenderPass; // For layout transition (COLOR_ATTACHMENT->PRESENT_SRC, MSAA=1 only)
    GfxRenderPass resolveRenderPass; // For final resolve pass (LOAD + resolve to swapchain)
    GfxRenderPipeline renderPipeline;
    GfxBindGroupLayout uniformBindGroupLayout;

    // Depth buffer
    GfxTexture depthTexture;
    GfxTextureView depthTextureView;

    // MSAA color buffer
    GfxTexture msaaColorTexture;
    GfxTextureView msaaColorTextureView;

    // Shared resources (not per-frame)
    GfxBuffer sharedUniformBuffer; // Single buffer for all frames
    size_t uniformAlignedSize; // Aligned size per frame

    // Framebuffers (one per swapchain image)
    PerFrameResources* frameResources;
    GfxSemaphore* renderFinishedSemaphores;
    GfxFramebuffer* framebuffers;
    uint32_t currentFrame;

    uint32_t windowWidth;
    uint32_t windowHeight;
    uint32_t previousWidth;
    uint32_t previousHeight;

    // State
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

#if USE_THREADING
    // Threading infrastructure
    pthread_t cubeThreads[CUBE_COUNT];
    CubeThreadData threadData[CUBE_COUNT];
    pthread_barrier_t recordBarrier;
    volatile bool threadsRunning;
    volatile uint32_t currentImageIndex;
#endif
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
static bool createRenderingResources(CubeApp* app);
static void destroyRenderingResources(CubeApp* app);
static bool createRenderPipeline(CubeApp* app);
static void destroyRenderPipeline(CubeApp* app);

static void updateCube(CubeApp* app, int cubeIndex);
static float getCurrentTime(void);
static GfxPlatformWindowHandle getPlatformWindowHandle(GLFWwindow* window);
static void* loadBinaryFile(const char* filepath, size_t* outSize);
static void* loadTextFile(const char* filepath, size_t* outSize);

static void recordCubeCommands(CubeApp* app, int cubeIndex, uint32_t imageIndex);
static void recordClearCommands(CubeApp* app, uint32_t imageIndex);
static void recordResolveCommands(CubeApp* app, uint32_t imageIndex);
static void recordLayoutTransition(CubeApp* app, uint32_t imageIndex);

#if USE_THREADING
static bool createThreading(CubeApp* app);
static void destroyThreading(CubeApp* app);
static void* cubeRecordThread(void* arg);
#endif

// Matrix/Vector math function declarations
static void matrixIdentity(Mat4* matrix);
static void matrixMultiply(Mat4* result, const Mat4* a, const Mat4* b);
static void matrixRotateX(Mat4* matrix, float angle);
static void matrixRotateY(Mat4* matrix, float angle);
static void matrixRotateZ(Mat4* matrix, float angle);
static void matrixPerspective(Mat4* matrix, float fov, float aspect, float nearPlane, float farPlane, GfxBackend backend);
static void matrixLookAt(Mat4* matrix, const Vec3* eye, const Vec3* center, const Vec3* up);
static bool vectorNormalize(Vec3* v);

// The public functions called from main
static bool init(CubeApp* app);
static void cleanup(CubeApp* app);
static void update(CubeApp* app, float deltaTime);
static void render(CubeApp* app);
static bool handleResize(CubeApp* app, uint32_t width, uint32_t height);

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
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

static bool createWindow(CubeApp* app, uint32_t width, uint32_t height)
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        LOG_ERROR("Failed to initialize GLFW");
        return false;
    }

    // Don't create OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    // Create window title with backend name
    const char* backendName = (app->settings.backend == GFX_BACKEND_VULKAN) ? "Vulkan" : "WebGPU";
    char windowTitle[128];
#if USE_THREADING
    snprintf(windowTitle, sizeof(windowTitle), "Cube Example Threaded - %s", backendName);
#else
    snprintf(windowTitle, sizeof(windowTitle), "Cube Example SingleThreaded - %s", backendName);
#endif

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
}

static void destroyWindow(CubeApp* app)
{
    if (app->window) {
        glfwDestroyWindow(app->window);
        app->window = NULL;
    }
    glfwTerminate();
}

static bool createGraphics(CubeApp* app)
{
    // Set up logging callback
    gfxSetLogCallback(logCallback, NULL);

    // Load the graphics backend BEFORE creating an instance
    // This is now decoupled - you load the backend API once at startup
    LOG_INFO("Loading graphics backend...");
    if (gfxLoadBackend(app->settings.backend) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to load any graphics backend");
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
    const char* deviceExtensions[] = { GFX_DEVICE_EXTENSION_SWAPCHAIN };
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
    GfxPlatformWindowHandle windowHandle = getPlatformWindowHandle(app->window);
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

    // Query surface capabilities to determine frames in flight
    if (gfxSurfaceGetInfo(app->surface, app->adapter, &app->surfaceInfo) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get surface info");
        return false;
    }

    LOG_INFO("Surface Info:");
    LOG_INFO("  Min Image Count: %u", app->surfaceInfo.minImageCount);
    LOG_INFO("  Max Image Count: %u", app->surfaceInfo.maxImageCount);
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
    // Allocate per-frame resources dynamically
    app->frameResources = (PerFrameResources*)calloc(app->swapchainInfo.imageCount, sizeof(PerFrameResources));
    if (!app->frameResources) {
        LOG_ERROR("Failed to allocate frame resources");
        return false;
    }

    // Create synchronization objects and command encoders for each frame in flight
    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        char label[64];
        PerFrameResources* frame = &app->frameResources[i];

        // Create semaphores
        snprintf(label, sizeof(label), "Image Available Semaphore %u", i);
        GfxSemaphoreDescriptor imageAvailableSemaphoreDesc = {
            .sType = GFX_STRUCTURE_TYPE_SEMAPHORE_DESCRIPTOR,
            .pNext = NULL,
            .label = label,
            .type = GFX_SEMAPHORE_TYPE_BINARY,
            .initialValue = 0
        };

        if (gfxDeviceCreateSemaphore(app->device, &imageAvailableSemaphoreDesc, &frame->imageAvailableSemaphore) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create image available semaphore %u", i);
            return false;
        }

        snprintf(label, sizeof(label), "Clear Finished Semaphore %u", i);
        GfxSemaphoreDescriptor clearFinishedSemaphoreDesc = {
            .sType = GFX_STRUCTURE_TYPE_SEMAPHORE_DESCRIPTOR,
            .pNext = NULL,
            .label = label,
            .type = GFX_SEMAPHORE_TYPE_BINARY,
            .initialValue = 0
        };

        if (gfxDeviceCreateSemaphore(app->device, &clearFinishedSemaphoreDesc, &frame->clearFinishedSemaphore) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create clear finished semaphore %u", i);
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

        // Create clear encoder
        snprintf(label, sizeof(label), "Clear Encoder Frame %u", i);
        GfxCommandEncoderDescriptor encoderDesc = {
            .sType = GFX_STRUCTURE_TYPE_COMMAND_ENCODER_DESCRIPTOR,
            .pNext = NULL,
            .label = label
        };

        if (gfxDeviceCreateCommandEncoder(app->device, &encoderDesc, &frame->clearEncoder) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create clear encoder %d", i);
            return false;
        }

        // Create command encoders - one per cube for parallel recording
        for (int cubeIdx = 0; cubeIdx < CUBE_COUNT; ++cubeIdx) {
            snprintf(label, sizeof(label), "Command Encoder Frame %u Cube %d", i, cubeIdx);
            GfxCommandEncoderDescriptor encoderDesc = {
                .sType = GFX_STRUCTURE_TYPE_COMMAND_ENCODER_DESCRIPTOR,
                .pNext = NULL,
                .label = label
            };

            if (gfxDeviceCreateCommandEncoder(app->device, &encoderDesc, &frame->cubeEncoders[cubeIdx]) != GFX_RESULT_SUCCESS) {
                LOG_ERROR("Failed to create command encoder %u cube %d", i, cubeIdx);
                return false;
            }
        }

        // Create resolve encoder
        snprintf(label, sizeof(label), "Resolve Encoder Frame %u", i);
        GfxCommandEncoderDescriptor resolveEncoderDesc = {
            .sType = GFX_STRUCTURE_TYPE_COMMAND_ENCODER_DESCRIPTOR,
            .pNext = NULL,
            .label = label
        };

        if (gfxDeviceCreateCommandEncoder(app->device, &resolveEncoderDesc, &frame->resolveEncoder) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create resolve encoder %u", i);
            return false;
        }

        // Create transition encoder (for COLOR_ATTACHMENT->PRESENT_SRC when MSAA=1)
        snprintf(label, sizeof(label), "Transition Encoder %u", i);
        GfxCommandEncoderDescriptor transitionEncoderDesc = {
            .sType = GFX_STRUCTURE_TYPE_COMMAND_ENCODER_DESCRIPTOR,
            .pNext = NULL,
            .label = label
        };

        if (gfxDeviceCreateCommandEncoder(app->device, &transitionEncoderDesc, &frame->transitionEncoder) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create transition encoder %u", i);
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
        if (frame->clearFinishedSemaphore) {
            gfxSemaphoreDestroy(frame->clearFinishedSemaphore);
            frame->clearFinishedSemaphore = NULL;
        }
        if (frame->imageAvailableSemaphore) {
            gfxSemaphoreDestroy(frame->imageAvailableSemaphore);
            frame->imageAvailableSemaphore = NULL;
        }
        if (frame->inFlightFence) {
            gfxFenceDestroy(frame->inFlightFence);
            frame->inFlightFence = NULL;
        }

        // Destroy command encoders
        if (frame->resolveEncoder) {
            gfxCommandEncoderDestroy(frame->resolveEncoder);
            frame->resolveEncoder = NULL;
        }
        if (frame->transitionEncoder) {
            gfxCommandEncoderDestroy(frame->transitionEncoder);
            frame->transitionEncoder = NULL;
        }
        for (int cubeIdx = 0; cubeIdx < CUBE_COUNT; ++cubeIdx) {
            if (frame->cubeEncoders[cubeIdx]) {
                gfxCommandEncoderDestroy(frame->cubeEncoders[cubeIdx]);
                frame->cubeEncoders[cubeIdx] = NULL;
            }
        }
        if (frame->clearEncoder) {
            gfxCommandEncoderDestroy(frame->clearEncoder);
            frame->clearEncoder = NULL;
        }
    }

    // Free the dynamically allocated array
    free(app->frameResources);
    app->frameResources = NULL;
}

static bool createSizeDependentResources(CubeApp* app, uint32_t width, uint32_t height)
{
    if (!createSwapchain(app, width, height)) {
        return false;
    }

    // Use actual swapchain dimensions (may differ from requested window size)
    uint32_t swapchainWidth = app->swapchainInfo.extent.width;
    uint32_t swapchainHeight = app->swapchainInfo.extent.height;

    if (!createRenderTargetTextures(app, swapchainWidth, swapchainHeight)) {
        return false;
    }

    if (!createRenderPass(app)) {
        return false;
    }

    if (!createFrameBuffers(app, swapchainWidth, swapchainHeight)) {
        return false;
    }

    return true;
}

static void destroySizeDependentResources(CubeApp* app)
{
    // Destroy framebuffers
    destroyFrameBuffers(app);

    // Destroy render passes (depend on swapchain format)
    destroyRenderPass(app);

    // Destroy render target textures (depth and MSAA)
    destroyRenderTargetTextures(app);

    // Destroy swapchain
    destroySwapchain(app);
}

static bool createRenderPass(CubeApp* app)
{
    // Create render pass (persistent, reusable across frames)
    // Define color attachment target - for cube passes that LOAD content
    GfxRenderPassColorAttachmentTarget colorTarget = {
        .format = app->swapchainInfo.format,
        .sampleCount = app->settings.msaaSampleCount,
        .ops = {
            .loadOp = GFX_LOAD_OP_LOAD,
            .storeOp = GFX_STORE_OP_STORE }, // STORE to preserve MSAA content across passes
        .finalLayout = GFX_TEXTURE_LAYOUT_COLOR_ATTACHMENT // Keep in COLOR_ATTACHMENT (renderPassFinal handles PRESENT_SRC)
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

    // Resolve target for intermediate passes (don't actually resolve, just keep framebuffer compatible)
    GfxRenderPassColorAttachmentTarget dummyResolveTarget = {
        .format = app->swapchainInfo.format,
        .sampleCount = GFX_SAMPLE_COUNT_1,
        .ops = {
            .loadOp = GFX_LOAD_OP_DONT_CARE,
            .storeOp = GFX_STORE_OP_DONT_CARE }, // Don't care - won't use this in intermediate passes
        .finalLayout = GFX_TEXTURE_LAYOUT_PRESENT_SRC
    };

    // Bundle color attachment - include resolve target for framebuffer compatibility
    GfxRenderPassColorAttachment colorAttachment = {
        .target = colorTarget,
        .resolveTarget = (app->settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? &dummyResolveTarget : NULL // Include for compatibility only when MSAA is used
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

    // Create clear render pass (loadOp=CLEAR)
    GfxRenderPassColorAttachmentTarget clearColorTarget = {
        .format = app->swapchainInfo.format,
        .sampleCount = app->settings.msaaSampleCount,
        .ops = {
            .loadOp = GFX_LOAD_OP_CLEAR,
            .storeOp = GFX_STORE_OP_STORE }, // STORE so it can be loaded by subsequent passes
        .finalLayout = GFX_TEXTURE_LAYOUT_COLOR_ATTACHMENT // Always COLOR_ATTACHMENT so cube passes can LOAD it
    };

    GfxRenderPassColorAttachment clearColorAttachment = {
        .target = clearColorTarget,
        .resolveTarget = (app->settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? &dummyResolveTarget : NULL // Include for framebuffer compatibility only when MSAA is used
    };

    GfxRenderPassDescriptor clearPassDesc = {
        .sType = GFX_STRUCTURE_TYPE_RENDER_PASS_DESCRIPTOR,
        .pNext = NULL,
        .label = "Clear Render Pass",
        .colorAttachments = &clearColorAttachment,
        .colorAttachmentCount = 1,
        .depthStencilAttachment = &depthAttachment
    };

    if (gfxDeviceCreateRenderPass(app->device, &clearPassDesc, &app->clearRenderPass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create clear render pass");
        return false;
    }

    // Create main render pass (loadOp=LOAD)

    GfxRenderPassDescriptor renderPassDesc = {
        .sType = GFX_STRUCTURE_TYPE_RENDER_PASS_DESCRIPTOR,
        .pNext = NULL,
        .label = "Cube Render Pass (LOAD)",
        .colorAttachments = &colorAttachment,
        .colorAttachmentCount = 1,
        .depthStencilAttachment = &depthAttachment
    };

    if (gfxDeviceCreateRenderPass(app->device, &renderPassDesc, &app->renderPass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create render pass");
        return false;
    }

    // Create transition render pass (for MSAA=1: COLOR_ATTACHMENT->PRESENT_SRC)
    if (app->settings.msaaSampleCount == GFX_SAMPLE_COUNT_1) {
        GfxRenderPassColorAttachmentTarget transitionColorTarget = {
            .format = app->swapchainInfo.format,
            .sampleCount = app->settings.msaaSampleCount,
            .ops = {
                .loadOp = GFX_LOAD_OP_LOAD, // Load existing content
                .storeOp = GFX_STORE_OP_STORE },
            .finalLayout = GFX_TEXTURE_LAYOUT_PRESENT_SRC
        };

        GfxRenderPassColorAttachment transitionColorAttachment = {
            .target = transitionColorTarget,
            .resolveTarget = NULL
        };

        // Depth attachment for transition pass - just to match framebuffer, not actually used
        GfxRenderPassDepthStencilAttachmentTarget transitionDepthTarget = {
            .format = DEPTH_FORMAT,
            .sampleCount = app->settings.msaaSampleCount,
            .depthOps = {
                .loadOp = GFX_LOAD_OP_DONT_CARE, // Don't care - not using depth
                .storeOp = GFX_STORE_OP_DONT_CARE },
            .stencilOps = { .loadOp = GFX_LOAD_OP_DONT_CARE, .storeOp = GFX_STORE_OP_DONT_CARE },
            .finalLayout = GFX_TEXTURE_LAYOUT_DEPTH_STENCIL_ATTACHMENT
        };

        GfxRenderPassDepthStencilAttachment transitionDepthAttachment = {
            .target = transitionDepthTarget,
            .resolveTarget = NULL
        };

        GfxRenderPassDescriptor transitionPassDesc = {
            .sType = GFX_STRUCTURE_TYPE_RENDER_PASS_DESCRIPTOR,
            .pNext = NULL,
            .label = "Layout Transition Pass",
            .colorAttachments = &transitionColorAttachment,
            .colorAttachmentCount = 1,
            .depthStencilAttachment = &transitionDepthAttachment
        };

        if (gfxDeviceCreateRenderPass(app->device, &transitionPassDesc, &app->transitionRenderPass) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create transition render pass");
            return false;
        }
    } else {
        app->transitionRenderPass = NULL;
    }

    // Create final resolve pass (loadOp=LOAD + resolve to swapchain)
    GfxRenderPassColorAttachment resolveColorAttachment = {
        .target = colorTarget, // Load from MSAA
        .resolveTarget = &resolveTarget // Resolve to swapchain
    };

    // Depth attachment for resolve pass - just LOAD (no clearing needed)
    GfxRenderPassDepthStencilAttachmentTarget resolveDepthTarget = {
        .format = DEPTH_FORMAT,
        .sampleCount = app->settings.msaaSampleCount,
        .depthOps = {
            .loadOp = GFX_LOAD_OP_LOAD, // Load existing depth
            .storeOp = GFX_STORE_OP_DONT_CARE }, // Don't need to store, just resolving color
        .stencilOps = { .loadOp = GFX_LOAD_OP_DONT_CARE, .storeOp = GFX_STORE_OP_DONT_CARE },
        .finalLayout = GFX_TEXTURE_LAYOUT_DEPTH_STENCIL_ATTACHMENT
    };

    GfxRenderPassDepthStencilAttachment resolveDepthAttachment = {
        .target = resolveDepthTarget,
        .resolveTarget = NULL
    };

    GfxRenderPassDescriptor resolvePassDesc = {
        .sType = GFX_STRUCTURE_TYPE_RENDER_PASS_DESCRIPTOR,
        .pNext = NULL,
        .label = "Resolve Render Pass",
        .colorAttachments = &resolveColorAttachment,
        .colorAttachmentCount = 1,
        .depthStencilAttachment = &resolveDepthAttachment
    };

    if (gfxDeviceCreateRenderPass(app->device, &resolvePassDesc, &app->resolveRenderPass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create resolve render pass");
        return false;
    }

    return true;
}

static void destroyRenderPass(CubeApp* app)
{
    if (app->resolveRenderPass) {
        gfxRenderPassDestroy(app->resolveRenderPass);
        app->resolveRenderPass = NULL;
    }
    if (app->transitionRenderPass) {
        gfxRenderPassDestroy(app->transitionRenderPass);
        app->transitionRenderPass = NULL;
    }
    if (app->clearRenderPass) {
        gfxRenderPassDestroy(app->clearRenderPass);
        app->clearRenderPass = NULL;
    }
    if (app->renderPass) {
        gfxRenderPassDestroy(app->renderPass);
        app->renderPass = NULL;
    }
}

static bool createSwapchain(CubeApp* app, uint32_t width, uint32_t height)
{
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
    LOG_INFO("Requested format: %d, Actual swapchain format: %d", COLOR_FORMAT, app->swapchainInfo.format);

    // Create render finished semaphores (one per swapchain image)
    app->renderFinishedSemaphores = (GfxSemaphore*)malloc(app->swapchainInfo.imageCount * sizeof(GfxSemaphore));
    if (!app->renderFinishedSemaphores) {
        LOG_ERROR("Failed to allocate renderFinishedSemaphores");
        return false;
    }
    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        app->renderFinishedSemaphores[i] = NULL;
    }

    for (uint32_t i = 0; i < app->swapchainInfo.imageCount; ++i) {
        char label[64];

        snprintf(label, sizeof(label), "Render Finished Semaphore Image %u", i);
        GfxSemaphoreDescriptor semaphoreDesc = {
            .sType = GFX_STRUCTURE_TYPE_SEMAPHORE_DESCRIPTOR,
            .pNext = NULL,
            .type = GFX_SEMAPHORE_TYPE_BINARY,
            .label = label,
        };

        if (gfxDeviceCreateSemaphore(app->device, &semaphoreDesc, &app->renderFinishedSemaphores[i]) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create render finished semaphore %u", i);
            return false;
        }
    }

    return true;
}

static void destroySwapchain(CubeApp* app)
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

    // Create MSAA color texture (is unused if app->settings.msaaSampleCount == 1)
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

    // Create MSAA color texture view (is unused if app->settings.msaaSampleCount == 1)
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
    // Allocate framebuffers array dynamically based on actual swapchain image count
    app->framebuffers = (GfxFramebuffer*)calloc(app->swapchainInfo.imageCount, sizeof(GfxFramebuffer));
    if (!app->framebuffers) {
        LOG_ERROR("Failed to allocate framebuffers");
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
            .renderPass = app->resolveRenderPass,
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
    // Create cube vertices (8 vertices for a cube)
    Vertex vertices[] = {
        // Front face
        { { -1.0f, -1.0f, 1.0f }, { 1.0f, 0.0f, 0.0f } }, // 0: Bottom-left
        { { 1.0f, -1.0f, 1.0f }, { 0.0f, 1.0f, 0.0f } }, // 1: Bottom-right
        { { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f } }, // 2: Top-right
        { { -1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 0.0f } }, // 3: Top-left

        // Back face
        { { -1.0f, -1.0f, -1.0f }, { 1.0f, 0.0f, 1.0f } }, // 4: Bottom-left
        { { 1.0f, -1.0f, -1.0f }, { 0.0f, 1.0f, 1.0f } }, // 5: Bottom-right
        { { 1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f } }, // 6: Top-right
        { { -1.0f, 1.0f, -1.0f }, { 0.5f, 0.5f, 0.5f } } // 7: Top-left
    };

    // Create cube indices (36 indices for 12 triangles)
    // All faces wound clockwise when viewed from outside
    uint16_t indices[] = {
        // Front face (Z+) - vertices 0,1,2,3
        0, 1, 2, 2, 3, 0,
        // Back face (Z-) - vertices 4,5,6,7 - FIXED
        5, 4, 7, 7, 6, 5,
        // Left face (X-) - vertices 4,0,3,7
        4, 0, 3, 3, 7, 4,
        // Right face (X+) - vertices 1,5,6,2
        1, 5, 6, 6, 2, 1,
        // Top face (Y+) - vertices 3,2,6,7
        3, 2, 6, 6, 7, 3,
        // Bottom face (Y-) - vertices 4,5,1,0
        4, 5, 1, 1, 0, 4
    };

    // Create vertex buffer
    GfxBufferDescriptor vertexBufferDesc = {
        .sType = GFX_STRUCTURE_TYPE_BUFFER_DESCRIPTOR,
        .pNext = NULL,
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
    gfxQueueWriteBuffer(app->queue, app->vertexBuffer, 0, vertices, sizeof(vertices));
    gfxQueueWriteBuffer(app->queue, app->indexBuffer, 0, indices, sizeof(indices));

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
    // Need space for CUBE_COUNT cubes per frame
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
        .type = GFX_BINDING_TYPE_UNIFORM_BUFFER,
        .count = 1,
        .uniformBuffer = {
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

    // Note: Bind groups are created in createPerFrameResources() after uniform buffer is ready

    return true;
}

static void destroyBindGroup(CubeApp* app)
{
    // Note: Bind groups are destroyed in destroyPerFrameResources
    // Only destroy the layout here
    if (app->uniformBindGroupLayout) {
        gfxBindGroupLayoutDestroy(app->uniformBindGroupLayout);
        app->uniformBindGroupLayout = NULL;
    }
}

static bool createShaders(CubeApp* app)
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
        { GFX_SHADER_SOURCE_SPIRV, "shaders/cube.vert.spv", "shaders/cube.frag.spv" },
        { GFX_SHADER_SOURCE_WGSL, "shaders/cube.vert.wgsl", "shaders/cube.frag.wgsl" }
    };

    for (size_t i = 0; i < ARRAY_SIZE(shaderFormats); ++i) {
        bool formatSupported = false;
        if (gfxDeviceSupportsShaderFormat(app->device, shaderFormats[i].format, &formatSupported) != GFX_RESULT_SUCCESS || !formatSupported) {
            continue;
        }

        LOG_INFO("Loading shaders: %s, %s", shaderFormats[i].vertexPath, shaderFormats[i].fragmentPath);

        if (shaderFormats[i].format == GFX_SHADER_SOURCE_SPIRV) {
            vertexCode = loadBinaryFile(shaderFormats[i].vertexPath, &vertexSize);
            fragmentCode = loadBinaryFile(shaderFormats[i].fragmentPath, &fragmentSize);
        } else {
            vertexCode = loadTextFile(shaderFormats[i].vertexPath, &vertexSize);
            fragmentCode = loadTextFile(shaderFormats[i].fragmentPath, &fragmentSize);
        }

        if (vertexCode && fragmentCode) {
            sourceType = shaderFormats[i].format;
            LOG_INFO("Successfully loaded shaders (vertex: %zu bytes, fragment: %zu bytes)",
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
        LOG_ERROR("Error: No supported shader format found or failed to load shaders");
        free(vertexCode);
        free(fragmentCode);
        return false;
    }

    // Create vertex shader
    GfxShaderDescriptor vertexShaderDesc = {
        .sType = GFX_STRUCTURE_TYPE_SHADER_DESCRIPTOR,
        .pNext = NULL,
        .label = "Cube Vertex Shader",
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

    // Create fragment shader
    GfxShaderDescriptor fragmentShaderDesc = {
        .sType = GFX_STRUCTURE_TYPE_SHADER_DESCRIPTOR,
        .pNext = NULL,
        .label = "Cube Fragment Shader",
        .sourceType = sourceType,
        .code = fragmentCode,
        .codeSize = fragmentSize,
        .entryPoint = "main"
    };

    if (gfxDeviceCreateShader(app->device, &fragmentShaderDesc, &app->fragmentShader) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create fragment shader");
        free(vertexCode);
        free(fragmentCode);
        return false;
    }

    free(vertexCode);
    free(fragmentCode);
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

bool createRenderingResources(CubeApp* app)
{
    if (!createGeometry(app)) {
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
    // Destroy render pipeline (depends on shaders and layouts)
    destroyRenderPipeline(app);

    // Destroy shaders
    destroyShaders(app);

    // Destroy bind groups and layouts
    destroyBindGroup(app);

    // Destroy uniform buffer
    destroyUniformBuffer(app);

    // Destroy geometry buffers
    destroyGeometry(app);
}

bool createRenderPipeline(CubeApp* app)
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
        .attributeCount = 2,
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
    GfxBindGroupLayout bindGroupLayouts[] = { app->uniformBindGroupLayout };

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
    translation.m[12] = -(float)CUBE_COUNT * 0.5 + (cubeIndex - 1) * 1.5f; // x offset: -3, 0, 3

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
    gfxQueueWriteBuffer(app->queue, app->sharedUniformBuffer, offset, &uniforms, sizeof(uniforms));
}

static GfxPlatformWindowHandle getPlatformWindowHandle(GLFWwindow* window)
{
    GfxPlatformWindowHandle handle = { 0 };
#if defined(__EMSCRIPTEN__)
    handle = gfxPlatformWindowHandleFromEmscripten("#canvas");
#elif defined(_WIN32)
    handle = gfxPlatformWindowHandleFromWin32(GetModuleHandle(NULL), glfwGetWin32Window(window));
#elif defined(__linux__)
    // handle = gfxPlatformWindowHandleFromXlib(glfwGetX11Display(), glfwGetX11Window(window));
    handle = gfxPlatformWindowHandleFromWayland(glfwGetWaylandDisplay(), glfwGetWaylandWindow(window));
#elif defined(__APPLE__)
    handle = gfxPlatformWindowHandleFromCocoaWindow(glfwGetCocoaWindow(window));
#endif
    return handle;
}

static float getCurrentTime(void)
{
#if defined(__EMSCRIPTEN__)
    return (float)emscripten_get_now() / 1000.0f;
#else
    return (float)glfwGetTime();
#endif
}

// Helper function to load binary files (SPIR-V shaders)
static void* loadBinaryFile(const char* filepath, size_t* outSize)
{
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
}

static void* loadTextFile(const char* filepath, size_t* outSize)
{
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
}

// Record commands for a single cube - called by worker threads OR main thread
static void recordCubeCommands(CubeApp* app, int cubeIndex, uint32_t imageIndex)
{
    PerFrameResources* frame = &app->frameResources[app->currentFrame];
    GfxCommandEncoder encoder = frame->cubeEncoders[cubeIndex];
    gfxCommandEncoderBegin(encoder);

    // Begin render pass using pre-created render pass and framebuffer
    GfxColor clearColor = { 0.1f, 0.2f, 0.3f, 1.0f };

    GfxRenderPassBeginDescriptor beginDesc = {
        .label = "Main Render Pass",
        .renderPass = app->renderPass,
        .framebuffer = app->framebuffers[imageIndex],
        .colorClearValues = &clearColor,
        .colorClearValueCount = 1,
        .depthClearValue = 1.0f,
        .stencilClearValue = 0
    };

    // Override load ops: first cube clears, others load
    // Note: This requires modifying the render pass dynamically, which isn't ideal
    // For now, we'll clear on first cube only by modifying the clear color alpha to signal

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

        // Set vertex buffer
        gfxRenderPassEncoderSetVertexBuffer(renderPass, 0, app->vertexBuffer, 0, app->vertexBufferInfo.size);

        // Set index buffer
        gfxRenderPassEncoderSetIndexBuffer(renderPass, app->indexBuffer, GFX_INDEX_FORMAT_UINT16, 0, app->indexBufferInfo.size);

        // Bind this cube's bind group and draw
        gfxRenderPassEncoderSetBindGroup(renderPass, 0, frame->uniformBindGroups[cubeIndex], NULL, 0);

        // Draw indexed (calculate index count from buffer size)
        uint32_t indexCount = app->indexBufferInfo.size / sizeof(uint16_t);
        gfxRenderPassEncoderDrawIndexed(renderPass, indexCount, 1, 0, 0, 0);

        // End render pass
        gfxRenderPassEncoderEnd(renderPass);
    }

    // Finish command encoding
    gfxCommandEncoderEnd(encoder);
}

// Record clear commands - just begin render pass with CLEAR and end
static void recordClearCommands(CubeApp* app, uint32_t imageIndex)
{
    PerFrameResources* frame = &app->frameResources[app->currentFrame];
    GfxCommandEncoder encoder = frame->clearEncoder;
    gfxCommandEncoderBegin(encoder);

    GfxColor clearColor = { 0.1f, 0.2f, 0.3f, 1.0f };

    GfxRenderPassBeginDescriptor beginDesc = {
        .label = "Clear Pass",
        .renderPass = app->clearRenderPass, // Use clear render pass
        .framebuffer = app->framebuffers[imageIndex],
        .colorClearValues = &clearColor,
        .colorClearValueCount = 1,
        .depthClearValue = 1.0f,
        .stencilClearValue = 0
    };

    GfxRenderPassEncoder renderPass;
    if (gfxCommandEncoderBeginRenderPass(encoder, &beginDesc, &renderPass) == GFX_RESULT_SUCCESS) {
        // Just clear - don't draw anything
        gfxRenderPassEncoderEnd(renderPass);
    }

    gfxCommandEncoderEnd(encoder);
}

// Record final resolve commands - just begin render pass with LOAD and let it resolve
static void recordResolveCommands(CubeApp* app, uint32_t imageIndex)
{
    PerFrameResources* frame = &app->frameResources[app->currentFrame];
    GfxCommandEncoder encoder = frame->resolveEncoder;
    gfxCommandEncoderBegin(encoder);

    GfxRenderPassBeginDescriptor beginDesc = {
        .label = "Final Resolve Pass",
        .renderPass = app->resolveRenderPass,
        .framebuffer = app->framebuffers[imageIndex],
        .colorClearValues = NULL, // Not used with LOAD
        .colorClearValueCount = 0,
        .depthClearValue = 1.0f,
        .stencilClearValue = 0
    };

    GfxRenderPassEncoder renderPass;
    if (gfxCommandEncoderBeginRenderPass(encoder, &beginDesc, &renderPass) == GFX_RESULT_SUCCESS) {
        // Just begin and end - the resolve happens automatically
        gfxRenderPassEncoderEnd(renderPass);
    }

    gfxCommandEncoderEnd(encoder);
}

// Record a simple layout transition from COLOR_ATTACHMENT to PRESENT_SRC (for MSAA=1 only)
static void recordLayoutTransition(CubeApp* app, uint32_t imageIndex)
{
    PerFrameResources* frame = &app->frameResources[app->currentFrame];
    GfxCommandEncoder encoder = frame->transitionEncoder;
    gfxCommandEncoderBegin(encoder);

    // Use an empty render pass to transition layout via initialLayout/finalLayout
    GfxRenderPassBeginDescriptor beginDesc = {
        .label = "Layout Transition Pass",
        .renderPass = app->transitionRenderPass,
        .framebuffer = app->framebuffers[imageIndex],
        .colorClearValues = NULL,
        .colorClearValueCount = 0,
        .depthClearValue = 1.0f,
        .stencilClearValue = 0
    };

    GfxRenderPassEncoder renderPass;
    if (gfxCommandEncoderBeginRenderPass(encoder, &beginDesc, &renderPass) == GFX_RESULT_SUCCESS) {
        // Empty pass - just begin and end to trigger layout transition
        gfxRenderPassEncoderEnd(renderPass);
    }

    gfxCommandEncoderEnd(encoder);
}

#if USE_THREADING
// Threading support functions
static bool createThreading(CubeApp* app)
{
    // Initialize barrier for CUBE_COUNT threads + 1 main thread
    if (pthread_barrier_init(&app->recordBarrier, NULL, CUBE_COUNT + 1) != 0) {
        LOG_ERROR("Failed to initialize pthread barrier");
        return false;
    }

    app->threadsRunning = true;

    // Create worker threads
    for (int i = 0; i < CUBE_COUNT; i++) {
        app->threadData[i].app = app;
        app->threadData[i].cubeIndex = i;
        app->threadData[i].barrier = &app->recordBarrier;

        if (pthread_create(&app->cubeThreads[i], NULL, cubeRecordThread, &app->threadData[i]) != 0) {
            LOG_ERROR("Failed to create cube thread %d", i);
            return false;
        }
    }

    LOG_INFO("Created %d worker threads for parallel command recording", CUBE_COUNT);
    return true;
}

static void destroyThreading(CubeApp* app)
{
    if (!app->threadsRunning) {
        return;
    }

    // Signal threads to exit
    app->threadsRunning = false;

    // Wake up threads
    pthread_barrier_wait(&app->recordBarrier);

    // Join threads
    for (int i = 0; i < CUBE_COUNT; i++) {
        pthread_join(app->cubeThreads[i], NULL);
    }

    pthread_barrier_destroy(&app->recordBarrier);
    LOG_INFO("Cleaned up worker threads");
}

static void* cubeRecordThread(void* arg)
{
    CubeThreadData* data = (CubeThreadData*)arg;
    LOG_INFO("Cube thread %d started", data->cubeIndex);

    while (data->app->threadsRunning) {
        // Wait for signal to start recording
        pthread_barrier_wait(data->barrier);

        if (!data->app->threadsRunning) {
            break;
        }

        // Record commands for this cube
        uint32_t imageIndex = data->app->currentImageIndex;
        recordCubeCommands(data->app, data->cubeIndex, imageIndex);

        // Wait for all threads to finish recording
        pthread_barrier_wait(data->barrier);
    }

    LOG_INFO("Cube thread %d exiting", data->cubeIndex);
    return NULL;
}
#endif

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

    // 5. Create per-frame resources (semaphores, fences, command encoders)
    if (!createPerFrameResources(app)) {
        LOG_ERROR("Failed to create per-frame resources");
        return false;
    }

    // 6. Create render pipeline
    if (!createRenderPipeline(app)) {
        LOG_ERROR("Failed to create render pipeline");
        return false;
    }

#if USE_THREADING
    // 7. Create threading infrastructure
    if (!createThreading(app)) {
        LOG_ERROR("Failed to create threading");
        return false;
    }
#endif

    // Initialize loop state
    app->currentFrame = 0;
    app->previousWidth = app->windowWidth;
    app->previousHeight = app->windowHeight;

    // Initialize animation state
    app->rotationAngleX = 0.0f;
    app->rotationAngleY = 0.0f;

    // Initialize FPS tracking
    app->fpsFrameCount = 0;
    app->fpsTimeAccumulator = 0.0f;
    app->fpsFrameTimeMin = FLT_MAX;
    app->fpsFrameTimeMax = 0.0f;

    app->lastFrameTime = getCurrentTime();

    LOG_INFO("Application initialized successfully!");
#if USE_THREADING
    LOG_INFO("Running with %d worker threads for parallel command recording", CUBE_COUNT);
#else
    LOG_INFO("Running in single-threaded mode");
#endif
    return true;
}

static void cleanup(CubeApp* app)
{
#if USE_THREADING
    // Clean up threading infrastructure first
    destroyThreading(app);
#endif

    // Wait for device to finish all GPU work before cleanup
    if (app->device) {
        gfxDeviceWaitIdle(app->device);
    }

    // Destroy in reverse order of creation for symmetry

    // 6. Destroy render pipeline (depends on render pass and resources)
    destroyRenderPipeline(app);

    // 5. Destroy per-frame resources (depends on uniform buffer and layouts)
    destroyPerFrameResources(app);

    // 4. Destroy rendering resources (textures, buffers, layouts)
    destroyRenderingResources(app);

    // 3. Destroy size-dependent resources (swapchain, framebuffers, render pass)
    destroySizeDependentResources(app);

    // 2. Destroy graphics context (surface, device, instance)
    destroyGraphics(app);

    // 1. Destroy window
    destroyWindow(app);
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

    // Record clear commands (always done by main thread)
    recordClearCommands(app, imageIndex);

#if USE_THREADING
    // Store image index for worker threads
    app->currentImageIndex = imageIndex;

    // Signal worker threads to start recording cube commands
    pthread_barrier_wait(&app->recordBarrier);

    // Wait for all threads to finish recording
    pthread_barrier_wait(&app->recordBarrier);

    // Submit clear encoder first (waits on imageAvailable, signals clearFinished)
    GfxSubmitDescriptor clearSubmit = {
        .sType = GFX_STRUCTURE_TYPE_SUBMIT_DESCRIPTOR,
        .pNext = NULL,
        .commandEncoders = &frame->clearEncoder,
        .commandEncoderCount = 1,
        .waitSemaphores = &frame->imageAvailableSemaphore,
        .waitStages = (const GfxPipelineStageFlags[]){ GFX_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT },
        .waitSemaphoreCount = 1,
        .signalSemaphores = &frame->clearFinishedSemaphore,
        .signalSemaphoreCount = 1,
        .signalFence = NULL
    };

    gfxQueueSubmit(app->queue, &clearSubmit);

    // Submit cube encoders (wait on clearFinished, signal renderFinished)
    GfxCommandEncoder cubeEncoderArray[CUBE_COUNT];
    for (int i = 0; i < CUBE_COUNT; i++) {
        cubeEncoderArray[i] = frame->cubeEncoders[i];
    }

    // Don't signal yet - either transition (no MSAA) or resolve (with MSAA) pass will signal
    GfxSubmitDescriptor cubesSubmit = {
        .sType = GFX_STRUCTURE_TYPE_SUBMIT_DESCRIPTOR,
        .pNext = NULL,
        .commandEncoders = cubeEncoderArray,
        .commandEncoderCount = CUBE_COUNT,
        .waitSemaphores = &frame->clearFinishedSemaphore,
        .waitStages = (const GfxPipelineStageFlags[]){ GFX_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT },
        .waitSemaphoreCount = 1,
        .signalSemaphores = NULL,
        .signalSemaphoreCount = 0,
        .signalFence = NULL
    };

    gfxQueueSubmit(app->queue, &cubesSubmit);

    // Record and submit layout transition (only when MSAA == 1)
    if (app->settings.msaaSampleCount == GFX_SAMPLE_COUNT_1) {
        recordLayoutTransition(app, imageIndex);

        GfxSubmitDescriptor transitionSubmit = {
            .sType = GFX_STRUCTURE_TYPE_SUBMIT_DESCRIPTOR,
            .pNext = NULL,
            .commandEncoders = &frame->transitionEncoder,
            .commandEncoderCount = 1,
            .waitSemaphores = NULL,
            .waitSemaphoreCount = 0,
            .signalSemaphores = &app->renderFinishedSemaphores[imageIndex],
            .signalSemaphoreCount = 1,
            .signalFence = frame->inFlightFence
        };

        gfxQueueSubmit(app->queue, &transitionSubmit);
    }

    // Record and submit final resolve pass (only when MSAA > 1)
    if (app->settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) {
        recordResolveCommands(app, imageIndex);

        GfxSubmitDescriptor resolveSubmit = {
            .sType = GFX_STRUCTURE_TYPE_SUBMIT_DESCRIPTOR,
            .pNext = NULL,
            .commandEncoders = &frame->resolveEncoder,
            .commandEncoderCount = 1,
            .waitSemaphores = NULL,
            .waitSemaphoreCount = 0,
            .signalSemaphores = &app->renderFinishedSemaphores[imageIndex],
            .signalSemaphoreCount = 1,
            .signalFence = frame->inFlightFence
        };

        gfxQueueSubmit(app->queue, &resolveSubmit);
    }
#else
    // Non-threaded path for WebGPU - record all cubes in ONE render pass
    GfxCommandEncoder encoder = frame->cubeEncoders[0];
    gfxCommandEncoderBegin(encoder);

    // Begin render pass with CLEAR (not using separate clear pass for WebGPU)
    GfxColor clearColor = { 0.1f, 0.2f, 0.3f, 1.0f };
    GfxRenderPassBeginDescriptor beginDesc = {
        .label = "Main Render Pass (All Cubes)",
        .renderPass = app->clearRenderPass, // Use clear pass to properly clear
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

        // Set viewport and scissor
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

        // Set vertex buffer once
        gfxRenderPassEncoderSetVertexBuffer(renderPass, 0, app->vertexBuffer, 0, app->vertexBufferInfo.size);

        // Set index buffer once
        gfxRenderPassEncoderSetIndexBuffer(renderPass, app->indexBuffer, GFX_INDEX_FORMAT_UINT16, 0, app->indexBufferInfo.size);

        // Calculate index count from buffer size
        uint32_t indexCount = app->indexBufferInfo.size / sizeof(uint16_t);

        // Draw all 3 cubes in ONE render pass
        for (int cubeIdx = 0; cubeIdx < CUBE_COUNT; cubeIdx++) {
            gfxRenderPassEncoderSetBindGroup(renderPass, 0, frame->uniformBindGroups[cubeIdx], NULL, 0);
            gfxRenderPassEncoderDrawIndexed(renderPass, indexCount, 1, 0, 0, 0);
        }

        gfxRenderPassEncoderEnd(renderPass);
    }

    gfxCommandEncoderEnd(encoder);

    // Submit single command buffer for WebGPU
    GfxSubmitDescriptor submitDescriptor = {
        .sType = GFX_STRUCTURE_TYPE_SUBMIT_DESCRIPTOR,
        .pNext = NULL,
        .commandEncoders = &encoder,
        .commandEncoderCount = 1,
        .waitSemaphores = &frame->imageAvailableSemaphore,
        .waitStages = (const GfxPipelineStageFlags[]){ GFX_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT },
        .waitSemaphoreCount = 1,
        .signalSemaphores = &app->renderFinishedSemaphores[imageIndex],
        .signalSemaphoreCount = 1,
        .signalFence = frame->inFlightFence
    };

    gfxQueueSubmit(app->queue, &submitDescriptor);
#endif

    // Present with synchronization
    GfxPresentDescriptor presentDescriptor = {
        .sType = GFX_STRUCTURE_TYPE_PRESENT_DESCRIPTOR,
        .pNext = NULL,
        .waitSemaphores = &app->renderFinishedSemaphores[imageIndex],
        .waitSemaphoreCount = 1
    };
    gfxSwapchainPresent(app->swapchain, &presentDescriptor);

    // Move to next frame
    app->currentFrame = (app->currentFrame + 1) % app->swapchainInfo.imageCount;
}

static bool handleResize(CubeApp* app, uint32_t width, uint32_t height)
{
    LOG_INFO("Resizing to %dx%d", width, height);

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

// Returns false if loop should exit
static bool mainLoopIteration(CubeApp* app)
{
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

// Command line argument parsing helpers
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

// Parse command line arguments
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
    LOG_INFO("=== Threaded Cube Example with Parallel Command Recording (C) ===");
    LOG_INFO("");

    CubeApp app = { 0 }; // Initialize all members to NULL/0

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
