// Threaded Cube Example (C++ with C API) - Parallel Command Recording with ThreadPool
// Uses C API (gfx/gfx.h) but C++ language features and ThreadPool for threading

#include <gfx/gfx.h>

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

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

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

// Constants
constexpr uint32_t WINDOW_WIDTH = 800;
constexpr uint32_t WINDOW_HEIGHT = 600;
constexpr uint32_t CUBE_COUNT = 12;
constexpr GfxFormat COLOR_FORMAT = GFX_FORMAT_B8G8R8A8_UNORM_SRGB;
constexpr GfxFormat DEPTH_FORMAT = GFX_FORMAT_DEPTH32_FLOAT;
#if defined(__EMSCRIPTEN__)
constexpr bool USE_THREADING = false;
#else
constexpr bool USE_THREADING = true;
#endif

// ThreadPool class for parallel command recording
class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads)
        : stop(false)
    {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });

                        if (stop && tasks.empty()) {
                            return;
                        }

                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    template <class F>
    auto Enqueue(F&& f) -> std::future<void>
    {
        auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
        std::future<void> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (stop) {
                throw std::runtime_error("Enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

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
        LOG_INFO("%s", message);
        break;
    }
}

// Math types for improved API clarity and type safety
struct Vec3 {
    float x, y, z;
};

struct Mat4 {
    std::array<std::array<float, 4>, 4> m;
};

// Vertex structure for cube
struct Vertex {
    Vec3 position;
    Vec3 color;
};

// Uniform buffer structure for transformations
struct UniformData {
    Mat4 model; // Model matrix
    Mat4 view; // View matrix
    Mat4 projection; // Projection matrix
};

// Settings structure for command-line arguments
struct Settings {
    GfxBackend backend;
    GfxSampleCount msaaSampleCount;
    bool vsync;
};

// Per-frame resources for threaded rendering
struct PerFrameResources {
    // Synchronization
    GfxSemaphore imageAvailableSemaphore = nullptr;
    GfxSemaphore clearFinishedSemaphore = nullptr;
    GfxFence inFlightFence = nullptr;

    // Command encoders
    GfxCommandEncoder clearEncoder = nullptr;
    GfxCommandEncoder resolveEncoder = nullptr;
    GfxCommandEncoder transitionEncoder = nullptr; // For COLOR_ATTACHMENT->PRESENT_SRC (MSAA=1)
    std::vector<GfxCommandEncoder> cubeEncoders; // One per cube

    // Bind groups
    std::vector<GfxBindGroup> uniformBindGroups; // One per cube
};

// Utility namespace for file loading and other helpers
namespace util {
std::vector<uint8_t> loadBinaryFile(const char* filepath);
std::string loadTextFile(const char* filepath);
} // namespace util

// Math namespace for matrix and vector operations
namespace math {
void matrixIdentity(Mat4& matrix);
void matrixPerspective(Mat4& matrix, float fov, float aspect, float nearPlane, float farPlane, GfxBackend backend);
void matrixLookAt(Mat4& matrix, const Vec3& eye, const Vec3& center, const Vec3& up);
void matrixRotateX(Mat4& matrix, float angle);
void matrixRotateY(Mat4& matrix, float angle);
void matrixMultiply(Mat4& result, const Mat4& a, const Mat4& b);
bool vectorNormalize(Vec3& v);
} // namespace math

// Main application class
class CubeApp {
public:
    explicit CubeApp(const Settings& settings);
    ~CubeApp() = default;

    bool init();
    void run();
    void cleanup();

private:
    bool createWindow(uint32_t width, uint32_t height);
    void destroyWindow();
    bool createGraphics();
    void destroyGraphics();
    bool createSizeDependentResources(uint32_t width, uint32_t height);
    void destroySizeDependentResources();
    bool createSwapchain(uint32_t width, uint32_t height);
    void destroySwapchain();
    bool createTextures(uint32_t width, uint32_t height);
    void destroyTextures();
    bool createRenderPass();
    void destroyRenderPass();
    bool createFramebuffers(uint32_t width, uint32_t height);
    void destroyFramebuffers();
    bool createGeometry();
    void destroyGeometry();
    bool createUniformBuffer();
    void destroyUniformBuffer();
    bool createShaders();
    void destroyShaders();
    bool createRenderingResources();
    void destroyRenderingResources();
    bool createRenderPipeline();
    void destroyRenderPipeline();
    bool createPerFrameResources();
    void destroyPerFrameResources();

    // Main loop
    void updateCube(int cubeIndex);
    void update(float deltaTime);
    void updateFPS(float deltaTime);
    void recordClearCommands(uint32_t imageIndex);
    void recordCubeCommands(int cubeIndex, uint32_t imageIndex);
    void recordResolveCommands(uint32_t imageIndex);
    void recordLayoutTransition(uint32_t imageIndex);
    void render();
    bool handleResize(uint32_t width, uint32_t height);
    float getCurrentTime();
    bool mainLoopIteration();
#if defined(__EMSCRIPTEN__)
    static void emscriptenMainLoop(void* userData);
#endif

    // Platform-specific
    GfxPlatformWindowHandle getPlatformWindowHandle();

    // Window callbacks
    static void errorCallback(int error, const char* description);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

private:
    Settings settings;

    // Public for GLFW callbacks
    GLFWwindow* window = nullptr;

    // Graphics resources
    GfxInstance instance = nullptr;
    GfxAdapter adapter = nullptr;
    GfxAdapterInfo adapterInfo = {};
    GfxDevice device = nullptr;
    GfxQueue queue = nullptr;
    GfxSurface surface = nullptr;
    GfxSwapchain swapchain = nullptr;
    GfxSwapchainInfo swapchainInfo = {};
    GfxSurfaceInfo surfaceInfo = {};
    uint32_t framesInFlight = 3; // Dynamic: set based on surface capabilities

    GfxBuffer vertexBuffer = nullptr;
    GfxBuffer indexBuffer = nullptr;
    GfxBufferInfo vertexBufferInfo = {};
    GfxBufferInfo indexBufferInfo = {};
    GfxShader vertexShader = nullptr;
    GfxShader fragmentShader = nullptr;
    GfxRenderPass clearRenderPass = nullptr;
    GfxRenderPass renderPass = nullptr;
    GfxRenderPass transitionRenderPass = nullptr; // For layout transition (MSAA=1)
    GfxRenderPass resolveRenderPass = nullptr;
    GfxRenderPipeline renderPipeline = nullptr;
    GfxBindGroupLayout uniformBindGroupLayout = nullptr;

    // Depth and MSAA resources
    GfxTexture depthTexture = nullptr;
    GfxTextureView depthTextureView = nullptr;
    GfxTexture msaaColorTexture = nullptr;
    GfxTextureView msaaColorTextureView = nullptr;

    // Uniform buffers
    GfxBuffer sharedUniformBuffer = nullptr;
    size_t uniformAlignedSize = 0;

    // Per-frame resources
    std::vector<GfxFramebuffer> framebuffers;
    std::vector<PerFrameResources> frameResources;
    std::vector<GfxSemaphore> renderFinishedSemaphores;
    uint32_t currentFrame = 0;

    uint32_t previousWidth = WINDOW_WIDTH;
    uint32_t previousHeight = WINDOW_HEIGHT;
    uint32_t windowWidth = WINDOW_WIDTH;
    uint32_t windowHeight = WINDOW_HEIGHT;

    // Animation state
    float rotationAngleX = 0.0f;
    float rotationAngleY = 0.0f;
    float elapsedTime = 0.0f;
    float lastFrameTime = 0.0f;

    // FPS tracking
    uint32_t fpsFrameCount = 0;
    float fpsTimeAccumulator = 0.0f;
    float fpsFrameTimeMin = FLT_MAX;
    float fpsFrameTimeMax = 0.0f;

    // Threading
    std::unique_ptr<ThreadPool> threadPool;
    std::atomic<uint32_t> currentImageIndex{ 0 };
};

CubeApp::CubeApp(const Settings& settings)
    : settings(settings)
{
}

bool CubeApp::init()
{
    // 1. Create window
    if (!createWindow(windowWidth, windowHeight)) {
        return false;
    }

    // 2. Create graphics context
    if (!createGraphics()) {
        return false;
    }

    // 3. Create size-dependent resources
    if (!createSizeDependentResources(windowWidth, windowHeight)) {
        return false;
    }

    // 4. Create rendering resources (geometry, uniform buffer, shaders, pipeline)
    if (!createRenderingResources()) {
        return false;
    }

    // 5. Create per-frame resources (sync objects, encoders, bind groups)
    if (!createPerFrameResources()) {
        return false;
    }

    // Initialize thread pool if using threading
    if constexpr (USE_THREADING) {
        threadPool = std::make_unique<ThreadPool>(CUBE_COUNT);
        LOG_INFO("Created ThreadPool with %u worker threads for parallel command recording", CUBE_COUNT);
    }

    previousWidth = windowWidth;
    previousHeight = windowHeight;
    lastFrameTime = getCurrentTime();

    // Initialize FPS tracking
    fpsFrameCount = 0;
    fpsTimeAccumulator = 0.0f;
    fpsFrameTimeMin = FLT_MAX;
    fpsFrameTimeMax = 0.0f;

    LOG_INFO("Application initialized successfully!");
    if constexpr (USE_THREADING) {
        LOG_INFO("Running with ThreadPool (%u threads) for parallel command recording", CUBE_COUNT);
    } else {
        LOG_INFO("Running in single-threaded mode");
    }
    LOG_INFO("Press ESC to exit\n");

    return true;
}

void CubeApp::run()
{
#if defined(__EMSCRIPTEN__)
    // Note: emscripten_set_main_loop_arg returns immediately and never blocks
    // Cleanup happens in emscriptenMainLoop when the loop exits
    // Execution continues in the browser event loop
    emscripten_set_main_loop_arg(CubeApp::emscriptenMainLoop, this, 0, 1);
#else
    while (mainLoopIteration()) {
        // Continue running
    }
#endif
}

void CubeApp::cleanup()
{
    // Wait for device to finish
    if (device) {
        gfxDeviceWaitIdle(device);
    }

    // Destroy threadPool before other resources
    threadPool.reset();

    // Destroy resources in reverse order of creation
    // 5. Destroy per-frame resources
    destroyPerFrameResources();

    // 4. Destroy size-dependent resources
    destroySizeDependentResources();

    // 3. Destroy rendering resources
    destroyRenderingResources();

    // 2. Destroy graphics resources
    destroyGraphics();

    // 1. Destroy window
    destroyWindow();
}

bool CubeApp::createWindow(uint32_t width, uint32_t height)
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        LOG_ERROR("Failed to initialize GLFW");
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    std::string backendName = (settings.backend == GFX_BACKEND_VULKAN) ? "Vulkan" : "WebGPU";
    std::string threadingInfo = USE_THREADING ? " (Threaded) - Parallel Command Recording" : "";
    std::string title = "Cube Example (C++ ThreadPool) - " + backendName + threadingInfo;

    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        LOG_ERROR("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }

    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);

    return true;
}

void CubeApp::destroyWindow()
{
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

bool CubeApp::createGraphics()
{
    // Set up logging callback
    gfxSetLogCallback(logCallback, nullptr);

    auto result = gfxLoadBackend(settings.backend);
    if (result != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to load graphics backend: %s", gfxResultToString(result));
        return false;
    }

    LOG_INFO("Loading graphics backend...");
    if (gfxLoadBackend(settings.backend) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to load graphics backend");
        return false;
    }
    LOG_INFO("Graphics backend loaded successfully!");

    // Create instance
    const char* instanceExtensions[] = { GFX_INSTANCE_EXTENSION_SURFACE, GFX_INSTANCE_EXTENSION_DEBUG };
    GfxInstanceDescriptor instanceDesc = {
        .backend = settings.backend,
        .applicationName = "Cube Example (C++ ThreadPool)",
        .applicationVersion = 1,
        .enabledExtensions = instanceExtensions,
        .enabledExtensionCount = ARRAY_SIZE(instanceExtensions)
    };

    if (gfxCreateInstance(&instanceDesc, &instance) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create graphics instance");
        return false;
    }

    // Get adapter
    GfxAdapterDescriptor adapterDesc = {
        .preference = GFX_ADAPTER_PREFERENCE_HIGH_PERFORMANCE
    };

    if (gfxInstanceRequestAdapter(instance, &adapterDesc, &adapter) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get graphics adapter");
        return false;
    }

    gfxAdapterGetInfo(adapter, &adapterInfo);
    LOG_INFO("Using adapter: %s", adapterInfo.name);
    LOG_INFO("  Vendor ID: 0x%04X, Device ID: 0x%04X", adapterInfo.vendorID, adapterInfo.deviceID);
    LOG_INFO("  Type: %s", (adapterInfo.adapterType == GFX_ADAPTER_TYPE_DISCRETE_GPU ? "Discrete GPU" : adapterInfo.adapterType == GFX_ADAPTER_TYPE_INTEGRATED_GPU ? "Integrated GPU"
                                   : adapterInfo.adapterType == GFX_ADAPTER_TYPE_CPU                                                                               ? "CPU"
                                                                                                                                                                   : "Unknown"));
    LOG_INFO("  Backend: %s", (adapterInfo.backend == GFX_BACKEND_VULKAN ? "Vulkan" : "WebGPU"));

    // Create device
    const char* deviceExtensions[] = { GFX_DEVICE_EXTENSION_SWAPCHAIN };
    GfxDeviceDescriptor deviceDesc = {
        .label = "Main Device",
        .enabledExtensions = deviceExtensions,
        .enabledExtensionCount = ARRAY_SIZE(deviceExtensions)
    };
    if (gfxAdapterCreateDevice(adapter, &deviceDesc, &device) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create device");
        return false;
    }

    // Query device limits
    GfxDeviceLimits limits;
    if (gfxDeviceGetLimits(device, &limits) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get device limits");
        return false;
    }
    LOG_INFO("Device Limits:");
    LOG_INFO("  Min Uniform Buffer Offset Alignment: %u bytes", limits.minUniformBufferOffsetAlignment);
    LOG_INFO("  Max Buffer Size: %llu bytes", (unsigned long long)limits.maxBufferSize);

    // Get queue
    if (gfxDeviceGetQueue(device, &queue) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get device queue");
        return false;
    }

    // Create surface
    GfxPlatformWindowHandle windowHandle = getPlatformWindowHandle();
    GfxSurfaceDescriptor surfaceDesc = {
        .label = "Main Surface",
        .windowHandle = windowHandle
    };

    if (gfxInstanceCreateSurface(instance, &surfaceDesc, &surface) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create surface");
        return false;
    }

    // Query surface capabilities to determine frames in flight
    if (gfxSurfaceGetInfo(surface, adapter, &surfaceInfo) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get surface info");
        return false;
    }

    LOG_INFO("  Image Count: min %u, max %u", surfaceInfo.minImageCount, surfaceInfo.maxImageCount);
    LOG_INFO("  Extent: min (%ux%u), max (%ux%u)", surfaceInfo.minExtent.width, surfaceInfo.minExtent.height, surfaceInfo.maxExtent.width, surfaceInfo.maxExtent.height);

    // Calculate frames in flight based on surface capabilities
    // Use min image count, but clamp to reasonable values (2-4 is typical)
    framesInFlight = surfaceInfo.minImageCount;
    if (framesInFlight < 2) {
        framesInFlight = 2;
    }
    if (framesInFlight > 4) {
        framesInFlight = 4;
    }
    LOG_INFO("Frames in flight: %u", framesInFlight);

    return true;
}

void CubeApp::destroyGraphics()
{
    if (surface) {
        gfxSurfaceDestroy(surface);
        surface = nullptr;
    }
    if (queue) {
        queue = nullptr; // Queue doesn't need explicit destruction
    }
    if (device) {
        gfxDeviceDestroy(device);
        device = nullptr;
    }
    if (adapter) {
        adapter = nullptr; // Adapter doesn't need explicit destruction
    }
    if (instance) {
        gfxInstanceDestroy(instance);
        instance = nullptr;
    }

    LOG_INFO("Unloading graphics backend...");
    gfxUnloadBackend(settings.backend);
}

bool CubeApp::createSwapchain(uint32_t width, uint32_t height)
{
    GfxSwapchainDescriptor swapchainDesc = {
        .label = "Main Swapchain",
        .surface = surface,
        .extent = { width, height },
        .format = COLOR_FORMAT,
        .usage = GFX_TEXTURE_USAGE_RENDER_ATTACHMENT,
        .presentMode = settings.vsync ? GFX_PRESENT_MODE_FIFO : GFX_PRESENT_MODE_IMMEDIATE,
        .imageCount = framesInFlight
    };

    if (gfxDeviceCreateSwapchain(device, &swapchainDesc, &swapchain) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create swapchain");
        return false;
    }

    if (gfxSwapchainGetInfo(swapchain, &swapchainInfo) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get swapchain info");
        return false;
    }

    LOG_INFO("Swapchain created: %ux%u, format: %d", swapchainInfo.extent.width, swapchainInfo.extent.height, swapchainInfo.format);

    // Create render finished semaphores (one per swapchain image)
    renderFinishedSemaphores.resize(swapchainInfo.imageCount, nullptr);
    GfxSemaphoreDescriptor semaphoreDesc = {
        .sType = GFX_STRUCTURE_TYPE_SEMAPHORE_DESCRIPTOR,
        .pNext = nullptr,
        .type = GFX_SEMAPHORE_TYPE_BINARY
    };

    for (uint32_t i = 0; i < swapchainInfo.imageCount; ++i) {
        std::string labelStr = "Render Finished Semaphore Image " + std::to_string(i);
        semaphoreDesc.label = labelStr.c_str();
        if (gfxDeviceCreateSemaphore(device, &semaphoreDesc, &renderFinishedSemaphores[i]) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create render finished semaphore %u", i);
            return false;
        }
    }

    return true;
}

void CubeApp::destroySwapchain()
{
    // Clean up render finished semaphores
    for (auto& semaphore : renderFinishedSemaphores) {
        if (semaphore) {
            gfxSemaphoreDestroy(semaphore);
            semaphore = nullptr;
        }
    }
    renderFinishedSemaphores.clear();

    if (swapchain) {
        gfxSwapchainDestroy(swapchain);
        swapchain = nullptr;
    }
}

bool CubeApp::createTextures(uint32_t width, uint32_t height)
{
    // Create depth texture
    GfxTextureDescriptor depthTextureDesc = {
        .label = "Depth Buffer",
        .type = GFX_TEXTURE_TYPE_2D,
        .size = { .width = width, .height = height, .depth = 1 },
        .arrayLayerCount = 1,
        .mipLevelCount = 1,
        .sampleCount = settings.msaaSampleCount,
        .format = DEPTH_FORMAT,
        .usage = GFX_TEXTURE_USAGE_RENDER_ATTACHMENT
    };

    if (gfxDeviceCreateTexture(device, &depthTextureDesc, &depthTexture) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create depth texture");
        return false;
    }

    GfxTextureViewDescriptor depthViewDesc = {
        .label = "Depth Buffer View",
        .viewType = GFX_TEXTURE_VIEW_TYPE_2D,
        .format = DEPTH_FORMAT,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };

    if (gfxTextureCreateView(depthTexture, &depthViewDesc, &depthTextureView) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create depth texture view");
        return false;
    }

    // Create MSAA color texture
    GfxTextureDescriptor msaaColorTextureDesc = {
        .label = "MSAA Color Buffer",
        .type = GFX_TEXTURE_TYPE_2D,
        .size = { .width = width, .height = height, .depth = 1 },
        .arrayLayerCount = 1,
        .mipLevelCount = 1,
        .sampleCount = settings.msaaSampleCount,
        .format = swapchainInfo.format,
        .usage = GFX_TEXTURE_USAGE_RENDER_ATTACHMENT
    };

    if (gfxDeviceCreateTexture(device, &msaaColorTextureDesc, &msaaColorTexture) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create MSAA color texture");
        return false;
    }

    GfxTextureViewDescriptor msaaColorViewDesc = {
        .label = "MSAA Color Buffer View",
        .viewType = GFX_TEXTURE_VIEW_TYPE_2D,
        .format = swapchainInfo.format,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1
    };

    if (gfxTextureCreateView(msaaColorTexture, &msaaColorViewDesc, &msaaColorTextureView) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create MSAA color texture view");
        return false;
    }

    return true;
}

void CubeApp::destroyTextures()
{
    if (msaaColorTextureView) {
        gfxTextureViewDestroy(msaaColorTextureView);
        msaaColorTextureView = nullptr;
    }
    if (msaaColorTexture) {
        gfxTextureDestroy(msaaColorTexture);
        msaaColorTexture = nullptr;
    }
    if (depthTextureView) {
        gfxTextureViewDestroy(depthTextureView);
        depthTextureView = nullptr;
    }
    if (depthTexture) {
        gfxTextureDestroy(depthTexture);
        depthTexture = nullptr;
    }
}

bool CubeApp::createFramebuffers(uint32_t width, uint32_t height)
{
    // Resize framebuffers vector to match swapchain image count
    framebuffers.resize(swapchainInfo.imageCount, nullptr);

    for (uint32_t i = 0; i < swapchainInfo.imageCount; ++i) {
        GfxTextureView backbuffer = nullptr;
        if (gfxSwapchainGetTextureView(swapchain, i, &backbuffer) != GFX_RESULT_SUCCESS || !backbuffer) {
            LOG_ERROR("Failed to get swapchain image view %u", i);
            return false;
        }

        GfxFramebufferAttachment fbColorAttachment = {
            .view = (settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? msaaColorTextureView : backbuffer,
            .resolveTarget = (settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? backbuffer : nullptr
        };

        GfxFramebufferAttachment fbDepthAttachment = {
            .view = depthTextureView,
            .resolveTarget = nullptr
        };

        std::string labelStr = "Framebuffer " + std::to_string(i);
        const char* label = labelStr.c_str();

        GfxFramebufferDescriptor fbDesc = {
            .label = label,
            .renderPass = resolveRenderPass,
            .colorAttachments = &fbColorAttachment,
            .colorAttachmentCount = 1,
            .depthStencilAttachment = fbDepthAttachment,
            .extent = { width, height }
        };

        if (gfxDeviceCreateFramebuffer(device, &fbDesc, &framebuffers[i]) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create framebuffer %u", i);
            return false;
        }
    }

    return true;
}

void CubeApp::destroyFramebuffers()
{
    for (auto& fb : framebuffers) {
        if (fb) {
            gfxFramebufferDestroy(fb);
            fb = nullptr;
        }
    }
}

bool CubeApp::createRenderPass()
{
    // Clear pass target
    GfxRenderPassColorAttachmentTarget clearColorTarget = {
        .format = swapchainInfo.format,
        .sampleCount = settings.msaaSampleCount,
        .ops = { .loadOp = GFX_LOAD_OP_CLEAR, .storeOp = GFX_STORE_OP_STORE },
        .finalLayout = GFX_TEXTURE_LAYOUT_COLOR_ATTACHMENT // Always COLOR_ATTACHMENT so subsequent passes can LOAD
    };

    // Load pass target
    GfxRenderPassColorAttachmentTarget colorTarget = {
        .format = swapchainInfo.format,
        .sampleCount = settings.msaaSampleCount,
        .ops = { .loadOp = GFX_LOAD_OP_LOAD, .storeOp = GFX_STORE_OP_STORE },
        .finalLayout = GFX_TEXTURE_LAYOUT_COLOR_ATTACHMENT // Keep in COLOR_ATTACHMENT (renderPassFinal handles PRESENT_SRC)
    };

    // Resolve target
    GfxRenderPassColorAttachmentTarget resolveTarget = {
        .format = swapchainInfo.format,
        .sampleCount = GFX_SAMPLE_COUNT_1,
        .ops = { .loadOp = GFX_LOAD_OP_DONT_CARE, .storeOp = GFX_STORE_OP_STORE },
        .finalLayout = GFX_TEXTURE_LAYOUT_PRESENT_SRC
    };

    GfxRenderPassColorAttachmentTarget dummyResolveTarget = {
        .format = swapchainInfo.format,
        .sampleCount = GFX_SAMPLE_COUNT_1,
        .ops = { .loadOp = GFX_LOAD_OP_DONT_CARE, .storeOp = GFX_STORE_OP_DONT_CARE },
        .finalLayout = GFX_TEXTURE_LAYOUT_PRESENT_SRC
    };

    GfxRenderPassDepthStencilAttachmentTarget depthTarget = {
        .format = DEPTH_FORMAT,
        .sampleCount = settings.msaaSampleCount,
        .depthOps = { .loadOp = GFX_LOAD_OP_CLEAR, .storeOp = GFX_STORE_OP_DONT_CARE },
        .stencilOps = { .loadOp = GFX_LOAD_OP_DONT_CARE, .storeOp = GFX_STORE_OP_DONT_CARE },
        .finalLayout = GFX_TEXTURE_LAYOUT_DEPTH_STENCIL_ATTACHMENT
    };

    GfxRenderPassDepthStencilAttachment depthAttachment = {
        .target = depthTarget,
        .resolveTarget = nullptr
    };

    // Clear render pass
    GfxRenderPassColorAttachment clearColorAttachment = {
        .target = clearColorTarget,
        .resolveTarget = (settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? &dummyResolveTarget : nullptr
    };

    GfxRenderPassDescriptor clearPassDesc = {
        .label = "Clear Render Pass",
        .colorAttachments = &clearColorAttachment,
        .colorAttachmentCount = 1,
        .depthStencilAttachment = &depthAttachment
    };

    if (gfxDeviceCreateRenderPass(device, &clearPassDesc, &clearRenderPass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create clear render pass");
        return false;
    }

    // Main render pass
    GfxRenderPassColorAttachment colorAttachment = {
        .target = colorTarget,
        .resolveTarget = (settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? &dummyResolveTarget : nullptr
    };

    GfxRenderPassDescriptor renderPassDesc = {
        .label = "Cube Render Pass (LOAD)",
        .colorAttachments = &colorAttachment,
        .colorAttachmentCount = 1,
        .depthStencilAttachment = &depthAttachment
    };

    if (gfxDeviceCreateRenderPass(device, &renderPassDesc, &renderPass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create render pass");
        return false;
    }

    // Transition render pass (MSAA=1: COLOR_ATTACHMENT->PRESENT_SRC)
    if (settings.msaaSampleCount == GFX_SAMPLE_COUNT_1) {
        GfxRenderPassColorAttachmentTarget transitionColorTarget = {
            .format = swapchainInfo.format,
            .sampleCount = settings.msaaSampleCount,
            .ops = { .loadOp = GFX_LOAD_OP_LOAD, .storeOp = GFX_STORE_OP_STORE },
            .finalLayout = GFX_TEXTURE_LAYOUT_PRESENT_SRC
        };

        GfxRenderPassColorAttachment transitionColorAttachment = {
            .target = transitionColorTarget,
            .resolveTarget = nullptr
        };

        // Depth attachment for framebuffer compatibility (not actually used)
        GfxRenderPassDepthStencilAttachmentTarget transitionDepthTarget = {
            .format = DEPTH_FORMAT,
            .sampleCount = settings.msaaSampleCount,
            .depthOps = { .loadOp = GFX_LOAD_OP_DONT_CARE, .storeOp = GFX_STORE_OP_DONT_CARE },
            .stencilOps = { .loadOp = GFX_LOAD_OP_DONT_CARE, .storeOp = GFX_STORE_OP_DONT_CARE },
            .finalLayout = GFX_TEXTURE_LAYOUT_DEPTH_STENCIL_ATTACHMENT
        };

        GfxRenderPassDepthStencilAttachment transitionDepthAttachment = {
            .target = transitionDepthTarget,
            .resolveTarget = nullptr
        };

        GfxRenderPassDescriptor transitionPassDesc = {
            .label = "Layout Transition Pass",
            .colorAttachments = &transitionColorAttachment,
            .colorAttachmentCount = 1,
            .depthStencilAttachment = &transitionDepthAttachment
        };

        if (gfxDeviceCreateRenderPass(device, &transitionPassDesc, &transitionRenderPass) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create transition render pass");
            return false;
        }
    }

    // Resolve render pass
    GfxRenderPassColorAttachment resolveColorAttachment = {
        .target = colorTarget,
        .resolveTarget = (settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) ? &resolveTarget : nullptr
    };

    GfxRenderPassDepthStencilAttachmentTarget resolveDepthTarget = {
        .format = DEPTH_FORMAT,
        .sampleCount = this->settings.msaaSampleCount,
        .depthOps = { .loadOp = GFX_LOAD_OP_LOAD, .storeOp = GFX_STORE_OP_DONT_CARE },
        .stencilOps = { .loadOp = GFX_LOAD_OP_DONT_CARE, .storeOp = GFX_STORE_OP_DONT_CARE },
        .finalLayout = GFX_TEXTURE_LAYOUT_DEPTH_STENCIL_ATTACHMENT
    };

    GfxRenderPassDepthStencilAttachment resolveDepthAttachment = {
        .target = resolveDepthTarget,
        .resolveTarget = nullptr
    };

    GfxRenderPassDescriptor resolvePassDesc = {
        .label = "Resolve Render Pass",
        .colorAttachments = &resolveColorAttachment,
        .colorAttachmentCount = 1,
        .depthStencilAttachment = &resolveDepthAttachment
    };

    if (gfxDeviceCreateRenderPass(device, &resolvePassDesc, &resolveRenderPass) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create resolve render pass");
        return false;
    }

    return true;
}

void CubeApp::destroyRenderPass()
{
    if (resolveRenderPass) {
        gfxRenderPassDestroy(resolveRenderPass);
        resolveRenderPass = nullptr;
    }
    if (transitionRenderPass) {
        gfxRenderPassDestroy(transitionRenderPass);
        transitionRenderPass = nullptr;
    }
    if (clearRenderPass) {
        gfxRenderPassDestroy(clearRenderPass);
        clearRenderPass = nullptr;
    }
    if (renderPass) {
        gfxRenderPassDestroy(renderPass);
        renderPass = nullptr;
    }
}

bool CubeApp::createSizeDependentResources(uint32_t width, uint32_t height)
{
    if (!createSwapchain(width, height)) {
        return false;
    }

    uint32_t swapchainWidth = swapchainInfo.extent.width;
    uint32_t swapchainHeight = swapchainInfo.extent.height;

    if (!createTextures(swapchainWidth, swapchainHeight)) {
        return false;
    }
    if (!createRenderPass()) {
        return false;
    }
    if (!createFramebuffers(swapchainWidth, swapchainHeight)) {
        return false;
    }

    return true;
}

void CubeApp::destroySizeDependentResources()
{
    destroyFramebuffers();
    destroyRenderPass();
    destroyTextures();
    destroySwapchain();
}

bool CubeApp::createGeometry()
{
    // Cube vertices
    Vertex vertices[] = {
        // Front face
        { { -1.0f, -1.0f, 1.0f }, { 1.0f, 0.0f, 0.0f } },
        { { 1.0f, -1.0f, 1.0f }, { 0.0f, 1.0f, 0.0f } },
        { { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f } },
        { { -1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 0.0f } },
        // Back face
        { { -1.0f, -1.0f, -1.0f }, { 1.0f, 0.0f, 1.0f } },
        { { 1.0f, -1.0f, -1.0f }, { 0.0f, 1.0f, 1.0f } },
        { { 1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f } },
        { { -1.0f, 1.0f, -1.0f }, { 0.5f, 0.5f, 0.5f } }
    };

    // Cube indices
    uint16_t indices[] = {
        0, 1, 2, 2, 3, 0, // Front
        5, 4, 7, 7, 6, 5, // Back
        4, 0, 3, 3, 7, 4, // Left
        1, 5, 6, 6, 2, 1, // Right
        3, 2, 6, 6, 7, 3, // Top
        4, 5, 1, 1, 0, 4 // Bottom
    };

    // Create vertex buffer
    GfxBufferDescriptor vertexBufferDesc = {
        .label = "Cube Vertices",
        .size = sizeof(vertices),
        .usage = GFX_FLAGS(GFX_BUFFER_USAGE_VERTEX | GFX_BUFFER_USAGE_COPY_DST),
        .memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL
    };

    if (gfxDeviceCreateBuffer(device, &vertexBufferDesc, &vertexBuffer) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create vertex buffer");
        return false;
    }
    gfxBufferGetInfo(vertexBuffer, &vertexBufferInfo);

    // Create index buffer
    GfxBufferDescriptor indexBufferDesc = {
        .label = "Cube Indices",
        .size = sizeof(indices),
        .usage = GFX_FLAGS(GFX_BUFFER_USAGE_INDEX | GFX_BUFFER_USAGE_COPY_DST),
        .memoryProperties = GFX_MEMORY_PROPERTY_DEVICE_LOCAL
    };

    if (gfxDeviceCreateBuffer(device, &indexBufferDesc, &indexBuffer) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create index buffer");
        return false;
    }
    gfxBufferGetInfo(indexBuffer, &indexBufferInfo);

    // Upload data
    gfxQueueWriteBuffer(queue, vertexBuffer, 0, vertices, sizeof(vertices));
    gfxQueueWriteBuffer(queue, indexBuffer, 0, indices, sizeof(indices));

    return true;
}

void CubeApp::destroyGeometry()
{
    if (indexBuffer) {
        gfxBufferDestroy(indexBuffer);
        indexBuffer = nullptr;
    }
    if (vertexBuffer) {
        gfxBufferDestroy(vertexBuffer);
        vertexBuffer = nullptr;
    }
}

bool CubeApp::createUniformBuffer()
{
    GfxDeviceLimits limits;
    if (gfxDeviceGetLimits(device, &limits) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to get device limits");
        return false;
    }

    size_t uniformSize = sizeof(UniformData);
    uniformAlignedSize = gfxAlignUp(uniformSize, limits.minUniformBufferOffsetAlignment);
    size_t totalBufferSize = uniformAlignedSize * framesInFlight * CUBE_COUNT;

    GfxBufferDescriptor uniformBufferDesc = {
        .label = "Shared Transform Uniforms",
        .size = totalBufferSize,
        .usage = GFX_FLAGS(GFX_BUFFER_USAGE_UNIFORM | GFX_BUFFER_USAGE_COPY_DST),
        .memoryProperties = GFX_FLAGS(GFX_MEMORY_PROPERTY_HOST_VISIBLE | GFX_MEMORY_PROPERTY_HOST_COHERENT)
    };

    if (gfxDeviceCreateBuffer(device, &uniformBufferDesc, &sharedUniformBuffer) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create shared uniform buffer");
        return false;
    }

    // Create bind group layout
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
        .label = "Uniform Bind Group Layout",
        .entries = &uniformLayoutEntry,
        .entryCount = 1
    };

    if (gfxDeviceCreateBindGroupLayout(device, &uniformLayoutDesc, &uniformBindGroupLayout) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create uniform bind group layout");
        return false;
    }

    return true;
}

void CubeApp::destroyUniformBuffer()
{
    if (uniformBindGroupLayout) {
        gfxBindGroupLayoutDestroy(uniformBindGroupLayout);
        uniformBindGroupLayout = nullptr;
    }

    if (sharedUniformBuffer) {
        gfxBufferDestroy(sharedUniformBuffer);
        sharedUniformBuffer = nullptr;
    }
}

bool CubeApp::createShaders()
{
    // Data-driven shader loading - try formats in order of preference
    struct ShaderFormat {
        GfxShaderSourceType format;
        const char* vertexPath;
        const char* fragmentPath;
    };

    const ShaderFormat shaderFormats[] = {
        { GFX_SHADER_SOURCE_SPIRV, "shaders/cube.vert.spv", "shaders/cube.frag.spv" },
        { GFX_SHADER_SOURCE_WGSL, "shaders/cube.vert.wgsl", "shaders/cube.frag.wgsl" }
    };

    GfxShaderSourceType sourceType;
    std::vector<uint8_t> vertexShaderData;
    std::vector<uint8_t> fragmentShaderData;
    bool shadersLoaded = false;

    for (const auto& format : shaderFormats) {
        LOG_INFO("Loading shaders: %s, %s", format.vertexPath, format.fragmentPath);

        if (format.format == GFX_SHADER_SOURCE_SPIRV) {
            vertexShaderData = util::loadBinaryFile(format.vertexPath);
            fragmentShaderData = util::loadBinaryFile(format.fragmentPath);
        } else {
            auto vertexText = util::loadTextFile(format.vertexPath);
            auto fragmentText = util::loadTextFile(format.fragmentPath);
            vertexShaderData.assign(vertexText.begin(), vertexText.end());
            fragmentShaderData.assign(fragmentText.begin(), fragmentText.end());
        }

        if (!vertexShaderData.empty() && !fragmentShaderData.empty()) {
            sourceType = format.format;
            LOG_INFO("Successfully loaded shaders (vertex: %zu bytes, fragment: %zu bytes)", vertexShaderData.size(), fragmentShaderData.size());
            shadersLoaded = true;
            break;
        }

        // Failed to load this format, clear and try next
        LOG_ERROR("Failed to load shaders for format");
        vertexShaderData.clear();
        fragmentShaderData.clear();
    }

    if (!shadersLoaded) {
        LOG_ERROR("No supported shader format found or failed to load shaders");
        return false;
    }

    // Create vertex shader
    GfxShaderDescriptor vertexShaderDesc = {
        .label = "Cube Vertex Shader",
        .sourceType = sourceType,
        .code = vertexShaderData.data(),
        .codeSize = vertexShaderData.size(),
        .entryPoint = "main"
    };

    if (gfxDeviceCreateShader(device, &vertexShaderDesc, &vertexShader) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create vertex shader");
        return false;
    }

    // Create fragment shader
    GfxShaderDescriptor fragmentShaderDesc = {
        .label = "Cube Fragment Shader",
        .sourceType = sourceType,
        .code = fragmentShaderData.data(),
        .codeSize = fragmentShaderData.size(),
        .entryPoint = "main"
    };

    if (gfxDeviceCreateShader(device, &fragmentShaderDesc, &fragmentShader) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create fragment shader");
        return false;
    }

    return true;
}

void CubeApp::destroyShaders()
{
    if (fragmentShader) {
        gfxShaderDestroy(fragmentShader);
        fragmentShader = nullptr;
    }
    if (vertexShader) {
        gfxShaderDestroy(vertexShader);
        vertexShader = nullptr;
    }
}

bool CubeApp::createRenderingResources()
{
    // 1. Create geometry (vertex/index buffers)
    if (!createGeometry()) {
        return false;
    }

    // 2. Create uniform buffer and layout
    if (!createUniformBuffer()) {
        return false;
    }

    // 3. Create shaders
    if (!createShaders()) {
        return false;
    }

    // 4. Create render pipeline
    if (!createRenderPipeline()) {
        return false;
    }

    return true;
}

void CubeApp::destroyRenderingResources()
{
    // Destroy pipeline
    destroyRenderPipeline();

    // Destroy shaders
    destroyShaders();

    // Destroy uniform buffer
    destroyUniformBuffer();

    // Destroy geometry
    destroyGeometry();
}

bool CubeApp::createRenderPipeline()
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

    GfxVertexBufferLayout vertexBufferLayout = {
        .arrayStride = sizeof(Vertex),
        .attributes = attributes,
        .attributeCount = ARRAY_SIZE(attributes),
        .stepMode = GFX_VERTEX_STEP_MODE_VERTEX
    };

    GfxVertexState vertexState = {
        .module = vertexShader,
        .entryPoint = "main",
        .buffers = &vertexBufferLayout,
        .bufferCount = 1
    };

    GfxColorTargetState colorTarget = {
        .format = swapchainInfo.format,
        .blend = nullptr,
        .writeMask = GFX_COLOR_WRITE_MASK_ALL
    };

    GfxFragmentState fragmentState = {
        .module = fragmentShader,
        .entryPoint = "main",
        .targets = &colorTarget,
        .targetCount = 1
    };

    GfxPrimitiveState primitiveState = {
        .topology = GFX_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .stripIndexFormat = GFX_INDEX_FORMAT_UNDEFINED,
        .frontFace = GFX_FRONT_FACE_COUNTER_CLOCKWISE,
        .cullMode = GFX_CULL_MODE_BACK,
        .polygonMode = GFX_POLYGON_MODE_FILL
    };

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

    GfxBindGroupLayout bindGroupLayouts[] = { uniformBindGroupLayout };

    GfxRenderPipelineDescriptor pipelineDesc = {
        .label = "Cube Render Pipeline",
        .renderPass = renderPass,
        .vertex = &vertexState,
        .fragment = &fragmentState,
        .primitive = &primitiveState,
        .depthStencil = &depthStencilState,
        .sampleCount = this->settings.msaaSampleCount,
        .bindGroupLayouts = bindGroupLayouts,
        .bindGroupLayoutCount = ARRAY_SIZE(bindGroupLayouts)
    };

    if (gfxDeviceCreateRenderPipeline(device, &pipelineDesc, &renderPipeline) != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to create render pipeline");
        return false;
    }

    return true;
}

void CubeApp::destroyRenderPipeline()
{
    if (renderPipeline) {
        gfxRenderPipelineDestroy(renderPipeline);
        renderPipeline = nullptr;
    }
}

bool CubeApp::createPerFrameResources()
{
    frameResources.resize(framesInFlight);

    for (size_t i = 0; i < framesInFlight; ++i) {
        auto& frame = frameResources[i];

        std::string labelStr;
        labelStr = "Image Available Semaphore " + std::to_string(i);
        GfxSemaphoreDescriptor imageAvailableSemaphoreDesc = {
            .label = labelStr.c_str(),
            .type = GFX_SEMAPHORE_TYPE_BINARY,
            .initialValue = 0
        };
        if (gfxDeviceCreateSemaphore(device, &imageAvailableSemaphoreDesc, &frame.imageAvailableSemaphore) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create image available semaphore %zu", i);
            return false;
        }

        labelStr = "Clear Finished Semaphore " + std::to_string(i);
        GfxSemaphoreDescriptor clearFinishedSemaphoreDesc = {
            .label = labelStr.c_str(),
            .type = GFX_SEMAPHORE_TYPE_BINARY,
            .initialValue = 0
        };
        if (gfxDeviceCreateSemaphore(device, &clearFinishedSemaphoreDesc, &frame.clearFinishedSemaphore) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create clear finished semaphore %zu", i);
            return false;
        }

        // Create fence
        labelStr = "In Flight Fence " + std::to_string(i);
        GfxFenceDescriptor fenceDesc = {
            .label = labelStr.c_str(),
            .signaled = true
        };
        if (gfxDeviceCreateFence(device, &fenceDesc, &frame.inFlightFence) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create in flight fence %zu", i);
            return false;
        }

        // Create clear encoder
        labelStr = "Clear Encoder Frame " + std::to_string(i);
        GfxCommandEncoderDescriptor encoderDesc = { .label = labelStr.c_str() };
        if (gfxDeviceCreateCommandEncoder(device, &encoderDesc, &frame.clearEncoder) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create clear encoder %zu", i);
            return false;
        }

        // Create cube encoders
        frame.cubeEncoders.resize(CUBE_COUNT);
        for (size_t cubeIdx = 0; cubeIdx < CUBE_COUNT; ++cubeIdx) {
            labelStr = "Command Encoder Frame " + std::to_string(i) + " Cube " + std::to_string(cubeIdx);
            GfxCommandEncoderDescriptor encoderDesc = { .label = labelStr.c_str() };
            if (gfxDeviceCreateCommandEncoder(device, &encoderDesc, &frame.cubeEncoders[cubeIdx]) != GFX_RESULT_SUCCESS) {
                LOG_ERROR("Failed to create command encoder %zu cube %zu", i, cubeIdx);
                return false;
            }
        }

        // Create resolve encoder
        labelStr = "Resolve Encoder Frame " + std::to_string(i);
        GfxCommandEncoderDescriptor resolveEncoderDesc = { .label = labelStr.c_str() };
        if (gfxDeviceCreateCommandEncoder(device, &resolveEncoderDesc, &frame.resolveEncoder) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create resolve encoder %zu", i);
            return false;
        }

        // Create transition encoder
        labelStr = "Transition Encoder Frame " + std::to_string(i);
        GfxCommandEncoderDescriptor transitionEncoderDesc = { .label = labelStr.c_str() };
        if (gfxDeviceCreateCommandEncoder(device, &transitionEncoderDesc, &frame.transitionEncoder) != GFX_RESULT_SUCCESS) {
            LOG_ERROR("Failed to create transition encoder %zu", i);
            return false;
        }

        // Create bind groups
        frame.uniformBindGroups.resize(CUBE_COUNT);
        for (size_t cubeIdx = 0; cubeIdx < CUBE_COUNT; ++cubeIdx) {
            std::string labelStr = "Uniform Bind Group Frame " + std::to_string(i) + " Cube " + std::to_string(cubeIdx);
            const char* label = labelStr.c_str();

            GfxBindGroupEntry uniformEntry = {
                .binding = 0,
                .type = GFX_BIND_GROUP_ENTRY_TYPE_BUFFER,
                .resource = {
                    .buffer = {
                        .buffer = sharedUniformBuffer,
                        .offset = (i * CUBE_COUNT + cubeIdx) * uniformAlignedSize,
                        .size = sizeof(UniformData) } }
            };

            GfxBindGroupDescriptor uniformBindGroupDesc = {
                .label = label,
                .layout = uniformBindGroupLayout,
                .entries = &uniformEntry,
                .entryCount = 1
            };

            if (gfxDeviceCreateBindGroup(device, &uniformBindGroupDesc, &frame.uniformBindGroups[cubeIdx]) != GFX_RESULT_SUCCESS) {
                LOG_ERROR("Failed to create uniform bind group %zu cube %zu", i, cubeIdx);
                return false;
            }
        }
    }

    return true;
}

void CubeApp::destroyPerFrameResources()
{
    // Wait for device idle
    if (device) {
        gfxDeviceWaitIdle(device);
    }

    // Destroy per-frame resources
    for (auto& frame : frameResources) {
        // Destroy bind groups
        for (auto& bindGroup : frame.uniformBindGroups) {
            if (bindGroup) {
                gfxBindGroupDestroy(bindGroup);
                bindGroup = nullptr;
            }
        }

        // Destroy cube encoders
        for (auto& encoder : frame.cubeEncoders) {
            if (encoder) {
                gfxCommandEncoderDestroy(encoder);
                encoder = nullptr;
            }
        }

        // Destroy clear and resolve encoders
        if (frame.clearEncoder) {
            gfxCommandEncoderDestroy(frame.clearEncoder);
            frame.clearEncoder = nullptr;
        }
        if (frame.resolveEncoder) {
            gfxCommandEncoderDestroy(frame.resolveEncoder);
            frame.resolveEncoder = nullptr;
        }
        if (frame.transitionEncoder) {
            gfxCommandEncoderDestroy(frame.transitionEncoder);
            frame.transitionEncoder = nullptr;
        }

        // Destroy synchronization objects
        if (frame.imageAvailableSemaphore) {
            gfxSemaphoreDestroy(frame.imageAvailableSemaphore);
            frame.imageAvailableSemaphore = nullptr;
        }
        if (frame.clearFinishedSemaphore) {
            gfxSemaphoreDestroy(frame.clearFinishedSemaphore);
            frame.clearFinishedSemaphore = nullptr;
        }
        if (frame.inFlightFence) {
            gfxFenceDestroy(frame.inFlightFence);
            frame.inFlightFence = nullptr;
        }
    }

    frameResources.clear();
}

void CubeApp::updateCube(int cubeIndex)
{
    UniformData uniforms{};

    // Create rotation matrices (combine X and Y rotations)
    // Each cube rotates slightly differently
    Mat4 rotX, rotY, tempModel, translation;
    math::matrixIdentity(tempModel);
    math::matrixRotateX(rotX, (rotationAngleX + cubeIndex * 30.0f) * M_PI / 180.0f);
    math::matrixRotateY(rotY, (rotationAngleY + cubeIndex * 45.0f) * M_PI / 180.0f);
    math::matrixMultiply(tempModel, rotY, rotX);

    // Position cubes side by side: left (-3, 0, 0), center (0, 0, 0), right (3, 0, 0)
    math::matrixIdentity(translation);
    translation.m[3][0] = -(float)CUBE_COUNT * 0.5f + (cubeIndex - 1) * 1.5f; // x offset

    // Apply translation after rotation
    math::matrixMultiply(uniforms.model, tempModel, translation);

    // Create view matrix (camera positioned at 0, 0, 10 looking at origin)
    Vec3 eye = { 0.0f, 0.0f, 10.0f }; // pulled back to see all cubes
    Vec3 center = { 0.0f, 0.0f, 0.0f }; // look at point
    Vec3 up = { 0.0f, 1.0f, 0.0f }; // up vector
    math::matrixLookAt(uniforms.view, eye, center, up);

    // Create projection matrix
    float aspect = (float)swapchainInfo.extent.width / (float)swapchainInfo.extent.height;
    math::matrixPerspective(uniforms.projection, 45.0f * M_PI / 180.0f, aspect, 0.1f, 100.0f, adapterInfo.backend);

    // Upload uniform data
    size_t offset = (currentFrame * CUBE_COUNT + cubeIndex) * uniformAlignedSize;
    gfxQueueWriteBuffer(queue, sharedUniformBuffer, offset, &uniforms, sizeof(uniforms));
}

void CubeApp::update(float deltaTime)
{
    // Accumulate total elapsed time
    elapsedTime += deltaTime;

    updateFPS(deltaTime);

    rotationAngleX += 45.0f * deltaTime;
    rotationAngleY += 30.0f * deltaTime;
    if (rotationAngleX >= 360.0f) {
        rotationAngleX -= 360.0f;
    }
    if (rotationAngleY >= 360.0f) {
        rotationAngleY -= 360.0f;
    }

    // Update uniforms for each cube
    for (int i = 0; i < CUBE_COUNT; ++i) {
        updateCube(i);
    }
}

void CubeApp::updateFPS(float deltaTime)
{
    fpsFrameCount++;
    fpsTimeAccumulator += deltaTime;

    if (deltaTime < fpsFrameTimeMin) {
        fpsFrameTimeMin = deltaTime;
    }
    if (deltaTime > fpsFrameTimeMax) {
        fpsFrameTimeMax = deltaTime;
    }

    // Log FPS every second
    if (fpsTimeAccumulator >= 1.0f) {
        float avgFPS = static_cast<float>(fpsFrameCount) / fpsTimeAccumulator;
        float avgFrameTime = (fpsTimeAccumulator / static_cast<float>(fpsFrameCount)) * 1000.0f;
        float minFPS = 1.0f / fpsFrameTimeMax;
        float maxFPS = 1.0f / fpsFrameTimeMin;
        LOG_INFO("FPS - Avg: %.1f, Min: %.1f, Max: %.1f | Frame Time - Avg: %.2f ms, Min: %.2f ms, Max: %.2f ms", avgFPS, minFPS, maxFPS, avgFrameTime, fpsFrameTimeMin * 1000.0f, fpsFrameTimeMax * 1000.0f);

        // Reset for next second
        fpsFrameCount = 0;
        fpsTimeAccumulator = 0.0f;
        fpsFrameTimeMin = FLT_MAX;
        fpsFrameTimeMax = 0.0f;
    }
}

void CubeApp::recordClearCommands(uint32_t imageIndex)
{
    auto& frame = frameResources[currentFrame];
    gfxCommandEncoderBegin(frame.clearEncoder);

    GfxColor clearColor = { 0.1f, 0.2f, 0.3f, 1.0f };

    GfxRenderPassBeginDescriptor beginDesc = {
        .label = "Clear Pass",
        .renderPass = clearRenderPass,
        .framebuffer = framebuffers[imageIndex],
        .colorClearValues = &clearColor,
        .colorClearValueCount = 1,
        .depthClearValue = 1.0f,
        .stencilClearValue = 0
    };

    GfxRenderPassEncoder renderPass;
    if (gfxCommandEncoderBeginRenderPass(frame.clearEncoder, &beginDesc, &renderPass) == GFX_RESULT_SUCCESS) {
        gfxRenderPassEncoderEnd(renderPass);
    }

    gfxCommandEncoderEnd(frame.clearEncoder);
}

void CubeApp::recordCubeCommands(int cubeIndex, uint32_t imageIndex)
{
    auto& frame = frameResources[currentFrame];
    GfxCommandEncoder encoder = frame.cubeEncoders[cubeIndex];
    gfxCommandEncoderBegin(encoder);

    GfxColor clearColor = { 0.1f, 0.2f, 0.3f, 1.0f };

    GfxRenderPassBeginDescriptor beginDesc = {
        .label = "Main Render Pass",
        .renderPass = renderPass,
        .framebuffer = framebuffers[imageIndex],
        .colorClearValues = &clearColor,
        .colorClearValueCount = 1,
        .depthClearValue = 1.0f,
        .stencilClearValue = 0
    };

    GfxRenderPassEncoder renderPass;
    if (gfxCommandEncoderBeginRenderPass(encoder, &beginDesc, &renderPass) == GFX_RESULT_SUCCESS) {
        gfxRenderPassEncoderSetPipeline(renderPass, renderPipeline);

        GfxViewport viewport = { 0.0f, 0.0f, (float)swapchainInfo.extent.width, (float)swapchainInfo.extent.height, 0.0f, 1.0f };
        GfxScissorRect scissor = { 0, 0, swapchainInfo.extent.width, swapchainInfo.extent.height };
        gfxRenderPassEncoderSetViewport(renderPass, &viewport);
        gfxRenderPassEncoderSetScissorRect(renderPass, &scissor);

        gfxRenderPassEncoderSetVertexBuffer(renderPass, 0, vertexBuffer, 0, vertexBufferInfo.size);
        gfxRenderPassEncoderSetIndexBuffer(renderPass, indexBuffer, GFX_INDEX_FORMAT_UINT16, 0, indexBufferInfo.size);

        gfxRenderPassEncoderSetBindGroup(renderPass, 0, frame.uniformBindGroups[cubeIndex], nullptr, 0);
        gfxRenderPassEncoderDrawIndexed(renderPass, 36, 1, 0, 0, 0);

        gfxRenderPassEncoderEnd(renderPass);
    }

    gfxCommandEncoderEnd(encoder);
}

void CubeApp::recordResolveCommands(uint32_t imageIndex)
{
    auto& frame = frameResources[currentFrame];
    gfxCommandEncoderBegin(frame.resolveEncoder);

    GfxRenderPassBeginDescriptor beginDesc = {
        .label = "Final Resolve Pass",
        .renderPass = resolveRenderPass,
        .framebuffer = framebuffers[imageIndex],
        .colorClearValues = nullptr,
        .colorClearValueCount = 0,
        .depthClearValue = 1.0f,
        .stencilClearValue = 0
    };

    GfxRenderPassEncoder renderPass;
    if (gfxCommandEncoderBeginRenderPass(frame.resolveEncoder, &beginDesc, &renderPass) == GFX_RESULT_SUCCESS) {
        gfxRenderPassEncoderEnd(renderPass);
    }

    gfxCommandEncoderEnd(frame.resolveEncoder);
}

void CubeApp::recordLayoutTransition(uint32_t imageIndex)
{
    auto& frame = frameResources[currentFrame];
    gfxCommandEncoderBegin(frame.transitionEncoder);

    // Use an empty render pass to transition layout via initialLayout/finalLayout
    GfxRenderPassBeginDescriptor beginDesc = {
        .label = "Layout Transition Pass",
        .renderPass = transitionRenderPass,
        .framebuffer = framebuffers[imageIndex],
        .colorClearValues = nullptr,
        .colorClearValueCount = 0,
        .depthClearValue = 1.0f,
        .stencilClearValue = 0
    };

    GfxRenderPassEncoder renderPass;
    if (gfxCommandEncoderBeginRenderPass(frame.transitionEncoder, &beginDesc, &renderPass) == GFX_RESULT_SUCCESS) {
        // Empty pass - just transitions layout
        gfxRenderPassEncoderEnd(renderPass);
    }

    gfxCommandEncoderEnd(frame.transitionEncoder);
}

void CubeApp::render()
{
    auto& frame = frameResources[currentFrame];

    gfxFenceWait(frame.inFlightFence, GFX_TIMEOUT_INFINITE);
    gfxFenceReset(frame.inFlightFence);

    uint32_t imageIndex;
    GfxResult result = gfxSwapchainAcquireNextImage(swapchain, GFX_TIMEOUT_INFINITE,
        frame.imageAvailableSemaphore, nullptr, &imageIndex);

    if (result != GFX_RESULT_SUCCESS) {
        LOG_ERROR("Failed to acquire swapchain image");
        return;
    }

    recordClearCommands(imageIndex);

    if constexpr (USE_THREADING) {
        // Store image index for threads
        currentImageIndex.store(imageIndex);

        // Record cube commands in parallel using ThreadPool
        std::vector<std::future<void>> futures;
        for (int cubeIdx = 0; cubeIdx < CUBE_COUNT; ++cubeIdx) {
            futures.push_back(threadPool->Enqueue([this, cubeIdx, imageIndex]() {
                recordCubeCommands(cubeIdx, imageIndex);
            }));
        }

        // Wait for all threads to finish
        for (auto& future : futures) {
            future.get();
        }

        // Wait stage for each wait semaphore in the submits below (Vulkan only; ignored by WebGPU)
        const GfxPipelineStageFlags colorOutputWaitStage = GFX_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT;

        // Submit clear encoder
        GfxSubmitDescriptor clearSubmit = {
            .commandEncoders = &frame.clearEncoder,
            .commandEncoderCount = 1,
            .waitSemaphores = &frame.imageAvailableSemaphore,
            .waitStages = &colorOutputWaitStage,
            .waitSemaphoreCount = 1,
            .signalSemaphores = &frame.clearFinishedSemaphore,
            .signalSemaphoreCount = 1,
            .signalFence = nullptr
        };
        gfxQueueSubmit(queue, &clearSubmit);

        // Submit cube encoders
        std::vector<GfxCommandEncoder> cubeEncoderArray(CUBE_COUNT);
        for (int i = 0; i < CUBE_COUNT; ++i) {
            cubeEncoderArray[i] = frame.cubeEncoders[i];
        }

        // When MSAA > 1, we need a resolve pass after cube rendering
        // When MSAA == 1, we render directly to swapchain so no resolve needed
        if (settings.msaaSampleCount > GFX_SAMPLE_COUNT_1) {
            GfxSubmitDescriptor cubesSubmit = {
                .commandEncoders = cubeEncoderArray.data(),
                .commandEncoderCount = static_cast<uint32_t>(cubeEncoderArray.size()),
                .waitSemaphores = &frame.clearFinishedSemaphore,
                .waitStages = &colorOutputWaitStage,
                .waitSemaphoreCount = 1,
                .signalSemaphores = nullptr,
                .signalSemaphoreCount = 0,
                .signalFence = nullptr
            };
            gfxQueueSubmit(queue, &cubesSubmit);

            // Submit resolve encoder
            recordResolveCommands(imageIndex);

            GfxSubmitDescriptor resolveSubmit = {
                .commandEncoders = &frame.resolveEncoder,
                .commandEncoderCount = 1,
                .waitSemaphores = nullptr,
                .waitSemaphoreCount = 0,
                .signalSemaphores = &renderFinishedSemaphores[imageIndex],
                .signalSemaphoreCount = 1,
                .signalFence = frame.inFlightFence
            };
            gfxQueueSubmit(queue, &resolveSubmit);
        } else {
            // No MSAA: submit cube rendering, then layout transition
            GfxSubmitDescriptor cubesSubmit = {
                .commandEncoders = cubeEncoderArray.data(),
                .commandEncoderCount = static_cast<uint32_t>(cubeEncoderArray.size()),
                .waitSemaphores = &frame.clearFinishedSemaphore,
                .waitStages = &colorOutputWaitStage,
                .waitSemaphoreCount = 1,
                .signalSemaphores = nullptr,
                .signalSemaphoreCount = 0,
                .signalFence = nullptr
            };
            gfxQueueSubmit(queue, &cubesSubmit);

            // Submit layout transition
            recordLayoutTransition(imageIndex);

            GfxSubmitDescriptor transitionSubmit = {
                .commandEncoders = &frame.transitionEncoder,
                .commandEncoderCount = 1,
                .waitSemaphores = nullptr,
                .waitSemaphoreCount = 0,
                .signalSemaphores = &renderFinishedSemaphores[imageIndex],
                .signalSemaphoreCount = 1,
                .signalFence = frame.inFlightFence
            };
            gfxQueueSubmit(queue, &transitionSubmit);
        }
    } else {
        // Non-threaded path for WebGPU
        GfxCommandEncoder encoder = frame.cubeEncoders[0];
        gfxCommandEncoderBegin(encoder);

        GfxColor clearColor = { 0.1f, 0.2f, 0.3f, 1.0f };
        GfxRenderPassBeginDescriptor beginDesc = {
            .label = "Main Render Pass (All Cubes)",
            .renderPass = clearRenderPass,
            .framebuffer = framebuffers[imageIndex],
            .colorClearValues = &clearColor,
            .colorClearValueCount = 1,
            .depthClearValue = 1.0f,
            .stencilClearValue = 0
        };

        GfxRenderPassEncoder renderPass;
        if (gfxCommandEncoderBeginRenderPass(encoder, &beginDesc, &renderPass) == GFX_RESULT_SUCCESS) {
            gfxRenderPassEncoderSetPipeline(renderPass, renderPipeline);

            GfxViewport viewport = { 0.0f, 0.0f, (float)swapchainInfo.extent.width, (float)swapchainInfo.extent.height, 0.0f, 1.0f };
            GfxScissorRect scissor = { 0, 0, swapchainInfo.extent.width, swapchainInfo.extent.height };
            gfxRenderPassEncoderSetViewport(renderPass, &viewport);
            gfxRenderPassEncoderSetScissorRect(renderPass, &scissor);

            gfxRenderPassEncoderSetVertexBuffer(renderPass, 0, vertexBuffer, 0, vertexBufferInfo.size);
            gfxRenderPassEncoderSetIndexBuffer(renderPass, indexBuffer, GFX_INDEX_FORMAT_UINT16, 0, indexBufferInfo.size);

            for (int cubeIdx = 0; cubeIdx < CUBE_COUNT; ++cubeIdx) {
                gfxRenderPassEncoderSetBindGroup(renderPass, 0, frame.uniformBindGroups[cubeIdx], nullptr, 0);
                gfxRenderPassEncoderDrawIndexed(renderPass, 36, 1, 0, 0, 0);
            }

            gfxRenderPassEncoderEnd(renderPass);
        }

        gfxCommandEncoderEnd(encoder);

        // Wait stage for the imageAvailable wait semaphore (Vulkan only; ignored by WebGPU)
        const GfxPipelineStageFlags colorOutputWaitStage = GFX_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT;
        GfxSubmitDescriptor submitDescriptor = {
            .commandEncoders = &encoder,
            .commandEncoderCount = 1,
            .waitSemaphores = &frame.imageAvailableSemaphore,
            .waitStages = &colorOutputWaitStage,
            .waitSemaphoreCount = 1,
            .signalSemaphores = &renderFinishedSemaphores[imageIndex],
            .signalSemaphoreCount = 1,
            .signalFence = frame.inFlightFence
        };
        gfxQueueSubmit(queue, &submitDescriptor);
    }

    // Present
    GfxPresentDescriptor presentDescriptor = {
        .sType = GFX_STRUCTURE_TYPE_PRESENT_DESCRIPTOR,
        .pNext = NULL,
        .waitSemaphores = &renderFinishedSemaphores[imageIndex],
        .waitSemaphoreCount = 1
    };
    gfxSwapchainPresent(swapchain, &presentDescriptor);

    currentFrame = (currentFrame + 1) % framesInFlight;
}

float CubeApp::getCurrentTime()
{
#if defined(__EMSCRIPTEN__)
    return (float)emscripten_get_now() / 1000.0f;
#else
    return (float)glfwGetTime();
#endif
}

bool CubeApp::mainLoopIteration()
{
    if (!window || glfwWindowShouldClose(window)) {
        return false;
    }

    glfwPollEvents();

    // Handle framebuffer resize
    if (previousWidth != windowWidth || previousHeight != windowHeight) {
        if (!handleResize(windowWidth, windowHeight)) {
            return false;
        }
        return true; // Skip rendering this frame
    }

    // Calculate delta time
    float currentTime = getCurrentTime();
    float deltaTime = currentTime - lastFrameTime;
    lastFrameTime = currentTime;

    update(deltaTime);
    render();

    return true;
}

bool CubeApp::handleResize(uint32_t width, uint32_t height)
{
    LOG_INFO("Resizing to %ux%u", width, height);

    windowWidth = width;
    windowHeight = height;

    // Wait for all in-flight frames to complete
    gfxDeviceWaitIdle(device);

    // Destroy per-frame resources first (they depend on swapchain image count)
    destroyPerFrameResources();

    // Recreate size-dependent resources (including swapchain)
    destroySizeDependentResources();
    if (!createSizeDependentResources(windowWidth, windowHeight)) {
        LOG_ERROR("Failed to recreate size-dependent resources after resize");
        return false;
    }

    // Recreate per-frame resources with new swapchain image count
    if (!createPerFrameResources()) {
        LOG_ERROR("Failed to recreate per-frame resources after resize");
        return false;
    }

    // Reset frame index to prevent out-of-bounds access
    currentFrame = 0;

    previousWidth = windowWidth;
    previousHeight = windowHeight;

    LOG_INFO("Successfully recreated resources for new size");
    return true;
}

GfxPlatformWindowHandle CubeApp::getPlatformWindowHandle()
{
    GfxPlatformWindowHandle handle = {}; // Use empty initializer for C++
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

void CubeApp::errorCallback(int error, const char* description)
{
    LOG_ERROR("GLFW Error %d: %s", error, description);
}

void CubeApp::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    auto* app = static_cast<CubeApp*>(glfwGetWindowUserPointer(window));
    if (app) {
        app->windowWidth = static_cast<uint32_t>(width);
        app->windowHeight = static_cast<uint32_t>(height);
    }
}

void CubeApp::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

namespace util {
std::vector<uint8_t> loadBinaryFile(const char* filepath)
{
    std::FILE* file = std::fopen(filepath, "rb");
    if (!file) {
        LOG_ERROR("Failed to open binary file: %s", filepath);
        return {};
    }

    std::fseek(file, 0, SEEK_END);
    long fileSize = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);

    if (fileSize <= 0) {
        LOG_ERROR("Invalid file size for binary file: %s", filepath);
        std::fclose(file);
        return {};
    }

    std::vector<uint8_t> buffer(fileSize);
    size_t bytesRead = std::fread(buffer.data(), 1, fileSize, file);
    std::fclose(file);

    if (bytesRead != static_cast<size_t>(fileSize)) {
        LOG_ERROR("Failed to read complete binary file: %s", filepath);
        return {};
    }

    return buffer;
}

std::string loadTextFile(const char* filepath)
{
    std::FILE* file = std::fopen(filepath, "r");
    if (!file) {
        LOG_ERROR("Failed to open text file: %s", filepath);
        return {};
    }

    std::fseek(file, 0, SEEK_END);
    long fileSize = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);

    if (fileSize <= 0) {
        LOG_ERROR("Invalid file size for text file: %s", filepath);
        std::fclose(file);
        return {};
    }

    std::string buffer(fileSize, '\0');
    size_t bytesRead = std::fread(&buffer[0], 1, fileSize, file);
    std::fclose(file);

    if (bytesRead != static_cast<size_t>(fileSize)) {
        LOG_ERROR("Failed to read complete text file: %s", filepath);
        return {};
    }

    return buffer;
}
} // namespace util

// Math namespace implementation
namespace math {
void matrixIdentity(Mat4& matrix)
{
    for (auto& row : matrix.m) {
        row.fill(0.0f);
    }
    matrix.m[0][0] = matrix.m[1][1] = matrix.m[2][2] = matrix.m[3][3] = 1.0f;
}

void matrixPerspective(Mat4& matrix, float fov, float aspect, float nearPlane, float farPlane, GfxBackend backend)
{
    for (auto& row : matrix.m) {
        row.fill(0.0f);
    }

    float f = 1.0f / std::tan(fov / 2.0f);

    matrix.m[0][0] = f / aspect;
    if (backend == GFX_BACKEND_VULKAN) {
        matrix.m[1][1] = -f; // Invert Y for Vulkan
    } else {
        matrix.m[1][1] = f;
    }
    matrix.m[2][2] = (farPlane + nearPlane) / (nearPlane - farPlane);
    matrix.m[2][3] = -1.0f;
    matrix.m[3][2] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
}

void matrixLookAt(Mat4& matrix, const Vec3& eye, const Vec3& center, const Vec3& up)
{
    // Calculate forward vector
    Vec3 forward = { center.x - eye.x, center.y - eye.y, center.z - eye.z };

    // Normalize forward vector
    if (!vectorNormalize(forward)) {
        matrixIdentity(matrix);
        return;
    }

    // Calculate right vector (forward cross up)
    Vec3 right = {
        forward.y * up.z - forward.z * up.y,
        forward.z * up.x - forward.x * up.z,
        forward.x * up.y - forward.y * up.x
    };

    // Normalize right vector
    if (!vectorNormalize(right)) {
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
    matrix.m[0][0] = right.x;
    matrix.m[0][1] = upCorrect.x;
    matrix.m[0][2] = -forward.x;
    matrix.m[0][3] = 0.0f;
    matrix.m[1][0] = right.y;
    matrix.m[1][1] = upCorrect.y;
    matrix.m[1][2] = -forward.y;
    matrix.m[1][3] = 0.0f;
    matrix.m[2][0] = right.z;
    matrix.m[2][1] = upCorrect.z;
    matrix.m[2][2] = -forward.z;
    matrix.m[2][3] = 0.0f;
    matrix.m[3][0] = -(right.x * eye.x + right.y * eye.y + right.z * eye.z);
    matrix.m[3][1] = -(upCorrect.x * eye.x + upCorrect.y * eye.y + upCorrect.z * eye.z);
    matrix.m[3][2] = forward.x * eye.x + forward.y * eye.y + forward.z * eye.z;
    matrix.m[3][3] = 1.0f;
}

void matrixRotateY(Mat4& matrix, float angle)
{
    float c = std::cos(angle);
    float s = std::sin(angle);

    matrixIdentity(matrix);
    matrix.m[0][0] = c;
    matrix.m[0][2] = s;
    matrix.m[2][0] = -s;
    matrix.m[2][2] = c;
}

void matrixRotateX(Mat4& matrix, float angle)
{
    float c = std::cos(angle);
    float s = std::sin(angle);

    matrixIdentity(matrix);
    matrix.m[1][1] = c;
    matrix.m[1][2] = -s;
    matrix.m[2][1] = s;
    matrix.m[2][2] = c;
}

void matrixMultiply(Mat4& result, const Mat4& a, const Mat4& b)
{
    Mat4 temp;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            temp.m[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                temp.m[i][j] += a.m[i][k] * b.m[k][j];
            }
        }
    }
    result = temp;
}

bool vectorNormalize(Vec3& v)
{
    const float epsilon = 1e-6f;
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

    if (len < epsilon) {
        return false;
    }

    v.x /= len;
    v.y /= len;
    v.z /= len;
    return true;
}
} // namespace math

static bool parseBackend(const char* backendStr, GfxBackend& outBackend)
{
    if (std::strcmp(backendStr, "vulkan") == 0) {
        outBackend = GFX_BACKEND_VULKAN;
        return true;
    } else if (std::strcmp(backendStr, "webgpu") == 0) {
        outBackend = GFX_BACKEND_WEBGPU;
        return true;
    } else {
        LOG_ERROR("Error: Unknown backend: %s", backendStr);
        LOG_ERROR("Valid values: vulkan, webgpu");
        return false;
    }
}

static bool parseMsaa(const char* msaaStr, GfxSampleCount& outSampleCount)
{
    int samples = std::atoi(msaaStr);
    switch (samples) {
    case 1:
        outSampleCount = GFX_SAMPLE_COUNT_1;
        break;
    case 2:
        outSampleCount = GFX_SAMPLE_COUNT_2;
        break;
    case 4:
        outSampleCount = GFX_SAMPLE_COUNT_4;
        break;
    case 8:
        outSampleCount = GFX_SAMPLE_COUNT_8;
        break;
    case 16:
        outSampleCount = GFX_SAMPLE_COUNT_16;
        break;
    case 32:
        outSampleCount = GFX_SAMPLE_COUNT_32;
        break;
    case 64:
        outSampleCount = GFX_SAMPLE_COUNT_64;
        break;
    default:
        LOG_ERROR("Error: Invalid MSAA sample count: %d", samples);
        LOG_ERROR("Valid values: 1, 2, 4, 8, 16, 32, 64");
        return false;
    }
    return true;
}

static bool parseVsync(const char* vsyncStr, bool& outVsync)
{
    int vsync = std::atoi(vsyncStr);
    if (vsync == 0) {
        outVsync = false;
        return true;
    } else if (vsync == 1) {
        outVsync = true;
        return true;
    } else {
        LOG_ERROR("Error: Invalid vsync value: %s", vsyncStr);
        LOG_ERROR("Valid values: 0 (off), 1 (on)");
        return false;
    }
}

static void printHelp(const char* programName)
{
    LOG_INFO("Usage: %s [options]", programName);
    LOG_INFO("Options:");
    LOG_INFO("  --backend [vulkan|webgpu]   Select graphics backend");
    LOG_INFO("  --msaa [1|2|4|8|16|32|64]   Select MSAA sample count");
    LOG_INFO("  --vsync [0|1]               VSync: 0=off, 1=on");
    LOG_INFO("  --help                      Show this help message");
}

bool parseArguments(int argc, char** argv, Settings& settings)
{
#if defined(__EMSCRIPTEN__)
    settings.backend = GFX_BACKEND_WEBGPU;
#else
    settings.backend = GFX_BACKEND_VULKAN;
#endif
    settings.msaaSampleCount = GFX_SAMPLE_COUNT_4;
    settings.vsync = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            i++;
            if (!parseBackend(argv[i], settings.backend)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--msaa") == 0 && i + 1 < argc) {
            i++;
            if (!parseMsaa(argv[i], settings.msaaSampleCount)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--vsync") == 0 && i + 1 < argc) {
            i++;
            if (!parseVsync(argv[i], settings.vsync)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--help") == 0) {
            printHelp(argv[0]);
            return false;
        } else {
            LOG_ERROR("Error: Unknown argument: %s", argv[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    LOG_INFO("=== Threaded Cube Example (C++ ThreadPool with C API) ===\n");

    Settings settings;
    if (!parseArguments(argc, argv, settings)) {
        return 0;
    }

    CubeApp app(settings);
    if (!app.init()) {
        LOG_ERROR("Failed to initialize application");
        app.cleanup();
        return -1;
    }
    app.run();
    app.cleanup();

    LOG_INFO("Example completed successfully!");
    return 0;
}
