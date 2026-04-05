#include <gfx_cpp/gfx.hpp>

#ifdef __ANDROID__
#include <android/log.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <time.h>

// Android logging macros
#define LOG_TAG "GFX_CUBE_CPP"
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOG_WARN(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOG_DEBUG(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#else
// Desktop logging macros
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

#include <cstdio>
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
#endif // __ANDROID__

#ifdef Success
#undef Success
#endif

#include <array>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr uint32_t WINDOW_WIDTH = 800;
static constexpr uint32_t WINDOW_HEIGHT = 600;
static constexpr size_t CUBE_COUNT = 3;
static constexpr gfx::Format COLOR_FORMAT = gfx::Format::B8G8R8A8UnormSrgb;
static constexpr gfx::Format DEPTH_FORMAT = gfx::Format::Depth32Float;

// Log callback function
static void logCallback(gfx::LogLevel level, const std::string& message)
{
    switch (level) {
    case gfx::LogLevel::Error:
        LOG_ERROR("%s", message.c_str());
        break;
    case gfx::LogLevel::Warning:
        LOG_WARN("%s", message.c_str());
        break;
    case gfx::LogLevel::Info:
        LOG_INFO("%s", message.c_str());
        break;
    case gfx::LogLevel::Debug:
        LOG_DEBUG("%s", message.c_str());
        break;
    default:
        LOG_INFO("%s", message.c_str());
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
    gfx::Backend backend;
    gfx::SampleCount msaaSampleCount;
    bool vsync;
};

// Per-frame resources
struct PerFrameResources {
    std::shared_ptr<gfx::Semaphore> imageAvailableSemaphore;
    std::shared_ptr<gfx::Fence> inFlightFence;
    std::shared_ptr<gfx::CommandEncoder> commandEncoder;
    std::vector<std::shared_ptr<gfx::BindGroup>> uniformBindGroups;
};

namespace math {
void matrixIdentity(Mat4& matrix);
void matrixPerspective(Mat4& matrix, float fovy, float aspect, float nearPlane, float farPlane, gfx::Backend backend);
void matrixLookAt(Mat4& matrix, const Vec3& eye, const Vec3& center, const Vec3& up);
void matrixRotateX(Mat4& matrix, float angle);
void matrixRotateY(Mat4& matrix, float angle);
void matrixMultiply(Mat4& result, const Mat4& a, const Mat4& b);
bool vectorNormalize(Vec3& v);
} // namespace math

class CubeApp {
public:
    explicit CubeApp(const Settings& settings);
    ~CubeApp() = default;

    bool init();
    void run();
    void cleanup();

#ifdef __ANDROID__
    // Android-specific setters/getters for android_main
    void setAndroidApp(struct android_app* app) { androidApp = app; }
    void setAnimating(bool value) { animating = value; }

    // Android callbacks (public because they're assigned to android_app function pointers)
    static void handleAppCommand(struct android_app* state, int32_t cmd);
    static int32_t handleInput(struct android_app* state, AInputEvent* event);
#endif

private:
    bool createWindow(uint32_t width, uint32_t height);
    void destroyWindow();
    bool createGraphics();
    void destroyGraphics();
    bool createPerFrameResources();
    void destroyPerFrameResources();
    bool createSizeDependentResources(uint32_t width, uint32_t height);
    void destroySizeDependentResources();
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

    void updateCube(int cubeIndex);
    void update(float deltaTime);
    void render();
    bool handleResize(uint32_t width, uint32_t height);
    void updateFPS(float deltaTime);
    float getCurrentTime();
    bool mainLoopIteration();
#if defined(__EMSCRIPTEN__)
    static void emscriptenMainLoop(void* userData);
#endif

    gfx::PlatformWindowHandle getPlatformWindowHandle();

#ifndef __ANDROID__
    static void errorCallback(int error, const char* description);
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
#endif

    std::vector<uint8_t> loadBinaryFile(const char* filepath);
    std::string loadTextFile(const char* filepath);

private:
    Settings settings;

#ifdef __ANDROID__
    struct android_app* androidApp = nullptr;
    bool animating = false;
#else
    GLFWwindow* window = nullptr;
#endif
    std::shared_ptr<gfx::Instance> instance;
    std::shared_ptr<gfx::Adapter> adapter;
    gfx::AdapterInfo adapterInfo; // Cached adapter info
    std::shared_ptr<gfx::Device> device;
    std::shared_ptr<gfx::Queue> queue;
    std::shared_ptr<gfx::Surface> surface;
    std::shared_ptr<gfx::Swapchain> swapchain;
    gfx::SwapchainInfo swapchainInfo;

    std::shared_ptr<gfx::Buffer> vertexBuffer;
    std::shared_ptr<gfx::Buffer> indexBuffer;
    gfx::BufferInfo vertexBufferInfo;
    gfx::BufferInfo indexBufferInfo;
    std::shared_ptr<gfx::Shader> vertexShader;
    std::shared_ptr<gfx::Shader> fragmentShader;
    std::shared_ptr<gfx::RenderPipeline> renderPipeline;
    std::shared_ptr<gfx::BindGroupLayout> uniformBindGroupLayout;

    // Depth buffer
    std::shared_ptr<gfx::Texture> depthTexture;
    std::shared_ptr<gfx::TextureView> depthTextureView;

    // MSAA color buffer
    std::shared_ptr<gfx::Texture> msaaColorTexture;
    std::shared_ptr<gfx::TextureView> msaaColorTextureView;

    std::shared_ptr<gfx::RenderPass> renderPass;

    uint32_t windowWidth = WINDOW_WIDTH;
    uint32_t windowHeight = WINDOW_HEIGHT;
    uint32_t previousWidth = WINDOW_WIDTH;
    uint32_t previousHeight = WINDOW_HEIGHT;

    std::shared_ptr<gfx::Buffer> sharedUniformBuffer;
    size_t uniformAlignedSize = 0;

    // Per-frame resources
    std::vector<std::shared_ptr<gfx::Framebuffer>> framebuffers;
    std::vector<PerFrameResources> frameResources;
    std::vector<std::shared_ptr<gfx::Semaphore>> renderFinishedSemaphores;
    size_t currentFrame = 0;

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
};

CubeApp::CubeApp(const Settings& settings)
    : settings(settings)
{
}

bool CubeApp::init()
{
    // Initialize in order of dependencies

    // 1. Create window
    if (!createWindow(windowWidth, windowHeight)) {
        LOG_ERROR("Failed to create window");
        return false;
    }

    // 2. Create graphics context (instance, adapter, device, surface)
    if (!createGraphics()) {
        LOG_ERROR("Failed to create graphics");
        destroyGraphics(); // Clean up partial initialization
        return false;
    }

    // 3. Create size-dependent resources (swapchain, framebuffers, render pass)
    if (!createSizeDependentResources(windowWidth, windowHeight)) {
        LOG_ERROR("Failed to create size-dependent resources");
        destroyGraphics();
        return false;
    }

    // 4. Create rendering resources (geometry, uniform buffer, shaders, pipeline)
    if (!createRenderingResources()) {
        LOG_ERROR("Failed to create rendering resources");
        destroySizeDependentResources();
        destroyGraphics();
        return false;
    }

    // 5. Create per-frame resources (sync objects and bind groups - depends on uniform buffer and layouts)
    if (!createPerFrameResources()) {
        LOG_ERROR("Failed to create per-frame resources");
        destroyRenderingResources();
        destroySizeDependentResources();
        destroyGraphics();
        return false;
    }

    LOG_INFO("Application initialized successfully!");
    LOG_INFO("Press ESC or close window to exit");

    // Initialize timing
    lastFrameTime = getCurrentTime();

    return true;
}

void CubeApp::run()
{
    // Run main loop (platform-specific)
#if defined(__EMSCRIPTEN__)
    // Note: emscripten_set_main_loop_arg returns immediately and never blocks
    // Cleanup happens in emscriptenMainLoop when the loop exits
    // Execution continues in the browser event loop
    emscripten_set_main_loop_arg(CubeApp::emscriptenMainLoop, this, 0, 1);
#else
    while (mainLoopIteration()) {
        // Loop continues until mainLoopIteration returns false
    }
#endif
}

void CubeApp::cleanup()
{
    // Wait for device to finish
    if (device) {
        device->waitIdle();
    }

    // Destroy resources in reverse order of creation
    // 7. Destroy per-frame resources (includes bind groups and sync objects)
    destroyPerFrameResources();

    // 6. Destroy size-dependent resources
    destroySizeDependentResources();

    // 5. Destroy rendering resources (pipeline, shaders, uniform buffer, geometry)
    destroyRenderingResources();

    // 4. Destroy graphics resources
    destroyGraphics();

    // 3. Destroy window
    destroyWindow();

    // 2. Unload backend
    gfx::unloadBackend(settings.backend);
}

bool CubeApp::createWindow(uint32_t width, uint32_t height)
{
#ifdef __ANDROID__
    // Android window is created by the system
    this->windowWidth = width;
    this->windowHeight = height;
    return true;
#else
    glfwSetErrorCallback(errorCallback);

    // Initialize GLFW
    if (!glfwInit()) {
        LOG_ERROR("Failed to initialize GLFW");
        return false;
    }

    // Create window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No OpenGL context
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    std::string backendName = (settings.backend == gfx::Backend::Vulkan) ? "Vulkan" : "WebGPU";
    std::string title = "Cube Example (C++ API) - " + backendName;
    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        LOG_ERROR("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }

    // Set up window resize callback
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetKeyCallback(window, keyCallback);

    this->windowWidth = width;
    this->windowHeight = height;

    return true;
#endif
}

void CubeApp::destroyWindow()
{
#ifdef __ANDROID__
    // Android window is managed by the system
#else
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
#endif
}

bool CubeApp::createGraphics()
{
    // Set up logging callback
    gfx::setLogCallback(logCallback);

    auto result = gfx::loadBackend(settings.backend);
    if (!gfx::isSuccess(result)) {
        LOG_ERROR("Failed to load graphics backend: %d", static_cast<int32_t>(result));
        return false;
    }

    try {
        gfx::InstanceDescriptor instanceDesc{
            .backend = settings.backend,
            .applicationName = "Rotating Cube Example (C++)",
            .applicationVersion = 1,
            .enabledExtensions = { gfx::INSTANCE_EXTENSION_SURFACE, gfx::INSTANCE_EXTENSION_DEBUG }
        };

        instance = gfx::createInstance(instanceDesc);
        if (!instance) {
            LOG_ERROR("Failed to create graphics instance");
            return false;
        }

        // Get adapter
        gfx::AdapterDescriptor adapterDesc{
            .preference = gfx::AdapterPreference::HighPerformance
        };

        adapter = instance->requestAdapter(adapterDesc);
        if (!adapter) {
            LOG_ERROR("Failed to get graphics adapter");
            return false;
        }

        // Query and store adapter info
        adapterInfo = adapter->getInfo();
        LOG_INFO("Using adapter: %s", adapterInfo.name.c_str());
        LOG_INFO("Backend: %s", (adapterInfo.backend == gfx::Backend::Vulkan ? "Vulkan" : "WebGPU"));
        LOG_INFO("  Vendor ID: 0x%04X, Device ID: 0x%04X", adapterInfo.vendorID, adapterInfo.deviceID);

        // Create device
        gfx::DeviceDescriptor deviceDesc{
            .label = "Main Device",
            .enabledExtensions = { gfx::DEVICE_EXTENSION_SWAPCHAIN }
        };

        device = adapter->createDevice(deviceDesc);
        if (!device) {
            LOG_ERROR("Failed to create device");
            return false;
        }

        queue = device->getQueue();

        gfx::SurfaceDescriptor surfaceDesc{
            .label = "Main Surface",
            .windowHandle = getPlatformWindowHandle()
        };

        surface = instance->createSurface(surfaceDesc);
        if (!surface) {
            LOG_ERROR("Failed to create surface");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize graphics: %s", e.what());
        return false;
    }
}

void CubeApp::destroyGraphics()
{
    surface.reset();
    queue.reset();
    device.reset();
    adapter.reset();
    instance.reset();
}

bool CubeApp::createSizeDependentResources(uint32_t width, uint32_t height)
{
    try {
        // Query surface capabilities to determine frame count
        auto surfaceInfo = surface->getInfo(adapter);
        LOG_INFO("Surface Info:");
        LOG_INFO("  Image Count: min %u, max %u", surfaceInfo.minImageCount, surfaceInfo.maxImageCount);
        LOG_INFO("  Extent: min (%u, %u), max (%u, %u)",
            surfaceInfo.minExtent.width, surfaceInfo.minExtent.height,
            surfaceInfo.maxExtent.width, surfaceInfo.maxExtent.height);

        // Create swapchain
        gfx::SwapchainDescriptor swapchainDesc{
            .label = "Main Swapchain",
            .surface = surface,
            .extent = { width, height },
            .format = COLOR_FORMAT,
            .usage = gfx::TextureUsage::RenderAttachment,
            .presentMode = settings.vsync ? gfx::PresentMode::Fifo : gfx::PresentMode::Immediate,
            .imageCount = surfaceInfo.minImageCount
        };

        swapchain = device->createSwapchain(swapchainDesc);
        if (!swapchain) {
            LOG_ERROR("Failed to create swapchain");
            return false;
        }

        // Get actual swapchain dimensions (may differ from requested)
        swapchainInfo = swapchain->getInfo();

        // Create render finished semaphores (one per swapchain image)
        renderFinishedSemaphores.resize(swapchainInfo.imageCount);
        for (uint32_t i = 0; i < swapchainInfo.imageCount; ++i) {
            gfx::SemaphoreDescriptor semDesc{
                .label = "Render Finished Semaphore Image " + std::to_string(i),
                .type = gfx::SemaphoreType::Binary
            };

            renderFinishedSemaphores[i] = device->createSemaphore(semDesc);
            if (!renderFinishedSemaphores[i]) {
                LOG_ERROR("Failed to create render finished semaphore %u", i);
                return false;
            }
        }
        uint32_t actualWidth = swapchainInfo.extent.width;
        uint32_t actualHeight = swapchainInfo.extent.height;

        // Create depth texture with MSAA using actual swapchain dimensions
        gfx::TextureDescriptor depthTextureDesc{
            .label = "Depth Buffer",
            .type = gfx::TextureType::Texture2D,
            .size = { actualWidth, actualHeight, 1 },
            .arrayLayerCount = 1,
            .mipLevelCount = 1,
            .sampleCount = settings.msaaSampleCount,
            .format = DEPTH_FORMAT,
            .usage = gfx::TextureUsage::RenderAttachment
        };

        depthTexture = device->createTexture(depthTextureDesc);
        if (!depthTexture) {
            LOG_ERROR("Failed to create depth texture");
            return false;
        }

        // Create depth texture view
        gfx::TextureViewDescriptor depthViewDesc{
            .label = "Depth Buffer View",
            .viewType = gfx::TextureViewType::View2D,
            .format = DEPTH_FORMAT,
            .baseMipLevel = 0,
            .mipLevelCount = 1,
            .baseArrayLayer = 0,
            .arrayLayerCount = 1
        };

        depthTextureView = depthTexture->createView(depthViewDesc);
        if (!depthTextureView) {
            LOG_ERROR("Failed to create depth texture view");
            return false;
        }

        // Create MSAA color texture using actual swapchain dimensions
        gfx::TextureDescriptor msaaColorTextureDesc{
            .label = "MSAA Color Buffer",
            .type = gfx::TextureType::Texture2D,
            .size = { actualWidth, actualHeight, 1 },
            .arrayLayerCount = 1,
            .mipLevelCount = 1,
            .sampleCount = settings.msaaSampleCount,
            .format = swapchainInfo.format,
            .usage = gfx::TextureUsage::RenderAttachment
        };

        msaaColorTexture = device->createTexture(msaaColorTextureDesc);
        if (!msaaColorTexture) {
            LOG_ERROR("Failed to create MSAA color texture");
            return false;
        }

        // Create MSAA color texture view
        gfx::TextureViewDescriptor msaaColorViewDesc{
            .label = "MSAA Color Buffer View",
            .viewType = gfx::TextureViewType::View2D,
            .format = swapchainInfo.format,
            .baseMipLevel = 0,
            .mipLevelCount = 1,
            .baseArrayLayer = 0,
            .arrayLayerCount = 1
        };

        msaaColorTextureView = msaaColorTexture->createView(msaaColorViewDesc);
        if (!msaaColorTextureView) {
            LOG_ERROR("Failed to create MSAA color texture view");
            return false;
        }

        // Create render pass
        gfx::RenderPassCreateDescriptor renderPassDesc{
            .label = "Main Render Pass"
        };

        // Color attachment
        gfx::RenderPassColorAttachment colorAttachment{
            .target = {
                .format = swapchainInfo.format,
                .sampleCount = settings.msaaSampleCount,
                .ops = {
                    .load = gfx::LoadOp::Clear,
                    .store = settings.msaaSampleCount != gfx::SampleCount::Count1 ? gfx::StoreOp::DontCare : gfx::StoreOp::Store },
                .finalLayout = settings.msaaSampleCount != gfx::SampleCount::Count1 ? gfx::TextureLayout::ColorAttachment : gfx::TextureLayout::PresentSrc }
        };

        if (settings.msaaSampleCount != gfx::SampleCount::Count1) {
            // MSAA: Add resolve target
            gfx::RenderPassColorAttachmentTarget resolveTarget{
                .format = swapchainInfo.format,
                .sampleCount = gfx::SampleCount::Count1,
                .ops = {
                    .load = gfx::LoadOp::DontCare,
                    .store = gfx::StoreOp::Store },
                .finalLayout = gfx::TextureLayout::PresentSrc
            };
            colorAttachment.resolveTarget = resolveTarget;
        }

        renderPassDesc.colorAttachments.push_back(colorAttachment);

        // Depth/stencil attachment
        gfx::RenderPassDepthStencilAttachment depthAttachment{
            .target = {
                .format = DEPTH_FORMAT,
                .sampleCount = settings.msaaSampleCount,
                .depthOps = {
                    .load = gfx::LoadOp::Clear,
                    .store = gfx::StoreOp::DontCare },
                .stencilOps = { .load = gfx::LoadOp::DontCare, .store = gfx::StoreOp::DontCare },
                .finalLayout = gfx::TextureLayout::DepthStencilAttachment }
        };

        renderPassDesc.depthStencilAttachment = depthAttachment;

        renderPass = device->createRenderPass(renderPassDesc);
        if (!renderPass) {
            LOG_ERROR("Failed to create render pass");
            return false;
        }

        // Create framebuffers for each swapchain image
        framebuffers.resize(swapchainInfo.imageCount);

        for (uint32_t i = 0; i < swapchainInfo.imageCount; ++i) {
            gfx::FramebufferDescriptor framebufferDesc{
                .label = "Framebuffer " + std::to_string(i),
                .renderPass = renderPass,
                .extent = { actualWidth, actualHeight }
            };

            // Color attachment
            if (settings.msaaSampleCount != gfx::SampleCount::Count1) {
                // MSAA: Single attachment with MSAA buffer and resolve target
                framebufferDesc.colorAttachments.push_back({ msaaColorTextureView, swapchain->getTextureView(i) });
            } else {
                // No MSAA: Attach swapchain image directly
                framebufferDesc.colorAttachments.push_back({ swapchain->getTextureView(i) });
            }

            // Depth attachment (must be a pointer)
            gfx::FramebufferDepthStencilAttachment depthAttachment{ depthTextureView };
            framebufferDesc.depthStencilAttachment = depthAttachment;

            framebuffers[i] = device->createFramebuffer(framebufferDesc);
            if (!framebuffers[i]) {
                LOG_ERROR("Failed to create framebuffer %u", i);
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Graphics initialization error: %s", e.what());
        return false;
    }
}

void CubeApp::destroyPerFrameResources()
{
    // Wait for device idle before destroying resources
    if (device) {
        device->waitIdle();
    }

    // Clean up per-frame resources
    for (auto& frame : frameResources) {
        // Clean up bind groups
        for (auto& bindGroup : frame.uniformBindGroups) {
            bindGroup.reset();
        }
        frame.uniformBindGroups.clear();

        // Clean up sync objects and command encoder
        frame.commandEncoder.reset();
        frame.inFlightFence.reset();
        frame.imageAvailableSemaphore.reset();
    }

    frameResources.clear();
}

bool CubeApp::createPerFrameResources()
{
    try {
        // Resize frameResources vector for dynamic frame count
        frameResources.resize(swapchainInfo.imageCount);

        // Create per-frame resources for each frame in flight
        for (size_t i = 0; i < swapchainInfo.imageCount; ++i) {
            auto& frame = frameResources[i];

            // Create binary semaphores for image availability and render completion
            gfx::SemaphoreDescriptor semDesc{
                .label = "Image Available Semaphore Frame " + std::to_string(i),
                .type = gfx::SemaphoreType::Binary
            };

            frame.imageAvailableSemaphore = device->createSemaphore(semDesc);
            if (!frame.imageAvailableSemaphore) {
                LOG_ERROR("Failed to create image available semaphore %zu", i);
                return false;
            }

            // Create fence (start signaled so first frame doesn't wait)
            gfx::FenceDescriptor fenceDesc{
                .label = "In Flight Fence Frame " + std::to_string(i),
                .signaled = true
            };

            frame.inFlightFence = device->createFence(fenceDesc);
            if (!frame.inFlightFence) {
                LOG_ERROR("Failed to create in flight fence %zu", i);
                return false;
            }

            // Create command encoder for this frame
            frame.commandEncoder = device->createCommandEncoder({ .label = "Command Encoder Frame " + std::to_string(i) });
            if (!frame.commandEncoder) {
                LOG_ERROR("Failed to create command encoder %zu", i);
                return false;
            }

            // Resize uniformBindGroups array for this frame
            frame.uniformBindGroups.resize(CUBE_COUNT);

            // Create bind groups (one per cube) using offsets into shared buffer
            for (size_t cubeIdx = 0; cubeIdx < CUBE_COUNT; ++cubeIdx) {
                gfx::BindGroupEntry uniformEntry{
                    .binding = 0,
                    .resource = sharedUniformBuffer,
                    .offset = (i * CUBE_COUNT + cubeIdx) * uniformAlignedSize,
                    .size = sizeof(UniformData)
                };

                gfx::BindGroupDescriptor uniformBindGroupDesc{
                    .label = "Uniform Bind Group Frame " + std::to_string(i) + " Cube " + std::to_string(cubeIdx),
                    .layout = uniformBindGroupLayout,
                    .entries = { uniformEntry }
                };

                frame.uniformBindGroups[cubeIdx] = device->createBindGroup(uniformBindGroupDesc);
                if (!frame.uniformBindGroups[cubeIdx]) {
                    LOG_ERROR("Failed to create uniform bind group %zu cube %zu", i, cubeIdx);
                    return false;
                }
            }
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create per-frame resources: %s", e.what());
        return false;
    }
}

void CubeApp::destroySizeDependentResources()
{
    // Clean up framebuffers and render pass
    framebuffers.clear();
    renderPass.reset();

    // Clean up size-dependent resources
    msaaColorTextureView.reset();
    msaaColorTexture.reset();
    depthTextureView.reset();
    depthTexture.reset();

    // Clean up render finished semaphores
    renderFinishedSemaphores.clear();

    // Also destroy swapchain to fully recreate it
    swapchain.reset();
}

bool CubeApp::createGeometry()
{
    try {
        // Create cube vertices (8 vertices for a cube)
        std::array<Vertex, 8> vertices = { {
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
        } };

        // Create cube indices (36 indices for 12 triangles)
        // All faces wound clockwise when viewed from outside
        std::array<uint16_t, 36> indices = { { // Front face (Z+) - vertices 0,1,2,3
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
            4, 5, 1, 1, 0, 4 } };

        // Create vertex buffer
        gfx::BufferDescriptor vertexBufferDesc{
            .label = "Cube Vertices",
            .size = sizeof(vertices),
            .usage = gfx::BufferUsage::Vertex | gfx::BufferUsage::CopyDst
        };

        vertexBuffer = device->createBuffer(vertexBufferDesc);
        if (!vertexBuffer) {
            LOG_ERROR("Failed to create vertex buffer");
            return false;
        }
        vertexBufferInfo = vertexBuffer->getInfo();

        // Create index buffer
        gfx::BufferDescriptor indexBufferDesc{
            .label = "Cube Indices",
            .size = sizeof(indices),
            .usage = gfx::BufferUsage::Index | gfx::BufferUsage::CopyDst
        };

        indexBuffer = device->createBuffer(indexBufferDesc);
        if (!indexBuffer) {
            LOG_ERROR("Failed to create index buffer");
            return false;
        }
        indexBufferInfo = indexBuffer->getInfo();

        // Upload vertex and index data
        queue->writeBuffer(vertexBuffer, 0, vertices.data(), sizeof(vertices));
        queue->writeBuffer(indexBuffer, 0, indices.data(), sizeof(indices));

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create geometry: %s", e.what());
        return false;
    }
}

void CubeApp::destroyGeometry()
{
    indexBuffer.reset();
    vertexBuffer.reset();
}

bool CubeApp::createUniformBuffer()
{
    try {
        // Create single large uniform buffer for all frames and cubes with proper alignment
        auto limits = device->getLimits();

        size_t uniformSize = sizeof(UniformData);
        uniformAlignedSize = gfx::utils::alignUp(uniformSize, limits.minUniformBufferOffsetAlignment);
        size_t totalBufferSize = uniformAlignedSize * swapchainInfo.imageCount * CUBE_COUNT;

        gfx::BufferDescriptor uniformBufferDesc{
            .label = "Shared Transform Uniforms",
            .size = totalBufferSize,
            .usage = gfx::BufferUsage::Uniform | gfx::BufferUsage::CopyDst
        };

        sharedUniformBuffer = device->createBuffer(uniformBufferDesc);
        if (!sharedUniformBuffer) {
            LOG_ERROR("Failed to create shared uniform buffer");
            return false;
        }

        // Create bind group layout for uniforms
        gfx::BindGroupLayoutEntry uniformLayoutEntry{
            .binding = 0,
            .visibility = gfx::ShaderStage::Vertex,
            .resource = gfx::BindGroupLayoutEntry::BufferBinding{
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(UniformData) }
        };

        gfx::BindGroupLayoutDescriptor uniformLayoutDesc{
            .label = "Uniform Bind Group Layout",
            .entries = { uniformLayoutEntry }
        };

        uniformBindGroupLayout = device->createBindGroupLayout(uniformLayoutDesc);
        if (!uniformBindGroupLayout) {
            LOG_ERROR("Failed to create uniform bind group layout");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create uniform buffer: %s", e.what());
        return false;
    }
}

void CubeApp::destroyUniformBuffer()
{
    uniformBindGroupLayout.reset();
    sharedUniformBuffer.reset();
}

bool CubeApp::createShaders()
{
    try {
        // Try shader formats in order of preference
        struct ShaderFormat {
            gfx::ShaderSourceType format;
            const char* vertexPath;
            const char* fragmentPath;
        };

        const ShaderFormat shaderFormats[] = {
            { gfx::ShaderSourceType::SPIRV, "shaders/cube.vert.spv", "shaders/cube.frag.spv" },
            { gfx::ShaderSourceType::WGSL, "shaders/cube.vert.wgsl", "shaders/cube.frag.wgsl" }
        };

        gfx::ShaderSourceType shaderSourceType;
        std::vector<uint8_t> vertexShaderCode;
        std::vector<uint8_t> fragmentShaderCode;
        bool shadersLoaded = false;

        for (const auto& format : shaderFormats) {
            if (!device->supportsShaderFormat(format.format)) {
                continue;
            }

            LOG_INFO("    Loading shaders: %s, %s", format.vertexPath, format.fragmentPath);

            if (format.format == gfx::ShaderSourceType::SPIRV) {
                vertexShaderCode = loadBinaryFile(format.vertexPath);
                fragmentShaderCode = loadBinaryFile(format.fragmentPath);
            } else {
                auto vertexWgsl = loadTextFile(format.vertexPath);
                auto fragmentWgsl = loadTextFile(format.fragmentPath);
                vertexShaderCode.assign(vertexWgsl.begin(), vertexWgsl.end());
                fragmentShaderCode.assign(fragmentWgsl.begin(), fragmentWgsl.end());
            }

            if (!vertexShaderCode.empty() && !fragmentShaderCode.empty()) {
                shaderSourceType = format.format;
                LOG_INFO("    Successfully loaded shaders (vertex: %zu bytes, fragment: %zu bytes)",
                    vertexShaderCode.size(), fragmentShaderCode.size());
                shadersLoaded = true;
                break;
            }

            // Failed to load this format, clear and try next
            LOG_ERROR("    Failed to load shaders for format");
            vertexShaderCode.clear();
            fragmentShaderCode.clear();
        }

        if (!shadersLoaded) {
            LOG_ERROR("    No supported shader format found or failed to load shaders");
            return false;
        }

        // Create vertex shader
        gfx::ShaderDescriptor vertexShaderDesc{
            .label = "Cube Vertex Shader",
            .sourceType = shaderSourceType,
            .code = vertexShaderCode,
            .entryPoint = "main"
        };

        vertexShader = device->createShader(vertexShaderDesc);
        if (!vertexShader) {
            LOG_ERROR("    Failed to create vertex shader");
            return false;
        }

        // Create fragment shader
        gfx::ShaderDescriptor fragmentShaderDesc{
            .label = "Cube Fragment Shader",
            .sourceType = shaderSourceType,
            .code = fragmentShaderCode,
            .entryPoint = "main"
        };

        fragmentShader = device->createShader(fragmentShaderDesc);
        if (!fragmentShader) {
            LOG_ERROR("    Failed to create fragment shader");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("    Exception in createShaders: %s", e.what());
        return false;
    }
}

void CubeApp::destroyShaders()
{
    fragmentShader.reset();
    vertexShader.reset();
}

bool CubeApp::createRenderingResources()
{
    try {
        LOG_INFO("Creating rendering resources...");

        // 1. Create geometry (vertex/index buffers)
        if (!createGeometry()) {
            LOG_ERROR("  Failed to create geometry");
            return false;
        }

        // 2. Create uniform buffer and layout
        if (!createUniformBuffer()) {
            LOG_ERROR("  Failed to create uniform buffer");
            return false;
        }

        // 3. Create shaders
        if (!createShaders()) {
            LOG_ERROR("  Failed to create shaders");
            return false;
        }

        // Initialize animation state
        rotationAngleX = 0.0f;
        rotationAngleY = 0.0f;

        // 4. Create render pipeline
        bool result = createRenderPipeline();
        if (!result) {
            LOG_ERROR("  Failed to create render pipeline");
        }
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("Resource creation error: %s", e.what());
        return false;
    }
}

void CubeApp::destroyRenderingResources()
{
    // 4. Destroy render pipeline
    destroyRenderPipeline();

    // 3. Destroy shaders
    destroyShaders();

    // 2. Destroy uniform buffer and layout
    destroyUniformBuffer();

    // 1. Destroy geometry
    destroyGeometry();
}

bool CubeApp::createRenderPipeline()
{
    try {
        // Define vertex buffer layout
        std::vector<gfx::VertexAttribute> attributes = {
            { .format = gfx::Format::R32G32B32Float,
                .offset = offsetof(Vertex, position),
                .shaderLocation = 0 },
            { .format = gfx::Format::R32G32B32Float,
                .offset = offsetof(Vertex, color),
                .shaderLocation = 1 }
        };

        gfx::VertexBufferLayout vertexLayout{
            .arrayStride = sizeof(Vertex),
            .attributes = attributes,
            .stepMode = gfx::VertexStepMode::Vertex
        };

        // Create render pipeline descriptor
        gfx::VertexState vertexState{
            .module = vertexShader,
            .entryPoint = "main",
            .buffers = { vertexLayout }
        };

        auto swapchainInfo = swapchain->getInfo();
        gfx::ColorTargetState colorTarget{
            .format = swapchainInfo.format,
            .writeMask = gfx::ColorWriteMask::All
        };

        gfx::FragmentState fragmentState{
            .module = fragmentShader,
            .entryPoint = "main",
            .targets = { colorTarget }
        };

        gfx::PrimitiveState primitiveState{
            .topology = gfx::PrimitiveTopology::TriangleList,
            .frontFace = gfx::FrontFace::CounterClockwise,
            .cullMode = gfx::CullMode::Back, // Enable back-face culling for 3D
            .polygonMode = gfx::PolygonMode::Fill
        };

        // Depth/stencil state - enable depth testing
        gfx::DepthStencilState depthStencilState{
            .format = gfx::Format::Depth32Float,
            .depthWriteEnabled = true,
            .depthCompare = gfx::CompareFunction::Less
        };

        gfx::RenderPipelineDescriptor pipelineDesc{
            .label = "Cube Pipeline",
            .renderPass = renderPass,
            .vertex = vertexState,
            .fragment = fragmentState,
            .primitive = primitiveState,
            .depthStencil = depthStencilState,
            .sampleCount = settings.msaaSampleCount,
            .bindGroupLayouts = { uniformBindGroupLayout } // Pass the bind group layout
        };

        renderPipeline = device->createRenderPipeline(pipelineDesc);
        if (!renderPipeline) {
            LOG_ERROR("Failed to create render pipeline");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Pipeline creation error: %s", e.what());
        return false;
    }
}

void CubeApp::destroyRenderPipeline()
{
    renderPipeline.reset();
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
    translation.m[3][0] = (cubeIndex - 1) * 3.0f; // x offset: -3, 0, 3

    // Apply translation after rotation
    math::matrixMultiply(uniforms.model, tempModel, translation);
    // Create view matrix (camera positioned at 0, 0, 10 looking at origin)
    Vec3 eye = { 0.0f, 0.0f, 10.0f }; // pulled back to see all 3 cubes
    Vec3 center = { 0.0f, 0.0f, 0.0f }; // look at point
    Vec3 up = { 0.0f, 1.0f, 0.0f }; // up vector
    math::matrixLookAt(uniforms.view, eye, center, up);

    // Create projection matrix
    auto swapchainInfo = swapchain->getInfo();
    float aspect = (float)swapchainInfo.extent.width / (float)swapchainInfo.extent.height;
    math::matrixPerspective(uniforms.projection, 45.0f * M_PI / 180.0f, aspect, 0.1f, 100.0f, settings.backend);

    // Upload uniform data to buffer at aligned offset
    // Formula: (frame * CUBE_COUNT + cube) * alignedSize
    size_t offset = (currentFrame * CUBE_COUNT + cubeIndex) * uniformAlignedSize;
    queue->writeBuffer(sharedUniformBuffer, offset, &uniforms, sizeof(uniforms));
}

void CubeApp::update(float deltaTime)
{
    // Accumulate total elapsed time
    elapsedTime += deltaTime;

    updateFPS(deltaTime);

    // Update rotation angles (both X and Y axes)
    rotationAngleX += deltaTime * 45.0f; // 45 degrees per second around X
    rotationAngleY += deltaTime * 30.0f; // 30 degrees per second around Y
    if (rotationAngleX >= 360.0f) {
        rotationAngleX -= 360.0f;
    }
    if (rotationAngleY >= 360.0f) {
        rotationAngleY -= 360.0f;
    }

    // Update uniforms for all CUBE_COUNT cubes BEFORE encoding
    for (int i = 0; i < CUBE_COUNT; ++i) {
        updateCube(i);
    }
}

void CubeApp::render()
{
    try {
        auto& frame = frameResources[currentFrame];

        // Wait for this frame's fence to be signaled
        auto waitResult = frame.inFlightFence->wait(gfx::TimeoutInfinite);
        if (!gfx::isSuccess(waitResult)) {
            throw std::runtime_error("Failed to wait for fence");
        }
        frame.inFlightFence->reset();

        // Acquire next image with explicit synchronization
        uint32_t imageIndex;
        auto result = swapchain->acquireNextImage(
            gfx::TimeoutInfinite,
            frame.imageAvailableSemaphore,
            nullptr,
            &imageIndex);

        if (result != gfx::Result::Success) {
            LOG_ERROR("Failed to acquire next image");
            return;
        }

        // Begin command encoder for reuse
        auto commandEncoder = frame.commandEncoder;
        commandEncoder->begin();

        // Begin render pass with the new API
        gfx::Color clearColor{ 0.1f, 0.2f, 0.3f, 1.0f }; // Dark blue background

        gfx::RenderPassBeginDescriptor renderPassBeginDesc{
            .framebuffer = framebuffers[imageIndex],
            .colorClearValues = { clearColor },
            .depthClearValue = 1.0f,
            .stencilClearValue = 0
        };

        {
            auto renderPassEncoder = commandEncoder->beginRenderPass(renderPassBeginDesc);

            // Set pipeline, bind groups, and buffers (using current frame's bind group)
            renderPassEncoder->setPipeline(renderPipeline);

            // Set viewport and scissor to fill the entire render target
            auto swapchainInfo = swapchain->getInfo();
            renderPassEncoder->setViewport({ 0.0f, 0.0f, static_cast<float>(swapchainInfo.extent.width), static_cast<float>(swapchainInfo.extent.height), 0.0f, 1.0f });
            renderPassEncoder->setScissorRect({ 0, 0, swapchainInfo.extent.width, swapchainInfo.extent.height });

            renderPassEncoder->setVertexBuffer(0, vertexBuffer, 0, vertexBufferInfo.size);
            renderPassEncoder->setIndexBuffer(indexBuffer, gfx::IndexFormat::Uint16, 0, indexBufferInfo.size);

            // Draw CUBE_COUNT cubes at different positions
            for (int i = 0; i < CUBE_COUNT; ++i) {
                // Bind the specific cube's bind group (no dynamic offsets)
                renderPassEncoder->setBindGroup(0, frame.uniformBindGroups[i]);

                // Draw indexed (36 indices for the cube)
                renderPassEncoder->drawIndexed(36, 1, 0, 0, 0);
            }
        } // renderPassEncoder destroyed here, ending the render pass

        // Finish command encoding
        commandEncoder->end();

        // Submit with explicit synchronization
        gfx::SubmitDescriptor submitDescriptor{
            .commandEncoders = { commandEncoder },
            .waitSemaphores = { frame.imageAvailableSemaphore },
            .signalSemaphores = { renderFinishedSemaphores[imageIndex] },
            .signalFence = frame.inFlightFence
        };

        auto submitResult = queue->submit(submitDescriptor);
        if (!gfx::isSuccess(submitResult)) {
            throw std::runtime_error("Failed to submit command buffer");
        }

        // Present with explicit synchronization
        gfx::PresentDescriptor presentDescriptor{
            .waitSemaphores = { renderFinishedSemaphores[imageIndex] }
        };

        result = swapchain->present(presentDescriptor);
        if (result != gfx::Result::Success) {
            LOG_ERROR("Failed to present");
        }

        // Advance to next frame
        currentFrame = (currentFrame + 1) % swapchainInfo.imageCount;
    } catch (const std::exception& e) {
        LOG_ERROR("Render error: %s", e.what());
    }
}

bool CubeApp::handleResize(uint32_t width, uint32_t height)
{
    LOG_INFO("Resizing to %dx%d", width, height);

    windowWidth = width;
    windowHeight = height;

    // Wait for all in-flight frames to complete
    device->waitIdle();

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
        LOG_INFO("FPS - Avg: %.1f, Min: %.1f, Max: %.1f | Frame Time - Avg: %.2f ms, Min: %.2f ms, Max: %.2f ms",
            avgFPS, minFPS, maxFPS, avgFrameTime, fpsFrameTimeMin * 1000.0f, fpsFrameTimeMax * 1000.0f);

        // Reset for next second
        fpsFrameCount = 0;
        fpsTimeAccumulator = 0.0f;
        fpsFrameTimeMin = FLT_MAX;
        fpsFrameTimeMax = 0.0f;
    }
}

float CubeApp::getCurrentTime()
{
#if defined(__ANDROID__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (float)ts.tv_sec + (float)ts.tv_nsec / 1000000000.0f;
#elif defined(__EMSCRIPTEN__)
    return (float)emscripten_get_now() / 1000.0f;
#else
    return (float)glfwGetTime();
#endif
}

gfx::PlatformWindowHandle CubeApp::getPlatformWindowHandle()
{
    gfx::PlatformWindowHandle handle{};
#if defined(__ANDROID__)
    handle = gfx::PlatformWindowHandle::fromAndroid(androidApp->window);
    LOG_INFO("Extracted Android handle: window=%p", androidApp->window);
#elif defined(__EMSCRIPTEN__)
    handle = gfx::PlatformWindowHandle::fromEmscripten("#canvas");
#elif defined(_WIN32)
    handle = gfx::PlatformWindowHandle::fromWin32(GetModuleHandle(nullptr), glfwGetWin32Window(window));
    LOG_DEBUG("Extracted Win32 handle: HWND=%p, HINSTANCE=%p", handle.handle.win32.hwnd, handle.handle.win32.hinstance);
#elif defined(__linux__)
    // handle = gfx::PlatformWindowHandle::fromXlib(glfwGetX11Display(), glfwGetX11Window(window));
    // LOG_DEBUG("Extracted Wayland handle: Window=%lld, Display=%p", handle.handle.xlib.window, handle.handle.xlib.display);
    handle = gfx::PlatformWindowHandle::fromWayland(glfwGetWaylandDisplay(), glfwGetWaylandWindow(window));
    LOG_DEBUG("Extracted Wayland handle: Surface=%p, Display=%p", handle.handle.wayland.surface, handle.handle.wayland.display);
#elif defined(__APPLE__)
    handle = gfx::PlatformWindowHandle::fromMetal(glfwGetCocoaWindow(window));
    LOG_DEBUG("Extracted Metal handle: Layer=%p", handle.handle.metal.layer);
#endif
    return handle;
}

#ifndef __ANDROID__
void CubeApp::errorCallback(int error, const char* description)
{
    LOG_ERROR("GLFW Error %d: %s", error, description);
}

void CubeApp::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    auto* app = static_cast<CubeApp*>(glfwGetWindowUserPointer(window));
    if (app) {
        app->windowWidth = static_cast<uint32_t>(width);
        app->windowHeight = static_cast<uint32_t>(height);
    }
}

void CubeApp::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods; // Suppress unused parameter warnings

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}
#endif // !__ANDROID__

std::vector<uint8_t> CubeApp::loadBinaryFile(const char* filepath)
{
#ifdef __ANDROID__
    if (!androidApp || !androidApp->activity || !androidApp->activity->assetManager) {
        LOG_ERROR("AssetManager not available for file: %s", filepath);
        return {};
    }

    AAssetManager* assetManager = androidApp->activity->assetManager;
    AAsset* asset = AAssetManager_open(assetManager, filepath, AASSET_MODE_BUFFER);
    if (!asset) {
        LOG_ERROR("Failed to open asset: %s", filepath);
        return {};
    }

    size_t size = AAsset_getLength(asset);
    std::vector<uint8_t> buffer(size);

    int bytesRead = AAsset_read(asset, buffer.data(), size);
    AAsset_close(asset);

    if (bytesRead < 0 || static_cast<size_t>(bytesRead) != size) {
        LOG_ERROR("Failed to read complete asset: %s", filepath);
        return {};
    }

    return buffer;
#else
    std::FILE* file = std::fopen(filepath, "rb");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filepath);
        return {};
    }

    // Get file size
    std::fseek(file, 0, SEEK_END);
    long fileSize = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);

    if (fileSize <= 0) {
        LOG_ERROR("Invalid file size for: %s", filepath);
        std::fclose(file);
        return {};
    }

    // Read file into vector
    std::vector<uint8_t> buffer(fileSize);
    size_t bytesRead = std::fread(buffer.data(), 1, fileSize, file);
    std::fclose(file);

    if (bytesRead != static_cast<size_t>(fileSize)) {
        LOG_ERROR("Failed to read complete file: %s", filepath);
        return {};
    }

    return buffer;
#endif
}

std::string CubeApp::loadTextFile(const char* filepath)
{
#ifdef __ANDROID__
    if (!androidApp || !androidApp->activity || !androidApp->activity->assetManager) {
        LOG_ERROR("AssetManager not available for file: %s", filepath);
        return {};
    }

    AAssetManager* assetManager = androidApp->activity->assetManager;
    AAsset* asset = AAssetManager_open(assetManager, filepath, AASSET_MODE_BUFFER);
    if (!asset) {
        LOG_ERROR("Failed to open asset: %s", filepath);
        return {};
    }

    size_t size = AAsset_getLength(asset);
    std::string buffer(size, '\0');

    int bytesRead = AAsset_read(asset, buffer.data(), size);
    AAsset_close(asset);

    if (bytesRead < 0 || static_cast<size_t>(bytesRead) != size) {
        LOG_ERROR("Failed to read complete asset: %s", filepath);
        return {};
    }

    return buffer;
#else
    std::FILE* file = std::fopen(filepath, "r");
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filepath);
        return {};
    }

    // Get file size
    std::fseek(file, 0, SEEK_END);
    long fileSize = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);

    if (fileSize <= 0) {
        LOG_ERROR("Invalid file size for: %s", filepath);
        std::fclose(file);
        return {};
    }

    // Read file into string
    std::string buffer(fileSize, '\0');
    size_t bytesRead = std::fread(buffer.data(), 1, fileSize, file);
    std::fclose(file);

    if (bytesRead != static_cast<size_t>(fileSize)) {
        LOG_ERROR("Failed to read complete file: %s", filepath);
        return {};
    }

    return buffer;
#endif
}

namespace math {
void matrixIdentity(Mat4& matrix)
{
    for (auto& row : matrix.m) {
        row.fill(0.0f);
    }
    matrix.m[0][0] = matrix.m[1][1] = matrix.m[2][2] = matrix.m[3][3] = 1.0f;
}

void matrixPerspective(Mat4& matrix, float fovy, float aspect, float nearPlane, float farPlane, gfx::Backend backend)
{
    for (auto& row : matrix.m) {
        row.fill(0.0f);
    }

    float f = 1.0f / std::tan(fovy / 2.0f);
    matrix.m[0][0] = f / aspect;
    if (backend == gfx::Backend::Vulkan) {
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

    // Check for zero-length forward vector
    if (!vectorNormalize(forward)) {
        matrixIdentity(matrix);
        return;
    }

    // Calculate right vector (cross product of forward and up)
    Vec3 right = {
        forward.y * up.z - forward.z * up.y,
        forward.z * up.x - forward.x * up.z,
        forward.x * up.y - forward.y * up.x
    };

    // Check for zero-length right vector (forward and up are parallel)
    if (!vectorNormalize(right)) {
        matrixIdentity(matrix);
        return;
    }

    // Calculate up vector (cross product of right and forward)
    Vec3 upCorrect = {
        right.y * forward.z - right.z * forward.y,
        right.z * forward.x - right.x * forward.z,
        right.x * forward.y - right.y * forward.x
    };

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

#ifdef __ANDROID__
// ============================================================================
// Android Platform Implementation
// ============================================================================

bool CubeApp::mainLoopIteration()
{
    static int frameCount = 0;

    int events;
    struct android_poll_source* source;

    // Poll all events
    while (ALooper_pollOnce(animating ? 0 : -1, nullptr, &events, reinterpret_cast<void**>(&source)) >= 0) {
        if (source != nullptr) {
            source->process(androidApp, source);
        }

        if (androidApp->destroyRequested != 0) {
            LOG_INFO("Destroy requested");
            cleanup();
            return false;
        }
    }

    if (animating && instance) {
        float currentTime = getCurrentTime();
        float deltaTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;

        update(deltaTime);
        render();
    }

    return true;
}

void CubeApp::handleAppCommand(struct android_app* state, int32_t cmd)
{
    auto* app = static_cast<CubeApp*>(state->userData);

    switch (cmd) {
    case APP_CMD_INIT_WINDOW:
        if (state->window != nullptr) {
            app->windowWidth = static_cast<uint32_t>(ANativeWindow_getWidth(state->window));
            app->windowHeight = static_cast<uint32_t>(ANativeWindow_getHeight(state->window));
            LOG_INFO("Window initialized: %dx%d", app->windowWidth, app->windowHeight);
            LOG_INFO("androidApp=%p, window=%p", state, state->window);

            if (!app->instance) {
                // First time init
                LOG_INFO("Starting initialization...");
                if (app->init()) {
                    app->animating = true;
                    LOG_INFO("Application initialized successfully! Animating set to true");
                } else {
                    LOG_ERROR("Failed to initialize graphics - app will not render");
                }
            } else {
                LOG_INFO("App already initialized, instance=%p", app->instance.get());
            }
        } else {
            LOG_WARN("APP_CMD_INIT_WINDOW but window is null");
        }
        break;

    case APP_CMD_TERM_WINDOW:
        LOG_INFO("Window terminating");
        app->animating = false;
        // Only destroy size-dependent resources, not full cleanup
        // The device will be cleaned up when the app actually exits
        if (app->device) {
            app->device->waitIdle();
            app->destroyPerFrameResources();
            app->destroySizeDependentResources();
        }
        break;

    case APP_CMD_GAINED_FOCUS:
        LOG_INFO("Gained focus");
        if (app->instance && app->device) {
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
        if (app->instance && app->device) {
            app->animating = true;
        }
        break;

    case APP_CMD_WINDOW_RESIZED:
        if (state->window != nullptr && app->instance) {
            uint32_t newWidth = static_cast<uint32_t>(ANativeWindow_getWidth(state->window));
            uint32_t newHeight = static_cast<uint32_t>(ANativeWindow_getHeight(state->window));
            if (!app->handleResize(newWidth, newHeight)) {
                LOG_ERROR("Failed to handle window resize");
            }
        }
        break;
    }
}

int32_t CubeApp::handleInput(struct android_app* state, AInputEvent* event)
{
    (void)state;
    (void)event;
    return 0;
}

void android_main(struct android_app* state)
{
    Settings settings;
    settings.backend = gfx::Backend::Vulkan;
    settings.msaaSampleCount = gfx::SampleCount::Count4;
    settings.vsync = true;

    CubeApp app(settings);
    app.setAndroidApp(state);
    app.setAnimating(false);

    state->userData = &app;
    state->onAppCmd = CubeApp::handleAppCommand;
    state->onInputEvent = CubeApp::handleInput;

    LOG_INFO("=== GFX Cube Example (Android C++) ===");

    app.run();
}

#else
// ============================================================================
// Desktop/Web Platform Implementation
// ============================================================================

bool CubeApp::mainLoopIteration()
{
    if (glfwWindowShouldClose(window)) {
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

#if defined(__EMSCRIPTEN__)
void CubeApp::emscriptenMainLoop(void* userData)
{
    CubeApp* app = (CubeApp*)userData;
    if (!app->mainLoopIteration()) {
        emscripten_cancel_main_loop();
        app->cleanup();
    }
}
#endif

static bool parseBackend(const char* backendStr, gfx::Backend& outBackend)
{
    if (strcmp(backendStr, "vulkan") == 0) {
        outBackend = gfx::Backend::Vulkan;
        return true;
    } else if (strcmp(backendStr, "webgpu") == 0) {
        outBackend = gfx::Backend::WebGPU;
        return true;
    } else {
        LOG_ERROR("Unknown backend: %s", backendStr);
        LOG_ERROR("Valid values: vulkan, webgpu");
        return false;
    }
}

static bool parseMsaa(const char* msaaStr, gfx::SampleCount& outSampleCount)
{
    int samples = atoi(msaaStr);
    switch (samples) {
    case 1:
        outSampleCount = gfx::SampleCount::Count1;
        break;
    case 2:
        outSampleCount = gfx::SampleCount::Count2;
        break;
    case 4:
        outSampleCount = gfx::SampleCount::Count4;
        break;
    case 8:
        outSampleCount = gfx::SampleCount::Count8;
        break;
    case 16:
        outSampleCount = gfx::SampleCount::Count16;
        break;
    case 32:
        outSampleCount = gfx::SampleCount::Count32;
        break;
    case 64:
        outSampleCount = gfx::SampleCount::Count64;
        break;
    default:
        LOG_ERROR("Invalid MSAA sample count: %d", samples);
        LOG_ERROR("Valid values: 1, 2, 4, 8, 16, 32, 64");
        return false;
    }
    return true;
}

static bool parseVsync(const char* vsyncStr, bool& outVsync)
{
    int vsync = atoi(vsyncStr);
    if (vsync == 0) {
        outVsync = false;
        return true;
    } else if (vsync == 1) {
        outVsync = true;
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

bool parseArguments(int argc, char** argv, Settings& settings)
{
#if defined(__EMSCRIPTEN__)
    settings.backend = gfx::Backend::WebGPU;
#else
    settings.backend = gfx::Backend::Vulkan;
#endif
    settings.msaaSampleCount = gfx::SampleCount::Count4;
    settings.vsync = true; // VSync on by default

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            i++;
            if (!parseBackend(argv[i], settings.backend)) {
                return false;
            }
        } else if (strcmp(argv[i], "--msaa") == 0 && i + 1 < argc) {
            i++;
            if (!parseMsaa(argv[i], settings.msaaSampleCount)) {
                return false;
            }
        } else if (strcmp(argv[i], "--vsync") == 0 && i + 1 < argc) {
            i++;
            if (!parseVsync(argv[i], settings.vsync)) {
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
    LOG_INFO("=== Cube Example with Unified Graphics API (C++) ===");

    Settings settings;
    if (!parseArguments(argc, argv, settings)) {
        printHelp(argv[0]);
        return -1;
    }

    CubeApp app(settings);
    if (!app.init()) {
        app.cleanup();
        return -1;
    }
    app.run();
    app.cleanup();

    return 0;
}

#endif // __ANDROID__