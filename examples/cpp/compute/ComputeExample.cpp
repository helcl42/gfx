#include <gfx_cpp/gfx.hpp>

#ifdef __ANDROID__
#include <android/log.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <time.h>

// Android logging macros
#define LOG_TAG "GFX_COMPUTE_CPP"
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOG_WARN(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOG_DEBUG(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#else
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
#endif // __ANDROID__

#ifdef Success
#undef Success
#endif

#ifdef None
#undef None
#endif

#include <array>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <memory>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr uint32_t WINDOW_WIDTH = 800;
static constexpr uint32_t WINDOW_HEIGHT = 600;
static constexpr uint32_t COMPUTE_TEXTURE_WIDTH = 512;
static constexpr uint32_t COMPUTE_TEXTURE_HEIGHT = 512;
static constexpr gfx::Format COLOR_FORMAT = gfx::Format::B8G8R8A8UnormSrgb;

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

// Uniform structures
struct ComputeUniformData {
    float time;
    float padding[3]; // WebGPU requires 16-byte alignment for uniform buffers
};

struct RenderUniformData {
    float postProcessStrength;
    float padding[3]; // WebGPU requires 16-byte alignment for uniform buffers
};

// Application settings/configuration
struct Settings {
    gfx::Backend backend;
    bool vsync;
};

// Per-frame resources
struct PerFrameResources {
    std::shared_ptr<gfx::Semaphore> imageAvailableSemaphore;
    std::shared_ptr<gfx::Fence> inFlightFence;
    std::shared_ptr<gfx::CommandEncoder> commandEncoder;
    std::shared_ptr<gfx::BindGroup> computeBindGroup;
    std::shared_ptr<gfx::Buffer> computeUniformBuffer;
    std::shared_ptr<gfx::BindGroup> renderBindGroup;
    std::shared_ptr<gfx::Buffer> renderUniformBuffer;
};

class ComputeApp {
public:
    explicit ComputeApp(const Settings& settings);
    ~ComputeApp() = default;

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
    bool createSwapchain(uint32_t width, uint32_t height);
    void destroySwapchain();
    bool createRenderPass();
    void destroyRenderPass();
    bool createFramebuffers();
    void destroyFramebuffers();

    bool createComputeTexture();
    void destroyComputeTexture();
    bool createComputeShaders();
    void destroyComputeShaders();
    bool createComputeBindGroupLayout();
    void destroyComputeBindGroupLayout();
    bool createComputePipeline();
    void destroyComputePipeline();
    bool transitionComputeTexture();
    bool createComputeResources();
    void destroyComputeResources();

    bool createSampler();
    void destroySampler();
    bool createRenderShaders();
    void destroyRenderShaders();
    bool createRenderBindGroupLayout();
    void destroyRenderBindGroupLayout();
    bool createRenderPipeline();
    void destroyRenderPipeline();
    bool createRenderResources();
    void destroyRenderResources();

    void update(float deltaTime);
    void updateFPS(float deltaTime);
    bool handleResize(uint32_t width, uint32_t height);
    void render();
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

    // Compute resources
    std::shared_ptr<gfx::Texture> computeTexture;
    std::shared_ptr<gfx::TextureView> computeTextureView;
    std::shared_ptr<gfx::Shader> computeShader;
    std::shared_ptr<gfx::ComputePipeline> computePipeline;
    std::shared_ptr<gfx::BindGroupLayout> computeBindGroupLayout;

    // Render resources (fullscreen quad)
    std::shared_ptr<gfx::Shader> vertexShader;
    std::shared_ptr<gfx::Shader> fragmentShader;
    std::shared_ptr<gfx::RenderPipeline> renderPipeline;
    std::shared_ptr<gfx::BindGroupLayout> renderBindGroupLayout;
    std::shared_ptr<gfx::Sampler> sampler;
    std::shared_ptr<gfx::RenderPass> renderPass;
    std::vector<std::shared_ptr<gfx::Framebuffer>> framebuffers;

    uint32_t windowWidth = WINDOW_WIDTH;
    uint32_t windowHeight = WINDOW_HEIGHT;
    uint32_t previousWidth = WINDOW_WIDTH;
    uint32_t previousHeight = WINDOW_HEIGHT;

    // Per-frame resources
    std::vector<PerFrameResources> frameResources;
    std::vector<std::shared_ptr<gfx::Semaphore>> renderFinishedSemaphores;
    size_t currentFrame = 0;

    // State
    float elapsedTime = 0.0f;
    float lastFrameTime = 0.0f;

    // FPS tracking
    uint32_t fpsFrameCount = 0;
    float fpsTimeAccumulator = 0.0f;
    float fpsFrameTimeMin = FLT_MAX;
    float fpsFrameTimeMax = 0.0f;
};

ComputeApp::ComputeApp(const Settings& settings)
    : settings(settings)
{
}

bool ComputeApp::init()
{
    if (!createWindow(WINDOW_WIDTH, WINDOW_HEIGHT)) {
        return false;
    }

    if (!createGraphics()) {
        return false;
    }

    if (!createSizeDependentResources(windowWidth, windowHeight)) {
        return false;
    }

    if (!createComputeResources()) {
        return false;
    }

    if (!createRenderResources()) {
        return false;
    }

    if (!createPerFrameResources()) {
        return false;
    }

    LOG_INFO("Application initialized successfully!");
    LOG_INFO("Press ESC to exit");

    return true;
}

void ComputeApp::run()
{
    // Run main loop (platform-specific)
#if defined(__EMSCRIPTEN__)
    // Note: emscripten_set_main_loop_arg returns immediately and never blocks
    // Cleanup happens in emscriptenMainLoop when the loop exits
    // Execution continues in the browser event loop
    emscripten_set_main_loop_arg(ComputeApp::emscriptenMainLoop, this, 0, 1);
#else
    while (mainLoopIteration()) {
        // Loop continues until mainLoopIteration returns false
    }
#endif
}

void ComputeApp::cleanup()
{
    if (device) {
        device->waitIdle();
    }

    // Destroy in reverse order of creation
    destroyPerFrameResources();
    destroyRenderResources();
    destroyComputeResources();
    destroySizeDependentResources();
    destroyGraphics();
    destroyWindow();
}

bool ComputeApp::createWindow(uint32_t width, uint32_t height)
{
#ifdef __ANDROID__
    if (!androidApp || !androidApp->window) {
        LOG_ERROR("Android app or window is null");
        return false;
    }

    windowWidth = static_cast<uint32_t>(ANativeWindow_getWidth(androidApp->window));
    windowHeight = static_cast<uint32_t>(ANativeWindow_getHeight(androidApp->window));
    LOG_INFO("Android window size: %ux%u", windowWidth, windowHeight);
    return true;
#else
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        LOG_ERROR("Failed to initialize GLFW");
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    const char* backendName = (settings.backend == gfx::Backend::Vulkan) ? "Vulkan" : "WebGPU";
    std::string windowTitle = std::string("Compute & Postprocess Example (C++) - ") + backendName;
    window = glfwCreateWindow(windowWidth, windowHeight, windowTitle.c_str(), nullptr, nullptr);
    if (!window) {
        LOG_ERROR("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }

    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetKeyCallback(window, keyCallback);

    return true;
#endif
}

void ComputeApp::destroyWindow()
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

bool ComputeApp::createGraphics()
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
            .applicationName = "Compute & Postprocess Example (C++)",
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

        // Create surface using native platform handles
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

void ComputeApp::destroyGraphics()
{
    surface.reset();
    queue.reset();
    device.reset();
    adapter.reset();
    instance.reset();
    gfx::unloadBackend(settings.backend);
}

bool ComputeApp::createPerFrameResources()
{
    try {
        frameResources.resize(swapchainInfo.imageCount);

        gfx::SemaphoreDescriptor semaphoreDesc{
            .type = gfx::SemaphoreType::Binary
        };

        gfx::FenceDescriptor fenceDesc{
            .signaled = true
        };

        // Create compute uniform buffer descriptor
        gfx::BufferDescriptor computeUniformBufferDesc{
            .label = "Compute Uniform Buffer",
            .size = sizeof(ComputeUniformData),
            .usage = gfx::BufferUsage::Uniform | gfx::BufferUsage::CopyDst
        };

        // Create render uniform buffer descriptor
        gfx::BufferDescriptor renderUniformBufferDesc{
            .label = "Render Uniform Buffer",
            .size = sizeof(RenderUniformData),
            .usage = gfx::BufferUsage::Uniform | gfx::BufferUsage::CopyDst
        };

        for (size_t i = 0; i < swapchainInfo.imageCount; ++i) {
            auto& frame = frameResources[i];

            // Create semaphore
            frame.imageAvailableSemaphore = device->createSemaphore(semaphoreDesc);
            if (!frame.imageAvailableSemaphore) {
                LOG_ERROR("Failed to create image available semaphore %zu", i);
                return false;
            }

            // Create fence
            frame.inFlightFence = device->createFence(fenceDesc);
            if (!frame.inFlightFence) {
                LOG_ERROR("Failed to create fence %zu", i);
                return false;
            }

            // Create command encoder
            frame.commandEncoder = device->createCommandEncoder({ .label = "Command Encoder " + std::to_string(i) });
            if (!frame.commandEncoder) {
                LOG_ERROR("Failed to create command encoder %zu", i);
                return false;
            }

            // Create compute uniform buffer
            frame.computeUniformBuffer = device->createBuffer(computeUniformBufferDesc);
            if (!frame.computeUniformBuffer) {
                LOG_ERROR("Failed to create compute uniform buffer %zu", i);
                return false;
            }

            // Create compute bind group
            gfx::BindGroupEntry textureEntry{};
            textureEntry.binding = 0;
            textureEntry.resource = computeTextureView;

            gfx::BindGroupEntry computeBufferEntry{};
            computeBufferEntry.binding = 1;
            computeBufferEntry.resource = frame.computeUniformBuffer;
            computeBufferEntry.offset = 0;
            computeBufferEntry.size = sizeof(ComputeUniformData);

            gfx::BindGroupDescriptor computeBindGroupDesc{
                .label = "Compute Bind Group " + std::to_string(i),
                .layout = computeBindGroupLayout,
                .entries = { textureEntry, computeBufferEntry }
            };

            frame.computeBindGroup = device->createBindGroup(computeBindGroupDesc);
            if (!frame.computeBindGroup) {
                LOG_ERROR("Failed to create compute bind group %zu", i);
                return false;
            }

            // Create render uniform buffer
            frame.renderUniformBuffer = device->createBuffer(renderUniformBufferDesc);
            if (!frame.renderUniformBuffer) {
                LOG_ERROR("Failed to create render uniform buffer %zu", i);
                return false;
            }

            // Create render bind group
            gfx::BindGroupEntry samplerBindEntry{};
            samplerBindEntry.binding = 0;
            samplerBindEntry.resource = sampler;

            gfx::BindGroupEntry textureBindEntry{};
            textureBindEntry.binding = 1;
            textureBindEntry.resource = computeTextureView;

            gfx::BindGroupEntry renderBufferEntry{};
            renderBufferEntry.binding = 2;
            renderBufferEntry.resource = frame.renderUniformBuffer;
            renderBufferEntry.offset = 0;
            renderBufferEntry.size = sizeof(RenderUniformData);

            gfx::BindGroupDescriptor renderBindGroupDesc{
                .label = "Render Bind Group " + std::to_string(i),
                .layout = renderBindGroupLayout,
                .entries = { samplerBindEntry, textureBindEntry, renderBufferEntry }
            };

            frame.renderBindGroup = device->createBindGroup(renderBindGroupDesc);
            if (!frame.renderBindGroup) {
                LOG_ERROR("Failed to create render bind group %zu", i);
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create per-frame resources: %s", e.what());
        return false;
    }
}

void ComputeApp::destroyPerFrameResources()
{
    if (device) {
        device->waitIdle();
    }

    for (auto& frame : frameResources) {
        frame.renderBindGroup.reset();
        frame.renderUniformBuffer.reset();
        frame.computeBindGroup.reset();
        frame.computeUniformBuffer.reset();
        frame.commandEncoder.reset();
        frame.inFlightFence.reset();
        frame.imageAvailableSemaphore.reset();
    }

    frameResources.clear();
}

bool ComputeApp::createSizeDependentResources(uint32_t width, uint32_t height)
{
    if (!createSwapchain(width, height)) {
        return false;
    }

    if (!createRenderPass()) {
        return false;
    }

    if (!createFramebuffers()) {
        return false;
    }

    return true;
}

void ComputeApp::destroySizeDependentResources()
{
    destroyFramebuffers();
    destroyRenderPass();
    destroySwapchain();
}

bool ComputeApp::createSwapchain(uint32_t width, uint32_t height)
{
    try {
        // Query surface capabilities
        auto surfaceInfo = surface->getInfo(adapter);
        LOG_INFO("Surface Info:");
        LOG_INFO("  Image Count: min %u, max %u", surfaceInfo.minImageCount, surfaceInfo.maxImageCount);

        gfx::SwapchainDescriptor swapchainDesc{
            .label = "Main Swapchain",
            .surface = surface,
            .extent = { width, height },
            .format = COLOR_FORMAT,
            .usage = gfx::TextureUsage::RenderAttachment,
            .presentMode = settings.vsync ? gfx::PresentMode::Fifo : gfx::PresentMode::Immediate,
            .imageCount = static_cast<uint32_t>(surfaceInfo.minImageCount)
        };

        swapchain = device->createSwapchain(swapchainDesc);
        if (!swapchain) {
            LOG_ERROR("Failed to create swapchain");
            return false;
        }

        // Get swapchain info and create render finished semaphores (one per swapchain image)
        swapchainInfo = swapchain->getInfo();
        renderFinishedSemaphores.resize(swapchainInfo.imageCount);

        gfx::SemaphoreDescriptor semaphoreDesc{
            .type = gfx::SemaphoreType::Binary
        };

        for (uint32_t i = 0; i < swapchainInfo.imageCount; ++i) {
            semaphoreDesc.label = "Render Finished Semaphore Image " + std::to_string(i);
            renderFinishedSemaphores[i] = device->createSemaphore(semaphoreDesc);
            if (!renderFinishedSemaphores[i]) {
                LOG_ERROR("Failed to create render finished semaphore %u", i);
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Swapchain creation error: %s", e.what());
        return false;
    }
}

void ComputeApp::destroySwapchain()
{
    // Clean up render finished semaphores
    for (size_t i = 0; i < renderFinishedSemaphores.size(); ++i) {
        renderFinishedSemaphores[i].reset();
    }
    renderFinishedSemaphores.clear();

    swapchain.reset();
}

bool ComputeApp::createRenderPass()
{
    try {
        auto swapchainInfo = swapchain->getInfo();

        gfx::RenderPassCreateDescriptor renderPassDesc{
            .label = "Main Render Pass"
        };

        gfx::RenderPassColorAttachment colorAttachment{};
        colorAttachment.target.format = swapchainInfo.format;
        colorAttachment.target.sampleCount = gfx::SampleCount::Count1;
        colorAttachment.target.ops.load = gfx::LoadOp::Clear;
        colorAttachment.target.ops.store = gfx::StoreOp::Store;
        colorAttachment.target.finalLayout = gfx::TextureLayout::PresentSrc;

        renderPassDesc.colorAttachments.push_back(colorAttachment);

        renderPass = device->createRenderPass(renderPassDesc);
        if (!renderPass) {
            LOG_ERROR("Failed to create render pass");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Render pass creation error: %s", e.what());
        return false;
    }
}

void ComputeApp::destroyRenderPass()
{
    renderPass.reset();
}

bool ComputeApp::createFramebuffers()
{
    try {
        auto swapchainInfo = swapchain->getInfo();
        framebuffers.resize(swapchainInfo.imageCount);

        for (uint32_t i = 0; i < swapchainInfo.imageCount; ++i) {
            gfx::FramebufferDescriptor framebufferDesc{
                .label = "Framebuffer " + std::to_string(i),
                .renderPass = renderPass,
                .colorAttachments = { { swapchain->getTextureView(i) } },
                .extent = { swapchainInfo.extent.width, swapchainInfo.extent.height }
            };

            framebuffers[i] = device->createFramebuffer(framebufferDesc);
            if (!framebuffers[i]) {
                LOG_ERROR("Failed to create framebuffer %u", i);
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Framebuffer creation error: %s", e.what());
        return false;
    }
}

void ComputeApp::destroyFramebuffers()
{
    framebuffers.clear();
}

bool ComputeApp::createComputeTexture()
{
    try {
        // Create compute output texture (storage image)
        gfx::TextureDescriptor textureDesc{
            .type = gfx::TextureType::Texture2D,
            .size = { COMPUTE_TEXTURE_WIDTH, COMPUTE_TEXTURE_HEIGHT, 1 },
            .mipLevelCount = 1,
            .sampleCount = gfx::SampleCount::Count1,
            .format = gfx::Format::R8G8B8A8Unorm,
            .usage = gfx::TextureUsage::StorageBinding | gfx::TextureUsage::TextureBinding
        };

        computeTexture = device->createTexture(textureDesc);
        if (!computeTexture) {
            LOG_ERROR("Failed to create compute texture");
            return false;
        }

        gfx::TextureViewDescriptor viewDesc{
            .viewType = gfx::TextureViewType::View2D,
            .format = gfx::Format::R8G8B8A8Unorm,
            .baseMipLevel = 0,
            .mipLevelCount = 1,
            .baseArrayLayer = 0,
            .arrayLayerCount = 1
        };

        computeTextureView = computeTexture->createView(viewDesc);
        if (!computeTextureView) {
            LOG_ERROR("Failed to create compute texture view");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create compute texture: %s", e.what());
        return false;
    }
}

void ComputeApp::destroyComputeTexture()
{
    computeTextureView.reset();
    computeTexture.reset();
}

bool ComputeApp::createComputeShaders()
{
    try {
        // Try shader formats in order of preference
        struct ShaderFormat {
            gfx::ShaderSourceType format;
            const char* shaderPath;
        };

        const ShaderFormat shaderFormats[] = {
            { gfx::ShaderSourceType::SPIRV, "shaders/generate.comp.spv" },
            { gfx::ShaderSourceType::WGSL, "shaders/generate.comp.wgsl" }
        };

        gfx::ShaderSourceType shaderSourceType;
        std::vector<uint8_t> shaderCode;
        bool shaderLoaded = false;

        for (const auto& format : shaderFormats) {
            if (!device->supportsShaderFormat(format.format)) {
                continue;
            }

            LOG_INFO("Loading compute shader: %s", format.shaderPath);

            if (format.format == gfx::ShaderSourceType::SPIRV) {
                shaderCode = loadBinaryFile(format.shaderPath);
            } else {
                auto wgsl = loadTextFile(format.shaderPath);
                shaderCode.assign(wgsl.begin(), wgsl.end());
            }

            if (!shaderCode.empty()) {
                shaderSourceType = format.format;
                LOG_INFO("Successfully loaded compute shader (%zu bytes)", shaderCode.size());
                shaderLoaded = true;
                break;
            }

            // Failed to load this format, clear and try next
            LOG_ERROR("Failed to load compute shader for format");
            shaderCode.clear();
        }

        if (!shaderLoaded) {
            LOG_ERROR("Error: No supported shader format found or failed to load shader");
            return false;
        }

        gfx::ShaderDescriptor shaderDesc{
            .label = "Compute Shader",
            .sourceType = shaderSourceType,
            .code = shaderCode,
            .entryPoint = "main"
        };

        computeShader = device->createShader(shaderDesc);
        if (!computeShader) {
            LOG_ERROR("Failed to create compute shader");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create compute shaders: %s", e.what());
        return false;
    }
}

void ComputeApp::destroyComputeShaders()
{
    computeShader.reset();
}

bool ComputeApp::createComputeBindGroupLayout()
{
    try {
        // Create compute bind group layout
        gfx::BindGroupLayoutEntry storageTextureEntry{
            .binding = 0,
            .visibility = gfx::ShaderStage::Compute,
            .count = 1,
            .resource = gfx::BindGroupLayoutEntry::StorageTextureBinding{
                .format = gfx::Format::R8G8B8A8Unorm,
                .access = gfx::StorageTextureAccess::WriteOnly,
                .viewDimension = gfx::TextureViewType::View2D,
            }
        };

        gfx::BindGroupLayoutEntry uniformBufferEntry{
            .binding = 1,
            .visibility = gfx::ShaderStage::Compute,
            .count = 1,
            .resource = gfx::BindGroupLayoutEntry::BufferBinding{
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(ComputeUniformData),
            }
        };

        gfx::BindGroupLayoutDescriptor computeLayoutDesc{
            .label = "Compute Bind Group Layout",
            .entries = { storageTextureEntry, uniformBufferEntry }
        };

        computeBindGroupLayout = device->createBindGroupLayout(computeLayoutDesc);
        if (!computeBindGroupLayout) {
            LOG_ERROR("Failed to create compute bind group layout");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create compute bind group layout: %s", e.what());
        return false;
    }
}

void ComputeApp::destroyComputeBindGroupLayout()
{
    computeBindGroupLayout.reset();
}

bool ComputeApp::createComputePipeline()
{
    try {
        // Create compute uniform buffers (one per frame in flight)
        gfx::BufferDescriptor computeUniformBufferDesc{
            .label = "Compute Uniform Buffer",
            .size = sizeof(ComputeUniformData),
            .usage = gfx::BufferUsage::Uniform | gfx::BufferUsage::CopyDst
        };

        // Create compute pipeline
        gfx::ComputePipelineDescriptor computePipelineDesc{
            .label = "Compute Pipeline",
            .compute = computeShader,
            .entryPoint = "main",
            .bindGroupLayouts = { computeBindGroupLayout }
        };

        computePipeline = device->createComputePipeline(computePipelineDesc);
        if (!computePipeline) {
            LOG_ERROR("Failed to create compute pipeline");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create compute pipeline: %s", e.what());
        return false;
    }
}

void ComputeApp::destroyComputePipeline()
{
    computePipeline.reset();
}

bool ComputeApp::transitionComputeTexture()
{
    try {
        // Transition compute texture from Undefined to ShaderReadOnly layout
        gfx::CommandEncoderDescriptor initEncoderDesc{
            .label = "Init Texture Transition"
        };
        auto initEncoder = device->createCommandEncoder(initEncoderDesc);
        if (!initEncoder) {
            LOG_ERROR("Failed to create command encoder for texture transition");
            return false;
        }

        initEncoder->begin();

        gfx::TextureBarrier initBarrier{};
        initBarrier.texture = computeTexture;
        initBarrier.oldLayout = gfx::TextureLayout::Undefined;
        initBarrier.newLayout = gfx::TextureLayout::ShaderReadOnly;
        initBarrier.srcStageMask = gfx::PipelineStage::TopOfPipe;
        initBarrier.dstStageMask = gfx::PipelineStage::FragmentShader;
        initBarrier.srcAccessMask = gfx::AccessFlags::None;
        initBarrier.dstAccessMask = gfx::AccessFlags::ShaderRead;
        initBarrier.baseMipLevel = 0;
        initBarrier.mipLevelCount = 1;
        initBarrier.baseArrayLayer = 0;
        initBarrier.arrayLayerCount = 1;

        gfx::PipelineBarrierDescriptor barrierDesc{
            .textureBarriers = { initBarrier }
        };
        initEncoder->pipelineBarrier(barrierDesc);
        initEncoder->end();

        gfx::FenceDescriptor initFenceDesc{
            .signaled = false
        };
        auto initFence = device->createFence(initFenceDesc);
        if (!initFence) {
            LOG_ERROR("Failed to create fence for texture transition");
            return false;
        }

        gfx::SubmitDescriptor submitDescriptor{
            .commandEncoders = { initEncoder },
            .signalFence = initFence
        };

        queue->submit(submitDescriptor);
        auto waitResult = initFence->wait(gfx::TimeoutInfinite);
        if (!gfx::isSuccess(waitResult)) {
            LOG_ERROR("Failed to wait for texture transition fence");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to transition compute texture: %s", e.what());
        return false;
    }
}

bool ComputeApp::createComputeResources()
{
    if (!createComputeTexture()) {
        return false;
    }

    if (!createComputeShaders()) {
        return false;
    }

    if (!createComputeBindGroupLayout()) {
        return false;
    }

    if (!createComputePipeline()) {
        return false;
    }

    if (!transitionComputeTexture()) {
        return false;
    }

    LOG_INFO("Compute resources created successfully");
    return true;
}

void ComputeApp::destroyComputeResources()
{
    destroyComputePipeline();
    destroyComputeBindGroupLayout();
    destroyComputeShaders();
    destroyComputeTexture();
}

bool ComputeApp::createSampler()
{
    try {
        // Create sampler
        gfx::SamplerDescriptor samplerDesc{
            .addressModeU = gfx::AddressMode::ClampToEdge,
            .addressModeV = gfx::AddressMode::ClampToEdge,
            .magFilter = gfx::FilterMode::Linear,
            .minFilter = gfx::FilterMode::Linear
        };

        sampler = device->createSampler(samplerDesc);
        if (!sampler) {
            LOG_ERROR("Failed to create sampler");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create sampler: %s", e.what());
        return false;
    }
}

void ComputeApp::destroySampler()
{
    sampler.reset();
}

bool ComputeApp::createRenderShaders()
{
    try {
        // Try shader formats in order of preference
        struct ShaderFormat {
            gfx::ShaderSourceType format;
            const char* vertexPath;
            const char* fragmentPath;
        };

        const ShaderFormat shaderFormats[] = {
            { gfx::ShaderSourceType::SPIRV, "shaders/fullscreen.vert.spv", "shaders/postprocess.frag.spv" },
            { gfx::ShaderSourceType::WGSL, "shaders/fullscreen.vert.wgsl", "shaders/postprocess.frag.wgsl" }
        };

        gfx::ShaderSourceType shaderSourceType;
        std::vector<uint8_t> vertexShaderCode, fragmentShaderCode;
        bool shadersLoaded = false;

        for (const auto& format : shaderFormats) {
            if (!device->supportsShaderFormat(format.format)) {
                continue;
            }

            LOG_INFO("Loading shaders: %s, %s", format.vertexPath, format.fragmentPath);

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
                LOG_INFO("Successfully loaded shaders (vertex: %zu bytes, fragment: %zu bytes)",
                    vertexShaderCode.size(), fragmentShaderCode.size());
                shadersLoaded = true;
                break;
            }

            // Failed to load this format, clear and try next
            LOG_ERROR("Failed to load shaders for format");
            vertexShaderCode.clear();
            fragmentShaderCode.clear();
        }

        if (!shadersLoaded) {
            LOG_ERROR("Error: No supported shader format found or failed to load shaders");
            return false;
        }

        gfx::ShaderDescriptor vertexShaderDesc{
            .label = "Vertex Shader",
            .sourceType = shaderSourceType,
            .code = vertexShaderCode,
            .entryPoint = "main"
        };

        vertexShader = device->createShader(vertexShaderDesc);
        if (!vertexShader) {
            LOG_ERROR("Failed to create vertex shader");
            return false;
        }

        gfx::ShaderDescriptor fragmentShaderDesc{
            .label = "Fragment Shader",
            .sourceType = shaderSourceType,
            .code = fragmentShaderCode,
            .entryPoint = "main"
        };

        fragmentShader = device->createShader(fragmentShaderDesc);
        if (!fragmentShader) {
            LOG_ERROR("Failed to create fragment shader");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create render shaders: %s", e.what());
        return false;
    }
}

void ComputeApp::destroyRenderShaders()
{
    fragmentShader.reset();
    vertexShader.reset();
}

bool ComputeApp::createRenderBindGroupLayout()
{
    try {
        // Create render bind group layout
        gfx::BindGroupLayoutEntry samplerEntry{
            .binding = 0,
            .visibility = gfx::ShaderStage::Fragment,
            .count = 1,
            .resource = gfx::BindGroupLayoutEntry::SamplerBinding{
                .type = gfx::SamplerBindingType::Filtering,
            }
        };

        gfx::BindGroupLayoutEntry textureEntry{
            .binding = 1,
            .visibility = gfx::ShaderStage::Fragment,
            .count = 1,
            .resource = gfx::BindGroupLayoutEntry::TextureBinding{
                .multisampled = false,
                .viewDimension = gfx::TextureViewType::View2D,
            }
        };

        gfx::BindGroupLayoutEntry uniformBufferEntry{
            .binding = 2,
            .visibility = gfx::ShaderStage::Fragment,
            .count = 1,
            .resource = gfx::BindGroupLayoutEntry::BufferBinding{
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(RenderUniformData),
            }
        };

        gfx::BindGroupLayoutDescriptor renderLayoutDesc{
            .label = "Render Bind Group Layout",
            .entries = { samplerEntry, textureEntry, uniformBufferEntry }
        };

        renderBindGroupLayout = device->createBindGroupLayout(renderLayoutDesc);
        if (!renderBindGroupLayout) {
            LOG_ERROR("Failed to create render bind group layout");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create render bind group layout: %s", e.what());
        return false;
    }
}

void ComputeApp::destroyRenderBindGroupLayout()
{
    renderBindGroupLayout.reset();
}

bool ComputeApp::createRenderPipeline()
{
    try {
        // Create render uniform buffers (one per frame in flight)
        gfx::BufferDescriptor renderUniformBufferDesc{
            .label = "Render Uniform Buffer",
            .size = sizeof(RenderUniformData),
            .usage = gfx::BufferUsage::Uniform | gfx::BufferUsage::CopyDst
        };

        // Create render pipeline
        gfx::VertexState vertexState{};
        vertexState.module = vertexShader;
        vertexState.entryPoint = "main";
        vertexState.buffers = {};

        gfx::ColorTargetState colorTarget{};
        colorTarget.format = swapchain->getInfo().format;
        colorTarget.writeMask = gfx::ColorWriteMask::All;

        gfx::FragmentState fragmentState{};
        fragmentState.module = fragmentShader;
        fragmentState.entryPoint = "main";
        fragmentState.targets.push_back(colorTarget);

        gfx::PrimitiveState primitiveState{};
        primitiveState.topology = gfx::PrimitiveTopology::TriangleList;
        primitiveState.frontFace = gfx::FrontFace::CounterClockwise;
        primitiveState.cullMode = gfx::CullMode::None;
        primitiveState.polygonMode = gfx::PolygonMode::Fill;

        gfx::RenderPipelineDescriptor pipelineDesc{
            .label = "Render Pipeline",
            .renderPass = renderPass,
            .vertex = vertexState,
            .fragment = fragmentState,
            .primitive = primitiveState,
            .sampleCount = gfx::SampleCount::Count1,
            .bindGroupLayouts = { renderBindGroupLayout }
        };

        renderPipeline = device->createRenderPipeline(pipelineDesc);
        if (!renderPipeline) {
            LOG_ERROR("Failed to create render pipeline");
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create render pipeline: %s", e.what());
        return false;
    }
}

void ComputeApp::destroyRenderPipeline()
{
    renderPipeline.reset();
}

bool ComputeApp::createRenderResources()
{
    if (!createRenderShaders()) {
        return false;
    }

    if (!createSampler()) {
        return false;
    }

    if (!createRenderBindGroupLayout()) {
        return false;
    }

    if (!createRenderPipeline()) {
        return false;
    }

    LOG_INFO("Render resources created successfully");
    return true;
}

void ComputeApp::destroyRenderResources()
{
    destroyRenderPipeline();
    destroyRenderBindGroupLayout();
    destroyRenderShaders();
    destroySampler();
}

void ComputeApp::update(float deltaTime)
{
    elapsedTime += deltaTime;
    updateFPS(deltaTime);
}

void ComputeApp::updateFPS(float deltaTime)
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

bool ComputeApp::handleResize(uint32_t width, uint32_t height)
{
    LOG_INFO("Resizing to %ux%u", width, height);

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

void ComputeApp::render()
{
    try {
        size_t frameIndex = currentFrame;
        auto& frame = frameResources[frameIndex];

        // Wait for previous frame
        auto waitResult = frame.inFlightFence->wait(gfx::TimeoutInfinite);
        if (!gfx::isSuccess(waitResult)) {
            throw std::runtime_error("Failed to wait for frame fence");
        }
        frame.inFlightFence->reset();

        // Acquire swapchain image
        uint32_t imageIndex = 0;
        auto result = swapchain->acquireNextImage(
            UINT64_MAX,
            frame.imageAvailableSemaphore,
            nullptr,
            &imageIndex);

        if (result != gfx::Result::Success) {
            LOG_ERROR("Failed to acquire swapchain image");
            return;
        }

        // Update compute uniforms for current frame
        ComputeUniformData computeUniforms{ .time = elapsedTime };
        queue->writeBuffer(frame.computeUniformBuffer, 0, &computeUniforms, sizeof(computeUniforms));

        // Update render uniforms for current frame
        RenderUniformData renderUniforms{
            .postProcessStrength = 0.5f + 0.5f * std::sin(elapsedTime * 0.5f)
        };
        queue->writeBuffer(frame.renderUniformBuffer, 0, &renderUniforms, sizeof(renderUniforms));

        // Begin command encoder
        auto encoder = frame.commandEncoder;
        encoder->begin();

        // Transition compute texture to GENERAL layout for compute shader write
        gfx::TextureBarrier readToWriteBarrier{
            .texture = computeTexture,
            .oldLayout = gfx::TextureLayout::ShaderReadOnly,
            .newLayout = gfx::TextureLayout::General,
            .srcStageMask = gfx::PipelineStage::FragmentShader,
            .dstStageMask = gfx::PipelineStage::ComputeShader,
            .srcAccessMask = gfx::AccessFlags::ShaderRead,
            .dstAccessMask = gfx::AccessFlags::ShaderWrite,
            .baseMipLevel = 0,
            .mipLevelCount = 1,
            .baseArrayLayer = 0,
            .arrayLayerCount = 1
        };
        encoder->pipelineBarrier({ .textureBarriers = { readToWriteBarrier } });

        // Compute pass: Generate pattern
        {
            gfx::ComputePassBeginDescriptor computePassDesc;
            computePassDesc.label = "Generate Pattern";
            auto computePass = encoder->beginComputePass(computePassDesc);
            computePass->setPipeline(computePipeline);
            computePass->setBindGroup(0, frame.computeBindGroup);

            uint32_t workGroupsX = (COMPUTE_TEXTURE_WIDTH + 15) / 16;
            uint32_t workGroupsY = (COMPUTE_TEXTURE_HEIGHT + 15) / 16;
            computePass->dispatch(workGroupsX, workGroupsY, 1);
        } // computePass destroyed here

        // Transition compute texture for shader read
        gfx::TextureBarrier computeToReadBarrier{
            .texture = computeTexture,
            .oldLayout = gfx::TextureLayout::General,
            .newLayout = gfx::TextureLayout::ShaderReadOnly,
            .srcStageMask = gfx::PipelineStage::ComputeShader,
            .dstStageMask = gfx::PipelineStage::FragmentShader,
            .srcAccessMask = gfx::AccessFlags::ShaderWrite,
            .dstAccessMask = gfx::AccessFlags::ShaderRead,
            .baseMipLevel = 0,
            .mipLevelCount = 1,
            .baseArrayLayer = 0,
            .arrayLayerCount = 1
        };
        encoder->pipelineBarrier({ .textureBarriers = { computeToReadBarrier } });

        // Render pass: Post-process and display
        gfx::Color clearColor{ 0.0f, 0.0f, 0.0f, 1.0f };

        gfx::RenderPassBeginDescriptor renderPassBeginDesc{
            .framebuffer = framebuffers[imageIndex],
            .colorClearValues = { clearColor }
        };

        {
            auto renderPassEncoder = encoder->beginRenderPass(renderPassBeginDesc);

            renderPassEncoder->setPipeline(renderPipeline);
            renderPassEncoder->setBindGroup(0, frame.renderBindGroup);

            // Set viewport and scissor to match the actual swapchain extent
            auto swapchainInfo = swapchain->getInfo();
            renderPassEncoder->setViewport({ 0.0f, 0.0f, static_cast<float>(swapchainInfo.extent.width), static_cast<float>(swapchainInfo.extent.height), 0.0f, 1.0f });
            renderPassEncoder->setScissorRect({ 0, 0, swapchainInfo.extent.width, swapchainInfo.extent.height });

            // Draw fullscreen quad (6 vertices, no buffers needed)
            renderPassEncoder->draw(6, 1, 0, 0);
        } // renderPassEncoder destroyed here

        encoder->end();

        // Submit
        gfx::SubmitDescriptor submitDescriptor{
            .commandEncoders = { encoder },
            .waitSemaphores = { frame.imageAvailableSemaphore },
            .signalSemaphores = { renderFinishedSemaphores[imageIndex] },
            .signalFence = frame.inFlightFence
        };

        auto submitResult = queue->submit(submitDescriptor);
        if (!gfx::isSuccess(submitResult)) {
            throw std::runtime_error("Failed to submit compute commands");
        }

        // Present
        gfx::PresentDescriptor presentDescriptor{
            .waitSemaphores = { renderFinishedSemaphores[imageIndex] }
        };

        result = swapchain->present(presentDescriptor);

        currentFrame = (currentFrame + 1) % swapchainInfo.imageCount;
    } catch (const std::exception& e) {
        LOG_ERROR("Render error: %s", e.what());
    }
}

float ComputeApp::getCurrentTime()
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

gfx::PlatformWindowHandle ComputeApp::getPlatformWindowHandle()
{
    gfx::PlatformWindowHandle handle{};

#if defined(__ANDROID__)
    handle = gfx::PlatformWindowHandle::fromAndroid(androidApp->window);
    LOG_INFO("Extracted Android handle: Window=%p", handle.handle.android.window);
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
void ComputeApp::errorCallback(int error, const char* description)
{
    LOG_ERROR("GLFW Error %d: %s", error, description);
}

void ComputeApp::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    auto* app = static_cast<ComputeApp*>(glfwGetWindowUserPointer(window));
    if (app) {
        app->windowWidth = static_cast<uint32_t>(width);
        app->windowHeight = static_cast<uint32_t>(height);
    }
}

void ComputeApp::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}
#endif // !__ANDROID__

std::vector<uint8_t> ComputeApp::loadBinaryFile(const char* filepath)
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
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        LOG_ERROR("Failed to open file: %s", filepath);
        return {};
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0);

    std::vector<uint8_t> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

    if (!file) {
        LOG_ERROR("Failed to read complete file: %s", filepath);
        return {};
    }

    return buffer;
#endif
}

std::string ComputeApp::loadTextFile(const char* filepath)
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

    std::fseek(file, 0, SEEK_END);
    long fileSize = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);

    if (fileSize <= 0) {
        LOG_ERROR("Invalid file size for: %s", filepath);
        std::fclose(file);
        return {};
    }

    std::string buffer(fileSize, '\0');
    size_t bytesRead = std::fread(const_cast<char*>(buffer.data()), 1, fileSize, file);
    std::fclose(file);

    if (bytesRead != static_cast<size_t>(fileSize)) {
        LOG_ERROR("Failed to read complete file: %s", filepath);
        return {};
    }

    return buffer;
#endif
}

#ifdef __ANDROID__
// ============================================================================
// Android Platform Implementation
// ============================================================================

bool ComputeApp::mainLoopIteration()
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
        if (frameCount == 0) {
            LOG_INFO("Starting render loop - first frame");
        }

        float currentTime = getCurrentTime();
        static float lastTime = 0.0f;
        if (lastTime == 0.0f) {
            lastTime = currentTime;
        }
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        update(deltaTime);
        render();

        frameCount++;
        if (frameCount % 60 == 0) {
            LOG_DEBUG("Rendered %d frames", frameCount);
        }
    } else if (frameCount == 0) {
        // Only log once when not rendering
        static bool logged = false;
        if (!logged) {
            LOG_WARN("Not rendering: animating=%d, instance=%p",
                animating, instance.get());
            logged = true;
        }
    }

    return true;
}

void ComputeApp::handleAppCommand(struct android_app* state, int32_t cmd)
{
    auto* app = static_cast<ComputeApp*>(state->userData);

    switch (cmd) {
    case APP_CMD_INIT_WINDOW:
        if (state->window != nullptr) {
            app->windowWidth = static_cast<uint32_t>(ANativeWindow_getWidth(state->window));
            app->windowHeight = static_cast<uint32_t>(ANativeWindow_getHeight(state->window));
            LOG_INFO("Window initialized: %ux%u", app->windowWidth, app->windowHeight);

            if (!app->instance) {
                if (!app->createWindow(app->windowWidth, app->windowHeight)) {
                    LOG_ERROR("Failed to create window");
                    return;
                }

                if (!app->createGraphics()) {
                    LOG_ERROR("Failed to create graphics");
                    return;
                }

                if (!app->createSizeDependentResources(app->windowWidth, app->windowHeight)) {
                    LOG_ERROR("Failed to create size-dependent resources");
                    return;
                }

                if (!app->createComputeResources()) {
                    LOG_ERROR("Failed to create compute resources");
                    return;
                }

                if (!app->createRenderResources()) {
                    LOG_ERROR("Failed to create render resources");
                    return;
                }

                LOG_INFO("Application initialized successfully");
            }

            app->animating = true;
        }
        break;

    case APP_CMD_TERM_WINDOW:
        LOG_INFO("Window terminated");
        app->animating = false;
        break;

    case APP_CMD_GAINED_FOCUS:
        LOG_INFO("Gained focus");
        app->animating = true;
        break;

    case APP_CMD_LOST_FOCUS:
        LOG_INFO("Lost focus");
        app->animating = false;
        break;
    }
}

int32_t ComputeApp::handleInput(struct android_app* state, AInputEvent* event)
{
    (void)state;
    (void)event;
    return 0;
}

void android_main(struct android_app* state)
{
    Settings settings;
    settings.backend = gfx::Backend::Vulkan;
    settings.vsync = true;

    ComputeApp app(settings);
    app.setAndroidApp(state);
    app.setAnimating(false);

    state->userData = &app;
    state->onAppCmd = ComputeApp::handleAppCommand;
    state->onInputEvent = ComputeApp::handleInput;

    LOG_INFO("=== GFX Compute Example (Android C++) ===");

    app.run();
}

#else
// ============================================================================
// Desktop/Web Platform Implementation
// ============================================================================

bool ComputeApp::mainLoopIteration()
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
void ComputeApp::emscriptenMainLoop(void* userData)
{
    ComputeApp* app = (ComputeApp*)userData;
    if (!app->mainLoopIteration()) {
        emscripten_cancel_main_loop();
        app->cleanup();
    }
}
#endif

static bool parseBackend(const char* backendStr, gfx::Backend& outBackend)
{
    if (std::strcmp(backendStr, "vulkan") == 0) {
        outBackend = gfx::Backend::Vulkan;
        return true;
    } else if (std::strcmp(backendStr, "webgpu") == 0) {
        outBackend = gfx::Backend::WebGPU;
        return true;
    } else {
        LOG_ERROR("Unknown backend: %s", backendStr);
        LOG_ERROR("Valid values: vulkan, webgpu");
        return false;
    }
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

static bool parseArguments(int argc, char** argv, Settings& settings)
{
#if defined(__EMSCRIPTEN__)
    settings.backend = gfx::Backend::WebGPU;
#else
    settings.backend = gfx::Backend::Vulkan;
#endif
    settings.vsync = true; // VSync on by default

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            i++;
            if (!parseBackend(argv[i], settings.backend)) {
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
            LOG_ERROR("Unknown argument: %s", argv[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    LOG_INFO("=== Compute & Postprocess Example (C++) ===");

    Settings settings;
    if (!parseArguments(argc, argv, settings)) {
        printHelp(argv[0]);
        return 0;
    }

    ComputeApp app(settings);

    if (!app.init()) {
        app.cleanup();
        return -1;
    }

    app.run();
    app.cleanup();

    LOG_INFO("Application terminated successfully");
    return 0;
}

#endif // __ANDROID__