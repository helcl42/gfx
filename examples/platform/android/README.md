# Android Platform Examples

Android examples using NativeActivity for both C and C++ APIs.

## Available Examples

### C Examples (`c/`)
- **cube/** - Rotating textured cube with depth testing
- **compute/** - Compute shader with animated texture

### C++ Examples (`cpp/`)
- **cube/** - Rotating textured cube with depth testing  
- **compute/** - Compute shader generating animated texture with post-processing

## Quick Start

```bash
# 1. Build native libraries and shaders
./scripts/build_android.sh

# 2. Build APK (C++ examples)
cd examples/platform/android/cpp/cube  # or: cpp/compute
./gradlew assembleDebug

# 3. Install on device
adb install app/build/outputs/apk/debug/app-debug.apk
```

## View Logs

```bash
adb logcat -s GFX_CUBE_CPP      # C++ cube
adb logcat -s GFX_COMPUTE_CPP   # C++ compute
```

## Requirements

- Android NDK r27+ (for C++20 support)
- Java 17 (for Gradle)
- Device with Vulkan support
