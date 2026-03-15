#!/bin/bash
# Build script for Android examples
# Usage: ./build_android.sh [ABI] [BUILD_TYPE]
#   ABI: arm64-v8a (default), armeabi-v7a, x86_64, x86
#   BUILD_TYPE: Release (default), Debug

set -e

# Configuration
ABI=${1:-arm64-v8a}
BUILD_TYPE=${2:-Release}
API_LEVEL=26  # Minimum for better C++20 support

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building GFX Android Examples ===${NC}"
echo "ABI: ${ABI}"
echo "Build Type: ${BUILD_TYPE}"
echo "API Level: ${API_LEVEL}"
echo ""

# Check for Android NDK - prefer NDK 27+ for C++20 support
if [ -z "$ANDROID_NDK_ROOT" ]; then
    # Try to auto-detect NDK in common locations
    if [ -d "$HOME/Library/Android/sdk/ndk/27.2.12479018" ]; then
        export ANDROID_NDK_ROOT="$HOME/Library/Android/sdk/ndk/27.2.12479018"
        echo -e "${YELLOW}Auto-detected NDK 27:${NC} $ANDROID_NDK_ROOT"
    elif [ -d "$ANDROID_HOME/ndk/27.2.12479018" ]; then
        export ANDROID_NDK_ROOT="$ANDROID_HOME/ndk/27.2.12479018"
        echo -e "${YELLOW}Auto-detected NDK 27:${NC} $ANDROID_NDK_ROOT"
    else
        echo -e "${RED}Error: ANDROID_NDK_ROOT environment variable not set${NC}"
        echo "Please set it to your Android NDK r27+ installation path:"
        echo "  export ANDROID_NDK_ROOT=/path/to/android-ndk-r27"
        echo ""
        echo "This project requires NDK r27+ for C++20 support (std::format)."
        echo "Install with: sdkmanager 'ndk;27.2.12479018'"
        exit 1
    fi
fi

if [ ! -d "$ANDROID_NDK_ROOT" ]; then
    echo -e "${RED}Error: ANDROID_NDK_ROOT directory does not exist: $ANDROID_NDK_ROOT${NC}"
    exit 1
fi

# Verify NDK version (should be r27+)
NDK_VERSION=$(basename "$ANDROID_NDK_ROOT")
if [[ ! "$NDK_VERSION" =~ ^(27|28|29|[3-9][0-9])\. ]]; then
    echo -e "${RED}Error: This project requires NDK r27+ for C++20 support${NC}"
    echo "Current NDK: $NDK_VERSION"
    echo "Location: $ANDROID_NDK_ROOT"
    echo ""
    echo "Please install NDK r27 or later:"
    echo "  sdkmanager 'ndk;27.2.12479018'"
    echo ""
    echo "Then set ANDROID_NDK_ROOT or let the script auto-detect it:"
    echo "  export ANDROID_NDK_ROOT=\$HOME/Library/Android/sdk/ndk/27.2.12479018"
    exit 1
fi

echo -e "${GREEN}Using Android NDK r${NDK_VERSION}:${NC} $ANDROID_NDK_ROOT"
echo ""

# Build directory
BUILD_DIR="build_android_${ABI}"

# Clean build directory if NDK version changed
if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
    CACHED_NDK=$(grep "CMAKE_ANDROID_NDK:PATH" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | cut -d'=' -f2 || echo "")
    if [ -n "$CACHED_NDK" ] && [ "$CACHED_NDK" != "$ANDROID_NDK_ROOT" ]; then
        echo -e "${YELLOW}NDK version changed, cleaning build directory...${NC}"
        echo "  Old: $CACHED_NDK"
        echo "  New: $ANDROID_NDK_ROOT"
        rm -rf "$BUILD_DIR"
    fi
fi

# Configure CMake
echo -e "${YELLOW}Configuring CMake...${NC}"
cmake -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="android-${API_LEVEL}" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_VULKAN_BACKEND=ON \
    -DBUILD_WEBGPU_BACKEND=OFF \
    -DBUILD_CPP_WRAPPER=ON \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_TESTS=OFF

echo ""
echo -e "${YELLOW}Building...${NC}"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE"

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "Output libraries:"
echo "  - $BUILD_DIR/libgfx.so"
echo "  - $BUILD_DIR/libgfx_cpp.so (if C++ wrapper enabled)"
echo "  - $BUILD_DIR/examples/platform/android/c/libcube_example_android_c.so"
echo ""
echo "Next steps:"
echo "  1. Navigate to a examples/platform/android directory go to a example folder"
echo "  2. Assemble APK using gradle (./gradlew assembleDebug or ./gradlew assembleRelease)"
echo "  3. Install APK using adb (adb install -r app/build/outputs/apk/debug/app-debug.apk)s"
echo ""
echo "For detailed instructions, see: examples/platform/android/README.md"
