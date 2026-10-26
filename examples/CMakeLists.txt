# Examples CMakeLists.txt
# This file contains all example applications for Gfx

cmake_minimum_required(VERSION 3.16)

# stb_image for texture loading
include(FetchContent)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(stb)

# Create an interface library for stb_image
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${stb_SOURCE_DIR})

# GLFW dependency for examples (not needed for Web, Android, or iOS)
if(NOT BUILD_FOR_WEB AND NOT ANDROID AND NOT IOS)
    find_package(glfw3 3.3 QUIET)
    
    if(NOT glfw3_FOUND)
        message(STATUS "GLFW not found, fetching via FetchContent...")
        include(FetchContent)
        FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG 3.4
            GIT_SHALLOW TRUE
        )
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(glfw)
    else()
        message(STATUS "Using system GLFW")
    endif()
endif()

# Gives a native example its own directory holding the executable, its WGSL shaders with their
# SPIR-V compiled by tint, and its textures, so no two examples write the same files
function(add_example_assets target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SHADERS;TEXTURES")
    set(example_dir "${CMAKE_CURRENT_BINARY_DIR}/${target}")
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${example_dir}")
    if(ARG_SHADERS AND NOT BUILD_WEBGPU_BACKEND)
        message(WARNING "No SPIR-V shaders for ${target}: compiling WGSL needs tint from the WebGPU backend")
    endif()

    set(outputs "")
    foreach(shader IN LISTS ARG_SHADERS)
        get_filename_component(shader_name "${shader}" NAME)
        set(wgsl_output "${example_dir}/shaders/${shader_name}")
        add_custom_command(
            OUTPUT "${wgsl_output}"
            COMMAND ${CMAKE_COMMAND} -E copy "${shader}" "${wgsl_output}"
            DEPENDS "${shader}"
            VERBATIM
        )
        list(APPEND outputs "${wgsl_output}")

        if(BUILD_WEBGPU_BACKEND)
            string(REGEX REPLACE "\\.wgsl$" ".spv" spirv_name "${shader_name}")
            set(spirv_output "${example_dir}/shaders/${spirv_name}")
            add_custom_command(
                OUTPUT "${spirv_output}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${example_dir}/shaders"
                COMMAND $<TARGET_FILE:tint_cmd_tint_cmd> "${shader}" -o "${spirv_output}" --format spirv --entry-point main
                DEPENDS "${shader}" tint_cmd_tint_cmd
                COMMENT "Compiling WGSL shader for ${target}: ${shader_name} -> ${spirv_name}"
                VERBATIM
            )
            list(APPEND outputs "${spirv_output}")
        endif()
    endforeach()

    foreach(texture IN LISTS ARG_TEXTURES)
        get_filename_component(texture_name "${texture}" NAME)
        set(texture_output "${example_dir}/textures/${texture_name}")
        add_custom_command(
            OUTPUT "${texture_output}"
            COMMAND ${CMAKE_COMMAND} -E copy "${texture}" "${texture_output}"
            DEPENDS "${texture}"
            VERBATIM
        )
        list(APPEND outputs "${texture_output}")
    endforeach()

    target_sources(${target} PRIVATE ${outputs})
endfunction()

# =============================================================================
# Web Examples (Emscripten)
# =============================================================================
if(BUILD_FOR_WEB)
    # Web Cube Example (Emscripten)
    add_executable(cube_example_web ${CMAKE_CURRENT_SOURCE_DIR}/c/cube/CubeExample.c)
    target_link_libraries(cube_example_web gfx stb_image)
    
    # Emscripten-specific flags for web cube example
    target_compile_options(cube_example_web PRIVATE -DEMSCRIPTEN -sDISABLE_EXCEPTION_CATCHING=0)
    target_link_options(cube_example_web PRIVATE
        --use-port=emdawnwebgpu
        -sUSE_GLFW=3
        -sASYNCIFY=1
        -sALLOW_MEMORY_GROWTH=1
        -sASSERTIONS=2
        -sSTACK_OVERFLOW_CHECK=2
        -sSAFE_HEAP=1
        -sDISABLE_EXCEPTION_CATCHING=0
        -sEXPORTED_RUNTIME_METHODS=requestFullscreen
        --preload-file=${CMAKE_CURRENT_SOURCE_DIR}/c/cube/shaders@shaders
        --preload-file=${CMAKE_CURRENT_SOURCE_DIR}/c/cube/textures@textures
        -g
        -O0
        --shell-file=${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html
    )
    
    # Set output to create HTML, JS, and WASM
    set_target_properties(cube_example_web PROPERTIES
        SUFFIX ".html"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/web"
    )
    
    message(STATUS "Web example will be built to: ${CMAKE_CURRENT_BINARY_DIR}/web/cube_example_web.html")
    
    # Web Compute Example (Emscripten)
    add_executable(compute_example_web ${CMAKE_CURRENT_SOURCE_DIR}/c/compute/ComputeExample.c)
    target_link_libraries(compute_example_web gfx)
    
    # Emscripten-specific flags for web compute example
    target_compile_options(compute_example_web PRIVATE -DEMSCRIPTEN -sDISABLE_EXCEPTION_CATCHING=0)
    target_link_options(compute_example_web PRIVATE
        --use-port=emdawnwebgpu
        -sUSE_GLFW=3
        -sASYNCIFY=1
        -sALLOW_MEMORY_GROWTH=1
        -sASSERTIONS=2
        -sSTACK_OVERFLOW_CHECK=2
        -sSAFE_HEAP=1
        -sDISABLE_EXCEPTION_CATCHING=0
        -sEXPORTED_RUNTIME_METHODS=requestFullscreen
        --preload-file=${CMAKE_CURRENT_SOURCE_DIR}/c/compute/shaders@shaders
        -g
        -O0
        --shell-file=${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html
    )
    
    # Set output to create HTML, JS, and WASM
    set_target_properties(compute_example_web PROPERTIES
        SUFFIX ".html"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/web"
    )
    
    message(STATUS "Web example will be built to: ${CMAKE_CURRENT_BINARY_DIR}/web/compute_example_web.html")
    
    # Web C++ Cube Example (Emscripten)
    if(BUILD_CPP_WRAPPER)
        add_executable(cube_example_cpp_web ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cube/CubeExample.cpp)
        target_link_libraries(cube_example_cpp_web gfx_cpp)
    
    # Emscripten-specific flags for web C++ cube example
    target_compile_options(cube_example_cpp_web PRIVATE -DEMSCRIPTEN -sDISABLE_EXCEPTION_CATCHING=0)
    target_link_options(cube_example_cpp_web PRIVATE
        --use-port=emdawnwebgpu
        -sUSE_GLFW=3
        -sASYNCIFY=1
        -sALLOW_MEMORY_GROWTH=1
        -sASSERTIONS=2
        -sSTACK_OVERFLOW_CHECK=2
        -sSAFE_HEAP=1
        -sDISABLE_EXCEPTION_CATCHING=0
        -sEXPORTED_RUNTIME_METHODS=requestFullscreen
        --preload-file=${CMAKE_CURRENT_SOURCE_DIR}/cpp/cube/shaders@shaders
        -g
        -O0
        --shell-file=${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html
    )
    
    # Set output to create HTML, JS, and WASM
    set_target_properties(cube_example_cpp_web PROPERTIES
        SUFFIX ".html"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/web"
    )
    
    message(STATUS "Web C++ cube example will be built to: ${CMAKE_CURRENT_BINARY_DIR}/web/cube_example_cpp_web.html")
    endif()  # BUILD_CPP_WRAPPER
    
    # Web C++ Compute Example (Emscripten)
    if(BUILD_CPP_WRAPPER)
        add_executable(compute_example_cpp_web ${CMAKE_CURRENT_SOURCE_DIR}/cpp/compute/ComputeExample.cpp)
        target_link_libraries(compute_example_cpp_web gfx_cpp)
        
        # Emscripten-specific flags for web C++ compute example
        target_compile_options(compute_example_cpp_web PRIVATE -DEMSCRIPTEN -sDISABLE_EXCEPTION_CATCHING=0)
        target_link_options(compute_example_cpp_web PRIVATE
            --use-port=emdawnwebgpu
            -sUSE_GLFW=3
            -sASYNCIFY=1
            -sALLOW_MEMORY_GROWTH=1
            -sASSERTIONS=0
            -sDISABLE_EXCEPTION_CATCHING=0
            -sEXPORTED_RUNTIME_METHODS=requestFullscreen
            --preload-file=${CMAKE_CURRENT_SOURCE_DIR}/c/compute/shaders@shaders
            -g
            -O0
            --shell-file=${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html
        )
        
        # Set output to create HTML, JS, and WASM
        set_target_properties(compute_example_cpp_web PROPERTIES
            SUFFIX ".html"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/web"
        )
        
        message(STATUS "Web C++ compute example will be built to: ${CMAKE_CURRENT_BINARY_DIR}/web/compute_example_cpp_web.html")
    endif()  # BUILD_CPP_WRAPPER

# =============================================================================
# Android Examples (NativeActivity)
# =============================================================================
elseif(ANDROID)
    message(STATUS "Building Android examples...")
    
    # Find Android libraries
    find_library(ANDROID_LIBRARY android)
    find_library(LOG_LIBRARY log)
    
    # Get native_app_glue source from NDK
    set(NATIVE_APP_GLUE_DIR "${ANDROID_NDK}/sources/android/native_app_glue")
    if(NOT EXISTS "${NATIVE_APP_GLUE_DIR}")
        message(FATAL_ERROR "Could not find native_app_glue at ${NATIVE_APP_GLUE_DIR}")
    endif()
    
    # Build native_app_glue as a static library
    add_library(native_app_glue STATIC
        "${NATIVE_APP_GLUE_DIR}/android_native_app_glue.c"
    )
    target_include_directories(native_app_glue PUBLIC
        "${NATIVE_APP_GLUE_DIR}"
    )
    target_link_libraries(native_app_glue
        ${ANDROID_LIBRARY}
        ${LOG_LIBRARY}
    )
    
    # Note: Shaders are compiled in desktop builds and copied by Gradle APK projects.
    # Android native libraries don't bundle shaders - they're packaged in the APK.
    
    # C Cube Example (Android)
    add_library(cube_example_android_c SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/c/cube/CubeExample.c
    )
    target_include_directories(cube_example_android_c PRIVATE
        ${CMAKE_SOURCE_DIR}/gfx/include
    )
    target_link_libraries(cube_example_android_c
        gfx
        stb_image
        -Wl,--whole-archive
        native_app_glue
        -Wl,--no-whole-archive
        ${ANDROID_LIBRARY}
        ${LOG_LIBRARY}
    )
    set_target_properties(cube_example_android_c PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/platform/android/c"
    )
    
    # C Compute Example (Android)
    add_library(compute_example_android_c SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/c/compute/ComputeExample.c
    )
    target_include_directories(compute_example_android_c PRIVATE
        ${CMAKE_SOURCE_DIR}/gfx/include
    )
    target_link_libraries(compute_example_android_c
        gfx
        -Wl,--whole-archive
        native_app_glue
        -Wl,--no-whole-archive
        ${ANDROID_LIBRARY}
        ${LOG_LIBRARY}
    )
    set_target_properties(compute_example_android_c PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/platform/android/c"
    )
    
    # C++ Examples (require C++ wrapper)
    if(BUILD_CPP_WRAPPER)
        # C++ Cube Example (Android)
        add_library(cube_example_android_cpp SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cube/CubeExample.cpp
        )
        target_include_directories(cube_example_android_cpp PRIVATE
            ${CMAKE_SOURCE_DIR}/gfx_cpp/include
        )
        target_link_libraries(cube_example_android_cpp
            gfx_cpp
            gfx
            -Wl,--whole-archive
            native_app_glue
            -Wl,--no-whole-archive
            ${ANDROID_LIBRARY}
            ${LOG_LIBRARY}
        )
        set_target_properties(cube_example_android_cpp PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/platform/android/cpp"
        )
        
        # C++ Compute Example (Android)
        add_library(compute_example_android_cpp SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/compute/ComputeExample.cpp
        )
        target_include_directories(compute_example_android_cpp PRIVATE
            ${CMAKE_SOURCE_DIR}/gfx_cpp/include
        )
        target_link_libraries(compute_example_android_cpp
            gfx_cpp
            gfx
            -Wl,--whole-archive
            native_app_glue
            -Wl,--no-whole-archive
            ${ANDROID_LIBRARY}
            ${LOG_LIBRARY}
        )
        set_target_properties(compute_example_android_cpp PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/platform/android/cpp"
        )
    endif()  # BUILD_CPP_WRAPPER
    
    message(STATUS "Android examples configured") 

# =============================================================================
# Native Examples (GLFW-based, not for Web or Android)
# =============================================================================
else()
    # Native C Cube Example
    add_executable(cube_example_c ${CMAKE_CURRENT_SOURCE_DIR}/c/cube/CubeExample.c)
    target_link_libraries(cube_example_c gfx glfw stb_image)
    if(ENABLE_ASAN AND NOT MSVC)
        target_link_options(cube_example_c PRIVATE -fsanitize=address)
    endif()
    add_example_assets(cube_example_c
        SHADERS
            ${CMAKE_CURRENT_SOURCE_DIR}/c/cube/shaders/cube_textured.vert.wgsl
            ${CMAKE_CURRENT_SOURCE_DIR}/c/cube/shaders/cube_textured.frag.wgsl
        TEXTURES
            ${CMAKE_CURRENT_SOURCE_DIR}/c/cube/textures/vulkan.png
    )

    # Native C Threaded Cube Example (uses pthreads)
    add_executable(cube_example_c_threaded ${CMAKE_CURRENT_SOURCE_DIR}/c/cube/CubeExampleThreaded.c)
    target_link_libraries(cube_example_c_threaded gfx glfw stb_image)
    if(NOT MSVC)
        target_link_libraries(cube_example_c_threaded pthread)
        if(ENABLE_ASAN)
            target_link_options(cube_example_c_threaded PRIVATE -fsanitize=address)
        endif()
    endif()
    add_example_assets(cube_example_c_threaded
        SHADERS
            ${CMAKE_CURRENT_SOURCE_DIR}/c/cube/shaders/cube.vert.wgsl
            ${CMAKE_CURRENT_SOURCE_DIR}/c/cube/shaders/cube.frag.wgsl
    )

    # Threaded C++ Cube Example (uses C API with C++ language features)
    add_executable(cube_example_cpp_threaded ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cube/CubeExampleThreaded.cpp)
    target_link_libraries(cube_example_cpp_threaded gfx glfw stb_image)
    if(NOT MSVC)
        target_link_libraries(cube_example_cpp_threaded pthread)
        if(ENABLE_ASAN)
            target_link_options(cube_example_cpp_threaded PRIVATE -fsanitize=address)
        endif()
    endif()
    add_example_assets(cube_example_cpp_threaded
        SHADERS
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cube/shaders/cube.vert.wgsl
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cube/shaders/cube.frag.wgsl
    )

    # C++ Cube Example (uses C++ wrapper)
    if(BUILD_CPP_WRAPPER)
        add_executable(cube_example_cpp ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cube/CubeExample.cpp)
        target_link_libraries(cube_example_cpp gfx_cpp glfw stb_image)
        add_dependencies(cube_example_cpp gfx_cpp)
        if(ENABLE_ASAN AND NOT MSVC)
            target_link_options(cube_example_cpp PRIVATE -fsanitize=address)
        endif()
        add_example_assets(cube_example_cpp
            SHADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cube/shaders/cube.vert.wgsl
                ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cube/shaders/cube.frag.wgsl
        )
    endif()  # BUILD_CPP_WRAPPER

    # C Compute Example
    add_executable(compute_example_c ${CMAKE_CURRENT_SOURCE_DIR}/c/compute/ComputeExample.c)
    target_link_libraries(compute_example_c gfx glfw)
    if(ENABLE_ASAN AND NOT MSVC)
        target_link_options(compute_example_c PRIVATE -fsanitize=address)
    endif()
    add_example_assets(compute_example_c
        SHADERS
            ${CMAKE_CURRENT_SOURCE_DIR}/c/compute/shaders/generate.comp.wgsl
            ${CMAKE_CURRENT_SOURCE_DIR}/c/compute/shaders/fullscreen.vert.wgsl
            ${CMAKE_CURRENT_SOURCE_DIR}/c/compute/shaders/postprocess.frag.wgsl
    )

    # C++ Compute Example
    if(BUILD_CPP_WRAPPER)
        add_executable(compute_example_cpp ${CMAKE_CURRENT_SOURCE_DIR}/cpp/compute/ComputeExample.cpp)
        target_link_libraries(compute_example_cpp gfx_cpp glfw)
        add_dependencies(compute_example_cpp gfx_cpp)
        if(ENABLE_ASAN AND NOT MSVC)
            target_link_options(compute_example_cpp PRIVATE -fsanitize=address)
        endif()
        add_example_assets(compute_example_cpp
            SHADERS
                ${CMAKE_CURRENT_SOURCE_DIR}/cpp/compute/shaders/generate.comp.wgsl
                ${CMAKE_CURRENT_SOURCE_DIR}/cpp/compute/shaders/fullscreen.vert.wgsl
                ${CMAKE_CURRENT_SOURCE_DIR}/cpp/compute/shaders/postprocess.frag.wgsl
        )
    endif()  # BUILD_CPP_WRAPPER
endif()  # BUILD_FOR_WEB
