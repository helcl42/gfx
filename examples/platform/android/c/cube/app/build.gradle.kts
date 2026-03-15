plugins {
    id("com.android.application")
}

android {
    namespace = "com.example.gfx.cube"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.gfx.cube"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            // Specify the ABIs we want to build for
            // Match this with your CMake build
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    // Point to prebuilt native libraries
    sourceSets {
        getByName("main") {
            jniLibs.srcDir("src/main/jniLibs")
        }
    }
}

dependencies {
    // No dependencies needed for native-only app
}

// Task to copy built .so files from CMake build to jniLibs
tasks.register<Copy>("copyNativeLibs") {
    description = "Copies native libraries from CMake build output to jniLibs directory"
    
    // Get the ABI from NDK config
    val abi = android.defaultConfig.ndk.abiFilters.firstOrNull() ?: "arm64-v8a"
    val abiToDir = mapOf(
        "arm64-v8a" to "arm64-v8a",
        "armeabi-v7a" to "armeabi-v7a",
        "x86_64" to "x86_64",
        "x86" to "x86"
    )
    val targetDir = file("src/main/jniLibs/${abiToDir[abi] ?: "arm64-v8a"}").absoluteFile
    
    // Try multiple possible build directory locations
    val possibleBuildDirs = listOf(
        file("${project.rootDir}/../../../../../build_android_${abi}").absoluteFile,
        file("${project.rootDir}/../../../../../build_android_arm64").absoluteFile,
        file("${project.rootDir}/../../../../../build_android").absoluteFile
    )
    
    val buildDir = possibleBuildDirs.firstOrNull { it.exists() }
        ?: possibleBuildDirs[0]  // Use first option for error message if none exist
    
    from(buildDir) {
        include("libgfx.so")
        include("libgfx.so.*")
    }
    from(file("$buildDir/examples/platform/android/c")) {
        include("libcube_example_android_c.so")
    }
    
    into(targetDir)
    
    doFirst {
        println("Copying native libraries...")
        println("  Target ABI: $abi")
        println("  Build directory: $buildDir")
        println("  Target directory: $targetDir")
        
        if (!buildDir.exists()) {
            val buildScript = file("${project.rootDir}/../../../../scripts/build_android.sh").absolutePath
            throw GradleException("""
                ❌ CMake build output not found at: $buildDir
                
                Please build native libraries first using the build script:
                  cd ${file("${project.rootDir}/../../../../").absolutePath}
                  bash scripts/build_android.sh $abi
                  
                Or manually with CMake:
                  cd ${file("${project.rootDir}/../../../../").absolutePath}
                  cmake -B build_android_$abi \
                    -G Ninja \
                    -DCMAKE_TOOLCHAIN_FILE=$${'$'}ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
                    -DANDROID_ABI=$abi \
                    -DANDROID_PLATFORM=android-26 \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DBUILD_VULKAN_BACKEND=ON \
                    -DBUILD_EXAMPLES=ON
                  cmake --build build_android_$abi
                
                Then retry: ./gradlew assembleDebug
            """.trimIndent())
        }
        
        // Check if the example library exists
        val exampleLib = file("$buildDir/examples/platform/android/c/libcube_example_android_c.so")
        if (!exampleLib.exists()) {
            throw GradleException("""
                ❌ Example library not found: ${exampleLib.absolutePath}
                
                The CMake build completed, but the example library is missing.
                Make sure you have:
                  - BUILD_EXAMPLES=ON in CMake configuration
                  - The build completed successfully
                
                Try rebuilding:
                  bash ${file("${project.rootDir}/../../../../scripts/build_android.sh").absolutePath} $abi Release
            """.trimIndent())
        }
    }
    
    doLast {
        val copiedFiles = targetDir.listFiles()?.toList() ?: emptyList()
        if (copiedFiles.isEmpty()) {
            throw GradleException("No files were copied to jniLibs. Check that the build directory contains .so files.")
        }
        println("✓ Copied ${copiedFiles.size} files to jniLibs:")
        copiedFiles.forEach { println("  - ${it.name} (${it.length() / 1024}KB)") }
    }
}

// Task to copy shaders from build to assets
tasks.register<Copy>("copyShaders") {
    description = "Copies shaders from build output to assets directory"
    
    val workspaceRoot = file("${project.rootDir}/../../../../..").absoluteFile
    val desktopBuildDir = file("$workspaceRoot/build").absoluteFile
    
    from("$desktopBuildDir/examples") {
        include("c_cube_shaders_cube_textured.vert.spv")
        include("c_cube_shaders_cube_textured.frag.spv")
        rename { it.replace("c_cube_shaders_", "") }
    }
    into("src/main/assets/shaders")
    
    doFirst {
        if (!desktopBuildDir.exists()) {
            throw GradleException("""
                Desktop build output not found at: $desktopBuildDir
                
                Please build shaders first:
                  cd $workspaceRoot
                  mkdir -p build && cd build
                  cmake .. && ninja
            """.trimIndent())
        }
    }
}

// Task to copy textures from source to assets
tasks.register<Copy>("copyTextures") {
    description = "Copies textures from source to assets directory"
    
    val workspaceRoot = file("${project.rootDir}/../../../../..").absoluteFile
    val examplesDir = file("$workspaceRoot/examples").absoluteFile
    
    from("$examplesDir/c/cube/textures") {
        include("vulkan.png")
    }
    into("src/main/assets/textures")
}

// Combined task for all assets
tasks.register("copyAssets") {
    description = "Copies all shaders and textures to assets directory"
    dependsOn("copyShaders", "copyTextures")
    
    doLast {
        val shaderDir = file("src/main/assets/shaders")
        val textureDir = file("src/main/assets/textures")
        val shaderFiles = shaderDir.listFiles()?.toList() ?: emptyList()
        val textureFiles = textureDir.listFiles()?.toList() ?: emptyList()
        println("Copied ${shaderFiles.size} shaders and ${textureFiles.size} textures:")
        shaderFiles.forEach { println("  - shaders/${it.name}") }
        textureFiles.forEach { println("  - textures/${it.name}") }
    }
}

// Make sure native libs and assets are copied before packaging
tasks.named("preBuild") {
    dependsOn("copyNativeLibs", "copyAssets")
}
