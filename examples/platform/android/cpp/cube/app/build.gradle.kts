plugins {
    id("com.android.application")
}

android {
    namespace = "com.example.gfx.cube.cpp"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.gfx.cube.cpp"
        minSdk = 26
        targetSdk = 35
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

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
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
        ?: possibleBuildDirs[0]
    
    from(buildDir) {
        include("libgfx.so")
        include("libgfx_cpp.so")
        include("libgfx.so.*")
        include("libgfx_cpp.so.*")
    }
    from(file("$buildDir/examples/platform/android/cpp")) {
        include("libcube_example_android_cpp.so")
    }
    
    into(targetDir)
    
    doFirst {
        println("Copying native libraries...")
        println("  Target ABI: $abi")
        println("  Build directory: $buildDir")
        println("  Target directory: $targetDir")
        
        if (!buildDir.exists()) {
            throw GradleException("""
                ❌ CMake build output not found at: $buildDir
                
                Please build native libraries first using the build script:
                  cd ${file("${project.rootDir}/../../../../").absolutePath}
                  bash scripts/build_android.sh $abi
                  
                Then retry: ./gradlew assembleDebug
            """.trimIndent())
        }
        
        val exampleLib = file("$buildDir/examples/platform/android/cpp/libcube_example_android_cpp.so")
        if (!exampleLib.exists()) {
            throw GradleException("""
                ❌ Example library not found: ${exampleLib.absolutePath}
                
                The CMake build completed, but the example library is missing.
                Make sure you have BUILD_EXAMPLES=ON and BUILD_CPP_WRAPPER=ON in CMake configuration.
                
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
        include("cpp_cube_shaders_cube.vert.spv")
        include("cpp_cube_shaders_cube.frag.spv")
        rename { it.replace("cpp_cube_shaders_", "") }
    }
    into("src/main/assets/shaders")
    
    doFirst {
        println("Copying shaders...")
        if (!desktopBuildDir.exists()) {
            throw GradleException("""
                ❌ Desktop build output not found at: $desktopBuildDir
                
                Please build shaders first:
                  cd $workspaceRoot
                  mkdir -p build && cd build
                  cmake .. && ninja
                  
                Then retry: ./gradlew assembleDebug
            """.trimIndent())
        }
    }
    
    doLast {
        val shaderDir = file("src/main/assets/shaders")
        val shaderFiles = shaderDir.listFiles()?.toList() ?: emptyList()
        println("✓ Copied ${shaderFiles.size} shaders:")
        shaderFiles.forEach { println("  - ${it.name}") }
    }
}

// Combined task for all assets
tasks.register("copyAssets") {
    description = "Copies all shaders to assets directory"
    dependsOn("copyShaders")
}

// Make sure native libs and assets are copied before packaging
tasks.named("preBuild") {
    dependsOn("copyNativeLibs", "copyAssets")
}
