val projectRoot = java.io.File(".").absoluteFile
println("Project root: $projectRoot")

val possibleBuildDirs = listOf(
    java.io.File("${projectRoot.parent}/../../../build_android_arm64-v8a").absoluteFile,
    java.io.File("${projectRoot.parent}/../../../../build_android_arm64-v8a").absoluteFile,
    java.io.File("${projectRoot.parent}/../../../build_android").absoluteFile,
    java.io.File("${projectRoot.parent}/../../../../build_android").absoluteFile
)

println("\nTrying paths:")
possibleBuildDirs.forEachIndexed { i, dir ->
    println("${i+1}. ${dir.absolutePath} - exists: ${dir.exists()}")
}

val buildDir = possibleBuildDirs.firstOrNull { it.exists() }
println("\nSelected: ${buildDir?.absolutePath ?: "NONE"}")

if (buildDir != null) {
    val gfxLib = java.io.File(buildDir, "libgfx.so")
    val exampleLib = java.io.File(buildDir, "examples/platform/android/cpp/libcube_example_android_cpp.so")
    println("\nChecking files:")
    println("  libgfx.so: ${gfxLib.exists()}")
    println("  example lib: ${exampleLib.exists()}")
}
