//! We use this to automatically compile CUDA C++ code when building.
//!
//! [Tutorial](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
//!
//! You must have this or equivalent in the PATH environment variable:
//! `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64`

use std::process::{Command, ExitStatus};
use cc;

/// See [These CUDA docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
/// for info about these flags.
fn main() {
    return
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed=cuda/cuda.cu");
    // println!("cargo:rustc-link-lib=cuda"); // todo: Is this required?
    // println!("cargo:rustc-link-lib=cuda"); // todo: Is this required?

    cc::Build::new()
        .cpp(true)
        // .cuda(true)
        // .cudart("shared") // todo: troubleshooting.
        // Generate code for RTX 2 series.
        // .flag("-gencode").flag("arch=compute_75,code=sm_75")
        // Generate code in parallel // todo: Do we want this?
        // .flag("-t0")
        //         .include("...")
        //         .flag("-Llibrary_path")
        //         .flag("-llibrary")
        //         .compile("...");
        .file("cuda/cuda.cu")
        .compile("cuda");

    return;

    // `nvcc --shared -o libtest.so test.cu --compiler-options '-fPIC' `



    // todo: generate lib (so/dll?) but not bin (.exe)
    // note: with clang++, either `-shared` or `-c` seems to be required; `-shared` may be better.
    // todo: `-c` working; `-shared` not.
    // Call: `clang++ cuda/cuda.cpp -c -o cuda.lib`
    let compilation_result = Command::new("nvcc")
        // `nvcc .\main.cu -o main.exe`
        .args(["cuda/cuda.cu", "--lib", "-gencode", "arch=compute_75,code=sm_75", "-o", "cuda.lib" ])
        .output().expect("Problem compiling the C++/CUDA code.");

    // let compilation_result = Command::new("clang++")
    //     // `nvcc .\main.cu -o main.exe`
    //     .args(["cuda/cuda.cpp", "-c", "-o", "cuda.lib" ])
    //     .output().expect("Problem compiling the C++/CUDA code.");

    if !compilation_result.status.success() {
        panic!("Compilation problem: {:?}", compilation_result);
    }
}
