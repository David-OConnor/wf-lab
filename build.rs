//! We use this to automatically compile CUDA C++ code when building.
//!
//! [Tutorial](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
//!
//! You must have this or equivalent in the PATH environment variable:
//! `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64`
//!

use std::{
    env,
    path::PathBuf,
    process::{Command, ExitStatus},
};
// use cc;

#[derive(Copy, Clone)]
pub enum GpuArchitecture {
    Rtx2,
    Rtx3,
    Rtx4,
}

impl GpuArchitecture {
    /// [Selecting architecture, by nVidia series](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
    pub fn gencode_val(&self) -> String {
        let version: u8 = match self {
            Self::Rtx2 => 75,
            Self::Rtx3 => 86,
            Self::Rtx4 => 89,
        };

        String::from(format!("arch=compute_{version},code=sm_{version}"))
    }
}

/// See [These CUDA docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
/// for info about these flags.
///
/// Compiles our CUDA program using Nvidia's NVCC compiler
/// [NVCC docs and list of commands](https://docs.nvidia.com/cuda/pdf/CUDA_Compiler_Driver_NVCC.pdf)
/// A lib using CC, in linux: https://github.com/termoshtt/link_cuda_kernel
///
/// Important: Must have LLVM>=16 installed.
// /// Must set the environment vars `CXX` to `nvcc` (clang++?), and `LIBCLANG_PATH` to`C:\Program Files\LLVM\bin`
/// Must set the environment vars `CXX` to `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64`
/// or similar, and `LIBCLANG_PATH` to`C:\Program Files\LLVM\bin`
fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed=src/cuda.cu");

    let architecture = GpuArchitecture::Rtx4;

    let compilation_result = Command::new("nvcc")
        .args([
            "src/cuda.cu",
            "-gencode",
            &architecture.gencode_val(),
            "-ptx",
        ])
        .output()
        .expect("Problem compiling the CUDA module.");

    if !compilation_result.status.success() {
        panic!("Compilation problem: {:?}", compilation_result);
    }
    return;

    // cc::Build::new()
    //     .cuda(true)
    //     .cudart("shared")  // Defaults to `static`.
    //     // Generate code for the GPU architecture
    //     .flag("-gencode").flag(&architecture.gencode_val())
    //     // Generate code in parallel // todo: Do we want this?
    //     .flag("-t0")
    //     //         .include("...")
    //     //         .flag("-Llibrary_path")
    //     //         .flag("-llibrary")
    //     //         .compile("...");
    //     // .cpp_link_stdlib("stdc++")
    //     .file("src/cuda.cu")
    //     .compile("cuda");
    //     // .compile("libcuda.a"); // Linux

    // println!("cargo:rustc-link-lib=msvc");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64'");

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");

    // println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    // println!("cargo:rustc-link-lib=cudart");
    // println!("cargo:rustc-link-lib=lcudart");
    // println!("cargo:rustc-link-lib=lcuda");

    // println!("cargo:rustc-link-lib=cuda");

    return;

    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!("cargo:rustc-link-search={}", out.display());

    // todo: `-ptx` nvcc flag?

    // `nvcc src/cuda.cu -gencode "arch=compute_75,code=sm_75" -t0 -c -o cuda.lib`
}
