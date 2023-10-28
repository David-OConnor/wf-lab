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
    // cc::Build::new()
    //     .cuda(true)
    //     // .cudart("static")
    //     // todo: Is expicitly specifying gencode required?
    //     // Generate code for RTX 2 series.
    //     .flag("-gencode").flag("arch=compute_75,code=sm_75")
    //     // Generate code in parallel
    //     .flag("-t0")
    //     .file("./cuda/cuda.cu")
    //     .compile("cuda_test");

    // todo: generate lib (so/dll?) but not bin (.exe)
    let compilation_result = Command::new("nvcc")
        // `nvcc .\main.cu -o main.exe`
        .args(["./cuda/cuda.cu", "-gencode", "arch=compute_75,code=sm_75", "-o", "cuda.exe" ])
        .output().expect("Problem compiling the C++/CUDA code.");

    if !compilation_result.status.success() {
        panic!("Compilation problem: {:?}", compilation_result);
    }
}