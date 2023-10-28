//! We use this to automatically compile CUDA C++ code when building.
//!
//! [Tutorial](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
//!
//! You must have this or equivalent in the PATH environment variable:
//! `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64`

use std::process::Command;
use cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .cudart("static")
        // todo: Is expicitly specifying gencode required?
        // Generate code for RTX 2 series.
        // .flag("-gencode").flag("arch=compute_75,code=sm_75")
        // Generate code in parallel
        .flag("-t0")
        .file("./cuda/cuda.cu")
        .compile("cuda_test");

    // Command::new("powershell")
    //     .args(["nvcc", "./cuda.cu", "-o", "cuda.exe"])
    //     .output()
    //     .expect("Failed to compile the CUDA program.");
}