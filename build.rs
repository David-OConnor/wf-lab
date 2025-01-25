//! We use this to automatically compile CUDA C++ code when building.
//!
//! [Tutorial](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
//!
//! You must have this or equivalent in the PATH environment variable:
//! `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64`
//!

use std::process::Command;

#[derive(Copy, Clone)]
pub enum GpuArchitecture {
    Rtx2,
    Rtx3,
    Rtx4,
    Rtx5,
}

impl GpuArchitecture {
    /// [Selecting architecture, by nVidia series](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
    pub fn gencode_val(&self) -> String {
        let version: u8 = match self {
            Self::Rtx2 => 75,
            Self::Rtx3 => 86,
            Self::Rtx4 => 89,
            Self::Rtx5 => 100,
        };

        String::from(format!("arch=compute_{version},code=sm_{version}"))
    }
}

/// See [These CUDA docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
/// for info about these flags.
///
/// Compiles our CUDA program using Nvidia's NVCC compiler
fn main() {
    #[cfg(not(feature = "cuda"))]
    return

    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed=src/cuda/cuda.cu");
    println!("cargo:rerun-if-changed=src/cuda/util.cu");

    let architecture = GpuArchitecture::Rtx4;

    let compilation_result = Command::new("nvcc")
        .args([
            "src/cuda/cuda.cu",
            "-gencode",
            &architecture.gencode_val(),
            "-ptx",
            "-O3", // optimized/release mode.
        ])
        .output()
        .expect("Problem compiling the CUDA module.");

    if !compilation_result.status.success() {
        panic!("Compilation problem: {:?}", compilation_result);
    }
}
