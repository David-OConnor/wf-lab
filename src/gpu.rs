//! GPU computation, via CUDA (not for graphics)
//!
//! nbody exmaple from nvidia: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
//! third party n-body example: https://github.com/harrism/mini-nbody
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DriverError, LaunchAsync, LaunchConfig};
use lin_alg2::f64::Vec3;

// #[repr(C)]
// struct Float3 {
//     x: f32,
//     y: f32,
//     z: f32,
// }

/// Convert a collection of `Vec3`s into Cuda arrays of their components.
fn alloc_vec3s(dev: &Arc<CudaDevice>, data: &[Vec3]) -> CudaSlice<f32> {
    let mut result = Vec::new();
    // todo: Ref etcs A/R; you are making a double copy here.
    for v in data {
        result.push(v.x as f32);
        result.push(v.y as f32);
        result.push(v.z as f32);
    }
    dev.htod_copy(result).unwrap()
}

/// Run coulomb attraction via the GPU. Calculates coulomb potentials between all combinations
/// of sample points and charge points, using a GPU kernel. Sums all charge points for a given
/// sample, and returns a potential-per-sample Vec.
pub fn run_coulomb_without_addition(
    dev: &Arc<CudaDevice>,
    posit_charges: &[Vec3],
    posit_samples: &[Vec3],
    charges: &[f64], // Corresponds 1:1 with `posit_charges`.
) -> Vec<f64> {
    // allocate buffers
    let n_charges = posit_charges.len();
    let n_samples = posit_samples.len();

    let posit_charges_ = alloc_vec3s(&dev, posit_charges);
    let posit_samples_ = alloc_vec3s(&dev, posit_samples);

    let charges: Vec<f32> = charges.iter().map(|c| *c as f32).collect();

    // let mut charges_gpu = dev.alloc_zeros::<f64>(N_CHARGES).unwrap();
    let mut charges_gpu = dev.alloc_zeros::<f32>(n_charges).unwrap();
    dev.htod_sync_copy_into(&charges, &mut charges_gpu).unwrap();

    let n = n_charges * n_samples;

    // todo: You will likely need 32-bit floats for performance purposes.
    // let mut coulomb_combos = dev.alloc_zeros::<f64>(N_CHARGES * N_SAMPLES).unwrap();
    let mut coulomb_combos = dev.alloc_zeros::<f32>(n).unwrap();

    let kernel = dev
        .get_func("cuda", "coulomb_kernel_without_addition")
        .unwrap();

    // The first parameter specifies the number of thread blocks. The second is the number of
    // threads in the thread block.
    // This must be a multiple of 32.
    // todo: Figure out how you want to divide up the block sizes, index, stride etc.
    // int blockSize = 256;
    // int numBlocks = (N + blockSize - 1) / blockSize;

    // VCoulomb<<<numBlocks, blockSize>>>(// ...);

    let cfg = LaunchConfig::for_num_elems(n as u32);

    // `for_num_elems`:
    // block_dim == 1024
    // grid_dim == (n + 1023) / 1024
    // shared_mem_bytes == 0

    const NUM_THREADS: u32 = 1024;
    // const NUM_THREADS: u32 = 1;
    let num_blocks = (n as u32 + NUM_THREADS - 1) / NUM_THREADS;

    // Self {
    //     grid_dim: (num_blocks, 1, 1),
    //     block_dim: (NUM_THREADS, 1, 1),
    //     shared_mem_bytes: 0,
    // }
    //     let cfg = LaunchConfig {
    //     grid_dim: (1, 1, 1),
    //     block_dim: (2, 2, 1),
    //     shared_mem_bytes: 0,
    // };

    // Custom launch config for 2-dimensional data (?)
    // let cfg = LaunchConfig {
    //     grid_dim: (num_blocks, 1, 1),
    //     block_dim: (NUM_THREADS, 1, 1),
    //     shared_mem_bytes: 0,
    // };

    //     dim3 threadsPerBlock(16, 16);
    //     dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    //     MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);

    unsafe {
        kernel.launch(
            cfg,
            (
                &mut coulomb_combos,
                &posit_charges_,
                &posit_samples_,
                &charges_gpu,
                n_charges,
                n_samples,
            ),
        )
    }
    .unwrap();

    let coulomb_combos_flat = dev.dtoh_sync_copy(&coulomb_combos).unwrap();

    println!("GPU coulomb data collected");

    // todo: Kernel for this A/R.
    let mut per_sample_flat = Vec::new();
    for i_sample in 0..n_samples {
        let mut charge_this_pt = 0.;
        for i_charge in 0..n_charges {
            charge_this_pt += coulomb_combos_flat[i_charge * n_samples + i_sample];
        }
        per_sample_flat.push(charge_this_pt as f64);
    }

    println!("Sum complete");
    per_sample_flat
}

/// Run coulomb attraction via the GPU. Computes per-sample potentials in paralle on the GPU; runs
/// the per-charge logic serial in the same kernel. This prevents needing to compute the sum on the CPU
/// afterwards. Returns a potential-per-sample Vec. Same API as the parallel+CPU approach above.
pub fn run_coulomb(
    dev: &Arc<CudaDevice>,
    posit_charges: &[Vec3],
    posit_samples: &[Vec3],
    charges: &[f64], // Corresponds 1:1 with `posit_charges`.
) -> Vec<f64> {
    // allocate buffers
    let n_charges = posit_charges.len();
    let n_samples = posit_samples.len();

    let posit_charges_ = alloc_vec3s(&dev, posit_charges);
    let posit_samples_ = alloc_vec3s(&dev, posit_samples);

    let charges: Vec<f32> = charges.iter().map(|c| *c as f32).collect();

    let mut charges_gpu = dev.alloc_zeros::<f32>(n_charges).unwrap();
    dev.htod_sync_copy_into(&charges, &mut charges_gpu).unwrap();

    let mut V_per_sample = dev.alloc_zeros::<f32>(n_samples).unwrap();

    let kernel = dev.get_func("cuda", "coulomb_kernel").unwrap();

    const NUM_THREADS: u32 = 1024;
    // const NUM_THREADS: u32 = 1;
    let num_blocks = (n_samples as u32 + NUM_THREADS - 1) / NUM_THREADS;

    let cfg = LaunchConfig::for_num_elems(n_samples as u32);

    // Custom launch config for 2-dimensional data (?)
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        kernel.launch(
            cfg,
            (
                &mut V_per_sample,
                &posit_charges_,
                &posit_samples_,
                &charges_gpu,
                n_charges,
                n_samples,
            ),
        )
    }
    .unwrap();

    let result = dev.dtoh_sync_copy(&V_per_sample).unwrap();

    println!("GPU coulomb data collected");

    result.iter().map(|v| *v as f64).collect()
}