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
fn allocate_vec3s(dev: &Arc<CudaDevice>, data: &[Vec3]) -> CudaSlice<f32> {
    let mut result = Vec::new();
    // todo: Ref etcs A/R; you are making a double copy here.
    for v in data {
        result.push(v.x as f32);
        result.push(v.y as f32);
        result.push(v.z as f32);
    }
    dev.htod_copy(result).unwrap()
}

/// Run coulomb attraction via the GPU.
pub fn run_coulomb(
    dev: &Arc<CudaDevice>,
    posit_charges: &[Vec3],
    posit_samples: &[Vec3],
    charges: &[f64], // Corresponds 1:1 with `posit_charges`.
) -> Vec<f64> {
    // You can load a function from a pre-compiled PTX like so:
    // dev.load_ptx(Ptx::from_file("./src/cuda.ptx"), "sin", &["sin_kernel"])?;

    // let a: CudaSlice<f64> = dev.alloc_zeros::<f64>(10)?;
    // let mut b = dev.alloc_zeros::<f64>(10)?;
    //
    // // you can do device to device copies of course
    // dev.dtod_copy(&a, &mut b)?;

    // allocate buffers
    let n_charges = posit_charges.len();
    let n_samples = posit_samples.len();

    let posit_charges_ = allocate_vec3s(&dev, posit_charges);
    let posit_samples_ = allocate_vec3s(&dev, posit_samples);

    let charges: Vec<f32> = charges.iter().map(|c| *c as f32).collect();

    // let mut charges_gpu = dev.alloc_zeros::<f64>(N_CHARGES).unwrap();
    let mut charges_gpu = dev.alloc_zeros::<f32>(n_charges).unwrap();
    // dev.htod_sync_copy_into(charges, &mut charges_gpu).unwrap();
    dev.htod_sync_copy_into(&charges, &mut charges_gpu).unwrap();

    // let a_dev = dev.htod_copy(a_host.into()).unwrap();
    // let mut b_dev = a_dev.clone();

    let n = n_charges * n_samples;

    // todo: You will likely need 32-bit floats for performance purposes.
    // let mut coulomb_combos = dev.alloc_zeros::<f64>(N_CHARGES * N_SAMPLES).unwrap();
    let mut coulomb_combos = dev.alloc_zeros::<f32>(n).unwrap();

    let kernel = dev.get_func("cuda", "coulomb_kernel").unwrap();

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
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

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

    // Copy back to the host:
    let coulomb_combos_flat = dev.dtoh_sync_copy(&coulomb_combos).unwrap();

    // println!("coulomb combos_flat: {:?}", coulomb_combos_flat);
    // Expected result. // (charge, sample)
    // (0, 0): 1.
    // (0, 1): 0.5
    // (0, 2): 0.333
    // (1, 0): 2.
    // (1, 1):2.
    // (1, 2): 0.666

    // 1., 0.5, 0.333, 2., 2., 0.666?  # This is the result we're getting.
    // 1., 2., 0.5, 2., 0.333, 0.666?

    println!("GPU coulomb data collected");

    // should be indexes 0+3, 1+4, 2+5
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

    // print!("Per sample flat: {:?}", per_sample_flat);
    // We are not getting this; we're getting [1.5, 2.3333333333333335, 2.6666666666666665]
    // Expected:
    // S0: 3.
    // S1: 2.5
    // S2: 1.

    per_sample_flat

    // Now, convert

    // todo: For now, handle the rest on CPU. Come back to GPU once this is working.

    // todo: Come back to this.
    // {
    //     let mut charges_at_each_sample_posit = dev.alloc_zeros::<f64>(N_SAMPLES).unwrap();
    //
    //     // Now, add, for each sample point, all appropriate charges
    //     // todo: Be careful about indexes and/or grids!
    //     let kernel = dev.get_func("cuda", "sum_coulomb_results_kernel").unwrap();
    //
    //     let cfg = LaunchConfig::for_num_elems((N_CHARGES * N_SAMPLES) as u32);
    //     unsafe {
    //         kernel.launch(
    //             cfg,
    //             (
    //                 &mut charges_at_each_sample_posit,
    //                 &coulomb_combos,
    //                 N_CHARGES,
    //                 N_SAMPLES,
    //             ),
    //         )
    //     }
    //         .unwrap();
    //
    //     // let a_host_2 = dev.sync_reclaim(a_dev)?;
    //     // let out_host = dev.sync_reclaim(b_dev)?;
    //

    // }

    // todo: Don't do this copy to CPU then back!

    // println!("OUT: {:?}", coulomb_combos_flat);

    // Next step: Either here or in `potential.rs`:
    // You have flat output of the 2d correspondances between each posit sample and posit charge.
    // Expand into an array of arrays. (2D) etc.

    // (Should we do this on CPU, or GPU? Likely GPU.)

    // Loop over, and sum the contributions  from each charge, for the corresponding sample point.

    // Map the result back  back from a flat list to a 3D list per sample point.

    // coulomb_combos_flat

    // // unsafe initialization of unset memory
    // let _: CudaSlice<f32> = unsafe { dev.alloc::<f32>(10) }.unwrap();

    // // this will have memory initialized as 0
    // let _: CudaSlice<f64> = dev.alloc_zeros::<f64>(10).unwrap();

    // initialize with a rust vec
    // let _: CudaSlice<usize> = dev.htod_copy(vec![0; 10]).unwrap();

    // // or finially, initialize with a slice. this is synchronous though.
    // let _: CudaSlice<u32> = dev.htod_sync_copy(&[1, 2, 3]).unwrap();
}
