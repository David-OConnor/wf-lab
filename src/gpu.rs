//! GPU computation, via CUDA (not for graphics)
//!
//! Note: We are currently using f64s on the GPU. This induces a roughly 10-20x performance
//! hit compared to f32. We'll switch to f32 as required, but this is still much faster
//! than performing coulomb operations on the CPU.
//!
use std::{sync::Arc, time::Instant};

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use lin_alg2::f64::Vec3;

use crate::basis_wfs::Sto;

// type FDev = f64; // This makes switching between f32 and f64 easier.
type FDev = f32; // This makes switching between f32 and f64 easier.

/// Convert a collection of `Vec3`s into Cuda arrays of their components.
fn alloc_vec3s(dev: &Arc<CudaDevice>, data: &[Vec3]) -> CudaSlice<FDev> {
    let mut result = Vec::new();
    // todo: Ref etcs A/R; you are making a double copy here.
    for v in data {
        result.push(v.x as FDev);
        result.push(v.y as FDev);
        result.push(v.z as FDev);
    }
    dev.htod_copy(result).unwrap()
}

/// Run coulomb attraction via the GPU. Computes per-sample potentials in paralle on the GPU; runs
/// the per-charge logic serial in the same kernel. This prevents needing to compute the sum on the CPU
/// afterwards. Returns a potential-per-sample Vec. Same API as the parallel+CPU approach above.
pub fn run_coulomb(
    dev: &Arc<CudaDevice>,
    posits_charge: &[Vec3],
    posits_sample: &[Vec3],
    charges: &[f64], // Corresponds 1:1 with `posit_charges`.
) -> Vec<f64> {
    let start = Instant::now();

    // allocate buffers
    let n_charges = posits_charge.len();
    let n_samples = posits_sample.len();

    let posit_charges_gpus = alloc_vec3s(&dev, posits_charge);
    let posits_sample_gpu = alloc_vec3s(&dev, posits_sample);

    // Note: This step is not required when using f64ss.
    let charges: Vec<FDev> = charges.iter().map(|c| *c as FDev).collect();

    let mut charges_gpu = dev.alloc_zeros::<FDev>(n_charges).unwrap();
    dev.htod_sync_copy_into(&charges, &mut charges_gpu).unwrap();

    let mut V_per_sample = dev.alloc_zeros::<FDev>(n_samples).unwrap();

    let kernel = dev.get_func("cuda", "coulomb_kernel").unwrap();

    let cfg = LaunchConfig::for_num_elems(n_samples as u32);

    let cfg = {
        const NUM_THREADS: u32 = 1024;
        let num_blocks = (n_samples as u32 + NUM_THREADS - 1) / NUM_THREADS;

        // Custom launch config for 2-dimensional data (?)
        LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    };

    unsafe {
        kernel.launch(
            cfg,
            (
                &mut V_per_sample,
                &posit_charges_gpus,
                &posits_sample_gpu,
                &charges_gpu,
                n_charges,
                n_samples,
            ),
        )
    }
    .unwrap();

    let result = dev.dtoh_sync_copy(&V_per_sample).unwrap();

    // Some profiling numbers for certain grid sizes.
    // 2D, f32: 99.144 ms
    // 3D, f32: 400.06 ms
    // 2D, f64: 1_658 ms
    // 3D, f64: 1_643 ms
    // 300 ms for both large and small sizes on f32 with std::sqrt???

    let time_diff = Instant::now() - start;
    println!("GPU coulomb data collected. Time: {:?}", time_diff);

    // This step is not required when using f64.
    result.iter().map(|v| *v as f64).collect()
    // result
}

/// Compute STO value and second derivative at a collection of sample points.
/// Assumes N=1 and real values for now.
pub(crate) fn sto_vals_derivs_multiple_bases(
    dev: &Arc<CudaDevice>,
    bases: &[Sto],
    posits_sample: &[Vec3],
) -> (Vec<f64>, Vec<f64>) {
    let n_samples = posits_sample.len();
    let n_bases = bases.len();

    let mut posits_nuc = Vec::new();
    let mut xis = Vec::new();
    let mut ns = Vec::new();
    let mut weights = Vec::new();
    for basis in bases {
        posits_nuc.push(basis.posit);
        xis.push(basis.xi as FDev);
        ns.push(basis.n);
        weights.push(basis.weight as FDev);
    }

    let posits_sample_gpu = alloc_vec3s(&dev, posits_sample);

    let mut psi = dev.alloc_zeros::<FDev>(n_samples).unwrap();
    let mut psi_pp = dev.alloc_zeros::<FDev>(n_samples).unwrap();

    let posits_nuc_gpu = alloc_vec3s(&dev, &posits_nuc);
    let mut xis_gpu = dev.alloc_zeros::<FDev>(n_bases).unwrap();
    let mut ns_gpu = dev.alloc_zeros::<u16>(n_bases).unwrap();
    let mut weights_gpu = dev.alloc_zeros::<FDev>(n_bases).unwrap();

    dev.htod_sync_copy_into(&xis, &mut xis_gpu).unwrap();
    dev.htod_sync_copy_into(&ns, &mut ns_gpu).unwrap();
    dev.htod_sync_copy_into(&weights, &mut weights_gpu).unwrap();

    let kernel = dev
        .get_func("cuda", "sto_val_deriv_multiple_bases_kernel")
        .unwrap();
    let cfg = LaunchConfig::for_num_elems(n_samples as u32);

    unsafe {
        kernel.launch(
            cfg,
            (
                &mut psi,
                &mut psi_pp,
                &posits_sample_gpu,
                &posits_nuc_gpu,
                &xis_gpu,
                &ns_gpu,
                &weights_gpu,
                n_samples,
                n_bases,
            ),
        )
    }
    .unwrap();

    let result_psi = dev.dtoh_sync_copy(&psi).unwrap();
    let result_psi_pp = dev.dtoh_sync_copy(&psi_pp).unwrap();

    // This step is not required when using f64.
    (
        result_psi.iter().map(|v| *v as f64).collect(),
        result_psi_pp.iter().map(|v| *v as f64).collect(),
    )
}

// todo: DRY with above
pub(crate) fn sto_vals_multiple_bases(
    dev: &Arc<CudaDevice>,
    bases: &[Sto],
    posits_sample: &[Vec3],
) -> Vec<f64> {
    let n_samples = posits_sample.len();
    let n_bases = bases.len();

    let mut posits_nuc = Vec::new();
    let mut xis = Vec::new();
    let mut ns = Vec::new();
    let mut weights = Vec::new();

    for basis in bases {
        posits_nuc.push(basis.posit);
        xis.push(basis.xi as FDev);
        ns.push(basis.n);
        weights.push(basis.weight as FDev);
    }

    let posits_sample_gpu = alloc_vec3s(&dev, posits_sample);

    let mut psi = dev.alloc_zeros::<FDev>(n_samples).unwrap();

    let posits_nuc_gpu = alloc_vec3s(&dev, &posits_nuc);
    let mut xis_gpu = dev.alloc_zeros::<FDev>(n_bases).unwrap();
    let mut ns_gpu = dev.alloc_zeros::<u16>(n_bases).unwrap();
    let mut weights_gpu = dev.alloc_zeros::<FDev>(n_bases).unwrap();

    dev.htod_sync_copy_into(&xis, &mut xis_gpu).unwrap();
    dev.htod_sync_copy_into(&ns, &mut ns_gpu).unwrap();
    dev.htod_sync_copy_into(&weights, &mut weights_gpu).unwrap();

    let kernel = dev
        .get_func("cuda", "sto_val_multiple_bases_kernel")
        .unwrap();
    let cfg = LaunchConfig::for_num_elems(n_samples as u32);

    unsafe {
        kernel.launch(
            cfg,
            (
                &mut psi,
                &posits_sample_gpu,
                &posits_nuc_gpu,
                &xis_gpu,
                &ns_gpu,
                &weights_gpu,
                n_samples,
                n_bases,
            ),
        )
    }
    .unwrap();

    let result_psi = dev.dtoh_sync_copy(&psi).unwrap();

    // This step is not required when using f64.
    result_psi.iter().map(|v| *v as f64).collect()
}

/// Compute STO value and second derivative at a collection of sample points.
/// Assumes N=1 and real values for now.
pub(crate) fn sto_vals_or_derivs(
    dev: &Arc<CudaDevice>,
    xi: f64,
    n: u16,
    posits_sample: &[Vec3],
    posit_nuc: Vec3,
    deriv: bool,
) -> Vec<f64> {
    // todo: DRY
    let n_samples = posits_sample.len();

    // todo: We may need to use f64 for numerical second derivatives.

    let posits_sample_gpu = alloc_vec3s(&dev, posits_sample);

    let mut psi = dev.alloc_zeros::<FDev>(n_samples).unwrap();

    let posit_nuc_gpu = dev
        .htod_sync_copy(&[
            posit_nuc.x as FDev,
            posit_nuc.y as FDev,
            posit_nuc.z as FDev,
        ])
        .unwrap();

    let kernel = dev.get_func("cuda", "sto_val_or_deriv_kernel").unwrap();

    // let cfg = LaunchConfig::for_num_elems(n_samples as u32);
    let cfg = {
        const NUM_THREADS: u32 = 768; // 1_024 is triggering `OUT_OF_RESOURCES`.
        let num_blocks = (n_samples as u32 + NUM_THREADS - 1) / NUM_THREADS;

        // Custom launch config for 2-dimensional data (?)
        LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    };

    unsafe {
        kernel.launch(
            cfg,
            (
                &mut psi,
                &posits_sample_gpu,
                &posit_nuc_gpu,
                xi as FDev,
                n,
                deriv,
                n_samples,
            ),
        )
    }
    .unwrap();

    let result_psi = dev.dtoh_sync_copy(&psi).unwrap();

    // This step is not required when using f64.
    result_psi.iter().map(|v| *v as f64).collect()
}

/// Compute STO value and second derivative at a collection of sample points.
/// Assumes N=1 and real values for now.
pub(crate) fn sto_vals_derivs(
    dev: &Arc<CudaDevice>,
    xi: f64,
    n: u16,
    posits_sample: &[Vec3],
    posit_nuc: Vec3,
) -> (Vec<f64>, Vec<f64>) {
    let n_samples = posits_sample.len();

    let posits_sample_gpu = alloc_vec3s(&dev, posits_sample);

    let mut psi = dev.alloc_zeros::<FDev>(n_samples).unwrap();
    let mut psi_pp = dev.alloc_zeros::<FDev>(n_samples).unwrap();

    let posit_nuc_gpu = dev
        .htod_sync_copy(&[
            posit_nuc.x as FDev,
            posit_nuc.y as FDev,
            posit_nuc.z as FDev,
        ])
        .unwrap();

    let kernel = dev.get_func("cuda", "sto_val_deriv_kernel").unwrap();

    let cfg = {
        const NUM_THREADS: u32 = 768; // 1_024 is triggering `OUT_OF_RESOURCES`.
        let num_blocks = (n_samples as u32 + NUM_THREADS - 1) / NUM_THREADS;

        // Custom launch config for 2-dimensional data (?)
        LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    };

    unsafe {
        kernel.launch(
            cfg,
            (
                &mut psi,
                &mut psi_pp,
                &posits_sample_gpu,
                &posit_nuc_gpu,
                xi as FDev,
                n,
                n_samples,
            ),
        )
    }
    .unwrap();

    let result_psi = dev.dtoh_sync_copy(&psi).unwrap();
    let result_psi_pp = dev.dtoh_sync_copy(&psi_pp).unwrap();

    // This step is not required when using f64.
    (
        result_psi.iter().map(|v| *v as f64).collect(),
        result_psi_pp.iter().map(|v| *v as f64).collect(),
    )
}
