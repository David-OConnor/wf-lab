//! GPU computation, via CUDA (not for graphics)

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DriverError, LaunchAsync, LaunchConfig};
use lin_alg2::f64::Vec3;

/// Convert a collection of `Vec3`s into Cuda arrays of their components.
fn allocate_vec3s(
    dev: &Arc<CudaDevice>,
    data: &[Vec3],
) -> (CudaSlice<f64>, CudaSlice<f64>, CudaSlice<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();

    // todo: Ref etcs; you are making a double copy here.
    for v in data {
        x.push(v.x);
        y.push(v.y);
        z.push(v.z);
    }

    (
        // todo: Explore other Cudarc data copying approaches
        // dev.dtoh_sync_copy(x).unwrap(),
        dev.htod_copy(x).unwrap(),
        dev.htod_copy(y).unwrap(),
        dev.htod_copy(z).unwrap(),
    )
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
    let N_CHARGES = posit_charges.len();
    let N_SAMPLES = posit_samples.len();

    let (posit_charges_x, posit_charges_y, posit_charges_z) = allocate_vec3s(&dev, posit_charges);
    let (posit_samples_x, posit_samples_y, posit_samples_z) = allocate_vec3s(&dev, posit_samples);

    let mut charges_gpu = dev.alloc_zeros::<f64>(N_CHARGES).unwrap();
    // dev.htod_sync_copy_into(charges, &mut charges_gpu).unwrap();
    dev.htod_sync_copy_into(charges, &mut charges_gpu).unwrap();

    // let a_dev = dev.htod_copy(a_host.into()).unwrap();
    // let mut b_dev = a_dev.clone();

    // todo: You will likely need 32-bit floats for performance purposes.
    let mut out = dev.alloc_zeros::<f64>(N_CHARGES * N_SAMPLES).unwrap();

    let kernel = dev.get_func("cuda", "coulomb_kernel").unwrap();

    let cfg = LaunchConfig::for_num_elems((N_CHARGES * N_SAMPLES) as u32);

    unsafe {
        kernel.launch(
            cfg,
            (
                &mut out,
                &posit_charges_x,
                &posit_charges_y,
                &posit_charges_z,
                &posit_samples_x,
                &posit_samples_y,
                &posit_samples_z,
                &charges_gpu,
                N_CHARGES,
                N_SAMPLES,
            ),
        )
    }
    .unwrap();

    // let a_host_2 = dev.sync_reclaim(a_dev)?;
    // let out_host = dev.sync_reclaim(b_dev)?;

    // Copy back to the host:
    let out_host = dev.dtoh_sync_copy(&out).unwrap();

    println!("OUT: {:?}", out_host);

    // Next step: Either here or in `potential.rs`:
    // You have flat output of the 2d correspondances between each posit sample and posit charge.
    // Expand into an array of arrays. (2D) etc.

    // (Should we do this on CPU, or GPU? Likely GPU.)


    // Loop over, and sum the contributions  from each charge, for the corresponding sample point.

    // Map the result back  back from a flat list to a 3D list per sample point.

    out_host

    // // unsafe initialization of unset memory
    // let _: CudaSlice<f32> = unsafe { dev.alloc::<f32>(10) }.unwrap();

    // // this will have memory initialized as 0
    // let _: CudaSlice<f64> = dev.alloc_zeros::<f64>(10).unwrap();

    // initialize with a rust vec
    // let _: CudaSlice<usize> = dev.htod_copy(vec![0; 10]).unwrap();

    // // or finially, initialize with a slice. this is synchronous though.
    // let _: CudaSlice<u32> = dev.htod_sync_copy(&[1, 2, 3]).unwrap();
}
