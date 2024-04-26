//! Contains code related to creating and combining potentials.

use lin_alg::f64::Vec3;

#[cfg(feature = "cuda")]
use crate::gpu;
use crate::{
    grid_setup::{Arr2dReal, Arr2dVec, Arr3dReal, Arr3dVec},
    iter_arr, iter_arr_2d,
    types::ComputationDevice,
    wf_ops::K_C,
};

// We use this to prevent numerical anomolies and divide-by-0 errors in coulomb interactions, where
// the positions are very close to each other.
const SOFTENING_FACTOR: f64 = 0.000000000000001;

/// Utility function used to flatten charge data prior to sending to the GPU.
fn flatten_charge(posits_charge: &Arr3dVec, values_charge: &Arr3dReal) -> (Vec<Vec3>, Vec<f64>) {
    let mut posits = Vec::new();
    let mut charges = Vec::new();

    let grid_n = values_charge.len();
    for (i, j, k) in iter_arr!(grid_n) {
        posits.push(posits_charge[i][j][k]);
        charges.push(values_charge[i][j][k]);
    }

    (posits, charges)
}

/// Computes potential field from nuclei, by calculating Coulomb potential.
/// Run this after changing these charges.
/// Does not modify per-electron charges; those are updated elsewhere, incorporating the
/// potential here, as well as from other electrons.
pub fn update_V_from_nuclei(
    // V_from_nuclei: &mut Arr3dReal,
    V_from_nuclei: &mut Arr2dReal,
    charges_nuc: &[(Vec3, f64)],
    // grid_posits: &Arr3dVec,
    grid_posits: &Arr2dVec,
    // Wave functions from other electrons, for calculating the Hartree potential.
) {
    let grid_n = grid_posits.len();
    // todo: CUDA

    // todo: You may need both variants: One for charge, one for render. (2d/3d)

    // for (i, j, k) in iter_arr!(grid_n) {
    //     let posit_sample = grid_posits[i][j][k];
    //
    //     V_from_nuclei[i][j][k] = 0.;
    //
    //     for (posit_charge, charge_amt) in charges_nuc.iter() {
    //         V_from_nuclei[i][j][k] += V_coulomb(*posit_charge, posit_sample, *charge_amt);
    //     }
    // }
    for (i, j) in iter_arr_2d!(grid_n) {
        let posit_sample = grid_posits[i][j];

        V_from_nuclei[i][j] = 0.;

        for (posit_charge, charge_amt) in charges_nuc.iter() {
            V_from_nuclei[i][j] += V_coulomb(*posit_charge, posit_sample, *charge_amt);
        }
    }
}

/// Update the potential field acting on a given electron. Run this after changing V nuclei,
/// or V from another electron.
pub(crate) fn update_V_acting_on_elec(
    // V_on_this_elec: &mut Arr3dReal,
    V_on_this_elec: &mut Arr2dReal,
    // V_from_nuclei: &Arr3dReal,
    V_from_nuclei: &Arr2dReal,
    // V_from_elecs: &Arr3dReal,
    V_from_elecs: &Arr2dReal,
    grid_n: usize,
) {
    // for (i, j, k) in iter_arr!(grid_n) {
    for (i, j) in iter_arr_2d!(grid_n) {
        // V_on_this_elec[i][j][k] = V_from_nuclei[i][j][k];
        // V_on_this_elec[i][j][k] += V_from_elecs[i][j][k];
        V_on_this_elec[i][j] = V_from_nuclei[i][j];
        V_on_this_elec[i][j] += V_from_elecs[i][j];
    }
}

// /// Update the potential field acting on a given electron, by summing already-calcualted V
// /// associated with other electrons. Run this after changing V nuclei,
// /// or V from another electron.
// pub(crate) fn update_V_acting_on_elec_1d(
//     V_on_this_elec: &mut [f64],
//     V_from_nuclei: &[f64],
//     V_from_elecs: &[Vec<f64>],
//     i_this_elec: usize,
//     grid_n: usize,
// ) {
//     for i in 0..grid_n {
//         V_on_this_elec[i] = V_from_nuclei[i];
//
//         for (i_other_elec, V_other_elec) in V_from_elecs.iter().enumerate() {
//             // Don't apply this own electron's charge to the V on it.
//             if i_this_elec == i_other_elec {
//                 continue;
//             }
//             V_on_this_elec[i] += V_other_elec[i];
//         }
//     }
//     println!("Complete");
// }

/// See `create_V`. This assumes the sample positions are flattened already, vice arranged in
/// 3D grids. Note that it returns a result, vice modifying in place.
pub(crate) fn create_V_1d_from_elecs(
    dev: &ComputationDevice,
    posits_sample: &[Vec3],
    charges_elec: &Arr3dReal,
    posits_charge: &Arr3dVec,
) -> Vec<f64> {
    match dev {
        #[cfg(feature = "cuda")]
        ComputationDevice::Gpu(cuda_dev) => {
            let (posits_charge_flat, charges_flat) = flatten_charge(posits_charge, charges_elec);

            // Calculate the charge from electrons using the GPU
            gpu::run_coulomb(cuda_dev, &posits_charge_flat, posits_sample, &charges_flat)
        }
        ComputationDevice::Cpu => {
            create_V_from_elec_1d_cpu(posits_sample, charges_elec, posits_charge)
        }
    }
}

/// Update the V associated with a single electron's charge.
/// This must be run after the charge from this electron is created from the wave function square.
/// We use the GPU, due to the large number of operations involved (3d set of sample points interacting
/// we each of a 3d set of charge points). We leave the option to calculate sample points only on
/// a 2D grid, but this may not be necessary given how fast the full operation is on GPU.
pub(crate) fn create_V_from_elecs(
    dev: &ComputationDevice,
    V_from_this_elec: &mut Arr3dReal,
    posits_sample: &Arr3dVec,
    posits_charge: &Arr3dVec,
    charges_elec: &Arr3dReal,
    grid_n_sample: usize,
    grid_n_charge: usize,
    twod_only: bool,
) {
    println!("Creating V from an electron on grid...");

    match dev {
        #[cfg(feature = "cuda")]
        ComputationDevice::Gpu(cuda_dev) => {
            let (posits_charge_flat, charges_flat) = flatten_charge(posits_charge, charges_elec);

            let mut posits_sample_flat = Vec::new();

            // Similar to `util::flatten_arr`, but With the 2d option.
            // Flatten sample positions, prior to passing to the kernel.
            for i_sample in 0..grid_n_sample {
                for j_sample in 0..grid_n_sample {
                    if twod_only {
                        // This makes it grid_n times faster, but only creates one Z-slice.
                        let k_sample = grid_n_sample / 2 + 1;
                        posits_sample_flat.push(posits_sample[i_sample][j_sample][k_sample]);
                    } else {
                        for k_sample in 0..grid_n_sample {
                            posits_sample_flat.push(posits_sample[i_sample][j_sample][k_sample]);
                        }
                    }
                }
            }

            let V_per_sample_flat = gpu::run_coulomb(
                cuda_dev,
                &posits_charge_flat,
                &posits_sample_flat,
                &charges_flat,
            );

            let grid_n_sq = grid_n_sample.pow(2);

            // Similar to `util::unflatten_arr`, but With the 2d option.
            // Repack into a 3D array. We do it here, vice in `gpu::run_coulomb`, since we use `run_coulomb`
            // for flat sample input as well.
            for i in 0..grid_n_sample {
                for j in 0..grid_n_sample {
                    if twod_only {
                        let k = grid_n_sample / 2 + 1;
                        let i_flat = i * grid_n_sample + j;
                        V_from_this_elec[i][j][k] = V_per_sample_flat[i_flat];
                    } else {
                        for k in 0..grid_n_sample {
                            let i_flat = i * grid_n_sq + j * grid_n_sample + k;
                            V_from_this_elec[i][j][k] = V_per_sample_flat[i_flat];
                        }
                    }
                }
            }
        }
        ComputationDevice::Cpu => {
            create_V_from_elec_grid_cpu(
                V_from_this_elec,
                &charges_elec,
                posits_sample,
                posits_charge,
                grid_n_sample,
                grid_n_charge,
                twod_only,
            );
        }
    }

    println!("V creation complete");
}

/// Single-point Coulomb potential, eg a hydrogen nuclei.
pub(crate) fn V_coulomb(posit_charge: Vec3, posit_sample: Vec3, charge: f64) -> f64 {
    let diff = posit_sample - posit_charge;
    let r = diff.magnitude();

    K_C * charge / (r + SOFTENING_FACTOR)
}

/// Helper fn to deal with 2d vs 3d electron V. Calculates Coulomb potential at every charge point,
/// at a single sample point.
fn V_from_grid_inner_cpu(
    V_from_this_elec: &mut Arr3dReal,
    charge_this_elec: &Arr3dReal,
    grid_posits: &Arr3dVec,
    grid_charge: &Arr3dVec,
    grid_n_charge: usize,
    posit: (usize, usize, usize),
) {
    let posit_sample = grid_posits[posit.0][posit.1][posit.2];

    // Iterate through this electron's (already computed) charge at every position in space,
    // comparing it to this position.

    V_from_this_elec[posit.0][posit.1][posit.2] = 0.;

    for (i, j, k) in iter_arr!(grid_n_charge) {
        let posit_charge = grid_charge[i][j][k];
        let charge = charge_this_elec[i][j][k];

        V_from_this_elec[posit.0][posit.1][posit.2] +=
            V_coulomb(posit_charge, posit_sample, charge);
    }
}

/// Update the V associated with a single electron's charge.
/// This must be run after the charge from this electron is created from the wave function square.
/// We expect the loop over charge positions to be larger than the one over V positions.
///
/// This is computationally intensive. The `twod_only` option can alleviate this by only
/// evaluating potential points on one plane.
/// Deprecated in favor of GPU.
pub(crate) fn create_V_from_elec_grid_cpu(
    V_from_this_elec: &mut Arr3dReal,
    charges_elec: &Arr3dReal,
    posits_sample: &Arr3dVec,
    posits_charge: &Arr3dVec,
    grid_n: usize,
    grid_n_charge: usize,
    twod_only: bool,
) {
    for i_sample in 0..grid_n {
        for j_sample in 0..grid_n {
            if twod_only {
                // This makes it grid_n times faster, but only creates one Z-slice.
                let k_sample = grid_n / 2 + 1;
                V_from_grid_inner_cpu(
                    V_from_this_elec,
                    charges_elec,
                    posits_sample,
                    posits_charge,
                    grid_n_charge,
                    (i_sample, j_sample, k_sample),
                )
            } else {
                for k_sample in 0..grid_n {
                    V_from_grid_inner_cpu(
                        V_from_this_elec,
                        charges_elec,
                        posits_sample,
                        posits_charge,
                        grid_n_charge,
                        (i_sample, j_sample, k_sample),
                    )
                }
            }
        }
    }
}

/// Create a potential on a set of sample points, from nuclei and electrons.
pub(crate) fn create_V_from_elec_1d_cpu(
    posits_sample: &[Vec3],
    charges_elec: &Arr3dReal,
    posits_charge: &Arr3dVec,
) -> Vec<f64> {
    let mut result = Vec::new();
    let grid_n_charge = posits_charge.len();

    for sample_pt in posits_sample {
        let mut V_sample = 0.;

        for (i, j, k) in iter_arr!(grid_n_charge) {
            let posit_charge = posits_charge[i][j][k];
            let charge = charges_elec[i][j][k];

            V_sample += V_coulomb(posit_charge, *sample_pt, charge);
        }

        result.push(V_sample);
    }

    result
}
