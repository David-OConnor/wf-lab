//! Contains code related to creating and combining potentials.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use lin_alg2::f64::Vec3;

use crate::{
    gpu,
    grid_setup::{Arr3dReal, Arr3dVec},
    wf_ops::K_C,
};

/// Utility function used to flatten position and charge data prior to sending
/// to the GPU. todo: Could use for position too if it werent for the charge values being included.
fn flatten_charge(
    posits_charge: &Arr3dVec,
    values_charge: &Arr3dReal,
    grid_n: usize,
) -> (Vec<Vec3>, Vec<f64>) {
    let mut posits = Vec::new();
    let mut charges = Vec::new();

    for i_charge in 0..grid_n {
        for j_charge in 0..grid_n {
            for k_charge in 0..grid_n {
                posits.push(posits_charge[i_charge][j_charge][k_charge]);
                charges.push(values_charge[i_charge][j_charge][k_charge]);
            }
        }
    }

    (posits, charges)
}

/// Create a potential on a set of sample points, from nuclei and electrons.
pub fn create_V_1d(
    posits_sample: &[Vec3],
    charges_fixed: &[(Vec3, f64)],
    charges_elec: &Arr3dReal,
    posits_charge: &Arr3dVec,
    grid_n_charge: usize,
) -> Vec<f64> {
    let mut V_to_match = Vec::new();

    for sample_pt in posits_sample {
        let mut V_sample = 0.;

        for (posit_nuc, charge) in charges_fixed {
            V_sample += V_coulomb(*posit_nuc, *sample_pt, *charge);
        }

        for i in 0..grid_n_charge {
            for j in 0..grid_n_charge {
                for k in 0..grid_n_charge {
                    let posit_charge = posits_charge[i][j][k];
                    let charge = charges_elec[i][j][k];

                    V_sample += V_coulomb(posit_charge, *sample_pt, charge);
                }
            }
        }

        V_to_match.push(V_sample);
    }

    V_to_match
}

pub fn create_V_1d_gpu(
    cuda_dev: &Arc<CudaDevice>,
    posits_sample: &[Vec3],
    charges_fixed: &[(Vec3, f64)],
    charges_elec: &Arr3dReal,
    posits_charge: &Arr3dVec,
    grid_n_charge: usize,
) -> Vec<f64> {
    let mut V_to_match = Vec::new();

    let (posits_charge_flat, charges_flat) =
        flatten_charge(posits_charge, charges_elec, grid_n_charge);

    // Calculate the charge from electrons using the GPU
    let mut per_sample_flat =
        gpu::run_coulomb(cuda_dev, &posits_charge_flat, &posits_sample, &charges_flat);

    // Add the charge from nucleii.
    for (i, sample_pt) in posits_sample.iter().enumerate() {
        for (posit_nuc, charge_nuc) in charges_fixed {
            per_sample_flat[i] += V_coulomb(*posit_nuc, *sample_pt, *charge_nuc);
        }
    }

    per_sample_flat
}

/// Computes potential field from nuclei, by calculating Coulomb potential.
/// Run this after changing these charges.
/// Does not modify per-electron charges; those are updated elsewhere, incorporating the
/// potential here, as well as from other electrons.
pub fn update_V_from_nuclei(
    V_from_nuclei: &mut Arr3dReal,
    charges_nuc: &[(Vec3, f64)],
    grid_posits: &Arr3dVec,
    grid_n: usize,
    // Wave functions from other electrons, for calculating the Hartree potential.
) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                let posit_sample = grid_posits[i][j][k];

                V_from_nuclei[i][j][k] = 0.;

                for (posit_charge, charge_amt) in charges_nuc.iter() {
                    V_from_nuclei[i][j][k] += V_coulomb(*posit_charge, posit_sample, *charge_amt);
                }
            }
        }
    }
}

pub fn update_V_from_nuclei_1d(
    V_from_nuclei: &mut [f64], // by posit
    charges_nuc: &[(Vec3, f64)],
    posits: &[Vec3],
    grid_n: usize,
    // Wave functions from other electrons, for calculating the Hartree potential.
) {
    for i in 0..grid_n {
        let posit_sample = posits[i];

        V_from_nuclei[i] = 0.;

        for (posit_charge, charge_amt) in charges_nuc.iter() {
            V_from_nuclei[i] += V_coulomb(*posit_charge, posit_sample, *charge_amt);
        }
    }
}

/// Update the potential field acting on a given electron. Run this after changing V nuclei,
/// or V from another electron.
pub(crate) fn update_V_acting_on_elec(
    V_on_this_elec: &mut Arr3dReal,
    V_from_nuclei: &Arr3dReal,
    V_from_elecs: &[Arr3dReal],
    i_this_elec: usize,
    grid_n: usize,
) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                V_on_this_elec[i][j][k] = V_from_nuclei[i][j][k];

                for (i_other_elec, V_other_elec) in V_from_elecs.iter().enumerate() {
                    // Don't apply this own electron's charge to the V on it.
                    if i_this_elec == i_other_elec {
                        continue;
                    }
                    V_on_this_elec[i][j][k] += V_other_elec[i][j][k];
                }
            }
        }
    }
}

/// Update the potential field acting on a given electron. Run this after changing V nuclei,
/// or V from another electron.
pub(crate) fn update_V_acting_on_elec_1d(
    V_on_this_elec: &mut [f64],
    V_from_nuclei: &[f64],
    V_from_elecs: &[Vec<f64>],
    i_this_elec: usize,
    grid_n: usize,
) {
    println!("Updating V on this elec (1d)...");
    for i in 0..grid_n {
        V_on_this_elec[i] = V_from_nuclei[i];

        for (i_other_elec, V_other_elec) in V_from_elecs.iter().enumerate() {
            // Don't apply this own electron's charge to the V on it.
            if i_this_elec == i_other_elec {
                continue;
            }
            V_on_this_elec[i] += V_other_elec[i];
        }
    }
    println!("Complete");
}

/// Helper fn to deal with 2d vs 3d electron V. Calculates Coulomb potential at every charge point,
/// at a single sample point.
fn V_from_grid_inner(
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

    for i_charge in 0..grid_n_charge {
        for j_charge in 0..grid_n_charge {
            for k_charge in 0..grid_n_charge {
                let posit_charge = grid_charge[i_charge][j_charge][k_charge];
                let charge = charge_this_elec[i_charge][j_charge][k_charge];

                V_from_this_elec[posit.0][posit.1][posit.2] +=
                    V_coulomb(posit_charge, posit_sample, charge);
            }
        }
    }
}

/// Update the V associated with a single electron's charge.
/// This must be run after the charge from this electron is created from the wave function square.
/// We expect the loop over charge positions to be larger than the one over V positions.
///
/// This is computationally intensive. The `twod_only` option can alleviate this by only
/// evaluating potential points on one plane.
pub(crate) fn create_V_from_elec_grid(
    V_from_this_elec: &mut Arr3dReal,
    charges_elec: &Arr3dReal,
    posits_sample: &Arr3dVec,
    posits_charge: &Arr3dVec,
    grid_n: usize,
    grid_n_charge: usize,
    twod_only: bool,
) {
    println!("Creating V from an electron on grid...");

    for i_sample in 0..grid_n {
        for j_sample in 0..grid_n {
            if twod_only {
                // This makes it grid_n times faster, but only creates one Z-slice.
                let k_sample = grid_n / 2 + 1;
                V_from_grid_inner(
                    V_from_this_elec,
                    charges_elec,
                    posits_sample,
                    posits_charge,
                    grid_n_charge,
                    (i_sample, j_sample, k_sample),
                )
            } else {
                for k_sample in 0..grid_n {
                    V_from_grid_inner(
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

    println!("V creation complete");
}

/// See `create_V_from_elec_grid`.
pub(crate) fn create_V_from_elec_grid_gpu(
    cuda_dev: &Arc<CudaDevice>,
    V_from_this_elec: &mut Arr3dReal,
    charge_elec: &Arr3dReal,
    posits_sample: &Arr3dVec,
    posits_charge: &Arr3dVec,
    grid_n: usize,
    grid_n_charge: usize,
    twod_only: bool,
) {
    println!("Creating V from an electron on grid (GPU)...");

    let (posits_charge_flat, charges_flat) =
        flatten_charge(posits_charge, charge_elec, grid_n_charge);

    let mut posits_sample_flat = Vec::new();

    for i_sample in 0..grid_n {
        for j_sample in 0..grid_n {
            if twod_only {
                // This makes it grid_n times faster, but only creates one Z-slice.
                let k_sample = grid_n / 2 + 1;
                posits_sample_flat.push(posits_sample[i_sample][j_sample][k_sample]);
            } else {
                for k_sample in 0..grid_n {
                    posits_sample_flat.push(posits_sample[i_sample][j_sample][k_sample]);
                }
            }
        }
    }

    let per_sample_flat = gpu::run_coulomb(
        cuda_dev,
        &posits_charge_flat,
        &posits_sample_flat,
        &charges_flat,
    );

    // Repack here, vice in `gpu::run_coulomb`, since we use `run_coulomb` for flat sample input as well.
    for i in 0..grid_n {
        for j in 0..grid_n {
            if twod_only {
                let k = grid_n / 2 + 1;
                let i_flat = i * grid_n + j; // QC etc
                V_from_this_elec[i][j][k] = per_sample_flat[i_flat];
            } else {
                for k in 0..grid_n {
                    let i_flat = i * grid_n + j * grid_n + k; // QC etc
                    V_from_this_elec[i][j][k] = per_sample_flat[i_flat];
                }
            }
        }
    }

    println!("V creation complete");
}

/// Single-point Coulomb potential, eg a hydrogen nuclei.
pub(crate) fn V_coulomb(posit_charge: Vec3, posit_sample: Vec3, charge: f64) -> f64 {
    let diff = posit_sample - posit_charge;
    let r = diff.magnitude();

    if r < 0.0000000000001 {
        return 0.; // todo: Is this the way to handle?
    }

    K_C * charge / r
}

/// Update the combined V; this is from nuclei, and all electrons.
/// Must be done after individual V from individual electrons are generated.
pub fn _update_V_combined(
    V_combined: &mut Arr3dReal,
    V_nuc: &Arr3dReal,
    V_elecs: &[Arr3dReal],
    grid_n: usize,
) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                V_combined[i][j][k] = V_nuc[i][j][k];

                for V_elec in V_elecs {
                    V_combined[i][j][k] += V_elec[i][j][k]
                }
            }
        }
    }
}
