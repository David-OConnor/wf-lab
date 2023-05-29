//! This module contains code for electron-electron interactions, including EM repulsion,
//! and exchange forces.

use core::f64::consts::FRAC_1_SQRT_2;

use crate::{
    types::{Arr3d, Arr3dReal, Arr3dVec},
    util,
    wf_ops::{self, Q_ELEC},
};

use lin_alg2::f64::Vec3;

/// Convert an array of ψ to one of electron charge, through space. Modifies in place
/// to avoid unecessary allocations.
pub(crate) fn update_charge_density_fm_psi(
    psi: &Arr3d,
    charge_density: &mut Arr3dReal,
    grid_n: usize,
) {
    println!("Creating electron charge for the active e-");

    // todo: Problem? Needs to sum to 1 over *all space*, not just in the grid.
    // todo: We can mitigate this by using a sufficiently large grid bounds, since the WF
    // todo goes to 0 at distance.

    // todo: Consequence of your irregular grid: Is this normalization process correct?

    // Normalize <ψ|ψ>
    let mut psi_sq_size = 0.;
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                psi_sq_size += psi[i][j][k].abs_sq();
            }
        }
    }

    let num_elecs = 1;
    // Save computation on this constant factor.
    let c = Q_ELEC * num_elecs as f64 / psi_sq_size;

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                charge_density[i][j][k] = psi[i][j][k].abs_sq() * c;
            }
        }
    }
}

// todo: Currently unused.
/// Update electron charge densities ψ, for every electron.
pub(crate) fn update_charge_densities_fm_psi(
    charges_fm_elecs: &mut [Arr3dReal],
    psi_per_electron: &[Arr3d],
    grid_n: usize,
) {
    for (i, psi) in psi_per_electron.iter().enumerate() {
        update_charge_density_fm_psi(psi, &mut charges_fm_elecs[i], grid_n)
    }
}

/// Find the (repulsive) potential from electron charge densities, for a given electron, at a given position.
/// The API uses a given index to represent a point in space, since we may combine this calculation in a loop
/// to find the potential from (Born-Oppenheimer) nuclei.
///
/// This should be run after charge densities are updated from psis.
pub(crate) fn find_hartree_V(
    charges_electron: &[Arr3dReal],
    // The position in the array of this electron, so we don't have it repel itself.
    i_this_elec: usize,
    posit_sample: Vec3,
    grid_posits: &Arr3dVec,
    i: usize,
    j: usize,
    k: usize,
    grid_n: usize,
) -> f64 {
    // Re why the electron interaction, in many cases, appears to be very small compared to protons: After thinking about it, the protons, being point charges (approximately) are pulling from a single direction. While most of the smudged out electron gets cancelled out in the area of interest
    // But, it should follow that at a distance, the electsron force and potential is as strong as the proton's
    // (Yet, at a distance, the electron and proton charges cancel each other out largely, unless it's an ion...)
    // So I guess it follows that the interesting bits are in the intermediate distances...

    let mut result = 0.;

    for (i_other_elec, charge_other_elec) in charges_electron.iter().enumerate() {
        if i_other_elec == i_this_elec {
            continue;
        }

        let mut test_sum = 0.;

        for i2 in 0..grid_n {
            for j2 in 0..grid_n {
                for k2 in 0..grid_n {
                    // Don't compare the same point to itself; will get a divide-by-zero error
                    // on the distance.
                    if i2 == i && j2 == j && k2 == k {
                        continue;
                    }

                    let posit_charge = grid_posits[i2][j2][k2];
                    let charge = charge_other_elec[i2][j2][k2];

                    test_sum += charge;

                    // println!("Charge: {:?}", charge);

                    // todo: This may not be quite right, ie matching the posit_sample grid with the i2, j2, k2 elec charges.
                    result += util::V_coulomb(posit_charge, posit_sample, charge);
                }
            }
        }

        // println!("Total Q: {}", test_sum);
    }

    result
}

/// Update the (nuclear and other-electron) potential for a single electron. This resets it to the V from nuclei,
/// plus V from the wavefunction -> charge density of other electrons. Does not include charge density
/// from own Psi.
/// This must be run after charge from the other electrons is created from the wave function square.
pub(crate) fn update_V_individual(
    V_this_elec: &mut Arr3dReal,
    V_nuclei: &Arr3dReal,
    charges_electron: &[Arr3dReal],
    // The position in the array of this electron, so we don't have it repel itself.
    i_this_elec: usize,
    grid_posits: &Arr3dVec,
    grid_n: usize,
) {
    println!("Updating V for {}", i_this_elec);

    // println!("C1: {}", charges_electron[0][6][5][5]);
    // println!("C2: {}", charges_electron[1][6][5][5]);

    // let mut test_nuc_sum = 0.;
    // let mut test_tot_sum = 0.;

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                let posit_sample = grid_posits[i][j][k];

                // Combine nuclear charge with charge from other electrons.
                V_this_elec[i][j][k] = V_nuclei[i][j][k];

                // println!("Vn: {}", V_nuclei[i][j][k]);

                // test_nuc_sum += V_nuclei[i][j][k];

                // println!("C3: {}", charges_electron[0][i][j][k]);
                // println!("C4: {}", charges_electron[1][i][j][k]);

                V_this_elec[i][j][k] += find_hartree_V(
                    charges_electron,
                    i_this_elec,
                    posit_sample,
                    grid_posits,
                    i,
                    j,
                    k,
                    grid_n,
                );

                // println!("N: {} T: {}",  V_nuclei[i][j][k], V_this_elec[i][j][k]);
                // test_tot_sum += V_this_elec[i][j][k];
            }
        }
    }

    println!("V update complete");
}

/// Calculate the result of exchange interactions between electrons.
pub(crate) fn calc_exchange(psis: &[Arr3d], result: &mut Arr3d, grid_n: usize) {
    // for i_a in 0..N {
    //     for j_a in 0..N {
    //         for k_a in 0..N {
    //             for i_b in 0..N {
    //                 for j_b in 0..N {
    //                     for k_b in 0..N {
    //
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    *result = Arr3d::new();

    for a in 0..grid_n {
        // todo: i, j, k for 3D
        for b in 0..grid_n {
            // This term will always be 0, so skipping  here may save calculation.
            if a == b {
                continue;
            }
            // Enumerate so we don't calculate exchange on a WF with itself.
            for (i_1, psi_1) in psis.iter().enumerate() {
                for (i_2, psi_2) in psis.iter().enumerate() {
                    // Don't calcualte exchange with self
                    if i_1 == i_2 {
                        continue;
                    }

                    // todo: THink this through. What index to update?
                    // result[a] += FRAC_1_SQRT_2 * (psi_1[a] * psi_2[b] - psi_2[a] * psi_1[b]);
                }
            }
        }
    }
}
