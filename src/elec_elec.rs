//! This module contains code for electron-electron interactions, including EM repulsion,
//! and exchange forces.

use std::f64::consts::FRAC_1_SQRT_2;

use crate::{
    complex_nums::Cplx,
    types::{new_data, SurfacesPerElec},
    types::{Arr3d, Arr3dReal, Arr3dVec},
    util,
    wf_ops::{self, PsiWDiffs, Q_ELEC},
};

use lin_alg2::f64::Vec3;

/// Represents a wave function of multiple electrons
/// todo: Incorporate spin.
pub struct WaveFunctionMultiElec {
    num_elecs: usize,
    /// A column-major slater determinant.
    // components_slater: Vec<Arr3d>

    /// Maps permutations of position index to wave function value.
    /// The outer vec has a value for each permutation of spacial coordinates, with length
    /// `num_elecs`. So, its length is (n_grid^3)^num_elecs
    psi_joint: Vec<(Vec<(usize, usize, usize)>, Cplx)>,
    /// This is the combined probability, created from `psi_joint`.
    pub psi_marginal: PsiWDiffs,
}

/// Be careful: (Wiki): "Only a small subset of all possible fermionic wave functions can be written
/// as a single Slater determinant, but those form an important and useful subset because of their simplicity."
/// What is the more general case?
/// todo: YOu may need to figure out how to deal with chareg density on an irregular grid.
impl WaveFunctionMultiElec {
    pub fn new(num_elecs: usize, n_grid: usize) -> Self {
        let data = new_data(n_grid);
        Self {
            num_elecs,
            psi_joint: Vec::new(),
            psi_marginal: PsiWDiffs::init(&data),
        }
    }

    /// `posits_by_elec` is indexed by the electron index. For example, for a 2-electron system, returns
    /// the probability of finding electron 0 in `posits_by_elec[0]`, and electron 1 in `posits_by_elec[1]`.
    pub fn populate_psi_combined(&mut self) {
        for (posits, val) in &self.psi_joint {
            for posit in posits {
                self.psi_marginal.on_pt[posit.0][posit.1][posit.2] += *val;
            }
        }
    }
    // let's say n = 2, and r spans 0, 1, 2, 3
    // we want to calc electron density at 2, or collect all relevant parts
    // val([x0, x1], [r0, r0, r0]),
    // val ([x0, x1], [r0, r0, r1]),
    // val([x0, x1], [r0, r0, r2]),
    // val([x0, x1], [r0, r0, 3]),
    // val([x0, x1], [r0, r1, 0]),

    // let's say, n=3. x = [x0, x1, x2]
    // posits = [r0, r1, r2] where we

    /// Set up values for the joint wavefunction; related to probabilities of the electrons
    /// being in a combination of positions.
    pub fn setup_joint_wf(&mut self, wfs: &[Arr3d], grid_n: usize) {
        // todo: If you run this function more often than generating the posits
        // todo store them somewhere;
        let mut posits = Vec::new();

        for i in 0..grid_n {
            for j in 0..grid_n {
                for k in 0..grid_n {
                    posits.push((i, j, k));
                }
            }
        }

        let mut psi_joint = Vec::new();

        let mut posit_permutations = Vec::new();
        // for 2 elecs:
        // [p0, p0], [p0, p1], [p0, p2].. etc, [p1, p0]... etc
        // todo: Hard-coded for 2 electrons. Find or use a general algorithm.

        // for r_i in 0..self.num_elecs {
        //     let mut permutation = Vec::new();
        //     for (i, r) in posits.iter().enumerate() {
        //         permutation[i] = r;
        //     }
        //     posit_permutations.push(permutation);
        // }

        for r0 in &posits {
            for r1 in &posits {
                posit_permutations.push(vec![*r0, *r1]);
            }
        }

        for permutation in posit_permutations {
            psi_joint.push((
                permutation.clone(),
                self.joint_wf_at_permutation(wfs, &permutation),
            ));
        }

        // for (i, x) in wfs.iter().enumerate() {
        //     for posit_combo in posit_permutations {
        //         self.val(x, posit_combo)
        //     }
        // }

        self.psi_joint = psi_joint;
    }

    /// Find the wave fucntion value associated with a single permutation of position. Eg, the one associated
    /// with the probability that electron 0 is in position 0 and electron 1 is in position 1.
    pub fn joint_wf_at_permutation(&mut self, x: &[Arr3d], r: &[(usize, usize, usize)]) -> Cplx {
        // hardcoded 2x2 to test
        return Cplx::from_real(FRAC_1_SQRT_2)
            * (x[0][r[0].0][r[0].1][r[0].2] * x[1][r[1].0][r[1].1][r[1].2]
                - x[1][r[0].0][r[0].1][r[0].2] * x[0][r[1].0][r[1].1][r[1].2]);

        // hardcoded 3x3 to test. // todo: QC norm const
        // 1. / 3.0.sqrt() * (
        //     x[0][i0][j0][k0] * x[1][i1][j1][k1] * x[2][i2][j2][k2] +
        //         x[1][i0][j0][k0] * x[2][i1][j1][k1] * x[0][i2][j2][k2] +
        //         x[2][i0][j0][k0] * x[0][i1][j1][k1] * x[1][i2][j2][k2] -
        //         x[0][i0][j0][k0] * x[2][i1][j1][k1] * x[1][i2][j2][k2] -
        //         x[1][i0][j0][k0] * x[0][i1][j1][k1] * x[2][i2][j2][k2] -
        //         x[2][i0][j0][k0] * x[1][i1][j1][k1] * x[0][i2][j2][k2]
        // );

        // for i_x in 0..self.num_elecs {
        //     let mut entry = 1.;
        //     for i_posit in 0..self.num_elecs {
        //         entry *= x[i_x]posits[i_posit]; // todo not quite right. Check the 3x3 example for how it goes.
        //     }
        // }
    }

    pub fn calc_charge_density(&self, posit: Vec3) -> f64 {
        0.
    }
}

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
