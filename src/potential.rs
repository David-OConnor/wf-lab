//! Contains code related to creating and combining potentials.

use crate::{
    types::{Arr3dReal, Arr3dVec},
    util,
};

use lin_alg2::f64::Vec3;

/// Computes V from nuclei, by calculating Coulomb potential.
/// Run this after changing these charges.
/// Does not modify per-electron charges; those are updated elsewhere, incorporating the
/// potential here, as well as from other electrons.
pub fn update_V_from_nuclei(
    V_nuclei: &mut Arr3dReal,
    charges_nuc: &[(Vec3, f64)],
    grid_posits: &Arr3dVec,
    grid_n: usize,
    // Wave functions from other electrons, for calculating the Hartree potential.
) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                let posit_sample = grid_posits[i][j][k];

                V_nuclei[i][j][k] = 0.;

                for (posit_charge, charge_amt) in charges_nuc.iter() {
                    V_nuclei[i][j][k] += util::V_coulomb(*posit_charge, posit_sample, *charge_amt);
                }
            }
        }
    }
}

/// Update the combined V; this is from nuclei, and all electrons.
/// Must be done after individual V from individual electrons are generated.
pub fn update_V_combined(
    V_combined: &mut Arr3dReal,
    V_nuc: &Arr3dReal,
    V_elecs: &[&Arr3dReal],
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

/// Update the V acting on a given electron. Run this after changing V nuclei, or V from another electron.
pub(crate) fn update_V_acting_on_elec(
    V_on_this_elec: &mut Arr3dReal,
    V_from_nuclei: &Arr3dReal,
    V_from_other_elecs: &[Arr3dReal],
    i_this_elec: usize,
    grid_n: usize,
) {
    println!("Updating V on this elec");
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                V_on_this_elec[i][j][k] = V_from_nuclei[i][j][k];

                for (i_other_elec, V_other_elec) in V_from_other_elecs.iter().enumerate() {
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

/// Update the V associated with a single electron's charge.
/// This must be run after the charge from this electron is created from the wave function square.
pub(crate) fn create_V_from_an_elec(
    V_from_this_elec: &mut Arr3dReal,
    charge_this_elec: &Arr3dReal,
    grid_posits: &Arr3dVec,
    grid_n: usize,
) {
    println!("Creating V from an electron...");

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                let posit_sample = grid_posits[i][j][k];

                // Iterate through this electron's (already computed) charge at every position in space,
                // comparing it to this position.

                V_from_this_elec[i][j][k] = 0.;

                for i_charge in 0..grid_n {
                    for j_charge in 0..grid_n {
                        for k_charge in 0..grid_n {
                            let posit_charge = grid_posits[i_charge][j_charge][k_charge];
                            let charge = charge_this_elec[i_charge][j_charge][k_charge];

                            V_from_this_elec[i][j][k] +=
                                util::V_coulomb(posit_charge, posit_sample, charge);
                        }
                    }
                }
            }
        }
    }

    println!("V creation complete");
}

// todo: This function may need a rework after your V refactor.
//
// /// Find the (repulsive) potential from electron charge densities, for a given electron, at a given position.
// /// The API uses a given index to represent a point in space, since we may combine this calculation in a loop
// /// to find the potential from (Born-Oppenheimer) nuclei.
// ///
// /// This should be run after charge densities are updated from psis.
// fn find_hartree_V(
//     charges_electron: &[Arr3dReal],
//     // The position in the array of this electron, so we don't have it repel itself.
//     i_this_elec: usize,
//     posit_sample: Vec3,
//     grid_posits: &Arr3dVec,
//     i: usize,
//     j: usize,
//     k: usize,
//     grid_n: usize,
// ) -> f64 {
//     // Re why the electron interaction, in many cases, appears to be very small compared to protons: After thinking about it, the protons, being point charges (approximately) are pulling from a single direction. While most of the smudged out electron gets cancelled out in the area of interest
//     // But, it should follow that at a distance, the electsron force and potential is as strong as the proton's
//     // (Yet, at a distance, the electron and proton charges cancel each other out largely, unless it's an ion...)
//     // So I guess it follows that the interesting bits are in the intermediate distances...
//
//     let mut result = 0.;
//
//     for (i_other_elec, charge_other_elec) in charges_electron.iter().enumerate() {
//         if i_other_elec == i_this_elec {
//             continue;
//         }
//
//         let mut test_sum = 0.;
//
//         for i2 in 0..grid_n {
//             for j2 in 0..grid_n {
//                 for k2 in 0..grid_n {
//                     // Don't compare the same point to itself; will get a divide-by-zero error
//                     // on the distance.
//                     if i2 == i && j2 == j && k2 == k {
//                         continue;
//                     }
//
//                     let posit_charge = grid_posits[i2][j2][k2];
//                     let charge = charge_other_elec[i2][j2][k2];
//
//                     test_sum += charge;
//
//                     // todo: This may not be quite right, ie matching the posit_sample grid with the i2, j2, k2 elec charges.
//                     result += util::V_coulomb(posit_charge, posit_sample, charge);
//                 }
//             }
//         }
//
//     }
//
//     result
// }
