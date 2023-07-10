//! Contains code related to creating and combining potentials.

use crate::{
    grid_setup::{Arr3dReal, Arr3dVec},
    util,
    wf_ops::K_C,
};

use lin_alg2::f64::Vec3;

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

/// Update the combined V; this is from nuclei, and all electrons.
/// Must be done after individual V from individual electrons are generated.
pub fn update_V_combined(
    V_combined: &mut Arr3dReal,
    V_nuc: &Arr3dReal,
    // V_elecs: &[&Arr3dReal],
    V_elecs: &[Arr3dReal],
    grid_n: usize,
) {
    // todo: QC this.
    // We combine electron Vs initially; this is required to prevent numerical errors. (??)
    // let mut V_from_elecs = grid_setup::new_data_real(grid_n);
    // for i in 0..grid_n {
    //     for j in 0..grid_n {
    //         for k in 0..grid_n {
    //             for V_elec in V_elecs {
    //                 V_from_elecs[i][j][k] += V_elec[i][j][k];
    //             }
    //         }
    //     }
    // }

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                V_combined[i][j][k] = V_nuc[i][j][k];

                for V_elec in V_elecs {
                    V_combined[i][j][k] += V_elec[i][j][k]
                }

                // todo temp
                // V_combined[i][j][k] += V_elecs[0][i][j][k];
                // V_combined[i][j][k] += V_from_elecs[i][j][k];

                // println!("Nuc: {}", V_nuc[i][j][k]);
                // println!("Elec: {}", V_from_elecs[i][j][k]);
            }
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
    // println!("Updating V on this elec...");
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
    // println!("Complete");
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

/// Update the V associated with a single electron's charge.
/// This must be run after the charge from this electron is created from the wave function square.
/// We expect the loop over charge positions to be larger than the one over V positions.
pub(crate) fn create_V_from_an_elec_grid(
    V_from_this_elec: &mut Arr3dReal,
    charge_this_elec: &Arr3dReal,
    grid_posits: &Arr3dVec,
    grid_posits_charge: &Arr3dVec,
    grid_n: usize,
    grid_n_charge: usize,
) {
    println!("Creating V from an electron on grid...");

    // todo: Are there any tricks or approximations we can use to make this less computationally-intense?
    // todo: Perhaps you could create an approximate analytic function of charge density over space,
    // todo then shoot rays or something out at evenly spaced angles from the sample pt??

    for i_sample in 0..grid_n {
        for j_sample in 0..grid_n {
            for k_sample in 0..grid_n {
                let posit_sample = grid_posits[i_sample][j_sample][k_sample];

                // Iterate through this electron's (already computed) charge at every position in space,
                // comparing it to this position.

                V_from_this_elec[i_sample][j_sample][k_sample] = 0.;

                for i_charge in 0..grid_n_charge {
                    for j_charge in 0..grid_n_charge {
                        for k_charge in 0..grid_n_charge {
                            // This will produce infinities due to 0 r.

                            // todo: You may still need a check here to prevent infinities.
                            // if i == i_charge && j == j_charge && k == k_charge {
                            //     continue;
                            // }

                            let posit_charge = grid_posits_charge[i_charge][j_charge][k_charge];
                            let charge = charge_this_elec[i_charge][j_charge][k_charge];

                            V_from_this_elec[i_sample][j_sample][k_sample] +=
                                V_coulomb(posit_charge, posit_sample, charge);
                        }
                    }
                }
            }
        }
    }

    println!("V creation complete");
}

/// Update the V associated with a single electron's charge.
/// This must be run after the charge from this electron is created from the wave function square.
/// We expect the loop over charge positions to be larger than the one over V positions.
///
/// This is (at least for now) only for the 1d eval data set, to save computation. This means
/// we can't currently visualize this potential.
pub(crate) fn create_V_from_an_elec(
    V_from_this_elec: &mut [f64],
    charge_this_elec: &Arr3dReal,
    grid_posits: &[Vec3],
    grid_posits_charge: &Arr3dVec,
    grid_n: usize,
    grid_n_charge: usize,
) {
    // println!("Creating V from an electron...");

    // todo: Are there any tricks or approximations we can use to make this less computationally-intense?
    // todo: Perhaps you could create an approximate analytic function of charge density over space,
    // todo then shoot rays or something out at evenly spaced angles from the sample pt??

    for i_sample in 0..grid_n {
        let posit_sample = grid_posits[i_sample];

        // Iterate through this electron's (already computed) charge at every position in space,
        // comparing it to this position.

        V_from_this_elec[i_sample] = 0.;

        for i_charge in 0..grid_n_charge {
            for j_charge in 0..grid_n_charge {
                for k_charge in 0..grid_n_charge {
                    // This will produce infinities due to 0 r.

                    let posit_charge = grid_posits_charge[i_charge][j_charge][k_charge];
                    let charge = charge_this_elec[i_charge][j_charge][k_charge];

                    V_from_this_elec[i_sample] += V_coulomb(posit_charge, posit_sample, charge);
                }
            }
        }
    }

    // println!("V creation complete");
}

/// Single-point Coulomb potential, eg a hydrogen nuclei.
pub(crate) fn V_coulomb(posit_charge: Vec3, posit_sample: Vec3, charge: f64) -> f64 {
    let diff = posit_sample - posit_charge;
    let r = diff.magnitude();

    if r < 0.0000000000001 {
        println!(
            "\nR lim. Charge: {:?}, sample: {:?}",
            posit_charge, posit_sample
        );
        return 0.; // todo: Is this the way to handle?
    }

    K_C * charge / r
}
