//! Calculations of intra-molecular and inter-molecular forces. Initial use case:
//! estimating force on two hydrogen atoms in a molecule.

use lin_alg::f64::Vec3;

use crate::{
    grid_setup::{Arr3dReal, Arr3dVec},
    iter_arr,
    wf_ops::{K_C, Q_ELEC, Q_PROT},
};


pub(crate) fn calc_force_on_nucs(
    nucs: &[(Vec3, f64)],
    charges_elecs: &[Arr3dReal],
    grid_charge: &Arr3dVec,
) -> Vec<Vec3> {
    let mut result = Vec::new();

    for (i, (nuc_posit, nuc_charge)) in nucs.iter().enumerate() {
        let mut f_on_this_nuc = Vec3::new_zero();

        // Calculate force from other nuclei.
        for (j, (nuc_posit_other, nuc_charge_other)) in nucs.iter().enumerate() {
            if i == j {
                continue
            }
            let posit_diff = *nuc_posit_other - *nuc_posit;
            let nuc_dist = posit_diff.magnitude();

            // Note: We factor out KC * Q_PROT, since they are present in every calculation.
            // Calculate the Coulomb force between nuclei.

            let f_mag = nuc_charge * nuc_charge_other / nuc_dist.powi(2);

            f_on_this_nuc += posit_diff * f_mag;
        }

        // Calculate force from electrons.

        // This variable is a component we can re-use, when calculating coulomb force.
        let f_elec_part = nuc_charge / grid_charge.len() as f64;

        for charge_elec in charges_elecs {
            // todo: GPU
            for (i, j, k) in iter_arr!(grid_charge.len()) {
                let posit_diff = grid_charge[i][j][Ik] - *nuc_posit;
                let dist_nuc = posit_diff.magnitude();

                let charge = charge_elec[i][j][k];

                // Force from both electroncs on nuc0.
                let f_mag = f_elec_part * charge / dist_nuc.powi(2);
                f_on_this_nuc += posit_diff * f_mag;
            }
        }
        result.push(f_on_this_nuc);
    }

    println!("Force on nucs: {:?}", result);
    result
}