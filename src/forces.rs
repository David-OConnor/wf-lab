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
                continue;
            }
            let posit_diff = *nuc_posit_other - *nuc_posit;
            let dist = posit_diff.magnitude();

            let posit_diff_unit = posit_diff / dist;

            // Note: We factor out KC * Q_PROT, since they are present in every calculation.
            // Calculate the Coulomb force between nuclei.

            let f_mag = nuc_charge * nuc_charge_other / dist.powi(2);

            f_on_this_nuc += posit_diff_unit * f_mag;
        }

        println!("Repulsion from other nucs: {:?}", f_on_this_nuc);

        // Calculate force from electrons.

        // This variable is a component we can re-use, when calculating coulomb force.
        // let f_elec_part = nuc_charge / grid_charge.len().pow(3) as f64;
        let f_elec_part = nuc_charge;

        for charge_elec in charges_elecs {
            // todo: GPU

            let mut elec_f = Vec3::new_zero(); // todo: Temp var for debug prints
            for (i, j, k) in iter_arr!(grid_charge.len()) {
                let posit_diff = grid_charge[i][j][k] - *nuc_posit;
                let dist = posit_diff.magnitude();

                let posit_diff_unit = posit_diff / dist;

                let charge = charge_elec[i][j][k];

                // Force from both electroncs on nuc0.
                let f_mag = f_elec_part * charge / dist.powi(2);
                f_on_this_nuc += posit_diff_unit * f_mag;

                elec_f += posit_diff_unit * f_mag;
            }
            println!("Attraction from this elec: {:?}", elec_f);
        }
        result.push(f_on_this_nuc);
    }

    println!("Force on nucs: {:?}", result);
    result
}
