//! Calculations of intra-molecular and inter-molecular forces. Initial use case:
//! estimating force on two hydrogen atoms in a molecule.

use lin_alg2::f64::Vec3;

use crate::{
    grid_setup::{Arr3dReal, Arr3dVec},
    iter_arr,
    wf_ops::{K_C, Q_ELEC, Q_PROT},
};

/// Returns the force on nuc0, then nuc1.
pub(crate) fn h2_force(
    charge_elec0: &Arr3dReal,
    charge_elec1: &Arr3dReal,
    posit_nuc0: Vec3,
    posit_nuc1: Vec3,
    grid_charge: &Arr3dVec,
) -> (Vec3, Vec3) {
    let diff_nuc = posit_nuc1 - posit_nuc0;
    let nuc_dist = diff_nuc.magnitude();

    let q_elec_part = Q_ELEC / grid_charge.len() as f64;

    // todo: QC order.
    // Initialize to force from the other nucleus;
    // Note: We factor out KC * Q_PROT, since they are present in every calculation.
    let mut f0 = diff_nuc * Q_PROT / nuc_dist.powi(2);
    let mut f1 = -f0;

    // Calculate the combined electron force on the nucleus.
    // todo: GPU
    for (i, j, k) in iter_arr!(grid_charge.len()) {
        let diff_elec_nuc0 = grid_charge[i][j][k] - posit_nuc0;
        let diff_elec_nuc1 = grid_charge[i][j][k] - posit_nuc1;
        let nuc_dist0 = diff_elec_nuc0.magnitude();
        let nuc_dist1 = diff_elec_nuc1.magnitude();

        let charge0 = charge_elec0[i][j][k];
        let charge1 = charge_elec1[i][j][k];

        // Force from both electroncs on nuc0.
        f0 += diff_elec_nuc0 * charge0 * q_elec_part / nuc_dist0;
        f0 += diff_elec_nuc0 * charge1 * q_elec_part / nuc_dist0;

        // Force from both electroncs on nuc1.
        f1 += diff_elec_nuc1 * charge0 * q_elec_part / nuc_dist1;
        f1 += diff_elec_nuc1 * charge1 * q_elec_part / nuc_dist1;
    }

    (f0 * K_C * Q_PROT, f1 * K_C * Q_PROT)
}
