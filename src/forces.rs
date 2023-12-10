//! Calculations of intra-molecular and inter-molecular forces. Initial use case:
//! estimating force on two hydrogen atoms in a molecule.

use lin_alg2::f64::Vec3;
use crate::grid_setup::Arr3dReal;
use crate::wf_ops::{K_C, Q_PROT};

/// Returns the force on nuc0, then nuc1.
pub(crate) fn h2_force(charge_elec0: &Arr3dReal, charge_elec1: &Arr3dReal, posit_nuc0: Vec3, posit_nuc1: Vec3) -> (Vec3, Vec3) {
    let nuc_dist = (posit_nuc1 - posit_nuc0).magnitude();

    // todo: QC order.
    /// Initialize to force from the other nucleus;
    let mut f0 = (posit_nuc1 - posit_nuc0) * K_C * Q_PROT * Q_PROT / nuc_dist.powi(2);
    let mut f1 = -f0;


    (f0, f1)
}