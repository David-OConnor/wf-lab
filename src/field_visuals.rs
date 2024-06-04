//! Visualizations of the EM field, including vector fields and flux lines.

use lin_alg::f64::Vec3;

use crate::{
    grid_setup::{new_data_vec, Arr3dReal, Arr3dVec},
    iter_arr, potential,
    wf_ops::Q_ELEC,
};

/// Generate a vectorfield of the gradient, from a charge density field. Note the convention of vectors
/// pointing towards positive charge, and away from negative charge.
/// todo: Custom type for type safety?
pub fn calc_gradient(
    charge_elecs: &Arr3dReal,
    charge_nucs: &[(Vec3, f64)],
    grid_gradient: &Arr3dVec,
    grid_charge: &Arr3dVec,
) -> Arr3dVec {
    let n_gradient = grid_gradient.len();
    let n_charge = grid_charge.len();

    // Assume even spacing for now. Adjust for a non-uniform grid, or remove
    // this in favor of a const if using analytic fns, A/R.
    let h_2 = (grid_gradient[2][0][0] - grid_gradient[0][0][0]).x;

    // todo: Combine from elecs and prots A/R here.

    // todo: We are starting out using the grid difference for now. Move to basis-based differences
    // todo as required: Much more accurate.

    // todo: Modify in place instead of creating a new array for result?
    let mut result = new_data_vec(n_gradient);

    for (i, j, k) in iter_arr!(n_gradient) {
        if i == 0
            || i == n_charge - 1
            || j == 0
            || j == n_gradient - 1
            || k == 0
            || k == n_gradient - 1
        {
            continue;
        }
        let posit_sample = grid_gradient[i][j][k];

        let mut E = Vec3::new_zero();

        // Add electron charge.
        for (i_charge, j_charge, k_charge) in iter_arr!(n_charge) {
            let posit_charge = grid_charge[i_charge][j_charge][k_charge];
            let charge_elecs = charge_elecs[i_charge][j_charge][k_charge];

            let E_scalar = potential::E_coulomb(posit_charge, posit_sample, charge_elecs);
            E += (posit_charge - posit_sample) * E_scalar;
        }

        // Add nucleus charge.
        for (posit_nuc, charge_nuc) in charge_nucs {
            let E_scalar = potential::E_coulomb(*posit_nuc, posit_sample, *charge_nuc);
            E += (*posit_nuc - posit_sample) * E_scalar;
        }

        // todo: Is this right? What is the quantity we are diffing? How does this work
        // todo point charges like the nucleii? Maybe we need to, for each point, calculate
        // todo a value based on Coulomb's law? Likely!

        result[i][j][k] = Vec3::new(
            (charge_elecs[i + 1][j][k] - charge_elecs[i - 1][j][k]) / h_2,
            (charge_elecs[i][j + 1][k] - charge_elecs[i][j - 1][k]) / h_2,
            (charge_elecs[i][j][k + 1] - charge_elecs[i][j][k - 1]) / h_2,
        );
    }

    result
}
