//! Visualizations of the EM field, including vector fields and flux lines.

use lin_alg::f64::Vec3;

use crate::{
    core_calcs::potential,
    grid_setup::{new_data_vec, Arr3dReal, Arr3dVec},
    iter_arr,
};
/// Calcualte the electric field on a grid.
pub(crate) fn calc_E_field(
    charge_elecs: &Arr3dReal,
    charge_nucs: &[(Vec3, f64)],
    grid_gradient: &Arr3dVec,
    grid_charge: &Arr3dVec,
) -> Arr3dVec {
    let n_gradient = grid_gradient.len();
    let n_charge = grid_charge.len();

    // todo: We are starting out using the grid difference for now. Move to basis-based differences
    // todo as required: Much more accurate.

    // todo: Modify in place instead of creating a new array for result?
    let mut result = new_data_vec(n_gradient);

    for (i, j, k) in iter_arr!(n_gradient) {
        let posit_sample = grid_gradient[i][j][k];

        // Calculate the electric field.
        let mut E = Vec3::new_zero();

        // Add electron charge.
        for (i_charge, j_charge, k_charge) in iter_arr!(n_charge) {
            let posit_charge = grid_charge[i_charge][j_charge][k_charge];
            let charge_elecs = charge_elecs[i_charge][j_charge][k_charge];

            let E_scalar = potential::E_coulomb(posit_charge, posit_sample, charge_elecs);
            E += (posit_sample - posit_charge) * E_scalar;
        }

        // Add nucleus charge.
        for (posit_nuc, charge_nuc) in charge_nucs {
            let E_scalar = potential::E_coulomb(*posit_nuc, posit_sample, *charge_nuc);
            E += (posit_sample - *posit_nuc) * E_scalar;
        }

        result[i][j][k] = E;
    }

    result
}

// /// Generate a vectorfield of the gradient, from a charge density field. Note the convention of vectors
// /// pointing towards positive charge, and away from negative charge.
// /// todo: Custom type for type safety?
// ///
// /// todo: Maybe it's the E field you want...
// pub fn calc_gradient(
//     charge_elecs: &Arr3dReal,
//     charge_nucs: &[(Vec3, f64)],
//     grid_gradient: &Arr3dVec,
//     grid_charge: &Arr3dVec,
// ) -> Arr3dJacobian {
//     let n_gradient = grid_gradient.len();
//     let n_charge = grid_charge.len();
//
//     // todo: We are starting out using the grid difference for now. Move to basis-based differences
//     // todo as required: Much more accurate.
//
//     let E_field = calc_E_field(charge_elecs, charge_nucs, grid_gradient, grid_charge);
//
//     // todo: Modify in place instead of creating a new array for result?
//     let mut result = new_data_vec(n_gradient);
//
//     // Assume even spacing for now. Adjust for a non-uniform grid, or remove
//     // this in favor of a const if using analytic fns, A/R.
//
//     // todo: If you use analytic functions, you can avoid diffing this coarse grid.
//     let h_2 = (grid_gradient[2][0][0] - grid_gradient[0][0][0]).x;
//
//     // Now, calculate the gradient.
//
//     for (i, j, k) in iter_arr!(n_gradient) {
//         if i == 0
//             || i == n_gradient - 1
//             || j == 0
//             || j == n_gradient - 1
//             || k == 0
//             || k == n_gradient - 1
//         {
//             continue;
//         }
//
//         result[i][j][k] = Vec3::new(
//             (E_field[i + 1][j][k] - E_field[i - 1][j][k]) / h_2,
//             (E_field[i][j + 1][k] - E_field[i][j - 1][k]) / h_2,
//             (E_field[i][j][k + 1] - E_field[i][j][k - 1]) / h_2,
//         );
//     }
//
//     result
// }

// /// Generate a vectorfield of the gradient, from a charge density field. Note the convention of vectors
// /// pointing towards positive charge, and away from negative charge.
// /// todo: Custom type for type safety?
// ///
// /// todo: Maybe it's the E field you want...
// pub fn calc_divergence(
//     charge_elecs: &Arr3dReal,
//     charge_nucs: &[(Vec3, f64)],
//     grid_gradient: &Arr3dVec,
//     grid_charge: &Arr3dVec,
// ) -> Arr3dVec {
//     let n_gradient = grid_gradient.len();
//     let n_charge = grid_charge.len();
//
//     // todo: We are starting out using the grid difference for now. Move to basis-based differences
//     // todo as required: Much more accurate.
//
//     let E_field = calc_E_field(charge_elecs, charge_nucs, grid_gradient, grid_charge);
//
//     // todo: Modify in place instead of creating a new array for result?
//     let mut result = new_data_vec(n_gradient);
//
//     // Assume even spacing for now. Adjust for a non-uniform grid, or remove
//     // this in favor of a const if using analytic fns, A/R.
//
//     // todo: If you use analytic functions, you can avoid diffing this coarse grid.
//     let h_2 = (grid_gradient[2][0][0] - grid_gradient[0][0][0]).x;
//
//     // Now, calculate the gradient.
//
//     for (i, j, k) in iter_arr!(n_gradient) {
//         if i == 0
//             || i == n_gradient - 1
//             || j == 0
//             || j == n_gradient - 1
//             || k == 0
//             || k == n_gradient - 1
//         {
//             continue;
//         }
//
//         result[i][j][k] = Vec3::new(
//             (E_field[i + 1][j][k] - E_field[i - 1][j][k]) / h_2,
//             (E_field[i][j + 1][k] - E_field[i][j - 1][k]) / h_2,
//             (E_field[i][j][k + 1] - E_field[i][j][k - 1]) / h_2,
//         );
//     }
//
//     result
// }
