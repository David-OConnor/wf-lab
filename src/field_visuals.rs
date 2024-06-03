//! Visualizations of the EM field, including vector fields and flux lines.

use lin_alg::f64::Vec3;

use crate::{
    grid_setup::{new_data_vec, Arr3dReal, Arr3dVec},
    iter_arr,
};

/// Generate a vectorfield of the gradient, from a charge density field. Note the convention of vectors
/// pointing towards positive charge, and away from negative charge.
/// todo: Custom type for type safety?
pub fn calc_gradient(charge_density: &Arr3dReal, grid: &Arr3dVec) -> Arr3dVec {
    let n = charge_density.len();

    // Assume even spacing for now. Adjust for a non-uniform grid, or remove
    // this in favor of a const if using analytic fns, A/R.
    let h_2 = (grid[2][0][0] - grid[0][0][0]).x;

    // todo: Combine from elecs and prots A/R here.

    // todo: We are starting out using the grid difference for now. Move to basis-based differences
    // todo as required: Much more accurate.

    // todo: Modify in place instead of creating a new array for result?
    let mut result = new_data_vec(n);

    for (i, j, k) in iter_arr!(n) {
        if i == 0 || i == n - 1 || j == 0 || j == n - 1 || k == 0 || k == n - 1 {
            continue;
        }

        // todo: Is this right? What is the quantity we are diffing? How does this work for
        // point charges like the nucleii?

        result[i][j][k] = Vec3::new(
            (charge_density[i + 1][j][k] - charge_density[i - 1][j][k]) / h_2,
            (charge_density[i][j + 1][k] - charge_density[i][j - 1][k]) / h_2,
            (charge_density[i][j][k + 1] - charge_density[i][j][k - 1]) / h_2,
        );
    }

    result
}
