//! Code for numerical finite-difference differentiation. We use this primarily to calculate
//! Ψ'', which is a component of the Schrodinger equation. We use analytic second derivatives
//! when available, and numerical ones when not.

use lin_alg2::f64::Vec3;

use crate::{
    basis_wfs::Basis,
    complex_nums::Cplx,
    grid_setup::{new_data, Arr3d},
    iter_arr,
};

// Used for calculating numerical ψ''.
// Smaller is more accurate. Too small might lead to numerical issues though (?)
// Applies to dx, dy, and dz
pub const H: f64 = 0.01;
pub const H_SQ: f64 = H * H;

/// Calcualte ψ'', numerically from ψ, using the finite diff method, for a single value.
/// Calculate ψ'' based on a numerical derivative of psi in 3D.
///
/// This solves, numerically, the eigenvalue equation for the Hamiltonian operator.
///
/// todo: This may replace the one below by using cached values of each wf at this point,
/// todo and neighbors.
pub(crate) fn _find_ψ_pp_meas(
    // todo: Combine these into a single struct a/r
    psi_on_pt: Cplx,
    psi_x_prev: Cplx,
    psi_x_next: Cplx,
    psi_y_prev: Cplx,
    psi_y_next: Cplx,
    psi_z_prev: Cplx,
    psi_z_next: Cplx,
) -> Cplx {
    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi_on_pt * 6.;

    result / H_SQ
}

pub(crate) fn _find_pp_real(
    psi_on_pt: f64,
    psi_x_prev: f64,
    psi_x_next: f64,
    psi_y_prev: f64,
    psi_y_next: f64,
    psi_z_prev: f64,
    psi_z_next: f64,
) -> f64 {
    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi_on_pt * 6.;

    result / H_SQ
}

/// Calcualte ψ'', numerically from ψ, using the finite diff method, for a single value.
/// Calculate ψ'' based on a numerical derivative of psi in 3D.
///
/// This solves, numerically, the eigenvalue equation for the Hamiltonian operator.
pub(crate) fn find_ψ_pp_num_fm_bases(
    posit_sample: Vec3,
    bases: &[Basis],
    // We pass this as an argument since it's likely already been calculated.
    ψ_sample_loc: Cplx,
) -> Cplx {
    let x_prev = Vec3::new(posit_sample.x - H, posit_sample.y, posit_sample.z);
    let x_next = Vec3::new(posit_sample.x + H, posit_sample.y, posit_sample.z);
    let y_prev = Vec3::new(posit_sample.x, posit_sample.y - H, posit_sample.z);
    let y_next = Vec3::new(posit_sample.x, posit_sample.y + H, posit_sample.z);
    let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - H);
    let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + H);

    let mut psi_x_prev = Cplx::new_zero();
    let mut psi_x_next = Cplx::new_zero();
    let mut psi_y_prev = Cplx::new_zero();
    let mut psi_y_next = Cplx::new_zero();
    let mut psi_z_prev = Cplx::new_zero();
    let mut psi_z_next = Cplx::new_zero();

    for basis in bases {
        psi_x_prev += basis.value(x_prev);
        psi_x_next += basis.value(x_next);
        psi_y_prev += basis.value(y_prev);
        psi_y_next += basis.value(y_next);
        psi_z_prev += basis.value(z_prev);
        psi_z_next += basis.value(z_next);
    }

    // Note: We currently handle norm downstream.

    (psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - ψ_sample_loc * 6.)
        / H_SQ
}

/// Differentiate a 3D complex array. Note that this may exhibit numerical inaccuracies due to the large
/// difference between grid points. Call this in series to calculate higher derivatives.
/// todo: This is a second deriv; not first!
pub(crate) fn differentiate_grid(
    data: &Arr3d,
    index: (usize, usize, usize),
    grid_spacing: f64, // ie H.
) -> Cplx {
    let n = data.len();
    let (i, j, k) = index;

    // We are unable to calculate values at the grid edge using this approach.
    if i == 0 || i == n - 1 || j == 0 || j == n - 1 || k == 0 || k == n - 1 {
        return Cplx::new_zero();
    }

    let sample_loc = data[i][j][k];

    let x_prev = data[i - 1][j][k];
    let x_next = data[i + 1][j][k];
    let y_prev = data[i][j - 1][k];
    let y_next = data[i][j + 1][k];
    let z_prev = data[i][j][k - 1];
    let z_next = data[i][j][k + 1];
    // Note: We currently handle norm downstream.

    (x_prev + x_next + y_prev + y_next + z_prev + z_next - sample_loc * 6.) / grid_spacing.powi(2)
}

pub(crate) fn differentiate_grid_all(data: &Arr3d, grid_spacing: f64) -> Arr3d {
    let n = data.len();
    let mut result = new_data(n);

    for (i, j, k) in iter_arr!(n) {
        result[i][j][k] = differentiate_grid(data, (i, j, k), grid_spacing);
    }

    result
}
