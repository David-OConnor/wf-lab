//! Code for numerical finite-difference differentiation. We use this primarily to calculate
//! Ψ'', which is a component of the Schrodinger equation. We use analytic second derivatives
//! when available, and numerical ones when not.

use lin_alg2::f64::Vec3;

use crate::{
    basis_wfs::Basis,
    complex_nums::Cplx,
    grid_setup::{new_data, Arr3d},
    iter_arr,
    types::{Derivatives, DerivativesSingle},
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

/// Differentiate a 3D complex array. Total derivative. (dx + dy + dz) Note that this may exhibit numerical inaccuracies due to the large
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

// todo: Similar fn from analytic bases.
impl Derivatives {
    /// Create a derivatives struct using numeric derivatives of a wave function
    pub fn numeric_fm_psi(psi: &Arr3d, grid_spacing: f64) -> Self {
        let n = psi.len();

        // For usue with midpoint first derivatives.
        let grid_spacing_sq = grid_spacing.powi(2);
        let mid_pt_diff = grid_spacing * 2.;

        let mut result = Self::new(n);

        for (i, j, k) in iter_arr!(n) {
            // We are unable to calculate values at the grid edge using this approach.
            if i == 0 || i == n - 1 || j == 0 || j == n - 1 || k == 0 || k == n - 1 {
                continue;
            }

            let on_pt = psi[i][j][k];
            let x_prev = psi[i - 1][j][k];
            let x_next = psi[i + 1][j][k];
            let y_prev = psi[i][j - 1][k];
            let y_next = psi[i][j + 1][k];
            let z_prev = psi[i][j][k - 1];
            let z_next = psi[i][j][k + 1];

            result.dx[i][j][k] = (x_next - x_prev) / mid_pt_diff;
            result.dy[i][j][k] = (y_next - y_prev) / mid_pt_diff;
            result.dz[i][j][k] = (z_next - z_prev) / mid_pt_diff;

            result.d2x[i][j][k] = (x_next + x_prev - on_pt * 2.) / grid_spacing_sq;
            result.d2y[i][j][k] = (y_next + y_prev - on_pt * 2.) / grid_spacing_sq;
            result.d2z[i][j][k] = (z_next + z_prev - on_pt * 2.) / grid_spacing_sq;

            result.d2_sum[i][j][k] =
                result.d2x[i][j][k] + result.d2y[i][j][k] + result.d2z[i][j][k];
        }

        result
    }
}
// todo: combine above adn this A/R. Vec of struct vs struct of vec dilemma too.

// todo: Similar fn from analytic bases.
impl DerivativesSingle {
    /// Create a derivativesSingle struct using numeric derivatives of a wave function
    pub fn numeric_fm_psi(psi: &Arr3d, index: (usize, usize, usize), grid_spacing: f64) -> Self {
        let n = psi.len();
        let (i, j, k) = index;

        // For usue with midpoint first derivatives.
        let grid_spacing_sq = grid_spacing.powi(2);
        let mid_pt_diff = grid_spacing * 2.;

        let mut result = Self::default();

        // We are unable to calculate values at the grid edge using this approach.
        if i == 0 || i == n - 1 || j == 0 || j == n - 1 || k == 0 || k == n - 1 {
            return result;
        }

        let on_pt = psi[i][j][k];
        let x_prev = psi[i - 1][j][k];
        let x_next = psi[i + 1][j][k];
        let y_prev = psi[i][j - 1][k];
        let y_next = psi[i][j + 1][k];
        let z_prev = psi[i][j][k - 1];
        let z_next = psi[i][j][k + 1];

        result.dx = (x_next - x_prev) / mid_pt_diff;
        result.dy = (y_next - y_prev) / mid_pt_diff;
        result.dz = (z_next - z_prev) / mid_pt_diff;

        result.d2x = (x_next + x_prev - on_pt * 2.) / grid_spacing_sq;
        result.d2y = (y_next + y_prev - on_pt * 2.) / grid_spacing_sq;
        result.d2z = (z_next + z_prev - on_pt * 2.) / grid_spacing_sq;

        result.d2_sum = result.d2x + result.d2y + result.d2z;

        result
    }
}
