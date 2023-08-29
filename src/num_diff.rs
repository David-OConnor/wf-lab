//! Code for numerical finite-difference differentiation.

use lin_alg2::f64::Vec3;

use crate::{
    basis_wfs::Basis,
    complex_nums::Cplx,
    grid_setup::{Arr3d, Arr3dVec},
    interp,
};

// Used for calculating numerical psi''.
// Smaller is more precise. Too small might lead to numerical issues though (?)
// Applies to dx, dy, and dz
pub const H: f64 = 0.001;
pub const H_SQ: f64 = H * H;

/// Calcualte ψ'', numerically from ψ, using the finite diff method, for a single value.
/// Calculate ψ'' based on a numerical derivative of psi in 3D.
///
/// This solves, numerically, the eigenvalue equation for the Hamiltonian operator.
///
/// todo: This may replace the one below by using cached values of each wf at this point,
/// todo and neighbors.
pub(crate) fn find_ψ_pp_meas(
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

pub(crate) fn find_pp_real(
    psi_on_pt: f64,
    psi_x_prev: f64,
    psi_x_next: f64,
    psi_y_prev: f64,
    psi_y_next: f64,
    psi_z_prev: f64,
    psi_z_next: f64,
    h: f64,
) -> f64 {
    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi_on_pt * 6.;

    result / h
}

/// Calcualte ψ'', numerically from ψ, using the finite diff method, for a single value.
/// Calculate ψ'' based on a numerical derivative of psi in 3D.
///
/// This solves, numerically, the eigenvalue equation for the Hamiltonian operator.
pub(crate) fn _find_ψ_pp_meas_fm_bases(
    posit_sample: Vec3,
    bases: &[Basis],
    psi_sample_loc: Cplx,
    psi_norm_sqrt: f64,
    weights: Option<&[f64]>, // todo: This API eneds work.
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

    for (basis_i, basis) in bases.iter().enumerate() {
        let weight = match weights {
            Some(w) => w[basis_i],
            None => basis.weight(),
        };

        psi_x_prev += basis.value(x_prev) * weight;
        psi_x_next += basis.value(x_next) * weight;
        psi_y_prev += basis.value(y_prev) * weight;
        psi_y_next += basis.value(y_next) * weight;
        psi_z_prev += basis.value(z_prev) * weight;
        psi_z_next += basis.value(z_next) * weight;
    }

    psi_x_prev = psi_x_prev / psi_norm_sqrt;
    psi_x_next = psi_x_next / psi_norm_sqrt;
    psi_y_prev = psi_y_prev / psi_norm_sqrt;
    psi_y_next = psi_y_next / psi_norm_sqrt;
    psi_z_prev = psi_z_prev / psi_norm_sqrt;
    psi_z_next = psi_z_next / psi_norm_sqrt;

    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi_sample_loc * 6.;

    result / H_SQ
}

/// Find ψ'', numerically from ψ on an evenly-spaced-rectangular grid
///
/// This solves, numerically, the eigenvalue equation for the Hamiltonian operator.
pub(crate) fn _find_ψ_pp_meas_fm_grid_reg(
    psi: &Arr3d,
    psi_pp_measured: &mut Arr3d,
    dx_sq: f64,
    grid_n: usize,
) {
    // Note re these edge-cases: Hopefully it doesn't matter, since the WF is flat around
    // the edges, if the boundaries are chosen appropriately.
    for i in 0..grid_n {
        if i == 0 || i == grid_n - 1 {
            continue;
        }
        for j in 0..grid_n {
            if j == 0 || j == grid_n - 1 {
                continue;
            }
            for k in 0..grid_n {
                if k == 0 || k == grid_n - 1 {
                    continue;
                }

                let psi_x_prev = psi[i - 1][j][k];
                let psi_x_next = psi[i + 1][j][k];
                let psi_y_prev = psi[i][j - 1][k];
                let psi_y_next = psi[i][j + 1][k];
                let psi_z_prev = psi[i][j][k - 1];
                let psi_z_next = psi[i][j][k + 1];

                let psi_this = psi[i][j][k];

                let finite_diff =
                    psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
                        - psi_this * 6.;

                psi_pp_measured[i][j][k] = finite_diff / dx_sq;
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum _PsiPVar {
    Total,
    X,
    Y,
    Z,
}

/// Apply the momentum operator to a wave function.
/// At the edges and corners, does not mutate the value. Note that this is simpler than finding
/// psi'' due to being linear. (first deriv vice second)
///
/// Note: It makes sense that this is entirely imaginary for the l=0 wave functions,
/// since those don't have angular momentum. There's only [real?] angular momentum
/// if the WF has a complex component. Todo: What is the significance of imaginary momentum?
///
/// todo: Can we use this to estimate which L and M values might be used as basis fns?
///
/// This solves, numerically, the eigenvalue equation for the momentum operator.
/// todo: Calculated version we can compare to? Maybe sum of momentum of basis fns used?
/// todo: ie find_ψ_p_calc in `wf_ops`.
/// Pψ = pψ . -iħ ψ' = Pψ. ψ' = piħ ψ
pub fn _find_ψ_p_meas_fm_grid_irreg(
    psi: &Arr3d,
    result: &mut Arr3d,
    grid_posits: &Arr3dVec,
    var: _PsiPVar,
    grid_n: usize,
) {
    // This is `i * ħ`. Const expr limits avail syntax.
    // const COEFF: Cplx = Cplx { real: 0., im: -ħ };

    for i in 0..grid_n {
        if i == 0 || i == grid_n - 1 {
            continue;
        }
        for j in 0..grid_n {
            if j == 0 || j == grid_n - 1 {
                continue;
            }
            for k in 0..grid_n {
                if k == 0 || k == grid_n - 1 {
                    continue;
                }

                match var {
                    _PsiPVar::Total => {
                        let dx = grid_posits[i + 1][j][k].x - grid_posits[i - 1][j][k].x;
                        let dy = grid_posits[i][j + 1][k].y - grid_posits[i][j - 1][k].y;
                        let dz = grid_posits[i][j][k + 1].z - grid_posits[i][j][k - 1].z;

                        let d_psi_d_x = (psi[i + 1][j][k] - psi[i - 1][j][k]) / dx;
                        let d_psi_d_y = (psi[i][j + 1][k] - psi[i][j - 1][k]) / dy;
                        let d_psi_d_z = (psi[i][j][k + 1] - psi[i][j][k - 1]) / dz;

                        result[i][j][k] = d_psi_d_x + d_psi_d_y + d_psi_d_z;
                    }
                    _PsiPVar::X => {
                        let dx = grid_posits[i + 1][j][k].x - grid_posits[i - 1][j][k].x;
                        let d_psi_d_x = (psi[i + 1][j][k] - psi[i - 1][j][k]) / dx;

                        result[i][j][k] = d_psi_d_x;
                    }
                    _PsiPVar::Y => {
                        let dy = grid_posits[i][j + 1][k].y - grid_posits[i][j - 1][k].y;
                        let d_psi_d_y = (psi[i][j + 1][k] - psi[i][j - 1][k]) / dy;

                        result[i][j][k] = d_psi_d_y;
                    }
                    _PsiPVar::Z => {
                        let dz = grid_posits[i][j][k + 1].z - grid_posits[i][j][k - 1].z;
                        let d_psi_d_z = (psi[i][j][k + 1] - psi[i][j][k - 1]) / dz;

                        result[i][j][k] = d_psi_d_z;
                    }
                }

                // todo: Are we solving for psi' here, or -i hbar psi' ?
                // todo: Multiply by `COEFF` if finding momentum, vice only the derivative.
            }
        }
    }
}

/// Find ψ'', numerically from ψ on an unevenly-spaced-rectangular grid. This is a generalized
/// version of teh regular-grid version, but is more performance-intensive. Computes an order-2
/// (quadratic) polynomial given each point and its 2 neighbors, on each axis. Uses only the
/// squared term to find the second derivative at each point.
///
/// This solves, numerically, the eigenvalue equation for the Hamiltonian operator.
pub(crate) fn find_ψ_pp_meas_fm_grid_irreg(
    psi: &Arr3d,
    psi_pp_measured: &mut Arr3d,
    grid_posits: &Arr3dVec,
    grid_n: usize,
) {
    // Note re these edge-cases: Hopefully it doesn't matter, since the WF is flat around
    // the edges, if the boundaries are chosen appropriately.
    // for i in 0..grid_n {
    for i in 0..grid_n {
        if i == 0 || i == grid_n - 1 {
            continue;
        }
        for j in 0..grid_n {
            if j == 0 || j == grid_n - 1 {
                continue;
            }
            for k in 0..grid_n {
                if k == 0 || k == grid_n - 1 {
                    continue;
                }

                let psi_this = psi[i][j][k];

                let psi_x_prev = psi[i - 1][j][k];
                let psi_x_next = psi[i + 1][j][k];
                let psi_y_prev = psi[i][j - 1][k];
                let psi_y_next = psi[i][j + 1][k];
                let psi_z_prev = psi[i][j][k - 1];
                let psi_z_next = psi[i][j][k + 1];

                // `p` here is grid position.
                let p_this = grid_posits[i][j][k];

                let p_x_prev = grid_posits[i - 1][j][k];
                let p_x_next = grid_posits[i + 1][j][k];
                let p_y_prev = grid_posits[i][j - 1][k];
                let p_y_next = grid_posits[i][j + 1][k];
                let p_z_prev = grid_posits[i][j][k - 1];
                let p_z_next = grid_posits[i][j][k + 1];

                // Create a local polynomial in each axis.
                // todo: Should you use more than 3 points for better precision?
                let a_x_real = interp::create_quadratic_term(
                    (p_x_prev.x, psi_x_prev.real),
                    (p_this.x, psi_this.real),
                    (p_x_next.x, psi_x_next.real),
                );

                let a_y_real = interp::create_quadratic_term(
                    (p_y_prev.y, psi_y_prev.real),
                    (p_this.y, psi_this.real),
                    (p_y_next.y, psi_y_next.real),
                );

                let a_z_real = interp::create_quadratic_term(
                    (p_z_prev.z, psi_z_prev.real),
                    (p_this.z, psi_this.real),
                    (p_z_next.z, psi_z_next.real),
                );

                let a_x_im = interp::create_quadratic_term(
                    (p_x_prev.x, psi_x_prev.im),
                    (p_this.x, psi_this.im),
                    (p_x_next.x, psi_x_next.im),
                );

                let a_y_im = interp::create_quadratic_term(
                    (p_y_prev.y, psi_y_prev.im),
                    (p_this.y, psi_this.im),
                    (p_y_next.y, psi_y_next.im),
                );

                let a_z_im = interp::create_quadratic_term(
                    (p_z_prev.z, psi_z_prev.im),
                    (p_this.z, psi_this.im),
                    (p_z_next.z, psi_z_next.im),
                );

                let a_x = Cplx::new(a_x_real, a_x_im);
                let a_y = Cplx::new(a_y_real, a_y_im);
                let a_z = Cplx::new(a_z_real, a_z_im);

                // Combine the dimensions by adding, as we do for a regular grid. The factor of 2 is
                // part of the differentiation process.
                let finite_diff = Cplx::from_real(2.) * (a_x + a_y + a_z);

                psi_pp_measured[i][j][k] = finite_diff;
            }
        }
    }
}

/// Calcualte ψ'' measured, using a discrete function, interpolated.
/// Calculate ψ'' based on a numerical derivative of psi
/// in 3D.
pub(crate) fn _find_ψ_pp_meas_from_interp(
    posit_sample: Vec3,
    psi: &Arr3d,
    grid_min: f64,
    grid_max: f64,
    i: usize,
    j: usize,
    k: usize,
    grid_n: usize,
) -> Cplx {
    let grid_dx = (grid_max - grid_min) / grid_n as f64;

    // todo: This function is producing sub-optimal results when interpolating at other
    // todo than teh grid fn. You need a better algo, or don't use this.

    let h2 = grid_dx / 8.; // todo temp!!! Not working for values other than dx...

    let x_prev = Vec3::new(posit_sample.x - h2, posit_sample.y, posit_sample.z);
    let x_next = Vec3::new(posit_sample.x + h2, posit_sample.y, posit_sample.z);
    let y_prev = Vec3::new(posit_sample.x, posit_sample.y - h2, posit_sample.z);
    let y_next = Vec3::new(posit_sample.x, posit_sample.y + h2, posit_sample.z);
    let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - h2);
    let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + h2);

    let psi_x_prev = interp::linear_3d_cplx(
        x_prev,
        (posit_sample.x - grid_dx, posit_sample.x),
        (posit_sample.y, posit_sample.y + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.z, posit_sample.z + grid_dx), // On the edge; arbitrary which box we picked
        // todo: Coordinate system consistency? Does it matter here?
        psi[i - 1][j + 1][k + 1],
        psi[i - 1][j][k + 1],
        psi[i][j + 1][k + 1],
        psi[i][j][k + 1],
        psi[i - 1][j + 1][k],
        psi[i - 1][j][k],
        psi[i][j + 1][k],
        psi[i][j][k],
    );

    let psi_x_next = interp::linear_3d_cplx(
        x_next,
        (posit_sample.x, posit_sample.x + grid_dx),
        (posit_sample.y, posit_sample.y + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.z, posit_sample.z + grid_dx), // On the edge; arbitrary which box we picked
        psi[i][j + 1][k + 1],
        psi[i][j][k + 1],
        psi[i + 1][j + 1][k + 1],
        psi[i + 1][j][k + 1],
        psi[i][j + 1][k],
        psi[i][j][k],
        psi[i + 1][j + 1][k],
        psi[i + 1][j][k],
    );

    let psi_y_prev = interp::linear_3d_cplx(
        y_prev,
        (posit_sample.x, posit_sample.x + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.y - grid_dx, posit_sample.y),
        (posit_sample.z, posit_sample.z + grid_dx), // On the edge; arbitrary which box we picked
        psi[i][j][k + 1],
        psi[i][j - 1][k + 1],
        psi[i + 1][j][k + 1],
        psi[i + 1][j - 1][k + 1],
        psi[i][j][k],
        psi[i][j - 1][k],
        psi[i + 1][j][k],
        psi[i + 1][j - 1][k],
    );

    let psi_y_next = interp::linear_3d_cplx(
        y_next,
        (posit_sample.x, posit_sample.x + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.y, posit_sample.y + grid_dx),
        (posit_sample.z, posit_sample.z + grid_dx), // On the edge; arbitrary which box we picked
        psi[i][j + 1][k + 1],
        psi[i][j][k + 1],
        psi[i + 1][j + 1][k + 1],
        psi[i + 1][j][k + 1],
        psi[i][j + 1][k],
        psi[i][j][k],
        psi[i + 1][j + 1][k],
        psi[i + 1][j][k],
    );

    let psi_z_prev = interp::linear_3d_cplx(
        z_prev,
        (posit_sample.x, posit_sample.x + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.y, posit_sample.y + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.z - grid_dx, posit_sample.z),
        psi[i][j + 1][k],
        psi[i][j][k],
        psi[i + 1][j + 1][k],
        psi[i + 1][j][k],
        psi[i][j + 1][k - 1],
        psi[i][j][k - 1],
        psi[i + 1][j + 1][k - 1],
        psi[i + 1][j][k - 1],
    );

    let psi_z_next = interp::linear_3d_cplx(
        z_next,
        (posit_sample.x, posit_sample.x + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.y, posit_sample.y + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.z, posit_sample.z + grid_dx),
        psi[i][j + 1][k + 1],
        psi[i][j][k + 1],
        psi[i + 1][j + 1][k + 1],
        psi[i + 1][j][k + 1],
        psi[i][j + 1][k],
        psi[i][j][k - 1],
        psi[i + 1][j + 1][k],
        psi[i + 1][j][k],
    );

    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi[i][j][k] * 6.;

    // result / H_SQ
    result / (h2 * h2)
}
