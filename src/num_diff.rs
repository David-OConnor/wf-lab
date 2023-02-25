//! Code for numerical finite-difference differentiation.

use lin_alg2::f64::Vec3;

use crate::{
    basis_wfs::Basis,
    complex_nums::Cplx,
    interp,
    rbf::Rbf,
    util::{self, Arr3d, Arr3dBasis},
    wf_ops::N,
    Arr3dReal,
};

// Used for calculating numerical psi''.
// Smaller is more precise. Too small might lead to numerical issues though (?)
// Applies to dx, dy, and dz
const H: f64 = 0.01;
const H_SQ: f64 = H * H;

/// Calcualte ψ'', numerically from ψ, using the finite diff method, for a single value.
/// Calculate ψ'' based on a numerical derivative of psi in 3D.
pub(crate) fn find_ψ_pp_meas_fm_bases(
    posit_sample: Vec3,
    bases: &[Basis],
    psi_sample_loc: Cplx,
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
        psi_x_prev += basis.value(x_prev) * basis.weight();
        psi_x_next += basis.value(x_next) * basis.weight();
        psi_y_prev += basis.value(y_prev) * basis.weight();
        psi_y_next += basis.value(y_next) * basis.weight();
        psi_z_prev += basis.value(z_prev) * basis.weight();
        psi_z_next += basis.value(z_next) * basis.weight();
    }

    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi_sample_loc * 6.;

    result / H_SQ
}

/// Find ψ'', numerically from ψ on an evenly-spaced-rectangular grid
pub(crate) fn find_ψ_pp_meas_fm_grid_reg(
    psi: &Arr3d,
    psi_pp_measured: &mut Arr3d,
    grid_posits: &[f64],
    dx_sq: f64,
) {
    // Note re these edge-cases: Hopefully it doesn't matter, since the WF is flat around
    // the edges, if the boundaries are chosen appropriately.
    for i in 0..N {
        if i == 0 || i == N - 1 {
            continue;
        }
        for j in 0..N {
            if j == 0 || j == N - 1 {
                continue;
            }
            for k in 0..N {
                if k == 0 || k == N - 1 {
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

/// Find ψ'', numerically from ψ on an unevenly-spaced-rectangular grid. This is a generalized
/// version of teh regular-grid version, but is more performance-intensive.
pub(crate) fn find_ψ_pp_meas_fm_grid_irreg(
    psi: &Arr3d,
    psi_pp_measured: &mut Arr3d,
    grid_posits: &Arr3dReal,
) {
    // Note re these edge-cases: Hopefully it doesn't matter, since the WF is flat around
    // the edges, if the boundaries are chosen appropriately.
    // for i in 0..N {
    for i in 0..N {
        if i == 0 || i == N - 1 {
            continue;
        }
        for j in 0..N {
            if j == 0 || j == N - 1 {
                continue;
            }
            for k in 0..N {
                if k == 0 || k == N - 1 {
                    continue;
                }

                let psi_x_prev = psi[i - 1][j][k];
                let psi_x_next = psi[i + 1][j][k];
                let psi_y_prev = psi[i][j - 1][k];
                let psi_y_next = psi[i][j + 1][k];
                let psi_z_prev = psi[i][j][k - 1];
                let psi_z_next = psi[i][j][k + 1];

                let psi_this = psi[i][j][k];

                // `p` here is grid position.
                let p_this = grid_posits[i][j][k];

                let p_x_prev = grid_posits[i - 1][j][k];
                let p_x_next = grid_posits[i + 1][j][k];
                let p_y_prev = grid_posits[i][j - 1][k];
                let p_y_next = grid_posits[i][j + 1][k];
                let p_z_prev = grid_posits[i][j][k - 1];
                let p_z_next = grid_posits[i][j][k + 1];

                // todo: QC how you're mixing the 3 dimensinos. Just add?

                let num = (psi_x_next - psi_this) / (p_x_next - p_this)
                    - (psi_this - psi_x_prev) / (p_this - p_x_prev)
                    + (psi_y_next - psi_this) / (p_y_next - p_this)
                    - (psi_this - psi_y_prev) / (p_this - p_y_prev)
                    + (psi_z_next - psi_this) / (p_z_next - p_this)
                    - (psi_this - psi_z_prev) / (p_this - p_z_prev);

                let denom = 0.5 * (p_x_next - p_x_prev + p_y_next - p_x_next + p_z_next - p_z_prev);

                let finite_diff = num / denom;

                psi_pp_measured[i][j][k] = finite_diff;
            }
        }
    }
}

/// Calcualte ψ'' measured, using a discrete function, interpolated.
/// Calculate ψ'' based on a numerical derivative of psi
/// in 3D.
pub(crate) fn find_ψ_pp_meas_from_interp(
    posit_sample: Vec3,
    psi: &Arr3d,
    grid_min: f64,
    grid_max: f64,
    i: usize,
    j: usize,
    k: usize,
) -> Cplx {
    let grid_dx = (grid_max - grid_min) / N as f64;

    // todo: This function is producing sub-optimal results when interpolating at other
    // todo than teh grid fn. You need a better algo, or don't use this.

    let h2 = grid_dx / 8.; // todo temp!!! Not working for values other than dx...

    let x_prev = Vec3::new(posit_sample.x - h2, posit_sample.y, posit_sample.z);
    let x_next = Vec3::new(posit_sample.x + h2, posit_sample.y, posit_sample.z);
    let y_prev = Vec3::new(posit_sample.x, posit_sample.y - h2, posit_sample.z);
    let y_next = Vec3::new(posit_sample.x, posit_sample.y + h2, posit_sample.z);
    let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - h2);
    let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + h2);

    // Given the points we're sampling are along the grid lines, we can (conveniently! do 1D
    // interpolation here, vice 3D.

    // todo: I think this 1d simplification won't work. I think perhaps even your
    // todo overall 3d approach won't work. But proper 3D interp will. (?)

    // let psi_x_prev = interp::linear_1d_cplx(
    //     x_prev.x,
    //     (posit_sample.x - grid_dx, posit_sample.x),
    //     psi[i - 1][j][k],
    //     psi[i][j][k],
    // );
    //
    // let psi_x_next = interp::linear_1d_cplx(
    //     x_next.x,
    //     (posit_sample.x, posit_sample.x + grid_dx),
    //     psi[i][j][k],
    //     psi[i + 1][j][k],
    // );
    //
    // let psi_y_prev = interp::linear_1d_cplx(
    //     y_prev.y,
    //     (posit_sample.y - grid_dx, posit_sample.y),
    //     psi[i][j - 1][k],
    //     psi[i][j][k],
    // );
    //
    // let psi_y_next = interp::linear_1d_cplx(
    //     y_next.y,
    //     (posit_sample.y, posit_sample.y + grid_dx),
    //     psi[i][j][k],
    //     psi[i][j + 1][k],
    // );
    //
    // let psi_z_prev = interp::linear_1d_cplx(
    //     z_prev.z,
    //     (posit_sample.z - grid_dx, posit_sample.z),
    //     psi[i][j][k - 1],
    //     psi[i][j][k],
    // );
    //
    // let psi_z_next = interp::linear_1d_cplx(
    //     z_next.z,
    //     (posit_sample.z, posit_sample.z + grid_dx),
    //     psi[i][j][k],
    //     psi[i][j][k + 1],
    // );

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

/// Calcualte ψ'' measured, using a discrete function, interpolated.
/// Calculate ψ'' based on a numerical derivative of psi
/// in 3D.
pub(crate) fn find_ψ_pp_meas_fm_rbf(posit_sample: Vec3, psi_sample: Cplx, rbf: &Rbf) -> Cplx {
    let h2 = 0.01;

    let x_prev = Vec3::new(posit_sample.x - h2, posit_sample.y, posit_sample.z);
    let x_next = Vec3::new(posit_sample.x + h2, posit_sample.y, posit_sample.z);
    let y_prev = Vec3::new(posit_sample.x, posit_sample.y - h2, posit_sample.z);
    let y_next = Vec3::new(posit_sample.x, posit_sample.y + h2, posit_sample.z);
    let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - h2);
    let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + h2);

    let psi_x_prev = rbf.interp_point(x_prev);
    let psi_x_next = rbf.interp_point(x_next);
    let psi_y_prev = rbf.interp_point(y_prev);
    let psi_y_next = rbf.interp_point(y_next);
    let psi_z_prev = rbf.interp_point(z_prev);
    let psi_z_next = rbf.interp_point(z_next);

    // todo: real only for now.

    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi_sample.real * 6.;

    // result / H_SQ
    Cplx::from_real(result / (h2 * h2)) // todo real temp
}

/// Calcualte ψ'' measured, using our polynomial/sin/exp bases
pub(crate) fn find_ψ_pp_meas_from_interp2(
    posit_sample: Vec3,
    psi: &Arr3d,
    bases: &Arr3dBasis,
    grid_min: f64,
    grid_max: f64,
    i: usize,
    j: usize,
    k: usize,
) -> Cplx {
    let grid_dx = (grid_max - grid_min) / N as f64;

    // todo: For now, use only the basis function at the point, since we're using
    // todo small diffs from it. In the future, consider if you'd like to interpolate
    // todo from the basis-functions at neighboring points weighted by dist to each.

    let h2 = grid_dx / 10.; // todo temp!!! Not working for values other than dx...

    let h2 = 0.001;

    let x_prev = Vec3::new(posit_sample.x - h2, posit_sample.y, posit_sample.z);
    let x_next = Vec3::new(posit_sample.x + h2, posit_sample.y, posit_sample.z);
    let y_prev = Vec3::new(posit_sample.x, posit_sample.y - h2, posit_sample.z);
    let y_next = Vec3::new(posit_sample.x, posit_sample.y + h2, posit_sample.z);
    let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - h2);
    let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + h2);

    // todo, since our bases for now produce real vals

    let psi_x_prev = Cplx::from_real(bases[i][j][k].value(x_prev));
    let psi_x_next = Cplx::from_real(bases[i][j][k].value(x_next));
    let psi_y_prev = Cplx::from_real(bases[i][j][k].value(y_prev));
    let psi_y_next = Cplx::from_real(bases[i][j][k].value(y_next));
    let psi_z_prev = Cplx::from_real(bases[i][j][k].value(z_prev));
    let psi_z_next = Cplx::from_real(bases[i][j][k].value(z_next));

    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi[i][j][k] * 6.;

    // result / H_SQ
    result / (h2 * h2)
}
