//! Contains code related to calculating the results of the basic eigenfunctions (H, L^2, L_z)
//! on ψ.

use lin_alg::{f64::Vec3, complex_nums::{Cplx, IM}};

use crate::{core_calcs::eigen_fns::KE_COEFF, types::DerivativesSingle};

/// Calculate H; the hamiltonian (Energy eigenfunction). H = (h^2/2m ∇^2 + V) ψ
pub(crate) fn calc_H(psi: Cplx, psi_pp: Cplx, V: f64) -> Cplx {
    // todo: why - V? Probably same as the "What's going on?" comment above. Maybe your V is inverted?
    psi_pp * KE_COEFF - psi * V
}

/// Calculate L^2, given derivatives. Used in one of the two momentum eigenfunctions. See Onenote:
/// Exploring the WF, part 9
/// todo: This is coming out incorrectly. Note that the `l_z` function above appears to work correctly.
pub(crate) fn calc_L_sq(posit: Vec3, d: &DerivativesSingle) -> Cplx {
    let x = posit.x;
    let y = posit.y;
    let z = posit.z;

    // todo: This is fine too; just a different representation.
    // let part0 =
    //     (d.d2y + d.d2z) * x.powi(2) + (d.d2x + d.d2z) * y.powi(2) + (d.d2x + d.d2y) * z.powi(2);

    // todo: This should be equivalent, and it seems to be so.
    let x_sq = x.powi(2);
    let y_sq = y.powi(2);
    let z_sq = z.powi(2);

    let part0 = d.d2z * (x_sq + y_sq) + d.d2x * (y_sq + z_sq) + d.d2y * (z_sq + x_sq);

    // // todo: Experimenting...
    // let l_sq_sq = part0.abs_sq();
    // return Cplx::from_real(l_sq_sq);

    // todo: This is wrong (Multiplying dpsi/dx * dpsi/dy. But shouldn't this come out to 0, so it doesn't matter??
    // let part1 = (d.dx * d.dy * x * y + d.dx * d.dz * x * z + d.dy * d.dz * y * z) * 2.;

    // todo: Guess
    let d_dydz = d.dy * d.dz;
    let d_dzdx = d.dz * d.dx;
    let d_dxdy = d.dx * d.dy;

    // Todo: My best guess: We need the cross terms. Ref ChatGpt. YOu need d^2/(dx dz) and d^2/(dzdx)
    let part1 = (d_dydz * y * z + d_dzdx * z * x + d_dxdy * x * y) * 2.;
    // println!("Part 1:  {:?}", part1); // todo: part1 0 from cancelling cross terms? Appears to be.

    // todo: Maybe you need to take the derivatives in series, instead of multiplying? Yea.
    // todo: dx * dx isn't (dpsi/dx)^2... it's the second derivative.

    // TS note: Part 1 seems to be 0 for several test cases. (Cross-deriv terms cancelling, likely)

    -part0
    // -part0 + part1
}

/// We choose z by convention; we need one of any dimension.
/// Uses the momentum eigenfunction p_a = -i ħ d/da. L_z = -iħ (x d/dy - y d/dx)
pub(crate) fn calc_L_z(posit: Vec3, d: &DerivativesSingle) -> Cplx {
    -IM * (d.dy * posit.x - d.dx * posit.y)
}

/// See `calc_L_z`.
pub(crate) fn calc_L_x(posit: Vec3, d: &DerivativesSingle) -> Cplx {
    -IM * (d.dz * posit.y - d.dy * posit.z)
}

/// See `calc_L_z`.
pub(crate) fn calc_L_y(posit: Vec3, d: &DerivativesSingle) -> Cplx {
    -IM * (d.dx * posit.z - d.dz * posit.x)
}
