//! Contains code related to angular momentum eigenfunctions.
//! Code here, for now, is using natural units. (no explicit hbar etc)

use lin_alg2::f64::Vec3;

use crate::{
    complex_nums::{Cplx, IM},
    types::DerivativesSingle,
};

/// We choose z by convention; we need one of any dimension.
/// Uses the momentum eigenfunction p_a = -i hbar d/da
pub(crate) fn calc_l_z(posit: Vec3, d: &DerivativesSingle) -> Cplx {
    -IM * (d.dy * posit.x - d.dx * posit.y)
}

/// Calculate L^2, given derivatives. Used in one of the two momentum eigenfunctions. See Onenote: Exploring the WF, part 9
pub(crate) fn calc_l_sq(posit: Vec3, d: &DerivativesSingle) -> Cplx {
    let x = posit.x;
    let y = posit.y;
    let z = posit.z;

    let part0 =
        (d.d2y + d.d2z) * -x.powi(2) - (d.d2x + d.d2z) * y.powi(2) - (d.d2x + d.d2y) * z.powi(2);

    let part1 = (d.dx * d.dy * x * y + d.dx * d.dz * x * z + d.dy * d.dz * y * z) * 2.;

    part0 + part1
}
