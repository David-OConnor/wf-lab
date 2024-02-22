//! Code related to Gaussian-Type Orbitals

use lin_alg::f64::Vec3;

use super::SphericalHarmonic;
use crate::complex_nums::Cplx;

/// A Slater-Type Orbital (STO). Includes a `n`: The quantum number; the effective
/// charge slater exponent (ζ) may be used to simulate "effective charge", which
/// can represent "electron shielding".(?)
/// At described in *Computational Physics* by T+J.
#[derive(Clone, Debug)]
pub struct Gto {
    pub posit: Vec3,
    pub alpha: f64,
    pub weight: f64,
    pub charge_id: usize,
    pub harmonic: SphericalHarmonic,
}

impl Gto {
    /// Calculate this basis function's value at a given point.
    /// Does not include weight.
    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        let diff = posit_sample - self.posit;
        let r_sq = diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2);

        Cplx::from_real((-self.alpha * r_sq).exp())
    }
}
