//! Code related to Gaussian-Type Orbitals

use lin_alg::f64::Vec3;

use super::SphericalHarmonic;
use crate::complex_nums::Cplx;

/// A Gaussian. We are currently experimenting with this for use as a general
/// modifier to the wave function, vice for approximating atomic orbitals.
#[derive(Clone, Debug)]
pub struct Gaussian {
    // pub charge_id: usize,
    pub posit: Vec3,
    // Height
    // pub a: f64,
    /// Standard deviation.
    pub c: f64,
    pub weight: f64,
    // pub harmonic: SphericalHarmonic,
}

impl Gaussian {
    /// Calculate this basis function's value at a given point.
    /// Does not include weight.
    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        let diff = posit_sample - self.posit;
        let r_sq = diff.magnitude_squared();

        // todO: How should we handle the complex part?

        // println!("TEST r{:?} w{} c{} V{}",r_sq, self.weight, self.c, self.weight * (-r_sq / (2. * self.c.powi(2))).exp());

        Cplx::from_real(self.weight * (-r_sq / (2. * self.c.powi(2))).exp())
    }
}
