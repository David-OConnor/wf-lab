//! Code related to Gaussian-Type Orbitals

use lin_alg::{complex_nums::Cplx, f64::Vec3};

use super::SphericalHarmonic;

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

    /// Calculate psi'' / psi, for a gaussian. QC this. Reference OneNote: Exploring The WF, part 12.
    pub fn psi_pp_div_psi(&self, posit_sample: Vec3) -> f64 {
        let diff = posit_sample - self.posit;

        -3. / self.c.powi(2) + (diff.x + diff.y + diff.z).powi(2) / self.c.powi(4)
    }

    /// Generate a gaussian to solve a V_acting_on - V_eigen difference
    /// Note: It appears that weight doesn't make a difference here...
    /// todo: But we have to take weight into account... Maybe due to how it balances
    /// todo with other bases?? Think this through.
    pub fn from_V_diff(V_acting_on: f64, V_eigen: f64, posit: Vec3) -> Self {
        // Hmm... -3. / c^2 = val with diff=0...

        // todo: QC the order.
        let diff = V_acting_on - V_eigen;

        let c = (-3. / diff).sqrt();

        let weight = 1.0;

        // todo: I think this overall approach is wrong, as you can't just add this
        // todo to the WF. You have to calculate what change to the WF would produce the
        // todo desired result.

        Self { posit, c, weight }
    }
}
