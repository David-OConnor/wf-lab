//! Code related to Hydrogen-like orbitals

use lin_alg::{complex_nums::Cplx, f64::Vec3};

use super::{SphericalHarmonic, A_0, Z_H};
use crate::util::{self, factorial};

/// A Hydrogen-atomic orbital. Note that unlike STOs, does not include an
/// effective charge.
/// todo: If this turns out to be teh same as an STO but with effective-
/// charge always equal to one, remove it.
#[derive(Clone, Debug)]
pub struct HOrbital {
    pub posit: Vec3,
    pub n: u16,
    pub harmonic: SphericalHarmonic,
    pub weight: f64,
    /// Somewhat degenerate with `posit`.
    pub charge_id: usize,
}

impl HOrbital {
    pub fn new(
        posit: Vec3,
        n: u16,
        harmonic: SphericalHarmonic,
        weight: f64,
        charge_id: usize,
    ) -> Self {
        assert!(harmonic.l < n);

        Self {
            posit,
            n,
            harmonic,
            weight,
            charge_id,
        }
    }

    /// Calculate the radial part of a basis function.
    /// We pass in `diff` and `r` to avoid duplicate calcs.
    /// https://chem.libretexts.org/Bookshelves/Inorganic_Chemistry/Map%3A_Inorganic_Chemistry_(Miessler_Fischer_Tarr)/
    /// 02%3A_Atomic_Structure/2.02%3A_The_Schrodinger_equation_particle_in_a_box_and_atomic_wavefunctions/2.2.02%3A_Quantum_Numbers_and_Atomic_Wave_Functions
    ///
    /// todo: That link above has errors! Ref this:
    /// https://chem.libretexts.org/Boo1kshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Map%3A_Physical_Chemistry_for_the_Biosciences_(Chang)
    /// /11%3A_Quantum_Mechanics_and_Atomic_Structure/11.10%3A_The_Schrodinger_Wave_Equation_for_the_Hydrogen_Atom
    /// [This ref](http://staff.ustc.edu.cn/~zqj/posts/Hydrogen-Wavefunction/) has a general equation
    /// at its top, separated into radial and angular parts.
    ///
    /// This reference seems accurate maybe?
    /// https://www.reed.edu/physics/courses/P342.S10/Physics342/page1/files/Lecture.24.pdf
    fn radial(&self, r: f64, l: u16) -> f64 {
        // todo: Once you've verified the general approach works, hard-code the first few values,
        // todo, and use the general approach for others.

        let n = self.n;
        let nf = n as f64;

        // https://docs.rs/scilib/latest/scilib/quantum/fn.radial_wavefunction.html
        // return scilib::quantum::radial_wavefunction(n.into(), l.into(), r);

        // Even though we normalize the combined wave function, we still need a norm term on individual
        // components here so the bases come out balanced relative to each other.
        let norm_term_num = (2. / (nf * A_0)).powi(3) * factorial(n - l - 1) as f64;
        let norm_term_denom = (2 * n as u64 * factorial(n + l).pow(3)) as f64;
        let norm_term = (norm_term_num / norm_term_denom).sqrt();

        // let L = Poly::laguerre((n - l - 1).into(), 2 * l + 1);
        let L = util::make_laguerre(n - l - 1, 2 * l + 1);

        return norm_term
            * (-r / (nf * A_0)).exp()
            * (2. * r / (nf * A_0)).powi(l.into())
            // * L.compute(2. * r / (nf * A_0));
            * L(2. * r / (nf * A_0));

        // Approach 2: Hard-coded for low n.

        // N is the normalization constant for the radial part
        let ρ = Z_H * r / A_0; // always r.
        let c = (Z_H * A_0).powf(3. / 2.); // always 1

        // Z is the charge of the nucleus.

        // function to solve for any n and l?
        // https://bingweb.binghamton.edu/~suzuki/QuantumMechanicsII/4-3_Radial-wave_function.pdf

        // todo: If this is correct, you can factor out (ZH/AO)^3/2 into its own part

        // todo: More abstractions based on rules?
        let part1 = match n {
            1 => 2.,
            2 => match l {
                0 => 2. * (2. - ρ),           // todo: z/a0 vice / 2a0?
                1 => 1. / 3.0_f64.sqrt() * ρ, // z/a0 vs /3a0?
                _ => panic!(),
            },
            // compare to this: https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Map%3A_Physical_Chemistry_for_the_Biosciences_(Chang)/11%3A_Quantum_Mechanics_and_Atomic_Structure/11.10%3A_The_Schrodinger_Wave_Equation_for_the_Hydrogen_Atom
            3 => match l {
                0 => 2. / 27. * (27. - 18. * ρ + 2. * ρ.powi(2)),
                1 => 1. / (81. * 3.0_f64.sqrt()) * (6. - ρ.powi(2)),
                2 => 1. / (81. * 15.0_f64.sqrt()) * ρ.powi(2),
                _ => panic!(),
            },
            _ => unimplemented!(), // todo: More
        };

        let part2 = (-ρ / n as f64).exp();

        part1 * part2 * c
    }

    /// Calculate this basis function's value at a given point.
    /// Does not include weight.
    /// todo: Is this normalized?
    /// https://quantummechanics.ucsd.edu/ph130a/130_notes/node233.html
    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        const EPS: f64 = 0.000000001;
        // todo: COnsider re-adding if aplicable to save computation, if you end up with lots of 0ed bases.
        // if self.weight.abs() < EPS {
        //     return Cplx::new_zero(); // saves some computation.
        // }

        let diff = posit_sample - self.posit;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        let radial = self.radial(r, self.harmonic.l);

        let diff = self.harmonic.orientation.inverse().rotate_vec(diff);

        // We use the "physics" (ISO) convention described here, where phi
        // covers the full way around, and theta goes half a turn from
        // the top.

        let θ = (diff.x.powi(2) + diff.y.powi(2)).sqrt().atan2(diff.z);
        let ϕ = diff.y.atan2(diff.x);

        let angular = self.harmonic.value(θ, ϕ);

        if angular.real.is_nan() {
            println!("Angular NAN: θ: {}, ϕ: {}", θ, ϕ);
        }

        // Normalization consts are applied to radial and angular parts separately.
        Cplx::from_real(radial) * angular
    }

    pub fn second_deriv(&self, posit_sample: Vec3) -> Cplx {
        // todo: Do this.
        Cplx::new_zero()
    }

    // pub fn psi_pp_div_psi(&self, posit_sample: Vec3) -> f64 {
    //     // todo: Do this.
    //     0.
    // }
}
