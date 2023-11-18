//! This module contains basis wave functions, which we combine in a LCAO
//! approach. For example, various Hydrogen orbitals.
//!
//! For calculated analytic H wave fns:
//! https://chem.libretexts.org/Courses/University_of_California_Davis/
//! UCD_Chem_107B%3A_Physical_Chemistry_for_Life_Scientists/Chapters/4%3A_Quantum_Theory/
//! 4.10%3A_The_Schr%C3%B6dinger_Wave_Equation_for_the_Hydrogen_Atom

// Some thoughts on directions to take (11 Mar 2023)
// - Determine how to add multiple basis fns (eg STOs or gauassians) to better approximate a WF,
// eg over a minimal basis set
// - Look into electron-electron interactions. Pauli exlcusion, (Is it effectively  a force??),
// - multiple electrons, fermions not stacking, spin etc.
// - second derivative term only in WF -- Is there a tie in to general relatively and gravity?
// - Metric (or other) tensor per point in space?

use std::f64::consts::PI;

use scilib::{self, math::polynomial::Poly};
// todo: There's also a WIP scilib Quantum lib that can do these H orbital calculations
// todo directly.

use lin_alg2::f64::{Quaternion, Vec3};

use crate::{
    complex_nums::{Cplx, IM},
    eigen_fns::KE_COEFF,
    util::{self, factorial},
};

// Hartree units.
const A_0: f64 = 1.;
const Z_H: f64 = 1.;

const PI_SQRT_INV: f64 = 0.5641895835477563;

// todo: Remove this enum if you use STOs as the only basis
#[derive(Clone, Debug)]
pub enum Basis {
    H(HOrbital),
    Gto(Gto),
    Sto(Sto),
}

impl Basis {
    /// These getters and setters allow access to common values (all but slater weight) without unpacking.
    pub fn posit(&self) -> Vec3 {
        match self {
            // Self::Sto(v) => v.posit,
            Self::H(v) => v.posit,
            Self::Gto(v) => v.posit,
            Self::Sto(v) => v.posit,
        }
    }

    pub fn posit_mut(&mut self) -> &mut Vec3 {
        match self {
            // Self::Sto(v) => &mut v.posit,
            Self::H(v) => &mut v.posit,
            Self::Gto(v) => &mut v.posit,
            Self::Sto(v) => &mut v.posit,
        }
    }

    pub fn n(&self) -> u16 {
        match self {
            // Self::Sto(v) => v.n,
            Self::H(v) => v.n,
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.n,
        }
    }

    pub fn n_mut(&mut self) -> &mut u16 {
        match self {
            // Self::Sto(v) => &mut v.n,
            Self::H(v) => &mut v.n,
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => &mut v.n,
        }
    }

    pub fn l(&self) -> u16 {
        match self {
            // Self::Sto(v) => v.harmonic.l,
            Self::H(v) => v.harmonic.l,
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(_v) => unimplemented!(),
        }
    }

    pub fn l_mut(&mut self) -> &mut u16 {
        match self {
            // Self::Sto(v) => &mut v.harmonic.l,
            Self::H(v) => &mut v.harmonic.l,
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(_v) => unimplemented!(),
        }
    }

    pub fn m(&self) -> i16 {
        match self {
            // Self::Sto(v) => v.harmonic.m,
            Self::H(v) => v.harmonic.m,
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(_v) => unimplemented!(),
        }
    }

    pub fn m_mut(&mut self) -> &mut i16 {
        match self {
            // Self::Sto(v) => &mut v.harmonic.m,
            Self::H(v) => &mut v.harmonic.m,
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(_v) => unimplemented!(),
        }
    }

    pub fn harmonic(&self) -> &SphericalHarmonic {
        match self {
            // Self::Sto(v) => &v.harmonic,
            Self::H(v) => &v.harmonic,
            Self::Gto(v) => &v.harmonic,
            Self::Sto(v) => &v.harmonic,
        }
    }

    pub fn harmonic_mut(&mut self) -> &mut SphericalHarmonic {
        match self {
            // Self::Sto(v) => &mut v.harmonic,
            Self::H(v) => &mut v.harmonic,
            Self::Gto(v) => &mut v.harmonic,
            Self::Sto(v) => &mut v.harmonic,
        }
    }

    pub fn weight(&self) -> f64 {
        match self {
            // Self::Sto(v) => v.weight,
            Self::H(v) => v.weight,
            Self::Gto(v) => v.weight,
            Self::Sto(v) => v.weight,
        }
    }

    pub fn xi(&self) -> f64 {
        match self {
            // Self::Sto(v) => v.weight,
            Self::H(v) => unimplemented!(),
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.xi,
        }
    }

    pub fn weight_mut(&mut self) -> &mut f64 {
        match self {
            // Self::Sto(v) => &mut v.weight,
            Self::H(v) => &mut v.weight,
            Self::Gto(v) => &mut v.weight,
            Self::Sto(v) => &mut v.weight,
        }
    }

    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        match self {
            // Self::Sto(v) => v.value(posit_sample),
            Self::H(v) => v.value(posit_sample),
            Self::Gto(v) => v.value(posit_sample),
            // Self::Sto(v) => v.value(posit_sample),
            Self::Sto(v) => v.value(posit_sample),
        }
    }

    /// Note: We must normalize afterwards.
    pub fn second_deriv(&self, posit_sample: Vec3) -> Cplx {
        match self {
            Self::Sto(v) => v.second_deriv(posit_sample),
            Self::H(v) => v.second_deriv(posit_sample),
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.second_deriv(posit_sample),
        }
    }

    pub fn psi_pp_div_psi(&self, posit_sample: Vec3) -> f64 {
        match self {
            Self::H(v) => v.psi_pp_div_psi(posit_sample),
            Self::Gto(v) => unimplemented!(),
            Self::Sto(v) => v.psi_pp_div_psi(posit_sample),
        }
    }

    pub fn V_p_from_psi(&self, posit_sample: Vec3) -> f64 {
        match self {
            Self::H(_v) => 0.,
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.V_p_from_psi(posit_sample),
        }
    }

    pub fn V_pp_from_psi(&self, posit_sample: Vec3) -> f64 {
        match self {
            Self::H(_v) => 0.,
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.V_pp_from_psi(posit_sample),
        }
    }

    pub fn charge_id(&self) -> usize {
        match self {
            // Self::Sto(v) => v.charge_id,
            Self::H(v) => v.charge_id,
            Self::Gto(v) => v.charge_id,
            Self::Sto(v) => v.charge_id,
        }
    }

    pub fn charge_id_mut(&mut self) -> &mut usize {
        match self {
            // Self::Sto(v) => &mut v.charge_id,
            Self::H(v) => &mut v.charge_id,
            Self::Gto(v) => &mut v.charge_id,
            Self::Sto(v) => &mut v.charge_id,
        }
    }

    pub fn descrip(&self) -> String {
        match self {
            // Self::Sto(_) => "STO",
            Self::H(_) => "H",
            Self::Gto(_) => "SO1",
            Self::Sto(_) => "SO2",
        }
        .to_owned()
    }
}

/// Represents a spherical harmonic, at a given l quantum numbers.
/// `l` represents the shape of the orbital.
/// orientation, we use a quaternion.
///
/// https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Spherical_harmonics
///
/// Todo: Real vs complex spherical harmonics.
/// For now, represents a complex spherical harmonic.
///
/// The base orientation is... // todo
#[derive(Clone, Debug)]
pub struct SphericalHarmonic {
    /// todo: Should we merge this back into `Basis`? Given l is used in the radial component calculation
    /// The quantum number the describes orbital shape.
    pub l: u16,
    /// The quantum number that...
    pub m: i16,
    /// Orientation.
    pub orientation: Quaternion,
}

impl SphericalHarmonic {
    pub fn new(l: u16, m: i16, orientation: Quaternion) -> Self {
        assert!(m.unsigned_abs() <= l);

        Self { l, m, orientation }
    }

    /// Calculate the value of the spherical harmonic at a given θ
    /// and ϕ.
    /// Note that we do not include a normalization constant,
    /// since we handle that at the basis fn level.
    ///
    /// https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    ///
    /// https://docs.rs/scilib/latest/scilib/quantum/index.html: Has an example general equation
    /// http://scipp.ucsc.edu/~haber/ph116C/SphericalHarmonics_12.pdf
    pub fn value(&self, θ: f64, ϕ: f64) -> Cplx {
        // todo: You could hard-code some of the more common ones like l = 0.

        // Approach 1: `scilib` module.
        let l = self.l;
        let m = self.m;

        // https://docs.rs/scilib/latest/scilib/quantum/fn.spherical_harmonics.html
        let result = scilib::quantum::spherical_harmonics(l.into(), m.into(), θ, ϕ);
        return Cplx {
            real: result.re,
            im: result.im,
        };

        // Approach 2: Poly module.

        // todo: QC this.
        // let P = Poly::gen_legendre(l.into(), m.into());

        // let part1 = (2 * l + 1) as f64 / (4. * PI) * factorial(l - m as u16) as f64
        //     / factorial(l + m as u16) as f64;

        // let part2 = (-1.0_f64).powi(m.into()) * part1.sqrt() * P.compute(θ.cos());

        // todo: is "im theta" i theta, or i * m theta?? Damn notation.
        // return Cplx::from_real(part2) * (IM * ϕ).exp();

        // Approach 3: Hard-coded. Can still use this for low values; may be more
        // performant.
        //
        // println!("θ: {} ϕ: {}, l: {}, m: {}", θ, ϕ, l, m);
        match l {
            0 => (0.5 * (1. / PI).sqrt()).into(),
            1 => match m {
                -1 => (-IM * ϕ).exp() * θ.sin() * 0.5 * (3. / (2. * PI)).sqrt(),
                0 => Cplx::from_real(θ.cos()) * 0.5 * (3. / PI).sqrt(),
                1 => (IM * ϕ).exp() * θ.sin() * -0.5 * (3. / (2. * PI)).sqrt(),
                _ => panic!("Invalid m quantum number"),
            },
            // todo: Norm consts.
            2 => match m {
                -2 => (-IM * -2. * ϕ).exp() * θ.sin().powi(2) * 0.25 * (15. / (2. * PI)).sqrt(),
                -1 => (-IM * ϕ).exp() * θ.sin() * θ.cos() * 0.5 * (15. / (2. * PI)).sqrt(),
                0 => (3. * (θ.cos().powi(2) - 1.) * 0.25 * (5. / PI).sqrt()).into(),
                1 => (IM * ϕ).exp() * θ.sin() * θ.cos() * -0.5 * (15. / (2. * PI)).sqrt(),
                2 => (IM * 2. * ϕ).exp() * θ.sin().sin().powi(2) * 0.25 * (15. / (2. * PI)).sqrt(),
                _ => panic!("Invalid m quantum number"),
            },
            3 => match m {
                -3 => (-IM * -3. * ϕ).exp() * θ.sin().powi(3) * 0.125 * (35. / PI).sqrt(),
                -2 => {
                    (-IM * -2. * ϕ).exp()
                        * θ.sin().powi(2)
                        * θ.cos()
                        * 0.25
                        * (105. / (2. * PI)).sqrt()
                }
                -1 => {
                    (-IM * ϕ).exp()
                        * θ.sin()
                        * (5. * θ.cos().powi(2) - 1.)
                        * 0.125
                        * (21. / PI).sqrt()
                }
                0 => (5. * θ.cos().powi(3) - 3. * θ.cos() * 0.25 * (7. / PI).sqrt()).into(),
                1 => {
                    (IM * ϕ).exp()
                        * θ.sin()
                        * (5. * θ.cos().powi(2) - 1.)
                        * -0.125
                        * (21. / PI).sqrt()
                }
                2 => {
                    (IM * 2. * ϕ).exp()
                        * θ.sin().powi(2)
                        * θ.cos()
                        * 0.25
                        * (105. / (2. * PI)).sqrt()
                }
                3 => (IM * 3. * ϕ).exp() * θ.sin().sin().powi(3) * -0.125 * (35. / PI).sqrt(),
                _ => panic!("Invalid m quantum number"),
            },
            _ => unimplemented!(),
        }
    }
}

impl Default for SphericalHarmonic {
    /// Return the l=0 spherical harmonic. (Spherically symmetric, with only a normalization const)
    fn default() -> Self {
        Self {
            l: 0,
            m: 0,
            orientation: Quaternion::new_identity(),
        }
    }
}

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

/// See Sebens: Electric Charge Density, equation 24
#[derive(Clone, Debug)]
pub struct Sto {
    pub posit: Vec3,
    pub n: u16,
    pub xi: f64,
    pub harmonic: SphericalHarmonic,
    pub weight: f64,
    pub charge_id: usize,
}

impl Sto {
    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        // todo: This currently ignores the spherical harmonic part; add that!
        let r = (posit_sample - self.posit).magnitude();

        // todo: Experimenting using lessons learned from H orbitals, to get  this working.
        return Cplx::from_real({
            let n = self.n;
            let nf = n as f64;
            let l = 0;

            let norm_term_num = (2. / (nf * A_0)).powi(3) * factorial(n - l - 1) as f64;
            let norm_term_denom = (2 * n as u64 * factorial(n + l).pow(3)) as f64;
            let norm_term = (norm_term_num / norm_term_denom).sqrt();

            // let L = Poly::laguerre((n - l - 1).into(), 2 * l + 1);
            let L = util::make_laguerre(n - l - 1, 2 * l + 1);

            norm_term
                * (-r / (nf * A_0)).exp()
                * (2. * r / (nf * A_0)).powi(l.into())
                // * L.compute(2. * r / (nf * A_0))
                * L(2. * r / (nf * A_0))
        });

        // return Cplx::from_real({
        //     let n = self.n;
        //     let l = 0;
        //     // N is the normalization constant for the radial part
        //     let ρ = Z_H * r / A_0; // always r.
        //     let c = (Z_H * A_0).powf(3. / 2.); // always 1
        //
        //     let part1 = match n {
        //         1 => 2.,
        //         2 => match l {
        //             0 => 2. * (2. - ρ),           // todo: z/a0 vice / 2a0?
        //             1 => 1. / 3.0_f64.sqrt() * ρ, // z/a0 vs /3a0?
        //             _ => panic!(),
        //         },
        //         // compare to this: https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Map%3A_Physical_Chemistry_for_the_Biosciences_(Chang)/11%3A_Quantum_Mechanics_and_Atomic_Structure/11.10%3A_The_Schrodinger_Wave_Equation_for_the_Hydrogen_Atom
        //         3 => match l {
        //             0 => 2. / 27. * (27. - 18. * ρ + 2. * ρ.powi(2)),
        //             1 => 1. / (81. * 3.0_f64.sqrt()) * (6. - ρ.powi(2)),
        //             2 => 1. / (81. * 15.0_f64.sqrt()) * ρ.powi(2),
        //             _ => panic!(),
        //         },
        //         _ => unimplemented!(), // todo: More
        //     };
        //
        //     let part2 = (-ρ / n as f64).exp();
        //
        //     part1 * part2 * c
        // });

        // Note: Divide xi bo A_0 if setting that to something other than 1.
        // todo: Confirm quantum number `n` isn't part of this term if you start using it.
        // ξ^(3/2) / √π
        let N = PI_SQRT_INV * self.xi.powf(1.5);

        // Note: Try ChatGPT: "User
        // What is the formula for the radial component of a slater-type-orbital, given distance from nucleus r,
        // and the n quantum number. Assume quantum numbers l and m are 0."
        // You get a more complicated N, involving a double factorial.

        // e^(-ξr/n)
        // todo: Which?
        let exp_term = (-self.xi * r / (self.n as f64)).exp();
        // e^(-ξr)
        // let exp_term = (-self.xi * r).exp();

        //   let L = Poly::laguerre((n - l - 1).into(), 2 * l + 1);
        // // let L = util::make_laguerre(n - l - 1, 2 * l + 1.);
        //
        // return
        //     * (-r / (n as f64 * A_0)).exp()
        //     * (2. * r / (n as f64 * A_0)).powi(l.into())
        //     * L.compute(2. * r / (n as f64 * A_0));

        // r^(n-1)
        let radial = Cplx::from_real(
            // Note: We're leaving out A_0, since it's 1.
            N * r.powi((self.n - 1).into()) * exp_term,
        );

        radial
    }

    /// Analytic second derivative using analytic basis functions.
    /// See OneNote: `Exploring the WF, part 6`. Hardcoded for n=1.
    /// todo: Deprecate A/R
    pub fn _second_deriv_n1(&self, posit_sample: Vec3) -> Cplx {
        // todo: This currently ignores the spherical harmonic part; add that!

        // Enter this in Wolfram Alpha: `second derivative of (1/sqrt(pi)) * \xi^(3/2) * e^(-\xi * sqrt(x^2 + y^2 + z^2)) with respect to x`
        let diff = posit_sample - self.posit;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        let N = PI_SQRT_INV * self.xi.powf(1.5);
        // Note: This variant uses the same form as our `value` fn, but assumes n = 1.

        let exp_term = (-self.xi * r).exp();

        let mut result = 0.;

        // Each part is the second deriv WRT to an orthogonal axis.
        for x in &[diff.x, diff.y, diff.z] {
            result += self.xi.powi(2) * x.powi(2) * exp_term / r.powi(2);
            result += self.xi * x.powi(2) * exp_term / r.powi(3);
            result -= self.xi * exp_term / r;
        }

        let radial = Cplx::from_real(N * result);

        radial
    }

    /// Analytic second derivative using analytic basis functions.
    /// See OneNote: `Exploring the WF, part 6`.
    pub fn second_deriv(&self, posit_sample: Vec3) -> Cplx {
        // todo: This currently ignores the spherical harmonic part; add that!

        // Enter this in Wolfram Alpha: `second derivative of (1/sqrt(pi)) * \xi^(3/2) * sqrt(x^2 + y^2 + z^2)^(n-1) * e^(-\xi * sqrt(x^2 + y^2 + z^2) / n) with respect to x`

        // code-shorteners, to help make these long equations easier to read. And precomputing  some terms.
        let diff = posit_sample - self.posit;
        let r_sq = diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2);
        let r = r_sq.sqrt();

        let n = self.n as f64;
        let xi = self.xi;

        let exp_term = (-xi * r / n).exp();

        let N = PI_SQRT_INV * xi.powf(1.5);
        let term0_coeff = N * r_sq.powf((n - 1.) / 2.);

        let mut result = 0.;

        // Each part is the second deriv WRT to an orthogonal axis.
        for x in &[diff.x, diff.y, diff.z] {
            let x_sq = x.powi(2);

            let term0 = xi.powi(2) * x.powi(2) * exp_term / (n.powi(2) * r_sq)
                + xi * x_sq * exp_term / (n * r.powi(3))
                - xi * exp_term / (n * r);

            let term1 = -2.
                * (n - 1.)
                * xi.powf(5. / 2.)
                * x_sq
                * r_sq.powf((n - 1.) / 2. - 1.5)
                * exp_term
                * PI_SQRT_INV
                / n;

            // The part to the left and right of the parenthesis.
            let term2a = PI_SQRT_INV * xi.powf(1.5) * exp_term;

            // (The portion inside the parenthesis, as described by Wolfram Alpha)
            let term2b =
                2. * ((n - 1.) / 2. - 1.) * (n - 1.) * x_sq * r_sq.powf((n - 1.) / 2. - 2.)
                    + (n - 1.) * r_sq.powf((n - 1.) / 2. - 1.);

            result += term0_coeff * term0 + term1 + term2a * term2b;
        }

        let radial = Cplx::from_real(result);

        radial
    }

    /// Saves some minor computations over calculating them individually, due to
    /// eliminated terms. Of note, the exponential term cancels out.
    /// todo: Is it really always real?
    pub fn psi_pp_div_psi(&self, posit_sample: Vec3) -> f64 {
        let diff = posit_sample - self.posit;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        let mut result = 0.;

        for x in &[diff.x, diff.y, diff.z] {
            result += self.xi.powi(2) * x.powi(2) / r.powi(2);
            result += self.xi * x.powi(2) / r.powi(3);
            result -= self.xi / r;
        }

        let radial = result;

        radial
    }

    pub fn V_p_from_psi(&self, posit_sample: Vec3) -> f64 {
        // From Wolfram alpha
        let diff = posit_sample - self.posit;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        let mut result = 0.;

        let xi = self.xi;

        // todo: DRY here
        result +=
            1. / r.powi(5) * xi * diff.x * (diff.y.powi(2) + diff.z.powi(2)) * (2. * xi * r + 3.);
        result +=
            1. / r.powi(5) * xi * diff.y * (diff.x.powi(2) + diff.z.powi(2)) * (2. * xi * r + 3.);
        result +=
            1. / r.powi(5) * xi * diff.z * (diff.x.powi(2) + diff.y.powi(2)) * (2. * xi * r + 3.);

        let radial = result;

        radial * KE_COEFF
    }

    pub fn V_pp_from_psi(&self, posit_sample: Vec3) -> f64 {
        let diff = posit_sample - self.posit;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        let mut result = 0.;

        let xi = self.xi;
        let x2 = diff.x.powi(2);
        let y2 = diff.y.powi(2);
        let z2 = diff.z.powi(2);

        // todo DRY here
        result += 1. / r.powi(7)
            * xi
            * (y2 + z2)
            * ((y2 + z2) * (2. * xi * r + 3.) - 6. * x2 * (xi * r + 2.));
        result += 1. / r.powi(7)
            * xi
            * (x2 + z2)
            * ((x2 + z2) * (2. * xi * r + 3.) - 6. * y2 * (xi * r + 2.));
        result += 1. / r.powi(7)
            * xi
            * (x2 + y2)
            * ((x2 + y2) * (2. * xi * r + 3.) - 6. * z2 * (xi * r + 2.));

        let radial = result;

        radial * KE_COEFF
    }
}

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
        const EPS: f64 = 0.0000001;
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

        // let θ = if diff.z > EPS {
        //     (diff.x.powi(2) + diff.y.powi(2)).sqrt().atan2(diff.z)
        // } else if diff.z < -EPS {
        //     (PI + diff.x.powi(2) + diff.y.powi(2)).sqrt().atan2(diff.z)
        // } else if diff.z.abs() < EPS && (diff.x * diff.y).abs() > 0. {
        // PI / 2.
        // } else {
        //     0. // todo: actually undefined.
        // };

        // let ϕ = if diff.x > EPS {
        //     diff.y.atan2(diff.x)
        // } else if diff.x < -EPS && diff.y >= EPS {
        //    diff.y.atan2(diff.x) + PI
        // } else if  diff.x < -EPS && diff.y < -EPS {
        //     diff.y.atan2(diff.x) - PI
        // } else if diff.x.abs() < EPS && diff.y > EPS {
        //     PI / 2.
        // } else if diff.x.abs() < EPS && diff.y < -EPS {
        //     -PI / 2.
        // } else {
        //     0. // todo: Actually undefined.
        // };

        // todo: Is this why you're unable to get m = -1 to be orthogonal to m = 1?

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

    pub fn psi_pp_div_psi(&self, posit_sample: Vec3) -> f64 {
        // todo: Do this.
        0.
    }
}
