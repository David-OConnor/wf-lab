//! This module contains basis wave functions, which we combine in a LCAO
//! approach. For example, various Hydrogen orbitals.
//!
//! For calculated analytic H wave fns:
//! https://chem.libretexts.org/Courses/University_of_California_Davis/
//! UCD_Chem_107B%3A_Physical_Chemistry_for_Life_Scientists/Chapters/4%3A_Quantum_Theory/
//! 4.10%3A_The_Schr%C3%B6dinger_Wave_Equation_for_the_Hydrogen_Atom

// todo: Bring in your own complex-num lib. Check prev code bases to find it.

use std::f64::consts::PI;

use lin_alg2::f64::{Quaternion, Vec3};

// Hartree units.
const A_0: f64 = 1.;
const Z_H: f64 = 1.;

#[derive(Copy, Clone, PartialEq)]
pub enum BasisFn {
    H100,
    H200,
    H300,
    H210(Vec3), // axis
    // H211,
    // H21M1,
    Sto(f64), // Slater exponent
}

impl BasisFn {
    /// The box is required, instead of a ref, to deal with the closures.
    pub fn f(&self) -> Box<dyn Fn(Vec3, Vec3) -> f64 + '_> {
        match self {
            Self::H100 => Box::new(h_wf_100),
            Self::H200 => Box::new(h_wf_200),
            Self::H300 => Box::new(h_wf_300),
            Self::H210(axis) => Box::new(|a, b| h_wf_210(a, b, *axis)),
            // Self::H211 => &h_wf_211,
            // Self::H21M1 => &h_wf_21m1,
            Self::Sto(slater_exp) => Box::new(|a, b| slater(a, b, *slater_exp)),
        }
    }

    pub fn descrip(&self) -> String {
        match self {
            // Self::H100 => "H100: n=1, l=0, m=0",
            // Self::H200 => "H200: n=2, l=0, m=0",
            // Self::H300 => "H300: n=3, l=0, m=0",
            // Self::H210(_) => "H210: n=2, l=1, m=0",
            // // Self::H211 => "H211: n=2, l=1, m=1",
            // // Self::H21M1 => "H21-1: n=2, l=1, m=-1",
            // Self::Sto(_) => "STO",
            Self::H100 => "H100",
            Self::H200 => "H200",
            Self::H300 => "H300",
            Self::H210(_) => "H210",
            // Self::H211 => "H211: n=2, l=1, m=1",
            // Self::H21M1 => "H21-1: n=2, l=1, m=-1",
            Self::Sto(_) => "STO",
        }
        .to_owned()
    }
}

fn r_from_pts(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let diff = posit_sample - posit_nuc;
    (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt()
}

/// Analytic solution for n=1, s orbital
pub fn h_wf_100(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let r = r_from_pts(posit_nuc, posit_sample);

    let ρ = Z_H * r / A_0;
    1. / PI.sqrt() * (Z_H / A_0).powf(3. / 2.) * (-ρ).exp()
}

/// Analytic solution for n=2, s orbital
pub fn h_wf_200(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let r = r_from_pts(posit_nuc, posit_sample);

    let ρ = Z_H * r / A_0;
    1. / (32. * PI).sqrt() * (Z_H / A_0).powf(3. / 2.) * (2. - ρ) * (-ρ / 2.).exp()
}

/// Analytic solution for n=3, s orbital
pub fn h_wf_300(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let r = r_from_pts(posit_nuc, posit_sample);

    let ρ = Z_H * r / A_0;

    1. / (81. * (3. * PI).sqrt())
        * (Z_H / A_0).powf(3. / 2.)
        * (27. - 18. * ρ + 2. * ρ.powi(2))
        * (-ρ / 3.).exp()
}

// We assume the axis is already normalized.
pub fn h_wf_210(posit_nuc: Vec3, posit_sample: Vec3, axis_through_lobes: Vec3) -> f64 {
    // We take Cos theta below, so no need for cos^-1 here.
    let diff = posit_sample - posit_nuc;
    (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

    let cos_θ = diff.to_normalized().dot(axis_through_lobes);

    let ρ = Z_H * r / A_0;
    1. / (32. * PI).sqrt() * (Z_H / A_0).powf(3. / 2.) * ρ * (-ρ / 2.).exp() * cos_θ
}

/// todo: Not required, since it's a rotation of h_wf_210?
pub fn h_wf_211(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let r = r_from_pts(posit_nuc, posit_sample);

    // todo wrong
    // We take Cos theta below, so no need for cos^-1 here.
    // todo: Not sure how we deal with diff phis?
    let sin_theta = posit_nuc.to_normalized().dot(posit_sample.to_normalized());

    let i: f64 = 0.; // todo!!
    let phi: f64 = 0.; // todo!

    let ρ = Z_H * r / A_0;
    1. / (64. * PI).sqrt()
        * (Z_H / A_0).powf(3. / 2.)
        * ρ
        * (-ρ / 2.).exp()
        * sin_theta
        * (i * phi).exp()
}

/// todo: Not required, since it's a rotation of h_wf_210?
/// todo: Dry with above! Only diff is one change in sign.
pub fn h_wf_21m1(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let r = r_from_pts(posit_nuc, posit_sample);

    // todo wrong
    // We take Cos theta below, so no need for cos^-1 here.
    // todo: Not sure how we deal with diff phis?
    let sin_theta = posit_nuc.to_normalized().dot(posit_sample.to_normalized());

    let i: f64 = 0.; // todo!!
    let phi: f64 = 0.; // todo!

    let ρ = Z_H * r / A_0;
    1. / (64. * PI).sqrt()
        * (Z_H / A_0).powf(3. / 2.)
        * ρ
        * (-ρ / 2.).exp()
        * sin_theta
        * (-i * phi).exp()
}

// todo: Below this, is an experimental new approach to generic basis fns

/// Represents a spherical harmonic, at given l and m quantum numbers.
/// Note that we do not represent degenerate orientations as separate
/// values.
/// todo: Instead of an enum, is there a procedural way to represent
/// *any* spherical harmonic using a single function?
#[derive(Clone, Copy, PartialEq)]
pub enum SphericalHarmonic {
    // todo: Should orientation be baked into this? Should we have a fn that
}

/// A Slater-Type Orbital (STO). Includes a `n`: The quantum number; the effective
/// charge slater exponent (ζ) may be used to simulate "effective charge", which
/// can represent "electron shielding".(?)
/// todo: Update to include angular part
pub struct SlaterOrbital {
    posit: Vec3,
    n: u32,
    harmonic: SphericalHarmonic,
    orientation: Quaternion,
    eff_charge: f64,
    weight: f64,
}

impl SlaterOrbital {
    /// Calculate this basis function's value at a given point.
    pub fn value(&self, posit_sample: Vec3) -> f64 {
        let diff = posit_sample - posit_nuc;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        // todo: Find formula for N, and confirm the one here is even correct.
        // N is the normalization constant.
        let N = (slater_exp.powi(3) / PI).sqrt(); // todo: Update base on n.
        let radial = r.powi(self.n - 1) * (-self.eff_charge * r).exp();

        let angular = 0.;

        N * radial * angular
    }
}

impl SphericalHarmonic {
    /// Calculate the value of the spherical harmonic at a given θ
    /// and ϕ. The base orientation is... // todo
    pub fn value(θ: f64, ϕ: f64, orientation: Quatenrion) {
        //
    }
}

/// A Hydrogen-atomic orbital. Note that unlike STOs, does not include an
/// effective charge.
/// todo: If this turns out to be teh same as an STO but with effective-
/// charge always equal to one, remove it.
pub struct HOrbital {
    posit: Vec3,
    n: u32,
    harmonic: SphericalHarmonic,
    orientation: Quaternion,
    weight: f64,
}

impl HOrbital {
    /// Calculate this basis function's value at a given point.
    pub fn value(&self, posit_sample: Vec3) -> f64 {
        let diff = posit_sample - posit_nuc;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        // N is the normalization constant.
        let N = 1. / (32. * PI).sqrt();

        // todo: Update this based on n.
        let ρ = Z_H * r / A_0;
        let radial = (Z_H / A_0).powf(3. / 2.) * ρ * (-ρ / 2.).exp();

        // let cos_theta = diff.to_normalized().dot(axis_through_lobes);

        let angular = 1.;

        N * radial * angular
    }
}

// /// A Slater-Type Orbital (STO). Includes a `n`: The quantum number; the effective
// /// charge slater exponent (ζ) may be used to simulate "effective charge", which
// /// can represent "electron shielding".(?)
// /// todo: Update to include angular part
// pub fn slater(
//     posit_nuc: Vec3,
//     posit_sample: Vec3,
//     n: u32,
//     eff_charge: f64,
//     harmonic: SphericalHarmonic,
//     orientation: Quaternion,
// ) -> f64 {
//     let diff = posit_sample - posit_nuc;
//     let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

//     // todo: Update the normalization constant based on n!

//     // todo: Find formula for N, and confirm the one here is even correct.
//     let N = (slater_exp.powi(3) / PI).sqrt(); // todo: Update base on n.
//     let radial = N * r.powi(n - 1) * (-eff_charge * r).exp();

//     let angular = 0.;

//     radial * angular
// }

// /// A basis function for hydrogen orbitals. Can be used to represent any
// /// H basis function.
// pub fn h_basis(
//     posit_nuc: Vec3,
//     posit_sample: Vec3,
//     n: u32,
//     harmonic: SphericalHarmonic,
//     orientation: Quaternion,
// ) -> f64 {
//     let diff = posit_sample - posit_nuc;
//     let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

//     let cos_theta = diff.to_normalized().dot(axis_through_lobes);

//     let ρ = Z_H * r / A_0;
//     1. / (32. * PI).sqrt() * (Z_H / A_0).powf(3. / 2.) * ρ * (-ρ / 2.).exp() * cos_theta
// }
