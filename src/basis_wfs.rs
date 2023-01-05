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

use crate::complex_nums::{Cplx, IM};

// Hartree units.
const A_0: f64 = 1.;
const Z_H: f64 = 1.;
//
// #[derive(Copy, Clone, PartialEq)]
// pub enum BasisFn {
//     H100,
//     H200,
//     H300,
//     H210(Vec3), // axis
//     // H211,
//     // H21M1,
//     Sto(f64), // Slater exponent
// }
//
// impl BasisFn {
//     /// The box is required, instead of a ref, to deal with the closures.
//     pub fn f(&self) -> Box<dyn Fn(Vec3, Vec3) -> f64 + '_> {
//         match self {
//             Self::H100 => Box::new(h_wf_100),
//             Self::H200 => Box::new(h_wf_200),
//             Self::H300 => Box::new(h_wf_300),
//             Self::H210(axis) => Box::new(|a, b| h_wf_210(a, b, *axis)),
//             // Self::H211 => &h_wf_211,
//             // Self::H21M1 => &h_wf_21m1,
//             Self::Sto(slater_exp) => Box::new(|a, b| slater(a, b, *slater_exp)),
//         }
//     }
//
//     pub fn descrip(&self) -> String {
//         match self {
//             // Self::H100 => "H100: n=1, l=0, m=0",
//             // Self::H200 => "H200: n=2, l=0, m=0",
//             // Self::H300 => "H300: n=3, l=0, m=0",
//             // Self::H210(_) => "H210: n=2, l=1, m=0",
//             // // Self::H211 => "H211: n=2, l=1, m=1",
//             // // Self::H21M1 => "H21-1: n=2, l=1, m=-1",
//             // Self::Sto(_) => "STO",
//             Self::H100 => "H100",
//             Self::H200 => "H200",
//             Self::H300 => "H300",
//             Self::H210(_) => "H210",
//             // Self::H211 => "H211: n=2, l=1, m=1",
//             // Self::H21M1 => "H21-1: n=2, l=1, m=-1",
//             Self::Sto(_) => "STO",
//         }
//         .to_owned()
//     }
// }
//
// fn r_from_pts(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
//     let diff = posit_sample - posit_nuc;
//     (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt()
// }
//
// /// Analytic solution for n=1, s orbital
// pub fn h_wf_100(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
//     let r = r_from_pts(posit_nuc, posit_sample);
//
//     let ρ = Z_H * r / A_0;
//     1. / PI.sqrt() * (Z_H / A_0).powf(3. / 2.) * (-ρ).exp()
// }
//
// /// Analytic solution for n=2, s orbital
// pub fn h_wf_200(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
//     let r = r_from_pts(posit_nuc, posit_sample);
//
//     let ρ = Z_H * r / A_0;
//     1. / (32. * PI).sqrt() * (Z_H / A_0).powf(3. / 2.) * (2. - ρ) * (-ρ / 2.).exp()
// }
//
// /// Analytic solution for n=3, s orbital
// pub fn h_wf_300(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
//     let r = r_from_pts(posit_nuc, posit_sample);
//
//     let ρ = Z_H * r / A_0;
//
//     1. / (81. * (3. * PI).sqrt())
//         * (Z_H / A_0).powf(3. / 2.)
//         * (27. - 18. * ρ + 2. * ρ.powi(2))
//         * (-ρ / 3.).exp()
// }
//
// // We assume the axis is already normalized.
// pub fn h_wf_210(posit_nuc: Vec3, posit_sample: Vec3, axis_through_lobes: Vec3) -> f64 {
//     // We take Cos theta below, so no need for cos^-1 here.
//     let diff = posit_sample - posit_nuc;
//     (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();
//
//     let cos_θ = diff.to_normalized().dot(axis_through_lobes);
//
//     let ρ = Z_H * r / A_0;
//     1. / (32. * PI).sqrt() * (Z_H / A_0).powf(3. / 2.) * ρ * (-ρ / 2.).exp() * cos_θ
// }
//
// /// todo: Not required, since it's a rotation of h_wf_210?
// pub fn h_wf_211(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
//     let r = r_from_pts(posit_nuc, posit_sample);
//
//     // todo wrong
//     // We take Cos theta below, so no need for cos^-1 here.
//     // todo: Not sure how we deal with diff phis?
//     let sin_theta = posit_nuc.to_normalized().dot(posit_sample.to_normalized());
//
//     let i: f64 = 0.; // todo!!
//     let phi: f64 = 0.; // todo!
//
//     let ρ = Z_H * r / A_0;
//     1. / (64. * PI).sqrt()
//         * (Z_H / A_0).powf(3. / 2.)
//         * ρ
//         * (-ρ / 2.).exp()
//         * sin_theta
//         * (i * phi).exp()
// }
//
// /// todo: Not required, since it's a rotation of h_wf_210?
// /// todo: Dry with above! Only diff is one change in sign.
// pub fn h_wf_21m1(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
//     let r = r_from_pts(posit_nuc, posit_sample);
//
//     // todo wrong
//     // We take Cos theta below, so no need for cos^-1 here.
//     // todo: Not sure how we deal with diff phis?
//     let sin_theta = posit_nuc.to_normalized().dot(posit_sample.to_normalized());
//
//     let i: f64 = 0.; // todo!!
//     let phi: f64 = 0.; // todo!
//
//     let ρ = Z_H * r / A_0;
//     1. / (64. * PI).sqrt()
//         * (Z_H / A_0).powf(3. / 2.)
//         * ρ
//         * (-ρ / 2.).exp()
//         * sin_theta
//         * (-i * phi).exp()
// }

// todo: Below this, is an experimental new approach to generic basis fns

// todo: Remove this enum if you use STOs as the only basis
pub enum Basis {
    Sto(Sto),
    H(HOrbital),
}

impl Basis {
    /// These getters allow access to common values (all but slater weight) without unpacking.
    pub fn posit(&self) -> Vec3 {
        match self {
            Self::Sto(v) => v.posit,
            Self::H(v) => v.posit,
        }
    }

    pub fn n(&self) -> u16 {
        match self {
            Self::Sto(v) => v.n,
            Self::H(v) => v.n,
        }
    }

    pub fn harmonic(&self) -> &SphericalHarmonic {
        match self {
            Self::Sto(v) => &v.harmonic,
            Self::H(v) => &v.harmonic,
        }
    }

    pub fn weight(&self) -> f64 {
        match self {
            Self::Sto(v) => v.weight,
            Self::H(v) => v.weight,
        }
    }

    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        match self {
            Self::Sto(v) => v.value(posit_sample),
            Self::H(v) => v.value(posit_sample),
        }
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
pub struct SphericalHarmonic {
    /// The quantum number the describes orbital shape.
    l: u8,
    /// The quantum number that...
    m: i8,
    /// Orientation.
    orientation: Quaternion,
}

// todo: If you use a continuous range, use a struct with parameter fields
// todo instead of an enum that contains discrete values. This is your
// todo likely approach.

impl SphericalHarmonic {
    pub fn new(l: u8, m: i8, orientation: Quaternion) -> Self {
        assert!(m.abs() as u8 <= l);

        Self { l, m, orientation }
    }

    /// Calculate the value of the spherical harmonic at a given θ
    /// and ϕ.
    /// Note that we do not include a normalization constant,
    /// since we handle that at the basis fn level.
    /// todo: ϕ as an `Option`?
    pub fn value(&self, θ: f64, ϕ: f64) -> Cplx {
        // todo: Hard-coded match arms for now.

        // todo: QC normalization const as it relates to radial part!

        // todo: Shortcut  vars or consts for repeated patterns A/R

        match self.l {
            0 => 1.0.into(),
            1 => match self.m {
                -1 => (-IM * ϕ).exp() * θ.sin(),
                0 => θ.cos().into(),
                1 => (IM * ϕ).exp() * θ.sin(),
                _ => panic!("Invalid m quantum number"),
            },
            2 => match self.m {
                -2 => (-IM * -2. * ϕ).exp() * θ.sin().powi(2),
                -1 => (-IM * ϕ).exp() * θ.sin() * θ.cos(),
                0 => (3. * θ.cos().powi(2) - 1.).into(),
                1 => (IM * ϕ).exp() * θ.sin() * θ.cos(),
                2 => (IM * 2. * ϕ).exp() * θ.sin().sin().powi(2),
                _ => panic!("Invalid m quantum number"),
            },
            3 => match self.m {
                -3 => (-IM * -3. * ϕ).exp() * θ.sin().powi(3),
                -2 => (-IM * -2. * ϕ).exp() * θ.sin().powi(2) * θ.cos(),
                -1 => (-IM * ϕ).exp() * θ.sin() * (5. * θ.cos().powi(2) - 1.),
                0 => (5. * θ.cos().powi(3) - 3. * θ.cos()).into(),
                1 => (IM * ϕ).exp() * θ.sin() * (5. * θ.cos().powi(2) - 1.),
                2 => (IM * 2. * ϕ).exp() * θ.sin().powi(2) * θ.cos(),
                3 => (IM * 3. * ϕ).exp() * θ.sin().sin().powi(3),
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
/// todo: Update to include angular part
pub struct Sto {
    posit: Vec3,
    n: u16,
    harmonic: SphericalHarmonic,
    eff_charge: f64,
    weight: f64,
}

impl Sto {
    pub fn new(
        posit: Vec3,
        n: u16,
        harmonic: SphericalHarmonic,
        eff_charge: f64,
        weight: f64,
    ) -> Self {
        Self {
            posit,
            n,
            harmonic,
            eff_charge,
            weight,
        }
    }

    /// Calculate this basis function's value at a given point.
    /// Does not include weight.
    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        let diff = posit_sample - self.posit;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        // todo: Find formula for N, and confirm the one here is even correct.
        // N is the normalization constant.
        let N = (self.eff_charge.powi(3) / PI).sqrt(); // todo: Update base on n.
        let radial = r.powi(self.n as i32 - 1) * (-self.eff_charge * r).exp();

        let θ = 0.;
        let ϕ = 0.;
        let angular = self.harmonic.value(θ, ϕ);

        Cplx::from_real(N) * Cplx::from_real(radial) * angular
    }
}

/// A Hydrogen-atomic orbital. Note that unlike STOs, does not include an
/// effective charge.
/// todo: If this turns out to be teh same as an STO but with effective-
/// charge always equal to one, remove it.
pub struct HOrbital {
    posit: Vec3,
    n: u16,
    harmonic: SphericalHarmonic,
    weight: f64,
}

impl HOrbital {
    pub fn new(posit: Vec3, n: u16, harmonic: SphericalHarmonic, weight: f64) -> Self {
        Self {
            posit,
            n,
            harmonic,
            weight,
        }
    }

    /// Calculate this basis function's value at a given point.
    /// Does not include weight.
    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        let diff = posit_sample - self.posit;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        // N is the normalization constant.
        let N = 1. / (32. * PI).sqrt();

        // todo: Update this based on n.
        let ρ = Z_H * r / A_0;
        let radial = (Z_H / A_0).powf(3. / 2.) * ρ * (-ρ / 2.).exp();

        // let cos_theta = diff.to_normalized().dot(axis_through_lobes);

        let θ = 0.;
        let ϕ = 0.;
        let angular = self.harmonic.value(θ, ϕ);

        Cplx::from_real(N) * Cplx::from_real(radial) * angular
    }
}

// /// A Slater-Type Orbital (STO). Includes a `n`: The quantum number; the effective
// /// charge slater exponent (ζ) may be used to simulate "effective charge", which
// /// can represent "electron shielding".(?)
// /// todo: Update to include angular part
// pub fn slater(
//     posit_nuc: Vec3,
//     posit_sample: Vec3,
//     n: u16,
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
