//! This module contains basis wave functions, which we combine in a LCAO
//! approach. For example, various Hydrogen orbitals.
//!
//! For calculated analytic H wave fns:
//! https://chem.libretexts.org/Courses/University_of_California_Davis/
//! UCD_Chem_107B%3A_Physical_Chemistry_for_Life_Scientists/Chapters/4%3A_Quantum_Theory/
//! 4.10%3A_The_Schr%C3%B6dinger_Wave_Equation_for_the_Hydrogen_Atom

// todo: Bring in your own complex-num lib. Check prev code bases to find it.

use std::f64::consts::PI;

use lin_alg2::f64::Vec3;

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
    let r = r_from_pts(posit_nuc, posit_sample);
    // We take Cos theta below, so no need for cos^-1 here.

    // todo: Duplicate creations of `diff`; here and in `r_from_pts`.
    let posit_sample_rel = posit_sample - posit_nuc;

    let cos_theta = posit_sample_rel.to_normalized().dot(axis_through_lobes);

    let ρ = Z_H * r / A_0;
    1. / (32. * PI).sqrt() * (Z_H / A_0).powf(3. / 2.) * ρ * (-ρ / 2.).exp() * cos_theta
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

/// Experimenting with Slater orbitals
pub fn slater(posit_nuc: Vec3, posit_sample: Vec3, slater_exp: f64) -> f64 {
    let r = r_from_pts(posit_nuc, posit_sample);

    (slater_exp.powi(3) / PI).sqrt() * (-slater_exp * r).exp()
}
