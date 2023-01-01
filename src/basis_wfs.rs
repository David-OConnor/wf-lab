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

const A_0: f64 = 1.;
const Z_H: f64 = 1.;

#[derive(Copy, Clone, PartialEq)]
pub enum BasisFn {
    H100,
    H200,
    H300,
    H210,
    H211,
    H21M1,
    // todo etc
}

impl BasisFn {
    pub fn f(&self) -> &dyn Fn(Vec3, Vec3) -> f64 {
        match self {
            Self::H100 => &h_wf_100,
            Self::H200 => &h_wf_200,
            Self::H300 => &h_wf_300,
            Self::H210 => &h_wf_210,
            Self::H211 => &h_wf_211,
            Self::H21M1 => &h_wf_21m1,
        }
    }

    pub fn descrip(&self) -> String {
        match self {
            Self::H100 => "H100: n=1, l=0, m=0",
            Self::H200 => "H200: n=2, l=0, m=0",
            Self::H300 => "H300: n=3, l=0, m=0",
            Self::H210 => "H210: n=2, l=1, m=0",
            Self::H211 => "H211: n=2, l=1, m=1",
            Self::H21M1 => "H21-1: n=2, l=1, m=-1",
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

pub fn h_wf_210(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let r = r_from_pts(posit_nuc, posit_sample);

    // todo wrong
    // We take Cos theta below, so no need for cos^-1 here.
    // todo: Not sure how we deal with diff phis?
    let cos_theta = posit_nuc.to_normalized().dot(posit_sample.to_normalized());

    let ρ = Z_H * r / A_0;
    1. / (32. * PI).sqrt() * (Z_H / A_0).powf(3. / 2.) * ρ * (-ρ / 2.).exp() * cos_theta
}

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

    // let ρ = Z_H * r / A_0;
    // 1. / PI.sqrt() * (Z_H / A_0).powf(3. / 2.) * (-ρ).exp()

    (slater_exp.powi(3) / PI).sqrt() * (-slater_exp * r).exp()
}
