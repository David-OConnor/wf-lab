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

// todo: There's also a WIP scilib Quantum lib that can do these H orbital calculations
// todo directly.
use lin_alg::f64::{Quaternion, Vec3};
use scilib;

use crate::complex_nums::{Cplx, IM};

// Hartree units.
const A_0: f64 = 1.;
const Z_H: f64 = 1.;

const PI_SQRT_INV: f64 = 0.5641895835477563;

pub(crate) mod gto;
pub(crate) mod h;
pub(crate) mod sto;

pub(crate) use gto::Gto;
pub(crate) use h::HOrbital;
pub(crate) use sto::Sto;

// todo: trait instead?
#[derive(Clone, Debug)]
pub enum Basis {
    H(HOrbital),
    Gto(Gto),
    Sto(Sto),
}

impl Basis {
    /// Constructor we use frequently
    pub fn new_sto(posit: Vec3, n: u16, xi: f64, weight: f64, charge_id: usize) -> Self {
        Self::Sto(Sto {
            posit,
            n,
            xi,
            weight,
            charge_id,
            harmonic: Default::default(), // todo
        })
    }

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

    pub fn xi_mut(&mut self) -> &mut f64 {
        match self {
            // Self::Sto(v) => &mut v.harmonic.l,
            Self::H(_v) => unimplemented!(),
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => &mut v.xi,
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
            Self::H(v) => &mut v.weight,
            Self::Gto(v) => &mut v.weight,
            Self::Sto(v) => &mut v.weight,
        }
    }

    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        match self {
            Self::H(v) => v.value(posit_sample),
            Self::Gto(v) => v.value(posit_sample),
            Self::Sto(v) => v.value(posit_sample),
            // Self::Sto(v) => Cplx::from_real(v.radial_type2(posit_sample)),
        }
    }

    pub fn dx(&self, posit_sample: Vec3) -> Cplx {
        match self {
            Self::H(v) => unimplemented!(),
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.dx(posit_sample),
        }
    }

    pub fn dy(&self, posit_sample: Vec3) -> Cplx {
        match self {
            Self::H(v) => unimplemented!(),
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.dy(posit_sample),
        }
    }

    pub fn dz(&self, posit_sample: Vec3) -> Cplx {
        match self {
            Self::H(v) => unimplemented!(),
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.dz(posit_sample),
        }
    }

    pub fn d2x(&self, posit_sample: Vec3) -> Cplx {
        match self {
            Self::H(v) => unimplemented!(),
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.d2x(posit_sample),
        }
    }

    pub fn d2y(&self, posit_sample: Vec3) -> Cplx {
        match self {
            Self::H(v) => unimplemented!(),
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.d2y(posit_sample),
        }
    }

    pub fn d2z(&self, posit_sample: Vec3) -> Cplx {
        match self {
            Self::H(v) => unimplemented!(),
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.d2z(posit_sample),
        }
    }

    /// Note: We must normalize afterwards.
    pub fn second_deriv(&self, posit_sample: Vec3) -> Cplx {
        match self {
            Self::H(v) => v.second_deriv(posit_sample),
            Self::Gto(_v) => unimplemented!(),
            Self::Sto(v) => v.second_deriv(posit_sample),
            // Self::Sto(v) => v.second_deriv_type2(posit_sample),
        }
    }
    //
    // pub fn psi_pp_div_psi(&self, posit_sample: Vec3) -> f64 {
    //     match self {
    //         Self::H(v) => v.psi_pp_div_psi(posit_sample),
    //         Self::Gto(v) => unimplemented!(),
    //         Self::Sto(v) => v.psi_pp_div_psi(posit_sample),
    //         // Self::Sto(v) => v.psi_pp_div_psi_type2(posit_sample),
    //     }
    // }

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
