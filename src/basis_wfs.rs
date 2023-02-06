//! This module contains basis wave functions, which we combine in a LCAO
//! approach. For example, various Hydrogen orbitals.
//!
//! For calculated analytic H wave fns:
//! https://chem.libretexts.org/Courses/University_of_California_Davis/
//! UCD_Chem_107B%3A_Physical_Chemistry_for_Life_Scientists/Chapters/4%3A_Quantum_Theory/
//! 4.10%3A_The_Schr%C3%B6dinger_Wave_Equation_for_the_Hydrogen_Atom

// todo: Bring in your own complex-num lib. Check prev code bases to find it.

use std::f64::consts::PI;

use crate::{
    complex_nums::{Cplx, IM},
    util::Arr3dBasis,
};

use lin_alg2::f64::{Quaternion, Vec3};


// Hartree units.
const A_0: f64 = 1.;
const Z_H: f64 = 1.;

// todo: Remove this enum if you use STOs as the only basis
#[derive(Clone)]
pub enum Basis {
    Sto(Sto),
    H(HOrbital),
}

impl Basis {
    /// These getters and setters allow access to common values (all but slater weight) without unpacking.
    pub fn posit(&self) -> Vec3 {
        match self {
            Self::Sto(v) => v.posit,
            Self::H(v) => v.posit,
        }
    }

    pub fn posit_mut(&mut self) -> &mut Vec3 {
        match self {
            Self::Sto(v) => &mut v.posit,
            Self::H(v) => &mut v.posit,
        }
    }

    pub fn n(&self) -> u16 {
        match self {
            Self::Sto(v) => v.n,
            Self::H(v) => v.n,
        }
    }

    pub fn n_mut(&mut self) -> &mut u16 {
        match self {
            Self::Sto(v) => &mut v.n,
            Self::H(v) => &mut v.n,
        }
    }

    pub fn l(&self) -> u16 {
        match self {
            Self::Sto(v) => v.harmonic.l,
            Self::H(v) => v.harmonic.l,
        }
    }

    pub fn l_mut(&mut self) -> &mut u16 {
        match self {
            Self::Sto(v) => &mut v.harmonic.l,
            Self::H(v) => &mut v.harmonic.l,
        }
    }

    pub fn m(&self) -> i16 {
        match self {
            Self::Sto(v) => v.harmonic.m,
            Self::H(v) => v.harmonic.m,
        }
    }

    pub fn m_mut(&mut self) -> &mut i16 {
        match self {
            Self::Sto(v) => &mut v.harmonic.m,
            Self::H(v) => &mut v.harmonic.m,
        }
    }

    pub fn harmonic(&self) -> &SphericalHarmonic {
        match self {
            Self::Sto(v) => &v.harmonic,
            Self::H(v) => &v.harmonic,
        }
    }

    pub fn harmonic_mut(&mut self) -> &mut SphericalHarmonic {
        match self {
            Self::Sto(v) => &mut v.harmonic,
            Self::H(v) => &mut v.harmonic,
        }
    }

    pub fn weight(&self) -> f64 {
        match self {
            Self::Sto(v) => v.weight,
            Self::H(v) => v.weight,
        }
    }

    pub fn weight_mut(&mut self) -> &mut f64 {
        match self {
            Self::Sto(v) => &mut v.weight,
            Self::H(v) => &mut v.weight,
        }
    }

    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        match self {
            Self::Sto(v) => v.value(posit_sample),
            Self::H(v) => v.value(posit_sample),
        }
    }

    pub fn charge_id(&self) -> usize {
        match self {
            Self::Sto(v) => v.charge_id,
            Self::H(v) => v.charge_id,
        }
    }

    pub fn charge_id_mut(&mut self) -> &mut usize {
        match self {
            Self::Sto(v) => &mut v.charge_id,
            Self::H(v) => &mut v.charge_id,
        }
    }

    pub fn descrip(&self) -> String {
        match self {
            Self::Sto(_) => "STO",
            Self::H(_) => "H",
        }
            .to_owned()
    }
}

impl PartialEq for Basis {
    /// Just compares if the main type is the same.
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::Sto(_) => match other {
                Self::Sto(_) => true,
                _ => false,
            },
            Self::H(_) => match other {
                Self::Sto(_) => false,
                _ => true,
            },
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
#[derive(Clone)]
pub struct SphericalHarmonic {
    /// The quantum number the describes orbital shape.
    pub l: u16,
    /// The quantum number that...
    pub m: i16,
    /// Orientation.
    pub orientation: Quaternion,
}

// todo: If you use a continuous range, use a struct with parameter fields
// todo instead of an enum that contains discrete values. This is your
// todo likely approach.

impl SphericalHarmonic {
    pub fn new(l: u16, m: i16, orientation: Quaternion) -> Self {
        assert!(m.abs() as u16 <= l);

        Self { l, m, orientation }
    }

    /// Calculate the value of the spherical harmonic at a given θ
    /// and ϕ.
    /// Note that we do not include a normalization constant,
    /// since we handle that at the basis fn level.
    ///
    /// https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    pub fn value(&self, θ: f64, ϕ: f64) -> Cplx {
        // todo: Hard-coded match arms for now.

        // todo: Shortcut  vars or consts for repeated patterns A/R

        // println!("θ: {} ϕ: {}, l: {}, m: {}", θ, ϕ, self.l, self.m);
        match self.l {
            0 => (0.5 * (1. / PI).sqrt()).into(),
            1 => match self.m {
                -1 => (-IM * ϕ).exp() * θ.sin() * 0.5 * (3. / (2. * PI)).sqrt(),
                0 => Cplx::from_real(θ.cos()) * 0.5 * (3. / PI).sqrt(),
                1 => (IM * ϕ).exp() * θ.sin() * -0.5 * (3. / (2. * PI)).sqrt(),
                _ => panic!("Invalid m quantum number"),
            },
            // todo: Norm consts.
            2 => match self.m {
                -2 => (-IM * -2. * ϕ).exp() * θ.sin().powi(2) * 0.25 * (15. / (2. * PI)).sqrt(),
                -1 => (-IM * ϕ).exp() * θ.sin() * θ.cos() * 0.5 * (15. / (2. * PI)).sqrt(),
                0 => (3. * (θ.cos().powi(2) - 1.) * 0.25 * (5. / PI).sqrt()).into(),
                1 => (IM * ϕ).exp() * θ.sin() * θ.cos() * -0.5 * (15. / (2. * PI)).sqrt(),
                2 => (IM * 2. * ϕ).exp() * θ.sin().sin().powi(2) * 0.25 * (15. / (2. * PI)).sqrt(),
                _ => panic!("Invalid m quantum number"),
            },
            3 => match self.m {
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
/// todo: Update to include angular part
#[derive(Clone)]
pub struct Sto {
    pub posit: Vec3,
    pub n: u16,
    pub harmonic: SphericalHarmonic,
    pub eff_charge: f64,
    pub weight: f64,
    /// Somewhat degenerate with `posit`.
    pub charge_id: usize,
}

impl Sto {
    pub fn new(
        posit: Vec3,
        n: u16,
        harmonic: SphericalHarmonic,
        eff_charge: f64,
        weight: f64,
        charge_id: usize,
    ) -> Self {
        assert!(harmonic.l < n);

        Self {
            posit,
            n,
            harmonic,
            eff_charge,
            weight,
            charge_id,
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
#[derive(Clone)]
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
    /// https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Map%3A_Physical_Chemistry_for_the_Biosciences_(Chang)/11%3A_Quantum_Mechanics_and_Atomic_Structure/11.10%3A_The_Schrodinger_Wave_Equation_for_the_Hydrogen_Atom
    fn radial(&self, r: f64, l: u16) -> f64 {
        // N is the normalization constant for the radial part
        let ρ = Z_H * r / A_0;

        let c = (Z_H * A_0).powf(3. / 2.);

        // Z is the charge of the nucleus.

        // function to solve for any n and l?
        // https://bingweb.binghamton.edu/~suzuki/QuantumMechanicsII/4-3_Radial-wave_function.pdf

        // todo: If this is correct, you can factor out (ZH/AO)^3/2 into its own part

        // todo: More abstractions based on rules?
        let part1 = match self.n {
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

        let part2 = (-ρ / self.n as f64).exp();

        part1 * part2 * c
    }

    /// Calculate this basis function's value at a given point.
    /// Does not include weight.
    /// https://quantummechanics.ucsd.edu/ph130a/130_notes/node233.html
    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        const EPS: f64 = 0.0001;
        if self.weight.abs() < EPS {
            return Cplx::new_zero(); // saves some computation.
        }

        let diff = posit_sample - self.posit;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        let radial = self.radial(r, self.harmonic.l);

        // let cos_theta = diff.to_normalized().dot(axis_through_lobes);

        let diff_r = self.harmonic.orientation.inverse().rotate_vec(diff);

        // todo: QC these.
        // https://en.wikipedia.org/wiki/Spherical_coordinate_system
        // let θ = (diff_r.z / r).acos();
        // let a = diff_r.x / (diff_r.x.powi(2) + diff_r.y.powi(2)).sqrt(); // For legibility.
        // let ϕ = diff_r.y.signum() * a.acos();

        // todo: Looks like the diff between above and below is which coord is which.
        // todo: They're also in different forms.

        // Alternative formulations using acos etc can result in NaNs.

        // alt take from https://keisan.casio.com/exec/system/1359533867:
        // let θ = diff_r.y.atan2(diff_r.x);
        // let ϕ = (diff_r.x.powi(2) + diff_r.y.powi(2)).sqrt().atan2(diff_r.z);

        let ϕ = diff_r.y.atan2(diff_r.x);
        let θ = (diff_r.x.powi(2) + diff_r.y.powi(2)).sqrt().atan2(diff_r.z);

        // The rub: θ = NaN!!!

        let angular = self.harmonic.value(θ, ϕ);

        if angular.real.is_nan() {
            println!("θ: {}, ϕ: {}", θ, ϕ);
        }

        // Normalization consts are applied to radial and angular parts separately.
        Cplx::from_real(radial) * angular
    }
}

/// Terms for this a sin + exponential basis, in a single dimension. Associated with a single
/// point, where this a reasonable approximation of the wave function locally.
#[derive(Clone, Debug, Default)]
pub struct SinExpBasisTerm {
    // todo: Complex exponential instead of sin? Probably.
    // pub sin_weight: f64, //todo: Taken care of by amplitude?
    pub sin_amp: f64,
    pub sin_freq: f64, // todo: Angular (ω) or normal?
    pub sin_phase: f64,
    pub decaying_exp_amp: f64,
    pub decaying_exp_rate: f64, // ie λ
    // todo: 0-center these functions, and add a shift for decaying exp?
    pub decaying_exp_shift: f64,
    /// Power 2
    pub poly_a: f64,
    /// Power 1
    pub poly_b: f64,
    /// Power 0
    pub poly_c: f64,

}

impl SinExpBasisTerm {
    // todo: Cplx if you use a complex exponential for sin.
    pub fn value(&self, posit: f64) -> f64 {
        self.sin_amp * (self.sin_freq * posit + self.sin_phase).sin()
            + self.decaying_exp_amp * (self.decaying_exp_rate * (posit - self.decaying_exp_shift)).exp() // todo dir of shift?
            + self.poly_a * posit.powi(2) + self.poly_b * posit + self.poly_c
    }

    /// Create a basis point here from neighboring values, in 1 dimension.
    /// todo: Cplx?
    pub fn from_neighbors(val_this: f64, val_prev: f64, val_next: f64, posit_this: f64, h: f64) -> Self {
        // Note that for decaying exponentials, the amplitude can be though of as used to define
        // at what value the equation takes at position = 0. Is this also a weight for balancing
        // with the sin term?

        // todo: If it's from an H basis, then I think this truly is teh analytic solution we
        // todo have for these exactly, at all positions. (?)

        //

        // todo: If we're fitting  a segment using interpolation of neighboring psi values,
        // todo is 3 points per axis good to find a fit?
        // todo: center using an additional offset?
        // todo: Do you only need 2 points since there are only 2 unknowns?
        // todo: Maybe you need 3 points, since there is a third unknown: The offset

        // todo: I think this is untenable for just complex exponentials.
        // todo another approach: Construct shiftless exponential from 2 pts
        // todo another: Take another stab at degree-2 polynomial interp.
        // todo: This brings up another Q: Can you treat dimensions independently?

        // 3 unknowns: B, λ, shift. 3 equations, from 3 points.
        // psi(x) = B·e^(-λ(x + shift))
        // psi(x + h) = B·e^(-λ(x + shift + h))
        // psi(x - h) = B·e^(-λ(x + shift - h))

        // A higher exp rate causes the equation to trend towards 0 more quickly as position increases.

        // https://math.stackexchange.com/questions/680646/get-polynomial-function-from-3-points
        let posit_prev = posit_this - h;
        let posit_next = posit_this + h;

        let poly_a_num = posit_prev * (val_next - val_this) + posit_this * (val_prev - val_next) + posit_next * (val_this - val_prev);
        let poly_a_denom = (posit_prev - posit_this) * (posit_prev - posit_next) * (posit_this - posit_next);

        let poly_a = poly_a_num / poly_a_denom;

        let poly_b = (val_this - val_prev) / (posit_this - posit_prev) - poly_a * (posit_prev + posit_this);

        let poly_c = val_prev - poly_a * posit_prev.powi(2) - poly_b * posit_prev;

        Self {
            poly_a,
            poly_b,
            poly_c,
            // decaying_exp_amp: B,
            // decaying_exp_rate: λ,
            // decaying_exp_shift: shift,
            ..Default::default() // No sin term for now.
        }
    }
}

/// Represents a local approximation of the wave function as a combination of terms that construct
/// a sin wave overlaid with a decaying exponential, at a given point.
/// Represents a point as A·sin(ωx + ϕ) + B·e^(-λx), for each of the 3 dimensions X, Y, Z.
///
/// todo note: Maybe polynomial term? It's in the H bases
/// Todo: Complex exponential to replace sin term? To replace *both* terms?
/// todo: Move this to another module if it works out
#[derive(Clone, Debug, Default)]
pub struct SinExpBasisPt {
    pub terms_x: SinExpBasisTerm,
    pub terms_y: SinExpBasisTerm,
    pub terms_z: SinExpBasisTerm,
}

impl SinExpBasisPt {
    // todo: Cplx if you use a complex exponential for sin.
    pub fn value(&self, posit: Vec3) -> f64 {
        self.terms_x.value(posit.x) + self.terms_y.value(posit.y) + self.terms_z.value(posit.z)
    }

    /// Create a basis point from equally-spaced neighbors, in 3 dimensions.
    /// Format of vals is (this, prev, next).
    /// todo: COmplex
    pub fn from_neighbors(vals_x: (f64, f64, f64),vals_y: (f64, f64, f64), vals_z: (f64, f64, f64), posit: Vec3, h: f64) -> Self {
        Self {
            terms_x: SinExpBasisTerm::from_neighbors(vals_x.0, vals_x.1, vals_x.2, posit.x, h),
            terms_y: SinExpBasisTerm::from_neighbors(vals_y.0, vals_y.1, vals_y.2, posit.y, h),
            terms_z: SinExpBasisTerm::from_neighbors(vals_z.0, vals_z.1, vals_z.2, posit.z, h),
        }
    }
}

/// todo: Cplx?
/// todo: Input args include indices, or not?
fn interp_from_sin_exp_basis(
    basis: &Arr3dBasis,
    pt: Vec3,
    i_tla: (usize, usize, usize), // todo: If you keep this apch, bundle
    i_tra: (usize, usize, usize),
    i_bla: (usize, usize, usize),
    i_bra: (usize, usize, usize),
    i_tlf: (usize, usize, usize),
    i_trf: (usize, usize, usize),
    i_blf: (usize, usize, usize),
    i_brf: (usize, usize, usize),
) -> f64 {
    let tla = basis[i_tla.0][i_tla.1][i_tla.2].value(pt);
    let tra = basis[i_tra.0][i_tra.1][i_tra.2].value(pt);
    let bla = basis[i_bla.0][i_bla.1][i_bla.2].value(pt);
    let bra = basis[i_bra.0][i_bra.1][i_bra.2].value(pt);
    
    let tlf = basis[i_tlf.0][i_tlf.1][i_tlf.2].value(pt);
    let trf = basis[i_trf.0][i_trf.1][i_trf.2].value(pt);
    let blf = basis[i_blf.0][i_blf.1][i_blf.2].value(pt);
    let brf = basis[i_brf.0][i_brf.1][i_brf.2].value(pt);

    tla * tla_weight +
        tra * tra_weight +
        bla * bla_weight +
        bra * bra_weight +
            tlf * tl_weight +
        trf * trf_weight +
        blf * blf_weight +
        brf * brf_weight
}
