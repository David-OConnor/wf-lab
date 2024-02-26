//! Scratchpad for Dirac-equation implementation.
//! Intent: See if this clarifies how to handle spin.
//!
//! Relation of E and p̂ for any particle: E^2 - p̂^2 c^c = m^2 c^4
//!
//! //! α_1^2 = α_2^2 = α_3^2 = β^2 = 1
//! α and β are 4x4 matrices. α = [0,σ, σ, 0]  β = [1, 0, 0, -1]
//!
//! iħ * dψ/dt = H_dirac ψ
//! H_dirac = cα (p̂ + e/C A) + βmc^2 + V(r)   V(r) ~= -e^2/r = (-e) * ϕ(r)
//! ϕ is the scaler potential, eg like V.
//!
//! ψ = [X, ϕ] X: Pauli.  [energy eigenstate. X and ϕ are 2-component spinors]
//! HX = EX.
//!
//! p̂ = -iħ∇
//! p̂^2 = -ħ^2∇^2
//! p̂^4 = ħ^4∇^4
//!
//! H = p̂^2/2m + V(r) [H0: Schrodinger eq] - p̂^4/(8m^3c^2) [(dH)_relativistic] +
//! 1/(2m^2c^2)(1/r)(dV/dr)S L [spin orbit coupling; S is spin angular momentum; L is orbital
//! angular momentum] + ħ^2 / (8m^2c^2) ∇^2 V [dH Darwin]
//!
//! Perterbation theory...
//!
//! https://arxiv.org/ftp/arxiv/papers/1709/1709.04338.pdf
//!
//!
//! "The density of electronic charge, which is
//! proportional to the square of Schroedinger's scalar amplitude function for the hydrogen atom, or
//! precisely a product of a function and its complex conjugate as ψ* ψ to take account of the
//! complex character, hence became proportional to a scalar product of Dirac's vectorial amplitude
//! function for each state with its complex conjugate vector."
//!
//! (For Darwin term) Vtilde(r) = V(r) + 1/10 * hbar^2/(m^2 c^2) * nalba^2 V
//!
//! Dirac psi amplitude: 4 components: 2 large; 2 small (vector)

use core::ops::{Add, Mul, Sub};

use lin_alg::f64::Vec3;
use na::{Matrix2, Matrix4};
use nalgebra as na;

use crate::{
    basis_wfs::Sto,
    complex_nums::{Cplx, IM},
    grid_setup,
    grid_setup::{Arr3d, Arr4d},
    iter_arr, iter_arr_4,
    num_diff::{H, H_2},
};

// Matrix operators: alpha, beta, gamma. Gamma is 2x2. alpha and beta are (at least?) 4x4

const C0: Cplx = Cplx::new_zero();
const C1: Cplx = Cplx { real: 1., im: 0. };

pub type SpinorVec = Vec<Vec<Vec<Vec<SpinorTypeB>>>>;
pub type SpinorVec3 = Vec<Vec<Vec<SpinorTypeB>>>;

#[derive(Clone)]
pub struct BasisSpinor {
    pub c0: Sto,
    pub c1: Sto,
    pub c2: Sto,
    pub c3: Sto,
}

impl BasisSpinor {
    pub(crate) fn get(&self, comp: ComponentPsi) -> &Sto {
        match comp {
            ComponentPsi::C0 => &self.c0,
            ComponentPsi::C1 => &self.c1,
            ComponentPsi::C2 => &self.c2,
            ComponentPsi::C3 => &self.c3,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Component {
    T,
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ComponentPsi {
    C0,
    C1,
    C2,
    C3,
}

/// Todo: Figure out how to use this...
/// A 4-component spinor wave function.
/// /// Ordering: Psi component, index
#[derive(Clone, Default)]
pub struct Spinor {
    /// Matter, spin α
    pub c0: Arr4d,
    /// Matter, spin β
    pub c1: Arr4d,
    /// Antimatter, spin α
    pub c2: Arr4d,
    /// Antimatter, spin β
    pub c3: Arr4d,
}

/// Ordering: Psi component, index
#[derive(Clone, Default)]
pub struct Spinor3 {
    /// Matter, spin α
    pub c0: Arr3d,
    /// Matter, spin β
    pub c1: Arr3d,
    /// Antimatter, spin α
    pub c2: Arr3d,
    /// Antimatter, spin β
    pub c3: Arr3d,
}

impl Spinor3 {
    pub(crate) fn get(&self, comp: ComponentPsi) -> &Arr3d {
        match comp {
            ComponentPsi::C0 => &self.c0,
            ComponentPsi::C1 => &self.c1,
            ComponentPsi::C2 => &self.c2,
            ComponentPsi::C3 => &self.c3,
        }
    }

    pub(crate) fn get_mut(&mut self, comp: ComponentPsi) -> &mut Arr3d {
        match comp {
            ComponentPsi::C0 => &mut self.c0,
            ComponentPsi::C1 => &mut self.c1,
            ComponentPsi::C2 => &mut self.c2,
            ComponentPsi::C3 => &mut self.c3,
        }
    }
}

/// Ordering: Index, psi component
#[derive(Clone, Default)]
pub struct SpinorTypeB {
    pub c0: Cplx,
    pub c1: Cplx,
    pub c2: Cplx,
    pub c3: Cplx,
}

impl Spinor3 {
    pub fn new(n: usize) -> Self {
        let data = grid_setup::new_data(n);

        Self {
            c0: data.clone(),
            c1: data.clone(),
            c2: data.clone(),
            c3: data,
        }
    }
}

/// Ordering, outside in: μ, psi component, index
#[derive(Clone, Default)]
pub struct SpinorDerivs {
    pub dt: Spinor,
    pub dx: Spinor,
    pub dy: Spinor,
    pub dz: Spinor,
}

impl SpinorDerivs {
    // pub fn new(spinor: &Spinor, grid_spacing: f64) -> Self {
    //     Self {
    //         dt: spinor.differentiate(spinor, Component::T, grid_spacing),
    //         dx: spinor.differentiate(spinor, Component::X, grid_spacing),
    //         dy: spinor.differentiate(spinor, Component::Y, grid_spacing),
    //         dz: spinor.differentiate(spinor, Component::Z, grid_spacing),
    //     }
    // }
}

/// Ordering, outside in: da, psi component, index
#[derive(Clone, Default)]
pub struct SpinorDerivs3 {
    pub dx: Spinor3,
    pub dy: Spinor3,
    pub dz: Spinor3,
}

impl SpinorDerivs3 {
    // pub fn new(spinor: &Spinor3, grid_spacing: f64) -> Self {
    //     Self {
    //         dx: spinor.differentiate(spinor, Component::X, grid_spacing),
    //         dy: spinor.differentiate(spinor, Component::Y, grid_spacing),
    //         dz: spinor.differentiate(spinor, Component::Z, grid_spacing),
    //     }
    // }
}

/// Ordering, outside in: index, dμ, psi component,
#[derive(Clone, Default)]
pub struct SpinorDerivsTypeB {
    pub dt: SpinorTypeB,
    pub dx: SpinorTypeB,
    pub dy: SpinorTypeB,
    pub dz: SpinorTypeB,
}

/// Ordering, outside in: index, da, psi component,
#[derive(Clone, Default)]
pub struct SpinorDerivsTypeB3 {
    pub dx: SpinorTypeB,
    pub dy: SpinorTypeB,
    pub dz: SpinorTypeB,
}

#[derive(Default, Clone)]
pub struct SpinorDerivsTypeE3Inner {
    pub dx: Cplx,
    pub dy: Cplx,
    pub dz: Cplx,
}

/// Ordering, outside in: index, psi component, da
#[derive(Clone, Default)]
pub struct SpinorDerivsTypeE3 {
    pub c0: SpinorDerivsTypeE3Inner,
    pub c1: SpinorDerivsTypeE3Inner,
    pub c2: SpinorDerivsTypeE3Inner,
    pub c3: SpinorDerivsTypeE3Inner,
}

impl SpinorDerivsTypeE3 {
    /// Modified version of `num_diff:DerivativesSingle::from_bases()`.
    pub(crate) fn from_bases(posit_sample: Vec3, bases: &[BasisSpinor]) -> Self {
        let mut result = Self::default();

        let x_prev = Vec3::new(posit_sample.x - H, posit_sample.y, posit_sample.z);
        let x_next = Vec3::new(posit_sample.x + H, posit_sample.y, posit_sample.z);
        let y_prev = Vec3::new(posit_sample.x, posit_sample.y - H, posit_sample.z);
        let y_next = Vec3::new(posit_sample.x, posit_sample.y + H, posit_sample.z);
        let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - H);
        let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + H);

        // for basis in bases {
        for (psi_comp, comp) in [
            (&mut result.c0, ComponentPsi::C0),
            (&mut result.c1, ComponentPsi::C1),
            (&mut result.c2, ComponentPsi::C2),
            (&mut result.c3, ComponentPsi::C3),
        ] {
            let mut psi_x_prev = Cplx::new_zero();
            let mut psi_x_next = Cplx::new_zero();
            let mut psi_y_prev = Cplx::new_zero();
            let mut psi_y_next = Cplx::new_zero();
            let mut psi_z_prev = Cplx::new_zero();
            let mut psi_z_next = Cplx::new_zero();

            for basis in bases {
                psi_x_prev += basis.get(comp).value(x_prev);
                psi_x_next += basis.get(comp).value(x_next);
                psi_y_prev += basis.get(comp).value(y_prev);
                psi_y_next += basis.get(comp).value(y_next);
                psi_z_prev += basis.get(comp).value(z_prev);
                psi_z_next += basis.get(comp).value(z_next);
            }
            psi_comp.dx = (psi_x_next - psi_x_prev) / H_2;
            psi_comp.dy = (psi_y_next - psi_y_prev) / H_2;
            psi_comp.dz = (psi_z_next - psi_z_prev) / H_2;
        }

        result
    }
}

/// Ordering, outside in: dμ, index, psi component
#[derive(Clone)]
pub struct SpinorDerivsTypeC {
    pub dt: SpinorVec,
    pub dx: SpinorVec,
    pub dy: SpinorVec,
    pub dz: SpinorVec,
}

/// Ordering, outside in: da, index, psi component
#[derive(Clone)]
pub struct SpinorDerivsTypeC3 {
    pub dx: SpinorVec3,
    pub dy: SpinorVec3,
    pub dz: SpinorVec3,
}

/// Ordering, outside in: psi component, dμ, index
/// Note that this is similar to `types::Derivatives`
#[derive(Clone, Default)]
pub struct SpinorDerivsTypeDInner {
    pub dt: Arr4d,
    pub dx: Arr4d,
    pub dy: Arr4d,
    pub dz: Arr4d,
}

/// Ordering, outside in: psi component, da, index
/// Note that this is similar to `types::Derivatives`
#[derive(Clone, Default)]
pub struct SpinorDerivsTypeDInner3 {
    pub dx: Arr3d,
    pub dy: Arr3d,
    pub dz: Arr3d,
}

impl SpinorDerivsTypeDInner3 {
    pub fn new(n: usize) -> Self {
        let data = grid_setup::new_data(n);

        Self {
            dx: data.clone(),
            dy: data.clone(),
            dz: data,
        }
    }
}

/// Ordering, outside in: psi component, dμ, index,
/// Note that this is similar to `types::Derivatives`, but with an additional outer layer
/// for the psi components.
#[derive(Clone)]
pub struct SpinorDerivsTypeD {
    pub c0: SpinorDerivsTypeDInner,
    pub c1: SpinorDerivsTypeDInner,
    pub c2: SpinorDerivsTypeDInner,
    pub c3: SpinorDerivsTypeDInner,
}

/// Ordering, outside in: psi component, da, index,
/// Note that this is similar to `types::Derivatives`, but with an additional outer layer
/// for the psi components.
#[derive(Clone, Default)]
pub struct SpinorDerivsTypeD3 {
    pub c0: SpinorDerivsTypeDInner3,
    pub c1: SpinorDerivsTypeDInner3,
    pub c2: SpinorDerivsTypeDInner3,
    pub c3: SpinorDerivsTypeDInner3,
}

impl SpinorDerivsTypeD3 {
    /// Update self using psi values. (Replaces existing content.)
    pub fn from_bases(&mut self, posit_sample: Vec3, bases: &[BasisSpinor]) {}

    pub fn new(n: usize) -> Self {
        let data = SpinorDerivsTypeDInner3::new(n);

        Self {
            c0: data.clone(),
            c1: data.clone(),
            c2: data.clone(),
            c3: data,
        }
    }

    pub(crate) fn get(&self, comp: ComponentPsi) -> &SpinorDerivsTypeDInner3 {
        match comp {
            ComponentPsi::C0 => &self.c0,
            ComponentPsi::C1 => &self.c1,
            ComponentPsi::C2 => &self.c2,
            ComponentPsi::C3 => &self.c3,
        }
    }

    pub(crate) fn get_mut(&mut self, comp: ComponentPsi) -> &mut SpinorDerivsTypeDInner3 {
        match comp {
            ComponentPsi::C0 => &mut self.c0,
            ComponentPsi::C1 => &mut self.c1,
            ComponentPsi::C2 => &mut self.c2,
            ComponentPsi::C3 => &mut self.c3,
        }
    }
}

impl Spinor {
    pub fn differentiate(&self, deriv: &mut Self, component: Component, grid_spacing: f64) {
        // For use with our midpoint formula.
        let n = self.c0.len();
        let diff = grid_spacing / 2.;

        for (i, j, k, l) in iter_arr_4!(n) {
            match component {
                Component::T => {
                    deriv.c0[i][j][k][l] =
                        (self.c0[i + 1][j][k][l] - self.c0[i - 1][j][k][l]) / diff;
                    deriv.c1[i][j][k][l] =
                        (self.c1[i + 1][j][k][l] - self.c1[i - 1][j][k][l]) / diff;
                    deriv.c2[i][j][k][l] =
                        (self.c2[i + 1][j][k][l] - self.c2[i - 1][j][k][l]) / diff;
                    deriv.c3[i][j][k][l] =
                        (self.c3[i + 1][j][k][l] - self.c3[i - 1][j][k][l]) / diff;
                }
                Component::X => {
                    deriv.c0[i][j][k][l] =
                        (self.c0[i][j + 1][k][l] - self.c0[i][j - 1][k][l]) / diff;
                    deriv.c1[i][j][k][l] =
                        (self.c1[i][j + 1][k][l] - self.c1[i][j - 1][k][l]) / diff;
                    deriv.c2[i][j][k][l] =
                        (self.c2[i][j + 1][k][l] - self.c2[i][j - 1][k][l]) / diff;
                    deriv.c3[i][j][k][l] =
                        (self.c3[i][j + 1][k][l] - self.c3[i][j - 1][k][l]) / diff;
                }
                Component::Y => {
                    deriv.c0[i][j][k][l] =
                        (self.c0[i][j][k + 1][l] - self.c0[i][j][k - 1][l]) / diff;
                    deriv.c1[i][j][k][l] =
                        (self.c1[i][j][k + 1][l] - self.c1[i][j][k - 1][l]) / diff;
                    deriv.c2[i][j][k][l] =
                        (self.c2[i][j][k + 1][l] - self.c2[i][j][k - 1][l]) / diff;
                    deriv.c3[i][j][k][l] =
                        (self.c3[i][j][k + 1][l] - self.c3[i][j][k - 1][l]) / diff;
                }
                Component::Z => {
                    deriv.c0[i][j][k][l] =
                        (self.c0[i][j][k][l + 1] - self.c0[i][j][k][l - 1]) / diff;
                    deriv.c1[i][j][k][l] =
                        (self.c1[i][j][k][l + 1] - self.c1[i][j][k][l - 1]) / diff;
                    deriv.c2[i][j][k][l] =
                        (self.c2[i][j][k][l + 1] - self.c2[i][j][k][l - 1]) / diff;
                    deriv.c3[i][j][k][l] =
                        (self.c3[i][j][k][l + 1] - self.c3[i][j][k][l - 1]) / diff;
                }
            }
        }
    }
}

// todo: DRY with above
impl Spinor3 {
    pub fn differentiate(&self, deriv: &mut Self, component: Component, grid_spacing: f64) {
        // For use with our midpoint formula.
        let diff = grid_spacing / 2.;
        let n = self.c0.len();

        for (i, j, k) in iter_arr!(n) {
            match component {
                Component::T => panic!("T is not avail on a 3D Spinor"),
                Component::X => {
                    deriv.c0[i][j][k] = (self.c0[i + 1][j][k] - self.c0[i - 1][j][k]) / diff;
                    deriv.c1[i][j][k] = (self.c1[i + 1][j][k] - self.c1[i - 1][j][k]) / diff;
                    deriv.c2[i][j][k] = (self.c2[i + 1][j][k] - self.c2[i - 1][j][k]) / diff;
                    deriv.c3[i][j][k] = (self.c3[i + 1][j][k] - self.c3[i - 1][j][k]) / diff;
                }
                Component::Y => {
                    deriv.c0[i][j][k] = (self.c0[i][j + 1][k] - self.c0[i][j - 1][k]) / diff;
                    deriv.c1[i][j][k] = (self.c1[i][j + 1][k] - self.c1[i][j - 1][k]) / diff;
                    deriv.c2[i][j][k] = (self.c2[i][j + 1][k] - self.c2[i][j - 1][k]) / diff;
                    deriv.c3[i][j][k] = (self.c3[i][j + 1][k] - self.c3[i][j - 1][k]) / diff;
                }
                Component::Z => {
                    deriv.c0[i][j][k] = (self.c0[i][j][k + 1] - self.c0[i][j][k - 1]) / diff;
                    deriv.c1[i][j][k] = (self.c1[i][j][k + 1] - self.c1[i][j][k - 1]) / diff;
                    deriv.c2[i][j][k] = (self.c2[i][j][k + 1] - self.c2[i][j][k - 1]) / diff;
                    deriv.c3[i][j][k] = (self.c3[i][j][k + 1] - self.c3[i][j][k - 1]) / diff;
                }
            }
        }
    }
}

impl Mul<Matrix4<Cplx>> for Spinor {
    type Output = Self;

    /// Multiply with γ on the left: γψ
    fn mul(self, rhs: Matrix4<Cplx>) -> Self::Output {
        let n = self.c0.len();
        let mut result = self.clone();

        // for (i, j, k) in iter_arr!(n) {
        for (i, j, k, l) in iter_arr_4!(n) {
            // Code simplifiers
            let a = self.c0[i][j][k][l];
            let b = self.c1[i][j][k][l];
            let c = self.c2[i][j][k][l];
            let d = self.c3[i][j][k][l];

            // todo: Confirm this indexing is in the correct order.
            // result.t[i][j][k] =
            //     rhs[(0, 0)] * a + rhs[(0, 1)] * b + rhs[(0, 2)] * c + rhs[(0, 3)] * d;
            // result.x[i][j][k] =
            //     rhs[(1, 0)] * a + rhs[(1, 1)] * b + rhs[(1, 2)] * c + rhs[(1, 3)] * d;
            // result.y[i][j][k] =
            //     rhs[(2, 0)] * a + rhs[(2, 1)] * b + rhs[(2, 2)] * c + rhs[(2, 3)] * d;
            // result.z[i][j][k] =
            //     rhs[(3, 0)] * a + rhs[(3, 1)] * b + rhs[(3, 2)] * c + rhs[(3, 3)] * d;

            // todo?
            result.c0[i][j][k][l] =
                rhs[(0, 0)] * a + rhs[(0, 1)] * b + rhs[(0, 2)] * c + rhs[(0, 3)] * d;
            result.c1[i][j][k][l] =
                rhs[(1, 0)] * a + rhs[(1, 1)] * b + rhs[(1, 2)] * c + rhs[(1, 3)] * d;
            result.c2[i][j][k][l] =
                rhs[(2, 0)] * a + rhs[(2, 1)] * b + rhs[(2, 2)] * c + rhs[(2, 3)] * d;
            result.c3[i][j][k][l] =
                rhs[(3, 0)] * a + rhs[(3, 1)] * b + rhs[(3, 2)] * c + rhs[(3, 3)] * d;
        }

        result
    }
}

impl Mul<Cplx> for &Spinor {
    type Output = Spinor;

    fn mul(self, rhs: Cplx) -> Self::Output {
        let n = self.c0.len();
        let mut result = self.clone();

        for (i, j, k, l) in iter_arr_4!(n) {
            result.c0[i][j][k][l] *= rhs;
            result.c1[i][j][k][l] *= rhs;
            result.c2[i][j][k][l] *= rhs;
            result.c3[i][j][k][l] *= rhs;
        }

        result
    }
}

impl Mul<Cplx> for Spinor {
    type Output = Self;

    fn mul(self, rhs: Cplx) -> Self::Output {
        let n = self.c0.len();
        let mut result = self.clone();

        for (i, j, k, l) in iter_arr_4!(n) {
            result.c0[i][j][k][l] *= rhs;
            result.c1[i][j][k][l] *= rhs;
            result.c2[i][j][k][l] *= rhs;
            result.c3[i][j][k][l] *= rhs;
        }

        result
    }
}

impl Add<&Self> for Spinor {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let n = self.c0.len();
        let mut result = self.clone();

        for (i, j, k, l) in iter_arr_4!(n) {
            result.c0[i][j][k][l] += rhs.c0[i][j][k][l];
            result.c1[i][j][k][l] += rhs.c1[i][j][k][l];
            result.c2[i][j][k][l] += rhs.c2[i][j][k][l];
            result.c3[i][j][k][l] += rhs.c3[i][j][k][l];
        }

        result
    }
}

impl Sub<&Self> for Spinor {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        let n = self.c0.len();
        let mut result = self.clone();

        for (i, j, k, l) in iter_arr_4!(n) {
            result.c0[i][j][k][l] -= rhs.c0[i][j][k][l];
            result.c1[i][j][k][l] -= rhs.c1[i][j][k][l];
            result.c2[i][j][k][l] -= rhs.c2[i][j][k][l];
            result.c3[i][j][k][l] -= rhs.c3[i][j][k][l];
        }

        result
    }
}

#[rustfmt::skip]
/// Return a gamma matrix. [Gamma matrices](https://en.wikipedia.org/wiki/Gamma_matrices)
fn gamma(mu: u8) -> Matrix4<Cplx> {
    match mu {
        0 => Matrix4::new(
            C1, C0, C0, C0,
            C0, C1, C0, C0,
            C0, C0, -C1, C0,
            C0, C0, C0, -C1
        ),
        1 => Matrix4::new(
            C0, C0, C0, C1,
            C0, C0, C1, C0,
            C0, -C1, C0, C0,
            -C1, C0, C0, C0
        ),
        2 => Matrix4::new(
            C0, C0, C0, -IM,
            C0, C0, IM, C0,
            C0, IM, C0, C0,
            -IM, C0, C0, C0
        ),
        3 => Matrix4::new(
            C1, C0, C1, C0,
            C0, C0, C0, -C1,
            -C1, C0, C0, C0,
            C0, C1, C0, C0
        ),
        // The identity matrix.
        4 => Matrix4::new(
            C1, C0, C0, C0,
            C0, C1, C0, C0,
            C0, C0, C1, C0,
            C0, C0, C0, C1
        ),
        // γ5 is not a proper member of the gamma group.
        5 => Matrix4::new(
            C0, C0, C1, C0,
            C0, C0, C0, C1,
            C1, C0, C0, C0,
            C0, C1, C0, C0
        ),
        _ => panic!("Invalid gamma matrix; must be 0-5."),
    }
}

#[rustfmt::skip]
fn a() {
    let gamma: Matrix2<f64> = Matrix2::new(
        0.0, 0.0,
        0.0, 0.0,
    );

    let beta: Matrix4<f64> = Matrix4::new(
        0.0, 0.0,0.0, 0.0,
        0.0, 0.0,0.0, 0.0,
        0.0, 0.0,0.0, 0.0,
        0.0, 0.0,0.0, 0.0,
    );

    // [Pauli matrices](https://en.wikipedia.org/wiki/Pauli_matrices)
    let sigma1: Matrix2<f64> = Matrix2::new(
        0.0, 1.,
        1., 0.0,
    );

    let sigma2: Matrix2<Cplx> = Matrix2::new(
        C0, -IM,
        IM, C0,
    );

    let sigma3: Matrix2<f64> = Matrix2::new(
        1., 0.,
        0., -1.,
    );

    // Note: The `m11` syntax works for `Matrix4<f64>`, but not <Cplx>.
    let alpha1: Matrix4<f64> = Matrix4::new(
        0., 0., sigma1[(1, 1)], sigma1[(1, 2)],
        0., 0., sigma1[(2, 1)], sigma1[(2, 2)],
        sigma1[(1, 1)], sigma1[(1, 2)], 0., 0.,
        sigma1[(2, 1)], sigma1[(2, 2)], 0., 0.
    );

    let alpha2: Matrix4<Cplx> = Matrix4::new(
        C0, C0, sigma2[(1, 1)], sigma2[(1, 2)],
        C0, C0, sigma2[(2, 1)], sigma2[(2, 2)],
        sigma2[(1, 1)], sigma2[(1, 2)], C0, C0,
        sigma2[(2, 1)], sigma2[(2, 2)], C0, C0
    );

    let alpha3: Matrix4<f64> = Matrix4::new(
        0., 0., sigma3[(1, 1)], sigma3[(1, 2)],
        0., 0.,sigma3[(2, 1)], sigma3[(2, 2)],
        sigma3[(1, 1)], sigma3[(1, 2)], 0., 0.,
        sigma3[(2, 1)], sigma3[(2, 2)], 0., 0.
    );

    let beta: Matrix4<f64> = Matrix4::new(
        1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., -1., 0.,
        0., 0., 0., -1.
    );
}

/// Calculate what psi should be, based on its derivatives, E, and V
/// todo: Other forms, ie this rearranged too
/// This is similar to the functions found in the `eigen_fns` module.
pub fn calc_psi(psi_calc: &mut Spinor3, derivs: &SpinorDerivsTypeD3, E: [f64; 4], V: [f64; 4]) {
    let n = psi_calc.c0.len();

    let dt0 = -IM * (E[0] - V[0]);
    let dt1 = -IM * (E[1] - V[1]);
    let dt2 = -IM * (E[2] - V[2]);
    let dt3 = -IM * (E[3] - V[3]);

    // for (i, j, k, l) in iter_arr_4!(n) {
    for (i, j, k) in iter_arr!(n) {
        // Code simplifiers
        let c0 = &derivs.c0;
        let c1 = &derivs.c1;
        let c2 = &derivs.c2;
        let c3 = &derivs.c3;

        let c0_dx = c0.dx[i][j][k];
        let c0_dy = c0.dy[i][j][k];
        let c0_dz = c0.dz[i][j][k];

        let c1_dx = c1.dx[i][j][k];
        let c1_dy = c1.dy[i][j][k];
        let c1_dz = c1.dz[i][j][k];

        let c2_dx = c2.dx[i][j][k];
        let c2_dy = c2.dy[i][j][k];
        let c2_dz = c2.dz[i][j][k];

        let c3_dx = c3.dx[i][j][k];
        let c3_dy = c3.dy[i][j][k];
        let c3_dz = c3.dz[i][j][k];

        psi_calc.c0[i][j][k] = dt0 - c3_dx + IM * c3_dy - c2_dz;
        psi_calc.c1[i][j][k] = dt1 - c2_dx - IM * c2_dy + c3_dz;
        psi_calc.c2[i][j][k] = -dt2 + c1_dx - IM * c1_dy + c0_dz;
        psi_calc.c3[i][j][k] = -dt3 + c0_dx + IM * c0_dy - c1_dz;
    }
}

// todo: Put back.
// /// Calculate the left-hand-side of the Dirac equation, of form (iħ γ^μ ∂_μ - mc) ψ = 0.
// /// We assume ħ = c = 1.
// /// todo: Adopt tensor shortcut fns as you have in the Gravity sim?
// /// // "rememer tho that hbar omega= E"
// pub fn dirac_lhs(psi: &PsiSpinor, grid_spacing: f64) {
//     let part0 = psi.differentiate(Component::T, grid_spacing) * gamma(0);
//     let part1 = psi.differentiate(Component::X, grid_spacing) * gamma(1);
//     let part2 = psi.differentiate(Component::Y, grid_spacing) * gamma(2);
//     let part3 = psi.differentiate(Component::Z, grid_spacing) * gamma(3);
//
//     (part0 - part1 - part2 - part3) * IM - psi * Cplx::from_real(M_ELEC);
// }
//
// todo: PUut back
// /// See Exploring the Wf, part 9 (OneNote)
// /// Calculate the expected psi_p, given psi. From the Dirac equation, with gamma matrices included.
// /// todo: Also the reverse, A/R
// pub fn psi_from_psi_p(d_t: &PsiSpinor, d_x: &PsiSpinor, d_y: &PsiSpinor, d_z: &PsiSpinor) -> PsiSpinor {
//     let mut result = PsiSpinor::default();
//
//     result.c0 = IM * (d_t.c0 + d_t.c1 - d_t.c2 - d_t.c3);
//     result.c1 = IM * -(d_x.c3 + d_x.c2 - d_x.c1 - d_x.c0);
//     result.c2 = -(-d_y.c3 + d_y.c2 + d_y.c1 - d_y.c0);
//     result.c3 = IM * -(d_z.c2 - d_z.c3 - d_z.c0 + d_z.c1);
//
//     result
// }

// /// See Exploring the Wf, part 9 (OneNote)
// /// Calculate the expected psi_p, given psi. From the Dirac equation, with gamma matrices included.
// /// todo: Also the reverse, A/R
// pub fn psi_p_from_psi(psi_p: &PsiSpinor) -> PsiSpinor {
//     let mut result = PsiSpinor::default();
//
//     result.t = IM * (psi_p.t);
//
//     result
// }
