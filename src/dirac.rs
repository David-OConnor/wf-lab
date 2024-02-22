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

use na::{Matrix2, Matrix4};
use nalgebra as na;

use crate::{
    complex_nums::{Cplx, IM},
    grid_setup,
    grid_setup::{Arr3d, Arr4d},
    iter_arr, iter_arr_4, util,
    wf_ops::M_ELEC,
};

// Matrix operators: alpha, beta, gamma. Gamma is 2x2. alpha and beta are (at least?) 4x4

const C0: Cplx = Cplx::new_zero();
const C1: Cplx = Cplx { real: 1., im: 0. };

#[derive(Clone, Copy, PartialEq)]
enum Component {
    T,
    X,
    Y,
    Z,
}

/// Todo: Figure out how to use this...
/// A 4-component spinor wave function.
#[derive(Clone, Default)]
pub struct PsiSpinor {
    /// Matter, spin α
    pub c0: Arr4d,
    /// Matter, spin β
    pub c1: Arr4d,
    /// Antimatter, spin α
    pub c2: Arr4d,
    /// Antimatter, spin β
    pub c3: Arr4d,
}

/// Ignores T; using E. viable? Will try.
#[derive(Clone, Default)]
pub struct PsiSpinor3D {
    /// Matter, spin α
    pub c0: Arr3d,
    /// Matter, spin β
    pub c1: Arr3d,
    /// Antimatter, spin α
    pub c2: Arr3d,
    /// Antimatter, spin β
    pub c3: Arr3d,
}

impl PsiSpinor3D {
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

impl PsiSpinor {
    pub fn differentiate(&self, component: Component, grid_spacing: f64) -> Self {
        let n = self.c0.len();
        let mut result = Self::default();

        // For use with our midpoint formula.
        let diff = grid_spacing / 2.;

        for (i, j, k, l) in iter_arr_4!(n) {
            match component {
                Component::T => {
                    result.c0[i][j][k][l] =
                        (self.c0[i + 1][j][k][l] - self.c0[i - 1][j][k][l]) / diff;
                    result.c1[i][j][k][l] =
                        (self.c1[i + 1][j][k][l] - self.c1[i - 1][j][k][l]) / diff;
                    result.c2[i][j][k][l] =
                        (self.c2[i + 1][j][k][l] - self.c2[i - 1][j][k][l]) / diff;
                    result.c3[i][j][k][l] =
                        (self.c3[i + 1][j][k][l] - self.c3[i - 1][j][k][l]) / diff;
                }
                Component::X => {
                    result.c0[i][j][k][l] =
                        (self.c0[i][j + 1][k][l] - self.c0[i][j - 1][k][l]) / diff;
                    result.c1[i][j][k][l] =
                        (self.c1[i][j + 1][k][l] - self.c1[i][j - 1][k][l]) / diff;
                    result.c2[i][j][k][l] =
                        (self.c2[i][j + 1][k][l] - self.c2[i][j - 1][k][l]) / diff;
                    result.c3[i][j][k][l] =
                        (self.c3[i][j + 1][k][l] - self.c3[i][j - 1][k][l]) / diff;
                }
                Component::Y => {
                    result.c0[i][j][k][l] =
                        (self.c0[i][j][k + 1][l] - self.c0[i][j][k - 1][l]) / diff;
                    result.c1[i][j][k][l] =
                        (self.c1[i][j][k + 1][l] - self.c1[i][j][k - 1][l]) / diff;
                    result.c2[i][j][k][l] =
                        (self.c2[i][j][k + 1][l] - self.c2[i][j][k - 1][l]) / diff;
                    result.c3[i][j][k][l] =
                        (self.c3[i][j][k + 1][l] - self.c3[i][j][k - 1][l]) / diff;
                }
                Component::Z => {
                    result.c0[i][j][k][l] =
                        (self.c0[i][j][k][l + 1] - self.c0[i][j][k][l - 1]) / diff;
                    result.c1[i][j][k][l] =
                        (self.c1[i][j][k][l + 1] - self.c1[i][j][k][l - 1]) / diff;
                    result.c2[i][j][k][l] =
                        (self.c2[i][j][k][l + 1] - self.c2[i][j][k][l - 1]) / diff;
                    result.c3[i][j][k][l] =
                        (self.c3[i][j][k][l + 1] - self.c3[i][j][k][l - 1]) / diff;
                }
            }
        }

        result
    }
}

impl Mul<Matrix4<Cplx>> for PsiSpinor {
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

impl Mul<Cplx> for &PsiSpinor {
    type Output = PsiSpinor;

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

impl Mul<Cplx> for PsiSpinor {
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

impl Add<&Self> for PsiSpinor {
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

impl Sub<&Self> for PsiSpinor {
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
