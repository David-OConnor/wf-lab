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

use na::{Matrix2, Matrix4};
use nalgebra as na;

use crate::{
    complex_nums::{Cplx, IM},
    grid_setup::Arr3d,
};

// Matrix operators: alpha, beta, gamma. Gamma is 2x2. alpha and beta are (at least?) 4x4

const C0: Cplx = Cplx::new_zero();
const C1: Cplx = Cplx::new(1., 0.);

/// Todo: Figure out how to use this...
/// A 4-component spinor wave function.
pub struct PsiSpinor {
    pub a: Arr3d,
    pub b: Arr3d,
    pub c: Arr3d,
    pub d: Arr3d,
}

impl PsiSpinor {
    /// Multiply with γ on the left: γψ
    pub fn multiply_with_gamma(&self, gamma: Matrix4<Cplx>) -> Self {}

    /// Take the (numeric) first derivative of this wave function. Can be called multiple times
    /// to calculate higher derivatives.
    pub fn deriv(&self) -> Self {}
}

// todo: Cplx?
#[rustfmt::skip]
/// Return a gamma matrix. [Gamma matrices](https://en.wikipedia.org/wiki/Gamma_matrices)
// fn gamma(mu: u8) -> Matrix4<f64> {
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
        
        
        // 0 => Matrix4::new(
        //     1., 0., 0., 0.,
        //     0., 1., 0., 0.,
        //     0., 0., -1., 0.,
        //     0., 0., 0., -1.
        // ),
        // 1 => Matrix4::new(
        //     0., 0., 0., 1.,
        //     0., 0., 1., 0.,
        //     0., -1., 0., 0.,
        //     -1., 0., 0., 0.
        // ),
        // // todo: SOrt out how to mix cplx/real
        // // 2 => Matrix4::new(
        // //     C0, C0, C0, -IM,
        // //     C0, C0, IM, C0,
        // //     C0, IM, C0, C0,
        // //     -IM, C0, C0, C0
        // // ),
        // 3 => Matrix4::new(
        //     1., 0., 1., 0.,
        //     0., 0., 0., -1.,
        //     -1., 0., 0., 0.,
        //     0., 1., 0., 0.
        // ),
        // // The identity matrix.
        // 4 => Matrix4::new(
        //     1., 0., 0., 0.,
        //     0., 1., 0., 0.,
        //     0., 0., 1., 0.,
        //     0., 0., 0., 1.
        // ),
        // // γ5 is not a proper member of the gamma group.
        // 5 => Matrix4::new(
        //     0., 0., 1., 0.,
        //     0., 0., 0., 1.,
        //     1., 0., 0., 0.,
        //     0., 1., 0., 0.
        // ),
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

/// Calculate the Dirac equation with form (i γ^μ ∂_μ - m) ψ = 0.
/// todo: How far does mu range?
/// todo: Adopt tensor shortcut fns as you have in the Gravity sim?
pub fn dirac_lhs(psi: &PsiSpinor, m: i8) -> Arr3d {
    // todo: Solve numerically.
    let psi_p = psi.clone();
    let psi_p2 = psi.clone();
    let psi_p3 = psi.clone();
    let psi_p4 = psi.clone();

    // todo: Sort out how this matrix multiplication works...
    // IM * (gamma(0) * psi + )
    psi.clone() // todo temp
}
