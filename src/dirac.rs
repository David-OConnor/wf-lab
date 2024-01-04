//! Scratchpad for Dirac-equation implementation.
//! Intent: See if this clarifies how to handle spin.
//!
//! Relation of E and p for any particle: E^2 - phat^2 c^c = m^2 c^4
//!
//! α_1^2 = α_2^2 = α_3^2 = β^2 = 1
//! α and β are 4x4 matrices. α = [0,σ, σ, 0]  β = [1, 0, 0, -1]

use na::{Matrix2, Matrix4};
use nalgebra as na;

use crate::complex_nums::{Cplx, IM};

// Matrix operators: alpha, beta, gamma. Gamma is 2x2. alpha and beta are (at least?) 4x4

const C0: Cplx = Cplx::new_zero();

#[rustfmt::skip]
fn a() {
    let gamma: Matrix2<f32> = Matrix2::new(
        0.0, 0.0,
        0.0, 0.0,
    );

    let beta: Matrix4<f32> = Matrix4::new(
        0.0, 0.0,0.0, 0.0,
        0.0, 0.0,0.0, 0.0,
        0.0, 0.0,0.0, 0.0,
        0.0, 0.0,0.0, 0.0,
    );

    // [Gamma matrices](https://en.wikipedia.org/wiki/Gamma_matrices)
    let gamma0: Matrix4<f32> = Matrix4::new(
        1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., -1., 0.,
        0., 0., 0., -1.
    );

    let gamma1: Matrix4<f32> = Matrix4::new(
        0., 0., 0., 1.,
        0., 0., 1., 0.,
        0., -1., 0., 0.,
        -1., 0., 0., 0.
    );

    let gamma2: Matrix4<Cplx> = Matrix4::new(
        C0, C0, C0, -IM,
        C0, C0, IM, C0,
        C0, IM, C0, C0,
        -IM, C0, C0, C0
    );

    let gamma3: Matrix4<f32> = Matrix4::new(
        1., 0., 1., 0.,
        0., 0., 0., -1.,
        -1., 0., 0., 0.,
        0., 1., 0., 0.
    );

    // The identity matrix.
    let gamma4: Matrix4<f32> = Matrix4::new(
        1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.
    );

    // γ5 is not a proper member of the gamma group.
    let gamma5: Matrix4<f32> = Matrix4::new(
        0., 0., 1., 0.,
        0., 0., 0., 1.,
        1., 0., 0., 0.,
        0., 1., 0., 0.
    );

    // [Pauli matrices](https://en.wikipedia.org/wiki/Pauli_matrices)
    let sigma1: Matrix2<f32> = Matrix2::new(
        0.0, 1.,
        1., 0.0,
    );

    // [Pauli matrices](https://en.wikipedia.org/wiki/Pauli_matrices)
    let sigma2: Matrix2<Cplx> = Matrix2::new(
        C0, -IM,
        IM, C0,
    );

    // [Pauli matrices](https://en.wikipedia.org/wiki/Pauli_matrices)
    let sigma3: Matrix2<f32> = Matrix2::new(
        1., 0.,
        0., -1.,
    );

    let alpha1: Matrix4<f32> = Matrix4::new(
        0., 0., sigma1.m11, sigma1.m12,
        0., 0., sigma1.m21, sigma1.m22,
        sigma1.m11, sigma1.m12, 0., 0.,
        sigma1.m21, sigma1.m22, 0., 0.
    );

    let alpha2: Matrix4<Cplx> = Matrix4::new(
        C0, C0, sigma2.m11, sigma2.m12,
        C0, C0, sigma2.m21, sigma2.m22,
        sigma2.m11, sigma2.m12, C0, C0,
        sigma2.m21, sigma2.m22, C0, C0
    );

    let alpha3: Matrix4<f32> = Matrix4::new(
        0., 0., sigma3.m11, sigma3.m12,
        0., 0., sigma3.m21, sigma3.m22,
        sigma3.m11, sigma3.m12, 0., 0.,
        sigma3.m21, sigma3.m22, 0., 0.
    );
    
    // etc for alpha 2 and 3.
    let beta: Matrix4<f32> = Matrix4::new(
        1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., -1., 0.,
        0., 0., 0., -1.
    );


}
