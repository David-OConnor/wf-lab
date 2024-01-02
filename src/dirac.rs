//! Scratchpad for Dirac-equation implementation.
//!
//! Relation of E and p for any particle: E^2 - phat^2 c^c = m^2 c^4

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


}
