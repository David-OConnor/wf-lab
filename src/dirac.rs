//! Scratchpad for Dirac-equation implementation.
//!
//! Relation of E and p for any particle: E^2 - phat^2 c^c = m^2 c^4

extern crate nalgebra as na;
use na::{Matrix2, Matrix4};


// Matrix operators: alpha, beta, gamma. Gamma is 2x2. alpha and beta are (at least?) 4x4

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

}