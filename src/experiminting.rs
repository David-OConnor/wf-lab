//! 2024-09-06

use lin_alg::f32::Vec3;
use crate::grid_setup::{Arr3dReal, Arr3dVec, new_data_real};

/// Given a set of atoms, find the charge density over 3D space at a given timestamp.
/// nuclei: (posit, charge)
pub fn get_electron_map(nuclei: Vec<(Vec3, f64)>, num_elecs: usize, grid: &Arr3dVec) -> Arr3dReal {
    let n = grid.len();

    let mut result = new_data_real(n);


    // The secret sauce here. How do we do this...

    result
}