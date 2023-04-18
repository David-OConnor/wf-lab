//! Experimental / shell module for finding basis functions given a potential,
//! or arrangement of nuclei.

// use crate::basis_wfs::{Basis, Sto};

use lin_alg2::f64::{Quaternion, Vec3};

// /// Returns a Vec of basis fns. Each has an associated position and orientation.
// pub(crate) fn find_basis_fns(nuclei: &[Nucleus]) -> Vec<(Basis, Vec3, Quaternion)> {
//
// }

use wf_lab::complex_nums::Cplx;
use crate::{
    basis_wfs::HOrbital,
    types::{Arr3dReal, Arr3dVec},
    wf_ops::{ħ},
};

// todo: Put back

//
// /// Experimental function to estimate bases for a wave function.
// /// See OneNote notebook for details.
// ///
// /// `dx` is for the grid
// ///
// /// todo: WIP
// pub fn find_bases(V: &Arr3dReal, bases: &[HOrbital], E: f64, mass: f64, grid_posits: &Arr3dVec, n: usize) {
//     // Let's start with a single x, y, and z, as we prototype. These values here are indices.
//     let x = 5;
//     let y = 5;
//     let z = 5;
//
//     // This term is a function of E and V.
//     let dx: f64 = 1.; // todo: find Dx. This may be a bit tricky given the irregular grid.
//     let f_e = 2. * mass / ħ.powi(2) * (E - V[x][y][z]) * dx.powi(2) + 6.;
//
//     for basis in bases {
//         for i in 0..n {
//             for j in 0..n {
//                 for k in 0..n {
//                     let posit = grid_posits[i][j][k];
//                 }
//             }
//         }
//     }
// }
//
// /// See OneNote. If this is 0, the solution is exact.
// pub fn compute_equality(V: &Arr3dReal, bases: &[HOrbital], E: f64, mass: f64, grid_posits: &Arr3dVec, n: usize) -> Cplx {
//     // Let's start with a single x, y, and z, as we prototype. These values here are indices.
//     let x = 5;
//     let y = 5;
//     let z = 5;
//
//     // This term is a function of E and V.
//     let dx: f64 = 1.; // todo: find Dx. This may be a bit tricky given the irregular grid.
//     let f_e = 2. * mass / ħ.powi(2) * (E - V[x][y][z]) * dx.powi(2) + 6.;
//
//     let mut result = Cplx::new_zero();
//
//     // todo: pick a point, or sum over all like this?
//
//     // todo: DRY C+P from basis_wfs finding psi'' diff
//
//     // todo: Is this effectively the same as your finding psi'' calc and meas, and scoring??!
//
//     // todo: Normalize each basis? Probably.
//
//
//     for basis in bases {
//
//         // todo: Do we just do an inner sum?
//         let mut result_this_basis = Cplx::new_zero(); //
//
//         for i in 0..n {
//             for j in 0..n {
//                 for k in 0..n {
//                     let posit_sample = grid_posits[i][j][k];
//
//                     let x_prev = Vec3::new(posit_sample.x - dx, posit_sample.y, posit_sample.z);
//                     let x_next = Vec3::new(posit_sample.x + dx, posit_sample.y, posit_sample.z);
//                     let y_prev = Vec3::new(posit_sample.x, posit_sample.y - dx, posit_sample.z);
//                     let y_next = Vec3::new(posit_sample.x, posit_sample.y + dx, posit_sample.z);
//                     let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - dx);
//                     let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + dx);
//
//                     let mut psi_x_prev = Cplx::new_zero();
//                     let mut psi_x_next = Cplx::new_zero();
//                     let mut psi_y_prev = Cplx::new_zero();
//                     let mut psi_y_next = Cplx::new_zero();
//                     let mut psi_z_prev = Cplx::new_zero();
//                     let mut psi_z_next = Cplx::new_zero();
//
//                     psi_x_prev = psi_x_prev / psi_norm_sqrt;
//                     psi_x_next = psi_x_next / psi_norm_sqrt;
//                     psi_y_prev = psi_y_prev / psi_norm_sqrt;
//                     psi_y_next = psi_y_next / psi_norm_sqrt;
//                     psi_z_prev = psi_z_prev / psi_norm_sqrt;
//                     psi_z_next = psi_z_next / psi_norm_sqrt;
//
//                     let term_a = f_e * basis.value(posit_sample) + basis.value(posit_sample) + psi_x_prev + psi_x_next +
//                         psi_y_prev + psi_y_next + psi_z_prev + psi_z_next;
//
//                     result += basis.weight * term_a;
//                 }
//             }
//         }
//     }
//
//     result
// }