//! This module contains code to calculate expected values of parameters
//! like psi'' and psi', based on eigenfunctions.
//!
//!
//! Observables and their eigenfunctions:
//! Energy. Hψ = Eψ. H = -ħ^2/2m ∇^2 + V. Eigenvalue: E.
//!
//! Momentum (linear). P ψ = p ψ. P = -iħ∇. Eigenvalue: p.
//! Todo: maybe we divide momentum into its 3 components.
//! P_x ψ = p_x ψ. P = -iħ d/dx. Eigenvalue: p_x
//! P_y ψ = p_y ψ. P = -iħ d/dy. Eigenvalue: p_y
//! P_z ψ = p_z ψ. P = -iħ d/dz. Eigenvalue: p_z
//!
//! Momentum (angular). L^2 ψ = ħ^2 l(l+1) ψ (uses l quantum number directly. Not what we want?)
//!
//! - L_x = y p_z - p_y z = -iħ(y d/dz - d/dy z). Eigenvalue: p_x?
//! - L_y = z p_x - p_z x = -iħ(z d/dx - d/dz x)
//! - L_z = x p_y - p_x y = -iħ(x d/dy - d/dx y)
//!
//! Position? Xψ = xψ. X = x??
//!
//! L^2 = d_psi_d_x^2 + d_psi_d_y^2 + d_psi_d_z^2

use crate::{
    complex_nums::{Cplx, IM},
    types::{Arr3d, Arr3dReal},
    wf_ops::{self, ħ},
};

pub const KE_COEFF: f64 = -2. * wf_ops::M_ELEC / (ħ * ħ);
pub const KE_COEFF_INV: f64 = 1. / KE_COEFF;

/// Calcualte psi'', calculated from psi, and E. Note that the V term used must include both
/// electron-electron interactions, and electron-proton interactions.
/// At a given i, j, k.
///
/// This solves, analytically, the eigenvalue equation for the Hamiltonian operator.
///
/// Hψ = Eψ. -ħ^2/2m * ψ'' + Vψ = Eψ. ψ'' = [(E - V) / (-ħ^2/2m)] ψ
pub fn find_ψ_pp_calc(psi: &Arr3d, V: &Arr3dReal, E: f64, i: usize, j: usize, k: usize) -> Cplx {
    // ψ(r1, r2) = ψ_a(r1)ψb(r2), wherein we are combining probabilities.
    // fermions: two identical fermions cannot occupy the same state.
    // ψ(r1, r2) = A[ψ_a(r1)ψ_b(r2) - ψ_b(r1)ψ_a(r2)]
    psi[i][j][k] * (E - V[i][j][k]) * KE_COEFF
}

// /// Calcualte dψ/dx, from ψ and L_x.
// /// L_y = z p_x - p_z x = -iħ(z d/dx - d/dz x)
// /// L_z = x p_y - p_x y = -iħ(x d/dy - d/dx y)
// ///
// /// dψ/dx = ((L_y / -iħ) + x dψ/dz)) / z
// ///
// /// dψ/dx = ((L_z / -iħ) - x dψ/dy)) / y
// ///
// /// L^2 = d_psi_d_x^2 + d_psi_d_y^2 + d_psi_d_z^2
// pub fn find_dψ_dx_calc(psi: &Arr3d, L_y: f64, i: usize, j: usize, k: usize) -> Cplx {
//     const COEFF: Cplx = Cplx { real: 0., im: -ħ };
//
//     // todo: This is tricky due to L being a vector quantity, with inter-dependent components.
//     // let val_a = ((L_y / COEFF) + x * d_psi_d_z) / z;
//     // let val_b = ((L_z / COEFF) - x * d_psi_d_y) / y;
//
//     psi[i][j][k] * p * COEFF
// }


// todo: $$
// -i \hbar \int  \psi^* \left( y \frac{\partial \psi}{\partial z} -  \frac{\partial (z \psi)}{\partial y} \right)dx dy dz
// $$

// ?