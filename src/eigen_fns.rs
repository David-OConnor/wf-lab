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
    complex_nums::Cplx,
    elec_elec::WaveFunctionMultiElec,
    types::{Arr3d, Arr3dReal},
    wf_ops::{self, ħ, Q_ELEC, Q_PROT, K_C},
    types::Arr3dVec,
};

use lin_alg2::f64::Vec3;

pub const KE_COEFF: f64 = -(ħ * ħ) / (2. & wf_ops::M_ELEC);
pub const KE_COEFF_INV: f64 = 1. / KE_COEFF;

/// Calcualte psi'', calculated from psi, and E. Note that the V term used must include both
/// electron-electron interactions, and electron-proton interactions.
/// At a given i, j, k.
///
/// This solves, analytically, the eigenvalue equation for the Hamiltonian operator.
///
/// Hψ = Eψ. -ħ^2/2m * ψ'' + Vψ = Eψ. ψ'' = [(E - V) / (-ħ^2/2m)] ψ
pub fn find_ψ_pp_calc(psi: &Arr3d, V: &Arr3dReal, E: f64, i: usize, j: usize, k: usize) -> Cplx {
    KE_COEFF_INV  * (E - V[i][j][k]) * psi[i][j][k]

    // (E - V)= psi'' / (psi * KE_COEFF)
}

/// Returns the *sum of psi'' from the 2 electrons*.
/// Note: V must be calculatd appropriately from the 3 relevant terms.
/// todo: MOre general one, not limited to to elecs
pub fn find_ψ_pp_calc_2_elec(psi_joint: &WaveFunctionMultiElec, V: &Arr3dReal, E: f64, i: usize, j: usize, k: usize) -> Cplx {
    psi[i][j][k] * (E - V[i][j][k]) * KE_COEFF_INV
}

/// Experimental function to calculate E from a 2-electron WF.
/// todo: How to
pub fn find_E_2_elec_at_pt(psi_joint: Cplx, psi_pp_0: Cplx, psi_pp_1: Cplx, posit_nuc: Vec3, posit_elec_0: Vec3, posit_elec_1: Vec3) {
    // Note: This uses a factor of 4 due to m = 2.
    const KE_COEFF_2_ELEC: f64 = - (ħ * ħ) / (4. * wf_ops::M_ELEC);

    let diff_e0_nuc = posit_elec_0 - posit_nuc;
    let diff_e1_nuc = posit_elec_1 - posit_nuc;
    let diff_e0_e1 = posit_elec_0 - posit_elec_1;

    let r0_nuc = diff_e0_nuc.magnitude();
    let r1_nuc = diff_e1_nuc.magnitude();
    let r0_1 = diff_e0_e1.magnitude();

    const C: f64 = K_C * Q_ELEC * Q_PROT;

    let V = C * -1. / r0_nuc - 1. / r1_nuc + 1. / r0_1;

    KE_COEFF * (psi_pp_0 + psi_pp_1) / psi_joint + V;
}

pub fn find_E_2_elec(psi_joint: &WaveFunctionMultiElec, psi_pp_0: &Arr3d, psi_pp_0: &Arr3d, grid_posits: &Arr3dVec, grid_n: usize) {
    // Note: This uses a factor of 4 due to m = 2.
    const KE_COEFF_2_ELEC_INV: f64 = - (ħ * ħ) / (4. * wf_ops::M_ELEC);

    for i0 in 0..grid_n {
        for j0 in 0..grid_n {
            for k0 in 0..grid_n {
                let f0 = grid_posits[i0][j0][k0];

                for i1 in 0..grid_n {
                    for j1 in 0..grid_n {
                        for k1 in 0..grid_n {
                            let r1 = grid_posits[i1][j1][k1];

                            // let E_at_posit = find_E_2_elec_at_pt(
                            //     psi_joint[]?
                            //     psi_joint[i][j][k],
                            //     psi_pp_a[i0][j0][k0],
                            //     psi_bb_b[i1][j1][k1],
                            //     r0, r1
                            // );
                        }
                    }
                }
            }
        }
    }
}

/// Todo WIP/probably wrong approach
/// L^2 ψ = ħ^2 l(l+1)ψ
pub fn find_spin(psi: &Arr3d, l: f64, L: f64, i: usize, j: usize, k: usize) -> Cplx {
    Cplx::new_zero()
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
