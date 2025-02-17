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

use lin_alg::{f64::Vec3, complex_nums::{Cplx, IM}};

use crate::{
    core_calcs::elec_elec::WaveFunctionMultiElec,
    grid_setup::{Arr3d, Arr3dVec},
    types::DerivativesSingle,
    wf_ops::{self, ħ, K_C, Q_ELEC, Q_PROT},
};

pub const KE_COEFF: f64 = -(ħ * ħ) / (2. * wf_ops::M_ELEC);
pub const KE_COEFF_INV: f64 = 1. / KE_COEFF;

/// Calcualte psi'', calculated from psi, and E. Note that the V term used must include both
/// electron-electron interactions, and electron-proton interactions.
/// V here is a potential field from the nucleus, so multiply it by the electron's
/// charge to find potential energy
/// At a given i, j, k.
///
/// This solves, analytically, the eigenvalue equation for the Hamiltonian operator.
///
/// Hψ = Eψ. -ħ^2/2m * ψ'' + Vψ = Eψ. ψ'' = [(E - V) / (-ħ^2/2m)] ψ
pub fn find_ψ_pp_calc(ψ: Cplx, V: f64, E: f64) -> Cplx {
    // Note that V input is potential field; we get potential energy by multiplying it
    // by the charge being acted on (the electron)
    // todo: Why q_elec here??
    ψ * (E - V * Q_ELEC) * KE_COEFF_INV
}

/// Returns the *sum of psi'' from the 2 electrons*.
/// Note: V must be calculatd appropriately from the 3 relevant terms.
/// todo: MOre general one, not limited to to elecs
pub fn find_ψ_pp_calc_2_elec(
    // psi_joint: &WaveFunctionMultiElec,
    psi_joint: Cplx,
    // V: &Arr3dReal,
    V: f64,
    // todo: Try with a single V here. If not working, try with V from parts, eg posits as below.
    E: f64,
    // i: usize,
    // j: usize,
    // k: usize,
) -> Cplx {
    psi_joint * (E - V * Q_ELEC) * KE_COEFF_INV
}

/// Experimental function to calculate E from a 2-electron WF, of Helium.
/// todo: How to
pub fn find_E_2_elec_at_pt(
    psi_joint: Cplx,
    // These psi''s are calculated by holding the other electron coordinate[s] constant,
    // and comparing to +/- dx values of the coordinate in question.
    psi_pp_0: Cplx,
    psi_pp_1: Cplx,
    posit_nuc: Vec3,
    posit_elec_0: Vec3,
    posit_elec_1: Vec3,
) -> f64 {
    // Note: This uses a factor of 4 due to m = 2.
    const KE_COEFF_2_ELEC: f64 = -(ħ * ħ) / (4. * wf_ops::M_ELEC);

    let r0_nuc = (posit_elec_0 - posit_nuc).magnitude();
    let r1_nuc = (posit_elec_1 - posit_nuc).magnitude();
    let r0_1 = (posit_elec_0 - posit_elec_1).magnitude();

    const C_PROT_ELEC: f64 = K_C * Q_PROT * Q_ELEC;
    const C_ELEC_ELEC: f64 = K_C * Q_ELEC * Q_ELEC;

    let V_energy = C_PROT_ELEC * (1. / r0_nuc + 1. / r1_nuc) + C_ELEC_ELEC / r0_1;

    // Alternative approach that may be more computationally efficience since we've already
    // calculated this.
    // todo: Once the above approach works,compare the 2. Or just print them out and confirm
    // todo they're the same.
    // todo: Possibly a no-go due to the 3 terms?
    // let V_energy = Q_ELEC * V_acting_on_this_elec;

    // Real since it's an eigenvalue of a Hermitian operator.
    ((psi_pp_0 + psi_pp_1) / psi_joint * KE_COEFF_2_ELEC + V_energy.into()).real
}

pub fn _find_E_2_elec(
    psi_joint: &WaveFunctionMultiElec,
    psi_pp_0: &Arr3d,
    psi_pp_1: &Arr3d,
    grid_posits: &Arr3dVec,
    grid_n: usize,
) {
    // Note: This uses a factor of 4 due to m = 2.
    const KE_COEFF_2_ELEC_INV: f64 = -(ħ * ħ) / (4. * wf_ops::M_ELEC);

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

// todo: Separate module for this work on V-based evaluation?

/// Calculate the V that must be acting on a given psi, and its (known to be accurate, eg numerical
/// differentiation) derivative.
pub fn calc_V_on_psi(psi: Cplx, psi_pp: Cplx, E: f64) -> f64 {
    // psi''/psi is always real, due to being an eigenvalue of a Hermitian operator.
    KE_COEFF * (psi_pp / psi).real - E
}

/// A mirror of `calc_V_on_psi`.
/// note: This is identical to calc_V_on_psi.
pub fn calc_E_on_psi(psi: Cplx, psi_pp: Cplx, V: f64) -> f64 {
    calc_V_on_psi(psi, psi_pp, V)
}

/// Alternative API, taking advantage of analytic psi''/psi
/// psipp_div_psi is always real; Hermitian.
pub fn _calc_V_on_psi2(psi_pp_div_psi: f64, E: f64) -> f64 {
    KE_COEFF * psi_pp_div_psi - E
}

/// Alternative API, taking advantage of analytic psi''/psi
/// psipp_div_psi is always real; Hermitian.
pub fn _calc_E_on_psi2(psi_pp_div_psi: f64, V: f64) -> f64 {
    _calc_V_on_psi2(psi_pp_div_psi, V)
}

/// todo: An experiment.
/// Similar to how to calculate V or psi'' from the H eigenfunction from a trial psi, calculate
/// y using the L_z eigenfunction, and x.
///
/// // todo: Something is off. Why does quantum number m appear here, but n doesn't appear for Schrodinger?
/// todo: That might be OK, with mħ filling the role of E.
///
/// L_z = mħ = -iħ (x d/dy - y d/dx). -y d/dx = mħ/(-iħ) - x d/dy.
/// y = (-mħ/(-iħ) + x d/dy) / d/dx
/// y = (-i m + x d/dy) / d/dx
pub(crate) fn calc_y_fm_Lz(d: &DerivativesSingle, m: i16, x: f64) -> Cplx {
    (-IM * m as f64 + d.dy * x) / d.dx
}
