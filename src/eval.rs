//! This module contains code related to evlauting a WF's accuracy.

use lin_alg2::f64::Vec3;

use crate::{complex_nums::Cplx, basis_wfs::Basis, grid_setup::{Arr3d, Arr3dReal}, eigen_fns};

/// Score using the fidelity of psi'' calculated vs measured; |<psi_trial | psi_true >|^2.
/// This requires normalizing the wave functions we're comparing.
/// todo: Curretly not working.
/// todo: I don't think you can use this approach comparing psi''s with fidelity, since they're
/// todo not normalizsble.
/// todo: Perhaps this isn't working because these aren't wave functions! psi is a WF;
/// psi'' is not
// fn wf_fidelity(sfcs: &Surfaces) -> f64 {
fn _fidelity(psi_pp_calc: &Arr3d, psi_pp_meas: &Arr3d, n: usize) -> f64 {
    // "The accuracy should be scored by the fidelity of the wavefunction compared
    // to the true wavefunction. Fidelity is defined as |<psi_trial | psi_true >|^2.
    // For normalized states, this will always be bounded from above by 1.0. So it's
    // lower than 1.0 for an imperfect variational function, but is 1 if you are
    // able to exactly express it.""

    // For normalization.
    let mut norm_calc = Cplx::new_zero();
    let mut norm_meas = Cplx::new_zero();

    const SCORE_THRESH: f64 = 100.;

    // Create normalization const.
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                // norm_sq_calc += sfcs.psi_pp_calculated[i][j][k].abs_sq();
                // norm_sq_meas += sfcs.psi_pp_measured[i][j][k].abs_sq();
                // todo: .real is temp
                if psi_pp_calc[i][j][k].real.abs() < SCORE_THRESH
                    && psi_pp_meas[i][j][k].real.abs() < SCORE_THRESH
                {
                    norm_calc += psi_pp_calc[i][j][k];
                    norm_meas += psi_pp_meas[i][j][k];
                }
            }
        }
    }

    // Now that we have both wave functions and normalized them, calculate fidelity.
    let mut result = Cplx::new_zero();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                // todo: .reals here may be a kludge and not working with complex psi.

                // todo: LHS should be conjugated.
                if psi_pp_calc[i][j][k].real.abs() < SCORE_THRESH
                    && psi_pp_meas[i][j][k].real.abs() < SCORE_THRESH
                {
                    result += psi_pp_calc[i][j][k] / norm_calc.real * psi_pp_calc[i][j][k]
                        / norm_calc.real;
                }
            }
        }
    }

    result.abs_sq()
}


/// Score a wave function by comparing the least-squares sum of its measured and
/// calculated second derivaties.
pub fn score_wf(psi_pp_calc: &[Cplx], psi_pp_meas: &[Cplx]) -> f64 {
    let mut result = 0.;

    for i in 0..psi_pp_calc.len() {
        // todo: Check if either individual is outside a thresh?
        let diff = psi_pp_calc[i] - psi_pp_meas[i];
        // todo: Ideally, examine its derivative too, but I'm not sure how with
        // todo this approach. I guess use the analytic values?

        result += diff.abs_sq();
    }

    result / psi_pp_calc.len() as f64
}

// /// Score a wave function by comparing its estimate of total potential acting on it: This
// /// should be 0 at infinity.
// pub fn score_wf_from_V(V_total: &Arr3dReal) -> f64 {
//     V_total[0][0][0]
// }
//
// /// Score a wave function by comparing its estimate of total potential acting on it: This
// /// should be 0 at infinity.
// pub fn score_wf_from_V_analytic(bases: &[Basis], E: f64) -> f64 {
//     let compare_val = 9999.; // Far from the nuclei; should be zero.
//     let compare_pt = Vec3::new(compare_val, compare_val, compare_val);
//
//     let mut psi = Cplx::new_zero();
//     let mut psi_pp = Cplx::new_zero();
//     for basis in bases {
//         psi += basis.value(compare_pt);
//         psi_pp += basis.second_deriv(compare_pt);
//     }
//
//     eigen_fns::calc_V_on_psi(psi, psi_pp, E)
// }
