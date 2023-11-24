//! This module contains code related to evlauting a WF's accuracy.

use crate::{complex_nums::Cplx, grid_setup::Arr3d, iter_arr};

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
    for (i, j, k) in iter_arr!(n) {
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

    // Now that we have both wave functions and normalized them, calculate fidelity.
    let mut result = Cplx::new_zero();

    for (i, j, k) in iter_arr!(n) {
        // todo: .reals here may be a kludge and not working with complex psi.

        // todo: LHS should be conjugated.
        if psi_pp_calc[i][j][k].real.abs() < SCORE_THRESH
            && psi_pp_meas[i][j][k].real.abs() < SCORE_THRESH
        {
            result += psi_pp_calc[i][j][k] / norm_calc.real * psi_pp_calc[i][j][k] / norm_calc.real;
        }
    }

    result.abs_sq()
}

/// Score a wave function by comparing the least-squares sum of its measured and
/// calculated second derivaties.
pub fn score_wf_from_psi_pp(psi_pp_calc: &[Cplx], psi_pp_meas: &[Cplx]) -> f64 {
    let mut result = 0.;

    for i in 0..psi_pp_calc.len() {
        // todo: Check if either individual is outside a thresh?
        let diff = psi_pp_calc[i] - psi_pp_meas[i];
        result += diff.abs_sq();
    }

    result / psi_pp_calc.len() as f64
}

/// Score a wave function by comparing the least-squares sum of the known potential acting onit,
/// and the estimated potential from the wave function.
pub fn score_wf_from_V(V_known: &[f64], V_from_psi: &[f64]) -> f64 {
    let mut result = 0.;

    for i in 0..V_known.len() {
        // todo: Check if either individual is outside a thresh?
        let diff = V_known[i] - V_from_psi[i];
        result += diff.abs();
    }

    result / V_known.len() as f64
}
