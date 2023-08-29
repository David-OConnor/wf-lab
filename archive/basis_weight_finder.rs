//! Experimental / shell module for finding basis functions given a potential,
//! or arrangement of nuclei.

// use std::collections::HashMap;

use itertools::Itertools;

use lin_alg2::f64::Vec3;
use wf_lab::types::{BasesEvaluated, BasesEvaluated1d};

use crate::{
    basis_wfs::Basis,
    eigen_fns, elec_elec, eval,
    grid_setup::{Arr3d, Arr3dReal, Arr3dVec, new_data},
    potential,
    types::{EvalDataPerElec, SurfacesPerElec, SurfacesShared},
    util,
    wf_ops::{self},
};

// Observation to explore: For Helium (what else?) Energy seems to be the value
// where psi'' calc has no part above 0, but barely. todo Can we use this?
// Is it a least-dampening phenomenon?

/// Adjust weights of coefficiants until score is minimized.
/// We use a gradient-descent approach to find local *score* minimum. (fidelity?)
/// We choose several start points to help find a global solution.
pub fn find_weights(
    charges_fixed: &Vec<(Vec3, f64)>,
    charges_elec: &mut [Arr3dReal],
    eval_data: &mut EvalDataPerElec,
    posits: &[Vec3],
    bases: &mut Vec<Basis>,
    bases_evaled: &mut BasesEvaluated1d,
    bases_evaled_charge: &mut Vec<Arr3d>,
    max_n: u16, // quantum number n
    V_from_nuclei: &[f64],
    V_from_elec: &mut [f64],
    grid_n_charge: usize,
    grid_posits_charge: &Arr3dVec,
    grid_n: usize,
) {
    wf_ops::initialize_bases(charges_fixed, bases, max_n);

    let norm = 1.; // todo temp! Figure this out.

    *bases_evaled = BasesEvaluated1d::new(bases, posits, norm);
    *bases_evaled_charge = wf_ops::arr_from_bases(bases, grid_posits_charge, grid_n_charge);

    // We use this 3D psi for calculating charge density.
    let mut psi_grid = new_data(grid_n_charge);

    // todo: Consider again using unweighted bases in your main logic. YOu removed it before
    // todo because it was bugged when you attempted it.

    // let mut basis_wfs_weighted: Vec<Arr3d> = Vec::new();

    // Approach: Take a handleful of approaches, eg evenly-spaced; maybe 2-3 per dimension
    // to start. From each, using gradient-descent to find a global minima of score.Then use the best of these.

    // These points iterate through all bases, so the total number of points
    // is initial_points^n_weights. These are where we begin our gradient-descents. More points allows
    // for better differentiation of global vs local minima (of WF score);

    // We will score the wave function, and along each dimension, in order to find the partial
    // derivatives. We will then follow the gradients to victory (?)
    // let initial_sample_weights = util::linspace((weight_min, weight_max), weight_vals_per_iter);

    let sample_weights = util::linspace((-0.8, 0.8), 6);
    // todo: sample_differnet xis.
    // let sample_xis = util::linspace((-1., 10), 20);

    let weight_permutations: Vec<Vec<f64>> = sample_weights
        .into_iter()
        .permutations(bases.len())
        .collect();

    // todo: Scoring in general. Consider measuring slope at each point; not just value.

    // let mut scores = HashMap::new();
    let mut scores = Vec::new();

    for weights in &weight_permutations {
        scores.push(score_weight_set(
            bases,
            bases_evaled,
            bases_evaled_charge,
            eval_data,
            &mut psi_grid,
            V_from_elec,
            &mut charges_elec[0],
            V_from_nuclei,
            &weights,
            posits,
            grid_posits_charge,
            grid_n,
            grid_n_charge,
        ))
    }

    let mut best_score = 9999.;
    let mut best_i = 0;
    for (i, score) in scores.iter().enumerate() {
        if *score < best_score {
            best_score = *score;
            best_i = i;
        }
    }
    println!(
        "Best weight set. Score: {} weights: {:?}",
        best_score, weight_permutations[best_i]
    )

    // wf_ops::update_wf_fm_bases_1d(bases, bases_evaled, eval_data, grid_n, None);

    // eval_data.score = eval::score_wf(&eval_data.psi_pp_calc, &eval_data.psi_pp_meas)
}
//
// fn gradient_descent() {
//     // Infinitessimal weight change, used for assessing derivatives.
//     const D_WEIGHT: f64 = 0.0001;
//
//     const NUM_DESCENTS: usize = 3; // todo
//     let descent_rate = 0.1; // todo? Factor for gradient descent based on the vector.
//

// Here: Isolated descent algo. Possibly put in a sep fn.
// This starts with a weight=1 n=1 orbital at each electron.

// Update the weights stored in bases with what we've set.
// We update other things like the grid-based values elsewhere after running this.
// for (i, basis) in bases.iter_mut().enumerate() {
//     *basis.weight_mut() = current_point[i];
// }
//     // For now, let's use a single starting point, and gradient-descent from it.
//     let mut current_point = vec![0.; bases.len()];
//     // for i in 0..charges_fixed.len() {
//     for i in 0..1 {
//         current_point[i] = 1.;
//     }

//     for _descent_num in 0..NUM_DESCENTS {
//         // This is our gradient. d_score / d_weight
//         let mut diffs = vec![0.; bases.len()];
//
//         // Iterate through each basis; calcualate a score using the same parameters as above,
//         // with the exception of a slightly small weight for the basis.
//         for (i_basis, _basis) in bases.iter().enumerate() {
//             // Scores from a small change along this dimension. basis = dimension
//             // Midpoint.
//             let mut point_shifted_left = current_point.clone();
//             let mut point_shifted_right = current_point.clone();
//
//             point_shifted_left[i_basis] -= D_WEIGHT;
//             point_shifted_right[i_basis] += D_WEIGHT;
//
//             let scores: Vec<f64> = [point_shifted_left, point_shifted_right]
//                 .iter()
//                 .map(|weights| {
//                     score_weight_set(
//                         bases,
//                         bases_evaled,
//                         bases_evaled_charge,
//                         eval_data,
//                         &mut psi_grid,
//                         V_from_elec,
//                         &mut charges_elec[0],
//                         V_from_nuclei,
//                         weights,
//                         posits,
//                         grid_posits_charge,
//                         grid_n,
//                         grid_n_charge,
//                     )
//                 })
//                 .collect();
//
//             let score_prev = scores[0];
//             let score_next = scores[1];
//
//             diffs[i_basis] = (score_next - score_prev) / (2. * D_WEIGHT);
//         }
//
//         // Now that we've computed our gradient, shift down it to the next point.
//         for i in 0..bases.len() {
//             // Direction: Diff is current pt score minus score from a slightly
//             // lower value of a given basis. If it's positive, it means the score gets better
//             // (lower) with a smaller weight, so *reduce* weight accordingly.
//
//             // Leave the n=1 weight for one of the fixed-charges fixed to a value of 1.
//             // Note that this may preclude solutions other than the ground state.
//             // This should help avoid the system seeking 0 on all weights.
//             if i == 0 {
//                 continue;
//             }
//
//             current_point[i] -= diffs[i] * descent_rate;
//         }
//
//         println!("\n\nDiffs: {:?}\n", diffs);
//         println!("Score: {:?}", eval_data.score);
//         println!("current pt: {:?}", current_point);
//     }
// }

/// Helper for finding weight gradient descent. Updates psi and psi'', calculate E, and score.
/// This function doesn't mutate any of the data.
fn score_weight_set(
    bases: &[Basis],
    bases_evaled: &BasesEvaluated1d,
    bases_evaled_charge: &[Arr3d],
    eval_data: &mut EvalDataPerElec,
    psi_grid: &mut Arr3d,
    V_from_elec: &mut [f64],
    charges_elec: &mut Arr3dReal,
    V_from_nuclei: &[f64],
    weights: &[f64],
    grid_posits_1d: &[Vec3],
    grid_posits_charge: &Arr3dVec,
    grid_n: usize,
    grid_n_charge: usize,
) -> f64 {
    // This updates psi, E, and both psi''s
    wf_ops::update_wf_fm_bases_1d(eval_data, bases_evaled, grid_n, weights, None);

    // Update psi on the grid
    wf_ops::mix_bases_no_diffs(psi_grid, bases_evaled_charge, grid_n, weights);

    update_elec_V(
        eval_data,
        V_from_elec,
        charges_elec,
        psi_grid,
        V_from_nuclei,
        grid_posits_1d,
        grid_posits_charge,
        grid_n,
        grid_n_charge,
    );

    // println!("E: {:?}", eval_data.E);

    eval_data.score = eval::score_wf(&eval_data.psi_pp_calc, &eval_data.psi_pp_meas);

    eval_data.score
}

/// Update electron data.
/// todo: hard-coded for 2 identical (opposite-spin) electrons, for now.
/// Note that we are using a single electron here, and using it
/// for both the charge and acted-on roles.
fn update_elec_V(
    eval_data: &mut EvalDataPerElec,
    V_from_elec: &mut [f64],
    charges_elec: &mut Arr3dReal,
    psi_grid_charge: &Arr3d,
    V_from_nuclei: &[f64],
    grid_posits_1d: &[Vec3],
    grid_posits_charge: &Arr3dVec,
    grid_n: usize,
    grid_n_charge: usize,
) {
    // todo: Confirm that the psi etc used by this fn are on the charge grid.
    // let mut psi_charge_grid = new_data(grid_n_charge);
    // wf_ops::mix_bases_no_diffs(
    //     psi: &mut psi_charge_grid,
    //     &state.bases[ae],
    //     &state.bases_evaluated_charge[ae],
    //     state.grid_n_charge,
    //     None,
    // );

    // Create charge using the trial weights.
    elec_elec::update_charge_density_fm_psi(charges_elec, psi_grid_charge, grid_n_charge);

    // Create a potential from this charge.
    potential::create_V_from_an_elec(
        V_from_elec,
        charges_elec,
        grid_posits_1d,
        grid_posits_charge,
        grid_n,
        grid_n_charge,
    );

    // let elec_v = V_from_elec0.clone();

    // This step is a variant on `update_V_acting_on_elec`, for our specialized
    // 2-identical-electrons setup here.
    for i in 0..grid_n {
        eval_data.V_acting_on_this[i] = V_from_nuclei[i] + V_from_elec[i];
    }

    for i in 0..grid_n {
        eval_data.psi_pp_calc[i] = eigen_fns::find_ψ_pp_calc(
            eval_data.psi.on_pt[i],
            eval_data.V_acting_on_this[i],
            eval_data.E,
        );
    }
}

// todo: Experimental; here be dragons. See onenote.
use crate::eigen_fns::{KE_COEFF, KE_COEFF_INV};
// /// todo: We are likely looking at solving a linear system of equations, eg with a matrix.
// pub fn find_coeffs(
//     xi: &[f64],
//     V: f64,
//     E: f64,
//
// ) -> Vec<f64> {
//
// }
