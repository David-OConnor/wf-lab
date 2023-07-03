//! Experimental / shell module for finding basis functions given a potential,
//! or arrangement of nuclei.

use crate::{
    basis_wfs::Basis,
    eval,
    grid_setup::{Arr3d, Arr3dVec},
    types::{EvalDataPerElec, SurfacesPerElec, SurfacesShared},
    wf_ops::{self, BasesEvaluated, BasesEvaluated1d},
};

use lin_alg2::f64::Vec3;

/// Adjust weights of coefficiants until score is minimized.
/// We use a gradient-descent approach to find local *score* minimum. (fidelity?)
/// We choose several start points to help find a global solution.
pub fn find_weights(
    charges_fixed: &Vec<(Vec3, f64)>,
    eval_data: &mut EvalDataPerElec,
    posits: &[Vec3],
    bases: &mut Vec<Basis>,
    bases_evaled: &mut BasesEvaluated1d,
    bases_evaled_charge: &mut Vec<Arr3d>,
    max_n: u16, // quantum number n
    grid_n_charge: usize,
    grid_posits_charge: &Arr3dVec,
    grid_n: usize,
) {
    wf_ops::initialize_bases(charges_fixed, bases, None, max_n);

    let norm = 1.; // todo temp! Figure this out.

    *bases_evaled = BasesEvaluated1d::new(bases, posits, norm);

    *bases_evaled_charge = wf_ops::arr_from_bases(bases, grid_posits_charge, grid_n_charge);

    // Infinitessimal weight change, used for assessing derivatives.
    const D_WEIGHT: f64 = 0.001;

    const NUM_DESCENTS: usize = 10; // todo
    let descent_rate = 2.; // todo? Factor for gradient descent based on the vector.

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

    // Here: Isolated descent algo. Possibly put in a sep fn.
    // This starts with a weight=1 n=1 orbital at each electron.

    // todo: Find what's diff each run.

    // For now, let's use a single starting point, and gradient-descent from it.
    let mut current_point = vec![0.; bases.len()];
    for i in 0..charges_fixed.len() {
        current_point[i] = 1.;
    }

    for _descent_num in 0..NUM_DESCENTS {
        // This is our gradient. d_score / d_weight
        let mut diffs = vec![0.; bases.len()];

        // Iterate through each basis; calcualate a score using the same parameters as above,
        // with the exception of a slightly small weight for the basis.
        for (i_basis, _basis) in bases.iter().enumerate() {

            // Scores from a small change along this dimension. basis = dimension
            // Midpoint.
            let mut point_shifted_left = current_point.clone();
            let mut point_shifted_right = current_point.clone();

            point_shifted_left[i_basis] -= D_WEIGHT;
            point_shifted_right[i_basis] += D_WEIGHT;

            let score_prev =
                score_weight_set(bases, eval_data, bases_evaled, &point_shifted_left, grid_n);

            let score_next =
                score_weight_set(bases, eval_data, bases_evaled, &point_shifted_right, grid_n);

            diffs[i_basis] = (score_next - score_prev) / (2. * D_WEIGHT);
        }

        // Now that we've computed our gradient, shift down it to the next point.
        for i in 0..bases.len() {
            // Direction: Diff is current pt score minus score from a slightly
            // lower value of a given basis. If it's positive, it means the score gets better
            // (lower) with a smaller weight, so *reduce* weight accordingly.

            // Leave the n=1 weight for one of the fixed-charges fixed to a value of 1.
            // Note that this may preclude solutions other than the ground state.
            // This should help avoid the system seeking 0 on all weights.
            if i == 0 {
                continue;
            }

            current_point[i] -= diffs[i] * descent_rate;
        }

        println!("\n\nDiffs: {:?}\n", diffs);
        println!("Score: {:?}", eval_data.score);
        println!("current pt: {:?}", current_point);
    }

    println!("\n\nResult: {:?}", current_point);

    // Update the weights stored in bases with what we've set.
    // We update other things like the grid-based values elsewhere after running this.
    for (i, basis) in bases.iter_mut().enumerate() {
        *basis.weight_mut() = current_point[i];
    }

    // wf_ops::update_wf_fm_bases_1d(bases, bases_evaled, eval_data, grid_n, None);

    // eval_data.score = eval::score_wf(&eval_data.psi_pp_calc, &eval_data.psi_pp_meas)
}

/// Helper for finding weight gradient descent. Updates psi and psi'', calculate E, and score.
/// This function doesn't mutate any of the data.
pub fn score_weight_set(
    bases: &[Basis],
    eval_data: &mut EvalDataPerElec,
    bases_evaled: &BasesEvaluated1d,
    weights: &[f64],
    grid_n: usize,
) -> f64 {
    wf_ops::update_wf_fm_bases_1d(
        bases,
        bases_evaled,
        eval_data,
        grid_n,
        Some(weights),
    );

    // println!("E: {:?}", eval_data.E);

    eval_data.score = eval::score_wf(&eval_data.psi_pp_calc, &eval_data.psi_pp_meas);

    eval_data.score
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