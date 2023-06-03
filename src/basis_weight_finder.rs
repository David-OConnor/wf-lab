//! Experimental / shell module for finding basis functions given a potential,
//! or arrangement of nuclei.

use crate::{
    basis_wfs::Basis,
    eval,
    types::{SurfacesPerElec, SurfacesShared},
    wf_ops::{self, BasisWfsUnweighted},
};

use lin_alg2::f64::Vec3;

/// Adjust weights of coefficiants until score is minimized.
/// We use a gradient-descent approach to find local *score* minimum. (fidelity?)
/// We choose several start points to help find a global solution.
pub fn find_weights(
    charges_fixed: &Vec<(Vec3, f64)>,
    bases: &mut Vec<Basis>,
    bases_visible: &mut Vec<bool>,
    basis_wfs_unweighted: &mut BasisWfsUnweighted,
    surfaces_shared: &SurfacesShared,
    surfaces_per_elec: &mut SurfacesPerElec,
    max_n: u16, // quantum number n
    grid_n: usize,
) {
    wf_ops::initialize_bases(charges_fixed, bases, bases_visible, max_n);

    *basis_wfs_unweighted = BasisWfsUnweighted::new(&bases, &surfaces_shared.grid_posits, grid_n);

    // Infinitessimal weight change, used for assessing derivatives.
    const D_WEIGHT: f64 = 0.01;

    const NUM_DESCENTS: usize = 5; // todo
    let mut descent_rate = 2.; // todo? Factor for gradient descent based on the vector.

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
    // current_point[2] = 0.6;
    // current_point[3] = -0.5;

    // For reasons not-yet determined, we appear to need to run these after initializing the weights,
    // even though they're include din the scoring algo. All 3 seem to be required.

    for _descent_num in 0..NUM_DESCENTS {
        // todo: This appears to be required, even though we set it in scores.
        // todo: Still not sur ewhy
        wf_ops::update_wf_fm_bases(
            bases,
            &basis_wfs_unweighted,
            surfaces_per_elec,
            grid_n,
            Some(&current_point),
        );

        // println!("Psi Before: {} {} {}", surfaces_per_elec.psi.on_pt[10][10][10],
        //          surfaces_per_elec.psi.on_pt[1][10][15],
        //          surfaces_per_elec.psi.on_pt[3][4][6]);

        // let score_this = score_weight_set(
        //     bases,
        //     *E,
        //     &surfaces_per_elec,
        //     grid_n,
        //     &basis_wfs_unweighted,
        //     &current_point,
        // );

        // println!("\n\nScore this: {:?}", score_this);
        // println!("Current weight: {:?}", current_point);

        // This is our gradient.
        let mut diffs = vec![0.; bases.len()];

        for n in 1..max_n + 1 {
            // Iterate through each basis; calcualate a score using the same parameters as above,
            // with the exception of a slightly small weight for the basis.
            for (i_basis, _basis) in bases.iter().enumerate() {
                // todo: Sloppy
                if _basis.n() != n {
                    continue;
                }

                // Scores from a small change along this dimension. basis = dimension
                // Midpoint.
                let mut point_shifted_left = current_point.clone();
                let mut point_shifted_right = current_point.clone();

                point_shifted_left[i_basis] -= D_WEIGHT;
                point_shifted_right[i_basis] += D_WEIGHT;

                let score_prev = score_weight_set(
                    bases,
                    &surfaces_per_elec,
                    grid_n,
                    &basis_wfs_unweighted,
                    &point_shifted_left,
                );

                let score_next = score_weight_set(
                    bases,
                    &surfaces_per_elec,
                    grid_n,
                    &basis_wfs_unweighted,
                    &point_shifted_right,
                );

                // println!("Score prev: {:?}", score_prev);

                // dscore / d_weight.
                // diffs[i_basis] = (score_this - score_prev) / D_WEIGHT;
                diffs[i_basis] = (score_next - score_prev) / (2. * D_WEIGHT);
            }
            // }

            println!("\nDiffs: {:?}\n", diffs);
            // println!("current pt: {:?}", current_point);

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
        }
    }

    println!("\n\nResult: {:?}", current_point);
    for (i, basis) in bases.iter_mut().enumerate() {
        *basis.weight_mut() = current_point[i];
    }

    wf_ops::update_wf_fm_bases(
        bases,
        &basis_wfs_unweighted,
        surfaces_per_elec,
        grid_n,
        None,
    );

    surfaces_per_elec.psi_pp_score = eval::score_wf(
        &surfaces_per_elec.psi_pp_calculated,
        &surfaces_per_elec.psi_pp_measured,
        grid_n,
    )
}

/// Helper for finding weight gradient descent. Updates psi and psi'', calculate E, and score.
/// This function doesn't mutate any of the data.
pub fn score_weight_set(
    bases: &[Basis],
    surfaces_per_elec: &SurfacesPerElec,
    grid_n: usize,
    basis_wfs_unweighted: &BasisWfsUnweighted,
    weights: &[f64],
) -> f64 {
    let mut surfaces = surfaces_per_elec.clone();

    // todo: Previously, you decoupled E from this so it wouldn't change the overall E.
    // todo, if, 3 June 2023, there are problems, look to this, and figure out how to handle.
    // todo: Likely, update_wf_fm_bases will need to take E explicilty instead of pulling form sfcs.
    wf_ops::update_wf_fm_bases(
        bases,
        &basis_wfs_unweighted,
        &mut surfaces,
        grid_n,
        Some(weights),
    );

    eval::score_wf(
        &surfaces.psi_pp_calculated,
        &surfaces.psi_pp_measured,
        grid_n,
    )
}
