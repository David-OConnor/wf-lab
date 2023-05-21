//! Experimental / shell module for finding basis functions given a potential,
//! or arrangement of nuclei.

use crate::{
    basis_wfs::{Basis, HOrbital},
    complex_nums::Cplx,
    eval,
    types::{Arr3d, Arr3dReal, Arr3dVec, SurfacesPerElec, SurfacesShared},
    wf_ops::{self, ħ, BasisWfsUnweighted},
};

use lin_alg2::f64::{Quaternion, Vec3};

/// Adjust weights of coefficiants until score is minimized.
/// We use a gradient-descent approach to find local *score* minimum. (fidelity?)
/// We choose several start points to help find a global solution.
pub fn find_weights(
    charges_fixed: &Vec<(Vec3, f64)>,
    bases: &mut Vec<Basis>,
    basis_wfs_unweighted: &mut BasisWfsUnweighted,
    E: &mut f64,
    surfaces_shared: &mut SurfacesShared,
    surfaces_per_elec: &mut SurfacesPerElec,
    max_n: u16, // quantum number n
    grid_n: usize,
) {
    let mut visible = Vec::new();
    wf_ops::initialize_bases(charges_fixed, bases, &mut visible, max_n);

    *basis_wfs_unweighted = BasisWfsUnweighted::new(&bases, &surfaces_shared.grid_posits, grid_n);

    // Infinitessimal weight change, used for assessing derivatives.
    const D_WEIGHT: f64 = 0.001;

    const NUM_DESCENTS: usize = 10; // todo
    let mut descent_rate = 0.1; // todo? Factor for gradient descent based on the vector.

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

    // For now, let's use a single starting point, and gradient-descent from it.
    let mut current_point = vec![0.; bases.len()];
    for i in 0..charges_fixed.len() {
        current_point[i] = 1.;
    }

    // For reasons not-yet determined, we appear to need to run these after initializing the weights,
    // even though they're include din the scoring algo. All 3 seem to be required.

    for _descent_num in 0..NUM_DESCENTS {
        //
        // // todo: TS. This should be handled by score_this. Get to the bottom of it, although
        // // todo you've spent a while without results.
        // wf_ops::update_wf_fm_bases(
        //     bases,
        //     &basis_wfs_unweighted,
        //     surfaces_per_elec,
        //     E,
        //     None,
        //     grid_n,
        //     Some(&current_point),
        // );

        // println!("Psi Before: {} {} {}", surfaces_per_elec.psi.on_pt[10][10][10],
        //          surfaces_per_elec.psi.on_pt[1][10][15],
        //          surfaces_per_elec.psi.on_pt[3][4][6]);

        let score_this = score_weight_set(
            bases,
            *E,
            &surfaces_per_elec,
            grid_n,
            &basis_wfs_unweighted,
            &current_point,
        );

        println!("\n\nScore this: {:?}", score_this);
        println!("Current weight: {:?}", current_point);

        // This is our gradient.
        let mut diffs = vec![0.; bases.len()];

        // todo:  WIth our current API, the finding psi'' numericaly
        // uses the weight field on bases to construct the nearby points.
        // todo: YOu currently have a mixed API between this weights Vec,
        // todo and that field. For now, update the weights field prior to scoring.

        // Iterate through each basis; calcualate a score using the same parameters as above,
        // with the exception of a slightly small weight for the basis.
        for (i_basis, _basis) in bases.iter().enumerate() {
            // Scores from a small change along this dimension. basis = dimension
            // We could use midpoint, but to save computation, we will do a simple 2-point.
            let mut point_shifted_left = current_point.clone();

            point_shifted_left[i_basis] -= D_WEIGHT;

            println!("\nPrev weight: {:?}", point_shifted_left);
            //
            // println!("Psi After: {} {} {}", surfaces_per_elec.psi.on_pt[10][10][10],
            //          surfaces_per_elec.psi.on_pt[1][10][15],
            //          surfaces_per_elec.psi.on_pt[3][4][6]);

            let score_prev = score_weight_set(
                bases,
                *E,
                &surfaces_per_elec,
                grid_n,
                &basis_wfs_unweighted,
                &point_shifted_left,
            );

            println!("Score prev: {:?}", score_prev);

            // dscore / d_weight.
            diffs[i_basis] = (score_this - score_prev) / D_WEIGHT;
        }

        println!("\nDiffs: {:?}\n", diffs);
        println!("current pt: {:?}", current_point);

        // Now that we've computed our gradient, shift down it to the next point.
        for i in 0..bases.len() {
            // Direction: Diff is current pt score minus score from a slightly
            // lower value of a given basis. If it's positive, it means the score gets better
            // (lower) with a smaller weight, so *reduce* weight accordingly.
            current_point[i] -= diffs[i] * descent_rate;
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
        E,
        None,
        grid_n,
        Some(&current_point),
    );
}

/// Helper for finding weight gradient descent. Updates psi and psi'', calculate E, and score.
/// This function doesn't mutate any of the data.
pub fn score_weight_set(
    bases: &[Basis],
    E: f64,
    surfaces_per_elec: &SurfacesPerElec,
    grid_n: usize,
    basis_wfs_unweighted: &BasisWfsUnweighted,
    weights: &[f64],
) -> f64 {
    let mut surfaces = surfaces_per_elec.clone();
    let mut E = E;

    wf_ops::update_wf_fm_bases(
        bases,
        &basis_wfs_unweighted,
        &mut surfaces,
        &mut E,
        None,
        grid_n,
        Some(weights),
    );

    eval::score_wf(
        &surfaces.psi_pp_calculated,
        &surfaces.psi_pp_measured,
        grid_n,
    )
}
