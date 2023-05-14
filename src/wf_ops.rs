//! This module contains the bulk of the wave-function evalution and solving logic.
//!

//! todo: Important question about state and quantum numbers. You can't have more than one elec
//! etc in the same state (Pauli exclusion / fermion rules), but how does this apply when multiple
//! todo nuclei are involved?

// todo: For your grid: Perhaps an optimal grid is one where the lines between grid
// todo points are as constant as you can? Ie are like contour lines? Ie pick points
// todo along lines of equal psi? (or psi^2?)

// todo: A thought: Maybe analyze psi'' diff, then figure out what combination of
// todo H basis fns added to psi approximate it, or move towards it?

// todo: If you switch to individual wfs per electon, spherical coords may make more sense, eg
// todo with RBF or otherwise.

// Project to do, from Eckkert @ discord Physics:
// "if you wanted to invent something, a Bayesian classical simulator that could tell which of its
// quantum-derived parameters needed finer calculation and which did them when necessary, is
// something that doesn't exist afik"

// todo: Something that would really help: A recipe for which basis wfs to add for a given
// todo potential.

use crate::{
    basis_wfs::{Basis, HOrbital, SphericalHarmonic},
    complex_nums::Cplx,
    eigen_fns,
    num_diff,
    // rbf::Rbf,
    types::{Arr3d, Arr3dReal, Arr3dVec, SurfacesPerElec, SurfacesShared},
    util::{self},
};
use std::f32::EPSILON;

use lin_alg2::f64::{Quaternion, Vec3};

// We use Hartree units: ħ, elementary charge, electron mass, and Bohr radius.
const K_C: f64 = 1.;
pub const Q_PROT: f64 = 1.;
pub const Q_ELEC: f64 = -1.;
pub const M_ELEC: f64 = 1.;
pub const ħ: f64 = 1.;

pub(crate) const NUDGE_DEFAULT: f64 = 0.01;

// Wave fn weights
pub const WEIGHT_MIN: f64 = -4.;
pub const WEIGHT_MAX: f64 = 4.;

// Compute these statically, to avoid continuous calls during excecution.

// Wave function number of values per edge.
// Memory use and some parts of computation scale with the cube of this.
// pub const N: usize = 20;

#[derive(Clone, Copy, Debug)]
pub enum Spin {
    Up,
    Dn,
}

/// - Computes V from "fixed" charges, ie nuclei. Run this after changing these charges.
/// Does not modify per-electron charges; those are updated elsewhere, incorporating the
/// potential here, as well as from other electrons.
pub fn update_V_fm_fixed_charges(
    charges_fixed: &[(Vec3, f64)],
    V_nuc_shared: &mut Arr3dReal,
    grid_min: &mut f64,
    grid_max: &mut f64,
    spacing_factor: f64,
    grid_posits: &mut Arr3dVec,
    n: usize,
    // Wave functions from other electrons, for calculating the Hartree potential.
    // charges_electron: &[Arr3dReal],
    // i_this_elec: usize,
) {
    // Set up the grid so that it smartly encompasses the charges, letting the WF go to 0
    // towards the edges
    let mut max_abs_val = 0.;
    for (posit, _) in charges_fixed {
        if posit.x.abs() > max_abs_val {
            max_abs_val = posit.x.abs();
        }
        if posit.y.abs() > max_abs_val {
            max_abs_val = posit.y.abs();
        }
        if posit.z.abs() > max_abs_val {
            max_abs_val = posit.z.abs();
        }
    }

    // const RANGE_PAD: f64 = 1.6;
    const RANGE_PAD: f64 = 5.8;
    // const RANGE_PAD: f64 = 15.;

    *grid_max = max_abs_val + RANGE_PAD;
    *grid_min = -*grid_max;
    update_grid_posits(grid_posits, *grid_min, *grid_max, spacing_factor, n);

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let posit_sample = grid_posits[i][j][k];

                V_nuc_shared[i][j][k] = 0.;

                for (posit_charge, charge_amt) in charges_fixed.iter() {
                    V_nuc_shared[i][j][k] += V_coulomb(*posit_charge, posit_sample, *charge_amt);
                }
                // Note: We update individual electron Vs (eg with fixed + all *other* elec Vs
                // in `elec_elec::update_V_individual()`.
                // sfcs.V[i][j][k] = sfcs.V[i][j][k]
            }
        }
    }
}

/// Calculate ψ* ψ
pub(crate) fn norm_sq(dest: &mut Arr3dReal, source: &Arr3d, n: usize) {
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                dest[i][j][k] = source[i][j][k].abs_sq();
            }
        }
    }
}

/// Normalize a wave function so that <ψ|ψ> = 1.
/// Returns the norm value for use in normalizing basis fns in psi''_measured calculation.
///
/// Note that due to phase symmetry, there are many ways to balance the normalization of the real
/// vice imaginary parts. Our implmentation (dividing both real and imag parts by norm square)
/// is one way.
fn normalize_wf(arr: &mut Arr3d, norm: f64, n: usize) -> f64 {
    const EPS: f64 = 0.000001;
    if norm.abs() < EPS {
        return 1.;
    }

    let norm_sqrt = norm.sqrt();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                // Note: Check the div impl for details.
                arr[i][j][k] = arr[i][j][k] / norm_sqrt;
            }
        }
    }

    norm_sqrt
}

/// - Computes a trial ψ from basis functions
/// - Computes ψ'' calculated, and measured from the trial ψ
/// Modifies in place to conserve memory. These operations are combined in the same function to
/// save computation, since they're often run at once, and can be iterated through using a single loop
/// through all grid points.
// pub fn update_wf_fm_bases(
//     bases: &[Basis],
//     basis_wfs: &[Arr3d],
//     sfcs: &mut SurfacesPerElec,
//     E: f64,
//     grid_posits: &Arr3dVec,
//     bases_visible: &[bool],
//     grid_n: usize,
// ) {
//     let mut norm = 0.;
//
//     for i in 0..grid_n {
//         for j in 0..grid_n {
//             for k in 0..grid_n {
//                 sfcs.psi[i][j][k] = Cplx::new_zero();
//
//                 for (i_basis, basis_wf) in basis_wfs.iter().enumerate() {
//                     let weight = if bases_visible[i_basis] {
//                         bases[i_basis].weight()
//                     } else {
//                         0.
//                     };
//                     sfcs.psi[i][j][k] += basis_wf[i][j][k] * weight;
//                 }
//
//                 norm += sfcs.psi[i][j][k].abs_sq();
//             }
//         }
//     }

pub fn update_wf_fm_bases(
    bases: &[Basis],
    sfcs: &mut SurfacesPerElec,
    E: f64,
    grid_posits: &Arr3dVec,
    bases_visible: &[bool],
    grid_n: usize,
) {
    // We intergrate normalization into the loops here, vice as a separate function,
    // to prevent excess loops from running.

    let mut norm = 0.;

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                let posit_sample = grid_posits[i][j][k];

                sfcs.psi[i][j][k] = Cplx::new_zero();

                for (basis_i, basis) in bases.iter().enumerate() {
                    let weight = if bases_visible[basis_i] {
                        basis.weight()
                    } else {
                        0.
                    };
                    sfcs.psi[i][j][k] += basis.value(posit_sample) * weight;
                }

                norm += sfcs.psi[i][j][k].abs_sq();

                // sfcs.psi_pp_calculated[i][j][k] =
                //     eigen_fns::find_ψ_pp_calc(&sfcs.psi, &sfcs.V, E, i, j, k);
                //
                // // Calculate psi'' based on a numerical derivative of psi
                // // in 3D.
                // // We can compute ψ'' measured this in the same loop here, since we're using an analytic
                // // equation for ψ; we can diff at arbitrary points vice only along a grid of pre-computed ψ.
                // sfcs.psi_pp_measured[i][j][k] = num_diff::find_ψ_pp_meas_fm_bases(
                //     posit_sample,
                //     bases,
                //     sfcs.psi[i][j][k],
                //     bases_visible,
                // );
            }
        }
    }

    // let psi_norm_sqrt = norm.sqrt();
    let psi_norm_sqrt = normalize_wf(&mut sfcs.psi, norm, grid_n);

    // Update psi_pps after normalization. We can't rely on cached wfs here, since we need to
    // take infinitessimal differences on the analytic basis equations to find psi'' measured.
    update_psi_pps_from_bases(
        bases,
        &sfcs.psi,
        &sfcs.V,
        &mut sfcs.psi_pp_calculated,
        &mut sfcs.psi_pp_measured,
        grid_posits,
        E,
        bases_visible,
        grid_n,
        psi_norm_sqrt,
        None,
    );
}

/// Update psi'' calc and psi'' measured, assuming we are using basis WFs. This is done
/// after the wave function is contructed and normalized.
///
/// We use a separate function from this since it's used separately in our basis-finding
/// algorithm
fn update_psi_pps_from_bases(
    // We split these arguments up instead of using surfaces to control mutability.
    bases: &[Basis],
    psi: &Arr3d,
    V: &Arr3dReal,
    psi_pp_calc: &mut Arr3d,
    psi_pp_meas: &mut Arr3d,
    grid_posits: &Arr3dVec,
    E: f64,
    bases_visible: &[bool],
    grid_n: usize,
    psi_norm_sqrt: f64,
    weights: Option<&[f64]>, // todo: This API eneds work.
) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                let posit_sample = grid_posits[i][j][k];

                // psi[i][j][k] = psi[i][j][k] / norm_sqrt;

                psi_pp_calc[i][j][k] = eigen_fns::find_ψ_pp_calc(psi, V, E, i, j, k);

                // Calculate psi'' based on a numerical derivative of psi
                // in 3D.
                // We can compute ψ'' measured this in the same loop here, since we're using an analytic
                // equation for ψ; we can diff at arbitrary points vice only along a grid of pre-computed ψ.
                psi_pp_meas[i][j][k] = num_diff::find_ψ_pp_meas_fm_bases(
                    posit_sample,
                    bases,
                    psi[i][j][k],
                    psi_norm_sqrt,
                    bases_visible,
                    weights,
                );
            }
        }
    }
}

/// Score using the fidelity of psi'' calculated vs measured; |<psi_trial | psi_true >|^2.
/// This requires normalizing the wave functions we're comparing.
/// todo: Curretly not working.
/// todo: I don't think you can use this approach comparing psi''s with fidelity, since they're
/// todo not normalizsble.
/// todo: Perhaps this isn't working because these aren't wave functions! psi is a WF;
/// psi'' is not
// fn wf_fidelity(sfcs: &Surfaces) -> f64 {
fn fidelity(sfcs: &SurfacesPerElec, n: usize) -> f64 {
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
                if sfcs.psi_pp_calculated[i][j][k].real.abs() < SCORE_THRESH
                    && sfcs.psi_pp_measured[i][j][k].real.abs() < SCORE_THRESH
                {
                    norm_calc += sfcs.psi_pp_calculated[i][j][k];
                    norm_meas += sfcs.psi_pp_measured[i][j][k];
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
                if sfcs.psi_pp_calculated[i][j][k].real.abs() < SCORE_THRESH
                    && sfcs.psi_pp_measured[i][j][k].real.abs() < SCORE_THRESH
                {
                    result += sfcs.psi_pp_calculated[i][j][k] / norm_calc.real
                        * sfcs.psi_pp_calculated[i][j][k]
                        / norm_calc.real;
                }
            }
        }
    }

    result.abs_sq()
}

/// Score a wave function by comparing the least-squares sum of its measured and
/// calculated second derivaties.
pub fn score_wf(sfcs: &SurfacesPerElec, n: usize) -> f64 {
    let mut result = 0.;

    // Avoids numerical precision issues. Without this, certain values of N will lead
    // to a bogus score. Values of N both too high and too low can lead to this. Likely due to
    // if a grid value is too close to a charge source, the value baloons.
    const SCORE_THRESH: f64 = 10.;

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                // todo: Check if either individual is outside a thresh?
                let diff = sfcs.psi_pp_calculated[i][j][k] - sfcs.psi_pp_measured[i][j][k];
                // let val = diff.real + diff.im; // todo: Do you want this, mag_sq, or something else?
                let val = diff.abs_sq();
                if val < SCORE_THRESH {
                    result += val;
                }
            }
        }
    }

    result
}

/// Single-point Coulomb potential, eg a hydrogen nuclei.
pub(crate) fn V_coulomb(posit_charge: Vec3, posit_sample: Vec3, charge: f64) -> f64 {
    let diff = posit_sample - posit_charge;
    let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

    -K_C * charge / r
}

/// Find the E that minimizes score, by narrowing it down. Note that if the relationship
/// between E and psi'' score isn't straightforward, this will converge on a local minimum.
pub fn find_E(sfcs: &mut SurfacesPerElec, E: &mut f64, grid_n: usize) {
    // todo: WHere to configure these mins and maxes
    let mut E_min = -2.;
    let mut E_max = 2.;
    let mut E_range_div2 = 2.;
    let vals_per_iter = 8;

    let num_iters = 10;

    for _ in 0..num_iters {
        let E_vals = util::linspace((E_min, E_max), vals_per_iter);
        let mut best_score = 100_000_000.;
        let mut best_E = 0.;

        for E_trial in E_vals {
            for i in 0..grid_n {
                for j in 0..grid_n {
                    for k in 0..grid_n {
                        sfcs.psi_pp_calculated[i][j][k] =
                            eigen_fns::find_ψ_pp_calc(&sfcs.psi, &sfcs.V, E_trial, i, j, k);
                    }
                }
            }

            let score = score_wf(sfcs, grid_n);
            if score < best_score {
                best_score = score;
                best_E = E_trial;
                *E = E_trial;
            }
        }

        E_min = best_E - E_range_div2;
        E_max = best_E + E_range_div2;
        E_range_div2 /= vals_per_iter as f64; // todo: May need a wider range than this.
    }
}

/// A crude low pass
pub fn smooth_array(arr: &mut Arr3d, smoothing_amt: f64, n: usize) {
    let orig = arr.clone();

    for i in 0..n {
        if i == 0 || i == n - 1 {
            continue;
        }
        for j in 0..n {
            if j == 0 || j == n - 1 {
                continue;
            }
            for k in 0..n {
                if k == 0 || k == n - 1 {
                    continue;
                }
                let neighbor_avg = (orig[i - 1][j][k]
                    + orig[i + 1][j][k]
                    + orig[i][j - 1][k]
                    + orig[i][j + 1][k]
                    + orig[i][j][k - 1]
                    + orig[i][j][k + 1])
                    / 6.;

                let diff_from_neighbors = neighbor_avg - arr[i][j][k];

                arr[i][j][k] += diff_from_neighbors * smoothing_amt;
            }
        }
    }
}

/// Update our grid positions
pub fn update_grid_posits(
    grid_posits: &mut Arr3dVec,
    grid_min: f64,
    grid_max: f64,
    spacing_factor: f64,
    n: usize,
) {
    let grid_lin = util::linspace((grid_min, grid_max), n);

    // Set up a grid with values that increase in distance the farther we are from the center.
    let mut grid_1d = vec![0.; n];

    for i in 0..n {
        let mut val = grid_lin[i].abs().powf(spacing_factor);
        if grid_lin[i] < 0. {
            val *= -1.; // square the magnitude only.
        }
        grid_1d[i] = val;
    }
    // println!("\n\nGRID 1D: {:.2?}", grid_1d);

    for (i, x) in grid_1d.iter().enumerate() {
        for (j, y) in grid_1d.iter().enumerate() {
            for (k, z) in grid_1d.iter().enumerate() {
                grid_posits[i][j][k] = Vec3::new(*x, *y, *z);
            }
        }
    }
}

/// [re]Create a set of basis functions, given fixed-charges representing nuclei.
/// Use this in main and lib inits, and when you add charges.
pub fn initialize_bases(
    charges_fixed: &Vec<(Vec3, f64)>,
    bases: &mut Vec<Basis>,
    bases_visible: &mut Vec<bool>,
    max_n: u16, // quantum number n
) {
    *bases = Vec::new();
    *bases_visible = Vec::new();

    // for (charge_id, (nuc_posit, _)) in charges_fixed.iter().enumerate() {
    for n in 1..max_n + 1 {
        for l in 0..n {
            for m in -(l as i16)..l as i16 + 1 {
                // This loop order allows the basis sliders to be sorted with like-electrons next to each other.
                for (charge_id, (nuc_posit, _)) in charges_fixed.iter().enumerate() {
                    bases.push(Basis::H(HOrbital {
                        posit: *nuc_posit,
                        n,
                        harmonic: SphericalHarmonic {
                            l,
                            m,
                            orientation: Quaternion::new_identity(),
                        },
                        weight: 0.,
                        charge_id,
                    }));

                    bases_visible.push(true);
                }
            }
        }
    }
}

/// Adjust weights of coefficiants until score is minimized.
/// We use a gradient-descent approach to find local *score* minimum. (fidelity?)
/// We choose several start points to help find a global solution.
pub fn find_weights(
    charges_fixed: &Vec<(Vec3, f64)>,
    bases: &mut Vec<Basis>,
    E: &mut f64,
    surfaces_shared: &mut SurfacesShared,
    surfaces_per_elec: &mut SurfacesPerElec,
    max_n: u16, // quantum number n
    grid_n: usize,
) {
    let mut visible = Vec::new();
    initialize_bases(charges_fixed, bases, &mut visible, max_n);

    let mut weight_min = -4.;
    let mut weight_max = 4.;

    // Infinitessimal weight change, used for assessing derivatives.
    const D_WEIGHT: f64 = 0.0001;

    const NUM_DESCENTS: usize = 1; // todo
    let mut descent_rate = 5.; // todo? Factor for gradient descent based on the vector.

    let basis_wfs_unweighted =
        create_bases_wfs_unweighted(bases, &surfaces_shared.grid_posits, grid_n);

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
    current_point[0] = 1.;
    current_point[1] = 1.;

    for descent_num in 0..NUM_DESCENTS {
        // todo: You could choose your initial points, where, for each atom, the n=1 bases
        // todo are +- 1 for each, and all other bases are 0. Ie repeat the whole process,
        // todo but for those initial points.

        let score_this = score_weight_set(
            bases,
            E,
            surfaces_shared,
            surfaces_per_elec,
            grid_n,
            &basis_wfs_unweighted,
            &current_point,
        );

        println!("This score: {:?}", score_this);

        // This is our gradient.
        let mut diffs = vec![0.; bases.len()];

        // todo:  WIth our current API, the finding psi'' numericaly
        // uses the weight field on bases to construct the nearby points.
        // todo: YOu currently have a mixed API between this weights Vec,
        // todo and that field. For now, update the weights field prior to scoring.

        for (i_basis, _basis) in bases.iter().enumerate() {
            // Scores from a small change along this dimension. basis = dimension
            // We could use midpoint, but to save computation, we will do a simple 2-point.
            let mut prev_point = current_point.clone();
            prev_point[i_basis] -= D_WEIGHT;

            // println!("Current: {:?}, prev: {:?}", current_point, prev_point);

            let score_prev = score_weight_set(
                bases,
                E,
                surfaces_shared,
                surfaces_per_elec,
                grid_n,
                &basis_wfs_unweighted,
                &prev_point,
            );

            println!("Score prev: {:?}", score_prev);

            diffs[i_basis] = (score_this - score_prev) * D_WEIGHT;
        }

        println!("Diffs: {:?}", diffs);
        println!("current pt: {:?}", current_point);
        // Now that we've computed our gradient, shift down it to the next point.
        for i in 0..bases.len() {
            // todo: Dir?
            current_point[i] -= diffs[i] * descent_rate;
        }
    }

    println!("Final result: {:?}", current_point);

    // Set our global weights to be the final descent result.
    for (i, basis) in bases.iter_mut().enumerate() {
        *basis.weight_mut() = current_point[i];
    }
    find_E(surfaces_per_elec, E, grid_n);
}

/// Helper for finding weight gradient descent. Returns a score at a given set of weights.
pub fn score_weight_set(
    bases: &[Basis],
    E: &mut f64,
    surfaces_shared: &mut SurfacesShared,
    surfaces_per_elec: &mut SurfacesPerElec,
    grid_n: usize,
    basis_wfs_unweighted: &[Arr3d],
    weights: &[f64],
) -> f64 {
    let mut norm = 0.;

    // println!("Bases: {:?}", bases);

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                surfaces_per_elec.psi[i][j][k] = Cplx::new_zero();

                for i_basis in 0..bases.len() {
                    surfaces_per_elec.psi[i][j][k] +=
                        basis_wfs_unweighted[i_basis][i][j][k] * weights[i_basis]
                }

                norm += surfaces_per_elec.psi[i][j][k].abs_sq();
            }
        }
    }

    let psi_norm_sqrt = normalize_wf(&mut surfaces_shared.psi, norm, grid_n);

    // kludge for API that needs work.
    let bases_visible = vec![true; bases.len()];

    // Find E before finding psi'' calc.
    find_E(surfaces_per_elec, E, grid_n);

    update_psi_pps_from_bases(
        bases,
        &surfaces_per_elec.psi,
        &surfaces_per_elec.V,
        &mut surfaces_per_elec.psi_pp_calculated,
        &mut surfaces_per_elec.psi_pp_measured,
        &surfaces_shared.grid_posits,
        *E,
        &bases_visible,
        grid_n,
        psi_norm_sqrt,
        Some(&weights),
    );

    score_wf(surfaces_per_elec, grid_n)
}

/// Create unweighted basis wave functions. Run this whenever we add or remove basis fns,
/// and when changing the grid size. Each basis will be normalized in this function.
pub fn create_bases_wfs_unweighted(
    bases: &[Basis],
    grid_posits: &Arr3dVec,
    grid_n: usize,
) -> Vec<Arr3d> {
    let mut result = Vec::new();

    for _ in 0..bases.len() {
        result.push(crate::types::new_data(grid_n))
    }

    for (basis_i, basis) in bases.iter().enumerate() {
        let mut norm = 0.;

        for i in 0..grid_n {
            for j in 0..grid_n {
                for k in 0..grid_n {
                    let posit_sample = grid_posits[i][j][k];

                    let val = basis.value(posit_sample);

                    // println!("VAL: {:?}", val);
                    // println!("P: {:?}", posit_sample);

                    result[basis_i][i][j][k] = val;
                    norm += val.abs_sq();
                }
            }
        }

        normalize_wf(&mut result[basis_i], norm, grid_n);
    }

    result
}
