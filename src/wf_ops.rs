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
    eigen_fns, eval,
    num_diff::{self, H, H_SQ},
    types,
    types::{Arr3d, Arr3dReal, Arr3dVec, SurfacesPerElec, SurfacesShared},
    util,
};

use crate::types::new_data;
use lin_alg2::f64::{Quaternion, Vec3};

// We use Hartree units: ħ, elementary charge, electron mass, and Bohr radius.
pub const K_C: f64 = 1.;
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

/// Computes V from "fixed" charges, ie nuclei, by calculating Coulomb potential.
/// Run this after changing these charges.
/// Does not modify per-electron charges; those are updated elsewhere, incorporating the
/// potential here, as well as from other electrons.
pub fn update_V_fm_fixed_charges(
    charges_fixed: &[(Vec3, f64)],
    V_nuc_shared: &mut Arr3dReal,
    grid_posits: &Arr3dVec,
    grid_n: usize,
    // Wave functions from other electrons, for calculating the Hartree potential.
    // charges_electron: &[Arr3dReal],
    // i_this_elec: usize,
) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                let posit_sample = grid_posits[i][j][k];

                V_nuc_shared[i][j][k] = 0.;

                for (posit_charge, charge_amt) in charges_fixed.iter() {
                    V_nuc_shared[i][j][k] +=
                        util::V_coulomb(*posit_charge, posit_sample, *charge_amt);
                }
            }
        }
    }
}

/// Mix bases together into a numerical wave function at each grid point, and at diffs.
/// This is our mixer from pre-calculated basis fucntions: Create psi, including at
/// neighboring points (used to numerically differentiate), from summing them with
/// their weights. Basis wfs must be initialized prior to running this, and weights must
/// be selected.
///
/// The resulting wave functions are normalized.
pub fn mix_bases(
    bases: &[Basis],
    basis_wfs: &BasisWfsUnweighted,
    psi: &mut PsiWDiffs,
    grid_n: usize,
    weights: Option<&[f64]>,
) {
    // We don't need to normalize the result using the full procedure; the basis-wfs are already
    // normalized, so divide by the cumulative basis weights.
    let mut weight_total = 0.;
    match weights {
        Some(w) => {
            for weight in w {
                weight_total += weight.abs();
            }
        }
        None => {
            for b in bases {
                weight_total += b.weight().abs();
            }
        }
    }

    let mut norm_scaler = 1. / weight_total;

    // Prevents NaNs and related complications.
    if weight_total.abs() < 0.000001 {
        norm_scaler = 0.;
    }

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                psi.on_pt[i][j][k] = Cplx::new_zero();
                psi.x_prev[i][j][k] = Cplx::new_zero();
                psi.x_next[i][j][k] = Cplx::new_zero();
                psi.y_prev[i][j][k] = Cplx::new_zero();
                psi.y_next[i][j][k] = Cplx::new_zero();
                psi.z_prev[i][j][k] = Cplx::new_zero();
                psi.z_next[i][j][k] = Cplx::new_zero();

                for i_basis in 0..bases.len() {
                    let mut weight = match weights {
                        Some(w) => w[i_basis],
                        None => bases[i_basis].weight(),
                    };

                    weight *= norm_scaler;

                    psi.on_pt[i][j][k] += basis_wfs.on_pt[i_basis][i][j][k] * weight;
                    psi.x_prev[i][j][k] += basis_wfs.x_prev[i_basis][i][j][k] * weight;
                    psi.x_next[i][j][k] += basis_wfs.x_next[i_basis][i][j][k] * weight;
                    psi.y_prev[i][j][k] += basis_wfs.y_prev[i_basis][i][j][k] * weight;
                    psi.y_next[i][j][k] += basis_wfs.y_next[i_basis][i][j][k] * weight;
                    psi.z_prev[i][j][k] += basis_wfs.z_prev[i_basis][i][j][k] * weight;
                    psi.z_next[i][j][k] += basis_wfs.z_next[i_basis][i][j][k] * weight;
                }
            }
        }
    }
}

/// This function combines mixing (pre-computed) numerical basis WFs with updating psi''.
/// it updates E as well.
///
/// - Computes a trial ψ from basis functions. Computes it at each grid point, as well as
/// the 6 offset ones along the 3 axis used to numerically differentiate.
/// - Computes ψ'' calculated, and measured from the trial ψ
pub fn update_wf_fm_bases(
    bases: &[Basis],
    basis_wfs: &BasisWfsUnweighted,
    sfcs: &mut SurfacesPerElec,
    E: &mut f64,
    grid_n: usize,
    weights: Option<&[f64]>,
) {
    mix_bases(bases, basis_wfs, &mut sfcs.psi, grid_n, weights);

    *E = find_E(sfcs, grid_n);

    // Update psi_pps after normalization. We can't rely on cached wfs here, since we need to
    // take infinitessimal differences on the analytic basis equations to find psi'' measured.
    update_psi_pps_from_bases(
        &sfcs.psi,
        &sfcs.V,
        &mut sfcs.psi_pp_calculated,
        &mut sfcs.psi_pp_measured,
        *E,
        grid_n,
    );
}

/// Run this after update E.
pub fn update_psi_pp_calc(
    // We split these arguments up instead of using surfaces to control mutability.
    psi: &Arr3d,
    V: &Arr3dReal,
    psi_pp_calc: &mut Arr3d,
    E: f64,
    grid_n: usize,
) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                psi_pp_calc[i][j][k] = eigen_fns::find_ψ_pp_calc(&psi, V, E, i, j, k);
            }
        }
    }
}

/// Update psi'' calc and psi'' measured, assuming we are using basis WFs. This is done
/// after the wave function is contructed and normalized, including at neighboring points.
///
/// We use a separate function from this since it's used separately in our basis-finding
/// algorithm
pub fn update_psi_pps_from_bases(
    // We split these arguments up instead of using surfaces to control mutability.
    psi: &PsiWDiffs,
    V: &Arr3dReal,
    psi_pp_calc: &mut Arr3d,
    psi_pp_meas: &mut Arr3d,
    E: f64,
    grid_n: usize,
) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                psi_pp_calc[i][j][k] = eigen_fns::find_ψ_pp_calc(&psi.on_pt, V, E, i, j, k);

                // Calculate psi'' based on a numerical derivative of psi
                // in 3D.
                // We can compute ψ'' measured this in the same loop here, since we're using an analytic
                // equation for ψ; we can diff at arbitrary points vice only along a grid of pre-computed ψ.

                // todo: Put back once you sorted out your anomoloous Psi behavior.
                psi_pp_meas[i][j][k] = num_diff::find_ψ_pp_meas_fm_unweighted_bases(
                    // todo: Combine into a single struct etc A/R.
                    psi.on_pt[i][j][k],
                    psi.x_prev[i][j][k],
                    psi.x_next[i][j][k],
                    psi.y_prev[i][j][k],
                    psi.y_next[i][j][k],
                    psi.z_prev[i][j][k],
                    psi.z_next[i][j][k],
                );
            }
        }
    }
}

/// Find the E that minimizes score, by narrowing it down. Note that if the relationship
/// between E and psi'' score isn't straightforward, this will converge on a local minimum.
pub fn find_E(sfcs: &SurfacesPerElec, grid_n: usize) -> f64 {
    // todo: WHere to configure these mins and maxes
    let mut result = 0.;

    let mut E_min = -2.;
    let mut E_max = 2.;
    let mut E_range_div2 = 2.;
    let vals_per_iter = 8;

    let num_iters = 10;

    let mut psi_pp_calc = new_data(grid_n);
    types::copy_array(&mut psi_pp_calc, &sfcs.psi_pp_calculated, grid_n);

    for _ in 0..num_iters {
        let E_vals = util::linspace((E_min, E_max), vals_per_iter);
        let mut best_score = 100_000_000.;
        let mut best_E = 0.;

        for E_trial in E_vals {
            for i in 0..grid_n {
                for j in 0..grid_n {
                    for k in 0..grid_n {
                        psi_pp_calc[i][j][k] =
                            eigen_fns::find_ψ_pp_calc(&sfcs.psi.on_pt, &sfcs.V, E_trial, i, j, k);
                    }
                }
            }

            let score = eval::score_wf(&psi_pp_calc, &sfcs.psi_pp_measured, grid_n);
            if score < best_score {
                best_score = score;
                best_E = E_trial;
                result = E_trial;
            }
        }

        E_min = best_E - E_range_div2;
        E_max = best_E + E_range_div2;
        E_range_div2 /= vals_per_iter as f64; // todo: May need a wider range than this.
    }

    result
}

/// Update our grid positions. Run this when we change grid bounds, resolution, or spacing.
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

    for (i, x) in grid_1d.iter().enumerate() {
        for (j, y) in grid_1d.iter().enumerate() {
            for (k, z) in grid_1d.iter().enumerate() {
                grid_posits[i][j][k] = Vec3::new(*x, *y, *z);
            }
        }
    }
}

/// [re]Create a set of basis functions, given fixed-charges representing nuclei.
/// Use this in main and lib inits, and when you add or remove charges.
pub fn initialize_bases(
    charges_fixed: &Vec<(Vec3, f64)>,
    bases: &mut Vec<Basis>,
    bases_visible: &mut Vec<bool>,
    max_n: u16, // quantum number n
) {
    // let mut prev_weights = Vec::new();
    // for basis in bases.iter() {
    //     prev_weights.push(basis.weight());
    // }

    *bases = Vec::new();
    println!("Initializing bases");

    // todo: We currently call this in some cases where it maybe isn't strictly necessarly;
    // todo for now as a kludge to preserve weights, we copy the prev weights.

    let mut i = 0;
    // for (charge_id, (nuc_posit, _)) in charges_fixed.iter().enumerate() {
    for n in 1..max_n + 1 {
        for l in 0..n {
            for m in -(l as i16)..l as i16 + 1 {
                // This loop order allows the basis sliders to be sorted with like-electrons next to each other.
                for (charge_id, (nuc_posit, _)) in charges_fixed.iter().enumerate() {
                    // let weight = if i < prev_weights.len() {
                    //     prev_weights[i]
                    // } else {
                    //     0.
                    // };

                    let weight = 0.;

                    bases.push(Basis::H(HOrbital {
                        posit: *nuc_posit,
                        n,
                        harmonic: SphericalHarmonic {
                            l,
                            m,
                            orientation: Quaternion::new_identity(),
                        },

                        weight,
                        charge_id,
                    }));
                    i += 1;
                }
                bases_visible.push(true);
            }
        }
    }
}

/// Group that includes psi at a point, and at points surrounding it, an infinetesimal difference
/// in both directions along each spacial axis.
#[derive(Clone)]
pub struct PsiWDiffs {
    pub on_pt: Arr3d,
    pub x_prev: Arr3d,
    pub x_next: Arr3d,
    pub y_prev: Arr3d,
    pub y_next: Arr3d,
    pub z_prev: Arr3d,
    pub z_next: Arr3d,
}

/// We use this to store numerical wave functions for each basis, both at sample points, and
/// a small amount along each axix, for calculating partial derivatives of psi''.
/// The `Vec` index corresponds to basis index.
#[derive(Clone)]
pub struct BasisWfsUnweighted {
    pub on_pt: Vec<Arr3d>,
    pub x_prev: Vec<Arr3d>,
    pub x_next: Vec<Arr3d>,
    pub y_prev: Vec<Arr3d>,
    pub y_next: Vec<Arr3d>,
    pub z_prev: Vec<Arr3d>,
    pub z_next: Vec<Arr3d>,
}

impl BasisWfsUnweighted {
    /// Create unweighted basis wave functions. Run this whenever we add or remove basis fns,
    /// and when changing the grid. This evaluates the analytic basis functions at
    /// each grid point. Each basis will be normalized in this function.
    /// Relatively computationally intensive.
    pub fn new(bases: &[Basis], grid_posits: &Arr3dVec, grid_n: usize) -> Self {
        let mut on_pt = Vec::new();
        let mut x_prev = Vec::new();
        let mut x_next = Vec::new();
        let mut y_prev = Vec::new();
        let mut y_next = Vec::new();
        let mut z_prev = Vec::new();
        let mut z_next = Vec::new();

        for _ in 0..bases.len() {
            on_pt.push(crate::types::new_data(grid_n));
            x_prev.push(crate::types::new_data(grid_n));
            x_next.push(crate::types::new_data(grid_n));
            y_prev.push(crate::types::new_data(grid_n));
            y_next.push(crate::types::new_data(grid_n));
            z_prev.push(crate::types::new_data(grid_n));
            z_next.push(crate::types::new_data(grid_n));
        }

        for (basis_i, basis) in bases.iter().enumerate() {
            let mut norm_pt = 0.;

            // let mut norm_x_prev = 0.;
            // let mut norm_x_next = 0.;
            // let mut norm_y_prev = 0.;
            // let mut norm_y_next = 0.;
            // let mut norm_z_prev = 0.;
            // let mut norm_z_next = 0.;

            for i in 0..grid_n {
                for j in 0..grid_n {
                    for k in 0..grid_n {
                        let posit_sample = grid_posits[i][j][k];

                        let posit_x_prev =
                            Vec3::new(posit_sample.x - H, posit_sample.y, posit_sample.z);
                        let posit_x_next =
                            Vec3::new(posit_sample.x + H, posit_sample.y, posit_sample.z);
                        let posit_y_prev =
                            Vec3::new(posit_sample.x, posit_sample.y - H, posit_sample.z);
                        let posit_y_next =
                            Vec3::new(posit_sample.x, posit_sample.y + H, posit_sample.z);
                        let posit_z_prev =
                            Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - H);
                        let posit_z_next =
                            Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + H);

                        let val_pt = basis.value(posit_sample);

                        let val_x_prev = basis.value(posit_x_prev);
                        let val_x_next = basis.value(posit_x_next);
                        let val_y_prev = basis.value(posit_y_prev);
                        let val_y_next = basis.value(posit_y_next);
                        let val_z_prev = basis.value(posit_z_prev);
                        let val_z_next = basis.value(posit_z_next);

                        on_pt[basis_i][i][j][k] = val_pt;
                        x_prev[basis_i][i][j][k] = val_x_prev;
                        x_next[basis_i][i][j][k] = val_x_next;
                        y_prev[basis_i][i][j][k] = val_y_prev;
                        y_next[basis_i][i][j][k] = val_y_next;
                        z_prev[basis_i][i][j][k] = val_z_prev;
                        z_next[basis_i][i][j][k] = val_z_next;

                        norm_pt += val_pt.abs_sq();
                        // norm_x_prev += val_x_prev.abs_sq();
                        // norm_x_next += val_x_next.abs_sq();
                        // norm_y_prev += val_y_prev.abs_sq();
                        // norm_y_next += val_y_next.abs_sq();
                        // norm_z_prev += val_z_prev.abs_sq();
                        // norm_z_next += val_z_next.abs_sq();
                    }
                }
            }

            util::normalize_wf(&mut on_pt[basis_i], norm_pt, grid_n);

            // note: Using individual norm consts appeares to produce incorrect results.

            // util::normalize_wf(&mut x_prev[basis_i], norm_x_prev, grid_n);
            // util::normalize_wf(&mut x_next[basis_i], norm_x_next, grid_n);
            // util::normalize_wf(&mut y_prev[basis_i], norm_y_prev, grid_n);
            // util::normalize_wf(&mut y_next[basis_i], norm_y_next, grid_n);
            // util::normalize_wf(&mut z_prev[basis_i], norm_z_prev, grid_n);
            // util::normalize_wf(&mut z_next[basis_i], norm_z_next, grid_n);
            //
            util::normalize_wf(&mut x_prev[basis_i], norm_pt, grid_n);
            util::normalize_wf(&mut x_next[basis_i], norm_pt, grid_n);
            util::normalize_wf(&mut y_prev[basis_i], norm_pt, grid_n);
            util::normalize_wf(&mut y_next[basis_i], norm_pt, grid_n);
            util::normalize_wf(&mut z_prev[basis_i], norm_pt, grid_n);
            util::normalize_wf(&mut z_next[basis_i], norm_pt, grid_n);
        }

        Self {
            on_pt,
            x_prev,
            x_next,
            y_prev,
            y_next,
            z_prev,
            z_next,
        }
    }
}
