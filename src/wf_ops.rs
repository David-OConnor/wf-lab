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
    num_diff::{self, H, H_SQ},
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
    grid_min: f64,
    grid_max: f64,
    spacing_factor: f64,
    grid_posits: &mut Arr3dVec,
    n: usize,
    // Wave functions from other electrons, for calculating the Hartree potential.
    // charges_electron: &[Arr3dReal],
    // i_this_elec: usize,
) {
    update_grid_posits(grid_posits, grid_min, grid_max, spacing_factor, n);

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
pub fn normalize_wf(arr: &mut Arr3d, norm: f64, n: usize) -> f64 {
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

/// Mix bases together into psi at each grid point, and at diffs.
/// This is our mixer from pre-calculated basis fucntions: Create psi from summing them with
/// their weights; do the same for the ones offset, used to numerically differtiate.
pub fn mix_bases(
    bases: &[Basis],
    basis_wfs: &BasisWfsUnweighted,
    psi: &mut PsiWDiffs,
    bases_visible: &[bool],
    grid_n: usize,
    weights: Option<&[f64]>,
) {
    let mut norm = 0.;

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
                    let weight = match weights {
                        Some(w) => w[i_basis],
                        None => {
                            if bases_visible[i_basis] {
                                bases[i_basis].weight()
                            } else {
                                0.
                            }
                        }
                    };

                    psi.on_pt[i][j][k] += basis_wfs.on_pt[i_basis][i][j][k] * weight;
                    psi.x_prev[i][j][k] += basis_wfs.x_prev[i_basis][i][j][k] * weight;
                    psi.x_next[i][j][k] += basis_wfs.x_next[i_basis][i][j][k] * weight;
                    psi.y_prev[i][j][k] += basis_wfs.y_prev[i_basis][i][j][k] * weight;
                    psi.y_next[i][j][k] += basis_wfs.y_next[i_basis][i][j][k] * weight;
                    psi.z_prev[i][j][k] += basis_wfs.z_prev[i_basis][i][j][k] * weight;
                    psi.z_next[i][j][k] += basis_wfs.z_next[i_basis][i][j][k] * weight;
                }

                // todo: How should normalization work? Perhaps use the same const for all.
                // todo: This is where it should be, but consider how.
                norm += psi.on_pt[i][j][k].abs_sq();
            }
        }
    }

    let _psi_norm_sqrt = normalize_wf(&mut psi.on_pt, norm, grid_n);
    let _psi_norm_sqrt = normalize_wf(&mut psi.x_prev, norm, grid_n);
    let _psi_norm_sqrt = normalize_wf(&mut psi.x_next, norm, grid_n);
    let _psi_norm_sqrt = normalize_wf(&mut psi.y_prev, norm, grid_n);
    let _psi_norm_sqrt = normalize_wf(&mut psi.y_next, norm, grid_n);
    let _psi_norm_sqrt = normalize_wf(&mut psi.z_prev, norm, grid_n);
    let _psi_norm_sqrt = normalize_wf(&mut psi.z_next, norm, grid_n);
}

/// - Computes a trial ψ from basis functions. Computes it at each grid point, as well as
/// the 6 offset ones along the 3 axis used to numerically differentiate.
/// - Computes ψ'' calculated, and measured from the trial ψ
/// Modifies in place to conserve memory. These operations are combined in the same function to
/// save computation, since they're often run at once, and can be iterated through using a single loop
/// through all grid points.
pub fn update_wf_fm_bases(
    bases: &[Basis],
    basis_wfs: &BasisWfsUnweighted,
    sfcs: &mut SurfacesPerElec,
    E: f64,
    // grid_posits: &Arr3dVec,
    bases_visible: &[bool],
    grid_n: usize,
    weights: &[f64],
) {
    mix_bases(bases, basis_wfs, &mut sfcs.psi, bases_visible, grid_n, None);

    // Update psi_pps after normalization. We can't rely on cached wfs here, since we need to
    // take infinitessimal differences on the analytic basis equations to find psi'' measured.
    update_psi_pps_from_bases(
        &sfcs.psi,
        &sfcs.V,
        &mut sfcs.psi_pp_calculated,
        &mut sfcs.psi_pp_measured,
        E,
        grid_n,
    );
}

/// Update psi'' calc and psi'' measured, assuming we are using basis WFs. This is done
/// after the wave function is contructed and normalized.
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
                            eigen_fns::find_ψ_pp_calc(&sfcs.psi.on_pt, &sfcs.V, E_trial, i, j, k);
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
    let mut prev_weights = Vec::new();
    for basis in bases.iter() {
        prev_weights.push(basis.weight());
    }

    *bases = Vec::new();
    *bases_visible = Vec::new();
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
                    let weight = if i < prev_weights.len() {
                        prev_weights[i]
                    } else {
                        0.
                    };

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

                    bases_visible.push(true);
                }
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
    /// and when changing the grid size. Each basis will be normalized in this function.
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

            let mut norm_x_prev = 0.;
            let mut norm_x_next = 0.;
            let mut norm_y_prev = 0.;
            let mut norm_y_next = 0.;
            let mut norm_z_prev = 0.;
            let mut norm_z_next = 0.;

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
                        norm_x_prev += val_x_prev.abs_sq();
                        norm_x_next += val_x_next.abs_sq();
                        norm_y_prev += val_y_prev.abs_sq();
                        norm_y_next += val_y_next.abs_sq();
                        norm_z_prev += val_z_prev.abs_sq();
                        norm_z_next += val_z_next.abs_sq();
                    }
                }
            }

            normalize_wf(&mut on_pt[basis_i], norm_pt, grid_n);

            // todo: Normalize all the same, or per const?
            // normalize_wf(&mut x_prev[basis_i], norm_x_prev, grid_n);
            // normalize_wf(&mut x_next[basis_i], norm_x_next, grid_n);
            // normalize_wf(&mut y_prev[basis_i], norm_y_prev, grid_n);
            // normalize_wf(&mut y_next[basis_i], norm_y_next, grid_n);
            // normalize_wf(&mut z_prev[basis_i], norm_z_prev, grid_n);
            // normalize_wf(&mut z_next[basis_i], norm_z_next, grid_n);

            normalize_wf(&mut x_prev[basis_i], norm_pt, grid_n);
            normalize_wf(&mut x_next[basis_i], norm_pt, grid_n);
            normalize_wf(&mut y_prev[basis_i], norm_pt, grid_n);
            normalize_wf(&mut y_next[basis_i], norm_pt, grid_n);
            normalize_wf(&mut z_prev[basis_i], norm_pt, grid_n);
            normalize_wf(&mut z_next[basis_i], norm_pt, grid_n);
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
