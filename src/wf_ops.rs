//! This module contains the bulk of the wave-function evalution and solving logic.
//!

//! todo: Important question about state and quantum numbers. You can't have more than one elec
//! etc in the same state (Pauli exclusion / fermion rules), but how does this apply when multiple
//! todo nuclei are involved?

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
    basis_wfs::Basis,
    complex_nums::Cplx,
    eigen_fns,
    num_diff,
    // rbf::Rbf,
    types::{Arr3d, Arr3dReal, Arr3dVec, SurfacesPerElec},
    util::{self},
};

use lin_alg2::f64::Vec3;

// We use Hartree units: ħ, elementary charge, electron mass, and Bohr radius.
const K_C: f64 = 1.;
pub const Q_PROT: f64 = 1.;
pub const Q_ELEC: f64 = -1.;
pub const M_ELEC: f64 = 1.;
pub const ħ: f64 = 1.;

pub(crate) const NUDGE_DEFAULT: f64 = 0.01;

// Compute these statically, to avoid continuous calls during excecution.

// Wave function number of values per edge.
// Memory use and some parts of computation scale with the cube of this.
pub const N: usize = 30;

#[derive(Clone, Copy, Debug)]
pub enum Spin {
    Up,
    Dn,
}

/// This is our main computation function for sfcs. It:
/// - Computes V from charges
/// - Computes a trial ψ from basis functions
/// - Computes ψ'' calculated, and measured from the trial ψ
/// Modifies in place to conserve memory. These operations are combined in the same function to
/// save computation, since they're often run at once, and can be iterated through using a single loop
/// through all grid points.
pub fn init_wf(
    bases: &[Basis],
    charges_fixed: &[(Vec3, f64)],
    sfcs: &mut SurfacesPerElec,
    E: f64,
    update_charges: bool,
    grid_min: &mut f64,
    grid_max: &mut f64,
    spacing_factor: f64,
    grid_posits: &mut Arr3dVec,
    bases_visible: &[bool],
    // Wave functions from other electrons, for calculating the Hartree potential.
    _charges_electron: &[Arr3dReal],
    _i_this_elec: usize,
) {
    // Set up the grid so that it smartly encompasses the charges, letting the WF go to 0
    // towards the edges
    // todo: For now, maintain a cubic grid centered on 0.
    if update_charges {
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
        update_grid_posits(grid_posits, *grid_min, *grid_max, spacing_factor);
    }

    // todo: Store these somewhere to save on computation? minor pt.
    // let grid_1d = util::linspace((*grid_min, *grid_max), N);

    // Our initial psi'' measured uses our analytic LCAO system, which doesn't have the
    // grid edge and precision issues of the fixed numerical grid we use to tune the trial
    // WF.
    // for (i, x) in grid_1d.iter().enumerate() {
    //     for (j, y) in grid_1d.iter().enumerate() {
    //         for (k, z) in grid_1d.iter().enumerate() {
    //             let posit_sample = Vec3::new(*x, *y, *z);

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                let posit_sample = grid_posits[i][j][k];

                // Calculate psi'' based on a numerical derivative of psi
                // in 3D.

                if update_charges {
                    sfcs.V[i][j][k] = 0.;

                    for (posit_charge, charge_amt) in charges_fixed.iter() {
                        sfcs.V[i][j][k] += V_coulomb(*posit_charge, posit_sample, *charge_amt);
                    }

                    // sfcs.V[i][j][k] += find_hartree_V(
                    //     charges_electron,
                    //     i_this_elec,
                    //     posit_sample,
                    //     grid_posits,
                    //     i,
                    //     j,
                    //     k,
                    // );
                }

                sfcs.psi[i][j][k] = Cplx::new_zero();

                for (basis_i, basis) in bases.iter().enumerate() {
                    let weight = if bases_visible[basis_i] {
                        basis.weight()
                    } else {
                        0.
                    };
                    sfcs.psi[i][j][k] += basis.value(posit_sample) * weight;
                }

                sfcs.psi_pp_calculated[i][j][k] =
                    eigen_fns::find_ψ_pp_calc(&sfcs.psi, &sfcs.V, E, i, j, k);

                // We can compute ψ'' measured this in the same loop here, since we're using an analytic
                // equation for ψ; we can diff at arbitrary points vice only along a grid of pre-computed ψ.
                sfcs.psi_pp_measured[i][j][k] = num_diff::find_ψ_pp_meas_fm_bases(
                    posit_sample,
                    bases,
                    sfcs.psi[i][j][k],
                    bases_visible,
                );
            }
        }
    }

    // // todo: Initial hack at updating our psi' to see what insight it may have, eg into momentum.
    // // todo: DRY on total vs each compoonent.
    // num_diff::find_ψ_p_meas_fm_grid_irreg(
    //     &sfcs.psi,
    //     &mut sfcs.psi_p_total_measured,
    //     &sfcs.grid_posits,
    //     num_diff::PsiPVar::Total,
    // );
    // num_diff::find_ψ_p_meas_fm_grid_irreg(
    //     &sfcs.psi,
    //     &mut sfcs.psi_px_measured,
    //     &sfcs.grid_posits,
    //     num_diff::PsiPVar::X,
    // );
    // num_diff::find_ψ_p_meas_fm_grid_irreg(
    //     &sfcs.psi,
    //     &mut sfcs.psi_py_measured,
    //     &sfcs.grid_posits,
    //     num_diff::PsiPVar::Y,
    // );
    // num_diff::find_ψ_p_meas_fm_grid_irreg(
    //     &sfcs.psi,
    //     &mut sfcs.psi_pz_measured,
    //     &sfcs.grid_posits,
    //     num_diff::PsiPVar::Z,
    // );
}

/// Score using the fidelity of psi'' calculated vs measured; |<psi_trial | psi_true >|^2.
/// This requires normalizing the wave functions we're comparing.
/// todo: Curretly not working.
/// todo: I don't think you can use this approach comparing psi''s with fidelity, since they're
/// todo not normalizsble.
// fn wf_fidelity(sfcs: &Surfaces) -> f64 {
fn fidelity(sfcs: &SurfacesPerElec) -> f64 {
    // "The accuracy should be scored by the fidelity of the wavefunction compared
    // to the true wavefunction. Fidelity is defined as |<psi_trial | psi_true >|^2.
    // For normalized states, this will always be bounded from above by 1.0. So it's
    // lower than 1.0 for an imperfect variational function, but is 1 if you are
    // able to exactly express it.""

    // For normalization.
    // let mut norm_sq_calc = 0.;
    // let mut norm_sq_meas = 0.;
    let mut norm_calc = Cplx::new_zero();
    let mut norm_meas = Cplx::new_zero();

    const SCORE_THRESH: f64 = 100.;

    // Create normalization const.
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
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

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
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
pub fn score_wf(sfcs: &SurfacesPerElec) -> f64 {
    let mut result = 0.;

    // Avoids numerical precision issues. Without this, certain values of N will lead
    // to a bogus score. Values of N both too high and too low can lead to this. Likely due to
    // if a grid value is too close to a charge source, the value baloons.
    const SCORE_THRESH: f64 = 10.;

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                // todo: Check if either individual is outside a thresh?

                let val =
                    (sfcs.psi_pp_calculated[i][j][k] - sfcs.psi_pp_measured[i][j][k]).abs_sq();
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
pub fn find_E(sfcs: &mut SurfacesPerElec, E: &mut f64) {
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
            for i in 0..N {
                for j in 0..N {
                    for k in 0..N {
                        sfcs.psi_pp_calculated[i][j][k] =
                            eigen_fns::find_ψ_pp_calc(&sfcs.psi, &sfcs.V, E_trial, i, j, k);
                    }
                }
            }

            let score = score_wf(sfcs);
            if score < best_score {
                best_score = score;
                best_E = E_trial;
                *E = E_trial;
            }
        }
        E_range_div2 /= vals_per_iter as f64; // todo: May need a wider range than this.
        E_min = best_E - E_range_div2;
        E_max = best_E + E_range_div2;
    }
}

/// A crude low pass
pub fn smooth_array(arr: &mut Arr3d, smoothing_amt: f64) {
    let orig = arr.clone();

    for i in 0..N {
        if i == 0 || i == N - 1 {
            continue;
        }
        for j in 0..N {
            if j == 0 || j == N - 1 {
                continue;
            }
            for k in 0..N {
                if k == 0 || k == N - 1 {
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
) {
    let grid_lin = util::linspace((grid_min, grid_max), N);

    // Set up a grid with values that increase in distance the farther we are from the center.
    let mut grid_1d = [0.; N];

    for i in 0..N {
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
