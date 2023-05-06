//! This module contains functionality to nudge a wave function to better match the
//! Schrodinger equation.

use crate::{
    basis_wfs::Basis,
    complex_nums::Cplx,
    eigen_fns, num_diff, types,
    types::{Arr3dVec, SurfacesPerElec},
    wf_ops,
};

// todo: Nudging is a good candidate for GPU. Try to impl in Vulkan / WGPU.

/// Apply a correction to the WF, in attempt to make our two psi''s closer.
/// Uses our numerically-calculated WF. Updates psi, and both psi''s.
pub fn nudge_wf(
    sfcs: &mut SurfacesPerElec,
    // wfs: &[Basis],
    // wfs: &[SlaterOrbital],
    // charges: &[(Vec3, f64)],
    nudge_amount: &mut f64,
    E: &mut f64,
    grid_min: f64,
    grid_max: f64,
    bases: &[Basis],
    grid_posits: &Arr3dVec,
    grid_n: usize,
) {
    let num_nudges = 1; // todo: Put back to 3 etc?
    let smooth_amt = 0.4;

    // Consider applying a lowpass after each nudge, and using a high nudge amount.
    // todo: Perhaps you lowpass the diffs... Make a grid of diffs, lowpass that,
    // todo, then nudge based on it. (?)
    // todo: Or, analyze diff result, and dynamically adjust nudge amount.

    // todo: Once out of the shower, look for more you can optimize out!

    // todo: Check for infinities etc around the edges

    // todo: Variational method and perterbation theory.

    // todo: Cheap lowpass for now on diff: Average it with its neighbors?

    // Find E before and after the nudge.
    wf_ops::find_E(sfcs, E, grid_n);

    // Really, the outliers are generally spiked very very high. (much higher than this)
    // This probably occurs near the nucleus.
    // Keep this low to avoid numerical precision issues.
    const OUTLIER_THRESH: f64 = 10.;

    // let x_vals = linspace((grid_min, grid_max), N);

    // todo: COnsider again if you can model how psi'' calc and measured
    // todo react to a change in psi, and try a nudge that sends them on a collision course

    // We revert to this if we've nudged too far.
    let mut psi_backup = sfcs.psi.clone();
    let mut psi_pp_calc_backup = sfcs.psi_pp_calculated.clone();
    let mut psi_pp_meas_backup = sfcs.psi_pp_measured.clone();
    let mut current_score = wf_ops::score_wf(sfcs, grid_n);

    // We use diff map so we can lowpass the entire map before applying corrections.
    let mut diff_map = types::new_data(grid_n);

    // `correction_fm_bases` is the difference between our nudged wave function, and what it
    // was from bases alone, without nudging.;
    let mut correction_fm_bases = types::new_data(grid_n);

    for _ in 0..num_nudges {
        // let mut diff_pre_smooth = new_data(N); // todo experimenting

        for i_a in 0..grid_n {
            for j in 0..grid_n {
                for k in 0..grid_n {
                    let diff = sfcs.psi_pp_calculated[i_a][j][k] - sfcs.psi_pp_measured[i_a][j][k];

                    // let psi_pp_calc_nudged = (sfcs.psi[i][j][k] + h.into())  * (E - sfcs.V[i][j][k]) * KE_COEFF;
                    // let psi_pp_meas_nudged = asdf
                    //
                    // let d_psi_pp_calc__d_psi = (psi_pp_calc_nudged - sfcs.psi_pp_calculated[i][j][k]) / h;
                    // let d_psi_pp_meas__d_psi = (psi_pp_meas_nudged - sfcs.psi_pp_measured[i][j][k]) / h;

                    // epxerimental approach to avoid anomolies. Likely from blown-up values
                    // near the nuclei.
                    // if diff.mag() > outlier_thresh {
                    if diff.real.abs() > OUTLIER_THRESH {
                        // Cheaper than mag()
                        // todo: Nudge amt?
                        continue;
                    }

                    diff_map[i_a][j][k] = diff;
                }
                // }
            } // todo: COmmenting this out, and adding one towards the bottom makes a dramatic improvement
              // todo, but why??!

            // Note: It turns out smoothing makes a big difference, as does the smoothing coefficient.
            // diff_pre_smooth = diff_map.clone();

            wf_ops::smooth_array(&mut diff_map, smooth_amt, grid_n);

            for i in 0..grid_n {
                for j in 0..grid_n {
                    for k in 0..grid_n {
                        // sfcs.psi[i][j][k] -= diff_map[i][j][k] * sfcs.nudge_amounts[i][j][k];
                        sfcs.psi[i][j][k] -= diff_map[i][j][k] * *nudge_amount;

                        sfcs.psi_pp_calculated[i][j][k] =
                            eigen_fns::find_ψ_pp_calc(&sfcs.psi, &sfcs.V, *E, i, j, k);
                    }
                }
            }

            // todo: Here lies one of the strange bracket mismatches that is helping our cause
            // todo (Uncomment one to engage the strange behavior)
        }

        // Calculated psi'' measured in a separate loop after updating psi, since it depends on
        // neighboring psi values as well.
        num_diff::find_ψ_pp_meas_fm_grid_irreg(
            &sfcs.psi,
            &mut sfcs.psi_pp_measured,
            grid_posits,
            grid_n,
        );

        // If you use individual nudges, evaluate how you want to handle this.
        let score = wf_ops::score_wf(sfcs, grid_n);

        // todo: Maybe update temp ones above instead of the main ones?
        if (score - current_score) > 0. {
            // We've nudged too much; revert.

            *nudge_amount *= 0.6;
            sfcs.psi = psi_backup.clone();
            sfcs.psi_pp_calculated = psi_pp_calc_backup.clone();
            sfcs.psi_pp_measured = psi_pp_meas_backup.clone();
        } else {
            // Our nudge was good; get a bit more aggressive.

            *nudge_amount *= 1.2;
            psi_backup = sfcs.psi.clone();
            psi_pp_calc_backup = sfcs.psi_pp_calculated.clone();
            psi_pp_meas_backup = sfcs.psi_pp_measured.clone();
            current_score = score;
            wf_ops::find_E(sfcs, E, grid_n);

            for i in 0..grid_n {
                for j in 0..grid_n {
                    for k in 0..grid_n {
                        let mut psi_bases = Cplx::new_zero();
                        let posit_sample = grid_posits[i][j][k];
                        for basis in bases {
                            psi_bases += basis.value(posit_sample) * basis.weight();
                        }

                        correction_fm_bases[i][j][k] = (sfcs.psi[i][j][k] - psi_bases) * 1.;
                    }
                }
            }
            sfcs.aux2 = correction_fm_bases.clone();
        }
        // todo: Update state's score here so we don't need to explicitly after.
    }

    sfcs.aux1 = diff_map;
}
