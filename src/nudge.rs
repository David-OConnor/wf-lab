//! This module contains functionality to nudge a wave function to better match the
//! Schrodinger equation.

use crate::{
    num_diff, util,
    wf_ops::{self, Surfaces, N},
};

use lin_alg2::f64::Vec3;

// todo: Nudging is a good candidate for GPU. Try to impl in Vulkan / WGPU.

/// Apply a correction to the WF, in attempt to make our two psi''s closer.
/// Uses our numerically-calculated WF. Updates psi, and both psi''s.
pub fn nudge_wf(
    sfcs: &mut Surfaces,
    // wfs: &[Basis],
    // wfs: &[SlaterOrbital],
    // charges: &[(Vec3, f64)],
    nudge_amount: &mut f64,
    E: &mut f64,
    grid_min: f64,
    grid_max: f64,
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
    wf_ops::find_E(sfcs, E);

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
    let mut current_score = wf_ops::score_wf(sfcs);

    // We use diff map so we can lowpass the entire map before applying corrections.
    let mut diff_map = wf_ops::new_data(N);

    let dx = (grid_max - grid_min) / N as f64;
    let grid_posits = util::linspace((grid_min, grid_max), N);

    for _ in 0..num_nudges {
        // let mut diff_pre_smooth = new_data(N); // todo experimenting

        for i_a in 0..N {
            for j in 0..N {
                for k in 0..N {
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

            wf_ops::smooth_array(&mut diff_map, smooth_amt);

            for i in 0..N {
                for j in 0..N {
                    for k in 0..N {
                        // sfcs.psi[i][j][k] -= diff_map[i][j][k] * sfcs.nudge_amounts[i][j][k];
                        sfcs.psi[i][j][k] -= diff_map[i][j][k] * *nudge_amount;

                        sfcs.psi_pp_calculated[i][j][k] =
                            wf_ops::find_ψ_pp_calc(&sfcs.psi, &sfcs.V, *E, i, j, k);
                    }
                }
            }

            // Calculated psi'' measured in a separate loop after updating psi, since it depends on
            // neighboring psi values as well.
            num_diff::find_ψ_pp_meas_fm_grid_irreg(
                &sfcs.psi,
                &mut sfcs.psi_pp_measured,
                &sfcs.grid_posits,
            );
            // todo: Here lies one of the strange bracket mismatches that is helping our cause
            // todo (Uncomment one to engage the strange behavior)
        }

        // If you use individual nudges, evaluate how you want to handle this.
        let score = wf_ops::score_wf(sfcs);

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
            wf_ops::find_E(sfcs, E);
        }
        // todo: Update state's score here so we don't need to explicitly after.
    }

    sfcs.aux1 = diff_map.clone();
}
