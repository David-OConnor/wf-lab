//! This program explores solving the wave equation for
//! arbitrary potentials. It visualizes the wave function in 3d, with user interaction.

// todo: Consider instead of H orbitals, use the full set of Slater basis
// functions, which are more general. Make a fn to generate them.

// todo: Hylleraas basis functions?

#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(uncommon_codepoints)]

use lin_alg2::f64::{Quaternion, Vec3};

mod basis_wfs;
mod complex_nums;
mod render;
mod ui;

use basis_wfs::{Basis, HOrbital, SphericalHarmonic, Sto};
use complex_nums::Cplx;

const NUM_SURFACES: usize = 6;

// We use Hartree units: ħ, elementary charge, electron mass, and Bohr radius.
const K_C: f64 = 1.;
const Q_PROT: f64 = 1.;
const Q_ELEC: f64 = -1.;
const M_ELEC: f64 = 1.;
const ħ: f64 = 1.;

// Compute these statically, to avoid continuous calls during excecution.
const KE_COEFF: f64 = -2. * M_ELEC / (ħ * ħ);
const KE_COEFF_INV: f64 = 1. / KE_COEFF;

// Wave function number of values per edge.
// Memory use and some parts of computation scale with the cube of this.
const N: usize = 50;

// Used for calculating numerical psi''.
// Smaller is more precise. Too small might lead to numerical issues though (?)
// Applies to dx, dy, and dz
const H: f64 = 0.01;
const H_SQ: f64 = H * H;

// Of note, these grids need to extend beyond where the WF curves, or we get poor
// results. I'm not sure why, but it may have to do with making sure there is
// little curvature at the edges.
// const GRID_MIN: f64 = -16.;
// const GRID_MAX: f64 = 16.;

// todo: Consider a spherical grid centered perhaps on the system center-of-mass, which
// todo less precision further away?

// type Arr3d = Vec<Vec<Vec<f64>>>;
type Arr3dReal = Vec<Vec<Vec<f64>>>;
type Arr3d = Vec<Vec<Vec<Cplx>>>;

/// Make a new 3D grid, as a nested Vec
fn new_data_real(n: usize) -> Arr3dReal {
    let mut z = Vec::new();
    z.resize(n, 0.);

    let mut y = Vec::new();
    y.resize(n, z);

    let mut x = Vec::new();
    x.resize(n, y);

    x
}

/// Make a new 3D grid, as a nested Vec
fn new_data(n: usize) -> Arr3d {
    let mut z = Vec::new();
    z.resize(n, Cplx::new_zero());
    // z.resize(N, 0.);

    let mut y = Vec::new();
    y.resize(n, z);

    let mut x = Vec::new();
    x.resize(n, y);

    x
}

/// Represents important data, in describe 3D arrays.
/// We use Vecs, since these can be large, and we don't want
/// to put them on the stack. Although, they are fixed-size.
/// todo: Change name?
pub struct Surfaces {
    pub V: Arr3dReal,
    pub psi: Arr3d,
    pub psi_pp_calculated: Arr3d,
    pub psi_pp_measured: Arr3d,
    /// Aux surfaces are for misc visualizations
    pub aux1: Arr3d,
    pub aux2: Arr3dReal,
    pub elec_charges: Vec<Arr3dReal>,
}

impl Default for Surfaces {
    /// Fills with 0.s
    fn default() -> Self {
        let data = new_data(N);
        let data_real = new_data_real(N);

        Self {
            V: data_real.clone(),
            psi: data.clone(),
            psi_pp_calculated: data.clone(),
            psi_pp_measured: data.clone(),
            aux1: data.clone(),
            aux2: data_real.clone(),
            // todo: For now, an empty one.
            elec_charges: vec![data_real],
        }
    }
}

pub struct State {
    /// todo: Combine wfs and nuclei in into single tuple etc to enforce index pairing?
    /// todo: Or a sub struct?
    /// Wave functions, with weights
    pub bases: Vec<Basis>,
    // pub wfs: Vec<SlaterOrbital>, // todo: Rename, eg `bases`
    // todo use an index for them.
    /// Nuclei. todo: H only for now.
    pub charges: Vec<(Vec3, f64)>,
    /// Computed surfaces, with name.
    pub surfaces: Surfaces,
    // pub surfaces: [&'static Arr3d; NUM_SURFACES],
    // pub surfaces: [Arr3d; NUM_SURFACES],
    /// Eg, least-squares over 2 or 3 dimensions between
    /// When visualizing a 2d wave function over X and Y, this is the fixed Z value.
    pub z_displayed: f64,
    /// Energy of the system
    pub E: f64,
    pub psi_pp_score: f64,
    /// Surface name
    pub surface_names: [String; NUM_SURFACES],
    pub show_surfaces: [bool; NUM_SURFACES],
    pub grid_n: usize,
    pub grid_min: f64,
    pub grid_max: f64,
    pub nudge_amount: f64,
    // vals for h_grid and h_grid_sq, need to be updated when grid changes. A cache.
    // For finding psi_pp_meas, using only values on the grid
    pub h_grid: f64,
    pub h_grid_sq: f64,
}

/// Score using the fidelity of psi'' calculated vs measured; |<psi_trial | psi_true >|^2.
/// This requires normalizing the wave functions we're comparing.
/// todo: Curretly not working.
/// todo: I don't think you can use this approach comparing psi''s with fidelity, since they're
/// todo not normalizsble.
// fn wf_fidelity(sfcs: &Surfaces) -> f64 {
fn fidelity(sfcs: &Surfaces) -> f64 {
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
fn score_wf(sfcs: &Surfaces) -> f64 {
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

/// Convert an array of Psi to one of electron charge through space. Modifies in place
/// to avoid unecessary allocations.
fn charge_density_fm_psi(psi: &Arr3d, V_elec: &mut Arr3dReal, num_elecs: usize) {
    // Normalize <ψ|ψ>
    let mut psi_sq_size = 0.;
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                psi_sq_size += psi[i][j][k].abs_sq();
            }
        }
    }

    // Save computation on this constant factor.
    let c = -Q_ELEC * num_elecs as f64 / psi_sq_size;

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                V_elec[i][j][k] = psi[i][j][k].abs_sq() * c;
            }
        }
    }
}

/// Convert an array of Psi to one of electron potential. Modifies in place
/// to avoid unecessary allocations. Not-normalized.
fn charge_density_fm_psi_one(psi: &Arr3d, num_elecs: usize, i: usize, j: usize, k: usize) -> f64 {
    // Save computation on this constant factor.
    let psi_sq_size = 1.; // todo: Wrong! This should be a normalization constant.
    let c = -Q_ELEC * num_elecs as f64 / psi_sq_size;
    let mag = psi[i][j][k].abs_sq();
    mag * c
}

/// Single-point Coulomb potential, eg a hydrogen nuclei.
fn V_coulomb(posit_charge: Vec3, posit_sample: Vec3, charge: f64) -> f64 {
    let diff = posit_sample - posit_charge;
    let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

    -K_C * charge / r
}

/// Create a set of values in a given range, with a given number of values.
/// Similar to `numpy.linspace`.
/// The result terminates one step before the end of the range.
fn linspace(range: (f64, f64), num_vals: usize) -> Vec<f64> {
    let step = (range.1 - range.0) / num_vals as f64;

    let mut result = Vec::new();

    let mut val = range.0;
    for _ in 0..num_vals {
        result.push(val);
        val += step;
    }

    result
}

/// Calcualte psi'', calculated from psi, and E.
/// At a given i, j, k.
fn find_ψ_pp_calc(psi: &Arr3d, V: &Arr3dReal, E: f64, i: usize, j: usize, k: usize) -> Cplx {
    // todo: Do you need to multiply KE_COEFF, or part of it, by the number of electrons
    // todo: in this WF?
    psi[i][j][k] * (E - V[i][j][k]) * KE_COEFF
}

/// Calcualte ψ'', numerically from ψ, using the finite diff method, for a single value.
fn find_ψ_pp_meas(
    psi: &Arr3d,
    posit_sample: Vec3,
    bases: &[Basis],
    charges: &[(Vec3, f64)],
    i: usize,
    j: usize,
    k: usize,
) -> Cplx {
    // Calculate ψ'' based on a numerical derivative of psi
    // in 3D.

    let x_prev = Vec3::new(posit_sample.x - H, posit_sample.y, posit_sample.z);
    let x_next = Vec3::new(posit_sample.x + H, posit_sample.y, posit_sample.z);
    let y_prev = Vec3::new(posit_sample.x, posit_sample.y - H, posit_sample.z);
    let y_next = Vec3::new(posit_sample.x, posit_sample.y + H, posit_sample.z);
    let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - H);
    let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + H);

    let mut psi_x_prev = Cplx::new_zero();
    let mut psi_x_next = Cplx::new_zero();
    let mut psi_y_prev = Cplx::new_zero();
    let mut psi_y_next = Cplx::new_zero();
    let mut psi_z_prev = Cplx::new_zero();
    let mut psi_z_next = Cplx::new_zero();

    for basis in bases {
        psi_x_prev += basis.value(x_prev) * basis.weight();
        psi_x_next += basis.value(x_next) * basis.weight();
        psi_y_prev += basis.value(y_prev) * basis.weight();
        psi_y_next += basis.value(y_next) * basis.weight();
        psi_z_prev += basis.value(z_prev) * basis.weight();
        psi_z_next += basis.value(z_next) * basis.weight();
    }

    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi[i][j][k] * 6.;

    result / H_SQ
}

/// Find the E that minimizes score, by narrowing it down. Note that if the relationship
/// between E and psi'' score isn't straightforward, this will converge on a local minimum.
fn find_E(sfcs: &mut Surfaces, E: &mut f64) {
    // todo: WHere to configure these mins and maxes
    let mut E_min = -2.;
    let mut E_max = 2.;
    let mut E_range_div2 = 2.;
    let vals_per_iter = 8;

    let num_iters = 10;

    for _ in 0..num_iters {
        let E_vals = linspace((E_min, E_max), vals_per_iter);
        let mut best_score = 100_000_000.;
        let mut best_E = 0.;

        for E_trial in E_vals {
            for i in 0..N {
                for j in 0..N {
                    for k in 0..N {
                        sfcs.psi_pp_calculated[i][j][k] =
                            find_ψ_pp_calc(&sfcs.psi, &sfcs.V, E_trial, i, j, k);
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

// /// Interpolate a value from a discrete wave function, assuming (what about curvature)
// fn interp_wf(psi: &Arr3d, posit_sample: Vec3) -> Cplx {
//     // Maybe a polynomial?
// }

/// A crude low pass
fn smooth_array(arr: &mut Arr3d, smoothing_amt: f64) {
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

/// Apply a correction to the WF, in attempt to make our two psi''s closer.
/// Uses our numerically-calculated WF. Updates psi, and both psi''s.
fn nudge_wf(
    sfcs: &mut Surfaces,
    wfs: &[Basis],
    // wfs: &[SlaterOrbital],
    charges: &[(Vec3, f64)],
    nudge_amount: &mut f64,
    E: &mut f64,
    grid_min: f64,
    grid_max: f64,
) {
    let num_nudges = 1;

    // let nudge_width = 0.1;

    // Consider applying a lowpass after each nudge, and using a high nudge amount.
    // todo: Perhaps you lowpass the diffs... Make a grid of diffs, lowpass that,
    // todo, then nudge based on it. (?)
    // todo: Or, analyze diff result, and dynamically adjust nudge amount.

    // todo: Once out of the shower, look for more you can optimize out!

    // todo: Check for infinities etc around the edges

    // todo: Variational method and perterbation theory.

    // todo: Cheap lowpass for now on diff: Average it with its neighbors?

    // Find E before and after the nudge.
    find_E(sfcs, E);

    // Really, the outliers are generally spiked very very high. (much higher than this)
    // This probably occurs near the nucleus.
    // Keep this low to avoid numerical precision issues.
    const OUTLIER_THRESH: f64 = 10.;

    let x_vals = linspace((grid_min, grid_max), N);

    // todo: COnsider again if you can model how psi'' calc and measured
    // todo react to a change in psi, and try a nudge that sends them on a collision course

    // We revert to this if we've nudged too far.
    let mut psi_backup = sfcs.psi.clone();
    let mut psi_pp_calc_backup = sfcs.psi_pp_calculated.clone();
    let mut psi_pp_meas_backup = sfcs.psi_pp_measured.clone();
    let mut current_score = score_wf(sfcs);

    // We use diff map so we can lowpass the entire map before applying corrections.
    let mut diff_map = new_data(N);

    let dx = (grid_max - grid_min) / N as f64;
    let divisor = (dx).powi(2);

    let h = 0.0001; // todo

    for _ in 0..num_nudges {
        // for _ in 0..num_nudges {
        for (i, x) in x_vals.iter().enumerate() {
            for (j, y) in x_vals.iter().enumerate() {
                for (k, z) in x_vals.iter().enumerate() {
                    // let posit_sample = Vec3::new(*x, *y, *z);

                    let diff = sfcs.psi_pp_calculated[i][j][k] - sfcs.psi_pp_measured[i][j][k];

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

                    diff_map[i][j][k] = diff;

                    // sfcs.psi[i][j][k] -= diff * *nudge_amount;
                    // sfcs.aux1[i][j][k] = diff;
                }
            }

            smooth_array(&mut diff_map, 0.4);

            // todo: DRY with eval_wf

            for (i, x) in x_vals.iter().enumerate() {
                if i == 0 || i == N - 1 {
                    continue;
                }
                for (j, y) in x_vals.iter().enumerate() {
                    if j == 0 || j == N - 1 {
                        continue;
                    }
                    for (k, z) in x_vals.iter().enumerate() {
                        if k == 0 || k == N - 1 {
                            continue;
                        }

                        sfcs.aux1[i][j][k] = diff_map[i][j][k]; // post smooth

                        sfcs.psi[i][j][k] -= diff_map[i][j][k] * *nudge_amount;

                        // todo: Experimenting with nudging neighbors too.
                        // sfcs.psi[i-1][j][k] += diff_map[i][j][k] * *nudge_amount;
                        // sfcs.psi[i+1][j][k] += diff_map[i][j][k] * *nudge_amount;
                        // sfcs.psi[i][j-1][k] += diff_map[i][j][k] * *nudge_amount;
                        // sfcs.psi[i][j+1][k] += diff_map[i][j][k] * *nudge_amount;
                        // sfcs.psi[i][j][k-1] += diff_map[i][j][k] * *nudge_amount;
                        // sfcs.psi[i][j][k+1] += diff_map[i][j][k] * *nudge_amount;

                        // let posit_sample = Vec3::new(*x, *y, *z);

                        // todo: Maybe you can wrap up the psi and /or psi calc into the psi_pp_meas fn?

                        // sfcs.psi_pp_measured[i][j][k] =
                        //     find_psi_pp_meas(&sfcs.psi, posit_sample, wfs, charges, gauss, i, j, k);

                        // let x_prev = Vec3::new(posit_sample.x - H, posit_sample.y, posit_sample.z);
                        // let x_next = Vec3::new(posit_sample.x + H, posit_sample.y, posit_sample.z);
                        // let y_prev = Vec3::new(posit_sample.x, posit_sample.y - H, posit_sample.z);
                        // let y_next = Vec3::new(posit_sample.x, posit_sample.y + H, posit_sample.z);
                        // let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - H);
                        // let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + H);

                        // let mut psi_x_prev = interp_wf(&sfcs.psi, x_prev);
                        // let mut psi_x_next = interp_wf(&sfcs.psi, x_next);
                        // let mut psi_y_prev = interp_wf(&sfcs.psi, y_prev);
                        // let mut psi_y_next = interp_wf(&sfcs.psi, y_next);
                        // let mut psi_z_prev = interp_wf(&sfcs.psi, z_prev);
                        // let mut psi_z_next = interp_wf(&sfcs.psi, z_next);

                        sfcs.psi_pp_calculated[i][j][k] =
                            find_ψ_pp_calc(&sfcs.psi, &sfcs.V, *E, i, j, k);

                        // Don't calcualte extrapolated edge diffs; they'll be sifniciantly off from calculated.

                        let mut psi_x_prev = sfcs.psi[i - 1][j][k];
                        let mut psi_x_next = sfcs.psi[i + 1][j][k];
                        let mut psi_y_prev = sfcs.psi[i][j - 1][k];
                        let mut psi_y_next = sfcs.psi[i][j + 1][k];
                        let mut psi_z_prev = sfcs.psi[i][j][k - 1];
                        let mut psi_z_next = sfcs.psi[i][j][k + 1];

                        let result = psi_x_prev
                            + psi_x_next
                            + psi_y_prev
                            + psi_y_next
                            + psi_z_prev
                            + psi_z_next
                            - sfcs.psi[i][j][k] * 6.;

                        sfcs.psi_pp_measured[i][j][k] = result / divisor;
                    }
                }
            }
        }
        let score = score_wf(sfcs);

        // todo: Maybe update temp ones above instead of the main ones?
        // if score > current_score {
        if (score - current_score) > 0. {
            // hacky
            // We've nudged too much; revert.
            *nudge_amount *= 0.5;
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
            find_E(sfcs, E);
        }
        // todo: Update state's score here so we don't need to explicitly after.
    }
}

/// todo: This should probably be a method on `State`.
/// This is our main computation function for sfcs.
/// Modifies in place to conserve memory.
fn eval_wf(
    bases: &[Basis],
    // bases: &[SlaterOrbital],
    charges: &[(Vec3, f64)],
    sfcs: &mut Surfaces,
    E: f64,
    update_charges: bool,
    grid_min: f64,
    grid_max: f64,
) {
    // output score. todo: Move score to a diff fn?
    // ) -> (Vec<(Arr3d, String)>, f64) {
    // Schrod eq for H:
    // V for hydrogen: K_C * Q_PROT / r

    // psi(r)'' = (E - V(r)) * 2*m/ħ**2 * psi(r)
    // psi(r) = (E - V(R))^-1 * ħ**2/2m * psi(r)''

    // todo: Store these somewhere to save on computation? minor pt.
    let vals_1d = linspace((grid_min, grid_max), N);

    // Our initial psi'' measured uses our analytic LCAO system, which doesn't have the
    // grid edge and precision issues of the fixed numerical grid we use to tune the trial
    // WF.
    for (i, x) in vals_1d.iter().enumerate() {
        for (j, y) in vals_1d.iter().enumerate() {
            for (k, z) in vals_1d.iter().enumerate() {
                let posit_sample = Vec3::new(*x, *y, *z);

                // Calculate psi'' based on a numerical derivative of psi
                // in 3D.

                if update_charges {
                    sfcs.V[i][j][k] = 0.;
                    for (posit_charge, charge_amt) in charges.iter() {
                        sfcs.V[i][j][k] += V_coulomb(*posit_charge, posit_sample, *charge_amt);
                    }

                    // Re why the electron interaction, in many cases, appears to be very small compared to protons: After thinking about it, the protons, being point charges (approximately) are pulling from a single direction. While most of the smudged out electron gets cancelled out in the area of interest
                    // But, it should follow that at a distance, the electsron force and potential is as strong as the proton's
                    // (Yet, at a distance, the electron and proton charges cancel each other out largely, unless it's an ion...)
                    // So I guess it follows that the interesting bits are in the intermediate distances...
                    // todo: Hard coded ito index 0.

                    // Oh boy... this will slow things down... Simulating a charge at every grid point.,
                    // acting on every other grid point.

                    // todo: This is going to be a deal breaker most likely.
                    // for (i2, x2) in vals_1d.iter().enumerate() {
                    //     for (j2, y2) in vals_1d.iter().enumerate() {
                    //         for (k2, z2) in vals_1d.iter().enumerate() {
                    //             let posit_sample_electron = Vec3::new(*x2, *y2, *z2);
                    //             // todo: This may not be quite right, ie matching the posit_sample grid with the i2, j2, k2 elec charges.
                    //             sfcs.V[i][j][k] += V_coulomb(posit_sample_electron, posit_sample, sfcs.elec_charges[0][i2][j2][k2]);
                    //         }
                    //     }
                    // }
                }

                sfcs.psi[i][j][k] = Cplx::new_zero();
                for basis in bases {
                    sfcs.psi[i][j][k] += basis.value(posit_sample) * basis.weight();
                }

                sfcs.psi_pp_calculated[i][j][k] = find_ψ_pp_calc(&sfcs.psi, &sfcs.V, E, i, j, k);

                sfcs.psi_pp_measured[i][j][k] =
                    find_ψ_pp_meas(&sfcs.psi, posit_sample, bases, charges, i, j, k);

                // sfcs.aux2[i][j][k] = charge_density_fm_psi_one(&sfcs.psi, 1, i, j, k);
            }
        }
    }
}

fn main() {
    let posit_charge_1 = Vec3::new(-1., 0., 0.);
    let posit_charge_2 = Vec3::new(1., 0., 0.);

    let neutral = Quaternion::new_identity();

    // todo: Clean up constructor sequene for these basis fns A/R.
    let wfs = vec![
        Basis::H(HOrbital::new(
            posit_charge_1,
            1,
            SphericalHarmonic::default(),
            1.,
            0,
        )),
        Basis::H(HOrbital::new(
            posit_charge_2,
            1,
            SphericalHarmonic::default(),
            1.,
            1,
        )),
        Basis::H(HOrbital::new(
            posit_charge_1,
            2,
            SphericalHarmonic::default(),
            0.,
            0,
        )),
        Basis::H(HOrbital::new(
            posit_charge_2,
            2,
            SphericalHarmonic::default(),
            0.,
            1,
        )),
        Basis::H(HOrbital::new(
            posit_charge_1,
            2,
            SphericalHarmonic::new(1, 0, neutral),
            0.,
            0,
        )),
        Basis::H(HOrbital::new(
            posit_charge_2,
            2,
            SphericalHarmonic::new(1, 0, neutral),
            0.,
            1,
        )),
        Basis::H(HOrbital::new(
            posit_charge_1,
            3,
            SphericalHarmonic::default(),
            0.,
            0,
        )),
        Basis::Sto(Sto::new(
            posit_charge_1,
            1,
            SphericalHarmonic::default(),
            1.,
            0.,
            1,
        )),
        // Basis::new(0, BasisFn::H100, posit_charge_1, 1.),
        // Basis::new(1, BasisFn::H100, posit_charge_2, -1.),
        // Basis::new(0, BasisFn::H200, posit_charge_1, 0.),
        // Basis::new(1, BasisFn::H200, posit_charge_2, 0.),
        // Basis::new(0, BasisFn::H210(x_axis), posit_charge_1, 0.),
        // Basis::new(1, BasisFn::H210(x_axis), posit_charge_2, 0.),
        // Basis::new(0, BasisFn::H300, posit_charge_1, 0.),
        // Basis::new(1, BasisFn::Sto(1.), posit_charge_2, 0.),
    ];

    // let gaussians = vec![Gaussian::new_symmetric(Vec3::new(0., 0., 0.), 0.1, 2.)];
    // let gaussians = Vec::new();

    // H ion nuc dist is I believe 2 bohr radii.
    // let charges = vec![(Vec3::new(-1., 0., 0.), Q_PROT), (Vec3::new(1., 0., 0.), Q_PROT)];
    let charges = vec![
        (posit_charge_1, Q_PROT),
        (posit_charge_2, Q_PROT),
        // (Vec3::new(0., 1., 0.), Q_ELEC),
    ];

    let z_displayed = 0.;
    let E = -0.7;
    let (grid_min, grid_max) = (-6., 6.);

    let h_grid = (grid_max - grid_min) / (N as f64);
    let h_grid_sq = h_grid.powi(2);

    let mut sfcs = Default::default();

    eval_wf(&wfs, &charges, &mut sfcs, E, true, grid_min, grid_max);

    let psi_pp_score = score_wf(&sfcs);

    let show_surfaces = [true, true, true, true, false, false];

    let surface_names = [
        "V".to_owned(),
        "ψ".to_owned(),
        "ψ'' calculated".to_owned(),
        "ψ'' measured".to_owned(),
        "Aux 1".to_owned(),
        "Aux 2".to_owned(),
    ];

    // let z = vec![4; N];
    // let y = vec![z; N];
    // let grid_divisions = vec![y; N];

    let state = State {
        bases: wfs,
        charges,
        surfaces: sfcs,
        E,
        z_displayed,
        psi_pp_score,
        surface_names,
        show_surfaces,
        // grid_divisions,
        // gaussians,
        grid_n: N,
        grid_min,
        grid_max,
        nudge_amount: 0.002,
        h_grid,
        h_grid_sq,
    };

    render::render(state);
}
