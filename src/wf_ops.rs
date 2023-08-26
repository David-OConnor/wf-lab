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
    basis_wfs::{Basis, Gto, HOrbital, SphericalHarmonic, Sto},
    complex_nums::Cplx,
    eigen_fns::{self, KE_COEFF}, eval, grid_setup,
    grid_setup::{new_data, Arr3d, Arr3dReal, Arr3dVec},
    num_diff::{self, H},
    types::{SurfacesPerElec, EvalDataPerElec},
    util,
};

use lin_alg2::f64::{Quaternion, Vec3};

// We use Hartree units: ħ, elementary charge, electron mass, and Bohr radius.
pub const K_C: f64 = 1.;
pub const Q_PROT: f64 = 1.;
pub const Q_ELEC: f64 = -1.;
pub const M_ELEC: f64 = 1.;
pub const ħ: f64 = 1.;

pub(crate) const NUDGE_DEFAULT: f64 = 0.01;

// Compute these statically, to avoid continuous calls during excecution.

// Wave function number of values per edge.
// Memory use and some parts of computation scale with the cube of this.
// pub const N: usize = 20;

#[derive(Clone, Copy, Debug)]
pub enum Spin {
    Up,
    Dn,
}

// todo: QC if the individual Vs you're adding here already have nuc baked in; I think they do!
// todo: You should likely split them off.

/// Mix bases together into a numerical wave function at each grid point, and at diffs.
/// This is our mixer from pre-calculated basis fucntions: Create psi, including at
/// neighboring points (used to numerically differentiate), from summing them with
/// their weights. Basis wfs must be initialized prior to running this, and weights must
/// be selected.
///
/// The resulting wave functions are normalized.
pub fn mix_bases(
    psi: &mut PsiWDiffs,
    bases_evaled: &BasesEvaluated,
    grid_n: usize,
    weights: &[f64],
) {
    // We don't need to normalize the result using the full procedure; the basis-wfs are already
    // normalized, so divide by the cumulative basis weights.
    let mut weight_total = 0.;
    for weight in weights {
        weight_total += weight.abs();
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
                psi.psi_pp_analytic[i][j][k] = Cplx::new_zero();
                psi.x_prev[i][j][k] = Cplx::new_zero();
                psi.x_next[i][j][k] = Cplx::new_zero();
                psi.y_prev[i][j][k] = Cplx::new_zero();
                psi.y_next[i][j][k] = Cplx::new_zero();
                psi.z_prev[i][j][k] = Cplx::new_zero();
                psi.z_next[i][j][k] = Cplx::new_zero();

                for (i_basis, weight) in weights.iter().enumerate() {
                    let scaled = weight * norm_scaler;

                    psi.on_pt[i][j][k] += bases_evaled.on_pt[i_basis][i][j][k] * scaled;
                    psi.psi_pp_analytic[i][j][k] +=
                        bases_evaled.psi_pp_analytic[i_basis][i][j][k] * scaled;
                    psi.x_prev[i][j][k] += bases_evaled.x_prev[i_basis][i][j][k] * scaled;
                    psi.x_next[i][j][k] += bases_evaled.x_next[i_basis][i][j][k] * scaled;
                    psi.y_prev[i][j][k] += bases_evaled.y_prev[i_basis][i][j][k] * scaled;
                    psi.y_next[i][j][k] += bases_evaled.y_next[i_basis][i][j][k] * scaled;
                    psi.z_prev[i][j][k] += bases_evaled.z_prev[i_basis][i][j][k] * scaled;
                    psi.z_next[i][j][k] += bases_evaled.z_next[i_basis][i][j][k] * scaled;
                }
            }
        }
    }
}

/// Eg for the psi-on-grid for charges, where we don't need to generate psi''.
// todo: DRY
pub fn mix_bases_no_diffs(psi: &mut Arr3d, bases_evaled: &[Arr3d], grid_n: usize, weights: &[f64]) {
    // We don't need to normalize the result using the full procedure; the basis-wfs are already
    // normalized, so divide by the cumulative basis weights.
    let mut weight_total = 0.;
    for weight in weights {
        weight_total += weight.abs();
    }

    let mut norm_scaler = 1. / weight_total;

    // Prevents NaNs and related complications.
    if weight_total.abs() < 0.000001 {
        norm_scaler = 0.;
    }

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                psi[i][j][k] = Cplx::new_zero();

                for (i_basis, weight) in weights.iter().enumerate() {
                    let scaled = weight * norm_scaler;

                    psi[i][j][k] += bases_evaled[i_basis][i][j][k] * scaled;
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
pub fn mix_bases_1d(
    psi: &mut PsiWDiffs1d,
    bases_evaled: &BasesEvaluated1d,
    grid_n: usize,
    weights: &[f64],
) {
    // We don't need to normalize the result using the full procedure; the basis-wfs are already
    // normalized, so divide by the cumulative basis weights.
    let mut weight_total = 0.;

    for weight in weights {
        weight_total += weight.abs();
    }

    let mut norm_scaler = 1. / weight_total;

    // Prevents NaNs and related complications.
    if weight_total.abs() < 0.000001 {
        norm_scaler = 0.;
    }

    for i in 0..grid_n {
        psi.on_pt[i] = Cplx::new_zero();
        psi.psi_pp_analytic[i] = Cplx::new_zero();
        psi.x_prev[i] = Cplx::new_zero();
        psi.x_next[i] = Cplx::new_zero();
        psi.y_prev[i] = Cplx::new_zero();
        psi.y_next[i] = Cplx::new_zero();
        psi.z_prev[i] = Cplx::new_zero();
        psi.z_next[i] = Cplx::new_zero();

        for (i_basis, weight) in weights.iter().enumerate() {
            let scaled = weight * norm_scaler;

            psi.on_pt[i] += bases_evaled.on_pt[i_basis][i] * scaled;
            psi.psi_pp_analytic[i] += bases_evaled.psi_pp_analytic[i_basis][i] * scaled;
            psi.x_prev[i] += bases_evaled.x_prev[i_basis][i] * scaled;
            psi.x_next[i] += bases_evaled.x_next[i_basis][i] * scaled;
            psi.y_prev[i] += bases_evaled.y_prev[i_basis][i] * scaled;
            psi.y_next[i] += bases_evaled.y_next[i_basis][i] * scaled;
            psi.z_prev[i] += bases_evaled.z_prev[i_basis][i] * scaled;
            psi.z_next[i] += bases_evaled.z_next[i_basis][i] * scaled;
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
    sfcs: &mut SurfacesPerElec,
    basis_wfs: &BasesEvaluated,
    E: f64,
    grid_n: usize,
    weights: &[f64],
) {
    mix_bases(&mut sfcs.psi, basis_wfs, grid_n, weights);

    // sfcs.E = find_E(sfcs, grid_n);
    // sfcs.E = 0.; // todo

    // Update psi_pps after normalization. We can't rely on cached wfs here, since we need to
    // take infinitessimal differences on the analytic basis equations to find psi'' measured.
    update_psi_pps(
        &sfcs.psi,
        &sfcs.V_acting_on_this,
        &mut sfcs.psi_pp_calculated,
        &mut sfcs.psi_pp_measured,
        E,
        grid_n,
    );
}

pub fn update_wf_fm_bases_1d(
    eval_data: &mut EvalDataPerElec,
    basis_wfs: &BasesEvaluated1d,
    grid_n: usize,
    weights: &[f64],
    E: Option<f64>,
) {
    mix_bases_1d(&mut eval_data.psi, basis_wfs, grid_n, weights);

    eval_data.E = match E {
        Some(E_) => E_,
        None => find_E(eval_data, grid_n),
    };

    // Update psi_pps after normalization. We can't rely on cached wfs here, since we need to
    // take infinitessimal differences on the analytic basis equations to find psi'' measured.
    update_psi_pps_1d(
        &eval_data.psi,
        &eval_data.V_acting_on_this,
        &mut eval_data.psi_pp_calc,
        &mut eval_data.psi_pp_meas,
        eval_data.E,
        grid_n,
    );
}

// /// Run this after update E.
// pub fn _update_psi_pp_calc(
//     // We split these arguments up instead of using surfaces to control mutability.
//     psi: &Arr3d,
//     V: &Arr3dReal,
//     psi_pp_calc: &mut Arr3d,
//     E: f64,
//     grid_n: usize,
// ) {
//     for i in 0..grid_n {
//         for j in 0..grid_n {
//             for k in 0..grid_n {
//                 psi_pp_calc[i][j][k] = eigen_fns::find_ψ_pp_calc(psi[i][j][k], V[i][j][k], E);
//             }
//         }
//     }
// }

/// Update psi'' calc and psi'' measured, assuming we are using basis WFs. This is done
/// after the wave function is contructed and normalized, including at neighboring points.
///
/// We use a separate function from this since it's used separately in our basis-finding
/// algorithm
pub fn update_psi_pps(
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
                psi_pp_calc[i][j][k] =
                    eigen_fns::find_ψ_pp_calc(psi.on_pt[i][j][k], V[i][j][k], E);

                // Calculate psi'' based on a numerical derivative of psi
                // in 3D.
                // We can compute ψ'' measured this in the same loop here, since we're using an analytic
                // equation for ψ; we can diff at arbitrary points vice only along a grid of pre-computed ψ.
                psi_pp_meas[i][j][k] = num_diff::find_ψ_pp_meas(
                    psi.on_pt[i][j][k],
                    psi.x_prev[i][j][k],
                    psi.x_next[i][j][k],
                    psi.y_prev[i][j][k],
                    psi.y_next[i][j][k],
                    psi.z_prev[i][j][k],
                    psi.z_next[i][j][k],
                );

                // todo experimenting
                // psi_pp_meas[i][j][k] = psi.psi_pp_analytic[i][j][k];
            }
        }
    }
}

pub fn update_psi_pps_1d(
    // We split these arguments up instead of using surfaces to control mutability.
    psi: &PsiWDiffs1d,
    V: &[f64],
    psi_pp_calc: &mut [Cplx],
    psi_pp_meas: &mut [Cplx],
    E: f64,
    grid_n: usize,
) {
    for i in 0..grid_n {
        psi_pp_calc[i] = eigen_fns::find_ψ_pp_calc(psi.on_pt[i], V[i], E);

        // Calculate psi'' based on a numerical derivative of psi
        // in 3D.
        // We can compute ψ'' measured this in the same loop here, since we're using an analytic
        // equation for ψ; we can diff at arbitrary points vice only along a grid of pre-computed ψ.
        psi_pp_meas[i] = num_diff::find_ψ_pp_meas(
            psi.on_pt[i],
            psi.x_prev[i],
            psi.x_next[i],
            psi.y_prev[i],
            psi.y_next[i],
            psi.z_prev[i],
            psi.z_next[i],
        );

        // todo experimenting
        // psi_pp_meas[i] = psi.psi_pp_analytic[i];
    }
}


/// Find the E that minimizes score, by narrowing it down. Note that if the relationship
/// between E and psi'' score isn't straightforward, this will converge on a local minimum.
/// Note: The only part of `eval_data` we mutate is psi'' calc.
pub fn find_E(data: &mut EvalDataPerElec, grid_n: usize) -> f64 {
    // todo: WHere to configure these mins and maxes
    let mut result = 0.;

    let mut E_min = -4.;
    let mut E_max = 4.;
    let mut E_range_div2 = 2.;
    let vals_per_iter = 8;

    let num_iters = 10;

    for _ in 0..num_iters {
        let E_vals = util::linspace((E_min, E_max), vals_per_iter);
        let mut best_score = 100_000_000.;
        let mut best_E = 0.;

        for E_trial in E_vals {
            for i in 0..grid_n {
                data.psi_pp_calc[i] = eigen_fns::find_ψ_pp_calc(
                    data.psi.on_pt[i],
                    data.V_acting_on_this[i],
                    E_trial,
                );
            }

            let score = eval::score_wf(&data.psi_pp_calc, &data.psi_pp_meas);

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

/// [re]Create a set of basis functions, given fixed-charges representing nuclei.
/// Use this in main and lib inits, and when you add or remove charges.
pub fn initialize_bases(
    charges_fixed: &[(Vec3, f64)],
    bases: &mut Vec<Basis>,
    bases_visible: Option<&mut Vec<bool>>,
    max_n: u16, // quantum number n
) {
    // let mut prev_weights = Vec::new();
    // for basis in bases.iter() {
    //     prev_weights.push(basis.weight());
    // }

    *bases = Vec::new();
    println!("Initializing bases");

    let mut visible = Vec::new();

    // todo: We currently call this in some cases where it maybe isn't strictly necessarly;
    // todo for now as a kludge to preserve weights, we copy the prev weights.
    for (charge_id, (nuc_posit, _)) in charges_fixed.iter().enumerate() {
        // See Sebens, for weights under equation 24; this is for Helium.
        // bases.push(Basis::Sto(Sto {
        //     posit: *nuc_posit,
        //     n: 1,
        //     xi: 1.41714,
        //     weight: 0.76837,
        //     charge_id,
        //     harmonic: Default::default(),
        // }));
        // bases.push(Basis::Sto(Sto {
        //     posit: *nuc_posit,
        //     n: 1,
        //     xi: 2.37682,
        //     weight: 0.22346,
        //     charge_id,
        //     harmonic: Default::default(),
        // }));
        // bases.push(Basis::Sto(Sto {
        //     posit: *nuc_posit,
        //     n: 1,
        //     xi: 4.39628,
        //     weight: 0.04082,
        //     charge_id,
        //     harmonic: Default::default(),
        // }));
        // bases.push(Basis::Sto(Sto {
        //     posit: *nuc_posit,
        //     n: 1,
        //     xi: 6.52699,
        //     weight: -0.00994,
        //     charge_id,
        //     harmonic: Default::default(),
        // }));
        // bases.push(Basis::Sto(Sto {
        //     posit: *nuc_posit,
        //     n: 1,
        //     xi: 7.94252,
        //     weight: 0.00230,
        //     charge_id,
        //     harmonic: Default::default(),
        // }));

        for _ in 0..bases.len() {
            visible.push(true);
        }
    }
    // if let Some(mut vis) = bases_visible {
    //     *vis = visible;
    // }
    // return; // todo temp

    for n in 1..max_n + 1 {
        for l in 0..n {
            for m in -(l as i16)..l as i16 + 1 {
                // This loop order allows the basis sliders to be sorted with like-electrons next to each other.
                for (charge_id, (nuc_posit, _)) in charges_fixed.iter().enumerate() {
                    let weight = if n == 1 { 1. } else { 0. };

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

                    //    pub posit: Vec3,
                    //     pub n: u16,
                    //     pub xi: f64,
                    //     pub weight: f64,
                    //     pub charge_id: usize,
                    //     pub harmonic: SphericalHarmonic,

                    // for xi in &[1., 2., 3., 4.] {
                    //     for xi in &[1.41714, 2.37682, 4.39628, 6.52699, 7.94252] {
                    //         bases.push(Basis::Sto(Sto {
                    //             posit: *nuc_posit,
                    //             n,
                    //             xi: *xi,
                    //             harmonic: SphericalHarmonic {
                    //                 l,
                    //                 m,
                    //                 orientation: Quaternion::new_identity(),
                    //             },
                    //             weight,
                    //             charge_id,
                    //         }));
                    //     }
                }

                visible.push(true);
            }
        }
    }

    // if let Some(mut vis) = bases_visible {
    //     *vis = visible;
    // }
}

/// Group that includes psi at a point, and at points surrounding it, an infinetesimal difference
/// in both directions along each spacial axis.
#[derive(Clone)]
pub struct PsiWDiffs {
    pub on_pt: Arr3d,
    pub psi_pp_analytic: Arr3d,
    pub x_prev: Arr3d,
    pub x_next: Arr3d,
    pub y_prev: Arr3d,
    pub y_next: Arr3d,
    pub z_prev: Arr3d,
    pub z_next: Arr3d,
}

impl PsiWDiffs {
    pub fn init(data: &Arr3d) -> Self {
        Self {
            on_pt: data.clone(),
            psi_pp_analytic: data.clone(),
            x_prev: data.clone(),
            x_next: data.clone(),
            y_prev: data.clone(),
            y_next: data.clone(),
            z_prev: data.clone(),
            z_next: data.clone(),
        }
    }
}

/// Group that includes psi at a point, and at points surrounding it, an infinetesimal difference
/// in both directions along each spacial axis.
#[derive(Clone)]
pub struct PsiWDiffs1d {
    pub on_pt: Vec<Cplx>,
    pub psi_pp_analytic: Vec<Cplx>,
    pub x_prev: Vec<Cplx>,
    pub x_next: Vec<Cplx>,
    pub y_prev: Vec<Cplx>,
    pub y_next: Vec<Cplx>,
    pub z_prev: Vec<Cplx>,
    pub z_next: Vec<Cplx>,
}

impl PsiWDiffs1d {
    pub fn init(data: &Vec<Cplx>) -> Self {
        Self {
            on_pt: data.clone(),
            psi_pp_analytic: data.clone(),
            x_prev: data.clone(),
            x_next: data.clone(),
            y_prev: data.clone(),
            y_next: data.clone(),
            z_prev: data.clone(),
            z_next: data.clone(),
        }
    }
}

/// We use this to store numerical wave functions for each basis, both at sample points, and
/// a small amount along each axix, for calculating partial derivatives of psi''.
/// The `Vec` index corresponds to basis index.
#[derive(Clone)]
pub struct BasesEvaluated {
    pub on_pt: Vec<Arr3d>,
    pub x_prev: Vec<Arr3d>,
    pub x_next: Vec<Arr3d>,
    pub y_prev: Vec<Arr3d>,
    pub y_next: Vec<Arr3d>,
    pub z_prev: Vec<Arr3d>,
    pub z_next: Vec<Arr3d>,
    pub psi_pp_analytic: Vec<Arr3d>, // Sneaking this approach in!
}

impl BasesEvaluated {
    /// Create unweighted basis wave functions. Run this whenever we add or remove basis fns,
    /// and when changing the grid. This evaluates the analytic basis functions at
    /// each grid point. Each basis will be normalized in this function.
    /// Relatively computationally intensive.
    pub fn new(bases: &[Basis], grid_posits: &Arr3dVec, grid_n: usize) -> Self {
        let mut on_pt = Vec::new();
        let mut psi_pp_analytic = Vec::new();
        let mut x_prev = Vec::new();
        let mut x_next = Vec::new();
        let mut y_prev = Vec::new();
        let mut y_next = Vec::new();
        let mut z_prev = Vec::new();
        let mut z_next = Vec::new();

        for _ in 0..bases.len() {
            on_pt.push(new_data(grid_n));
            psi_pp_analytic.push(new_data(grid_n));
            x_prev.push(new_data(grid_n));
            x_next.push(new_data(grid_n));
            y_prev.push(new_data(grid_n));
            y_next.push(new_data(grid_n));
            z_prev.push(new_data(grid_n));
            z_next.push(new_data(grid_n));
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

                        psi_pp_analytic[basis_i][i][j][k] = basis.second_deriv(posit_sample);

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
            psi_pp_analytic,
        }
    }
}

/// We use this to store numerical wave functions for each basis, both at sample points, and
/// a small amount along each axix, for calculating partial derivatives of psi''.
/// The outer `Vec` index corresponds to basis index. Inner corresponds to position.
#[derive(Clone)]
pub struct BasesEvaluated1d {
    pub on_pt: Vec<Vec<Cplx>>,
    pub x_prev: Vec<Vec<Cplx>>,
    pub x_next: Vec<Vec<Cplx>>,
    pub y_prev: Vec<Vec<Cplx>>,
    pub y_next: Vec<Vec<Cplx>>,
    pub z_prev: Vec<Vec<Cplx>>,
    pub z_next: Vec<Vec<Cplx>>,
    pub psi_pp_analytic: Vec<Vec<Cplx>>, // Sneaking this approach in!
}

impl BasesEvaluated1d {
    /// Create unweighted basis wave functions. Run this whenever we add or remove basis fns,
    /// and when changing the grid. This evaluates the analytic basis functions at
    /// each grid point. Each basis will be normalized in this function.
    /// Relatively computationally intensive.
    ///
    /// `norm` should be calculated either numerically from a full grid, or analytically
    /// from the basis functions.
    pub fn new(bases: &[Basis], grid_posits: &[Vec3], norm: f64) -> Self {
        let mut on_pt = Vec::new();
        let mut x_prev = Vec::new();
        let mut x_next = Vec::new();
        let mut y_prev = Vec::new();
        let mut y_next = Vec::new();
        let mut z_prev = Vec::new();
        let mut z_next = Vec::new();
        let mut psi_pp_analytic = Vec::new();

        for _ in 0..bases.len() {
            on_pt.push(vec![Cplx::new_zero(); grid_posits.len()]);
            psi_pp_analytic.push(vec![Cplx::new_zero(); grid_posits.len()]);
            x_prev.push(vec![Cplx::new_zero(); grid_posits.len()]);
            x_next.push(vec![Cplx::new_zero(); grid_posits.len()]);
            y_prev.push(vec![Cplx::new_zero(); grid_posits.len()]);
            y_next.push(vec![Cplx::new_zero(); grid_posits.len()]);
            z_prev.push(vec![Cplx::new_zero(); grid_posits.len()]);
            z_next.push(vec![Cplx::new_zero(); grid_posits.len()]);
        }

        for (basis_i, basis) in bases.iter().enumerate() {
            for i in 0..grid_posits.len() {
                let posit_sample = grid_posits[i];

                let posit_x_prev = Vec3::new(posit_sample.x - H, posit_sample.y, posit_sample.z);
                let posit_x_next = Vec3::new(posit_sample.x + H, posit_sample.y, posit_sample.z);
                let posit_y_prev = Vec3::new(posit_sample.x, posit_sample.y - H, posit_sample.z);
                let posit_y_next = Vec3::new(posit_sample.x, posit_sample.y + H, posit_sample.z);
                let posit_z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - H);
                let posit_z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + H);

                let val_pt = basis.value(posit_sample);

                let val_x_prev = basis.value(posit_x_prev);
                let val_x_next = basis.value(posit_x_next);
                let val_y_prev = basis.value(posit_y_prev);
                let val_y_next = basis.value(posit_y_next);
                let val_z_prev = basis.value(posit_z_prev);
                let val_z_next = basis.value(posit_z_next);

                on_pt[basis_i][i] = val_pt / norm;
                x_prev[basis_i][i] = val_x_prev / norm;
                x_next[basis_i][i] = val_x_next / norm;
                y_prev[basis_i][i] = val_y_prev / norm;
                y_next[basis_i][i] = val_y_next / norm;
                z_prev[basis_i][i] = val_z_prev / norm;
                z_next[basis_i][i] = val_z_next / norm;

                psi_pp_analytic[basis_i][i] = basis.second_deriv(posit_sample);
            }
        }

        Self {
            on_pt,
            x_prev,
            x_next,
            y_prev,
            y_next,
            z_prev,
            z_next,
            psi_pp_analytic,
        }
    }
}

/// Similar to `BasisWfs::new` but without the diffs.
pub fn arr_from_bases(bases: &[Basis], grid_posits: &Arr3dVec, grid_n: usize) -> Vec<Arr3d> {
    let mut result = Vec::new();

    for _ in 0..bases.len() {
        result.push(new_data(grid_n));
    }

    for (basis_i, basis) in bases.iter().enumerate() {
        let mut norm = 0.;

        for i in 0..grid_n {
            for j in 0..grid_n {
                for k in 0..grid_n {
                    let posit_sample = grid_posits[i][j][k];

                    let val = basis.value(posit_sample);

                    result[basis_i][i][j][k] = val;
                    norm += val.abs_sq();
                }
            }
        }

        util::normalize_wf(&mut result[basis_i], norm, grid_n);
    }

    result
}

fn analytically_integrate_psi_sq(bases: &[Basis]) {
    // Try Sympy. WolFram ALpha is causing standard computation time to be exceeded, even for a single
    // integra.
}

/// Calulate the electron potential that must be acting on a given enectron, given its
/// wave function. This is the potential from *other* electroncs in the system. This is, perhaps
/// something that could be considered DFT. (?)
///
/// Note: psi'' here is associated with psi'' measured, vice calculated.
pub fn calculate_v_elec(V_elec: &mut Arr3dReal, V_total: &mut Arr3dReal, psi: &Arr3d, psi_pp: &Arr3d, E: f64, V_nuc: &Arr3dReal, grid_n: usize) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                // todo: Hermitian operator has real eigenvalues. How are we able to discard
                // todo the imaginary parts here? Is psi'' / psi always real?

                V_total[i][j][k] = eigen_fns::calc_V_on_psi(psi[i][j][k], psi_pp[i][j][k], E);

                // todo: This appears to match with our displayed V, but I don't understand why. Electron sign?
                V_total[i][j][k] = -V_total[i][j][k];
                V_elec[i][j][k] = V_total[i][j][k] - V_nuc[i][j][k];
            }
        }
    }
    println!("V (total) from psi at edge: {}", V_total[0][0][0]);

    // todo: Aux 3: Try a numerical derivative of potential; examine for smoothness.

    // Note: By dividing by Q_C, we can get the sum of psi^2 from other elecs, I think?
    // Then a formula where (psi_0^2 + psi_1^2) * Q_C = the charge density, which is associated with the
    // V_elec we calculate here.

    // todo: How do we go from V_elec to charge density? That may be key.
}