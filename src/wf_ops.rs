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
    basis_wfs::{Basis, Sto},
    complex_nums::Cplx,
    eigen_fns::{self, KE_COEFF},
    gpu,
    grid_setup::{new_data, Arr3d, Arr3dReal, Arr3dVec},
    num_diff::{self},
    types::{BasesEvaluated, PsiWDiffs, PsiWDiffs1d, SurfacesPerElec},
    util,
};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

use crate::grid_setup::new_data_real;
use lin_alg2::f64::Vec3;

// We use Hartree units: ħ, elementary charge, electron mass, and Bohr radius.
pub const K_C: f64 = 1.;
pub const Q_PROT: f64 = 1.;
pub const Q_ELEC: f64 = -1.;
pub const M_ELEC: f64 = 1.;
pub const ħ: f64 = 1.;

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
/// This is our mixer from pre-calculated basis functions: Create psi, including at
/// neighboring points (used to numerically differentiate), from summing them with
/// their weights. Basis wfs must be initialized prior to running this, and weights must
/// be selected.
///
/// The resulting wave functions are normalized.
pub fn mix_bases_w_diffs(
    psi: &mut PsiWDiffs,
    bases_evaled: &BasesEvaluated,
    grid_n: usize,
    weights: &[f64],
) {
    // todo: This assumption may not be correct.
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

    // todo temp TS
    // let norm_scaler = 1.;

    let mut norm = 0.;

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

                norm += psi.on_pt[i][j][k].abs_sq();
            }
        }
    }

    util::normalize_wf(&mut psi.on_pt, norm);
    util::normalize_wf(&mut psi.psi_pp_analytic, norm);
}

/// Eg for the psi-on-grid for charges, where we don't need to generate psi''.
// todo: DRY
pub fn mix_bases(psi: &mut Arr3d, bases_evaled: &[Arr3d], grid_n: usize, weights: &[f64]) {
    // We don't need to normalize the result using the full procedure; the basis-wfs are already
    // normalized, so divide by the cumulative basis weights.
    let mut weight_total = 0.;
    for weight in weights {
        weight_total += weight.abs();
    }

    let mut norm_scaler = 1. / weight_total;

    // Prevents NaNs and related complications.
    if weight_total.abs() < 0.00000001 {
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

/// 2 in one, to remove an unecessarly loop.
pub fn mix_bases_update_charge_density(
    psi: &mut Arr3d,
    charge_density: &mut Arr3dReal,
    bases_evaled: &[Arr3d],
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
    if weight_total.abs() < 0.00000001 {
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
                charge_density[i][j][k] = psi[i][j][k].abs_sq() * Q_ELEC;
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
    sfcs: &mut SurfacesPerElec,
    basis_wfs: &BasesEvaluated,
    E: f64,
    grid_n: usize,
    weights: &[f64],
) {
    mix_bases_w_diffs(&mut sfcs.psi, basis_wfs, grid_n, weights);
    // mix_bases_no_diffs(&mut sfcs.psi.on_pt, &basis_wfs.on_pt, grid_n, weights);

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
                psi_pp_meas[i][j][k] = psi.psi_pp_analytic[i][j][k];
            }
        }
    }
}

/// [re]Create a set of basis functions, given fixed-charges representing nuclei.
/// Use this in main and lib inits, and when you add or remove charges.
pub fn initialize_bases(
    charges_fixed: &[(Vec3, f64)],
    bases: &mut Vec<Basis>,
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
    for (charge_id, (nuc_posit, _)) in charges_fixed.iter().enumerate() {
        // See Sebens, for weights under equation 24; this is for Helium.

        for (xi, weight) in [
            // (1.41714, 0.76837),
            // (2.37682, 0.22346),
            // (4.39628, 0.04082),
            // (6.52699, -0.00994),
            // (7.94252, 0.00230),
            (1., 1.),
            // (1.7, 0.),
            // (1.6, 0.),
            (2., 0.),
            (3., 0.),
            // (4., 0.),
            // (5., 0.),
            // (6., 0.),
            // (7., 0.),
            // (8., 0.),
            // (9., 0.),
        ] {
            for n in 1..3 {
                bases.push(Basis::Sto(Sto {
                    posit: *nuc_posit,
                    n: 1,
                    xi,
                    weight,
                    charge_id,
                    harmonic: Default::default(),
                }));
            }
        }
    }

    // for n in 1..max_n + 1 {
    //     for l in 0..n {
    //         for m in -(l as i16)..l as i16 + 1 {
    //             // This loop order allows the basis sliders to be sorted with like-electrons next to each other.
    //             for (charge_id, (nuc_posit, _)) in charges_fixed.iter().enumerate() {
    //                 let weight = if n == 1 { 1. } else { 0. };
    //
    //                 bases.push(Basis::H(HOrbital {
    //                     posit: *nuc_posit,
    //                     n,
    //                     harmonic: SphericalHarmonic {
    //                         l,
    //                         m,
    //                         orientation: Quaternion::new_identity(),
    //                     },
    //
    //                     weight,
    //                     charge_id,
    //                 }));
    //
    //                 //    pub posit: Vec3,
    //                 //     pub n: u16,
    //                 //     pub xi: f64,
    //                 //     pub weight: f64,
    //                 //     pub charge_id: usize,
    //                 //     pub harmonic: SphericalHarmonic,
    //
    //                 // for xi in &[1., 2., 3., 4.] {
    //                 //     for xi in &[1.41714, 2.37682, 4.39628, 6.52699, 7.94252] {
    //                 //         bases.push(Basis::Sto(Sto {
    //                 //             posit: *nuc_posit,
    //                 //             n,
    //                 //             xi: *xi,
    //                 //             harmonic: SphericalHarmonic {
    //                 //                 l,
    //                 //                 m,
    //                 //                 orientation: Quaternion::new_identity(),
    //                 //             },
    //                 //             weight,
    //                 //             charge_id,
    //                 //         }));
    //                 //     }
    //             }
    //
    //         }
    //     }
    // }
}

/// Create psi using basis functions. Only creates psi; doesn't create
/// its derivatives, and combines from all bases. We use this primarily as part of the central dogma
/// pipeline; ie for creating charge from the electron cloud.
/// todo: This currently keeps the bases unmixed. Do we want 2 variants: One mixed, one unmixed?
pub fn create_psi_from_bases(
    dev: &Arc<CudaDevice>,
    bases: &[Basis],
    grid_posits: &Arr3dVec,
    grid_n: usize,
) -> Vec<Arr3d> {
    let mut result = Vec::new();
    for _ in 0..bases.len() {
        result.push(new_data(grid_n));
    }

    let posits_flat = util::flatten_arr(grid_posits, grid_n);

    for (basis_i, basis) in bases.iter().enumerate() {
        let psi_flat = gpu::sto_vals(dev, basis.xi(), basis.n(), &posits_flat, basis.posit());

        let mut norm = 0.;

        // This is similar to util::unflatten, but with coercing to cplx.
        let grid_n_sq = grid_n.pow(2);

        for i in 0..grid_n {
            for j in 0..grid_n {
                for k in 0..grid_n {
                    let i_flat = i * grid_n_sq + j * grid_n + k;
                    result[basis_i][i][j][k] = Cplx::from_real(psi_flat[i_flat]);
                    norm += result[basis_i][i][j][k].abs_sq(); // todo: Handle norm on GPU?
                }
            }
        }

        util::normalize_wf(&mut result[basis_i], norm);
    }
    result
}

/// Another loop-saving combo.
pub fn create_psi_from_bases_mix_update_charge_density(
    dev: &Arc<CudaDevice>,
    charge_density: &mut Arr3dReal,
    bases: &[Basis],
    grid_posits: &Arr3dVec,
    grid_n: usize,
) -> Vec<Arr3d> {
    let mut result = Vec::new();
    for _ in 0..bases.len() {
        result.push(new_data(grid_n));
    }

    let posits_flat = util::flatten_arr(grid_posits, grid_n);

    for (basis_i, basis) in bases.iter().enumerate() {
        let psi_flat = gpu::sto_vals(dev, basis.xi(), basis.n(), &posits_flat, basis.posit());

        let mut norm = 0.;

        // This is similar to util::unflatten, but with coercing to cplx.
        let grid_n_sq = grid_n.pow(2);

        for i in 0..grid_n {
            for j in 0..grid_n {
                for k in 0..grid_n {
                    let i_flat = i * grid_n_sq + j * grid_n + k;

                    result[basis_i][i][j][k] = Cplx::from_real(psi_flat[i_flat]);
                    norm += result[basis_i][i][j][k].abs_sq(); // todo: Handle norm on GPU?

                    // todo: Figure out if this is feasible given the normalization
                    // charge_density[i][j][k] = psi[i][j][k].abs_sq() * Q_ELEC;
                }
            }
        }

        util::normalize_wf(&mut result[basis_i], norm);
    }
    result
}

/// Calulate the electron potential that must be acting on a given electron, given its
/// wave function. This is the potential from *other* electroncs in the system. This is, perhaps
/// something that could be considered DFT. (?)
///
/// Note: psi'' here is associated with psi'' measured, vice calculated.
pub fn calculate_v_elec(
    V_elec: &mut Arr3dReal,
    V_total: &mut Arr3dReal,
    psi: &Arr3d,
    psi_pp: &Arr3d,
    E: f64,
    V_nuc: &Arr3dReal,
) {
    let grid_n = psi.len();

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                // todo: Hermitian operator has real eigenvalues. How are we able to discard
                // todo the imaginary parts here? Is psi'' / psi always real?
                V_total[i][j][k] = eigen_fns::calc_V_on_psi(psi[i][j][k], psi_pp[i][j][k], E);
                V_elec[i][j][k] = V_total[i][j][k] - V_nuc[i][j][k];
            }
        }
    }

    // todo: Aux 3: Try a numerical derivative of potential; examine for smoothness.

    // Note: By dividing by Q_C, we can get the sum of psi^2 from other elecs, I think?
    // Then a formula where (psi_0^2 + psi_1^2) * Q_C = the charge density, which is associated with the
    // V_elec we calculate here.

    // todo: How do we go from V_elec to charge density? That may be key.
}

// todo: For finding E, you should consider varying it until V at the edges, analyticaally, is 0.

/// Calculate E from a trial wave function. We assume V goes to 0 at +/- ∞
/// note that this appears to approach the eneryg, but doens't hit it.
pub fn E_from_trial(bases: &[Basis], V_corner: f64, posit_corner: Vec3) -> f64 {
    // Important: eval_pt should be close to +- infinity, but doing so may cause numerical issues
    // as both psi and psi'' go to 0.
    // todo: Ideally with analytic bases, we use the analytic second derivative, but we're currently
    // todo having trouble with it.

    let mut psi = Cplx::new_zero();
    let mut psi_pp = Cplx::new_zero();
    // let mut psi_pp_div_psi = 0.;

    // todo: Pass in weights as an arg?
    let mut weights = Vec::new();
    for basis in bases {
        weights.push(basis.weight());
    }

    for (i, basis) in bases.iter().enumerate() {
        let weight = Cplx::from_real(weights[i]);
        // let weight = weights[i];

        // psi_pp_div_psi += weight * basis.psi_pp_div_psi(posit_corner);
        psi += weight * basis.value(posit_corner);
        psi_pp += weight * basis.second_deriv(posit_corner);
    }

    // todo: WIth this psi_pp_div_psi shortcut, you appear to be getting normalization issues.

    // todo: Why do we need to flip the sign?
    KE_COEFF * (psi_pp / psi).real - V_corner
}

/// Convert an array of ψ to one of electron charge, through space. This is used to calculate potential
/// from an electron. (And potential energy between electrons) Modifies in place
/// to avoid unecessary allocations.
/// `psi` must be normalized.
pub(crate) fn update_charge_density_fm_psi(
    charge_density: &mut Arr3dReal,
    psi_on_charge_grid: &Arr3d,
    grid_n_charge: usize,
) {
    // Note: We need to sum to 1 over *all space*, not just in the grid.
    // We can mitigate this by using a sufficiently large grid bounds, since the WF
    // goes to 0 at distance.

    // todo: YOu may need to model in terms of areas vice points; this is likely
    // todo a factor on irregular grids.

    for i in 0..grid_n_charge {
        for j in 0..grid_n_charge {
            for k in 0..grid_n_charge {
                charge_density[i][j][k] = psi_on_charge_grid[i][j][k].abs_sq() * Q_ELEC;
            }
        }
    }
}

/// Combine electron charges into a single array, not to include the electron acted on.
pub(crate) fn combine_electron_charges(
    elec_id: usize,
    charges_electron: &[Arr3dReal],
    grid_n_charge: usize,
) -> Arr3dReal {
    let mut result = new_data_real(grid_n_charge);

    for (i_charge, charge_from_elec) in charges_electron.iter().enumerate() {
        if i_charge == elec_id {
            continue;
        }
        let mut sum = 0.; // todo confirming
        for i in 0..grid_n_charge {
            for j in 0..grid_n_charge {
                for k in 0..grid_n_charge {
                    result[i][j][k] += charge_from_elec[i][j][k];
                    sum += charge_from_elec[i][j][k];
                }
            }
        }
        println!("Charge sum (should be -1): {:?}", sum);
    }

    result
}
