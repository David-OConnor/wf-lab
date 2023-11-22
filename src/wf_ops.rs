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

#[cfg(features = "cuda")]
use cudarc::driver::CudaDevice;
use lin_alg2::f64::Vec3;

use crate::{
    basis_wfs::{Basis, Sto},
    complex_nums::Cplx,
    eigen_fns::{self, KE_COEFF},
    grid_setup::{new_data_real, Arr3d, Arr3dReal, Arr3dVec},
    num_diff,
    types::ComputationDevice,
    util::{self, unflatten_arr},
};

#[cfg(features = "cuda")]
use crate::gpu;
use crate::util::EPS_DIV0;

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

pub fn mix_bases(
    psi: &mut Arr3d,
    mut psi_pp: Option<&mut Arr3d>,
    psi_per_basis: &[Arr3d],
    psi_pp_per_basis: Option<&[Arr3d]>,
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
    if weight_total.abs() < EPS_DIV0 {
        norm_scaler = 0.;
    }

    // let mut norm_scaler = 1.; // todo temp/TS.

    // todo: GPU option?
    let mut norm = 0.; // todo temp/TS

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                psi[i][j][k] = Cplx::new_zero();
                if let Some(pp) = psi_pp.as_mut() {
                    pp[i][j][k] = Cplx::new_zero();
                }

                for (i_basis, weight) in weights.iter().enumerate() {
                    let scaler = weight * norm_scaler;

                    psi[i][j][k] += psi_per_basis[i_basis][i][j][k] * scaler;
                    norm += psi[i][j][k].abs_sq(); // todo: Experimenting

                    if let Some(pp) = psi_pp.as_mut() {
                        pp[i][j][k] +=
                            psi_pp_per_basis.as_ref().unwrap()[i_basis][i][j][k] * scaler;
                    }
                }
            }
        }
    }

    // todo: Evaluate if you need/want this.
    // util::normalize_arr(psi, norm);
    // if psi_pp.is_some() {
    //     util::normalize_arr(psi_pp.as_mut().unwrap(), norm);
    // }
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
    if weight_total.abs() < EPS_DIV0 {
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
            for n in 1..max_n + 1 {
                bases.push(Basis::Sto(Sto {
                    posit: *nuc_posit,
                    n,
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

/// Create psi, and optionally psi'', using basis functions. Does not mix bases; creates these
/// values per-basis.
/// todo: This currently keeps the bases unmixed. Do we want 2 variants: One mixed, one unmixed?
pub fn update_wf_from_bases(
    dev: &ComputationDevice,
    psi: &mut [Arr3d],
    mut psi_pp: Option<&mut [Arr3d]>,
    bases: &[Basis],
    grid_posits: &Arr3dVec,
    grid_n: usize,
) {
    let mut norm = 0.;

    match dev {
        #[cfg(features = "cuda")]
        ComputationDevice::Gpu(cuda_dev) => {
            let posits_flat = util::flatten_arr(grid_posits, grid_n);

            // This basis loop is instead the dev selection so we don't calc posits_flat on CPU.
            // Unfortunately, it adds duplicated code.
            for (basis_i, basis) in bases.iter().enumerate() {
                let psi_flat = gpu::sto_vals_or_derivs(
                    cuda_dev,
                    basis.xi(),
                    basis.n(),
                    &posits_flat,
                    basis.posit(),
                    false,
                );

                // todo: Use the combined kernnel once you sort out out-of-resources.
                let psi_pp_flat = if let Some(_) = psi_pp {
                    Some(gpu::sto_vals_or_derivs(
                        cuda_dev,
                        basis.xi(),
                        basis.n(),
                        &posits_flat,
                        basis.posit(),
                        true,
                    ))
                } else {
                    None
                };

                let grid_n_sq = grid_n.pow(2);

                // This is similar to util::unflatten, but with norm involved.
                // todo: Try to use unflatten.
                for i in 0..grid_n {
                    for j in 0..grid_n {
                        for k in 0..grid_n {
                            let i_flat = i * grid_n_sq + j * grid_n + k;
                            psi[basis_i][i][j][k] = Cplx::from_real(psi_flat[i_flat]);

                            if let Some(pp) = psi_pp.as_mut() {
                                pp[basis_i][i][j][k] =
                                    Cplx::from_real(psi_pp_flat.as_ref().unwrap()[i_flat]);
                            }

                            norm += psi[basis_i][i][j][k].abs_sq(); // todo: Handle norm on GPU?
                        }
                    }
                }
                util::normalize_arr(&mut psi[basis_i], norm);
                if psi_pp.is_some() {
                    util::normalize_arr(&mut psi_pp.as_mut().unwrap()[basis_i], norm);
                }
            }
        }
        ComputationDevice::Cpu => {
            for (basis_i, basis) in bases.iter().enumerate() {
                for i in 0..grid_n {
                    for j in 0..grid_n {
                        for k in 0..grid_n {
                            let posit_sample = grid_posits[i][j][k];
                            psi[basis_i][i][j][k] = basis.value(posit_sample);

                            if let Some(ref mut pp) = psi_pp {
                                if basis.n() >= 2 {
                                    // todo: Once working, apply to GPU as well.
                                    pp[basis_i][i][j][k] = num_diff::find_ψ_pp_num_fm_bases(
                                        posit_sample,
                                        &[basis.clone()], // todo: Don't clone
                                        psi[basis_i][i][j][k],
                                    );
                                } else {
                                    pp[basis_i][i][j][k] = basis.second_deriv(posit_sample);
                                }
                            }

                            norm += psi[basis_i][i][j][k].abs_sq();
                        }
                    }
                }
                util::normalize_arr(&mut psi[basis_i], norm);
                if psi_pp.is_some() {
                    util::normalize_arr(&mut psi_pp.as_mut().unwrap()[basis_i], norm);
                }
            }
        }
    }
}

// todo: put back A/R

// /// Another loop-saving combo.
// pub fn create_psi_from_bases_mix_update_charge_density(
//     dev: ComputationDevice,
//     charge_density: &mut Arr3dReal,
//     bases: &[Basis],
//     grid_posits: &Arr3dVec,
//     grid_n: usize,
// ) -> Vec<Arr3d> {
//     let mut result = Vec::new();
//     for _ in 0..bases.len() {
//         result.push(new_data(grid_n));
//     }
//
//     let posits_flat = util::flatten_arr(grid_posits, grid_n);
//
//     for (basis_i, basis) in bases.iter().enumerate() {
//
//         let psi_flat = gpu::sto_vals_or_derivs(
//             dev,
//             basis.xi(),
//             basis.n(),
//             &posits_flat,
//             basis.posit(),
//             false,
//         );
//
//         let mut norm = 0.;
//
//         // This is similar to util::unflatten, but with coercing to cplx.
//         let grid_n_sq = grid_n.pow(2);
//
//         for i in 0..grid_n {
//             for j in 0..grid_n {
//                 for k in 0..grid_n {
//                     let i_flat = i * grid_n_sq + j * grid_n + k;
//
//                     result[basis_i][i][j][k] = Cplx::from_real(psi_flat[i_flat]);
//                     norm += result[basis_i][i][j][k].abs_sq(); // todo: Handle norm on GPU?
//
//                     // todo: Figure out if this is feasible given the normalization
//                     // charge_density[i][j][k] = psi[i][j][k].abs_sq() * Q_ELEC;
//                 }
//             }
//         }
//
//         util::normalize_arr(&mut result[basis_i], norm);
//     }
//     result
// }

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

/// Update V-from-psi, and psi''-calculated, based on psi, and V acting on a given electron.
pub fn update_eigen_vals(
    V_elec: &mut Arr3dReal,
    V_total: &mut Arr3dReal,
    psi_pp_calculated: &mut Arr3d,
    psi: &Arr3d,
    psi_pp: &Arr3d,
    V_acting_on_this: &Arr3dReal,
    E: f64,
    V_nuc: &Arr3dReal,
) {
    let grid_n = psi.len();

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                V_total[i][j][k] = eigen_fns::calc_V_on_psi(psi[i][j][k], psi_pp[i][j][k], E);
                V_elec[i][j][k] = V_total[i][j][k] - V_nuc[i][j][k];

                psi_pp_calculated[i][j][k] =
                    eigen_fns::find_ψ_pp_calc(psi[i][j][k], V_acting_on_this[i][j][k], E)
            }
        }
    }

    // todo: Aux 3: Try a numerical derivative of potential; examine for smoothness.

    // Note: By dividing by Q_C, we can get the sum of psi^2 from other elecs, I think?
    // Then a formula where (psi_0^2 + psi_1^2) * Q_C = the charge density, which is associated with the
    // V_elec we calculate here.

    // todo: How do we go from V_elec to charge density? That may be key.
}

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

/// Get STO values, and second-derivative values, using the CPU.
pub(crate) fn sto_vals_derivs_cpu(
    psi: &mut Arr3d,
    psi_pp: &mut Arr3d,
    grid_posits: &Arr3dVec,
    basis: &Basis,
    grid_n: usize,
) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                let posit_sample = grid_posits[i][j][k];

                psi[i][j][k] = basis.value(posit_sample);
                psi_pp[i][j][k] = basis.second_deriv(posit_sample);
            }
        }
    }
}
