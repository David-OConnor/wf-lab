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

use lin_alg::f64::Vec3;

#[cfg(feature = "cuda")]
use crate::gpu;
use crate::{
    basis_wfs::Basis,
    complex_nums::Cplx,
    core_calcs::{
        eigen_fns::{self},
        eigen_raw,
    },
    dirac,
    dirac::{BasisSpinor, CompPsi, Spinor3, SpinorDerivsTypeD3, SpinorDerivsTypeE3},
    grid_setup::{new_data_real, Arr2d, Arr2dReal, Arr2dVec, Arr3d, Arr3dReal, Arr3dVec},
    iter_arr, iter_arr_2d, num_diff,
    types::{ComputationDevice, Derivatives, Derivatives2D, DerivativesSingle},
    util::{self, MAX_PSI_FOR_NORM},
};

// We use Hartree units: ħ, elementary charge, electron mass, and Bohr radius.
pub const K_C: f64 = 1.;
pub const Q_PROT: f64 = 1.;
pub const Q_ELEC: f64 = -1.;
pub const M_ELEC: f64 = 1.;
pub const ħ: f64 = 1.;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spin {
    Alpha,
    Beta,
}

impl Spin {
    pub fn to_string(&self) -> String {
        match self {
            Self::Alpha => "α",
            Self::Beta => "β",
        }
        .to_string()
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum DerivCalc {
    Analytic,
    Numeric,
}

impl Default for DerivCalc {
    fn default() -> Self {
        Self::Numeric
    }
}

/// Create psi, and optionally derivatives using basis functions. Creates one psi per basis. Does not mix bases;
/// creates these values per-basis.
pub fn wf_from_bases(
    dev: &ComputationDevice,
    psi_per_basis: &mut [Arr2d],
    derivs_per_basis: &mut [Derivatives2D], // None for charge calcs.
    bases: &[Basis],
    grid_posits: &Arr2dVec,
    deriv_calc: DerivCalc,
) {
    println!("Starting WF from bases...");
    let grid_n = grid_posits.len();

    // Setting up posits_flat here prevents repetition between CUDA and CPU code below.
    #[cfg(feature = "cuda")]
    let posits_flat = match dev {
        ComputationDevice::Gpu(_) => Some(util::flatten_arr_2d(grid_posits, grid_n)),
        ComputationDevice::Cpu => None,
    };

    for (basis_i, basis) in bases.iter().enumerate() {
        // todo: Temp forcing CPU only while we confirm numerical stability issues with f32
        // todo on GPU aren't causing a problem.

        match dev {
            #[cfg(feature = "cuda")]
            ComputationDevice::Gpu(cuda_dev) => {
                let (psi_flat, psi_pp_flat) =
                    // Calculate both using the same kernel.
                    gpu::sto_vals_derivs(
                        cuda_dev,
                        basis.xi(),
                        basis.n(),
                        &posits_flat.as_ref().unwrap(),
                        basis.posit(),
                    );

                let grid_n_sq = grid_n.pow(2);

                // This is similar to util::unflatten, but with norm involved.
                for (i, j) in iter_arr_2d!(grid_n) {
                    let i_flat = i * grid_n_sq + j;
                    psi_per_basis[basis_i][i][j] = Cplx::from_real(psi_flat[i_flat]);

                    derivs_per_basis[basis_i].d2_sum[i][j] = Cplx::from_real(psi_pp_flat[i_flat]);
                }
            }
            ComputationDevice::Cpu => {
                for (i, j) in iter_arr_2d!(grid_n) {
                    let posit_sample = grid_posits[i][j];

                    psi_per_basis[basis_i][i][j] = basis.value(posit_sample);
                    let b = [basis.clone()];

                    let d =
                        calc_derivs_cpu(psi_per_basis[basis_i][i][j], &b, posit_sample, deriv_calc);

                    derivs_per_basis[basis_i].dx[i][j] = d.dx;
                    derivs_per_basis[basis_i].dy[i][j] = d.dy;
                    derivs_per_basis[basis_i].dz[i][j] = d.dz;
                    derivs_per_basis[basis_i].d2x[i][j] = d.d2x;
                    derivs_per_basis[basis_i].d2y[i][j] = d.d2y;
                    derivs_per_basis[basis_i].d2z[i][j] = d.d2z;
                    derivs_per_basis[basis_i].d2_sum[i][j] = d.d2_sum;

                    // todo: Impl your Derivatives construction from GPU as well, but we'll use CPU for calculating
                    // todo these for now.
                }
            }
        }
    }
    println!("WF from bases complete.");
}

/// todo: DRY
pub fn wf_from_bases_charge(
    dev: &ComputationDevice,
    psi_per_basis: &mut [Arr3d],
    bases: &[Basis],
    grid_posits: &Arr3dVec,
) {
    println!("Starting WF from bases (charge)...");
    let grid_n = grid_posits.len();

    // Setting up posits_flat here prevents repetition between CUDA and CPU code below.
    #[cfg(feature = "cuda")]
    let posits_flat = match dev {
        ComputationDevice::Gpu(_) => Some(util::flatten_arr(grid_posits, grid_n)),
        ComputationDevice::Cpu => None,
    };

    for (basis_i, basis) in bases.iter().enumerate() {
        // todo: Temp forcing CPU only while we confirm numerical stability issues with f32
        // todo on GPU aren't causing a problem.

        let mut norm = 0.;

        match dev {
            #[cfg(feature = "cuda")]
            ComputationDevice::Gpu(cuda_dev) => {
                let psi_flat = gpu::sto_vals_or_derivs(
                    cuda_dev,
                    basis.xi(),
                    basis.n(),
                    &posits_flat.as_ref().unwrap(),
                    basis.posit(),
                    false,
                );

                let grid_n_sq = grid_n.pow(2);

                // This is similar to util::unflatten, but with norm involved.
                for (i, j, k) in iter_arr!(grid_n) {
                    let i_flat = i * grid_n_sq + j * grid_n + k;
                    psi_per_basis[basis_i][i][j][k] = Cplx::from_real(psi_flat[i_flat]);

                    util::add_to_norm(&mut norm, psi_per_basis[basis_i][i][j][k]);
                }
            }
            ComputationDevice::Cpu => {
                for (i, j, k) in iter_arr!(grid_n) {
                    let posit_sample = grid_posits[i][j][k];

                    psi_per_basis[basis_i][i][j][k] = basis.value(posit_sample);
                    util::add_to_norm(&mut norm, psi_per_basis[basis_i][i][j][k]);
                }
            }
        }

        // This normalization makes balancing the bases more intuitive, but isn't strictly required
        // in the way normalizing the composite (squared) wave function is prior to generating charge.
        util::normalize_arr(&mut psi_per_basis[basis_i], norm);
    }
    println!("WF from bases complete.");
}

#[allow(unused)]
/// Create psi, and optionally derivatives using basis functions. Creates one psi per basis. Does not mix bases;
/// creates these values per-basis.
/// todo: This currently keeps the bases unmixed. Do we want 2 variants: One mixed, one unmixed?
pub fn wf_from_bases_spinor(
    dev: &ComputationDevice,
    psi_per_basis: &mut [Spinor3],
    mut derivs_per_basis: Option<&mut [SpinorDerivsTypeD3]>, // psi component, da, index
    bases: &[BasisSpinor],
    grid_posits: &Arr3dVec,
) {
    // todo: For performance reasons while not using it.
    return;
    let grid_n = grid_posits.len();

    // Setting up posits_flat here prevents repetition between CUDA and CPU code below.
    #[cfg(feature = "cuda")]
    let posits_flat = match dev {
        ComputationDevice::Gpu(_) => Some(util::flatten_arr(grid_posits, grid_n)),
        ComputationDevice::Cpu => None,
    };

    for (basis_i, basis) in bases.iter().enumerate() {
        let mut norm = [0.; 4];

        match dev {
            #[cfg(feature = "cuda")]
            ComputationDevice::Gpu(cuda_dev) => {
                unimplemented!()
            }
            ComputationDevice::Cpu => {
                for (i, j, k) in iter_arr!(grid_n) {
                    let posit_sample = grid_posits[i][j][k];

                    for comp in [CompPsi::C0, CompPsi::C1, CompPsi::C2, CompPsi::C3] {
                        psi_per_basis[basis_i].get_mut(comp)[i][j][k] =
                            // basis.get(comp).value(posit_sample);

                            // We don't use the basis-integrated weight for non-DiracWFs. Let's do it here
                            // to make manipulating the fns easier at first.
                            basis.get(comp).value(posit_sample) * basis.get(comp).weight;
                    }

                    // todo: A hack to zero out the middle wave functions
                    // psi_per_basis[basis_i].c1[i][j][k] = Cplx::new_zero();
                    // psi_per_basis[basis_i].c2[i][j][k] = Cplx::new_zero();
                    // psi_per_basis[basis_i].c3[i][j][k] = Cplx::new_zero();

                    let b = [basis.clone()];

                    if let Some(ref mut derivs) = derivs_per_basis {
                        // todo: We are copying some of our non-dirac awkward ordering mistakes...
                        let diffs = SpinorDerivsTypeE3::from_bases(posit_sample, &b);

                        // Re-arrange the order of data to fit out API.
                        for comp in [CompPsi::C0, CompPsi::C1, CompPsi::C2, CompPsi::C3] {
                            derivs[basis_i].get_mut(comp).dx[i][j][k] = diffs.get(comp).dx;
                            derivs[basis_i].get_mut(comp).dy[i][j][k] = diffs.get(comp).dy;
                            derivs[basis_i].get_mut(comp).dz[i][j][k] = diffs.get(comp).dz;
                        }
                    }

                    for (i, comp) in [CompPsi::C0, CompPsi::C1, CompPsi::C2, CompPsi::C3]
                        .into_iter()
                        .enumerate()
                    {
                        util::add_to_norm(&mut norm[i], psi_per_basis[basis_i].get(comp)[i][j][k]);
                    }
                }
            }
        }

        // todo: Temp removing normalization from spinors.

        // util::normalize_arr(&mut psi_per_basis[basis_i].c0, norm[0]);
        // todo: Temp removed norm on middle vals for psi=0 trial wf.
        // util::normalize_arr(&mut psi_per_basis[basis_i].c1, norm[1]);
        // util::normalize_arr(&mut psi_per_basis[basis_i].c2, norm[2]);
        // util::normalize_arr(&mut psi_per_basis[basis_i].c3, norm[3]);

        // if let Some(derivs_mut) = derivs_per_basis.as_mut() {
        //     for (i, comp) in [CompPsi::C0, CompPsi::C1, CompPsi::C2, CompPsi::C3]
        //         .into_iter()
        //         .enumerate()
        //     {
        //         util::normalize_arr(&mut derivs_mut[basis_i].get_mut(comp).dx, norm[i]);
        //         util::normalize_arr(&mut derivs_mut[basis_i].get_mut(comp).dy, norm[i]);
        //         util::normalize_arr(&mut derivs_mut[basis_i].get_mut(comp).dz, norm[i]);
        //     }
        // }
    }
}

/// Mix previously-evaluated basis into a single wave function, with optional psi''. We generally
/// use psi'' when evaluating sample positions, but not when evaluating psi to be fed into
/// electron charge.
pub fn mix_bases(
    psi: &mut Arr2d,
    charge_density: &mut Arr2dReal,
    derivs: &mut Derivatives2D, // Not required for charge generation.
    psi_per_basis: &[Arr2d],
    derivs_per_basis: &[Derivatives2D], // Not required for charge generation.
    weights: &[f64],
) {
    // todo: GPU option?
    let mut norm = 0.;
    let grid_n = psi.len();

    // todo: Experimenting with a norm approach for 2D arrays, assuming each basis is normalized.
    let mut weight_total = 0.;
    for w in weights {
        weight_total += w;
    }

    for (i, j) in iter_arr_2d!(grid_n) {
        psi[i][j] = Cplx::new_zero();
        // todo: This is avoidable by a reversed Derivatives struct.
        derivs.dx[i][j] = Cplx::new_zero();
        derivs.dy[i][j] = Cplx::new_zero();
        derivs.dz[i][j] = Cplx::new_zero();
        derivs.d2x[i][j] = Cplx::new_zero();
        derivs.d2y[i][j] = Cplx::new_zero();
        derivs.d2z[i][j] = Cplx::new_zero();
        derivs.d2_sum[i][j] = Cplx::new_zero();

        for (i_basis, weight) in weights.iter().enumerate() {
            let scaler = *weight;
            // let scaler = *weight / weight_total;

            psi[i][j] += psi_per_basis[i_basis][i][j] * scaler;

            // todo: This is avoidable by a reversed Derivatives struct.
            derivs.dx[i][j] += derivs_per_basis[i_basis].dx[i][j] * scaler;
            derivs.dy[i][j] += derivs_per_basis[i_basis].dy[i][j] * scaler;
            derivs.dz[i][j] += derivs_per_basis[i_basis].dz[i][j] * scaler;
            derivs.d2x[i][j] += derivs_per_basis[i_basis].d2x[i][j] * scaler;
            derivs.d2y[i][j] += derivs_per_basis[i_basis].d2y[i][j] * scaler;
            derivs.d2z[i][j] += derivs_per_basis[i_basis].d2z[i][j] * scaler;
            derivs.d2_sum[i][j] += derivs_per_basis[i_basis].d2_sum[i][j] * scaler;
        }

        // The nuclear option: You can use a LUT. Probably a function of n, l, and xi.
        // let abs_sq = psi[i][j].abs_sq();
        // if abs_sq < MAX_PSI_FOR_NORM {
        //     norm += abs_sq; // todo: Handle norm on GPU?
        // } else {
        //     println!("Exceeded norm thresh in mix: {:?}", abs_sq);
        // }
    }

    {
        // todo: Probalby won't work, as you need to normalize the wf squared.?
        for weight in weights {
            norm += weight.abs();
        }

        // todo: We can't normalize using a 2D grid alone. Experimenting.
        util::normalize_arr_2d(psi, norm);
        util::normalize_arr_2d(&mut derivs.dx, norm);
        util::normalize_arr_2d(&mut derivs.dy, norm);
        util::normalize_arr_2d(&mut derivs.dz, norm);

        util::normalize_arr_2d(&mut derivs.d2x, norm);
        util::normalize_arr_2d(&mut derivs.d2y, norm);
        util::normalize_arr_2d(&mut derivs.d2z, norm);

        util::normalize_arr_2d(&mut derivs.d2_sum, norm);
    }

    // todo: Is this the appropriate place for this?
    // Set up charge density (psi^2 * -1), for the purposes of rendering.
    for (i, j) in iter_arr_2d!(grid_n) {
        charge_density[i][j] = psi[i][j].abs_sq() * Q_ELEC;
    }
}

// todo: DRY, while we sort out 2D vs 3D eval. Maybe it's better this way anyhow...
// todo: Sort out DRY between this and above.
pub fn mix_bases_charge(
    psi: &mut Arr3d,
    charge_density: &mut Arr3dReal,
    psi_per_basis: &[Arr3d],
    weights: &[f64],
) {
    let mut norm = 0.;
    let grid_n = psi.len();

    for (i, j, k) in iter_arr!(grid_n) {
        psi[i][j][k] = Cplx::new_zero();
        for (i_basis, weight) in weights.iter().enumerate() {
            let scaler = *weight;

            psi[i][j][k] += psi_per_basis[i_basis][i][j][k] * scaler;
        }

        let abs_sq = psi[i][j][k].abs_sq();
        if abs_sq < MAX_PSI_FOR_NORM {
            norm += abs_sq; // todo: Handle norm on GPU?
        } else {
            println!("Exceeded norm thresh in mix: {:?}", abs_sq);
        }
    }

    // println!("3D norm: {:?}.", norm);

    util::normalize_arr(psi, norm);

    // Update charge density as well, from this electron's wave function.
    charge_from_psi(
        charge_density,
        psi,
        grid_n, // Render; not charge grid.
    )
}

#[allow(unused)]
pub fn mix_bases_spinor(
    psi: &mut Spinor3,
    mut charge_density: Option<&mut Arr3dReal>,
    mut derivs: Option<&mut SpinorDerivsTypeD3>, // Not required for charge generation.
    psi_per_basis: &[Spinor3],
    derivs_per_basis: Option<&[SpinorDerivsTypeD3]>, // Not required for charge generation.
    weights: &[f64],
    // todo: Experiment with this API A/R
) {
    // todo: For performance reasons while not using it.
    return;

    // println!("WEIGHTS: {:?}", weights);
    // todo: GPU option?
    let mut norm = [0.; 4];
    let grid_n = psi.c0.len();

    // todo: Like with spinor ops in the above fn, this is repetative/DRY.
    for (i, j, k) in iter_arr!(grid_n) {
        psi.c0[i][j][k] = Cplx::new_zero();
        psi.c1[i][j][k] = Cplx::new_zero();
        psi.c2[i][j][k] = Cplx::new_zero();
        psi.c3[i][j][k] = Cplx::new_zero();

        if let Some(d) = derivs.as_mut() {
            for comp in [CompPsi::C0, CompPsi::C1, CompPsi::C2, CompPsi::C3] {
                d.get_mut(comp).dx[i][j][k] = Cplx::new_zero();
                d.get_mut(comp).dy[i][j][k] = Cplx::new_zero();
                d.get_mut(comp).dz[i][j][k] = Cplx::new_zero();
            }
        }

        for (i_basis, weight) in weights.iter().enumerate() {
            let scaler = *weight;

            for comp in [CompPsi::C0, CompPsi::C1, CompPsi::C2, CompPsi::C3] {
                psi.get_mut(comp)[i][j][k] += psi_per_basis[i_basis].get(comp)[i][j][k] * scaler;
            }

            if let Some(d) = derivs.as_mut() {
                let per_basis = derivs_per_basis.as_ref().unwrap();

                for comp in [CompPsi::C0, CompPsi::C1, CompPsi::C2, CompPsi::C3] {
                    d.get_mut(comp).dx[i][j][k] +=
                        per_basis[i_basis].get(comp).dx[i][j][k] * scaler;
                    d.get_mut(comp).dy[i][j][k] +=
                        per_basis[i_basis].get(comp).dy[i][j][k] * scaler;
                    d.get_mut(comp).dz[i][j][k] +=
                        per_basis[i_basis].get(comp).dz[i][j][k] * scaler;
                }
            }
        }

        // todo: Four separate abs_sqs like this?
        let abs_sq = [
            psi.c0[i][j][k].abs_sq(),
            psi.c1[i][j][k].abs_sq(),
            psi.c2[i][j][k].abs_sq(),
            psi.c3[i][j][k].abs_sq(),
        ];

        for (i, val) in abs_sq.iter().enumerate() {
            if val < &MAX_PSI_FOR_NORM {
                norm[i] += val; // todo: Handle norm on GPU?
            } else {
                println!("Exceeded norm thresh in mix: {:?}", val);
            }
        }
    }

    // todo: Temp removed norm from spinors.
    // for (i, comp) in [&mut psi.c0, &mut psi.c1, &mut psi.c2, &mut psi.c3]
    //     .into_iter()
    //     .enumerate()
    // {
    //     util::normalize_arr(comp, norm[i]);
    // }
    //
    // if let Some(derivs_mut) = derivs.as_mut() {
    //     for (i, comp) in [CompPsi::C0, CompPsi::C1, CompPsi::C2, CompPsi::C3]
    //         .into_iter()
    //         .enumerate()
    //     {
    //         util::normalize_arr(&mut derivs_mut.get_mut(comp).dx, norm[i]);
    //         util::normalize_arr(&mut derivs_mut.get_mut(comp).dy, norm[i]);
    //         util::normalize_arr(&mut derivs_mut.get_mut(comp).dz, norm[i]);
    //     }
    // }

    // Update charge density as well, from this electron's wave function.

    // todo
    // if charge_density.is_some() {
    //     charge_from_psi(
    //         charge_density.as_mut().unwrap(),
    //         psi,
    //         grid_n, // Render; not charge grid.
    //     )
    // }
}

/// Convert an array of ψ to one of electron charge, through space. This is used to calculate potential
/// from an electron. (And potential energy between electrons) Modifies in place
/// to avoid unecessary allocations.
/// `psi` must be normalized prior to being passed to this.
pub(crate) fn charge_from_psi(
    charge: &mut Arr3dReal,
    psi_on_charge_grid: &Arr3d,
    grid_n_charge: usize,
) {
    // Note: We need to sum to 1 over *all space*, not just in the grid.
    // We can mitigate this by using a sufficiently large grid bounds, since the WF
    // goes to 0 at distance.

    // todo: YOu may need to model in terms of areas vice points; this is likely
    // todo a factor on irregular grids.

    for (i, j, k) in iter_arr!(grid_n_charge) {
        charge[i][j][k] = psi_on_charge_grid[i][j][k].abs_sq() * Q_ELEC;
    }
}

/// Update V-from-psi, and psi''-calculated, based on psi, and V acting on a given electron.
/// todo: When do we use this, vice `calcualte_v_elec`?
pub fn update_eigen_vals(
    V_elec: &mut Arr2dReal,
    V_total: &mut Arr2dReal,
    V_diff: &mut Arr2dReal,
    psi_pp_calculated: &mut Arr2d,
    psi: &Arr2d,
    derivs: &Derivatives2D,
    V_acting_on_this: &Arr2dReal,
    E: f64,
    V_nuc: &Arr2dReal,
    // todo: New with angular p eigenvals
    grid_posits: &Arr2dVec,
    H: &mut Arr2d,
    L_sq: &mut Arr2d,
    L_z: &mut Arr2d,
) {
    let grid_n = psi.len();

    // for (i, j, k) in iter_arr!(grid_n) {
    //     V_total[i][j][k] = eigen_fns::calc_V_on_psi(psi[i][j][k], derivs.d2_sum[i][j][k], E);
    //     V_elec[i][j][k] = V_total[i][j][k] - V_nuc[i][j][k];
    //
    //     // todo: Another case where a reversed Derivs API would help.
    //     let derivs_single = DerivativesSingle {
    //         dx: derivs.dx[i][j][k],
    //         dy: derivs.dy[i][j][k],
    //         dz: derivs.dz[i][j][k],
    //         d2x: derivs.d2x[i][j][k],
    //         d2y: derivs.d2y[i][j][k],
    //         d2z: derivs.d2z[i][j][k],
    //         d2_sum: derivs.d2_sum[i][j][k],
    //     };
    //
    //     let temp = derivs_single.d2x + derivs_single.d2y + derivs_single.d2z;
    //     // H[i][j][k] = eigen_fns::calc_H(psi[i][j][k], derivs_single.d2_sum,V_acting_on_this[i][j][k]);
    //     H[i][j][k] = eigen_raw::calc_H(psi[i][j][k], temp, V_acting_on_this[i][j][k]);
    //
    //     L_sq[i][j][k] = eigen_raw::calc_L_sq(grid_posits[i][j][k], &derivs_single);
    //     L_z[i][j][k] = eigen_raw::calc_L_z(grid_posits[i][j][k], &derivs_single);
    //
    //     psi_pp_calculated[i][j][k] =
    //         eigen_fns::find_ψ_pp_calc(psi[i][j], V_acting_on_this[i][j], E)
    // }

    for (i, j) in iter_arr_2d!(grid_n) {
        V_total[i][j] = eigen_fns::calc_V_on_psi(psi[i][j], derivs.d2_sum[i][j], E);
        V_elec[i][j] = V_total[i][j] - V_nuc[i][j];

        // todo: Experimenting
        V_diff[i][j] = V_acting_on_this[i][j] - V_total[i][j];

        // todo: Another case where a reversed Derivs API would help.
        let derivs_single = DerivativesSingle {
            dx: derivs.dx[i][j],
            dy: derivs.dy[i][j],
            dz: derivs.dz[i][j],
            d2x: derivs.d2x[i][j],
            d2y: derivs.d2y[i][j],
            d2z: derivs.d2z[i][j],
            d2_sum: derivs.d2_sum[i][j],
        };

        H[i][j] = eigen_raw::calc_H(psi[i][j], derivs_single.d2_sum, V_acting_on_this[i][j]);
        L_sq[i][j] = eigen_raw::calc_L_sq(grid_posits[i][j], &derivs_single);
        L_z[i][j] = eigen_raw::calc_L_z(grid_posits[i][j], &derivs_single);

        psi_pp_calculated[i][j] = eigen_fns::find_ψ_pp_calc(psi[i][j], V_acting_on_this[i][j], E)
    }
}

#[allow(unused)]
/// For now, a thin wrapper.
/// todo: When do we use this, vice `calcualte_v_elec`?
pub fn update_eigen_vals_spinor(
    psi_calc: &mut Spinor3,
    derivs: &SpinorDerivsTypeD3,
    E: [f64; 4],
    // V: [f64; 4],
    V: &Arr3dReal,
) {
    // todo: For performance reasons while not using it.
    return;
    // todo: For now, a thin wrapper.
    dirac::calc_psi(psi_calc, derivs, E, V);
}

/// Calculate E using the bases save functions. We assume V goes to 0 at +/- ∞, so
/// edges are, perhaps, a good sample point for this calculation.
pub fn calc_E_from_bases(
    bases: &[Basis],
    V_corner: f64,
    posit_corner: Vec3,
    deriv_calc: DerivCalc,
) -> f64 {
    // Important: eval_pt should be close to +- infinity, but doing so may cause numerical issues
    // as both psi and psi'' go to 0.

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

        let psi_this = weight * basis.value(posit_corner);
        psi += psi_this;

        psi_pp += second_deriv_cpu(psi_this, basis, posit_corner, deriv_calc);
    }

    // todo: WIth the psi_pp_div_psi shortcut, you appear to be getting normalization issues.
    eigen_fns::calc_E_on_psi(psi, psi_pp, V_corner)
}

/// Combine electron charges into a single array, not to include the electron acted on.
pub(crate) fn combine_electron_charges(
    elec_id: usize,
    charges_by_electron: &[Arr3dReal],
    grid_n_charge: usize,
) -> Arr3dReal {
    // todo: Consider modify in place instead of creating something new.
    let mut result = new_data_real(grid_n_charge);

    for (i_charge, charge_from_elec) in charges_by_electron.iter().enumerate() {
        if i_charge == elec_id {
            continue;
        }
        let mut sum = 0.; // todo confirmation heuristic

        for (i, j, k) in iter_arr!(grid_n_charge) {
            result[i][j][k] += charge_from_elec[i][j][k];
            sum += charge_from_elec[i][j][k];
        }
        println!("Charge sum (should be -1): {:.5}", sum);
    }

    result
}

/// Get basis values, and derivatives, using the CPU.
pub(crate) fn calc_vals_derivs_cpu(
    psi: &mut Arr3d,
    // psi_pp: &mut Arr3d,
    derivs: &mut Derivatives,
    grid_posits: &Arr3dVec,
    basis: &Basis,
    grid_n: usize,
    deriv_calc: DerivCalc,
) {
    // *derivs = Derivatives::from_bases(psi, &[basis.clone()], grid_posits);

    let b = [basis.clone()]; // Our API takes &[], not &[&]]
    for (i, j, k) in iter_arr!(grid_n) {
        let posit = grid_posits[i][j][k];

        psi[i][j][k] = basis.value(posit);
        // psi_pp[i][j][k] = calc_derivs_cpu(psi[i][j][k], basis, posit_sample, deriv_calc);

        let d = calc_derivs_cpu(psi[i][j][k], &b, posit, deriv_calc);
        derivs.dx[i][j][k] = d.dx;
        derivs.dy[i][j][k] = d.dy;
        derivs.dz[i][j][k] = d.dz;
        derivs.d2x[i][j][k] = d.d2x;
        derivs.d2y[i][j][k] = d.d2y;
        derivs.d2z[i][j][k] = d.d2z;
        derivs.d2_sum[i][j][k] = d.d2_sum;
    }
}

/// Helper function to help manage numerical vs analytic derivatives. Operates at a single location.
pub(crate) fn calc_derivs_cpu(
    psi: Cplx,
    basis: &[Basis],
    posit: Vec3,
    deriv_calc: DerivCalc,
) -> DerivativesSingle {
    if deriv_calc == DerivCalc::Numeric {
        DerivativesSingle::from_bases(posit, basis, psi)
    } else {
        unimplemented!()
    }

    // todo: Put back if you are able to get analytic derivs of harmonics etc.
    // if basis.n() >= 3 || basis.harmonic().l > 0 {
    //     // num_diff::second_deriv_fm_bases(posit, &[basis.clone()], psi)
    //     DerivativesSingle::from_bases(posit, basis, psi)
    // } else {
    //     unimplemented!()
    //     // basis.second_deriv(posit)
    // }
}

/// Helper fn to help manage numerical vs analytic second derivs. Use this for when we only require
/// the summed second derivative, vice all derivatives.
pub(crate) fn second_deriv_cpu(
    psi: Cplx,
    basis: &Basis,
    posit: Vec3,
    deriv_calc: DerivCalc,
) -> Cplx {
    // todo temp, until we get analytic second derivs with harmonics.
    if deriv_calc == DerivCalc::Numeric {
        num_diff::second_deriv_fm_bases(posit, &[basis.clone()], psi)
    } else {
        unimplemented!()
    }

    // if basis.n() >= 3 || basis.harmonic().l > 0 {
    //     num_diff::second_deriv_fm_bases(posit, &[basis.clone()], psi)
    // } else {
    //     basis.second_deriv(posit)
    // }
}

// /// Helper function to help manage numerical vs analytic derivatives,
// pub fn psi_pp_div_psi_cpu(psi: Cplx, basis: &Basis, posit: Vec3, deriv_calc: DerivCalc) -> f64 {
//     // todo temp, until we get analytic second derivs with harmonics.
//     let psi_pp = num_diff::second_deriv_fm_bases(posit, &[basis.clone()], psi);
//
//     if deriv_calc == DerivCalc::Numeric {
//         return (psi_pp / psi).real;
//     }
//
//     if basis.n() >= 3 || basis.harmonic().l > 0 {
//         let psi_pp = num_diff::second_deriv_fm_bases(posit, &[basis.clone()], psi);
//         (psi_pp / psi).real
//     } else {
//         basis.psi_pp_div_psi(posit)
//     }
// }

// todo: Come back to. Broke during 2D conv, adn unused.
// /// Update surfaces releated to multi-electron wave functions. This includes things related to
// /// spin, and charge density of all electrons. This should be run after changing any electron wave
// /// function, or nucleus charge.
// pub(crate) fn update_combined(
//     shared: &mut SurfacesShared,
//     per_elec: &[SurfacesPerElec],
//     grid_n: usize,
// ) {
//     // Remove previous V from electrons.
//     shared.V_total = shared.V_from_nuclei.clone();
//     shared.psi_alpha = new_data(grid_n);
//     shared.psi_beta = new_data(grid_n);
//     shared.charge_alpha = new_data_real(grid_n);
//     shared.charge_beta = new_data_real(grid_n);
//     shared.charge_density_all = new_data_real(grid_n);
//     shared.spin_density = new_data_real(grid_n);
//     // todo: psi_all too?
//
//     for i_elec in 0..per_elec.len() {
//         for (i, j, k) in iter_arr!(grid_n) {
//             // todo: Handle this.
//             // shared.V_total[i][j][k] += V_from_elecs[i_elec][i][j][k];
//
//             // todo: Raise this if out of the triple loop?
//             match per_elec[i_elec].spin {
//                 Spin::Alpha => {
//                     shared.psi_alpha[i][j][k] += per_elec[i_elec].psi[i][j][k];
//                     shared.charge_alpha[i][j][k] += per_elec[i_elec].charge_density[i][j][k];
//
//                     shared.spin_density[i][j][k] += per_elec[i_elec].charge_density[i][j][k];
//                 }
//                 Spin::Beta => {
//                     shared.psi_beta[i][j][k] += per_elec[i_elec].psi[i][j][k];
//                     shared.charge_beta[i][j][k] += per_elec[i_elec].charge_density[i][j][k];
//
//                     shared.spin_density[i][j][k] -= per_elec[i_elec].charge_density[i][j][k];
//                 }
//             }
//
//             shared.charge_density_all[i][j][k] += per_elec[i_elec].charge_density[i][j][k];
//         }
//     }
// }
