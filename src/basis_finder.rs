//! Used to find weights and Xis for STOs.

use lin_alg::f64::Vec3;
use ndarray::prelude::*;
use ndarray_linalg::SVD;

use crate::{
    basis_wfs::{Basis, Sto},
    complex_nums::Cplx,
    eigen_fns::{calc_E_on_psi, calc_V_on_psi, KE_COEFF_INV},
    grid_setup::{new_data, new_data_real, Arr3dReal, Arr3dVec},
    iter_arr,
    potential::{self, V_coulomb},
    types::ComputationDevice,
    util, wf_ops,
    wf_ops::{DerivCalc, Q_ELEC},
};

/// Normalize a set of weights, to keep the values reasonable and consistent when viewing.
fn normalize_weights(weights: &[f64]) -> Vec<f64> {
    // This approach prevents clipping our UI sliders.
    // todo: Rust's `max` doesn't work with floats. Manually implmementing.
    let mut norm = 0.;
    for weight in weights {
        if weight.abs() > norm {
            norm = weight.abs();
        }
    }
    // This normalization is to keep values reasonable relative to each other; it's subjective.
    let normalize_to = 0.7;

    let mut result = Vec::new();
    for weight in weights {
        // The multiplication factor here keeps values from scaling too heavily.
        result.push(*weight * normalize_to / norm);
    }

    result
}

pub fn generate_sample_pts() -> Vec<Vec3> {
    // It's important that these sample points span points both close and far from the nuclei.

    // Go low to validate high xi, but not too low, Icarus. We currently seem to have trouble below ~0.5 dist.
    let sample_dists = [10., 9., 8., 7., 6., 5., 4., 3.5, 3., 2.5, 2., 1.5, 1., 0.8];

    let mut result = Vec::new();
    for dist in sample_dists {
        // todo: Add back in the others, but for now skipping other dims due to spherical potentials.
        result.push(Vec3::new(dist, 0., 0.));
        // sample_pts_all.push(Vec3::new(0., dist, 0.));
        // sample_pts_all.push(Vec3::new(0., 0., dist));
    }

    result
}

fn find_E_from_base_xi(
    base_xi: f64,
    V_corner: f64,
    posit_corner: Vec3,
    deriv_calc: DerivCalc,
) -> f64 {
    // Now that we've identified the base Xi, use it to calculate the energy of the system.
    // (The energy appears to be determined primarily by it.)
    let base_sto = Basis::Sto(Sto {
        posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
        n: 1,
        xi: base_xi,
        weight: 1.,
        charge_id: 0,
        harmonic: Default::default(),
    });

    let psi_corner = base_sto.value(posit_corner);
    let psi_pp_corner = wf_ops::second_deriv_cpu(psi_corner, &base_sto, posit_corner, deriv_calc);

    calc_E_on_psi(psi_corner, psi_pp_corner, V_corner)
}

fn find_base_xi_E_common(
    V_corner: f64,
    posit_corner: Vec3,
    V_sample: f64,
    posit_sample: Vec3,
    deriv_calc: DerivCalc, // base_xi_specified: f64,
) -> (f64, f64) {
    let mut best_xi_i = 0;
    let mut smallest_diff = f64::MAX;

    // This isn't perhaps an ideal approach, but try it to find the baseline xi.
    let trial_base_xis = util::linspace((1., 3.), 200);

    let mut Es = vec![0.; trial_base_xis.len()];

    for (i, xi_trial) in trial_base_xis.iter().enumerate() {
        let sto = Basis::Sto(Sto {
            posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
            n: 1,                    // todo: Maybe don't hard-set.,
            xi: *xi_trial,
            weight: 1.,
            charge_id: 0,
            harmonic: Default::default(),
        });

        let psi_corner = sto.value(posit_corner);
        let psi_pp_corner = wf_ops::second_deriv_cpu(psi_corner, &sto, posit_corner, deriv_calc);

        Es[i] = calc_E_on_psi(psi_corner, psi_pp_corner, V_corner);

        // We know the corner matches from how we set E. Let's try a different point, and
        // see how close it is.
        let psi = sto.value(posit_sample);
        let psi_pp = wf_ops::second_deriv_cpu(psi, &sto, posit_sample, deriv_calc);

        let V_from_psi = calc_V_on_psi(psi, psi_pp, Es[i]);

        let diff = (V_from_psi - V_sample).abs();

        if diff < smallest_diff {
            best_xi_i = i;
            smallest_diff = diff;
        }
    }

    let base_xi = trial_base_xis[best_xi_i];
    println!(
        "Assessed base xi: {:.3}, E there: {:.3}",
        base_xi, Es[best_xi_i]
    );

    (base_xi, Es[best_xi_i])
}

fn find_base_xi_E(
    charges_fixed: &[(Vec3, f64)],
    charge_elec: &Arr3dReal,
    grid_charge: &Arr3dVec,
    deriv_calc: DerivCalc,
) -> (f64, f64) {
    const SAMPLE_DIST: f64 = 15.;
    let posit_corner = Vec3::new(SAMPLE_DIST, SAMPLE_DIST, SAMPLE_DIST);
    let posit_sample = Vec3::new(SAMPLE_DIST, 0., 0.);

    let mut V_corner = 0.;
    let mut V_sample = 0.;

    for (posit_nuc, charge) in charges_fixed {
        V_sample += V_coulomb(*posit_nuc, posit_sample, *charge);
        V_corner += V_coulomb(*posit_nuc, posit_corner, *charge);
    }

    for (i, j, k) in iter_arr!(charge_elec.len()) {
        let charge = charge_elec[i][j][k];
        // todo: This seems to be the problem with your updated code. This must go back, once
        // todo the system works again from nuc alone.
        V_sample += V_coulomb(grid_charge[i][j][k], posit_sample, charge);
        V_corner += V_coulomb(grid_charge[i][j][k], posit_corner, charge);
    }

    find_base_xi_E_common(V_corner, posit_corner, V_sample, posit_sample, deriv_calc)
}

/// An attempt by evaluating the closest fit using different base xis.
///
/// Alternative take: // For n=1: E = -(xi^2)/2.
/// for n=2: E =
///
/// But how much of this is just the base, vs a blend? And how do they blend?
fn find_base_xi_E2(
    V_to_match: &[f64],
    sample_pts: &[Vec3],
    // non_base_xis: &[f64],
    bases: &Vec<Basis>,
    charges_fixed: &[(Vec3, f64)],
    charge_elec: &Arr3dReal,
    grid_charge: &Arr3dVec,
    deriv_calc: DerivCalc,
) -> (f64, f64) {
    const SAMPLE_DIST: f64 = 20.;
    // let posit_corner = Vec3::new(SAMPLE_DIST, SAMPLE_DIST, SAMPLE_DIST);
    //
    // let V_corner = {
    //     let mut V = 0.;
    //
    //     for (posit_nuc, charge) in charges_fixed {
    //         V += V_coulomb(*posit_nuc, posit_corner, *charge);
    //     }
    //     for (i, j, k) in iter_arr!(charge_elec.len()) {
    //         let charge = charge_elec[i][j][k];
    //         V += V_coulomb(grid_charge[i][j][k], posit_corner, charge);
    //     }
    //     V
    // };

    // let trial_base_xis = util::linspace((0.5, 1.9), 100);
    let trial_base_xis = util::linspace((1.38, 1.41), 10);
    let trial_Es = util::linspace((-0.1, -0.45), 100);

    let mut best_score = 999999.;
    let mut best_xi = 0.;
    let mut best_E = 0.;

    // let mut xis = vec![0.]; // This initial value is the base xi we update with each trial one.
    // for xi in non_base_xis {
    //     xis.push(*xi);
    // }

    let mut bases = bases.clone();

    for xi_trial in &trial_base_xis {
        // xis[0] = *xi_trial;
        *bases[0].xi_mut() = *xi_trial;

        // let base_sto = Basis::Sto(Sto {
        //     posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
        //     n: 1,                    // todo: Maybe don't hard-set.,
        //     xi: *xi_trial,
        //     weight: 1.,
        //     charge_id: 0,
        //     harmonic: Default::default(),
        // });

        // let psi_corner = base_sto.value(posit_corner);
        // let psi_pp_corner = wf_ops::second_deriv(psi_corner, &base_sto, posit_corner);

        // let E_trial = calc_E_on_psi(psi_corner, psi_pp_corner, V_corner);

        for E_trial in &trial_Es {
            let E_trial = *E_trial;

            let bases_this_e =
                find_bases_system_of_eqs(&V_to_match, &bases, &sample_pts, E_trial, deriv_calc);

            let score = score_fit(V_to_match, sample_pts, &bases_this_e, E_trial, deriv_calc);

            println!("Xi: {:.3}, E: {:.3} Score: {:.5}", xi_trial, E_trial, score);

            if score < best_score {
                best_score = score;
                best_xi = *xi_trial;
                best_E = E_trial;
            }
        }
    }

    (best_xi, best_E)
}

/// Stos passed are (xi, weight)
fn find_charge_trial_wf(
    stos: &[(f64, f64)],
    grid_charge: &Arr3dVec,
    grid_n_charge: usize,
) -> Arr3dReal {
    let mut result = new_data_real(grid_n_charge);
    let mut trial_other_elecs_wf = Vec::new();
    for (xi, weight) in stos {
        trial_other_elecs_wf.push(Basis::Sto(Sto {
            posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
            n: 1,
            xi: *xi,
            weight: *weight,
            charge_id: 0,
            harmonic: Default::default(),
        }));
    }

    let mut psi_trial_charge_grid = new_data(grid_n_charge);
    let mut norm = 0.;

    for (i, j, k) in iter_arr!(grid_n_charge) {
        let posit_sample = grid_charge[i][j][k];
        let mut psi = Cplx::new_zero();

        for basis in &trial_other_elecs_wf {
            psi += basis.value(posit_sample);
        }

        psi_trial_charge_grid[i][j][k] = psi;
        norm += psi.abs_sq();
    }

    // Note: We're saving a loop by not calling `elec_elec::update_charge_density_fm_psi`. here,
    // since we need to normalize anyway.

    for (i, j, k) in iter_arr!(grid_n_charge) {
        result[i][j][k] = psi_trial_charge_grid[i][j][k].abs_sq() * Q_ELEC / norm
    }

    result
}

/// See Onenote: `Exploring the WF, part 7`.
/// `V_to_match` indices correspond to `sample_pts`.
///
/// We solve the system F W = V, where F is a matrix of
/// psi'' / psi, with columns for samples position and rows for xi.
/// W is the weight vector we are solving for, and V is V from charge, with h^2/2m and E included.
fn find_bases_system_of_eqs(
    V_to_match: &[f64],
    // bases: &[Sto],
    bases: &[Basis],
    sample_pts: &[Vec3],
    E: f64,
    deriv_calc: DerivCalc,
) -> Vec<Basis> {
    // Bases, from xi, are the rows; positions are the columns.
    // Set this up as a column-major Vec, for use with nalgebra.
    // Note: We are currently using ndarray, which has a row-major constructor;
    // todo: Consider switching, although this col-maj approach prevents reconstructing the STO
    // todo for each value.
    let mut psi_mat_ = Vec::new();
    let mut psi_pp_mat_ = Vec::new();

    for basis in bases {
        let sto = Basis::Sto(Sto {
            posit: basis.posit(),
            n: basis.n(),
            xi: basis.xi(),
            weight: 1., // Weight is 1 here.
            charge_id: basis.charge_id(),
            harmonic: Default::default(), // todo
        });

        // Sample positions are the columns.
        for posit_sample in sample_pts {
            // todo: Real-only for now while building the algorithm, but in general, these are complex.
            let psi = sto.value(*posit_sample);
            let psi_pp = wf_ops::second_deriv_cpu(psi, &sto, *posit_sample, deriv_calc);
            psi_mat_.push(psi.real);
            psi_pp_mat_.push(psi_pp.real);
        }
    }

    let shape = (bases.len(), sample_pts.len());

    let psi_mat = Array::from_shape_vec(shape, psi_mat_).unwrap();
    let psi_mat = psi_mat.t();
    let psi_pp_mat = Array::from_shape_vec(shape, psi_pp_mat_).unwrap();
    let psi_pp_mat = psi_pp_mat.t();

    let rhs: Vec<f64> = V_to_match.iter().map(|V| KE_COEFF_INV * (V + E)).collect();

    let rhs_vec = Array::from_vec(rhs.clone());

    let mat_to_solve = &psi_pp_mat - Array2::from_diag(&rhs_vec).dot(&psi_mat);

    // you can use an iterative method, which will be more efficient for larger matrices where full svd is
    // unfeasible. also check that rust's svd sorts singular values in non-increasing order (pretty sure it does)
    let svd = mat_to_solve.svd(false, true).unwrap();
    let weights = svd.2.unwrap().slice(s![-1, ..]).to_vec();

    let weights_normalized = normalize_weights(&weights);

    let mut result = Vec::new();

    for (i, basis) in bases.iter().enumerate() {
        result.push(Basis::Sto(Sto {
            posit: basis.posit(),
            n: basis.n(),
            xi: basis.xi(),
            weight: weights_normalized[i],
            charge_id: basis.charge_id(),
            harmonic: Default::default(), // todo
        }));
    }

    println!("\nBasis result:");
    for r in &result {
        println!("Xi: {:.2} Weight: {:.2}", r.xi(), r.weight());
    }

    result
}

/// Evaluate an estimated wave function, based on least-squares distance of V trial vs the V it's matching.
/// We may use this for determining base wave function.
fn score_fit(
    V_to_match: &[f64],
    sample_pts: &[Vec3],
    bases_to_eval: &[Basis],
    E: f64,
    deriv_calc: DerivCalc,
) -> f64 {
    let mut V_from_psi = vec![0.; sample_pts.len()];

    for (i, sample_pt) in sample_pts.iter().enumerate() {
        let mut psi_this_pt = Cplx::new_zero();
        let mut psi_pp_this_pt = Cplx::new_zero();

        for basis in bases_to_eval {
            let psi = basis.value(*sample_pt);
            psi_this_pt += psi;
            psi_pp_this_pt += wf_ops::second_deriv_cpu(psi, basis, *sample_pt, deriv_calc);
        }
        V_from_psi[i] = calc_V_on_psi(psi_this_pt, psi_pp_this_pt, E);
    }

    let mut score = 0.;
    for i in 0..sample_pts.len() {
        // println!("Diff: {:?}", (V_from_psi[i] - V_to_match[i]).powi(2));
        println!(
            "Diff: {:?}",
            ((V_from_psi[i] - V_to_match[i]) / V_to_match[i]).powi(2)
        );
        // score += (V_from_psi[i] - V_to_match[i]).powi(2);
        score += ((V_from_psi[i] - V_to_match[i]) / V_to_match[i]).powi(2);
    }

    score
}

/// Find a wave function, composed of STOs, that match a given potential. The potential is calculated
/// in this function from charges associated with nuclei, and other electrons; these must be calculated
/// prior to passing in as parameters.
pub fn run(
    dev_charge: &ComputationDevice,
    charges_fixed: &[(Vec3, f64)],
    charge_elec: &Arr3dReal,
    grid_charge: &Arr3dVec,
    sample_pts: &[Vec3],
    bases: &Vec<Basis>,
    deriv_calc: DerivCalc,
) -> (Vec<Basis>, f64) {
    let mut bases = bases.clone();

    println!("Bases: {:?}", bases);

    let V_to_match = {
        let mut V =
            potential::create_V_1d_from_elecs(dev_charge, sample_pts, charge_elec, grid_charge);

        println!("V from elecs: {:?}", V);

        // let mut V_to_match = vec![0.; sample_pts.len()]; // todo temp to TS our solver

        // Add the V from nucleii charges.
        for (i, sample_pt) in sample_pts.iter().enumerate() {
            for (posit_nuc, charge_nuc) in charges_fixed {
                V[i] += V_coulomb(*posit_nuc, *sample_pt, *charge_nuc);
            }
        }
        println!("V from both: {:?}", V);
        V
    };

    let (base_xi, E) = find_base_xi_E(charges_fixed, charge_elec, grid_charge, deriv_calc);
    // xis[0]);
    // let (base_xi, E) = find_base_xi_E2(
    //     &V_to_match,
    //     sample_pts,
    //     &xis,
    //     charges_fixed,
    //     charge_elec,
    //     grid_charge,
    // );
    //
    // xis[0] = base_xi;

    *bases[0].xi_mut() = base_xi;

    // Code regarding trial wave functions. We are currently not using it, assuming we can converge
    // on a solution from an arbitrary starting point. This seems acceptable for Helium.
    {
        // let mut trial_electron_v = new_data_real(grid_elecd)

        // let other_elecs_charge = find_charge_trial_wf(
        //     &[
        //         (1.0, 1.),
        //         (2., -0.01),
        //         (3., -0.01),
        //         (4., -0.01),
        //         (5., -0.01),
        //         (6., -0.01),
        //     ],
        //     grid_charge,
        //     grid_n_charge,
        // );
    }

    // todo: The above re trial other elec WF or V should be in a wrapper that iterates new
    // todo charge densities based on this trial.

    // let bases = find_bases_system_of_eqs(&V_to_match, &xis, &sample_pts, E);
    let bases = find_bases_system_of_eqs(&V_to_match, &bases, &sample_pts, E, deriv_calc);

    (bases, E)
}
