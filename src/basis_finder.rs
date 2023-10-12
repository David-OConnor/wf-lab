//! Used to find weights and Xis for STOs.

use lin_alg2::f64::Vec3;

use crate::{
    basis_wfs::{Basis, Sto},
    complex_nums::Cplx,
    eigen_fns::{calc_E_on_psi, calc_V_on_psi},
    grid_setup::{new_data, new_data_real, Arr3dReal, Arr3dVec},
    num_diff, potential, util,
    wf_ops::Q_ELEC,
};

use crate::eigen_fns::KE_COEFF_INV;
use nalgebra::{DMatrix, DVector};

use ndarray::prelude::*;
use ndarray_linalg::{Solve, SVD};



fn find_E_from_base_xi(base_xi: f64, V_corner: f64, posit_corner: Vec3) -> f64 {
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

    // todo: Helper fn for this process? Lots of DRY.
    let psi_corner = base_sto.value(posit_corner);
    let psi_pp_corner = base_sto.second_deriv(posit_corner);

    calc_E_on_psi(psi_corner, psi_pp_corner, V_corner)
}

fn find_base_xi_E_common(
    V_corner: f64,
    posit_corner: Vec3,
    V_sample: f64,
    posit_sample: Vec3,
    base_xi_specified: f64,
) -> (f64, f64) {
    // todo: Try at different ns, one the `value` and `second_deriv` there for STOs are set up appropritaely.

    let mut best_xi_i = 0;
    let mut smallest_diff = 99999.;
    // This isn't perhaps an ideal apperoach, but try it to find the baseline xi.
    let trial_base_xis = util::linspace((1., 5.), 100);
    for (i, trial_xi) in trial_base_xis.iter().enumerate() {
        let sto = Basis::Sto(Sto {
            posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
            n: 1,
            xi: *trial_xi,
            weight: 1.,
            charge_id: 0,
            harmonic: Default::default(),
        });

        let psi_corner = sto.value(posit_corner);
        let psi_pp_corner = sto.second_deriv(posit_corner);
        let E_ = calc_E_on_psi(psi_corner, psi_pp_corner, V_corner);

        // We know the corner matches from how we set E. Let's try a different point.

        let psi = sto.value(posit_sample);
        let psi_pp = sto.second_deriv(posit_sample);
        let V_from_psi = calc_V_on_psi(psi, psi_pp, E_);

        let diff = (V_from_psi - V_sample).abs();
        if diff < smallest_diff {
            best_xi_i = i;
            smallest_diff = diff;
        }
    }

    let base_xi = trial_base_xis[best_xi_i];

    // todo temp?
    let base_xi = base_xi_specified;

    let E = find_E_from_base_xi(base_xi, V_corner, posit_corner);

    (base_xi, E)
}

fn find_base_xi_E(V: &Arr3dReal, grid_posits: &Arr3dVec, base_xi_specified: f64) -> (f64, f64) {
    // Set energy so that at a corner, (or edge, ie as close to +/- infinity as we have given a grid-based V)
    // V calculated from this basis matches the potential at this point.
    let posit_corner = grid_posits[0][0][0];
    let posit_sample = grid_posits[20][0][0];

    let V_corner = V[0][0][0];
    // todo: Very rough and hard-set!
    let V_sample = V[20][0][0];

    find_base_xi_E_common(V_corner, posit_corner, V_sample, posit_sample, base_xi_specified)
}

fn find_base_xi_E_type2(
    charges_fixed: &[(Vec3, f64)],
    charge_elec: &Arr3dReal,
    grid_charge: &Arr3dVec,
    base_xi_specified: f64,
) -> (f64, f64) {
    let posit_corner = Vec3::new(30., 30., 30.);
    let posit_sample = Vec3::new(30., 0., 0.);

    let mut V_corner = 0.;
    let mut V_sample = 0.;

    for (posit_nuc, charge) in charges_fixed {
        V_sample += potential::V_coulomb(*posit_nuc, posit_sample, *charge);
        V_corner += potential::V_coulomb(*posit_nuc, posit_corner, *charge);
    }

    for i in 0..charge_elec.len() {
        for j in 0..charge_elec.len() {
            for k in 0..charge_elec.len() {
                let charge = charge_elec[i][j][k];
                V_sample += potential::V_coulomb(grid_charge[i][j][k], posit_sample, charge);
                V_corner += potential::V_coulomb(grid_charge[i][j][k], posit_corner, charge);
            }
        }
    }

    find_base_xi_E_common(V_corner, posit_corner, V_sample, posit_sample, base_xi_specified)
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
    for i in 0..grid_n_charge {
        for j in 0..grid_n_charge {
            for k in 0..grid_n_charge {
                let posit_sample = grid_charge[i][j][k];
                let mut psi = Cplx::new_zero();

                for basis in &trial_other_elecs_wf {
                    psi += basis.value(posit_sample);
                }

                psi_trial_charge_grid[i][j][k] = psi;
                norm += psi.abs_sq();
            }
        }
    }

    // Note: We're saving a loop by not calling `elec_elec::update_charge_density_fm_psi`. here,
    // since we need to normalize anyway.
    for i in 0..grid_n_charge {
        for j in 0..grid_n_charge {
            for k in 0..grid_n_charge {
                result[i][j][k] = psi_trial_charge_grid[i][j][k].abs_sq() * Q_ELEC / norm
            }
        }
    }

    result
}

fn generate_sample_pts() -> Vec<Vec3> {
    // It's important that these sample points span points both close and far from the nuclei.

    // Go low to validate high xi, but not too low, Icarus. We currently seem to have trouble below ~0.5 dist.
    let sample_dists = [
        // 10., 5., 3., 2., 1.5, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1
        10., 9., 8., 7., 6., 5., 4., 3.5, 3., 2.5, 2., 1.5, 1., 0.8, 0.7, 0.6, 0.6, 0.4
    ];

    // println!("\nSample dists: {:?}", sample_dists);

    let mut result = Vec::new();
    for dist in sample_dists {
        // todo: Add back in the others, but for now skipping other dims due to spherical potentials.
        result.push(Vec3::new(dist, 0., 0.));
        // sample_pts_all.push(Vec3::new(0., dist, 0.));
        // sample_pts_all.push(Vec3::new(0., 0., dist));
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
    xis: &[f64],
    sample_pts: &[Vec3],
    E: f64,
) -> Vec<Basis> {
    // Bases, from xi, are the rows; positions are the columns.
    // Set this up as a column-major Vec, for use with nalgebra.
    // Note: We are currently using ndarray, which has a row-major constructor;
    // todo: Consider switching, although this col-maj approach prevents reconstructing the STO
    // todo for each value.
    let mut psi_mat_ = Vec::new();
    let mut psi_pp_mat_ = Vec::new();

    for xi in xis {
        let sto = Basis::Sto(Sto {
            posit: Vec3::new_zero(), // todo: Hard-coded for now.
            n: 1,
            xi: *xi,
            weight: 1., // Weight is 1 here.
            charge_id: 0,
            harmonic: Default::default(),
        });

        // todo: Experimenting with non-square matrix
        // Sample positions are the columns.
        // for posit_sample in &sample_pts[i_range.clone()] {
        for posit_sample in sample_pts {
            // todo: Real-only for now while building the algorithm, but in general, these are complex.
            psi_mat_.push(sto.value(*posit_sample).real);
            psi_pp_mat_.push(sto.second_deriv(*posit_sample).real);
        }
    }

    let psi_mat = Array::from_shape_vec((xis.len(), sample_pts.len()), psi_mat_).unwrap();
    let psi_mat = psi_mat.t();
    let psi_pp_mat = Array::from_shape_vec((xis.len(), sample_pts.len()), psi_pp_mat_).unwrap();
    let psi_pp_mat = psi_pp_mat.t();

    let rhs: Vec<f64> = V_to_match
        .iter()
        .map(|V| KE_COEFF_INV * (V + E))
        .collect();

    let rhs_vec = Array::from_vec(rhs.clone());

    let mat_to_solve = &psi_pp_mat - Array2::from_diag(&rhs_vec).dot(&psi_mat);

    // you can use an iterative method, which will be more efficient for larger matrices where full svd is
    // unfeasible. also check that rust's svd sorts singular values in non-increasing order (pretty sure it does)
    let svd = mat_to_solve.svd(false, true).unwrap();
    let mut weights = svd.2.unwrap().slice(s![-1, ..]).to_vec();

    // Normalize re base xi.


    // let base_weight = weights[0];
    // let base_val = 0.2;

    // This approach prevents clipping our UI sliders.
    // todo: Rust's `max` doesn't work with floats. Manually implmementing.
    let mut highest_weight = 0.;
    for weight in &weights {
        if weight.abs() > highest_weight {
            highest_weight = *weight;
        }
    }
    let base_weight = highest_weight;

    let normalize_to = 1.2;

    let mut weights_normalized = Vec::new();
    for weight in &weights {
        // The multiplication factor here keeps values from scaling too heavily.
        weights_normalized.push(*weight * normalize_to / base_weight);
    }

    println!("\nXis: {:.3?}", xis);
    println!("Sample pts: {:?}", &sample_pts);
    println!("V to match: {:?}", &V_to_match);

    println!("\nWeights: {:.6?}", weights);
    println!("Weights normalized: {:.6?}\n", weights_normalized);

    let mut result = Vec::new();

    for (i, xi) in xis.iter().enumerate() {
        result.push(Basis::Sto(Sto {
            posit: Vec3::new_zero(), // todo: Hard-coded for now.
            n: 1,
            xi: *xi,
            weight: weights_normalized[i],
            charge_id: 0,
            harmonic: Default::default(),
        }));
    }

    result
}

/// Find a wave function, composed of STOs, that match a given potential.
// pub fn find_stos(V: &Arr3dReal, grid_posits: &Arr3dVec) -> (Vec<Basis>, f64) {
pub fn find_stos(
    charges_fixed: &[(Vec3, f64)],
    charge_elec: &Arr3dReal,
    grid_charge: &Arr3dVec,
    grid_n_charge: usize,
    // todo: We are experimenting with using the input xis and output for helium
    // Note that this is an intermediate step while we manually experiment with
    // todo convergence algos and trial WFs.
    xis: &[f64],
) -> (Vec<Basis>, f64) {
    // let additional_xis = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    // let xis = [1.41714, 2.37682, 4.39628, 6.52699, 7.94252, 5.];
    // let xis = [1.5, 2.37682, 4.39628, 6.52699, 7.94252];

    let mut xis = Vec::from(xis); // todo: Experimenting with adding more
    xis.push(7.);
    // xis.push(8.);
    // xis.push(9.);

    let (base_xi, E) = find_base_xi_E_type2(charges_fixed, charge_elec, grid_charge, xis[0]);
    println!("\nBase xi: {}. E: {}\n", base_xi, E);

    // todo: Move this part about trial elec V A/R
    // todo: Frame in terms of a trial psi via STOs instead? Equivlanet. Less direct,
    // todo but may be easier to construct
    // let mut trial_electron_v = new_data_real(grid_elecd)

    let other_elecs_charge = find_charge_trial_wf(
        &[
            (1.0, 1.),
            (2., -0.01),
            (3., -0.01),
            (4., -0.01),
            (5., -0.01),
            (6., -0.01),
        ],
        grid_charge,
        grid_n_charge,
    );

    // todo: The above re trial other elec WF or V should be in a wrapper that iterates new
    // todo charge densities based on this trial.

    let sample_pts = generate_sample_pts();

    let V_to_match = potential::create_V_1d(
        &sample_pts,
        charges_fixed,
        charge_elec,
        grid_charge,
        grid_n_charge,
    );

    // let bases = find_bases_system_of_eqs(&V_to_match, &additional_xis, base_xi, &sample_pts, E);
    let bases = find_bases_system_of_eqs(&V_to_match, &xis, &sample_pts, E);

    (bases, E)
}
