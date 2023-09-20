//! Used to find weights and Xis for STOs.

use graphics::event::Force::Normalized;
use lin_alg2::f64::Vec3;

use crate::{
    basis_wfs::{Basis, Sto},
    complex_nums::Cplx,
    eigen_fns::{calc_E_on_psi, calc_V_on_psi},
    grid_setup::{new_data, new_data_real, Arr3dReal, Arr3dVec},
    num_diff, potential, util,
    wf_ops::Q_ELEC,
};

// use ndarray::prelude::*;
// use ndarray_linalg::Solve;
use crate::eigen_fns::{KE_COEFF, KE_COEFF_INV};
use nalgebra::{
    ArrayStorage, DMatrix, DVector, Dynamic, Matrix, OMatrix, OVector, SMatrix, SVector,
    VecStorage, U2, U3,
};

// Norms for a given base xi. Computed numerically. These are relative to each other.
// Note: Using norms by dividing and summing from discrete grid sizes. It appearse that it's the ratios
// that matter. Higher grid size, or a true analytic norm would be better.
//
// taken @ grid size = 200
const NORM_TABLE: [(f64, f64); 19] = [
    (1., 25112.383452512597),
    // These are useful as base xis.
    (1.1, 18876.50698245556),
    (1.2, 14542.570125492253),
    (1.3, 11439.026097691232),
    (1.4, 9159.01676430535),
    (1.5, 7446.719975215648),
    (1.6, 6135.947110455885),
    (1.7, 5115.598212989685),
    (1.8, 4309.500917234906),
    (1.9, 3664.248999866286),
    //
    (2., 3141.6457036898487),
    (3., 930.9220618000895),
    (4., 392.80528311671713),
    (5., 201.1943766693086),
    (6., 116.51377521465731),
    (7., 73.45759681761108),
    (8., 49.29721719256224),
    (9., 34.71069659544023),
    (10., 25.39268269325724),
];

/// Create an order-2 polynomial based on 2 or 3 calibration points.
/// `a` is the ^2 term, `b` is the linear term, `c` is the constant term.
/// This is a general mathematical function, and can be derived using a system of equations.
fn create_polynomial_terms(pt0: (f64, f64), pt1: (f64, f64), pt2: (f64, f64)) -> (f64, f64, f64) {
    let a_num = pt0.0 * (pt2.1 - pt1.1) + pt1.0 * (pt0.1 - pt2.1) + pt2.0 * (pt1.1 - pt0.1);

    let a_denom = (pt0.0 - pt1.0) * (pt0.0 - pt2.0) * (pt1.0 - pt2.0);

    let a = a_num / a_denom;
    let b = (pt1.1 - pt0.1) / (pt1.0 - pt0.0) - a * (pt0.0 + pt1.0);
    let c = pt0.1 - a * pt0.0.powi(2) - b * pt0.0;

    (a, b, c)
}

pub fn map_linear(val: f64, range_in: (f64, f64), range_out: (f64, f64)) -> f64 {
    // todo: You may be able to optimize calls to this by having the ranges pre-store
    // todo the total range vals.
    // todo: The real shape is exponential.
    let portion = (val - range_in.0) / (range_in.1 - range_in.0);

    portion * (range_out.1 - range_out.0) + range_out.0
}

/// Interpolate from our norm table.
/// todo: This currently uses linear interpolation, which isn't correct. But, better than
/// todo not interpolating.
fn find_sto_norm(xi: f64) -> f64 {
    let t_len = NORM_TABLE.len();

    for i in 0..t_len {
        // i cutoff here is so we don't hit the right end of the table.
        if i < NORM_TABLE.len() - 2 && xi < NORM_TABLE[i + 1].0 {
            let (a, b, c) =
                create_polynomial_terms(NORM_TABLE[i], NORM_TABLE[i + 1], NORM_TABLE[i + 2]);
            return a * xi.powi(2) + b * xi + c;
        }
    }

    println!("Fallthrough on norm table");
    let (a, b, c) = create_polynomial_terms(
        NORM_TABLE[t_len - 3],
        NORM_TABLE[t_len - 2],
        NORM_TABLE[t_len - 1],
    );

    a * xi.powi(2) + b * xi + c
}

/// Experimental; very.
fn numerical_psi_ps(trial_base_sto: &Basis, grid_posits: &Arr3dVec, V: &Arr3dReal, E: f64) {
    let H = grid_posits[1][0][0].x - grid_posits[0][0][0].x;
    let V_pp_corner = num_diff::find_pp_real(
        V[1][1][1], V[0][1][1], V[2][1][1], V[1][0][1], V[1][2][1], V[1][1][0], V[1][1][2], H,
    );

    // todo QC this
    let V_p_corner = (V[2][1][1] - V[0][1][1])
        + (V[1][2][1] - V[1][0][1])
        + (V[1][1][2] - V[1][1][0]) / (2. * H);

    // let V_pp_psi = trial_base_sto.V_pp_from_psi(posit_corner_offset);
    // let V_p_psi = trial_base_sto.V_p_from_psi(posit_corner_offset);

    // todo: Let's do a cheeky numeric derivative of oV from psi until we're confident the analytic approach
    // todo works.

    // todo well, this is a mess, but it's easy enough to evaluate.
    let posit_x_prev = grid_posits[0][1][1];
    let posit_x_next = grid_posits[2][1][1];
    let posit_y_prev = grid_posits[1][0][1];
    let posit_y_next = grid_posits[1][2][1];
    let posit_z_prev = grid_posits[1][1][0];
    let posit_z_next = grid_posits[1][1][2];

    let psi_x_prev = trial_base_sto.value(posit_x_prev);
    let psi_pp_x_prev = trial_base_sto.second_deriv(posit_x_prev);
    let psi_x_next = trial_base_sto.value(posit_x_next);
    let psi_pp_x_next = trial_base_sto.second_deriv(posit_x_next);

    let psi_y_prev = trial_base_sto.value(posit_y_prev);
    let psi_pp_y_prev = trial_base_sto.second_deriv(posit_y_prev);
    let psi_y_next = trial_base_sto.value(posit_y_next);
    let psi_pp_y_next = trial_base_sto.second_deriv(posit_y_next);

    let psi_z_prev = trial_base_sto.value(posit_z_prev);
    let psi_pp_z_prev = trial_base_sto.second_deriv(posit_z_prev);
    let psi_z_next = trial_base_sto.value(posit_z_next);
    let psi_pp_z_next = trial_base_sto.second_deriv(posit_z_next);

    let V_p_psi = ((calc_V_on_psi(psi_x_next, psi_pp_x_next, E)
        - calc_V_on_psi(psi_x_prev, psi_pp_x_prev, E))
        + (calc_V_on_psi(psi_y_next, psi_pp_y_next, E)
            - calc_V_on_psi(psi_y_prev, psi_pp_y_prev, E))
        + (calc_V_on_psi(psi_z_next, psi_pp_z_next, E)
            - calc_V_on_psi(psi_z_prev, psi_pp_z_prev, E)))
        / (2. * H);

    println!("V' corner: Blue {}  Grey {}", V_p_corner, V_p_psi);
    // println!("V'' corner: Blue {}  Grey {}", V_pp_corner, V_pp_psi);
}

fn find_base_xi_E_common(
    V_corner: f64,
    posit_corner: Vec3,
    V_sample: f64,
    posit_sample: Vec3,
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

    let E = calc_E_on_psi(psi_corner, psi_pp_corner, V_corner);

    (base_xi, E)
}

fn find_base_xi_E(V: &Arr3dReal, grid_posits: &Arr3dVec) -> (f64, f64) {
    // Set energy so that at a corner, (or edge, ie as close to +/- infinity as we have given a grid-based V)
    // V calculated from this basis matches the potential at this point.
    let posit_corner = grid_posits[0][0][0];
    let posit_sample = grid_posits[20][0][0];

    let V_corner = V[0][0][0];
    // todo: Very rough and hard-set!
    let V_sample = V[20][0][0];

    find_base_xi_E_common(V_corner, posit_corner, V_sample, posit_sample)
}

fn find_base_xi_E_type2(
    charges_fixed: &[(Vec3, f64)],
    charge_elec: &Arr3dReal,
    grid_charge: &Arr3dVec,
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

    find_base_xi_E_common(V_corner, posit_corner, V_sample, posit_sample)
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

fn generate_sample_pts_per_xi() -> Vec<Vec<Vec3>> {
    //  It's important that these sample points span points both close and far from the nuclei.

    // let sample_dists = vec![0.1, 0.3, 0.8, 1., 1.5, 2., 3., 4., 8., 15.];

    // 1: 3
    // 2: 1.8
    // 3: 1.3
    // 4: 1.0
    // 5: 0.75
    // 6: 0.63

    // These sample distances correspon d to xi.
    // todo: Hard-coded for now, starting at xi=1. Note that different base xis will throw this off!
    let sample_dists_per_xi = [
        3., 2., 1.5, 1., 0.75, 0.5, // 6
        0.4, 0.35, 0.3, 0.25,
    ];

    let mut result = Vec::new();
    for dist in sample_dists_per_xi {
        let mut set = Vec::new();
        // for scaler in &[0.4, 1., 1.4] {
        for scaler in &[0.3, 0.5, 1.] {
            set.push(Vec3::new(dist * scaler, 0., 0.));
            set.push(Vec3::new(0., dist * scaler, 0.));
            set.push(Vec3::new(0., 0., dist * scaler));
        }
        result.push(set);
    }

    result
}

fn generate_sample_pts() -> Vec<Vec3> {
    // It's important that these sample points span points both close and far from the nuclei.

    // Go low to validate high xi, but not too low, Icarus.
    let sample_dists = [
        // 3., 2.5, 2.0, 1.7, 1.3, 1.0, 0.75, 0.63, 0.5, 0.45, 0.42, 0.4, 0.35, 0.3,
        3., 2.0, 1.3, 1.0, 0.63, 0.5, 0.45, 0.42, 0.4, 0.35, 0.3,
    ];

    let mut result = Vec::new();
    for dist in sample_dists {
        // todo: Add back in the others, but for now skipping other dims due to spherical potentials.
        result.push(Vec3::new(dist, 0., 0.));
        // sample_pts_all.push(Vec3::new(0., dist, 0.));
        // sample_pts_all.push(Vec3::new(0., 0., dist));
    }

    result
}

/// Using a set of points to evaluate a basis at, tailored for a specifix xi.
fn make_ref_V_per_xi(
    sample_pt_sets: &[Vec<Vec3>],
    charges_fixed: &[(Vec3, f64)],
    charge_elec: &Arr3dReal,
    grid_charge: &Arr3dVec,
    grid_n_charge: usize,
) -> Vec<Vec<f64>> {
    let mut V_to_match_outer = Vec::new(); // Outer: By xi.

    for sample_pts in sample_pt_sets {
        let mut V_to_match_inner = Vec::new(); // By posit. (eg the 3 posits per dist defined above)

        for posit_sample in sample_pts {
            let mut V_sample = 0.;

            for (posit_nuc, charge) in charges_fixed {
                V_sample += potential::V_coulomb(*posit_nuc, *posit_sample, *charge);
            }

            for i in 0..grid_n_charge {
                for j in 0..grid_n_charge {
                    for k in 0..grid_n_charge {
                        let posit_charge = grid_charge[i][j][k];
                        let charge = charge_elec[i][j][k];

                        V_sample += potential::V_coulomb(posit_charge, *posit_sample, charge);
                    }
                }
            }
            V_to_match_inner.push(V_sample);
        }

        V_to_match_outer.push(V_to_match_inner);
    }

    V_to_match_outer
}

/// Using a flat set of sample points.
fn make_ref_V_flat(
    sample_pts: &[Vec3],
    charges_fixed: &[(Vec3, f64)],
    charge_elec: &Arr3dReal,
    grid_charge: &Arr3dVec,
    grid_n_charge: usize,
) -> Vec<f64> {
    let mut V_to_match = Vec::new();

    for sample_pt in sample_pts {
        let mut V_sample = 0.;

        for (posit_nuc, charge) in charges_fixed {
            V_sample += potential::V_coulomb(*posit_nuc, *sample_pt, *charge);
        }

        for i in 0..grid_n_charge {
            for j in 0..grid_n_charge {
                for k in 0..grid_n_charge {
                    let posit_charge = grid_charge[i][j][k];
                    let charge = charge_elec[i][j][k];

                    V_sample += potential::V_coulomb(posit_charge, *sample_pt, charge);
                }
            }
        }

        V_to_match.push(V_sample);
    }

    V_to_match
}

fn find_bases(
    V_to_match: &[f64],
    additional_xis: &[f64],
    base_sto: &Basis,
    base_xi: f64,
    E: f64,
    sample_pts_all: &[Vec3],
) -> Vec<Basis> {
    const EPS: f64 = 0.0001;
    let weights = util::linspace((-1., 1.), 200);
    let base_norm = find_sto_norm(base_xi);

    let mut bases = Vec::new();
    for xi in additional_xis {
        // if *xi <= base_xi + EPS {
        //     continue;
        // }

        bases.push(Basis::Sto(Sto {
            posit: Vec3::new_zero(),
            n: 1,
            xi: *xi,
            weight: 0.,
            charge_id: 0,
            harmonic: Default::default(),
        }))
    }

    for (i_xi, xi) in additional_xis.iter().enumerate() {
        if *xi <= base_xi + EPS {
            continue;
        }

        // Re-generate `psi_other_basis`, because we add a new base each time.
        let mut psi_other_bases = Vec::new();
        let mut psi_pp_other_bases = Vec::new();

        // For this xi, calculate psi and psi'' for the already-added bases.
        // for pt in &sample_pt_sets[i_xi] {
        for pt in sample_pts_all {
            // todo: Is V linear with these? Can we simply calculate V_other directly? Try once this approach works.
            let mut psi = Cplx::new_zero();
            let mut psi_pp = Cplx::new_zero();

            const EPS: f64 = 0.00001;

            // todo: DRY here between the base basis and the others.
            let weight = Cplx::from_real(base_sto.weight()) / base_norm;

            psi += weight * base_sto.value(*pt);
            psi_pp += weight * base_sto.second_deriv(*pt);

            for basis in &bases {
                if basis.weight().abs() < EPS {
                    continue;
                }
                let weight = Cplx::from_real(basis.weight()) / find_sto_norm(basis.xi());

                psi += weight * basis.value(*pt);
                psi_pp += weight * basis.second_deriv(*pt);
            }

            psi_other_bases.push(psi);
            psi_pp_other_bases.push(psi_pp);
        }

        let mut best_weight_i = 0;
        let mut smallest_diff = 9999.;

        for (i_weight, weight) in weights.iter().enumerate() {
            let sto = Basis::Sto(Sto {
                posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
                n: 1,
                xi: *xi,
                weight: *weight,
                charge_id: 0,
                harmonic: Default::default(),
            });

            let mut cum_diff_this_set = 0.;

            // for (i_pt, pt) in sample_pt_sets[i_xi].iter().enumerate() {
            for (i_pt, pt) in sample_pts_all.iter().enumerate() {
                let weight_ = Cplx::from_real(*weight) / find_sto_norm(*xi);

                let mut psi = psi_other_bases[i_pt] + weight_ * sto.value(*pt);

                let mut psi_pp = psi_pp_other_bases[i_pt] + weight_ * sto.second_deriv(*pt);

                let V_from_psi = calc_V_on_psi(psi, psi_pp, E);
                // let V_to_match = V_to_match_per_xi[i_xi][i_pt];
                let V_to_match = V_to_match[i_pt];

                if *xi < 5. {
                    // println!(
                    //     "V. Xi: {}, Weight: {:.3}, From psi: {:.3}, To match: {:.3} psi {:.3} psi'' {:.3}",
                    //     xi,
                    //     weight,
                    //     V_from_psi,
                    //     V_to_match,
                    //     // (V_from_psi - V_to_match).abs()
                    //     psi.real,
                    //     psi_pp.real,
                    // );

                    // println!("PREV V: {:.4}", calc_V_on_psi(psi_other_bases[i_pt], psi_pp_other_bases[i_pt], E))

                    // println!("Aux. psi_other {:.6} psi_pp_other: {:.6}, psi_this: {:.6}, psi_pp_this: {:.6}\n",
                    //          psi_other_bases[i_pt].real, psi_pp_other_bases[i_pt].real, Cplx::from_real(*weight) * sto.value(*pt).real / norm, Cplx::from_real(*weight) * sto.second_deriv(*pt).real / norm);
                }

                let this_diff = (V_from_psi - V_to_match).abs();

                cum_diff_this_set += this_diff;
            }

            if cum_diff_this_set < smallest_diff {
                best_weight_i = i_weight;
                smallest_diff = cum_diff_this_set;
            }
        }

        let best_weight = weights[best_weight_i];

        *bases[i_xi].weight_mut() = best_weight;
    }

    for basis in &bases {
        if let Basis::Sto(sto) = basis {
            println!("Xi: {}, weight: {}", sto.xi, sto.weight);
        }
    }

    bases
}

/// See Onenote: `Exploring the WF, part 7`.
/// `V_to_match` indices correspond to `sample_pts`.
///
/// We solve the system F W = V, where F is a matrix of
/// psi'' / psi, with columns for samples position and rows for xi.
/// W is the weight vector we are solving for, and V is V from charge, with h^2/2m and E included.
fn find_bases_system_of_eqs(
    V_to_match: &[f64],
    additional_xis: &[f64],
    base_xi: f64,
    E: f64,
    sample_pts: &[Vec3],
) -> Vec<Basis> {
    let mut bases = Vec::new();

    // type PSI_RATIO_MAT = DMatrix<f64, Dynamic, Dynamic>;
    // type V_CHARGE_VEC = OVector<f64, Dynamic>;

    // Construct our matrix F, or psi_pp / psi ratios.

    // todo: TS
    let mut xis = vec![base_xi];
    let mut xis = vec![];
    for xi in additional_xis {
        // if xi > &base_xi {
        xis.push(*xi);
        // }
    }

    // Bases, from xi, are the rows; positions are the columns.
    // Set this up as a column-major Vec, for use with nalgebra.
    let mut psi_ratio_mat_ = Vec::new();
    for xi in &xis {
        let norm = Cplx::from_real(find_sto_norm(*xi));
        let sto = Basis::Sto(Sto {
            posit: Vec3::new_zero(), // todo: Hard-coded for now.
            n: 1,
            xi: *xi,
            weight: 1., // Weight is 1 here.
            charge_id: 0,
            harmonic: Default::default(),
        });

        // Sample positions are the columns.
        for posit_sample in &sample_pts[..xis.len()] {
            let psi = norm * sto.value(*posit_sample);
            let psi_pp = norm * sto.second_deriv(*posit_sample);

            psi_ratio_mat_.push((psi_pp / psi).real);
        }
    }

    let psi_ratio_mat = DMatrix::from_vec(xis.len(), xis.len(), psi_ratio_mat_);
    // let psi_ratio_mat: SMatrix<f64, 11, 11> = SMatrix::from_vec(psi_ratio_mat_);

    // println!("Psi ratio mat: {}", psi_ratio_mat);

    // Set the charge vector V.
    //         // todo: CHeck teh sign on E. You still have an anomoly here. On paper, it shows - E,
    //         // todo but you've been inverting this in practice to make it work.
    //         // v_charge_vec_.push(KE_COEFF_INV * (V - E));
    let v_charge_vec_: Vec<f64> = V_to_match.iter().map(|V| KE_COEFF_INV * (V + E)).collect();

    let v_charge_vec = DVector::from_vec(v_charge_vec_[..xis.len()].to_owned());

    // Solve for the weights vector. https://nalgebra.org/docs/user_guide/decompositions_and_lapack

    let decomp = psi_ratio_mat.lu();
    // let decomp = psi_ratio_mat.cholesky().unwrap();
    let mut weights = decomp
        .solve(&v_charge_vec)
        .expect("Linear resolution failed.");

    // Normalize re base xi.
    let base_weight = weights[0];
    for weight in &mut weights {
        *weight /= base_weight;
    }

    println!("Xis: {:.3?}", xis);
    println!("Weights: {:.6}", weights);

    // todo: See nalgebra Readme on BLAS etc as-required if you wish to optomize.

    bases
}

/// Find a wave function, composed of STOs, that match a given potential.
// pub fn find_stos(V: &Arr3dReal, grid_posits: &Arr3dVec) -> (Vec<Basis>, f64) {
pub fn find_stos(
    charges_fixed: &[(Vec3, f64)],
    charge_elec: &Arr3dReal,
    grid_charge: &Arr3dVec,
    grid_n_charge: usize,
) -> (Vec<Basis>, f64) {
    // let (base_xi, E) = find_base_xi_E(V, grid_posits);
    let (base_xi, E) = find_base_xi_E_type2(charges_fixed, charge_elec, grid_charge);
    println!("\nBase xi: {}. E: {}\n", base_xi, E);

    // todo: Move this part about trial elec V A/R
    // todo: Frame in terms of a trial psi via STOs instead? Equivlanet. Less direct,
    // todo but may be easier to construct
    // let mut trial_electron_v = new_data_real(grid_elecd)

    // Rough, observational area of where each Xi should match:
    // 1: outer edge (?)5+
    // 2: 4.5
    // 3: 4.
    // 4: 2.
    // 5: 0.5
    // 6: 0.25?
    // 7:
    // 8:
    // 9:
    // 10:

    // I think maybe find the -3dB point or similar, when looking at the derivative?

    // Second derivative peaks: 1/xi
    // 1: 1
    // 2: 0.5
    // 3: 0.333
    // 4: 0.25
    // 5: 0.2

    // Eyeballing a dropoff point on first derivative: (rough) This seems to match the dimple in the
    // 3d plots.

    // 1: 3
    // 2: 1.8
    // 3: 1.3
    // 4: 1.0
    // 5: 0.75
    // 6: 0.63

    let other_elecs_charge = find_charge_trial_wf(
        &vec![
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

    // Now, add more xis:

    let additional_xis = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];

    let base_sto = Basis::Sto(Sto {
        posit: charges_fixed[0].0, // todo: Hard-coded for a single nuc.
        n: 1,
        xi: base_xi,
        weight: 1.,
        charge_id: 0,
        harmonic: Default::default(),
    });

    // let sample_pt_sets = generate_sample_pts_per_xi();
    let sample_pts = generate_sample_pts();

    // let V_to_match_per_xi = make_ref_V_per_xi(
    //     &sample_pt_sets,
    //     charges_fixed,
    //     charge_elec,
    //     grid_charge,
    //     grid_n_charge,
    // );

    let V_to_match = make_ref_V_flat(
        &sample_pts,
        charges_fixed,
        charge_elec,
        grid_charge,
        grid_n_charge,
    );

    // todo: 18 Sep 2023: try this - Once you have base xi, find the lin combo that solves
    // for each sample pt. One sample pt per basis. Matrix / etc approach.

    let bases = find_bases_system_of_eqs(&V_to_match, &additional_xis, base_xi, E, &sample_pts);

    (bases, E)
}
