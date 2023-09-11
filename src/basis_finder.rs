//! Used to find weights and Xis for STOs.

use lin_alg2::f64::Vec3;

use crate::{
    basis_wfs::{Basis, Sto},
    complex_nums::Cplx,
    eigen_fns::{self, calc_E_on_psi, calc_V_on_psi},
    elec_elec,
    grid_setup::{new_data, new_data_real, Arr3d, Arr3dReal, Arr3dVec},
    num_diff, potential, util, wf_ops,
    wf_ops::{Q_ELEC, Q_PROT},
};

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

fn find_charge_trial_wf(grid_charge: &Arr3dVec, grid_n_charge: usize) -> Arr3dReal {
    let mut result = new_data_real(grid_n_charge);
    {
        let mut trial_other_elecs_wf = vec![Basis::Sto(Sto {
            posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
            n: 1,
            xi: 1.5,
            weight: 1.,
            charge_id: 0,
            harmonic: Default::default(),
        })];

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

    let other_elecs_charge = find_charge_trial_wf(grid_charge, grid_n_charge);

    // todo: The above re trial other elec WF or V should be in a wrapper that iterates new
    // todo charge densities based on this trial.

    let base_sto = Basis::Sto(Sto {
        posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
        n: 1,
        xi: base_xi,
        weight: 1.,
        charge_id: 0,
        harmonic: Default::default(),
    });

    // Now, add more xis:

    let additional_xis = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
    let weights = util::linspace((-1., 1.), 200);

    // Note: Using norms by dividing and summing from discrete grid sizes. It appearse that it's the ratios
    // that matter. Higher grid size, or a true analytic norm would be better.
    // taken @ size = 100
    // todo: Try to get even higher, but this is probably fine.
    let norms = [
        5424.308042880133, // 1
        678.6725787289182, // 2
        201.19437666856675,
        84.9987267609355,
        43.64765306169169,
        25.392682693257417,
        16.127608754901615,
        10.942927647325494,
        7.824996490004794,
        5.8437728521755785, // 10
    ];

    // todo: Not ideal. Set up sample points at specific intervals, then evaluate
    // todo V at those intervals; don't use the grid.

    // tod: It's important that these sample points span points both close and far from the nuclei.

    // let sample_dists = vec![0.1, 0.3, 0.8, 1., 1.5, 2., 3., 4., 8., 15.];

    // Each of these distances corresponds to a xi.
    // let mut sample_dists = Vec::new();
    for xi in &additional_xis {}

    // 1: 3
    // 2: 1.8
    // 3: 1.3
    // 4: 1.0
    // 5: 0.75
    // 6: 0.63

    // These sample distances correspon d to xi.
    // todo: Hard-coded for now, starting at xi=1. Note that different base xis will throw this off!
    let sample_dists_per_xi = [
        3., // 1.8,
        2.5, 1.3, 1., 0.75, 0.63, // 6
        0.5, 0.45, 0.42, 0.4,
    ];

    let sample_dists = [
        3., 2.5, 2.0, 1.7, 1.3, 1.0, 0.75, 0.63, // 6
        0.5, 0.45, 0.42, 0.4, 0.35, 0.3,
    ];

    let mut sample_pt_sets = Vec::new();
    for dist in sample_dists_per_xi {
        sample_pt_sets.push(vec![
            Vec3::new(dist, 0., 0.),
            Vec3::new(0., dist, 0.),
            Vec3::new(0., 0., dist),
        ]);
    }

    let mut sample_pts_all = Vec::new();
    for dist in sample_dists {
        sample_pts_all.push(Vec3::new(dist, 0., 0.));
        sample_pts_all.push(Vec3::new(0., dist, 0.));
        sample_pts_all.push(Vec3::new(0., 0., dist));
    }

    let mut V_to_match_outer = Vec::new(); // Outer: By xi.

    for sample_pts in &sample_pt_sets {
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

    let mut bases = vec![base_sto];
    for xi in &additional_xis {
        if xi <= &base_xi {
            continue;
        }

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
        if xi <= &base_xi {
            continue;
        }

        // Re-generate `psi_other_basis`, because we add a new base each time.
        let mut psi_other_bases = Vec::new();
        let mut psi_pp_other_bases = Vec::new();

        // For this xi, calculate psi and psi'' for the already-added bases.
        // for pt in &sample_pt_sets[i_xi] {
        for pt in &sample_pts_all {
            // todo: Is V linear with these? Can we simply calculate V_other directly? Try once this approach works.
            let mut psi = Cplx::new_zero();
            let mut psi_pp = Cplx::new_zero();

            const EPS: f64 = 0.00001;

            for (i_basis, basis) in bases.iter().enumerate() {
                if basis.weight().abs() < EPS {
                    continue;
                }
                let weight = Cplx::from_real(basis.weight()) / norms[i_basis];

                psi += weight * basis.value(*pt);
                psi_pp += weight * basis.second_deriv(*pt);
            }

            // let psi = Cplx::new_zero(); // todo temp! Experimenting to make sure absolute weight doesn' tmatter.
            // let psi_pp = Cplx::new_zero(); // todo tmep!!
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

            for (i_pt, pt) in sample_pt_sets[i_xi].iter().enumerate() {
                let weight_ = Cplx::from_real(*weight) / norms[i_xi];

                let mut psi = psi_other_bases[i_pt] + weight_ * sto.value(*pt);

                let mut psi_pp = psi_pp_other_bases[i_pt] + weight_ * sto.second_deriv(*pt);

                let V_from_psi = calc_V_on_psi(psi, psi_pp, E);
                let V_to_match = V_to_match_outer[i_xi][i_pt];

                if *xi < 5. {
                    // println!("V. From psi: {}, To match: {}", V_from_psi, V_to_match);
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

    (bases, E)
}
