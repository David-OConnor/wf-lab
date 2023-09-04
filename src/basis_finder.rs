//! Used to find weights and Xis for STOs.

use lin_alg2::f64::Vec3;

use crate::{
    basis_wfs::{Basis, Sto},
    complex_nums::Cplx,
    eigen_fns::{self, calc_E_on_psi, calc_V_on_psi},
    grid_setup::Arr3dReal,
    grid_setup::Arr3dVec,
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
    const XI_INITIAL: f64 = 1.;

    // Base STO is the one with the lowest xi.
    // let mut trial_base_sto = Basis::Sto(Sto {
    //     posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
    //     n: 1,
    //     xi: XI_INITIAL,
    //     weight: 1.,
    //     // todo: As-required. This is used to associate it's position with a nucleus's.
    //     // todo: If we have nuc position, what purpose does it servce.
    //     charge_id: 0,
    //     harmonic: Default::default(),
    // });

    // let psi_corner = trial_base_sto.value(posit_corner);
    // let psi_pp_corner = trial_base_sto.second_deriv(posit_corner);

    let mut best_xi_i = 0;
    let mut smallest_diff = 99999.;
    // This isn't perhaps an ideal apperoach, but try it to find the baseline xi.
    let trial_base_xis = util::linspace((1., 6.), 100);
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
                // todo: Wrong. YHou must normalize. Check out existing code you haeve.
                V_sample += potential::V_coulomb(grid_charge[i][j][k], posit_sample, Q_ELEC);
                V_corner += potential::V_coulomb(grid_charge[i][j][k], posit_corner, Q_ELEC);
            }
        }
    }

    find_base_xi_E_common(V_corner, posit_corner, V_sample, posit_sample)
}

/// Find a wave function, composed of STOs, that match a given potential.
// pub fn find_stos(V: &Arr3dReal, grid_posits: &Arr3dVec) -> (Vec<Basis>, f64) {
pub fn find_stos(
    charges_fixed: &[(Vec3, f64)],
    charge_elec: &Arr3dReal,
    grid_charge: &Arr3dVec,
) -> (Vec<Basis>, f64) {
    // let (base_xi, E) = find_base_xi_E(V, grid_posits);
    let (base_xi, E) = find_base_xi_E_type2(charges_fixed, charge_elec, grid_charge);
    println!("\nBase xi: {}. E: {}\n", base_xi, E);

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
    let weights = util::linspace((-1.3, 1.3), 200);

    // todo: Not ideal. Set up sample points at specific intervals, then evaluate
    // todo V at those intervals; don't use the grid.

    // tod: It's critical these sample points span points both close and far from the nuclei.

    let sample_dists = vec![0.1, 0.3, 0.8, 1., 1.5, 2., 3., 4., 8., 15.];
    let mut grid_posits = Vec::new();
    for dist in sample_dists {
        grid_posits.push(Vec3::new(dist, 0., 0.));
        grid_posits.push(Vec3::new(0., dist, 0.));
        grid_posits.push(Vec3::new(0., 0., dist));
    }

    // let sample_pts = vec![
    //     // grid_posits[20][20][20],
    //     grid_posits[14][14][14],
    //     grid_posits[13][14][12],
    //     grid_posits[12][12][12],
    //     grid_posits[10][10][10],
    //     grid_posits[9][9][9],
    //     grid_posits[8][8][8],
    //     grid_posits[7][7][7],
    // ];
    //
    // let V_samples = vec![
    //     // V[20][20][20],
    //     V[14][14][14],
    //     V[13][14][12],
    //     V[12][12][12],
    //     V[10][10][10],
    //     V[9][9][9],
    //     V[8][8][8],
    //     V[7][7][7],
    // ];

    let mut V_samples = Vec::new();

    for posit_sample in &grid_posits {
        // todo: Dry with above! Find a helper you alreayd have, or make one.
        let mut V_sample = 0.;

        for (posit_nuc, charge) in charges_fixed {
            // todo: For now, we assume nuclei are at 0.
            V_sample += potential::V_coulomb(*posit_nuc, *posit_sample, *charge);
        }

        for i in 0..charge_elec.len() {
            for j in 0..charge_elec.len() {
                for k in 0..charge_elec.len() {
                    // todo: Wrong. YHou must normalize. Check out existing code you haeve.
                    V_sample += potential::V_coulomb(grid_charge[i][j][k], *posit_sample, Q_ELEC);
                }
            }
        }
        V_samples.push(V_sample);
    }

    println!(
        "SAMPLE: {:?} {} {} {}",
        grid_posits[0].magnitude(),
        grid_posits[1].magnitude(),
        grid_posits[2].magnitude(),
        grid_posits[3].magnitude()
    );

    // todo: Check these sample pts for 0

    let mut bases = vec![base_sto.clone()];

    const EPS: f64 = 0.00000001;

    for xi in &additional_xis {
        if xi <= &base_xi {
            continue;
        }

        // Update this for each Xi, since we add a basis each time.else {     // Calculate the value from the previously-selected bases at the sample points.
        let mut psi_other_bases = Vec::new();
        let mut psi_pp_other_bases = Vec::new();

        for pt in &grid_posits {
            let mut psi = Cplx::new_zero();
            let mut psi_pp = Cplx::new_zero();

            // Prevents NaNs. Alternatively, don't use the origin as a point.
            // if pt.magnitude() < EPS {
            //     continue
            // }

            for basis in &bases {
                psi += Cplx::from_real(basis.weight()) * basis.value(*pt);
                psi_pp += Cplx::from_real(basis.weight()) * basis.second_deriv(*pt);
            }

            psi_other_bases.push(psi);
            psi_pp_other_bases.push(psi_pp);
        }

        let mut best_weight_i = 0;
        let mut smallest_diff = 9999.;

        // todo: Exerimental. Consider V from psi, and target V to be equivlant if they're within this
        // todo value of each other.
        const V_DIFF_THRESH: f64 = 0.0001;

        let mut best_match_count = 0;
        let mut best_weight_i = 0;

        for (i_weight, weight) in weights.iter().enumerate() {
            let sto = Basis::Sto(Sto {
                posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
                n: 1,
                xi: *xi,
                weight: *weight,
                charge_id: 0,
                harmonic: Default::default(),
            });

            let mut num_pts_close = 0;

            for (i_pt, pt) in grid_posits.iter().enumerate() {
                let psi = psi_other_bases[i_pt] + Cplx::from_real(*weight) * sto.value(*pt);

                let psi_pp =
                    psi_pp_other_bases[i_pt] + Cplx::from_real(*weight) * sto.second_deriv(*pt);

                // psi is a denominator; prevents nans.
                // todo: You could abort all pts on this weight.
                // if psi.abs_sq() < EPS {
                //     println!("V eps violation. xi: {}", xi);
                //     diff = 999999.;
                //     // continue;
                // }

                let V_from_psi = calc_V_on_psi(psi, psi_pp, E);

                // if V_from_psi.is_nan() {
                //     println!("NaN. xi: {}", xi);
                //     diff = 999999.;
                // }

                // // todo: experimenting with taking the weight that has the single lowest score.
                // let this_diff = (V_from_psi - V_samples[i_pt]).abs();
                // if this_diff < smallest_diff {
                //     best_weight_i = i_weight;
                //     smallest_diff = this_diff;
                // }

                let diff = (V_from_psi - V_samples[i_pt]).abs();
                if diff < V_DIFF_THRESH {
                    num_pts_close += 1;
                }
                //
                // if *xi > 1.9 && *xi < 2.1 {
                //     println!("Weight: {} V Psi: {} V sample: {}, diff: {}", weight, V_from_psi, V_samples[i_pt],
                //              (V_from_psi - V_samples[i_pt]).abs())
                // }
            }

            if num_pts_close > best_match_count {
                best_weight_i = i_weight;
                best_match_count = num_pts_close;
            }
        }

        let best_weight = weights[best_weight_i];
        bases.push(Basis::Sto(Sto {
            posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
            n: 1,
            xi: *xi,
            weight: best_weight,
            charge_id: 0,
            harmonic: Default::default(),
        }))
    }

    for basis in &bases {
        if let Basis::Sto(sto) = basis {
            println!("Xi: {}, weight: {}", sto.xi, sto.weight);
        }
    }

    (bases, E)
}
