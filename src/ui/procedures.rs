//! This module contains code called by the GUI that modifies state (eg wave functions).
//! Components here may be called in one or more places.

use graphics::{EngineUpdates, Scene};

use crate::{
    basis_finder, basis_init,
    grid_setup::{new_data, Arr2dReal, Arr2dVec, Arr3d, Arr3dReal},
    potential, render,
    types::SurfacesPerElec,
    wf_ops, ActiveElec, State,
};

pub fn update_E_or_V(
    sfcs: &mut SurfacesPerElec,
    V_from_nuclei: &Arr2dReal,
    E: f64,
    grid_posits: &Arr2dVec,
) {
    wf_ops::update_eigen_vals(
        &mut sfcs.V_elec_eigen,
        &mut sfcs.V_total_eigen,
        &mut sfcs.V_diff,
        &mut sfcs.psi_pp_calculated,
        &sfcs.psi,
        &sfcs.derivs,
        &sfcs.V_acting_on_this,
        E,
        V_from_nuclei,
        grid_posits,
        &mut sfcs.psi_fm_H,
        &mut sfcs.psi_fm_L2,
        &mut sfcs.psi_fm_Lz,
    );

    // todo: Put back A/R
    // wf_ops::update_eigen_vals_spinor(
    //     &mut sfcs.spinor_calc,
    //     &sfcs.spinor_derivs,
    //     [E; 4], // todo temp
    //     V_from_nuclei,
    // );
}

/// Set up our basis-function based trial wave function.
pub fn update_basis_weights(state: &mut State, ae_: usize) {
    // If symmetry is enabled update weights for all electrons; not just the active one.
    let elecs_i = if state.ui.weight_symmetry {
        let mut r = Vec::new();
        for i in 0..state.num_elecs {
            r.push(i);
        }
        r
    } else {
        vec![ae_]
    };

    for ae in elecs_i {
        let weights: Vec<f64> = state.bases[ae].iter().map(|b| b.weight()).collect();

        let sfcs = &mut state.surfaces_per_elec[ae];

        // Prevents double borrow-mut error
        let psi = &mut sfcs.psi;
        let charge_density_2d = &mut sfcs.charge_density_2d;
        let psi_pp = &mut sfcs.derivs;
        let spinor = &mut sfcs.spinor;
        let spinor_derivs = &mut sfcs.spinor_derivs;

        wf_ops::mix_bases(
            psi,
            charge_density_2d,
            psi_pp,
            &sfcs.psi_per_basis,
            &sfcs.derivs_per_basis,
            &weights,
        );

        wf_ops::mix_bases_spinor(
            spinor,
            None, // todo
            Some(spinor_derivs),
            &sfcs.spinor_per_basis,
            Some(&sfcs.spinor_derivs_per_basis),
            &weights,
        );

        wf_ops::update_eigen_vals(
            &mut sfcs.V_elec_eigen,
            &mut sfcs.V_total_eigen,
            &mut sfcs.V_diff,
            &mut sfcs.psi_pp_calculated,
            &sfcs.psi,
            &sfcs.derivs,
            &sfcs.V_acting_on_this,
            sfcs.E,
            &state.surfaces_shared.V_from_nuclei,
            &state.surfaces_shared.grid_posits,
            &mut sfcs.psi_fm_H,
            &mut sfcs.psi_fm_L2,
            &mut sfcs.psi_fm_Lz,
        );
    }

    // todo: A/R
    // wf_ops::update_eigen_vals_spinor(
    //     &mut sfcs.spinor_calc,
    //     spinor_derivs,
    //     [-0.5; 4], // todo temp
    //     &sfcs.V_acting_on_this,
    // );

    // // For now, we are setting the V elec that must be acting on this WF if it were to be valid.
    // wf_ops::calculate_v_elec(
    //     &mut sfcs.V_elec_eigen,
    //     &mut sfcs.V_total_eigen,
    //     &sfcs.psi,
    //     &sfcs.psi_pp_evaluated,
    //     &sfcs.psi_pp_div_psi_evaluated,
    //     state.surfaces_shared.E,
    //     &state.surfaces_shared.V_from_nuclei,
    // );

    // let mut charge_density = &mut sfcs.charge_density;

    // if state.ui.auto_gen_elec_V {
    //     let mut psi_charge_grid = new_data(state.grid_n_charge);
    //
    //     wf_ops::mix_bases_charge(
    //         &mut psi_charge_grid,
    //         &mut charge_density,
    //         &state.psi_charge[ae],
    //         &weights,
    //     );
    //
    //     wf_ops::charge_from_psi(
    //         &mut state.charges_from_electron[ae],
    //         &psi_charge_grid,
    //         state.grid_n_charge,
    //     );
    // }

    // todo: Come back to: Broke after 2D refactor.
    // {
    //     // todo: Is this an appropriate place to test this visualization?
    //     let mut s_orbital = new_data(state.grid_n_render);
    //     let n = state.grid_n_render;
    //     let s_basis = Basis::Sto(Sto {
    //         posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
    //         n: 1,
    //         xi: 1.,
    //         weight: 1.,
    //         charge_id: 0,
    //         harmonic: Default::default(),
    //     });
    //
    //     let mut norm = 0.;
    //     for (i, j, k) in iter_arr!(n) {
    //         let posit = state.surfaces_shared.grid_posits[i][j][k];
    //         s_orbital[i][j][k] = s_basis.value(posit);
    //         util::add_to_norm(&mut norm, s_orbital[i][j][k]);
    //     }
    //
    //     util::normalize_arr(&mut s_orbital, norm);
    //
    //     for (i, j, k) in iter_arr!(n) {
    //         // println!("S: {}", s_orbital[i][j][k]);
    //         sfcs.orb_sub[i][j][k] = sfcs.psi[i][j][k] - s_orbital[i][j][k];
    //     }
    // }
}

/// Run this when we add bases, change basis parameters other than weight etc.
pub fn update_evaluated_wfs(state: &mut State, ae: usize) {
    let sfcs = &mut state.surfaces_per_elec[ae];

    // Prevents double borrow-mut error
    let psi = &mut sfcs.psi_per_basis;
    let psi_pp = &mut sfcs.derivs_per_basis;

    let spinor = &mut sfcs.spinor_per_basis;
    let spinor_derivs = &mut sfcs.spinor_derivs_per_basis;

    wf_ops::wf_from_bases(
        &state.dev_psi,
        psi,
        psi_pp,
        &state.bases[ae],
        &state.surfaces_shared.grid_posits,
        state.deriv_calc,
    );

    // todo: A/R
    // wf_ops::wf_from_bases_spinor(
    //     &state.dev_psi,
    //     spinor,
    //     Some(spinor_derivs),
    //     &state.bases_spinor[ae],
    //     &state.surfaces_shared.grid_posits,
    // );

    // Note: This is a heavy operation, because it's 3D, while our other surfaces are 2D.
    wf_ops::wf_from_bases_charge(
        &state.dev_psi,
        &mut state.psi_charge[ae],
        &state.bases[ae],
        &state.surfaces_shared.grid_posits_charge,
    );
}

pub fn update_fixed_charges(state: &mut State, scene: &mut Scene) {
    potential::update_V_from_nuclei(
        &mut state.surfaces_shared.V_from_nuclei,
        &state.nucleii,
        &state.surfaces_shared.grid_posits,
    );

    // Reinintialize bases due to the added charges, since we initialize bases centered
    // on the charges.
    // Note: An alternative would be to add the new bases without 0ing the existing ones.
    for elec_i in 0..state.surfaces_per_elec.len() {
        // todo: Kludge for Li
        let n = if elec_i > 1 { 2 } else { 1 };
        basis_init::initialize_bases(&mut state.bases[elec_i], &state.nucleii, n);

        potential::update_V_acting_on_elec(
            &mut state.surfaces_per_elec[elec_i].V_acting_on_this,
            &state.surfaces_shared.V_from_nuclei,
            &state.V_from_elecs[elec_i],
            state.grid_n_render,
        );
    }

    // Update sphere entity locations.
    render::update_entities(
        &state.nucleii,
        &state.surface_descs_per_elec,
        scene,
        &state.charge_density_balls,
    );
}

/// Create the electric charge from a single electron's wave function squared. Note that this
/// is a combination of mixing bases to get the wave function, and generating charge from this.
pub fn create_elec_charge(
    charge_electron: &mut Arr3dReal,
    psi_charge_per_basis: &[Arr3d],
    weights: &[f64],
    grid_n_charge: usize,
) {
    let mut psi_charge_grid = new_data(grid_n_charge);

    wf_ops::mix_bases_charge(
        &mut psi_charge_grid,
        charge_electron,
        psi_charge_per_basis,
        weights,
    );

    wf_ops::charge_from_psi(charge_electron, &psi_charge_grid, grid_n_charge);
}

pub(crate) fn update_V_acting_on_elec(state: &mut State, ae: usize) {
    if state.ui.create_3d_electron_V || state.ui.create_2d_electron_V {
        // First, create V based on electron charge.
        let charges_other_elecs =
            wf_ops::combine_electron_charges(ae, &state.charges_from_electron, state.grid_n_charge);

        potential::create_V_from_elecs(
            &state.dev_charge,
            &mut state.V_from_elecs[ae],
            &state.surfaces_shared.grid_posits,
            &state.surfaces_shared.grid_posits_charge,
            &charges_other_elecs,
            state.grid_n_render,
            state.grid_n_charge,
            // state.ui.create_2d_electron_V,
            // state.ui.z_displayed,
        );

        potential::update_V_acting_on_elec(
            &mut state.surfaces_per_elec[ae].V_acting_on_this,
            &state.surfaces_shared.V_from_nuclei,
            &state.V_from_elecs[ae],
            state.grid_n_render,
        );
    }
}

pub fn update_meshes(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    engine_updates.meshes = true;

    let render_multi_elec = match state.ui.active_elec {
        ActiveElec::PerElec(_) => false,
        ActiveElec::Combined => true,
    };

    let active_elec = match state.ui.active_elec {
        ActiveElec::Combined => 0,
        ActiveElec::PerElec(v) => v,
    };

    render::update_meshes(
        &state.surfaces_shared,
        &state.surfaces_per_elec[active_elec],
        state.ui.z_displayed,
        scene,
        &state.surfaces_shared.grid_posits,
        state.ui.mag_phase,
        &state.charges_from_electron[active_elec],
        state.grid_n_render,
        render_multi_elec,
        &state.surface_descs_per_elec,
        &state.surface_descs_combined,
        state.ui.hidden_axis,
    );
}

/// Each loop run, make sure we are only updating things relevant for these calculations.
/// Notably, we need to update our 3D charge grid using the central dogma, but not the 3D sample grid.
///
/// We can use this to assist in general refactoring of our fundamental operations.
pub(crate) fn he_solver(state: &mut State) {
    let sample_pts = basis_finder::generate_sample_pts();

    for i in 0..3 {
        let ae = i % 2;

        let charges_other_elecs =
            wf_ops::combine_electron_charges(ae, &state.charges_from_electron, state.grid_n_charge);

        // let xis: Vec<f64> = state.bases[elec_id].iter().map(|b| b.xi()).collect();

        let (bases, E) = basis_finder::run(
            &state.dev_charge,
            &state.nucleii,
            &charges_other_elecs,
            &state.surfaces_shared.grid_posits_charge,
            &sample_pts,
            &state.bases[ae],
            state.deriv_calc,
        );

        state.surfaces_per_elec[ae].E = E;
        state.bases[ae] = bases;
        state.ui.active_elec = ActiveElec::PerElec(ae);

        wf_ops::wf_from_bases_charge(
            &state.dev_psi,
            &mut state.psi_charge[ae],
            &state.bases[ae],
            &state.surfaces_shared.grid_posits_charge,
        );

        // We don't need to mix bases here, and that's handled at the end of the loop
        // by the `updated_weights` flag.

        // let mut psi_charge_grid = new_data(state.grid_n_charge);
        let weights: Vec<f64> = state.bases[ae].iter().map(|b| b.weight()).collect();

        create_elec_charge(
            &mut state.charges_from_electron[ae],
            &state.psi_charge[ae],
            &weights,
            state.grid_n_charge,
        );
    }

    // Update the 2D or 3D V grids once, at the end.
    for e_id in 0..2 {
        let weights: Vec<f64> = state.bases[e_id].iter().map(|b| b.weight()).collect();
        create_elec_charge(
            &mut state.charges_from_electron[e_id],
            &state.psi_charge[e_id],
            &weights,
            state.grid_n_charge,
        );
    }
}
