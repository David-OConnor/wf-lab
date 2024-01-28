//! This module contains code called by the GUI that modifies state (eg wave functions).
//! Components here may be called in one or more places.

use graphics::{EngineUpdates, Scene};

use crate::{
    basis_finder,
    grid_setup::{new_data, Arr3d, Arr3dReal},
    iter_arr, potential, render,
    types::SurfacesPerElec,
    util, wf_ops, ActiveElec, State,
};

pub fn update_E_or_V(
    sfcs: &mut SurfacesPerElec,
    V_from_nuclei: &Arr3dReal,
    grid_n_render: usize,
    E: f64,
) {
    wf_ops::update_eigen_vals(
        &mut sfcs.V_elec_eigen,
        &mut sfcs.V_total_eigen,
        &mut sfcs.psi_pp_calculated,
        &sfcs.psi,
        &sfcs.derivs,
        &sfcs.psi_pp_div_psi_evaluated,
        &sfcs.V_acting_on_this,
        E,
        V_from_nuclei,
    );
}

/// Set up our basis-function based trial wave function.
pub fn update_basis_weights(state: &mut State, ae: usize) {
    let weights: Vec<f64> = state.bases[ae].iter().map(|b| b.weight()).collect();

    let sfcs = &mut state.surfaces_per_elec[ae];

    // Prevents double borrow-mut error
    let psi = &mut sfcs.psi;
    let charge_density = &mut sfcs.charge_density;
    let psi_pp = &mut sfcs.derivs;
    let psi_pp_div_psi = &mut sfcs.psi_pp_div_psi_evaluated;

    wf_ops::mix_bases(
        psi,
        Some(charge_density),
        Some(psi_pp),
        Some(psi_pp_div_psi),
        &sfcs.psi_per_basis,
        Some(&sfcs.derivs_per_basis),
        Some(&sfcs.psi_pp_div_psi_per_basis),
        state.grid_n_render,
        &weights,
        // Some(&mut state.surfaces_shared),
    );

    wf_ops::update_eigen_vals(
        &mut sfcs.V_elec_eigen,
        &mut sfcs.V_total_eigen,
        &mut sfcs.psi_pp_calculated,
        &sfcs.psi,
        &sfcs.derivs,
        &sfcs.psi_pp_div_psi_evaluated,
        &sfcs.V_acting_on_this,
        state.surfaces_shared.E,
        &state.surfaces_shared.V_from_nuclei,
    );

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

    if state.ui.auto_gen_elec_V {
        let mut psi_charge_grid = new_data(state.grid_n_charge);

        wf_ops::mix_bases(
            &mut psi_charge_grid,
            None,
            None,
            None,
            &state.psi_charge[ae],
            None,
            None,
            state.grid_n_charge,
            &weights,
            // None,
        );

        wf_ops::charge_from_psi(
            &mut state.charges_from_electron[ae],
            &psi_charge_grid,
            state.grid_n_charge,
        );
    }
}

/// Run this when we add bases, change basis parameters other than weight etc.
pub fn update_evaluated_wfs(state: &mut State, ae: usize) {
    let sfcs = &mut state.surfaces_per_elec[ae];

    // Prevents double borrow-mut error
    let psi = &mut sfcs.psi_per_basis;
    let psi_pp = &mut sfcs.derivs_per_basis;
    let psi_pp_div_psi = &mut sfcs.psi_pp_div_psi_per_basis;

    wf_ops::wf_from_bases(
        &state.dev_psi,
        psi,
        Some(psi_pp),
        Some(psi_pp_div_psi),
        &state.bases[ae],
        &state.surfaces_shared.grid_posits,
        state.grid_n_render,
        state.deriv_calc,
    );

    wf_ops::wf_from_bases(
        &state.dev_psi,
        &mut state.psi_charge[ae],
        None,
        None,
        &state.bases[ae],
        &state.surfaces_shared.grid_posits_charge,
        state.grid_n_charge,
        state.deriv_calc,
    );
}

pub fn update_fixed_charges(state: &mut State, scene: &mut Scene) {
    potential::update_V_from_nuclei(
        &mut state.surfaces_shared.V_from_nuclei,
        &state.charges_fixed,
        &state.surfaces_shared.grid_posits,
        state.grid_n_render,
    );

    // Reinintialize bases due to the added charges, since we initialize bases centered
    // on the charges.
    // Note: An alternative would be to add the new bases without 0ing the existing ones.
    for elec_i in 0..state.surfaces_per_elec.len() {
        wf_ops::initialize_bases(
            &mut state.bases[elec_i],
            &state.charges_fixed,
            state.max_basis_n,
        );

        potential::update_V_acting_on_elec(
            &mut state.surfaces_per_elec[elec_i].V_acting_on_this,
            &state.surfaces_shared.V_from_nuclei,
            &state.V_from_elecs[elec_i],
            state.grid_n_render,
        );
    }

    // Update sphere entity locations.
    render::update_entities(&state.charges_fixed, &state.surface_descs_per_elec, scene);
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

    wf_ops::mix_bases(
        &mut psi_charge_grid,
        None,
        None,
        None,
        psi_charge_per_basis,
        None,
        None,
        grid_n_charge,
        weights,
        // None,
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
            state.ui.create_2d_electron_V,
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
    );
}

/// Each loop run, make sure we are only updating things relevant for these calculations.
/// Notably, we need to update our 3D charge grid using the central dogma, but not the 3D sample grid.
///
/// We can use this to assist in general refactoring of our fundamental operations.
pub(crate) fn he_solver(state: &mut State) {
    let sample_pts = basis_finder::generate_sample_pts();

    for i in 0..3 {
        let elec_id = i % 2;

        let charges_other_elecs = wf_ops::combine_electron_charges(
            elec_id,
            &state.charges_from_electron,
            state.grid_n_charge,
        );

        // let xis: Vec<f64> = state.bases[elec_id].iter().map(|b| b.xi()).collect();

        let (bases, E) = basis_finder::run(
            &state.dev_charge,
            &state.charges_fixed,
            &charges_other_elecs,
            &state.surfaces_shared.grid_posits_charge,
            state.grid_n_charge,
            &sample_pts,
            // &xis,
            &state.bases[elec_id],
            state.deriv_calc,
        );

        state.surfaces_shared.E = E;
        state.bases[elec_id] = bases;
        state.ui.active_elec = ActiveElec::PerElec(elec_id);

        wf_ops::wf_from_bases(
            &state.dev_psi,
            &mut state.psi_charge[elec_id],
            None,
            None,
            &state.bases[elec_id],
            &state.surfaces_shared.grid_posits_charge,
            state.grid_n_charge,
            state.deriv_calc,
        );

        // We don't need to mix bases here, and that's handled at the end of the loop
        // by the `updated_weights` flag.

        // let mut psi_charge_grid = new_data(state.grid_n_charge);
        let weights: Vec<f64> = state.bases[elec_id].iter().map(|b| b.weight()).collect();

        create_elec_charge(
            &mut state.charges_from_electron[elec_id],
            &state.psi_charge[elec_id],
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

    // create_elec_charge(state, 0);
    // create_elec_charge(state, 1);
}
