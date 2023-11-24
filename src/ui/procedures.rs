//! This module contains code called by the GUI that modifies state (eg wave functions).
//! Components here may be called in one or more places.

use graphics::{EngineUpdates, Scene};

use crate::{
    basis_finder,
    eigen_fns,
    elec_elec::{PositIndex, WaveFunctionMultiElec},
    grid_setup::{new_data, Arr3dReal},
    potential,
    render,
    // types::BasesEvaluated,
    types::SurfacesPerElec,
    wf_ops,
    ActiveElec,
    State,
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
        &sfcs.psi_pp_evaluated,
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
    let psi_pp = &mut sfcs.psi_pp_evaluated;

    wf_ops::mix_bases(
        psi,
        Some(psi_pp),
        &sfcs.psi_per_basis,
        Some(&sfcs.psi_pp_per_basis),
        state.grid_n_render,
        &weights,
    );

    wf_ops::update_eigen_vals(
        &mut sfcs.V_elec_eigen,
        &mut sfcs.V_total_eigen,
        &mut sfcs.psi_pp_calculated,
        &sfcs.psi,
        &sfcs.psi_pp_evaluated,
        &sfcs.V_acting_on_this,
        state.surfaces_shared.E,
        &state.surfaces_shared.V_from_nuclei,
    );

    // For now, we are setting the V elec that must be acting on this WF if it were to be valid.
    wf_ops::calculate_v_elec(
        &mut sfcs.V_elec_eigen,
        &mut sfcs.V_total_eigen,
        &sfcs.psi,
        &sfcs.psi_pp_evaluated,
        state.surfaces_shared.E,
        &state.surfaces_shared.V_from_nuclei,
    );

    if state.ui.auto_gen_elec_V {
        let mut psi_charge_grid = new_data(state.grid_n_charge);

        wf_ops::mix_bases(
            &mut psi_charge_grid,
            None,
            &state.psi_charge[ae],
            None,
            state.grid_n_charge,
            &weights,
        );

        wf_ops::charge_from_psi(
            &mut state.charges_electron[ae],
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
    let psi_pp = &mut sfcs.psi_pp_per_basis;

    wf_ops::update_wf_from_bases(
        &state.dev,
        psi,
        Some(psi_pp),
        &state.bases[ae],
        &state.surfaces_shared.grid_posits,
        state.grid_n_render,
    );

    wf_ops::update_wf_from_bases(
        &state.dev,
        &mut state.psi_charge[ae],
        None,
        &state.bases[ae],
        &state.surfaces_shared.grid_posits_charge,
        state.grid_n_charge,
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
            &state.V_from_elecs,
            state.grid_n_render,
        );
    }

    // Update sphere entity locations.
    render::update_entities(&state.charges_fixed, &state.surface_data, scene);
}

/// Create the electric charge from a single electron's wave function squared. Note that this
/// is a combination of mixing bases to get the wave function, and generating charge from this.
pub fn create_elec_charge(state: &mut State, ae: usize) {
    let mut psi_charge_grid = new_data(state.grid_n_charge);

    let weights: Vec<f64> = state.bases[ae].iter().map(|b| b.weight()).collect();
    wf_ops::mix_bases(
        &mut psi_charge_grid,
        None,
        &state.psi_charge[ae],
        None,
        state.grid_n_charge,
        &weights,
    );

    wf_ops::charge_from_psi(
        &mut state.charges_electron[ae],
        &psi_charge_grid,
        state.grid_n_charge,
    );
}

pub(crate) fn update_V_acting_on_elec(state: &mut State, ae: usize) {
    if state.ui.create_3d_electron_V || state.ui.create_2d_electron_V {
        // First, create V based on electron charge.
        let charges_other_elecs =
            wf_ops::combine_electron_charges(ae, &state.charges_electron, state.grid_n_charge);

        potential::create_V_from_elecs(
            &state.dev,
            &mut state.V_from_elecs[ae],
            &state.surfaces_shared.grid_posits,
            &state.surfaces_shared.grid_posits_charge,
            &charges_other_elecs,
            state.grid_n_render,
            state.grid_n_charge,
            state.ui.create_2d_electron_V,
            ae,
        );

        potential::update_V_acting_on_elec(
            &mut state.surfaces_per_elec[ae].V_acting_on_this,
            &state.surfaces_shared.V_from_nuclei,
            &state.V_from_elecs,
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
        &state.charges_electron[active_elec],
        state.grid_n_render,
        render_multi_elec,
    );
}

/// Each loop run, make sure we are only updating things relevant for these calculations.
/// Notably, we need to update our 3D charge grid using the central dogma, but not the 3D sample grid.
///
/// We can use this to assist in general refactoring of our fundamental operations.
pub(crate) fn he_solver(state: &mut State) {
    for i in 0..8 {
        let elec_id = i % 2;

        let charges_other_elecs =
            wf_ops::combine_electron_charges(elec_id, &state.charges_electron, state.grid_n_charge);

        let sample_pts = basis_finder::generate_sample_pts();
        let xis: Vec<f64> = state.bases[elec_id].iter().map(|b| b.xi()).collect();

        let (bases, E) = basis_finder::run(
            &state.dev,
            &state.charges_fixed,
            &charges_other_elecs,
            &state.surfaces_shared.grid_posits_charge,
            state.grid_n_charge,
            &sample_pts,
            &xis,
        );

        state.surfaces_shared.E = E;
        state.bases[elec_id] = bases;
        state.ui.active_elec = ActiveElec::PerElec(elec_id);

        // todo: Consider combining these 2 things: evaluating the WF at each basis, and
        // todo mixing the bases. The clincher is how to handle normalization. Maybe normalize the charge
        // todo density grid after the fact?
        // wf_ops::create_psi_from_bases_mix_update_charge_density(
        //     &state.dev,
        //     &mut state.charges_electron[elec_id],
        //     &state.bases[elec_id],
        //     &state.surfaces_shared.grid_posits_charge,
        //     state.grid_n_charge,
        // );

        wf_ops::update_wf_from_bases(
            &state.dev,
            &mut state.psi_charge[elec_id],
            None,
            &state.bases[elec_id],
            &state.surfaces_shared.grid_posits_charge,
            state.grid_n_charge,
        );

        let mut psi_charge_grid = new_data(state.grid_n_charge);
        let weights: Vec<f64> = state.bases[elec_id].iter().map(|b| b.weight()).collect();

        wf_ops::mix_bases_update_charge_density(
            &mut psi_charge_grid,
            &mut state.charges_electron[elec_id],
            &state.psi_charge[elec_id],
            state.grid_n_charge,
            &weights,
        );
    }

    // Update the 2D or 3D V grids once, at the end.
    create_elec_charge(state, 0);
    create_elec_charge(state, 1);
}
