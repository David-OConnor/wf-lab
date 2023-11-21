//! This module contains code called by the GUI that modifies state (eg wave functions).
//! Components here may be called in one or more places.

use graphics::{EngineUpdates, Scene};

use crate::{
    basis_finder, eigen_fns,
    elec_elec::{PositIndex, WaveFunctionMultiElec},
    grid_setup::{new_data, Arr3dReal},
    potential, render,
    types::BasesEvaluated,
    types::SurfacesPerElec,
    wf_ops, ActiveElec, State,
};

pub fn update_E_or_V(
    sfcs: &mut SurfacesPerElec,
    V_from_nuclei: &Arr3dReal,
    grid_n_render: usize,
    E: f64,
) {
    for i in 0..grid_n_render {
        for j in 0..grid_n_render {
            for k in 0..grid_n_render {
                sfcs.psi_pp_calculated[i][j][k] = eigen_fns::find_ψ_pp_calc(
                    sfcs.psi.on_pt[i][j][k],
                    sfcs.V_acting_on_this[i][j][k],
                    E,
                )
            }
        }
    }

    // For now, we are setting the V elec that must be acting on this WF if it were to be valid.
    wf_ops::calculate_v_elec(
        &mut sfcs.aux1,
        &mut sfcs.aux2,
        &sfcs.psi.on_pt,
        &sfcs.psi_pp_evaluated,
        E,
        V_from_nuclei,
    );
}

/// Set up our basis-function based trial wave function.
pub fn update_basis_weights(state: &mut State, ae: usize) {
    let weights: Vec<f64> = state.bases[ae].iter().map(|b| b.weight()).collect();

    wf_ops::update_wf_fm_bases(
        &mut state.surfaces_per_elec[ae],
        &state.bases_evaluated[ae],
        state.surfaces_shared.E,
        state.grid_n_render,
        &weights,
    );

    // For now, we are setting the V elec that must be acting on this WF if it were to be valid.
    let sfcs = &mut state.surfaces_per_elec[ae];
    wf_ops::calculate_v_elec(
        &mut sfcs.aux1,
        &mut sfcs.aux2,
        &sfcs.psi.on_pt,
        &sfcs.psi_pp_evaluated,
        state.surfaces_shared.E,
        &state.surfaces_shared.V_from_nuclei,
    );
}

/// Run this when we add bases, change basis parameters other than weight etc.
pub fn update_evaluated_wfs(state: &mut State, ae: usize) {
    state.bases_evaluated[ae] = BasesEvaluated::initialize_with_psi(
        &state.dev,
        &state.bases[ae],
        &state.surfaces_shared.grid_posits,
        state.grid_n_render,
    );

    wf_ops::create_psi_from_bases(
        &state.dev,
        &mut state.bases_evaluated_charge[ae],
        None,
        &state.bases[ae],
        &state.surfaces_shared.grid_posits_charge,
        state.grid_n_charge,
    );
}

pub fn update_fixed_charges(state: &mut State) {
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
            &state.charges_fixed,
            &mut state.bases[elec_i],
            // Some(&mut state.bases_visible[elec_i]),
            2,
        );

        potential::update_V_acting_on_elec(
            &mut state.surfaces_per_elec[elec_i].V_acting_on_this,
            &state.surfaces_shared.V_from_nuclei,
            &state.V_from_elecs,
            elec_i,
            state.grid_n_render,
        );
    }
}

pub fn create_V_from_elec(state: &mut State, ae: usize) {
    let mut psi_charge_grid = new_data(state.grid_n_charge);

    let weights: Vec<f64> = state.bases[ae].iter().map(|b| b.weight()).collect();
    wf_ops::mix_bases(
        &mut psi_charge_grid,
        &state.bases_evaluated_charge[ae],
        state.grid_n_charge,
        &weights,
    );

    wf_ops::update_charge_density_fm_psi(
        &mut state.charges_electron[ae],
        &psi_charge_grid,
        state.grid_n_charge,
    );

    if state.ui.create_3d_electron_V || state.ui.create_2d_electron_V {
        potential::create_V_from_elec(
            &state.dev,
            &mut state.V_from_elecs[ae],
            &state.surfaces_shared.grid_posits,
            &state.surfaces_shared.grid_posits_charge,
            &state.charges_electron[ae],
            state.grid_n_render,
            state.grid_n_charge,
            state.ui.create_2d_electron_V,
        );
    }
}

pub fn update_V_acting_on_elec(state: &mut State, scene: &mut Scene, ae: usize) {
    if state.ui.create_3d_electron_V || state.ui.create_2d_electron_V {
        potential::update_V_acting_on_elec(
            &mut state.surfaces_per_elec[ae].V_acting_on_this,
            &state.surfaces_shared.V_from_nuclei,
            &state.V_from_elecs,
            ae,
            state.grid_n_render,
        );
    }

    if state.ui.auto_gen_elec_V {
        // state.surfaces_shared.E = wf_ops::find_E(
        //     &mut state.eval_data_per_elec[ae],
        //     state.eval_data_shared.grid_n,
        // );
    }

    // todo: Kludge to update sphere entity locs; DRY
    match state.ui.active_elec {
        ActiveElec::PerElec(ae) => {
            render::update_entities(&state.charges_fixed, &state.surface_data, scene);
        }
        ActiveElec::Combined => (),
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

        wf_ops::create_psi_from_bases(
            &state.dev,
            &mut state.bases_evaluated_charge[elec_id],
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
            &state.bases_evaluated_charge[elec_id],
            state.grid_n_charge,
            &weights,
        );
    }

    // Update the 2D or 3D V grids once, at the end.
    create_V_from_elec(state, 0);
    create_V_from_elec(state, 1);
}
