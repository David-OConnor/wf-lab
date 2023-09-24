//! This module contains code called by the GUI that modifies state (eg wave functions).
//! Components here may be called in one or more places.

use graphics::{EngineUpdates, Scene};

use crate::{
    eigen_fns,
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

    // todo: Not working for some things lik eV?
    // eval_data.score = eval::score_wf_from_psi_pp(&eval_data.psi_pp_calc, &eval_data.psi_pp_meas);

    // For now, we are setting the V elec that must be acting on this WF if it were to be valid.
    wf_ops::calculate_v_elec(
        &mut sfcs.aux1,
        &mut sfcs.aux2,
        &sfcs.psi.on_pt,
        &sfcs.psi_pp_measured,
        // eval_data.E,
        E,
        V_from_nuclei,
    );
}

pub fn update_basis_weights(state: &mut State, ae: usize) {
    // Set up our basis-function based trial wave function.
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
        &sfcs.psi_pp_measured,
        state.surfaces_shared.E,
        &state.surfaces_shared.V_from_nuclei,
    );
}

/// Run this when we add bases, change basis parameters other than weight etc.
pub fn update_evaluated_wfs(state: &mut State, ae: usize) {
    state.bases_evaluated[ae] = BasesEvaluated::new(
        &state.bases[ae],
        &state.surfaces_shared.grid_posits,
        state.grid_n_render,
    );

    state.bases_evaluated_charge[ae] = wf_ops::arr_from_bases(
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

pub fn create_V_from_elec(state: &mut State, scene: &mut Scene, ae: usize) {
    let mut psi_charge_grid = new_data(state.grid_n_charge);

    let weights: Vec<f64> = state.bases[ae].iter().map(|b| b.weight()).collect();
    wf_ops::mix_bases_no_diffs(
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
        potential::create_V_from_an_elec_grid(
            &mut state.V_from_elecs[ae],
            &state.charges_electron[ae],
            &state.surfaces_shared.grid_posits,
            &state.surfaces_shared.grid_posits_charge,
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

/// todo: Needs rework
pub fn _combine_wfs(state: &mut State) {
    potential::_update_V_combined(
        &mut state.surfaces_shared.V_total,
        &state.surfaces_shared.V_from_nuclei,
        &state.V_from_elecs,
        state.grid_n_render,
    );

    let mut per_elec_wfs = Vec::new();
    for sfc in &state.surfaces_per_elec {
        per_elec_wfs.push(&sfc.psi);
    }

    state
        .surfaces_shared
        .psi
        .setup_joint_wf(&per_elec_wfs, state.grid_n_render);

    // todo: COnsider a more DFT-like approach: Instead of reducing a WF to
    // todo individual electrons, consider varying E like you do for a single elec.
    // todo: Immediate obstacle: How to deal with V? Sum from all??
    // todo: Try retrofitting your single-electron setup with a modified V
    // todo that includes all elecs...

    wf_ops::update_psi_pps(
        &state.surfaces_shared.psi.psi_marginal,
        &state.surfaces_shared.V_total,
        &mut state.surfaces_shared.psi_pp_calculated,
        &mut state.surfaces_shared.psi_pp_measured,
        state.surfaces_shared.E,
        state.grid_n_render,
    );

    // Experiment to calc E.
    for i in 7..10 {
        for j in 7..10 {
            for k in 7..10 {
                let i1 = 9;
                let j1 = 10;
                let k1 = 11;

                let posit_0 = PositIndex::new(i, j, k);
                let posit_1 = PositIndex::new(i1, j1, k1);

                // Hold r1 constant, and differentiate r0
                let psi_pp_r0 = WaveFunctionMultiElec::calc_psi_pp(
                    &posit_0,
                    &posit_1,
                    &state.surfaces_per_elec[0].psi,
                    &state.surfaces_per_elec[1].psi,
                    0,
                );

                let psi_pp_r1 = WaveFunctionMultiElec::calc_psi_pp(
                    &posit_0,
                    &posit_1,
                    &state.surfaces_per_elec[0].psi,
                    &state.surfaces_per_elec[1].psi,
                    1,
                );

                // Naive HT product
                let psi = posit_0.index(&state.surfaces_per_elec[0].psi.on_pt)
                    * posit_1.index(&state.surfaces_per_elec[1].psi.on_pt);

                // todo: The neighboring vals as well.
                state.surfaces_shared.psi.psi_marginal.on_pt[i][j][k] = psi;

                let E = eigen_fns::find_E_2_elec_at_pt(
                    psi,
                    psi_pp_r0,
                    psi_pp_r1,
                    state.charges_fixed[0].0,
                    state.surfaces_shared.grid_posits[i][j][k],
                    state.surfaces_shared.grid_posits[i1][j1][k1],
                );

                println!("E: {:?}", E);
            }
        }
    }
}
