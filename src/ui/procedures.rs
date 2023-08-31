//! This module contains code called by the GUI that modifies state (eg wave functions).
//! Components here may be called in one or more places.

use graphics::{EngineUpdates, Scene};

use crate::{
    eigen_fns,
    elec_elec::{self, PositIndex, WaveFunctionMultiElec},
    eval,
    grid_setup::{new_data, Arr3dReal},
    potential, render,
    types::{BasesEvaluated, BasesEvaluated1d},
    types::{EvalDataPerElec, SurfacesPerElec},
    wf_ops, ActiveElec, State,
};

pub fn update_E_or_V(
    eval_data: &mut EvalDataPerElec,
    sfcs: &mut SurfacesPerElec,
    V_from_nuclei: &Arr3dReal,
    grid_n_1d: usize,
    grid_n_render: usize,
) {
    for i in 0..grid_n_render {
        for j in 0..grid_n_render {
            for k in 0..grid_n_render {
                sfcs.psi_pp_calculated[i][j][k] = eigen_fns::find_ψ_pp_calc(
                    sfcs.psi.on_pt[i][j][k],
                    sfcs.V_acting_on_this[i][j][k],
                    eval_data.E,
                )
            }
        }
    }

    for i in 0..grid_n_1d {
        eval_data.psi_pp_calc[i] = eigen_fns::find_ψ_pp_calc(
            eval_data.psi.on_pt[i],
            eval_data.V_acting_on_this[i],
            eval_data.E,
        );
    }

    // todo: Not working for some things lik eV?
    eval_data.score = eval::score_wf_from_psi_pp(&eval_data.psi_pp_calc, &eval_data.psi_pp_meas);

    // For now, we are setting the V elec that must be acting on this WF if it were to be valid.
    wf_ops::calculate_v_elec(
        &mut sfcs.aux1,
        &mut sfcs.aux2,
        &sfcs.psi.on_pt,
        &sfcs.psi_pp_measured,
        eval_data.E,
        V_from_nuclei,
        grid_n_render,
    );
}

pub fn update_basis_weights(state: &mut State, ae: usize) {
    // let mut weights = vec![0.; state.bases[ae].len()];
    // // Syncing procedure pending a better API.
    // for (i, basis) in state.bases[ae].iter().enumerate() {
    //     weights[i] = basis.weight();
    // }

    // Set up our basis-function based trial wave function.
    let weights: Vec<f64> = state.bases[ae].iter().map(|b| b.weight()).collect();
    wf_ops::update_wf_fm_bases(
        &mut state.surfaces_per_elec[ae],
        &state.bases_evaluated[ae],
        state.eval_data_per_elec[ae].E,
        state.grid_n_render,
        &weights,
    );

    let E = if state.adjust_E_with_weights {
        None
    } else {
        Some(state.eval_data_per_elec[ae].E)
    };

    let weights: Vec<f64> = state.bases[ae].iter().map(|b| b.weight()).collect();
    wf_ops::update_wf_fm_bases_1d(
        &mut state.eval_data_per_elec[ae],
        &state.bases_evaluated_1d[ae],
        state.eval_data_shared.grid_n,
        &weights,
        E,
    );

    state.eval_data_per_elec[ae].score = eval::score_wf_from_psi_pp(
        &state.eval_data_per_elec[ae].psi_pp_calc,
        &state.eval_data_per_elec[ae].psi_pp_meas,
    );

    // For now, we are setting the V elec that must be acting on this WF if it were to be valid.
    let sfcs = &mut state.surfaces_per_elec[ae];
    wf_ops::calculate_v_elec(
        &mut sfcs.aux1,
        &mut sfcs.aux2,
        &sfcs.psi.on_pt,
        &sfcs.psi_pp_measured,
        state.eval_data_per_elec[ae].E,
        &state.surfaces_shared.V_from_nuclei,
        state.grid_n_render,
    );

    let E_from_V = wf_ops::E_from_trial(
        &state.bases[ae],
        state.surfaces_per_elec[ae].V_acting_on_this[0][0][0],
        state.surfaces_shared.grid_posits[0][0][0],
    );
    // println!("E from V: {}", E_from_V);
}

pub fn update_evaluated_wfs(state: &mut State, ae: usize) {
    state.bases_evaluated[ae] = BasesEvaluated::new(
        &state.bases[ae],
        &state.surfaces_shared.grid_posits,
        state.grid_n_render,
    );

    let norm = 1.; // todo temp!!!
    state.bases_evaluated_1d[ae] =
        BasesEvaluated1d::new(&state.bases[ae], &state.eval_data_shared.posits, norm);

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

    potential::update_V_from_nuclei_1d(
        &mut state.eval_data_shared.V_from_nuclei,
        &state.charges_fixed,
        &state.eval_data_shared.posits,
        state.eval_data_shared.grid_n,
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

        potential::update_V_acting_on_elec_1d(
            &mut state.eval_data_per_elec[elec_i].V_acting_on_this,
            &state.eval_data_shared.V_from_nuclei,
            &state.V_from_elecs_1d,
            elec_i,
            state.eval_data_shared.grid_n,
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

    elec_elec::update_charge_density_fm_psi(
        &mut state.charges_electron[ae],
        &psi_charge_grid,
        state.grid_n_charge,
    );

    // todo: Better appraoch?
    /*            let mut weights = Vec::new();
    for basis in &state.bases[ae] {
        weights.push(basis.weight());
    }*/

    potential::create_V_from_an_elec(
        &mut state.V_from_elecs_1d[ae],
        &state.charges_electron[ae],
        &state.eval_data_shared.posits,
        &state.surfaces_shared.grid_posits_charge,
        state.eval_data_shared.grid_n,
        state.grid_n_charge,
    );

    if state.create_3d_electron_V {
        potential::create_V_from_an_elec_grid(
            &mut state.V_from_elecs[ae],
            &state.charges_electron[ae],
            &state.surfaces_shared.grid_posits,
            &state.surfaces_shared.grid_posits_charge,
            state.grid_n_render,
            state.grid_n_charge,
        );
    }

    // todo: Kludge to update sphere entity locs; DRY
    match state.ui_active_elec {
        ActiveElec::PerElec(ae) => {
            render::update_entities(
                &state.charges_fixed,
                &state.surface_data,
                &state.eval_data_per_elec[ae].psi_pp_calc,
                &state.eval_data_per_elec[ae].psi_pp_meas,
                &state.eval_data_shared.posits,
                scene,
            );
        }
        ActiveElec::Combined => (),
    }
}

pub fn update_V_acting_on_elec(state: &mut State, scene: &mut Scene, ae: usize) {
    if state.create_3d_electron_V {
        potential::update_V_acting_on_elec(
            &mut state.surfaces_per_elec[ae].V_acting_on_this,
            &state.surfaces_shared.V_from_nuclei,
            &state.V_from_elecs,
            ae,
            state.grid_n_render,
        );
    }

    potential::update_V_acting_on_elec_1d(
        &mut state.eval_data_per_elec[ae].V_acting_on_this,
        &state.eval_data_shared.V_from_nuclei,
        &state.V_from_elecs_1d,
        ae,
        state.eval_data_shared.grid_n,
    );

    if state.auto_gen_elec_V {
        state.eval_data_per_elec[ae].E = wf_ops::find_E(
            &mut state.eval_data_per_elec[ae],
            state.eval_data_shared.grid_n,
        );
    }

    // todo: Kludge to update sphere entity locs; DRY
    match state.ui_active_elec {
        ActiveElec::PerElec(ae) => {
            render::update_entities(
                &state.charges_fixed,
                &state.surface_data,
                &state.eval_data_per_elec[ae].psi_pp_calc,
                &state.eval_data_per_elec[ae].psi_pp_meas,
                &state.eval_data_shared.posits,
                scene,
            );
        }
        ActiveElec::Combined => (),
    }
}

pub fn update_meshes(state: &mut State, scene: &mut Scene, engine_updates: &mut EngineUpdates) {
    engine_updates.meshes = true;

    let render_multi_elec = match state.ui_active_elec {
        ActiveElec::PerElec(_) => false,
        ActiveElec::Combined => true,
    };

    let active_elec = match state.ui_active_elec {
        ActiveElec::Combined => 0,
        ActiveElec::PerElec(v) => v,
    };

    render::update_meshes(
        &state.surfaces_shared,
        &state.surfaces_per_elec[active_elec],
        state.ui_z_displayed,
        scene,
        &state.surfaces_shared.grid_posits,
        state.mag_phase,
        &state.charges_electron[active_elec],
        state.grid_n_render,
        render_multi_elec,
    );
}

/// todo: Needs rework
pub fn combine_wfs(state: &mut State) {
    potential::update_V_combined(
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

                // code shortener
                // let psi = &state
                //     .surfaces_shared
                //     .psi
                //     .psi_joint;

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

                // let psi = state
                //     .surfaces_shared
                //     .psi
                //     .psi_joint
                //     .get(&(PositIndex::new(i, j, k), PositIndex::new(i1, j1, k1)))
                //     .unwrap()
                //     .on_pt;

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

    // state
    //     .surfaces_shared
    //     .psi
    //     .populate_psi_marginal(state.grid_n);
    //

    // // todo: June 25, 2023: Trying a diff approach, more like single-elec approach.
    // for i in 0..state.grid_n {
    //     for j in 0..state.grid_n {
    //         for k in 0..state.grid_n {
    //
    //         }
    //     }
    // }
}
