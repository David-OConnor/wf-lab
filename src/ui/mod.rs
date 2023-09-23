use std::f64::consts::TAU;

use egui::{self, Color32, RichText, Ui};
use graphics::{EngineUpdates, Scene};
use lin_alg2::f64::Vec3;

use crate::grid_setup::new_data_real;
use crate::{
    basis_finder, basis_wfs::Basis, eigen_fns, elec_elec, eval, grid_setup::new_data, potential,
    render, wf_ops, ActiveElec, State,
};

pub(crate) mod procedures;

const UI_WIDTH: f32 = 300.;
const SIDE_PANEL_SIZE: f32 = 400.;
const SLIDER_WIDTH: f32 = 260.;
const SLIDER_WIDTH_ORIENTATION: f32 = 100.;

const E_MIN: f64 = -4.5;
const E_MAX: f64 = 0.2;

// Wave fn weights
pub const WEIGHT_MIN: f64 = -1.;
pub const WEIGHT_MAX: f64 = 1.9;

const _L_MIN: f64 = -3.;
const _L_MAX: f64 = 3.;

// sets range of -size to +size
const GRID_SIZE_MIN: f64 = 0.;
const GRID_SIZE_MAX: f64 = 40.;

const ITEM_SPACING: f32 = 18.;
const FLOAT_EDIT_WIDTH: f32 = 40.;

const NUDGE_MIN: f64 = 0.;
const NUDGE_MAX: f64 = 0.2;

fn text_edit_float(val: &mut f64, _default: f64, ui: &mut Ui) {
    let mut entry = val.to_string();

    let response = ui.add(egui::TextEdit::singleline(&mut entry).desired_width(FLOAT_EDIT_WIDTH));
    if response.changed() {
        *val = entry.parse::<f64>().unwrap_or(0.);
    }
}

// /// Create a slider to adjust E. Note that this is a shared fn due to repetition between per-elec,
// /// and combined.
// /// todo: Not feasible to use this due to borrow-checker issues with mixed mutability of structf fields;
// /// todo so, we repeat ourselves between shared and per-elec C.
// fn _E_slider(
//     ui: &mut Ui,
//     E: &mut f64,
//     psi: &Arr3d,
//     V: &Arr3dReal,
//     psi_pp_calc: &mut Arr3d,
//     psi_pp_meas: &Arr3d,
//     // psi_pp_calc: &mut [Cplx],
//     // psi_pp_meas: &mut [Cplx],
//     score: &mut f64,
//     grid_n: usize,
//     updated_meshes: &mut bool,
// ) {
//     ui.add(
//         egui::Slider::from_get_set(E_MIN..=E_MAX, |v| {
//             if let Some(v_) = v {
//                 *E = v_;
//
//                 // wf_ops::update_psi_pp_calc_grid(
//                 //     psi_pp_calc,
//                 //     psi,
//                 //     V,
//                 //     *E,
//                 //     grid_n,
//                 // );
//
//                 // *score = eval::score_wf(psi_pp_calc, psi_pp_meas, grid_n);
//                 // *score = eval::score_wf(psi_pp_calc, psi_pp_meas);
//                 *score = 0.; // todo t
//
//                 // state.psi_p_score[active_elec] = 0.; // todo!
//
//                 *updated_meshes = true;
//             }
//
//             *E
//         })
//             .text("E"),
//     );
// }

/// Ui elements that allow adding, removing, and changing the point
/// charges that form our potential.
fn charge_editor(
    charges: &mut Vec<(Vec3, f64)>,
    basis_fns: &mut [Basis],
    updated_unweighted_basis_wfs: &mut bool,
    updated_basis_weights: &mut bool,
    updated_charges: &mut bool,
    updated_entities: &mut bool,
    ui: &mut Ui,
) {
    let mut charge_removed = None;

    for (i, (posit, val)) in charges.iter_mut().enumerate() {
        // We store prev posit so we can know to update entities
        // when charge posit changes.
        let prev_posit = *posit;
        let prev_charge = *val;

        ui.horizontal(|ui| {
            text_edit_float(&mut posit.x, 0., ui);
            text_edit_float(&mut posit.y, 0., ui);
            text_edit_float(&mut posit.z, 0., ui);

            if prev_posit != *posit {
                *updated_unweighted_basis_wfs = true;
                *updated_basis_weights = true;
                *updated_charges = true;
                *updated_entities = true;

                // Updated basis centers based on the updated charge positions.
                for basis in basis_fns.iter_mut() {
                    if basis.charge_id() == i {
                        *basis.posit_mut() = *posit;
                    }
                }
            }

            ui.add_space(20.);
            text_edit_float(val, crate::Q_PROT, ui);

            if prev_charge != *val {
                *updated_unweighted_basis_wfs = true;
                *updated_basis_weights = true;
                *updated_charges = true;
                *updated_entities = true;
            }

            if ui.button(RichText::new("❌").color(Color32::RED)).clicked() {
                // Don't remove from a collection we're iterating over.
                charge_removed = Some(i);
                *updated_unweighted_basis_wfs = true;
                *updated_basis_weights = true;
                *updated_charges = true;
                // Update entities due to charge sphere placement.
                *updated_entities = true;
            }
        });
    }

    if let Some(charge_i_removed) = charge_removed {
        charges.remove(charge_i_removed);
        *updated_unweighted_basis_wfs = true;
        *updated_basis_weights = true;
        *updated_charges = true;
        *updated_entities = true;
    }

    if ui.add(egui::Button::new("Add charge")).clicked() {
        charges.push((Vec3::new_zero(), crate::Q_PROT));
        *updated_unweighted_basis_wfs = true;
        *updated_basis_weights = true;
        *updated_charges = true;
        *updated_entities = true;
    }
}

/// Ui elements that allow mixing various basis WFs.
fn basis_fn_mixer(
    state: &mut State,
    updated_basis_weights: &mut bool,
    updated_unweighted_basis_wfs: &mut bool,
    ui: &mut Ui,
    active_elec: usize,
) {
    // Select with charge (and its position) this basis fn is associated with.

    egui::containers::ScrollArea::vertical()
        .max_height(500.)
        .show(ui, |ui| {
            // We use this Vec to avoid double-mutable borrow issues.
            let mut bases_modified = Vec::new();

            let weights: Vec<f64> = state.bases[active_elec]
                .iter()
                .map(|b| b.weight())
                .collect();

            for (basis_i, basis) in state.bases[active_elec].iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    // Checkbox to immediately hide or show the basis.

                    // if ui
                    //     .checkbox(&mut state.bases_visible[active_elec][id], "")
                    //     .clicked()
                    // {
                    //     *updated_basis_weights = true;
                    // }

                    ui.spacing_mut().slider_width = SLIDER_WIDTH_ORIENTATION; // Only affects sliders in this section.

                    // `prev...` is to check if it changed below.
                    let prev_charge_id = basis.charge_id();

                    // Pair WFs with charge positions.
                    egui::ComboBox::from_id_source(basis_i + 1_000)
                        .width(30.)
                        .selected_text(basis.charge_id().to_string())
                        .show_ui(ui, |ui| {
                            for (charge_i, (_charge_posit, _amt)) in
                                state.charges_fixed.iter().enumerate()
                            {
                                ui.selectable_value(
                                    basis.charge_id_mut(),
                                    charge_i,
                                    charge_i.to_string(),
                                );
                            }
                        });

                    if basis.charge_id() != prev_charge_id {
                        *basis.posit_mut() = state.charges_fixed[basis.charge_id()].0;
                        *updated_basis_weights = true;
                        *updated_unweighted_basis_wfs = true;
                    }

                    // todo: If the basis is the general S0 type etc, replace the n, l etc sliders with
                    // todo chi and c, and make GO's implementation of n(), l() etc unimplemented!.

                    match basis {
                        Basis::H(_b) => {
                            // todo: Intead of the getter methods, maybe call the params directly?
                            let n_prev = basis.n();
                            let l_prev = basis.l();
                            let m_prev = basis.m();

                            // todo: Helper fn to reduce DRY here.
                            ui.heading("n:");
                            let mut entry = basis.n().to_string(); // angle
                            let response =
                                ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
                            if response.changed() {
                                *basis.n_mut() = entry.parse().unwrap_or(1);
                            }

                            ui.heading("l:");
                            let mut entry = basis.l().to_string(); // angle
                            let response =
                                ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
                            if response.changed() {
                                *basis.l_mut() = entry.parse().unwrap_or(0);
                            }

                            ui.heading("m:");
                            let mut entry = basis.m().to_string(); // angle
                            let response =
                                ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
                            if response.changed() {
                                *basis.m_mut() = entry.parse().unwrap_or(0);
                            }

                            if basis.n() != n_prev || basis.l() != l_prev || basis.m() != m_prev {
                                // Enforce quantum number constraints.
                                if basis.l() >= basis.n() {
                                    *basis.l_mut() = basis.n() - 1;
                                }
                                if basis.m() < -(basis.l() as i16) {
                                    *basis.m_mut() = -(basis.l() as i16)
                                } else if basis.m() > basis.l() as i16 {
                                    *basis.m_mut() = basis.l() as i16
                                }

                                *updated_unweighted_basis_wfs = true;
                                *updated_basis_weights = true;
                            }
                        }
                        Basis::Gto(b) => {
                            ui.heading("α:");
                            let mut entry = b.alpha.to_string(); // angle
                            let response =
                                ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
                            if response.changed() {
                                b.alpha = entry.parse().unwrap_or(1.);
                            }
                        } // Basis::Sto(_b) => (),
                        Basis::Sto(b) => {
                            let n_prev = b.n;
                            let l_prev = b.harmonic.l;
                            let m_prev = b.harmonic.m;
                            let xi_prev = b.xi;
                            // ui.heading("c:");
                            // let mut entry = b.c.to_string(); // angle
                            // let response =
                            //     ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
                            // if response.changed() {
                            //     b.c = entry.parse().unwrap_or(1.);
                            // }
                            ui.heading("n:");
                            let mut entry = b.n.to_string(); // angle
                            let response =
                                ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
                            if response.changed() {
                                b.n = entry.parse().unwrap_or(1);
                            }

                            ui.heading("l:");
                            let mut entry = b.harmonic.l.to_string(); // angle
                            let response =
                                ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
                            if response.changed() {
                                b.harmonic.l = entry.parse().unwrap_or(0);
                            }

                            ui.heading("m:");
                            let mut entry = b.harmonic.m.to_string(); // angle
                            let response =
                                ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
                            if response.changed() {
                                b.harmonic.m = entry.parse().unwrap_or(0);
                            }

                            ui.heading("ξ:");
                            // todo: It's easier to use ints.
                            const XI_INT_FACTOR: f64 = 100.;
                            let mut val = (b.xi * XI_INT_FACTOR) as u32; // angle
                            let mut entry = val.to_string();
                            let response = ui.add(
                                egui::TextEdit::singleline(&mut entry)
                                    .desired_width(FLOAT_EDIT_WIDTH),
                            );
                            if response.changed() {
                                val = entry.parse().unwrap_or(0);
                                b.xi = (val as f64) / XI_INT_FACTOR;
                            }

                            // todo: DRY with H
                            if b.n != n_prev || b.harmonic.l != l_prev || b.harmonic.m != m_prev {
                                // Enforce quantum number constraints.
                                if b.harmonic.l >= b.n {
                                    b.harmonic.l = b.n - 1;
                                }
                                if b.harmonic.m < -(b.harmonic.l as i16) {
                                    b.harmonic.m = -(b.harmonic.l as i16)
                                } else if b.harmonic.m > b.harmonic.l as i16 {
                                    b.harmonic.m = b.harmonic.l as i16
                                }

                                *updated_unweighted_basis_wfs = true;
                                *updated_basis_weights = true;
                            }

                            if b.xi != xi_prev {
                                *updated_unweighted_basis_wfs = true;
                                *updated_basis_weights = true;
                            }
                        }
                    }

                    // Note: We've replaced the below rotation-slider code with just using combinations of
                    // different m
                    // For now, we use an azimuth, elevation API for orientation.
                    //     if basis.l() >= 1 && basis.weight().abs() > 0.00001 {
                    //         let mut euler = basis.harmonic().orientation.to_euler();

                    //         // todo: DRY between the 3.
                    //         ui.add(
                    //             // Offsets are to avoid gimball lock.
                    //             egui::Slider::from_get_set(-TAU / 4.0 + 0.001..=TAU / 4.0 - 0.001, |v| {
                    //                 if let Some(v_) = v {
                    //                     euler.pitch = v_;
                    //                     basis.harmonic_mut().orientation = Quaternion::from_euler(&euler);
                    //                     *updated_wfs = true;
                    //                 }

                    //                 euler.pitch
                    //             })
                    //             .text("P"),
                    //         );
                    //         ui.add(
                    //             egui::Slider::from_get_set(-TAU / 2.0..=TAU / 2.0, |v| {
                    //                 if let Some(v_) = v {
                    //                     euler.roll = v_;
                    //                     basis.harmonic_mut().orientation = Quaternion::from_euler(&euler);
                    //                     *updated_wfs = true;
                    //                 }

                    //                 euler.roll
                    //             })
                    //             .text("R"),
                    //         );
                    //         ui.add(
                    //             egui::Slider::from_get_set(0.0..=TAU, |v| {
                    //                 if let Some(v_) = v {
                    //                     euler.yaw = v_;
                    //                     basis.harmonic_mut().orientation = Quaternion::from_euler(&euler);
                    //                     *updated_wfs = true;
                    //                 }

                    //                 euler.yaw
                    //             })
                    //             .text("Y"),
                    //         );
                    //     }
                });

                // todo: Text edit or dropdown for n.

                ui.add(
                    egui::Slider::from_get_set(WEIGHT_MIN..=WEIGHT_MAX, |v| {
                        if let Some(v_) = v {
                            *basis.weight_mut() = v_;
                            *updated_basis_weights = true;

                            bases_modified.push(basis_i);

                            if state.auto_gen_elec_V {
                                // todo: DRY (C+P) from `procedures` ideally use this:
                                // todo procedures::create_V_from_elec(state, active_elec);
                                // todo, but borrow checker issues when borrowing all of state

                                // todo: It may be advantagerous to cache `psi_charge_grid` somewwhere,
                                // todo eg in SurfacesPerElec.
                                let mut psi_charge_grid = new_data(state.grid_n_charge);

                                wf_ops::mix_bases_no_diffs(
                                    &mut psi_charge_grid,
                                    &state.bases_evaluated_charge[active_elec],
                                    state.grid_n_charge,
                                    &weights,
                                );

                                elec_elec::update_charge_density_fm_psi(
                                    &mut state.charges_electron[active_elec],
                                    &psi_charge_grid,
                                    state.grid_n_charge,
                                );

                                potential::create_V_from_an_elec(
                                    &mut state.V_from_elecs_1d[active_elec],
                                    &state.charges_electron[active_elec],
                                    &state.eval_data_shared.posits,
                                    &state.surfaces_shared.grid_posits_charge,
                                    state.eval_data_shared.grid_n,
                                    state.grid_n_charge,
                                );
                            }
                        }

                        basis.weight()
                    })
                    .text("Wt"),
                );
            }

            if state.weight_symmetry {
                for elec_i in 0..state.num_elecs {
                    if elec_i == active_elec {
                        continue;
                    }
                    for basis_i in &bases_modified {
                        *state.bases[elec_i][*basis_i].weight_mut() =
                            state.bases[active_elec][*basis_i].weight();
                        // *updated_basis_weights = true; // Make this update for all!

                        // todo: DRY-filled C+P from `if updated_basis_weights` below! Fix this with
                        // todo a better API

                        let ae = elec_i;

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

                        let weights: Vec<f64> =
                            state.bases[ae].iter().map(|b| b.weight()).collect();
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

                        // updated_meshes = true;
                    }
                }
            }
        });
}

/// Add buttons and other UI items at the bottom of the window.
fn bottom_items(
    ui: &mut Ui,
    state: &mut State,
    scene: &mut Scene,
    ae: usize,
    updated_meshes: &mut bool,
    updated_basis_weights: &mut bool,
    updated_E_or_V: &mut bool,
) {
    ui.horizontal(|ui| {
        if ui.add(egui::Button::new("Nudge WF")).clicked() {
            // crate::nudge::nudge_wf(
            //     &mut state.surfaces_per_elec[active_elec],
            //     &mut state.nudge_amount[active_elec],
            //     &state.bases[active_elec],
            //     &state.surfaces_shared.grid_posits,
            //     state.grid_n_render,
            // );

            *updated_meshes = true;

            // state.surfaces_per_elec[active_elec].psi_pp_score = eval::score_wf(
            //     &state.surfaces_per_elec[active_elec].psi_pp_calculated,
            //     &state.surfaces_per_elec[active_elec].psi_pp_measured,
            //     state.grid_n_render,
            // );

            state.eval_data_per_elec[ae].score = eval::score_wf_from_psi_pp(
                &state.eval_data_per_elec[ae].psi_pp_calc,
                &state.eval_data_per_elec[ae].psi_pp_meas,
            );
        }

        // if ui.add(egui::Button::new("Empty e- charge")).clicked() {
        //     state.charges_electron[active_elec] = grid_setup::new_data_real(state.grid_n);
        //
        //     *updated_meshes = true;
        // }

        if ui
            .add(egui::Button::new("Create V from this elec"))
            .clicked()
        {
            procedures::create_V_from_elec(state, scene, ae);
        }

        if ui
            .add(egui::Button::new("Update V acting on this elec"))
            .clicked()
        {
            procedures::update_V_acting_on_elec(state, scene, ae);

            *updated_meshes = true;
            *updated_E_or_V = true;
        }

        if ui.add(egui::Button::new("Find E")).clicked() {
            // state.eval_data_per_elec[ae].E = wf_ops::find_E(
            //     &mut state.eval_data_per_elec[ae],
            //     state.eval_data_shared.grid_n,
            // );

            state.eval_data_per_elec[ae].E = wf_ops::E_from_trial(
                &state.bases[ae],
                state.surfaces_per_elec[ae].V_acting_on_this[0][0][0],
                state.surfaces_shared.grid_posits[0][0][0],
            );

            *updated_E_or_V = true;
            *updated_meshes = true;
        }
    });

    // if ui.add(egui::Button::new("Find weights")).clicked() {
    //     basis_weight_finder::find_weights(
    //         &state.charges_fixed,
    //         &mut state.charges_electron,
    //         &mut state.eval_data_per_elec[ae],
    //         &state.eval_data_shared.posits,
    //         &mut state.bases[ae],
    //         &mut state.bases_evaluated_1d[ae],
    //         &mut state.bases_evaluated_charge[ae],
    //         state.max_basis_n,
    //         &state.eval_data_shared.V_from_nuclei,
    //         &mut state.V_from_elecs_1d[ae],
    //         state.grid_n_charge,
    //         &state.surfaces_shared.grid_posits_charge,
    //         state.eval_data_shared.grid_n,
    //     );

    //     *updated_E_or_V = true;
    //     *updated_basis_weights = true;
    //     *updated_meshes = true;
    // }

    if ui.add(egui::Button::new("Find STO bases")).clicked() {
        // todo: Place this somewhere? `potential module`?
        let mut charges_other_electrons = new_data_real(state.grid_n_charge);

        for (i_charge, charge_from_elec) in state.charges_electron.iter().enumerate() {
            if i_charge == ae {
                continue;
            }
            let mut sum = 0.; // todo confirming
            for i in 0..state.grid_n_charge {
                for j in 0..state.grid_n_charge {
                    for k in 0..state.grid_n_charge {
                        charges_other_electrons[i][j][k] += charge_from_elec[i][j][k];
                        sum += charge_from_elec[i][j][k];
                    }
                }
            }
            println!("Charge sum (should be -1): {:?}", sum);
        }

        let (bases, E) = basis_finder::find_stos(
            &state.charges_fixed,
            &charges_other_electrons,
            &state.surfaces_shared.grid_posits_charge,
            state.grid_n_charge,
        );

        // todo: Which of these are we using?
        state.eval_data_per_elec[ae].E = E;

        // todo: We'd need to re-calcualte the bases here if replacing them with non-defaults.
        // state.bases[ae] = bases;

        *updated_E_or_V = true;
        *updated_basis_weights = true;
        *updated_meshes = true;
    }
}

/// This function draws the (immediate-mode) GUI.
/// [UI items](https://docs.rs/egui/latest/egui/struct.Ui.html#method.heading)
pub fn ui_handler(state: &mut State, cx: &egui::Context, scene: &mut Scene) -> EngineUpdates {
    let mut engine_updates = EngineUpdates::default();

    let panel = egui::SidePanel::left(0) // ID must be unique among panels.
        .default_width(SIDE_PANEL_SIZE);

    panel.show(cx, |ui| {
        engine_updates.ui_size = ui.available_width();
        ui.spacing_mut().item_spacing = egui::vec2(10.0, 12.0);

        // todo: Wider values on larger windows?
        // todo: Delegate the numbers etc here to consts etc.
        ui.spacing_mut().slider_width = SLIDER_WIDTH;

        ui.set_max_width(UI_WIDTH);

        // ui.heading("Wavefunction Lab");

        // ui.heading("Show surfaces:");

        let mut updated_fixed_charges = false;
        let mut updated_evaluated_wfs = false;
        let mut updated_basis_weights = false;
        let mut updated_E_or_V = false;
        let mut updated_meshes = false;

        ui.horizontal(|ui| {
            let mut entry = state.grid_n_render.to_string();

            // Box to adjust grid n.
            let response =
                ui.add(egui::TextEdit::singleline(&mut entry).desired_width(FLOAT_EDIT_WIDTH));
            if response.changed() {
                let result = entry.parse::<usize>().unwrap_or(20);
                state.grid_n_render = result;

                let (
                    charges_electron,
                    V_from_elecs,
                    bases_evaluated,
                    bases_evaluated_charge,
                    surfaces_shared,
                    surfaces_per_elec,
                ) = crate::init_from_grid(
                    state.grid_range_render,
                    state.grid_range_charge,
                    state.sample_factor_render,
                    state.grid_n_render,
                    state.grid_n_charge,
                    &state.bases,
                    &state.charges_fixed,
                    state.num_elecs,
                );

                state.charges_electron = charges_electron;
                state.V_from_elecs = V_from_elecs;
                state.bases_evaluated = bases_evaluated;
                state.bases_evaluated_charge = bases_evaluated_charge;
                state.surfaces_shared = surfaces_shared;
                state.surfaces_per_elec = surfaces_per_elec;

                for elec_i in 0..state.surfaces_per_elec.len() {
                    wf_ops::initialize_bases(
                        &state.charges_fixed,
                        &mut state.bases[elec_i],
                        // Some(&mut state.bases_visible[elec_i]),
                        2,
                    );
                }

                updated_evaluated_wfs = true;
                updated_meshes = true;
            }

            ui.add_space(20.);

            let prev_active_elec = state.ui_active_elec;
            // Combobox to select the active electron, or select the combined wave functino.
            let active_elec_text = match state.ui_active_elec {
                ActiveElec::Combined => "C".to_owned(),
                ActiveElec::PerElec(i) => i.to_string(),
            };

            egui::ComboBox::from_id_source(0)
                .width(30.)
                .selected_text(active_elec_text)
                .show_ui(ui, |ui| {
                    // A value to select the composite wave function.
                    ui.selectable_value(&mut state.ui_active_elec, ActiveElec::Combined, "C");

                    // A value for each individual electron
                    for i in 0..state.surfaces_per_elec.len() {
                        ui.selectable_value(
                            &mut state.ui_active_elec,
                            ActiveElec::PerElec(i),
                            i.to_string(),
                        );
                    }
                });

            // Show the new active electron's meshes, if it changed.
            if state.ui_active_elec != prev_active_elec {
                updated_meshes = true;
            }

            // if ui
            //     .checkbox(&mut state.ui_render_all_elecs, "Render all elecs")
            //     .clicked()
            // {
            //     updated_meshes = true;
            // }

            if ui
                .checkbox(&mut state.mag_phase, "Show mag, phase")
                .clicked()
            {
                updated_meshes = true;
            }

            if ui
                .checkbox(&mut state.adjust_E_with_weights, "Auto adjust E")
                .clicked()
            {}
        });

        ui.horizontal(|ui| {
            if ui
                .checkbox(&mut state.auto_gen_elec_V, "Auto elec V")
                .clicked()
            {}

            if ui
                .checkbox(&mut state.weight_symmetry, "Weight sym")
                .clicked()
            {}

            if ui
                .checkbox(&mut state.create_2d_electron_V, "2D elec V")
                .clicked()
            {}

            if ui
                .checkbox(&mut state.create_3d_electron_V, "3D elec V")
                .clicked()
            {}
        });

        ui.add_space(ITEM_SPACING);

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                for (i, data) in state.surface_data.iter_mut().enumerate() {
                    if i > 3 {
                        continue;
                    }
                    if ui.checkbox(&mut data.visible, &data.name).clicked() {
                        engine_updates.entities = true;
                    }
                }
            });
            // todo DRY
            ui.vertical(|ui| {
                for (i, data) in state.surface_data.iter_mut().enumerate() {
                    if i <= 3 || i > 6 {
                        continue;
                    }
                    if ui.checkbox(&mut data.visible, &data.name).clicked() {
                        engine_updates.entities = true;
                    }
                }
            });

            ui.vertical(|ui| {
                for (i, data) in state.surface_data.iter_mut().enumerate() {
                    if i <= 6 {
                        continue;
                    }
                    if ui.checkbox(&mut data.visible, &data.name).clicked() {
                        engine_updates.entities = true;
                    }
                }
            });
        });

        ui.add(
            // -0.1 is a kludge.
            egui::Slider::from_get_set(
                state.grid_range_render.0..=state.grid_range_render.1 - 0.1,
                |v| {
                    if let Some(v_) = v {
                        state.ui_z_displayed = v_;
                        updated_meshes = true;
                    }

                    state.ui_z_displayed
                },
            )
            .text("Z slice"),
        );

        ui.add(
            egui::Slider::from_get_set(-TAU / 2.0..=TAU / 2.0, |v| {
                if let Some(v_) = v {
                    state.visual_rotation = v_;
                    updated_meshes = true;
                }

                state.visual_rotation
            })
            .text("Visual rotation"),
        );

        ui.add(
            egui::Slider::from_get_set(GRID_SIZE_MIN..=GRID_SIZE_MAX, |v| {
                if let Some(v_) = v {
                    state.grid_range_render = (-v_, v_);

                    // state.h_grid = (state.grid_max - state.grid_min) / (N as f64);
                    // state.h_grid_sq = state.h_grid.powi(2);

                    updated_basis_weights = true;
                    updated_evaluated_wfs = true;
                    updated_fixed_charges = true; // Seems to be required.
                    updated_meshes = true;
                }

                state.grid_range_render.1
            })
            .text("Grid range"),
        );

        match state.ui_active_elec {
            ActiveElec::PerElec(ae) => {
                ui.heading(format!(
                    "ψ'' score: {:.10}",
                    state.eval_data_per_elec[ae].score
                ));

                ui.add(
                    egui::Slider::from_get_set(E_MIN..=E_MAX, |v| {
                        if let Some(v_) = v {
                            state.eval_data_per_elec[ae].E = v_;

                            // Update psi'' calc for evaluation
                            for i in 0..state.eval_data_shared.grid_n {
                                state.eval_data_per_elec[ae].psi_pp_calc[i] =
                                    eigen_fns::find_ψ_pp_calc(
                                        state.eval_data_per_elec[ae].psi.on_pt[i],
                                        state.eval_data_per_elec[ae].V_acting_on_this[i],
                                        state.eval_data_per_elec[ae].E,
                                    );
                            }

                            updated_meshes = true;
                            updated_E_or_V = true;

                            // todo: Temp debugging
                            // println!("\n\n\nV nuc: {:?}", state.eval_data_shared.V_from_nuclei);
                            //
                            // println!("\nPsi: {:?}", state.eval_data_per_elec[ae].psi.on_pt);
                            // println!("\nCalc: {:?}", state.eval_data_per_elec[ae].psi_pp_calc);
                            // println!("\nMeas: {:?}", state.eval_data_per_elec[ae].psi_pp_meas);
                            // println!("\nV from: {:?}", state.V_from_elecs_1d[ae]);
                            // println!(
                            //     "\nV on: {:?}",
                            //     state.eval_data_per_elec[ae].V_acting_on_this
                            // );
                            //
                            // println!("\nV_ from elecs global {:?}", state.V_from_elecs_1d[ae]);
                            //
                            // println!("\nBasess eval {:?}", state.eval_data_per_elec[ae].psi.on_pt);
                        }

                        state.eval_data_per_elec[ae].E
                    })
                    .text("E"),
                );

                ui.add(
                    // -0.1 is a kludge.
                    egui::Slider::from_get_set(NUDGE_MIN..=NUDGE_MAX, |v| {
                        if let Some(v_) = v {
                            state.nudge_amount[ae] = v_;
                        }

                        state.nudge_amount[ae]
                    })
                    .text("Nudge amount")
                    .logarithmic(true),
                );

                ui.add_space(ITEM_SPACING);

                ui.heading("Charges:");

                charge_editor(
                    &mut state.charges_fixed,
                    &mut state.bases[ae],
                    &mut updated_evaluated_wfs,
                    &mut updated_basis_weights,
                    &mut updated_fixed_charges,
                    &mut engine_updates.entities,
                    ui,
                );

                ui.add_space(ITEM_SPACING);

                ui.heading("Basis functions and weights:");

                basis_fn_mixer(
                    state,
                    &mut updated_basis_weights,
                    &mut updated_evaluated_wfs,
                    ui,
                    ae,
                );

                ui.add_space(ITEM_SPACING);

                bottom_items(
                    ui,
                    state,
                    scene,
                    ae,
                    &mut updated_meshes,
                    &mut updated_basis_weights,
                    &mut updated_E_or_V,
                );

                // Code below handles various updates that were flagged above.

                if updated_fixed_charges {
                    procedures::update_fixed_charges(state);
                }

                if updated_evaluated_wfs {
                    procedures::update_evaluated_wfs(state, ae);
                    updated_meshes = true;
                }

                if updated_basis_weights {
                    procedures::update_basis_weights(state, ae);
                    updated_meshes = true;
                }

                if updated_E_or_V {
                    procedures::update_E_or_V(
                        &mut state.eval_data_per_elec[ae],
                        &mut state.surfaces_per_elec[ae],
                        &state.surfaces_shared.V_from_nuclei,
                        state.eval_data_shared.grid_n,
                        state.grid_n_render,
                    );
                }
            }

            ActiveElec::Combined => {
                // DRY with per-elec E slider, but unable to delegate to our function due to
                // borrow-checker issues.

                // ui.heading(format!(
                //     "ψ'' score: {:.10}",
                //     state.surfaces_shared.psi_pp_score
                // ));

                ui.add(
                    egui::Slider::from_get_set(E_MIN..=E_MAX, |v| {
                        if let Some(v_) = v {
                            state.surfaces_shared.E = v_;

                            updated_E_or_V = true;
                            updated_meshes = true;
                        }

                        state.surfaces_shared.E
                    })
                    .text("E"),
                );

                // Multiply wave functions together, and stores in Shared surfaces.
                // todo: This is an approximation
                if ui.add(egui::Button::new("Combine wavefunctions")).clicked() {
                    procedures::combine_wfs(state);
                    updated_meshes = true;
                    engine_updates.entities = true;
                }
            }
        }
        // Code here runs for both multi-electron, and combined states

        // Track using a variable to avoid mixing mutable and non-mutable borrows to
        // surfaces.
        if engine_updates.entities {
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

        if updated_meshes {
            procedures::update_meshes(state, scene, &mut engine_updates);
        }
    });

    engine_updates
}
