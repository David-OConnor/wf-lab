use std::f64::consts::TAU;

use egui::{self, Color32, RichText};
use graphics::{EngineUpdates, Scene};
use lin_alg2::f64::{Quaternion, Vec3};

use crate::{
    basis_weight_finder, basis_wfs::Basis, eigen_fns, elec_elec, render, types, wf_ops, State,
};

const UI_WIDTH: f32 = 300.;
const SIDE_PANEL_SIZE: f32 = 400.;
const SLIDER_WIDTH: f32 = 260.;
const SLIDER_WIDTH_ORIENTATION: f32 = 100.;

const E_MIN: f64 = -2.0;
const E_MAX: f64 = 0.2;

const L_MIN: f64 = -3.;
const L_MAX: f64 = 3.;

// sets range of -size to +size
const GRID_SIZE_MIN: f64 = 0.;
const GRID_SIZE_MAX: f64 = 40.;

const ITEM_SPACING: f32 = 18.;
const FLOAT_EDIT_WIDTH: f32 = 24.;

const NUDGE_MIN: f64 = 0.;
const NUDGE_MAX: f64 = 0.2;

fn text_edit_float(val: &mut f64, default: f64, ui: &mut egui::Ui) {
    let mut entry = val.to_string();

    let response = ui.add(egui::TextEdit::singleline(&mut entry).desired_width(FLOAT_EDIT_WIDTH));
    if response.changed() {
        *val = entry.parse::<f64>().unwrap_or(0.);
    }
}

/// Ui elements that allow adding, removing, and changing the point
/// charges that form our potential.
fn charge_editor(
    charges: &mut Vec<(Vec3, f64)>,
    basis_fns: &mut Vec<Basis>,
    updated_unweighted_basis_wfs: &mut bool,
    updated_basis_weights: &mut bool,
    updated_charges: &mut bool,
    updated_entities: &mut bool,
    ui: &mut egui::Ui,
) {
    let mut charge_removed = None;

    for (i, (posit, val)) in charges.iter_mut().enumerate() {
        // We store prev posit so we can know to update entities
        // when charge posit changes.
        let prev_posit = posit.clone();
        let prev_charge = val.clone();

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
    ui: &mut egui::Ui,
    // engine_updates: &mut EngineUpdates,
    // scene: &mut Scene,
) {
    // Select with charge (and its position) this basis fn is associated with.
    egui::containers::ScrollArea::vertical()
        .max_height(400.)
        .show(ui, |ui| {
            for (id, basis) in state.bases[state.ui_active_elec].iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    // Checkbox to immediately hide or show the basis.

                    if ui
                        .checkbox(&mut state.bases_visible[state.ui_active_elec][id], "")
                        .clicked()
                    {
                        *updated_basis_weights = true;
                    }

                    ui.spacing_mut().slider_width = SLIDER_WIDTH_ORIENTATION; // Only affects sliders in this section.

                    // `prev...` is to check if it changed below.
                    let prev_charge_id = basis.charge_id();

                    // Pair WFs with charge positions.
                    egui::ComboBox::from_id_source(id + 1_000)
                        .width(30.)
                        .selected_text(basis.charge_id().to_string())
                        .show_ui(ui, |ui| {
                            for (mut charge_i, (_charge_posit, _amt)) in
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

                    // egui::ComboBox::from_id_source(id + 2_000)
                    //     .width(30.)
                    //     .selected_text(basis.n().to_string())
                    //     .show_ui(ui, |ui| {
                    //         for i in 1..4 {
                    //             ui.selectable_value(basis.n_mut(), i, i.to_string());
                    //         }
                    //     });
                    //
                    // ui.heading("l:");
                    //
                    // egui::ComboBox::from_id_source(id + 3_000)
                    //     .width(30.)
                    //     .selected_text(basis.l().to_string())
                    //     .show_ui(ui, |ui| {
                    //         for i in 0..basis.n() {
                    //             ui.selectable_value(basis.l_mut(), i, i.to_string());
                    //         }
                    //     });
                    //
                    // ui.heading("m:");
                    //
                    // egui::ComboBox::from_id_source(id + 4_000)
                    //     .width(30.)
                    //     .selected_text(basis.m().to_string())
                    //     .show_ui(ui, |ui| {
                    //         for i in -1 * basis.l() as i16..basis.l() as i16 + 1 {
                    //             ui.selectable_value(basis.m_mut(), i, i.to_string());
                    //         }
                    //     });

                    if basis.n() != n_prev || basis.l() != l_prev || basis.m() != m_prev {
                        // Enforce quantum number constraints.
                        if basis.l() >= basis.n() {
                            *basis.l_mut() = basis.n() - 1;
                        }
                        if basis.m() < -1 * basis.l() as i16 {
                            *basis.m_mut() = -1 * basis.l() as i16
                        } else if basis.m() > basis.l() as i16 {
                            *basis.m_mut() = basis.l() as i16
                        }

                        *updated_unweighted_basis_wfs = true;
                        *updated_basis_weights = true;
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
                    egui::Slider::from_get_set(wf_ops::WEIGHT_MIN..=wf_ops::WEIGHT_MAX, |v| {
                        if let Some(v_) = v {
                            *basis.weight_mut() = v_;
                            *updated_basis_weights = true;
                        }

                        basis.weight()
                    })
                    .text("Wt"),
                );
            }
        });
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

        // Selector to choose the active electron
        let prev_active_elec = state.ui_active_elec;

        // todo: Is this where the surfaces combined vs individual should be?

        // todo: Unused.
        // let surfaces = if state.ui_render_all_elecs {
        //     &state.surfaces_shared
        // } else {
        //     &state.surfaces_per_elec[state.ui_active_elec]
        // };

        let mut updated_charges = false;
        let mut updated_unweighted_basis_wfs = false;
        let mut updated_basis_weights = false;
        let mut updated_meshes = false;

        ui.horizontal(|ui| {
            let mut entry = state.grid_n.to_string();

            let response =
                ui.add(egui::TextEdit::singleline(&mut entry).desired_width(FLOAT_EDIT_WIDTH));
            if response.changed() {
                let result = entry.parse::<usize>().unwrap_or(20);
                state.grid_n = result;

                // Now, update all things that depend on n.

                // todo: Start sloppy C+P from main. Put this into a fn etc.
                let arr_real = types::new_data_real(state.grid_n);

                // These must be initialized from wave functions later.
                let charges_electron = vec![arr_real.clone(), arr_real];

                let sfcs_one_elec = crate::SurfacesPerElec::new(state.grid_n);

                let mut surfaces_per_elec = vec![sfcs_one_elec.clone(), sfcs_one_elec];

                let mut grid_min = -2.;
                let mut grid_max = 2.; // todo: Is this used, or overridden?
                let spacing_factor = 1.6;

                let Es = [-0.7, -0.7]; // todo?

                let mut surfaces_shared =
                    crate::SurfacesShared::new(grid_min, grid_max, spacing_factor, state.grid_n);
                surfaces_shared.combine_psi_parts(&surfaces_per_elec, &Es, state.grid_n);

                state.surfaces_shared = surfaces_shared;
                state.surfaces_per_elec = surfaces_per_elec;

                state.bases_unweighted = wf_ops::BasisWfsUnweighted::new(
                    &state.bases[state.ui_active_elec],
                    &state.surfaces_shared.grid_posits,
                    state.grid_n,
                );

                // todo end sloppy C+P

                updated_basis_weights = true; // todo?
                updated_unweighted_basis_wfs = true; // todo?
                updated_charges = true; // todo?
                updated_meshes = true;
            }

            ui.add_space(20.);

            egui::ComboBox::from_id_source(0)
                .width(30.)
                .selected_text(state.ui_active_elec.to_string())
                .show_ui(ui, |ui| {
                    for i in 0..state.surfaces_per_elec.len() {
                        ui.selectable_value(&mut state.ui_active_elec, i, i.to_string());
                    }
                });

            // Show the new active electron's meshes, if it changed.
            if state.ui_active_elec != prev_active_elec {
                updated_meshes = true;
            }

            if ui
                .checkbox(&mut state.ui_render_all_elecs, "Render all elecs")
                .clicked()
            {
                updated_meshes = true;
            }

            if ui
                .checkbox(&mut state.mag_phase, "Show mag, phase")
                .clicked()
            {
                updated_meshes = true;
            }
        });

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                for (i, name) in state.surface_names.iter_mut().enumerate() {
                    if i > 3 {
                        continue;
                    }
                    let show = &mut state.show_surfaces[i];
                    if ui.checkbox(show, &*name).clicked() {
                        engine_updates.entities = true;
                    }
                }
            });
            // todo DRY
            ui.vertical(|ui| {
                for (i, name) in state.surface_names.iter_mut().enumerate() {
                    if i <= 3 || i > 6 {
                        continue;
                    }
                    let show = &mut state.show_surfaces[i];
                    if ui.checkbox(show, &*name).clicked() {
                        engine_updates.entities = true;
                    }
                }
            });

            ui.vertical(|ui| {
                for (i, name) in state.surface_names.iter_mut().enumerate() {
                    if i <= 6 {
                        continue;
                    }
                    let show = &mut state.show_surfaces[i];
                    if ui.checkbox(show, &*name).clicked() {
                        engine_updates.entities = true;
                    }
                }
            });
        });

        // ui.add_space(ITEM_SPACING);

        ui.heading(format!(
            "ψ'' score: {:.10}",
            state.psi_pp_score[state.ui_active_elec]
        ));

        ui.add(
            egui::Slider::from_get_set(E_MIN..=E_MAX, |v| {
                if let Some(v_) = v {
                    state.E[state.ui_active_elec] = v_;

                    for i in 0..state.grid_n {
                        for j in 0..state.grid_n {
                            for k in 0..state.grid_n {
                                state.surfaces_per_elec[state.ui_active_elec].psi_pp_calculated[i]
                                    [j][k] = eigen_fns::find_ψ_pp_calc(
                                    &state.surfaces_per_elec[state.ui_active_elec].psi.on_pt,
                                    &state.surfaces_per_elec[state.ui_active_elec].V,
                                    state.E[state.ui_active_elec],
                                    i,
                                    j,
                                    k,
                                )
                            }
                        }
                    }

                    state.psi_pp_score[state.ui_active_elec] = wf_ops::score_wf(
                        &state.surfaces_per_elec[state.ui_active_elec],
                        state.grid_n,
                    );
                    // state.psi_p_score[state.ui_active_elec] = 0.; // todo!

                    updated_meshes = true;
                }

                state.E[state.ui_active_elec]
            })
            .text("E"),
        );
        //
        // // todo: DRY!!
        // ui.add(
        //     egui::Slider::from_get_set(L_MIN..=L_MAX, |v| {
        //         if let Some(v_) = v {
        //             state.L_2 = v_;
        //
        //             for i in 0..N {
        //                 for j in 0..N {
        //                     for k in 0..N {
        //                         state.surfaces.psi_pp_calculated[i][j][k] =
        //                             eigen_fns::find_ψ_pp_calc(
        //                                 &state.surfaces.psi[state.ui_active_elec],
        //                                 &state.surfaces.V,
        //                                 state.E,
        //                                 i,
        //                                 j,
        //                                 k,
        //                             )
        //                     }
        //                 }
        //             }
        //
        //             state.psi_pp_score = wf_ops::score_wf(&state.surfaces);
        //             state.psi_p_score = 0.; // todo!
        //
        //             render::update_meshes(&state.surfaces, state.ui_z_displayed, scene);
        //             engine_updates.meshes = true;
        //         }
        //
        //         state.L_2
        //     })
        //     .text("L^2"),
        // );
        //
        // // todo: DRY!!
        // ui.add(
        //     egui::Slider::from_get_set(L_MIN..=L_MAX, |v| {
        //         if let Some(v_) = v {
        //             state.L_x = v_;
        //
        //             for i in 0..N {
        //                 for j in 0..N {
        //                     for k in 0..N {
        //                         state.surfaces.psi_pp_calculated[i][j][k] =
        //                             eigen_fns::find_ψ_pp_calc(
        //                                 &state.surfaces.psi[state.ui_active_elec],
        //                                 &state.surfaces.V,
        //                                 state.E,
        //                                 i,
        //                                 j,
        //                                 k,
        //                             )
        //                     }
        //                 }
        //             }
        //
        //             state.psi_pp_score = wf_ops::score_wf(&state.surfaces);
        //             state.psi_p_score = 0.; // todo!
        //
        //             render::update_meshes(&state.surfaces, state.ui_z_displayed, scene);
        //             engine_updates.meshes = true;
        //         }
        //
        //         state.L_x
        //     })
        //     .text("L_x"),
        // );
        //
        // // todo: DRY!!
        // ui.add(
        //     egui::Slider::from_get_set(L_MIN..=L_MAX, |v| {
        //         if let Some(v_) = v {
        //             state.L_y = v_;
        //
        //             for i in 0..N {
        //                 for j in 0..N {
        //                     for k in 0..N {
        //                         state.surfaces.psi_pp_calculated[i][j][k] =
        //                             eigen_fns::find_ψ_pp_calc(
        //                                 &state.surfaces.psi[state.ui_active_elec],
        //                                 &state.surfaces.V,
        //                                 state.E,
        //                                 i,
        //                                 j,
        //                                 k,
        //                             )
        //                     }
        //                 }
        //             }
        //
        //             state.psi_pp_score = wf_ops::score_wf(&state.surfaces);
        //             state.psi_p_score = 0.; // todo!
        //
        //             render::update_meshes(&state.surfaces, state.ui_z_displayed, scene);
        //             engine_updates.meshes = true;
        //         }
        //
        //         state.L_y
        //     })
        //     .text("L_y"),
        // );
        //
        // // todo: DRY!!
        // ui.add(
        //     egui::Slider::from_get_set(L_MIN..=L_MAX, |v| {
        //         if let Some(v_) = v {
        //             state.L_z = v_;
        //
        //             for i in 0..N {
        //                 for j in 0..N {
        //                     for k in 0..N {
        //                         state.surfaces.psi_pp_calculated[i][j][k] =
        //                             eigen_fns::find_ψ_pp_calc(
        //                                 &state.surfaces.psi[state.ui_active_elec],
        //                                 &state.surfaces.V,
        //                                 state.E,
        //                                 i,
        //                                 j,
        //                                 k,
        //                             )
        //                     }
        //                 }
        //             }
        //
        //             state.psi_pp_score = wf_ops::score_wf(&state.surfaces);
        //             state.psi_p_score = 0.; // todo!
        //
        //             render::update_meshes(&state.surfaces, state.ui_z_displayed, scene);
        //             engine_updates.meshes = true;
        //         }
        //
        //         state.L_z
        //     })
        //     .text("L_z"),
        // );

        ui.add(
            // -0.1 is a kludge.
            egui::Slider::from_get_set(state.grid_min..=state.grid_max - 0.1, |v| {
                if let Some(v_) = v {
                    state.ui_z_displayed = v_;
                    updated_meshes = true;
                }

                state.ui_z_displayed
            })
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
                    state.grid_min = -v_;
                    state.grid_max = v_;

                    // state.h_grid = (state.grid_max - state.grid_min) / (N as f64);
                    // state.h_grid_sq = state.h_grid.powi(2);

                    updated_basis_weights = true;
                    updated_unweighted_basis_wfs = true;
                    updated_charges = true; // Seems to be required.
                    updated_meshes = true;
                }

                state.grid_max
            })
            .text("Grid range"),
        );

        ui.add(
            // -0.1 is a kludge.
            egui::Slider::from_get_set(NUDGE_MIN..=NUDGE_MAX, |v| {
                if let Some(v_) = v {
                    state.nudge_amount[state.ui_active_elec] = v_;
                }

                state.nudge_amount[state.ui_active_elec]
            })
            .text("Nudge amount")
            .logarithmic(true),
        );

        ui.add_space(ITEM_SPACING);

        ui.heading("Charges:");

        charge_editor(
            &mut state.charges_fixed,
            &mut state.bases[state.ui_active_elec],
            &mut updated_unweighted_basis_wfs,
            &mut updated_basis_weights,
            &mut updated_charges,
            &mut engine_updates.entities,
            ui,
        );

        ui.add_space(ITEM_SPACING);

        ui.heading("Basis functions and weights:");

        basis_fn_mixer(
            state,
            &mut updated_basis_weights,
            &mut updated_unweighted_basis_wfs,
            ui,
        );

        ui.add_space(ITEM_SPACING);

        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("Nudge WF")).clicked() {
                crate::nudge::nudge_wf(
                    &mut state.surfaces_per_elec[state.ui_active_elec],
                    // &state.bases,
                    // &state.charges,
                    &mut state.nudge_amount[state.ui_active_elec],
                    &mut state.E[state.ui_active_elec],
                    state.grid_min,
                    state.grid_max,
                    &state.bases[state.ui_active_elec],
                    &state.surfaces_shared.grid_posits,
                    state.grid_n,
                );

                updated_meshes = true;

                state.psi_pp_score[state.ui_active_elec] = crate::wf_ops::score_wf(
                    &state.surfaces_per_elec[state.ui_active_elec],
                    state.grid_n,
                );

                // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);
                // state.psi_pp_score  = crate::eval_wf(&state.wfs, &state.charges, state.E);

                // *updated_wfs = true;
            }

            if ui.add(egui::Button::new("Create e- charge")).clicked() {
                elec_elec::update_charge_density_fm_psi(
                    &state.surfaces_per_elec[state.ui_active_elec].psi.on_pt,
                    &mut state.charges_electron[state.ui_active_elec],
                    state.grid_n,
                );

                updated_basis_weights = true;
                updated_unweighted_basis_wfs = true;
                updated_charges = true;
            }

            if ui.add(egui::Button::new("Empty e- charge")).clicked() {
                state.charges_electron[state.ui_active_elec] = types::new_data_real(state.grid_n);

                updated_basis_weights = true;
                updated_unweighted_basis_wfs = true;
                updated_charges = true;
            }

            if ui.add(egui::Button::new("Create e- V")).clicked() {
                elec_elec::update_V_individual(
                    &mut state.surfaces_per_elec[state.ui_active_elec].V,
                    &state.surfaces_shared.V_fixed_charges,
                    &state.charges_electron,
                    state.ui_active_elec,
                    &state.surfaces_shared.grid_posits,
                    state.grid_n,
                );

                updated_basis_weights = true;
                updated_unweighted_basis_wfs = true;
                updated_charges = true;
            }

            if ui.add(egui::Button::new("Find E")).clicked() {
                wf_ops::find_E(
                    &mut state.surfaces_per_elec[state.ui_active_elec],
                    &mut state.E[state.ui_active_elec],
                    state.grid_n,
                );

                updated_meshes = true;

                state.psi_pp_score[state.ui_active_elec] =
                    wf_ops::score_wf(&state.surfaces_per_elec[state.ui_active_elec], state.grid_n);

                // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);
                // state.psi_pp_score  = crate::eval_wf(&state.wfs, &state.charges, state.E);

                // *updated_wfs = true;
            }
        });

        if ui.add(egui::Button::new("Find weights")).clicked() {
            basis_weight_finder::find_weights(
                &state.charges_fixed,
                &mut state.bases[state.ui_active_elec],
                &mut state.bases_unweighted,
                &mut state.E[state.ui_active_elec],
                &mut state.surfaces_shared,
                &mut state.surfaces_per_elec[state.ui_active_elec],
                3,
                state.grid_n,
            );

            // todo: These may not be required if handled by `find_weights`.
            // updated_basis_weights = true;
            // updated_unweighted_basis_wfs = true;
            updated_meshes = true;
        }

        // Code below handles various updates that were flagged above.

        if updated_charges {
            // Reinintialize bases due to the added charges
            // Note: An alternative would be to add the new bases without 0ing the existing ones.
            wf_ops::initialize_bases(
                &mut state.charges_fixed,
                &mut state.bases[state.ui_active_elec],
                &mut state.bases_visible[state.ui_active_elec],
                2,
            );

            wf_ops::update_V_fm_fixed_charges(
                &state.charges_fixed,
                &mut state.surfaces_shared.V_fixed_charges,
                &mut state.grid_min,
                &mut state.grid_max,
                state.spacing_factor,
                &mut state.surfaces_shared.grid_posits,
                state.grid_n,
            );

            // Replace indiv sfc charges with this. A bit of a kludge, perhaps
            for sfc in &mut state.surfaces_per_elec {
                types::copy_array_real(
                    &mut sfc.V,
                    &state.surfaces_shared.V_fixed_charges,
                    state.grid_n,
                );
            }
        }

        if updated_unweighted_basis_wfs {
            engine_updates.meshes = true;

            state.bases_unweighted = wf_ops::BasisWfsUnweighted::new(
                &state.bases[state.ui_active_elec],
                &state.surfaces_shared.grid_posits,
                state.grid_n,
            );
        }

        if updated_basis_weights {
            engine_updates.meshes = true;

            let mut weights = vec![0.; state.bases[state.ui_active_elec].len()];
            // Syncing procedure pending a better API.
            for (i, basis) in state.bases[state.ui_active_elec].iter().enumerate() {
                weights[i] = basis.weight();
            }

            // Set up our basis-function based trial wave function.
            wf_ops::update_wf_fm_bases(
                &state.bases[state.ui_active_elec],
                &state.bases_unweighted,
                &mut state.surfaces_per_elec[state.ui_active_elec],
                state.E[state.ui_active_elec],
                // &mut state.surfaces_shared.grid_posits,
                &state.bases_visible[state.ui_active_elec],
                state.grid_n,
                &weights,
            );

            state.psi_pp_score[state.ui_active_elec] =
                wf_ops::score_wf(&state.surfaces_per_elec[state.ui_active_elec], state.grid_n);

            updated_meshes = true;
        }

        if updated_meshes {
            engine_updates.meshes = true;

            render::update_meshes(
                &state.surfaces_shared,
                &state.surfaces_per_elec[state.ui_active_elec],
                state.ui_z_displayed,
                scene,
                &state.surfaces_shared.grid_posits,
                state.mag_phase,
                state.grid_n,
            );
        }
        // Track using a variable to avoid mixing mutable and non-mutable borrows to
        // surfaces.
        if engine_updates.entities {
            render::update_entities(&state.charges_fixed, &state.show_surfaces, scene);
        }
    });

    engine_updates
}
