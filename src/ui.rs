use std::f64::consts::TAU;

use egui::{self, Color32, RichText};

use graphics::{EngineUpdates, Scene};

use crate::{
    basis_wfs::Basis,
    eigen_fns, render,
    wf_ops::{self, N},
    State,
};

use lin_alg2::f64::{EulerAngle, Quaternion, Vec3};
use wf_lab::types;

const UI_WIDTH: f32 = 300.;
const SIDE_PANEL_SIZE: f32 = 400.;
const SLIDER_WIDTH: f32 = 260.;
const SLIDER_WIDTH_ORIENTATION: f32 = 100.;

const E_MIN: f64 = -1.2;
const E_MAX: f64 = 0.2;

const L_MIN: f64 = -3.;
const L_MAX: f64 = 3.;

// Wave fn weights
const WEIGHT_MIN: f64 = -2.;
const WEIGHT_MAX: f64 = 2.;

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
    updated_wfs: &mut bool,
    updated_charges: &mut bool,
    updated_entities: &mut bool,
    ui: &mut egui::Ui,
) {
    // todo: Scroll area?

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
                *updated_wfs = true;
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
                *updated_wfs = true;
                *updated_charges = true;
                *updated_entities = true;
            }

            if ui.button(RichText::new("❌").color(Color32::RED)).clicked() {
                // Don't remove from a collection we're iterating over.
                charge_removed = Some(i);
                *updated_wfs = true;
                *updated_charges = true;
                // Update entities due to charge sphere placement.
                *updated_entities = true;
            }
        });
    }

    if let Some(charge_i_removed) = charge_removed {
        charges.remove(charge_i_removed);
        *updated_wfs = true;
        *updated_charges = true;
        *updated_entities = true;
    }

    if ui.add(egui::Button::new("Add charge")).clicked() {
        charges.push((Vec3::new_zero(), crate::Q_PROT));
        *updated_wfs = true;
        *updated_charges = true;
        *updated_entities = true;
    }
}

/// Ui elements that allow mixing various basis WFs.
fn basis_fn_mixer(
    state: &mut State,
    updated_wfs: &mut bool,
    ui: &mut egui::Ui,
    // engine_updates: &mut EngineUpdates,
    // scene: &mut Scene,
) {
    // Select with charge (and its position) this basis fn is associated with.
    egui::containers::ScrollArea::vertical().show(ui, |ui| {
        for (id, basis) in state.bases[0].iter_mut().enumerate() {
            ui.horizontal(|ui| {
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
                    *updated_wfs = true;
                }

                ui.heading("n:");

                let n_prev = basis.n();
                let l_prev = basis.l();
                let m_prev = basis.m();

                // todo: Helper fn to reduce DRY here.
                egui::ComboBox::from_id_source(id + 2_000)
                    .width(30.)
                    .selected_text(basis.n().to_string())
                    .show_ui(ui, |ui| {
                        for i in 1..4 {
                            ui.selectable_value(basis.n_mut(), i, i.to_string());
                        }
                    });

                ui.heading("l:");

                egui::ComboBox::from_id_source(id + 3_000)
                    .width(30.)
                    .selected_text(basis.l().to_string())
                    .show_ui(ui, |ui| {
                        for i in 0..basis.n() {
                            ui.selectable_value(basis.l_mut(), i, i.to_string());
                        }
                    });

                ui.heading("m:");

                egui::ComboBox::from_id_source(id + 4_000)
                    .width(30.)
                    .selected_text(basis.m().to_string())
                    .show_ui(ui, |ui| {
                        for i in -1 * basis.l() as i16..basis.l() as i16 + 1 {
                            ui.selectable_value(basis.m_mut(), i, i.to_string());
                        }
                    });

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

                    *updated_wfs = true;
                }

                // For now, we use an azimuth, elevation API for orientation.
                // todo: Not working
                if basis.l() >= 1 && basis.weight().abs() > 0.00001 {
                    let mut euler = basis.harmonic().orientation.to_euler();

                    // todo: DRY between the 3.
                    ui.add(
                        // Offsets are to avoid gimball lock.
                        egui::Slider::from_get_set(-TAU / 4.0 + 0.001..=TAU / 4.0 - 0.001, |v| {
                            if let Some(v_) = v {
                                euler.pitch = v_;
                                basis.harmonic_mut().orientation = Quaternion::from_euler(&euler);
                                *updated_wfs = true;
                            }

                            euler.pitch
                        })
                        .text("P"),
                    );
                    ui.add(
                        egui::Slider::from_get_set(-TAU / 2.0..=TAU / 2.0, |v| {
                            if let Some(v_) = v {
                                euler.roll = v_;
                                basis.harmonic_mut().orientation = Quaternion::from_euler(&euler);
                                *updated_wfs = true;
                            }

                            euler.roll
                        })
                        .text("R"),
                    );
                    ui.add(
                        egui::Slider::from_get_set(0.0..=TAU, |v| {
                            if let Some(v_) = v {
                                euler.yaw = v_;
                                basis.harmonic_mut().orientation = Quaternion::from_euler(&euler);
                                *updated_wfs = true;
                            }

                            euler.yaw
                        })
                        .text("Y"),
                    );
                }
            });

            // todo: Text edit or dropdown for n.

            ui.add(
                egui::Slider::from_get_set(WEIGHT_MIN..=WEIGHT_MAX, |v| {
                    if let Some(v_) = v {
                        *basis.weight_mut() = v_;
                        *updated_wfs = true;
                    }

                    basis.weight()
                })
                .text("Wt"),
            );

            // todo: Text edits etc for all the fields.

            // match basis.f {
            //     BasisFn::Sto(mut slater_exp) => {
            //         let mut entry = slater_exp.to_string();
            //
            //         let response =
            //             ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
            //         if response.changed() {
            //             let exp = entry.parse::<f64>().unwrap_or(1.);
            //             basis.f = BasisFn::Sto(exp);
            //             *updated_wfs = true;
            //         }
            //     }
            //     BasisFn::H210(mut axis) => {
            //         // todo: DRY
            //         let mut entry = "0.".to_owned(); // angle
            //
            //         let response =
            //             ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
            //         if response.changed() {
            //             let angle = entry.parse::<f64>().unwrap_or(0.);
            //
            //             // todo: locked to Z rotation axis for now.
            //             let rotation_axis = Vec3::new(0., 0., 1.);
            //             let rotation = Quaternion::from_axis_angle(rotation_axis, angle);
            //             let new_axis = rotation.rotate_vec(Vec3::new(1., 0., 0.));
            //             basis.f = BasisFn::H210(new_axis);
            //
            //             *updated_wfs = true;
            //         }
            //     }
            //     _ => (),
            // }
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
                    if i <= 3 {
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

        let mut updated_wfs = false;
        let mut updated_charges = false;

        ui.heading(format!("ψ'' score: {:.10}", state.psi_pp_score[0]));

        ui.add(
            egui::Slider::from_get_set(E_MIN..=E_MAX, |v| {
                if let Some(v_) = v {
                    state.E[0] = v_;

                    for i in 0..N {
                        for j in 0..N {
                            for k in 0..N {
                                state.surfaces.psi_pp_calculated[0][i][j][k] =
                                    eigen_fns::find_ψ_pp_calc(
                                        &state.surfaces.psi[0],
                                        &state.surfaces.V[0],
                                        state.E[0],
                                        i,
                                        j,
                                        k,
                                    )
                            }
                        }
                    }

                    state.psi_pp_score[0] = crate::wf_ops::score_wf(&state.surfaces);
                    // state.psi_p_score[0] = 0.; // todo!

                    render::update_meshes(&state.surfaces, state.ui_z_displayed, scene);
                    engine_updates.meshes = true;
                }

                state.E[0]
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
        //                                 &state.surfaces.psi[0],
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
        //                                 &state.surfaces.psi[0],
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
        //                                 &state.surfaces.psi[0],
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
        //                                 &state.surfaces.psi[0],
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
                    engine_updates.meshes = true;

                    render::update_meshes(
                        &state.surfaces,
                        state.ui_z_displayed,
                        scene,
                        // state.grid_min,
                        // state.grid_max,
                    );
                }

                state.ui_z_displayed
            })
            .text("Z slice"),
        );

        ui.add(
            egui::Slider::from_get_set(GRID_SIZE_MIN..=GRID_SIZE_MAX, |v| {
                if let Some(v_) = v {
                    state.grid_min = -v_;
                    state.grid_max = v_;

                    // state.h_grid = (state.grid_max - state.grid_min) / (N as f64);
                    // state.h_grid_sq = state.h_grid.powi(2);

                    updated_wfs = true;
                    updated_charges = true; // Seems to be required.
                }

                state.grid_max
            })
            .text("Grid range"),
        );

        ui.add(
            // -0.1 is a kludge.
            egui::Slider::from_get_set(NUDGE_MIN..=NUDGE_MAX, |v| {
                if let Some(v_) = v {
                    state.nudge_amount[0] = v_;
                }

                state.nudge_amount[0]
            })
            .text("Nudge amount")
            .logarithmic(true),
        );

        ui.add_space(ITEM_SPACING);

        ui.heading("Charges:");

        charge_editor(
            &mut state.charges_fixed,
            &mut state.bases[0],
            &mut updated_wfs,
            &mut updated_charges,
            &mut engine_updates.entities,
            ui,
        );

        ui.add_space(ITEM_SPACING);

        ui.heading("Wave functions and weights:");

        // basis_fn_mixer(state, &mut updated_wfs, ui, &mut engine_updates, scene);
        basis_fn_mixer(state, &mut updated_wfs, ui);

        ui.add_space(ITEM_SPACING);

        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("Nudge WF")).clicked() {
                crate::nudge::nudge_wf(
                    &mut state.surfaces,
                    // &state.bases,
                    // &state.charges,
                    &mut state.nudge_amount[0],
                    &mut state.E[0],
                    state.grid_min,
                    state.grid_max,
                    &state.bases[0],
                );

                render::update_meshes(
                    &state.surfaces,
                    state.ui_z_displayed,
                    scene,
                    // state.grid_min,
                    // state.grid_max,
                ); // todo!
                engine_updates.meshes = true;

                state.psi_pp_score[0] = crate::wf_ops::score_wf(&state.surfaces);

                // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);
                // state.psi_pp_score  = crate::eval_wf(&state.wfs, &state.charges, state.E);

                // *updated_wfs = true;
            }

            if ui.add(egui::Button::new("Create e- V")).clicked() {
                // hard-coded as first item for now.
                crate::wf_ops::charge_density_fm_psi(
                    &state.surfaces.psi[0],
                    &mut state.surfaces.elec_charges[0],
                    1,
                );

                updated_wfs = true;
                updated_charges = true;
            }

            if ui.add(egui::Button::new("Empty e- V")).clicked() {
                // hard-coded as first item for now.
                state.surfaces.elec_charges[0] = types::new_data_real(N);

                updated_wfs = true;
                updated_charges = true;
            }

            if ui.add(egui::Button::new("Find E")).clicked() {
                wf_ops::find_E(&mut state.surfaces, &mut state.E[0]);

                render::update_meshes(
                    &state.surfaces,
                    state.ui_z_displayed,
                    scene,
                    // state.grid_min,
                    // state.grid_max,
                );
                engine_updates.meshes = true;

                state.psi_pp_score[0] = wf_ops::score_wf(&state.surfaces);

                // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);
                // state.psi_pp_score  = crate::eval_wf(&state.wfs, &state.charges, state.E);

                // *updated_wfs = true;
            }
        });

        if updated_wfs {
            engine_updates.meshes = true;

            wf_ops::init_wf(
                &state.bases[0],
                // &state.gaussians,
                &state.charges_fixed,
                &mut state.surfaces,
                state.E[0],
                updated_charges,
                &mut state.grid_min,
                &mut state.grid_max,
                state.spacing_factor,
            );

            state.psi_pp_score[0] = wf_ops::score_wf(&state.surfaces);

            render::update_meshes(
                &state.surfaces,
                state.ui_z_displayed,
                scene,
                // state.grid_min,
                // state.grid_max,
            ); // todo!
        }

        // Track using a variable to avoid mixing mutable and non-mutable borrows to
        // surfaces.
        if engine_updates.entities {
            render::update_entities(&state.charges_fixed, &state.show_surfaces, scene);
            // todo
        }
    });

    engine_updates
}
