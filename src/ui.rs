use std::f64::consts::TAU;

use egui::{self, Color32, RichText};

use graphics::{EngineUpdates, Scene};

// use crate::{basis_wfs::BasisFn, render, State, N};
use crate::{
    basis_wfs::{Basis, HOrbital, SphericalHarmonic, Sto},
    render, State, N,
};

use lin_alg2::f64::{Quaternion, Vec3};

const UI_WIDTH: f32 = 300.;
const SIDE_PANEL_SIZE: f32 = 400.;
const SLIDER_WIDTH: f32 = 260.;

const E_MIN: f64 = -1.2;
const E_MAX: f64 = 0.2;

// Wave fn weights
const WEIGHT_MIN: f64 = -2.;
const WEIGHT_MAX: f64 = 2.;

const ITEM_SPACING: f32 = 18.;
const FLOAT_EDIT_WIDTH: f32 = 24.;

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
    engine_updates: &mut EngineUpdates,
    scene: &mut Scene,
) {
    // Select with charge (and its position) this basis fn is associated with.
    egui::containers::ScrollArea::vertical().show(ui, |ui| {
        for (id, basis) in state.bases.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                // `prev...` is to check if it changed below.
                let prev_charge_id = basis.charge_id();

                // Pair WFs with charge positions.
                egui::ComboBox::from_id_source(id + 1_000)
                    .width(30.)
                    .selected_text(basis.charge_id().to_string())
                    .show_ui(ui, |ui| {
                        for (mut charge_i, (_charge_posit, _amt)) in
                            state.charges.iter().enumerate()
                        {
                            ui.selectable_value(
                                basis.charge_id_mut(),
                                charge_i,
                                charge_i.to_string(),
                            );
                        }
                    });

                if basis.charge_id() != prev_charge_id {
                    *basis.posit_mut() = state.charges[basis.charge_id()].0;
                    *updated_wfs = true;
                }

                // let mut selected = basis;

                // Maybe a basis-type enum?
                // let mut h_basis = Basis::H(HOrbital::new(basis.posit(), basis.n(), basis.harmonic().clone(), basis.weight(), basis.charge_id()));
                // let mut sto_basis = Basis::Sto(Sto::new(basis.posit(), basis.n(), basis.harmonic().clone(), 1.,  basis.weight(), basis.charge_id()));
                //
                // egui::ComboBox::from_id_source(id + 2_000)
                //     .width(60.)
                //     .selected_text(basis.descrip())
                //     .show_ui(ui, |ui| {
                //         // ui.selectable_value(&mut selected, BasisFn::H100, BasisFn::H100.descrip());
                //         // ui.selectable_value(&mut selected, BasisFn::H200, BasisFn::H200.descrip());
                //         // ui.selectable_value(&mut selected, BasisFn::H300, BasisFn::H300.descrip());
                //         ui.selectable_value(
                //             basis,
                //             &mut h_basis,
                //             "STO orbital"
                //         );
                //         ui.selectable_value(
                //             basis,
                //             &mut sto_basis,
                //             "H orbital"
                //         );
                //         // ui.selectable_value(&mut selected, BasisFn::H211, BasisFn::H211.descrip());
                //         // ui.selectable_value(&mut selected, BasisFn::H21M1, BasisFn::H21M1.descrip());
                //     });

                // Just comparing type, based on partialEq impl
                // todo: Put back etc
                // if selected != basis {
                //     // basis = selected;
                //     *updated_wfs = true;
                // }

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
                ui.heading("Yaw");

                // todo: Not working
                if basis.l() >= 1 && basis.weight().abs() > 0.00001 {
                    let mut euler = basis.harmonic().orientation.to_euler();

                    ui.add(
                        egui::Slider::from_get_set(WEIGHT_MIN..=WEIGHT_MAX, |v| {
                            if let Some(v_) = v {
                                euler.yaw = v_;
                                // phi, psi, theta??
                                basis.harmonic_mut().orientation = Quaternion::from_euler(&euler);
                                *updated_wfs = true;
                            }

                            euler.yaw
                        })
                        .text("Yaw"),
                    );
                }
                //
                // egui::ComboBox::from_id_source(id + 5_000)
                //     .width(30.)
                //     .selected_text(basis.m().to_string())
                //     .show_ui(ui, |ui| {
                //         for i in 0.0..TAU {
                //             ui.selectable_value(&mut az, i, i.to_string()); // todo: Float format?
                //         }
                //     });

                // todo: If this works, do pitch and roll too.
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
                    if i > 2 {
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
                    if i <= 2 {
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

        ui.heading(format!("ψ'' score: {:.10}", state.psi_pp_score));

        ui.add(
            egui::Slider::from_get_set(E_MIN..=E_MAX, |v| {
                if let Some(v_) = v {
                    state.E = v_;

                    // todo: Don't update all meshes! We only need
                    // to update psi_pp_calculated.
                    // todo: YOu should probably have a delegated standalone
                    // todo fn for this, probably one that accepts i, j, k
                    // todo: YOu'd call it from nudge, eval, and here.

                    for i in 0..N {
                        for j in 0..N {
                            for k in 0..N {
                                state.surfaces.psi_pp_calculated[i][j][k] = crate::find_psi_pp_calc(
                                    &state.surfaces.psi,
                                    &state.surfaces.V,
                                    state.E,
                                    i,
                                    j,
                                    k,
                                )
                            }
                        }
                    }

                    // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);
                    // state.psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, state.E);
                    state.psi_pp_score = crate::score_wf(&state.surfaces);

                    render::update_meshes(&state.surfaces, state.z_displayed, scene);
                    engine_updates.meshes = true;
                }

                state.E
            })
            .text("E"),
        );

        ui.add(
            // -0.1 is a kludge.
            egui::Slider::from_get_set(crate::GRID_MIN..=crate::GRID_MAX - 0.1, |v| {
                if let Some(v_) = v {
                    state.z_displayed = v_;
                    engine_updates.meshes = true;

                    render::update_meshes(&state.surfaces, state.z_displayed, scene);
                }

                state.z_displayed
            })
            .text("Z slice"),
        );

        ui.add(
            // -0.1 is a kludge.
            egui::Slider::from_get_set(0.0..=0.1, |v| {
                if let Some(v_) = v {
                    state.nudge_amount = v_;
                }

                state.nudge_amount
            })
            .text("Nudge amount")
            .logarithmic(true),
        );

        // ui.add_space(ITEM_SPACING);

        let mut updated_wfs = false;
        let mut updated_charges = false;

        ui.heading("Charges:");

        charge_editor(
            &mut state.charges,
            &mut updated_wfs,
            &mut updated_charges,
            &mut engine_updates.entities,
            ui,
        );

        ui.add_space(ITEM_SPACING);

        ui.heading("Wave functions and weights:");

        basis_fn_mixer(state, &mut updated_wfs, ui, &mut engine_updates, scene);

        ui.add_space(ITEM_SPACING);

        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("Nudge WF")).clicked() {
                crate::nudge_wf(
                    &mut state.surfaces,
                    &state.bases,
                    &state.charges,
                    &mut state.nudge_amount,
                    &mut state.E,
                );

                render::update_meshes(&state.surfaces, state.z_displayed, scene); // todo!
                engine_updates.meshes = true;

                state.psi_pp_score = crate::score_wf(&state.surfaces);

                // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);
                // state.psi_pp_score  = crate::eval_wf(&state.wfs, &state.charges, state.E);

                // *updated_wfs = true;
            }

            if ui.add(egui::Button::new("Create e- V")).clicked() {
                // hard-coded as first item for now.
                crate::elec_V_density_fm_psi(
                    &state.surfaces.psi,
                    &mut state.surfaces.elec_charges[0],
                    1,
                );

                updated_wfs = true;
                updated_charges = true;
            }

            if ui.add(egui::Button::new("Empty e- V")).clicked() {
                // hard-coded as first item for now.
                state.surfaces.elec_charges[0] = crate::new_data_real(crate::N);

                updated_wfs = true;
                updated_charges = true;
            }

            if ui.add(egui::Button::new("Find E")).clicked() {
                crate::find_E(&mut state.surfaces, &mut state.E);

                render::update_meshes(&state.surfaces, state.z_displayed, scene);
                engine_updates.meshes = true;

                state.psi_pp_score = crate::score_wf(&state.surfaces);

                // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);
                // state.psi_pp_score  = crate::eval_wf(&state.wfs, &state.charges, state.E);

                // *updated_wfs = true;
            }
        });

        if updated_wfs {
            engine_updates.meshes = true;

            crate::eval_wf(
                &state.bases,
                // &state.gaussians,
                &state.charges,
                &mut state.surfaces,
                state.E,
                updated_charges,
            );

            state.psi_pp_score = crate::score_wf(&state.surfaces);

            render::update_meshes(&state.surfaces, state.z_displayed, scene); // todo!
        }

        // Track using a variable to avoid mixing mutable and non-mutable borrows to
        // surfaces.
        if engine_updates.entities {
            render::update_entities(&state.charges, &state.show_surfaces, scene);
            // todo
        }
    });

    engine_updates
}
