use egui::{self, Color32, RichText};

use graphics::{EngineUpdates, Scene};

use crate::{basis_wfs::BasisFn, render, State, N};

use lin_alg2::f64::{Quaternion, Vec3};

const UI_WIDTH: f32 = 300.;
const SIDE_PANEL_SIZE: f32 = 400.;
const SLIDER_WIDTH: f32 = 260.;

const E_MIN: f64 = -2.;
const E_MAX: f64 = 2.;

// Wave fn weights
const WEIGHT_MIN: f64 = -2.;
const WEIGHT_MAX: f64 = 2.;

const ITEM_SPACING: f32 = 16.;

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
                                state.surfaces.psi_pp_calculated[i][j][k] =
                                    crate::find_psi_pp_calc(&state.surfaces, state.E, i, j, k)
                            }
                        }
                    }

                    // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);
                    // state.psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, state.E);
                    state.psi_pp_score = crate::score_wf(&state.surfaces, state.E);

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

        // ui.add_space(ITEM_SPACING);

        ui.heading("Wave functions and weights:");

        let mut updated_wfs = false;

        egui::containers::ScrollArea::vertical().show(ui, |ui| {

            for (id, basis) in state.wfs.iter_mut().enumerate() {
                // Clone here so we can properly check if it changed below.
                let mut selected = basis.f.clone();

                egui::ComboBox::from_id_source(id)
                    .width(60.)
                    .selected_text(basis.f.descrip())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut selected, BasisFn::H100, BasisFn::H100.descrip());
                        ui.selectable_value(&mut selected, BasisFn::H200, BasisFn::H200.descrip());
                        ui.selectable_value(&mut selected, BasisFn::H300, BasisFn::H300.descrip());
                        ui.selectable_value(
                            &mut selected,
                            BasisFn::H210(Vec3::new(1., 0., 0.)),
                            BasisFn::H210(Vec3::new(1., 0., 0.)).descrip(),
                        );
                        ui.selectable_value(
                            &mut selected,
                            BasisFn::Sto(1.),
                            BasisFn::Sto(1.).descrip(),
                        );
                        // ui.selectable_value(&mut selected, BasisFn::H211, BasisFn::H211.descrip());
                        // ui.selectable_value(&mut selected, BasisFn::H21M1, BasisFn::H21M1.descrip());
                    });

                if selected != basis.f {
                    basis.f = selected;
                    updated_wfs = true;
                }

                ui.add(
                    egui::Slider::from_get_set(WEIGHT_MIN..=WEIGHT_MAX, |v| {
                        if let Some(v_) = v {
                            basis.weight = v_;
                            updated_wfs = true;
                        }

                        basis.weight
                    })
                    .text("Weight"),
                );

                match basis.f {
                    BasisFn::Sto(mut slater_exp) => {
                        let mut entry = slater_exp.to_string();

                        let response =
                            ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
                        if response.changed() {
                            let exp = entry.parse::<f64>().unwrap_or(1.);
                            basis.f = BasisFn::Sto(exp);
                            updated_wfs = true;
                        }
                    }
                    BasisFn::H210(mut axis) => {
                        // todo: DRY
                        let mut entry = "0.".to_owned(); // angle

                        let response =
                            ui.add(egui::TextEdit::singleline(&mut entry).desired_width(16.));
                        if response.changed() {
                            let angle = entry.parse::<f64>().unwrap_or(0.);

                            // todo: locked to Z rotation axis for now.
                            let rotation_axis = Vec3::new(0., 0., 1.);
                            let rotation = Quaternion::from_axis_angle(rotation_axis, angle);
                            let new_axis = rotation.rotate_vec(Vec3::new(1., 0., 0.));
                            basis.f = BasisFn::H210(new_axis);

                            updated_wfs = true;
                        }
                    }
                    _ => (),
                }
            }

            ui.add_space(ITEM_SPACING);

            if ui.add(egui::Button::new("Nudge WF")).clicked() {
                crate::nudge_wf(
                    &mut state.surfaces,
                    &state.wfs,
                    &state.charges,
                    &mut state.gaussians,
                    state.E,
                );

                // todo: DRY
                engine_updates.meshes = true;

                state.psi_pp_score = crate::score_wf(&state.surfaces, state.E);

                // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);
                // state.psi_pp_score  = crate::eval_wf(&state.wfs, &state.charges, state.E);

                render::update_meshes(
                    &state.surfaces,
                    state.z_displayed,
                    // &mut state.surfaces,
                    scene,
                );
            }
        });

        if updated_wfs {
            engine_updates.meshes = true;

            // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);

            crate::eval_wf(
                &state.wfs,
                &state.gaussians,
                &state.charges,
                &mut state.surfaces,
                state.E,
            );

            state.psi_pp_score = crate::score_wf(&state.surfaces, state.E);

            render::update_meshes(&state.surfaces, state.z_displayed, scene);
        }

        // Track using a variable to avoid mixing mutable and non-mutable borrows to
        // surfaces.
        if engine_updates.entities {
            render::update_entities(&state.show_surfaces, scene);
        }
    });

    engine_updates
}
