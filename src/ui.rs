use egui::{self, Color32, RichText};

use graphics::{EngineUpdates, Scene};

use crate::{render, State, N, NUM_SURFACES};

const UI_WIDTH: f32 = 300.;
const SIDE_PANEL_SIZE: f32 = 400.;
const SLIDER_WIDTH: f32 = 260.;

const E_MIN: f64 = -2.;
const E_MAX: f64 = 2.;

// Wave fn weights
const WEIGHT_MIN: f64 = -3.;
const WEIGHT_MAX: f64 = 3.;

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

        ui.heading("Wavefunction Lab");

        ui.heading("Show surfaces:");

        for (i, name) in state.surface_names.iter_mut().enumerate() {
            let mut show = &mut state.show_surfaces[i];
            if ui.checkbox(&mut show, &*name).clicked() {
                engine_updates.entities = true;
            }
        }

        ui.add_space(ITEM_SPACING);

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
                                unsafe {
                                    crate::psi_pp_calculated[i][j][k] =
                                        crate::find_psi_pp_calc(state.E, i, j, k)
                                };
                            }
                        }
                    }

                    // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);
                    // state.psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, state.E);
                    state.psi_pp_score = crate::score_wf(&state.surfaces[2], &state.surfaces[3]);

                    render::update_meshes(&state.surfaces, state.z_displayed, scene);
                    engine_updates.meshes = true;
                }

                state.E
            })
            .text("Energy"),
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

        ui.add_space(ITEM_SPACING);

        ui.heading("H orbitals:");

        ui.heading("Wave functions and Weights:");

        // We use this var to avoid mutable/unmutable borrow conflicts
        let mut updated_wfs = false;
        for wf in state.wfs.iter_mut() {
            let mut wf_entry = wf.0.to_string();

            let response = ui.add(egui::TextEdit::singleline(&mut wf_entry).desired_width(16.));
            if response.changed() {
                match wf_entry.parse() {
                    Ok(v) => {
                        wf.0 = v;
                        updated_wfs = true;
                    }
                    Err(_) => (),
                }
            }

            ui.add(
                egui::Slider::from_get_set(WEIGHT_MIN..=WEIGHT_MAX, |v| {
                    if let Some(v_) = v {
                        wf.1 = v_;
                        updated_wfs = true;
                    }

                    wf.1
                })
                .text("Weight"),
            );
        }

        ui.add_space(ITEM_SPACING);

        if ui.add(egui::Button::new("Nudge WF")).clicked() {
            // crate::nudge_wf(&mut state.surfaces[1], &state.surfaces[2], &state.surfaces[3]);
            // crate::nudge_wf(&mut state.surfaces);
            crate::nudge_wf(&state.wfs, &state.charges, state.E);

            // todo: DRY
            engine_updates.meshes = true;

            state.psi_pp_score = crate::score_wf(&state.surfaces[2], &state.surfaces[3]);

            // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);
            // state.psi_pp_score  = crate::eval_wf(&state.wfs, &state.charges, state.E);

            render::update_meshes(&state.surfaces, state.z_displayed, scene);
        }

        if updated_wfs {
            engine_updates.meshes = true;

            // let psi_pp_score = crate::eval_wf(&state.wfs, &state.charges, &mut state.surfaces, state.E);

            crate::eval_wf(&state.wfs, &state.charges, state.E);

            state.psi_pp_score = crate::score_wf(&state.surfaces[2], &state.surfaces[3]);

            render::update_meshes(&state.surfaces, state.z_displayed, scene);
        }

        ui.add_space(ITEM_SPACING);

        ui.heading(format!("ψ'' score: {:.7}", state.psi_pp_score));

        // Track using a variable to avoid mixing mutable and non-mutable borrows to
        // surfaces.
        if engine_updates.entities {
            render::update_entities(&state.surfaces, &state.show_surfaces, scene);
        }
    });

    engine_updates
}
