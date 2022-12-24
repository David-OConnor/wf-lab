use egui::{self, Color32, RichText};

use graphics::{EngineUpdates, Scene};

use crate::{render, State};

const UI_WIDTH: f32 = 300.;
const SIDE_PANEL_SIZE: f32 = 400.;
const SLIDER_WIDTH: f32 = 200.;

const E_MIN: f64 = -3.;
const E_MAX: f64 = 3.;
const Z_MIN: f64 = -10.;
const Z_MAX: f64 = 5.;

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

        for (i, ((_surface, name), show)) in state.surfaces.iter_mut().enumerate() {
            if ui.checkbox(show, &*name).clicked() {
                engine_updates.entities = true;
            }
        }

        ui.add(
            egui::Slider::from_get_set(E_MIN..=E_MAX, |v| {
                if let Some(v_) = v {
                    state.E = v_;
                    engine_updates.meshes = true;
                    // engine_updates.entities = true; // todo?

                    // todo: Don't reset shown state.
                    let data = crate::eval_wf(state.z, state.E);

                    render::update_meshes(&state.surfaces, scene);

                    state.surfaces = data.0.into_iter().map(|s| (s, true)).collect();

                    state.psi_pp_score = data.1;

                    render::update_meshes(&state.surfaces, scene);
                }

                state.E
            })
            .text("Energy"),
        );

        ui.add(
            egui::Slider::from_get_set(Z_MIN..=Z_MAX, |v| {
                if let Some(v_) = v {
                    state.z = v_;
                    engine_updates.meshes = true;
                    // engine_updates.entities = true; // todo?
                    // todo: Don't reset shown state.
                    let data = crate::eval_wf(state.z, state.E);

                    state.surfaces = data.0.into_iter().map(|s| (s, true)).collect();
                    state.psi_pp_score = data.1;

                    render::update_meshes(&state.surfaces, scene);
                }

                state.z
            })
            .text("Z slice"),
        );

        ui.heading(format!("ψ'' score: {:.4}", state.psi_pp_score));

        // Track using a variable to avoid mixing mutable and non-mutable borrows to
        // surfaces.
        if engine_updates.entities {
            render::update_entities(&state.surfaces, scene);
        }
    });

    engine_updates
}
