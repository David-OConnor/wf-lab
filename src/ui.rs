use egui::{self, Color32, RichText};

use graphics::{EngineUpdates, Scene};

use crate::{render, State};

const UI_WIDTH: f32 = 300.;
const SIDE_PANEL_SIZE: f32 = 400.;
const SLIDER_WIDTH: f32 = 200.;

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

        let mut updated_shown = false;
        for (i, ((_surface, name), show)) in state.surfaces.iter_mut().enumerate() {
            if ui.checkbox(show, &*name).clicked() {
                updated_shown = true;
                engine_updates.entities = true;
            }
        }

        // Track using a variable to avoid mixing mutable and non-mutable borrows to
        // surfaces.
        if updated_shown {
            render::update_entities(&state.surfaces, scene);
        }
    });

    engine_updates
}
