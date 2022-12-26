use egui::{self, Color32, RichText};

use graphics::{EngineUpdates, Scene};

use crate::{render, State, N, NUM_SURFACES};

const UI_WIDTH: f32 = 300.;
const SIDE_PANEL_SIZE: f32 = 400.;
const SLIDER_WIDTH: f32 = 200.;

const E_MIN: f64 = -4.;
const E_MAX: f64 = 4.;

// Wave fn weights
const WEIGHT_MIN: f64 = -6.;
const WEIGHT_MAX: f64 = 6.;

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
                    engine_updates.meshes = true;

                    let (surfaces, _names, psi_pp_score) =
                        crate::eval_wf(&state.wfs, &state.nuclei, state.E);

                    state.surfaces = surfaces;
                    state.psi_pp_score = psi_pp_score;

                    render::update_meshes(&state.surfaces, state.z_displayed, scene);
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

        ui.heading("Weights:");

        // We use this var to avoid mutable/unmutable borrow conflicts
        let mut updated_wfs = false;
        for wf in state.wfs.iter_mut() {
            let mut wf_entry = wf.0.to_string();

            let response = ui.add(egui::TextEdit::singleline(&mut wf_entry).desired_width(16.));
            if response.changed() {
                let mut ok = false;
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

        if updated_wfs {
            engine_updates.meshes = true;

            let (surfaces, _names, psi_pp_score) =
                crate::eval_wf(&state.wfs, &state.nuclei, state.E);

            state.surfaces = surfaces;
            state.psi_pp_score = psi_pp_score;

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
