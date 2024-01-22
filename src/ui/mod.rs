use std::f64::consts::TAU;

use egui::{self, Button, Color32, RichText, Ui};
use graphics::{EngineUpdates, Scene};
use lin_alg2::f64::Vec3;

use crate::{
    basis_finder,
    basis_wfs::Basis,
    grid_setup::{new_data, new_data_real, Arr3d, Arr3dReal, Arr3dVec},
    potential, render,
    types::ComputationDevice,
    wf_ops,
    wf_ops::{DerivCalc, Spin},
    ActiveElec, State,
};

pub(crate) mod procedures;

const UI_WIDTH: f32 = 280.;
const SIDE_PANEL_SIZE: f32 = 400.;
const SLIDER_WIDTH: f32 = 260.;
const SLIDER_WIDTH_ORIENTATION: f32 = 100.;

const E_MIN: f64 = -4.5;
const E_MAX: f64 = 0.1;

// Wave fn weights
pub const WEIGHT_MIN: f64 = -1.7;
pub const WEIGHT_MAX: f64 = 1.3;

const _L_MIN: f64 = -3.;
const _L_MAX: f64 = 3.;

// sets range of -size to +size
const GRID_SIZE_MIN: f64 = 0.;
const GRID_SIZE_MAX: f64 = 40.;

const ITEM_SPACING: f32 = 18.;
const FLOAT_EDIT_WIDTH: f32 = 40.;

fn text_edit_float(val: &mut f64, _default: f64, ui: &mut Ui) {
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
    basis_fns: &mut [Basis],
    updated_evaled_wfs: &mut bool,
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
                *updated_evaled_wfs = true;
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
                *updated_evaled_wfs = true;
                *updated_basis_weights = true;
                *updated_charges = true;
                *updated_entities = true;
            }

            if ui.button(RichText::new("❌").color(Color32::RED)).clicked() {
                // Don't remove from a collection we're iterating over.
                charge_removed = Some(i);
                *updated_evaled_wfs = true;
                *updated_basis_weights = true;
                *updated_charges = true;
                // Update entities due to charge sphere placement.
                *updated_entities = true;
            }
        });
    }

    if let Some(charge_i_removed) = charge_removed {
        charges.remove(charge_i_removed);
        *updated_evaled_wfs = true;
        *updated_basis_weights = true;
        *updated_charges = true;
        *updated_entities = true;
    }

    if ui.add(Button::new("Add charge")).clicked() {
        charges.push((Vec3::new_zero(), crate::Q_PROT));
        *updated_evaled_wfs = true;
        *updated_basis_weights = true;
        *updated_charges = true;
        *updated_entities = true;
    }
}

/// Ui elements that allow mixing various basis WFs.
fn basis_fn_mixer(
    state: &mut State,
    updated_basis_weights: &mut bool,
    ui: &mut Ui,
    active_elec: usize,
) {
    // Select with charge (and its position) this basis fn is associated with.

    egui::containers::ScrollArea::vertical()
        .max_height(500.)
        .show(ui, |ui| {
            // We use this Vec to avoid double-mutable borrow issues.
            let mut bases_modified = Vec::new();

            for (basis_i, basis) in state.bases[active_elec].iter_mut().enumerate() {
                let mut recalc_this_basis = false;

                ui.horizontal(|ui| {
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

                        recalc_this_basis = true;
                        *updated_basis_weights = true;
                        // *updated_unweighted_basis_wfs = true;
                    }

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

                                // *updated_unweighted_basis_wfs = true;
                                recalc_this_basis = true;
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

                                // *updated_unweighted_basis_wfs = true;
                                recalc_this_basis = true;
                                *updated_basis_weights = true;
                            }

                            if b.xi != xi_prev {
                                // *updated_unweighted_basis_wfs = true;
                                recalc_this_basis = true;
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

                // Only update this particular basis; not all.
                if recalc_this_basis {
                    // Note: Extra memory use from this re-allocoating and cloning.
                    let mut temp_psi = vec![new_data(state.grid_n_render)];
                    let mut temp_psi_pp = vec![new_data(state.grid_n_render)];
                    let mut temp_psi_pp_div_psi = vec![new_data_real(state.grid_n_render)];

                    wf_ops::wf_from_bases(
                        &state.dev_psi,
                        &mut temp_psi,
                        Some(&mut temp_psi_pp),
                        Some(&mut temp_psi_pp_div_psi),
                        &[basis.clone()],
                        &state.surfaces_shared.grid_posits,
                        state.grid_n_render,
                        state.deriv_calc,
                    );

                    state.surfaces_per_elec[active_elec].psi_per_basis[basis_i] =
                        temp_psi.remove(0);
                    state.surfaces_per_elec[active_elec].psi_pp_per_basis[basis_i] =
                        temp_psi_pp.remove(0);
                    state.surfaces_per_elec[active_elec].psi_pp_div_psi_per_basis[basis_i] =
                        temp_psi_pp_div_psi.remove(0);
                }

                ui.add(
                    egui::Slider::from_get_set(WEIGHT_MIN..=WEIGHT_MAX, |v| {
                        if let Some(v_) = v {
                            *basis.weight_mut() = v_;
                            *updated_basis_weights = true;

                            bases_modified.push(basis_i);
                        }

                        basis.weight()
                    })
                    .text("Wt"),
                );

                // Re-compute this basis WF. Eg, after changing n, l, m, xi, or the associated electron.
                if ui.add(Button::new("C")).clicked() {}
            }

            if state.ui.weight_symmetry {
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

                        // wf_ops::update_wf_fm_bases(
                        //     &mut state.surfaces_per_elec[ae],
                        //     &state.bases_evaluated[ae],
                        //     state.surfaces_shared.E,
                        //     state.grid_n_render,
                        //     &weights,
                        // );

                        let E = if state.ui.adjust_E_with_weights {
                            None
                        } else {
                            Some(state.surfaces_shared.E)
                        };

                        let weights: Vec<f64> =
                            state.bases[ae].iter().map(|b| b.weight()).collect();
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
    updated_evaluated_wfs: &mut bool,
) {
    ui.horizontal(|ui| {
        // if ui.add(Button::new("Empty e- charge")).clicked() {
        //     state.charges_electron[active_elec] = grid_setup::new_data_real(state.grid_n);
        //
        //     *updated_meshes = true;
        // }

        if ui
            .add(Button::new("Create charge from this elec"))
            .clicked()
        {
            let weights: Vec<f64> = state.bases[ae].iter().map(|b| b.weight()).collect();
            procedures::create_elec_charge(
                &mut state.charges_from_electron[ae],
                &state.psi_charge[ae],
                &weights,
                state.grid_n_charge,
            );
        }

        if ui
            .add(Button::new("Update V acting on this elec"))
            .clicked()
        {
            procedures::update_V_acting_on_elec(state, ae);

            *updated_E_or_V = true;
            *updated_meshes = true;
        }

        if ui.add(Button::new("Find E")).clicked() {
            state.surfaces_shared.E = wf_ops::calc_E_from_bases(
                &state.bases[ae],
                state.surfaces_per_elec[ae].V_acting_on_this[0][0][0],
                state.surfaces_shared.grid_posits[0][0][0],
                state.deriv_calc,
            );

            *updated_E_or_V = true;
            *updated_meshes = true;
        }
    });

    if ui.add(Button::new("Find STO bases")).clicked() {
        let charges_other_elecs =
            wf_ops::combine_electron_charges(ae, &state.charges_from_electron, state.grid_n_charge);

        let sample_pts = basis_finder::generate_sample_pts();
        // let xis: Vec<f64> = state.bases[ae].iter().map(|b| b.xi()).collect();

        let (bases, E) = basis_finder::run(
            &state.dev_charge,
            &state.charges_fixed,
            &charges_other_elecs,
            &state.surfaces_shared.grid_posits_charge,
            state.grid_n_charge,
            &sample_pts,
            &state.bases[ae],
            state.deriv_calc,
        );

        state.surfaces_shared.E = E;

        state.bases[ae] = bases;
        // todo: Only reculate ones that are new; this recalculates all, when it's unlikely we need to do that.
        *updated_evaluated_wfs = true;

        *updated_E_or_V = true;
        *updated_basis_weights = true;
    }

    if ui.add(Button::new("He solver")).clicked() {
        procedures::he_solver(state);

        // todo: Only reculate ones that are new; this recalculates all, when it's unlikely we need to do that.
        *updated_evaluated_wfs = true;
        *updated_E_or_V = true;
        *updated_basis_weights = true;
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
                    bases_evaluated_charge,
                    surfaces_shared,
                    surfaces_per_elec,
                ) = crate::init_from_grid(
                    &state.dev_psi,
                    &state.dev_charge,
                    state.grid_range_render,
                    state.grid_range_charge,
                    state.sample_factor_render,
                    state.grid_n_render,
                    state.grid_n_charge,
                    &state.bases,
                    &state.charges_fixed,
                    state.num_elecs,
                    state.deriv_calc,
                );

                state.charges_from_electron = charges_electron;
                state.V_from_elecs = V_from_elecs;
                state.psi_charge = bases_evaluated_charge;
                state.surfaces_shared = surfaces_shared;
                state.surfaces_per_elec = surfaces_per_elec;

                for elec_i in 0..state.surfaces_per_elec.len() {
                    wf_ops::initialize_bases(&mut state.bases[elec_i], &state.charges_fixed, 2);
                }

                updated_evaluated_wfs = true;
                updated_meshes = true;
            }

            ui.add_space(20.);

            let prev_active_elec = state.ui.active_elec;
            // Combobox to select the active electron, or select the combined wave functino.
            let active_elec_text = match state.ui.active_elec {
                ActiveElec::Combined => "C".to_owned(),
                ActiveElec::PerElec(i) => i.to_string(),
            };

            egui::ComboBox::from_id_source(0)
                .width(30.)
                .selected_text(active_elec_text)
                .show_ui(ui, |ui| {
                    // A value to select the composite wave function.
                    ui.selectable_value(&mut state.ui.active_elec, ActiveElec::Combined, "C");

                    // A value for each individual electron
                    for i in 0..state.surfaces_per_elec.len() {
                        ui.selectable_value(
                            &mut state.ui.active_elec,
                            ActiveElec::PerElec(i),
                            i.to_string(),
                        );
                    }
                });

            // Show the new active electron's meshes, if it changed.
            if state.ui.active_elec != prev_active_elec {
                // Create charge from all other electrons.
                for ae in 0..state.surfaces_per_elec.len() {
                    // if ae == state.ui.active_elec {
                    //     continue
                    // }
                    let weights: Vec<f64> = state.bases[ae].iter().map(|b| b.weight()).collect();
                    procedures::create_elec_charge(
                        &mut state.charges_from_electron[ae],
                        &state.psi_charge[ae],
                        &weights,
                        state.grid_n_charge,
                    );
                }

                updated_E_or_V = true;
                updated_meshes = true;
            }

            if ui
                .checkbox(&mut state.ui.mag_phase, "Show mag, phase")
                .clicked()
            {
                updated_meshes = true;
            }

            if ui
                .checkbox(&mut state.ui.adjust_E_with_weights, "Auto adjust E")
                .clicked()
            {}
        });

        ui.horizontal(|ui| {
            // if ui
            //     .checkbox(&mut state.ui.auto_gen_elec_V, "Auto elec V")
            //     .clicked()
            // {}

            let mut deriv_numeric = state.deriv_calc == DerivCalc::Numeric;

            if ui.checkbox(&mut deriv_numeric, "ψ'' num").clicked() {
                // todo: recalc all bases.
                state.deriv_calc = if deriv_numeric {
                    DerivCalc::Numeric
                } else {
                    DerivCalc::Analytic
                };
            }

            if ui
                .checkbox(&mut state.ui.weight_symmetry, "Weight sym")
                .clicked()
            {}

            if ui
                .checkbox(&mut state.ui.create_2d_electron_V, "2D elec V")
                .clicked()
            {}

            if ui
                .checkbox(&mut state.ui.create_3d_electron_V, "3D elec V")
                .clicked()
            {}
        });

        ui.add_space(ITEM_SPACING);

        ui.horizontal(|ui| {
            // todo: Move this A/R. Function?
            match state.ui.active_elec {
                ActiveElec::PerElec(_) => {
                    ui.vertical(|ui| {
                        for (i, data) in state.surface_descs_per_elec.iter_mut().enumerate() {
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
                        for (i, data) in state.surface_descs_per_elec.iter_mut().enumerate() {
                            if i <= 3 || i > 6 {
                                continue;
                            }
                            if ui.checkbox(&mut data.visible, &data.name).clicked() {
                                engine_updates.entities = true;
                            }
                        }
                    });

                    ui.vertical(|ui| {
                        for (i, data) in state.surface_descs_per_elec.iter_mut().enumerate() {
                            if i <= 6 {
                                continue;
                            }
                            if ui.checkbox(&mut data.visible, &data.name).clicked() {
                                engine_updates.entities = true;
                            }
                        }
                    });
                }
                // todo: This whole section is DRY!
                ActiveElec::Combined => {
                    ui.vertical(|ui| {
                        for (i, data) in state.surface_descs_combined.iter_mut().enumerate() {
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
                        for (i, data) in state.surface_descs_combined.iter_mut().enumerate() {
                            if i <= 3 || i > 6 {
                                continue;
                            }
                            if ui.checkbox(&mut data.visible, &data.name).clicked() {
                                engine_updates.entities = true;
                            }
                        }
                    });

                    ui.vertical(|ui| {
                        for (i, data) in state.surface_descs_combined.iter_mut().enumerate() {
                            if i <= 6 {
                                continue;
                            }
                            if ui.checkbox(&mut data.visible, &data.name).clicked() {
                                engine_updates.entities = true;
                            }
                        }
                    });

                    // We currently use separate surfaces for these, vice the toggles.
                    // if ui
                    //     .checkbox(&mut state.ui.display_alpha, "Show α")
                    //     .clicked()
                    // {
                    //     updated_meshes = true;
                    // }
                    // if ui
                    //     .checkbox(&mut state.ui.display_beta, "Show β")
                    //     .clicked()
                    // {
                    //     updated_meshes = true;
                    // }
                }
            }
        });

        ui.add(
            // -0.1 is a kludge.
            egui::Slider::from_get_set(
                state.grid_range_render.0..=state.grid_range_render.1 - 0.1,
                |v| {
                    if let Some(v_) = v {
                        state.ui.z_displayed = v_;
                        updated_meshes = true;
                    }

                    state.ui.z_displayed
                },
            )
            .text("Z slice"),
        );

        ui.add(
            egui::Slider::from_get_set(-TAU / 2.0..=TAU / 2.0, |v| {
                if let Some(v_) = v {
                    state.ui.visual_rotation = v_;
                    updated_meshes = true;
                }

                state.ui.visual_rotation
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
                }

                state.grid_range_render.1
            })
            .text("Grid range"),
        );

        match state.ui.active_elec {
            ActiveElec::PerElec(ae) => {
                // ui.heading(format!(
                //     "ψ'' score: {:.10}",
                //     state.eval_data_per_elec[ae].score
                // ));

                ui.add(
                    egui::Slider::from_get_set(E_MIN..=E_MAX, |v| {
                        if let Some(v_) = v {
                            state.surfaces_shared.E = v_;

                            updated_meshes = true;
                            updated_E_or_V = true;
                        }

                        state.surfaces_shared.E
                    })
                    .text("E"),
                );

                let prev_spin = state.surfaces_per_elec[ae].spin;
                // Combobox to select the active electron, or select the combined wave functino.
                egui::ComboBox::from_id_source(0)
                    .width(30.)
                    .selected_text(prev_spin.to_string())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.surfaces_per_elec[ae].spin,
                            Spin::Alpha,
                            Spin::Alpha.to_string(),
                        );
                        ui.selectable_value(
                            &mut state.surfaces_per_elec[ae].spin,
                            Spin::Beta,
                            Spin::Beta.to_string(),
                        );
                    });

                if prev_spin != state.surfaces_per_elec[ae].spin {
                    wf_ops::update_combined(
                        &mut state.surfaces_shared,
                        &state.surfaces_per_elec,
                        state.grid_n_render,
                    );
                    updated_meshes = true;
                }

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

                basis_fn_mixer(state, &mut updated_basis_weights, ui, ae);

                ui.add_space(ITEM_SPACING);

                bottom_items(
                    ui,
                    state,
                    scene,
                    ae,
                    &mut updated_meshes,
                    &mut updated_basis_weights,
                    &mut updated_E_or_V,
                    &mut updated_evaluated_wfs,
                );

                // Code below handles various updates that were flagged above.

                if updated_fixed_charges {
                    procedures::update_fixed_charges(state, scene);
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
                        // &mut state.eval_data_per_elec[ae],
                        &mut state.surfaces_per_elec[ae],
                        &state.surfaces_shared.V_from_nuclei,
                        // state.eval_data_shared.grid_n,
                        state.grid_n_render,
                        state.surfaces_shared.E,
                    );
                }
            }

            ActiveElec::Combined => {
                // ui.add(
                //     egui::Slider::from_get_set(E_MIN..=E_MAX, |v| {
                //         if let Some(v_) = v {
                //             state.surfaces_shared.E = v_;
                //
                //             updated_E_or_V = true;
                //             updated_meshes = true;
                //         }
                //
                //         state.surfaces_shared.E
                //     })
                //     .text("E"),
                // );

                // Multiply wave functions together, and stores in Shared surfaces.
                // todo: This is an approximation
                // if ui.add(Button::new("Combine wavefunctions")).clicked() {
                //     procedures::combine_wfs(state);
                //     updated_meshes = true;
                //     engine_updates.entities = true;
                // }
            }
        }
        // Code here runs for both multi-electron, and combined states

        // Track using a variable to avoid mixing mutable and non-mutable borrows to
        // surfaces.
        if engine_updates.entities {
            match state.ui.active_elec {
                ActiveElec::PerElec(ae) => {
                    render::update_entities(
                        &state.charges_fixed,
                        &state.surface_descs_per_elec,
                        scene,
                    );
                }
                ActiveElec::Combined => {
                    render::update_entities(
                        &state.charges_fixed,
                        &state.surface_descs_combined,
                        scene,
                    );
                }
            }
        }

        if updated_meshes {
            procedures::update_meshes(state, scene, &mut engine_updates);
        }
    });

    engine_updates
}
