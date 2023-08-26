//! This module handles 3D rendering, and the event loop.

use core::f32::consts::TAU;

use graphics::{
    self, Camera, ControlScheme, DeviceEvent, EngineUpdates, Entity, InputSettings, LightType,
    Lighting, Mesh, PointLight, Scene, UiLayout, UiSettings,
};

use lin_alg2::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
};

use crate::{complex_nums::Cplx, grid_setup::{new_data_real, Arr3d, Arr3dReal, Arr3dVec}, types::{SurfacesPerElec, SurfacesShared}, util, State, SurfaceData, NUM_SURFACES};

type Color = (f32, f32, f32);

const WINDOW_TITLE: &str = "ψ lab";
const WINDOW_SIZE_X: f32 = 1_600.;
const WINDOW_SIZE_Y: f32 = 1_200.;
const RENDER_DIST: f32 = 200.;
const BACKGROUND_COLOR: Color = (0.5, 0.5, 0.5);
// const SIDE_PANEL_SIZE: f32 = 400.;

const COLOR_POS_CHARGE: Color = (1., 0., 0.);
const COLOR_NEG_CHARGE: Color = (0., 0., 1.);

const COLOR_PSI_PP_CALC_1D: Color = (0., 1., 0.);
const COLOR_PSI_PP_MEAS_1D: Color = (0., 0.5, 0.5);

const CHARGE_SPHERE_SIZE: f32 = 0.05;

const SURFACE_COLORS: [Color; 10] = [
    (0., 0., 1.),
    (0., 0.5, 0.2),
    (1., 0., 0.),
    (0., 0.5, 0.5),
    (0., 0.3, 0.8),
    (0.5, 0.5, 0.),
    (0.33, 0.33, 0.333),
    (0.5, 0., 0.5),
    (0.5, 0.4, 0.2),
    (0.5, 0.4, 0.3),
];

const SURFACE_SHINYNESS: f32 = 10.5;
const CHARGE_SHINYNESS: f32 = 3.;

// To make the WF and other surfaces more visually significant.
const PSI_SCALER: f32 = 120.;
const PSI_SQ_SCALER: f32 = 1_000.;
const PSI_PP_SCALER: f32 = 20.;

const ELEC_CHARGE_SCALER: f32 = 600.;
// const ELEC_V_SCALER: f32 = 1100.;
const V_SCALER: f32 = 2.;

fn event_handler(
    _state: &mut State,
    _event: DeviceEvent,
    _scene: &mut Scene,
    _dt: f32,
) -> EngineUpdates {
    // match event {
    //     DeviceEvent::Key(key) => {}
    //     _ => (),
    // }
    EngineUpdates::default()
}

/// This runs each frame. Currently, no updates.
fn render_handler(_state: &mut State, _scene: &mut Scene, _dt: f32) -> EngineUpdates {
    // EngineUpdates::default()

    EngineUpdates {
        // compute: true,
        ..Default::default()
    }
}

/// Utility function to linearly map an input value to an output
pub fn _map_linear(val: f64, range_in: (f64, f64), range_out: (f64, f64)) -> f64 {
    // todo: You may be able to optimize calls to this by having the ranges pre-store
    // todo the total range vals.
    let portion = (val - range_in.0) / (range_in.1 - range_in.0);

    portion * (range_out.1 - range_out.0) + range_out.0
}

/// Generate a f32 mesh from a 3d F64 mesh, using a z slice. Replaces the z value with function value.
fn prepare_2d_mesh_real(
    posits: &Arr3dVec,
    vals: &Arr3dReal,
    z_i: usize,
    scaler: f32,
    grid_n: usize,
) -> Vec<Vec<Vec3>> {
    // todo: DRY from new_data fns. We have to repeat due to using f32 type instead of f64,
    // todo and 2d vice 3d.
    let mut y = Vec::new();
    y.resize(grid_n, Vec3::new_zero());

    let scaler = scaler / 30. * grid_n as f32; // todo why, without this, does higher grid n mean smaller?

    let mut result = Vec::new();
    result.resize(grid_n, y);

    for i in 0..grid_n {
        for j in 0..grid_n {
            result[i][j] = Vec3::new(
                posits[i][j][z_i].x as f32,
                posits[i][j][z_i].y as f32,
                vals[i][j][z_i] as f32 * scaler,
            );
        }
    }

    result
}

/// Generate a 2d f32 mesh from a 3d F64 mesh, using a z slice.
fn prepare_2d_mesh(
    posits: &Arr3dVec,
    vals: &Arr3d,
    z_i: usize,
    scaler: f32,
    mag_phase: bool,
    imag: bool,
    grid_n: usize,
) -> Vec<Vec<Vec3>> {
    // todo: DRY from new_data fns. We have to repeat due to using f32 type instead of f64.
    let mut y = Vec::new();
    y.resize(grid_n, Vec3::new_zero());

    let scaler = scaler / 30. * grid_n as f32; // todo why, without this, does higher grid n mean smaller?

    let mut result = Vec::new();
    result.resize(grid_n, y);

    for i in 0..grid_n {
        for j in 0..grid_n {
            //todo: Instead of real and imag, split by mag and phase??
            let val = if imag {
                if mag_phase {
                    vals[i][j][z_i].phase() / (scaler as f64)
                } else {
                    vals[i][j][z_i].im
                }
            } else if mag_phase {
                vals[i][j][z_i].mag()
            } else {
                vals[i][j][z_i].real
            };

            result[i][j] = Vec3::new(
                posits[i][j][z_i].x as f32,
                posits[i][j][z_i].y as f32,
                val as f32 * scaler,
            );
        }
    }

    result
}

/// Updates meshes. For example, when updating a plot due to changing parameters.
/// Note that this is where we decide which Z to render.
pub fn update_meshes(
    surfaces_shared: &SurfacesShared,
    surfaces: &SurfacesPerElec,
    z_displayed: f64,
    scene: &mut Scene,
    grid_posits: &Arr3dVec,
    mag_phase: bool,
    charges_electron: &Arr3dReal,
    grid_n: usize,
    // Render the combined wave function from all electrons.
    render_multi_elec: bool,
) {
    // Our meshes are defined in terms of a start point,
    // and a step. Adjust the step to center the grid at
    // the renderer's center.
    // const SFC_MESH_START: f32 = -4.;
    // let sfc_mesh_start = grid_min as f32; // todo: Sync graphics and atomic coords?
    // let sfc_mesh_step: f32 = -2. * sfc_mesh_start / N as f32;

    // `z_displayed` is a value float. Convert this to an index. Rounds to the nearest integer.
    // let z_i = map_linear(z_displayed, (grid_min, grid_max), (0., N as f64)) as usize;
    // todo: Using your new system, you can't use a linear map here!

    let mut z_i = 0;
    for i in 0..grid_posits.len() {
        if grid_posits[0][0][i].z > z_displayed {
            z_i = i;
            break;
        }
    }

    let mut meshes = Vec::new();

    // todo: Fix the DRY between multi and single-elec renders
    if render_multi_elec {
        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh_real(grid_posits, &surfaces_shared.V_total, z_i, V_SCALER, grid_n),
            true,
        ));

        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh(
                grid_posits,
                &surfaces_shared.psi.psi_marginal.on_pt,
                z_i,
                PSI_SCALER,
                mag_phase,
                false,
                grid_n,
            ),
            true,
        ));

        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh(
                grid_posits,
                &surfaces_shared.psi.psi_marginal.on_pt,
                z_i,
                PSI_SCALER,
                mag_phase,
                true,
                grid_n,
            ),
            true,
        ));

        let mut psi_sq = new_data_real(grid_n);
        util::norm_sq(&mut psi_sq, &surfaces_shared.psi.psi_marginal.on_pt, grid_n);

        // todo: Lots of DRY here that is fixable between multi-elec and single-elec
        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh_real(grid_posits, &psi_sq, z_i, PSI_SQ_SCALER, grid_n),
            true,
        ));

        for (scaler, sfc) in [
            (PSI_PP_SCALER, &surfaces_shared.psi_pp_calculated),
            (PSI_PP_SCALER, &surfaces_shared.psi_pp_measured),
        ] {
            meshes.push(Mesh::new_surface(
                &prepare_2d_mesh(grid_posits, sfc, z_i, scaler, mag_phase, false, grid_n),
                true,
            ));

            meshes.push(Mesh::new_surface(
                &prepare_2d_mesh(grid_posits, sfc, z_i, scaler, mag_phase, true, grid_n),
                true,
            ));
        }

        // meshes.push(Mesh::new_surface(
        //     &prepare_2d_mesh_real(
        //         grid_posits,
        //         charges_electron,
        //         z_i,
        //         ELEC_CHARGE_SCALER,
        //         grid_n,
        //     ),
        //     true,
        // ));
    } else {
        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh_real(
                grid_posits,
                &surfaces.V_acting_on_this,
                z_i,
                V_SCALER,
                grid_n,
            ),
            true,
        ));

        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh(
                grid_posits,
                &surfaces.psi.on_pt,
                z_i,
                PSI_SCALER,
                mag_phase,
                false,
                grid_n,
            ),
            true,
        ));

        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh(
                grid_posits,
                &surfaces.psi.on_pt,
                z_i,
                PSI_SCALER,
                mag_phase,
                true,
                grid_n,
            ),
            true,
        ));

        let mut psi_sq = new_data_real(grid_n);
        util::norm_sq(&mut psi_sq, &surfaces.psi.on_pt, grid_n);

        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh_real(grid_posits, &psi_sq, z_i, PSI_SQ_SCALER, grid_n),
            true,
        ));

        for (scaler, sfc) in [
            (PSI_PP_SCALER, &surfaces.psi_pp_calculated),
            (PSI_PP_SCALER, &surfaces.psi_pp_measured),
        ] {
            meshes.push(Mesh::new_surface(
                &prepare_2d_mesh(grid_posits, sfc, z_i, scaler, mag_phase, false, grid_n),
                true,
            ));

            meshes.push(Mesh::new_surface(
                &prepare_2d_mesh(grid_posits, sfc, z_i, scaler, mag_phase, true, grid_n),
                true,
            ));
        }

        // Experimenting with V_elec from a given psi.
        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh_real(grid_posits, &surfaces.aux1, z_i, V_SCALER, grid_n),
            true,
        ));

        // Experimenting with V_elec from a given psi.
        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh_real(grid_posits, &surfaces.aux2, z_i, V_SCALER, grid_n),
            true,
        ));

        // meshes.push(Mesh::new_surface(
        //     &prepare_2d_mesh_real(
        //         grid_posits,
        //         charges_electron,
        //         z_i,
        //         ELEC_CHARGE_SCALER,
        //         grid_n,
        //     ),
        //     true,
        // ));
    }

    meshes.push(Mesh::new_sphere(CHARGE_SPHERE_SIZE, 12, 12));

    scene.meshes = meshes;
}

/// Updates entities, but not meshes. For example, when hiding or
/// showing a mesh.
/// Note that currently, we update charge displays along
/// with surfaces.
pub fn update_entities(
    charges: &[(Vec3F64, f64)],
    surface_data: &[SurfaceData],
    psi_pp_calc_1d: &[Cplx],
    psi_pp_meas_1d: &[Cplx],
    posits_1d: &[Vec3F64],
    scene: &mut Scene,
) {
    let mut entities = Vec::new();
    for i in 0..NUM_SURFACES {
        if !surface_data[i].visible {
            continue;
        }
        entities.push(Entity::new(
            i,
            Vec3::new_zero(),
            Quaternion::new_identity(),
            1.,
            SURFACE_COLORS[i],
            SURFACE_SHINYNESS,
        ));
    }

    for (posit, val) in charges {
        entities.push(Entity::new(
            NUM_SURFACES, // Index 1 after surfaces.
            Vec3::new(
                posit.x as f32,
                // We invert Y and Z due to diff coord systems
                // between the meshes and the renderer.
                posit.z as f32,
                posit.y as f32,
            ),
            Quaternion::new_identity(),
            6.,
            // todo: More fine-grained shading
            if *val > 0. {
                COLOR_POS_CHARGE
            } else {
                COLOR_NEG_CHARGE
            },
            CHARGE_SHINYNESS,
        ));
    }

    // todo: Im comps if ticked
    // todo: How to handle Z comp?

    // todo: This needs to be called whenever you update eleval data per elec
    // todo as of now, it is not updated appropriately.
    for (i, posit) in posits_1d.iter().enumerate() {
        entities.push(Entity::new(
            NUM_SURFACES, // Index 1 after surfaces.
            Vec3::new(
                posit.x as f32,
                // We invert Y and Z due to diff coord systems
                // between the meshes and the renderer.
                psi_pp_calc_1d[i].real as f32,
                posit.y as f32,
            ),
            Quaternion::new_identity(),
            3.,
            COLOR_PSI_PP_CALC_1D,
            CHARGE_SHINYNESS,
        ));

        entities.push(Entity::new(
            NUM_SURFACES, // Index 1 after surfaces.
            Vec3::new(
                posit.x as f32,
                psi_pp_meas_1d[i].real as f32,
                posit.y as f32,
            ),
            Quaternion::new_identity(),
            4.,
            COLOR_PSI_PP_MEAS_1D,
            CHARGE_SHINYNESS,
        ));
    }

    scene.entities = entities;
}

/// Entry point to our render and event loop.
pub fn render(state: State) {
    let mut scene = Scene {
        meshes: Vec::new(),   // updated below.
        entities: Vec::new(), // updated beloCw.
        camera: Camera {
            fov_y: TAU / 8.,
            position: Vec3::new(0., 10., -40.),
            far: RENDER_DIST,
            orientation: Quaternion::from_axis_angle(Vec3::new(1., 0., 0.), TAU / 16.),
            ..Default::default()
        },
        lighting: Lighting {
            ambient_color: [-1., 1., 1., 0.5],
            ambient_intensity: 0.03,
            point_lights: vec![
                // // Light from above. The sun?
                // PointLight {
                //     type_: LightType::Omnidirectional,
                //     position: Vec3::new(0., 100., 0.),
                //     diffuse_color: [0.6, 0.4, 0.3, 1.],
                //     specular_color: [0.6, 0.4, 0.3, 1.],
                //     diffuse_intensity: 4_000.,
                //     specular_intensity: 10_000.,
                // },

                // Light from above and to a side.
                PointLight {
                    type_: LightType::Omnidirectional,
                    position: Vec3::new(30., 50., 30.),
                    diffuse_color: [0.3, 0.4, 0.5, 1.],
                    specular_color: [0.3, 0.4, 0.5, 1.],
                    diffuse_intensity: 8_000.,
                    specular_intensity: 30_000.,
                },
                // Light from below
                PointLight {
                    type_: LightType::Omnidirectional,
                    position: Vec3::new(20., -50., 0.),
                    diffuse_color: [0.3, 0.4, 0.5, 1.],
                    specular_color: [0.3, 0.4, 0.5, 1.],
                    diffuse_intensity: 5_000.,
                    specular_intensity: 20_000.,
                },
            ],
        },
        background_color: BACKGROUND_COLOR,
        window_size: (WINDOW_SIZE_X, WINDOW_SIZE_Y),
        window_title: WINDOW_TITLE.to_owned(),
    };

    // todo: Is this where we want to do the sum? Probably not!!!

    // todo!
    // let surfaces = if state.ui_render_all_elecs {
    //     &state.surfaces_shared
    // } else {
    //     &state.surfaces_per_elec[state.ui_active_elec]
    // };

    let active_elec_init = 0; // todo?

    let surfaces = &state.surfaces_per_elec[active_elec_init];

    update_meshes(
        &state.surfaces_shared,
        surfaces,
        state.ui_z_displayed,
        &mut scene,
        &state.surfaces_shared.grid_posits,
        state.mag_phase,
        &state.charges_electron[active_elec_init],
        state.grid_n_render,
        false,
    );

    update_entities(
        &state.charges_fixed,
        &state.surface_data,
        &state.eval_data_per_elec[active_elec_init].psi_pp_calc,
        &state.eval_data_per_elec[active_elec_init].psi_pp_meas,
        &state.eval_data_shared.posits,
        &mut scene,
    );

    let input_settings = InputSettings {
        initial_controls: ControlScheme::FreeCamera,
        ..Default::default()
    };
    let ui_settings = UiSettings {
        layout: UiLayout::Left,
        // todo: How to handle this? For blocking keyboard and moues inputs when over the UI.
        // width: gui::UI_WIDTH as f64, // todo: Not working correctly.
        size: 0., // todo: Bad API here.
        icon_path: Some("./resources/icon.png".to_owned()),
    };

    graphics::run(
        state,
        scene,
        input_settings,
        ui_settings,
        render_handler,
        event_handler,
        crate::ui::ui_handler,
        include_str!("shader_compute.wgsl"),
    );
}
