//! This module handles 3D rendering, and the event loop.

use core::f32::consts::TAU;

use graphics::{
    self, Camera, ControlScheme, DeviceEvent, EngineUpdates, Entity, InputSettings, LightType,
    Lighting, Mesh, PointLight, Scene, UiSettings,
};

use lin_alg2::{
    f32::{Quaternion, Vec3},
    f64::Vec3 as Vec3F64,
};

use crate::{State, GRID_MAX, GRID_MIN};

type Color = (f32, f32, f32);

const WINDOW_TITLE: &str = "ψ lab";
const WINDOW_SIZE_X: f32 = 1_600.;
const WINDOW_SIZE_Y: f32 = 1_000.;
const RENDER_DIST: f32 = 100.;
const BACKGROUND_COLOR: Color = (0.5, 0.5, 0.5);
const SIDE_PANEL_SIZE: f32 = 400.;

const COLOR_POS_CHARGE: Color = (1., 0., 0.);
const COLOR_NEG_CHARGE: Color = (0., 0., 1.);

const CHARGE_SPHERE_SIZE: f32 = 0.05;

const SURFACE_COLORS: [Color; 7] = [
    (0., 0., 1.),
    (0., 0.5, 0.2),
    (1., 0., 0.),
    (0., 0.5, 0.5),
    (0.5, 0.5, 0.),
    (0.33, 0.33, 0.333),
    (0.5, 0., 0.5),
];

const SURFACE_SHINYNESS: f32 = 1.5;
const CHARGE_SHINYNESS: f32 = 3.;

const PSI_SCALER: f32 = 4.; // to make WF more visually significant.
const ELEC_V_SCALER: f32 = 3_000_000.; // to make WF more visually significant.

// Our meshes are defined in terms of a start point,
// and a step. Adjust the step to center the grid at
// the renderer's center.
// const SFC_MESH_START: f32 = -4.;
const SFC_MESH_START: f32 = crate::GRID_MIN as f32; // todo: Sync graphics and atomic coords?
const SFC_MESH_STEP: f32 = -2. * SFC_MESH_START / crate::N as f32;

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
    EngineUpdates::default()
}

/// Utility function to linearly map an input value to an output
pub fn map_linear(val: f64, range_in: (f64, f64), range_out: (f64, f64)) -> f64 {
    // todo: You may be able to optimize calls to this by having the ranges pre-store
    // todo the total range vals.
    let portion = (val - range_in.0) / (range_in.1 - range_in.0);

    portion * (range_out.1 - range_out.0) + range_out.0
}

/// Generate a 2d f32 mesh from a 3d F64 mesh, using a z slice.
fn prepare_2d_mesh_real(surface: &crate::Arr3dReal, z_i: usize, scaler: f32) -> Vec<Vec<f32>> {
    let mut result = Vec::new();
    for i in 0..crate::N {
        let mut y_vec = Vec::new();
        for j in 0..crate::N {
            y_vec.push(surface[i][j][z_i] as f32 * scaler); // Convert from f64.
        }
        result.push(y_vec);
    }

    result
}

/// Generate a 2d f32 mesh from a 3d F64 mesh, using a z slice.
fn prepare_2d_mesh(surface: &crate::Arr3d, z_i: usize, scaler: f32) -> Vec<Vec<f32>> {
    let mut result = Vec::new();
    for i in 0..crate::N {
        let mut y_vec = Vec::new();
        for j in 0..crate::N {
            // We are only plotting the real part for now.
            y_vec.push(surface[i][j][z_i].real as f32 * scaler); // Convert from f64.
        }
        result.push(y_vec);
    }

    result
}

/// Updates meshes. For example, when updating a plot due to changing parameters.
/// Note that this is where we decide which Z to render.
pub fn update_meshes(
    surfaces: &crate::Surfaces,
    // surfaces: &[crate::Arr3d; crate::NUM_SURFACES],
    z_displayed: f64,
    scene: &mut Scene,
) {
    // `z_displayed` is a value float. Convert this to an index.
    let z_i = map_linear(z_displayed, (GRID_MIN, GRID_MAX), (0., crate::N as f64)) as usize;

    let mut meshes = Vec::new();

    // todo: You shouldn't update each mesh every time! Only update
    // todo the meshes that change.

    meshes.push(Mesh::new_surface(
        &prepare_2d_mesh_real(&surfaces.V, z_i, 1.),
        // todo: Center! Maybe offset in entities.
        SFC_MESH_START,
        SFC_MESH_STEP,
        true,
    ));

    meshes.push(Mesh::new_surface(
        &prepare_2d_mesh(&surfaces.psi, z_i, PSI_SCALER),
        // todo: Center! Maybe offset in entities.
        SFC_MESH_START,
        SFC_MESH_STEP,
        true,
    ));

    for sfc in [
        &surfaces.psi_pp_calculated,
        &surfaces.psi_pp_measured,
        &surfaces.aux1,
        // &surfaces.aux2,
    ] {
        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh(sfc, z_i, 1.),
            // todo: Center! Maybe offset in entities.
            SFC_MESH_START,
            SFC_MESH_STEP,
            true,
        ));
    }

    meshes.push(Mesh::new_surface(
        &prepare_2d_mesh_real(&surfaces.aux2, z_i, ELEC_V_SCALER),
        // todo: Center! Maybe offset in entities.
        SFC_MESH_START,
        SFC_MESH_STEP,
        true,
    ));

    meshes.push(Mesh::new_sphere(CHARGE_SPHERE_SIZE, 8, 8));

    scene.meshes = meshes;
}

/// Updates entities, but not meshes. For example, when hiding or
/// showing a mesh.
/// Note that currently, we update charge displays along
/// with surfaces.
pub fn update_entities(charges: &[(Vec3F64, f64)], show_surfaces: &[bool], scene: &mut Scene) {
    let mut entities = Vec::new();
    for i in 0..crate::NUM_SURFACES {
        if !show_surfaces[i] {
            continue;
        }
        entities.push(Entity::new(
            i,
            Vec3::new_zero(),
            // Quaternion::from_axis_angle(Vec3::new(-1., 0., 0.), TAU / 4.),
            Quaternion::new_identity(),
            1.,
            SURFACE_COLORS[i],
            SURFACE_SHINYNESS,
        ));
    }

    for (posit, val) in charges {
        entities.push(Entity::new(
            crate::NUM_SURFACES, // Index 1 after surfaces.
            // Todo: You may need to scale posit's Z.
            // todo: Heper for this?
            Vec3::new(
                posit.x as f32,
                // We invert Y and Z due to diff coord systems
                // between the meshes and the renderer.
                posit.z as f32,
                posit.y as f32,
            ),
            Quaternion::new_identity(),
            1.,
            // todo: More fine-grained shading
            if *val > 0. {
                COLOR_POS_CHARGE
            } else {
                COLOR_NEG_CHARGE
            },
            CHARGE_SHINYNESS,
        ));
    }

    scene.entities = entities;
}

/// Entry point to our render and event loop.
pub fn render(state: State) {
    let mut scene = Scene {
        meshes: Vec::new(),   // updated below.
        entities: Vec::new(), // updated below.
        camera: Camera {
            fov_y: TAU / 8.,
            position: Vec3::new(0., 6., -15.),
            far: RENDER_DIST,
            orientation: Quaternion::from_axis_angle(Vec3::new(1., 0., 0.), TAU / 16.),
            ..Default::default()
        },
        lighting: Lighting {
            ambient_color: [-1., 1., 1., 0.5],
            ambient_intensity: 0.03,
            point_lights: vec![
                // Light from above. The sun?
                PointLight {
                    type_: LightType::Omnidirectional,
                    position: Vec3::new(0., 100., 0.),
                    diffuse_color: [0.6, 0.4, 0.3, 1.],
                    specular_color: [0.6, 0.4, 0.3, 1.],
                    diffuse_intensity: 10_000.,
                    specular_intensity: 10_000.,
                },
                PointLight {
                    type_: LightType::Omnidirectional,
                    position: Vec3::new(30., 100., 30.),
                    diffuse_color: [0.3, 0.4, 0.5, 1.],
                    specular_color: [0.3, 0.4, 0.5, 1.],
                    diffuse_intensity: 10_000.,
                    specular_intensity: 10_000.,
                },
            ],
        },
        background_color: BACKGROUND_COLOR,
        window_size: (WINDOW_SIZE_X, WINDOW_SIZE_Y),
        window_title: WINDOW_TITLE.to_owned(),
    };

    update_meshes(&state.surfaces, state.z_displayed, &mut scene);
    update_entities(&state.charges, &state.show_surfaces, &mut scene);

    let input_settings = InputSettings {
        initial_controls: ControlScheme::FreeCamera,
        ..Default::default()
    };
    let ui_settings = UiSettings {
        // todo: How to handle this? For blocking keyboard and moues inputs when over the UI.
        // width: gui::UI_WIDTH as f64, // todo: Not working correctly.
        width: 500.,
        icon_path: None,
    };

    graphics::run(
        state,
        scene,
        input_settings,
        ui_settings,
        render_handler,
        event_handler,
        crate::ui::ui_handler,
    );
}
