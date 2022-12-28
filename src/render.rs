//! This module handles 3D rendering, and the event loop.

use core::f32::consts::TAU;

use graphics::{
    self, Camera, ControlScheme, DeviceEvent, EngineUpdates, Entity, InputSettings, LightType,
    Lighting, Mesh, PointLight, Scene, UiSettings,
};

use lin_alg2::f32::{Quaternion, Vec3};

use crate::{State, GRID_MAX, GRID_MIN};

const WINDOW_TITLE: &str = "ψ lab";
const WINDOW_SIZE_X: f32 = 1_100.;
const WINDOW_SIZE_Y: f32 = 800.;
const RENDER_DIST: f32 = 100.;
const BACKGROUND_COLOR: (f32, f32, f32) = (0.5, 0.5, 0.5);
const SIDE_PANEL_SIZE: f32 = 400.;

const SURFACE_COLORS: [(f32, f32, f32); 7] = [
    (0., 0., 1.),
    (0., 1., 0.),
    (1., 0., 0.),
    (0., 0.5, 0.5),
    (0.5, 0.5, 0.),
    (0.5, 0.5, 0.),
    (0.5, 0., 0.5),
];

const SURFACE_SHINYNESS: f32 = 1.;

fn event_handler(
    state: &mut State,
    event: DeviceEvent,
    scene: &mut Scene,
    dt: f32,
) -> EngineUpdates {
    // todo: Higher level api from winit or otherwise instead of scancode?
    let mut entities_changed = false;

    // let rotation_amt = crate::BOND_ROTATION_SPEED * dt as f64;

    match event {
        DeviceEvent::Key(key) => {}
        _ => (),
    }
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
fn prepare_2d_mesh(surface: &crate::Arr3d, z_i: usize) -> Vec<Vec<f32>> {
    let mut result = Vec::new();
    for i in 0..crate::N {
        let mut y_vec = Vec::new();
        for j in 0..crate::N {
            y_vec.push(surface[i][j][z_i] as f32); // Convert from f64.
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

    for sfc in [
        &surfaces.V,
        &surfaces.psi,
        &surfaces.psi_pp_calculated,
        &surfaces.psi_pp_measured,
    ] {
        meshes.push(Mesh::new_surface(
            &prepare_2d_mesh(sfc, z_i),
            -4.,
            0.1,
            true,
        ));
    }

    // for surface in surfaces.into_iter() {
    //     let mut surface_2d = Vec::new();
    //     // todo: Temp: Converting arr to vec. And indexing to the correct Z value.
    //     for x in *surface {
    //         let mut x_vec = Vec::new();
    //         for y in x {
    //             x_vec.push(y[z_i] as f32); // Convert from f64.
    //         }
    //         surface_2d.push(x_vec);
    //     }

    //     meshes.push(Mesh::new_surface(&surface_2d, -4., 0.1, true));
    // }
    scene.meshes = meshes;
}

/// Updates entities, but not meshes. For example, when hiding or showing a mesh.
pub fn update_entities(
    // surfaces: &[crate::Arr3d; crate::NUM_SURFACES],
    show_surfaces: &[bool; crate::NUM_SURFACES],
    scene: &mut Scene,
) {
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
    scene.entities = entities;
}

/// Entry point to our render and event loop.
pub fn render(state: State) {
    let mut scene = Scene {
        meshes: Vec::new(),   // updated below.
        entities: Vec::new(), // updated below.
        camera: Camera {
            fov_y: TAU as f32 / 8.,
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
        ..Default::default()
    };

    update_meshes(&state.surfaces, state.z_displayed, &mut scene);
    update_entities(&state.show_surfaces, &mut scene);

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
