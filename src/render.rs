//! This module handles 3D rendering, and the event loop.

use core::f32::consts::TAU;

use graphics::{
    self, Camera, ControlScheme, DeviceEvent, ElementState, EngineUpdates, Entity, InputSettings,
    LightType, Lighting, Mesh, PointLight, Scene, UiSettings,
};

use lin_alg2::f32::{Vec3, Quaternion};

use crate::State;

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
    let mut lighting_changed = false;

    // let rotation_amt = crate::BOND_ROTATION_SPEED * dt as f64;

    match event {
        DeviceEvent::Key(key) => {}
        _ => (),
    }
    EngineUpdates::default()
}

/// This runs each frame. Update our time-based simulation here.
fn render_handler(state: &mut State, scene: &mut Scene, dt: f32) -> EngineUpdates {
    let mut entities_changed = false;

    // scene.entities = generate_entities(&state);

    entities_changed = true;

    EngineUpdates {
        entities: entities_changed,
        camera: false,
        lighting: false,
    }
}

pub fn render(state: State, surfaces: &Vec<crate::arr_2d>) {
    let mut meshes = Vec::new();
    let mut entities = Vec::new();

    for (i, surface) in surfaces.into_iter().enumerate() {
        let mut surface_vec = Vec::new();
        // todo: Temp: Converting arr to vec.
        for row in surface {
            let mut row_vec = Vec::new();
            for val in row {
                row_vec.push(*val);
            }
            surface_vec.push(row_vec);
        }

        meshes.push(
            Mesh::new_surface(
                &surface_vec,
                -4.,
                0.1,
                true,
            )
        );

        entities.push(
            Entity::new(
                i,
                Vec3::new_zero(),
                // Quaternion::from_axis_angle(Vec3::new(-1., 0., 0.), TAU / 4.),
                Quaternion::new_identity(),
                1.,
                SURFACE_COLORS[i],
                SURFACE_SHINYNESS,
            )
        );
    }

    let scene = Scene {
        meshes,
        entities,
        camera: Camera {
            fov_y: TAU as f32 / 8.,
            position: Vec3::new(0., 6., -15.),
            far: RENDER_DIST,
            // orientation: QuatF32::from
            ..Default::default()
        },
        lighting: Lighting {
            ambient_color: [-1., 1., 1., 0.5],
            ambient_intensity: 0.05,
            point_lights: vec![
                // Light from above. The sun?
                PointLight {
                    type_: LightType::Omnidirectional,
                    position: Vec3::new(0., 20., 0.),
                    diffuse_color: [0.6, 0.4, 0.3, 1.],
                    specular_color: [0.6, 0.4, 0.3, 1.],
                    diffuse_intensity: 15_000.,
                    specular_intensity: 15_000.,
                },
            ],
        },
        background_color: BACKGROUND_COLOR,
        window_size: (WINDOW_SIZE_X, WINDOW_SIZE_Y),
        window_title: WINDOW_TITLE.to_owned(),
        ..Default::default()
    };

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
