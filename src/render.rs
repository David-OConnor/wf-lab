//! This module handles 3D rendering, and the event loop.

use core::f32::consts::TAU;

use graphics::{
    self, Camera, ControlScheme, DeviceEvent, EngineUpdates, Entity, InputSettings, LightType,
    Lighting, Mesh, PointLight, Scene, UiLayout, UiSettings,
};
use lin_alg::{
    f32::{Quaternion, Vec3, UP},
    f64::Vec3 as Vec3F64,
};

use crate::{
    grid_setup::{Arr2d, Arr2dReal, Arr2dVec, Arr3dReal, Arr3dVec},
    iter_arr,
    state::State,
    types::{SurfacesPerElec, SurfacesShared},
    ui::ui_handler,
    Axis, SurfaceDesc, SurfaceToRender,
};

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

const CHARGE_SPHERE_SIZE: f32 = 0.01;
const CHARGE_DENSITY_SPHERE_SIZE: f32 = 0.04;

const SURFACE_COLORS: [Color; 18] = [
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
    (0.3, 0.9, 0.3),
    (0.2, 0.8, 0.3),
    (0.8, 0.2, 0.3),
    (0.8, 0.2, 0.3),
    (0.8, 0.5, 0.8),
    (0.8, 0.2, 0.8),
    (0.1, 0.5, 0.9),
    (0.2, 0.2, 0.9),
];

const SURFACE_SHINYNESS: f32 = 10.5;
const CHARGE_SHINYNESS: f32 = 3.;

// To make the WF and other surfaces more visually significant.
// const PSI_SCALER: f32 = 100.;
const PSI_SCALER: f32 = 5.;
const CHARGE_DENSITY_SCALER: f32 = 10_000.;
const PSI_PP_SCALER: f32 = 5.;

// const V_SCALER: f32 = 1.;
const V_SCALER: f32 = 0.05;

pub(crate) const N_CHARGE_BALLS: usize = 1_000;

/// Add entities other than surfaces. For example: A gradient field, markers for grid positions,
/// markers for charge density etc.
fn add_non_surface_entities(
    entities: &mut Vec<Entity>,
    charge_density_balls: &[Vec3F64],
    gradient: &Arr3dVec,
    grid_posits_gradient: &Arr3dVec,
    mesh_sphere_i: usize,
    vector_arrow_i: usize,
) {
    // A grid, which helps visualizations.
    for grid_marker in &[
        Vec3::new(1., -1., 0.),
        Vec3::new(1., 0., 0.),
        Vec3::new(1., 1., 0.),
        Vec3::new(1., 2., 0.),
        Vec3::new(2., -1., 0.),
        Vec3::new(2., 0., 0.),
        Vec3::new(2., 1., 0.),
        Vec3::new(2., 2., 0.),
        Vec3::new(3., -1., 0.),
        Vec3::new(3., 0., 0.),
        Vec3::new(3., 1., 0.),
        Vec3::new(3., 2., 0.),
    ] {
        entities.push(Entity::new(
            mesh_sphere_i,
            *grid_marker,
            Quaternion::new_identity(),
            2.,
            // todo: More fine-grained shading
            (1., 0., 1.),
            CHARGE_SHINYNESS,
        ));
    }

    for posit in charge_density_balls {
        entities.push(Entity::new(
            mesh_sphere_i,
            Vec3::new(
                posit.x as f32,
                // todo: QC this
                // We invert Y and Z due to diff coord systems
                // between the meshes and the renderer.
                posit.z as f32,
                posit.y as f32,
            ),
            Quaternion::new_identity(),
            0.5,
            (0.2, 1., 0.5),
            CHARGE_SHINYNESS,
        ));
    }

    // Add a vector field
    // todo: Flux lines in adddition to or instead of this?
    for (i, j, k) in iter_arr!(grid_posits_gradient.len()) {
        let posit = grid_posits_gradient[i][j][k];

        // Swap Y and Z axis due to the renderer's different coord system.
        let posit = Vec3::new(posit.x as f32, posit.z as f32, posit.y as f32);

        let mut grad = gradient[i][j][k];

        // Note: We don't need any information on the rotation around the arrow's axis,
        // so we have an unused degree of freedom in this quaternion.
        // todo: QC this. I believe the starting vec should be oriented with the arrow in
        // todo the mesh. (Although you may have to do a coordinate conversion.
        let arrow_orientation = Quaternion::from_unit_vecs(
            Vec3::new(0., 1., 0.), // this instead of "UP" due to the different coordinate system.
            Vec3::new(grad.x as f32, grad.z as f32, grad.y as f32).to_normalized(),
        );

        entities.push(Entity::new(
            vector_arrow_i,
            posit,
            arrow_orientation,
            // todo: Auto-scaling, based on grid density, average arrow len etc.
            grad.magnitude() as f32 * 0.2,
            (0.3, 1., 0.5),
            CHARGE_SHINYNESS,
        ));
    }
}

fn event_handler(
    _state: &mut State,
    _event: DeviceEvent,
    _scene: &mut Scene,
    _dt: f32,
) -> EngineUpdates {
    EngineUpdates::default()
}

/// This runs each frame. Currently, no updates.
fn render_handler(_state: &mut State, _scene: &mut Scene, _dt: f32) -> EngineUpdates {
    EngineUpdates::default()
}

/// Generate a f32 mesh from a 3d F64 mesh, using a z slice. Replaces the z value with function value.
fn prepare_2d_mesh_real(
    posits: &Arr2dVec,
    vals: &Arr2dReal,
    scaler: f32,
    grid_n: usize,
    axis_hidden: Axis,
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
            let x = posits[i][j].x as f32;
            let y = posits[i][j].y as f32;
            let z = posits[i][j].z as f32;
            let val = vals[i][j] as f32 * scaler;
            // I believe these should match the order used when setting up the 2D grid.

            // note: By altering the values here, we can change the orientation of the plot.
            result[i][j] = match axis_hidden {
                Axis::X => Vec3::new(y, z, val),
                Axis::Y => Vec3::new(z, x, val),
                Axis::Z => Vec3::new(x, y, val),
            };
        }
    }

    result
}

/// Generate a 2d f32 mesh from a 3d F64 mesh, using a z slice.
fn prepare_2d_mesh(
    posits: &Arr2dVec,
    vals: &Arr2d,
    scaler: f32,
    mag_phase: bool,
    imag: bool,
    grid_n: usize,
    axis_hidden: Axis,
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
                    vals[i][j].phase() / (scaler as f64)
                } else {
                    vals[i][j].im
                }
            } else if mag_phase {
                vals[i][j].mag()
            } else {
                vals[i][j].real
            };

            // todo: DRY
            let x = posits[i][j].x as f32;
            let y = posits[i][j].y as f32;
            let z = posits[i][j].z as f32;
            let val = val as f32 * scaler;

            // I believe these should match the order used when setting up the 2D grid.
            // note: By altering the values here, we can change the orientation of the plot.
            result[i][j] = match axis_hidden {
                Axis::X => Vec3::new(y, z, val),
                Axis::Y => Vec3::new(z, x, val),
                Axis::Z => Vec3::new(x, y, val),
            };
        }
    }

    result
}

/// Updates meshes. For example, when updating a plot due to changing parameters.
/// Note that this is where we decide which Z to render. THis includes both surface meshes,
/// and static ones like spheres and arrows.
pub fn update_meshes(
    surfaces_shared: &SurfacesShared,
    surfaces: &SurfacesPerElec,
    z_displayed: f64,
    scene: &mut Scene,
    grid_posits: &Arr2dVec,
    mag_phase: bool,
    charges_electron: &Arr3dReal,
    grid_n: usize,
    // Render the combined wave function from all electrons.
    render_multi_elec: bool,
    sfc_descs_per_elec: &[SurfaceDesc],
    sfc_descs_combined: &[SurfaceDesc],
    axis_hidden: Axis,
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

    // let mut z_i = 0;
    // for i in 0..grid_posits.len() {
    //     // if grid_posits[0][0][i].z > z_displayed {
    //     if grid_posits[0][i].z > z_displayed {
    //         z_i = i;
    //         break;
    //     }
    // }

    let mut meshes = Vec::new();

    // todo: Fix the DRY between multi and single-elec renders
    // todo: Use sfc_descs_combined to set these.
    if render_multi_elec {
        // meshes.push(Mesh::new_surface(
        //     // &prepare_2d_mesh_real(grid_posits, &surfaces_shared.V_total, z_i, V_SCALER, grid_n),
        //     &prepare_2d_mesh_real(grid_posits, &surfaces_shared.V_total, V_SCALER, grid_n),
        //     true,
        // ));

        // meshes.push(Mesh::new_surface(
        //     &prepare_2d_mesh(
        //         grid_posits,
        //         &surfaces_shared.psi_alpha,
        //         // z_i,
        //         PSI_SCALER,
        //         mag_phase,
        //         false,
        //         grid_n,
        //     ),
        //     true,
        // ));
        //
        // meshes.push(Mesh::new_surface(
        //     &prepare_2d_mesh(
        //         grid_posits,
        //         &surfaces_shared.psi_beta,
        //         // z_i,
        //         PSI_SCALER,
        //         mag_phase,
        //         false,
        //         grid_n,
        //     ),
        //     true,
        // ));

        // meshes.push(Mesh::new_surface(
        //     &prepare_2d_mesh(
        //         grid_posits,
        //         &surfaces_shared.psi_alpha,
        //         z_i,
        //         PSI_SCALER,
        //         mag_phase,
        //         true,
        //         grid_n,
        //     ),
        //     true,
        // ));
        //
        // meshes.push(Mesh::new_surface(
        //     &prepare_2d_mesh(
        //         grid_posits,
        //         &surfaces_shared.psi_beta,
        //         z_i,
        //         PSI_SCALER,
        //         mag_phase,
        //         true,
        //         grid_n,
        //     ),
        //     true,
        // ));

        // meshes.push(Mesh::new_surface(
        //     &prepare_2d_mesh_real(
        //         grid_posits,
        //         &surfaces_shared.charge_alpha,
        //         // z_i,
        //         CHARGE_DENSITY_SCALER,
        //         grid_n,
        //     ),
        //     true,
        // ));
        //
        // meshes.push(Mesh::new_surface(
        //     &prepare_2d_mesh_real(
        //         grid_posits,
        //         &surfaces_shared.charge_beta,
        //         // z_i,
        //         CHARGE_DENSITY_SCALER,
        //         grid_n,
        //     ),
        //     true,
        // ));
        //
        // meshes.push(Mesh::new_surface(
        //     &prepare_2d_mesh_real(
        //         grid_posits,
        //         &surfaces_shared.charge_density_all,
        //         // z_i,
        //         CHARGE_DENSITY_SCALER,
        //         grid_n,
        //     ),
        //     true,
        // ));
        //
        // meshes.push(Mesh::new_surface(
        //     &prepare_2d_mesh_real(
        //         grid_posits,
        //         &surfaces_shared.spin_density,
        //         // z_i,
        //         CHARGE_DENSITY_SCALER,
        //         grid_n,
        //     ),
        //     true,
        // ));
    } else {
        for sfc in sfc_descs_per_elec {
            match sfc.surface {
                SurfaceToRender::V => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh_real(
                            grid_posits,
                            &surfaces.V_acting_on_this,
                            // z_i,
                            V_SCALER,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }
                SurfaceToRender::Psi => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.psi,
                            PSI_SCALER,
                            mag_phase,
                            false,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }
                SurfaceToRender::PsiIm => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.psi,
                            PSI_SCALER,
                            mag_phase,
                            true,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }
                SurfaceToRender::ChargeDensity => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh_real(
                            grid_posits,
                            &surfaces.charge_density_2d,
                            CHARGE_DENSITY_SCALER,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }

                SurfaceToRender::PsiPpCalc => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.psi_pp_calculated,
                            PSI_PP_SCALER,
                            mag_phase,
                            false,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }

                SurfaceToRender::PsiPpCalcIm => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.psi_pp_calculated,
                            PSI_PP_SCALER,
                            mag_phase,
                            true,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }

                SurfaceToRender::PsiPpMeas => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.derivs.d2_sum,
                            PSI_PP_SCALER,
                            mag_phase,
                            false,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }

                SurfaceToRender::PsiPpMeasIm => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.derivs.d2_sum,
                            PSI_PP_SCALER,
                            mag_phase,
                            true,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }

                SurfaceToRender::ElecVFromPsi => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh_real(
                            grid_posits,
                            &surfaces.V_elec_eigen,
                            V_SCALER,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }

                SurfaceToRender::TotalVFromPsi => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh_real(
                            grid_posits,
                            &surfaces.V_total_eigen,
                            V_SCALER,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }

                SurfaceToRender::VDiff => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh_real(
                            grid_posits,
                            &surfaces.V_diff,
                            V_SCALER,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }

                // SurfaceToRender::ElecVFromPsi => {
                //     meshes.push(Mesh::new_surface(
                //         &prepare_2d_mesh_real(grid_posits, &surfaces.aux3, z_i, V_SCALER, grid_n),
                //         true,
                //     ));
                // }

                // todo: Put back and add enum var A/R
                // SurfaceToRender::OrbBug => {
                //     // todo: we have hijacked spinor psi0 to be our H-orbital subtraction view (Apr 2024)
                //     meshes.push(Mesh::new_surface(
                //         &prepare_2d_mesh(
                //             grid_posits,
                //             &surfaces.orb_sub,
                //             z_i,
                //             PSI_SCALER,
                //             mag_phase,
                //             false,
                //             grid_n,
                //         ),
                //         true,
                //     ));
                // }
                SurfaceToRender::PsiSpinor0 => {
                    // Spinors (Trival wavefunction)
                    // meshes.push(Mesh::new_surface(
                    //     &prepare_2d_mesh(
                    //         grid_posits,
                    //         &surfaces.spinor.c0,
                    //         // z_i,
                    //         PSI_SCALER,
                    //         mag_phase,
                    //         false,
                    //         grid_n,
                    //     ),
                    //     true,
                    // ));
                }
                SurfaceToRender::PsiSpinor1 => {
                    // meshes.push(Mesh::new_surface(
                    //     &prepare_2d_mesh(
                    //         grid_posits,
                    //         &surfaces.spinor.c1,
                    //         // z_i,
                    //         PSI_SCALER,
                    //         mag_phase,
                    //         false,
                    //         grid_n,
                    //     ),
                    //     true,
                    // ));
                }
                SurfaceToRender::PsiSpinor2 => {
                    // meshes.push(Mesh::new_surface(
                    //     &prepare_2d_mesh(
                    //         grid_posits,
                    //         &surfaces.spinor.c2,
                    //         // z_i,
                    //         PSI_SCALER,
                    //         mag_phase,
                    //         false,
                    //         grid_n,
                    //     ),
                    //     true,
                    // ));
                }
                SurfaceToRender::PsiSpinor3 => {
                    // meshes.push(Mesh::new_surface(
                    //     &prepare_2d_mesh(
                    //         grid_posits,
                    //         &surfaces.spinor.c3,
                    //         // z_i,
                    //         PSI_SCALER,
                    //         mag_phase,
                    //         false,
                    //         grid_n,
                    //     ),
                    //     true,
                    // ));
                }
                SurfaceToRender::PsiSpinorCalc0 => {
                    // Calculated spinors, from the Dirac equation and trial (spinor) wavefunction.
                    // meshes.push(Mesh::new_surface(
                    //     &prepare_2d_mesh(
                    //         grid_posits,
                    //         &surfaces.spinor_calc.c0,
                    //         // z_i,
                    //         PSI_SCALER,
                    //         mag_phase,
                    //         false,
                    //         grid_n,
                    //     ),
                    //     true,
                    // ));
                }
                SurfaceToRender::PsiSpinorCalc1 => {
                    // meshes.push(Mesh::new_surface(
                    //     &prepare_2d_mesh(
                    //         grid_posits,
                    //         &surfaces.spinor_calc.c1,
                    //         // z_i,
                    //         PSI_SCALER,
                    //         mag_phase,
                    //         false,
                    //         grid_n,
                    //     ),
                    //     true,
                    // ));
                }
                SurfaceToRender::PsiSpinorCalc2 => {
                    // meshes.push(Mesh::new_surface(
                    //     &prepare_2d_mesh(
                    //         grid_posits,
                    //         &surfaces.spinor_calc.c2,
                    //         // z_i,
                    //         PSI_SCALER,
                    //         mag_phase,
                    //         false,
                    //         grid_n,
                    //     ),
                    //     true,
                    // ));
                }
                SurfaceToRender::PsiSpinorCalc3 => {
                    // meshes.push(Mesh::new_surface(
                    //     &prepare_2d_mesh(
                    //         grid_posits,
                    //         &surfaces.spinor_calc.c3,
                    //         // z_i,
                    //         PSI_SCALER,
                    //         mag_phase,
                    //         false,
                    //         grid_n,
                    //     ),
                    //     true,
                    // ));
                }
                SurfaceToRender::H => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.psi_fm_H,
                            PSI_SCALER,
                            mag_phase,
                            false,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }
                SurfaceToRender::HIm => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.psi_fm_H,
                            PSI_SCALER,
                            mag_phase,
                            true,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }
                SurfaceToRender::LSq => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.psi_fm_L2,
                            PSI_SCALER,
                            mag_phase,
                            false,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }
                SurfaceToRender::LSqIm => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.psi_fm_L2,
                            PSI_SCALER,
                            mag_phase,
                            true,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }

                SurfaceToRender::LZ => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.psi_fm_Lz,
                            PSI_SCALER,
                            mag_phase,
                            false,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }

                SurfaceToRender::LZIm => {
                    meshes.push(Mesh::new_surface(
                        &prepare_2d_mesh(
                            grid_posits,
                            &surfaces.psi_fm_Lz,
                            PSI_SCALER,
                            mag_phase,
                            true,
                            grid_n,
                            axis_hidden,
                        ),
                        true,
                    ));
                }
                SurfaceToRender::VPElec => unimplemented!(), // todo: Put these in A/R, with enum variants
                                                             // SurfaceToRender::V => {
                                                             //     // todo: Likely temp.
                                                             //     meshes.push(Mesh::new_surface(
                                                             //         &prepare_2d_mesh(
                                                             //             grid_posits,
                                                             //             &surfaces.derivs.dx,
                                                             //             z_i,
                                                             //             PSI_SCALER,
                                                             //             mag_phase,
                                                             //             false,
                                                             //             grid_n,
                                                             //         ),
                                                             //         true,
                                                             //     ));
                                                             // }
                                                             //
                                                             // SurfaceToRender::V => {
                                                             //     meshes.push(Mesh::new_surface(
                                                             //         &prepare_2d_mesh(
                                                             //             grid_posits,
                                                             //             &surfaces.derivs.dy,
                                                             //             z_i,
                                                             //             PSI_SCALER,
                                                             //             mag_phase,
                                                             //             false,
                                                             //             grid_n,
                                                             //         ),
                                                             //         true,
                                                             //     ));
                                                             // }
                                                             // SurfaceToRender::V => {
                                                             //     meshes.push(Mesh::new_surface(
                                                             //         &prepare_2d_mesh(
                                                             //             grid_posits,
                                                             //             &surfaces.derivs.dz,
                                                             //             z_i,
                                                             //             PSI_SCALER,
                                                             //             mag_phase,
                                                             //             false,
                                                             //             grid_n,
                                                             //         ),
                                                             //         true,
                                                             //     ));
                                                             // }
                                                             // SurfaceToRender::V => {
                                                             //     meshes.push(Mesh::new_surface(
                                                             //         &prepare_2d_mesh(
                                                             //             grid_posits,
                                                             //             &surfaces.derivs.d2x,
                                                             //             z_i,
                                                             //             PSI_PP_SCALER,
                                                             //             mag_phase,
                                                             //             false,
                                                             //             grid_n,
                                                             //         ),
                                                             //         true,
                                                             //     ));
                                                             // }
                                                             // SurfaceToRender::V => {
                                                             //     meshes.push(Mesh::new_surface(
                                                             //         &prepare_2d_mesh(
                                                             //             grid_posits,
                                                             //             &surfaces.derivs.d2y,
                                                             //             z_i,
                                                             //             PSI_PP_SCALER,
                                                             //             mag_phase,
                                                             //             false,
                                                             //             grid_n,
                                                             //         ),
                                                             //         true,
                                                             //     ));
                                                             // }
                                                             // SurfaceToRender::V => {
                                                             //     meshes.push(Mesh::new_surface(
                                                             //         &prepare_2d_mesh(
                                                             //             grid_posits,
                                                             //             &surfaces.derivs.d2z,
                                                             //             z_i,
                                                             //             PSI_PP_SCALER,
                                                             //             mag_phase,
                                                             //             false,
                                                             //             grid_n,
                                                             //         ),
                                                             //         true,
                                                             //     ));
                                                             // }
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
        }
    }

    meshes.push(Mesh::new_sphere(CHARGE_SPHERE_SIZE, 12, 12));
    meshes.push(Mesh::new_arrow(2., 0.2, 12));

    scene.meshes = meshes;
}

/// Updates entities, but not meshes. For example, when hiding or
/// showing a mesh.
/// Note that currently, we update charge displays along
/// with surfaces.
pub fn update_entities(
    charges: &[(Vec3F64, f64)],
    surface_descs: &[SurfaceDesc],
    scene: &mut Scene,
    charge_density_balls: &[Vec3F64],
    gradient: &Arr3dVec,
    grid_posits_gradient: &Arr3dVec,
) {
    let num_sfcs = surface_descs.len();

    let mesh_sphere_i = num_sfcs;
    let vector_arrow_i = num_sfcs + 1;

    let mut entities = Vec::new();
    for i in 0..num_sfcs {
        if !surface_descs[i].visible {
            continue;
        }
        entities.push(Entity::new(
            i,
            Vec3::new_zero(),
            Quaternion::new_identity(),
            1.,
            SURFACE_COLORS[i % SURFACE_COLORS.len()],
            SURFACE_SHINYNESS,
        ));
    }

    for (posit, val) in charges {
        entities.push(Entity::new(
            mesh_sphere_i,
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

    add_non_surface_entities(
        &mut entities,
        charge_density_balls,
        gradient,
        grid_posits_gradient,
        mesh_sphere_i,
        vector_arrow_i,
    );

    // todo: Im comps if ticked

    scene.entities = entities;
}

/// Entry point to our render and event loop.
pub fn render(state: State) {
    let mut scene = Scene {
        meshes: Vec::new(),   // updated below.
        entities: Vec::new(), // updated beloCw.
        camera: Camera {
            fov_y: TAU / 8.,
            position: Vec3::new(0., 10., -20.),
            far: RENDER_DIST,
            orientation: Quaternion::from_axis_angle(Vec3::new(1., 0., 0.), TAU / 16.),
            ..Default::default()
        },
        lighting: Lighting {
            ambient_color: [-1., 1., 1., 0.5],
            ambient_intensity: 0.03,
            point_lights: vec![
                // Light from above
                PointLight {
                    type_: LightType::Omnidirectional,
                    position: Vec3::new(20., 20., 100.),
                    diffuse_color: [0.3, 0.4, 0.5, 1.],
                    specular_color: [0.3, 0.4, 0.5, 1.],
                    diffuse_intensity: 1_000.,
                    specular_intensity: 2_000.,
                },
                // Light from below
                PointLight {
                    type_: LightType::Omnidirectional,
                    position: Vec3::new(-20., 20., -100.),
                    diffuse_color: [0.3, 0.4, 0.5, 1.],
                    specular_color: [0.3, 0.4, 0.5, 1.],
                    diffuse_intensity: 1_000.,
                    specular_intensity: 2_000.,
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
    if state.surfaces_per_elec.len() > 0 {
        let surfaces = &state.surfaces_per_elec[active_elec_init];

        update_meshes(
            &state.surfaces_shared,
            surfaces,
            state.ui.z_displayed,
            &mut scene,
            &state.surfaces_shared.grid_posits,
            state.ui.mag_phase,
            &state.charges_from_electron[active_elec_init],
            state.grid_n_render,
            false,
            &state.surface_descs_per_elec,
            &state.surface_descs_combined,
            state.ui.hidden_axis,
        );
    }

    update_entities(
        &state.nucleii,
        &state.surface_descs_per_elec,
        &mut scene,
        &state.charge_density_balls,
        &state.surfaces_shared.elec_field_gradient,
        &state.surfaces_shared.grid_posits_gradient,
    );

    let input_settings = InputSettings {
        initial_controls: ControlScheme::FreeCamera,
        ..Default::default()
    };
    let ui_settings = UiSettings {
        layout: UiLayout::Left,
        icon_path: Some("./resources/icon.png".to_owned()),
    };

    graphics::run(
        state,
        scene,
        input_settings,
        ui_settings,
        render_handler,
        event_handler,
        ui_handler,
    );
}
