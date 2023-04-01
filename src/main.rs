#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(uncommon_codepoints)]

//! This program explores solving the wave equation for
//! arbitrary potentials. It visualizes the wave function in 3d, with user interaction.

// todo: Consider instead of H orbitals, use the full set of Slater basis
// functions, which are more general. Make a fn to generate them.

// todo: Hylleraas basis functions?

// Idea, once your nudging etc code is faster and more accurate:
// Create template WFs based on ICs. Ie, multi-charge potential starting points
// based on a formula. Maybe they are something akin to a 3D LUT. So the converging
// goes faster.

//  When applying your force via electorn density (Sim code; maybe not this lib),
// You may need to interpolate to avoid quantized (not in the way we need!) positions
// at the grid you chose. Linear is fine.

use lin_alg2::f64::{Quaternion, Vec3};

mod basis_fn_finder;
mod basis_wfs;
mod complex_nums;
mod eigen_fns;
mod interp;
mod nudge;
mod num_diff;
mod render;
mod types;
mod ui;
mod util;
mod wf_ops;

use basis_wfs::{Basis, HOrbital, SphericalHarmonic, Sto};
use complex_nums::Cplx;
use wf_ops::{ħ, M_ELEC, N, Q_PROT};

use types::{Arr3d, Arr3dReal, Arr3dVec, Surfaces};

const NUM_SURFACES: usize = 6;

// todo: Consider a spherical grid centered perhaps on the system center-of-mass, which
// todo less precision further away?

pub struct State {
    pub grid_posits: Arr3dVec,
    /// Eg, Nuclei (position, charge amt), per the Born-Oppenheimer approximation. Charges over space
    /// due to electrons are stored in `Surfaces`.
    pub charges_fixed: Vec<(Vec3, f64)>,
    /// Computed surfaces, per electron. These span 3D space, are are quite large in memory. Contains various
    /// data including the grid spacing, psi, psi'', V etc.
    /// Vec iterates over the different electrons.
    pub surfaces: Vec<Surfaces>,
    /// The sum of all surfaces
    pub surfaces_combined: Surfaces,
    /// todo: Combine bases and nuclei in into single tuple etc to enforce index pairing?
    /// todo: Or a sub struct?
    /// Wave functions, with weights. Per-electron. (Outer Vec iterates over electrons; inner over
    /// bases per-electron)
    pub bases: Vec<Vec<Basis>>,
    /// Used to toggle precense of a basi, effectively setting its weight ot 0 without losing the stored
    /// weight value.
    pub bases_visible: Vec<Vec<bool>>,
    /// Energy eigenvalue of the Hamiltonian; per electron.
    /// todo: You may need separate eigenvalues per electron-WF if you go that route.
    pub E: Vec<f64>,
    /// Amount to nudge next; stored based on sensitivity of previous nudge. Per-electron.
    pub nudge_amount: Vec<f64>,
    /// Wave function score, evaluated by comparing psi to psi'' from numerical evaluation, and
    /// from the Schrodinger equation. Per-electron. todo: Consider replacing with the standard
    /// todo evaluation of "wavefunction fidelity".
    pub psi_pp_score: Vec<f64>,
    /// Surface name
    pub surface_names: [String; NUM_SURFACES],
    pub show_surfaces: [bool; NUM_SURFACES],
    pub grid_n: usize,
    pub grid_min: f64,
    pub grid_max: f64,
    /// 1.0 is an evenly-spaced grid. A higher value spreads out the grid; high values
    /// mean increased non-linearity, with higher spacing farther from the center.
    pub spacing_factor: f64,

    /// When visualizing a 2d wave function over X and Y, this is the fixed Z value rendered.
    /// We only display a slice, since we are viewing a 4d object as a 3d rendering.
    pub ui_z_displayed: f64,
    /// The electron UI controls adjust.
    pub ui_active_elec: usize,
    /// if true, render the composite surfaces for all electrons, vice only the active one.
    pub ui_render_all_elecs: bool,
    /// Rotation of the visual, around either the X or Y axis; used to better visualize
    /// cases that would normally need to be panned through using hte Z-slic slider.
    pub visual_rotation: f64,
    //
    // Below this are mainly experimental/WIP items
    //
    // /// Unused for now
    // pub psi_p_score: f64,
    // /// Angular momentum (L) of the system (eigenvalue)
    // pub L_2: f64,// todo: These l values are currently unused.
    // pub L_x: f64,
    // pub L_y: f64,
    // pub L_z: f64,
}

// /// Interpolate a value from a discrete wave function, assuming (what about curvature)
// fn interp_wf(psi: &Arr3d, posit_sample: Vec3) -> Cplx {
//     // Maybe a polynomial?
// }

fn main() {
    let posit_charge_1 = Vec3::new(-1., 0., 0.);
    let posit_charge_2 = Vec3::new(1., 0., 0.);

    let neutral = Quaternion::new_identity();

    // todo: Clean up constructor sequene for these basis fns A/R.
    let wfs = vec![
        Basis::H(HOrbital::new(
            posit_charge_1,
            1,
            SphericalHarmonic::default(),
            1.,
            0,
        )),
        Basis::H(HOrbital::new(
            posit_charge_2,
            1,
            SphericalHarmonic::default(),
            0.,
            1,
        )),
        Basis::H(HOrbital::new(
            posit_charge_1,
            2,
            SphericalHarmonic::default(),
            0.,
            0,
        )),
        Basis::H(HOrbital::new(
            posit_charge_2,
            2,
            SphericalHarmonic::default(),
            0.,
            1,
        )),
        Basis::H(HOrbital::new(
            posit_charge_1,
            2,
            SphericalHarmonic::new(1, 0, neutral),
            0.,
            0,
        )),
        Basis::H(HOrbital::new(
            posit_charge_2,
            2,
            SphericalHarmonic::new(1, 0, neutral),
            0.,
            1,
        )),
        Basis::H(HOrbital::new(
            posit_charge_1,
            3,
            SphericalHarmonic::default(),
            0.,
            0,
        )),
        Basis::Sto(Sto::new(
            posit_charge_1,
            1,
            SphericalHarmonic::default(),
            1.,
            0.,
            1,
        )),
    ];

    let ui_active_elec = 0;

    let visible = vec![true, true, true, true, true, true, true, true];
    let bases_visible = vec![visible.clone(), visible];

    // H ion nuc dist is I believe 2 bohr radii.
    // let charges = vec![(Vec3::new(-1., 0., 0.), Q_PROT), (Vec3::new(1., 0., 0.), Q_PROT)];
    let charges = vec![
        (posit_charge_1, Q_PROT * 2.), // helium
                                       // (posit_charge_2, Q_PROT),
                                       // (Vec3::new(0., 1., 0.), Q_ELEC),
    ];

    let z_displayed = 0.;

    let E = -0.7;
    let L_2 = 1.;
    let L_x = 1.;
    let L_y = 1.;
    let L_z = 1.;

    let (mut grid_min, mut grid_max) = (-2.5, 2.5);

    // // todo: Deprecate h_grid once your alternative works.
    // let h_grid = (grid_max - grid_min) / (N as f64);
    // let h_grid_sq = h_grid.powi(2);

    let mut sfcs = Surfaces::default();

    let spacing_factor = 1.6;
    // let spacing_factor = 1.;

    let mut grid_posits = types::new_data_vec(N);
    wf_ops::update_grid_posits(&mut grid_posits, grid_min, grid_max, spacing_factor);

    // todo: Short-term experiment
    // Set up an initial charge of a s0 Hydrogen orbital. Computationally intensive to use any of
    // these charges, but
    let mut psi_h00 = types::new_data(N);

    let h00 = Basis::H(HOrbital::new(
        posit_charge_1,
        1,
        SphericalHarmonic::default(),
        1.,
        0,
    ));

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                let posit_sample = grid_posits[i][j][k];

                psi_h00[i][j][k] = h00.value(posit_sample) * h00.weight();
            }
        }
    }

    let mut charge_density = types::new_data_real(N);
    wf_ops::charge_density_fm_psi(&psi_h00, &mut charge_density, 1);

    // sfcs.elec_charges = vec![charge_density]; // todo: removed
    // todo: end short-term experiment

    wf_ops::init_wf(
        &wfs,
        &charges,
        &mut sfcs,
        E,
        true,
        &mut grid_min,
        &mut grid_max,
        spacing_factor,
        &mut grid_posits,
        &bases_visible[ui_active_elec],
    );

    let psi_p_score = 0.; // todo T
    let psi_pp_score = wf_ops::score_wf(&sfcs);

    let show_surfaces = [true, true, true, true, false, false];

    let surface_names = [
        "V".to_owned(),
        "ψ".to_owned(),
        "ψ'' calculated".to_owned(),
        "ψ'' measured".to_owned(),
        // "ψ' calculated".to_owned(),
        // "ψ' measured".to_owned(),
        "Aux 1".to_owned(),
        "Aux 2".to_owned(),
    ];

    let state = State {
        grid_posits,
        charges_fixed: charges,
        bases: vec![wfs.clone(), wfs.clone()],
        bases_visible,
        surfaces: vec![sfcs.clone(), sfcs.clone()],
        surfaces_combined: sfcs.clone(),
        E: vec![E, E],
        nudge_amount: vec![wf_ops::NUDGE_DEFAULT, wf_ops::NUDGE_DEFAULT],
        psi_pp_score: vec![psi_pp_score, psi_pp_score],
        surface_names,
        show_surfaces,
        grid_n: N,
        grid_min,
        grid_max,
        spacing_factor,

        ui_z_displayed: 0.,
        ui_active_elec,
        ui_render_all_elecs: false,
        visual_rotation: 0.,
        // gaussians,
        // L_2,
        // L_x,
        // L_y,
        // L_z,
        // z_displayed,
        // psi_p_score,
    };

    render::render(state);
}
