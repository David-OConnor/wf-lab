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

mod basis_wfs;
mod complex_nums;
mod eigen_fns;
mod interp;
mod nudge;
mod num_diff;
mod rbf;
mod render;
mod types;
mod ui;
mod util;
mod wf_ops;
mod basis_fn_finder;

use basis_wfs::{Basis, HOrbital, SphericalHarmonic, Sto};
use complex_nums::Cplx;
use wf_ops::{ħ, M_ELEC, N, Q_PROT};

use types::{Arr3d, Arr3dReal, Surfaces};

const NUM_SURFACES: usize = 8;

// todo: Consider a spherical grid centered perhaps on the system center-of-mass, which
// todo less precision further away?

pub struct State {
    /// todo: Combine wfs and nuclei in into single tuple etc to enforce index pairing?
    /// todo: Or a sub struct?
    /// Wave functions, with weights
    pub bases: Vec<Basis>,
    // todo use an index for them.
    /// Nuclei. todo: H only for now.
    pub charges: Vec<(Vec3, f64)>,
    /// Computed surfaces, with name.
    pub surfaces: Surfaces,
    /// Eg, least-squares over 2 or 3 dimensions between
    /// When visualizing a 2d wave function over X and Y, this is the fixed Z value.
    pub z_displayed: f64,
    /// Energy of the system (eigenvalue); per electron.
    /// todo: You may need separate eigenvalues per electron-WF if you go that route.
    pub E: Vec<f64>,
    /// Angular momentum (L) of the system (eigenvalue)
    pub L_2: f64,// todo: These l values are currently unused.
    pub L_x: f64,
    pub L_y: f64,
    pub L_z: f64,
    /// Unused for now
    pub psi_p_score: f64,
    /// per electron.
    pub psi_pp_score: Vec<f64>,
    /// Surface name
    pub surface_names: [String; NUM_SURFACES],
    pub show_surfaces: [bool; NUM_SURFACES],
    pub grid_n: usize,
    pub grid_min: f64,
    pub grid_max: f64,
    pub nudge_amount: f64,
    // vals for h_grid and h_grid_sq, need to be updated when grid changes. A cache.
    // For finding psi_pp_meas, using only values on the grid
    // pub h_grid: f64,
    // pub h_grid_sq: f64,
    /// 1.0 is an evenly-spaced grid.
    pub spacing_factor: f64,
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
                let posit_sample = sfcs.grid_posits[i][j][k];

                psi_h00[i][j][k] = h00.value(posit_sample) * h00.weight();
            }
        }
    }

    let mut charge_density = types::new_data_real(N);
    wf_ops::charge_density_fm_psi(&psi_h00, &mut charge_density, 1);

    // sfcs.elec_charges = vec![charge_density]; // todo: removed
    // todo: end short-term experiment

    // let spacing_factor = 2.;
    let spacing_factor = 1.;
    wf_ops::update_grid_posits(&mut sfcs.grid_posits, grid_min, grid_max, spacing_factor);

    wf_ops::init_wf(
        &wfs,
        &charges,
        &mut sfcs,
        E,
        true,
        &mut grid_min,
        &mut grid_max,
        spacing_factor,
    );

    let psi_p_score = 0.; // todo T
    let psi_pp_score = wf_ops::score_wf(&sfcs);

    let show_surfaces = [true, true, true, true, false, false, false, false];

    let surface_names = [
        "V".to_owned(),
        "ψ".to_owned(),
        "ψ'' calculated".to_owned(),
        "ψ'' measured".to_owned(),
        "ψ' calculated".to_owned(),
        "ψ' measured".to_owned(),
        "Aux 1".to_owned(),
        "Aux 2".to_owned(),
    ];

    // let z = vec![4; N];
    // let y = vec![z; N];
    // let grid_divisions = vec![y; N];

    let state = State {
        bases: wfs,
        charges,
        surfaces: sfcs,
        E,
        L_2,
        L_x,
        L_y,
        L_z,
        z_displayed,
        psi_p_score,
        psi_pp_score,
        surface_names,
        show_surfaces,
        // grid_divisions,
        // gaussians,
        grid_n: N,
        grid_min,
        grid_max,
        nudge_amount: wf_ops::NUDGE_DEFAULT,
        // h_grid,
        // h_grid_sq,
        spacing_factor,
    };

    render::render(state);
}
