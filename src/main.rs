#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(uncommon_codepoints)]
#![allow(confusable_idents)]
#![allow(non_upper_case_globals)]
#![allow(clippy::needless_range_loop)]

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

// Consider an adjustable N, so you can use a coarse initial estimate, then refine.

use lin_alg2::f64::{Quaternion, Vec3};

mod basis_fn_finder;
mod basis_wfs;
mod complex_nums;
mod eigen_fns;
mod elec_elec;
mod interp;
mod nudge;
mod num_diff;
mod render;
mod types;
mod ui;
mod util;
mod wf_ops;

use basis_wfs::{Basis, HOrbital, SphericalHarmonic, Sto};
use types::{Arr3dReal, SurfacesPerElec, SurfacesShared};
use wf_ops::Q_PROT;

const NUM_SURFACES: usize = 10;
const GRID_N_DEFAULT: usize = 20;

// todo: Consider a spherical grid centered perhaps on the system center-of-mass, which
// todo less precision further away?

pub struct State {
    /// Eg, Nuclei (position, charge amt), per the Born-Oppenheimer approximation. Charges over space
    /// due to electrons are stored in `Surfaces`.
    pub charges_fixed: Vec<(Vec3, f64)>,
    /// Charges from electrons, over 3d space. Computed from <ψ|ψ> // todo: Alternatively, we could just
    /// todo use the sfcs.psi and compute from that.
    pub charges_electron: Vec<Arr3dReal>,
    /// Surfaces that are not electron-specific.
    pub surfaces_shared: SurfacesShared,
    /// Computed surfaces, per electron. These span 3D space, are are quite large in memory. Contains various
    /// data including the grid spacing, psi, psi'', V etc.
    /// Vec iterates over the different electrons.
    pub surfaces_per_elec: Vec<SurfacesPerElec>,
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
    /// Visuals for complex fields default to real/imaginary. Enabling this
    /// switches this to magnitude and phase.
    pub mag_phase: bool,
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

    let charges_fixed = vec![
        (posit_charge_1, Q_PROT * 1.), // helium
                                       // (posit_charge_2, Q_PROT),
                                       // (Vec3::new(0., 1., 0.), Q_ELEC),
    ];

    // Outer of these is per-elec.
    let mut bases = vec![Vec::new()];
    let mut bases_visible = vec![Vec::new()];
    let max_n = 2;
    wf_ops::initialize_bases(&charges_fixed, &mut bases[0], &mut bases_visible[0], max_n);

    let ui_active_elec = 0;
    // H ion nuc dist is I believe 2 bohr radii.
    // let charges = vec![(Vec3::new(-1., 0., 0.), Q_PROT), (Vec3::new(1., 0., 0.), Q_PROT)];

    let grid_n = GRID_N_DEFAULT;

    let arr_real = types::new_data_real(grid_n);

    // These must be initialized from wave functions later.
    let charges_electron = vec![arr_real.clone(), arr_real];

    let E = -0.7;
    let L_2 = 1.;
    let L_x = 1.;
    let L_y = 1.;
    let L_z = 1.;

    let Es = vec![E, E];

    let sfcs_one_elec = SurfacesPerElec::new(grid_n);

    let mut surfaces_per_elec = vec![sfcs_one_elec.clone(), sfcs_one_elec];

    let mut grid_min = -2.;
    let mut grid_max = 2.; // todo: Is this used, or overridden?
    let spacing_factor = 1.6;

    let mut surfaces_shared = SurfacesShared::new(grid_min, grid_max, spacing_factor, grid_n);
    surfaces_shared.combine_psi_parts(&surfaces_per_elec, &Es, grid_n);

    wf_ops::update_V_fm_fixed_charges(
        &charges_fixed,
        &mut surfaces_shared.V_fixed_charges,
        &mut grid_min,
        &mut grid_max,
        spacing_factor,
        &mut surfaces_shared.grid_posits,
        grid_n,
    );

    // todo: For now and for here at least, make all individual V = to fixed V at init.
    for sfc in &mut surfaces_per_elec {
        types::copy_array_real(&mut sfc.V, &surfaces_shared.V_fixed_charges, grid_n);
    }

    // Set up our basis-function based trial wave function.
    wf_ops::update_wf_fm_bases(
        &bases[0],
        &mut surfaces_per_elec[ui_active_elec],
        Es[ui_active_elec],
        &mut surfaces_shared.grid_posits,
        &bases_visible[ui_active_elec],
        grid_n,
    );

    let psi_p_score = 0.; // todo T
    let psi_pp_score_one = wf_ops::score_wf(&surfaces_per_elec[ui_active_elec], grid_n);

    let psi_pp_score = vec![psi_pp_score_one, psi_pp_score_one];

    let show_surfaces = [
        true, true, false, false, true, false, true, false, false, false,
    ];

    let surface_names = [
        "V".to_owned(),
        "ψ".to_owned(),
        "ψ im".to_owned(),
        "ψ²".to_owned(),
        "ψ'' calc".to_owned(),
        "ψ'' calc im".to_owned(),
        "ψ'' meas".to_owned(),
        "ψ'' meas im".to_owned(),
        // "ψ' calculated".to_owned(),
        // "ψ' measured".to_owned(),
        "Aux 1".to_owned(),
        "Aux 2".to_owned(),
    ];

    let state = State {
        charges_fixed,
        charges_electron,
        bases,
        bases_visible,
        surfaces_shared,
        surfaces_per_elec,
        E: Es,
        nudge_amount: vec![wf_ops::NUDGE_DEFAULT, wf_ops::NUDGE_DEFAULT],
        psi_pp_score,
        surface_names,
        show_surfaces,
        grid_n,
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
        mag_phase: false,
    };

    render::render(state);
}
