#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(uncommon_codepoints)]
#![allow(confusable_idents)]
#![allow(non_upper_case_globals)]
#![allow(clippy::needless_range_loop)]

//! This program explores solving the wave equation for
//! arbitrary potentials. It visualizes the wave function in 3d, with user interaction.

//  Consider instead of H orbitals, use the full set of Slater basis
// functions, which are more general.

// Idea, once your nudging etc code is faster and more accurate:
// Create template WFs based on ICs. Ie, multi-charge potential starting points
// based on a formula. Maybe they are something akin to a 3D LUT. So the converging
// goes faster.

// When applying your force via electorn density (Sim code; maybe not this lib),
// You may need to interpolate to avoid quantized (not in the way we need!) positions
// at the grid you chose. Linear is fine.

use lin_alg2::f64::Vec3;

mod basis_weight_finder;
mod basis_wfs;
mod complex_nums;
mod eigen_fns;
mod elec_elec;
mod eval;
mod interp;
mod nudge;
mod num_diff;
mod potential;
mod render;
mod types;
mod ui;
mod util;
mod wf_ops;

use basis_wfs::Basis;
use types::{Arr3dReal, SurfacesPerElec, SurfacesShared};
use wf_lab::types::new_data_real;
use wf_ops::Q_PROT;

const NUM_SURFACES: usize = 10;
const GRID_N_DEFAULT: usize = 20;

// todo: Consider a spherical grid centered perhaps on the system center-of-mass, which
// todo less precision further away?

#[derive(Clone, Copy, PartialEq)]
pub enum ActiveElec {
    PerElec(usize),
    Combined,
}

pub struct State {
    /// Eg, Nuclei (position, charge amt), per the Born-Oppenheimer approximation. Charges over space
    /// due to electrons are stored in `Surfaces`.
    pub charges_fixed: Vec<(Vec3, f64)>,
    /// Charges from electrons, over 3d space. Computed from <ψ|ψ>.
    /// This is not part of `SurfacesPerElec` since we use all values at once (or all except one)
    /// when calculating the potential. (easier to work with API)
    pub charges_electron: Vec<Arr3dReal>,
    /// Also stored here vice part of per-elec structs due to borrow-limiting on struct fields.
    pub V_from_elecs: Vec<Arr3dReal>,
    /// Surfaces that are not electron-specific.
    pub surfaces_shared: SurfacesShared,
    /// Computed surfaces, per electron. These span 3D space, are are quite large in memory. Contains various
    /// data including the grid spacing, psi, psi'', V etc.
    /// Vec iterates over the different electrons.
    pub surfaces_per_elec: Vec<SurfacesPerElec>,
    /// Wave functions, with weights. Per-electron. (Outer Vec iterates over electrons; inner over
    /// bases per-electron)
    pub bases: Vec<Vec<Basis>>,
    /// Basis wave functions. Perhaps faster to cache these (at the cost of more memory use, rather than
    /// compute their value each time we change weights...) Per-electron.
    /// todo: This should probably be in one of the surfaces.
    pub bases_unweighted: Vec<wf_ops::BasisWfsUnweighted>,
    /// Used to toggle precense of a bases, effectively setting its weight ot 0 without losing the stored
    /// weight value. Per-electron.
    pub bases_visible: Vec<Vec<bool>>,
    /// Amount to nudge next; stored based on sensitivity of previous nudge. Per-electron.
    pub nudge_amount: Vec<f64>,
    /// Wave function score, evaluated by comparing psi to psi'' from numerical evaluation, and
    /// from the Schrodinger equation. Per-electron.
    // pub psi_pp_score: Vec<f64>,
    pub surface_data: [SurfaceData; NUM_SURFACES],
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
    pub ui_active_elec: ActiveElec,
    // /// if true, render the composite surfaces for all electrons, vice only the active one.
    // pub ui_render_all_elecs: bool,
    /// Rotation of the visual, around either the X or Y axis; used to better visualize
    /// cases that would normally need to be panned through using hte Z-slic slider.
    pub visual_rotation: f64,
    /// Visuals for complex fields default to real/imaginary. Enabling this
    /// switches this to magnitude and phase.
    pub mag_phase: bool,
    /// When finding and initializing basis, this is the maximum n quantum number.
    pub max_basis_n: u16,
    pub num_elecs: usize,
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

/// Set up the grid so that it smartly encompasses the charges, letting the WF go to 0
/// towards the edges
fn choose_grid_limits(charges_fixed: &[(Vec3, f64)]) -> (f64, f64) {
    let mut max_abs_val = 0.;
    for (posit, _) in charges_fixed {
        if posit.x.abs() > max_abs_val {
            max_abs_val = posit.x.abs();
        }
        if posit.y.abs() > max_abs_val {
            max_abs_val = posit.y.abs();
        }
        if posit.z.abs() > max_abs_val {
            max_abs_val = posit.z.abs();
        }
    }

    // const RANGE_PAD: f64 = 5.8;
    const RANGE_PAD: f64 = 2.5;

    let grid_max = max_abs_val + RANGE_PAD;
    let grid_min = -grid_max;

    // update_grid_posits(grid_posits, *grid_min, *grid_max, spacing_factor, n);
    //
    // let mut grid_min = -5.; // todo ts
    // let mut grid_max = 5.; // todo t

    (grid_min, grid_max)
}

// /// Interpolate a value from a discrete wave function, assuming (what about curvature)
// fn interp_wf(psi: &Arr3d, posit_sample: Vec3) -> Cplx {
//     // Maybe a polynomial?
// }

pub struct SurfaceData {
    pub name: String,
    pub visible: bool,
}

impl SurfaceData {
    /// This constructor simplifies syntax.
    pub fn new(name: &str, visible: bool) -> Self {
        Self {
            name: name.to_owned(),
            visible,
        }
    }
}

/// Run this whenever n changes. Ie, at init, or when n changes in the GUI.
pub fn init_from_grid(
    grid_min: f64,
    grid_max: f64,
    spacing_factor: f64,
    grid_n: usize,
    bases: &[Vec<Basis>],
    charges_fixed: &[(Vec3, f64)],
    num_electrons: usize,
) -> (
    Vec<Arr3dReal>,
    Vec<Arr3dReal>,
    Vec<wf_ops::BasisWfsUnweighted>,
    SurfacesShared,
    Vec<SurfacesPerElec>,
    // Vec<f64>,
) {
    let arr_real = new_data_real(grid_n);

    let sfcs_one_elec = SurfacesPerElec::new(grid_n);

    let mut surfaces_per_elec = vec![sfcs_one_elec.clone(), sfcs_one_elec];

    let mut surfaces_shared =
        SurfacesShared::new(grid_min, grid_max, spacing_factor, grid_n, num_electrons);
    // surfaces_shared.combine_psi_parts(&surfaces_per_elec, &Es, grid_n);

    wf_ops::update_grid_posits(
        &mut surfaces_shared.grid_posits,
        grid_min,
        grid_max,
        spacing_factor,
        grid_n,
    );

    potential::update_V_from_nuclei(
        &mut surfaces_shared.V_from_nuclei,
        charges_fixed,
        &surfaces_shared.grid_posits,
        grid_n,
    );

    for (elec_i, electron) in surfaces_per_elec.iter_mut().enumerate() {
        potential::update_V_acting_on_elec(
            &mut electron.V_acting_on_this,
            &surfaces_shared.V_from_nuclei,
            &[], // Not ready to apply V from other elecs yet.
            elec_i,
            grid_n,
        );
    }

    let basis_wfs_unweighted = wf_ops::BasisWfsUnweighted::new(
        &bases[0], // todo: A bit of a kludge
        &surfaces_shared.grid_posits,
        grid_n,
    );

    // These must be initialized from wave functions later.
    let mut bases_unweighted = Vec::new();
    let mut charges_electron = Vec::new();
    let mut V_from_elecs = Vec::new();
    // let mut psi_pp_score = Vec::new();

    for i_elec in 0..num_electrons {
        charges_electron.push(arr_real.clone());
        V_from_elecs.push(arr_real.clone());
        bases_unweighted.push(basis_wfs_unweighted.clone());

        // Set up our basis-function based trial wave function.
        wf_ops::update_wf_fm_bases(
            // todo: Handle the multi-electron case instead of hard-coding 0.
            &bases[0],
            &bases_unweighted[i_elec],
            &mut surfaces_per_elec[i_elec],
            // &surfaces_shared.grid_posits,
            grid_n,
            None,
        );

        surfaces_per_elec[i_elec].psi_pp_score = eval::score_wf(
            &surfaces_per_elec[i_elec].psi_pp_calculated,
            &surfaces_per_elec[i_elec].psi_pp_calculated,
            grid_n,
        );

        // psi_pp_score.push(psi_pp_score_one);
    }

    (
        charges_electron,
        V_from_elecs,
        bases_unweighted,
        surfaces_shared,
        surfaces_per_elec,
        // psi_pp_score,
    )
}

fn main() {
    let posit_charge_1 = Vec3::new(0., 0., 0.);
    let _posit_charge_2 = Vec3::new(1., 0., 0.);

    let charges_fixed = vec![
        (posit_charge_1, Q_PROT * 2.), // helium
                                       // (posit_charge_2, Q_PROT),
                                       // (Vec3::new(0., 1., 0.), Q_ELEC),
    ];

    let max_basis_n = 3;

    let ui_active_elec = 0;

    let num_elecs = 2;

    // Outer of these is per-elec.
    let mut bases = Vec::new();
    let mut bases_visible = Vec::new();

    // let E_start = -0.7;
    // let mut Es = Vec::new();

    for _ in 0..num_elecs {
        bases.push(Vec::new());
        bases_visible.push(Vec::new());
        // Es.push(E_start);
    }

    wf_ops::initialize_bases(
        &charges_fixed,
        &mut bases[ui_active_elec],
        &mut bases_visible[ui_active_elec],
        max_basis_n,
    );

    // todo: This is getting weird re multiple electrons; perhaps you should switch
    // todo an approach where bases don't have weights, but you use a separate
    // weights array.
    for i_elec in 1..num_elecs {
        bases[i_elec] = bases[0].clone();
        bases_visible[i_elec] = bases_visible[0].clone();
    }

    // H ion nuc dist is I believe 2 bohr radii.
    // let charges = vec![(Vec3::new(-1., 0., 0.), Q_PROT), (Vec3::new(1., 0., 0.), Q_PROT)];

    let (grid_min, grid_max) = choose_grid_limits(&charges_fixed);
    // let spacing_factor = 1.6;
    let spacing_factor = 1.6;
    let grid_n = GRID_N_DEFAULT;

    let (charges_electron, V_from_elecs, bases_unweighted, surfaces_shared, surfaces_per_elec) =
        init_from_grid(
            grid_min,
            grid_max,
            spacing_factor,
            grid_n,
            // ui_active_elec,
            &bases,
            &charges_fixed,
            num_elecs,
        );

    let surface_data = [
        SurfaceData::new("V", true),
        SurfaceData::new("ψ", true),
        SurfaceData::new("ψ im", false),
        SurfaceData::new("ψ²", false),
        SurfaceData::new("ψ'' calc", true),
        SurfaceData::new("ψ'' calc im", false),
        SurfaceData::new("ψ'' meas", true),
        SurfaceData::new("ψ'' meas im", false),
        SurfaceData::new("Aux 1", false),
        SurfaceData::new("Aux 2", false),
    ];

    let state = State {
        charges_fixed,
        charges_electron,
        V_from_elecs,
        bases,
        bases_unweighted,
        bases_visible,
        surfaces_shared,
        surfaces_per_elec,
        nudge_amount: vec![wf_ops::NUDGE_DEFAULT, wf_ops::NUDGE_DEFAULT],
        surface_data,
        grid_n,
        grid_min,
        grid_max,
        spacing_factor,
        ui_z_displayed: 0.,
        ui_active_elec: ActiveElec::PerElec(ui_active_elec),
        visual_rotation: 0.,
        mag_phase: false,
        max_basis_n,
        num_elecs,
    };

    render::render(state);
}
