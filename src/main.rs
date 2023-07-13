#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(uncommon_codepoints)]
#![allow(confusable_idents)]
#![allow(non_upper_case_globals)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

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
mod grid_setup;
mod interp;
// mod nudge;
mod num_diff;
mod potential;
mod render;
mod types;
mod ui;
mod util;
mod wf_ops;

use crate::{
    basis_wfs::Basis,
    grid_setup::{Arr3d, Arr3dReal},
    types::{EvalDataPerElec, EvalDataShared, SurfacesPerElec, SurfacesShared},
    wf_ops::Q_PROT,
};

const NUM_SURFACES: usize = 10;

const SPACING_FACTOR_DEFAULT: f64 = 1.7;
const GRID_N_DEFAULT: usize = 50;
const GRID_N_CHARGE_DEFAULT: usize = 40;

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
    pub V_from_elecs_1d: Vec<Vec<f64>>,
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
    pub bases_evaluated: Vec<wf_ops::BasesEvaluated>,
    pub bases_evaluated_1d: Vec<wf_ops::BasesEvaluated1d>,
    /// Similar to `bases_evaluated`, but on the charge grid. We don't need diffs for this.
    /// Outer is per-electron. Inner is per-basis
    /// todo: Do we want/need per-electron here?
    pub bases_evaluated_charge: Vec<Vec<Arr3d>>,
    /// Used to toggle precense of a bases, effectively setting its weight ot 0 without losing the stored
    /// weight value. Per-electron.
    pub bases_visible: Vec<Vec<bool>>,
    /// Amount to nudge next; stored based on sensitivity of previous nudge. Per-electron.
    pub nudge_amount: Vec<f64>,
    /// Wave function score, evaluated by comparing psi to psi'' from numerical evaluation, and
    /// from the Schrodinger equation. Per-electron.
    // pub psi_pp_score: Vec<f64>,
    pub surface_data: [SurfaceData; NUM_SURFACES],
    pub grid_n_render: usize,
    /// This charge grid is generally denser than the main grid. This allows more fidelity for
    /// modelling electron charge, without evaluating the wave function at too many points.
    pub grid_n_charge: usize,
    pub grid_range_render: (f64, f64),
    pub grid_range_charge: (f64, f64),
    /// These are the points we evaluate our psi'' score at, and where we generate net potential,
    /// ie from electrons, at. Per-electron
    pub eval_data_per_elec: Vec<EvalDataPerElec>,
    pub eval_data_shared: EvalDataShared,
    /// 1.0 is an evenly-spaced grid. A higher value spreads out the grid; high values
    /// mean increased non-linearity, with higher spacing farther from the center.
    /// This only (currently) applies to the main grid, with a uniform grid set for
    /// charge density.
    pub sample_factor_render: f64,
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
    /// If set, E is automatically optimized after manually adjusting basis weights.
    pub adjust_E_with_weights: bool,
    /// Automatically generate the (1d) V (And acted on V?) when changing bases.
    pub auto_gen_elec_V: bool,
    /// When updating the weight of basis for an electron, update that weight
    /// for all electrons that share the same n. (When applicable?)
    pub weight_symmetry: bool,
    /// This is very computationally intensive, but can help visualize and debug electron-electron
    /// repulsion.
    pub create_3d_electron_V: bool,
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
    grid_range: (f64, f64),
    grid_range_charge: (f64, f64),
    spacing_factor: f64,
    grid_n: usize,
    grid_n_charge: usize,
    bases: &[Vec<Basis>],
    charges_fixed: &[(Vec3, f64)],
    num_electrons: usize,
) -> (
    Vec<Arr3dReal>,
    Vec<Arr3dReal>,
    Vec<wf_ops::BasesEvaluated>,
    Vec<Vec<Arr3d>>,
    SurfacesShared,
    Vec<SurfacesPerElec>,
    // Vec<f64>,
) {
    let arr_real = grid_setup::new_data_real(grid_n);

    let sfcs_one_elec = SurfacesPerElec::new(grid_n);

    let mut surfaces_per_elec = vec![sfcs_one_elec.clone(), sfcs_one_elec];

    let mut surfaces_shared = SurfacesShared::new(
        grid_range,
        grid_range_charge,
        spacing_factor,
        grid_n,
        grid_n_charge,
        num_electrons,
    );
    // surfaces_shared.combine_psi_parts(&surfaces_per_elec, &Es, grid_n);

    grid_setup::update_grid_posits(
        &mut surfaces_shared.grid_posits,
        grid_range,
        spacing_factor,
        grid_n,
    );

    grid_setup::update_grid_posits(
        &mut surfaces_shared.grid_posits_charge,
        grid_range_charge,
        1.,
        grid_n_charge,
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

    let bases_evaluated_one = wf_ops::BasesEvaluated::new(
        &bases[0], // todo: A bit of a kludge
        &surfaces_shared.grid_posits,
        grid_n,
    );

    let bases_evaluated_charge_one = wf_ops::arr_from_bases(
        &bases[0], // todo: A bit of a kludge
        &surfaces_shared.grid_posits_charge,
        grid_n_charge,
    );

    // These must be initialized from wave functions later.
    let mut bases_evaluated = Vec::new();
    let mut bases_evaluated_charge = Vec::new();
    let mut charges_electron = Vec::new();
    let mut V_from_elecs = Vec::new();
    // let mut psi_pp_score = Vec::new();

    // todo: YOu may not need the "bases_evaluated" per-elec.
    for i_elec in 0..num_electrons {
        charges_electron.push(grid_setup::new_data_real(grid_n_charge));
        V_from_elecs.push(arr_real.clone());
        bases_evaluated.push(bases_evaluated_one.clone());
        bases_evaluated_charge.push(bases_evaluated_charge_one.clone());

        // Set up our basis-function based trial wave function.
        // todo: Handle the multi-electron case instead of hard-coding 0.
        let weights: Vec<f64> = bases[0].iter().map(|b| b.weight()).collect();
        wf_ops::update_wf_fm_bases(
            &mut surfaces_per_elec[i_elec],
            &bases_evaluated[i_elec],
            // &surfaces_shared.grid_posits,
            -0.5,
            grid_n,
            &weights,
        );
    }

    (
        charges_electron,
        V_from_elecs,
        bases_evaluated,
        bases_evaluated_charge,
        surfaces_shared,
        surfaces_per_elec,
    )
}

pub fn init_1d(
    bases: &[Vec<Basis>],
    charges_fixed: &[(Vec3, f64)],
    num_electrons: usize,
) -> (
    EvalDataShared,
    Vec<EvalDataPerElec>,
    Vec<Vec<f64>>,
    Vec<wf_ops::BasesEvaluated1d>,
) {
    let mut eval_data_shared = EvalDataShared::new(&charges_fixed);
    let eval_data_one = EvalDataPerElec::new(eval_data_shared.grid_n);

    let mut eval_data_per_elec = Vec::new();
    for _ in 0..num_electrons {
        eval_data_per_elec.push(eval_data_one.clone());
    }

    potential::update_V_from_nuclei_1d(
        &mut eval_data_shared.V_from_nuclei,
        charges_fixed,
        &eval_data_shared.posits,
        eval_data_shared.grid_n,
    );

    // for (elec_i, _electron) in surfaces_per_elec.iter_mut().enumerate() {
    for elec_i in 0..num_electrons {
        potential::update_V_acting_on_elec_1d(
            &mut eval_data_per_elec[elec_i].V_acting_on_this,
            &eval_data_shared.V_from_nuclei,
            &[], // Not ready to apply V from other elecs yet.
            elec_i,
            eval_data_shared.grid_n,
        );
    }

    let norm = 1.; // todo: Figure this out!
    let bases_evaluated_one = wf_ops::BasesEvaluated1d::new(
        &bases[0], // todo: A bit of a kludge
        &eval_data_shared.posits,
        norm,
    );

    // These must be initialized from wave functions later.
    let mut bases_evaluated = Vec::new();
    let mut V_from_elecs = Vec::new();
    // let mut psi_pp_score = Vec::new();

    // todo: YOu may not need the "bases_evaluated" per-elec.
    for i_elec in 0..num_electrons {
        V_from_elecs.push(vec![0.; eval_data_shared.grid_n]);
        bases_evaluated.push(bases_evaluated_one.clone());

        // Set up our basis-function based trial wave function.

        let weights: Vec<f64> = bases[i_elec].iter().map(|b| b.weight()).collect();
        wf_ops::update_wf_fm_bases_1d(
            &mut eval_data_per_elec[i_elec],
            // todo: Handle the multi-electron case instead of hard-coding 0.
            &bases_evaluated[i_elec],
            eval_data_shared.grid_n,
            &weights,
            None,
        );
    }

    (
        eval_data_shared,
        eval_data_per_elec,
        V_from_elecs,
        bases_evaluated,
    )
}

fn main() {
    let posit_charge_1 = Vec3::new(0., 0., 0.);
    let _posit_charge_2 = Vec3::new(1., 0., 0.);

    let nuclei = vec![
        (posit_charge_1, Q_PROT * 2.), // helium
                                       // (posit_charge_1, Q_PROT * 1.), // Hydrogen
                                       // (posit_charge_2, Q_PROT),
                                       // (Vec3::new(0., 1., 0.), Q_ELEC),
    ];

    let max_basis_n = 1;

    let ui_active_elec = 0;

    let num_elecs = 2;
    // let num_elecs = 1;

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
        &nuclei,
        &mut bases[ui_active_elec],
        Some(&mut bases_visible[ui_active_elec]),
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

    let (grid_min, grid_max) = grid_setup::choose_grid_limits(&nuclei);

    // todo next up: Figure out why you get incorrect answers if these 2 grids don't line up.
    // todo: FOr now, you can continue with matching them if you wish.
    let (grid_min_render, grid_max_render) = (-4., 4.);
    let (grid_min_charge, grid_max_charge) = (-10., 10.);

    // let spacing_factor = 1.6;
    // Currently, must be one as long as used with elec-elec charge.
    let spacing_factor = SPACING_FACTOR_DEFAULT;

    let grid_n = GRID_N_DEFAULT;
    let grid_n_charge = GRID_N_CHARGE_DEFAULT;

    let (
        charges_electron,
        V_from_elecs,
        bases_evaluated,
        bases_evaluated_charge,
        surfaces_shared,
        surfaces_per_elec,
    ) = init_from_grid(
        (grid_min_render, grid_max_render),
        (grid_min_charge, grid_max_charge),
        spacing_factor,
        grid_n,
        grid_n_charge,
        // ui_active_elec,
        &bases,
        &nuclei,
        num_elecs,
    );

    let (eval_data_shared, eval_data_per_elec, V_from_elecs_1d, bases_evaluated_1d) =
        init_1d(&bases, &nuclei, num_elecs);

    let surface_data = [
        SurfaceData::new("V", false),
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
        charges_fixed: nuclei,
        charges_electron,
        V_from_elecs,
        V_from_elecs_1d,
        bases,
        bases_evaluated,
        bases_evaluated_1d,
        bases_evaluated_charge,
        bases_visible,
        surfaces_shared,
        surfaces_per_elec,
        nudge_amount: vec![wf_ops::NUDGE_DEFAULT, wf_ops::NUDGE_DEFAULT],
        surface_data,
        grid_n_render: grid_n,
        grid_n_charge,
        grid_range_render: (grid_min_render, grid_max_render),
        grid_range_charge: (grid_min_charge, grid_max_charge),
        eval_data_shared,
        eval_data_per_elec,
        sample_factor_render: spacing_factor,
        ui_z_displayed: 0.,
        ui_active_elec: ActiveElec::PerElec(ui_active_elec),
        visual_rotation: 0.,
        mag_phase: false,
        max_basis_n,
        num_elecs,
        adjust_E_with_weights: false,
        auto_gen_elec_V: false,
        weight_symmetry: true,
        create_3d_electron_V: false,
    };

    render::render(state);
}
