#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(uncommon_codepoints)]
#![allow(confusable_idents)]
#![allow(non_upper_case_globals)]
#![allow(non_ascii_idents)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

//! This program explores solving the wave equation for
//! arbitrary potentials. It visualizes the wave function in 3d, with user interaction.

// When applying your force via electron density (Sim code; maybe not this lib),
// You may need to interpolate to avoid quantized (not in the way we need!) positions
// at the grid you chose. Linear is fine.

// sys is the raw ffi apis generated with bindgen
// result is a very small wrapper around sys to return Result from each function
// result is a very small wrapper around sys to return Result from each function
// safe is a wrapper around result/sys to provide safe abstractions

// Focus; Lighter-weight models
// https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
// https://github.com/biubug6/Pytorch_Retinaface

// Note on STOs: It appears that, for n=1: E = -(xi^2)/2.

#[cfg(feature = "cuda")]
use cudarc::{driver::CudaDevice, nvrtc::Ptx};
use lin_alg2::f64::Vec3;

mod angular_p;
mod basis_finder;
mod basis_wfs;
mod complex_nums;
mod dirac;
mod eigen_fns;
mod elec_elec;
mod eval;
mod forces;
#[cfg(feature = "cuda")]
mod gpu;
mod grid_setup;
mod interp;
mod num_diff;
mod potential;
mod render;
mod types;
mod ui;
mod util;
mod wf_ops;

use crate::{
    basis_wfs::Basis,
    grid_setup::{new_data, new_data_real, Arr3d, Arr3dReal},
    types::{ComputationDevice, SurfacesPerElec, SurfacesShared},
    ui::procedures,
    wf_ops::{DerivCalc, Spin, Q_PROT},
};

const NUM_SURFACES_PER_ELEC: usize = 11;

const SPACING_FACTOR_DEFAULT: f64 = 1.;
const GRID_MAX_RENDER: f64 = 5.;
const GRID_MAX_CHARGE: f64 = 12.;
const GRID_N_RENDER_DEFAULT: usize = 50;
const GRID_N_CHARGE_DEFAULT: usize = 61;

// todo: Consider a spherical grid centered perhaps on the system center-of-mass, which
// todo less precision further away?

#[derive(Clone, Copy, PartialEq)]
pub enum ActiveElec {
    PerElec(usize),
    Combined,
}

/// Ui state, and adjustable settings.
pub struct StateUi {
    /// When visualizing a 2d wave function over X and Y, this is the fixed Z value rendered.
    /// We only display a slice, since we are viewing a 4d object as a 3d rendering.
    pub z_displayed: f64,
    /// The electron UI controls adjust.
    pub active_elec: ActiveElec,
    /// Rotation of the visual, around either the X or Y axis; used to better visualize
    /// cases that would normally need to be panned through using hte Z-slic slider.
    pub visual_rotation: f64,
    /// Visuals for complex fields default to real/imaginary. Enabling this
    /// switches this to magnitude and phase.
    pub mag_phase: bool,
    /// If set, E is automatically optimized after manually adjusting basis weights.
    pub adjust_E_with_weights: bool,
    /// Automatically generate the (1d) V (And acted on V?) when changing bases.
    pub auto_gen_elec_V: bool,
    /// When updating the weight of basis for an electron, update that weight
    /// for all electrons that share the same n. (When applicable?)
    pub weight_symmetry: bool,
    /// This is very computationally intensive, but can help visualize and debug electron-electron
    /// repulsion.
    pub create_2d_electron_V: bool,
    pub create_3d_electron_V: bool,
    /// Allows toggling spin-alpha electrons.
    pub display_alpha: bool,
    /// Allows toggling spin-beta electrons.
    pub display_beta: bool,
}

impl Default for StateUi {
    fn default() -> Self {
        Self {
            z_displayed: 0.,
            active_elec: ActiveElec::PerElec(0),
            visual_rotation: 0.,
            mag_phase: false,
            adjust_E_with_weights: false,
            auto_gen_elec_V: false,
            weight_symmetry: false,
            // todo: Having issues choosing the correct 2D one; 3D for now.
            create_2d_electron_V: false,
            create_3d_electron_V: true,
            display_alpha: true,
            display_beta: true,
        }
    }
}

pub struct State {
    /// Computation device for evaluating the very expensive charge potential computation.
    pub dev_charge: ComputationDevice,
    /// Computation device for evaluating psi. (And psi''?)
    pub dev_psi: ComputationDevice,
    pub deriv_calc: DerivCalc,
    /// Eg, Nuclei (position, charge amt), per the Born-Oppenheimer approximation. Charges over space
    /// due to electrons are stored in `Surfaces`.
    pub charges_fixed: Vec<(Vec3, f64)>,
    /// Charges from electrons, over 3d space. Each value is the charge created by a single electron. Computed from <ψ|ψ>.
    /// This is not part of `SurfacesPerElec` since we use all values at once (or all except one)
    /// when calculating the potential. (easier to work with API)
    pub charges_from_electron: Vec<Arr3dReal>,
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
    /// Similar to `bases_evaluated`, but on the charge grid. We don't need diffs for this.
    /// Outer is per-electron. Inner is per-basis
    pub psi_charge: Vec<Vec<Arr3d>>,
    // /// Amount to nudge next; stored based on sensitivity of previous nudge. Per-electron.
    // pub nudge_amount: Vec<f64>,
    pub surface_descs_per_elec: Vec<SurfaceDesc>,
    pub surface_descs_combined: Vec<SurfaceDesc>,
    pub grid_n_render: usize,
    /// This charge grid is generally denser than the main grid. This allows more fidelity for
    /// modelling electron charge, without evaluating the wave function at too many points.
    pub grid_n_charge: usize,
    pub grid_range_render: (f64, f64),
    pub grid_range_charge: (f64, f64),
    /// 1.0 is an evenly-spaced grid. A higher value spreads out the grid; high values
    /// mean increased non-linearity, with higher spacing farther from the center.
    /// This only (currently) applies to the main grid, with a uniform grid set for
    /// charge density.
    pub sample_factor_render: f64,
    /// When finding and initializing basis, this is the maximum n quantum number.
    pub max_basis_n: u16,
    pub num_elecs: usize,
    pub ui: StateUi,
}

pub struct SurfaceDesc {
    pub name: String,
    pub visible: bool,
}

impl SurfaceDesc {
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
    dev_psi: &ComputationDevice,
    dev_charge: &ComputationDevice,
    grid_range: (f64, f64),
    grid_range_charge: (f64, f64),
    spacing_factor: f64,
    grid_n_sample: usize,
    grid_n_charge: usize,
    bases_per_elec: &[Vec<Basis>],
    charges_fixed: &[(Vec3, f64)],
    num_electrons: usize,
    deriv_calc: DerivCalc,
) -> (
    Vec<Arr3dReal>,
    Vec<Arr3dReal>,
    Vec<Vec<Arr3d>>,
    SurfacesShared,
    Vec<SurfacesPerElec>,
) {
    let arr_real = new_data_real(grid_n_sample);

    let sfcs_one_elec = SurfacesPerElec::new(bases_per_elec[0].len(), grid_n_sample, Spin::Alpha);

    let mut surfaces_per_elec = Vec::new();
    for _ in 0..num_electrons {
        surfaces_per_elec.push(sfcs_one_elec.clone());
    }

    let mut surfaces_shared = SurfacesShared::new(
        grid_range,
        grid_range_charge,
        spacing_factor,
        grid_n_sample,
        grid_n_charge,
        num_electrons,
    );

    grid_setup::update_grid_posits(
        &mut surfaces_shared.grid_posits,
        grid_range,
        spacing_factor,
        grid_n_sample,
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
        grid_n_sample,
    );

    // These must be initialized from wave functions later.
    let mut psi_charge_all_elecs = Vec::new();
    let mut charges_electron = Vec::new();
    let mut V_from_elecs = Vec::new();

    for i_elec in 0..num_electrons {
        charges_electron.push(new_data_real(grid_n_charge));
        V_from_elecs.push(arr_real.clone());

        // todo: Call procedures::update_bases_weights etc here.
        let sfcs = &mut surfaces_per_elec[i_elec];

        // Assigning vars prevents multiple-borrow-mut vars.
        let psi = &mut sfcs.psi_per_basis;
        let psi_pp = &mut sfcs.derivs_per_basis;
        // let psi_pp_div_psi = &mut sfcs.psi_pp_div_psi_per_basis;

        wf_ops::wf_from_bases(
            dev_psi,
            psi,
            Some(psi_pp),
            // Some(psi_pp_div_psi),
            &bases_per_elec[i_elec],
            &surfaces_shared.grid_posits,
            grid_n_sample,
            deriv_calc,
        );

        let psi = &mut sfcs.psi;
        let charge_density = &mut sfcs.charge_density;
        let psi_pp = &mut sfcs.derivs;
        // let psi_pp_div_psi = &mut sfcs.psi_pp_div_psi_evaluated;

        let weights: Vec<f64> = bases_per_elec[i_elec].iter().map(|b| b.weight()).collect();
        wf_ops::mix_bases(
            psi,
            Some(charge_density),
            Some(psi_pp),
            // Some(psi_pp_div_psi),
            &sfcs.psi_per_basis,
            Some(&sfcs.derivs_per_basis),
            // Some(&sfcs.psi_pp_div_psi_per_basis),
            grid_n_sample,
            &weights,
            // Some(&mut surfaces_shared),
        );

        wf_ops::update_eigen_vals(
            &mut sfcs.V_elec_eigen,
            &mut sfcs.V_total_eigen,
            &mut sfcs.psi_pp_calculated,
            &sfcs.psi,
            &sfcs.derivs,
            // &sfcs.psi_pp_div_psi_evaluated,
            &sfcs.V_acting_on_this,
            surfaces_shared.E,
            &surfaces_shared.V_from_nuclei,
        );

        let mut psi_charge = Vec::new();

        for _ in 0..bases_per_elec[i_elec].len() {
            // Handle the charge-grid-evaluated psi.
            psi_charge.push(new_data(grid_n_charge));
        }
        wf_ops::wf_from_bases(
            dev_psi,
            &mut psi_charge,
            None,
            // None,
            &bases_per_elec[i_elec],
            &surfaces_shared.grid_posits_charge,
            grid_n_charge,
            deriv_calc,
        );

        procedures::create_elec_charge(
            &mut charges_electron[i_elec],
            &psi_charge,
            &weights,
            grid_n_charge,
        );

        psi_charge_all_elecs.push(psi_charge);

        // todo: Create electron V here
        // potential::create_V_from_elecs(
        //     dev,
        //     &mut V_from_elecs[i_elec],
        //     &state.surfaces_shared.grid_posits,
        //     &state.surfaces_shared.grid_posits_charge,
        //     &charges_other_elecs,
        //     state.grid_n_render,
        //     state.grid_n_charge,
        //     state.ui.create_2d_electron_V,
        // );

        for electron in &mut surfaces_per_elec {
            // todo: Come back to A/R
            potential::update_V_acting_on_elec(
                &mut electron.V_acting_on_this,
                &surfaces_shared.V_from_nuclei,
                // &[], // Not ready to apply V from other elecs yet.
                &new_data_real(grid_n_sample), // Not ready to apply V from other elecs yet.
                grid_n_sample,
            );
        }
    }

    wf_ops::update_combined(&mut surfaces_shared, &surfaces_per_elec, grid_n_sample);

    (
        charges_electron,
        V_from_elecs,
        psi_charge_all_elecs,
        surfaces_shared,
        surfaces_per_elec,
    )
}

fn main() {
    #[cfg(feature = "cuda")]
    let dev_charge = {
        // This is compiled in `build_`.
        let cuda_dev = CudaDevice::new(0).unwrap();
        cuda_dev
            .load_ptx(
                Ptx::from_file("./cuda.ptx"),
                "cuda",
                &[
                    "coulomb_kernel",
                    "sto_val_or_deriv_kernel",
                    // "sto_deriv_kernel", // todo: Temp workaround for out-of-resources errors.
                    "sto_val_deriv_multiple_bases_kernel",
                    "sto_val_multiple_bases_kernel",
                    "sto_val_deriv_kernel",
                ],
            )
            .unwrap();

        // println!("Using the GPU for computations.");
        ComputationDevice::Gpu(cuda_dev)
    };

    #[cfg(not(feature = "cuda"))]
    let dev_charge = ComputationDevice::Cpu;

    let dev_psi = ComputationDevice::Cpu;

    let ui_active_elec = 0;
    let max_basis_n = 1;
    let num_elecs = 1;

    let posit_charge_1 = Vec3::new(0., 0., 0.);
    let posit_charge_2 = Vec3::new(0.4, 0., 0.);

    let nuclei = vec![
        (posit_charge_1, Q_PROT * num_elecs as f64),
        // (posit_charge_2, Q_PROT * num_elecs as f64),
    ];

    // Outer of these is per-elec.
    let mut bases_per_elec = Vec::new();

    for _ in 0..num_elecs {
        bases_per_elec.push(Vec::new());
    }

    wf_ops::initialize_bases(&mut bases_per_elec[ui_active_elec], &nuclei, max_basis_n);

    // todo: This is getting weird re multiple electrons; perhaps you should switch
    // todo an approach where bases don't have weights, but you use a separate
    // weights array.
    for i_elec in 1..num_elecs {
        bases_per_elec[i_elec] = bases_per_elec[0].clone();
    }

    // H ion nuc dist is I believe 2 bohr radii.
    // let charges = vec![(Vec3::new(-1., 0., 0.), Q_PROT), (Vec3::new(1., 0., 0.), Q_PROT)];

    // let (grid_min, grid_max) = grid_setup::choose_grid_limits(&nuclei);

    // todoFigure out why you get incorrect answers if these 2 grids don't line up.
    // todo: FOr now, you can continue with matching them if you wish.
    let (grid_min_render, grid_max_render) = (-GRID_MAX_RENDER, GRID_MAX_RENDER);
    let (grid_min_charge, grid_max_charge) = (-GRID_MAX_CHARGE, GRID_MAX_CHARGE);

    // let spacing_factor = 1.6;
    // Currently, must be one as long as used with elec-elec charge.
    let spacing_factor = SPACING_FACTOR_DEFAULT;

    let grid_n = GRID_N_RENDER_DEFAULT;
    let grid_n_charge = GRID_N_CHARGE_DEFAULT;

    let psi_pp_calc = DerivCalc::Numeric;

    let (charges_electron, V_from_elecs, psi_charge, surfaces_shared, surfaces_per_elec) =
        init_from_grid(
            &dev_charge,
            &dev_psi,
            (grid_min_render, grid_max_render),
            (grid_min_charge, grid_max_charge),
            spacing_factor,
            grid_n,
            grid_n_charge,
            &bases_per_elec,
            &nuclei,
            num_elecs,
            psi_pp_calc,
        );

    let surface_descs_per_elec = vec![
        SurfaceDesc::new("V", true),
        SurfaceDesc::new("ψ", false),
        SurfaceDesc::new("ψ im", false),
        SurfaceDesc::new("ρ", false),
        SurfaceDesc::new("ψ'' calc", false),
        SurfaceDesc::new("ψ'' calc im", false),
        SurfaceDesc::new("ψ'' meas", false),
        SurfaceDesc::new("ψ'' meas im", false),
        SurfaceDesc::new("Elec V from ψ ", false),
        SurfaceDesc::new("Total V from ψ", true),
        SurfaceDesc::new("V'_elec", false),
        SurfaceDesc::new("ψ (L)", false),
        SurfaceDesc::new("ψ_z (L)", false),
    ];

    let surface_descs_combined = vec![
        SurfaceDesc::new("V", true),
        SurfaceDesc::new("ψ_α", false),
        SurfaceDesc::new("ψ_β", false),
        SurfaceDesc::new("ψ_α im", false),
        SurfaceDesc::new("ψ_β im", false),
        SurfaceDesc::new("ρ_α", false),
        SurfaceDesc::new("ρ_β", false),
        SurfaceDesc::new("ρ", true),
        SurfaceDesc::new("ρ spin", true),
    ];

    let state = State {
        dev_charge,
        dev_psi,
        deriv_calc: psi_pp_calc,
        charges_fixed: nuclei,
        charges_from_electron: charges_electron,
        V_from_elecs,
        bases: bases_per_elec,
        psi_charge,
        surfaces_shared,
        surfaces_per_elec,
        surface_descs_per_elec,
        surface_descs_combined,
        grid_n_render: grid_n,
        grid_n_charge,
        grid_range_render: (grid_min_render, grid_max_render),
        grid_range_charge: (grid_min_charge, grid_max_charge),
        sample_factor_render: spacing_factor,
        max_basis_n,
        num_elecs,
        ui: Default::default(),
    };

    render::render(state);
}
