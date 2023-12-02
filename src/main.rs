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

use std::mem;

#[cfg(feature = "cuda")]
use cudarc::{driver::CudaDevice, nvrtc::Ptx};

use lin_alg2::f64::Vec3;

mod basis_finder;
mod basis_wfs;
mod complex_nums;
mod eigen_fns;
mod elec_elec;
mod eval;
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
    wf_ops::Q_PROT,
};

const NUM_SURFACES: usize = 11;

const SPACING_FACTOR_DEFAULT: f64 = 1.;
const GRID_MAX_RENDER: f64 = 16.;
const GRID_MAX_CHARGE: f64 = 18.;
const GRID_N_RENDER_DEFAULT: usize = 80;
const GRID_N_CHARGE_DEFAULT: usize = 91;

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
            create_2d_electron_V: true,
            create_3d_electron_V: false,
        }
    }
}

pub struct State {
    pub dev: ComputationDevice,
    /// Eg, Nuclei (position, charge amt), per the Born-Oppenheimer approximation. Charges over space
    /// due to electrons are stored in `Surfaces`.
    pub charges_fixed: Vec<(Vec3, f64)>,
    /// Charges from electrons, over 3d space. Each value is the charge created by a single electron. Computed from <ψ|ψ>.
    /// This is not part of `SurfacesPerElec` since we use all values at once (or all except one)
    /// when calculating the potential. (easier to work with API)
    pub charges_from_electron: Vec<Arr3dReal>,
    /// Also stored here vice part of per-elec structs due to borrow-limiting on struct fields.
    pub V_from_elecs: Vec<Arr3dReal>, // todo: Now that we have charge_from_elec, do we want this? Probably
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
    pub surface_data: [SurfaceData; NUM_SURFACES],
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
    dev: &ComputationDevice,
    grid_range: (f64, f64),
    grid_range_charge: (f64, f64),
    spacing_factor: f64,
    grid_n_sample: usize,
    grid_n_charge: usize,
    bases_per_elec: &[Vec<Basis>],
    charges_fixed: &[(Vec3, f64)],
    num_electrons: usize,
) -> (
    Vec<Arr3dReal>,
    Vec<Arr3dReal>,
    Vec<Vec<Arr3d>>,
    SurfacesShared,
    Vec<SurfacesPerElec>,
) {
    let arr_real = new_data_real(grid_n_sample);

    let sfcs_one_elec = SurfacesPerElec::new(bases_per_elec[0].len(), grid_n_sample, grid_n_charge);

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
        let psi_pp = &mut sfcs.psi_pp_per_basis;

        wf_ops::wf_from_bases(
            dev,
            psi,
            Some(psi_pp),
            &bases_per_elec[i_elec],
            &surfaces_shared.grid_posits,
            grid_n_sample,
        );

        let psi = &mut sfcs.psi;
        let psi_pp = &mut sfcs.psi_pp_evaluated;

        let weights: Vec<f64> = bases_per_elec[i_elec].iter().map(|b| b.weight()).collect();
        wf_ops::mix_bases(
            psi,
            Some(psi_pp),
            &sfcs.psi_per_basis,
            Some(&sfcs.psi_pp_per_basis),
            grid_n_sample,
            &weights,
        );

        wf_ops::update_eigen_vals(
            &mut sfcs.V_elec_eigen,
            &mut sfcs.V_total_eigen,
            &mut sfcs.psi_pp_calculated,
            &sfcs.psi,
            &sfcs.psi_pp_evaluated,
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
            dev,
            &mut psi_charge,
            None,
            &bases_per_elec[i_elec],
            &surfaces_shared.grid_posits_charge,
            grid_n_charge,
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
    let dev = {
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

        println!("Using the GPU for computations.");
        ComputationDevice::Gpu(cuda_dev)
    };

    #[cfg(not(feature = "cuda"))]
    let dev = ComputationDevice::Cpu;

    let posit_charge_1 = Vec3::new(0., 0., 0.);
    let _posit_charge_2 = Vec3::new(1., 0., 0.);

    let nuclei = vec![
        // (posit_charge_1, Q_PROT * 2.), // helium
        (posit_charge_1, Q_PROT * 3.), // lithium
                                       // (posit_charge_1, Q_PROT * 1.), // Hydrogen
                                       // (posit_charge_2, Q_PROT),
                                       // (Vec3::new(0., 1., 0.), Q_ELEC),
    ];

    let ui_active_elec = 0;
    let max_basis_n = 2;
    let num_elecs = 3;

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

    let (charges_electron, V_from_elecs, psi_charge, surfaces_shared, surfaces_per_elec) =
        init_from_grid(
            &dev,
            (grid_min_render, grid_max_render),
            (grid_min_charge, grid_max_charge),
            spacing_factor,
            grid_n,
            grid_n_charge,
            &bases_per_elec,
            &nuclei,
            num_elecs,
        );

    let surface_data = [
        SurfaceData::new("V", true),
        SurfaceData::new("ψ", false),
        SurfaceData::new("ψ im", false),
        SurfaceData::new("ψ²", false),
        SurfaceData::new("ψ'' calc", false),
        SurfaceData::new("ψ'' calc im", false),
        SurfaceData::new("ψ'' meas", false),
        SurfaceData::new("ψ'' meas im", false),
        SurfaceData::new("Elec V from ψ ", false),
        SurfaceData::new("Total V from ψ", true),
        SurfaceData::new("V'_elec", false),
    ];
    //
    // let mut charge_from_elecs = Vec::new();
    // for _ in 0..num_elecs {
    //     charge_from_elecs.push(new_data_real(grid_n_charge));
    // }

    let state = State {
        dev,
        charges_fixed: nuclei,
        charges_from_electron: charges_electron,
        V_from_elecs,
        bases: bases_per_elec,
        psi_charge,
        surfaces_shared,
        surfaces_per_elec,
        surface_data,
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
