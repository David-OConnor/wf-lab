#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(uncommon_codepoints)]
#![allow(confusable_idents)]
#![allow(non_upper_case_globals)]
#![allow(non_ascii_idents)]
//  todo: Ideally temp, while we experiment. Loads of unused functions.
#![allow(dead_code)]
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

// April 2024 shower thought: On what defines the quantum numbers. If you separate these, you meet
// the Spin statistics/exclusion/exchange requirements.

// *********

// n: Energy, as used in wave equations.

// L^2ψ = hbar^2 l (l+1) ψ

// l: Orbital angular momentum. Related to shape?

// m: phase change over angle?? WF is real while m=0.  (starting guess)
//  Observations. If m=0, no phase change. m=1: Phase rotates once/tau. m=2: Phase rotates twice/tau.
//  if m=negative, the rotation is reversed.

// A related property for `m`: Perhaps phase must change continously around the circle. Not linearly perse,
// but along a given radial, phase must be equal?

// todo: Perhaps worth trying: Can you set up an equation where phase must meet the constraints you
// todo have for m above, then solve the rest to evaluate a trial wave function?

// spin: Related to angular momentum, and perhaps we can treat in an at-hoc manner. (?) Related to
// orientation of spin axis related to the orbital plane?

// *********

// APril 2024
// Consider a phase space model. This may mean, in addition to your 3d space model (grid) of x, y, z,
// you include px, py, and pz as well. (Think through how this would work, and what benefits it provides)
// Note that E + (px^2 + py^2 + pz^2)/2m

// Also: Can we model space as a discrete (3D, 4D with time etc) grid with dx = h or hbar? Then consider
// the possible states to be these discrete grid items. Sounds unfeasible: h = 1.616255×10−35 m, which is
// *much* smaller than the hartree unit scale we tend to model atoms with.

#[cfg(feature = "cuda")]
use cudarc::{driver::CudaDevice, nvrtc::Ptx};
use lin_alg::f64::Vec3;

mod angular_p;
mod basis_finder;
mod basis_init;
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
    dirac::BasisSpinor,
    grid_setup::{new_data, new_data_real, Arr3d, Arr3dReal},
    types::{ComputationDevice, SurfacesPerElec, SurfacesShared},
    ui::procedures,
    wf_ops::{DerivCalc, Spin, Q_PROT},
};

const NUM_SURFACES_PER_ELEC: usize = 11;

const SPACING_FACTOR_DEFAULT: f64 = 1.;
const GRID_MAX_RENDER: f64 = 10.;
const GRID_MAX_CHARGE: f64 = 12.;
const GRID_N_RENDER_DEFAULT: usize = 60;
const GRID_N_CHARGE_DEFAULT: usize = 61;

const RENDER_SPINOR: bool = false;
const RENDER_L: bool = true;

// todo: Consider a spherical grid centered perhaps on the system center-of-mass, which
// todo less precision further away?

// 2/17/2024. A thought: Consider the atom as the base stable unit. Molecules are composed  of atoms.
// With this in mind, perhaps you are not seeking to model using arbitrary potentials: You are seeking
// to model using atoms as the baseline, and molecules composed of discrete atoms.

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
    pub bases_spinor: Vec<Vec<BasisSpinor>>,
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
    // /// When finding and initializing basis, this is the maximum n quantum number.
    // pub max_basis_n: u16,
    pub num_elecs: usize,
    pub ui: StateUi,
}

impl State {
    pub fn new(
        num_elecs: usize,
        dev_psi: ComputationDevice,
        dev_charge: ComputationDevice,
    ) -> Self {
        println!("Initializing state...");

        let posit_charge_1 = Vec3::new(0., 0., 0.);
        let posit_charge_2 = Vec3::new(0.4, 0., 0.);

        let nuclei = vec![
            (posit_charge_1, Q_PROT * num_elecs as f64),
            // (posit_charge_2, Q_PROT * num_elecs as f64),
        ];

        // Outer of these is per-elec.
        let mut bases_per_elec = Vec::new();
        let mut bases_per_elec_spinor = Vec::new();

        // Initialize bases.
        for i_elec in 0..num_elecs {
            let mut bases_this_elec = Vec::new();
            let mut bases_this_elec_spinor = Vec::new();

            // todo: Kludge for Li
            let n = if i_elec > 1 { 2 } else { 1 };

            basis_init::initialize_bases(&mut bases_this_elec, &nuclei, n);
            wf_ops::initialize_bases_spinor(&mut bases_this_elec_spinor, &nuclei, n);

            bases_per_elec.push(bases_this_elec);
            bases_per_elec_spinor.push(bases_this_elec_spinor);
        }

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

        println!("Initializing from grid...");
        let (charges_electron, V_from_elecs, psi_charge, surfaces_shared, surfaces_per_elec) =
            init_from_grid(
                &dev_psi,
                &dev_charge,
                (grid_min_render, grid_max_render),
                (grid_min_charge, grid_max_charge),
                spacing_factor,
                grid_n,
                grid_n_charge,
                &bases_per_elec,
                &bases_per_elec_spinor,
                &nuclei,
                num_elecs,
                psi_pp_calc,
            );

        println!("Grid init complete.");

        let mut surface_descs_per_elec = vec![
            SurfaceDesc::new(SurfaceToRender::V, true),
            SurfaceDesc::new(SurfaceToRender::Psi, false),
            SurfaceDesc::new(SurfaceToRender::PsiIm, false),
            SurfaceDesc::new(SurfaceToRender::ChargeDensity, false),
            SurfaceDesc::new(SurfaceToRender::PsiPpCalc, false),
            SurfaceDesc::new(SurfaceToRender::PsiPpCalcIm, false),
            SurfaceDesc::new(SurfaceToRender::PsiPpMeas, false),
            SurfaceDesc::new(SurfaceToRender::PsiPpMeasIm, false),
            SurfaceDesc::new(SurfaceToRender::ElecVFromPsi, false),
            SurfaceDesc::new(SurfaceToRender::TotalVFromPsi, true),
            // SurfaceDesc::new(SurfaceToRender::VPElec, false),
            SurfaceDesc::new(SurfaceToRender::H, false),
            SurfaceDesc::new(SurfaceToRender::HIm, false),
        ];

        if RENDER_L {
            surface_descs_per_elec.append(&mut vec![
                SurfaceDesc::new(SurfaceToRender::LSq, false),
                SurfaceDesc::new(SurfaceToRender::LSqIm, false),
                SurfaceDesc::new(SurfaceToRender::LZ, false),
                SurfaceDesc::new(SurfaceToRender::LZIm, false),
                // todo: These likely temp to verify.
                // SurfaceDesc::new("dx", false),
                // SurfaceDesc::new("dy", false),
                // SurfaceDesc::new("dz", false),
                // SurfaceDesc::new("d2x", false),
                // SurfaceDesc::new("d2y", false),
                // SurfaceDesc::new("d2z", false),
            ])
        }

        if RENDER_SPINOR {
            surface_descs_per_elec.append(&mut vec![
                SurfaceDesc::new(SurfaceToRender::PsiSpinor0, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinor1, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinor2, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinor3, false),
                // Calculated, to compare to the trial.
                SurfaceDesc::new(SurfaceToRender::PsiSpinorCalc0, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinorCalc0, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinorCalc0, false),
                SurfaceDesc::new(SurfaceToRender::PsiSpinorCalc0, false),
            ])
        }

        // todo: Come back to this, and add appropriate SurfaceToRender variants next time you use this.
        let surface_descs_combined = vec![
            //     SurfaceDesc::new("V", true),
            //     SurfaceDesc::new("ψ_α", false),
            //     SurfaceDesc::new("ψ_β", false),
            //     SurfaceDesc::new("ψ_α im", false),
            //     SurfaceDesc::new("ψ_β im", false),
            //     SurfaceDesc::new("ρ_α", false),
            //     SurfaceDesc::new("ρ_β", false),
            //     SurfaceDesc::new("ρ", true),
            //     SurfaceDesc::new("ρ spin", true),
        ];

        println!("State init complete.");

        Self {
            dev_charge,
            dev_psi,
            deriv_calc: psi_pp_calc,
            charges_fixed: nuclei,
            charges_from_electron: charges_electron,
            V_from_elecs,
            bases: bases_per_elec,
            bases_spinor: bases_per_elec_spinor,
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
            // max_basis_n,
            num_elecs,
            ui: Default::default(),
        }
    }
}

pub struct SurfaceDesc {
    pub surface: SurfaceToRender,
    pub visible: bool,
}

impl SurfaceDesc {
    /// This constructor simplifies syntax.
    pub fn new(surface: SurfaceToRender, visible: bool) -> Self {
        Self { surface, visible }
    }
}

/// Run this whenever n changes. Ie, at init, or when n changes in the GUI.
/// // todo: Refactor/rethink this fn. It's kind of a param mess.
pub fn init_from_grid(
    dev_psi: &ComputationDevice,
    dev_charge: &ComputationDevice,
    grid_range: (f64, f64),
    grid_range_charge: (f64, f64),
    spacing_factor: f64,
    grid_n_sample: usize,
    grid_n_charge: usize,
    bases_per_elec: &[Vec<Basis>],
    bases_per_elec_spinor: &[Vec<BasisSpinor>],
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
        let spinor = &mut sfcs.spinor_per_basis;
        let spinor_derivs = &mut sfcs.spinor_derivs_per_basis;

        wf_ops::wf_from_bases(
            dev_psi,
            psi,
            Some(psi_pp),
            &bases_per_elec[i_elec],
            &surfaces_shared.grid_posits,
            deriv_calc,
        );

        wf_ops::wf_from_bases_spinor(
            dev_psi,
            spinor,
            Some(spinor_derivs),
            &bases_per_elec_spinor[i_elec],
            &surfaces_shared.grid_posits,
        );

        let psi = &mut sfcs.psi;
        let charge_density = &mut sfcs.charge_density;
        let psi_pp = &mut sfcs.derivs;
        let spinor = &mut sfcs.spinor;
        let spinor_derivs = &mut sfcs.spinor_derivs;

        let weights: Vec<f64> = bases_per_elec[i_elec].iter().map(|b| b.weight()).collect();
        wf_ops::mix_bases(
            psi,
            Some(charge_density),
            Some(psi_pp),
            &sfcs.psi_per_basis,
            Some(&sfcs.derivs_per_basis),
            &weights,
        );

        wf_ops::mix_bases_spinor(
            spinor,
            None, // todo
            Some(spinor_derivs),
            &sfcs.spinor_per_basis,
            Some(&sfcs.spinor_derivs_per_basis),
            &weights,
        );

        wf_ops::update_eigen_vals(
            &mut sfcs.V_elec_eigen,
            &mut sfcs.V_total_eigen,
            &mut sfcs.psi_pp_calculated,
            &sfcs.psi,
            &sfcs.derivs,
            // &sfcs.psi_pp_div_psi_evaluated,
            &sfcs.V_acting_on_this,
            sfcs.E,
            &surfaces_shared.V_from_nuclei,
            &surfaces_shared.grid_posits,
            &mut sfcs.psi_fm_H,
            &mut sfcs.psi_fm_L2,
            &mut sfcs.psi_fm_Lz,
        );

        wf_ops::update_eigen_vals_spinor(
            &mut sfcs.spinor_calc,
            spinor_derivs,
            [-0.5; 4], // todo temp
            &sfcs.V_acting_on_this,
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
            &bases_per_elec[i_elec],
            &surfaces_shared.grid_posits_charge,
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

// todo: Move this A/R
#[derive(Clone, Copy)]
pub enum SurfaceToRender {
    V,
    /// These may be overloaded to mean mag/phase
    Psi,
    PsiIm,
    ChargeDensity,
    PsiPpCalc,
    PsiPpCalcIm,
    PsiPpMeas,
    PsiPpMeasIm,
    ElecVFromPsi,
    TotalVFromPsi,
    VPElec,
    /// Hamiltonian
    H,
    HIm,
    /// Angular momentum
    LSq,
    LSqIm,
    LZ,
    LZIm,
    // Spinor
    // todo: Im of these A/R
    PsiSpinor0,
    PsiSpinor1,
    PsiSpinor2,
    PsiSpinor3,
    PsiSpinorCalc0,
    PsiSpinorCalc1,
    PsiSpinorCalc2,
    PsiSpinorCalc3,
}

impl SurfaceToRender {
    pub fn name(&self) -> String {
        match self {
            Self::V => "V",
            Self::Psi => "ψ",
            Self::PsiIm => "ψ im",
            Self::ChargeDensity => "ρ",
            Self::PsiPpCalc => "ψ'' calc",
            Self::PsiPpCalcIm => "ψ'' calc im",
            Self::PsiPpMeas => "ψ'' meas",
            Self::PsiPpMeasIm => "ψ'' meas im",
            Self::ElecVFromPsi => "Elec V from ψ",
            Self::TotalVFromPsi => "V from ψ",
            Self::VPElec => "V' elec",
            Self::H => "H",
            Self::HIm => "H im",
            Self::LSq => "L^2",
            Self::LSqIm => "L^2 im",
            Self::LZ => "L_z",
            Self::LZIm => "L_z im",
            Self::PsiSpinor0 => "ψ0",
            Self::PsiSpinor1 => "ψ1",
            Self::PsiSpinor2 => "2",
            Self::PsiSpinor3 => "ψ3",
            Self::PsiSpinorCalc0 => "ψ0_c",
            Self::PsiSpinorCalc1 => "ψ_c1",
            Self::PsiSpinorCalc2 => "ψ2_c",
            Self::PsiSpinorCalc3 => "ψ3_c",
        }
        .to_string()
    }
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

    let num_elecs = 1;

    render::render(State::new(num_elecs, dev_psi, dev_charge));
}
