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
//!
//! [Eliminating electron self-repulision](https://arxiv.org/pdf/2206.09472)
//! [Quantum electrostatics](https://arxiv.org/pdf/2003.07473)

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

mod basis_finder;
// mod basis_init;
mod basis_wfs;
mod complex_nums;
mod core_calcs;
mod dirac;
mod eval;
mod experiminting;
mod field_visuals;
mod forces;
mod gauss_padding;
#[cfg(feature = "cuda")]
mod gpu;
mod grid_setup;
mod interp;
mod num_diff;
mod presets;
mod render;
mod state;
mod types;
mod ui;
mod util;
mod wf_ops;

use crate::{state::State, types::ComputationDevice, wf_ops::Q_PROT};

const SPACING_FACTOR_DEFAULT: f64 = 1.;

const GRID_MAX_RENDER: f64 = 3.;
const GRID_MAX_CHARGE: f64 = 8.;
const GRID_MAX_GRADIENT: f64 = GRID_MAX_RENDER;

const GRID_N_RENDER_DEFAULT: usize = 70;

// todo: We have this temporarily high for force calc troubleshooting
const GRID_N_CHARGE_DEFAULT: usize = 61;
const GRID_N_GRADIENT_DEFAULT: usize = 8;
// const GRID_N_CHARGE_DEFAULT: usize = 91;

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

#[derive(Clone, Copy, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
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
    /// Allows rotating the view, vice using a Z-axis slider alone to view the third dimension
    pub hidden_axis: Axis,
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
            hidden_axis: Axis::Z,
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
    /// The difference between V acting on this electron, and V calculated from the Eigen function:
    /// V - V from psi
    VDiff,
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
    // Experimenting
    // /// A gradient of the electric field.
    // ElecFieldGradient,
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
            Self::VDiff => "dV",
            Self::PsiSpinor0 => "ψ0",
            Self::PsiSpinor1 => "ψ1",
            Self::PsiSpinor2 => "2",
            Self::PsiSpinor3 => "ψ3",
            Self::PsiSpinorCalc0 => "ψ0_c",
            Self::PsiSpinorCalc1 => "ψ_c1",
            Self::PsiSpinorCalc2 => "ψ2_c",
            Self::PsiSpinorCalc3 => "ψ3_c",
            // Self::ElecFieldGradient => "E ∇",
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
    // let num_elecs = 0;

    // render::render(State::new(num_elecs, dev_psi, dev_charge));
    render::render(State::new(dev_psi, dev_charge));
}
