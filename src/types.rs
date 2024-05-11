use std::{ops::Add, sync::Arc};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
use lin_alg::f64::Vec3;

use crate::{
    basis_wfs::Basis,
    complex_nums::Cplx,
    dirac::{Spinor3, SpinorDerivsTypeD3, SpinorDerivsTypeDInner3},
    elec_elec::WaveFunctionMultiElec,
    grid_setup::{
        self, new_data, new_data_2d, new_data_2d_real, new_data_2d_vec, new_data_real,
        new_data_vec, Arr2d, Arr2dReal, Arr2dVec, Arr3d, Arr3dReal, Arr3dVec,
    },
    wf_ops,
    wf_ops::{DerivCalc, Spin},
    Axis,
};

#[derive(Debug, Clone)]
pub enum ComputationDevice {
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu(Arc<CudaDevice>),
}

impl Default for ComputationDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

pub struct SurfacesShared {
    /// Represents points on a grid, for our non-uniform grid.
    // pub grid_posits: Arr3dVec,
    pub grid_posits: Arr2dVec,
    pub grid_posits_charge: Arr3dVec,
    /// Potential from nuclei, and all electrons
    pub V_total: Arr3dReal,
    /// Potential from nuclei only. We use this as a baseline for individual electron
    /// potentials, prior to summing over V from other electrons.
    // pub V_from_nuclei: Arr3dReal,
    pub V_from_nuclei: Arr2dReal,
    // todo: This may not be a good model: the wave function isn't a function of position
    // mapped to a value for multi-elecs. It's a function of a position for each elec.
    pub psi: WaveFunctionMultiElec,
    // pub E: f64,
    // /// 2023-08-17: Another attempt at a 3d-grid-based save function
    // pub psi_numeric: Arr3d,
    /// In case we want to explore something like DFT
    // pub charge_density_dft: Arr3dReal,
    /// todo: We are experimenting with separating psi by spin, but otherwise combining for electrons.
    pub psi_alpha: Arr3d,
    pub psi_beta: Arr3d,
    /// Charge density for all spin alpha electrons.
    pub charge_alpha: Arr3dReal,
    /// Charge density for all spin beta electrons.
    pub charge_beta: Arr3dReal,
    // Splitting up by charge density and spin density should be equivalent to splitting by
    // spin up and spin down.
    /// Electron density total
    pub charge_density_all: Arr3dReal,
    /// Charge density for all spin beta electrons.
    pub spin_density: Arr3dReal,
}

impl SurfacesShared {
    pub fn new(
        grid_range: (f64, f64),
        grid_range_charge: (f64, f64),
        spacing_factor: f64,
        n_grid: usize,
        n_grid_charge: usize,
        num_elecs: usize,
        axis_hidden: Axis,
    ) -> Self {
        let data = new_data(n_grid);
        let data_real = new_data_real(n_grid);
        let data_real_2d = new_data_2d_real(n_grid);

        // let mut grid_posits = new_data_vec(n_grid);
        let mut grid_posits = new_data_2d_vec(n_grid);
        // grid_setup::update_grid_posits(&mut grid_posits, grid_range, spacing_factor, n_grid);
        // todo: z_init at 0.; We will need to run this whenever we change the z slider.
        grid_setup::update_grid_posits_2d(
            &mut grid_posits,
            grid_range,
            spacing_factor,
            0.,
            n_grid,
            axis_hidden,
        );

        let mut grid_posits_charge = new_data_vec(n_grid_charge);

        // spacing factor is always 1 for charge grid. (for now at least)
        grid_setup::update_grid_posits(
            &mut grid_posits_charge,
            grid_range_charge,
            1.,
            n_grid_charge,
        );

        Self {
            grid_posits,
            grid_posits_charge,
            V_total: data_real.clone(),
            V_from_nuclei: data_real_2d.clone(),
            psi: WaveFunctionMultiElec::new(num_elecs, n_grid),
            // E: -0.50,
            // psi_numeric: data,
            // charge_density_dft: data_real.clone(),
            psi_alpha: data.clone(),
            psi_beta: data,
            charge_alpha: data_real.clone(),
            charge_beta: data_real.clone(),
            charge_density_all: data_real.clone(),
            spin_density: data_real,
        }
    }
}

#[derive(Clone, Default, Debug)]
/// A set of derivatives at a single point
/// Organization: index, da
pub struct DerivativesSingle {
    pub dx: Cplx,
    pub dy: Cplx,
    pub dz: Cplx,
    pub d2x: Cplx,
    pub d2y: Cplx,
    pub d2z: Cplx,
    pub d2_sum: Cplx,
}

impl Add<&Self> for DerivativesSingle {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let mut result = self;

        result.dx = result.dx + rhs.dx;
        result.dy = result.dy + rhs.dy;
        result.dz = result.dz + rhs.dz;
        result.d2x = result.d2x + rhs.d2x;
        result.d2y = result.d2y + rhs.d2y;
        result.d2z = result.d2z + rhs.d2z;
        result.d2_sum = result.d2_sum + rhs.d2_sum;

        result
    }
}

/// Used for storing various derivatives, on grids, used in our eigenfunctions. Most are used only
/// by the momentum eigen functions.
///
/// It would be easier in some cases to store an Arr3D of DerivativesSingle, but it makes other parts we have intractable,
/// or very complicated
#[derive(Clone)]
pub struct Derivatives {
    pub dx: Arr3d,
    pub dy: Arr3d,
    pub dz: Arr3d,
    pub d2x: Arr3d,
    pub d2y: Arr3d,
    pub d2z: Arr3d,
    /// Used in the energy eigenfunction, ie Schrodinger equation.
    pub d2_sum: Arr3d,
}

impl Derivatives {
    pub fn new(grid_n: usize) -> Self {
        let data = new_data(grid_n);

        Self {
            dx: data.clone(),
            dy: data.clone(),
            dz: data.clone(),
            d2x: data.clone(),
            d2y: data.clone(),
            d2z: data.clone(),
            d2_sum: data,
        }
    }
}

#[derive(Clone)]
pub struct Derivatives2D {
    pub dx: Arr2d,
    pub dy: Arr2d,
    pub dz: Arr2d,
    pub d2x: Arr2d,
    pub d2y: Arr2d,
    pub d2z: Arr2d,
    /// Used in the energy eigenfunction, ie Schrodinger equation.
    pub d2_sum: Arr2d,
}

impl Derivatives2D {
    pub fn new(grid_n: usize) -> Self {
        let data = new_data_2d(grid_n);

        Self {
            dx: data.clone(),
            dy: data.clone(),
            dz: data.clone(),
            d2x: data.clone(),
            d2y: data.clone(),
            d2z: data.clone(),
            d2_sum: data,
        }
    }
}

// // todo: Switch to this A/R. Likely cleaner code.
// /// Used for storing various derivatives, on grids, used in our eigenfunctions. Most are used only
// /// by the momentum eigen functions.
// #[derive(Clone, Default)]
// pub struct Derivatives {
//     pub v: Arr3dDeriv,
// }

// /// Extract references to the `d2_sum` field, for use with APIs that accept &[Arr3d], eg for psi_pp.
// pub fn extract_d2_sum(derivs: &[Derivatives]) -> Vec<&Arr3d> {
//     derivs.iter().map(|d| d.d2_sum).collect()
// }
//
// /// Extract references to the `d2_sum` field, for use with APIs that accept &[Arr3d], eg for psi_pp.
// pub fn extract_d2_sum_mut(derivs: &mut [Derivatives]) -> Vec<&mut Arr3d> {
//     derivs.iter().map(|d| d.d2_sum).collect()
// }

/// Represents important data, in describe 3D arrays.
/// We use Vecs, since these can be large, and we don't want
/// to put them on the stack. Although, they are fixed-size.
///
/// `Vec`s here generally mean per-electron.
#[derive(Clone)]
pub struct SurfacesPerElec {
    pub spin: Spin,
    /// V from the nucleii, and all other electrons. Does not include this electron's charge.
    /// We use this as a cache instead of generating it on the fly.
    // pub V_acting_on_this: Arr3dReal,
    pub V_acting_on_this: Arr2dReal,
    pub psi: Arr2d,
    /// Electron charge * normalized psi^2
    pub charge_density_2d: Arr2dReal,
    /// Electron charge * normalized psi^2
    pub charge_density: Arr3dReal,
    /// From the Schrodinger equation based on psi and the other parameters.
    pub psi_pp_calculated: Arr2d,
    /// From an analytic or numeric computation from basis functions.
    // pub derivs: Derivatives,
    pub derivs: Derivatives2D,
    // todo: An experiment where we analytically calculate this directly.
    // pub psi_pp_div_psi_evaluated: Arr3dReal,
    pub psi_per_basis: Vec<Arr2d>,
    pub derivs_per_basis: Vec<Derivatives2D>,
    // // todo: An experiment where we analytically calculate this directly.
    // pub psi_pp_div_psi_per_basis: Vec<Arr3dReal>,
    /// Aux surfaces are for misc visualizations
    pub V_elec_eigen: Arr2dReal,
    pub V_total_eigen: Arr2dReal,
    pub aux3: Arr2dReal,
    /// Hamiltonian (Schrodinger equation energery eigenfunction)
    pub psi_fm_H: Arr2d,
    // experiments with angular momentum
    pub psi_fm_L2: Arr2d,
    pub psi_fm_Lz: Arr2d,
    // todo: Update these types A/R based on which ordering variant you use.
    /// Trial wave function spinor
    pub spinor: Spinor3,
    /// Spinor, as calcualted using the trial wave function, and the dirac equation, using
    /// its derivatives.
    pub spinor_calc: Spinor3,
    pub spinor_derivs: SpinorDerivsTypeD3,
    pub spinor_per_basis: Vec<Spinor3>,
    pub spinor_derivs_per_basis: Vec<SpinorDerivsTypeD3>,
    pub E_dirac: (f64, f64, f64, f64),
    // todo: Temp experimenting with making orbitals from modded H ones. This may be used  to evaluate
    // todo how to modify an H orbital to arrive at the ones we've solved for.
    pub orb_sub: Arr3d,
    pub E: f64,
}

impl SurfacesPerElec {
    /// Fills with 0.s
    pub fn new(num_bases: usize, grid_n: usize, spin: Spin) -> Self {
        let data = new_data(grid_n);
        let data_2d = new_data_2d(grid_n);
        let data_real = new_data_real(grid_n);
        let data_2d_real = new_data_2d_real(grid_n);
        // let derivs = Derivatives::new(grid_n);
        let derivs = Derivatives2D::new(grid_n);

        let spinor_d_inner = SpinorDerivsTypeDInner3 {
            dx: data.clone(),
            dy: data.clone(),
            dz: data.clone(),
        };

        let derivs_spinor = SpinorDerivsTypeD3 {
            c0: spinor_d_inner.clone(),
            c1: spinor_d_inner.clone(),
            c2: spinor_d_inner.clone(),
            c3: spinor_d_inner,
        };

        // Set up a regular grid using this; this will allow us to convert to an irregular grid
        // later, once we've verified this works.

        let mut psi_per_basis = Vec::new();
        let mut derivs_per_basis = Vec::new();
        let mut spinor_per_basis = Vec::new();
        let mut spinor_derivs_per_basis = Vec::new();

        let mut psi_pp_div_psi_per_basis = Vec::new();
        for _ in 0..num_bases {
            psi_per_basis.push(data_2d.clone());
            derivs_per_basis.push(derivs.clone());
            psi_pp_div_psi_per_basis.push(data_real.clone());

            spinor_per_basis.push(Spinor3 {
                c0: data.clone(),
                c1: data.clone(),
                c2: data.clone(),
                c3: data.clone(),
            });
            spinor_derivs_per_basis.push(derivs_spinor.clone());
        }

        Self {
            spin,
            V_acting_on_this: data_2d_real.clone(),
            psi: data_2d.clone(),
            charge_density_2d: data_2d_real.clone(),
            charge_density: data_real.clone(),
            psi_pp_calculated: data_2d.clone(),
            // derivs: data.clone(),
            derivs,
            // psi_pp_div_psi_evaluated: data_real.clone(),
            psi_per_basis,
            // derivs_per_basis: psi_pp_per_basis,
            derivs_per_basis,
            // psi_pp_div_psi_per_basis,
            // charge: new_data_real(n_grid_charge),
            V_elec_eigen: data_2d_real.clone(),
            V_total_eigen: data_2d_real.clone(),
            aux3: data_2d_real,
            psi_fm_H: data_2d.clone(),
            psi_fm_L2: data_2d.clone(),
            psi_fm_Lz: data_2d.clone(),
            spinor: Spinor3::new(grid_n),
            spinor_calc: Spinor3::new(grid_n),
            // todo: This is likely wrong; need to populate with grid_n.
            spinor_derivs: SpinorDerivsTypeD3::new(grid_n),
            spinor_per_basis,
            spinor_derivs_per_basis,
            E_dirac: (0., 0., 0., 0.),
            orb_sub: data.clone(),
            E: -0.5,
        }
    }
}

// todo: What do you use this for?
pub struct EvalDataShared {
    pub posits: Vec<Vec3>,
    pub V_from_nuclei: Vec<f64>,
    pub grid_n: usize, // len of data here and in associated per-elec data.
}

impl EvalDataShared {
    pub fn new(nuclei: &[(Vec3, f64)]) -> Self {
        let posits = grid_setup::find_sample_points(nuclei);
        let n = posits.len();

        let mut V_from_nuclei = Vec::new();
        for _ in 0..n {
            V_from_nuclei.push(0.);
        }

        Self {
            posits,
            V_from_nuclei,
            grid_n: n,
        }
    }
}

// todo: This was unused, and broke after the 2D change.

// /// We use this to store numerical wave functions for each basis, both at sample points, and
// /// a small amount along each axix, for calculating partial derivatives of psi''.
// /// The `Vec` index corresponds to basis index.
// #[derive(Clone)]
// pub struct _BasesEvaluated {
//     pub on_pt: Vec<Arr3d>,
//     pub psi_pp_analytic: Vec<Arr3d>,
//     pub x_prev: Vec<Arr3d>,
//     pub x_next: Vec<Arr3d>,
//     pub y_prev: Vec<Arr3d>,
//     pub y_next: Vec<Arr3d>,
//     pub z_prev: Vec<Arr3d>,
//     pub z_next: Vec<Arr3d>,
// }
//
// impl _BasesEvaluated {
//     /// Create unweighted basis wave functions. Run this whenever we add or remove basis fns,
//     /// and when changing the grid. This evaluates the analytic basis functions at
//     /// each grid point. Each basis will be normalized in this function.
//     /// Relatively computationally intensive.
//     ///
//     /// Update Nov 2023: This initializer only updates `psi` and `psi_pp_analytic`: Compute
//     /// numerical diffs after.
//     pub fn initialize_with_psi(
//         dev: &ComputationDevice,
//         bases: &[Basis],
//         grid_posits: &Arr3dVec,
//         grid_n: usize,
//         deriv_calc: DerivCalc,
//     ) -> Self {
//         let mut on_pt = Vec::new();
//
//         let mut psi_pp_analytic = Vec::new();
//         let mut x_prev = Vec::new();
//         let mut x_next = Vec::new();
//         let mut y_prev = Vec::new();
//         let mut y_next = Vec::new();
//         let mut z_prev = Vec::new();
//         let mut z_next = Vec::new();
//         let mut derivs = Vec::new();
//
//         let derivs_empty = Derivatives::new(grid_n);
//
//         // let empty = new_data(grid_n);
//         let empty = new_data_2d(grid_n);
//
//         for _ in 0..bases.len() {
//             on_pt.push(empty.clone());
//             psi_pp_analytic.push(empty.clone());
//             derivs.push(derivs_empty.clone());
//             x_prev.push(empty.clone());
//             x_next.push(empty.clone());
//             y_prev.push(empty.clone());
//             y_next.push(empty.clone());
//             z_prev.push(empty.clone());
//             z_next.push(empty.clone());
//         }
//
//         wf_ops::wf_from_bases(
//             dev,
//             &mut on_pt,
//             &mut derivs,
//             bases,
//             grid_posits,
//             deriv_calc,
//         );
//         //
//         // wf_ops::wf_from_bases_spinor(
//         //     dev,
//         //     &mut on_pt,
//         //     Some(&mut derivs),
//         //     bases,
//         //     grid_posits,
//         //     deriv_calc,
//         // );
//
//         Self {
//             on_pt,
//             x_prev,
//             x_next,
//             y_prev,
//             y_next,
//             z_prev,
//             z_next,
//             psi_pp_analytic,
//         }
//     }
//
//     pub fn update_psi_pp_numerics(
//         &mut self,
//         bases: &[Basis],
//         grid_posits: &Arr3dVec,
//         grid_n: usize,
//     ) {
//         // num_diff::update_psi_pp(self, bases, grid_posits, grid_n);
//     }
// }

/// Group that includes psi at a point, and at points surrounding it, an infinetesimal difference
/// in both directions along each spacial axis.
#[derive(Clone)]
pub struct _PsiWDiffs1d {
    pub on_pt: Vec<Cplx>,
    pub psi_pp_analytic: Vec<Cplx>,
    pub x_prev: Vec<Cplx>,
    pub x_next: Vec<Cplx>,
    pub y_prev: Vec<Cplx>,
    pub y_next: Vec<Cplx>,
    pub z_prev: Vec<Cplx>,
    pub z_next: Vec<Cplx>,
}

impl _PsiWDiffs1d {
    pub fn init(data: &Vec<Cplx>) -> Self {
        Self {
            on_pt: data.clone(),
            psi_pp_analytic: data.clone(),
            x_prev: data.clone(),
            x_next: data.clone(),
            y_prev: data.clone(),
            y_next: data.clone(),
            z_prev: data.clone(),
            z_next: data.clone(),
        }
    }
}

/// Group that includes psi at a point, and at points surrounding it, an infinetesimal difference
/// in both directions along each spacial axis.
#[derive(Clone)]
pub struct _PsiWDiffs {
    pub on_pt: Arr3d,
    pub psi_pp_analytic: Arr3d,
    pub x_prev: Arr3d,
    pub x_next: Arr3d,
    pub y_prev: Arr3d,
    pub y_next: Arr3d,
    pub z_prev: Arr3d,
    pub z_next: Arr3d,
}

impl _PsiWDiffs {
    pub fn init(data: &Arr3d) -> Self {
        Self {
            on_pt: data.clone(),
            psi_pp_analytic: data.clone(),
            x_prev: data.clone(),
            x_next: data.clone(),
            y_prev: data.clone(),
            y_next: data.clone(),
            z_prev: data.clone(),
            z_next: data.clone(),
        }
    }
}
