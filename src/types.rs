use crate::{
    complex_nums::Cplx,
    num_diff,
    wf_ops::{self, N, NUDGE_DEFAULT},
};

use lin_alg2::f64::Vec3;

pub struct SurfacesShared {
    /// Represents points on a grid, for our non-uniform grid.
    pub grid_posits: Arr3dVec,
    /// Potential from nuclei, and all electrons
    pub V_combined: Arr3dReal,
    /// Potential from nuclei only. We use this as a baseline for individual electron
    /// potentials, prior to summing over V from other electrons.
    pub V_fixed_charges: Arr3dReal,
    /// `psi` etc here are combined from all individual electron wave functions.
    pub psi: Arr3d,
    // pub psi_pp_calculated: Arr3d, // todo??
    // todo: Do you want this?
    pub psi_pp_measured: Arr3d,
}

impl SurfacesShared {
    pub fn new(grid_min: f64, grid_max: f64, spacing_factor: f64) -> Self {
        let data = new_data(N);
        let data_real = new_data_real(N);

        let mut grid_posits = new_data_vec(N);
        wf_ops::update_grid_posits(&mut grid_posits, grid_min, grid_max, spacing_factor);

        Self {
            grid_posits,
            V_combined: data_real.clone(),
            V_fixed_charges: data_real,
            psi: data.clone(),
            // psi_pp_calculated: data.clone(),
            psi_pp_measured: data,
        }
    }

    /// Update `psi` etc from that of individual electrons
    pub fn combine_psi_parts(&mut self, per_elec: &[SurfacesPerElec], E: &[f64]) {
        // Todo: V. Before you update psi_pp_calc.

        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    self.psi[i][j][k] = Cplx::new_zero();

                    for (part_i, part) in per_elec.iter().enumerate() {
                        self.psi[i][j][k] += part.psi[i][j][k];

                        // todo: Come back to this once you figure out how to handle V here.
                        // todo: Maybe you don't have psi_pp_calc here.
                        // self.psi_pp_calculated[i][j][k] =
                        //     eigen_fns::find_ψ_pp_calc(&self.psi, &self.V, E[part_i], i, j, k)
                    }
                }
            }
        }

        num_diff::find_ψ_pp_meas_fm_grid_irreg(
            &self.psi,
            &mut self.psi_pp_measured,
            &self.grid_posits,
        );

        // todo: What to do with score?
        // let score = wf_ops::score_wf(self); // todo

        // todo: You probably need a score_total and e_total in state.
    }
}

/// Represents important data, in describe 3D arrays.
/// We use Vecs, since these can be large, and we don't want
/// to put them on the stack. Although, they are fixed-size.
///
/// `Vec`s here generally mean per-electron.
#[derive(Clone)]
pub struct SurfacesPerElec {
    /// V from the nucleus, and all *other* electrons
    pub V: Arr3dReal,
    pub psi: Arr3d,
    pub psi_pp_calculated: Arr3d,
    pub psi_pp_measured: Arr3d,
    /// Individual nudge amounts, per point of ψ. Real, since it's scaled by the diff
    /// between psi'' measured and calcualted, which is complex.
    pub nudge_amounts: Arr3dReal,

    /// Aux surfaces are for misc visualizations
    pub aux1: Arr3d,
    pub aux2: Arr3d,
    //
    // Below this, are mostly unused/experimental terms.
    //
    // todo: You quantize with n already associated with H and energy. Perhaps the next step
    // todo is to quantize with L and associated angular momentum, as a second check on your
    // todo WF, and a tool to identify its validity.
    // /// These momentum terms are currently unused.
    // pub psi_p_calculated: Arr3d,
    // pub psi_p_total_measured: Arr3d,
    // pub psi_px_measured: Arr3d,
    // pub psi_py_measured: Arr3d,
    // pub psi_pz_measured: Arr3d,
    // /// Todo: Plot both real and imaginary momentum components? (or mag + phase?)
    // pub momentum: Arr3d,
    // /// todo: Experimental representation as a local analytic eq at each point.
    // pub bases: Vec<Arr3dBasis>,

    //todo: Evaluating per-electron wave fns.
    // pub psis_per_elec: Vec<Arr3d>,
}

impl Default for SurfacesPerElec {
    /// Fills with 0.s
    fn default() -> Self {
        let data = new_data(N);
        let data_real = new_data_real(N);

        let mut default_nudges = data_real.clone();
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    default_nudges[i][j][k] = NUDGE_DEFAULT;
                }
            }
        }

        // Set up a regular grid using this; this will allow us to convert to an irregular grid
        // later, once we've verified this works.

        Self {
            V: data_real.clone(),
            psi: data.clone(),
            psi_pp_calculated: data.clone(),
            psi_pp_measured: data.clone(),
            nudge_amounts: default_nudges,

            aux1: data.clone(),
            aux2: data,
            // psi_prev: data.clone(),
            // bases: new_data_basis(N),
            // momentum: data.clone(),
            // psi_p_calculated: data.clone()],
            // psi_p_total_measured: data.clone(),
            // psi_px_measured: data.clone(),
            // psi_py_measured: data.clone(),
            // psi_pz_measured: data.clone(),
        }
    }
}

// type Arr3d = Vec<Vec<Vec<f64>>>;
pub type Arr3dReal = Vec<Vec<Vec<f64>>>;

pub type Arr3d = Vec<Vec<Vec<Cplx>>>;
// pub type Arr3dBasis = Vec<Vec<Vec<SinExpBasisPt>>>;

pub type Arr3dVec = Vec<Vec<Vec<Vec3>>>;

/// Make a new 3D grid, as a nested Vec
pub fn new_data(n: usize) -> Arr3d {
    let mut z = Vec::new();
    z.resize(n, Cplx::new_zero());
    // z.resize(N, 0.);

    let mut y = Vec::new();
    y.resize(n, z);

    let mut x = Vec::new();
    x.resize(n, y);

    x
}

/// Make a new 3D grid, as a nested Vec
pub fn new_data_real(n: usize) -> Arr3dReal {
    let mut z = Vec::new();
    z.resize(n, 0.);

    let mut y = Vec::new();
    y.resize(n, z);

    let mut x = Vec::new();
    x.resize(n, y);

    x
}

// pub fn new_data_basis(n: usize) -> Arr3dBasis {
//     let mut z = Vec::new();
//     z.resize(n, SinExpBasisPt::default());
//
//     let mut y = Vec::new();
//     y.resize(n, z);
//
//     let mut x = Vec::new();
//     x.resize(n, y);
//
//     x
// }

/// Make a new 3D grid of position vectors, as a nested Vec
pub fn new_data_vec(n: usize) -> Arr3dVec {
    let mut z = Vec::new();
    z.resize(n, Vec3::new_zero());

    let mut y = Vec::new();
    y.resize(n, z);

    let mut x = Vec::new();
    x.resize(n, y);

    x
}

pub fn copy_array_real(dest: &mut Arr3dReal, source: &Arr3dReal, grid_n: usize) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                dest[i][j][k] = source[i][j][k];
            }
        }
    }
}