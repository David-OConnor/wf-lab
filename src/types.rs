use crate::{
    basis_wfs::SinExpBasisPt,
    complex_nums::Cplx,
    wf_ops::{self, N, NUDGE_DEFAULT},
};

use lin_alg2::f64::Vec3;

/// Represents important data, in describe 3D arrays.
/// We use Vecs, since these can be large, and we don't want
/// to put them on the stack. Although, they are fixed-size.
///
/// `Vec`s here generally mean per-electron.
/// todo: Change name?
pub struct Surfaces {
    /// Represents points on a grid, for our non-uniform grid.
    pub grid_posits: Arr3dVec,
    /// todo: V here is per electron, *explicitly so an electron doesn't interact with itself*.
    pub V: Vec<Arr3dReal>,
    /// Per electron
    pub psi: Vec<Arr3d>,
    pub psi_pp_calculated: Vec<Arr3d>,
    pub psi_pp_measured: Vec<Arr3d>,
    /// Individual nudge amounts, per point of ψ. Real, since it's scaled by the diff
    /// between psi'' measured and calcualted, which is complex.
    pub nudge_amounts: Vec<Arr3dReal>,
    /// Electric charge at each point in space, from each electron.
    pub elec_charges: Vec<Arr3dReal>,
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

impl Default for Surfaces {
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
        let grid_posits = new_data_vec(N);

        Self {
            grid_posits,
            V: vec![data_real.clone()],
            psi: vec![data.clone(), data.clone()],
            // psi: data.clone(),
            psi_pp_calculated: vec![data.clone()],
            psi_pp_measured: vec![data.clone()],
            aux1: data.clone(),
            aux2: data.clone(),
            nudge_amounts: vec![default_nudges],
            // psi_prev: data.clone(),
            elec_charges: Vec::new(),
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
pub type Arr3dBasis = Vec<Vec<Vec<SinExpBasisPt>>>;

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

pub fn new_data_basis(n: usize) -> Arr3dBasis {
    let mut z = Vec::new();
    z.resize(n, SinExpBasisPt::default());

    let mut y = Vec::new();
    y.resize(n, z);

    let mut x = Vec::new();
    x.resize(n, y);

    x
}

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
