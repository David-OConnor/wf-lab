use crate::{
    complex_nums::Cplx,
    elec_elec::WaveFunctionMultiElec,
    grid_setup::{self, new_data, new_data_real, new_data_vec, Arr3d, Arr3dReal, Arr3dVec},
    wf_ops::{self, PsiWDiffs, NUDGE_DEFAULT},
};

use crate::wf_ops::PsiWDiffs1d;
use lin_alg2::f64::Vec3;

pub struct SurfacesShared {
    /// Represents points on a grid, for our non-uniform grid.
    pub grid_posits: Arr3dVec,
    pub grid_posits_charge: Arr3dVec,
    /// Potential from nuclei, and all electrons
    pub V_total: Arr3dReal,
    /// Potential from nuclei only. We use this as a baseline for individual electron
    /// potentials, prior to summing over V from other electrons.
    pub V_from_nuclei: Arr3dReal,
    // todo: This may not be a good model: the wave function isn't a function of position
    // mapped to a value for multi-elecs. It's a function of a position for each elec.
    /// `psi` etc here are combined from all individual electron wave functions.
    // pub psi: Arr3d,
    pub psi: WaveFunctionMultiElec,
    pub psi_pp_calculated: Arr3d,
    pub psi_pp_measured: Arr3d,
    pub E: f64,
    /// 2023-08-17: Another attempt at a 3d-grid-based save function
    pub psi_numeric: Arr3d,
    /// In case we want to explore something like DFT
    pub charge_density_dft: Arr3dReal,
    // pub psi_pp_score: f64,
}

impl SurfacesShared {
    pub fn new(
        grid_range: (f64, f64),
        grid_range_charge: (f64, f64),
        spacing_factor: f64,
        n_grid: usize,
        n_grid_charge: usize,
        num_elecs: usize,
    ) -> Self {
        let data = new_data(n_grid);
        let data_real = new_data_real(n_grid);

        let mut grid_posits = new_data_vec(n_grid);
        grid_setup::update_grid_posits(&mut grid_posits, grid_range, spacing_factor, n_grid);

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
            V_from_nuclei: data_real.clone(),
            // psi: data.clone(),
            psi: WaveFunctionMultiElec::new(num_elecs, n_grid),
            // psi_pp_calculated: data.clone(),
            psi_pp_measured: data.clone(),
            psi_pp_calculated: data.clone(),
            E: -0.50,
            psi_numeric: data,
            charge_density_dft: data_real,
            // psi_pp_score: 1.,
        }
    }
    //
    // /// Update `psi` etc from that of individual electrons
    // /// Relevant for combining from multiplie elecs
    // pub fn combine_psi_parts(&mut self, per_elec: &[SurfacesPerElec], E: &[f64], grid_n: usize) {
    //     // Todo: V. Before you update psi_pp_calc.
    //
    //     for i in 0..grid_n {
    //         for j in 0..grid_n {
    //             for k in 0..grid_n {
    //                 self.psi[i][j][k] = Cplx::new_zero();
    //
    //                 for (part_i, part) in per_elec.iter().enumerate() {
    //                     self.psi[i][j][k] += part.psi.on_pt[i][j][k];
    //
    //                     // todo: Come back to this once you figure out how to handle V here.
    //                     // todo: Maybe you don't have psi_pp_calc here.
    //                     // self.psi_pp_calculated[i][j][k] =
    //                     //     eigen_fns::find_ψ_pp_calc(&self.psi, &self.V, E[part_i], i, j, k)
    //                 }
    //             }
    //         }
    //     }
    //
    //     num_diff::find_ψ_pp_meas_fm_grid_irreg(
    //         &self.psi,
    //         &mut self.psi_pp_measured,
    //         &self.grid_posits,
    //         grid_n,
    //     );
    //
    //     // todo: What to do with score?
    //     // let score = wf_ops::score_wf(self); // todo
    //
    //     // todo: You probably need a score_total and e_total in state.
    // }
}

/// Represents important data, in describe 3D arrays.
/// We use Vecs, since these can be large, and we don't want
/// to put them on the stack. Although, they are fixed-size.
///
/// `Vec`s here generally mean per-electron.
#[derive(Clone)]
pub struct SurfacesPerElec {
    /// V from this electron's charge only.
    // pub V_from_this: Arr3dReal,
    /// V from the nucleii, and all other electrons. Does not include this electron's charge.
    /// We use this as a cache instead of generating it on the fly.
    pub V_acting_on_this: Arr3dReal,
    // pub psi: Arr3d,
    pub psi: PsiWDiffs,
    pub psi_pp_calculated: Arr3d,
    pub psi_pp_measured: Arr3d,
    /// Individual nudge amounts, per point of ψ. Real, since it's scaled by the diff
    /// between psi'' measured and calcualted, which is complex.
    pub nudge_amounts: Arr3dReal,
    // pub elec_charge: Arr3dReal,
    // pub E: f64,
    // pub psi_pp_score: f64,
    /// Aux surfaces are for misc visualizations
    pub aux1: Arr3dReal,
    pub aux2: Arr3dReal,
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

impl SurfacesPerElec {
    /// Fills with 0.s
    pub fn new(n_grid: usize) -> Self {
        let data = new_data(n_grid);
        let data_real = new_data_real(n_grid);

        let mut default_nudges = data_real.clone();
        for i in 0..n_grid {
            for j in 0..n_grid {
                for k in 0..n_grid {
                    default_nudges[i][j][k] = NUDGE_DEFAULT;
                }
            }
        }

        // Set up a regular grid using this; this will allow us to convert to an irregular grid
        // later, once we've verified this works.

        let psi = PsiWDiffs::init(&data);

        Self {
            // V_from_this: data_real.clone(),
            V_acting_on_this: data_real.clone(),
            psi,
            psi_pp_calculated: data.clone(),
            psi_pp_measured: data.clone(),
            // E: -0.50,
            // psi_pp_score: 1.,
            nudge_amounts: default_nudges,
            aux1: data_real.clone(),
            aux2: data_real,
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

/// Stores information used to evaluate a wave function at specific points.
#[derive(Clone)]
pub struct EvalDataPerElec {
    /// Posits are the 3D-space positions the other values are sampled at.
    /// V acting on
    pub V_acting_on_this: Vec<f64>,
    // pub V_from_this: Vec<f64>,
    // pub psi: Vec<Cplx>,
    pub psi: PsiWDiffs1d,
    pub psi_pp_calc: Vec<Cplx>,
    pub psi_pp_meas: Vec<Cplx>,
    pub E: f64, // todo: Is this a good place?
    pub score: f64,
}

impl EvalDataPerElec {
    pub fn new(n: usize) -> Self {
        let mut V_acting_on_this = Vec::new();
        // let mut V_from_this = Vec::new();
        let mut psi_pp_calc = Vec::new();
        let mut psi_pp_meas = Vec::new();

        for _ in 0..n {
            V_acting_on_this.push(0.);
            // V_from_this.push(0.);
            psi_pp_calc.push(Cplx::new_zero());
            psi_pp_meas.push(Cplx::new_zero());
        }

        Self {
            V_acting_on_this,
            // V_from_this,
            psi: PsiWDiffs1d::init(&psi_pp_calc), // 0 Vec.
            psi_pp_calc,
            psi_pp_meas,
            E: 0.,
            score: 0.,
        }
    }
}
