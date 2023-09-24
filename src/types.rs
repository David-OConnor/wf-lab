use crate::{
    complex_nums::Cplx,
    elec_elec::WaveFunctionMultiElec,
    grid_setup::{self, new_data, new_data_real, new_data_vec, Arr3d, Arr3dReal, Arr3dVec},
    util::normalize_wf,
    wf_ops::NUDGE_DEFAULT,
};

use crate::basis_wfs::Basis;
use crate::num_diff::H;
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
    pub psi: WaveFunctionMultiElec,
    pub psi_pp_calculated: Arr3d,
    pub psi_pp_measured: Arr3d,
    pub E: f64,
    /// 2023-08-17: Another attempt at a 3d-grid-based save function
    pub psi_numeric: Arr3d,
    /// In case we want to explore something like DFT
    pub charge_density_dft: Arr3dReal,
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
    /// V from the nucleii, and all other electrons. Does not include this electron's charge.
    /// We use this as a cache instead of generating it on the fly.
    pub V_acting_on_this: Arr3dReal,
    pub psi: PsiWDiffs,
    pub psi_pp_calculated: Arr3d,
    pub psi_pp_measured: Arr3d,
    /// Individual nudge amounts, per point of ψ. Real, since it's scaled by the diff
    /// between psi'' measured and calcualted, which is complex.
    pub nudge_amounts: Arr3dReal,
    /// Aux surfaces are for misc visualizations
    pub aux1: Arr3dReal,
    pub aux2: Arr3dReal,
    pub aux3: Arr3dReal,
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
            nudge_amounts: default_nudges,
            aux1: data_real.clone(),
            aux2: data_real.clone(),
            aux3: data_real,
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

/// We use this to store numerical wave functions for each basis, both at sample points, and
/// a small amount along each axix, for calculating partial derivatives of psi''.
/// The `Vec` index corresponds to basis index.
#[derive(Clone)]
pub struct BasesEvaluated {
    pub on_pt: Vec<Arr3d>,
    pub x_prev: Vec<Arr3d>,
    pub x_next: Vec<Arr3d>,
    pub y_prev: Vec<Arr3d>,
    pub y_next: Vec<Arr3d>,
    pub z_prev: Vec<Arr3d>,
    pub z_next: Vec<Arr3d>,
    pub psi_pp_analytic: Vec<Arr3d>, // Sneaking this approach in!
}

impl BasesEvaluated {
    /// Create unweighted basis wave functions. Run this whenever we add or remove basis fns,
    /// and when changing the grid. This evaluates the analytic basis functions at
    /// each grid point. Each basis will be normalized in this function.
    /// Relatively computationally intensive.
    pub fn new(bases: &[Basis], grid_posits: &Arr3dVec, grid_n: usize) -> Self {
        let mut on_pt = Vec::new();
        let mut psi_pp_analytic = Vec::new();
        let mut x_prev = Vec::new();
        let mut x_next = Vec::new();
        let mut y_prev = Vec::new();
        let mut y_next = Vec::new();
        let mut z_prev = Vec::new();
        let mut z_next = Vec::new();

        for _ in 0..bases.len() {
            on_pt.push(new_data(grid_n));
            psi_pp_analytic.push(new_data(grid_n));
            x_prev.push(new_data(grid_n));
            x_next.push(new_data(grid_n));
            y_prev.push(new_data(grid_n));
            y_next.push(new_data(grid_n));
            z_prev.push(new_data(grid_n));
            z_next.push(new_data(grid_n));
        }

        for (basis_i, basis) in bases.iter().enumerate() {
            let mut norm = 0.;

            for i in 0..grid_n {
                for j in 0..grid_n {
                    for k in 0..grid_n {
                        let posit_sample = grid_posits[i][j][k];

                        let posit_x_prev =
                            Vec3::new(posit_sample.x - H, posit_sample.y, posit_sample.z);
                        let posit_x_next =
                            Vec3::new(posit_sample.x + H, posit_sample.y, posit_sample.z);
                        let posit_y_prev =
                            Vec3::new(posit_sample.x, posit_sample.y - H, posit_sample.z);
                        let posit_y_next =
                            Vec3::new(posit_sample.x, posit_sample.y + H, posit_sample.z);
                        let posit_z_prev =
                            Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - H);
                        let posit_z_next =
                            Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + H);

                        let val_pt = basis.value(posit_sample);

                        let val_x_prev = basis.value(posit_x_prev);
                        let val_x_next = basis.value(posit_x_next);
                        let val_y_prev = basis.value(posit_y_prev);
                        let val_y_next = basis.value(posit_y_next);
                        let val_z_prev = basis.value(posit_z_prev);
                        let val_z_next = basis.value(posit_z_next);

                        on_pt[basis_i][i][j][k] = val_pt;
                        x_prev[basis_i][i][j][k] = val_x_prev;
                        x_next[basis_i][i][j][k] = val_x_next;
                        y_prev[basis_i][i][j][k] = val_y_prev;
                        y_next[basis_i][i][j][k] = val_y_next;
                        z_prev[basis_i][i][j][k] = val_z_prev;
                        z_next[basis_i][i][j][k] = val_z_next;

                        norm += val_pt.abs_sq();

                        psi_pp_analytic[basis_i][i][j][k] = basis.second_deriv(posit_sample);
                    }
                }
            }
            //
            // let mut xi = 0.;
            // if let Basis::Sto(sto) = basis {
            //     xi = sto.xi;
            // }
            // todo: Use this line, with high grid n, for finding the norm for analytic basis wfs.
            // println!("Norm for xi={}: {norm_pt}", xi);

            // normalize_wf(&mut on_pt[basis_i], norm);
            // normalize_wf(&mut psi_pp_analytic[basis_i], norm);

            // note: Using individual norm consts for the prevs and next appears to produce incorrect results.
            // normalize_wf(&mut x_prev[basis_i], norm);
            // normalize_wf(&mut x_next[basis_i], norm);
            // normalize_wf(&mut y_prev[basis_i], norm);
            // normalize_wf(&mut y_next[basis_i], norm);
            // normalize_wf(&mut z_prev[basis_i], norm);
            // normalize_wf(&mut z_next[basis_i], norm);
        }

        Self {
            on_pt,
            x_prev,
            x_next,
            y_prev,
            y_next,
            z_prev,
            z_next,
            psi_pp_analytic,
        }
    }
}

/// Group that includes psi at a point, and at points surrounding it, an infinetesimal difference
/// in both directions along each spacial axis.
#[derive(Clone)]
pub struct PsiWDiffs1d {
    pub on_pt: Vec<Cplx>,
    pub psi_pp_analytic: Vec<Cplx>,
    pub x_prev: Vec<Cplx>,
    pub x_next: Vec<Cplx>,
    pub y_prev: Vec<Cplx>,
    pub y_next: Vec<Cplx>,
    pub z_prev: Vec<Cplx>,
    pub z_next: Vec<Cplx>,
}

impl PsiWDiffs1d {
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
pub struct PsiWDiffs {
    pub on_pt: Arr3d,
    pub psi_pp_analytic: Arr3d,
    pub x_prev: Arr3d,
    pub x_next: Arr3d,
    pub y_prev: Arr3d,
    pub y_next: Arr3d,
    pub z_prev: Arr3d,
    pub z_next: Arr3d,
}

impl PsiWDiffs {
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
