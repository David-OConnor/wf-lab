//! This module contains code for electron-electron interactions, including EM repulsion,
//! and exchange forces.

use std::f64::consts::FRAC_1_SQRT_2;

use crate::{
    complex_nums::Cplx,
    types::new_data,
    types::{Arr3d, Arr3dReal, Arr3dVec},
    util, wf_ops,
    wf_ops::{PsiWDiffs, Q_ELEC},
};

use lin_alg2::f64::Vec3;

/// This struct helps keep syntax more readable
#[derive(Clone, Copy, PartialEq)]
pub struct PositIndex {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl PositIndex {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }

    pub fn index(&self, wf: &Arr3d) -> Cplx {
        wf[self.x][self.y][self.z]
    }
}

/// Variant of `PsiWDiffs`, to fit into our multi-elec WF data type.
#[derive(Clone, Default)]
pub struct CplxWDiffs {
    pub on_pt: Cplx,
    pub x_prev: Cplx,
    pub x_next: Cplx,
    pub y_prev: Cplx,
    pub y_next: Cplx,
    pub z_prev: Cplx,
    pub z_next: Cplx,
}

/// Represents a wave function of multiple electrons
/// ψ(r1, r2) = ψ_a(r1)ψb(r2), wherein we are combining probabilities.
/// fermions: two identical fermions cannot occupy the same state.
/// ψ(r1, r2) = A[ψ_a(r1)ψ_b(r2) - ψ_b(r1)ψ_a(r2)]
/// todo: Incorporate spin.
pub struct WaveFunctionMultiElec {
    num_elecs: usize,
    /// A column-major slater determinant.
    // components_slater: Vec<Arr3d>

    /// Maps permutations of position index to wave function value.
    /// The outer vec has a value for each permutation of spacial coordinates, with length
    /// `num_elecs`. So, its length is (n_grid^3)^num_elecs
    psi_joint: Vec<(Vec<PositIndex>, CplxWDiffs)>,
    /// This is the combined probability, created from `psi_joint`.
    pub psi_marginal: PsiWDiffs,
}

/// Be careful: (Wiki): "Only a small subset of all possible fermionic wave functions can be written
/// as a single Slater determinant, but those form an important and useful subset because of their simplicity."
/// What is the more general case?
/// todo: YOu may need to figure out how to deal with chareg density on an irregular grid.
impl WaveFunctionMultiElec {
    pub fn new(num_elecs: usize, n_grid: usize) -> Self {
        let data = new_data(n_grid);
        Self {
            num_elecs,
            psi_joint: Vec::new(),
            psi_marginal: PsiWDiffs::init(&data),
        }
    }

    /// Create the wave function that, when squared, represents electron charge density.
    /// This is constructed by integrating out the electrons over position space.
    /// todo: Fix this description, and code A/R.
    pub fn populate_psi_marginal(&mut self, grid_n: usize) {
        println!("Populating psi marginal...");
        // for i in 0..grid_n {
        //     for j in 0..grid_n {
        //         for k in 0..grid_n {
        //             self.psi_marginal.on_pt = 1.;
        //         }
        //     }
        // }

        // todo: Maybe there's no wf, but there is a wf squared?

        // Todo: What is this? Likely a function of n_electrons and n_points. Maybe multiply them,
        // todo then subtract one?
        let n_components_per_posit = (grid_n.pow(3) * self.num_elecs) as f64;

        // Start clean
        let data = new_data(grid_n);
        self.psi_marginal = PsiWDiffs::init(&data);

        for (posits, wf_val) in &self.psi_joint {
            for posit in posits {
                // posit.index_mut(&mut self.psi_marginal.on_pt) += *wf_val;
                // self.psi_marginal.on_pt[posit.x][posit.y][posit.z] += *wf_val;
                // self.psi_marginal.on_pt[posit.x][posit.y][posit.z] += *wf_val / n_components_per_posit;

                // self.psi_marginal.on_pt[posit.x][posit.y][posit.z] += *wf_val;

                // todo: Experimenting with treating the abs_square as what we combine, vice making a "marginal"
                // todo combined WF we square to find charge density
                // Note: We are hijacking psi_marginal to be this squared value. (Noting that
                // we are still storing it as a complex value)

                // todo: This produces a visible result (as opposed to when we don't square it...), but
                // todo: how do we score this using psi''s?
                // todo: (The impliciation here is perhaps the various values are variously positive and negative,
                // todo, so they cancel if not squared)
                self.psi_marginal.on_pt[posit.x][posit.y][posit.z] +=
                    Cplx::from_real(wf_val.on_pt.abs_sq());

                self.psi_marginal.x_prev[posit.x][posit.y][posit.z] +=
                    Cplx::from_real(wf_val.x_prev.abs_sq());
                self.psi_marginal.x_next[posit.x][posit.y][posit.z] +=
                    Cplx::from_real(wf_val.x_next.abs_sq());
                self.psi_marginal.y_prev[posit.x][posit.y][posit.z] +=
                    Cplx::from_real(wf_val.y_prev.abs_sq());
                self.psi_marginal.y_next[posit.x][posit.y][posit.z] +=
                    Cplx::from_real(wf_val.y_next.abs_sq());
                self.psi_marginal.z_prev[posit.x][posit.y][posit.z] +=
                    Cplx::from_real(wf_val.z_prev.abs_sq());
                self.psi_marginal.z_next[posit.x][posit.y][posit.z] +=
                    Cplx::from_real(wf_val.z_next.abs_sq());
            }
        }

        // Normalize the wave function.
        let mut norm = 0.;

        for i in 0..grid_n {
            for j in 0..grid_n {
                for k in 0..grid_n {
                    norm += self.psi_marginal.on_pt[i][j][k].abs_sq();
                }
            }
        }

        util::normalize_wf(&mut self.psi_marginal.on_pt, norm, grid_n);

        util::normalize_wf(&mut self.psi_marginal.x_prev, norm, grid_n);
        util::normalize_wf(&mut self.psi_marginal.x_next, norm, grid_n);
        util::normalize_wf(&mut self.psi_marginal.y_prev, norm, grid_n);
        util::normalize_wf(&mut self.psi_marginal.y_next, norm, grid_n);
        util::normalize_wf(&mut self.psi_marginal.z_prev, norm, grid_n);
        util::normalize_wf(&mut self.psi_marginal.z_next, norm, grid_n);

        println!("Some vals: {:?}", self.psi_marginal.on_pt[5][6][4]);
        println!("Some vals: {:?}", self.psi_marginal.on_pt[1][3][6]);
        // println!("Some vals: {:?}", self.psi_marginal.on_pt[7][10][2]);
        println!("Some vals: {:?}", self.psi_marginal.on_pt[6][4][3]);

        // todo: set up diffs. This is a relatively coarse numerical diff vice the analytic ones we use
        // todo for the individual electrons.
        for i in 0..grid_n {}
        // self.psi_marginal.x_prev =

        // todo: Set up prevs and nexts.

        //             psi: &PsiWDiffs,
        // V: &Arr3dReal,
        // psi_pp_calc: &mut Arr3d,
        // psi_pp_meas: &mut Arr3d,
        // E: f64,
        // grid_n: usize,

        println!("Complete");
    }
    // let's say n = 2, and r spans 0, 1, 2, 3
    // we want to calc electron density at 2, or collect all relevant parts
    // val([x0, x1], [r0, r0, r0]),
    // val ([x0, x1], [r0, r0, r1]),
    // val([x0, x1], [r0, r0, r2]),
    // val([x0, x1], [r0, r0, 3]),
    // val([x0, x1], [r0, r1, 0]),

    // let's say, n=3. x = [x0, x1, x2]
    // posits = [r0, r1, r2] where we

    /// Set up values for the joint wavefunction; related to probabilities of the electrons
    /// being in a permutation of positions.
    pub fn setup_joint_wf(&mut self, wfs: &[PsiWDiffs], grid_n: usize) {
        println!("Setting up psi joint...");
        // todo: If you run this function more often than generating the posits
        // todo store them somewhere;
        let mut posits = Vec::new();

        for i in 0..grid_n {
            for j in 0..grid_n {
                for k in 0..grid_n {
                    posits.push(PositIndex::new(i, j, k));
                }
            }
        }

        self.psi_joint = Vec::new();

        let mut posit_permutations = Vec::new();
        // for 2 elecs:
        // [p0, p0], [p0, p1], [p0, p2].. etc, [p1, p0]... etc
        // todo: Hard-coded for 2 electrons. Find or use a general algorithm.

        // for r_i in 0..self.num_elecs {
        //     let mut permutation = Vec::new();
        //     for (i, r) in posits.iter().enumerate() {
        //         permutation[i] = r;
        //     }
        //     posit_permutations.push(permutation);
        // }

        // todo: Combinations instead of permutations? Would that solve your WF-is-cancelled problem
        // todo without squaring?

        // let mut combo_indexes_added = Vec::new();

        // todo: Reason-out adn write out to determine if you want permutations or combinations. Ie,
        // are permutatinos with order reversed the same? If the WF is the same? If different?

        // for (i0, r0) in posits.iter().enumerate() {
        //     for (i1, r1) in posits.iter().enumerate() {
        for r0 in &posits {
            for r1 in &posits {
                // if combo_indexes_added.contains(&vec![i1, i0]) {
                //     continue;
                // }
                // combo_indexes_added.push(vec![i0, i1]);

                posit_permutations.push(vec![*r0, *r1]);
            }
        }

        for permutation in posit_permutations {
            // println!("j{}", self.joint_wf_at_permutation(wfs, &permutation));
            self.psi_joint.push((
                permutation.clone(),
                CplxWDiffs {
                    // on_pt: self.joint_wf_at_permutation(wfs.on_pt, &permutation),
                    // x_prev:self.joint_wf_at_permutation(wfs.x_prev, &permutation),
                    // x_next:self.joint_wf_at_permutation(wfs.x_next, &permutation),
                    // y_prev:self.joint_wf_at_permutation(wfs.y_prev, &permutation),
                    // y_next:self.joint_wf_at_permutation(wfs.y_next, &permutation),
                    // z_prev:self.joint_wf_at_permutation(wfs.z_prev, &permutation),
                    // z_next:self.joint_wf_at_permutation(wfs.z_next, &permutation),

                    // todo: Temp approach to mitigate our API. Hard-coded for 2-elecs.
                    on_pt: self
                        .joint_wf_at_permutation(&vec![&wfs[0].on_pt, &wfs[1].on_pt], &permutation),
                    x_prev: self.joint_wf_at_permutation(
                        &vec![&wfs[0].x_prev, &wfs[1].x_prev],
                        &permutation,
                    ),
                    x_next: self.joint_wf_at_permutation(
                        &vec![&wfs[0].x_prev, &wfs[1].x_next],
                        &permutation,
                    ),
                    y_prev: self.joint_wf_at_permutation(
                        &vec![&wfs[0].y_prev, &wfs[1].y_prev],
                        &permutation,
                    ),
                    y_next: self.joint_wf_at_permutation(
                        &vec![&wfs[0].y_prev, &wfs[1].y_next],
                        &permutation,
                    ),
                    z_prev: self.joint_wf_at_permutation(
                        &vec![&wfs[0].z_prev, &wfs[1].z_prev],
                        &permutation,
                    ),
                    z_next: self.joint_wf_at_permutation(
                        &vec![&wfs[0].z_prev, &wfs[1].z_next],
                        &permutation,
                    ),
                },
            ));
        }

        println!("Complete");
    }

    /// Find the wave fucntion value associated with a single permutation of position. Eg, the one associated
    /// with the probability that electron 0 is in position 0 and electron 1 is in position 1.
    // pub fn joint_wf_at_permutation(&self, x: &[Arr3d], r: &[PositIndex]) -> Cplx {
    pub fn joint_wf_at_permutation(&self, x: &[&Arr3d], r: &[PositIndex]) -> Cplx {
        // todo: Hard-coded for 2-elecs of opposite spin; ie no exchange interaction.

        // so, for posits 0, 1
        // we have permutations
        // 00 01 10 11
        // x0[0] * x1[0] + x0[0] * x1[1] + x0[1] x1[0] + x0[1] * x1[1]

        return r[0].index(&x[0]) * r[1].index(&x[1]);

        // hardcoded 2x2 to test

        // println!("a {}", r[0].index(&x[0]));
        // println!("b {}", r[1].index(&x[1]));
        // println!("c {}", r[1].index(&x[0]));
        // println!("d {}", r[0].index(&x[1]));
        // todo: Incorporate spin.
        return Cplx::from_real(FRAC_1_SQRT_2)
            * (r[0].index(&x[0]) * r[1].index(&x[1]) - r[1].index(&x[0]) * r[0].index(&x[1]));

        // hardcoded 3x3 to test. // todo: QC norm const
        // 1. / 3.0.sqrt() * (
        //     x[0][i0][j0][k0] * x[1][i1][j1][k1] * x[2][i2][j2][k2] +
        //         x[1][i0][j0][k0] * x[2][i1][j1][k1] * x[0][i2][j2][k2] +
        //         x[2][i0][j0][k0] * x[0][i1][j1][k1] * x[1][i2][j2][k2] -
        //         x[0][i0][j0][k0] * x[2][i1][j1][k1] * x[1][i2][j2][k2] -
        //         x[1][i0][j0][k0] * x[0][i1][j1][k1] * x[2][i2][j2][k2] -
        //         x[2][i0][j0][k0] * x[1][i1][j1][k1] * x[0][i2][j2][k2]
        // );

        // for i_x in 0..self.num_elecs {
        //     let mut entry = 1.;
        //     for i_posit in 0..self.num_elecs {
        //         entry *= x[i_x]posits[i_posit]; // todo not quite right. Check the 3x3 example for how it goes.
        //     }
        // }
    }

    pub fn calc_charge_density(&self, posit: Vec3) -> f64 {
        0.
    }
}

/// Convert an array of ψ to one of electron charge, through space. Modifies in place
/// to avoid unecessary allocations.
pub(crate) fn update_charge_density_fm_psi(
    psi: &Arr3d,
    charge_density: &mut Arr3dReal,
    grid_n: usize,
) {
    println!("Creating electron charge for the active e- ...");

    // todo: Problem? Needs to sum to 1 over *all space*, not just in the grid.
    // todo: We can mitigate this by using a sufficiently large grid bounds, since the WF
    // todo goes to 0 at distance.

    // todo: Consequence of your irregular grid: Is this normalization process correct?

    // Normalize <ψ|ψ>
    let mut psi_sq_size = 0.;
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                psi_sq_size += psi[i][j][k].abs_sq();
            }
        }
    }

    let num_elecs = 1;
    // Save computation on this constant factor.
    let c = Q_ELEC * num_elecs as f64 / psi_sq_size;

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                charge_density[i][j][k] = psi[i][j][k].abs_sq() * c;
            }
        }
    }
    println!("Complete");
}

// todo: Currently unused.
/// Update electron charge densities ψ, for every electron.
pub(crate) fn update_charge_densities_fm_psi(
    charges_fm_elecs: &mut [Arr3dReal],
    psi_per_electron: &[Arr3d],
    grid_n: usize,
) {
    for (i, psi) in psi_per_electron.iter().enumerate() {
        update_charge_density_fm_psi(psi, &mut charges_fm_elecs[i], grid_n)
    }
}

/// Calculate the result of exchange interactions between electrons.
pub(crate) fn calc_exchange(psis: &[Arr3d], result: &mut Arr3d, grid_n: usize) {
    // for i_a in 0..N {
    //     for j_a in 0..N {
    //         for k_a in 0..N {
    //             for i_b in 0..N {
    //                 for j_b in 0..N {
    //                     for k_b in 0..N {
    //
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    *result = Arr3d::new();

    for a in 0..grid_n {
        // todo: i, j, k for 3D
        for b in 0..grid_n {
            // This term will always be 0, so skipping  here may save calculation.
            if a == b {
                continue;
            }
            // Enumerate so we don't calculate exchange on a WF with itself.
            for (i_1, psi_1) in psis.iter().enumerate() {
                for (i_2, psi_2) in psis.iter().enumerate() {
                    // Don't calcualte exchange with self
                    if i_1 == i_2 {
                        continue;
                    }

                    // todo: THink this through. What index to update?
                    // result[a] += FRAC_1_SQRT_2 * (psi_1[a] * psi_2[b] - psi_2[a] * psi_1[b]);
                }
            }
        }
    }
}
