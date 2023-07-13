//! This module contains code for electron-electron interactions, including EM repulsion,
//! and exchange forces.

use std::{collections::HashMap, f64::consts::FRAC_1_SQRT_2};

use crate::{
    basis_wfs::Basis,
    complex_nums::Cplx,
    grid_setup::{new_data, Arr3d, Arr3dReal, Arr3dVec},
    num_diff, util,
    wf_ops::{BasesEvaluated, PsiWDiffs, Q_ELEC},
};

use lin_alg2::f64::Vec3;

/// This struct helps keep syntax more readable
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
    /// todo: Consdier moving to something other than HashMap for performacne reasons
    // psi_joint: Vec<(Vec<PositIndex>, CplxWDiffs)>,
    // todo: Hard-coded for 2-elecs
    // todo: Perhaps a faster way: A vec that is split up based on number of electrons,
    // todo wherin you index it much like your posits,but flattened. (Would need to be 1d) so
    // todo as to avoid hard-coding a specific num of elecs
    pub psi_joint: HashMap<(PositIndex, PositIndex), CplxWDiffs>,
    /// This is the combined probability, created from `psi_joint`.
    pub psi_marginal: PsiWDiffs,
    /// Maybe temp; experimenting;
    individual_elec_wfs: Vec<PsiWDiffs>,
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
            // psi_joint: Vec::new(),
            psi_joint: HashMap::new(),
            psi_marginal: PsiWDiffs::init(&data),
            individual_elec_wfs: Vec::new(),
        }
    }

    /// Create the wave function that, when squared, represents electron charge density.
    /// This is constructed by integrating out the electrons over position space.
    /// todo: Fix this description, and code A/R.
    pub fn populate_psi_marginal(&mut self, grid_n: usize) {
        // todo: This is perhaps what `populate_psi_marginal` should be:
        // todo: Experimenting with using our single-electron approach here, slightly modified
        for i in 0..grid_n {
            for j in 0..grid_n {
                for k in 0..grid_n {
                    let posit_0 = PositIndex::new(i, j, k);

                    self.psi_marginal.on_pt[i][j][k] = Cplx::new_zero();
                    self.psi_marginal.x_prev[i][j][k] = Cplx::new_zero();
                    self.psi_marginal.x_next[i][j][k] = Cplx::new_zero();
                    self.psi_marginal.y_prev[i][j][k] = Cplx::new_zero();
                    self.psi_marginal.y_next[i][j][k] = Cplx::new_zero();
                    self.psi_marginal.z_prev[i][j][k] = Cplx::new_zero();
                    self.psi_marginal.z_next[i][j][k] = Cplx::new_zero();

                    for i1 in 0..grid_n {
                        for j1 in 0..grid_n {
                            for k1 in 0..grid_n {
                                let posit_1 = PositIndex::new(i1, j1, k1);

                                // todo: Hard-coded for 2 elec.
                                let entry = self.psi_joint.get(&(posit_0, posit_1)).unwrap();

                                self.psi_marginal.on_pt[i][j][k] += entry.on_pt;

                                self.psi_marginal.x_prev[i][j][k] += entry.x_prev;
                                self.psi_marginal.x_next[i][j][k] += entry.x_next;
                                self.psi_marginal.y_prev[i][j][k] += entry.y_prev;
                                self.psi_marginal.y_next[i][j][k] += entry.y_next;
                                self.psi_marginal.z_prev[i][j][k] += entry.z_prev;
                                self.psi_marginal.z_next[i][j][k] += entry.z_next;

                                // self.psi_marginal.on_pt[i][j][k] += posit_0.index(&state.surfaces_per_elec[0].psi.on_pt)
                                //     * posit_1.index(&state.surfaces_per_elec[1].psi.on_pt);
                                //
                                // self.psi_marginal.x_prev[i][j][k] += posit_0.index(&state.surfaces_per_elec[0].psi.x_prev)
                                //     * posit_1.index(&state.surfaces_per_elec[1].psi.x_prev);
                                // self.psi_marginal.x_next[i][j][k] += posit_0.index(&state.surfaces_per_elec[0].psi.x_next)
                                //     * posit_1.index(&state.surfaces_per_elec[1].psi.x_next);
                                // self.psi_marginal.y_prev[i][j][k] += posit_0.index(&state.surfaces_per_elec[0].psi.y_prev)
                                //     * posit_1.index(&state.surfaces_per_elec[1].psi.y_prev);
                                // self.psi_marginal.y_next[i][j][k] += posit_0.index(&state.surfaces_per_elec[0].psi.y_next)
                                //     * posit_1.index(&state.surfaces_per_elec[1].psi.y_next);
                                // self.psi_marginal.z_prev[i][j][k] += posit_0.index(&state.surfaces_per_elec[0].psi.z_prev)
                                //     * posit_1.index(&state.surfaces_per_elec[1].psi.z_prev);
                                // self.psi_marginal.z_next[i][j][k] += posit_0.index(&state.surfaces_per_elec[0].psi.z_next)
                                //     * posit_1.index(&state.surfaces_per_elec[1].psi.z_next);
                            }
                        }
                    }
                }
            }
        }
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
    pub fn setup_joint_wf(&mut self, wfs: &[&PsiWDiffs], grid_n: usize) {
        println!("Setting up psi joint...");

        // Experimenting with a different approach.
        for wf in wfs {
            self.individual_elec_wfs.push((*wf).clone());
        }
        // return;

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

        // self.psi_joint = Vec::new();
        self.psi_joint = HashMap::new();

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
            // Trying HM

            // todo: Hard-coded for 2 elecs.
            self.psi_joint.insert(
                (permutation[0], permutation[1]),
                // Trying a HashMap
                CplxWDiffs {
                    on_pt: self
                        .joint_wf_at_permutation(&[&wfs[0].on_pt, &wfs[1].on_pt], &permutation),
                    ..Default::default()
                },
            );

            // self.psi_joint.push((
            //     permutation.clone(),
            //     CplxWDiffs {
            //         // on_pt: self.joint_wf_at_permutation(wfs.on_pt, &permutation),
            //         // x_prev:self.joint_wf_at_permutation(wfs.x_prev, &permutation),
            //         // x_next:self.joint_wf_at_permutation(wfs.x_next, &permutation),
            //         // y_prev:self.joint_wf_at_permutation(wfs.y_prev, &permutation),
            //         // y_next:self.joint_wf_at_permutation(wfs.y_next, &permutation),
            //         // z_prev:self.joint_wf_at_permutation(wfs.z_prev, &permutation),
            //         // z_next:self.joint_wf_at_permutation(wfs.z_next, &permutation),
            //
            //         // todo: Temp approach to mitigate our API. Hard-coded for 2-elecs.
            //         on_pt: self
            //             .joint_wf_at_permutation(&vec![&wfs[0].on_pt, &wfs[1].on_pt], &permutation),
            //             ..Deafult::default()
            //         // todo: Come back to this as required
            //
            //         // x_prev: self.joint_wf_at_permutation(
            //         //     &vec![&wfs[0].x_prev, &wfs[1].x_prev],
            //         //     &permutation,
            //         // ),
            //         // x_next: self.joint_wf_at_permutation(
            //         //     &vec![&wfs[0].x_next, &wfs[1].x_next],
            //         //     &permutation,
            //         // ),
            //         // y_prev: self.joint_wf_at_permutation(
            //         //     &vec![&wfs[0].y_prev, &wfs[1].y_prev],
            //         //     &permutation,
            //         // ),
            //         // y_next: self.joint_wf_at_permutation(
            //         //     &vec![&wfs[0].y_next, &wfs[1].y_next],
            //         //     &permutation,
            //         // ),
            //         // z_prev: self.joint_wf_at_permutation(
            //         //     &vec![&wfs[0].z_prev, &wfs[1].z_prev],
            //         //     &permutation,
            //         // ),
            //         // z_next: self.joint_wf_at_permutation(
            //         //     &vec![&wfs[0].z_next, &wfs[1].z_next],
            //         //     &permutation,
            //         // ),
            //     },
            // ));
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

        return r[0].index(x[0]) * r[1].index(x[1]);

        // hardcoded 2x2 to test

        // println!("a {}", r[0].index(x[0]));
        // println!("b {}", r[1].index(x[1]));
        // println!("c {}", r[1].index(x[0]));
        // println!("d {}", r[0].index(x[1]));
        // todo: Incorporate spin.
        return Cplx::from_real(FRAC_1_SQRT_2)
            * (r[0].index(x[0]) * r[1].index(x[1]) - r[1].index(x[0]) * r[0].index(x[1]));

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

    /// Calculate psi_pp, numerically
    /// todo: Hard-coded for 2 elecs
    /// todo: DRY from num_diff.
    // pub fn calc_psi_pp(&self, p0: &PositIndex, p1: &PositIndex, posit_wrt: usize) -> Cplx {
    // pub fn calc_psi_pp(&self, p0: Vec3, p1: Vec3, bases: &[&[Basis]], bases_unweighted: &BasisWfsUnweighted,
    //     //                    grid_n: usize, posit_wrt: usize) -> Cplx {

    pub fn calc_psi_pp(
        p0: &PositIndex,
        p1: &PositIndex,
        psi0: &PsiWDiffs,
        psi1: &PsiWDiffs,
        posit_wrt: usize,
    ) -> Cplx {
        // Naive Hartree product
        let on_pt = p0.index(&psi0.on_pt) * p1.index(&psi1.on_pt);

        return match posit_wrt {
            0 => num_diff::find_ψ_pp_meas(
                on_pt,
                p0.index(&psi0.x_prev) * p1.index(&psi1.on_pt),
                p0.index(&psi0.x_next) * p1.index(&psi1.on_pt),
                p0.index(&psi0.y_prev) * p1.index(&psi1.on_pt),
                p0.index(&psi0.y_next) * p1.index(&psi1.on_pt),
                p0.index(&psi0.z_prev) * p1.index(&psi1.on_pt),
                p0.index(&psi0.z_next) * p1.index(&psi1.on_pt),
            ),
            1 => num_diff::find_ψ_pp_meas(
                on_pt,
                p0.index(&psi0.on_pt) * p1.index(&psi1.x_prev),
                p0.index(&psi0.on_pt) * p1.index(&psi1.x_next),
                p0.index(&psi0.on_pt) * p1.index(&psi1.y_prev),
                p0.index(&psi0.on_pt) * p1.index(&psi1.y_next),
                p0.index(&psi0.on_pt) * p1.index(&psi1.z_prev),
                p0.index(&psi0.on_pt) * p1.index(&psi1.z_next),
            ),
            _ => unimplemented!(),
        };
    }

    pub fn calc_charge_density(&self, posit: Vec3) -> f64 {
        0.
    }
}

/// Convert an array of ψ to one of electron charge, through space. This is used to calculate potential
/// from an electron. (And potential energy between electrons) Modifies in place
/// to avoid unecessary allocations.
/// `psi` must be normalized.
pub(crate) fn update_charge_density_fm_psi(
    charge_density: &mut Arr3dReal,
    psi_on_charge_grid: &Arr3d,
    grid_n_charge: usize,
) {
    // Note: We need to sum to 1 over *all space*, not just in the grid.
    // We can mitigate this by using a sufficiently large grid bounds, since the WF
    // goes to 0 at distance.

    // todo: YOu may need to model in terms of areas vice points; this is likely
    // todo a factor on irregular grids.

    for i in 0..grid_n_charge {
        for j in 0..grid_n_charge {
            for k in 0..grid_n_charge {
                charge_density[i][j][k] = psi_on_charge_grid[i][j][k].abs_sq() * Q_ELEC;
            }
        }
    }
}
