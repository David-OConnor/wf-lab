//! This program explores solving the wave equation for
//! arbitrary potentials. It visualizes the wave function in 3d, with user interaction.

// todo: Consider instead of H orbitals, use the full set of Slater basis
// functions, which are more general. Make a fn to generate them.

#![allow(non_snake_case)]

use std::f64::consts::PI;

use lin_alg2::f64::Vec3;

use basis_wfs::BasisFn;

mod basis_wfs;
mod render;
mod ui;

const NUM_SURFACES: usize = 4; // V, psi, psi_pp_calculated, psi_pp_measured

const K_C: f64 = 1.;
const Q_PROT: f64 = 1.;
const Q_ELEC: f64 = -1.;
const M_ELEC: f64 = 1.; // todo: Which?
const ħ: f64 = 1.;

// Compute these statically, to avoid continuous calls during excecution.
const KE_COEFF: f64 = -2. * M_ELEC / (ħ * ħ);
const KE_COEFF_INV: f64 = 1. / KE_COEFF;

// Wave function number of values per edge.
// Memory use and some parts of computation scale with the cube of this.
// Note: Using this as our fine grid. We will potentially subdivide it once
// or twice per axis, hence the multiple of 4 constraint.
const N: usize = 21 * 4;
// Used for calculating numerical psi''.
// Smaller is more precise. Applies to dx, dy, and dz
const H: f64 = 0.00001;
const GRID_MIN: f64 = -4.;
const GRID_MAX: f64 = 4.;

// For finding psi_pp_meas, using only values on the grid
const H_GRID: f64 = (GRID_MAX - GRID_MIN) / (N as f64);
const H_GRID_SQ: f64 = H_GRID * H_GRID;

type Arr3d = Vec<Vec<Vec<f64>>>;
// type WfType = dyn Fn(Vec3, Vec3) -> f64;
// type wf_type = dyn Fn(Vec3, Vec3) -> f64;
// type wf_type = fn(Vec3, Vec3) -> f64;

/// Make a new 3D grid, as a nested Vec
fn new_data() -> Arr3d {
    let mut z = Vec::new();
    z.resize(N, 0.);

    let mut y = Vec::new();
    y.resize(N, z);

    let mut x = Vec::new();
    x.resize(N, y);

    x
}

/// Represents important data, in describe 3D arrays.
/// We use Vecs, since these can be large, and we don't want
/// to put them on the stack. Although, they are fixed-size.
/// todo: Change name?
pub struct Surfaces {
    pub V: Arr3d,
    pub psi: Arr3d,
    pub psi_pp_calculated: Arr3d,
    pub psi_pp_measured: Arr3d,
}

impl Default for Surfaces {
    /// Fills with 0.s
    fn default() -> Self {
        let mut data = new_data();

        Self {
            V: data.clone(),
            psi: data.clone(),
            psi_pp_calculated: data.clone(),
            psi_pp_measured: data,
        }
    }
}

/// Represents a gaussian function.
#[derive(Clone, Copy)]
pub struct Gaussian {
    pub a_x: f64,
    pub b_x: f64,
    pub c_x: f64,
    pub a_y: f64,
    pub b_y: f64,
    pub c_y: f64,
    pub a_z: f64,
    pub b_z: f64,
    pub c_z: f64,
}

impl Gaussian {
    /// Helper fn
    fn val_1d(x: f64, a: f64, b: f64, c: f64) -> f64 {
        let part_1 = (x - b).powi(2) / (2. * c.powi(2));
        a * (-part_1).exp()
    }

    pub fn val(&self, posit: Vec3) -> f64 {
        // todo: QC how this works in 3d
        Self::val_1d(posit.x, self.a_x, self.b_x, self.c_x)
            + Self::val_1d(posit.y, self.a_y, self.b_y, self.c_y)
            + Self::val_1d(posit.z, self.a_z, self.b_z, self.c_z)
    }
}

// #[derive(Default)]
pub struct State {
    /// todo: Combine wfs and nuclei in into single tuple etc to enforce index pairing?
    /// todo: Or a sub struct?
    /// Wave functions, with weights
    // pub wfs: Vec<(impl Fn(Vec3, Vec3) -> f64 + 'static, f64)>,
    pub wfs: Vec<(BasisFn, f64)>,
    // todo use an index for them.
    /// Nuclei. todo: H only for now.
    pub charges: Vec<(Vec3, f64)>,
    /// Computed surfaces, with name.
    pub surfaces: Surfaces,
    // pub surfaces: [&'static Arr3d; NUM_SURFACES],
    // pub surfaces: [Arr3d; NUM_SURFACES],
    /// Eg, least-squares over 2 or 3 dimensions between
    /// When visualizing a 2d wave function over X and Y, this is the fixed Z value.
    pub z_displayed: f64,
    /// Energy of the system
    pub E: f64,
    pub psi_pp_score: f64,
    /// Surface name
    pub surface_names: [String; NUM_SURFACES],
    pub show_surfaces: [bool; NUM_SURFACES],
    /// This defines how our 3D grid is subdivided in different areas.
    /// todo: FIgure this out.
    /// todo: We should possibly remove grid divisions, as we may move
    /// today away from the grid for all but plotting.
    pub grid_divisions: Vec<Vec<Vec<u8>>>,
    /// Experimenting with gaussians; if this works out, it should possibly be
    /// combined with the BasisFn (wfs field).
    pub gaussians: Vec<Gaussian>,
}

/// Score using wavefunction fidelity.
fn score_wf(sfcs: &Surfaces, E: f64) -> f64 {
    // "The accuracy should be scored by the fidelity of the wavefunction compared
    // to the true wavefunction. Fidelity is defined as |<psi_trial | psi_true >|^2.
    // For normalized states, this will always be bounded from above by 1.0. So it's
    // lower than 1.0 for an imperfect variational function, but is 1 if you are
    // able to exactly express it.""

    let mut fidelity = 0.;

    // For normalization.
    let mut norm_sq_trial = 0.;
    let mut norm_sq_meas = 0.;

    // todo: Dkn't allocate each time?
    let mut psi_fm_meas = new_data();

    const EPS: f64 = 0.0001;

    // Create normalization const.
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                norm_sq_trial += sfcs.psi[i][j][k].powi(2);

                // Numerical anomolies that should balance a very low number by
                // a very high one don't work out here; set to 0.(?)
                // todo: QC if this is really solving your problem with spike ring.
                psi_fm_meas[i][j][k] = if (E - sfcs.V[i][j][k]).abs() < EPS {
                    0.
                } else {
                    KE_COEFF_INV / (E - sfcs.V[i][j][k]) * sfcs.psi_pp_measured[i][j][k]
                };

                norm_sq_meas += psi_fm_meas[i][j][k].powi(2);
            }
        }
    }

    let norm_trial = norm_sq_trial.sqrt();
    let norm_meas = norm_sq_meas.sqrt();

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                fidelity +=
                    (psi_fm_meas[i][j][k] / norm_sq_meas) * (sfcs.psi[i][j][k] / norm_trial);
            }
        }
    }

    // sfcs.V = psi_fm_meas; // todo temp!!!

    fidelity.powi(2)
}

/// Single-point Coulomb potential, eg a hydrogen nuclei.
fn V_coulomb(posit_nuc: Vec3, posit_sample: Vec3, charge: f64) -> f64 {
    let diff = posit_sample - posit_nuc;
    let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

    -K_C * charge / r
}

/// Create a set of values in a given range, with a given number of values.
/// Similar to `numpy.linspace`.
/// The result terminates one step before the end of the range.
fn linspace(range: (f64, f64), num_vals: usize) -> Vec<f64> {
    let step = (range.1 - range.0) / num_vals as f64;

    let mut result = Vec::new();

    let mut val = range.0;
    for _ in 0..num_vals {
        result.push(val);
        val += step;
    }

    result
}

/// Calcualte psi'', calculated from psi, and E.
/// At a given i, j, k.
fn find_psi_pp_calc(sfcs: &Surfaces, E: f64, i: usize, j: usize, k: usize) -> f64 {
    (E - sfcs.V[i][j][k]) * KE_COEFF * sfcs.psi[i][j][k]
}

/// Calcualte psi'', measured, using the finite diff method, for a single value.
fn find_psi_pp_meas(
    psi: &Arr3d,
    psi_pp_measured: &Arr3d,
    // posit_sample: Vec3,
    // wfs: &Vec<(usize, f64)>,
    // charges: &Vec<(Vec3, f64)>,
    // E: f64,
    i: usize,
    j: usize,
    k: usize,
) -> f64 {
    // Using purely the numerical psi, we are now limited to the grid, for now.

    // let x_prev = Vec3::new(posit_sample.x - H_GRID, posit_sample.y, posit_sample.z);
    // let x_next = Vec3::new(posit_sample.x + H_GRID, posit_sample.y, posit_sample.z);
    // let y_prev = Vec3::new(posit_sample.x, posit_sample.y - H_GRID, posit_sample.z);
    // let y_next = Vec3::new(posit_sample.x, posit_sample.y + H_GRID, posit_sample.z);
    // let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - H_GRID);
    // let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + H_GRID);

    let mut psi_x_prev = 0.;
    let mut psi_x_next = 0.;
    let mut psi_y_prev = 0.;
    let mut psi_y_next = 0.;
    let mut psi_z_prev = 0.;
    let mut psi_z_next = 0.;

    // If on an edge, contue the WF around the edge's value using
    // a constant slope, to avoid. Alternatively, we could mirror the value.

    // 1, 2, 5 //  8
    // 5, 2, 1 // 8
    let psi_this = psi[i][j][k]; // code shortener

    let psi_x_prev = if i == 0 {
        2. * psi_this - psi[i + 1][j][k]
        // mirror: psi[i + 1][j][k]
    } else {
        psi[i - 1][j][k]
    };
    let psi_x_next = if i == N - 1 {
        2. * psi_this - psi[i - 1][j][k]
    } else {
        psi[i + 1][j][k]
    };
    let psi_y_prev = if j == 0 {
        2. * psi_this - psi[i][j + 1][k]
    } else {
        psi[i][j - 1][k]
    };
    let psi_y_next = if j == N - 1 {
        2. * psi_this - psi[i][j - 1][k]
    } else {
        psi[i][j + 1][k]
    };
    let psi_z_prev = if k == 0 {
        2. * psi_this - psi[i][j][k + 1]
    } else {
        psi[i][j][k - 1]
    };
    let psi_z_next = if k == N - 1 {
        2. * psi_this - psi[i][j][k - 1]
    } else {
        psi[i][j][k + 1]
    };

    let mut result =
        psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next - 6. * psi_this;

    result / H_GRID_SQ

    // let mut result = 0.;
    // result = psi_x_prev + psi_x_next - 2. * sfcs.psi[i][j][k];
    // result = psi_y_prev + psi_y_next - 2. * sfcs.psi[i][j][k];
    // result = psi_z_prev + psi_z_next - 2. * sfcs.psi[i][j][k];
    // result /= H_GRID.powi(2); // todo: Hard-code this in a const etc.

    // result
}

/// Apply a correction to the WF, in attempt to make our two psi''s closer.
/// Uses our numerically-calculated WF. Updates psi, and both psi''s.
fn nudge_wf(sfcs: &mut Surfaces, E: f64) {
    let mut nudge_amount = 0.0001;

    let num_nudges = 100;
    let d_psi = 0.001;

    // todo: Once out of the shower, look for more you can optimize out!

    // todo: Check for infinities etc around the edges

    // todo: Variational method and perterbation theory.

    for _ in 0..num_nudges {
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    if i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1 {
                        // continue
                    }

                    // todo: Maybe check if diff is belwo an eps, then take no action

                    // Note that changing psi affects both these things.
                    // todo: QC how this works.
                    // let diff = pp_calculated[i][j][k] - pp_measured[i][j][k];
                    // let diff = surfaces[2][i][j][k] - surfaces[3][i][j][k];
                    let psi_pp_diff =
                        sfcs.psi_pp_calculated[i][j][k] - sfcs.psi_pp_measured[i][j][k];

                    // From this diff in psi'' and E, calculate the psi change required
                    // to make this diff (Using the Schrodinger Eq, which may be a good enough
                    // proxy, assuming we started with a good trial WF)

                    // let psi_diff_guess = 1. / (E - sfcs.V[i][j][k]) * KE_COEFF_INV * sfcs.psi_pp_diff[i][j][k];

                    // Reasoning out steps:
                    // - We mus find out how to change psi'' (measured) by diff, and apply this
                    // change to psi. Does this mean taking a deriv of psi''-calculated at this [i][j][k]?
                    //- Plug `psi_pp_diff` into this deriv, and pull psi out of it. Use that new psi.

                    // - I think we need d_psi / d_psi''. Then we multiply that by the diff
                    // (Probably scaled down)

                    // todo: Consider applying a lowpass filter to the data after each nudge,
                    // to remove high-frequency noise added by the nudges.

                    // Increment psi, and see how psi'' calc reacts, from the Schrodinger calculation.
                    // This is how psi'' calculated reacts to a change in psi.
                    let d_psi_pp_calc__d_psi = (E - sfcs.V[i][j][k]) * KE_COEFF;
                    let d_psi__d_psi_pp_calc = 1. / (E - sfcs.V[i][j][k]) * KE_COEFF_INV;

                    // todo: I'm suspicious of this approach, since it doesn't take into account
                    // our nudges in neighbors.
                    // sfcs.psi[i][j][k] = d_psi;

                    let psi_pp_next_meas =
                        find_psi_pp_meas(&sfcs.psi, &sfcs.psi_pp_measured, i, j, k);
                    // sfcs.psi[i][j][k] -= d_psi;

                    let d_psi__d_psi_pp_meas =
                        d_psi / (psi_pp_next_meas - sfcs.psi_pp_measured[i][j][k]);

                    // println!(
                    // "Calc: {:.6} Meas: {:.6}",
                    // d_psi__d_psi_pp_calc, d_psi__d_psi_pp_meas
                    // );

                    let d_psi_d_psi_pp_diff = d_psi__d_psi_pp_calc - d_psi__d_psi_pp_meas;
                    // let d_psi_d_psi_pp_diff = d_psi__d_psi_pp_meas - d_psi__d_psi_pp_calc;

                    // println!("G {}", psi_diff_guess);
                    // Move down to create upward curvature at this pt, etc.
                    // sfcs.psi[i][j][k] -= nudge_amount * psi_pp_diff;

                    // let nudge_amt = -d_psi_d_psi_pp_diff * psi_pp_diff * 0.01;
                    let nudge = psi_pp_diff * nudge_amount;

                    sfcs.psi[i][j][k] -= nudge;
                    // sfcs.psi[i+1][j][k] = nudge_amount;
                    // sfcs.psi[i-1][j][k] = nudge_amount;
                    // sfcs.psi[i][j+1][k] = nudge_amount;
                    // sfcs.psi[i][j-1][k] = nudge_amount;
                    // sfcs.psi[i][j][k+1] = nudge_amount;
                    // sfcs.psi[i][j][k-1] = nudge_amount;

                    // sfcs.psi[i][j][k] = d_psi__d_psi_pp_meas * psi_pp_diff * 0.2;

                    // Now that we've updated psi, calculatd a new psi_pp_calulated,
                    // based on the energy.
                    // todo: Massage E here??
                    sfcs.psi_pp_calculated[i][j][k] = find_psi_pp_calc(sfcs, E, i, j, k);
                }
            }
        }

        // We must solve for psi IVO our sample point before measuring psi_pp.
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    sfcs.psi_pp_measured[i][j][k] =
                        find_psi_pp_meas(&sfcs.psi, &sfcs.psi_pp_measured, i, j, k);
                }
            }
        }
    }
}

/// todo: This should probably be a method on `State`.
/// This is our main computation function for sfcs.
/// Modifies in place to conserve memory.
fn eval_wf(
    wfs: &[(BasisFn, f64)],
    gauss: &Vec<Gaussian>,
    charges: &[(Vec3, f64)],
    sfcs: &mut Surfaces,
    E: f64,
) {
    // output score. todo: Move score to a diff fn?
    // ) -> (Vec<(Arr3d, String)>, f64) {
    // Schrod eq for H:
    // V for hydrogen: K_C * Q_PROT / r

    // psi(r)'' = (E - V(r)) * 2*m/ħ**2 * psi(r)
    // psi(r) = (E - V(R))^-1 * ħ**2/2m * psi(r)''

    let x_vals = linspace((GRID_MIN, GRID_MAX), N);
    let y_vals = linspace((GRID_MIN, GRID_MAX), N);
    let z_vals = linspace((GRID_MIN, GRID_MAX), N);

    // Our initial psi'' measured uses our analytic LCAO system, which doesn't have the
    // grid edge and precision issues of the fixed numerical grid we use to tune the trial
    // WF.
    for (i, x) in x_vals.iter().enumerate() {
        for (j, y) in y_vals.iter().enumerate() {
            for (k, z) in z_vals.iter().enumerate() {
                let posit_sample = Vec3::new(*x, *y, *z);

                // Calculate psi'' based on a numerical derivative of psi
                // in 3D.

                let x_prev = Vec3::new(posit_sample.x - H, posit_sample.y, posit_sample.z);
                let x_next = Vec3::new(posit_sample.x + H, posit_sample.y, posit_sample.z);
                let y_prev = Vec3::new(posit_sample.x, posit_sample.y - H, posit_sample.z);
                let y_next = Vec3::new(posit_sample.x, posit_sample.y + H, posit_sample.z);
                let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - H);
                let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + H);

                let mut psi_x_prev = 0.;
                let mut psi_x_next = 0.;
                let mut psi_y_prev = 0.;
                let mut psi_y_next = 0.;
                let mut psi_z_prev = 0.;
                let mut psi_z_next = 0.;

                sfcs.V[i][j][k] = 0.;
                sfcs.psi[i][j][k] = 0.;

                for (i_charge, (posit_charge, charge_amt)) in charges.iter().enumerate() {
                    let (basis, weight) = &wfs[i_charge];

                    let wf = basis.f();

                    sfcs.psi[i][j][k] += wf(*posit_charge, posit_sample) * weight;

                    sfcs.V[i][j][k] += V_coulomb(*posit_charge, posit_sample, *charge_amt);

                    psi_x_prev += wf(*posit_charge, x_prev) * weight;
                    psi_x_next += wf(*posit_charge, x_next) * weight;
                    psi_y_prev += wf(*posit_charge, y_prev) * weight;
                    psi_y_next += wf(*posit_charge, y_next) * weight;
                    psi_z_prev += wf(*posit_charge, z_prev) * weight;
                    psi_z_next += wf(*posit_charge, z_next) * weight;
                }

                for gauss_basis in gauss {
                    sfcs.psi[i][j][k] += gauss_basis.val(posit_sample);
                }

                sfcs.psi_pp_calculated[i][j][k] = find_psi_pp_calc(sfcs, E, i, j, k);

                sfcs.psi_pp_measured[i][j][k] =
                    psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
                        - 6. * sfcs.psi[i][j][k];

                sfcs.psi_pp_measured[i][j][k] /= H.powi(2)
            }
        }
    }
}

fn main() {
    let wfs = vec![
        (BasisFn::H100, 1.),
        (BasisFn::H100, 1.),
        (BasisFn::H100, 1.),
    ];

    let gaussians = vec![Gaussian {
        a_x: 0.,
        b_x: 0.,
        c_x: 0.,
        a_y: 0.,
        b_y: 0.,
        c_y: 0.,
        a_z: 0.,
        b_z: 0.,
        c_z: 0.,
    }];

    // H ion nuc dist is I believe 2 bohr radii.
    // let charges = vec![(Vec3::new(-1., 0., 0.), Q_PROT), (Vec3::new(1., 0., 0.), Q_PROT)];
    let charges = vec![
        (Vec3::new(-1., 0., 0.), Q_PROT),
        (Vec3::new(1., 0., 0.), Q_PROT),
        // (Vec3::new(0., 1., 0.), Q_ELEC),
    ];

    let z_displayed = 0.;
    let E = -0.7;

    let mut sfcs = Default::default();

    eval_wf(&wfs, &gaussians, &charges, &mut sfcs, E);

    let psi_pp_score = score_wf(&sfcs, E);

    let show_surfaces = [true, true, true, true];

    let surface_names = [
        "V".to_owned(),
        "ψ".to_owned(),
        "ψ'' calculated".to_owned(),
        "ψ'' measured".to_owned(),
    ];

    let z = vec![4; N];
    let y = vec![z; N];
    let grid_divisions = vec![y; N];

    let state = State {
        wfs,
        charges,
        surfaces: sfcs,
        E,
        z_displayed,
        psi_pp_score,
        surface_names,
        show_surfaces,
        grid_divisions,

        gaussians,
    };

    render::render(state);
}
