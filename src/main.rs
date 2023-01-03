//! This program explores solving the wave equation for
//! arbitrary potentials. It visualizes the wave function in 3d, with user interaction.

// todo: Consider instead of H orbitals, use the full set of Slater basis
// functions, which are more general. Make a fn to generate them.

// todo: Hylleraas basis functions?

#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]

use lin_alg2::f64::Vec3;

use basis_wfs::BasisFn;

mod basis_wfs;
mod render;
mod ui;

const NUM_SURFACES: usize = 6;

// We use Hartree units: ħ, elementary charge, electron mass, and Bohr radius.
const K_C: f64 = 1.;
const Q_PROT: f64 = 1.;
const Q_ELEC: f64 = -1.;
const M_ELEC: f64 = 1.;
const ħ: f64 = 1.;

// Compute these statically, to avoid continuous calls during excecution.
const KE_COEFF: f64 = -2. * M_ELEC / (ħ * ħ);
const KE_COEFF_INV: f64 = 1. / KE_COEFF;

// Wave function number of values per edge.
// Memory use and some parts of computation scale with the cube of this.
// Note: Using this as our fine grid. We will potentially subdivide it once
// or twice per axis, hence the multiple of 4 constraint.
// const N: usize = 21 * 4;
const N: usize = 50;

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
    /// Aux surfaces are for misc visualizations
    pub aux1: Arr3d,
    pub aux2: Arr3d,
}

impl Default for Surfaces {
    /// Fills with 0.s
    fn default() -> Self {
        let data = new_data();

        Self {
            V: data.clone(),
            psi: data.clone(),
            psi_pp_calculated: data.clone(),
            psi_pp_measured: data.clone(),
            aux1: data.clone(),
            aux2: data,
        }
    }
}

/// Represents a gaussian function.
/// "The parameter a is the height of the curve's peak, b is the
/// position of the center of the peak, and c (the standard deviation,
/// sometimes called the Gaussian RMS width) controls the width of the
/// "bell"."
#[derive(Clone, Copy)]
pub struct Gaussian {
    /// Aka `b`.
    pub posit: Vec3,
    pub a_x: f64,
    pub a_y: f64,
    pub a_z: f64,
    pub c_x: f64,
    pub c_y: f64,
    pub c_z: f64,
}

impl Gaussian {
    pub fn new_symmetric(posit: Vec3, a: f64, c: f64) -> Self {
        Self {
            posit,
            a_x: a,
            a_y: a,
            a_z: a,
            c_x: c,
            c_y: c,
            c_z: c,
        }
    }

    /// Helper fn
    fn val_1d(x: f64, a: f64, b: f64, c: f64) -> f64 {
        let part_1 = (x - b).powi(2) / (2. * c.powi(2));
        a * (-part_1).exp()
    }

    pub fn val(&self, posit_sample: Vec3) -> f64 {
        let diff = posit_sample - self.posit;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        // todo: QC how this works in 3d
        Self::val_1d(r, self.a_x, self.posit.x, self.c_x)
        // + Self::val_1d(posit_sample.y, self.a_y, self.posit.y, self.c_y)
        // + Self::val_1d(posit_sample.z, self.a_z, self.posit.z, self.c_z)
    }
}

pub struct Basis {
    pub f: BasisFn,
    pub posit: Vec3,
    pub weight: f64,
}

impl Basis {
    pub fn new(f: BasisFn, posit: Vec3, weight: f64) -> Self {
        Self { f, posit, weight }
    }
}

// #[derive(Default)]
pub struct State {
    /// todo: Combine wfs and nuclei in into single tuple etc to enforce index pairing?
    /// todo: Or a sub struct?
    /// Wave functions, with weights
    // pub wfs: Vec<(impl Fn(Vec3, Vec3) -> f64 + 'static, f64)>,
    pub wfs: Vec<Basis>, // todo: Rename, eg `bases`
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
    // /// This defines how our 3D grid is subdivided in different areas.
    // /// todo: FIgure this out.
    // /// todo: We should possibly remove grid divisions, as we may move
    // /// today away from the grid for all but plotting.
    // pub grid_divisions: Vec<Vec<Vec<u8>>>,
    /// Experimenting with gaussians; if this works out, it should possibly be
    /// combined with the BasisFn (wfs field).
    pub gaussians: Vec<Gaussian>,
}

/// Score using the fidelity of psi'' calculated vs measured; |<psi_trial | psi_true >|^2
fn score_wf(sfcs: &Surfaces, E: f64) -> f64 {
    // "The accuracy should be scored by the fidelity of the wavefunction compared
    // to the true wavefunction. Fidelity is defined as |<psi_trial | psi_true >|^2.
    // For normalized states, this will always be bounded from above by 1.0. So it's
    // lower than 1.0 for an imperfect variational function, but is 1 if you are
    // able to exactly express it.""

    // For normalization.
    let mut norm_sq_calc = 0.;
    let mut norm_sq_meas = 0.;

    // todo: Dkn't allocate each time?
    // let mut psi_fm_meas = new_data();

    // const EPS: f64 = 0.0001;

    // Create normalization const.
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                // norm_sq_trial += sfcs.psi[i][j][k].powi(2);

                // Numerical anomolies that should balance a very low number by
                // a very high one don't work out here; set to 0.(?)
                // todo: QC if this is really solving your problem with spike ring.
                // psi_fm_meas[i][j][k] = if (E - sfcs.V[i][j][k]).abs() < EPS {
                //     0.
                // } else {
                //     KE_COEFF_INV / (E - sfcs.V[i][j][k]) * sfcs.psi_pp_measured[i][j][k]
                // };

                // norm_sq_meas += psi_fm_meas[i][j][k].powi(2);

                norm_sq_calc += sfcs.psi_pp_calculated[i][j][k].powi(2);
                norm_sq_meas += sfcs.psi_pp_measured[i][j][k].powi(2);
            }
        }
    }

    let norm_calc = norm_sq_calc.sqrt();
    let norm_meas = norm_sq_meas.sqrt();

    let mut fidelity = 0.;

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                fidelity +=
                    // (psi_fm_meas[i][j][k] / norm_sq_meas) * (sfcs.psi[i][j][k] / norm_trial);
                    // sfcs.psi_pp_calculated[i][j][k] / norm_sq_calc * sfcs.psi_pp_measured[i][j][k] / norm_sq_meas;
                    // (sfcs.psi_pp_calculated[i][j][k] / norm_sq_calc) * (sfcs.psi_pp_calculated[i][j][k] / norm_sq_calc);
                    // (sfcs.psi_pp_measured[i][j][k] / norm_sq_meas) * (sfcs.psi_pp_measured[i][j][k] / norm_sq_meas);

                    // todo: I can't get "fidelity" working; bakc to leaset-squares
                    (sfcs.psi_pp_calculated[i][j][k] - sfcs.psi_pp_measured[i][j][k]).powi(2)
            }
        }
    }

    // fidelity.powi(2)
    fidelity
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
    posit_sample: Vec3,
    bases: &[Basis],
    charges: &[(Vec3, f64)],
    gauss: &[Gaussian],
    i: usize,
    j: usize,
    k: usize,
) -> f64 {
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

    for basis in bases {
        let wf = basis.f.f();

        psi_x_prev += wf(basis.posit, x_prev) * basis.weight;
        psi_x_next += wf(basis.posit, x_next) * basis.weight;
        psi_y_prev += wf(basis.posit, y_prev) * basis.weight;
        psi_y_next += wf(basis.posit, y_next) * basis.weight;
        psi_z_prev += wf(basis.posit, z_prev) * basis.weight;
        psi_z_next += wf(basis.posit, z_next) * basis.weight;
    }

    for gauss_basis in gauss {
        psi_x_prev += gauss_basis.val(x_prev);
        psi_x_next += gauss_basis.val(x_next);
        psi_y_prev += gauss_basis.val(y_prev);
        psi_y_next += gauss_basis.val(y_next);
        psi_z_prev += gauss_basis.val(z_prev);
        psi_z_next += gauss_basis.val(z_next);
    }

    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - 6. * psi[i][j][k];

    result / H.powi(2)
}

/// Apply a correction to the WF, in attempt to make our two psi''s closer.
/// Uses our numerically-calculated WF. Updates psi, and both psi''s.
fn nudge_wf(
    sfcs: &mut Surfaces,
    wfs: &[Basis],
    charges: &[(Vec3, f64)],
    gauss: &mut Vec<Gaussian>,
    E: f64,
) {
    let nudge_amount = 0.0001;

    let num_nudges = 1;
    let d_psi = 0.001;

    let nudge_width = 0.1;

    // todo: Once out of the shower, look for more you can optimize out!

    // todo: Check for infinities etc around the edges

    // todo: Variational method and perterbation theory.

    let x_vals = linspace((GRID_MIN, GRID_MAX), N);
    let y_vals = linspace((GRID_MIN, GRID_MAX), N);
    let z_vals = linspace((GRID_MIN, GRID_MAX), N);

    // todo: DRY with eval_wf

    let skip = 20;

    // for _ in 0..num_nudges {
    for (i, x) in x_vals.iter().enumerate() {
        // if i % skip != 0 {
        //     continue;
        // }
        for (j, y) in y_vals.iter().enumerate() {
            // if j % skip != 0 {
            //     continue;
            // }
            for (k, z) in z_vals.iter().enumerate() {
                // if k % skip != 0 {
                //     continue;
                // }
                let posit_sample = Vec3::new(*x, *y, *z);

                let diff = sfcs.psi_pp_calculated[i][j][k] - sfcs.psi_pp_measured[i][j][k];

                sfcs.aux1[i][j][k] = diff;
                // todo: aux 2 should be calcing psi from this.

                // let psi_fm_meas =
                //     KE_COEFF_INV / (E - sfcs.V[i][j][k]) * sfcs.psi_pp_measured[i][j][k];

                let psi_fm_meas = if (E - sfcs.V[i][j][k]).abs() < 0.01
                    || sfcs.psi_pp_measured[i][j][k].abs() < 0.01
                {
                    0.
                } else {
                    KE_COEFF_INV / (E - sfcs.V[i][j][k]) * sfcs.psi_pp_measured[i][j][k]
                };

                let psi_fm_diff = KE_COEFF_INV / (E - sfcs.V[i][j][k]) * diff;

                sfcs.aux1[i][j][k] = diff;

                // sfcs.aux2[i][j][k] = psi_fm_meas;
                sfcs.aux2[i][j][k] = psi_fm_diff;

                let a = diff * nudge_amount;

                // gauss.push(Gaussian {
                //     posit: posit_sample,
                //     a_x: a,
                //     a_y: a,
                //     a_z: a,
                //     c_x: nudge_width,
                //     c_y: nudge_width,
                //     c_z: nudge_width,
                // })
            }
        }
        // }

        // for (i, x) in x_vals.iter().enumerate() {
        //     for (j, y) in y_vals.iter().enumerate() {
        //         for (k, z) in z_vals.iter().enumerate() {
        //             let posit_sample = Vec3::new(*x, *y, *z);

        //             // todo: Maybe you can wrap up the psi and /or psi calc into the psi_pp_meas fn?

        //             sfcs.psi[i][j][k] = 0.;

        //             for (i_charge, (posit_charge, charge_amt)) in charges.iter().enumerate() {
        //                 let (basis, weight) = &wfs[i_charge];

        //                 let wf = basis.f();

        //                 sfcs.psi[i][j][k] += wf(*posit_charge, posit_sample) * weight;
        //             }

        //             sfcs.psi_pp_measured[i][j][k] =
        //                 find_psi_pp_meas(&sfcs.psi, posit_sample, wfs, charges, gauss, i, j, k);

        //             sfcs.psi_pp_calculated[i][j][k] = find_psi_pp_calc(sfcs, E, i, j, k);
        //         }
        //     }
        // }
    }
}

/// todo: This should probably be a method on `State`.
/// This is our main computation function for sfcs.
/// Modifies in place to conserve memory.
fn eval_wf(
    bases: &[Basis],
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

    // todo: Store these somewhere to save on computation; minor pt.
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

                sfcs.V[i][j][k] = 0.;
                for (posit_charge, charge_amt) in charges.iter() {
                    sfcs.V[i][j][k] += V_coulomb(*posit_charge, posit_sample, *charge_amt);
                }

                sfcs.psi[i][j][k] = 0.;
                for basis in bases {
                    sfcs.psi[i][j][k] += basis.f.f()(basis.posit, posit_sample) * basis.weight;
                }

                for gauss_basis in gauss {
                    sfcs.psi[i][j][k] += gauss_basis.val(posit_sample);
                }

                sfcs.psi_pp_calculated[i][j][k] = find_psi_pp_calc(sfcs, E, i, j, k);

                // todo: By delegating psi_pp_measured, we are causing an additional loop
                // through charges, wfs, gauss etc.

                sfcs.psi_pp_measured[i][j][k] =
                    find_psi_pp_meas(&sfcs.psi, posit_sample, bases, charges, gauss, i, j, k);
            }
        }
    }
}

fn main() {
    let x_axis = Vec3::new(1., 0., 0.);
    let posit_charge_1 = Vec3::new(-1., 0., 0.);
    let posit_charge_2 = Vec3::new(-2., 0., 0.);

    let wfs = vec![
        Basis::new(BasisFn::H100, posit_charge_1, 1.),
        Basis::new(BasisFn::H100, posit_charge_2, -1.),
        Basis::new(BasisFn::H200, posit_charge_1, 0.),
        Basis::new(BasisFn::H200, posit_charge_2, 0.),
        Basis::new(BasisFn::H210(x_axis), posit_charge_1, 0.),
        Basis::new(BasisFn::H210(x_axis), posit_charge_2, 0.),
        Basis::new(BasisFn::H300, posit_charge_1, 0.),
        Basis::new(BasisFn::Sto(1.), posit_charge_2, 0.),
    ];

    // let gaussians = vec![Gaussian::new_symmetric(Vec3::new(0., 0., 0.), 0.1, 2.)];
    let gaussians = Vec::new();

    // H ion nuc dist is I believe 2 bohr radii.
    // let charges = vec![(Vec3::new(-1., 0., 0.), Q_PROT), (Vec3::new(1., 0., 0.), Q_PROT)];
    let charges = vec![
        (posit_charge_1, Q_PROT),
        (posit_charge_2, Q_PROT),
        // (Vec3::new(0., 1., 0.), Q_ELEC),
    ];

    let z_displayed = 0.;
    let E = -0.7;

    let mut sfcs = Default::default();

    eval_wf(&wfs, &gaussians, &charges, &mut sfcs, E);

    let psi_pp_score = score_wf(&sfcs, E);

    let show_surfaces = [true, true, true, true, false, false];

    let surface_names = [
        "V".to_owned(),
        "ψ".to_owned(),
        "ψ'' calculated".to_owned(),
        "ψ'' measured".to_owned(),
        "Aux 1".to_owned(),
        "Aux 2".to_owned(),
    ];

    // let z = vec![4; N];
    // let y = vec![z; N];
    // let grid_divisions = vec![y; N];

    let state = State {
        wfs,
        charges,
        surfaces: sfcs,
        E,
        z_displayed,
        psi_pp_score,
        surface_names,
        show_surfaces,
        // grid_divisions,
        gaussians,
    };

    render::render(state);
}
