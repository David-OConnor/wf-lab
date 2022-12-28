//! This program explores solving the wave equation for
//! arbitrary potentials. It visualizes the wave function in 3d, with user interaction.

// todo: Imaginary part of WFs?

#![allow(non_snake_case)]

use std::{boxed::Box, f64::consts::PI};

use lin_alg2::f64::{Quaternion, Vec3};

mod render;
mod ui;

const NUM_SURFACES: usize = 4; // V, psi, psi_pp_calculated, psi_pp_measured

const A_0: f64 = 1.;
const Z_H: f64 = 1.;
const K_C: f64 = 1.;
const Q_PROT: f64 = 1.;
const Q_ELEC: f64 = -1.;
// const  M_ELEC: f64 = 5.45e-4
const M_ELEC: f64 = 1.; // todo: Which?
const ħ: f64 = 1.;

// Wave function number of values per edge.
// Memory use and some parts of computation scale with the cube of this.
const N: usize = 76;
// Used for calculating numerical psi''.
// Smaller is more precise. Applies to dx, dy, and dz
const H: f64 = 0.00001;
const GRID_MIN: f64 = -3.;
const GRID_MAX: f64 = 3.;

// type Arr2d = [[f64; N]; N];
// type Arr3d = [[[f64; N]; N]; N];
type Arr3d = Vec<Vec<Vec<f64>>>;
type wf_type = dyn Fn(Vec3, Vec3) -> f64;
// type wf_type = dyn Fn(Vec3, Vec3) -> f64;
// type wf_type = fn(Vec3, Vec3) -> f64;

// todo: Consider static allocation instead of vecs when possible.

// todo: troubleshooting stack overflow issues
// static mut V_vals: Arr3d = [[[0.; N]; N]; N];
// static mut psi: Arr3d = [[[0.; N]; N]; N];
// static mut psi_pp_measured: Arr3d = [[[0.; N]; N]; N];
// static mut psi_pp_calculated: Arr3d = [[[0.; N]; N]; N];

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
        let mut z = Vec::new();
        z.resize(N, 0.);

        let mut y = Vec::new();
        y.resize(N, z);

        let mut x = Vec::new();
        x.resize(N, y);

        // for i in 0..target[0].len() {
        //     for j in 0..target[1].len() {
        //         for k in 0..target[2].len() {
        //             result += (attempt[i][j][k] - target[i][j][k]).powi(2);
        //         }
        //     }
        // }

        Self {
            V: x.clone(),
            psi: x.clone(),
            psi_pp_calculated: x.clone(),
            psi_pp_measured: x,
        }
    }
}

// #[derive(Default)]
pub struct State {
    /// todo: Combine wfs and nuclei in into single tuple etc to enforce index pairing?
    /// todo: Or a sub struct?
    /// Wave functions, with weights
    // pub wfs: Vec<(impl Fn(Vec3, Vec3) -> f64 + 'static, f64)>,
    pub wfs: Vec<(usize, f64)>, // todo: currently unable to store wfs, so
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
}

/// Score using a least-squares regression.
fn score_wf(target: &Arr3d, attempt: &Arr3d) -> f64 {
    let mut result = 0.;

    for i in 0..target[0].len() {
        for j in 0..target[1].len() {
            for k in 0..target[2].len() {
                result += (attempt[i][j][k] - target[i][j][k]).powi(2);
            }
        }
    }

    result / target.len().pow(3) as f64
}

/// Single-point Coulomb potential, eg a hydrogen nuclei.
fn V_coulomb(posit_nuc: Vec3, posit_sample: Vec3, charge: f64) -> f64 {
    let diff = posit_sample - posit_nuc;
    let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

    -K_C * charge / r
}

/// https://chem.libretexts.org/Courses/University_of_California_Davis/UCD_Chem_107B%3A_Physical_Chemistry_for_Life_Scientists/Chapters/4%3A_Quantum_Theory/
/// 4.10%3A_The_Schr%C3%B6dinger_Wave_Equation_for_the_Hydrogen_Atom
/// Analytic solution for n=1, s orbital
fn h_wf_100(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let diff = posit_sample - posit_nuc;
    let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

    let ρ = Z_H * r / A_0;
    1. / PI.sqrt() * (Z_H / A_0).powf(3. / 2.) * (-ρ).exp()
    // 1. / sqrt(pi) * 1./ A_0.powf(3. / 2.) * (-ρ).exp()
}

/// Analytic solution for n=2, s orbital
fn h_wf_200(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let diff = posit_sample - posit_nuc;
    let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

    let ρ = Z_H * r / A_0;
    1. / (32. * PI).sqrt() * (Z_H / A_0).powf(3. / 2.) * (2. - ρ) * (-ρ / 2.).exp()
}

fn h_wf_210(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let diff = posit_sample - posit_nuc;
    let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

    // todo wrong
    // We take Cos theta below, so no need for cos^-1 here.
    // todo: Not sure how we deal with diff phis?
    let cos_theta = posit_nuc.to_normalized().dot(posit_sample.to_normalized());

    let ρ = Z_H * r / A_0;
    1. / (32. * PI).sqrt() * (Z_H / A_0).powf(3. / 2.) * ρ * (-ρ / 2.).exp() * cos_theta
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
fn find_psi_pp_calc(surfaces: &Surfaces, E: f64, i: usize, j: usize, k: usize) -> f64 {
    unsafe { (E - surfaces.V[i][j][k]) * -2. * M_ELEC / ħ.powi(2) * surfaces.psi[i][j][k] }
}

/// Calcualte psi'', measured, using the finite diff method, for a single value.
fn find_psi_pp_meas(
    surfaces: &Surfaces,
    posit_sample: Vec3,
    // wfs: &Vec<(usize, f64)>,
    // charges: &Vec<(Vec3, f64)>,
    E: f64,
    i: usize,
    j: usize,
    k: usize,
) -> f64 {
    // Using purely the numerical psi, we are now limited to the grid, for now.
    let H_GRID = (GRID_MAX - GRID_MIN) / (N as f64);

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

    // todo: At this point, you've done everything here from eval_wf except
    // todo th potential, and you're modifying psi in place instead
    // of creating it from basis fns.

    if i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1 {
        return surfaces.psi_pp_measured[i][j][k];
    }

    // for (i_charge, (posit_charge, charge_amt)) in charges.into_iter().enumerate() {
    // todo: THis is wrong. You've changed the *grid-based* (surface)
    // psi. You can no longer use these wave funcs!
    // todo: This is what you need to attack next.

    // let (wf_i, weight) = wfs[i_charge];

    // let wf = match wf_i {
    //     1 => h_wf_100,
    //     2 => h_wf_200,
    //     _ => h_wf_210,
    // };

    // psi_x_prev += wf(*posit_charge, x_prev) * weight;
    // psi_x_next += wf(*posit_charge, x_next) * weight;
    // psi_y_prev += wf(*posit_charge, y_prev) * weight;
    // psi_y_next += wf(*posit_charge, y_next) * weight;
    // psi_z_prev += wf(*posit_charge, z_prev) * weight;
    // psi_z_next += wf(*posit_charge, z_next) * weight;

    psi_x_prev += surfaces.psi[i - 1][j][k];
    psi_x_next += surfaces.psi[i + 1][j][k];
    psi_y_prev += surfaces.psi[i][j - 1][k];
    psi_y_next += surfaces.psi[i][j + 1][k];
    psi_z_prev += surfaces.psi[i][j][k - 1];
    psi_z_next += surfaces.psi[i][j][k + 1];

    let mut result = 0.;
    result += psi_x_prev + psi_x_next - 2. * surfaces.psi[i][j][k];
    result += psi_y_prev + psi_y_next - 2. * surfaces.psi[i][j][k];
    result += psi_z_prev + psi_z_next - 2. * surfaces.psi[i][j][k];
    result /= H_GRID.powi(2); // todo: Hard-code this in a const etc.

    result
}

/// Apply a correction to the WF, in attempt to make our two psi''s closer.
/// Uses our numerically-calculated WF.
// fn nudge_wf(wf: &mut Arr3d, pp_calculated: &Arr3d, pp_measured: &Arr3d) {
// fn nudge_wf(surfaces: &mut [&mut Arr3d; NUM_SURFACES]) {
fn nudge_wf(surfaces: &mut Surfaces, E: f64) {
    // fn nudge_wf(wfs: &Vec<(usize, f64)>, charges: &Vec<(Vec3, f64)>, E: f64) {
    let nudge_amount = 0.001;

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                // todo: Maybe check if diff is belwo an eps, then take no action

                // Note that changing psi affects both these things.
                // todo: QC how this works.
                // let diff = pp_calculated[i][j][k] - pp_measured[i][j][k];
                // let diff = surfaces[2][i][j][k] - surfaces[3][i][j][k];
                let diff = surfaces.psi_pp_calculated[i][j][k] - surfaces.psi_pp_measured[i][j][k];

                // Move down to create upward curvature at this pt, etc.
                // wf[i][j][k] -= nudge_amount * diff;
                surfaces.psi[i][j][k] -= nudge_amount * diff;

                // Now that we've updated psi, calculatd a new psi_pp_calulated,
                // based on the energy.
                // todo: Massage E here??
                surfaces.psi_pp_calculated[i][j][k] = find_psi_pp_calc(&surfaces, E, i, j, k);
            }
        }
    }

    // Now, update psi_pp_measured.
    // for i in 0..N {
    //     for j in 0..N {
    //         for k in 0..N {

    // todo: More DRY with eval_wf!
    let x_vals = linspace((GRID_MIN, GRID_MAX), N);
    let y_vals = linspace((GRID_MIN, GRID_MAX), N);
    let z_vals = linspace((GRID_MIN, GRID_MAX), N);

    for (i, x) in x_vals.iter().enumerate() {
        for (j, y) in y_vals.iter().enumerate() {
            for (k, z) in z_vals.iter().enumerate() {
                let posit_sample = Vec3::new(*x, *y, *z);

                surfaces.psi_pp_measured[i][j][k] =
                        // find_psi_pp_meas(posit_sample, wfs, charges, E, i, j, k);
                        find_psi_pp_meas(surfaces, posit_sample, E, i, j, k);
            }
        }
    }
}

/// todo: This should probably be a method on `State`.
/// This is our main computation function for surfaces.
/// Modifies in place to conserve memory.
fn eval_wf(
    // wfs: &Vec<(wf_type, f64)>,
    wfs: &Vec<(usize, f64)>,
    charges: &Vec<(Vec3, f64)>,
    surfaces: &mut Surfaces,
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

    let potential_fn = V_coulomb;

    for (i, x) in x_vals.iter().enumerate() {
        for (j, y) in y_vals.iter().enumerate() {
            for (k, z) in z_vals.iter().enumerate() {
                let posit_sample = Vec3::new(*x, *y, *z);

                // Calculate psi'' based on a numerical derivative of psi
                // in 3D.

                // todo: Use find_psi_pp_meas() here if you can, but wouldu require
                // todo an additional look through charges and wf.

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

                surfaces.V[i][j][k] = 0.;
                surfaces.psi[i][j][k] = 0.;

                for (i_charge, (posit_charge, charge_amt)) in charges.into_iter().enumerate() {
                    let (wf_i, weight) = wfs[i_charge];

                    let wf = match wf_i {
                        1 => h_wf_100,
                        2 => h_wf_200,
                        _ => h_wf_210,
                    };

                    surfaces.psi[i][j][k] += wf(*posit_charge, posit_sample) * weight;

                    surfaces.V[i][j][k] += potential_fn(*posit_charge, posit_sample, *charge_amt);

                    psi_x_prev += wf(*posit_charge, x_prev) * weight;
                    psi_x_next += wf(*posit_charge, x_next) * weight;
                    psi_y_prev += wf(*posit_charge, y_prev) * weight;
                    psi_y_next += wf(*posit_charge, y_next) * weight;
                    psi_z_prev += wf(*posit_charge, z_prev) * weight;
                    psi_z_next += wf(*posit_charge, z_next) * weight;
                }

                surfaces.psi_pp_calculated[i][j][k] = find_psi_pp_calc(&surfaces, E, i, j, k);

                surfaces.psi_pp_measured[i][j][k] = 0.;
                surfaces.psi_pp_measured[i][j][k] +=
                    psi_x_prev + psi_x_next - 2. * surfaces.psi[i][j][k];
                surfaces.psi_pp_measured[i][j][k] +=
                    psi_y_prev + psi_y_next - 2. * surfaces.psi[i][j][k];
                surfaces.psi_pp_measured[i][j][k] +=
                    psi_z_prev + psi_z_next - 2. * surfaces.psi[i][j][k];
                surfaces.psi_pp_measured[i][j][k] /= H.powi(2); // todo: Hard-code this in a const etc.
            }
        }
    }
    // psi_pp_measured[i] = 0.25 * psi_x_prev2 + psi_x_next2 + psi_y_prev2 + psi_y_next2 + \
    // psi_z_prev2 + psi_z_next2 - 6. * psi[i]
}

fn main() {
    let wfs = vec![
        (1, 1.),
        (1, -1.),
        // (1, 1.),
        // (h_wf_100, 1.),
        // (h_wf_100, 1.),
    ];
    // let nuclei = vec![Vec3::new(-0.5, 0., 0.), Vec3::new(0.5, 0., 0.)];
    // let nuclei = vec![Vec3::new(0., 0., 0.)];
    // H ion nuc dist is I believe 2 bohr radii.
    // let charges = vec![(Vec3::new(-1., 0., 0.), Q_PROT), (Vec3::new(1., 0., 0.), Q_PROT)];
    let charges = vec![
        (Vec3::new(-0.5, 0., 0.), Q_PROT),
        (Vec3::new(0.5, 0., 0.), Q_PROT),
        // (Vec3::new(0., 1., 0.), Q_ELEC),
    ];

    let z_displayed = 0.;
    let E = -0.7;

    // let (surfaces, surface_names, psi_pp_score) = eval_wf(&wfs, &charges, E);

    // This our sole allocation for the surfaces. We use a `Box` to assign this memory to the heap,
    // since it might get large otherwise. An alternative is to use Vecs, but this would
    // use more allocations. (?)

    // let mut surfaces2 = Box::new([
    //     [[[0.; N]; N]; N],
    //     // [[[0.; N]; N]; N],
    //     // [[[0.; N]; N]; N],
    //     // [[[0.; N]; N]; N],
    // ]);

    // let surfaces = unsafe { [&V_vals, &psi, &psi_pp_calculated, &psi_pp_measured] };

    let mut surfaces = Default::default();

    // let psi_pp_score = eval_wf(&wfs, &charges, &mut surfaces, E);
    eval_wf(&wfs, &charges, &mut surfaces, E);

    let psi_pp_score = score_wf(&surfaces.psi_pp_calculated, &surfaces.psi_pp_measured);

    let show_surfaces = [true, true, true, true];

    let surface_names = [
        "V".to_owned(),
        "ψ".to_owned(),
        "ψ'' calculated".to_owned(),
        "ψ'' measured".to_owned(),
    ];

    let state = State {
        wfs,
        charges,
        surfaces,
        E,
        z_displayed,
        psi_pp_score,
        surface_names,
        show_surfaces,
    };

    render::render(state);
}
