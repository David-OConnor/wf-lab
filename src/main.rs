//! This program explores solving the wave equation for
//! arbitrary potentials. It visualizes the wave function in 3d, with user interaction.

// todo: Imaginary part of WFs?

#![allow(non_snake_case)]

use core::f64::consts::{PI, TAU};

use lin_alg2::f64::{Quaternion, Vec3};

mod render;
mod ui;

const A_0: f64 = 1.;
const Z_H: f64 = 1.;
const K_C: f64 = 1.;
const Q_PROT: f64 = 1.;
const Q_ELEC: f64 = -1.;
// const  M_ELEC: f64 = 5.45e-4
const M_ELEC: f64 = 1.; // todo: Which?
const ħ: f64 = 1.;

const N: usize = 100;

type arr_2d = [[f64; N]; N];

#[derive(Default)]
pub struct State {
    /// Surface, show if true; hide if false.
    pub surfaces: Vec<((arr_2d, String), bool)>,
    /// Eg, least-squares over 2 or 3 dimensions between
    /// When visualizing a 2d wave function over X and Y, this is the fixed Z value.
    pub z: f64,
    /// Energy of the system
    pub E: f64,
    pub psi_pp_score: f64,
}

/// Score using a least-squares regression.
fn score_wf(target: &arr_2d, attempt: &arr_2d) -> f64 {
    let mut result = 0.;

    for i in 0..target[0].len() {
        for j in 0..target[1].len() {
            result += attempt[i][j] - target[i][j];
        }
    }

    result / target.len() as f64
}

/// Hydrogen potential.
fn V_h(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let diff = posit_sample - posit_nuc;
    let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

    -K_C * Q_PROT / r
}

/// Analytic solution for n=1, s orbital
fn h_wf_100(posit_nuc: Vec3, posit_sample: Vec3) -> f64 {
    let diff = posit_sample - posit_nuc;
    let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

    let ρ = Z_H * r / A_0;
    1. / PI.sqrt() * (Z_H / A_0).powf(3. / 2.) * (-ρ).exp()
    // 1. / sqrt(pi) * 1./ A_0.powf(3. / 2.) * (-ρ).exp()
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

fn eval_wf(z: f64, E: f64) -> (Vec<(arr_2d, String)>, f64) {
    // Schrod eq for H:
    // V for hydrogen: K_C * Q_PROT / r

    // psi(r)'' = (E - V(r)) * 2*m/ħ**2 * psi(r)
    // psi(r) = (E - V(R))^-1 * ħ**2/2m * psi(r)''

    let mut V_vals = [[0.; N]; N];
    let mut psi = [[0.; N]; N];
    let mut psi_pp_expected = [[0.; N]; N];
    let mut psi_pp_measured = [[0.; N]; N];

    let x_vals = linspace((-4., 4.), N);
    let y_vals = linspace((-4., 4.), N);

    // Used for calculating numerical psi''.
    // Smaller is more precise. Applies to dx, dy, and dz
    let h = 0.00001; // aka dx

    // let wf = wf_osc;
    let wf = h_wf_100;

    let potential_fn = V_h;
    // potential_fn = V_osc

    // letnuclei = [Vec3(2., 0., 0.)];
    // H ion nuc dist is I believe 2 bohr radii.
    let nuclei = [Vec3::new(-0.5, 0., 0.), Vec3::new(0.5, 0., 0.)];

    // psi = wf(posit_nuc, Vec3::n ))

    for (i, x) in x_vals.iter().enumerate() {
        for (j, y) in y_vals.iter().enumerate() {
            let posit_sample = Vec3::new(*x, *y, z); // todo: Inject z in a diff way.sc

            let mut V = 0.;

            for nuc in nuclei {
                // todo: Naiive superposition
                psi[i][j] += wf(nuc, posit_sample);

                V += potential_fn(nuc, posit_sample);
            }
            V_vals[i][j] = V;

            psi_pp_expected[i][j] = (E - V) * -2. * M_ELEC / ħ.powi(2) * psi[i][j];

            // Calculate psi'' based on a numerical derivative of psi
            // in 3D.

            let x_prev = Vec3::new(posit_sample.x - h, posit_sample.y, posit_sample.z);
            let x_next = Vec3::new(posit_sample.x + h, posit_sample.y, posit_sample.z);
            let y_prev = Vec3::new(posit_sample.x, posit_sample.y - h, posit_sample.z);
            let y_next = Vec3::new(posit_sample.x, posit_sample.y + h, posit_sample.z);
            let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - h);
            let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + h);

            let mut psi_x_prev = 0.;
            let mut psi_x_next = 0.;
            let mut psi_y_prev = 0.;
            let mut psi_y_next = 0.;
            let mut psi_z_prev = 0.;
            let mut psi_z_next = 0.;

            for nuc in nuclei {
                psi_x_prev += wf(nuc, x_prev);
                psi_x_next += wf(nuc, x_next);
                psi_y_prev += wf(nuc, y_prev);
                psi_y_next += wf(nuc, y_next);
                psi_z_prev += wf(nuc, z_prev);
                psi_z_next += wf(nuc, z_next);
            }
            // println!("{}", psi_x_prev);

            psi_pp_measured[i][j] = 0.;
            psi_pp_measured[i][j] += psi_x_prev + psi_x_next - 2. * psi[i][j];
            psi_pp_measured[i][j] += psi_y_prev + psi_y_next - 2. * psi[i][j];
            psi_pp_measured[i][j] += psi_z_prev + psi_z_next - 2. * psi[i][j];
            psi_pp_measured[i][j] /= h.powi(2);
        }
    }
    // psi_pp_measured[i] = 0.25 * psi_x_prev2 + psi_x_next2 + psi_y_prev2 + psi_y_next2 + \
    // psi_z_prev2 + psi_z_next2 - 6. * psi[i]

    // println!("Psi: {:?}", psi);

    // todo: You should score over all 3D, not just this 2D slice.
    let score = score_wf(&psi_pp_expected, &psi_pp_measured);

    (
        vec![
            (V_vals, "V".to_owned()),
            (psi, "ψ".to_owned()),
            (psi_pp_expected, "ψ'' expected".to_owned()),
            (psi_pp_measured, "ψ'' measured".to_owned()),
        ],
        score,
    )
}

fn main() {
    let z = 0.;
    let E = -0.5;

    let data = eval_wf(z, E);
    let surfaces = data.0.into_iter().map(|s| (s, true)).collect();

    let mut state = State {
        surfaces,
        E,
        z,
        psi_pp_score: data.1,
    };

    render::render(state);
}
