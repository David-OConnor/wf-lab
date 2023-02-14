//! This module contains the bulk of the wave-function evalution and solving logic.

use crate::{
    basis_wfs::{Basis, SinExpBasisPt},
    complex_nums::Cplx,
    interp,
    util::{self, Arr3d, Arr3dBasis, Arr3dReal},
    rbf::Rbf,
};

use lin_alg2::f64::Vec3;

// We use Hartree units: ħ, elementary charge, electron mass, and Bohr radius.
const K_C: f64 = 1.;
pub const Q_PROT: f64 = 1.;
const Q_ELEC: f64 = -1.;
pub const M_ELEC: f64 = 1.;
pub const ħ: f64 = 1.;
const KE_COEFF_INV: f64 = 1. / KE_COEFF;

pub(crate) const NUDGE_DEFAULT: f64 = 0.01;

// Compute these statically, to avoid continuous calls during excecution.
const KE_COEFF: f64 = -2. * M_ELEC / (ħ * ħ);

// Wave function number of values per edge.
// Memory use and some parts of computation scale with the cube of this.
pub const N: usize = 40;

// Used for calculating numerical psi''.
// Smaller is more precise. Too small might lead to numerical issues though (?)
// Applies to dx, dy, and dz
const H: f64 = 0.01;
const H_SQ: f64 = H * H;

/// This is our main computation function for sfcs. It:
/// - Computes V from charges
/// - Computes a trial ψ from basis functions
/// - Computes ψ'' calculated, and measured from the trial ψ
/// Modifies in place to conserve memory. These operations are combined in the same function to
/// save computation, since they're often run at once, and can be iterated through using a single loop
/// through all grid points.
pub fn init_wf(
    bases: &[Basis],
    charges: &[(Vec3, f64)],
    sfcs: &mut Surfaces,
    E: f64,
    update_charges: bool,
    grid_min: &mut f64,
    grid_max: &mut f64,
) {
    // output score. todo: Move score to a diff fn?
    // ) -> (Vec<(Arr3d, String)>, f64) {
    // Schrod eq for H:
    // V for hydrogen: K_C * Q_PROT / r

    // psi(r)'' = (E - V(r)) * 2*m/ħ**2 * psi(r)
    // psi(r) = (E - V(R))^-1 * ħ**2/2m * psi(r)''

    const RANGE_PAD: f64 = 5.;

    // Set up the grid so that it smartly encompasses the charges, letting the WF go to 0
    // towards the edges
    // todo: For now, maintain a cubic grid centered on 0.
    if update_charges {
        let mut max_abs_val = 0.;
        for (posit, _) in charges {
            if posit.x.abs() > max_abs_val {
                max_abs_val = posit.x.abs();
            }
            if posit.y.abs() > max_abs_val {
                max_abs_val = posit.y.abs();
            }
            if posit.z.abs() > max_abs_val {
                max_abs_val = posit.z.abs();
            }
        }

        *grid_max = max_abs_val + RANGE_PAD;
        *grid_min = -*grid_max;
    }

    // todo: Store these somewhere to save on computation? minor pt.
    let grid_1d = util::linspace((*grid_min, *grid_max), N);

    // Our initial psi'' measured uses our analytic LCAO system, which doesn't have the
    // grid edge and precision issues of the fixed numerical grid we use to tune the trial
    // WF.
    for (i, x) in grid_1d.iter().enumerate() {
        for (j, y) in grid_1d.iter().enumerate() {
            for (k, z) in grid_1d.iter().enumerate() {
                let posit_sample = Vec3::new(*x, *y, *z);

                // Calculate psi'' based on a numerical derivative of psi
                // in 3D.

                if update_charges {
                    sfcs.V[i][j][k] = 0.;
                    for (posit_charge, charge_amt) in charges.iter() {
                        sfcs.V[i][j][k] += V_coulomb(*posit_charge, posit_sample, *charge_amt);
                    }

                    // Re why the electron interaction, in many cases, appears to be very small compared to protons: After thinking about it, the protons, being point charges (approximately) are pulling from a single direction. While most of the smudged out electron gets cancelled out in the area of interest
                    // But, it should follow that at a distance, the electsron force and potential is as strong as the proton's
                    // (Yet, at a distance, the electron and proton charges cancel each other out largely, unless it's an ion...)
                    // So I guess it follows that the interesting bits are in the intermediate distances...
                    // todo: Hard coded ito index 0.

                    // Oh boy... this will slow things down... Simulating a charge at every grid point.,
                    // acting on every other grid point.

                    // todo: This is going to be a deal breaker most likely.
                    // for (i2, x2) in vals_1d.iter().enumerate() {
                    //     for (j2, y2) in vals_1d.iter().enumerate() {
                    //         for (k2, z2) in vals_1d.iter().enumerate() {
                    //             let posit_sample_electron = Vec3::new(*x2, *y2, *z2);
                    //             // todo: This may not be quite right, ie matching the posit_sample grid with the i2, j2, k2 elec charges.
                    //             sfcs.V[i][j][k] += V_coulomb(posit_sample_electron, posit_sample, sfcs.elec_charges[0][i2][j2][k2]);
                    //         }
                    //     }
                    // }
                }

                sfcs.psi[i][j][k] = Cplx::new_zero();

                for basis in bases {
                    sfcs.psi[i][j][k] += basis.value(posit_sample) * basis.weight();
                }

                sfcs.psi_pp_calculated[i][j][k] = find_ψ_pp_calc(&sfcs.psi, &sfcs.V, E, i, j, k);

                // We can compute ψ'' measured this in the same loop here, since we're using an analytic
                // equation for ψ; we can diff at arbitrary points vice only along a grid of pre-computed ψ.

                // todo: Commented out; see below while testing our new bases
                sfcs.psi_pp_measured[i][j][k] =
                    find_ψ_pp_meas_fm_bases(posit_sample, bases, sfcs.psi[i][j][k]);
            }
        }
    }

    // todo: Start RBF test

    let charge_posits: Vec<Vec3> = charges.into_iter().map(|c| c.0).collect();

    let rbf = interp::setup_rbf_interp(&charge_posits, bases);

    // todo: Testing numerical diff based on RBF.
    for (i, x) in grid_1d.iter().enumerate() {
        if i == 0 || i == N - 1 {
            continue;
        }
        for (j, y) in grid_1d.iter().enumerate() {
            if j == 0 || j == N - 1 {
                continue;
            }
            for (k, z) in grid_1d.iter().enumerate() {
                if k == 0 || k == N - 1 {
                    continue;
                }

                let posit_sample = Vec3::new(*x, *y, *z);

                // todo: Set psi from thsi too to see if it looks right. Zooming in to demonstrate
                // todo interp.
                sfcs.psi[i][j][k] = Cplx::from_real(rbf.interp_point(Vec3::new(
                    posit_sample.x / 1.,
                    posit_sample.y / 1.,
                    posit_sample.z / 1.,
                )));

                sfcs.psi_pp_measured[i][j][k] =
                    find_ψ_pp_meas_fm_rbf(posit_sample, sfcs.psi[i][k][k], &rbf, *grid_min, *grid_max);
            }
        }
    }

    // Code to test interpolation.
    let b1 = &bases[0];

    let rbf_compare_pts = vec![
        Vec3::new(0., 0., 0.),
        Vec3::new(1., 0., 0.),
        Vec3::new(1., 1., 0.),
        Vec3::new(0.5, 0.5, 0.5),
    ];

    for pt in &rbf_compare_pts {
        println!(
            "\nBasis: {:.5} \n Rbf: {:.5}",
            b1.value(*pt).real * b1.weight(),
            rbf.interp_point(*pt),
        );
    }

    // todo end RBF test

    // todo: Start sinexpbasispt testing

    let grid_dx = (*grid_max - *grid_min) / N as f64;

    // Populate our exp/sin/poly bases, now that psi is populated. (These bases are functions of psi
    // at a given point, and neighbors)
    for (i, x) in grid_1d.iter().enumerate() {
        if i == 0 || i == N - 1 {
            continue;
        }
        for (j, y) in grid_1d.iter().enumerate() {
            if j == 0 || j == N - 1 {
                continue;
            }
            for (k, z) in grid_1d.iter().enumerate() {
                if k == 0 || k == N - 1 {
                    continue;
                }

                let posit_sample = Vec3::new(*x, *y, *z);

                // // todo: Consider the API to create these, eg arbitrary points, or fixed dists.
                // sfcs.bases[i][j][k] = SinExpBasisPt::from_neighbors(
                //     // todo: Do you need to incorporate the full 27 points around instead of just the 6
                //     // todo faces? Can you get a better interp from edges and corners?? Would probably
                //     // todo need a more expressive basis for that to work.
                //     (
                //         sfcs.psi[i - 1][j][k].real,
                //         sfcs.psi[i][j][k].real,
                //         sfcs.psi[i + 1][j][k].real,
                //     ),
                //     (
                //         sfcs.psi[i][j - 1][k].real,
                //         sfcs.psi[i][j][k].real,
                //         sfcs.psi[i][j + 1][k].real,
                //     ),
                //     (
                //         sfcs.psi[i][j][k - 1].real,
                //         sfcs.psi[i][j][k].real,
                //         sfcs.psi[i][j][k + 1].real,
                //     ),
                //     posit_sample,
                //     grid_dx,
                // );
                //
                // // todo: testing our new approach.
                // sfcs.psi_pp_measured[i][j][k] = find_ψ_pp_meas_from_interp2(
                //     posit_sample,
                //     &sfcs.psi,
                //     &sfcs.bases,
                //     *grid_min,
                //     *grid_max,
                //     i,
                //     j,
                //     k,
                // );
            }
        }
    }

    // todo: end sinexpbasispt testing
}

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

/// Score using the fidelity of psi'' calculated vs measured; |<psi_trial | psi_true >|^2.
/// This requires normalizing the wave functions we're comparing.
/// todo: Curretly not working.
/// todo: I don't think you can use this approach comparing psi''s with fidelity, since they're
/// todo not normalizsble.
// fn wf_fidelity(sfcs: &Surfaces) -> f64 {
fn fidelity(sfcs: &Surfaces) -> f64 {
    // "The accuracy should be scored by the fidelity of the wavefunction compared
    // to the true wavefunction. Fidelity is defined as |<psi_trial | psi_true >|^2.
    // For normalized states, this will always be bounded from above by 1.0. So it's
    // lower than 1.0 for an imperfect variational function, but is 1 if you are
    // able to exactly express it.""

    // For normalization.
    // let mut norm_sq_calc = 0.;
    // let mut norm_sq_meas = 0.;
    let mut norm_calc = Cplx::new_zero();
    let mut norm_meas = Cplx::new_zero();

    const SCORE_THRESH: f64 = 100.;

    // Create normalization const.
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                // norm_sq_calc += sfcs.psi_pp_calculated[i][j][k].abs_sq();
                // norm_sq_meas += sfcs.psi_pp_measured[i][j][k].abs_sq();
                // todo: .real is temp
                if sfcs.psi_pp_calculated[i][j][k].real.abs() < SCORE_THRESH
                    && sfcs.psi_pp_measured[i][j][k].real.abs() < SCORE_THRESH
                {
                    norm_calc += sfcs.psi_pp_calculated[i][j][k];
                    norm_meas += sfcs.psi_pp_measured[i][j][k];
                }
            }
        }
    }

    // Now that we have both wave functions and normalized them, calculate fidelity.
    let mut result = Cplx::new_zero();

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                // todo: .reals here may be a kludge and not working with complex psi.

                // todo: LHS should be conjugated.
                if sfcs.psi_pp_calculated[i][j][k].real.abs() < SCORE_THRESH
                    && sfcs.psi_pp_measured[i][j][k].real.abs() < SCORE_THRESH
                {
                    result += sfcs.psi_pp_calculated[i][j][k] / norm_calc.real
                        * sfcs.psi_pp_calculated[i][j][k]
                        / norm_calc.real;
                }
            }
        }
    }

    result.abs_sq()
}

/// Score a wave function by comparing the least-squares sum of its measured and
/// calculated second derivaties.
pub fn score_wf(sfcs: &Surfaces) -> f64 {
    let mut result = 0.;

    // Avoids numerical precision issues. Without this, certain values of N will lead
    // to a bogus score. Values of N both too high and too low can lead to this. Likely due to
    // if a grid value is too close to a charge source, the value baloons.
    const SCORE_THRESH: f64 = 10.;

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                // todo: Check if either individual is outside a thresh?

                let val =
                    (sfcs.psi_pp_calculated[i][j][k] - sfcs.psi_pp_measured[i][j][k]).abs_sq();
                if val < SCORE_THRESH {
                    result += val;
                }
            }
        }
    }

    result
}

/// Convert an array of Psi to one of electron potential. Modifies in place
/// to avoid unecessary allocations. Not-normalized.
fn charge_density_fm_psi_one(psi: &Arr3d, num_elecs: usize, i: usize, j: usize, k: usize) -> f64 {
    // Save computation on this constant factor.
    let psi_sq_size = 1.; // todo: Wrong! This should be a normalization constant.
    let c = -Q_ELEC * num_elecs as f64 / psi_sq_size;
    let mag = psi[i][j][k].abs_sq();
    mag * c
}

/// Single-point Coulomb potential, eg a hydrogen nuclei.
fn V_coulomb(posit_charge: Vec3, posit_sample: Vec3, charge: f64) -> f64 {
    let diff = posit_sample - posit_charge;
    let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

    -K_C * charge / r
}

/// Calcualte psi'', calculated from psi, and E.
/// At a given i, j, k.
pub fn find_ψ_pp_calc(psi: &Arr3d, V: &Arr3dReal, E: f64, i: usize, j: usize, k: usize) -> Cplx {
    // todo: Do you need to multiply KE_COEFF, or part of it, by the number of electrons
    // todo: in this WF?
    psi[i][j][k] * (E - V[i][j][k]) * KE_COEFF
}

/// Calcualte ψ'', numerically from ψ, using the finite diff method, for a single value.
/// Calculate ψ'' based on a numerical derivative of psi
/// in 3D.
pub(crate) fn find_ψ_pp_meas_fm_bases(
    posit_sample: Vec3,
    bases: &[Basis],
    psi_sample_loc: Cplx,
) -> Cplx {
    let x_prev = Vec3::new(posit_sample.x - H, posit_sample.y, posit_sample.z);
    let x_next = Vec3::new(posit_sample.x + H, posit_sample.y, posit_sample.z);
    let y_prev = Vec3::new(posit_sample.x, posit_sample.y - H, posit_sample.z);
    let y_next = Vec3::new(posit_sample.x, posit_sample.y + H, posit_sample.z);
    let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - H);
    let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + H);

    let mut psi_x_prev = Cplx::new_zero();
    let mut psi_x_next = Cplx::new_zero();
    let mut psi_y_prev = Cplx::new_zero();
    let mut psi_y_next = Cplx::new_zero();
    let mut psi_z_prev = Cplx::new_zero();
    let mut psi_z_next = Cplx::new_zero();

    for basis in bases {
        psi_x_prev += basis.value(x_prev) * basis.weight();
        psi_x_next += basis.value(x_next) * basis.weight();
        psi_y_prev += basis.value(y_prev) * basis.weight();
        psi_y_next += basis.value(y_next) * basis.weight();
        psi_z_prev += basis.value(z_prev) * basis.weight();
        psi_z_next += basis.value(z_next) * basis.weight();
    }

    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi_sample_loc * 6.;

    result / H_SQ
}

/// Calcualte ψ'' measured, using a discrete function, interpolated.
/// Calculate ψ'' based on a numerical derivative of psi
/// in 3D.
pub(crate) fn find_ψ_pp_meas_fm_rbf(
    posit_sample: Vec3,
    psi_sample: Cplx,
    rbf: &Rbf,
    grid_min: f64,
    grid_max: f64,
) -> Cplx {
    let grid_dx = (grid_max - grid_min) / N as f64;

    let h2 = grid_dx / 1.; // todo temp?

    let x_prev = Vec3::new(posit_sample.x - h2, posit_sample.y, posit_sample.z);
    let x_next = Vec3::new(posit_sample.x + h2, posit_sample.y, posit_sample.z);
    let y_prev = Vec3::new(posit_sample.x, posit_sample.y - h2, posit_sample.z);
    let y_next = Vec3::new(posit_sample.x, posit_sample.y + h2, posit_sample.z);
    let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - h2);
    let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + h2);

    let psi_x_prev = rbf.interp_point(x_prev);
    let psi_x_next = rbf.interp_point(x_next);
    let psi_y_prev = rbf.interp_point(y_prev);
    let psi_y_next = rbf.interp_point(y_next);
    let psi_z_prev = rbf.interp_point(z_prev);
    let psi_z_next = rbf.interp_point(z_next);

    // todo: real only for now.

    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi_sample.real * 6.;

    // result / H_SQ
    Cplx::from_real(result / (h2 * h2)) // todo real temp
}

/// Calcualte ψ'' measured, using a discrete function, interpolated.
/// Calculate ψ'' based on a numerical derivative of psi
/// in 3D.
pub(crate) fn find_ψ_pp_meas_from_interp(
    posit_sample: Vec3,
    psi: &Arr3d,
    grid_min: f64,
    grid_max: f64,
    i: usize,
    j: usize,
    k: usize,
) -> Cplx {
    let grid_dx = (grid_max - grid_min) / N as f64;

    // todo: This function is producing sub-optimal results when interpolating at other
    // todo than teh grid fn. You need a better algo, or don't use this.

    let h2 = grid_dx / 8.; // todo temp!!! Not working for values other than dx...

    let x_prev = Vec3::new(posit_sample.x - h2, posit_sample.y, posit_sample.z);
    let x_next = Vec3::new(posit_sample.x + h2, posit_sample.y, posit_sample.z);
    let y_prev = Vec3::new(posit_sample.x, posit_sample.y - h2, posit_sample.z);
    let y_next = Vec3::new(posit_sample.x, posit_sample.y + h2, posit_sample.z);
    let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - h2);
    let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + h2);

    // Given the points we're sampling are along the grid lines, we can (conveniently! do 1D
    // interpolation here, vice 3D.

    // todo: I think this 1d simplification won't work. I think perhaps even your
    // todo overall 3d approach won't work. But proper 3D interp will. (?)

    // let psi_x_prev = interp::linear_1d_cplx(
    //     x_prev.x,
    //     (posit_sample.x - grid_dx, posit_sample.x),
    //     psi[i - 1][j][k],
    //     psi[i][j][k],
    // );
    //
    // let psi_x_next = interp::linear_1d_cplx(
    //     x_next.x,
    //     (posit_sample.x, posit_sample.x + grid_dx),
    //     psi[i][j][k],
    //     psi[i + 1][j][k],
    // );
    //
    // let psi_y_prev = interp::linear_1d_cplx(
    //     y_prev.y,
    //     (posit_sample.y - grid_dx, posit_sample.y),
    //     psi[i][j - 1][k],
    //     psi[i][j][k],
    // );
    //
    // let psi_y_next = interp::linear_1d_cplx(
    //     y_next.y,
    //     (posit_sample.y, posit_sample.y + grid_dx),
    //     psi[i][j][k],
    //     psi[i][j + 1][k],
    // );
    //
    // let psi_z_prev = interp::linear_1d_cplx(
    //     z_prev.z,
    //     (posit_sample.z - grid_dx, posit_sample.z),
    //     psi[i][j][k - 1],
    //     psi[i][j][k],
    // );
    //
    // let psi_z_next = interp::linear_1d_cplx(
    //     z_next.z,
    //     (posit_sample.z, posit_sample.z + grid_dx),
    //     psi[i][j][k],
    //     psi[i][j][k + 1],
    // );

    let psi_x_prev = interp::linear_3d_cplx(
        x_prev,
        (posit_sample.x - grid_dx, posit_sample.x),
        (posit_sample.y, posit_sample.y + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.z, posit_sample.z + grid_dx), // On the edge; arbitrary which box we picked
        // todo: Coordinate system consistency? Does it matter here?
        psi[i - 1][j + 1][k + 1],
        psi[i - 1][j][k + 1],
        psi[i][j + 1][k + 1],
        psi[i][j][k + 1],
        psi[i - 1][j + 1][k],
        psi[i - 1][j][k],
        psi[i][j + 1][k],
        psi[i][j][k],
    );

    let psi_x_next = interp::linear_3d_cplx(
        x_next,
        (posit_sample.x, posit_sample.x + grid_dx),
        (posit_sample.y, posit_sample.y + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.z, posit_sample.z + grid_dx), // On the edge; arbitrary which box we picked
        psi[i][j + 1][k + 1],
        psi[i][j][k + 1],
        psi[i + 1][j + 1][k + 1],
        psi[i + 1][j][k + 1],
        psi[i][j + 1][k],
        psi[i][j][k],
        psi[i + 1][j + 1][k],
        psi[i + 1][j][k],
    );

    let psi_y_prev = interp::linear_3d_cplx(
        y_prev,
        (posit_sample.x, posit_sample.x + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.y - grid_dx, posit_sample.y),
        (posit_sample.z, posit_sample.z + grid_dx), // On the edge; arbitrary which box we picked
        psi[i][j][k + 1],
        psi[i][j - 1][k + 1],
        psi[i + 1][j][k + 1],
        psi[i + 1][j - 1][k + 1],
        psi[i][j][k],
        psi[i][j - 1][k],
        psi[i + 1][j][k],
        psi[i + 1][j - 1][k],
    );

    let psi_y_next = interp::linear_3d_cplx(
        y_next,
        (posit_sample.x, posit_sample.x + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.y, posit_sample.y + grid_dx),
        (posit_sample.z, posit_sample.z + grid_dx), // On the edge; arbitrary which box we picked
        psi[i][j + 1][k + 1],
        psi[i][j][k + 1],
        psi[i + 1][j + 1][k + 1],
        psi[i + 1][j][k + 1],
        psi[i][j + 1][k],
        psi[i][j][k],
        psi[i + 1][j + 1][k],
        psi[i + 1][j][k],
    );

    let psi_z_prev = interp::linear_3d_cplx(
        z_prev,
        (posit_sample.x, posit_sample.x + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.y, posit_sample.y + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.z - grid_dx, posit_sample.z),
        psi[i][j + 1][k],
        psi[i][j][k],
        psi[i + 1][j + 1][k],
        psi[i + 1][j][k],
        psi[i][j + 1][k - 1],
        psi[i][j][k - 1],
        psi[i + 1][j + 1][k - 1],
        psi[i + 1][j][k - 1],
    );

    let psi_z_next = interp::linear_3d_cplx(
        z_next,
        (posit_sample.x, posit_sample.x + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.y, posit_sample.y + grid_dx), // On the edge; arbitrary which box we picked
        (posit_sample.z, posit_sample.z + grid_dx),
        psi[i][j + 1][k + 1],
        psi[i][j][k + 1],
        psi[i + 1][j + 1][k + 1],
        psi[i + 1][j][k + 1],
        psi[i][j + 1][k],
        psi[i][j][k - 1],
        psi[i + 1][j + 1][k],
        psi[i + 1][j][k],
    );

    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi[i][j][k] * 6.;

    // result / H_SQ
    result / (h2 * h2)
}

/// Calcualte ψ'' measured, using our polynomial/sin/exp bases
pub(crate) fn find_ψ_pp_meas_from_interp2(
    posit_sample: Vec3,
    psi: &Arr3d,
    bases: &Arr3dBasis,
    grid_min: f64,
    grid_max: f64,
    i: usize,
    j: usize,
    k: usize,
) -> Cplx {
    let grid_dx = (grid_max - grid_min) / N as f64;

    // todo: For now, use only the basis function at the point, since we're using
    // todo small diffs from it. In the future, consider if you'd like to interpolate
    // todo from the basis-functions at neighboring points weighted by dist to each.

    let h2 = grid_dx / 10.; // todo temp!!! Not working for values other than dx...

    let h2 = 0.001;

    let x_prev = Vec3::new(posit_sample.x - h2, posit_sample.y, posit_sample.z);
    let x_next = Vec3::new(posit_sample.x + h2, posit_sample.y, posit_sample.z);
    let y_prev = Vec3::new(posit_sample.x, posit_sample.y - h2, posit_sample.z);
    let y_next = Vec3::new(posit_sample.x, posit_sample.y + h2, posit_sample.z);
    let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - h2);
    let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + h2);

    // todo, since our bases for now produce real vals

    let psi_x_prev = Cplx::from_real(bases[i][j][k].value(x_prev));
    let psi_x_next = Cplx::from_real(bases[i][j][k].value(x_next));
    let psi_y_prev = Cplx::from_real(bases[i][j][k].value(y_prev));
    let psi_y_next = Cplx::from_real(bases[i][j][k].value(y_next));
    let psi_z_prev = Cplx::from_real(bases[i][j][k].value(z_prev));
    let psi_z_next = Cplx::from_real(bases[i][j][k].value(z_next));

    let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
        - psi[i][j][k] * 6.;

    // result / H_SQ
    result / (h2 * h2)
}

/// Convert an array of Psi to one of electron charge through space. Modifies in place
/// to avoid unecessary allocations.
pub fn charge_density_fm_psi(psi: &Arr3d, charge_density: &mut Arr3dReal, num_elecs: usize) {
    // Normalize <ψ|ψ>
    let mut psi_sq_size = 0.;
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                psi_sq_size += psi[i][j][k].abs_sq();
            }
        }
    }

    // Save computation on this constant factor.
    let c = -Q_ELEC * num_elecs as f64 / psi_sq_size;

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                charge_density[i][j][k] = psi[i][j][k].abs_sq() * c;
            }
        }
    }
}

/// Find the E that minimizes score, by narrowing it down. Note that if the relationship
/// between E and psi'' score isn't straightforward, this will converge on a local minimum.
pub fn find_E(sfcs: &mut Surfaces, E: &mut f64) {
    // todo: WHere to configure these mins and maxes
    let mut E_min = -2.;
    let mut E_max = 2.;
    let mut E_range_div2 = 2.;
    let vals_per_iter = 8;

    let num_iters = 10;

    for _ in 0..num_iters {
        let E_vals = util::linspace((E_min, E_max), vals_per_iter);
        let mut best_score = 100_000_000.;
        let mut best_E = 0.;

        for E_trial in E_vals {
            for i in 0..N {
                for j in 0..N {
                    for k in 0..N {
                        sfcs.psi_pp_calculated[i][j][k] =
                            find_ψ_pp_calc(&sfcs.psi, &sfcs.V, E_trial, i, j, k);
                    }
                }
            }

            let score = score_wf(sfcs);
            if score < best_score {
                best_score = score;
                best_E = E_trial;
                *E = E_trial;
            }
        }
        E_range_div2 /= vals_per_iter as f64; // todo: May need a wider range than this.
        E_min = best_E - E_range_div2;
        E_max = best_E + E_range_div2;
    }
}

/// A crude low pass
pub fn smooth_array(arr: &mut Arr3d, smoothing_amt: f64) {
    let orig = arr.clone();

    for i in 0..N {
        if i == 0 || i == N - 1 {
            continue;
        }
        for j in 0..N {
            if j == 0 || j == N - 1 {
                continue;
            }
            for k in 0..N {
                if k == 0 || k == N - 1 {
                    continue;
                }
                let neighbor_avg = (orig[i - 1][j][k]
                    + orig[i + 1][j][k]
                    + orig[i][j - 1][k]
                    + orig[i][j + 1][k]
                    + orig[i][j][k - 1]
                    + orig[i][j][k + 1])
                    / 6.;

                let diff_from_neighbors = neighbor_avg - arr[i][j][k];

                arr[i][j][k] += diff_from_neighbors * smoothing_amt;
            }
        }
    }
}

/// Represents important data, in describe 3D arrays.
/// We use Vecs, since these can be large, and we don't want
/// to put them on the stack. Although, they are fixed-size.
/// todo: Change name?
pub struct Surfaces {
    pub V: Arr3dReal,
    pub psi: Arr3d,
    pub psi_pp_calculated: Arr3d,
    pub psi_pp_measured: Arr3d,
    /// Aux surfaces are for misc visualizations
    pub aux1: Arr3d,
    pub aux2: Arr3dReal,
    /// Individual nudge amounts, per point of ψ. Real, since it's scaled by the diff
    /// between psi'' measured and calcualted, which is complex.
    pub nudge_amounts: Arr3dReal,
    // /// Used to revert values back after nudging.
    // pub psi_prev: Arr3d, // todo: Probably
    /// Electric charge at each point in space. Probably will be unused
    /// todo going forward, since this is *very* computationally intensive
    pub elec_charges: Vec<Arr3dReal>,
    /// todo: Experimental representation as a local analytic eq at each point.
    pub bases: Arr3dBasis,
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

        Self {
            V: data_real.clone(),
            psi: data.clone(),
            psi_pp_calculated: data.clone(),
            psi_pp_measured: data.clone(),
            aux1: data.clone(),
            aux2: data_real.clone(),
            nudge_amounts: default_nudges,
            // psi_prev: data.clone(),
            elec_charges: vec![data_real.clone()],
            bases: new_data_basis(N),
        }
    }
}
