//! Code for interpolation. Includes 1D interpolation functions that are linear
//! and of 2nd and 3rd order polynomials, and extensions of these to 2D and 3D.

// todo: This possibly needs complex support. Perhaps as a wrapper since you can treat the real and
// todo complex parts separately.

use std::f64::consts::TAU;

use crate::{complex_nums::Cplx, util};

use lin_alg2::f64::Vec3;

/// Complex wrapper for 3d linear interpolation.
pub fn linear_3d_cplx(
    posit_sample: Vec3,
    x_range: (f64, f64),
    y_range: (f64, f64),
    z_range: (f64, f64),
    v_up_l_f: Cplx,
    v_dn_l_f: Cplx,
    v_up_r_f: Cplx,
    v_dn_r_f: Cplx,
    v_up_l_a: Cplx,
    v_dn_l_a: Cplx,
    v_up_r_a: Cplx,
    v_dn_r_a: Cplx,
) -> Cplx {
    let real = linear_3d(
        posit_sample,
        x_range,
        y_range,
        z_range,
        v_up_l_f.real,
        v_dn_l_f.real,
        v_up_r_f.real,
        v_dn_r_f.real,
        v_up_l_a.real,
        v_dn_l_a.real,
        v_up_r_a.real,
        v_dn_r_a.real,
    );

    let im = linear_3d(
        posit_sample,
        x_range,
        y_range,
        z_range,
        v_up_l_f.im,
        v_dn_l_f.im,
        v_up_r_f.im,
        v_dn_r_f.im,
        v_up_l_a.im,
        v_dn_l_a.im,
        v_up_r_a.im,
        v_dn_r_a.im,
    );

    Cplx { real, im }
}

/// Naming convention: Up means positive Y. Right means positive X. Forward means positive Z.
/// This must be adhered to when positioning the value paremeters.
pub fn linear_3d(
    posit_sample: Vec3,
    x_range: (f64, f64),
    y_range: (f64, f64),
    z_range: (f64, f64),
    v_up_l_f: f64,
    v_dn_l_f: f64,
    v_up_r_f: f64,
    v_dn_r_f: f64,
    v_up_l_a: f64,
    v_dn_l_a: f64,
    v_up_r_a: f64,
    v_dn_r_a: f64,
) -> f64 {
    let f_face = linear_2d(
        (posit_sample.x, posit_sample.y),
        x_range,
        y_range,
        v_up_l_f,
        v_dn_l_f,
        v_up_r_f,
        v_dn_r_f,
    );
    let a_face = linear_2d(
        (posit_sample.x, posit_sample.y),
        x_range,
        y_range,
        v_up_l_a,
        v_dn_l_a,
        v_up_r_a,
        v_dn_r_a,
    );

    // Up for portion x and L for portion y chosen arbitrarily; doesn't matter for regular grids.
    linear_1d(posit_sample.z, z_range, a_face, f_face)
}

/// Naming convention: Up means positive Y. Right means positive X.
fn linear_2d(
    posit_sample: (f64, f64),
    x_range: (f64, f64),
    y_range: (f64, f64),
    v_up_l: f64,
    v_dn_l: f64,
    v_up_r: f64,
    v_dn_r: f64,
) -> f64 {
    // Up for portion x and L for portion y chosen arbitrarily; doesn't matter for regular grids.
    // let portion_x = (val.0 - x_range.0) / (x_range.1 - x_range.0);
    // todo: portion code is duped in `linear_1d` and below
    // We interpolate on the X axis first; could do Y instead.
    let t_edge = linear_1d(posit_sample.0, x_range, v_up_l, v_up_r);
    let b_edge = linear_1d(posit_sample.0, x_range, v_dn_l, v_dn_r);

    // Up for portion x and L for portion y chosen arbitrarily; doesn't matter for regular grids.
    linear_1d(posit_sample.1, y_range, b_edge, t_edge)
}

pub fn linear_1d_cplx(posit_sample: f64, range: (f64, f64), val_l: Cplx, val_r: Cplx) -> Cplx {
    let real = linear_1d(posit_sample, range, val_l.real, val_r.real);
    let im = linear_1d(posit_sample, range, val_l.im, val_r.im);

    Cplx { real, im }
}

/// Utility function to linearly map an input value to an output
/// (Similar to `map_linear` function you use in other projects, but with different
/// organization of input parameters)
pub fn linear_1d(posit_sample: f64, range: (f64, f64), val_l: f64, val_r: f64) -> f64 {
    // todo: You may be able to optimize calls to this by having the ranges pre-store
    // todo the total range vals.
    let portion = (posit_sample - range.0) / (range.1 - range.0);

    portion * (val_r - val_l) + val_l
}

/// Compute the result of a Lagrange polynomial of order 3.
/// Algorithm created from the `P(x)` eq
/// [here](https://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html).
/// todo: Figure out how to just calculate the coefficients for a more
/// todo flexible approach. More eloquent, but tough to find info on compared
/// todo to this approach.
/// todo: For coefficients, maybe try here:
/// https://math.stackexchange.com/questions/680646/get-polynomial-function-from-3-points
/// todo: Not too tough to work it out yourself too...
fn langrange_o3_1d(posit_sample: f64, pt0: (f64, f64), pt1: (f64, f64), pt2: (f64, f64)) -> f64 {
    let mut result = 0.;

    let x = [pt0.0, pt1.0, pt2.0];
    let y = [pt0.1, pt1.1, pt2.1];

    for j in 0..3 {
        let mut c = 1.;
        for i in 0..3 {
            if j == i {
                continue;
            }
            c *= (posit_sample - x[i]) / (x[j] - x[i]);
        }
        result += y[j] * c;
    }

    result
}

// /// Utility function to linearly map an input value to an output
// pub fn map_linear(range_in: (f64, f64), range_out: (f64, f64), val: f64) -> f64 {
//     // todo: You may be able to optimize calls to this by having the ranges pre-store
//     // todo the total range vals.
//     let portion = (val - range_in.0) / (range_in.1 - range_in.0);
//
//     portion * (range_out.1 - range_out.0) + range_out.0
// }

/// todo: Experimental
/// Estimate the value of psi at a given point, given its value defined
/// at other points of arbitrary spacing and alignment.
fn psi_at_pt(charges: Vec3, grid_vals: &[(Vec3, Cplx)]) -> Cplx {
    Cplx::new_zero()
}

fn test_rbf(charges: &[Vec3], grid_rng: (f64, f64)) {
    // Determine how to set up our sample points
    const N_LATS: usize = 10;
    const N_LONS: usize = 10;

    const N_DISTS: usize = 8;

    const ANGLE_BW_LATS: f64 = (TAU / 2.) / N_RADIALS;
    const ANGLE_BW_LONS: f64 = TAU / N_RADIALS;

    const DIST_CONST: f64 = 0.05; // c^n_dists = max_dist

    const N_SAMPLES: usize = N_LATS * N_LONS * N_DISTS * charges.len();

    // todo: Dist falloff, since we use more dists closer to the nuclei?

    // `xobs` is a an array of X, Y pts. Rust equiv type might be
    // &[Vec3]
    let mut xobs = [Vec3; N_SAMPLES];

    let mut i = 0;

    for lat_i in 0..N_LATS {
        let theta = lat_i * ANGLE_BW_LATS; // todo which is which?
                                           // We don't use dist = 0.

        for lon_i in 0..N_LONS {
            let phi = lon_i * ANGLE_BW_LONS; // todo which is which?

            for dist_i in 1..N_DISTS + 1 {
                // Don't use ring @ r=0.
                // r = exp(DIST_DECAY_EXP * dist_i) * DIST_CONST
                let r = dist_i.powi(2) * DIST_CONST;

                for (charge_i, charge_posit) in charges.into_iter().enumerate() {
                    xobs[i + charge_i] = util::spherical_to_cart(*charge_posit, theta, phi, r);
                }

                i += charges.len();
            }
        }
    }

    // z_slice = ctr[2]

    // `yobs` is the function values at each of the sample points.
    // eg `&[Cplx]`
    let mut yobs = [f64; N_SAMPLES]; // todo: Cplx?

    // Iterate over our random sample of points
    for (i, grid_pt) in xobs.iter().enumerate() {
        yobs[i] = h100(nuc1, Vec3(grid_pt[0], grid_pt[1], 0.))
            + h100(nuc2, Vec3(grid_pt[0], grid_pt[1], 0.));
    }

    //
    //
    // xgrid = np.mgrid[grid_min:grid_max:50j, grid_min:grid_max:50j]
    //
    // xflat = xgrid.reshape(2, -1).T
    //
    //
    // yflat = RBFInterpolator(xobs, yobs, kernel='cubic')(xflat)
    //
    // ygrid = yflat.reshape(50, 50)
}
