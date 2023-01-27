//! Code for interpolation. Includes 1D interpolation functions that are linear
//! and of 2nd and 3rd order polynomials, and extensions of these to 2D and 3D.

// todo: This possibly needs complex support. Perhaps as a wrapper since you can treat the real and
// todo complex parts separately.

use crate::complex_nums::Cplx;

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
    // Up for portion x and L for portion y chosen arbitrarily; doesn't matter for regular grids.

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
    linear_1d(posit_sample.z, (z_range.0, f_face), (z_range.1, a_face))
}

/// These points are (X, Y, function value)
// fn linear_2d(val: (f64, f64), pt_up_l: (f64, f64, f64), pt_dn_l: (f64, f64, f64), pt_up_r: (f64, f64, f64), pt_dn_r: (f64, f64, f64)) -> f64 {
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
    let t_edge = linear_1d(posit_sample.0, (x_range.0, v_up_l), (x_range.1, v_up_r));
    let b_edge = linear_1d(posit_sample.0, (x_range.0, v_dn_l), (x_range.1, v_dn_r));

    // Up for portion x and L for portion y chosen arbitrarily; doesn't matter for regular grids.
    linear_1d(posit_sample.1, (y_range.0, t_edge), (y_range.1, b_edge))
}

/// Compute the result of a Lagrange polynomial of order 3.
/// Algorithm created from the `P(x)` eq
/// [here](https://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html).
/// todo: Figure out how to just calculate the coefficients for a more
/// todo flexible approach. More eloquent, but tough to find info on compared
/// todo to this approach.
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

/// Utility function to linearly map an input value to an output
/// (Similar to `map_linear` function you use in other projects, but with different
/// organization of input parameters)
fn linear_1d(posit_sample: f64, pt0: (f64, f64), pt1: (f64, f64)) -> f64 {
    // todo: You may be able to optimize calls to this by having the ranges pre-store
    // todo the total range vals.
    let portion = (posit_sample - pt0.0) / (pt1.0 - pt0.0);

    portion * (pt1.1 - pt0.1) + pt0.1
}

// /// Utility function to linearly map an input value to an output
// pub fn map_linear(range_in: (f64, f64), range_out: (f64, f64), val: f64) -> f64 {
//     // todo: You may be able to optimize calls to this by having the ranges pre-store
//     // todo the total range vals.
//     let portion = (val - range_in.0) / (range_in.1 - range_in.0);
//
//     portion * (range_out.1 - range_out.0) + range_out.0
// }
