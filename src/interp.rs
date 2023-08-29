#![allow(unused)]

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

/// Create an order-2 polynomial based on 3 points. (1D: pts are (input, output).
/// `a` is the ^2 term, `b` is the linear term, `c` is the constant term.
/// This is a general mathematical function, and can be derived using a system of equations.
pub fn _create_polynomial_terms(
    pt0: (f64, f64),
    pt1: (f64, f64),
    pt2: (f64, f64),
) -> (f64, f64, f64) {
    let a_num = pt0.0 * (pt2.1 - pt1.1) + pt1.0 * (pt0.1 - pt2.1) + pt2.0 * (pt1.1 - pt0.1);

    let a_denom = (pt0.0 - pt1.0) * (pt0.0 - pt2.0) * (pt1.0 - pt2.0);

    let a = a_num / a_denom;

    let b = (pt1.1 - pt0.1) / (pt1.0 - pt0.0) - a * (pt0.0 + pt1.0);

    let c = pt0.1 - a * pt0.0.powi(2) - b * pt0.0;

    (a, b, c)
}

/// Create the quadratic term of an order-2 polynomial from 3 points. Useful for when we're
/// only looking for the second-derivative at a point; saves computation on linear and constant
/// terms which are discarded during differentiation.
/// todo: Complex?
pub fn create_quadratic_term(pt0: (f64, f64), pt1: (f64, f64), pt2: (f64, f64)) -> f64 {
    let a_num = pt0.0 * (pt2.1 - pt1.1) + pt1.0 * (pt0.1 - pt2.1) + pt2.0 * (pt1.1 - pt0.1);

    let a_denom = (pt0.0 - pt1.0) * (pt0.0 - pt2.0) * (pt1.0 - pt2.0);

    a_num / a_denom
}

/// todo: Experimental
/// Estimate the value of psi at a given point, given its value defined
/// at other points of arbitrary spacing and alignment.
fn _psi_at_pt(charges: Vec3, grid_vals: &[(Vec3, Cplx)]) -> Cplx {
    Cplx::new_zero()
}
