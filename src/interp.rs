//! Code for interpolation. Includes 1D interpolation functions that are linear
//! and of 2nd and 3rd order polynomials, and extensions of these to 2D and 3D.

/// These points are (X, Y, function value)
fn linear_2d(val: (f64, f64), pt_up_l: (f64, f64, f64), pt_dn_l: (f64, f64, f64), pt_up_r: (f64, f64, f64), pt_dn_r: (f64, f64, f64)) -> f64 {
    // Up for portion x and L for portion y chosen arbitrarily; doesn't matter for regular grids.
    // let portion_x = (val.0 - pt_up_l.0) / (pt_up_r.0 - pt_up_l.0);
    // todo: portion code is duped in `linear_1d` and below

    // We interpolate on the X axis first; could do Y instead.
    let t_edge = linear_1d(val.0, (pt_up_l.0, pt_up_l.2), (pt_up_r.0, pt_up_r.2));
    let b_edge = linear_1d(val.0, (pt_dn_l.0, pt_dn_l.2), (pt_dn_r.0, pt_dn_r.2));
    // let l_edge = linear_1d(val.1, (pt_up_l.1, pt_up_l.2), (pt_dn_l.1, pt_dn_l.2));
    // let r_edge = linear_1d(val.1, (pt_up_r.1, pt_up_r.2), (pt_dn_r.1, pt_dn_r.2));

    // Up for portion x and L for portion y chosen arbitrarily; doesn't matter for regular grids.
    linear_1d(val.1, (pt_up_l.1, t_edge), (pt_dn_l.1, b_edge))
}

/// Compute the result of a Lagrange polynomial of order 3.
/// Algorithm created from the `P(x)` eq
/// [here](https://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html).
/// todo: Figure out how to just calculate the coefficients for a more
/// todo flexible approach. More eloquent, but tough to find info on compared
/// todo to this approach.
fn langrange_o3_1d(val: f64, pt0: (f64, f64), pt1: (f64, f64), pt2: (f64, f64)) -> f64 {
    let mut result = 0.;

    let x = [pt0.0, pt1.0, pt2.0];
    let y = [pt0.1, pt1.1, pt2.1];

    for j in 0..3 {
        let mut c = 1.;
        for i in 0..3 {
            if j == i {
                continue;
            }
            c *= (val - x[i]) / (x[j] - x[i]);
        }
        result += y[j] * c;
    }

    result
}

/// Utility function to linearly map an input value to an output
/// (Similar to `map_linear` function you use in other projects, but with different
/// organization of input parameters)
pub fn linear_1d(val: f64, pt0: (f64, f64), pt1: (f64, f64)) -> f64 {
    // todo: You may be able to optimize calls to this by having the ranges pre-store
    // todo the total range vals.
    let portion = (val - pt0.0) / (pt1.0 - pt0.0);

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