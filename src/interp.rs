//! Code for interpolation. Includes 1D interpolation functions that are linear
//! and of 2nd and 3rd order polynomials, and extensions of these to 2D and 3D.

// todo: This possibly needs complex support. Perhaps as a wrapper since you can treat the real and
// todo complex parts separately.

use crate::{basis_wfs::Basis, complex_nums::Cplx};

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
pub fn create_polynomial_terms(
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
fn _psi_at_pt(charges: Vec3, grid_vals: &[(Vec3, Cplx)]) -> Cplx {
    Cplx::new_zero()
}

//
// /// Set up Radial Basis Function (RBF) interpolation, with spherical grids centered around
// /// nuclei, and a higher concentration of shells closer to nuclei.
// pub fn setup_rbf_interp(charge_posits: &[Vec3], bases: &[Basis]) -> rbf::Rbf {
//     // Determine how to set up our sample points
//     // Should use roughly half the number of lats as lons.
//     const N_LATS: usize = 10;
//     const N_LONS: usize = N_LATS * 2;
//
//     const N_DISTS: usize = 14;
//
//     const ANGLE_BW_LATS: f64 = (TAU / 2.) / N_LATS as f64;
//     const ANGLE_BW_LONS: f64 = TAU / N_LONS as f64;
//
//     // A smaller value here will allow for more shells close to the nucleii.
//     const DIST_CONST: f64 = 0.03; // c^n_dists = max_dist
//
//     let n_samples = N_LATS * N_LONS * N_DISTS * charge_posits.len();
//
//     // todo: Dist falloff, since we use more dists closer to the nuclei?
//
//     // `xobs` is a an array of X, Y pts. Rust equiv type might be
//     // &[Vec3]
//     let mut xobs = Vec::new();
//     // for _ in 0..n_samples {
//     //     xobs.push(Vec3::new_zero());
//     // }
//
//     let mut i = 0;
//
//     // For latitudes, don't include the poles, since they create degenerate points.
//     for lat_i in 1..N_LATS {
//         // thata is latitudes; phi longitudes.
//         let theta = lat_i as f64 * ANGLE_BW_LATS;
//         // We don't use dist = 0.
//         for lon_i in 0..N_LONS {
//             let phi = lon_i as f64 * ANGLE_BW_LONS; // todo which is which?
//
//             for dist_i in 1..N_DISTS + 1 {
//                 // r = exp(DIST_DECAY_EXP * dist_i) * DIST_CONST
//                 let r = (dist_i as f64).powi(2) * DIST_CONST;
//
//                 if lat_i == 1 && lon_i == 0 {
//                     println!("R: {:?}", r);
//                 }
//
//                 for (_charge_i, charge_posit) in charge_posits.into_iter().enumerate() {
//                     // xobs[i + charge_i] = util::spherical_to_cart(*charge_posit, theta, phi, r);
//                     xobs.push(util::spherical_to_cart(*charge_posit, theta, phi, r));
//
//                     // Just once per distance, add a pole at the top and bottom (lat = 0 and TAU/2
//                     if lat_i == 1 && lon_i == 0 {
//                         xobs.push(util::spherical_to_cart(*charge_posit, 0., 0., r)); // bottom
//                         xobs.push(util::spherical_to_cart(*charge_posit, TAU / 2., 0., r));
//                         // top
//                     }
//                 }
//
//                 i += charge_posits.len();
//             }
//         }
//     }
//
//     // z_slice = ctr[2]
//
//     // `yobs` is the function values at each of the sample points.
//
//     // todo: Cplx?
//     let mut yobs = Vec::new();
//     for _ in 0..n_samples {
//         yobs.push(0.);
//     }
//
//     // Iterate over our random sample of points
//     for (i, grid_pt) in xobs.iter().enumerate() {
//         for basis in bases {
//             // todo: discarding Im part for now
//             yobs[i] += (basis.value(*grid_pt) * basis.weight()).real;
//         }
//     }
//     //
//     // println!("Xobs: ");
//     // for obs in &xobs {
//     //     println!("x:{:.2} y:{:.2} z:{:.2}", obs.x, obs.y, obs.z);
//     // }
//
//     // println!("\n\nY obs: {:?}", yobs);
//
//     // From an intial test: Linear is OK. So is cubic. thin_plate is best
//     let rbf = Rbf::new(xobs, yobs, "linear", None);
//
//     rbf
//
//     //
//     //
//     // xgrid = np.mgrid[grid_min:grid_max:50j, grid_min:grid_max:50j]
//     //
//     // xflat = xgrid.reshape(2, -1).T
//     //
//     //
//     // yflat = RBFInterpolator(xobs, yobs, kernel='cubic')(xflat)
//     //
//     // ygrid = yflat.reshape(50, 50)
// }
