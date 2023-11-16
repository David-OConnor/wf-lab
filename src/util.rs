use crate::grid_setup::{Arr3d, Arr3dReal, Arr3dVec};

use lin_alg2::f64::Vec3;

/// Create a set of values in a given range, with a given number of values.
/// Similar to `numpy.linspace`.
/// The result terminates one step before the end of the range.
pub fn linspace(range: (f64, f64), num_vals: usize) -> Vec<f64> {
    let step = (range.1 - range.0) / num_vals as f64;

    let mut result = Vec::new();

    let mut val = range.0;
    for _ in 0..num_vals {
        result.push(val);
        val += step;
    }

    result
}

/// theta and r are anchored to the centern point. The center point and returned value
/// are in global, cartesian coords.
pub fn _polar_to_cart(ctr: (f64, f64), theta: f64, r: f64) -> (f64, f64) {
    let x = ctr.0 + theta.cos() * r;
    let y = ctr.1 + theta.cos() * r;

    (x, y)
}

/// Converts spherical coordinates to cartesian. θ is inclination (lat). φ is azimuth (lon).
/// θ is on a scale of 0 to τ/2. φ is on a scale of 0 to τ.
pub fn _spherical_to_cart(ctr: Vec3, θ: f64, φ: f64, r: f64) -> Vec3 {
    let x = ctr.x + r * θ.sin() * φ.cos();
    let y = ctr.y + r * θ.sin() * φ.sin();
    let z = ctr.z + r * θ.cos();

    Vec3::new(x, y, z)
}

//
// /// Quadratic interpolation in 3D, using spline interpolation.
// /// Note: Assumes a cubic set of values. If we make it non-cubic in the future, range needs
// /// to include a value for each axis.
// pub fn interpolate_spline3pt(surface: Arr3d, posit: Vec3, sfc_range: (f64, f64)) -> f64 {
//     // let vals_1d = linspace((grid_min, grid_max), N);
//
//
// }

// langrange, as an alternative to spline:

// You can use Lagrange polynomial interpolation: https://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html.
// So, let's say you have (1, 5), (2, 25), (3, 7) as your three points. You do 5((x - 2)(x - 3))/((1 - 2)(1 - 3)) + 25((x - 1)(x - 3))/((2 - 1)(2 - 3)) + 7((x - 1)(x - 2))/((3 - 1)(3 - 2)).
// Note the pattern. Each main term has one of the y values multiplied by a fraction.
// Then you have a numerator with (x - [insert one of the x values, unless it's the one that corresponds to the y value you're multiplying in front]).
// Then, you have a denominator with ([x value corresponding to the y value in front] - [some other x value]).
// That simplifies to f(x) = -19x² + 77x - 53.
// f(1) = -19(1)² + 77(1) - 53 = -19 + 77 - 53 = 5
// f(2) = -19(2)² + 77(2) - 53 = -76 + 154 - 53 = 25
// f(3) = -19(3)² + 77(3) - 53 = -171 + 231 - 53 = 7

// 2: https://www.wolframalpha.com/input?i=InterpolatingPolynomial%5B%7B%7Bx1%2C+y1%7D%2C+%7Bx2%2C+y2%7D%7D%2C+x%5D
// 3: https://www.wolframalpha.com/input?i=InterpolatingPolynomial%5B%7B%7Bx1%2C+y1%7D%2C+%7Bx2%2C+y2%7D%2C+%7Bx3%2C+y3%7D%7D%2C+x%5D
//

/// Generate a laguerre polynomial for a given value. Used in the radial component of Hydrogen basis functions.
/// https://www.cfm.brown.edu/people/dobrush/am34/Mathematica/ch7/laguerre.html
///
/// This has info on General form: https://planetmath.org/laguerrepolynomial
///
/// Wikipedia has info on generating arbitrary ones:
/// https://en.wikipedia.org/wiki/Laguerre_polynomials
pub(crate) fn _make_laguerre(n: u16, alhpa: f64) -> impl Fn(f64) -> f64 {
    // todo: For now, we've just hard-coded some values for low n.

    // todo: You may need the generalized Laguerre polynomial; QC this.

    // todo: It may actually be a generalized Lagureer run on 2r/(n A_0)

    fn result(x: f64) -> f64 {
        x + 1.
    }

    result

    // match n {
    //     0 => 1.,
    //     1 => 1. - x,
    //     2 => 1. / 2. * (x.powi(2) - 4. * x + 2),
    //     3 => 1. / 6. * (-x.powi(3) + 9. * x.powi(2) - 18. * x + 6.),
    //     4 => 1. / factorial(4) * (x.powi(4) - 16. * x.powi(3) + 72. * x.powi(2) - 96. * x + 24.),
    //     5 => {
    //         1. / factorial(5)
    //             * (-x.powi(5) * 25. * x.powi(4) - 200. * x.powi(3) + 600. * x.powi(2) - 600. * x
    //                 + 120.)
    //     }
    //     6 => {
    //         1. / factorial(6)
    //             * (x.powi(6) - 36. * x.powi(5) + 450. * x.powi(4) - 2_400. * x.powi(3)
    //                 + 5_400. * x.powi(2)
    //                 - 4_320 * x
    //                 + 720.)
    //     }
    //     _ => unimplemented!(),
    // }
}

/// Generate a Legendre polynomial for a given value. Used in the angular component of Hydrogen basis functions.
pub(crate) fn _legendre(n: u16, x: f64) -> f64 {
    // todo: For now, we've just hard-coded some values for low n.

    // todo: You may need the generalized Laguerre polynomial; QC this.

    match n {
        0 => 1.,
        1 => 1. - x,
        2 => 1. / 2. * (x.powi(2) - 4. * x + 2.),
        3 => 1. / 6. * (-x.powi(3) + 9. * x.powi(2) - 18. * x + 6.),
        4 => {
            1. / factorial(4) as f64
                * (x.powi(4) - 16. * x.powi(3) + 72. * x.powi(2) - 96. * x + 24.)
        }
        5 => {
            1. / factorial(5) as f64
                * (-x.powi(5) * 25. * x.powi(4) - 200. * x.powi(3) + 600. * x.powi(2) - 600. * x
                    + 120.)
        }
        6 => {
            1. / factorial(6) as f64
                * (x.powi(6) - 36. * x.powi(5) + 450. * x.powi(4) - 2_400. * x.powi(3)
                    + 5_400. * x.powi(2)
                    - 4_320. * x
                    + 720.)
        }
        _ => unimplemented!(),
    }
}

// todo: If you use a continuous range, use a struct with parameter fields
// todo instead of an enum that contains discrete values. This is your
// todo likely approach.

// todo: Move to util

/// Compute factorial using a LUT.
pub(crate) fn factorial(val: u16) -> u64 {
    match val {
        0 => 1,
        1 => 1,
        2 => 2,
        3 => 6,
        4 => 24,
        5 => 120,
        6 => 720,
        7 => 5040,
        8 => 40_320,
        9 => 362_880,
        10 => 3_628_800,
        11 => 39_916_800,
        12 => 479_001_600,
        _ => unimplemented!(),
    }
}

/// Calculate ψ* ψ
pub(crate) fn norm_sq(dest: &mut Arr3dReal, source: &Arr3d, grid_n: usize) {
    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                dest[i][j][k] = source[i][j][k].abs_sq();
            }
        }
    }
}

/// Normalize a wave function so that <ψ|ψ> = 1.
/// Returns the norm value for use in normalizing basis fns in psi''_measured calculation.
///
/// Note that due to phase symmetry, there are many ways to balance the normalization of the real
/// vice imaginary parts. Our implmentation (dividing both real and imag parts by norm square)
/// is one way.
pub(crate) fn normalize_wf(arr: &mut Arr3d, norm: f64) {
    const EPS: f64 = 0.0000001;
    if norm.abs() < EPS {
        return;
    }

    // todo: Why are we dividing by norm.sqrt() instead of norm?
    // todo: Seems to be required, but try to understand it.
    let norm_sqrt = norm.sqrt();

    let grid_n = arr.len();

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                // Note: Check the div impl for details.
                arr[i][j][k] = arr[i][j][k] / norm_sqrt;
            }
        }
    }
    // norm_sqrt
}

/// Flatten 3D data, prior passing to a GPU kernel.
pub(crate) fn flatten_arr(vals_3d: &Arr3dVec, grid_n: usize) -> Vec<Vec3> {
    let mut result = Vec::new();

    for i_sample in 0..grid_n {
        for j_sample in 0..grid_n {
            for k_sample in 0..grid_n {
                result.push(vals_3d[i_sample][j_sample][k_sample]);
            }
        }
    }

    result
}

/// Unflatted 3D data, after getting results from a GPU kernel.
pub(crate) fn unflatten_arr(result: &mut Arr3dReal, vals_flat: &[f64], grid_n: usize) {
    let grid_n_sq = grid_n.pow(2);

    for i in 0..grid_n {
        for j in 0..grid_n {
            for k in 0..grid_n {
                let i_flat = i * grid_n_sq + j * grid_n + k;
                result[i][j][k] = vals_flat[i_flat];
            }
        }
    }
}
//
// pub(crate) fn flatten_arr_vec(vals_3d: &Arr3dVec, grid_n: usize) -> Vec<Vec3> {
//     let mut result = Vec::new();
//
//     // Flatten sample positions, prior to passing to the kernel.
//     for i_sample in 0..grid_n {
//         for j_sample in 0..grid_n {
//             for k_sample in 0..grid_n {
//                 result.push(vals_3d[i_sample][j_sample][k_sample]);
//             }
//         }
//     }
//
//     result
// }
//
// pub(crate) fn unflatten_arr_vec(result: &mut Arr3dVec, vals_flat: &[Vec3], grid_n: usize) {
//     let grid_n_sq = grid_n.pow(2);
//
//     for i in 0..grid_n {
//         for j in 0..grid_n {
//             for k in 0..grid_n {
//                 let i_flat = i * grid_n_sq + j * grid_n + k;
//                 result[i][j][k] = vals_flat[i_flat];
//             }
//         }
//     }
// }
