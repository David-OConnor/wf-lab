use crate::{basis_wfs::SinExpBasisPt, complex_nums::Cplx, wf_ops::N};

use lin_alg2::f64::Vec3;

// type Arr3d = Vec<Vec<Vec<f64>>>;
pub type Arr3dReal = Vec<Vec<Vec<f64>>>;
pub type Arr3d = Vec<Vec<Vec<Cplx>>>;
pub type Arr3dVec = Vec<Vec<Vec<Vec3>>>;
pub type Arr3dBasis = Vec<Vec<Vec<SinExpBasisPt>>>;

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
pub fn polar_to_cart(ctr: (f64, f64), theta: f64, r: f64) -> (f64, f64) {
    let x = ctr.0 + theta.cos() * r;
    let y = ctr.1 + theta.cos() * r;

    (x, y)
}

/// Converts spherical coordinates to cartesian. θ is inclination (lat). φ is azimuth (lon).
/// θ is on a scale of 0 to τ/2. φ is on a scale of 0 to τ.
pub fn spherical_to_cart(ctr: Vec3, θ: f64, φ: f64, r: f64) -> Vec3 {
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

// todo: 3rd-order interpolation as well.

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