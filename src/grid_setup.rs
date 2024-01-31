//! This module contains code for setting up grids and sample points.

use lin_alg2::f64::Vec3;

use crate::{complex_nums::Cplx, iter_arr, types::DerivativesSingle, util};

// type Arr3d = Vec<Vec<Vec<f64>>>;
pub type Arr3dReal = Vec<Vec<Vec<f64>>>;
pub type Arr3dDeriv = Vec<Vec<Vec<DerivativesSingle>>>;

pub type Arr3d = Vec<Vec<Vec<Cplx>>>;
pub type Arr4d = Vec<Vec<Vec<Vec<Cplx>>>>;
// pub type Arr3dBasis = Vec<Vec<Vec<SinExpBasisPt>>>;

pub type Arr3dVec = Vec<Vec<Vec<Vec3>>>;

/// Find sample points for evaluating wave functions, based on nuclei positions.
/// Attempts to choose a minimal set of points that can accuruately be used
/// to assess trial wave functions, without introducing numerical instabilities.
pub(crate) fn find_sample_points(nuclei: &[(Vec3, f64)]) -> Vec<Vec3> {
    // todo: Mass-dependent?

    const X_PLUS: Vec3 = Vec3::new(1., 0., 0.);
    const Y_PLUS: Vec3 = Vec3::new(0., 1., 0.);
    const Z_PLUS: Vec3 = Vec3::new(0., 0., 1.);

    let sample_dists = [0.5, 1., 2., 3., 4., 5., 6., 7.];

    // todo: This is rough. Figure out how to do this.
    let mut result = Vec::new();

    for (nuc_posit, _charge) in nuclei {
        for dist in sample_dists {
            result.push(*nuc_posit + X_PLUS * dist);
            result.push(*nuc_posit + X_PLUS * dist);
            result.push(*nuc_posit + X_PLUS * dist);

            result.push(*nuc_posit + Y_PLUS * dist);
            result.push(*nuc_posit + Y_PLUS * dist);
            result.push(*nuc_posit + Y_PLUS * dist);

            result.push(*nuc_posit + Z_PLUS * dist);
            result.push(*nuc_posit + Z_PLUS * dist);
            result.push(*nuc_posit + Z_PLUS * dist);
        }
    }

    result
}

/// Set up a grid that  smartly encompasses the charges, letting the WF go to 0
/// towards the edges
pub(crate) fn choose_grid_limits(charges_fixed: &[(Vec3, f64)]) -> (f64, f64) {
    let mut max_abs_val = 0.;
    for (posit, _) in charges_fixed {
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

    const RANGE_PAD: f64 = 5.8;
    // const RANGE_PAD: f64 = 14.;

    let grid_max = max_abs_val + RANGE_PAD;

    // todo: temp
    let grid_max = 10.0;

    let grid_min = -grid_max;

    (grid_min, grid_max)
}

/// Update our grid positions. Run this when we change grid bounds, resolution, or spacing.
pub fn update_grid_posits(
    grid_posits: &mut Arr3dVec,
    grid_range: (f64, f64),
    spacing_factor: f64,
    n: usize,
) {
    let grid_lin = util::linspace((grid_range.0, grid_range.1), n);

    // Set up a grid with values that increase in distance the farther we are from the center.
    let mut grid_1d = vec![0.; n];

    for i in 0..n {
        let mut val = grid_lin[i].abs().powf(spacing_factor);
        if grid_lin[i] < 0. {
            val *= -1.; // square the magnitude only.
        }
        grid_1d[i] = val;
    }

    for (i, x) in grid_1d.iter().enumerate() {
        for (j, y) in grid_1d.iter().enumerate() {
            for (k, z) in grid_1d.iter().enumerate() {
                grid_posits[i][j][k] = Vec3::new(*x, *y, *z);
            }
        }
    }
}

/// Make a new 3D grid, as a nested Vec
pub fn new_data(n: usize) -> Arr3d {
    let mut z = Vec::new();
    z.resize(n, Cplx::new_zero());

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

/// Make a new 3D grid of position vectors, as a nested Vec
pub fn new_data_vec(n: usize) -> Arr3dVec {
    let mut z = Vec::new();
    z.resize(n, Vec3::new_zero());

    let mut y = Vec::new();
    y.resize(n, z);

    let mut x = Vec::new();
    x.resize(n, y);

    x
}

/// Make a new 3D grid of position vectors, as a nested Vec
pub fn new_data_deriv(n: usize) -> Arr3dDeriv {
    let mut z = Vec::new();
    z.resize(n, DerivativesSingle::default());

    let mut y = Vec::new();
    y.resize(n, z);

    let mut x = Vec::new();
    x.resize(n, y);

    x
}

pub fn copy_array(dest: &mut Arr3d, source: &Arr3d, grid_n: usize) {
    for (i, j, k) in iter_arr!(grid_n) {
        dest[i][j][k] = source[i][j][k];
    }
}
