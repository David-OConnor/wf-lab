use crate::complex_nums::Cplx;

// type Arr3d = Vec<Vec<Vec<f64>>>;
pub type Arr3dReal = Vec<Vec<Vec<f64>>>;
pub type Arr3d = Vec<Vec<Vec<Cplx>>>;

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
