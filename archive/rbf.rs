//! From https://github.com/mthh/rbf_interp

use rulinalg::{matrix::Matrix, vector::Vector};

use lin_alg2::f64::Vec3;

#[derive(Debug, Clone)]
pub struct Rbf {
    pub obs_points: Vec<Vec3>,
    pub fn_vals: Vec<f64>,
    pub weights: Vector<f64>,
    pub distance_fn: fn(f64, f64) -> f64,
    pub epsilon: f64,
}

impl Rbf {
    pub fn new(
        obs_points: Vec<Vec3>,
        fn_vals: Vec<f64>,
        dist_fn_name: &str, // todo: Use an enum.
        epsilon: Option<f64>,
    ) -> Self {
        let distance_fn = match dist_fn_name {
            "linear" => distance_linear,
            "cubic" => distance_cubic,
            "thin_plate" => distance_thin_plate,
            "quintic" => distance_quintic,
            "gaussian" => distance_gaussian,
            "multiquadratic" => distance_multiquadratic,
            "inverse_multiquadratic" => distance_inverse_multiquadratic,
            &_ => panic!("Invalid function name!"),
        };

        let nb_pts = obs_points.len();

        let mut mat = vec![0.; nb_pts * nb_pts];
        for j in 0..nb_pts {
            for i in 0..nb_pts {
                mat[j * nb_pts + i] = _norm(&obs_points[i], &obs_points[j]);
            }
        }

        let eps = if epsilon.is_some() {
            epsilon.unwrap()
        } else {
            let _nb = nb_pts as f64;
            sum_all(&mat) / (_nb.powi(2) - _nb)
        };

        for ix in 0..(nb_pts * nb_pts) {
            mat[ix] = distance_fn(mat[ix], eps);
        }

        let mut values: Vec<f64> = Vec::with_capacity(nb_pts);
        for i in 0..nb_pts {
            values.push(fn_vals[i]);
        }

        let mat = Matrix::new(nb_pts, nb_pts, mat);
        let vec = Vector::new(values);

        let weights = mat.solve(vec).unwrap();

        Self {
            obs_points,
            fn_vals,
            distance_fn,
            epsilon: eps,
            weights,
        }
    }

    pub fn interp_point(&self, pt: Vec3) -> f64 {
        let mut distances: Vec<f64> = Vec::with_capacity(self.obs_points.len());

        for point in &self.obs_points {
            let a = _norm(&pt, point);
            distances.push((self.distance_fn)(a, self.epsilon));
        }

        let dist = Vector::new(distances);
        let r = &dist.elemul(&self.weights);
        r.sum()
    }
}

fn sum_all(mat: &Vec<f64>) -> f64 {
    let mut s = 0.;
    for &v in mat {
        s = s + v;
    }
    s
}

fn _norm(pa: &Vec3, pb: &Vec3) -> f64 {
    ((pa.x - pb.x).powi(2) + (pa.y - pb.y).powi(2) + (pa.z - pb.z).powi(2)).sqrt()
}

fn distance_linear(r: f64, _eps: f64) -> f64 {
    r
}

fn distance_cubic(r: f64, _eps: f64) -> f64 {
    r.powi(3)
}

fn distance_quintic(r: f64, _eps: f64) -> f64 {
    r.powi(5)
}

fn distance_thin_plate(r: f64, _eps: f64) -> f64 {
    if r == 0. {
        0.
    } else {
        r.powi(2) * r.ln()
    }
}

fn distance_gaussian(r: f64, eps: f64) -> f64 {
    1. / ((r / eps).powi(2) + 1.).exp()
}

#[inline(always)]
fn distance_inverse_multiquadratic(r: f64, eps: f64) -> f64 {
    1. / ((r / eps).powi(2) + 1.).sqrt()
}

#[inline(always)]
fn distance_multiquadratic(r: f64, eps: f64) -> f64 {
    ((r / eps).powi(2) + 1.).sqrt()
}
