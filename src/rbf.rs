//! From https://github.com/mthh/rbf_interp

use rulinalg::{matrix::Matrix, vector::Vector};

use lin_alg2::f64::Vec3;

#[derive(Debug, Clone)]
pub struct Rbf<'a> {
    obs_points: &'a [Vec3],
    fn_vals: &'a [f64],
    weights: Vector<f64>,
    distance_fn: fn(f64, f64) -> f64,
    epsilon: f64,
}

impl<'a> Rbf<'a> {
    pub fn new(
        obs_points: &'a [Vec3],
        fn_vals: &'a [f64],
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

        // for j in 0..nb_pts {
        //     for i in 0..nb_pts {
        //         mat[j * nb_pts + i] = distance_func(mat[j * nb_pts + i], eps);
        //     }
        // }

        for ix in 0..(nb_pts * nb_pts) {
            mat[ix] = distance_fn(mat[ix], eps);
        }

        let mut values: Vec<f64> = Vec::with_capacity(nb_pts);
        for i in 0..nb_pts {
            values.push(fn_vals[i]);
        }

        let mat = Matrix::new(nb_pts, nb_pts, mat);
        let vec = Vector::new(values);
        // let weights = mat.solve(vec).unwrap().into_iter().collect::<Vec<Float>>();
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

        for point in self.obs_points {
            let a = _norm(&pt, point);
            distances.push((self.distance_fn)(a, self.epsilon));
        }

        let dist = Vector::new(distances);
        let r = &dist.elemul(&self.weights);
        r.sum()
    }
}

/// Function allowing to compute the choosen radial basis interpolation
/// from scattered data on a grid defined by its Bbox and by its resolution
/// on the x and y axis.
pub fn rbf_interpolation(
    reso_x: u32,
    reso_y: u32,
    reso_z: u32,
    bbox: &Bbox,
    obs_points: &[Vec3],
    fn_vals: &[f64],
    func_name: &str,
    epsilon: Option<f64>,
) -> Vec<(f64, f64, f64, f64)> {
    let rx = reso_x as f64;
    let ry = reso_y as f64;
    let rz = reso_z as f64;

    let x_step = (bbox.max_x - bbox.min_x) / rx;
    let y_step = (bbox.max_y - bbox.min_y) / ry;
    let z_step = (bbox.max_z - bbox.min_z) / rz;

    // `plots` is (x, y, z, fn value).
    let mut plots = Vec::with_capacity((reso_x * reso_y * reso_z) as usize);

    let rbf = Rbf::new(obs_points, fn_vals, func_name, epsilon);

    for i in 0..reso_x {
        for j in 0..reso_y {
            for k in 0..reso_z {
                let x = bbox.min_x + x_step * i as f64;
                let y = bbox.min_y + y_step * j as f64;
                let z = bbox.min_z + z_step * k as f64;

                let value = rbf.interp_point(Vec3::new(x, y, z));

                plots.push((x, y, z, value));
            }
        }
    }

    plots
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

fn distance_linear(r: f64, epsilon: f64) -> f64 {
    r
}

fn distance_cubic(r: f64, epsilon: f64) -> f64 {
    r.powi(3)
}

fn distance_quintic(r: f64, epsilon: f64) -> f64 {
    r.powi(5)
}

fn distance_thin_plate(r: f64, epsilon: f64) -> f64 {
    if r == 0. {
        0.
    } else {
        r.powi(2) * r.ln()
    }
}

fn distance_gaussian(r: f64, epsilon: f64) -> f64 {
    1. / ((r / epsilon).powi(2) + 1.).exp()
}

#[inline(always)]
fn distance_inverse_multiquadratic(r: f64, epsilon: f64) -> f64 {
    1. / ((r / epsilon).powi(2) + 1.).sqrt()
}

#[inline(always)]
fn distance_multiquadratic(r: f64, epsilon: f64) -> f64 {
    ((r / epsilon).powi(2) + 1.).sqrt()
}

#[derive(Debug, Clone, Copy)]
/// A 3-dimensional bounding box.
pub struct Bbox {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
    pub min_z: f64,
    pub max_z: f64,
}

impl Bbox {
    pub fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64, min_z: f64, max_z: f64) -> Self {
        Bbox {
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
        }
    }

    pub fn from_points(obs_points: &[Vec3]) -> Self {
        let (mut min_x, mut max_x, mut min_y, mut max_y, mut min_z, mut max_z) = (
            99999999., -99999999., 99999999., -99999999., 999999., -999999.,
        );
        for pt in obs_points {
            if pt.x > max_x {
                max_x = pt.x;
            } else if pt.x < min_x {
                min_x = pt.x;
            }
            if pt.y > max_y {
                max_y = pt.y;
            } else if pt.y < min_y {
                min_y = pt.y;
            }
            if pt.z > max_z {
                max_z = pt.z;
            } else if pt.z < min_z {
                min_z = pt.z;
            }
        }
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
        }
    }
}
