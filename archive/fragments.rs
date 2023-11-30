//! Misc code fragments we no longer use, but are available for reference or re-implementation in the future.

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



// /// Calcualte ψ'' measured, using a discrete function, interpolated.
// /// Calculate ψ'' based on a numerical derivative of psi
// /// in 3D.
// pub(crate) fn find_ψ_pp_meas_fm_rbf(posit_sample: Vec3, psi_sample: Cplx, rbf: &Rbf) -> Cplx {
//     let h2 = 0.01;
//
//     let x_prev = Vec3::new(posit_sample.x - h2, posit_sample.y, posit_sample.z);
//     let x_next = Vec3::new(posit_sample.x + h2, posit_sample.y, posit_sample.z);
//     let y_prev = Vec3::new(posit_sample.x, posit_sample.y - h2, posit_sample.z);
//     let y_next = Vec3::new(posit_sample.x, posit_sample.y + h2, posit_sample.z);
//     let z_prev = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z - h2);
//     let z_next = Vec3::new(posit_sample.x, posit_sample.y, posit_sample.z + h2);
//
//     let psi_x_prev = rbf.interp_point(x_prev);
//     let psi_x_next = rbf.interp_point(x_next);
//     let psi_y_prev = rbf.interp_point(y_prev);
//     let psi_y_next = rbf.interp_point(y_next);
//     let psi_z_prev = rbf.interp_point(z_prev);
//     let psi_z_next = rbf.interp_point(z_next);
//
//     // todo: real only for now.
//
//     let result = psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
//         - psi_sample.real * 6.;
//
//     // result / H_SQ
//     Cplx::from_real(result / (h2 * h2)) // todo real temp
// }


// /// Run this after update E.
// pub fn _update_psi_pp_calc(
//     // We split these arguments up instead of using surfaces to control mutability.
//     psi: &Arr3d,
//     V: &Arr3dReal,
//     psi_pp_calc: &mut Arr3d,
//     E: f64,
//     grid_n: usize,
// ) {
//     for i in 0..grid_n {
//         for j in 0..grid_n {
//             for k in 0..grid_n {
//                 psi_pp_calc[i][j][k] = eigen_fns::find_ψ_pp_calc(psi[i][j][k], V[i][j][k], E);
//             }
//         }
//     }
// }


// (From `basis_finder`):
/// Create an order-2 polynomial based on 2 or 3 calibration points.
/// `a` is the ^2 term, `b` is the linear term, `c` is the constant term.
/// This is a general mathematical function, and can be derived using a system of equations.
fn _create_polynomial_terms(pt0: (f64, f64), pt1: (f64, f64), pt2: (f64, f64)) -> (f64, f64, f64) {
    let a_num = pt0.0 * (pt2.1 - pt1.1) + pt1.0 * (pt0.1 - pt2.1) + pt2.0 * (pt1.1 - pt0.1);

    let a_denom = (pt0.0 - pt1.0) * (pt0.0 - pt2.0) * (pt1.0 - pt2.0);

    let a = a_num / a_denom;
    let b = (pt1.1 - pt0.1) / (pt1.0 - pt0.0) - a * (pt0.0 + pt1.0);
    let c = pt0.1 - a * pt0.0.powi(2) - b * pt0.0;

    (a, b, c)
}


/// Experimental; very.
fn _numerical_psi_ps(trial_base_sto: &Basis, grid_posits: &Arr3dVec, V: &Arr3dReal, E: f64) {
    let H = grid_posits[1][0][0].x - grid_posits[0][0][0].x;
    let V_pp_corner = num_diff::find_pp_real(
        V[1][1][1], V[0][1][1], V[2][1][1], V[1][0][1], V[1][2][1], V[1][1][0], V[1][1][2], H,
    );

    // todo QC this
    let V_p_corner = (V[2][1][1] - V[0][1][1])
        + (V[1][2][1] - V[1][0][1])
        + (V[1][1][2] - V[1][1][0]) / (2. * H);

    // let V_pp_psi = trial_base_sto.V_pp_from_psi(posit_corner_offset);
    // let V_p_psi = trial_base_sto.V_p_from_psi(posit_corner_offset);

    // todo: Let's do a cheeky numeric derivative of oV from psi until we're confident the analytic approach
    // todo works.

    // todo well, this is a mess, but it's easy enough to evaluate.
    let posit_x_prev = grid_posits[0][1][1];
    let posit_x_next = grid_posits[2][1][1];
    let posit_y_prev = grid_posits[1][0][1];
    let posit_y_next = grid_posits[1][2][1];
    let posit_z_prev = grid_posits[1][1][0];
    let posit_z_next = grid_posits[1][1][2];

    let psi_x_prev = trial_base_sto.value(posit_x_prev);
    let psi_pp_x_prev = trial_base_sto.second_deriv(posit_x_prev);
    let psi_x_next = trial_base_sto.value(posit_x_next);
    let psi_pp_x_next = trial_base_sto.second_deriv(posit_x_next);

    let psi_y_prev = trial_base_sto.value(posit_y_prev);
    let psi_pp_y_prev = trial_base_sto.second_deriv(posit_y_prev);
    let psi_y_next = trial_base_sto.value(posit_y_next);
    let psi_pp_y_next = trial_base_sto.second_deriv(posit_y_next);

    let psi_z_prev = trial_base_sto.value(posit_z_prev);
    let psi_pp_z_prev = trial_base_sto.second_deriv(posit_z_prev);
    let psi_z_next = trial_base_sto.value(posit_z_next);
    let psi_pp_z_next = trial_base_sto.second_deriv(posit_z_next);

    let V_p_psi = ((calc_V_on_psi(psi_x_next, psi_pp_x_next, E)
        - calc_V_on_psi(psi_x_prev, psi_pp_x_prev, E))
        + (calc_V_on_psi(psi_y_next, psi_pp_y_next, E)
        - calc_V_on_psi(psi_y_prev, psi_pp_y_prev, E))
        + (calc_V_on_psi(psi_z_next, psi_pp_z_next, E)
        - calc_V_on_psi(psi_z_prev, psi_pp_z_prev, E)))
        / (2. * H);

    println!("V' corner: Blue {}  Grey {}", V_p_corner, V_p_psi);
    // println!("V'' corner: Blue {}  Grey {}", V_pp_corner, V_pp_psi);
}

fn find_base_xi_E_discrete(
    V: &Arr3dReal,
    grid_posits: &Arr3dVec,
    // base_xi_specified: f64
) -> (f64, f64) {
    // Set energy so that at a corner, (or edge, ie as close to +/- infinity as we have given a grid-based V)
    // V calculated from this basis matches the potential at this point.
    let index_halfway = V[0].len() / 2;

    let posit_corner = grid_posits[0][0][0];
    let posit_sample = grid_posits[index_halfway][0][0];

    let V_corner = V[0][0][0];
    let V_sample = V[index_halfway][0][0];

    find_base_xi_E_common(
        V_corner,
        posit_corner,
        V_sample,
        posit_sample,
        // base_xi_specified,
    )
}