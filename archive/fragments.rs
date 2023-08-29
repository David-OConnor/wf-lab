//! Misc code fragments we no longer use, but are available for reference or re-implementation in the future.

// /// Utility function to linearly map an input value to an output
// pub fn map_linear(range_in: (f64, f64), range_out: (f64, f64), val: f64) -> f64 {
//     // todo: You may be able to optimize calls to this by having the ranges pre-store
//     // todo the total range vals.
//     let portion = (val - range_in.0) / (range_in.1 - range_in.0);
//
//     portion * (range_out.1 - range_out.0) + range_out.0
// }


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