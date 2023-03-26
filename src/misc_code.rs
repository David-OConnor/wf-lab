//! This module contains code we currently don't use, but may use later. it's not attached
//! to the library or binary, so code here doesn't have to compile.


/// Initialize a wave function using a charge-centric coordinate system, using RBF
/// interpolation.
fn init_wf_rbf(rbf: &Rbf, charges: &[(Vec3, f64)], bases: &[Basis], E: f64) {
    // todo: Start RBF testing using its own grid

    let mut psi_pp_calc_rbf = Vec::new();
    let mut psi_pp_meas_rbf = Vec::new();

    for (i, sample_pt) in rbf.obs_points.iter().enumerate() {
        let psi_sample = rbf.fn_vals[i];

        // calc(psi: &Arr3d, V: &Arr3dReal, E: f64, i: usize, j: usize, k: usize) -> Cplx {

        let V_sample = {
            let mut result = 0.;
            for (posit_charge, charge_amt) in charges.iter() {
                result += V_coulomb(*posit_charge, *sample_pt, *charge_amt);
            }

            result
        };

        let calc = psi_sample * (E - V_sample) * eigen_fns::KE_COEFF;

        psi_pp_calc_rbf.push(calc);
        psi_pp_meas_rbf.push(num_diff::find_ψ_pp_meas_fm_rbf(
            *sample_pt,
            Cplx::from_real(psi_sample),
            &rbf,
        ));
    }

    println!(
        "Comp1: {:?}, {:?}",
        psi_pp_calc_rbf[100], psi_pp_meas_rbf[100]
    );
    println!(
        "Comp2: {:?}, {:?}",
        psi_pp_calc_rbf[10], psi_pp_meas_rbf[10]
    );
    println!(
        "Comp3: {:?}, {:?}",
        psi_pp_calc_rbf[20], psi_pp_meas_rbf[20]
    );
    println!(
        "Comp4: {:?}, {:?}",
        psi_pp_calc_rbf[30], psi_pp_meas_rbf[30]
    );
    println!(
        "Comp5: {:?}, {:?}",
        psi_pp_calc_rbf[40], psi_pp_meas_rbf[40]
    );

    // Code to test interpolation.
    let b1 = &bases[0];

    let rbf_compare_pts = vec![
        Vec3::new(1., 0., 0.),
        Vec3::new(1., 1., 0.),
        Vec3::new(0., 0., 1.),
        Vec3::new(5., 0.5, 0.5),
        Vec3::new(6., 5., 0.5),
        Vec3::new(6., 0., 0.5),
    ];

    println!("\n");
    for pt in &rbf_compare_pts {
        println!(
            "\nBasis: {:.5} \n Rbf: {:.5}",
            b1.value(*pt).real * b1.weight(),
            rbf.interp_point(*pt),
        );
    }
}