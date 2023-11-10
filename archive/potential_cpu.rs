/// Helper fn to deal with 2d vs 3d electron V. Calculates Coulomb potential at every charge point,
/// at a single sample point.
fn V_from_grid_inner(
    V_from_this_elec: &mut Arr3dReal,
    charge_this_elec: &Arr3dReal,
    grid_posits: &Arr3dVec,
    grid_charge: &Arr3dVec,
    grid_n_charge: usize,
    posit: (usize, usize, usize),
) {
    let posit_sample = grid_posits[posit.0][posit.1][posit.2];

    // Iterate through this electron's (already computed) charge at every position in space,
    // comparing it to this position.

    V_from_this_elec[posit.0][posit.1][posit.2] = 0.;

    for i_charge in 0..grid_n_charge {
        for j_charge in 0..grid_n_charge {
            for k_charge in 0..grid_n_charge {
                let posit_charge = grid_charge[i_charge][j_charge][k_charge];
                let charge = charge_this_elec[i_charge][j_charge][k_charge];

                V_from_this_elec[posit.0][posit.1][posit.2] +=
                    V_coulomb(posit_charge, posit_sample, charge);
            }
        }
    }
}

/// Update the V associated with a single electron's charge.
/// This must be run after the charge from this electron is created from the wave function square.
/// We expect the loop over charge positions to be larger than the one over V positions.
///
/// This is computationally intensive. The `twod_only` option can alleviate this by only
/// evaluating potential points on one plane.
/// Deprecated in favor of GPU.
pub(crate) fn create_V_from_elec_grid(
    V_from_this_elec: &mut Arr3dReal,
    charges_elec: &Arr3dReal,
    posits_sample: &Arr3dVec,
    posits_charge: &Arr3dVec,
    grid_n: usize,
    grid_n_charge: usize,
    twod_only: bool,
) {
    println!("Creating V from an electron on grid...");

    for i_sample in 0..grid_n {
        for j_sample in 0..grid_n {
            if twod_only {
                // This makes it grid_n times faster, but only creates one Z-slice.
                let k_sample = grid_n / 2 + 1;
                V_from_grid_inner(
                    V_from_this_elec,
                    charges_elec,
                    posits_sample,
                    posits_charge,
                    grid_n_charge,
                    (i_sample, j_sample, k_sample),
                )
            } else {
                for k_sample in 0..grid_n {
                    V_from_grid_inner(
                        V_from_this_elec,
                        charges_elec,
                        posits_sample,
                        posits_charge,
                        grid_n_charge,
                        (i_sample, j_sample, k_sample),
                    )
                }
            }
        }
    }

    println!("V creation complete");
}


/// Create a potential on a set of sample points, from nuclei and electrons.
pub fn create_V_1d(
    posits_sample: &[Vec3],
    charges_fixed: &[(Vec3, f64)],
    charges_elec: &Arr3dReal,
    posits_charge: &Arr3dVec,
    grid_n_charge: usize,
) -> Vec<f64> {
    let mut V_to_match = Vec::new();

    for sample_pt in posits_sample {
        let mut V_sample = 0.;

        for (posit_nuc, charge) in charges_fixed {
            V_sample += V_coulomb(*posit_nuc, *sample_pt, *charge);
        }

        for i in 0..grid_n_charge {
            for j in 0..grid_n_charge {
                for k in 0..grid_n_charge {
                    let posit_charge = posits_charge[i][j][k];
                    let charge = charges_elec[i][j][k];

                    V_sample += V_coulomb(posit_charge, *sample_pt, charge);
                }
            }
        }

        V_to_match.push(V_sample);
    }

    V_to_match
}
