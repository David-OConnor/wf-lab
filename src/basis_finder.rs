//! Used to find weights and Xis for STOs.

use lin_alg2::f64::Vec3;

use crate::{
    basis_wfs::{Basis, Sto},
    eigen_fns,
    grid_setup::Arr3dReal,
    grid_setup::Arr3dVec,
    util, wf_ops,
};

/// Find a wave function, composed of STOs, that match a given potential.
pub fn find_stos(V: &Arr3dReal, grid_posits: &Arr3dVec) -> (Vec<Basis>, f64) {
    let mut bases = Vec::new();
    let E = 0.;

    const XI_INITIAL: f64 = 2.;

    // Base STO is the one with the lowest xi.
    let mut trial_base_sto = Basis::Sto(Sto {
        posit: Vec3::new_zero(), // todo: Hard-coded for a single nuc at 0.
        n: 1,
        xi: XI_INITIAL,
        weight: 1.,
        // todo: As-required. This is used to associate it's position with a nucleus's.
        // todo: If we have nuc position, what purpose does it servce.
        charge_id: 0,
        harmonic: Default::default(),
    });

    // Set energy so that at a corner, (or edge, ie as close to +/- infinity as we have given a grid-based V)
    // V calculated from this basis matches the potential at this point.

    let V_corner = V[0][0][0];

    let psi_corner = trial_base_sto.value(grid_posits[0][0][0]);
    let psi_pp_corner = trial_base_sto.second_deriv(grid_posits[0][0][0]);

    let E_corner_match = eigen_fns::calc_E_on_psi(psi_corner, psi_pp_corner, V_corner);

    println!("E from corner match: {:?}", E_corner_match);

    (bases, E)
}
