#![allow(non_snake_case)]

//! We use this module to define exports for when
/// using this package as a library.
pub mod basis_wfs;
pub mod complex_nums;
pub mod nudge;
pub mod types;
pub mod wf_ops;


use crate::{
    basis_wfs::{Basis, HOrbital, SphericalHarmonic},
    types::{Arr3d, Arr3dReal},
};

use lin_alg2::f64::Vec3;

/// Create trial wave functions for a given point-charge distibution. Currently
/// a rough approach using low-energy STOs centered on the charges.
fn create_trial_wfs(charges: &[(Vec3, f64)]) -> Vec<Basis> {
    let mut result = Vec::new();

    // todo: For now, we are hard-coding weights of +1. A next-step approximation may be
    // todo try various combinations of weights (either +1 and -1, or by tweaking the sliders)
    // todo then starting with the one that gives the best initial score.
    for (id, (charge_posit, charge_amt)) in charges.into_iter().enumerate() {
        result.push(Basis::H(HOrbital::new(
            *charge_posit,
            1,
            SphericalHarmonic::default(),
             // todo no idea if this even is better than setting to 1.
            if charge_amt > &0. { 1. } else { -1. },
            id, // todo unused?
        )))
    }

    result
}

/// Interface from external programs. Main API for solving the wave function.
// pub fn psi_from_V(V: &Arr3dReal, grid_bounds: (f64, f64)) -> Arr3d {
pub fn psi_from_V(charges: &[(Vec3, f64)], grid_bounds: (f64, f64)) -> Arr3d {
    // todo: Input is V, or charges? We use charges for now, since it
    // saves a pass in our initial WF. Perhaps though, we want to pass V intact.
    // todo: Output is psi, or psi^2?

    let mut sfcs = Default::default();

    let wfs = create_trial_wfs(charges);

    let mut E = 0.5;

    // Set up the potential, ψ, and ψ'' (measured and calculated) for the potential from input charges,
    // and our basis-function based trial wave function.
    wf_ops::eval_wf(
        &wfs,
        &charges,
        &mut sfcs,
        E,
        true,
        grid_bounds.0,
        grid_bounds.1,
    );

    nudge::nudge_wf(
        &mut sfcs,
        &mut 0.1,
        &mut E,
        grid_bounds.0,
        grid_bounds.1,
    );

    // let psi_pp_score = wf_ops::score_wf(&sfcs);

    sfcs.psi
}
