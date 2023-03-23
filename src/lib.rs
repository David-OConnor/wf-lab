#![allow(non_snake_case)]

//! We use this module to define exports for when
/// using this package as a library.
pub mod basis_wfs;
pub mod complex_nums;
pub mod interp;
pub mod nudge;
mod num_diff;
mod rbf;
pub mod types;
pub mod util;
pub mod wf_ops;

use basis_wfs::{Basis, HOrbital, SphericalHarmonic};

use lin_alg2::f64::Vec3;

pub use types::Arr3d;
pub use types::Arr3dReal;
pub use types::Surfaces;

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
pub fn psi_from_pt_charges(
    charges: &[(Vec3, f64)],
    grid_bounds: &mut (f64, f64),
    spacing_factor: f64,
    bases: &[Basis],
) -> Arr3d {
    // todo: Input is V, or charges? We use charges for now, since it
    // saves a pass in our initial WF. Perhaps though, we want to pass V intact.
    // todo: Output is psi, or psi^2?

    let mut sfcs = Surfaces::default();
    wf_ops::update_grid_posits(
        &mut sfcs.grid_posits,
        grid_bounds.0,
        grid_bounds.1,
        spacing_factor,
    );

    let wfs = create_trial_wfs(charges);

    let mut E = 0.5;

    // todo: grids that aren't centered at 0? Non-cube grids?

    let spacing_factor = 1.;

    // Set up the potential, ψ, and ψ'' (measured and calculated) for the potential from input charges,
    // and our basis-function based trial wave function.
    wf_ops::init_wf(
        &wfs,
        &charges,
        &mut sfcs,
        E,
        true,
        &mut grid_bounds.0,
        &mut grid_bounds.1,
        spacing_factor,
    );

    // todo: Temp removing nudge to test performance

    nudge::nudge_wf(
        &mut sfcs,
        &mut 0.1,
        &mut E,
        grid_bounds.0,
        grid_bounds.1,
        &bases,
    );

    // let psi_pp_score = wf_ops::score_wf(&sfcs);

    sfcs.psis_per_elec[0].clone()
}
