#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(uncommon_codepoints)]
#![allow(confusable_idents)]
#![allow(non_upper_case_globals)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

//! We use this module to define exports for when
/// using this package as a library.
pub mod basis_wfs;
pub mod complex_nums;
mod eigen_fns;
pub mod elec_elec;
pub mod eval;
mod grid_setup;
pub mod interp;
pub mod nudge;
mod num_diff;
pub mod potential;
pub mod types;
pub mod util;
pub mod wf_ops;

use basis_wfs::{Basis, HOrbital, SphericalHarmonic};
use grid_setup::{Arr3d, Arr3dVec};
use types::SurfacesPerElec;

use lin_alg2::f64::Vec3;

/// Create trial wave functions for a given point-charge distibution. Currently
/// a rough approach using low-energy STOs centered on the charges.
fn create_trial_wfs(charges: &[(Vec3, f64)]) -> Vec<Basis> {
    let mut result = Vec::new();

    // todo: For now, we are hard-coding weights of +1. A next-step approximation may be
    // todo try various combinations of weights (either +1 and -1, or by tweaking the sliders)
    // todo then starting with the one that gives the best initial score.
    for (id, (charge_posit, charge_amt)) in charges.iter().enumerate() {
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
    charges_fixed: &[(Vec3, f64)],
    grid_range: &mut (f64, f64),
    spacing_factor: f64,
    bases: &[Basis],
) -> Arr3d {
    // todo: Input is V, or charges? We use charges for now, since it
    // saves a pass in our initial WF. Perhaps though, we want to pass V intact.
    // todo: Output is psi, or psi^2?

    let grid_n = 30;

    let mut sfcs = SurfacesPerElec::new(grid_n);

    let mut grid_posits = grid_setup::new_data_vec(grid_n);

    grid_setup::update_grid_posits(&mut grid_posits, *grid_range, spacing_factor, grid_n);

    let wfs = create_trial_wfs(charges_fixed);

    // todo: grids that aren't centered at 0? Non-cube grids?

    let arr_real = grid_setup::new_data_real(grid_n);

    let mut V_shared = arr_real.clone();

    // These must be initialized from wave functions later.
    let charges_electron = vec![arr_real.clone(), arr_real];

    let ui_active_elec = 0;

    // Set up the potential, ψ, and ψ'' (measured and calculated) for the potential from input charges,
    potential::update_V_from_nuclei(&mut V_shared, charges_fixed, &grid_posits, grid_n);

    let bases_unweighted = wf_ops::BasesEvaluated::new(bases, &grid_posits, grid_n);

    // Set up our basis-function based trial wave function.
    wf_ops::update_wf_fm_bases(&wfs, &bases_unweighted, &mut sfcs, grid_n, None);

    sfcs.psi.on_pt.clone()
}
