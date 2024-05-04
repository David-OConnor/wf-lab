//! This module initializes basis weights.

use lin_alg::f64::Vec3;

use crate::basis_wfs::Basis;

#[allow(dead_code)]
/// [re]Create a set of basis functions, given fixed-charges representing nuclei.
/// Use this in main and lib inits, and when you add or remove charges.
pub fn initialize_bases(
    bases: &mut Vec<Basis>,
    charges_fixed: &[(Vec3, f64)],
    n: u16, // quantum number n
) {
    *bases = Vec::new();

    let weights_h = vec![
        (1., 0.7),
        (2., 0.),
        // (2.5, 0.),
        (3., 0.),
        // (3.5, 0.),
        (4., 0.),
        // (4.5, 0.),
        (5., 0.),
        // (5.5, 0.),
        (6., 0.),
        // (7., 0.),
        (8., 0.),
        // (9., 0.),
        (10., 0.),
    ];

    let weights_he = vec![
        (1., 0.45),
        (2., -0.02),
        (3., -0.25),
        (4., -0.01),
        (5., -0.32),
        (6., 0.17),
        (8., -0.61),
        (10., -0.05),
    ];

    // todo: No normalization on the WF of each orbital. Reason about this.
    // todo: This is just visual for your 2D mesh right, but you are using
    // todo your 2D mesh to assess compatibility... Eign fn cals (eg V and psi'') are invariant
    // todo to normalization (other than for electron charge), but your basis mixing
    // todo needs to take normalization into account...

    // Note: These commented-out ones are from when we numerically normalized the 3d WFs.
    let weights_he_no_norm = vec![
        (1., 0.45),
        (2., -0.02),
        (3., -0.25),
        (4., -0.01),
        (5., -0.32),
        (6., 0.17),
        (8., -0.61),
        (10., -0.05),
    ];

    let weights_li_inner = vec![
        (1., 0.32),
        (2., -0.60),
        (3., -0.17),
        (4., 0.32),
        (5., -0.26),
        (6., 0.10),
        (8., -0.02),
        (10., 0.01),
    ];

    let weights_li_outer = vec![
        // WIP for lithium:
        (1., 1.),
        (2., 0.51),
        (3., -0.16),
        (4., -0.17),
        (5., -1.26),
        (6., -0.83),
        (8., -0.25),
        (10., -0.75),
    ];

    // todo: Update these
    let weights_he_no_norm = vec![
        (1., 0.77),
        (2., -0.01),
        (3., 0.05),
        (4., -0.062),
        (5., 0.20),
        (6., -0.12),
        (8., -0.03),
        (10., -0.53),
    ];

    let weights_li_inner_no_norm = vec![
        (1., 0.32),
        (2., -0.60),
        (3., -0.17),
        (4., 0.32),
        (5., -0.26),
        (6., 0.10),
        (8., -0.02),
        (10., 0.01),
    ];

    let weights_li_outer_no_norm = vec![
        // WIP for lithium:
        (1., 1.),
        (2., 0.51),
        (3., -0.16),
        (4., -0.17),
        (5., -1.26),
        (6., -0.83),
        (8., -0.25),
        (10., -0.75),
    ];

    // todo: We currently call this in some cases where it maybe isn't strictly necessarly;
    // todo for now as a kludge to preserve weights, we copy the prev weights.
    for (charge_id, (nuc_posit, _)) in charges_fixed.iter().enumerate() {
        // See Sebens, for weights under equation 24; this is for Helium.

        let weights = if n == 1 {
            // &weights_h
            &weights_he_no_norm
            // &weights_li_inner_no_norm
        } else {
            &weights_li_outer_no_norm
        };

        for (xi, weight) in weights {
            bases.push(Basis::new_sto(*nuc_posit, n, *xi, *weight, charge_id));
        }
    }
}
