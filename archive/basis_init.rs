//! This module initializes basis weights.

use lin_alg::f64::Vec3;

use crate::{
    basis_wfs::{Basis, Sto},
    dirac::BasisSpinor,
};

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

    let weights_h2 = vec![
        // Also: Gaussian at midpoint, C=0.5, weight=0.2
        (1., 0.7),
        (2., 0.2),
        // (2.5, 0.),
        (3., 0.05),
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

    // todo: Which of these he weights is right?
    let weights_he_no_norm_ = vec![
        (1., 0.77),
        (2., -0.01),
        (3., 0.05),
        (4., -0.062),
        (5., 0.20),
        (6., -0.12),
        (8., -0.03),
        (10., -0.53),
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

    let weights_li_h_li_outer = vec![
        (1., 1.),
        (2., 0.),
        (3., 0.),
        (4., 0.),
        (5., 0.),
        (6., 0.),
        (8., 0.),
        (10., 0.),
    ];

    let weights_li_h_li_inner = vec![
        (1., 1.),
        (2., 0.),
        (3., 0.),
        (4., 0.),
        (5., 0.),
        (6., 0.),
        (8., 0.),
        (10., 0.),
    ];

    let weights_li_h_h = vec![
        (1., 1.),
        (2., 0.),
        (3., 0.),
        (4., 0.),
        (5., 0.),
        (6., 0.),
        (8., 0.),
        (10., 0.),
    ];

    // todo: Re-do this function to be more flexible.

    // Experimenting
    // bases.push(Basis::new_gauss(Vec3::new_zero(), 0.5, 0.2));

    // todo: We currently call this in some cases where it maybe isn't strictly necessarly;
    // todo for now as a kludge to preserve weights, we copy the prev weights.
    for (nuc_id, (nuc_posit, _)) in charges_fixed.iter().enumerate() {
        // See Sebens, for weights under equation 24; this is for Helium.

        // Experimenting with Li_h
        let weights = if nuc_id == 0 {
            // h nucleus.
            &weights_li_h_h
        } else {
            if n == 1 {
                &weights_li_h_li_inner
            } else {
                &weights_li_h_li_outer
            }
        };

        //
        // let weights = if n == 1 {
        //     // &weights_h
        //     &weights_h2
        //     // &weights_he_no_norm
        //     // &weights_li_inner_no_norm
        // } else {
        //     // &weights_li_outer
        //     &weights_li_h_li_outer
        // };

        for (xi, weight) in weights {
            bases.push(Basis::new_sto(*nuc_posit, n, *xi, *weight, nuc_id));
        }
    }
}

pub fn initialize_bases_spinor(
    bases: &mut Vec<BasisSpinor>,
    charges_fixed: &[(Vec3, f64)],
    max_n: u16, // quantum number n
) {
    // todo: For performance reasons while not using it.
    return;
    *bases = Vec::new();

    // todo: We currently call this in some cases where it maybe isn't strictly necessarly;
    // todo for now as a kludge to preserve weights, we copy the prev weights.
    for (charge_id, (nuc_posit, _)) in charges_fixed.iter().enumerate() {
        // See Sebens, for weights under equation 24; this is for Helium.

        for (xi, weight) in [
            (1., 1.),
            // (2., 0.),
            // (3., 0.),
            // (4., 0.),
            // (5., 0.),
            // (6., 0.),
            // (7., 0.),
        ] {
            for n in 1..max_n + 1 {
                let sto = Sto {
                    posit: *nuc_posit,
                    n,
                    xi,
                    weight,
                    charge_id,
                    harmonic: Default::default(),
                };

                let mut sto_zero = sto.clone();
                sto_zero.weight = 0.;

                let mut sto_neg = sto.clone();
                sto_neg.weight = -sto.weight;

                bases.push(BasisSpinor {
                    c0: sto.clone(),
                    c1: sto_zero.clone(),
                    c2: sto_zero.clone(),
                    c3: sto_neg,
                });
            }
        }
    }
}
