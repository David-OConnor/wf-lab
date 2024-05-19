//! Configuration presets

use lin_alg::f64::{Quaternion, Vec3};

use crate::basis_wfs::{SphericalHarmonic, Sto};

#[derive(Debug, Clone)]
pub struct NucPreset {
    pub posit: Vec3,
    /// We assume a neutral charge, so this is also charge.
    pub num_elecs: usize,
    /// todo: Weights instead?
    pub bases: Vec<Sto>,
}

#[derive(Debug, Clone)]
pub struct Preset {
    pub name: String,
    pub nuclei: Vec<NucPreset>,
}

impl Preset {
    pub fn make_h() -> Self {
        Self {
            name: "H".to_owned(),
            nuclei: vec![NucPreset {
                posit: Vec3::new_zero(),
                num_elecs: 1,
                bases: vec![Sto {
                    posit: Vec3::new_zero(),
                    n: 1,
                    xi: 1.,
                    harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                    weight: 0.7,
                    charge_id: 0,
                }],
            }],
        }
    }

    pub fn make_h_anion() -> Self {
        // todo:  Update A/R.
        let sto = Sto {
            posit: Vec3::new_zero(),
            n: 1,
            xi: 1.,
            harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
            weight: 0.7,
            charge_id: 0,
        };

        Self {
            name: "H-".to_owned(),
            nuclei: vec![NucPreset {
                posit: Vec3::new_zero(),
                num_elecs: 2,
                // todo: What should these actually be?
                // todo: also note: There are (at least) two solutions.
                bases: vec![sto.clone(), sto],
            }],
        }
    }

    pub fn make_h2() -> Self {
        let data = vec![
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

        let sto_data: Vec<_> = data.iter().map(|d| StoData::new(1, d.0, d.1)).collect();
        let stos = build_stos(&sto_data);

        let nuc_0_posit = Vec3::new(-0.7, 0., 0.);
        let nuc_1_posit = Vec3::new(0.7, 0., 0.);

        let mut bases_0 = stos.clone();
        let mut bases_1 = stos.clone();

        for b in &mut bases_0 {
            b.posit = nuc_0_posit;
        }

        for b in &mut bases_1 {
            b.posit = nuc_1_posit;
        }

        Self {
            name: "H2".to_owned(),
            nuclei: vec![
                NucPreset {
                    posit: nuc_0_posit,
                    num_elecs: 1,
                    bases: bases_0,
                },
                NucPreset {
                    posit: nuc_1_posit,
                    num_elecs: 1,
                    bases: bases_1,
                },
            ],
        }
    }

    pub fn make_he() -> Self {
        let data = vec![
            (1., 0.45),
            (2., -0.02),
            (3., -0.25),
            (4., -0.01),
            (5., -0.32),
            (6., 0.17),
            (8., -0.61),
            (10., -0.05),
        ];
        // todo: Or is it this?
        // vec![
        //     (1., 0.77),
        //     (2., -0.01),
        //     (3., 0.05),
        //     (4., -0.062),
        //     (5., 0.20),
        //     (6., -0.12),
        //     (8., -0.03),
        //     (10., -0.53),
        // ];

        let sto_data: Vec<_> = data.iter().map(|d| StoData::new(1, d.0, d.1)).collect();
        let stos = build_stos(&sto_data);

        Self {
            name: "He".to_owned(),
            nuclei: vec![NucPreset {
                posit: Vec3::new_zero(),
                num_elecs: 2,
                bases: stos,
            }],
        }
    }

    pub fn make_li() -> Self {
        Self {
            name: "Li".to_owned(),
            nuclei: vec![NucPreset {
                posit: Vec3::new_zero(),
                num_elecs: 3,
                /// todo: Update these from bases_init.
                bases: vec![
                    Sto {
                        posit: Vec3::new_zero(),
                        n: 1,
                        xi: 1.,
                        harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                        weight: 0.7,
                        charge_id: 0,
                    },
                    Sto {
                        posit: Vec3::new_zero(),
                        n: 1,
                        xi: 1.,
                        harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                        weight: 0.7,
                        charge_id: 0,
                    },
                    Sto {
                        posit: Vec3::new_zero(),
                        n: 2,
                        xi: 1.,
                        harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                        weight: 0.7,
                        charge_id: 0,
                    },
                ],
            }],
        }
    }

    /// Lithium hidride
    pub fn make_li_h() -> Self {
        Self {
            name: "LiH".to_owned(),
            nuclei: vec![
                NucPreset {
                    posit: Vec3::new(-1.5, 0., 0.),
                    num_elecs: 1,
                    /// todo: Update these from bases_init.
                    bases: vec![Sto {
                        posit: Vec3::new_zero(),
                        n: 1,
                        xi: 1.,
                        harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                        weight: 0.7,
                        charge_id: 0,
                    }],
                },
                NucPreset {
                    posit: Vec3::new(-1.5, 0., 0.),
                    num_elecs: 3,
                    /// todo: Update these from bases_init.
                    bases: vec![
                        Sto {
                            posit: Vec3::new_zero(),
                            n: 1,
                            xi: 1.,
                            harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                            weight: 0.7,
                            charge_id: 0,
                        },
                        Sto {
                            posit: Vec3::new_zero(),
                            n: 1,
                            xi: 1.,
                            harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                            weight: 0.7,
                            charge_id: 0,
                        },
                        Sto {
                            posit: Vec3::new_zero(),
                            n: 2,
                            xi: 1.,
                            harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                            weight: 0.7,
                            charge_id: 0,
                        },
                    ],
                },
            ],
        }
    }
}

/// We use this to represent what is required to build an STO.
struct StoData {
    pub n: u16,
    pub xi: f64,
    pub weight: f64,
}

impl StoData {
    pub fn new(n: u16, xi: f64, weight: f64) -> Self {
        Self { n, xi, weight }
    }
}

fn build_stos(data: &[StoData]) -> Vec<Sto> {
    let mut result = Vec::new();

    for d in data {
        result.push(Sto {
            posit: Vec3::new_zero(), // todo ?
            n: d.n,
            xi: d.xi,
            harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()), // todo ?
            weight: d.weight,
            charge_id: 0, // todo ?
        })
    }

    result
}
