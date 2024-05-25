//! Configuration presets

use lin_alg::f64::{Quaternion, Vec3};

use crate::basis_wfs::{Basis, Gaussian, SphericalHarmonic, Sto};

#[derive(Debug, Clone)]
pub struct NucPreset {
    pub posit: Vec3,
    pub num_protons: u8,
    // /// We assume a neutral charge, so this is also charge.
    // pub num_elecs: usize,
    // /// todo: Weights instead?
    // pub bases: Vec<Sto>,
}

#[derive(Debug, Clone)]
pub struct Preset {
    pub name: String,
    pub nuclei: Vec<NucPreset>,
    pub elecs: Vec<Vec<Basis>>

}

impl Preset {
    pub fn make_h() -> Self {
        Self {
            name: "H".to_owned(),
            nuclei: vec![NucPreset {
                posit: Vec3::new_zero(),
                num_protons: 1,
                // num_elecs: 1,
                // bases: vec![Sto {
                //     posit: Vec3::new_zero(),
                //     n: 1,
                //     xi: 1.,
                //     harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                //     weight: 0.7,
                //     charge_id: 0,
                // }],
            }],
            elecs: vec![vec![Basis::Sto(Sto {
                posit: Vec3::new_zero(),
                n: 1,
                xi: 1.,
                harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                weight: 0.7,
                nuc_id: 0,
            })],]
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
            nuc_id: 0,
        };

        Self {
            name: "H-".to_owned(),
            nuclei: vec![NucPreset {
                posit: Vec3::new_zero(),
                num_protons: 1,
                // num_elecs: 2,
                // todo: What should these actually be?
                // todo: also note: There are (at least) two solutions.
                // bases: vec![sto.clone(), sto],
            }],
            elecs: vec![vec![Basis::Sto(sto.clone())], vec![Basis::Sto(sto)]],
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
            // (6., 0.),
            // (7., 0.),
            // (8., 0.),
            // (9., 0.),
            // (10., 0.),
        ];

        let sto_data: Vec<_> = data.iter().map(|d| StoData::new(0,1, d.0, d.1)).collect();
        let stos = build_stos(&sto_data);

        let nuc_0_posit = Vec3::new(-0.7, 0., 0.);
        let nuc_1_posit = Vec3::new(0.7, 0., 0.);

        let mut stos_0 = stos.clone();
        let mut stos_1 = stos.clone();

        for b in &mut stos_0 {
            b.posit = nuc_0_posit;
            b.nuc_id = 0;
        }

        for b in &mut stos_1 {
            b.posit = nuc_1_posit;
            b.nuc_id = 1;
        }

        let mut bases_0 = Vec::new();
        for s in stos_0{
            bases_0.push(Basis::Sto(s));
        }
        let mut bases_1 = Vec::new();
        for s in stos_1{
            bases_1.push(Basis::Sto(s));
        }

        Self {
            name: "H2".to_owned(),
            nuclei: vec![
                NucPreset {
                    posit: nuc_0_posit,
                    num_protons: 1,
                    // num_elecs: 1,
                    // bases: bases_0,
                },
                NucPreset {
                    posit: nuc_1_posit,
                    num_protons: 1,
                    // num_elecs: 1,
                    // bases: bases_1,
                },
            ],
            elecs: vec![bases_0, bases_1]
        }
    }

    pub fn make_h2_cation() -> Self {
        let data = vec![
            // Also: Gaussian at midpoint, C=0.5, weight=0.2
            (0, 1., 0.7),
            (0, 2., 0.6),
            (0, 3., -0.18),
            (0, 4., 0.06),
            (0, 5., 0.),
            (1, 1., 0.7),
            (1, 2., 0.6),
            (1, 3., -0.18),
            (1, 4., 0.06),
            (1, 5., 0.),
        ];

        let sto_data: Vec<_> = data.iter().map(|d| StoData::new(d.0, 1, d.1, d.2)).collect();
        let stos = build_stos(&sto_data);

        // The experimental value is 2.00 for the distance; ours is significantly off. Can this be
        // explained by our fit not being great?
        let nuc_0_posit = Vec3::new(-0.92, 0., 0.);
        let nuc_1_posit = Vec3::new(0.92, 0., 0.);

        let mut stos = stos.clone();

        for b in &mut stos {
            if b.nuc_id == 0 {
                b.posit = nuc_0_posit
            } else {
                b.posit = nuc_1_posit
            }
        }

        let mut bases = Vec::new();
        for s in stos {
            bases.push(Basis::Sto(s));
        }


        bases.push(Basis::G(Gaussian{
            posit: Vec3::new_zero(),
            c: 0.40,
            weight: 0.20,
        }));

        Self {
            name: "H2+".to_owned(),
            nuclei: vec![
                NucPreset {
                    posit: nuc_0_posit,
                    num_protons: 1,
                },
                NucPreset {
                    posit: nuc_1_posit,
                    num_protons: 1,
                },
            ],
            elecs: vec![bases]
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

        let sto_data: Vec<_> = data.iter().map(|d| StoData::new(0,1, d.0, d.1)).collect();
        let stos = build_stos(&sto_data);

        let mut bases = Vec::new();
        for s in stos {
            bases.push(Basis::Sto(s));
        }

        Self {
            name: "He".to_owned(),
            nuclei: vec![NucPreset {
                posit: Vec3::new_zero(),
                num_protons: 2,
                // num_elecs: 2,
                // bases: stos,
            }],
            elecs: vec![bases.clone(), bases.clone()]
        }
    }

    pub fn make_li() -> Self {
        let weights_outer = vec![
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

        let weights_inner = vec![
            (1., 0.32),
            (2., -0.60),
            (3., -0.17),
            (4., 0.32),
            (5., -0.26),
            (6., 0.10),
            (8., -0.02),
            (10., 0.01),
        ];

        let sto_outer: Vec<_> = weights_outer.iter().map(|d| StoData::new(0,2, d.0, d.1)).collect();
        let sto_inner: Vec<_> = weights_inner.iter().map(|d| StoData::new(0,1, d.0, d.1)).collect();

        let stos_outer = build_stos(&sto_outer);
        let stos_inner = build_stos(&sto_inner);

        let mut bases_inner = Vec::new();
        for s in stos_inner {
            bases_inner.push(Basis::Sto(s));
        }
        let mut bases_outer = Vec::new();
        for s in stos_outer {
            bases_outer.push(Basis::Sto(s));
        }

        Self {
            name: "Li".to_owned(),
            nuclei: vec![NucPreset {
                posit: Vec3::new_zero(),
                num_protons: 3,
                // num_elecs: 3,
                // todo: Update these from bases_init.
                // bases: vec![
                //     Sto {
                //         posit: Vec3::new_zero(),
                //         n: 1,
                //         xi: 1.,
                //         harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                //         weight: 0.7,
                //         nuc_id: 0,
                //     },
                //     Sto {
                //         posit: Vec3::new_zero(),
                //         n: 1,
                //         xi: 1.,
                //         harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                //         weight: 0.7,
                //         nuc_id: 0,
                //     },
                //     Sto {
                //         posit: Vec3::new_zero(),
                //         n: 2,
                //         xi: 1.,
                //         harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                //         weight: 0.7,
                //         nuc_id: 0,
                //     },
                // ],
            }],
            elecs: vec![bases_inner.clone(), bases_inner, bases_outer],
        }
    }

    /// Lithium hidride
    pub fn make_li_h() -> Self {
        Self {
            name: "LiH".to_owned(),
            nuclei: vec![
                NucPreset {
                    posit: Vec3::new(-1.5, 0., 0.),
                    num_protons: 1,
                },
                NucPreset {
                    posit: Vec3::new(1.5, 0., 0.),
                    num_protons: 3,
                },
            ],
            elecs: vec![], // todo
        }
    }
}

/// We use this to represent what is required to build an STO.
struct StoData {
    pub nuc_id: usize,
    pub n: u16,
    pub xi: f64,
    pub weight: f64,
}

impl StoData {
    pub fn new(nuc_id: usize, n: u16, xi: f64, weight: f64) -> Self {
        Self { nuc_id, n, xi, weight }
    }
}

fn build_stos(data: &[StoData]) -> Vec<Sto> {
    let mut result = Vec::new();

    for d in data {
        result.push(Sto {
            nuc_id: d.nuc_id,
            posit: Vec3::new_zero(),
            n: d.n,
            xi: d.xi,
            harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
            weight: d.weight,
        })
    }

    result
}
