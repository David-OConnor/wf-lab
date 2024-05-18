//! Configuration presets

use lin_alg::f64::{Quaternion, Vec3};

use crate::basis_wfs::{SphericalHarmonic, Sto};

#[derive(Debug)]
pub struct NucPreset {
    pub posit: Vec3,
    /// We assume a neutral charge, so this is also charge.
    pub num_elecs: usize,
    /// todo: Weights instead?
    pub bases: Vec<Sto>,
}

#[derive(Debug)]
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
            name: "H2-".to_owned(),
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
        let sto = Sto {
            posit: Vec3::new_zero(),
            n: 1,
            xi: 1.,
            harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
            weight: 0.7,
            charge_id: 0,
        };

        Self {
            name: "H2".to_owned(),
            nuclei: vec![
                NucPreset {
                    posit: Vec3::new(-0.7, 0., 0.),
                    num_elecs: 1,
                    // todo: Update this, from basis_init.
                    bases: vec![sto.clone()],
                },
                NucPreset {
                    posit: Vec3::new(0.7, 0., 0.),
                    num_elecs: 1,
                    // todo: Update this, from basis_init.
                    bases: vec![sto.clone()],
                },
            ],
        }
    }

    pub fn make_he() -> Self {
        Self {
            name: "He".to_owned(),
            nuclei: vec![NucPreset {
                posit: Vec3::new_zero(),
                num_elecs: 2,
                // todo: Update these from bases_init.
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
                ],
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
