//! Configuration presets

use lin_alg::f64::{Quaternion, Vec3};

use crate::basis_wfs::{SphericalHarmonic, Sto};

#[derive(Debug)]
pub struct Preset {
    pub name: String,
    pub nuclei_posits: Vec<Vec3>,
    pub num_elecs: usize,
    pub bases: Vec<Sto>,
}

impl Preset {
    pub fn make_h() -> Self {
        Self {
            name: "H".to_owned(),
            nuclei_posits: vec![Vec3::new_zero()],
            num_elecs: 1,
            bases: vec![Sto {
                posit: Vec3::new_zero(),
                n: 1,
                xi: 1.,
                harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                weight: 0.7,
                charge_id: 0,
            }],
        }
    }

    pub fn make_h_ion() -> Self {
        Self {
            name: "H2-".to_owned(),
            // todo: Dist?
            nuclei_posits: vec![
                Vec3 {
                    x: -0.2,
                    y: 0.,
                    z: 0.,
                },
                Vec3 {
                    x: 0.2,
                    y: 0.,
                    z: 0.,
                },
            ],
            num_elecs: 1,
            bases: vec![Sto {
                posit: Vec3::new_zero(),
                n: 1,
                xi: 1.,
                harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                weight: 0.7,
                charge_id: 0,
            }],
        }
    }

    pub fn make_h2() -> Self {
        Self {
            name: "H2".to_owned(),
            nuclei_posits: vec![
                Vec3 {
                    x: -0.7,
                    y: 0.,
                    z: 0.,
                },
                Vec3 {
                    x: 0.7,
                    y: 0.,
                    z: 0.,
                },
            ],
            num_elecs: 2,
            bases: vec![Sto {
                posit: Vec3::new_zero(),
                n: 1,
                xi: 1.,
                harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                weight: 0.7,
                charge_id: 0,
            }],
        }
    }

    pub fn make_he() -> Self {
        Self {
            name: "He".to_owned(),
            nuclei_posits: vec![Vec3::new_zero()],
            num_elecs: 2,
            // todo
            bases: vec![Sto {
                posit: Vec3::new_zero(),
                n: 1,
                xi: 1.,
                harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                weight: 0.7,
                charge_id: 0,
            }],
        }
    }

    pub fn make_li() -> Self {
        Self {
            name: "Li".to_owned(),
            nuclei_posits: vec![Vec3::new_zero()],
            num_elecs: 3,
            // todo
            bases: vec![Sto {
                posit: Vec3::new_zero(),
                n: 1,
                xi: 1.,
                harmonic: SphericalHarmonic::new(0, 0, Quaternion::new_identity()),
                weight: 0.7,
                charge_id: 0,
            }],
        }
    }
}
