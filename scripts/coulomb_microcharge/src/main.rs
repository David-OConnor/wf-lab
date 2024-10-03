use rand::Rng; // Add this to use the random number generator
use std::f64::consts::TAU;

use lin_alg::f64::Vec3;
use lin_alg::f32::Vec3 as Vec3f32;
use rerun::{RecordingStream, RecordingStreamError, Image, Points3D, demo_util::grid, external::glam, Position3D};

const M_ELEC: f64 = 1.;
const Q_ELEC: f64 = -1.;

#[derive(Debug)]
struct SnapShot {
    pub time: f64,
    // pub nucs: Vec<Nucleus>,
    // todo: To save memory, you could store the snapshots as f32; we only need f64 precision
    // todo during the integration.
    // pub elecs: Vec<Electron>,
    pub elec_posits: Vec<Vec3f32>,
    pub nuc_posits: Vec<Vec3f32>,
}

#[derive(Debug)]
struct Nucleus {
    pub mass: f64,
    pub charge: f64, // todo: Integer?
    pub posit: Vec3,
}

#[derive(Clone, Debug)]
struct Electron {
    pub posit: Vec3,
    pub v: Vec3,
    pub a: Vec3,
}

/// Calculate the Coulomb acceleration on a particle, from a single other particle.
fn accel_coulomb(
    posit_acted_on: Vec3,
    posit_actor: Vec3,
    q_acted_on: f64,
    q_actor: f64,
    mass_acted_on: f64,
) -> Vec3 {
    let posit_diff = posit_actor - posit_acted_on;
    let dist = posit_diff.magnitude();

    let posit_diff_unit = posit_diff / dist;

    // Note: We factor out KC * Q_PROT, since they are present in every calculation.
    // Calculate the Coulomb force between nuclei.

    let f_mag = q_acted_on * q_actor / dist.powi(2);

    posit_diff_unit * f_mag / mass_acted_on
}

fn integrate_rk4(elecs: &mut [Electron], dt: f64) {
    for elec in elecs.iter_mut() {
        // Step 1: Calculate the k-values for position and velocity
        let k1_v = elec.a * dt;
        let k1_posit = elec.v * dt;

        let k2_v = (elec.a) * dt;
        let k2_posit = (elec.v + k1_v * 0.5) * dt;

        let k3_v = (elec.a) * dt;
        let k3_posit = (elec.v + k2_v * 0.5) * dt;

        let k4_v = (elec.a) * dt;
        let k4_posit = (elec.v + k3_v) * dt;

        // Step 2: Update position and velocity using weighted average of k-values
        elec.v += (k1_v + k2_v * 2. + k3_v * 2. + k4_v) / 6.;
        elec.posit += (k1_posit + k2_posit * 2. + k3_posit * 2. + k4_posit) / 6.;
    }
}

/// Set up initial condition for electron positions and velocities. This may have
/// significant effects on the outcome.
fn make_initial_elecs(n_elecs: usize) -> Vec<Electron> {
    let mut result = Vec::new();

    let mut rng = rand::thread_rng();

    let distribution_center = Vec3::new_zero(); // todo: Nucleus etc.
    let distribution_radius = 2.;

    for i in 0..num_elecs {
        // Generate random spherical coordinates
        let r = distribution_radius * rng.gen::<f64>().cbrt();  // Random radius scaled within [0, distribution_radius]
        let theta = rng.gen_range(0.0..TAU);              // Random angle theta in [0, 2*pi]
        let phi = rng.gen_range(0.0..TAU/2.);                      // Random angle phi in [0, pi]

        // Convert spherical coordinates to Cartesian coordinates
        let x = r * phi.sin() * theta.cos();
        let y = r * phi.sin() * theta.sin();
        let z = r * phi.cos();

        let posit = distribution_center + Vec3::new(x, y, z);

        result.push(Electron {
            posit,
            v: Vec3::new_zero(),
            a: Vec3::new_zero(),
        })
    }

    result
}

fn run(n_elecs: usize, n_timesteps: usize, dt: f64) -> Vec<Snapshot> {
    let mut snapshots = Vec::new();

    let charge_per_elec = Q_ELEC / num_elecs as f64;

    let nuc = Nucleus {
        mass: 2_000., // todo temp?
        charge: -Q_ELEC,
        posit: Vec3::new_zero(),
    };

    let elecs = make_initial_elecs(n_elecs);

    for snap_i in 0..n_timesteps {
        let len = elecs.len();
        for elec_acted_on in 0..len {
            let mut a = Vec3::new_zero();

            // Force of other elecs on this elec.
            for elec_actor in 0..len {
                // Split the slice to get mutable and immutable elements
                // let (acted_on, actor) = elecs.split_at_mut(i);

                if elec_acted_on == elec_actor {
                    continue;
                }

                a += accel_coulomb(
                    elecs[elec_acted_on].posit,
                    elecs[elec_actor].posit,
                    Q_ELEC,
                    Q_ELEC,
                    M_ELEC,
                );
            }

            // Nuc force on this elec.
            a += accel_coulomb(
                elecs[elec_acted_on].posit,
                nuc.posit,
                Q_ELEC,
                -Q_ELEC,
                M_ELEC,
            );
            elecs[elec_acted_on].a = a;
        }

        // todo: Euler for now; improve.
        // for elec in &mut elecs {
        //     elec.v += elec.a * dt;
        //     elec.posit += elec.v * dt;
        // }
        integrate_rk4(&mut elecs, dt);

        snapshots.push(SnapShot {
            time: snap_i as f64 * dt,
            elec_posits: elecs.iter().map(|e| Vec3f32::new(e.posit.x as f32, e.posit.y as f32, e.posit.z as f32)).collect(),
            nuc_posits: vec![Vec3f32::new(nuc.posit.x as f32, nuc.posit.y as f32, nuc.posit.z as f32)]
        })
    }

    snapshots
}

/// Render, by logging to Rerun.
fn render(snapshots: &[SnapShot]) -> Result<(), RecordingStreamError> {
    let rec = rerun::RecordingStreamBuilder::new("rerun_example_minimal").spawn()?;

    for snap in snapshots {
        let mut positions: Vec<Position3D> = snap
            .elec_posits
            .iter()
            .map(|s| Position3D::new(s.x, s.y, s.z))
            .collect();

        for nuc_posit in &snap.nuc_posits {
            positions.push(Position3D::new(nuc_posit.x, nuc_posit.y, nuc_posit.z))
        }


        let posits2 = grid(glam::Vec3::splat(-10.0), glam::Vec3::splat(10.0), 10);
        let colors = grid(glam::Vec3::ZERO, glam::Vec3::splat(255.0), 10)
            .map(|v| rerun::Color::from_rgb(v.x as u8, v.y as u8, v.z as u8));

        let points = Points3D::new(positions)
            // let points = Points3D::new(posits2)
            //     .with_colors(colors)
            .with_radii([0.5]);

        // let points = Points3D {
        //     positions,
        //     radii: Some()
        //     colors: None,
        //     labels: None,
        //     class_ids: None,
        //     keypoint_ids: None,
        // };

        // let rec = RecordingStream::global(rerun::StoreKind::Recording).unwrap();

        rec.log(format!("Elecs T:{}", snap.time), &points)?;
    }

    Ok(())
}

fn main() {
    // todo: Statically allocate?
    println!("Building snapshots...");
    let snapshots = run(100, 1_000, 1.);
    println!("Complete. Rendering...");

    render(&snapshots);

    // todo: Render snapshots.
}
