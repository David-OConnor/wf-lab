use rand::Rng; // Add this to use the random number generator
use std::f64::consts::TAU;

use lin_alg::f64::Vec3;
use lin_alg::f32::Vec3 as Vec3f32;
use rerun::{RecordingStream, RecordingStreamError, Image, Points3D, demo_util::grid, external::glam, Position3D};

// todo: What if you have total electric charge different from a whole number? Misc elec "particles"
// todo zipping around in the either, to be capture.


// todo: Next, try having the electrons add up to more than 1 charge.
// todo: Try to quantify the elec density, so you can compare it to Schrodinger.

const M_ELEC: f64 = 1.;
const Q_ELEC: f64 = -1.;

// If two particles are closer to each other than this, don't count accel, or cap it.
const MIN_DIST: f64 = 0.001;
// SKip these many frames in rerun, to keep it from hanging.
const RERUN_SKIP_AMOUNT: usize = 200;

// Don't calculate force if particles are farther than this from each other. Computation saver.
// const MAX_DIST: f64 = 0.1;

#[derive(Debug)]
struct SnapShot {
    pub time: f64,
    // todo: To save memory, you could store the snapshots as f32; we only need f64 precision
    // todo during the integration.
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

// (radius, charge_in_radius)
fn density(elec_posits: &[Vec3f32], q_per_elec: f64, center_ref: Vec3f32) -> Vec<(f32, f32)> {
    // Must be ascending radii for the below code to work.
    let per_elec = q_per_elec as f32;

    // Equidistant ish, for now.
    let mut result = vec![
        ((0.0, 0.2), 0.),
         ((0.2, 0.4),0.),
        ((0.4, 0.6),0.),
        ((0.8, 1.), 0.),
         ((1., 1.2),0.),
        ((1.2, 1.4),0.),
         ((1.4, 1.6), 0.),
        ((1.6, 1.8), 0.),
        ((1.8, 2.0), 0.),
        ((2.0, 9999.), 0.),
    ];

    for (r_bound, q) in &mut result {
        for posit in elec_posits {
            let mag = (center_ref - *posit).magnitude();
            if mag < *r_bound.1 && mag > *r_bound.0 {
                *q += per_elec;
            }
        }
    }

    result
}

/// Calculate the Coulomb acceleration on a particle, from a single other particle.
fn accel_coulomb(
    posit_acted_on: Vec3,
    posit_actor: Vec3,
    q_acted_on: f64,
    q_actor: f64,
    mass_acted_on: f64,
) -> Vec3 {
    let posit_diff = posit_acted_on - posit_actor;
    let dist = posit_diff.magnitude();

    let posit_diff_unit = posit_diff / dist;

    // Note: We factor out KC * Q_PROT, since they are present in every calculation.
    // Calculate the Coulomb force between nuclei.

    let f_mag = q_acted_on * q_actor / dist.powi(2);

    if dist < MIN_DIST {
        println!("Min dist: {:?} Force: {:?}", dist, f_mag);
        return Vec3::new_zero();
    }

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

    for i in 0..n_elecs {
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

fn run(n_elecs: usize, n_timesteps: usize, dt: f64) -> Vec<SnapShot> {
    let mut snapshots = Vec::new();

    let charge_per_elec = Q_ELEC / n_elecs as f64;

    let nuc = Nucleus {
        mass: 2_000., // todo temp?
        charge: -Q_ELEC,
        posit: Vec3::new_zero(),
    };

    let mut elecs = make_initial_elecs(n_elecs);

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
                    charge_per_elec,
                    charge_per_elec,
                    M_ELEC,
                );
            }

            // Nuc force on this elec.
            a += accel_coulomb(
                elecs[elec_acted_on].posit,
                nuc.posit,
                charge_per_elec,
                -Q_ELEC,
                M_ELEC,
            );
            elecs[elec_acted_on].a = a;
        }

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

    for (i, snap) in snapshots.iter().enumerate() {
        if i % RERUN_SKIP_AMOUNT != 0 {
            continue
        }

        let mut positions: Vec<Position3D> = snap
            .elec_posits
            .iter()
            .map(|s| Position3D::new(s.x, s.y, s.z))
            .collect();

        for nuc_posit in &snap.nuc_posits {
            positions.push(Position3D::new(nuc_posit.x, nuc_posit.y, nuc_posit.z))
        }

        let colors = grid(glam::Vec3::ZERO, glam::Vec3::splat(255.0), 10)
            .map(|v| rerun::Color::from_rgb(v.x as u8, v.y as u8, v.z as u8));

        let points = Points3D::new(positions)
            // let points = Points3D::new(posits2)
            //     .with_colors(colors)
            .with_radii([0.02]);

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

        // rec.set_timepoint(snap.time);
        rec.set_time_seconds("test", snap.time);
    }

    Ok(())
}

fn main() {
    // todo: Statically allocate?
    println!("Building snapshots...");
    let snapshots = run(100, 50_000, 0.005);
    println!("Complete. Rendering...");

    render(&snapshots);

    // todo: Render snapshots.
}
