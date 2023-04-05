//! This module contains code we currently don't use, but may use later. it's not attached
//! to the library or binary, so code here doesn't have to compile.


/// Initialize a wave function using a charge-centric coordinate system, using RBF
/// interpolation.
fn init_wf_rbf(rbf: &Rbf, charges: &[(Vec3, f64)], bases: &[Basis], E: f64) {
    // todo: Start RBF testing using its own grid

    let mut psi_pp_calc_rbf = Vec::new();
    let mut psi_pp_meas_rbf = Vec::new();

    for (i, sample_pt) in rbf.obs_points.iter().enumerate() {
        let psi_sample = rbf.fn_vals[i];

        // calc(psi: &Arr3d, V: &Arr3dReal, E: f64, i: usize, j: usize, k: usize) -> Cplx {

        let V_sample = {
            let mut result = 0.;
            for (posit_charge, charge_amt) in charges.iter() {
                result += V_coulomb(*posit_charge, *sample_pt, *charge_amt);
            }

            result
        };

        let calc = psi_sample * (E - V_sample) * eigen_fns::KE_COEFF;

        psi_pp_calc_rbf.push(calc);
        psi_pp_meas_rbf.push(num_diff::find_ψ_pp_meas_fm_rbf(
            *sample_pt,
            Cplx::from_real(psi_sample),
            &rbf,
        ));
    }

    println!(
        "Comp1: {:?}, {:?}",
        psi_pp_calc_rbf[100], psi_pp_meas_rbf[100]
    );
    println!(
        "Comp2: {:?}, {:?}",
        psi_pp_calc_rbf[10], psi_pp_meas_rbf[10]
    );
    println!(
        "Comp3: {:?}, {:?}",
        psi_pp_calc_rbf[20], psi_pp_meas_rbf[20]
    );
    println!(
        "Comp4: {:?}, {:?}",
        psi_pp_calc_rbf[30], psi_pp_meas_rbf[30]
    );
    println!(
        "Comp5: {:?}, {:?}",
        psi_pp_calc_rbf[40], psi_pp_meas_rbf[40]
    );

    // Code to test interpolation.
    let b1 = &bases[0];

    let rbf_compare_pts = vec![
        Vec3::new(1., 0., 0.),
        Vec3::new(1., 1., 0.),
        Vec3::new(0., 0., 1.),
        Vec3::new(5., 0.5, 0.5),
        Vec3::new(6., 5., 0.5),
        Vec3::new(6., 0., 0.5),
    ];

    println!("\n");
    for pt in &rbf_compare_pts {
        println!(
            "\nBasis: {:.5} \n Rbf: {:.5}",
            b1.value(*pt).real * b1.weight(),
            rbf.interp_point(*pt),
        );
    }
}


/// Terms for this a sin + exponential basis, in a single dimension. Associated with a single
/// point, where this a reasonable approximation of the wave function locally.
#[derive(Clone, Debug, Default)]
pub struct SinExpBasisTerm {
    // todo: Complex exponential instead of sin? Probably.
    // pub sin_weight: f64, //todo: Taken care of by amplitude?
    pub sin_amp: f64,
    pub sin_freq: f64, // todo: Angular (ω) or normal?
    pub sin_phase: f64,
    pub decaying_exp_amp: f64,
    pub decaying_exp_rate: f64, // ie λ
    // todo: 0-center these functions, and add a shift for decaying exp?
    pub decaying_exp_shift: f64,
    /// Power 2
    pub poly_a: f64,
    /// Power 1
    pub poly_b: f64,
    /// Power 0
    pub poly_c: f64,
}

impl SinExpBasisTerm {
    // todo: Cplx if you use a complex exponential for sin.
    pub fn value(&self, posit: f64) -> f64 {
        self.sin_amp * (self.sin_freq * posit + self.sin_phase).sin()
            + self.decaying_exp_amp * (self.decaying_exp_rate * (posit - self.decaying_exp_shift)).exp() // todo dir of shift?
            + self.poly_a * posit.powi(2) + self.poly_b * posit + self.poly_c
    }

    /// Create a basis point here from neighboring values, in 1 dimension.
    /// todo: Cplx?
    pub fn from_neighbors(v_prev: f64, v_this: f64, v_next: f64, p_this: f64, h: f64) -> Self {
        // Note that for decaying exponentials, the amplitude can be though of as used to define
        // at what value the equation takes at position = 0. Is this also a weight for balancing
        // with the sin term?

        // todo: If it's from an H basis, then I think this truly is teh analytic solution we
        // todo have for these exactly, at all positions. (?)

        //

        // todo: If we're fitting  a segment using interpolation of neighboring psi values,
        // todo is 3 points per axis good to find a fit?
        // todo: center using an additional offset?
        // todo: Do you only need 2 points since there are only 2 unknowns?
        // todo: Maybe you need 3 points, since there is a third unknown: The offset

        // todo: I think this is untenable for just complex exponentials.
        // todo another approach: Construct shiftless exponential from 2 pts
        // todo another: Take another stab at degree-2 polynomial interp.
        // todo: This brings up another Q: Can you treat dimensions independently?

        // todo: Consider having the coeffs defined with the point being = 0, with actual coords
        // todo stored, and applied as an offset. This could mitigate numerical precision issues,
        // todo especially if using f32s on the GPU.

        // 3 unknowns: B, λ, shift. 3 equations, from 3 points.
        // psi(x) = B·e^(-λ(x + shift))
        // psi(x + h) = B·e^(-λ(x + shift + h))
        // psi(x - h) = B·e^(-λ(x + shift - h))

        // A higher exp rate causes the equation to trend towards 0 more quickly as position increases.

        // todo: If you want to make this system centered on 0 for numerical floating point reasons,
        // todo, it may be as simple as setting posit_this here to 0, and offsetting downstream.
        // let posit_this = 0.;

        // https://math.stackexchange.com/questions/680646/get-polynomial-function-from-3-points
        // Good article: https://cohost.org/tomforsyth/post/982199-polynomial-interpola
        let p_prev = p_this - h;
        let p_next = p_this + h;

        let poly_a_num =
            p_prev * (v_next - v_this) + p_this * (v_prev - v_next) + p_next * (v_this - v_prev);

        let poly_a_denom = (p_prev - p_this) * (p_prev - p_next) * (p_this - p_next);

        let a = poly_a_num / poly_a_denom;

        let poly_b = (v_this - v_prev) / (p_this - p_prev) - a * (p_prev + p_this);

        let poly_c = v_prev - a * p_prev.powi(2) - poly_b * p_prev;

        Self {
            poly_a: a,
            poly_b,
            poly_c,
            // decaying_exp_amp: B,
            // decaying_exp_rate: λ,
            // decaying_exp_shift: shift,
            ..Default::default() // No sin term for now.
        }
    }
}

/// Represents a local approximation of the wave function as a combination of terms that construct
/// a sin wave overlaid with a decaying exponential, at a given point.
/// Represents a point as A·sin(ωx + ϕ) + B·e^(-λx), for each of the 3 dimensions X, Y, Z.
///
/// todo note: Maybe polynomial term? It's in the H bases
/// Todo: Complex exponential to replace sin term? To replace *both* terms?
/// todo: Move this to another module if it works out
#[derive(Clone, Debug, Default)]
pub struct SinExpBasisPt {
    pub terms_x: SinExpBasisTerm,
    pub terms_y: SinExpBasisTerm,
    pub terms_z: SinExpBasisTerm,
}

impl SinExpBasisPt {
    // todo: Cplx if you use a complex exponential for sin.
    pub fn value(&self, posit: Vec3) -> f64 {
        self.terms_x.value(posit.x) + self.terms_y.value(posit.y) + self.terms_z.value(posit.z)
    }

    /// Create a basis point from equally-spaced neighbors, in 3 dimensions.
    /// Format of vals is (this, prev, next).
    /// todo: COmplex
    pub fn from_neighbors(
        vals_x: (f64, f64, f64),
        vals_y: (f64, f64, f64),
        vals_z: (f64, f64, f64),
        posit: Vec3,
        h: f64,
    ) -> Self {
        Self {
            terms_x: SinExpBasisTerm::from_neighbors(vals_x.0, vals_x.1, vals_x.2, posit.x, h),
            terms_y: SinExpBasisTerm::from_neighbors(vals_y.0, vals_y.1, vals_y.2, posit.y, h),
            terms_z: SinExpBasisTerm::from_neighbors(vals_z.0, vals_z.1, vals_z.2, posit.z, h),
        }
    }
}

/// todo: Cplx?
/// todo: Input args include indices, or not?
fn _interp_from_sin_exp_basis(
    basis: &Arr3dBasis,
    pt: Vec3,
    i_tla: (usize, usize, usize), // todo: If you keep this apch, bundle
    i_tra: (usize, usize, usize),
    i_bla: (usize, usize, usize),
    i_bra: (usize, usize, usize),
    i_tlf: (usize, usize, usize),
    i_trf: (usize, usize, usize),
    i_blf: (usize, usize, usize),
    i_brf: (usize, usize, usize),
) -> f64 {
    let tla = basis[i_tla.0][i_tla.1][i_tla.2].value(pt);
    let tra = basis[i_tra.0][i_tra.1][i_tra.2].value(pt);
    let bla = basis[i_bla.0][i_bla.1][i_bla.2].value(pt);
    let bra = basis[i_bra.0][i_bra.1][i_bra.2].value(pt);

    let tlf = basis[i_tlf.0][i_tlf.1][i_tlf.2].value(pt);
    let trf = basis[i_trf.0][i_trf.1][i_trf.2].value(pt);
    let blf = basis[i_blf.0][i_blf.1][i_blf.2].value(pt);
    let brf = basis[i_brf.0][i_brf.1][i_brf.2].value(pt);

    // todo

    let tla_weight = 1.;
    let tra_weight = 1.;
    let bra_weight = 1.;
    let bla_weight = 1.;

    let tlf_weight = 1.;
    let trf_weight = 1.;
    let brf_weight = 1.;
    let blf_weight = 1.;

    tla * tla_weight
        + tra * tra_weight
        + bla * bla_weight
        + bra * bra_weight
        + tlf * tlf_weight
        + trf * trf_weight
        + blf * blf_weight
        + brf * brf_weight
}
