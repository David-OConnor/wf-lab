use lin_alg::f64::Vec3;

use super::{SphericalHarmonic, A_0};
use crate::{
    complex_nums::Cplx,
    eigen_fns::KE_COEFF,
    util::{self, factorial, EPS_DIV0},
};

/// See Sebens: Electric Charge Density, equation 24
#[derive(Clone, Debug)]
pub struct Sto {
    pub posit: Vec3,
    pub n: u16,
    pub xi: f64,
    pub harmonic: SphericalHarmonic,
    pub weight: f64,
    pub charge_id: usize,
}

impl Sto {
    pub fn value(&self, posit_sample: Vec3) -> Cplx {
        Cplx::from_real(self.radial(posit_sample)) * self.angular(posit_sample)
    }

    /// Calculate the angular portion of a basis function's value at a given point.
    /// Does not include weight.
    /// https://quantummechanics.ucsd.edu/ph130a/130_notes/node233.html
    pub fn angular(&self, posit_sample: Vec3) -> Cplx {
        let diff = posit_sample - self.posit;

        let diff = self.harmonic.orientation.inverse().rotate_vec(diff);

        // We use the "physics" (ISO) convention described here, where phi
        // covers the full way around, and theta goes half a turn from
        // the top.

        let θ = (diff.x.powi(2) + diff.y.powi(2)).sqrt().atan2(diff.z);
        let ϕ = diff.y.atan2(diff.x);

        self.harmonic.value(θ, ϕ)
    }

    fn norm_term_lut(n: u16, l: u16, xi: f64) -> f64 {
        return 2.;

        // todo: temp
        if n == 1 {
            return 2.;
        }
        if n == 2 {
            return 1.;
        }

        1.
    }

    /// todo: Should this be a function of xi too? Very likely.
    /// todo: This seems to work for n=1, but fails at n=2.
    fn norm_term(n: u16, l: u16) -> f64 {
        // return Self::norm_term_lut(n, l, 1.); // todo temp
        let nf = n as f64;

        let norm_term_num = (2. / nf).powi(3) * factorial(n - l - 1) as f64;
        let norm_term_denom = (2 * n as u64 * factorial(n + l).pow(3)) as f64;

        // println!("NT {:?}", (norm_term_num / norm_term_denom).sqrt());
        (norm_term_num / norm_term_denom).sqrt()
    }

    /// From I. Complete and orthonormal sets of exponential-type orbitals
    /// with noninteger principal quantum numbers
    /// https://arxiv.org/pdf/2205.02317.pdf
    /// This is currently based off Equation 30 in that paper.
    /// See also `scripts/sto_test.py`.
    fn sto_generalized(&self, posit_sample: Vec3) -> f64 {
        let r = (posit_sample - self.posit).magnitude();

        let n = self.n;
        let l = self.harmonic.l;

        // Instead of xi, we will substitute in eps.
        let eps = self.xi;

        let nf = n as f64;

        let n_st = nf + eps;
        let l_st = self.harmonic.l as f64 + eps;

        let eps3 = 1; // todo: A/R. What is its range? f64

        // let norm_num = (2. * xi).powi(3) * gamma(n - l - eps3 + 1);
        // let norm_denom = gamma(n + l + eps3 + 1);
        //
        // let norm_term = (norm_num / norm_denom).sqrt();

        // Diff between this and radial: Radial includes xi in the exp. (And n vice n*)
        let exp_term = (-r / nf).exp();

        // todo: Here and in `radial`: Cache `make_laguere. You don't need to run it each posit.

        // todo: See note on 2*eps3 vs 1*eps3.
        // let L = util::make_laguerre(n_st - l_st - eps3, 2 * l + 2 * eps3);
        // todo: Fractional laguerre? required for eps3 non-int.

        // todo: Confirm your floating point math here.
        let L = util::make_laguerre2(n - l - eps3, (2 * l) as f64 + eps3 as f64);

        let polynomial_term = (2. * r / n_st).powf(l_st + eps3 as f64 - 1.) * L(2. * r / n_st);

        // Note: This norm term is wrong.
        Self::norm_term(n, l) * polynomial_term * exp_term
    }

    pub fn radial(&self, posit_sample: Vec3) -> f64 {
        // todo tmep
        // return self.sto_generalized(posit_sample);

        // todo: This currently ignores the spherical harmonic part; add that!
        let r = (posit_sample - self.posit).magnitude();

        let n = self.n;
        let l = self.harmonic.l;
        let nf = n as f64;

        let exp_term = (-self.xi * r / (nf * A_0)).exp();

        // Note: [The OP here](https://chemistry.stackexchange.com/questions/164478/why-are-slater-type-orbitals-used-for-atomic-calculations-instead-of-hydrogen-li)
        // contains a different form; worth examining. It includes both xi and the Laguerre term,
        // which most forms I've found don't.

        let L = util::make_laguerre(n - l - 1, 2 * l + 1);

        // let polynomial_term = (2. * r / (nf * A_0)).powi(l.into()) * L(2. * r / (nf * A_0));
        let polynomial_term = (2. * r / nf).powi(l.into()) * L(2. * r / nf);

        // n_L = n - l - 1
        // b = 2r / (n*A0)
        // α = 2l + 1

        // n_L=0: L(x) = 1.
        // n_L=1: L(x) = α + 1. - b,
        // n_L=2: L(x) = b.powi(2) / 2. - (α + 2.) * b + (α + 1.) * (α + 2.) / 2.,

        // If n=1, l=0, L= 1.
        // If n=2, l=0, L = 2. - b
        // If n=2, l=1, L = 1.
        // If n=3, l=0, L = b^2 / 2 - 3b + 3

        // For the second derivative, try this in Wolfram Alpha:
        // n=1, l=0
        // `second derivative of exp(-r * xi / n) with respect to x where r=sqrt(x^2 + y^2 + z^2)`

        // n=2, l=0
        // `second derivative of exp(-r * xi / n) * (2-b) with respect to x where r=sqrt(x^2 + y^2 + z^2) and b=2*r/n`

        // n=2, l=1
        // `second derivative of 2r/n * exp(-r * xi / n) with respect to x where r=sqrt(x^2 + y^2 + z^2)
        // Or, with n in place: `second derivative of r * exp(-r * xi / 2) with respect to x where r=sqrt(x^2 + y^2 + z^2)`

        // n=3, l=0
        // todo: Redo this.
        // `second derivative of exp(-r * xi / n) * (b^2 / 2 - 3b + 3) with respect to x where r=sqrt(x^2 + y^2 + z^2) and b=2*r/n`

        // Example ChatGPT query:
        // "Please calculate the second derivative of "(see above)". Please output the result in
        // the form of a rust function. Please use the variable `r` in the resulting code when
        // possible, vice splitting it into its x, y, and z components."
        //
        // Note: In some cases when using Wolfram Alpha, don't define `b=` etc - just put in the full equation, or only use r=...;
        // You can often copy things from WA to ChatGPT, to get them in the form of a rust fn.

        Self::norm_term(n, l) * polynomial_term * exp_term
    }

    /// Analytic first derivative, with respect to x.
    pub fn dx(&self, posit_sample: Vec3) -> Cplx {
        // todo
        Cplx::new_zero()
    }

    /// Analytic first derivative, with respect to y.
    pub fn dy(&self, posit_sample: Vec3) -> Cplx {
        Cplx::new_zero()
    }

    /// Analytic first derivative, with respect to z.
    pub fn dz(&self, posit_sample: Vec3) -> Cplx {
        Cplx::new_zero()
    }

    /// Analytic second derivative, with respect to x.
    pub fn d2x(&self, posit_sample: Vec3) -> Cplx {
        Cplx::new_zero()
    }

    /// Analytic second derivative, with respect to y.
    pub fn d2y(&self, posit_sample: Vec3) -> Cplx {
        Cplx::new_zero()
    }

    /// Analytic second derivative, with respect to z.
    pub fn d2z(&self, posit_sample: Vec3) -> Cplx {
        Cplx::new_zero()
    }

    /// Analytic second derivative using analytic basis functions. (Sum of sub-components)
    /// See OneNote: `Exploring the WF, part 6`.
    ///
    /// See also `value()`, for information on appropriate ChatGPT queries to calculate
    /// this analytic second derivative.
    pub fn second_deriv(&self, posit_sample: Vec3) -> Cplx {
        // todo: This currently ignores the spherical harmonic part; add that!

        // Enter this in Wolfram Alpha: `second derivative of (1/sqrt(pi)) * \xi^(3/2) * r^(n-1) * e^(-\xi * r / n) with respect to x where r=sqrt(x^2 + y^2 + z^2)`

        // code-shorteners, to help make these long equations easier to read. And precomputing  some terms.
        let diff = posit_sample - self.posit;
        let r_sq = diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2);

        if r_sq < EPS_DIV0 {
            return Cplx::new_zero();
        }
        let r = r_sq.sqrt();

        let n = self.n;
        let l = self.harmonic.l;
        let nf = n as f64;
        let xi = self.xi;

        let exp_term = (-xi * r / nf).exp();

        let mut result = 0.;

        // Each part is the second deriv WRT to an orthogonal axis.
        for x in [diff.x, diff.y, diff.z] {
            let x_sq = x.powi(2);

            if n == 1 {
                let term1 = xi.powi(2) * x_sq * exp_term / r_sq;
                let term2 = xi * x_sq * exp_term / r_sq.powf(1.5);
                let term3 = -xi * exp_term / r;

                result += term1 + term2 + term3
            } else if n == 2 && l == 0 {
                let term1 = (2.0 - r)
                    * ((xi.powi(2) * x_sq * exp_term) / (4. * r_sq)
                        + (xi * x_sq * exp_term) / (2. * r_sq.powf(1.5))
                        - (xi * exp_term) / (2. * r));

                let term2 = (4. * xi * x_sq * exp_term) / (4. * r_sq);

                let term3 = -(2. * (1. / r - x_sq / r_sq.powf(1.5)) * exp_term) / 2.;

                result += term1 + term2 + term3
            } else if n == 2 && l == 1 {
                let term1 = (xi.powi(2) * x.powi(2) * exp_term) / (4. * r_sq);
                let term2 = (xi * x.powi(2) * exp_term) / (2. * r_sq.powf(1.5));
                let term3 = (xi * exp_term) / (2. * r);
                let term4 = (-xi * x.powi(2) * exp_term) / r_sq;
                let term5 = (1.0 / r - x.powi(2) / r_sq.powf(1.5)) * exp_term;

                result += r * (term1 + term2 - term3) - term4 + term5
            } else {
                unimplemented!("Second deriv unimplemented for this n and l.")
            }
        }

        Cplx::from_real(Self::norm_term(n, l) * result)
    }

    // todo: analytic first derivs, and individual second deriv components.

    // /// Saves some minor computations over calculating them individually, due to
    // /// eliminated terms. Of note, the exponential term cancels out.
    // ///
    // /// Note: It appears that this is always real (Hermitian eigenvalues?)
    // /// Important: We currently don't use this: It may be incompatible with the way we
    // /// mix bases, due to OOP of adding and dividing mattering.
    // pub fn psi_pp_div_psi(&self, posit_sample: Vec3) -> f64 {
    //     let r = (posit_sample - self.posit).magnitude();
    //
    //     let xi = self.xi;
    //
    //     // this is our ideal approach
    //     if self.n == 1 {
    //         xi.powi(2) - 2. * xi / r
    //     } else if self.n == 2 && self.harmonic.l == 0 {
    //         xi.powi(2) / 4. + xi / (2. - r) - xi / r - 2. / ((2. - r) * r)
    //     } else if self.n == 2 && self.harmonic.l == 1 {
    //         0.
    //     } else {
    //         unimplemented!()
    //     }
    // }

    pub fn V_p_from_psi(&self, posit_sample: Vec3) -> f64 {
        // From Wolfram alpha // todo: What query? What is this?
        // todo: Update A/R if you need based on new sto formula?
        let diff = posit_sample - self.posit;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        let mut result = 0.;

        let xi = self.xi;

        // todo: DRY here
        result +=
            1. / r.powi(5) * xi * diff.x * (diff.y.powi(2) + diff.z.powi(2)) * (2. * xi * r + 3.);
        result +=
            1. / r.powi(5) * xi * diff.y * (diff.x.powi(2) + diff.z.powi(2)) * (2. * xi * r + 3.);
        result +=
            1. / r.powi(5) * xi * diff.z * (diff.x.powi(2) + diff.y.powi(2)) * (2. * xi * r + 3.);

        let radial = result;

        radial * KE_COEFF
    }

    pub fn V_pp_from_psi(&self, posit_sample: Vec3) -> f64 {
        // todo: What query? What is this?
        // todo: Update A/R if you need based on new sto formula?
        let diff = posit_sample - self.posit;
        let r = (diff.x.powi(2) + diff.y.powi(2) + diff.z.powi(2)).sqrt();

        let mut result = 0.;

        let xi = self.xi;
        let x2 = diff.x.powi(2);
        let y2 = diff.y.powi(2);
        let z2 = diff.z.powi(2);

        // todo DRY here
        result += 1. / r.powi(7)
            * xi
            * (y2 + z2)
            * ((y2 + z2) * (2. * xi * r + 3.) - 6. * x2 * (xi * r + 2.));
        result += 1. / r.powi(7)
            * xi
            * (x2 + z2)
            * ((x2 + z2) * (2. * xi * r + 3.) - 6. * y2 * (xi * r + 2.));
        result += 1. / r.powi(7)
            * xi
            * (x2 + y2)
            * ((x2 + y2) * (2. * xi * r + 3.) - 6. * z2 * (xi * r + 2.));

        let radial = result;

        radial * KE_COEFF
    }
}
