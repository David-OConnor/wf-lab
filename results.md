# This file contains wave functions, and related results.

## Helium, from STOs
Method: Manual. Function used:
```rust
pub fn radial(&self, posit_sample: Vec3) -> f64 {
        // todo: This currently ignores the spherical harmonic part; add that!
        let r = (posit_sample - self.posit).magnitude();
        let n = self.n;
        let l = self.harmonic.l;
        let nf = n as f64;
        let exp_term = (-self.xi * r / (nf * A_0)).exp();
        let L = util::make_laguerre(n - l - 1, 2 * l + 1);
        let polynomial_term = (2. * r / (nf * A_0)).powi(l.into()) * L(2. * r / (nf * A_0));
        Self::norm_term(n, l) * polynomial_term * exp_term
    }
```

From STOs, with no spherical harmonic.
ξ: 1.0 Weight: 0.45
ξ: 2.0 Weight: 0.0
ξ: 3.0 Weight: -0.21
ξ: 4.0 Weight: -0.01
ξ: 5.0 Weight: -0.32
ξ: 6.0 Weight: 0.10
ξ: 8.0 Weight: -0.61
ξ: 10.0 Weight: -0.05
