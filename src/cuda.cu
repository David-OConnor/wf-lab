// #include <math.h>
#include <initializer_list>

// todo: Header file.
#include "util.cu"


__device__
dtype norm_term(uint16_t n, uint16_t l) {
    double norm_term_num = std::pow(2.0f / n, 3) * factorial(n - l - 1);
    double norm_term_denom = std::pow(2 * n * factorial(n + l), 3);
    return std::sqrt(norm_term_num / norm_term_denom);
}

// Note that this is for the radial component only, with n=1. Real. See CPU side for a ref.
__device__
dtype sto_val(dtype3 posit_sample, dtype3 posit_nuc, dtype xi, uint16_t n, uint16_t l) {
//     dtype N = PI_SQRT_INV * std::pow(xi, 1.5f);
//
    dtype r = calc_dist(posit_sample, posit_nuc);
//
//     dtype radial = N * std::pow(r, n - 1) * std::exp(-xi * r / n);
//     return radial;

    dtype exp_term = std::exp(-r / (n * A_0));

    uint16_t lg_l = n - l - 1;
    uint16_t lg_r = 2 * l + 1;
    dtype lg_input = 2.f * r / n;

    dtype polynomial_term = std::pow(2.f * r / n, l) * laguerre(lg_l, lg_r, lg_input);

    return norm_term(n, l)
        * polynomial_term
        * exp_term;
}

// Note that this is for the radial component only. Real. See CPU side for a ref.
__device__
dtype sto_second_deriv(dtype3 posit_sample, dtype3 posit_nuc, dtype xi, uint16_t n, uint16_t l) {
    dtype3 diff;
    diff.x = posit_sample.x - posit_nuc.x;
    diff.y = posit_sample.y - posit_nuc.y;
    diff.z = posit_sample.z - posit_nuc.z;

    dtype r_sq = std::pow(diff.x, 2) + std::pow(diff.y, 2) + std::pow(diff.z, 2);

    if (r_sq < EPS_DIV0) {
        return 0.f;
    }

    dtype r = std::sqrt(r_sq);

    dtype exp_term = std::exp(-xi * r / n);
    dtype laguerre_param = 2.f * r / n;

    dtype result = 0.;

    for (auto x : {diff.x, diff.y, diff.z}) {
        double x_sq = std::pow(x, 2);

        if (n == 1 && l == 0) {
            double term1 = std::pow(xi, 2) * x_sq * exp_term / (std::pow(n, 2) * r_sq);
            double term2 = xi * x_sq * exp_term / (n * std::pow(r_sq, 1.5f));
            double term3 = -xi * exp_term / (n * r);
        
            result += term1 + term2 + term3;
        } else if (n == 2 && l == 0) {
            result += exp_term * (2.f - laguerre_param);
        }
    }

    return norm_term(n, l) * result;
}


// In this approach, we parallelize operations per sample, but run the
// charge computations in serial, due to the cumulative addition step. This appears
// to be much faster in practice, likely due to the addition being offloaded
// to the CPU in the other approach.
extern "C" __global__
void coulomb_kernel(
    dtype *out,
    dtype3 *posits_charge,
    dtype3 *posits_sample,
    dtype *charges,
    size_t N_charges,
    size_t N_samples
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_sample = index; i_sample < N_samples; i_sample += stride) {
        // Compute the sum serially, as it may not be possible to naively apply it in parallel,
        // and we may still be saturating GPU cores given the large number of samples.
        for (size_t i_charge = 0; i_charge < N_charges; i_charge++) {
            dtype3 posit_charge = posits_charge[i_charge];
            dtype3 posit_sample = posits_sample[i_sample];

            if (i_sample < N_samples) {
                out[i_sample] += coulomb(posit_charge, posit_sample, charges[i_charge]);
            }
        }
    }
}


// Note that this is for the radial component only, with n=1. Real.
extern "C" __global__
void sto_val_or_deriv_kernel(
    dtype *out,
    dtype3 *posits_sample,
    dtype3 posit_nuc,
    dtype xi,
    uint16_t n,
//     bool deriv,
    size_t N_samples
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    bool deriv = false; // todo TS OUT OF RESOURCES.

    for (size_t i = index; i < N_samples; i += stride) {
        if (deriv == true) {
            out[i] = sto_second_deriv(posits_sample[i], posit_nuc, xi, n, 0);
        } else {
            out[i] = sto_val(posits_sample[i], posit_nuc, xi, n, 0);
        }
    }
}

// Temp workaround for out-of-resources error on the combined or or and ones.
extern "C" __global__
void sto_deriv_kernel(
    dtype *out,
    dtype3 *posits_sample,
    dtype3 posit_nuc,
    dtype xi,
    uint16_t n,
    size_t N_samples
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_samples; i += stride) {
        out[i] = sto_second_deriv(posits_sample[i], posit_nuc, xi, n, 0);
    }
}

// We combine value and derivative computations here to reduce IO between host and device.
extern "C" __global__
void sto_val_deriv_kernel(
    dtype *out_val,
    dtype *out_second_deriv,
    dtype3 *posits_sample,
    dtype3 posit_nuc,
    dtype xi,
    uint16_t n,
    size_t N_samples
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_sample = index; i_sample < N_samples; i_sample += stride) {
        out_val[i_sample] = sto_val(posits_sample[i_sample], posit_nuc, xi, n, 0);
        out_second_deriv[i_sample] = sto_second_deriv(posits_sample[i_sample], posit_nuc, xi, n, 0);
    }
}

extern "C" __global__
void sto_val_multiple_bases_kernel(
    dtype *out_val,
    dtype3 *posits_sample,
    dtype3 *posits_nuc,
    dtype *xis,
    uint16_t *n,
    dtype *weights,
    size_t N_samples,
    size_t N_bases
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_sample = index; i_sample < N_samples; i_sample += stride) {
        for (size_t i_basis = 0; i_basis < N_bases; i_basis++) {
            out_val[i_sample] += sto_val(posits_sample[i_sample], posits_nuc[i_basis], xis[i_basis], n[i_basis], 0) * weights[i_basis];
        }
    }
}


// Combines these 2 operations, as they're likely to be done on the same data set.
extern "C" __global__
void sto_val_deriv_multiple_bases_kernel(
    dtype *out_val,
    dtype *out_second_deriv,
    dtype3 *posits_sample,
    dtype3 posit_nuc,
    dtype *xis,
    uint16_t *n,
    dtype *weights,
    size_t N_samples,
    size_t N_bases
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_sample = index; i_sample < N_samples; i_sample += stride) {
        for (size_t i_basis = 0; i_basis < N_bases; i_basis++) {
            out_val[i_sample] += sto_val(posits_sample[i_sample], posit_nuc, xis[i_basis], n[i_basis], 0) * weights[i_basis];
            out_second_deriv[i_sample] += sto_second_deriv(posits_sample[i_sample], posit_nuc, xis[i_basis], n[i_basis], 0) * weights[i_basis];
        }
    }
}
