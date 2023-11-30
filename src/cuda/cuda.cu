// #include <math.h>
#include <initializer_list>

// todo: Header file.
#include "util.cu"
// #include "util.cuh"


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

    dtype exp_term = std::exp(-xi * r / n);

    uint16_t lg_l = n - l - 1;
    uint16_t lg_r = 2 * l + 1;
    dtype lg_input = 2.f * r / n;

    dtype polynomial_term = std::pow(2.f * r / n, l) * laguerre(lg_l, lg_r, lg_input);

    return norm_term(n, l)
        * polynomial_term
        * exp_term;
}

__device__
double sto_val_f64(double3 posit_sample, double3 posit_nuc, double xi, uint16_t n, uint16_t l) {
    double r = calc_dist_f64(posit_sample, posit_nuc);

    double exp_term = std::exp(-xi * r / n);

    uint16_t lg_l = n - l - 1;
    uint16_t lg_r = 2 * l + 1;
    double lg_input = 2.f * r / n;

    double polynomial_term = std::pow(2.f * r / n, l) * laguerre(lg_l, lg_r, lg_input);

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

// We use `double` here, due to numerical problems with `float`.
__device__
dtype find_psi_pp_num(
    dtype3 posit_sample_,
    dtype3 posit_nuc_,
//     dtype xi,
//     double3 posit_sample_,
//     double3 posit_nuc_,
    double xi,
    uint16_t n,
    uint16_t l,
//     dtype psi_sample_loc
    double psi_sample_loc
) {
//     dtype3 x_prev;
//     dtype3 x_next;
//     dtype3 y_prev;
//     dtype3 y_next;
//     dtype3 z_prev;
//     dtype3 z_next;

    double3 posit_sample;
    double3 posit_nuc;

   posit_sample.x = static_cast<double>(posit_sample_.x);
   posit_sample.y = static_cast<double>(posit_sample_.y);
   posit_sample.z = static_cast<double>(posit_sample_.z);
   posit_nuc.x = static_cast<double>(posit_nuc_.x);
   posit_nuc.y = static_cast<double>(posit_nuc_.y);
   posit_nuc.z = static_cast<double>(posit_nuc_.z);

    double3 x_prev;
    double3 x_next;
    double3 y_prev;
    double3 y_next;
    double3 z_prev;
    double3 z_next;

    x_prev.x = posit_sample.x - H;
    x_prev.y = posit_sample.y;
    x_prev.z = posit_sample.z;

    x_next.x = posit_sample.x + H;
    x_next.y = posit_sample.y;
    x_next.z = posit_sample.z;

    y_prev.x = posit_sample.x;
    y_prev.y = posit_sample.y - H;
    y_prev.z = posit_sample.z;

    y_next.x = posit_sample.x;
    y_next.y = posit_sample.y + H;
    y_next.z = posit_sample.z;

    z_prev.x = posit_sample.x;
    z_prev.y = posit_sample.y;
    z_prev.z = posit_sample.z - H;

    z_next.x = posit_sample.x;
    z_next.y = posit_sample.y;
    z_next.z = posit_sample.z + H;

    dtype psi_x_prev = sto_val_f64(x_prev, posit_nuc, xi, n, l);
    dtype psi_x_next = sto_val_f64(x_next, posit_nuc, xi, n, l);
    dtype psi_y_prev = sto_val_f64(y_prev, posit_nuc, xi, n, l);
    dtype psi_y_next = sto_val_f64(y_next, posit_nuc, xi, n, l);
    dtype psi_z_prev = sto_val_f64(z_prev, posit_nuc, xi, n, l);
    dtype psi_z_next = sto_val_f64(z_next, posit_nuc, xi, n, l);


    return static_cast<float>(psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next
//         - psi_sample_loc * 6.f)
        - psi_sample_loc * 6.)
        / H_SQ;
}

// In this approach, we parallelize operations per sample, but run the
// charge computations in serial, due to the cumulative addition step. This appears
// to be much faster in practice, likely due to the addition being offloaded
// to the CPU in the other approach.
extern "C" __global__
void coulomb_kernel(
    float *out,
    float3 *posits_charge,
    float3 *posits_sample,
    float *charges,
    size_t N_charges,
    size_t N_samples
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_sample = index; i_sample < N_samples; i_sample += stride) {
        // Compute the sum serially, as it may not be possible to naively apply it in parallel,
        // and we may still be saturating GPU cores given the large number of samples.
        for (size_t i_charge = 0; i_charge < N_charges; i_charge++) {
            float3 posit_charge = posits_charge[i_charge];
            float3 posit_sample = posits_sample[i_sample];

            if (i_sample < N_samples) {
                out[i_sample] += coulomb(posit_charge, posit_sample, charges[i_charge]);
            }
        }
    }
}

// Note that this is for the radial component only, with n=1. Real.
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
// __launch_bounds__(256, 2)
extern "C" __global__
void sto_val_or_deriv_kernel(
    dtype *out,
    dtype3 *posits_sample,
    dtype3 posit_nuc,
    dtype xi,
    uint16_t n,
    bool deriv,
    size_t N_samples
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = index; i < N_samples; i += stride) {
        if (deriv == true) {
            if (n >= 2) {
                // todo: Ideally, don't re-calc on pt here: Pass in, since you've likely
                // todo already calculated it.
                dtype psi_on_pt = sto_val(posits_sample[i], posit_nuc, xi, n, 0);
                out[i] = find_psi_pp_num(posits_sample[i], posit_nuc, xi, n, 0, psi_on_pt);
            } else {
                out[i] = sto_second_deriv(posits_sample[i], posit_nuc, xi, n, 0);
            }
        } else {
            out[i] = sto_val(posits_sample[i], posit_nuc, xi, n, 0);
        }
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

    for (size_t i = index; i < N_samples; i += stride) {
        out_val[i] = sto_val(posits_sample[i], posit_nuc, xi, n, 0);

        if (n >= 2) {
            out_second_deriv[i] = find_psi_pp_num(posits_sample[i], posit_nuc, xi, n, 0, out_val[i]);
        } else {
            out_second_deriv[i] = sto_second_deriv(posits_sample[i], posit_nuc, xi, n, 0);
        }
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
