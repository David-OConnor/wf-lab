// #include <math.h>
#include <initializer_list>

// todo: Consider separate files for coulomb, and STO calculations.

// https://developer.nvidia.com/blog/even-easier-introduction-cuda/

// Allows easy switching between float and double.
// #define dtype double
// #define dtype3 double3
#define dtype float
#define dtype3 float3

__device__
const dtype SOFTENING_FACTOR = 0.000000000001f;
__device__
const dtype PI_SQRT_INV = 0.5641895835477563f;
// const double PI_SQRT_INV = 1 / std::sqrt(M_PI);

// todo: Cuda's `threadIdx` can be 1D, 2D, or 3D. 2D may be a better fit here.
// 1D with packing/unpacking is fine, but 2D would be perhaps cleaner. Experiment.

// extern "C" __global__ void matmul(dtype* A, dtype* B, dtype* C, int N) {
//     // Syntax example using 2D inputs.
//     size_t ROW = blockIdx.y * blockDim.y + threadIdx.y;
//     size_t COL = blockIdx.x * blockDim.x + threadIdx.x;
//
//     dtype tmpSum = 0;
//
//     if (ROW < N && COL < N) {
//         // each thread computes one element of the block sub-matrix
//         for (size_t i = 0; i < N; i++) {
//             tmpSum += A[ROW * N + i] * B[i * N + COL];
//         }
//     }
//     C[ROW * N + COL] = tmpSum;
// }

__device__
dtype laguerre_0(uint16_t n, dtype alpha, dtype x) {
    return 1.
}

__device__
dtype laguerre_1(uint16_t n, dtype alpha, dtype x) {
    return alpha + 1. - x;
}

__device__
dtype laguerre_2(uint16_t n, dtype alpha, dtype x) {
    return std::exp(x, 2) / 2. - (alpha + 2.) * x + (alpha + 1.) * (alpha + 2.) / 2.;
}
// todo: More A/R.


__device__
dtype calc_dist(dtype3 point0, dtype3 point1) {
    dtype3 diff;
    diff.x = point0.x - point1.x;
    diff.y = point0.y - point1.y;
    diff.z = point0.z - point1.z;

    return std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
}


__device__
dtype coulomb(dtype3 q0, dtype3 q1, dtype charge) {
    dtype r = calc_dist(q0, q1);

    return 1.f * charge / (r + SOFTENING_FACTOR);
}


// Note that this is for the radial component only, with n=1. Real. See CPU side for a ref.
__device__
dtype sto_val(dtype3 posit_sample, dtype3 posit_nuc, dtype xi, uint16_t n) {
    dtype N = PI_SQRT_INV * std::pow(xi, 1.5f);

    dtype r = calc_dist(posit_sample, posit_nuc);

    dtype radial = N * std::pow(r, n - 1) * std::exp(-xi * r / n);
    return radial;
}


// Note that this is for the radial component only, with n=1. Real. See CPU side for a ref.
// todo: Deprecate A/R
__device__
dtype sto_second_deriv_n1(dtype3 posit_sample, dtype3 posit_nuc, dtype xi) {
    dtype N = PI_SQRT_INV * pow(xi, 1.5);

    dtype3 diff;
    diff.x = posit_sample.x - posit_nuc.x;
    diff.y = posit_sample.y - posit_nuc.y;
    diff.z = posit_sample.z - posit_nuc.z;

    dtype r = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    dtype exp_term = std::exp(-xi * r);

    dtype result = 0.;

    for (dtype x : {diff.x, diff.y, diff.z}) {
        result += std::pow(xi, 2) * std::pow(x, 2) * exp_term / std::pow(r, 2);
        result += xi * std::pow(x, 2) * exp_term / std::pow(r, 3);
        result -= xi * exp_term / r;
    }

    dtype radial = N * result;
    return radial;
}

// Note that this is for the radial component only. Real. See CPU side for a ref.
__device__
dtype sto_second_deriv(dtype3 posit_sample, dtype3 posit_nuc, dtype xi, uint16_t n) {
    dtype3 diff;
    diff.x = posit_sample.x - posit_nuc.x;
    diff.y = posit_sample.y - posit_nuc.y;
    diff.z = posit_sample.z - posit_nuc.z;

    dtype r_sq = std::pow(diff.x, 2) + std::pow(diff.y, 2) + std::pow(diff.z, 2);
    dtype r = std::sqrt(r_sq);

    dtype exp_term = std::exp(-xi * r / n);

    dtype N = PI_SQRT_INV * std::pow(xi, 1.5f);
    dtype term0_coeff = N * std::pow(r_sq, (n - 1.f) / 2.f);

    dtype result = 0.f;

    // Each part is the second deriv WRT to an orthogonal axis.
    for (dtype x : {diff.x, diff.y, diff.z}) {
        dtype x_sq = std::pow(x, 2);

        dtype term0 = std::pow(xi, 2) * std::pow(x, 2) * exp_term / (std::pow(n, 2) * r_sq)
            + xi * x_sq * exp_term / (n * std::pow(r, 3))
            - xi * exp_term / (n * r);

        dtype term1 = -2.f
            * (n - 1.f)
            * std::pow(xi, 5.f / 2.f)
            * x_sq
            * std::pow(r_sq, (n - 1.f) / 2.f - 1.5f)
            * exp_term
            * PI_SQRT_INV
            / n;

        // The part to the left and right of the parenthesis.
        dtype term2a = PI_SQRT_INV * std::pow(xi, 1.5f) * exp_term;

        // (The portion inside the parenthesis, as described by Wolfram Alpha)
        dtype term2b =
            2.f * ((n - 1.f) / 2.f - 1.f) * (n - 1.f) * x_sq * std::pow(r_sq, (n - 1.f) / 2.f - 2.f)
                + (n - 1.f) * std::pow(r_sq, (n - 1.f) / 2.f - 1.f);

        result += term0_coeff * term0 + term1 + term2a * term2b;
    }

    return result;
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


extern "C" __global__
void sto_val_kernel(
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
        out[i] = sto_val(posits_sample[i], posit_nuc, xi, 1);
    }
}


// Note that this is for the radial component only, with n=1. Real.
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
        out[i] = sto_second_deriv(posits_sample[i], posit_nuc, xi, n);
    }
}

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
        out_val[i_sample] += sto_val(posits_sample[i_sample], posit_nuc, xi, n);
        out_second_deriv[i_sample] += sto_second_deriv(posits_sample[i_sample], posit_nuc, xi, n);
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
            out_val[i_sample] += sto_val(posits_sample[i_sample], posits_nuc[i_basis], xis[i_basis], n[i_basis]) * weights[i_basis];
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
            out_val[i_sample] += sto_val(posits_sample[i_sample], posit_nuc, xis[i_basis], n[i_basis]) * weights[i_basis];
            out_second_deriv[i_sample] += sto_second_deriv(posits_sample[i_sample], posit_nuc, xis[i_basis], n[i_basis]) * weights[i_basis];
        }
    }
}
