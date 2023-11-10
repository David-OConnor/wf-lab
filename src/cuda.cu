// #include <iostream>
// #include <math.h>

// https://developer.nvidia.com/blog/even-easier-introduction-cuda/

// Allows easy switching between float and double.
#define dtype double
#define dtype3 double3

__device__
const dtype SOFTENING_FACTOR = 0.000000000001;

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
dtype coulomb(dtype3 a, dtype3 b, dtype charge) {
    dtype3 diff;
    diff.x = a.x - b.x;
    diff.y = a.y - b.y;
    diff.z = a.z - b.z;

    dtype r = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    return 1. * charge / (r + SOFTENING_FACTOR);
}


// todo: Cuda's `threadIdx` can be 1D, 2D, or 3D. 2D may be a better fit here.
// 1D with packing/unpacking is fine, but 2D would be perhaps cleaner. Experiment.

extern "C" __global__
void coulomb_kernel(
    dtype *out,
    dtype3 *posits_charge,
    dtype3 *posits_sample,
    dtype *charges,
    size_t N_charges,
    size_t N_samples
) {
    // In this approach, we parallelize operations per sample, but run the
    // charge computations in serial, due to the cumulative addition step. This appears
    // to be much faster in practice, likely due to the addition being offloaded
    // to the CPU in the other approach.

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_sample = index; i_sample < N_samples; i_sample += stride) {
        // Compute the sum serially, as it may not be possible to naively apply it in parallel,
        // and we may still be saturing GPU cores given the large number of samples.
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
void coulomb_kernel_without_addition(
    dtype *out,
    dtype3 *posits_charge,
    dtype3 *posits_sample,
    dtype *charges,
    size_t N_charges,
    size_t N_samples
) {
    // In this approach, we calculate all interaction combinations on the GPU in parallel, then sum per-sample
    // interactions afterwards on the host.
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < N_charges * N_samples; i += stride) {
        // todo: Consider using a grid to make the code more readable,
        // but for now, flat is fine.
        // size_t i_charge = blockIdx.y*blockDim.y+threadIdx.y;
        // size_t i_sample = blockIdx.x*blockDim.x+threadIdx.x;

        // Compute the sum serially, as it may not be possible to naively apply it in parallel.
        size_t i_charge = i / N_samples;
        size_t i_sample = i % N_samples;

        dtype3 posit_charge = posits_charge[i_charge];
        dtype3 posit_sample = posits_sample[i_sample];

        if (i_charge < N_charges && i_sample < N_samples) {
            out[i] = coulomb(posit_charge, posit_sample, charges[i_charge]);
        }
    }
}