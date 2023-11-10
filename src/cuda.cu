// #include <iostream>
// #include <math.h>

// https://developer.nvidia.com/blog/even-easier-introduction-cuda/

#define SOFTENING_FACTOR 0.0000000000001f


extern "C" __global__ void matmul(float* A, float* B, float* C, int N) {
    // Syntax example using 2D inputs.
    size_t ROW = blockIdx.y * blockDim.y + threadIdx.y;
    size_t COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (size_t i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}


__device__
float coulomb(float3 a, float3 b, float charge) {
    float3 diff;
    diff.x = a.x - b.x;
    diff.y = a.y - b.y;
    diff.z = a.z - b.z;

    float r = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    if (r < 0.0000000001f) { // todo: Softening not working properly.
        return 0.f;
    }

    // todo: Softening not working properly.
//     return 1.f * charge / (r + SOFTENING_FACTOR);
    return 1.f * charge / r;
}


// todo: Cuda's `threadIdx` can be 1D, 2D, or 3D. 2D may be a better fit here.
// 1D with packing/unpacking is fine, but 2D would be perhaps cleaner. Experiment.

extern "C" __global__
void coulomb_kernel(
    float *out,
    float3 *posits_charge,
    float3 *posits_sample,
    float *charges,
    size_t N_charges,
    size_t N_samples
) {
    // In this approach, we parallelize operations per sample, but run the
    // charge computations in serial, due to the cumulative addition step.

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i_sample = index; i_sample < N_samples; i_sample += stride) {
        // todo: Consider using a grid to make the code more readable,
        // but for now, flat is fine.
        // size_t i_charge = blockIdx.y*blockDim.y+threadIdx.y;
        // size_t i_sample = blockIdx.x*blockDim.x+threadIdx.x;

        // Compute the sum serially, as it may not be possible to naively apply it in parallel.

        for (size_t i_charge = 0; i_charge < N_charges; i_charge++) {
            float3 posit_charge = posits_charge[i_charge];
            float3 posit_sample = posits_sample[i_sample];

            if (i_sample < N_samples) {
                out[i_sample] += coulomb(posit_charge, posit_sample, charges[i_charge]);
            }
        }
    }
}



extern "C" __global__
void coulomb_kernel_without_addition(
    float *out,
    float3 *posits_charge,
    float3 *posits_sample,
    float *charges,
    size_t N_charges,
    size_t N_samples
) {
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

        float3 posit_charge = posits_charge[i_charge];
        float3 posit_sample = posits_sample[i_sample];

        if (i_charge < N_charges && i_sample < N_samples) {
            out[i] = coulomb(posit_charge, posit_sample, charges[i_charge]);
        }
    }
}

// todo: We should perform this addition here without passing to the host
// todo in between.

extern "C" __global__
void sum_coulomb_results_kernel(
    // For a given sample point, sum coulomb calculations from a number of charge points.
    double out,
    double *coulomb_combos,
    size_t N_charges,
    size_t N_samples
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N_charges) {
//         out[lkj] += coulomb_combos[i];
    }
}
