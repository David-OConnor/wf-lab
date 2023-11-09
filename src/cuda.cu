// #include <iostream>
// #include <math.h>

// https://developer.nvidia.com/blog/even-easier-introduction-cuda/


extern "C" __global__ void matmul(float* A, float* B, float* C, int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
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

    if (r < 0.0000000000001f) {
       return 0.f; // todo: Is this the way to handle?
    }

    return 1.f * charge / r;
}

extern "C" __global__
void coulomb_kernel(
    float *out,
    float3 *posits_charge,
    float3 *posits_sample,
    float *charges,
    int N_charges,
    int N_samples
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N_charges * N_samples; i+= stride) {
//         int i = blockIdx.x * blockDim.x + threadIdx.x;

        // todo: Consider using a grid to make the code more readable,
        // but for now, flat is fine.

    //     int i_charge = blockIdx.y*blockDim.y+threadIdx.y;
    //     int i_sample = blockIdx.x*blockDim.x+threadIdx.x;

        int i_charge = i / N_samples;
        int i_sample = i % N_samples;

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
    int N_charges,
    int N_samples
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N_charges) {
//         out[lkj] += coulomb_combos[i];
    }
}
