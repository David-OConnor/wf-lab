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

    // c note: Omitting the f is double; including is f32.
//         if (r < 0.0000000000001) {
    if (r < 0.0000000000001f) {
       return 0.; // todo: Is this the way to handle?
    }

    return 1.f * charge / r;
}

extern "C" __global__
void coulomb_kernel(
//     double *out,
//     double *posits_charge_x,
//     double *posits_charge_y,
//     double *posits_charge_z,
//     double *posits_sample_x,
//     double *posits_sample_y,
//     double *posits_sample_z,
//     double *charges,
    float *out,
    float *posits_charge_x,
    float *posits_charge_y,
    float *posits_charge_z,
    float *posits_sample_x,
    float *posits_sample_y,
    float *posits_sample_z,
//     float3 *posits_charge,
//     float3 *posits_sample,
    float *charges,
    int N_charges,
    int N_samples
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // todo: Consider using a grid to make the code more readable,
    // but for now, flat is fine.

//     int i_charge = blockIdx.y*blockDim.y+threadIdx.y;
//     int i_sample = blockIdx.x*blockDim.x+threadIdx.x;

    // todo: QC rounding
    int i_charge = i / N_samples;
    int i_sample = i % N_samples;
    
    float3 posit_charge;
    posit_charge.x = posits_charge_x[i_charge];
    posit_charge.y = posits_charge_y[i_charge];
    posit_charge.z = posits_charge_z[i_charge];

    float3 posit_sample;
    posit_sample.x = posits_sample_x[i_sample];
    posit_sample.y = posits_sample_y[i_sample];
    posit_sample.z = posits_sample_z[i_sample];

//     float3 posit_charge = posits_charge[i_charge];
//     float3 posit_sample = posits_sample[i_sample];

    // int stride = blockDim.x * gridDim.x;
    // for (int i = index; i < n; i+= stride)
    //     y[i] = x[i] + y[i];
    // todo: Check for out of bounds
//     y[index] = x[index] * 2.0f;

    if (i_charge < N_charges && i_sample < N_samples) {
        out[i] = coulomb(posit_charge, posit_sample, charges[i_charge]);
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

__device__
float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
    float3 r;
    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    float EPS2  = 0.0000000001f;
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);

    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

// __device__
// float3 tile_calculation(float4 myPosition, float3 accel) {
//     int i;
//
//     extern __shared__
//     float4[] shPosition;
//
//     for (i = 0; i < blockDim.x; i++) {
//         accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
//     }
//     return accel;
// }
//
// __global__ void calculate_forces(void *devX, void *devA) {
//    extern __shared__ float4[]
//    shPosition;
//
//    float4 *globalX = (float4 *)devX;
//    float4 *globalA = (float4 *)devA;
//    float4 myPosition;
//    int i, tile;
//
//    float3 acc = {0.0f, 0.0f, 0.0f};
//
//    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
//    myPosition = globalX[gtid];
//
//    for (i = 0, tile = 0; i < N; i += p, tile++) {
//         int idx = tile * blockDim.x + threadIdx.x;
//             shPosition[threadIdx.x] = globalX[idx];
//             __syncthreads();
//             acc = tile_calculation(myPosition, acc);
//             __syncthreads();
//         }
//
//    // Save the result in global memory for the integration step.
//    float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
//    globalA[gtid] = acc4;
//
// }