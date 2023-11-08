#include <iostream>
#include <math.h>

// https://developer.nvidia.com/blog/even-easier-introduction-cuda/

struct Vec3 {
    double x;
    double y;
    double z;
};


extern "C" __global__
void sin_kernel(float *out, const float *inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = sin(inp[i]);
    }
}


// // todo: We may need to use floats here vice double, or suffer a large performance hit.
// // todo: Research this.
// double VCoulomb(Vec3 posit_charge, Vec3 posit_sample, double charge) {
//     Vec3 diff = {
//        posit_charge.x - posit_sample.x,
//        posit_charge.y - posit_sample.y,
//        posit_charge.z - posit_sample.z,
//     };
//
//     // todo: Does this work with CUDA?
//     double r = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
//
//     // c note: Omitting the f is double; including is f32.
//     if (r < 0.0000000000001) {
//         return 0.; // todo: Is this the way to handle?
//     }
//
//     return 1. * charge / r;
// }


extern "C" __global__
void coulomb_kernel(
    double *out,
    double *posits_charge_x,
    double *posits_charge_y,
    double *posits_charge_z,
    double *posits_sample_x,
    double *posits_sample_y,
    double *posits_sample_z,
    double *charges,
    int N_charges,
    int N_samples
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // todo: QC rounding
    int i_charge = i / N_charges;
    int i_sample = i % N_samples;

    // int stride = blockDim.x * gridDim.x;
    // for (int i = index; i < n; i+= stride)
    //     y[i] = x[i] + y[i];
    // todo: Check for out of bounds
//     y[index] = x[index] * 2.0f;

    if (i_charge < N_charges && i_sample < N_samples) {
        double diff_x = posits_charge_x[i_charge] - posits_sample_x[i_sample];
        double diff_y = posits_charge_y[i_charge] - posits_sample_y[i_sample];
        double diff_z = posits_charge_z[i_charge] - posits_sample_z[i_sample];

        // todo: Does this work with CUDA?
        double r = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

        // c note: Omitting the f is double; including is f32.
        if (r < 0.0000000000001) {
           out[i] = 0.; // todo: Is this the way to handle?
        }

        out[i] = 1. * charges[i_charge] / r;
    }
}

extern "C" __global__
void sum_coulomb_results_kernel(
    // For a given sample point, sum coulomb calculations from a number of charge points.
    double out,
    double *charges_this_pt,
    int N_charges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N_charges) {
        out += charges_this_pt[i];
    }
}


// extern "C" void runVCoulomb(double *posit_charge, double *posit_sample, double charge) {
//     print("Calculating coulomb potential using CDUA...");
//
//     int N = sizeof(posit_charge);
//
//     // Allocate Unified Memory -- accessible from CPU or GPU
//     // float *x, *y;
//     cudaMallocManaged(&posit_charge, N*sizeof(double));
//     cudaMallocManaged(&posit_sample, N*sizeof(double));
//
//     // initialize x and y arrays on the host
//     for (int i = 0; i < N; i++) {
//         x[i] = float(i);
//         // y[i] = 2.0f * float(i);
//     }
//
//     // The first parameter specifies the number of thread blocks. The second is the number of
//     // threads in the thread block.
//     // This must be a multiple of 32.
//     // todo: Figure out how you want to divide up the block sizes, index, stride etc.
//     int blockSize = 256;
//     int numBlocks = (N + blockSize - 1) / blockSize;
//
//     VCoulomb<<<numBlocks, blockSize>>>(posit_charge, posit_sample, charge);
//
//     // Wait for GPU to finish before accessing on host
//     cudaDeviceSynchronize();
//
// //     for (int i=0; i < 10; i++) {
// //         std::cout << "Val @ " << i << ": " << pos[i] << std::endl;
// //     }
//
//     // Free memory
//     cudaFree(posit_charge);
//     cudaFree(posit_sample);
//  }