#include <iostream>
#include <math.h>

// https://developer.nvidia.com/blog/even-easier-introduction-cuda/

extern "C" __global__
void sin_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}

// CUDA Kernel function to add the elements of two arrays on the GPU
// extern "C" __global__
// void coulomb_parallel(float *posit_charges, float *posit_samples) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//
//     // for (int i = index; i < n; i+= stride)
//     //     y[i] = x[i] + y[i];
//     // todo: Check for out of bounds
//     y[index] = x[index] * 2.0f;
// }


struct Vec3 {
    double x;
    double y;
    double z;
};



// todo: We may need to use floats here vice double, or suffer a large performance hit.
// todo: Research this.
double VCoulomb(Vec3 posit_charge, Vec3 posit_sample, double charge) {
// double fn V_coulomb(posit_charge: std::array<int, 3>, posit_sample: std::array<int, 3>, charge: double) {
    Vec3 diff = {
       posit_charge.x - posit_sample.x,
       posit_charge.y - posit_sample.y,
       posit_charge.z - posit_sample.z,
    };
    double r = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    // c note: Omitting the f is double; including is f32.
    if (r < 0.0000000000001) {
        return 0.; // todo: Is this the way to handle?
    }

    return 1. * charge / r;
}

// `extern "C" prepended allows functions to be called from C code (And therefore Rust FFI)
extern "C" void ffi_test() {
    std::cout << "FFI TEST" << std::endl;
}

void print(std::string text) {
    std::cout << text << std::endl;
}


extern "C" void runVCoulomb(double *posit_charge, double *posit_sample, double charge) {
    print("Calculating coulomb potential using CDUA...");

    int N = sizeof(posit_charge);

    // Allocate Unified Memory -- accessible from CPU or GPU
    // float *x, *y;
    cudaMallocManaged(&posit_charge, N*sizeof(double));
    cudaMallocManaged(&posit_sample, N*sizeof(double));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = float(i);
        // y[i] = 2.0f * float(i);
    }

    // The first parameter specifies the number of thread blocks. The second is the number of
    // threads in the thread block.
    // This must be a multiple of 32.
    // todo: Figure out how you want to divide up the block sizes, index, stride etc.
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    VCoulomb<<<numBlocks, blockSize>>>(posit_charge, posit_sample, charge);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

//     for (int i=0; i < 10; i++) {
//         std::cout << "Val @ " << i << ": " << pos[i] << std::endl;
//     }

    // Free memory
    cudaFree(posit_charge);
    cudaFree(posit_sample);
 }