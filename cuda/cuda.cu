#include <iostream>
#include <math.h>

// https://developer.nvidia.com/blog/even-easier-introduction-cuda/

// CUDA Kernel function to add the elements of two arrays on the GPU
// __global__
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
double V_coulomb(Vec3 posit_charge, Vec3 posit_sample, double charge) {
// double fn V_coulomb(posit_charge: std::array<int, 3>, posit_sample: std::array<int, 3>, charge: double) {
//     double diff = posit_sample - posit_charge;
//     double r = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];

    double r = 1.;

    // c note: Omitting the f is double; including is f32.
    if (r < 0.0000000000001f) {
        return 0.; // todo: Is this the way to handle?
    }

    return 1. * charge / r;
}

// `extern "C" prepended allows functions to be called from C code (And therefore Rust FFI)
extern "C" void ffi_test() {
    std::cout << "FFI TEST" << std::endl;
}


extern "C" int cuda_main(void) {
    ffi_test();
    return 0;

    std::cout << "Hello C++; Hello Cu" << std::endl;

    int N = 1<<20; // 1M elements

    // Allocate Unified Memory -- accessible from CPU or GPU
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = float(i);
        // y[i] = 2.0f * float(i);
    }

    // The first parameter specifies the number of thread blocks. The second is the number of
    // threads in the thread block.
    // This must be a multiple of 32.
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

//     add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    // float maxError = 0.0f;
    // for (int i = 0; i < N; i++) {
    // maxError = fmax(maxError, fabs(y[i]-3.0f));
    // std::cout << "Max error: " << maxError << std::endl;
    // }

    for (int i=0; i < 10; i++) {
        std::cout << "Val @ " << i << ": " << y[i] << std::endl;
    }

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
 }