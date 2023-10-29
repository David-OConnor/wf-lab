#include <iostream>
#include <math.h>

// https://developer.nvidia.com/blog/even-easier-introduction-cuda/

// // CUDA Kernel function to add the elements of two arrays on the GPU
// __global__
// void add(int n, float *x, float *y)
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//
//   // for (int i = index; i < n; i+= stride)
//   //     y[i] = x[i] + y[i];
//   // todo: Check for out of bounds
//     y[index] = x[index] * 2.0f;
// }


// `extern "C" prepended allows functions to be called from C code (And therefore Rust FFI)
//extern "C" void ffi_test() {
void ffi_test() {
//    std::cout << "FFI TEST" << std::endl;
    printf("FFI TEST C");
}


 int main(void)
 {
//   std::cout << "Hello C++; Hello Cu" << std::endl;
//
//     int N = 1<<20; // 1M elements
//
//     // Allocate Unified Memory -- accessible from CPU or GPU
//     float *x, *y;
//     cudaMallocManaged(&x, N*sizeof(float));
//     cudaMallocManaged(&y, N*sizeof(float));
//
//     // initialize x and y arrays on the host
//     for (int i = 0; i < N; i++) {
//       x[i] = float(i);
//       // y[i] = 2.0f * float(i);
//     }
//
//   // The first parameter specifies the number of thread blocks. The second is the number of
//   // threads in the thread block.
//   // This must be a multiple of 32.
//   int blockSize = 256;
//   int numBlocks = (N + blockSize - 1) / blockSize;
//
//   add<<<numBlocks, blockSize>>>(N, x, y);
//
//   // Wait for GPU to finish before accessing on host
//   cudaDeviceSynchronize();
//
//   // Check for errors (all values should be 3.0f)
//   // float maxError = 0.0f;
//   // for (int i = 0; i < N; i++) {
//     // maxError = fmax(maxError, fabs(y[i]-3.0f));
//     // std::cout << "Max error: " << maxError << std::endl;
//   // }
//
//   for (int i=0; i < 10; i++) {
//     std::cout << "Val @ " << i << ": " << y[i] << std::endl;
//   }
//
//   // Free memory
//   cudaFree(x);
//   cudaFree(y);
//
//   return 0;da" << std::endl;
//
 }