#ifndef UTIL_H
#define UTIL_H

// #include <cstdint>

// Forward declaration of CUDA types
struct float3;
struct double3;

// Allows easy switching between float and double.
// #define dtype double
// #define dtype3 double3
#define dtype float
#define dtype3 float3

// Declaration of constants
extern __device__ const dtype SOFTENING_FACTOR;
extern __device__ const dtype PI_SQRT_INV;
extern __device__ const dtype A_0;
extern __device__ const dtype EPS_DIV0;
extern __device__ const dtype H;
extern __device__ const dtype H_SQ;

// Function declarations
__device__ dtype laguerre(uint16_t n, uint16_t alpha, dtype x);
__device__ dtype calc_dist(dtype3 point0, dtype3 point1);
__device__ uint32_t factorial(uint8_t val);
__device__ dtype coulomb(dtype3 q0, dtype3 q1, dtype charge);

#endif // UTIL_H
