// This module contains constants and utility functions related to the kernels we use.


// Allows easy switching between float and double.
using dtype = float;
using dtype3 = float3;

__device__
const dtype SOFTENING_FACTOR_SQ = 0.000000000001f;
__device__
const dtype PI_SQRT_INV = 0.5641895835477563f;
__device__
const dtype A_0 = 1.f;
__device__
const dtype EPS_DIV0 = 0.00000000001f;
__device__
const double H = 0.01;
__device__
const double H_SQ = 0.01 * 0.01;

__device__
dtype laguerre(uint16_t n, uint16_t alpha, dtype x) {
    if (n == 0) {
        return 1.f;
    } else if (n == 1) {
        return alpha + 1.f - x;
    } else if (n == 2) {
        return std::pow(x, 2) / 2.f - (alpha + 2.f) * x + (alpha + 1.f) * (alpha + 2.f) / 2.f;
    } else {
        return 0.f; // todo: Implement.
    }
}

__device__
dtype calc_dist(dtype3 point0, dtype3 point1) {
    dtype3 diff;
    diff.x = point0.x - point1.x;
    diff.y = point0.y - point1.y;
    diff.z = point0.z - point1.z;

    return std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
}

__device__
uint32_t factorial(uint8_t val) {
    // todo: 64-bit a/r
    if (val == 0) {
        return 1;
    }
    if (val == 1) {
        return 1;
    }
    if (val == 2) {
        return 1;
    }
    if (val == 3) {
        return 6;
    }
    if (val == 4) {
        return 24;
    }
    if (val == 5) {
        return 120;
    }
    if (val == 6) {
        return 720;
    }
    if (val == 7) {
        return 5040;
    }
    if (val == 8) {
        return 40320;
    }
    if (val == 9) {
        return 362880;
    }
    if (val == 10) {
        return 3628800;
    }

    // todo: More A/R
    return 0.;
}


__device__
dtype coulomb(dtype3 q0, dtype3 q1, dtype charge) {
    float r = calc_dist(q0, q1);

    return 1.f * charge / (r + SOFTENING_FACTOR_SQ);
}