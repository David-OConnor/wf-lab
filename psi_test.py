# For troubleshooting our Rust STO basis solver.

from numpy import exp
import numpy as np
from typing import List

KE_COEFF = -(1. * 1.) / (2. * 1.);
KE_COEFF_INV = 1. / KE_COEFF;

NORMS = [
    # 84.92786880965572
    10.942926831589256,  # xi=2
    3.6260051729267344,  # 3
    1.9298933867204193,  # 4
    1.3787825712919901,  # 5
    1.1666246143992656, # 6
]

def value(xi: float, posit_sample: (float, float, float)) -> float:
    diff = posit_sample
    r = (diff[0]**2 + diff[1]**2 + diff[2])**(1/2)

    return exp(-xi * r / (1 * 1))
    

def second_deriv(xi: float, posit_sample: (float, float, float)) -> float:
    diff = posit_sample
    r = (diff[0]**2 + diff[1]**2 + diff[2])**(1/2)

    exp_term = exp(-xi * r)

    result = 0.
    for coord in [diff[0], diff[1], diff[2]]:
        result += xi**2 * coord**2 * exp_term / r**2
        result += xi * coord**2 * exp_term / r**3
        result -= xi * exp_term / r

    return result
    

def find_bases(
    V_to_match: List[float],
    additional_xis: List[float],
    E: float,
    sample_pts,
):
    bases = []

    xis = []
    for xi in additional_xis: # todo temp
        xis.append(xi)

    psi_ratio_mat = np.zeros([len(sample_pts), len(sample_pts)])

    for i, xi in enumerate(xis):
        norm = NORMS[i]

        print(f"XI: {xi} norm: {norm}")

        for j, posit_sample in enumerate(sample_pts):
            psi = norm * value(xi, posit_sample)
            psi_pp = norm * second_deriv(xi, posit_sample)

            psi_ratio_mat[j][i] = (psi_pp / psi)


    

    v_charge_vec = np.array([KE_COEFF_INV * (V + E) for V in V_to_match])

    weights = np.linalg.solve(psi_ratio_mat, v_charge_vec)

    # Normalize re base xi.
    # base_weight = weights[0]
    # for i, weight in enumerate(weights):
    #     weights[i] /= base_weight


    # w_inv_approach = np.linalg.inv(psi_ratio_mat) @ v_charge_vec
    # for i, weight in enumerate(w_inv_approach):
    #     weights[i] /= base_weight

    print(f"\nPsi ratio mat: {psi_ratio_mat}")
    print(f"\nWeights: {weights}")
    print(f"\nV: {v_charge_vec}")

    print(f"\n A @ w: {psi_ratio_mat @ weights}")
    # print(f"\n A @ w (inv approach): {psi_ratio_mat @ w_inv_approach}")
    # print(f"\n A^(-1) @ V: {w_inv_approach}")

    # todo: See nalgebra Readme on BLAS etc as-required if you wish to optomize.

    return bases



find_bases(
    # todo: Consider calculating V from nuclei and electrons if you still have trouble.
    [0.5, 0.6666666666666666, 1.0, 1.3333333333333333, 2.0],
    [2., 3., 4., 5., 6.],
    -2.,
    [
        # (5., 0., 0.),
        (4., 0., 0.),
        (3., 0., 0.),
        (2., 0., 0.),
        (1.5, 0., 0.),
        (1.0, 0., 0.),
        # (0.5, 0., 0.),
        # (0.25, 0., 0.),
    ],
)