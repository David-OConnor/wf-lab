# For troubleshooting our Rust STO basis solver.

from numpy import exp
import numpy as np
from typing import List

from scipy import linalg

KE_COEFF = -(1. * 1.) / (2. * 1.)
KE_COEFF_INV = 1. / KE_COEFF

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
    
    N = len(sample_pts)

    xis = []
    for xi in additional_xis: # todo temp
        xis.append(xi)

    psi_mat = np.zeros([N, N])
    psi_pp_mat = np.zeros([N, N])

    for i, xi in enumerate(xis):
        for j, posit_sample in enumerate(sample_pts):
            psi_mat[j][i] = value(xi, posit_sample)
            psi_pp_mat[j][i] = second_deriv(xi, posit_sample)

    rhs = np.array([KE_COEFF_INV * (V + E) for V in V_to_match])

    mat_to_solve = psi_pp_mat - (np.diag(rhs) @ psi_mat)

    # todo: This may no longer be accurate
    # Scipy methods note: 'generic' seesm to work. Sym, hermitian produce bad results. `positive definite` fails.
    # Numpy works. Generic uses the `GESV` LAPACK routine.

    weights = np.linalg.solve(mat_to_solve, np.zeros(N))
    weights_scipy = linalg.solve(mat_to_solve, np.zeros(N), assume_a='gen')
#     weights_scipy = linalg.solve(mat_to_solve, np.zeros(N), assume_a='her')

    s, v, r = np.linalg.svd(mat_to_solve)
    print(f"\nS {s}")
    print(f"\nV {v}")
    print(f"\nR {r}")

    # Normalize re base xi.
    # base_weight = weights[0]
    # for i, weight in enumerate(weights):
    #     weights[i] /= base_weight

    # for i, weight in enumerate(w_inv_approach):
    #     weights[i] /= base_weight

    print(f"\nPsi  mat: {psi_mat}")
    print(f"\nPsi''o mat: {psi_pp_mat}")
    print(f"\nMat to solve: {mat_to_solve}")
    print(f"\nrhs: {rhs}")

    print(f"\nWeights: {weights}")
    print(f"\nWeights Scipy: {weights_scipy}")
    print(f"\nWeights svd: {r[-1]}")

    print(f"\nA @ w: {mat_to_solve @ weights}\n")

    print(f"det: {np.linalg.det(mat_to_solve)} inv: {np.linalg.inv(mat_to_solve)}")

    # todo: See nalgebra Readme on BLAS etc as-required if you wish to optomize.

    return bases



find_bases(
    # todo: Consider calculating V from nuclei and electrons if you still have trouble.
    [0.5, 0.6666666666666666, 1.0],
    [2., 3., 4.,],
    -2.,
    [
        # (5., 0., 0.),
        (4., 0., 0.),
        (3., 0., 0.),
        (2., 0., 0.),
#         (1.5, 0., 0.),
#         (1.0, 0., 0.),
        # (0.5, 0., 0.),
        # (0.25, 0., 0.),
    ],
)


# w_a = 1.
# w_b = 0.7
#
# psi_a = value(1., (2., 0., 0.))
# psi_pp_a = second_deriv(1., (2., 0., 0.))
# rat_a = psi_pp_a / psi_a
# E = -1.12
#
# psi_b = value(2., (2., 0., 0.))
# psi_pp_b = second_deriv(2., (2., 0., 0.))
# rat_b = psi_pp_b / psi_b
#
# psi = psi_a + psi_b
# psi_pp = psi_pp_a + psi_pp_b
# rat_combined = psi_pp / psi
#
# rat_individual = rat_a + rat_b
#
# print(f"Rat combined: {rat_combined}, Rat individual: {rat_individual}")
#
# V_combined = KE_COEFF * rat_combined - E
# V_individual = KE_COEFF * rat_individual - E
#
# # Combined is the correct approach.  Sum befor dividing.
# # Dividing before summing is wrong
# print(f"V combined: {V_combined} Individual: {V_individual}")