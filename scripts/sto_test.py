"""
Experiments with generalized STOs

From I. Complete and orthonormal sets of exponential-type orbitals
with noninteger principal quantum numbers
https://arxiv.org/pdf/2205.02317.pdf
"""

import numpy as np
from numpy import exp, sqrt, linspace
from matplotlib import pyplot as plt
from numpy import ndarray as array
from scipy.special import genlaguerre
from math import factorial


def gamma(a: float, x: float) -> float:
    """Eq 9, Γ. An extension of the factorial function to complex numbers."""
    result = 0

    dt = 0.01
    num_vals = 1_000

    t = x
    for i in range(num_vals):
        result += t ** (a-1) * exp(-t) * dt

        t += dt

    return result


def sto_generalized(xi: float, n: int, posit_sample: array) -> float:
    """This is currently based off Equation 17 and 30 in that paper."""
    l = 0
    posit = np.array([0., 0., 0.])
    r = np.linalg.norm(posit_sample - posit)

    eps = 1  # Eps is between 0 and 1.

    # Star definitions: See Eq 16.
    n_st = n + eps
    l_st = l + eps
    zeta = 1./n_st

    eps3 = 1; # Set to 1 for use with eq 17.

    term0_num = (2. * zeta)**3 * gamma(n_st - l_st - eps3 + 1, 0)
#     term0_denom = gamma(n_st + l_st + eps3 + 1, 0)  # eq30
    term0_denom = 2 * n_st * gamma(n_st + l_st + 1, 0)  # eq17

    term0 = sqrt(term0_num / term0_denom)

    term1 = (2. * zeta * r) ** (l_st + eps3 - 1)

    exp_term = exp(-zeta * r)

#     L = genlaguerre(n_st - l_st - eps3, 2 * l_st + 2 * eps3)  # eq30
    L = genlaguerre(n_st - l_st - 1, 2 * l_st  + 1)  # Eq 17

    polynomial_term = L(2. * zeta * r)

    return term0 * term1 * exp_term * polynomial_term


def radial(xi: float, n: int, posit_sample: array) -> float:
    l = 0
    posit = np.array([0., 0., 0.])
    r = np.linalg.norm(posit_sample - posit)

    exp_term = exp(-xi * r / n)

    L = genlaguerre(n - l - 1, 2 * l + 1)

    polynomial_term = (2. * r / n)**l * L(2. * r / n)


    return polynomial_term * exp_term


def main():
	# We'll say y and z = 0, for radial functions.
    x = linspace(0, 10, 1000)
    n = 2

    for xi in [1., 1.2, 1.4, 1.6, 1.8, 2., 10.]:
        values = np.zeros(x.size)
        for i in range(x.size):
#             values[i] = radial(xi, n, np.array([x[i], 0., 0.]))
            values[i] = sto_generalized(xi, n, np.array([x[i], 0., 0.]))

        plt.plot(x, values)

    plt.show()


main()