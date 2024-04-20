"""
Experiments with generalized STOs

From I. Complete and orthonormal sets of exponential-type orbitals
with noninteger principal quantum numbers
https://arxiv.org/pdf/2205.02317.pdf
"""

from dataclasses import dataclass
import numpy as np
from math import pi
from numpy import exp, sqrt, linspace
from matplotlib import pyplot as plt
from numpy import ndarray as array
from scipy.special import genlaguerre, gamma
from math import factorial


def gamma_(a: float, x: float=0) -> float:
    """Eq 9, Γ. An extension of the factorial function to complex numbers."""
    result = 0

    dt = 0.01
    num_vals = 1_000

    t = x
    for i in range(num_vals):
        print("T", t ** (a-1) * exp(-t) * dt)
        result += t ** (a-1) * exp(-t) * dt

        t += dt

    return result

@dataclass
class GenStoTerms:
    """Set up variables for a generalized STO that can be used across multiple positions.
    These are the terms that don't depend on position (or r)."""
    n_st: float
    l_st: float
    zeta: float
    term0: float

    @classmethod
    def new(cls, eps: float, n: int) -> "GenStoTerms":
        l = 0  # todo: A/R

        # Star definitions: See Eq 16.
        n_st = n + eps
        l_st = l + eps
        zeta = 1. / n_st


            # todo temp
        #     zeta = 1
        #     n_st = 1 / zeta
        #     zeta = 1 # todo temp!

        eps3 = 1; # Set to 1 for use with eq 17.

        term0_num = (2. * zeta)**3 * gamma(n_st - l_st - eps3 + 1)

        term0_denom = 2 * n_st * gamma(n_st + l_st + 1)  # eq17
        # term0_denom = gamma(n_st + l_st + eps3 + 1)  # eq30

        # term0 here is eq 18 or 30
        term0 = sqrt(term0_num / term0_denom)

        return cls(n_st, l_st, zeta, term0)

def sto_generalized(t: GenStoTerms, posit_sample: array) -> float:
    """This is currently based off Equation 17 and 30 in that paper."""

    posit = np.array([0., 0., 0.])
    r = np.linalg.norm(posit_sample - posit)

    eps3 = 1; # Set to 1 for use with eq 17.

    term1 = (2. * t.zeta * r) ** (t.l_st + eps3 - 1)

    exp_term = exp(-t.zeta * r)

    lg_term_a = t.n_st - t.l_st - 1  # Eq 17
    lg_term_b = 2 * t.l_st + 1  # Eq 17

   # lg_term_a = n_st - l_st - eps3,   # Eq 30
   # lg_term_b = 2 * l_st + 2 * eps3  # eq 30

    if lg_term_a < 0:  # Generally happens very close to 0.
#         print("LG A: ", lg_term_a)
        lg_term_a = 0

    # Temporary hack
    lg_term_a = round(lg_term_a)  # Eg so we don't get an error when A = 0.9999

#     if lg_term_b < 0:  # Generally happens very close to 0.
#         print("LG B: ", lg_term_b)
#         lg_term_b = 0

    L = genlaguerre(lg_term_a, lg_term_b)

    polynomial_term = L(2. * t.zeta * r)

    R = t.term0 * term1 * exp_term * polynomial_term

#     return R
#     return 4. * pi * r**2 * R
    return 4. * pi * r**2 * R**2


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
    x = linspace(0, 15, 100)
    n = 2

#     for xi in [1., 1.2, 1.4, 1.6, 1.8, 2., 10.]:
    for eps in [0., 0.1, 0.3, 0.5, 0.7, 0.9]:
        values = np.zeros(x.size)

        sto_terms = GenStoTerms.new(eps, n)

        for i in range(x.size):
#             values[i] = radial(xi, n, np.array([x[i], 0., 0.]))
            values[i] = sto_generalized(sto_terms, np.array([x[i], 0., 0.]))

        plt.plot(x, values)

    plt.show()


main()