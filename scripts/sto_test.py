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
from numpy import ndarray, array
from scipy.special import genlaguerre, gamma
from math import factorial

H = 0.03
H_SQ = H**2

@dataclass
class GenStoTerms:
    """
    Set up variables for a generalized STO that can be used across multiple positions.
    These are the terms that don't depend on position (or r). See equation 30.
    """
    n_st: float
    l_st: float
    zeta: float
    norm_term: float
    L: genlaguerre

    @classmethod
    def new(cls, eps: float, n: int) -> "GenStoTerms":
        l = 0

        # Star definitions: See Eq 16.
        n_st = n + eps
        l_st = l + eps
        zeta = 1.0 / n_st

        norm_num = (2.0 * zeta) ** 3 * gamma(n_st - l_st - eps3 + 1)

        norm_denom = gamma(n_st + l_st + eps3 + 1)

        # norm_term here is eq 31.
        norm_term = sqrt(norm_num / norm_denom)

        lg_term_a = n_st - l_st - eps3

        # Difference between this and normal STO: 2* factor for eps3. (1 makes it equal to sto)
        # We are setting it to 1 for now.
#         lg_term_b = 2 * l_st + 2 * eps3
        lg_term_b = 2 * l_st + 1 * eps3

        print(
            f"n: {n_st}, l: {l_st}, lg_a: {lg_term_a}, lg_b: {lg_term_b}"
        )

        if lg_term_a < 0:  # Generally happens very close to 0.
            lg_term_a = 0

        # Temporary hack
        lg_term_a = round(lg_term_a)  # Eg so we don't get an error when A = 0.9999

        # todo: fractional laguerre?
        L = genlaguerre(lg_term_a, lg_term_b)

        norm_term = 1  # Experimenting

        return cls(n_st, l_st, zeta, norm_term, L)

def gamma_(a: float, x: float = 0) -> float:
    """Eq 9, Γ. An extension of the factorial function to complex numbers."""
    result = 0

    dt = 0.01
    num_vals = 1_000

    t = x
    for i in range(num_vals):
        print("T", t ** (a - 1) * exp(-t) * dt)
        result += t ** (a - 1) * exp(-t) * dt

        t += dt

    return result


# If eps3 = 1, this is the same as eq17. (per the paper...)
eps3 = 1


def second_deriv(sto_terms: GenStoTerms, posit: ndarray, psi_sample_loc: float) -> float:
    """A numerical second derivative."""
    px = posit[0]
    py = posit[1]
    pz = posit[2]

    p_x_prev = array([px - H, py, pz])
    p_x_next = array([px + H, py, pz])
    p_y_prev = array([px, py - H, pz])
    p_y_next = array([px, py + H, pz])
    p_z_prev = array([px, py, pz - H])
    p_z_next = array([px, py, pz + H])

    psi_x_prev = sto_generalized(sto_terms, p_x_prev)
    psi_x_next = sto_generalized(sto_terms, p_x_next)
    psi_y_prev = sto_generalized(sto_terms, p_y_prev)
    psi_y_next = sto_generalized(sto_terms, p_y_next)
    psi_z_prev = sto_generalized(sto_terms, p_z_prev)
    psi_z_next = sto_generalized(sto_terms, p_z_next)

    return (psi_x_prev + psi_x_next + psi_y_prev + psi_y_next + psi_z_prev + psi_z_next - psi_sample_loc * 6) / H_SQ


def sto_generalized(t: GenStoTerms, posit_sample: ndarray) -> float:
    """This is currently based off Equation 30 in that paper."""

    posit = array([0.0, 0.0, 0.0])
    r = np.linalg.norm(posit_sample - posit)

    # Diff between this and radial: Radial includes xi in the exp. (And n vice n*)
    exp_term = exp(-t.zeta * r)

    # Diff between this and radial: Addition of eps3 term, and n vice n*, l vice l*.
    polynomial_term = (2.0 * t.zeta * r) ** (t.l_st + eps3 - 1) * t.L(2.0 * t.zeta * r)

    R = t.norm_term * exp_term * polynomial_term

    return R



def radial(xi: float, n: int, posit_sample: ndarray) -> float:
    l = 0
    posit = array([0.0, 0.0, 0.0])
    r = np.linalg.norm(posit_sample - posit)

    exp_term = exp(-xi * r / n)

    L = genlaguerre(n - l - 1, 2 * l + 1)

    polynomial_term = (2.0 * r / n) ** l * L(2.0 * r / n)

    return polynomial_term * exp_term


def main():
    # We'll say y and z = 0, for radial functions.
    x = linspace(0, 3, 10000)
    n = 2

    #     for xi in [1., 1.2, 1.4, 1.6, 1.8, 2., 10.]:
    for eps in [0., 0.1, 0.3, 0.5, 0.7, 0.9]:
        values = np.zeros(x.size)

        sto_terms = GenStoTerms.new(eps, n)

        for i in range(x.size):
            posit = array([x[i], 0.0, 0.0])
            R = sto_generalized(sto_terms, posit)

            r = np.linalg.norm(x[i])
#             values[i] = 4.0 * pi * r**2 * R**2
#             values[i] = R

            psi_pp = second_deriv(sto_terms, posit, R)

            values[i] = psi_pp / R
#             values[i] = psi_pp
#             values[i] = R

        plt.plot(x, values)

#     values_trad = np.zeros(x.size)
#     for i in range(x.size):
#         R = radial(1., n, array([x[i], 0.0, 0.0]))
#
#         r = np.linalg.norm(x[i])
#         values_trad[i] = sto_terms.norm_term * 4.0 * pi * r**2 * R**2
#
#     plt.plot(x, values_trad)

    plt.ylim(-10, 20)

    plt.show()


main()
