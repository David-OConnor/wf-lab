# This file is a lab aimed at solving the Schrodinger equation
# for arbitrary potentials, in 3D.
#
# Some methods to consider:
#
# - Density Functional theory
# - Finite Element Method
# - Diffusion Monte carlo
# - Hartree Fock
# - DVR
# - Spectral methods
# Kohn-Sham / plane-wave basis

# Article comparing plane waves to atomic orbitals:
# https://davidbowler.github.io/AtomisticSimulations/blog/basis-sets-in-dft
# Has some notable detractions about plane waves. Good info in general
# comparing the 2.

# https://chem.libretexts.org/Courses/University_of_California_Davis/UCD_Chem_107B%3A_Physical_Chemistry_for_Life_Scientists/Chapters/4%3A_Quantum_Theory/4.10%3A_The_Schr%C3%B6dinger_Wave_Equation_for_the_Hydrogen_Atom

# Good resource overall?: http://davidbowler.github.io/AtomisticSimulations/index.html


# todo: Imaginary part of wf?

# Basis sets:
# - psi values (Is this called real-space?)
# - plane waves
# - gaussians
# - atomic orbitals
# - Slater-type orbitals (STO)

# I think of these, perhaps Slater orbitals are a good place to start.
# Note the Kato therom: That there will be a cusp at every nucleus. This
# is great info, if true. (Ground state only??)

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from numpy import sqrt, pi, cos, ndarray, exp
import math

TAU = 2. * pi

A_0 = 1.
Z_H = 1.
K_C = 1.
Q_PROT = 1.
Q_ELEC = -1.
# M_ELEC = 5.45e-4
M_ELEC = 1.  # todo: Which?
ħ = 1.

@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)


def score_wf(target: ndarray, attempt: ndarray) -> float:
    """Score using a least-squares regression."""
    return sum((attempt - target)**2) / target.size

def score_wf_2d(target: ndarray, attempt: ndarray) -> float:
    """Score using a least-squares regression."""
    result = 0.
    for i in range(target[0].size):
        for j in range(target[1].size):
            result += attempt[i, j] - target[i, j]

    return result / target.size


def V_h(posit_nuc: Vec3, posit_sample: Vec3) -> float:
    """Hydrogen potential"""
    diff = posit_sample - posit_nuc
    r = sqrt(diff.x**2 + diff.y**2 + diff.z**2)

    return -K_C * Q_PROT / r


def V_h2_ion(posit_nuc: Vec3, posit_sample: Vec3) -> float:
    """Hydrogen ion potential"""
    diff = posit_sample - posit_nuc
    r = sqrt(diff.x**2 + diff.y**2 + diff.z**2)

    # todo: Why negative? Seems we need it.
    return -K_C * Q_PROT / r


def V_osc(posit_nuc: Vec3, posit_sample: Vec3) -> float:
    """Quantum oscillator potential"""
    diff = posit_sample - posit_nuc
    r = sqrt(diff.x**2 + diff.y**2 + diff.z**2)

    return (r/10)**2


def h_wf_100(posit_nuc: Vec3, posit_sample: Vec3) -> float:
    """Analytic solution for n=1, s orbital"""
    diff = posit_sample - posit_nuc
    r = sqrt(diff.x**2 + diff.y**2 + diff.z**2)

    ρ = Z_H * r / A_0
    return 1. / sqrt(pi) * (Z_H / A_0)**(3. / 2.) * exp(-ρ)
    # return 1. / sqrt(pi) * 1./ A_0**(3. / 2.) * exp(-ρ)


def h_wf_200(posit_nuc: Vec3, posit_sample: Vec3) -> float:
    diff = posit_sample - posit_nuc
    r = sqrt(diff.x**2 + diff.y**2 + diff.z**2)

    ρ = Z_H * r / A_0
    return 1. / sqrt(32.*pi) * (Z_H / A_0)**(3. / 2.) * (2. - ρ) * exp(-ρ/2.)


def h_wf_210(posit_nuc: Vec3, posit_sample: Vec3, theta: float=1/2. * TAU) -> float:
    diff = posit_sample - posit_nuc
    r = sqrt(diff.x**2 + diff.y**2 + diff.z**2)

    ρ = Z_H * r / A_0
    return 1. / sqrt(32.*pi) * (Z_H / A_0)**(3. / 2.) * ρ * exp(-ρ/2.) * cos(theta)


def wf_osc(posit_nuc: Vec3, posit_sample: Vec3, theta: float=1/2. * TAU) -> float:
    """Quantum harmonic oscillator"""
    diff = posit_sample - posit_nuc
    r = sqrt(diff.x**2 + diff.y**2 + diff.z**2)

    n = 0
    omega = 1

    part1 = 1./(sqrt(2**n * math.factorial(n)))
    part2 = (M_ELEC * omega / (pi * hbar))**(1/4)
    part3 = exp(-(M_ELEC * omega * r**2 / (2 * hbar)))
    part4 = H_n * sqrt((M_ELEC * omega / hbar) * r)

    # todo: Incomplete

    return part1 * part2 * part3 * part4


def plot_1d(x_vals: ndarray, data: list[ndarray]):
    for d in data:
        plt.plot(x_vals, data)

    plt.ylim(-0.5, 0.8)

    plt.show()


def plot_2d(x_vals: ndarray, y_vals: ndarray, data: list[ndarray]):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')

    x_plot, y_plot = np.meshgrid(x_vals, y_vals)

    for d in data:
        ax.plot_surface(x_plot, y_plot, d)

    ax.set_aspect('equal')
    ax.set_xlim(-4., 4.)
    ax.set_ylim(-4., 4.)
    ax.set_zlim(-0.5, 0.8)

    plt.show()


# def gauss():
#     """Putting gaussian-related code here for now; don't expect to use it."""
#     # Gauss consts.
#     a = 1./sqrt(pi)  # Height
#     b = 0.  # center
#     c = 1.5  # Width

#     # def make_gaussian_1d(x: ndarray, a: float, b: float, c: float) -> np.ndarray:
# #     return a * exp(-(x - b)**2. / (2. * c**2))


# def make_gaussian_3d(posit_nuc, posit_sample: Vec3, a: float, b: float, c: float) -> float:
#     diff = posit_sample - posit_nuc
#     r = sqrt(diff.x**2 + diff.y**2 + diff.z**2)

#     return a * exp(-(r - b)**2. / (2. * c**2))


def main():
    # Let's stick to 3D. And try the H WF as a test that it
    # works, to validate our method.

    # posit_nuc = Vec3(0., 0., 0.)

    # Schrod eq for H: 
    # V for hydrogen: K_C * Q_PROT / r

    # psi(r)'' = (E - V(r)) * 2*m/ħ**2 * psi(r)
    # psi(r) = (E - V(R))^-1 * ħ**2/2m * psi(r)''

    N = 100

    V_vals = np.zeros(N)
    psi = np.zeros(N)
    psi_pp_expected = np.zeros(N)
    psi_pp_measured = np.zeros(N)

    V_vals_2d = np.zeros((N, N))
    psi_2d = np.zeros((N, N))
    psi_pp_expected_2d = np.zeros((N, N))
    psi_pp_measured_2d = np.zeros((N, N))

    x_vals = np.linspace(-4., 4., N)
    y_vals = np.linspace(-4., 4., N)

    # Used for calculating numerical psi''.
    # Smaller is more precise. Applies to dx, dy, and dz
    h = 0.000001  # aka dx

    # wf = wf_osc
    wf = h_wf_100
    # wf = lambda poisit_nuc, posit_sample: make_gaussian_3d(posit_nuc, posit_sample, a, b, c)

    potential_fn = V_h
    # potential_fn = V_osc

    E = -0.05
    # E = -1/8 # n = 2
    # E = -0.05514706 # n = 3

    # nuclei = [Vec3(2., 0., 0.)]
     # H ion nuc dist is I believe 2 bohr radii.
    nuclei = [Vec3(-0.5, 0., 0.), Vec3(0.5, 0., 0.)]

    # psi = wf(posit_nuc, Vec3::n ))

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            posit_sample = Vec3(x, y, 0.)

            V = 0.

            # Note: If using an analytic etc psi, put this line back.
            # psi[i] = wf(posit_nuc, posit_sample)
            for nuc in nuclei:
                # todo: Naiive superposition
                psi_2d[i, j] += wf(nuc, posit_sample)

                V += potential_fn(nuc, posit_sample)
                # V_vals[i] = V
            V_vals_2d[i, j] = V

            # psi_pp_expected[i] = (E - V) * -2. * M_ELEC / ħ**2 * psi[i]
            psi_pp_expected_2d[i, j] = (E - V) * -2. * M_ELEC / ħ**2 * psi_2d[i, j]

            # Calculate psi'' based on a numerical derivative of psi
            # in 3D.

            # todo: You need to subtract and scale by an appropriate dt on your adding
            # todo and subtracting from x, y etc.

            # todo: second deriv is perhaps avg of the above? Validate this.
            # found in prev notes. Test on 1D, 2D, 3D.

            x_prev = Vec3(posit_sample.x - h, posit_sample.y, posit_sample.z)
            x_next = Vec3(posit_sample.x + h, posit_sample.y, posit_sample.z)
            y_prev = Vec3(posit_sample.x, posit_sample.y - h, posit_sample.z)
            y_next = Vec3(posit_sample.x, posit_sample.y + h, posit_sample.z)
            z_prev = Vec3(posit_sample.x, posit_sample.y, posit_sample.z - h)
            z_next = Vec3(posit_sample.x, posit_sample.y, posit_sample.z + h)

            psi_x_prev = 0.
            psi_x_next = 0.
            psi_y_prev = 0.
            psi_y_next = 0.
            psi_z_prev = 0.
            psi_z_next = 0.

            for nuc in nuclei:
                psi_x_prev += wf(nuc, x_prev)
                psi_x_next += wf(nuc, x_next)
                psi_y_prev += wf(nuc, y_prev)
                psi_y_next += wf(nuc, y_next)
                psi_z_prev += wf(nuc, z_prev)
                psi_z_next += wf(nuc, z_next)

            # print(psi_x_prev)

            # Could combine these, but this split-by-axis is easier to understand.

            # todo: Is this how you compensate for h??. We removed the 1/4 term and added h^2.
            
            # Initialize using the second derivative temr
            # psi_pp_measured[i] = 0.
            # psi_pp_measured[i] += (psi_x_prev + psi_x_next - 2. * psi[i])
            # psi_pp_measured[i] += (psi_y_prev + psi_y_next - 2. * psi[i])
            # psi_pp_measured[i] += (psi_z_prev + psi_z_next - 2. * psi[i])
            # psi_pp_measured[i] /= h**2

            psi_pp_measured_2d[i, j] = 0.
            psi_pp_measured_2d[i, j] += psi_x_prev + psi_x_next - 2. * psi_2d[i, j]
            psi_pp_measured_2d[i, j] += psi_y_prev + psi_y_next - 2. * psi_2d[i, j]
            psi_pp_measured_2d[i, j] += psi_z_prev + psi_z_next - 2. * psi_2d[i, j]
            psi_pp_measured_2d[i, j] /= h**2

    # print(f"ψ'' Score: {round(score_wf(psi_pp_expected, psi_pp_measured), 5)}")
    print(f"ψ'' Score: {round(score_wf_2d(psi_pp_expected_2d, psi_pp_measured_2d), 5)}")

    # Maybe for each trial WF, find the E that minimizes least-squared between two psi''s?

    # plot_1d(x_vals, [V_vals, psi, psi_pp_expected, psi_pp_measured])

    plot_2d(x_vals, y_vals, [
        # V_vals_2d,
        # psi_2d,
        psi_pp_expected_2d,
        psi_pp_measured_2d
    ]) 


main()