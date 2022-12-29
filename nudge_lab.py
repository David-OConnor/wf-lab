from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from numpy import sqrt, pi, cos, ndarray, exp, sin
import math

TAU = 2. * pi
N = 1000
NUDGE_AMT = 0.1
NUM_NUDGES = 3
h = (20/N)

def main():
	x = np.linspace(-10, 10, N)
	psi = sin(x)

	psipp_calc = -sin(x) + sin(2. * x) * 0.5

	psipp_meas = np.zeros(N)
	for i in range(N):
		if i == 0 or i == N-1:
			continue
		psipp_meas[i] = (psi[i-1] + psi[i+1] - 2. * psi[i]) / h**2

	plt.plot(psi)
	plt.plot(psipp_calc)
	plt.plot(psipp_meas)
	plt.show()

	plt.ylim(-2., 2.)

	for _ in range(NUM_NUDGES):

		diff = psipp_calc - psipp_meas
		psi -= diff * NUDGE_AMT

		psipp_meas = np.zeros(N)
		for i in range(N):
			if i == 0 or i == N-1:
				continue
			psipp_meas[i] = (psi[i-1] + psi[i+1] - 2. * psi[i]) / h**2

		# plt.plot(x)
	plt.plot(psi)
	plt.plot(psipp_calc)
	plt.plot(psipp_meas)

	plt.show()


main()