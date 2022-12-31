from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from numpy import sqrt, pi, cos, ndarray, exp, sin
import math

TAU = 2. * pi
N = 1000
NUDGE_AMT = 0.0001
NUM_NUDGES = 20
h = (20/N)

EPS_DIFF = 0.0001 # todo: tie to nudge amt?

def main():
	x = np.linspace(-10, 10, N)
	psi = sin(x)

	psipp_calc = -sin(x) + sin(2. * x) * 0.2

	psipp_meas = np.zeros(N)
	for i in range(N):
		if i == 0 or i == N-1:
			continue

		psipp_meas[i] = (psi[i-1] + psi[i+1] - 2. * psi[i]) / h**2

	# plt.plot(psi)
	# plt.plot(psipp_calc)
	# plt.plot(psipp_meas)
	# plt.show()

	plt.ylim(-2., 2.)

	for _ in range(NUM_NUDGES):

		diff = psipp_calc - psipp_meas

		for i in range(N):
			if i == 0 or i == N-1:
				continue

			psi[i] -= diff[i] * NUDGE_AMT
			psi[i+1] += diff[i] * NUDGE_AMT
			psi[i-1] += diff[i] * NUDGE_AMT

		psipp_meas = np.zeros(N)
		for i in range(N):
			if i == 0 or i == N-1:
				continue
			# if np.abs(diff[i]) < EPS_DIFF:
			# 	continue
			psipp_meas[i] = (psi[i-1] + psi[i+1] - 2. * psi[i]) / h**2

	plt.plot(psi)
	plt.plot(psipp_calc)
	plt.plot(psipp_meas)

	plt.show()


main()