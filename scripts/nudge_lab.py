from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from numpy import sqrt, pi, cos, ndarray, exp, sin
import math

TAU = 2. * pi
N = 1000
NUDGE_AMT = 0.001
NUM_NUDGES = 4
h = (20/N)

EPS_DIFF = 0.001 # todo: tie to nudge amt?

def main():
	x = np.linspace(-10, 10, N)
	psi = sin(x)

	psipp_calc = -sin(x) + sin(2. * x) * 0.2

	psipp_meas = np.zeros(N)
	for i in range(N):
		if i == 0:
			psi_prev = 2. * psi[i] - psi[i+1]
		else:
			psi_prev = psi[i-1]

		if i == N - 1:
			psi_next = 2. * psi[i] - psi[i-1]
		else:
			psi_next = psi[i+1]

		psipp_meas[i] = (psi_prev + psi_next - 2. * psi[i]) / h**2

	# plt.plot(psi)
	# plt.plot(psipp_calc)
	# plt.plot(psipp_meas)
	# plt.show()

	plt.ylim(-2., 2.)

	for _ in range(NUM_NUDGES):

		diff = psipp_calc - psipp_meas

		for i in range(N):
			psi[i] -= diff[i] * NUDGE_AMT
			if i != 0:
				psi[i-1] += diff[i] * NUDGE_AMT

			if i != N-1:
				psi[i+1] += diff[i] * NUDGE_AMT
			

		psipp_meas = np.zeros(N)
		for i in range(N):
			if i == 0:
				psi_prev = 2. * psi[i] - psi[i+1]
			else:
				psi_prev = psi[i-1]

			if i == N - 1:
				psi_next = 2. * psi[i] - psi[i-1]
			else:
				psi_next = psi[i+1]

			# if np.abs(diff[i]) < EPS_DIFF:
			# 	continue
			psipp_meas[i] = (psi_prev + psi_next - 2. * psi[i]) / h**2

	plt.plot(psi)
	plt.plot(psipp_calc)
	plt.plot(psipp_meas)

	plt.show()


main()