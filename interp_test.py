import numpy as np 
import matplotlib.pyplot as plt
from numpy import sqrt, sin, cos, exp, pi, log
import matplotlib as mpl

import math

from scipy.spatial import Delaunay
from scipy.interpolate import griddata, RBFInterpolator

from scipy.stats.qmc import Halton # from RBF offial example

from dataclasses import dataclass

TAU = 2* pi

N = 20

MAX = -3
MIN = 3

Ni = 40
Pi = np.random.rand(Ni, 2)
Xi, Yi = Pi[:,0], Pi[:,1]
Zi = np.random.rand(Ni)

@dataclass
class Vec3:
	x: float
	y: float
	z: float


@dataclass
class Vec2:
	x: float
	y: float


def h100(nuc: Vec3, sample: Vec3) -> float:
	diff = Vec3 (
		sample.x - nuc.x,
		sample.y - nuc.y,
		sample.z - nuc.z,
	)	

	r = sqrt(diff.x**2 + diff.y**2 + diff.z**2)

	return 2. * np.exp(-r) # todo? * r? r^2?




def test_fn(x: float, y: float) -> float:
	r = sqrt(x**2 + y**2)

	return r


def lin_interp(pt0, pt1, x) -> float:
	port = x / (pt1[0] - pt0[0])

	return port * (pt1[1] - pt0[1]) + pt0[1]

# todo: 3d one with spherical coords to cart
# theta and r are anchored to the centern point. The center point and returned value
# are in global, cartesian coords.
def polar_to_cart(ctr: Vec2, theta: float, r: float) -> Vec2:
	x = ctr.x + cos(theta) * r
	y = ctr.y + sin(theta) * r
	
	return Vec2(x, y)

# todo: WHich convention?
def spherical_to_cart(ctr: Vec3, θ: float, φ: float, r: float) -> Vec3:
	x= ctr.x + r * sin(φ) * cos(θ)
	y= ctr.y + r * sin(φ) * sin(θ)
	z= ctr.z + r * cos(φ)
	
	return Vec3(x, y, z)



def test_rbf():
	# todo: 2D to start; then move to 3D

	grid_max = 3.
	grid_min = -grid_max

	# Determine how to set up our sample points
	N_RADIALS = 10
	N_DISTS = 8
	MAX_DIST = 4.

	ANGLE_BW_RADS = TAU / N_RADIALS
	# DIST_BW_RINGS = MAX_RNG / N_DISTS # todo: Fixed sit

	DIST_CONST = 0.05 # c^n_dists = max_dist
	# DIST_CONST = math.log(MAX_DIST, 2)
	# DIST_DECAY_EXP= 0.2

	NUM_NUCS = 2

	nuc1 = Vec3(-1.5, 0., 0.)
	nuc2 = Vec3(1.5, 0., 0.)

	N_SAMPLES = N_RADIALS * N_DISTS * NUM_NUCS

	# todo: Dist falloff, since we use more dists closer to the nuclei?

	# `xobs` is a an array of X, Y pts. Rust equiv type might be
	# &[Vec3]
	xobs = np.zeros((N_SAMPLES, 2))

	i = 0

	for radial_i in range(N_RADIALS):
		theta = radial_i * ANGLE_BW_RADS
		# We don't use dist = 0.

		for dist_i in range(1, N_DISTS + 1): # Don't use ring @ r=0.
			# todo: Evenly spaced for now
			# r = 1./dist_i * DIST_BW_RINGS
			# r = exp(-DIST_DECAY_EXP * dist_i) * DIST_CONST
			r = dist_i**2 * DIST_CONST

			# todo: center around each nuc!!
			polar1 = polar_to_cart(nuc1, theta, r)
			polar2 = polar_to_cart(nuc2, theta, r)

			xobs[i] = np.array([polar1.x, polar1.y])
			xobs[i + 1] = np.array([polar2.x, polar2.y])

			i += NUM_NUCS

	# z_slice = ctr[2]

		# `yobs` is the function values at each of the sample points.
	# eg `&[Cplx]`
	yobs = np.zeros(N_SAMPLES)
	# Iterate over our random sample of points
	for i, grid_pt in enumerate(xobs):
		yobs[i] = h100(nuc1, Vec3(grid_pt[0], grid_pt[1], 0.)) + \
			h100(nuc2, Vec3(grid_pt[0], grid_pt[1], 0.))
	

	# Plot a slice on a 2d grid.

	xgrid = np.mgrid[grid_min:grid_max:50j, grid_min:grid_max:50j]

	xflat = xgrid.reshape(2, -1).T


	yflat = RBFInterpolator(xobs, yobs, kernel='cubic')(xflat)

	ygrid = yflat.reshape(50, 50)

	fig, ax = plt.subplots()

	ax.pcolormesh(*xgrid, ygrid, vmin=grid_min, vmax=grid_max, shading='gouraud')

	p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)

	fig.colorbar(p)

	plt.show()


def main():

	test_rbf()

	return

	Pi = np.array([Xi, Yi]).transpose()
	tri = Delaunay(Pi)
	plt.triplot(Xi, Yi , tri.simplices.copy())
	plt.plot(Xi, Yi, "or", label = "Data")
	plt.grid()
	plt.legend()
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()

	###

	N2 = 100
	x = np.linspace(0., 1., N2)
	y = np.linspace(0., 1., N2)
	X, Y = np.meshgrid(x, y)
	P = np.array([X.flatten(), Y.flatten() ]).transpose()
	plt.plot(Xi, Yi, "or", label = "Data")
	plt.triplot(Xi, Yi , tri.simplices.copy())
	plt.plot(X.flatten(), Y.flatten(), "g,", label = "Z = ?")
	plt.legend()
	plt.grid()
	plt.show()


	###


	Z_cubic = griddata(Pi, Zi, P, method = "cubic").reshape([N2, N2])
	plt.contourf(X, Y, Z_cubic, 50, cmap = mpl.cm.jet)
	plt.colorbar()
	plt.contour(X, Y, Z_cubic, 20, colors = "k")
	#plt.triplot(Xi, Yi , tri.simplices.copy(), color = "k")
	plt.plot(Xi, Yi, "or", label = "Data")
	plt.legend()
	plt.grid()
	plt.show()


	### split ##

	vals = np.zeros((N, N))

	grid_dx = (MAX - MIN) / N

	ctr = (N / 2, N / 2)

	grid_1d = np.linspace(MIN, MAX, N);

	for i, x in enumerate(grid_1d):
		for j, y in enumerate(grid_1d):
			vals[i][j] = test_fn(x, y)

	plt.imshow(vals)
	plt.show()




main()