from math import pi, sin, cos
from utils import Vector

RT = 6371000.0  # Earth radius [m]
mp = 1.67e-27  # mass of proton [kg]
qe = 1.602e-19  # charge of proton [C]
phi = 11.70 * pi / 180.0  # Magnetic dipole tilt [rad]
mu = -7.94e22 * Vector([0.0, sin(phi), cos(phi)])  # Earth's magnetic moment [A m2]
ROdip = Vector([0.0, 0.0, 0.0])  # Dipole location
MO = 1.0e-7  # mu0/4pi
er = Vector([1, 0, 0]) # first unitay vector of the basis for spherical coordinates
