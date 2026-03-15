import numpy as np
from numerical_methods import (
    RK4_numpy as RK4
)

# ==== Key constants ====
RT = 6371000.0  # Earth radius [m]
m_p = 1.67e-27  # mass of proton [kg]
qe = 1.602e-19  # charge of proton [C]
phi = 11.70 * np.pi / 180.0  # Magnetic dipole tilt [rad]
mu = -7.94e22 * np.array(
    [0.0, np.sin(phi), np.cos(phi)]
)  # Earth's magnetic moment [A m2]
ROdip = np.array([0.0, 0.0, 0.0])  # Dipole location
MO = 1.0e-7  # mu0/4pi


# ===== Magnetic field definition =====
def B(R, RO, mu):
    # R is in meters
    r = R - RO
    rmag = np.linalg.norm(r)
    Bfield = MO * (3.0 * r * np.dot(mu, r) / rmag**5 - mu / rmag**3)
    return Bfield


# ===== Time steps =====
dt = 0.01  # small enough to resolve trajectory
tf = 500.0  # total simulation time
Nsteps = int(tf / dt)

# ===== Vector initialization =====
t = np.linspace(0, tf, Nsteps)
rp = np.zeros((len(t), 3))
vp = np.zeros((len(t), 3))

# ===== Charged particle setup =====
m = 4.0 * m_p
q = 2.0 * qe

# ===== Initial conditions =====
t[0] = 0.0
rp[0, :] = np.array([5.0, 5.0, 5.0]) * RT  # position in meters
vp[0, :] = np.array([1e4, 2e4, 0.0])  # velocity in m/s

# ===== Run RK4 =====
rp, vp, t = RK4(rp, vp, t, dt, Nsteps, q, m, B, ROdip, mu)

# ===== Plot trajectory =====
# normalize only for plotting in Earth radii
# rp_plot = normalize_r_position(rp)
# plot_trajectory(rp_plot)

# plot_trajectory(rp)
print("Initialisation completed successfully!")
