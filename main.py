from Initialisation import *
from numerical_methods import RK4_numpy as RK4

# === RK4 implementation ===
rp, vp, t = RK4(rp, vp, t, dt, Nsteps, q, m, B, ROdip, mu)
