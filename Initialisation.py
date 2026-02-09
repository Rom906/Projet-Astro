import numpy as np
import matplotlib.pyplot as plt
from fonctions_RK4 import RK4, plot_trajectory
# ==== Def Key constants ====
RT= 6371000. # Earth radius [m]
m_p = 1.67E-27 # mass of proton [kg]
m_e = 9.109E-31 # mass of electron [kg]
qe = 1.602E-19 # charge of proton [c]
phi = 11.70*np.pi/180. # Magnetic dipole tilt [rad]
th = 23.67*np.pi/180.# Earth's obliquity [rad]
mu=-7.94e22*np.array([.0, np.sin(phi), np.cos(phi)]) # Earth's magnetic moment [A m2]
ROdip = np.array ([0.0, 0.0, 0.0]) # Dipole moment location
MO=1.0E-7 #mu0/4pi
# =====Def. magnetic field at point r ====
def B(R,RO,mu) :
    r = np.array ( [R[0]-RO [0] , R[1]-RO [1] , R [2]-RO [2]]) *RT
    rmag = np.sqrt (r[0] ** 2 + r [1] ** 2+ r[2] ** 2)
    Bfield=MO*(3.0*r*np.dot(mu,r)/(rmag ** 5)-mu/(rmag ** 3))
    return Bfield

# ===setup time steps====
dt = 0.1
tf = 5000.
Nsteps = int (tf/dt)

#==== vector initialization====
t = np.linspace(0, tf, Nsteps)
rp = np.zeros((len(t), 3))
vp = np.zeros ((len(t), 3))

# ====== Setup a charged particle====
m = 4.0*m_p
q = 2.0*qe

#==== Define initial conditions====
t[0] = 0.
rp [0,:] = np.array([5. , 5., 5.])
vp [0,:] = np.array([1.,1., 1.])

rp, vp, t = RK4(rp, vp, t, dt, Nsteps, q, m, B, ROdip, mu)
plot_trajectory(rp)