import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from utils import Vector


# ==== Euler single step iteration ====
def Euler_v2(R, V, q, m, B, ROdip, mu, dt):
    """
    Single step Euler method for particle motion in a magnetic field.
    
    This function performs one iteration of the Euler integration scheme.
    
    Parameters:
    -----------
    R : Vector
        Position vector (3D)
    V : Vector
        Velocity vector (3D)
    q : float
        Charge of the particle
    m : float
        Mass of the particle
    B : function
        Magnetic field function B(r, RO, mu)
    ROdip : ndarray
        Dipole location
    mu : ndarray
        Magnetic moment vector
    dt : float
        Time step
    
    Returns:
    --------
    R_new : Vector
        Updated position vector
    V_new : Vector
        Updated velocity vector
    KE : float
        Kinetic energy
    """
    
    # Convert Vector to numpy array for B field calculation
    R_array = np.array(R.coordinates)
    V_array = np.array(V.coordinates)
    
    # Update position: R[i] = R[i-1] + V[i-1] * dt
    R_new = R + V * dt
    
    # Calculate magnetic field at current position
    B_field = B(R_array, ROdip, mu)

    # Calculate acceleration: a = (q/m) * (V × B)
    # Using Vector's cross product operator @
    V_Vector = Vector(V_array.tolist())
    B_Vector = Vector(B_field.tolist())
    a_Vector = V_Vector @ B_Vector
    a_array = np.array(a_Vector.coordinates) * (q / m)
    a = Vector(a_array.tolist())

    # Update velocity: V[i] = V[i-1] + a * dt
    V_new = V + a * dt

    # Calculate kinetic energy: KE = 0.5 * m * ||V||²
    V_new_array = np.array(V_new.coordinates)
    KE = 0.5 * m * np.dot(V_new_array, V_new_array)
    
    return R_new, V_new, KE
