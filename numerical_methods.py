"""
Consolidated module for all numerical integration methods.

This module contains all numerical methods for solving differential equations and integrating
particle trajectories in electromagnetic fields. Includes:
- Euler methods (single step and full integration)
- Runge-Kutta 4 (RK4) methods
- Dormand-Prince adaptive method
- Velocity Verlet symplectic integrator
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from utils import Vector
from typing import Callable, List


# ==================== EULER METHODS ====================

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


def Euler(rp, vp, t, dt, Nsteps, q, m, B, ROdip, mu):
    """
    Euler method for integrating particle motion in a magnetic field using numpy arrays.

    Parameters:
    -----------
    rp : ndarray
        Position array (Nsteps, 3)
    vp : ndarray
        Velocity array (Nsteps, 3)
    t : ndarray
        Time array (Nsteps,)
    dt : float
        Time step
    Nsteps : int
        Number of steps
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
    
    Returns:
    --------
    rp : ndarray
        Position array
    vp : ndarray
        Velocity array
    t : ndarray
        Time array
    KE : ndarray
        Kinetic energy array
    comp_time : float
        Computation time in seconds
    """

    start_time = time.time()

    # Initialize kinetic energy array
    KE = np.zeros(Nsteps)
    KE[0] = 0.5 * m * np.dot(vp[0, :], vp[0, :])

    # Integration loop using Euler method
    for i in range(1, Nsteps):
        # Update position: r(t+dt) = r(t) + v(t)*dt
        rp[i] = rp[i - 1, :] + vp[i - 1, :] * dt

        # Calculate acceleration: a = (q/m) * (v × B)
        ap = q / m * np.cross(vp[i - 1, :], B(rp[i - 1, :], ROdip, mu))

        # Update velocity: v(t+dt) = v(t) + a(t)*dt
        vp[i] = vp[i - 1, :] + ap * dt

        # Update time
        t[i] = dt * i

        # Calculate kinetic energy: KE = 0.5 * m * v²
        KE[i] = 0.5 * m * np.dot(vp[i, :], vp[i, :])

    comp_time = time.time() - start_time

    # Final Printout
    print(f"\n=== Euler Method Statistics ===")
    print(f"Number of points: {Nsteps}")
    print(f"Computation time: {comp_time:.4f} s")
    print(f"Time step (dt): {dt} s")
    print(f"Sampling time (t): {t[-1]:.2f} s")
    print(f"===============================\n")

    return rp, vp, t, KE, comp_time


# ==================== RUNGE-KUTTA 4 (RK4) METHODS ====================

def RK4_numpy(rp, vp, t, dt, Nsteps, q, m, B, ROdip, mu):
    """
    Legacy RK4 implementation operating on numpy arrays.

    Parameters kept similar to the original code but with an explicit Nsteps parameter.
    
    Parameters:
    -----------
    rp : ndarray
        Position array
    vp : ndarray
        Velocity array
    t : ndarray
        Time array
    dt : float
        Time step
    Nsteps : int
        Number of steps
    q : float
        Charge
    m : float
        Mass
    B : function
        Magnetic field function
    ROdip : ndarray
        Dipole location
    mu : ndarray
        Magnetic moment
        
    Returns:
    --------
    tuple : (rp, vp, t) updated arrays
    """
    for i in range(1, Nsteps):
        rp1 = rp[i - 1, :]
        vp1 = vp[i - 1, :]
        ap1 = q / m * np.cross(vp1, B(rp1, ROdip, mu))

        rp2 = rp[i - 1, :] + 0.5 * vp1 * dt
        vp2 = vp[i - 1, :] + 0.5 * ap1 * dt
        ap2 = q / m * np.cross(vp2, B(rp2, ROdip, mu))

        rp3 = rp[i - 1, :] + 0.5 * vp2 * dt
        vp3 = vp[i - 1, :] + 0.5 * ap2 * dt
        ap3 = q / m * np.cross(vp3, B(rp3, ROdip, mu))

        rp4 = rp[i - 1, :] + vp3 * dt
        vp4 = vp[i - 1, :] + ap3 * dt
        ap4 = q / m * np.cross(vp4, B(rp4, ROdip, mu))

        rp[i] = rp[i - 1, :] + (dt / 6.0) * (vp1 + 2 * vp2 + 2 * vp3 + vp4)
        vp[i] = vp[i - 1, :] + (dt / 6.0) * (ap1 + 2 * ap2 + 2 * ap3 + ap4)
        t[i] = dt * i
    return rp, vp, t


def RK4(vector_list, differential_equation, t, h, number_of_steps):
    """
    Runge-Kutta 4 single step compatible with `compute_solution`.

    Expected inputs:
      - vector_list: List[Vector] where vector_list[0] is the most recent state Y_n.
      - differential_equation: function f(t, Y) -> Vector representing Y'.
      - t: current time (float)
      - h: timestep (float)
      - number_of_steps: not used for single-step RK4 but kept for API compatibility.

    Parameters:
    -----------
    vector_list : List[Vector]
        List of previous solution vectors
    differential_equation : Callable
        Function representing the differential equation
    t : float
        Current time
    h : float
        Time step
    number_of_steps : int
        Number of steps (for compatibility)
        
    Returns:
    --------
    Vector
        Estimated Y_{n+1}
    """
    # The most recent state
    Y = vector_list[0]

    # k1 = f(t, Y)
    k1 = differential_equation(t, Y)

    # k2 = f(t + h/2, Y + k1 * (h/2))
    k2 = differential_equation(t + h / 2.0, Y + k1 * (h / 2.0))

    # k3 = f(t + h/2, Y + k2 * (h/2))
    k3 = differential_equation(t + h / 2.0, Y + k2 * (h / 2.0))

    # k4 = f(t + h, Y + k3 * h)
    k4 = differential_equation(t + h, Y + k3 * h)

    # Combine to produce next value
    Y_next = Y + (k1 * (h / 6.0) + k2 * (h / 3.0) + k3 * (h / 3.0) + k4 * (h / 6.0))

    return Y_next


# ==================== DORMAND-PRINCE METHOD ====================

def dormand_prince(
    u_i: List[Vector], f: Callable[[Vector, float], Vector], ti: float, h: float, m: int
) -> Vector:
    """
    Dormand-Prince 5(4) method for solving differential equations.

    This is an explicit Runge-Kutta method with 7 stages, providing both 5th and 4th order solutions.
    With fixed step size h.

    Parameters:
    -----------
    u_i : List[Vector]
        List of previous solution vectors. For single-step methods, only u_i[-1] is used.
    f : Callable[[Vector, float], Vector]
        The differential equation function: y' = f(y, t)
    ti : float
        Current time point
    h : float
        Fixed time step size
    m : int
        Number of steps used by the method (not used for single-step Dormand-Prince, kept for compatibility)

    Returns:
    --------
    Vector
        The solution at time ti + h
    """

    # Extract the current state (last element of u_i list)
    y = u_i[-1]
    t = ti

    # Dormand-Prince 5(4) coefficients
    # Butcher tableau coefficients
    c = [0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0]

    # a[i][j] coefficients for computing k_i found on wikipedia page for Dormand-Prince method
    a = [
        [],
        [0.2],
        [3.0 / 40.0, 9.0 / 40.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
        [
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ],
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ]

    # Weights for 5th order solution
    b5 = [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ]

    # Compute the 7 stages (k values)
    k = []
    for i in range(7):
        # Compute the argument for f
        y_stage = y.copy()
        for j in range(i):
            y_stage = y_stage + (k[j] * (a[i][j] * h))

        # Evaluate f at the stage point
        t_stage = t + c[i] * h
        k_i = f(t_stage, y_stage)
        k.append(k_i)

    # Compute the 5th order solution
    y_next = y.copy()
    for i in range(7):
        y_next = y_next + (k[i] * (b5[i] * h))

    return y_next


# ==================== VELOCITY VERLET METHOD ====================

def velocity_verlet(
    u_i: List[Vector], f: Callable[[Vector, float], Vector], ti: float, h: float, m: int
) -> Vector:
    """
    Velocity Verlet method for solving Hamiltonian systems of differential equations.

    This is a 2nd order symplectic integrator, particularly efficient for systems
    that decompose into position and velocity components.

    The method is time-reversible and energy-conserving, making it ideal for long-term
    simulations and problems in classical mechanics and electromagnetism.

    Parameters:
    -----------
    u_i : List[Vector]
        List of previous solution vectors. For single-step methods, only u_i[-1] is used.
        Expected structure: u_i[-1] = [position_vector, velocity_vector]
    f : Callable[[Vector, float], Vector]
        The differential equation function: Y' = f(τ, Y)
        With Y = [r, v], returns [dr/dτ, dv/dτ] = [velocity, acceleration]
    ti : float
        Current time point (in normalized time τ)
    h : float
        Fixed time step size
    m : int
        Number of steps used by the method (not used for Velocity Verlet, kept for compatibility)

    Returns:
    --------
    Vector
        The solution at time ti + h, structured as [position_new, velocity_new]

    Algorithm:
    ----------
    For a system with structure Y = [r, v] where:
    - dr/dτ = velocity (dY[0]/dτ) = f(...)[0]
    - dv/dτ = acceleration (dY[1]/dτ) = f(...)[1]

    The Velocity Verlet scheme works as follows:

    1. v(τ + h/2) = v(τ) + (h/2) * a(τ)          [half-step velocity]
    2. r(τ + h)   = r(τ) + h * v(τ + h/2)        [full-step position]
    3. a(τ + h)   = f(τ + h, [r_new, v_{h/2}])[1] [acceleration at new position]
    4. v(τ + h)   = v(τ + h/2) + (h/2) * a(τ + h) [full-step velocity]

    Notes:
    ------
    - Symplectic integrator: preserves the structure of Hamiltonian systems
    - Error is O(h³) per step, O(h²) for global error
    - Energy drift is minimal even for very long simulations
    - Time-reversible and volume-preserving in phase space
    """

    # Extract current state (last element of u_i list)
    y_current = u_i[-1]

    # Extract position and velocity components
    # Y = [position, velocity]
    r_current = y_current[0]  # Current position
    v_current = y_current[1]  # Current velocity

    # Step 1: Evaluate derivatives at current time
    f_current = f(ti, y_current)
    # f_current[0] = dr/dτ = velocity (already have it)
    # f_current[1] = dv/dτ = acceleration
    a_current = f_current[1]

    # Step 2: Half-step velocity update
    v_half = v_current + a_current * (h / 2.0)

    # Step 3: Full-step position update
    r_new = r_current + v_half * h

    # Step 4: Evaluate derivatives at new position
    y_new_intermediate = Vector([r_new, v_half])
    f_new = f(ti + h, y_new_intermediate)
    a_new = f_new[1]

    # Step 5: Full-step velocity update
    v_new = v_half + a_new * (h / 2.0)

    # Return the updated state vector [r_new, v_new]
    return Vector([r_new, v_new])


# ==================== VISUALIZATION FUNCTIONS ====================

def plot_trajectory(rp):
    """
    Plot the 3D trajectory of the particle.

    Parameters:
    -----------
    rp : ndarray
        Position array (Nsteps, 3)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(rp[:, 0], rp[:, 1], rp[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Particle Trajectory (Euler Method)")
    plt.show()
