import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


# ==== Euler implementation ====
def Euler(rp, vp, t, dt, Nsteps, q, m, B, ROdip, mu):
    """
    Euler method for integrating particle motion in a magnetic field.

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


# ==== Trajectory visualization ====
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
