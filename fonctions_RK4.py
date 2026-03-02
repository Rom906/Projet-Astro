import numpy as np
import matplotlib.pyplot as plt

RT = 6371000.0  # Earth radius [m]


# ==== RK4 implementation for numpy arrays ====


def normalize_r_position(rp):
    return rp / RT


def RK4_numpy(rp, vp, t, dt, Nsteps, q, m, B, ROdip, mu):
    """
    Legacy RK4 implementation operating on numpy arrays.

    Parameters kept similar to the original code but with an explicit Nsteps parameter.
    Returns updated (rp, vp, t).
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


# ==== RK4 implementation compatible with compute_solution and Vector objects ====


def RK4(vector_list, differential_equation, t, h, number_of_steps):
    """
    Runge-Kutta 4 single step compatible with `compute_solution`.

    Expected inputs:
      - vector_list: List[Vector] where vector_list[0] is the most recent state Y_n.
      - differential_equation: function f(t, Y) -> Vector representing Y'.
      - t: current time (float)
      - h: timestep (float)
      - number_of_steps: not used for single-step RK4 but kept for API compatibility.

    Returns:
      - Vector: the estimated Y_{n+1}
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
