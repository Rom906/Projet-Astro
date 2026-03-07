import typing
import numpy as np
from utils import Vector


def adams(
    liste_ui: typing.List[Vector],
    f: typing.Callable[[float, Vector], Vector],
    ti: float,
    h: float,
    m: int,
) -> Vector:
    """Calculate the next ui of the list, using the previous ones

    Parameters
    ----------
    liste ui : List['Vector']
        all ui to calculate ui+1, in order (u0,u1,u2 etc...)
    f : function
        ODE (ordinary differential equation) that takes in entry a time and a vector and returns a vector
    ti : float
        temps i avec la relation ti+1 = ti + h
    h : int
        step
    m : int
        number of previous step used"""

    betas = [1, 1 / 2, 1 / 12, 1 / 24]
    alphas = [(1,), (3, -1), (23, -16, 5), (55, -59, 37, -9)]

    beta_choisi = betas[m - 1]
    alpha_choisi = alphas[m - 1]

    sum = Vector(
        [
            Vector([0 for i in range(liste_ui[0][0].dimension)])
            for i in range(liste_ui[0].dimension)
        ]
    )

    for i in range(m):
        alpha_i = alpha_choisi[i]
        u_pred = liste_ui[-i]
        t_pred = ti - (i - 1) * h

        produit = alpha_i * f(t_pred, u_pred)

        sum += produit

    ui_plus_1 = liste_ui[-1] + beta_choisi * h * sum

    return ui_plus_1


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
