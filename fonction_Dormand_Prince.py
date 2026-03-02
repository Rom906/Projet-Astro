from utils import Vector
from typing import Callable, List


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
        k_i = f(y_stage, t_stage)
        k.append(k_i)

    # Compute the 5th order solution
    y_next = y.copy()
    for i in range(7):
        y_next = y_next + (k[i] * (b5[i] * h))

    return y_next
