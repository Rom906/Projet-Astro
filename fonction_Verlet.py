from utils import Vector
from typing import Callable, List


def velocity_verlet(
    u_i: List[Vector], f: Callable[[Vector, float], Vector], ti: float, h: float, m: int
) -> Vector:
    """
    Velocity Verlet-like method for solving generic systems of differential equations.

    This is a 2nd order predictor-corrector integrator that approximates the exact solution
    of dy/dt = f(y, t) using a Modified Euler (Heun) scheme with velocity-like structure.

    This approach is efficient for systems with rapidly varying components and maintains
    good stability properties for long-term simulations.

    Parameters:
    -----------
    u_i : List[Vector]
        List of previous solution vectors. For single-step methods, only u_i[-1] is used.
    f : Callable[[Vector, float], Vector]
        The differential equation function: dy/dt = f(y, t)
    ti : float
        Current time point
    h : float
        Fixed time step size
    m : int
        Number of steps used by the method (not used for single-step method, kept for compatibility)

    Returns:
    --------
    Vector
        The solution at time ti + h

    Algorithm:
    ----------
    This method uses a two-stage predictor-corrector approach:

    1. y_pred = y(ti) + h * f(ti, y(ti))           [predictor step]
    2. f_pred = f(ti + h, y_pred)                   [evaluate f at predictor]
    3. y_new = y(ti) + (h/2) * [f(ti, y(ti)) + f_pred]  [corrector step (trapezoidal)]

    Notes:
    ------
    - 2nd order method: O(h³) local truncation error, O(h²) global error
    - Good stability for non-stiff problems
    - Compatible with generic systems of differential equations
    - Less sensitive to system structure than specialized Verlet methods
    """

    # Extract current state (last element of u_i list)
    y = u_i[-1]

    # Step 1: Evaluate f at current time
    f_current = f(ti, y)

    # Step 2: Predictor step - forward Euler
    y_pred = y + f_current * h

    # Step 3: Evaluate f at predicted point
    f_pred = f(ti + h, y_pred)

    # Step 4: Corrector step - trapezoidal rule (average of two slopes)
    y_new = y + (f_current + f_pred) * (h / 2.0)

    return y_new
