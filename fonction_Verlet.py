from utils import Vector
from typing import Callable, List


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
