import typing
from typing import Callable, List, Tuple
from utils import Vector
from typing import List, Callable


def adams(
    liste_ui: List[Vector],
    f: Callable[[float, Vector], Vector],
    ti: float,
    h: float,
    number_of_steps: int,
) -> Vector:
    """
    Calculates an integration step using explicit Adams-Bashforth

    param previous_conditions: all ui to calculate ui+1, in order (u0,u1,u2 etc...)
    type previous_conditions: List['Vector']
    param f: function of the ODE Y' = f(Y) that takes in entry a time and a vector and returns a vector
    type f: function
    param ti: initial time with relation t_{i+1} = t_i + h
    type ti: float
    param h: step size
    type h: int
    param number_of_steps: number of previous step used
    type number_of_steps: int
    rtype: Vector
    """

    betas = [1, 1 / 2, 1 / 12, 1 / 24]
    alphas = [(1,), (3, -1), (23, -16, 5), (55, -59, 37, -9)]

    beta_choisi = betas[number_of_steps - 1]
    alpha_choisi = alphas[number_of_steps - 1]

    sum = Vector(
        [
            Vector([0 for i in range(liste_ui[0][0].dimension)])
            for i in range(liste_ui[0].dimension)
        ]
    )

    for i in range(number_of_steps):
        alpha_i = alpha_choisi[i]
        u_pred = liste_ui[-i]
        t_pred = ti - (i - 1) * h

        produit = alpha_i * f(t_pred, u_pred)

        sum += produit

    ui_plus_1 = liste_ui[-1] + beta_choisi * h * sum

    return ui_plus_1


def euler(
    previous_conditions: List[Vector],
    f: Callable[[float, Vector], Vector],
    ti: float,
    h: float,
    number_of_steps: int,
) -> Vector:
    """
    Calculates an integration step using explicit Euler

    param previous_conditions: all ui to calculate ui+1, in order (u0,u1,u2 etc...)
    type previous_conditions: List['Vector']
    param f: function of the ODE Y' = f(Y) that takes in entry a time and a vector and returns a vector
    type f: function
    param ti: initial time with relation t_{i+1} = t_i + h
    type ti: float
    param h: step size
    type h: int
    param number_of_steps: number of previous step used
    type number_of_steps: int
    rtype: Vector
    """
    Y = previous_conditions[0]
    Y_next = Y + h * f(ti, Y)
    return Y_next


def RK4(previous_conditions, f, t, h, number_of_steps):
    """
    Calculates an integration step using RK4

    param previous_conditions: all ui to calculate ui+1, in order (u0,u1,u2 etc...)
    type previous_conditions: List['Vector']
    param f: function of the ODE Y' = f(Y) that takes in entry a time and a vector and returns a vector
    type f: function
    param ti: initial time with relation t_{i+1} = t_i + h
    type ti: float
    param h: step size
    type h: int
    param number_of_steps: number of previous step used
    type number_of_steps: int
    rtype: Vector
    """
    # The most recent state
    Y = previous_conditions[0]

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


def dormand_prince(
    previous_conditions: List[Vector],
    f: Callable[[Vector, float], Vector],
    ti: float,
    h: float,
    number_of_steps: int,
) -> Vector:
    """
    Calculates an integration step using Dormand-Prince

    param previous_conditions: all ui to calculate ui+1, in order (u0,u1,u2 etc...)
    type previous_conditions: List['Vector']
    param f: function of the ODE Y' = f(Y) that takes in entry a time and a vector and returns a vector
    type f: function
    param ti: initial time with relation t_{i+1} = t_i + h
    type ti: float
    param h: step size
    type h: int
    param number_of_steps: number of previous step used
    type number_of_steps: int
    rtype: Vector
    """

    # Extract the current state (last element of previous_conditions list)
    Y = previous_conditions[-1]
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
        Y_stage = y.copy()
        for j in range(i):
            Y_stage = Y_stage + (k[j] * (a[i][j] * h))

        # Evaluate f at the stage point
        t_stage = t + c[i] * h
        k_i = f(t_stage, Y_stage)
        k.append(k_i)

    # Compute the 5th order solution
    Y_next = Y.copy()
    for i in range(7):
        Y_next = Y_next + (k[i] * (b5[i] * h))

    return Y_next


def velocity_verlet(
    previous_conditions: List[Vector], f: Callable[[Vector, float], Vector], ti: float, h: float, number_of_steps: int
) -> Vector:
    """
    Calculates an integration step using Velocity-Verlet

    param previous_conditions: all ui to calculate ui+1, in order (u0,u1,u2 etc...)
    type previous_conditions: List['Vector']
    param f: function of the ODE Y' = f(Y) that takes in entry a time and a vector and returns a vector
    type f: function
    param ti: initial time with relation t_{i+1} = t_i + h
    type ti: float
    param h: step size
    type h: int
    param number_of_steps: number of previous step used
    type number_of_steps: int
    rtype: Vector
    """

    # Extract current state (last element of previous_conditions list)
    Y_current = previous_conditions[-1]

    # Extract position and velocity components
    # Y = [position, velocity]
    r_current = Y_current[0]  # Current position
    v_current = Y_current[1]  # Current velocity

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
    Y_new_intermediate = Vector([r_new, v_half])
    f_new = f(ti + h, Y_new_intermediate)
    a_new = f_new[1]

    # Step 5: Full-step velocity update
    v_new = v_half + a_new * (h / 2.0)

    # Return the updated state vector [r_new, v_new]
    return Vector([r_new, v_new])


def Heun(
    previous_conditions: List[Vector],
    f: Callable[[Vector, float], Vector],
    t: float,
    h: float,
    number_of_steps: int,
) -> Vector:
    """
    Calculates an integration step using Heun

    param previous_conditions: all ui to calculate ui+1, in order (u0,u1,u2 etc...)
    type previous_conditions: List['Vector']
    param f: function of the ODE Y' = f(Y) that takes in entry a time and a vector and returns a vector
    type f: function
    param ti: initial time with relation t_{i+1} = t_i + h
    type ti: float
    param h: step size
    type h: int
    param number_of_steps: number of previous step used
    type number_of_steps: int
    rtype: Vector
    """
    # Extract current state Y_n (last element of history)
    Y_current = prev_steps[-1]
    t_next = t + h

    # Extract position and velocity components explicitly
    # Y_current is a Vector where Y[0] = position, Y[1] = velocity
    r_current = Y_current[0]
    v_current = Y_current[1]

    # --- Step 1: Predictor (Explicit Euler) ---

    # Calculate slopes at current time: k1 = f(t_n, Y_n) = [r_n, v_n]
    k1 = f(t, Y_current)
    v_slope_1 = k1[0]  # Should be v_current
    a_slope_1 = k1[1]  # Acceleration at t

    # Estimate next state components using Euler
    # r_pred = r_n + h * v_n
    r_pred = r_current + v_slope_1 * h
    # v_pred = v_n + h * a_n
    v_pred = v_current + a_slope_1 * h

    # Construct the predicted state vector
    Y_predict = Vector([r_pred, v_pred])

    # --- Step 2: Corrector (Explicit Average) ---

    # Calculate slopes at the predicted state: k2 = f(t_{n+1}, Y_pred)
    k2 = f(t_next, Y_predict)
    v_slope_2 = k2[0]  # Velocity at predicted state
    a_slope_2 = k2[1]  # Acceleration at predicted state

    # Compute final state components using the average of the two slopes
    # r_{n+1} = r_n + (h/2) * (v_n + v_pred)
    r_new = r_current + (v_slope_1 + v_slope_2) * (h / 2.0)

    # v_{n+1} = v_n + (h/2) * (a_n + a_pred)
    v_new = v_current + (a_slope_1 + a_slope_2) * (h / 2.0)

    # Reconstruct the result vector [position, velocity]
    Y_new = Vector([r_new, v_new])

    return Y_new
