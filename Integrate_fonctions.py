import typing
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


def euler(
    liste_ui: typing.List[Vector],
    f: typing.Callable[[float, Vector], Vector],
    ti: float,
    h: float,
    m: int,
) -> Vector:
    vector = liste_ui[0]
    ui_plus_un = vector + h * f(ti, vector)
    return ui_plus_un


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


def dormand_prince(
    u_i: typing.List[Vector],
    f: typing.Callable[[Vector, float], Vector],
    ti: float,
    h: float,
    m: int,
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
