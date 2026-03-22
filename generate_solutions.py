from utils import Vector
from typing import Callable, List, Tuple
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from math import pi, cos, sin
import time
import numpy as np


def get_intervall(steps: int, minimum: float, maximum: float) -> List[float]:
    """
    return a liste of the specified number of values evenly spaced between "minimum" and "maximum" included

    :param steps: number of steps
    :type steps: int
    :param minimum: intervall start
    :type minimum: float
    :param maximum: intervall stop
    :type maximum: float
    :return: a list of steps value evenly spaced between "minimum" and "maximum" included
    :rtype: List[float]
    """
    intervall = []
    h = (maximum - minimum) / (steps - 1)
    for i in range(steps):
        intervall.append(minimum + h * i)
    return intervall


def compute_solution(
    model: Callable[
        [List[Vector], Callable[[float, Vector], Vector], float, float, int], Vector
    ],
    differential_equation: Callable[[float, Vector], Vector],
    steps: int,
    minimum: float,
    maximum: float,
    initial_conditions: Vector,
    multiple_steps_method: bool = False,
    number_of_steps: int = 1,
    variable_steps: bool = False,
    variation_threshold: int = 0.5
) -> Tuple[List[Vector], List[float]]:
    """
    compute an approximated solution of the given differential equation using the given model between min and max in a specified number of steps

    :param model: function representing the model used to approximate the solution. It needs to take a specified amount of previous steps to calculate the next one
    :type model: Callable[[List[Vector], differential_equation_type, float, float, int], Vector]
    :param differential_equation: represent the differential equation system to approximate. It is a function which represent the f in the equation y' = f(y, t)
    :type differential_equation: Callable[[Vector, float], Vector]
    :param steps: number of steps used to approximate the solution
    :type steps: int
    :param minimum: the value where we start to compute the approximate solution of the differential equation
    :type minimum: float
    :param maximum: the value where we stop to compute the approximate solution of the differential equation
    :type maximum: float
    :param initial_conditions: the initial values of the differential equation system
    :type initial_conditions: Vector
    :param multiple_steps_method: if true, means that the model used is using multiple steps to compute the solution
    :type multiple_steps_method: bool
    :param number_of_steps: if the method is using multiple steps, it is the maximum number of step used by it
    :type number_of_step: int
    :param variable_steps: if true, means that the steps sise adapts to change
    :type variable_steps: bool
    :param variation_threshold: if the step size is variable, it is the minimum tolerated variation between steps without which the step size is unchanged
    :type variation_threshold: int
    :return: a list of "steps" approximated value of the differential equation solution
    :rtype: List[Vector]
    """
    start_time = time.time()
    solution: List[Vector] = [initial_conditions]
    h = (maximum - minimum) / (steps - 1)
    start = minimum + h

    if multiple_steps_method:
        for i in range(1, number_of_steps):
            vector_list = []
            for j in range(i):
                vector_list.append(solution[-i])
            solution.append(
                model(vector_list, differential_equation, minimum + h * i, h, i)
            )
        start = minimum - h * number_of_steps

    intervall_not_full = get_intervall(steps - 1, start, maximum)
    intervall = [minimum] + intervall_not_full

    ti = 0
    while ti < maximum:
        vector_list = []
        for i in range(number_of_steps):
            vector_list.append(solution[-1 - i])
        new_step = model(vector_list, differential_equation, ti, h, number_of_steps)
        solution.append(new_step)
        new_step_pos = new_step[0]
        last_step_pos = solution[-1][0]
        variation_btw_steps = max([(new_step_pos[0] - last_step_pos[i]) / last_step_pos[i] for i in range(len(solution))])
        if variation_btw_steps > 0.5:
            

    comp_time = time.time() - start_time
    print("\n=== Computation Statistics ===")
    print(f"Method: {model.__name__}")
    print(f"Number of points: {len(solution)}")
    print(f"Computation time: {comp_time:.4f} s")
    print("==============================\n")

    return solution, intervall


def compute_solution_trash_points(
    model: Callable[
        [List[Vector], Callable[[float, Vector], Vector], float, float, int], Vector
    ],
    differential_equation: Callable[[float, Vector], Vector],
    steps: int,
    minimum: float,
    maximum: float,
    initial_conditions: Vector,
    multiple_steps_method: bool = False,
    number_of_steps: int = 1,
    ratio: int = 1,
) -> Tuple[List[Vector], List[float]]:
    """
    compute an approximated solution of the given differential equation using the given model between min and max in a specified number of steps. This method also keep a limited amount of position points allowing it to consume less memory. The catch is that you need to set a ration number higher than the number of step used or it wont work

    :param model: function representing the model used to approximate the solution. It needs to take a specified amount of previous steps to calculate the next one
    :type model: Callable[[List[Vector], differential_equation_type, float, float, int], Vector]
    :param differential_equation: represent the differential equation system to approximate. It is a function which represent the f in the equation y' = f(y, t)
    :type differential_equation: Callable[[Vector, float], Vector]
    :param steps: number of steps used to approximate the solution
    :type steps: int
    :param minimum: the value where we start to compute the approximate solution of the differential equation
    :type minimum: float
    :param maximum: the value where we stop to compute the approximate solution of the differential equation
    :type maximum: float
    :param initial_conditions: the initial values of the differential equation system
    :type initial_conditions: Vector
    :param multiple_steps_method: if true, means that the model used is using multiple steps to compute the solution
    :type multiple_steps_method: bool
    :param number_of_steps: if the method is using multiple steps, it is the maximum number of step used by it
    :type number_of_step: int
    :param ratio: number of point keeped during computation. If 1 all points will be keeped, if 2 only one out of 2, ...
    :type ratio: int
    :return: a list of "steps" approximated value of the differential equation solution
    :rtype: List[Vector]
    """
    start_time = time.time()
    solution: List[Vector] = [initial_conditions]
    h = (maximum - minimum) / (steps - 1)
    start = minimum + h

    if multiple_steps_method:
        for i in range(1, number_of_steps):
            vector_list = []
            for j in range(i):
                vector_list.append(solution[-i])
            solution.append(
                model(vector_list, differential_equation, minimum + h * i, h, i)
            )
        start = minimum - h * number_of_steps

    intervall = [minimum] + get_intervall(steps, start, maximum)

    counter = 0
    treesold = ratio
    time_index_deleted = []
    for i in range(1, len(intervall)):
        ti = intervall[i]
        vector_list = []
        for i in range(number_of_steps):
            vector_list.append(solution[-1 - i])
        solution.append(
            model(vector_list, differential_equation, ti, h, number_of_steps)
        )

        if treesold >= 0:
            treesold -= 1
        else:
            if counter >= ratio:
                counter = 1
                for i in range(ratio - 1):
                    solution.pop(len(solution) - 2 * ratio + i)
                    time_index_deleted.append(len(solution) - 2 * ratio + i)

            else:
                counter += 1

    for i in range(len(time_index_deleted) - 1, -1, -1):
        intervall.pop(time_index_deleted[i])

    comp_time = time.time() - start_time
    print("\n=== Computation Statistics ===")
    print(f"Method: {model.__name__}")
    print(f"Number of points: {len(solution)}")
    print(f"Computation time: {comp_time:.4f} s")
    print("==============================\n")

    return solution, intervall


def plot_x_solution(time: List[float], solution: List[Vector]) -> None:
    """
    plot the x coordinates of the computed solution

    :param time: The list of the different time where the soltution was computed
    :type time: List[float]
    :param solution: The computed solution
    :type solution: List[Vector]
    """
    x_coordinate = []
    for vector in solution:
        x_coordinate.append(vector[0][0])
    sb.lineplot(x=time, y=x_coordinate)
    plt.title("x coordinate of the particule trajectory")
    plt.grid(True)
    plt.show()


def plot_y_solution(time: List[float], solution: List[Vector]) -> None:
    """
    plot the y coordinates of the computed solution

    :param time: The list of the different time where the soltution was computed
    :type time: List[float]
    :param solution: The computed solution
    :type solution: List[Vector]
    """
    y_coordinate = []
    for vector in solution:
        y_coordinate.append(vector[0][1])
    sb.lineplot(x=time, y=y_coordinate)
    plt.title("y coordinate of the particule trajectory")
    plt.grid(True)
    plt.show()


def plot_z_solution(time: List[float], solution: List[Vector]) -> None:
    """
    plot the z coordinates of the computed solution

    :param time: The list of the different time where the soltution was computed
    :type time: List[float]
    :param solution: The computed solution
    :type solution: List[Vector]
    """
    z_coordinate = []
    for vector in solution:
        z_coordinate.append(vector[0][2])
    sb.lineplot(x=time, y=z_coordinate)
    plt.title("z coordinate of the particule trajectory")
    plt.grid(True)
    plt.show()


def plot_error(
    approximated_solution: List[Vector], exact_solution: List[Vector], time: List[float]
) -> None:
    """
    plot the error on the position during time of the computed solution compared to an exact (or almost exact) solution

    :param approximated_solution: the computed solution with the choosen model
    :type approximated_solution: List[vector]
    :param exact_solution: the exact solution of the equation
    :type exact_solution: List[Vector]
    """
    error = []
    for i in range(len(exact_solution)):
        error.append(abs(approximated_solution[i][0] - exact_solution[i][0]) ** 2)
    sb.lineplot(x=time, y=error)
    plt.title("Model error during time")
    plt.grid(True)
    plt.show()


def plot_3d(positions: List[Vector], initial_velocity: Vector = None) -> None:
    """
    make a 3D plot of an ordonated liste of position to represent the trajectory of the studied system.
    Includes initial position/velocity information and start/end point markers.

    :param positions: the list of the different position
    :type positions: List[Vector]
    :param initial_velocity: optional initial velocity vector for display
    :type initial_velocity: Vector or None
    """
    figure = go.Figure()

    # Plot the trajectory
    x = []
    y = []
    z = []
    for i in range(len(positions)):
        x.append(positions[i][0])
        y.append(positions[i][1])
        z.append(positions[i][2])
    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="blue", width=1, dash="solid"),
            name="Trajectory",
        )
    )

    # Add starting point (green)
    figure.add_trace(
        go.Scatter3d(
            x=[positions[0][0]],
            y=[positions[0][1]],
            z=[positions[0][2]],
            mode="markers",
            marker=dict(size=10, color="green"),
            name="Start Point",
            showlegend=True,
        )
    )

    # Add ending point (red)
    figure.add_trace(
        go.Scatter3d(
            x=[positions[-1][0]],
            y=[positions[-1][1]],
            z=[positions[-1][2]],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="End Point",
            showlegend=True,
        )
    )

    # Add a sphere at (0, 0, 0) representing earth
    r = 1
    phi = get_intervall(30, 0, 2 * pi)
    theta = get_intervall(15, 0, pi)
    xe = []
    ye = []
    ze = []
    for i in range(len(phi)):
        row_x = []
        row_y = []
        row_z = []
        for j in range(len(theta)):
            row_x.append(r * cos(phi[i]) * sin(theta[j]))
            row_y.append(r * sin(phi[i]) * sin(theta[j]))
            row_z.append(r * cos(theta[j]))
        xe.append(row_x)
        ye.append(row_y)
        ze.append(row_z)
    figure.add_trace(go.Surface(x=xe, y=ye, z=ze, showscale=False, name="Earth"))

    # Add annotations for initial conditions
    initial_pos_text = f"Initial Position:<br>x={positions[0][0]:.4f}<br>y={positions[0][1]:.4f}<br>z={positions[0][2]:.4f}"
    if initial_velocity is not None:
        velocity_text = (
            f"<br>Initial Velocity:<br>"
            f"vx={initial_velocity[0]:.3e}<br>"
            f"vy={initial_velocity[1]:.3e}<br>"
            f"vz={initial_velocity[2]:.3e}"
        )
        initial_pos_text += velocity_text

    figure.add_annotation(
        text=initial_pos_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        bgcolor="rgba(200, 255, 200, 0.8)",
        bordercolor="green",
        borderwidth=2,
        font=dict(size=10),
    )

    # Add annotation for final position
    final_pos_text = f"Final Position:<br>x={positions[-1][0]:.4f}<br>y={positions[-1][1]:.4f}<br>z={positions[-1][2]:.4f}"
    figure.add_annotation(
        text=final_pos_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.52,
        showarrow=False,
        bgcolor="rgba(255, 200, 200, 0.8)",
        bordercolor="red",
        borderwidth=2,
        font=dict(size=10),
    )

    # Set parameters
    figure.update_layout(
        scene=dict(
            xaxis=dict(title="x", showgrid=True, zeroline=True),
            yaxis=dict(title="y", showgrid=True, zeroline=True),
            zaxis=dict(title="z", showgrid=True, zeroline=True),
            aspectmode="data",
        ),
        title="Particle Trajectory in Magnetic Field",
        showlegend=True,
        legend=dict(x=0.7, y=0.9),
    )

    # Show figure
    figure.show()


def plot_kinetic_energy(
    velocity: List[Vector], time_list: List[float], mp: float
) -> None:
    """
    plot the kinetic energy during time using velocity vector
    :param velocity: the list of velocity vectors
    :type velocity: List[Vector]
    :param time_list: the different time associated to the velocity vetors
    :type time_list: List[float]
    :param mp: the mass of the particle
    :type mp: float
    """
    kinetic_energy = []
    for i in range(len(velocity)):
        kinetic_energy.append(1 / 2 * mp * abs(velocity[i]) ** 2)
    sb.lineplot(x=time_list, y=kinetic_energy)
    plt.title("System kinetic energy during time")
    plt.grid(True)
    plt.show()


def plot_2d_projections(positions_list, velocities_list=None, title="Projections 2D"):
    """
    Generates 3 stacked 2D projection plots.

    :param positions_list: List of Vectors [x, y, z]
    :param velocities_list: List of Vectors [vx, vy, vz] (Optional).
                            If provided, displays phase space plots (v vs pos).
                            If absent, displays geometric projections (y vs x).
    """
    # Extracting data
    x = np.array([p.coordinates[0] for p in positions_list])
    y = np.array([p.coordinates[1] for p in positions_list])
    z = np.array([p.coordinates[2] for p in positions_list])

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    fig.suptitle(title, fontsize=16)

    # Color and style
    color = "navy"
    linewidth = 0.1
    point_size = 0.5

    if velocities_list:
        # It's possible let just none for the velocity list but the result will be just 2D position
        # --- Space mode phase (v vs pos) ---
        vx = np.array([v.coordinates[0] for v in velocities_list])
        vy = np.array([v.coordinates[1] for v in velocities_list])
        vz = np.array([v.coordinates[2] for v in velocities_list])

        # Graph 1: vx vs x
        axs[0].plot(x, vx, ".", markersize=point_size, color=color, alpha=0.5)
        axs[0].set_ylabel(r"$v_x$")
        axs[0].set_title(r"Projection Phase Space: $v_x$ vs $x$")
        axs[0].grid(True, alpha=0.3)

        # Graph 2: vy vs y
        axs[1].plot(y, vy, ".", markersize=point_size, color=color, alpha=0.5)
        axs[1].set_ylabel(r"$v_y$")
        axs[1].set_title(r"Projection Phase Space: $v_y$ vs $y$")
        axs[1].grid(True, alpha=0.3)

        # Graph 3: vz vs z
        axs[2].plot(z, vz, ".", markersize=point_size, color=color, alpha=0.5)
        axs[2].set_ylabel(r"$v_z$")
        axs[2].set_xlabel(r"Position ($R_T$)")
        axs[2].set_title(r"Projection Phase Space: $v_z$ vs $z$")
        axs[2].grid(True, alpha=0.3)

    else:
        # --- Projection mode geometrical (pos vs pos) ---

        # Graph 1: y vs x (top view)
        axs[0].plot(x, y, ".", markersize=point_size, color=color, alpha=0.5)
        axs[0].set_ylabel(r"$y$ [$R_T$]")
        axs[0].set_title(r"Projection Plan XY (Top View)")
        axs[0].grid(True, alpha=0.3)
        axs[0].set_aspect("equal")

        # Graph 2: z vs y (side view)
        axs[1].plot(y, z, ".", markersize=point_size, color=color, alpha=0.5)
        axs[1].set_ylabel(r"$z$ [$R_T$]")
        axs[1].set_title(r"Projection Plan YZ (Side View)")
        axs[1].grid(True, alpha=0.3)
        axs[1].set_aspect("equal")

        # Graph3: z vs x (front view)
        axs[2].plot(x, z, ".", markersize=point_size, color=color, alpha=0.5)
        axs[2].set_ylabel(r"$z$ [$R_T$]")
        axs[2].set_xlabel(r"$x$ [$R_T$]")
        axs[2].set_title(r"Projection Plan XZ (Front View)")
        axs[2].grid(True, alpha=0.3)
        axs[2].set_aspect("equal")

    plt.tight_layout()
    plt.show()
