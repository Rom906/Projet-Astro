from utils import Vector
from constants import RT
from scientific_notation import ScientificNotation
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
    initial_conditions: Vector,
    max_n_steps: int,
    initial_step_size: int,
    multiple_steps_method: bool = False,
    model_n_steps: int = 1,
    model_order: int = 4,
    ratio: int = 1,
    variable_steps: bool = False,
    tolerated_variation: int = 0.05
) -> Tuple[List[Vector], List[float]]:
    """
    compute an approximated solution of the given differential equation using the given model between min and max in a specified number of steps. This method also keep a limited amount of position points allowing it to consume less memory. The catch is that you need to set a ration number higher than the number of step used or it wont work

    :param model: function representing the model used to approximate the solution. It needs to take a specified amount of previous steps to calculate the next one
    :type model: Callable[[List[Vector], differential_equation_type, float, float, int], Vector]
    :param differential_equation: represent the differential equation system to approximate. It is a function which represent the f in the equation y' = f(y, t)
    :type differential_equation: Callable[[Vector, float], Vector]
    :param initial_conditions: the initial values of the differential equation system
    :type initial_conditions: Vector
    :param max_n_steps: number of steps used to approximate the solution
    :type max_n_steps: int
    :param initial_step_size: initial guess for appropriate step nice, not modified if non-variable steps
    :type initial_step_size: int/float
    :param multiple_steps_method: if true, means that the model used is using multiple steps to compute the solution
    :type multiple_steps_method: bool
    :param model_n_steps: if the method is using multiple steps, it is the maximum number of step used by it
    :type model_n_steps: int
    :param model_order: convergence order of the model
    :type model_order: int
    :param ratio: number of point keeped during computation. If 1 all points will be keeped, if 2 only one out of 2, ...
    :type ratio: int
    :param variable_steps: if true, means that the steps sise adapts to change
    :type variable_steps: bool
    :param tolerated_variation: if the step size is variable, it is the maximum tolerated variation between steps without which the step size is unchanged
    :type tolerated_variation: float
    :return: a list of "steps" approximated value of the differential equation solution
    :rtype: List[Vector]
    """
    start_time = time.time()
    solution: List[Vector] = [initial_conditions]
    time_index = [0]
    h = initial_step_size

    if multiple_steps_method:
        for i in range(1, model_n_steps):
            vector_list = []
            for j in range(i):
                vector_list.append(solution[-i])
            solution.append(
                model(vector_list, differential_equation, minimum + h * i, h, i)
            )
            time_index.append(minimum + h * i)

    counter = 0
    treshold = ratio
    time_index_deleted = []
    n_steps = 0
    ti = 0
    while n_steps < max_n_steps - 1:
        print(n_steps, h)
        vector_list = []
        for i in range(model_n_steps):
            vector_list.append(solution[-1 - i])
        new_step_large = model(
            vector_list, differential_equation, ti, h, model_n_steps
        )
        new_step_pos_large = new_step_large[0]
        if variable_steps:
            half_step_fine = model(
                vector_list, differential_equation, ti, h / 2, model_n_steps
            )
            new_step_fine = model(
                [half_step_fine],
                differential_equation,
                ti + h / 2,
                h / 2,
                model_n_steps,
            )
            new_step_pos_fine = new_step_fine[0]
            max_variation = 0
            for i in range(len(new_step_pos_large.coordinates)):
                variation = abs(new_step_pos_large[i] - new_step_pos_fine[i])
                if variation > max_variation:
                    max_variation = variation
            if max_variation < tolerated_variation:
                solution.append(new_step_large)
                n_steps += 1
                ti += h
                time_index.append(ti)
            if max_variation != 0:
                h *= 0.9 * (tolerated_variation / max_variation) ** (1 / (model_order + 1))
            if max_variation >= tolerated_variation:
                new_step = model(
                    vector_list, differential_equation, ti, h, model_n_steps
                )
                solution.append(new_step)
                n_steps += 1
                ti += h
                time_index.append(ti)
        else:
            solution.append(new_step_large)
            n_steps += 1
            ti += h
            time_index.append(ti)

        if treshold >= 0:
            treshold -= 1
        else:
            if counter >= ratio:
                counter = 1
                for i in range(ratio - 1):
                    solution.pop(len(solution) - 2 * ratio + i)
                    time_index.pop(len(solution) - 2 * ratio + i)
            else:
                counter += 1

    comp_time = time.time() - start_time
    print("\n=== Computation Statistics ===")
    print(f"Method: {model.__name__}")
    print(f"Number of points: {len(solution)}")
    print(f"Computation time: {comp_time:.4f} s")
    print("==============================\n")

    return solution, time_index

def compute_solution_no_trash_points(
    model: Callable[
        [List[Vector], Callable[[float, Vector], Vector], float, float, int],
        Vector,
    ],
    differential_equation: Callable[[float, Vector], Vector],
    steps: int,
    minimum: float,
    maximum: float,
    initial_conditions: Vector,
    multiple_steps_method: bool = False,
    number_of_steps: int = 1,
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

    for ti in intervall:
        vector_list = []
        for i in range(number_of_steps):
            vector_list.append(solution[-1 - i])
        solution.append(
            model(vector_list, differential_equation, ti, h, number_of_steps)
        )

    comp_time = time.time() - start_time
    print("\n=== Computation Statistics ===")
    print(f"Method: {model.__name__}")
    print(f"Number of points: {len(solution)}")
    print(f"Computation time: {comp_time:.4f} s")
    print("==============================\n")

    return solution, intervall


def compute_solution_trash_points(
    model: Callable[
        [List[Vector], Callable[[float, Vector], Vector], float, float, int],
        Vector,
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
    approximated_solution: List[Vector],
    exact_solution: List[Vector],
    time: List[float],
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
    figure.add_trace(go.Surface(x=xe, y=ye, z=ze, showscale=False, name="Earth", colorscale=[[0, 'blue'], [1, 'blue']]))
    position_x = round(positions[0][0], 3)
    position_y = round(positions[0][1], 3)
    position_z = round(positions[0][2], 3)
    # Add annotations for initial conditions
    initial_pos_text = f"Initial Position (en RT):<br>x={position_x:.3f}<br>y={position_y:.3f}<br>z={position_z:.3f}"
    if initial_velocity is not None:
        velocity_text = (
            f"<br>Initial Velocity:<br>"
            f"vx={ScientificNotation(initial_velocity[0], 'm').to_scientific_notation()}<br>"
            f"vy={ScientificNotation(initial_velocity[1], 'm').to_scientific_notation()}<br>"
            f"vz={ScientificNotation(initial_velocity[2], 'm').to_scientific_notation()}"
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
    final_pos_text = f"Final Position:<br>x={ScientificNotation(RT * positions[-1][0], 'm').to_scientific_notation()}<br>y={ScientificNotation(RT * positions[-1][1], 'm').to_scientific_notation()}<br>z={ScientificNotation(RT * positions[-1][2], 'm').to_scientific_notation()}"
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
    figure.update_traces(showlegend=True)
    figure.update_layout(
        scene=dict(
            xaxis=dict(title="x/RT", showgrid=True, zeroline=True),
            yaxis=dict(title="y/RT", showgrid=True, zeroline=True),
            zaxis=dict(title="z/RT", showgrid=True, zeroline=True),
            aspectmode="data",
        ),
        title="Particle Trajectory in Magnetic Field",
        showlegend=True,
        legend=dict(x=0.7, y=0.9),
    )

    # Show figure
    figure.show()


def plot_3d_v2(
    positions: List[Vector],
    initial_velocity: Vector = None,
    magnetic_moment: Vector = None,
) -> None:
    """
    Enhanced 3D plot of particle trajectory in magnetic field with magnetic dipole moment vector.
    Includes initial position/velocity information, start/end point markers, and magnetic moment visualization.

    :param positions: the list of different positions
    :type positions: List[Vector]
    :param initial_velocity: optional initial velocity vector for display
    :type initial_velocity: Vector or None
    :param magnetic_moment: optional magnetic moment vector (default: mu from constants)
    :type magnetic_moment: Vector or None
    """
    from constants import mu

    if magnetic_moment is None:
        magnetic_moment = mu

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

    # Add a sphere at (0, 0, 0) representing Earth with realistic continents
    r = 1
    phi = get_intervall(50, 0, 2 * pi)
    theta = get_intervall(25, 0, pi)
    xe = []
    ye = []
    ze = []
    sc = []  # surfacecolor (0..1) to map to terrain types

    for i in range(len(phi)):
        row_x = []
        row_y = []
        row_z = []
        row_sc = []
        for j in range(len(theta)):
            # Convert spherical coords to Cartesian
            lon = phi[i]  # longitude [0, 2π]
            lat = theta[j]  # latitude [0, π], where π/2 is equator

            px = r * cos(lon) * sin(lat)
            py = r * sin(lon) * sin(lat)
            pz = r * cos(lat)

            row_x.append(px)
            row_y.append(py)
            row_z.append(pz)

            # --- Realistic terrain generation with harmonious continents ---
            # Convert latitude to degrees (-90 to +90)
            lat_deg = (lat - pi / 2) * 180 / pi  # Range: [-90°, 90°]
            # Convert longitude to degrees (0 to 360)
            lon_deg = lon * 180 / pi

            # Start with OCEAN (default)
            terrain = 0.15

            # --- Simplified Perlin-like noise for natural continents ---
            # Create a base continental noise pattern
            base_noise = (
                0.4 * sin(lon_deg * pi / 180) * cos(lat_deg * pi / 180)
                + 0.3 * sin(2 * lon_deg * pi / 180)
                + 0.2 * cos(3 * lat_deg * pi / 180)
                + 0.1 * sin(5 * lon_deg * pi / 180)
            )

            # Reduce ocean coverage at certain latitudes (where continents exist)
            if -60 <= lat_deg <= 75:
                # Apply continent pattern more strongly
                continent_presence = base_noise + 0.15 * sin(lat_deg * pi / 180)

                if continent_presence > -0.1:
                    # Progressively map noise to terrain values
                    terrain = 0.50 + continent_presence * 0.25

            # --- Make polar regions slightly higher terrain (greenland-like) ---
            if lat_deg > 75:
                if base_noise > -0.2:
                    terrain = 0.70  # Greenish at far north
            elif lat_deg < -75:
                if base_noise > -0.2:
                    terrain = 0.70  # Greenish at far south

            # Clamp to [0, 1]
            terrain = max(0.15, min(1.0, terrain))
            row_sc.append(terrain)

        xe.append(row_x)
        ye.append(row_y)
        ze.append(row_z)
<<<<<<< HEAD
    figure.add_trace(go.Surface(x=xe, y=ye, z=ze, showscale=False, name="Earth", colorscale=[[0, 'blue'], [1, 'blue']]))
    
=======
        sc.append(row_sc)

    # Enhanced colorscale - Blue and Green only (no white)
    earth_colorscale = [
        [0.00, "rgb(0, 30, 100)"],  # Deep ocean
        [0.25, "rgb(20, 100, 180)"],  # Ocean blue
        [0.50, "rgb(60, 150, 50)"],  # Green forest
        [0.75, "rgb(80, 180, 80)"],  # Light green
        [1.00, "rgb(100, 200, 100)"],  # Bright green
    ]

    figure.add_trace(
        go.Surface(
            x=xe,
            y=ye,
            z=ze,
            surfacecolor=sc,
            colorscale=earth_colorscale,
            cmin=0,
            cmax=1,
            showscale=False,
            name="Earth",
            lighting=dict(ambient=0.6, diffuse=0.6, roughness=0.9, specular=0.1),
            lightposition=dict(x=100, y=200, z=0),
        )
    )

>>>>>>> 5448aeabca727522f26132efa026e2cfb0582342
    # Add magnetic moment vector at North Pole (0, 0, 1)
    # Normalize and scale the magnetic moment for visualization
    mu_normalized = magnetic_moment.normalized()
    mu_magnitude = abs(magnetic_moment)
    # Scale factor for visualization (proportional to magnitude but visible on plot)
    scale_factor = 2.0
    mu_scaled = mu_normalized * scale_factor

    north_pole = [0, 0, 1]
    mu_end = [north_pole[i] + mu_scaled[i] for i in range(3)]

    figure.add_trace(
        go.Scatter3d(
            x=[north_pole[0], mu_end[0]],
            y=[north_pole[1], mu_end[1]],
            z=[north_pole[2], mu_end[2]],
            mode="lines",
            line=dict(color="purple", width=4),
            marker=dict(size=8, color="purple"),
            name="Magnetic Moment",
            showlegend=True,
        )
    )

    position_x = round(positions[0][0], 3)
    position_y = round(positions[0][1], 3)
    position_z = round(positions[0][2], 3)

    # Add annotations for initial conditions
    initial_pos_text = f"Initial Position (en RT):<br>x={position_x:.3f}<br>y={position_y:.3f}<br>z={position_z:.3f}"
    if initial_velocity is not None:
        velocity_text = (
            f"<br>Initial Velocity:<br>"
            f"vx={ScientificNotation(initial_velocity[0], 'm.s^-1').to_scientific_notation()}<br>"
            f"vy={ScientificNotation(initial_velocity[1], 'm.s^-1').to_scientific_notation()}<br>"
            f"vz={ScientificNotation(initial_velocity[2], 'm.s^-1').to_scientific_notation()}"
        )
        initial_pos_text += velocity_text

    # Add magnetic moment information
    mu_magnitude_formatted = ScientificNotation(
        mu_magnitude, "A·m²"
    ).to_scientific_notation()
    magnetic_moment_text = f"<br><br>Magnetic Moment: {mu_magnitude_formatted}"
    initial_pos_text += magnetic_moment_text

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
    final_pos_text = f"Final Position:<br>x={ScientificNotation(RT * positions[-1][0], 'm').to_scientific_notation()}<br>y={ScientificNotation(RT * positions[-1][1], 'm').to_scientific_notation()}<br>z={ScientificNotation(RT * positions[-1][2], 'm').to_scientific_notation()}"
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
    figure.update_traces(showlegend=True)
    figure.update_layout(
        scene=dict(
            xaxis=dict(title="x/RT", showgrid=True, zeroline=True),
            yaxis=dict(title="y/RT", showgrid=True, zeroline=True),
            zaxis=dict(title="z/RT", showgrid=True, zeroline=True),
            aspectmode="data",
        ),
        title="Particle Trajectory in Magnetic Field with Dipole Moment",
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


import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_kinetic_energy_v2(
    velocity: List["Vector"], time_list: List[float], mp: float
) -> None:

    # --- Compute Ke/m
    ke = np.array([0.5 * (abs(v) ** 2) for v in velocity])

    # --- Scale from first value
    ke0 = ke[0]
    exponent = int(np.floor(np.log10(abs(ke0))))
    scale = 10**exponent

    ke_scaled = ke / scale

    # --- Reliable zone
    critical_index = len(ke) - 1
    for i, val in enumerate(ke):
        if abs(val - ke0) / ke0 > 0.1:
            critical_index = i - 1
            break

    critical_time = time_list[critical_index]

    # --- Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axvspan(
        time_list[0],
        critical_time,
        alpha=0.3,
        color="lightgreen",
        label="Reliable zone (<10%)",
    )

    ax.plot(time_list, ke_scaled, linewidth=2, label="Ke/m (J/kg)")

    # Labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Ke/m (×10^{exponent} J/kg)")
    ax.set_title(f"Kinetic Energy per mass (m = {mp} (kg))")

    # force global scale
    ax.set_ylim(0, np.max(ke_scaled) * 1.1)

    # Remove offset
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    # Legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()


def plot_kinetic_energy_multiple(velocity_list, time_list, mp):

    fig, ax = plt.subplots(figsize=(12, 6))

    all_ke_final = []

    first_velocity = velocity_list[0]
    first_ke = np.array([0.5 * (abs(v) ** 2) for v in first_velocity])
    first_ke0 = first_ke[0] if len(first_ke) > 0 else 1
    exponent = int(np.floor(np.log10(abs(first_ke0)))) if first_ke0 != 0 else 0

    for sol_idx, velocity in enumerate(velocity_list):
        # --- Compute Ke/m
        ke = np.array([0.5 * (abs(v) ** 2) for v in velocity])

        # --- Reliable zone (ΔE/E0 < 0.1)
        ke0 = ke[0] if len(ke) > 0 else 1
        critical_index = len(ke) - 1
        if ke0 != 0:
            for j, val in enumerate(ke):
                if abs(val - ke0) / ke0 > 0.1:
                    critical_index = j - 1
                    break

        ke_final = ke[: critical_index + 1]
        time_final = (
            time_list[sol_idx][: critical_index + 1] if sol_idx < len(time_list) else []
        )
        valid_mask = ke_final > 0
        ke_valid = ke_final[valid_mask]
        time_valid = np.array(time_final)[valid_mask] if len(time_final) > 0 else []

        if len(ke_valid) > 0:
            all_ke_final.extend(ke_valid)
            ax.plot(
                time_valid,
                ke_valid,
                linewidth=2,
                marker="o",
                markersize=3,
                label=f"Solution {sol_idx+1}",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Ke/m (×10^{exponent} J/kg)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(f"Kinetic Energy per mass (m = {mp} kg)")

    ax.grid(True, alpha=0.3, which="both", linestyle="-", linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.8, 1])
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


def saved_plot_kinetic_energy(
    velocity: List[Vector], time_list: List[float], mp: float, save_name: str
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
    plt.savefig(save_name)


def saved_plot_2d_projections(
    positions_list,
    save_name: str,
    velocities_list=None,
    title="Projections 2D",
):
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
    plt.savefig(save_name)


def plot_3d_multi(
    positions_list: List[List[Vector]], magnetic_moment: Vector = None
) -> None:
    """
    3D plot of multiple particle trajectories in magnetic field with magnetic dipole moment vector.
    Trajectories are colored with a plasma gradient to distinguish between particles.
    Includes Earth sphere and magnetic moment visualization.

    :param positions_list: list of trajectory lists, where each trajectory is a list of Vectors [x, y, z]
    :type positions_list: List[List[Vector]]
    :param magnetic_moment: optional magnetic moment vector (default: mu from constants)
    :type magnetic_moment: Vector or None
    """
    from constants import mu
    import matplotlib.cm as cm

    if magnetic_moment is None:
        magnetic_moment = mu

    figure = go.Figure()

    # Generate plasma colormap for particles
    num_particles = len(positions_list)
    plasma_colors = cm.get_cmap("plasma")

    # Plot each trajectory
    for particle_idx, positions in enumerate(positions_list):
        # Compute color for this particle on the plasma gradient
        color_val = (
            particle_idx / max(1, num_particles - 1) if num_particles > 1 else 0.5
        )
        rgba = plasma_colors(color_val)
        # Convert RGBA to hex color for plotly
        color_hex = (
            f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.8)"
        )

        # Extract coordinates
        x = []
        y = []
        z = []
        for pos in positions:
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])

        # Plot trajectory with reduced line width
        figure.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color=color_hex, width=0.5, dash="solid"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add starting point (smaller)
        figure.add_trace(
            go.Scatter3d(
                x=[positions[0][0]],
                y=[positions[0][1]],
                z=[positions[0][2]],
                mode="markers",
                marker=dict(size=3, color=color_hex),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add ending point (smaller)
        figure.add_trace(
            go.Scatter3d(
                x=[positions[-1][0]],
                y=[positions[-1][1]],
                z=[positions[-1][2]],
                mode="markers",
                marker=dict(size=3, color=color_hex),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add Earth sphere
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

    # Add magnetic moment vector at North Pole (0, 0, 1)
    mu_normalized = magnetic_moment.normalized()
    scale_factor = 2.0
    mu_scaled = mu_normalized * scale_factor

    north_pole = [0, 0, 1]
    mu_end = [north_pole[i] + mu_scaled[i] for i in range(3)]

    figure.add_trace(
        go.Scatter3d(
            x=[north_pole[0], mu_end[0]],
            y=[north_pole[1], mu_end[1]],
            z=[north_pole[2], mu_end[2]],
            mode="lines+markers",
            line=dict(color="purple", width=4),
            marker=dict(size=8, color="purple"),
            name="Magnetic Moment",
            showlegend=True,
        )
    )

    # Set parameters
    figure.update_layout(
        scene=dict(
            xaxis=dict(title="x/RT", showgrid=True, zeroline=True),
            yaxis=dict(title="y/RT", showgrid=True, zeroline=True),
            zaxis=dict(title="z/RT", showgrid=True, zeroline=True),
            aspectmode="data",
        ),
        title="Multiple Particle Trajectories in Magnetic Field",
        showlegend=True,
        legend=dict(x=0.7, y=0.9),
    )

    # Show figure
    figure.show()
