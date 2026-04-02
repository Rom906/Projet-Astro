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


def saved_plot_2d_projections_color(
    positions_list: List[Vector], 
    save_name: str, 
    velocities_list: List[Vector] = None, 
    title: str = "2D Projections", 
    coordinate_system: str = "cartesian", 
    save: bool = True, 
    time_list: List[float] = None
) -> None:
    """
    Generates and saves (or displays) a figure containing three 2D projection plots. 
    Depending on the mode, it visualizes either Phase Space trajectories (Velocity vs Position) 
    or Geometric Projections (Position vs Position). 
    
    In Phase Space mode, trajectories are rendered as continuous lines with a color gradient 
    representing the evolution of time, using Matplotlib LineCollections for performance. 
    In Geometric mode, trajectories are rendered as scatter plots.

    :param positions_list: List of position vectors (Vector objects) representing the trajectory coordinates.
    :type positions_list: List[Vector]
    :param save_name: Base name of the file to save the figure (without extension). Ignored if save=False.
    :type save_name: str
    :param velocities_list: List of velocity vectors (Vector objects). If provided, enables Phase Space mode.
    :type velocities_list: List[Vector], optional
    :param title: Main title displayed at the top of the figure.
    :type title: str, optional
    :param coordinate_system: Coordinate system to use for projection. Options: "cartesian", "spherical", or "intrinsic".
    :type coordinate_system: str, optional
    :param save: If True, saves the figure to disk as '{save_name}.png'. If False, displays the figure interactively.
    :type save: bool, optional
    :param time_list: List of time values corresponding to each position/velocity point. Required for color gradient in Phase Space mode and mandatory for "intrinsic" mode.
    :type time_list: List[float], optional
    :raises ValueError: If "intrinsic" mode is selected but time_list or velocities_list are missing.
    """
    
    # --- Handle Intrinsic Mode (not pertinent now and not realy finished) ---
    
    if coordinate_system == "intrinsic":
        s, vs = None, None
        if time_list is None:
            raise ValueError("Intrinsic mode requires the 'time_list' parameter (list of dt).")
        if velocities_list is None:
            raise ValueError("Intrinsic mode requires 'velocities_list' to calculate velocity magnitude.")
        
        s, vs = calculate_curvilinear_coordinates(positions_list, velocities_list, time_list)
        
        # Replace x, y, z with s for the 3 plots.
        x, y, z = s, s, s
        vx, vy, vz = vs, vs, vs
        
        labels_pos = [r"$s$ [RT]", r"$s$ [RT]", r"$s$ [RT]"]
        labels_vel = [r"$v_s$ [RT]", r"$v_s$ [RT]", r"$v_s$ [RT]"]
        titles = [r"Intrinsic Phase Space: $v_s$ vs $s$", r"(Identical View)", r"(Identical View)"]
        
    # --- Handle Spherical Mode --- (Most pertinent)
    elif coordinate_system == "spherical":
        # Extract base Cartesian data
        x = np.array([p.coordinates[0] for p in positions_list])
        y = np.array([p.coordinates[1] for p in positions_list])
        z = np.array([p.coordinates[2] for p in positions_list])

        if coordinate_system == "spherical":
            # --- Conversion from Cartesian to Spherical --- (an amelioration is to use a function)
            orig_x = x
            orig_y = y
            orig_z = z
            
            rho = np.sqrt(orig_x**2 + orig_y**2)
            r = np.sqrt(orig_x**2 + orig_y**2 + orig_z**2)
            
            # Avoid division by zero
            r = np.where(r == 0, 1e-9, r)
            rho = np.where(rho == 0, 1e-9, rho)

            # Position Coordinates (r, theta, phi)
            x = r
            y = np.arccos(orig_z / r)           # theta (Colatitude)
            z = np.arctan2(orig_y, orig_x)      # phi (Longitude)

            if velocities_list:
                vx = np.array([v.coordinates[0] for v in velocities_list])
                vy = np.array([v.coordinates[1] for v in velocities_list])
                vz = np.array([v.coordinates[2] for v in velocities_list])

                # Radial Speed (vr)
                vr = (orig_x * vx + orig_y * vy + orig_z * vz) / r

                # Colatitudinal Speed (v_theta)
                v_theta = (orig_z * (orig_x * vx + orig_y * vy) / rho - rho * vz) / r
                
                # Azimuthal Speed (v_phi)
                v_phi = (orig_x * vy - orig_y * vx) / rho

                vx, vy, vz = vr, v_theta, v_phi

                labels_pos = [r"$r$ [$R_T$]", r"$\theta$ [rad]", r"$\phi$ [rad]"]
                labels_vel = [r"$v_r$ [$R_T$]", r"$v_\theta$ [$R_T$]", r"$v_\phi$ [$R_T$]"]
                titles = [r"Phase Space: $v_r$ vs $r$", r"Phase Space: $v_\theta$ vs $\theta$", r"Phase Space: $v_\phi$ vs $\phi$"]
            else:
                labels_pos = [r"$r$ [$R_T$]", r"$\theta$ [rad]", r"$\phi$ [rad]"]
                titles = [r"Projection: $\theta$ vs $r$", r"Projection: $\phi$ vs $\theta$", r"Projection: $\phi$ vs $r$"]

        # --- Handle Cartesian Mode --- (can be useful)
        else: # cartesian
            if velocities_list:
                vx = np.array([v.coordinates[0] for v in velocities_list])
                vy = np.array([v.coordinates[1] for v in velocities_list])
                vz = np.array([v.coordinates[2] for v in velocities_list])

            labels_pos = [r"$x$ [$R_T$]", r"$y$ [$R_T$]", r"$z$ [$R_T$]"]
            labels_vel = [r"$v_x$", r"$v_y$", r"$v_z$"]
            titles = [r"Phase Space Projection: $v_x$ vs $x$", r"Phase Space Projection: $v_y$ vs $y$", r"Phase Space Projection: $v_z$ vs $z$"]

    # --- Plot Configuration ---
    # To add time in the graph we want a series of segments (posn,veln) -> (posn+1, veln+1) and attribute one color to each segment. We will plot a series of segments
    
    # Normalise time_list and set color tab
    time_list=np.array(time_list)
    norm = Normalize(vmin=time_list.min(), vmax=time_list.max())
    cmap = plt.cm.viridis

    # Create array of couples (pos,vel) for all tree graphs
    points_graph1 = np.array([x, vx]).T.reshape(-1, 1, 2)
    points_graph2 = np.array([y, vy]).T.reshape(-1, 1, 2)
    points_graph3 = np.array([z, vz]).T.reshape(-1, 1, 2)
    # Create the segments
    segments1 = np.concatenate([points_graph1[:-1], points_graph1[1:]], axis=1)
    segments2 = np.concatenate([points_graph2[:-1], points_graph2[1:]], axis=1)
    segments3 = np.concatenate([points_graph3[:-1], points_graph3[1:]], axis=1)
    

    # -- Creation of Collections --- (plt.plot can't change colors so we use collections)
    lc1 = LineCollection(segments1, cmap=cmap, norm=norm)
    lc1.set_array(time_list[:-1]) # Set color to the segments
    lc2 = LineCollection(segments2, cmap=cmap, norm=norm)
    lc2.set_array(time_list[:-1])
    lc3 = LineCollection(segments3, cmap=cmap, norm=norm)
    lc3.set_array(time_list[:-1])

    # --- Define plot and axes ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    # --- Point color and size definition --- (for the last part, don't know if it's pertinent to keep)
    color = "navy"
    point_size = 0.5

    # --- Plot graphs vel in fonction of position --- (With color to show the time)
    if velocities_list or coordinate_system == "intrinsic":
        # --- Phase Space Mode (v vs pos) ---

        # Graph 1
        axs[0].add_collection(lc1)
        axs[0].set_ylabel(labels_vel[0])
        axs[0].set_title(titles[0])
        axs[0].grid(True, alpha=0.3)
        axs[0].set_xlim(x.min(), x.max())
        axs[0].set_ylim(vx.min(), vx.max())

        # Graph 2
        axs[1].add_collection(lc2)
        axs[1].set_ylabel(labels_vel[1])
        axs[1].set_title(titles[1])
        axs[1].grid(True, alpha=0.3)
        axs[1].set_xlim(y.min(), y.max())
        axs[1].set_ylim(vy.min(), vy.max())

        # Graph 3
        axs[2].add_collection(lc3)
        axs[2].set_ylabel(labels_vel[2])
        axs[2].set_xlabel(labels_pos[2]) # Display position unit on X axis
        axs[2].set_title(titles[2])
        axs[2].grid(True, alpha=0.3)
        axs[2].set_xlim(z.min(), z.max())
        axs[2].set_ylim(vz.min(), vz.max())

    else:
        # --- Geometric Projection Mode (pos vs pos) ---
        
        # Graph 1: y vs x
        axs[0].plot(x, y, ".", markersize=point_size, color=color, alpha=0.5)
        axs[0].set_xlabel(labels_pos[0])
        axs[0].set_ylabel(labels_pos[1])
        axs[0].set_title(titles[0])
        axs[0].grid(True, alpha=0.3)
        axs[0].set_aspect("equal")

        # Graph 2: z vs y
        axs[1].plot(y, z, ".", markersize=point_size, color=color, alpha=0.5)
        axs[1].set_xlabel(labels_pos[1])
        axs[1].set_ylabel(labels_pos[2])
        axs[1].set_title(titles[1])
        axs[1].grid(True, alpha=0.3)
        axs[1].set_aspect("equal")

        # Graph 3: z vs x
        axs[2].plot(x, z, ".", markersize=point_size, color=color, alpha=0.5)
        axs[2].set_xlabel(labels_pos[0])
        axs[2].set_ylabel(labels_pos[2])
        axs[2].set_title(r"Projection XZ Plane (Front View)")
        axs[2].grid(True, alpha=0.3)
        axs[2].set_aspect("equal")

    # --- Add color bar ---
    cbar = fig.colorbar(lc1, ax=axs.tolist(), shrink=0.95) 
    cbar.set_label('Temps (s)')

    # --- Save as a file or show --- (to change the save location don't hesitate, add ../ or any type of redirection)
    if save:
        plt.savefig(f"{save_name}.png")
    else:
        plt.show()