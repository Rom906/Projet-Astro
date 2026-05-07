from generate_solutions import plot_3d_collisions_only
from normalization import NormalizationParameters, differential_equation_normalized, convert_to_normalized, convert_to_dimensional, convert_electric_field_to_normalized
from integration_functions import RK4
from utils import Vector
from constants import RT, mp, MO, qe, mu
from typing import List, Callable, Tuple
import time
from monte_carlo import molecules_list, test_collision, field_contribution
from atmospheric_model import O, O2, H, HE, AR, N2
from random import randrange
from multiprocessing import Pool

NA = 6

def compute_solution_trash_points_by_steps(
    model: Callable[
        [List[Vector], Callable[[float, Vector], Vector], float, float, int], Vector
    ],
    differential_equation: Callable[[float, Vector], Vector],
    initial_conditions: Vector,
    max_n_steps: int,
    initial_step_size: int,
    params: NormalizationParameters,
    model_n_steps: int = 1,
    model_order: int = 4,
    tolerated_variation: float = 0.05
) -> Tuple[Vector, float, int]:
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
    h = initial_step_size
    pos_vel = initial_conditions
    n_steps = 0
    ti = 0
    collisions = False
    E = convert_electric_field_to_normalized(Vector([0, 0, 0]), params)
    while not collisions:
        if n_steps % 100 == 0:
            print(f"Step : {n_steps} / {max_n_steps}")
        # E = 0
        # for i in range() in range(other_particles):
        #     E += field_contribution(-qe, pos_vel[0], other_particles[i][-1][0])
        new_step_large = model(
        [pos_vel], differential_equation, ti, h, model_n_steps, E=E
        )
        new_step_pos_large = new_step_large[0]

        half_step_fine = model(
            [pos_vel], differential_equation, ti, h / 2, model_n_steps, E=E
        )
        new_step_fine = model(
            [half_step_fine],
            differential_equation,
            ti + h / 2,
            h / 2,
            model_n_steps,
            E=E
        )
        new_step_pos_fine = new_step_fine[0]
        max_variation = 0
        for i in range(len(new_step_pos_large.coordinates)):
            variation = abs(new_step_pos_large[i] - new_step_pos_fine[i])
            if variation > max_variation:
                max_variation = variation
        if max_variation < tolerated_variation:
            pos_vel = new_step_large
            n_steps += 1
            ti += h
        if max_variation != 0:
            h *= 0.9 * (tolerated_variation / max_variation) ** (1 / (model_order + 1))
        if max_variation >= tolerated_variation:
            new_step = model(
                [pos_vel], differential_equation, ti, h, model_n_steps, E=E
            )
            pos_vel = new_step
            n_steps += 1
            ti += h
        for i in range(len(molecules_list)):
            denormalized_posvel = convert_to_dimensional(pos_vel[0], pos_vel[1], params)
            conditions = Vector([denormalized_posvel[0], denormalized_posvel[1]])
            
            if test_collision(conditions, i):
                collisions = True
                molecule = i
        if n_steps > max_n_steps:
            molecule = NA
            break
        if abs(pos_vel[0]) > 50:
            molecule = NA
            break

    comp_time = time.time() - start_time
    print("\n=== Computation Statistics ===")
    print(f"Method: {model.__name__}")
    print(f"Number of points: {n_steps}")
    print(f"Computation time: {comp_time:.4f} s")
    print("==============================\n")

    return pos_vel, ti, molecule


if __name__ == "__main__":
    parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
    initial_conditions = []
    for i in range(32):
        x_start = 10
        y_start = randrange(-5, 6)
        z_start = randrange(-5, 6)
        vx_start = -randrange(1, 11) / 100
        vy_start = randrange(-1, 1) / 100
        vz_start = randrange(-1, 1) / 100
        initial_position = RT * Vector([x_start, y_start, z_start])
        initial_velocity = RT * Vector([vx_start, vy_start, vz_start])
        initial_condition = convert_to_normalized(initial_position, initial_velocity, parameters)
        initial_condition = Vector([initial_condition[0], initial_condition[1]])
        initial_conditions.append((RK4, differential_equation_normalized, initial_condition, 100000, 1, parameters))

    with Pool() as p:
        results = p.starmap(compute_solution_trash_points_by_steps, initial_conditions)

    collisions_positions = []
    times = []
    molecule_collisions_index = []
    for result in results:
        collisions_positions.append(result[0][0])
        times.append(result[1])
        molecule_collisions_index.append(result[2])
    plot_3d_collisions_only(collisions_positions, molecule_collisions_index)
