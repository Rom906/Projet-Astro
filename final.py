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

def compute_solution_trash_points_by_steps(initial_conditions: Vector) -> Tuple[Vector, float, int]:
    start_time = time.time()
    params = NormalizationParameters(RT, qe / mp, MO, abs(mu))
    tolerated_variation = 0.05
    model_n_steps = 1
    model_order = 4
    max_n_steps = 10000
    initial_step_size = 0.01
    h = initial_step_size
    pos_vel = initial_conditions
    n_steps = 0
    ti = 0
    collisions = False
    E = convert_electric_field_to_normalized(Vector([0, 0, 0]), params)
    # while not collisions:
    #     E = Vector([0, 0, 0])
    #     for i in range(other_particles):
    #         E += field_contribution(-qe, pos_vel[0], other_particles[i][-1][0])
    for i in range(len(initial_conditions)):
        pos_vel = pos_vels[0]
        new_step_large = model(
        [pos_vel], differential_equation, ti, h, model_n_steps, E=E
        )
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
        initial_conditions[i][1] = pos_vel


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
        initial_conditions = Vector([initial_condition[0], initial_condition[1]])

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
