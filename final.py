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

def compute_solution(initial_conditions: Vector) -> Tuple[Vector, float, int]:
    start_time = time.time()
    params = NormalizationParameters(RT, qe / mp, MO, abs(mu))
    tolerated_variation = 0.05
    max_n_steps = 10000
    h = 0.01
    n_steps = 0
    ti = 0
    for i in range(len(initial_conditions)):
        E = Vector([0, 0, 0])
        pos_vel = initial_conditions[i][0]
        for j in range(len(initial_conditions)):
            if pos_vel[0] != initial_conditions[j][0][0]:
                E += field_contribution(-qe, pos_vel[0], initial_conditions[j][0][0])
        E = convert_electric_field_to_normalized(Vector([0, 0, 0]), params)
        new_pos_vel = RK4([pos_vel], differential_equation_normalized, ti, h, 1, E=E)
        initial_conditions[i][1] = new_pos_vel


    comp_time = time.time() - start_time
    print("\n=== Computation Statistics ===")
    print(f"Computation time: {comp_time:.4f} s")
    print("==============================\n")


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
        initial_condition = [Vector([initial_condition[0], initial_condition[1]]), None]
        initial_conditions.append(initial_condition)
