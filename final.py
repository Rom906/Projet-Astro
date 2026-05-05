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
from multiprocessing import Pool, Manager, Barrier
import os


def compute_solution(index: int) -> None:
    if is_calculated[index]:
        params = NormalizationParameters(RT, qe / mp, MO, abs(mu))
        h = 0.01
        ti = 0
        global initial_conditions
        initial_condition = initial_conditions[index]
        E = Vector([0, 0, 0])
        pos_vel = initial_condition[0]
        for j in range(len(initial_conditions)):
            if pos_vel[0] != initial_conditions[j][0][0]:
                E += field_contribution(-qe, pos_vel[0], initial_conditions[j][0][0])
        E = convert_electric_field_to_normalized(E, params)
        new_pos_vel = RK4([pos_vel], differential_equation_normalized, ti, h, 1, E=E)  # type: ignore
        initial_conditions[index] = [pos_vel, new_pos_vel]
        for i in range(6):
            denormalized_posvel = convert_to_dimensional(new_pos_vel[0], new_pos_vel[1], params)
            conditions = Vector([denormalized_posvel[0], denormalized_posvel[1]])
            if test_collision(conditions, i):
                collision_points[index] = True


def init(shared, b, cp, ic):
    global barrier
    barrier = b
    global initial_conditions
    initial_conditions = shared
    global collision_points
    collision_points = cp
    global is_calculated
    is_calculated = ic



def compute_thread(indexs: List[int], nombre_pas):
    for i in range(nombre_pas):
        if i % 100 == 0:
            print(f"Step : {i} / 100000")
        try:
            for index in indexs:
                compute_solution(index)
        except Exception as e:
            print(e)
        barrier.wait()
        for index in indexs:
            if is_calculated[index]:
                new_vector = initial_conditions[index][1]
                if abs(new_vector[0]) >= 50:
                    is_calculated[index] = False
                initial_conditions[index] = [new_vector, None]
        barrier.wait()


if __name__ == "__main__":
    parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
    nombre_particules = 4
    nombre_pas = 100000
    with Manager() as manager:
        initial_conditions = manager.list()
        collision_points = manager.list()
        is_calculated = manager.list()
        for i in range(nombre_particules):
            collision_points.append(False)
            is_calculated.append(True)
        for i in range(nombre_particules):
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
        thread_count: int = os.cpu_count()  # type: ignore
        print(thread_count)
        arguments_all = []
        intervall_size = nombre_particules // thread_count
        remaining_threads = nombre_particules % thread_count
        barrier = Barrier(thread_count)
        for i in range(thread_count):
            liste_thread = [j for j in range(i * intervall_size, (i + 1) * intervall_size)]
            if i < remaining_threads:
                liste_thread.append(intervall_size * thread_count + i)
            arguments_all.append((liste_thread, nombre_pas))
        with Pool(initializer=init, initargs=(initial_conditions, barrier, collision_points, is_calculated)) as p:
            p.starmap(compute_thread, arguments_all)
        
        vectors = []
        for i in range(len(initial_conditions)):
            vectors.append(initial_conditions[i][0][0])
        collisions = []
        for i in range(nombre_particules):
            if collision_points[i]:
                collisions.append(1)
            else:
                collisions.append(6)
        plot_3d_collisions_only(vectors, collisions)
