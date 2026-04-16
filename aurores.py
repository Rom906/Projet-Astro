from utils import Vector
from math import exp
from constants import RT
from random import uniform
from integration_functions import RK4


def collision_test(conditions: Vector, concentration_ni, molecule: dict, ):
    n = concentration_ni(abs(conditions[0]), molecule["index"])
    s = molecule["cross_section"]
    v = conditions[i] - maxwell_boltzmann()
    collision_rate = n * s * v
    return uniform() > collision_rate


def maxwell_boltzmann():
    return Vector([0, 0, 0])


H0 = 8000  # m
p0 = 130025  # Pa
R = 8.314  # J.K-1.mol-1
T0 = 273.15  # K
eff_sec_n2 = 8e-19  # m²


def atm_model(position: float):
    n = p0 * (1 + exp(-(abs(position) - 1) / H0)) / (R * T0)  # assuming spherical coords
    eff_sec = eff_sec_n2 / RT
    return {"n": n, "eff_sec": eff_sec}


def compute_collisional_trajectory(initial_conditions: Vector, differential_equation, h, max_collisions):
    n_collisions = 0
    solution = [initial_conditions]
    collision_positions = []
    t = 0
    while n_collisions < max_collisions:
        new = RK4([solution[-1]], differential_equation, t, h, 1)
        t += h
        if collision_test(new[0]):
            for i in range(3):
                new[1][i] = -new[1][i]
            collision_positions.append(new[0])
            n_collisions += 1
        print(n_collisions)
        solution.append(new)

    return solution, collision_positions
