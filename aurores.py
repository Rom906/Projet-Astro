from utils import Vector
from math import exp
from constants import RT
from typing import Callable, List, Tuple
from random import uniform
from integration_functions import RK4
from atmospheric_model import concentration_ni, O2, O, N2, H, HE, AR

molecules_index = [O, O2, H, HE, AR, N2]

def test_collision(conditions: List[Vector], molecule_index) -> bool:
    n = concentration_ni(abs(conditions[0]), molecules_index[molecule_index])
    s = cross_section(molecules_index[molecule_index], 0)
    v = conditions[1] - maxwell_boltzmann()
    collision_rate = n * s * abs(v)
    return uniform(0, 1) < collision_rate.real


def draw_maxwell_boltzmann_velocity(central_velocity, velocity_range):
    initial_velocity = Vector([initial_velocity, 0, 0])

    vx = np.random.normal(loc=central_velocity[0], scale=velocity_range)
    vy = np.random.normal(loc=central_velocity[1], scale=velocity_range)
    vz = np.random.normal(loc=central_velocity[2], scale=velocity_range)

    random_particle_speed = Vector([vx, vy, vz])
    return vitesse_particule_random


def cross_section(molecule, electron_nrg):
    if molecule == "O":
        return 1.0**(-17) #13.62eV
    elif molecule == "O2":
        return 10**(-19)
    elif molecule == "H":
        return 3.0186 * 10**(-19) #13eV 1s2 -> 1s2p
    elif molecule == "HE":
        return 3.5 * 10**(-17) #20.6eV 1S2 -> 1s2p
    elif molecule == "AR":
        return 2.5 * 10**(-20) #circa 20eV
    elif molecule == "N2":
        return 10**(-20)



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
