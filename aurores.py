from utils import Vector
from math import exp
from constants import RT
from random import random, uniform
from integration_functions import RK4
from generate_solutions import plot_3d

def collision_test(position: Vector):
    collision_probability = atm_model(position)["n"] * atm_model["surface"]
    return random() > collision_probability

H0 = 8000 #m
p0 = 130025 #Pa
R = 8.314 #J.K-1.mol-1
T0 = 273.15 #K

def atm_model(position: float):
    n = p0 * (1 + exp(-(position[0] - RT)/ H0)) / (R * T0) #assuming spherical coords
    surface = 1
    return {"n": n, "surface": surface}

def compute_collisional_trajectory(initial_conditions: Vector, differential_equation, h, max_collisions):
    n_collisions = 0
    solution = [initial_conditions]
    collision_positions = []
    t = 0
    while n_collisions < max_collisions:
        new = RK4([solution[-1]], differential_equation, t, h, 1)
        t += h
        if collision_test(new[0]): #assuming spherical coords
            for i in range(3):
                new[1][i] = new[1][i]
            collision_positions.append(new[0])
            n_collisions += 1
        solution.append(new)

    return solution, collision_positions
