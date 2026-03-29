from aurores import compute_collisional_trajectory
from generate_solutions import plot_3d
from normalization import (
    NormalizationParameters,
    differential_equation_normalized,
    convert_to_normalized
)
from utils import Vector, convert_spherical_to_cartesian
from constants import RT, mp, MO, qe, mu
from random import randrange
from math import pi

parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
initial_position = Vector([2*RT, 0, pi/2])
initial_velocity = RT * Vector([1, 0, 0])
initial_conditions = convert_to_normalized(initial_position, initial_velocity, parameters)
initial_conditions = Vector([initial_conditions[0], initial_conditions[1]])

solution, collision_points = compute_collisional_trajectory(initial_conditions, differential_equation_normalized, 0.01, 100)

trajectory = []
for vector in solution:
    trajectory.append(convert_spherical_to_cartesian(vector[0]))

plot_3d(trajectory)