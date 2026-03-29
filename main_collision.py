from aurores import compute_collisional_trajectory
from generate_solutions import compute_solution, plot_3d, compute_solution_trash_points, saved_plot_kinetic_energy, saved_plot_2d_projections
from normalization import (
    NormalizationParameters,
    differential_equation_normalized,
    convert_to_normalized
)
from utils import Vector, save_time_interval, save_to_csv
from constants import RT, mp, MO, qe, mu
from random import randrange

parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
initial_position = Vector([-4 * RT, -1 * RT, -6 * RT])
initial_velocity = RT * Vector([0.05, 0.05, 0.05])
initial_conditions = convert_to_normalized(
    initial_position, initial_velocity, parameters
)
print(initial_conditions)
compute_collisional_trajectory(initial_conditions, differential_equation_normalized, 0.1, 10)