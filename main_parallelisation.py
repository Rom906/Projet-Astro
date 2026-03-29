from generate_solutions import compute_solution, plot_3d, compute_solution_trash_points, saved_plot_kinetic_energy, saved_plot_2d_projections
from normalization import (
    NormalizationParameters,
    differential_equation_normalized,
    convert_to_normalized
)
from Integrate_fonctions import adams, euler, RK4, dormand_prince, Heun
from utils import Vector, save_time_interval, save_to_csv
from constants import RT, mp, MO, qe, mu
from random import randrange
from multiprocessing import Pool


def compute_solution_specific(initial_conditions: Vector, parameters: NormalizationParameters, save_name_KE: str, save_name_phase: str, save_name_position: str, save_name_velocity: str, save_name_time_intervall: str):
    solution_normalized, time_noramlized = compute_solution_trash_points(
        RK4,
        differential_equation_normalized,
        100000,
        0,
        20000,
        initial_conditions,
        False,
        1,
        100,
        variable_steps=True
    )

    position = []
    velocity = []
    for i in range(len(solution_normalized)):
        position_denormalize, velocity_denormalized = solution_normalized[i][0], solution_normalized[i][1]
        position.append(position_denormalize)
        velocity.append(velocity_denormalized)

    time = parameters.rescale_normalized_time_intervall(time_noramlized)

    saved_plot_kinetic_energy(velocity, time, mp, save_name_KE)
    saved_plot_2d_projections(position, save_name_phase, velocity)
    save_to_csv(position, save_name_position)
    save_to_csv(velocity, save_name_velocity)
    save_time_interval(time, save_name_time_intervall)


if __name__ == "__main__":
    parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
    initial_conditions = []
    for i in range(4):
        x_start = randrange(3, 21)
        y_start = randrange(-20, 21)
        z_start = randrange(-20, 21)
        vx_start = randrange(1, 11) / 100
        vy_start = randrange(-10, 11) / 100
        vz_start = randrange(-10, 11) / 100
        initial_position = RT * Vector([x_start, y_start, z_start])
        initial_velocity = RT * Vector([vx_start, vy_start, vz_start])
        initial_condition = convert_to_normalized(initial_position, initial_velocity, parameters)
        initial_condition = Vector([initial_condition[0], initial_condition[1]])
        initial_conditions.append((initial_condition, parameters, f"save_KE_{i}_{x_start}_{y_start}_{z_start}_{vx_start}_{vy_start}_{vz_start}.png", f"save_phase_{i}_{x_start}_{y_start}_{z_start}_{vx_start}_{vy_start}_{vz_start}.png", f"save_position_{i}_{x_start}_{y_start}_{z_start}_{vx_start}_{vy_start}_{vz_start}.csv", f"save_time_{i}_{x_start}_{y_start}_{z_start}_{vx_start}_{vy_start}_{vz_start}.csv", f"save_KE_{i}_{x_start}_{y_start}_{z_start}_{vx_start}_{vy_start}_{vz_start}.csv"))

    with Pool() as p:
        p.starmap(compute_solution_specific, initial_conditions)
