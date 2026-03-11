from generate_solutions import compute_solution, plot_3d
from normalization import (
    NormalizationParameters,
    differential_equation_normalized,
    convert_to_dimensional_time_only,
    convert_to_normalized
)
from Integrate_fonctions import adams, euler, RK4, dormand_prince
from utils import Vector
from constants import RT, mp, MO, qe, mu


parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
initial_position = Vector([0, 0.925e8, 0])
initial_velocity = 4e5 * 3 ** (1 / 2) * Vector([1, 1, 1])
initial_conditions = convert_to_normalized(initial_position, initial_velocity, parameters)
initial_conditions = Vector([initial_conditions[0], initial_conditions[1]])
print(initial_conditions)
solution_normalized, time_noramlized = compute_solution(
    RK4,
    differential_equation_normalized,
    200000,
    0,
    100000,
    initial_conditions,
)

position = []
velocity = []
for i in range(len(solution_normalized)):
    position_denormalize, velocity_denormalized = solution_normalized[i][0], solution_normalized[i][1]
    position.append(position_denormalize)
    velocity.append(velocity_denormalized)

time = parameters.rescale_normalized_time_intervall(time_noramlized)

plot_3d(position)
