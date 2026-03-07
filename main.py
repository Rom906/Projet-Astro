from generate_solutions import compute_solution
from normalization import (
    NormalizationParameters,
    differential_equation_normalized,
    convert_to_dimensional,
)
from Integrate_fonctions import adams, euler, RK4, dormand_prince
from utils import Vector
from constants import RT, mp, MO, qe, mu


parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
solution_normalized, time_noramlized = compute_solution(
    RK4,
    differential_equation_normalized,
    2000000,
    0,
    1000,
    Vector([Vector([0, 1, 0]), Vector([0, -0.05, -0.05])]),
)

position_normalized = solution_normalized[0]
velocity_normalized = solution_normalized[1]
position, velocity = convert_to_dimensional(
    position_normalized, velocity_normalized, parameters
)
time = parameters.rescale_normalized_time_intervall(time_noramlized)
