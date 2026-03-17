from generate_solutions import (
    compute_solution,
    compute_solution_trash_points,
)
from normalization import (
    NormalizationParameters,
    differential_equation_normalized,
    convert_to_normalized,
)
from Integrate_fonctions import adams, euler, RK4, dormand_prince
from utils import Vector
from constants import RT, mp, MO, qe, mu
from comparatif_solutions_csv_fonctions import sauv_reference_json

parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
initial_position = Vector([-4 * RT, -1 * RT, -6 * RT])
initial_velocity = RT * Vector([0.1, 0.1, 0.1])
initial_conditions = convert_to_normalized(
    initial_position, initial_velocity, parameters
)
initial_conditions = Vector([initial_conditions[0], initial_conditions[1]])
print(initial_conditions)
solution_normalized, time_noramlized = compute_solution_trash_points(
    RK4,
    differential_equation_normalized,
    200000000,
    0,
    10000000,
    initial_conditions,
    False,
    1,
    100,
)

FICHIER_REF = "reference_rk4.json"

sauv_reference_json(FICHIER_REF, time_noramlized, solution_normalized)
