from utils import (
    Vector,
    save_time_interval,
    save_to_csv,
    load_time_interval,
    load_from_csv,
)
from Integrate_fonctions import adams, RK4, euler, dormand_prince, Heun, velocity_verlet

from constants import RT, qe, mp, MO, mu
from normalization import (
    NormalizationParameters,
    differential_equation_normalized,
    convert_to_normalized,
)

parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
initial_position = Vector([-4 * RT, -1 * RT, -6 * RT])
initial_velocity = RT * Vector([0.1, 0.1, 0.1])
initial_conditions = convert_to_normalized(
    initial_position, initial_velocity, parameters
)
initial_conditions = Vector([initial_conditions[0], initial_conditions[1]])
print("Conditions initiales :", initial_conditions)

