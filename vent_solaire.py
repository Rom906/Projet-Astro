import numpy as np
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

N_particules = 100

vitesse_moyenne = Vector([400000.0, 0.0, 0.0])

vitesse_thermique = 40000.0  # Écart-type de 40 km/s

vitesse_moyenne_norm = parameters.normalize_velocity(vitesse_moyenne)

vitesse_termique_nomr =

liste_conditions_initiales = []


for i in range(N_particules):
    vx = 
    vy = 
    vz = 

    vitesse_particule = Vector([vx, vy, vz])

    ci_particule = Vector([initial_conditions[0], vitesse_particule])
    liste_conditions_initiales.append(ci_particule)

print(f"{N_particules} particules générées avec succès !")

