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
from generate_solutions import compute_solution_trash_points, plot_3d

parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
initial_position = Vector([-4 * RT, -1 * RT, -6 * RT])
initial_velocity = RT * Vector([0.1, 0.1, 0.1])
initial_conditions = convert_to_normalized(
    initial_position, initial_velocity, parameters
)
initial_conditions = Vector([initial_conditions[0], initial_conditions[1]])

N_particules = 10

vitesse_vent_lent = Vector([40.0, 40.0, 40.0])
vitesse_vent_rapide = Vector([65.0, 65.0, 65.0])

vitesse_thermique = 4.0
facteur_conversion = parameters.R0 / parameters.T
vitesse_termique_nomr = vitesse_thermique * facteur_conversion

vitesse_moyenne_norm = parameters.normalize_velocity(vitesse_vent_lent)


liste_conditions_initiales = []


for i in range(N_particules):
    vx = np.random.normal(loc=vitesse_moyenne_norm[0], scale=vitesse_termique_nomr)
    vy = np.random.normal(loc=vitesse_moyenne_norm[0], scale=vitesse_termique_nomr)
    vz = np.random.normal(loc=vitesse_moyenne_norm[0], scale=vitesse_termique_nomr)

    vitesse_particule = Vector([vx, vy, vz])

    ci_particule = Vector([initial_conditions[0], vitesse_particule])
    liste_conditions_initiales.append(ci_particule)

print(f"{N_particules} particules générées avec succès !\n")

# for i in range(5):
#     print(f"ci pour la {i}eme particule :")
#     print(f"- position : {liste_conditions_initiales[0][0]}")
#     print(f"- vitesse : {liste_conditions_initiales[0][1]}\n")

liste_solutions = []

for i in range(len(liste_conditions_initiales)):
    solution_normalized, time_normalized = compute_solution_trash_points(
        RK4,
        differential_equation_normalized,
        2000000,
        0,
        10000000,
        liste_conditions_initiales[i],
        False,
        1,
        100,
    )
    liste_solutions.append(solution_normalized)

# fonction à completer dans generate_solutions
plot_3d_plusieurs_particules(liste_solutions[0])
