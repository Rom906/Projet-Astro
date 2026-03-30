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
from generate_solutions import compute_solution_trash_points, plot_3d_multi

parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
initial_position = Vector([-4 * RT, -1 * RT, -6 * RT])
initial_velocity = RT * Vector([0.0628, 0, 0])
initial_conditions = convert_to_normalized(
    initial_position, initial_velocity, parameters
)
initial_conditions = Vector([initial_conditions[0], initial_conditions[1]])
print("Conditions initiales normalisées :", initial_conditions)
pos_norm_base = initial_conditions[0]
vit_norm_base = initial_conditions[1]


vitesse_thermique = 40000
vitesse_thermique_norm = parameters.normalize_velocity(
    Vector([vitesse_thermique, 0, 0])
)[0]


N_particules = 10
liste_conditions_initiales = []


for i in range(N_particules):
    vx = np.random.normal(loc=vit_norm_base[0], scale=vitesse_thermique_norm)
    vy = np.random.normal(loc=vit_norm_base[1], scale=vitesse_thermique_norm)
    vz = np.random.normal(loc=vit_norm_base[2], scale=vitesse_thermique_norm)

    vitesse_particule = Vector([vx, vy, vz])
    ci_particule = Vector([pos_norm_base, vitesse_particule])
    liste_conditions_initiales.append(ci_particule)

print(f"CI de {N_particules} particules générées avec succès !\n")

for i in range(5):
    print(f"ci pour la {i}eme particule :")
    print(f"- position : {liste_conditions_initiales[i][0]}")
    print(f"- vitesse : {liste_conditions_initiales[i][1]}\n")

liste_solutions = []

for i in range(len(liste_conditions_initiales)):
    solution_normalized, time_normalized = compute_solution_trash_points(
        RK4,
        differential_equation_normalized,
        200000,
        0,
        4000000,
        liste_conditions_initiales[i],
        False,
        1,
        100,
    )
    liste_solutions.append(solution_normalized)
    print(f"Point de {i+1}eme particule générés avec succès !\n")

# fonction à completer dans generate_solutions
plot_3d_multi(liste_solutions, initial_position=pos_norm_base, initial_reference_velocity=vit_norm_base)
