import numpy as np
from utils import (
    Vector,
    save_time_interval,
    save_to_csv,
    load_time_interval,
    load_from_csv,
)
from integration_functions import (
    adams,
    RK4,
    euler,
    dormand_prince,
    Heun,
    velocity_verlet,
)

from constants import RT, qe, mp, MO, mu
from normalization import (
    NormalizationParameters,
    differential_equation_normalized,
    convert_to_normalized,
)
from generate_solutions import compute_solution_trash_points, plot_3d_multi
from time import time
from multiprocessing import Pool


def plot_multi_particules(
    N_particules=10,
    cercle=3,
    distance_cercle=5,
    nombre_points=1000,
    intervalle_temps=100000,
    ratio_sur_100=1,
    variable_steps=True,
    tolerated_variation=0.05,
):

    parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))

    rayon_cercle = cercle * RT

    # initial_position = Vector([-6 * RT, -3 * RT, -10 * RT])
    initial_velocity = Vector([400000, 0, 0])

    vitesse_thermique = 40000  # m/s

    liste_conditions_initiales = []

    for i in range(N_particules):
        x = -distance_cercle * RT
        r_alea = rayon_cercle * np.sqrt(np.random.uniform(0, 1))
        theta_aleatoire = np.random.uniform(0, 2 * np.pi)
        y = r_alea * np.cos(theta_aleatoire)
        z = r_alea * np.sin(theta_aleatoire)

        vx = np.random.normal(loc=initial_velocity[0], scale=vitesse_thermique)
        vy = np.random.normal(loc=initial_velocity[1], scale=vitesse_thermique)
        vz = np.random.normal(loc=initial_velocity[2], scale=vitesse_thermique)

        pos_particule = Vector([x, y, z])
        vitesse_particule = Vector([vx, vy, vz])
        ci_particule = Vector([pos_particule, vitesse_particule])
        liste_conditions_initiales.append(ci_particule)

    for i in range(len(liste_conditions_initiales)):
        ci = liste_conditions_initiales[i]
        ci_normalized = convert_to_normalized(ci[0], ci[1], parameters)
        liste_conditions_initiales[i] = Vector([ci_normalized[0], ci_normalized[1]])

    print(f"CI de {N_particules} particules générées avec succès !\n")

    if N_particules > 5:
        for i in range(5):
            print(f"ci pour la {i}eme particule :")
            print(f"- position : {liste_conditions_initiales[i][0]}")
            print(f"- vitesse : {liste_conditions_initiales[i][1]}\n")

    initial_conditions = []
    for condition in liste_conditions_initiales:
        initial_conditions.append(
            (
                RK4,
                differential_equation_normalized,
                nombre_points,
                0,
                intervalle_temps,
                condition,
                False,
                1,
                ratio_sur_100,
                # variable_steps,
                # tolerated_variation,
            )
        )

    liste_solutions = []

    with Pool() as p:
        results = p.starmap(compute_solution_trash_points, initial_conditions)
    for result in results:
        solution_normalized, time_normalized = result
        liste_solutions.append(solution_normalized)

    return (
        N_particules,
        liste_solutions,
        cercle,
        distance_cercle,
        nombre_points,
        intervalle_temps,
        ratio_sur_100,
    )


temps_initial = time()

if __name__ == "__main__":
    (
        N_particules,
        liste_solutions,
        cercle,
        distance_cercle,
        nombre_points,
        intervalle_temps,
        ratio_sur_100,
    ) = plot_multi_particules(
        N_particules=100,
        cercle=3,
        distance_cercle=10,
        nombre_points=100000,
        intervalle_temps=10000000,
        ratio_sur_100=1,
        variable_steps=True,
        tolerated_variation=0.05,
    )

    plot_3d_multi(
        N_particules,
        liste_solutions,
        cercle=cercle,
        distance_cercle=distance_cercle,
        nombre_points=nombre_points,
        intervalle_temps=intervalle_temps,
        ratio_sur_100=ratio_sur_100,
        color="multi",
        epaisseur=1,
    )

temps_execution = time() - temps_initial
print(f"Temps d'exécution : {temps_execution:.2f} secondes")
