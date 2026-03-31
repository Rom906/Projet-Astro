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


def plot_multi_particules(
    N_particules=10,
    cercle=3,
    distance_cercle=5,
    nombre_points=1000,
    intervalle_temps=100000,
    ratio_sur_100=1,
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

    for i in range(5):
        print(f"ci pour la {i}eme particule :")
        print(f"- position : {liste_conditions_initiales[i][0]}")
        print(f"- vitesse : {liste_conditions_initiales[i][1]}\n")

    liste_solutions = []

    for i in range(len(liste_conditions_initiales)):
        solution_normalized, time_normalized = compute_solution_trash_points(
            RK4,
            differential_equation_normalized,
            nombre_points,
            0,
            intervalle_temps,
            liste_conditions_initiales[i],
            False,
            ratio_sur_100,
            100,
        )
        liste_solutions.append(solution_normalized)
        print(f"Point de {i+1}eme particule générés avec succès !\n")

    plot_3d_multi(
        liste_solutions,
        cercle=cercle,
        distance_cercle=distance_cercle,
        nombre_points=nombre_points,
        intervalle_temps=intervalle_temps,
        ratio_sur_100=ratio_sur_100,
    )


if __name__ == "__main__":
    plot_multi_particules(
        N_particules=20,
        cercle=5,
        distance_cercle=8,
        nombre_points=4000,
        intervalle_temps=1000000,
        ratio_sur_100=1,
    )
