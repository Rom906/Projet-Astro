from utils import (
    Vector,
    save_time_interval,
    save_to_csv,
    load_time_interval,
    load_from_csv,
)
from Integrate_fonctions import adams, RK4, euler, dormand_prince, Heun, velocity_verlet
from comparatif_solutions_csv_fonctions import generer_recap_csv
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


if __name__ == "__main__":
    FICHIER_POS_REF = "reference_position_rk4.csv"
    FICHIER_VEL_REF = "reference_velocity_rk4.csv"
    FICHIER_TIME_REF = "time_interval_rk4.csv"
    FICHIER_CSV = "resultats_comparatif.csv"
    ETAPES = 2000000

    print("Chargement des fichiers de référence...")

    # 1. On charge séparément les positions et les vitesses
    ref_positions = load_from_csv(FICHIER_POS_REF)
    ref_velocities = load_from_csv(FICHIER_VEL_REF)

    # 2. On reconstruit la solution complète [Vector(pos), Vector(vel)] attendue par le calcul d'erreur
    solution_ref = []
    for p, v in zip(ref_positions, ref_velocities):
        solution_ref.append(Vector([p, v]))

    # 3. On charge le temps "rescalé" et on le remet en normalisé pour qu'il soit compatible avec le recap
    resultat_time_rescaled = load_time_interval(FICHIER_TIME_REF)
    resultat_time = parameters.normalize_time_intervall(resultat_time_rescaled)

    methodes_a_tester = [
        {"nom": "Euler", "fonction": euler, "multipas": False, "nb_pas": 1},
        {"nom": "Runge-Kutta 4", "fonction": RK4, "multipas": False, "nb_pas": 1},
        {"nom": "Adams-Bashforth 4", "fonction": adams, "multipas": True, "nb_pas": 4},
        {
            "nom": "Dormand-Prince",
            "fonction": dormand_prince,
            "multipas": False,
            "nb_pas": 1,
        },
        {"nom": "Heun", "fonction": Heun, "multipas": False, "nb_pas": 1},
        {
            "nom": "Velocity Verlet",
            "fonction": velocity_verlet,
            "multipas": False,
            "nb_pas": 1,
        },
    ]

    generer_recap_csv(
        FICHIER_CSV,
        methodes_a_tester,
        differential_equation_normalized,
        ETAPES,
        0,
        10000000,
        initial_conditions,
        solution_ref,
        resultat_time,
    )
