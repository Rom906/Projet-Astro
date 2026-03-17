import os
import time
import csv
from math import pi
from utils import Vector
from Integrate_fonctions import adams, RK4, euler, dormand_prince, Heun, velocity_verlet
from generate_solutions import compute_solution,compute_solution_trash_points,
from comparatif_solutions_csv_fonctions import (
    recup_reference_json,
    calculer_variation_energie,
    calculer_erreur_trajectoire,
    generer_recap_csv,
)
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
print(initial_conditions)


if __name__ == "__main__":
    FICHIER_REF = "reference_rk4.json"
    FICHIER_CSV = "resultats_comparatif.csv"
    ETAPES = 2000000
    

    # Vérification de la présence du fichier de référence
    solution_ref = None
    print(f"Fichier '{FICHIER_REF}' trouvé. L'erreur de trajectoire sera calculée.")
    resultat = recup_reference_json(FICHIER_REF)
    solution_ref = resultat[1]

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
        FICHIER_CSV, methodes_a_tester, differential_equation_normalized, ETAPES, 0, 10000000, initial_conditions, solution_ref
    )
