import json
import csv
import time
from typing import List, Callable, Tuple, Dict, Any, Optional
from utils import Vector
from generate_solutions import compute_solution, compute_solution_trash_points
from normalization import (
    NormalizationParameters,
    convert_to_normalized,
    differential_equation_normalized,
)


def calculer_variation_energie(solution: List[Vector]) -> float:
    """
    Calcule la variation moyenne relative de l'énergie cinétique.
    Dans un champ magnétique pur, l'énergie cinétique (proportionnelle au carré de la vitesse) doit être constante.

    :param solution: la solution calculée par une méthode
    :type solution: List[Vector]
    :return: le pourcentage moyen de variation de l'énergie par rapport à l'état initial
    :rtype: float
    """
    vitesse_initiale_carre = abs(solution[0][1]) ** 2
    if vitesse_initiale_carre == 0:
        return 0.0

    energies = []
    for etat in solution:
        vitesse_actuelle_carre = abs(etat[1]) ** 2
        variation = (
            abs(vitesse_actuelle_carre - vitesse_initiale_carre)
            / vitesse_initiale_carre
        )
        energies.append(variation)

    return sum(energies) / len(energies)


def calculer_erreur_trajectoire(
    solution_test: List[Vector],
    solution_reference: List[Vector],
    solution_test_time: Optional[List[float]] = None,
    solution_reference_time: Optional[List[float]] = None,
) -> float:
    erreurs = []
    indice_ref = 0
    n_ref = len(solution_reference_time)

    for position, t in zip(solution_test, solution_test_time):

        while indice_ref <= n_ref - 1 and solution_reference_time[indice_ref + 1] < t:
            indice_ref += 1

        t0 = solution_reference_time[indice_ref]
        t1 = solution_reference_time[indice_ref + 1]

        # intervalle de temps entre les deux points de référence
        if t1 != t0:
            u = (t - t0) / (t1 - t0)
        else:
            u = 0.0

        # calcul position de référence à l'instant t par interpolation linéaire
        pos_ref_t = (
            solution_reference[indice_ref][0] * (1 - u)
            + solution_reference[indice_ref + 1][0] * u
        )

        erreurs.append(abs(position[0] - pos_ref_t))

    return sum(erreurs) / len(erreurs) if erreurs else 0.0


def formater_nombre(valeur: float) -> str:
    """Formate un nombre avec 4 décimales max"""
    texte = f"{valeur:.4f}"
    # Supprime les zéros inutiles à la fin
    texte = texte.rstrip("0").rstrip(".")
    # Si la valeur était exactement 0, le texte devient vide
    if not texte:
        texte = "0"
    return texte


def formater_pourcentage(valeur: float) -> str:
    if abs(valeur) <= 100:
        return f"{int(round(valeur))} %"
    else:
        # Notation scientifique avec 2 décimales
        texte = f"{valeur:.2e}"
        return f"{texte.replace('.', ',')} %"


def formater_erreur(valeur: float) -> str:
    if valeur == 0:
        return "0 u"
    elif 0.001 <= abs(valeur) < 10000:
        # Arrondi classique à 3 décimales (ex: 3,385)
        texte = f"{valeur:.3f}".rstrip("0").rstrip(".")
        return f"{texte.replace('.', ',')} u"
    else:
        # Notation scientifique pour les grosses erreurs
        texte = f"{valeur:.2e}"
        return f"{texte.replace('.', ',')} u"


def generer_recap_csv(
    fichier_csv: str,
    methodes: List[Dict[str, Any]],
    equation_differentielle: Callable,
    etapes: int,
    minimum: float,
    maximum: float,
    conditions_initiales: Vector,
    solution_reference: List[Vector],
    solution_reference_time: Optional[List[float]],
) -> None:
    """
    Exécute plusieurs méthodes d'intégration, calcule leurs performances et génère un fichier CSV récapitulatif.

    :param fichier_csv: le nom du fichier CSV de sortie (ex: "resultats.csv")
    :type fichier_csv: str
    :param methodes: une liste de dictionnaires décrivant les méthodes à tester
    :type methodes: List[Dict[str, Any]]
    :param equation_differentielle: la fonction f(t, Y) de l'équation différentielle
    :type equation_differentielle: Callable
    :param etapes: le nombre d'étapes de calcul
    :type etapes: int
    :param minimum: le temps de début
    :type minimum: float
    :param maximum: le temps de fin
    :type maximum: float
    :param conditions_initiales: le vecteur d'état initial [position, vitesse]
    :type conditions_initiales: Vector
    :param solution_reference: la solution exacte chargée au préalable pour comparer les trajectoires
    :type solution_reference: List[Vector]
    """

    resultats = []

    print("\nLancement du recap...")
    for methode in methodes:
        print(f"Test de {methode['nom']}...")
        temps_debut = time.time()

        solution_test, times_test = compute_solution_trash_points(
            methode["fonction"],
            differential_equation_normalized,
            etapes,
            0,
            100000,
            conditions_initiales,
            methode["multipas"],
            methode["nb_pas"],
        )

        temps_execution = time.time() - temps_debut

        # Formatage du temps : arrondi à la seconde près
        secondes_totales = int(round(temps_execution))
        minutes, secondes = divmod(secondes_totales, 60)
        temps_execution_propre = f"{minutes}m {secondes}s"

        # Calcul et formatage de l'énergie
        valeur_delta_energie = calculer_variation_energie(solution_test) * 100
        delta_energie = formater_pourcentage(valeur_delta_energie)

        # Calcul et formatage de l'erreur
        valeur_erreur_traj = calculer_erreur_trajectoire(
            solution_test,
            solution_reference,
            times_test,
            solution_reference_time,
        )
        erreur_traj = formater_erreur(valeur_erreur_traj)
        resultats.append(
            {
                "Methode": methode["nom"],
                "Temps Execution": temps_execution_propre,
                "Erreur Trajectoire Moyenne (u)": erreur_traj,
                "Variation Energie Moyenne (%)": delta_energie,
            }
        )

    # Génération du CSV
    en_tetes = [
        "Methode",
        "Temps Execution",
        "Erreur Trajectoire Moyenne (u)",
        "Variation Energie Moyenne (%)",
    ]
    with open(fichier_csv, mode="w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=en_tetes, delimiter=";")
        writer.writeheader()
        writer.writerows(resultats)

    print(f"\nRecap terminé ! Résultats sauvegardés dans '{fichier_csv}'.")
