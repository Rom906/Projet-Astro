import json
import csv
import time
from typing import List, Callable, Tuple, Dict, Any
from utils import Vector
from generate_solutions import compute_solution, compute_solution_trash_points
from normalization import (
    NormalizationParameters,
    convert_to_normalized,
    differential_equation_normalized,
)


def sauv_reference_json(
    fichier_sortie: str, temps: List[float], solution: List[Vector]
) -> None:
    """
    Sauvegarde la trajectoire et les vitesses de référence dans un fichier JSON.

    :param fichier_sortie: le nom du fichier JSON de destination (ex: "reference.json")
    :type fichier_sortie: str
    :param temps: la liste des temps correspondant à chaque étape
    :type temps: List[float]
    :param solution: la liste des états [position, vitesse] à chaque étape
    :type solution: List[Vector]
    """
    donnees = {
        "temps": temps,
        "positions": [sol[0].coordinates for sol in solution],
        "vitesses": [sol[1].coordinates for sol in solution],
    }

    with open(fichier_sortie, "w") as f:
        json.dump(donnees, f)
    print(f"Référence sauvegardée avec succès dans {fichier_sortie}")


def recup_reference_json(fichier_entree: str) -> Tuple[List[float], List[Vector]]:
    """
    Charge une trajectoire de référence depuis un fichier JSON.

    :param fichier_entree: le nom du fichier JSON à lire
    :type fichier_entree: str
    :return: un tuple contenant la liste des temps et la liste des états [position, vitesse]
    :rtype: Tuple[List[float], List[Vector]]
    """
    with open(fichier_entree, "r") as f:
        donnees = json.load(f)

    temps = donnees["temps"]
    solution = []

    for pos, vit in zip(donnees["positions"], donnees["vitesses"]):
        etat = Vector([Vector(pos), Vector(vit)])
        solution.append(etat)

    return temps, solution


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
    solution_test: List[Vector], solution_reference: List[Vector]
) -> float:
    """
    Calcule l'écart spatial moyen entre la trajectoire testée et la trajectoire de référence.
    S'adapte automatiquement à la liste la plus courte si les tailles diffèrent.
    """
    taille_min = min(len(solution_test), len(solution_reference))

    erreurs = []
    for i in range(taille_min):
        distance = abs(solution_test[i][0] - solution_reference[i][0])
        erreurs.append(distance)

    return sum(erreurs) / len(erreurs)


def generer_recap_csv(
    fichier_csv: str,
    methodes: List[Dict[str, Any]],
    equation_differentielle: Callable,
    etapes: int,
    minimum: float,
    maximum: float,
    conditions_initiales: Vector,
    solution_reference: List[Vector],
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

        solution_test = compute_solution_trash_points(
            methode["fonction"],
            differential_equation_normalized,
            etapes,
            0,
            100,
            conditions_initiales,
            methode["multipas"],
            methode["nb_pas"],
        )[0]

        temps_execution = time.time() - temps_debut
        minutes, secondes = divmod(
            temps_execution, 60
        )  # divmode renvoie le quotient et le reste de la division, ici pour convertir les secondes en minutes et secondes
        temps_execution_propre = f"{int(minutes)}m {secondes:.2f}s"

        delta_energie = (
            calculer_variation_energie(solution_test) * 100
        )  # en pourcentage
        delta_energie = f"{delta_energie:.2e} %"

        # Calcul de l'erreur uniquement si la référence est disponible

        erreur_traj = calculer_erreur_trajectoire(solution_test, solution_reference)
        erreur_traj = f"{erreur_traj:.2e}u"
        resultats.append(
            {
                "Methode": methode["nom"],
                "Temps Execution (s)": temps_execution_propre,
                "Erreur Trajectoire Moyenne": erreur_traj,
                "Variation Energie Moyenne": delta_energie,
            }
        )

    # Génération du CSV
    en_tetes = [
        "Methode",
        "Temps Execution (min et s)",
        "Delta Erreur Trajectoire Moyenne (u : unités de distance normalisée)",
        "Variation Energie Moyenne (%)",
    ]
    with open(fichier_csv, mode="w", newline="") as f_csv:
        # Ajoute le délimiteur ici :
        writer = csv.DictWriter(f_csv, fieldnames=en_tetes, delimiter=";")
        writer.writeheader()
        writer.writerows(resultats)

    print(f"\nRecap terminé ! Résultats sauvegardés dans '{fichier_csv}'.")
