# Documentation

## Introduction
Ce répositoire GitHub ressemble l'ensemble de ce qui a été codé dans le cadre du projet Aurores du S4 2026 par Arthur FORESTIER, James GOLDIE, Roman GREZES-MOLÉNAT, Maxime LELOUP, Oscar LOUSTEAU, Julien PIERNAS et Téo PUJOL, sous l'encadrement de Carlo SCHIMD.

## Structure
Cette arborescence comporte de nombreuses branches correspondant aux travail de chacun mais la branche principale est `main`. Celui-ci contient toutes les fonctionnalités abouties du projet.

### `generate_solutions.py`
Ce fichier est rassemble toutes les fonctions qui

### `integration_functions.py`
Ce fichier contient l'ensemble des méthodes numériques d'intégration qui ont été comparé au cours du projet : Adams-Bashforth, Euler explicite, RK4, Dormand-Prince, Fast Verlet et Heun. La méthode numérique qui a été la plus efficace d'après les tests est Heun mais à l'heure actuelle le pas variable de la fonction `compute_solution` ne fonctionne qu'avec RK4.

Toutes les méthodes codées sont formattés pour être appliquée au cours d'une intégration de `compute_solution` et permettent donc d'approximer la valeur de la fonction recherché à un temps distant d'un pas `h` par rapport à la dernière valeur initiale fournie. Pour en savoir plus sur les méthodes numériques, nous recommandons le livre <u>Méthodes Numériques Pour l'Ingénieur</u> de Max CERF.