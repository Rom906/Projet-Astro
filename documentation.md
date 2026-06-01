# Documentation

## Introduction
Ce répositoire GitHub ressemble l'ensemble de ce qui a été codé dans le cadre du projet Aurores du S4 2026 par Arthur FORESTIER, James GOLDIE, Roman GREZES-MOLÉNAT, Maxime LELOUP, Oscar LOUSTEAU, Julien PIERNAS et Téo PUJOL, sous l'encadrement de Carlo SCHIMD.

## Structure
Cette arborescence comporte de nombreuses branches correspondant aux travail de chacun mais la branche principale est `main`. Celui-ci contient toutes les fonctionnalités abouties du projet.

### `generate_solutions.py`
Ce fichier est rassemble toutes les fonctions qui apporte directement des résultats physiques pour l'étude de la trajectoire d'une particule. Cela inclue la fonction `compute_solution` qui réalise l'intégration de la trajectoire à partir de condition initiales ainsi que de nombreuses fonctions qui apporte des éléments pour l'analyse d'une trajectoire, comme `plot_3d`, `plot_kinetic_energy` ou `plot_2d_projections`.

### `integration_functions.py`
Ce fichier contient l'ensemble des méthodes numériques d'intégration qui ont été comparé au cours du projet : Adams-Bashforth, Euler explicite, RK4, Dormand-Prince, Velocity Verlet et Heun. La méthode numérique qui a été la plus efficace d'après les tests est Heun mais à l'heure actuelle le pas variable de la fonction `compute_solution` ne fonctionne qu'avec RK4.

Toutes les méthodes codées sont formattés pour être appliquée au cours d'une intégration de `compute_solution` et permettent donc d'approximer la valeur de la fonction recherché à un temps distant d'un pas `h` par rapport à la dernière valeur initiale fournie. Elles peuvent résoudre toutes équation différentielle ordinaire de la forme `u' = f(u)`, au changement de variable près. Pour en savoir plus sur les méthodes numériques, nous recommandons le livre <u>Méthodes Numériques Pour l'Ingénieur</u> de Max CERF.

## `atmospheric_model.py`
Ce fichier contient toutes les fonctions qui portent sur les composantes du modèle atmosphérique de Jacchia (cf rapport). Ces fonctions permettent notamment de calculer la densité en nombre d'une particule atmosphérique donnée pour la fréquence de collision du modèle de Monte Carlo et la température à une altitude donnée pour évaluer une distribution de Maxwell-Boltzmann.

## `monte_carlo.py`
Ce fichier contient des fonctions nécessaires pour réaliser un test de collision, ie un tirage aléatoire en fonction d'une fréquence de collision. On appelle alors ce test au cours d'une intégration de trajectoire pour localiser des points de collisions d'une particule.

## `parallelisation_with_E.py`

## Pistes d'amélioration
Le code présent permet de calculer numériquement la trajectoire d'une ou plusieurs particules chargées. Il inclue plusieurs optimisations, tel que la réduction en mémoire, le pas variable et la parallélisation CPU. Cependant, ces optimisations se sont avérées insuffisantes pour simuler des aurores polaires entières (cf rapport). Il faudrait donc trouver conjointement une manière d'augmenter la puissance de calcul du code et une nouvelle approche physique moins lourde computativement.

En ce qui concerne le gain en puissance, il pourrait par exemple être pertienent d'appliquer de la parallélisation GPU et/ou de compiler le code pour ne pas perdre en temps d'interprétation. Cela peut se faire directement sur `Python` ce qui serait le plus simple car le code existant dansce language et tout étudiant.e en MPCI maîtrise ce language. Néanmoins, `Python` n'a pas le language le plus adapté pour ce genre d'optimisation bas-niveau donc il pourrait pertinent de passer sur un autre language, comme le `C` ou le `Fortran` de manière totale ou hybride.

En ce qui concerne la modification du problème physique, il existe de nombreuses approches différentes à explorer. Une première étape pourrait être d'améliorer le modèle qui a été proposé. Avec un parralélisation GPU et des conditions initiales différentes, il serait peut-être possible d'observer des aurores statiques avec l'outil proposé. Il est à noter que la méthode actuelle inclue un champ électrique entre les particules qui est calculé en O(n²) en calculant la force de Coulomb entre chaque particule mais que cela n'est pas très optimal. D'autres approches, telles que celles suggéré par Baranoski et al dans <u>Simulating the Aurora Borealis</u>, peuvent être beaucoup plus optimales. Il serait notamment possible de calculer le champ électrique à partir d'une distribution de charge calculé sur une grille 3D à l'aide de l'équation de Poisson résolué avec une méthode numérique sur les équations aux dérivées partielles. Cette approche permettrait de rapprocher la modèle entièrement particulaire de notre projet vers un modèle incluant des variables macroscopiques. 

Enfin, il pourrait également être pertinent de revoir des éléments qui datent du début du projet, telles que la stabilité des conditions initiales et la champ magnétique. Nous avons en effet étudié différentes conditions initiales pour tenter de déterminer lesquelles permettent d'obtenir des trajectoires stables et/ou d'observer un piégeage dans les ceintures de Van Hallen. Cependant les études que nous avons menés meriterait d'être explorés davantage et automatiser pour dresser un constat plus claire et rigureux sur la question des conditions initiales. De plus, nous avons implémenté en fin de projet un nouveau champ magnétique non uniforme, tel que proposé dans le modèle de Luhmann-Friesen, sans pour autant l'inclure dans les simulation de trajectoire. Il serait donc pertinent de la manipuler pour voir s'il apporte des résultats différents. Par ailleurs, d'autres modèles plus complexes sont envisageables, tel que World Magnetic Model (WMM). Il serait donc intéressant de soit développer des modèles de champs magnétiques plus sophistiqués, soit d'employer les modèles numériques existants de manière coordonnée avec le code. Cela s'applique également au modèle atmosphérique, actuellement analytique sous la forme du modèle de Jacchia, qui pourrait directment tirer profit des données du modèle NRLMSIS. Ce modèle, ainsi que celui des collisions de Monte Carlo, meriterait peut-être aussi d'inclure une composante de POO.