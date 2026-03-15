# Consolidation des Méthodes Numériques

## Résumé des Changements

Toutes les méthodes de résolution numérique ont été consolidées dans un seul fichier : **`numerical_methods.py`**.

## Fichiers Créés

### `numerical_methods.py`
Nouveau fichier consolidé contenant :
- **Méthodes Euler**:
  - `Euler_v2()` - Single step pour Vector objects
  - `Euler()` - Full integration pour numpy arrays

- **Méthodes Runge-Kutta 4 (RK4)**:
  - `RK4()` - Single step compatible avec `compute_solution` (Vector objects)
  - `RK4_numpy()` - Legacy implementation pour numpy arrays
  - `normalize_r_position()` - Fonction utilitaire

- **Méthode Dormand-Prince**:
  - `dormand_prince()` - 5(4) adaptive method pour Vector objects

- **Méthode Velocity Verlet**:
  - `velocity_verlet()` - Symplectic integrator pour Vector objects

- **Visualisation**:
  - `plot_trajectory()` - Plot 3D de trajectoires

## Fichiers Modifiés

### **Imports Updated**:
| Fichier | Changement |
|---------|-----------|
| `test.py` | `from fonctions_RK4 import RK4` → `from numerical_methods import RK4` |
| `test_DP.py` | `from fonction_Dormand_Prince import dormand_prince` → `from numerical_methods import dormand_prince` |
| `test_verlet.py` | `from fonction_Verlet import velocity_verlet` → `from numerical_methods import velocity_verlet` |
| `main.py` | `from fonctions_RK4 import RK4` → `from numerical_methods import RK4_numpy as RK4` |
| `Initialisation.py` | `from fonctions_RK4 import RK4, ...` → `from numerical_methods import RK4_numpy as RK4, ...` |

### **Initialisation.py**:
- Mis à jour imports pour utiliser `numerical_methods`
- Remplacé `plot_trajectory_3D()` (inexistante) par `plot_trajectory()`
- Commenté l'appel au plotting pour éviter les erreurs d'affichage GUI

## Fichiers Obsolètes

Les fichiers suivants peuvent être supprimés (mais sont laissés pour archive) :
- `Euler_v2.py` - Consolidé dans `numerical_methods.py`
- `fonction_Dormand_Prince.py` - Consolidé dans `numerical_methods.py`
- `fonction_Verlet.py` - Consolidé dans `numerical_methods.py`
- `fonctions_Euler.py` - Consolidé dans `numerical_methods.py`
- `fonctions_RK4.py` - Consolidé dans `numerical_methods.py`

## Fichiers Inchangés

- `fonctions.py` - Contient les utilitaires généraux et fonctions de plotting (non affectées)
- `utils.py` - Classe Vector (non affectée)
- `normalized_equations.py` - Équations normalisées (non affectée)

## Validation

✅ Tous les tests ont réussi :
- test.py : RK4 method - **OK**
- test_DP.py : Dormand-Prince method - **OK**
- test_verlet.py : Velocity Verlet method - **OK**
- Initialisation.py : Full RK4_numpy integration - **OK**

## Avantages de la Consolidation

1. **Organisation centralisée** : Toutes les méthodes numériques au même endroit
2. **Maintenance simplifiée** : Modifications à un seul fichier
3. **Dépendances claires** : Imports explicites et unifiés
4. **Code DRY** : Import `import` au lieu d'importer de plusieurs fichiers sources
5. **Facilite l'évolution** : Ajout de nouvelles méthodes plus simple
