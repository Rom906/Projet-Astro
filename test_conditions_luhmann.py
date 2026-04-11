"""
Conditions initiales pour mettre en avant la modification de Luhmann sur le champ B.

La modification de Luhmann ajoute un terme de queue magnétosphérique:
    BT * sign(μ·û) * x̂ 
qui crée une asymétrie nord-sud.

Pour observer clairement cette différence, on utilise des conditions initiales
qui permettent à la particule de traverser les régions où cet effet est maximal.
"""

from utils import Vector
from normalization import NormalizationParameters
from constants import RT, mp, MO, qe, mu
from generate_solutions import compute_solution_trash_points, plot_3d_v2, plot_2d_projections
from normalization import differential_equation_normalized, convert_to_normalized
from integration_functions import RK4


# ============================================================================
# CONDITION 1: TRAJECTOIRE DE LA QUEUE (zone d'influence majeure de Luhmann)
# ============================================================================
# Position: Loin dans la queue (côté nuit)
# Vitesse: Petite, pour rester longtemps dans la région affectée
# Effet attendu: Forte asymétrie causée par le terme Luhmann

print("\n" + "="*70)
print("CONDITION 1: TRAJECTOIRE DE QUEUE (Luhmann maximal)")
print("="*70)

parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))

# Position dans la queue, loin de la Terre
initial_position_1 = Vector([-8 * RT, 0 * RT, 0 * RT])
initial_velocity_1 = RT * Vector([0.005, 0.002, 0.003])

initial_conditions_1 = convert_to_normalized(
    initial_position_1, initial_velocity_1, parameters
)
initial_conditions_1 = Vector([initial_conditions_1[0], initial_conditions_1[1]])
print(f"Position initiale: {initial_position_1}")
print(f"Vitesse initiale: {initial_velocity_1}")
print(f"Position normalisée: {initial_conditions_1[0]}")
print(f"Vitesse normalisée: {initial_conditions_1[1]}")

# Calcul SANS modification Luhmann
print("\n--- Sans modification Luhmann (dipôle seul) ---")
solution_without_luhmann_1, time_1 = compute_solution_trash_points(
    RK4,
    lambda tau, Y: differential_equation_normalized(
        tau, Y, mu_direction=mu.normalized(), add_tail=False
    ),
    35000, 0, 20000000, initial_conditions_1, False, 1, 10,
)

# Calcul AVEC modification Luhmann
print("--- Avec modification Luhmann (dipôle + queue) ---")
solution_with_luhmann_1, time_1 = compute_solution_trash_points(
    RK4,
    lambda tau, Y: differential_equation_normalized(
        tau, Y, mu_direction=mu.normalized(), add_tail=True
    ),
    35000, 0, 20000000, initial_conditions_1, False, 1, 10,
)

# Extraction des positions
position_without_1 = [solution_without_luhmann_1[i][0] for i in range(len(solution_without_luhmann_1))]
position_with_1 = [solution_with_luhmann_1[i][0] for i in range(len(solution_with_luhmann_1))]

print(f"\nNombre de points de la trajectoire: {len(position_without_1)}")


# ============================================================================
# CONDITION 2: TRAJECTOIRE ÉQUATORIALE (symétrie visible)
# ============================================================================
# Position: Plan équatorial, proche distance
# Vitesse: Modérée, pour explorer les variations de B
# Effet attendu: Déviation asymétrique due à Luhmann vs symétrie dipolaire

print("\n" + "="*70)
print("CONDITION 2: TRAJECTOIRE ÉQUATORIALE (Luhmann visible)")
print("="*70)

initial_position_2 = Vector([-4 * RT, 2 * RT, 0 * RT])
initial_velocity_2 = RT * Vector([0.01, 0.005, 0.001])

initial_conditions_2 = convert_to_normalized(
    initial_position_2, initial_velocity_2, parameters
)
initial_conditions_2 = Vector([initial_conditions_2[0], initial_conditions_2[1]])
print(f"Position initiale: {initial_position_2}")
print(f"Vitesse initiale: {initial_velocity_2}")

# SANS Luhmann
print("\n--- Sans modification Luhmann (dipôle seul) ---")
solution_without_luhmann_2, time_2 = compute_solution_trash_points(
    RK4,
    lambda tau, Y: differential_equation_normalized(
        tau, Y, mu_direction=mu.normalized(), add_tail=False
    ),
    35000, 0, 20000000, initial_conditions_2, False, 1, 10,
)

# AVEC Luhmann
print("--- Avec modification Luhmann (dipôle + queue) ---")
solution_with_luhmann_2, time_2 = compute_solution_trash_points(
    RK4,
    lambda tau, Y: differential_equation_normalized(
        tau, Y, mu_direction=mu.normalized(), add_tail=True
    ),
    35000, 0, 20000000, initial_conditions_2, False, 1, 10,
)

position_without_2 = [solution_without_luhmann_2[i][0] for i in range(len(solution_without_luhmann_2))]
position_with_2 = [solution_with_luhmann_2[i][0] for i in range(len(solution_with_luhmann_2))]

print(f"\nNombre de points de la trajectoire: {len(position_without_2)}")


# ============================================================================
# CONDITION 3: TRAJECTOIRE POLAIRE (effet directionnel du terme x̂)
# ============================================================================
# Position: Région polaire
# Vitesse: Direction perpendiculaire au dipôle
# Effet attendu: Effet du terme x̂ qui brise la symétrie

print("\n" + "="*70)
print("CONDITION 3: TRAJECTOIRE POLAIRE (asymétrie nord-sud)")
print("="*70)

initial_position_3 = Vector([-2 * RT, -4 * RT, -6 * RT])
initial_velocity_3 = RT * Vector([0.01, 0.01, 0.01])

initial_conditions_3 = convert_to_normalized(
    initial_position_3, initial_velocity_3, parameters
)
initial_conditions_3 = Vector([initial_conditions_3[0], initial_conditions_3[1]])
print(f"Position initiale: {initial_position_3}")
print(f"Vitesse initiale: {initial_velocity_3}")

# SANS Luhmann
print("\n--- Sans modification Luhmann (dipôle seul) ---")
solution_without_luhmann_3, time_3 = compute_solution_trash_points(
    RK4,
    lambda tau, Y: differential_equation_normalized(
        tau, Y, mu_direction=mu.normalized(), add_tail=False
    ),
    350000, 0, 20000000, initial_conditions_3, False, 1, 10,
)

# AVEC Luhmann
print("--- Avec modification Luhmann (dipôle + queue) ---")
solution_with_luhmann_3, time_3 = compute_solution_trash_points(
    RK4,
    lambda tau, Y: differential_equation_normalized(
        tau, Y, mu_direction=mu.normalized(), add_tail=True
    ),
    350000, 0, 20000000, initial_conditions_3, False, 1, 10,
)

position_without_3 = [solution_without_luhmann_3[i][0] for i in range(len(solution_without_luhmann_3))]
position_with_3 = [solution_with_luhmann_3[i][0] for i in range(len(solution_with_luhmann_3))]

print(f"\nNombre de points de la trajectoire: {len(position_without_3)}")


# ============================================================================
# RÉSUMÉ DES CONDITIONS INITIALES
# ============================================================================
print("\n" + "="*70)
print("RÉSUMÉ: CONDITIONS INITIALES POUR MONTRER LA MODIFICATION LUHMANN")
print("="*70)

conditions_summary = {
    "Condition 1 (Queue)": {
        "position": initial_position_1.coordinates,
        "velocity": [v/RT for v in initial_velocity_1.coordinates],
        "description": "Zone de la queue où Luhmann a maximal effet"
    },
    "Condition 2 (Équatoriale)": {
        "position": initial_position_2.coordinates,
        "velocity": [v/RT for v in initial_velocity_2.coordinates],
        "description": "Plan équatorial pour voir l'asymétrie nord-sud"
    },
    "Condition 3 (Polaire)": {
        "position": initial_position_3.coordinates,
        "velocity": [v/RT for v in initial_velocity_3.coordinates],
        "description": "Région polaire pour effet du terme directif x̂"
    },
}

for cond_name, cond_data in conditions_summary.items():
    print(f"\n{cond_name}:")
    print(f"  Position: {cond_data['position']}")
    print(f"  Vitesse (en unités de RT/s): {cond_data['velocity']}")
    print(f"  → {cond_data['description']}")


# ============================================================================
# VISUALISATIONS AVEC plot_3d
# ============================================================================
print("\n" + "="*70)
print("GÉNÉRATION DES GRAPHIQUES 3D")
print("="*70)

import numpy as np

print("\n1️⃣  Plot Condition 1 (Queue) - Sans Luhmann...")
plot_3d_v2(position_without_1, initial_velocity_1, magnetic_moment=mu)

print("\n1️⃣  Plot Condition 1 (Queue) - Avec Luhmann...")
plot_3d_v2(position_with_1, initial_velocity_1, magnetic_moment=mu)

print("\n2️⃣  Plot Condition 2 (Équatoriale) - Sans Luhmann...")
plot_3d_v2(position_without_2, initial_velocity_2, magnetic_moment=mu)

print("\n2️⃣  Plot Condition 2 (Équatoriale) - Avec Luhmann...")
plot_3d_v2(position_with_2, initial_velocity_2, magnetic_moment=mu)

print("\n3️⃣  Plot Condition 3 (Polaire) - Sans Luhmann...")
plot_3d_v2(position_without_3, initial_velocity_3, magnetic_moment=mu)

print("\n3️⃣  Plot Condition 3 (Polaire) - Avec Luhmann...")
plot_3d_v2(position_with_3, initial_velocity_3, magnetic_moment=mu)

print("\n" + "="*70)
print("✅ GRAPHIQUES 3D GÉNÉRÉS AVEC SUCCÈS")
print("="*70)
