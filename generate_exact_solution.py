from generate_solutions import (
    compute_solution,
    compute_solution_trash_points,
)
from normalization import (
    NormalizationParameters,
    differential_equation_normalized,
    convert_to_normalized,
    convert_to_dimensional_time_only,
)
from Integrate_fonctions import RK4
from utils import Vector, save_time_interval, save_to_csv
from constants import RT, mp, MO, qe, mu

parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
initial_position = Vector([-4 * RT, -1 * RT, -6 * RT])
initial_velocity = RT * Vector([0.1, 0.1, 0.1])
initial_conditions = convert_to_normalized(
    initial_position, initial_velocity, parameters
)
initial_conditions = Vector([initial_conditions[0], initial_conditions[1]])
print("Conditions initiales :", initial_conditions)

solution_normalized, time_normalized = compute_solution_trash_points(
    RK4,
    differential_equation_normalized,
    20000000,
    0,
    10000000,
    initial_conditions,
    False,
    1,
    100,
)

# Extraction et conversion pour la sauvegarde
position = []
velocity = []
for i in range(len(solution_normalized)):
    # Attention: convert_to_dimensional_time_only garde la position normalisée, seule la vitesse est modifiée
    pos_denorm, vel_denorm = convert_to_dimensional_time_only(
        solution_normalized[i][0], solution_normalized[i][1], parameters
    )
    position.append(pos_denorm)
    velocity.append(vel_denorm)

# Mise à l'échelle du temps
time_rescaled = parameters.rescale_normalized_time_intervall(time_normalized)

# Sauvegarde dans 3 fichiers séparés
save_to_csv(position, "reference_position_rk4.csv")
save_to_csv(velocity, "reference_velocity_rk4.csv")
save_time_interval(time_rescaled, "time_interval_rk4.csv")

print("Génération de la solution exacte terminée avec succès !")
