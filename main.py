ffrom generate_solutions import (
    compute_solution,
    plot_3d,
    compute_solution_trash_points,
    plot_kinetic_energy,
)
from normalization import (
    NormalizationParameters,
    differential_equation_normalized,
    convert_to_dimensional_time_only,
    convert_to_normalized,
)
from Integrate_fonctions import adams, euler, RK4, dormand_prince
from utils import Vector
from constants import RT, mp, MO, qe, mu
from math import inf


parameters = NormalizationParameters(RT, qe / mp, MO, abs(mu))
initial_position = Vector([-4 * RT, -1 * RT, -6 * RT])
initial_velocity = RT * Vector([0.1, 0.1, 0.1])
initial_conditions = convert_to_normalized(
    initial_position, initial_velocity, parameters
)
initial_conditions = Vector([initial_conditions[0], initial_conditions[1]])
print(initial_conditions)
solution_normalized, time_noramlized = compute_solution_trash_points(
    RK4,
    differential_equation_normalized,
    2000000,
    0,
    10000000,
    initial_conditions,
    False,
    1,
    100,
)

print(len(solution_normalized))

position = []
velocity = []
for i in range(len(solution_normalized)):
    position_denormalize, velocity_denormalized = (
        solution_normalized[i][0],
        solution_normalized[i][1],
    )
    position.append(position_denormalize)
    velocity.append(velocity_denormalized)

max_x = position[0][0]
max_y = position[0][1]
max_z = position[0][2]

min_x = position[0][0]
min_y = position[0][1]
min_z = position[0][2]

for i in range(len(position)):
    if max_x < position[i][0]:
        max_x = position[i][0]
    elif min_x > position[i][0]:
        min_x = position[i][0]
    if max_y < position[i][1]:
        max_y = position[i][1]
    elif min_y > position[i][1]:
        min_y = position[i][1]
    if max_z < position[i][2]:
        max_z = position[i][2]
    elif min_z > position[i][2]:
        min_z = position[i][2]

print(min_x, max_x)
print(min_y, max_y)
print(min_z, max_z)

time = parameters.rescale_normalized_time_intervall(time_noramlized)

plot_3d(position)
plot_kinetic_energy(velocity, time, mp)