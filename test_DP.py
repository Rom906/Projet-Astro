from numerical_methods import dormand_prince
from fonctions import plot_3d, compute_solution
from utils import Vector
from math import pi

m = Vector([100, 100, 100])
q = 0.1
mp = 0.1
mu0 = 0.1

vector_CI = Vector([Vector([0.001, -0.001, 0.001]), Vector([-3, 0, 0])])


def f(t, Y: Vector):
    y_0: Vector = Y[0]
    y_1: Vector = Y[1]
    y_1_norm = y_1.normalized()
    f_0 = (
        (q / mp)
        * (mu0 / (4 * pi * (abs(y_1) ** 3)))
        * y_0
        @ (3 * ((m * (y_1_norm)) * y_1_norm - m))
    )
    f_1 = y_0
    return Vector([f_0, f_1])


solutions = compute_solution(dormand_prince, f, 100000, 0, 1000000, vector_CI, False, 1)
ploted_position = []
for i in range(0, len(solutions), 100):
    ploted_position.append(solutions[i][0])  # Position is [0], velocity is [1]
    
initial_velocity = vector_CI[1]  # Get initial velocity from conditions
plot_3d(ploted_position, initial_velocity=initial_velocity)
