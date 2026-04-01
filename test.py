from generate_solutions import plot_3d, compute_solution, compute_solution_trash_points
from utils import Vector
from math import pi
from Integrate_fonctions import adams, RK4, euler, dormand_prince

m = Vector([100, 100, 100])
q = 0.1
mp = 0.1
mu0 = 0.1

vector_CL = Vector([Vector([0.001, -0.001, 0.001]), Vector([-3, 0, 0])])


def f(t, Y: Vector):
    y_0: Vector = Y[0]
    y_1: Vector = Y[1]
    y_1_norm = y_1.normalized()
    f_0 = (
        (q / mp)
        * (mu0 / (4 * pi * (abs(y_1) ** 3)))
        * y_0
        @ (3 * ((m * y_1_norm) * y_1_norm - m))
    )
    f_1 = y_0
    return Vector([f_0, f_1])


# test comparatif

# solutions_1 = compute_solution(adams, f, 10000000, 0, 1000000, vector_CL, False, 1)
<<<<<<< HEAD
<<<<<<< HEAD
solutions_2 = compute_solution_trash_points(RK4, f, 10000, 0, 1000, vector_CL, variable_steps=True)[0]
=======
solutions_2 = compute_solution(RK4, f, 100000, 0, 10000, vector_CL, False, 1)[0]
>>>>>>> main
=======
solutions_2 = compute_solution_trash_points(RK4, f, 10000, 0, 1000, vector_CL, variable_steps=True)[0]
>>>>>>> 3c10bac6b66c9701768e7b6aef24dddf28597952

# ploted_position_1 = []
# for i in range(0, len(solutions_1), 100):
#     ploted_position_1.append(solutions_1[i][1])
# plot_3d(ploted_position_1)

ploted_position_2 = []
for i in range(0, len(solutions_2), 100):
    ploted_position_2.append(solutions_2[i][1])
plot_3d(ploted_position_2)
