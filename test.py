from fonction_adams_bashforth import adams
from fonctions import plot_3d, compute_solution
from utils import Vector
from math import pi
from RK4 import RK4
from euler import euler

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
solutions_2 = compute_solution(euler, f, 100000, 0, 7000, vector_CL, False, 1)

# ploted_position_1 = []
# for i in range(0, len(solutions_1), 100):
#     ploted_position_1.append(solutions_1[i][1])
# plot_3d(ploted_position_1)

ploted_position_2 = []
for i in range(0, len(solutions_2), 100):
    ploted_position_2.append(solutions_2[i][1])
plot_3d(ploted_position_2)
