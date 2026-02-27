from fonction_adams_bashforth import adams
from fonctions import plot_3d, compute_solution
from utils import Vector
from math import pi

m = Vector([1, 1, 1])
q = 1
mp = 1
mu0 = 1
er = Vector([1, 0, 0])

vector_CI = Vector([Vector([-10, 0, 0]), Vector([1, 1, 1])])


def f(t, Y):
    y_0 = Y[0]
    y_1 = Y[1]
    f_0 = ((q / mp) * (mu0 / (4 * pi * y_0[0] ** 3)) * y_1) @ (3 * (m * er) * er - m)
    f_1 = y_0
    return Vector([f_0, f_1])


solutions = compute_solution(adams, f, 100, -10, 10, vector_CI, True, 4)
