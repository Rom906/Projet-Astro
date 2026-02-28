from fonction_adams_bashforth import adams
from fonctions import plot_3d, compute_solution
from utils import Vector
from math import pi

m = Vector([1, 1, 1])
q = 1
mp = 1
mu0 = 1

vector_CI = Vector([Vector([5, 2, 0]), Vector([0.5, 0.5, 0.5])])


def f(t, Y):
    v = Y[0]
    r = Y[1]

    er = r.normalized()

    B = (mu0 / (4 * pi * v[0] ** 3)) * (3 * (m * er) * er - m)

    f_0 = (q / mp) * (v @ B)
    f_1 = v

    return Vector([f_0, f_1])


solutions = compute_solution(adams, f, 100000, -1, 1, vector_CI, True, 4)
ploted_position = []
for i in range(len(solutions)):
    ploted_position.append(solutions[i][1])
plot_3d(ploted_position)
