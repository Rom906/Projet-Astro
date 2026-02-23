from utils import Vector
from math import pi


def methode_de_collocation_de_degre_trois(x_0, f, t_0, t_1):
    T = t_1 - t_0
    x_0_prime = T * f(x_0)
    x_1 = (19 / 7) * x_0
    x_1_prime = T * f(x_1)
    a_0 = x_0
    a_1 = x_0_prime
    a_2 = -3 * x_0 - 2 * x_0_prime + 3 * x_1 - x_1_prime
    a_3 = 2 * x_0 + x_0_prime - 2 * x_1 + x_1_prime
    return [a_0, a_1, a_2, a_3]


q = 1
mp = 2
mu0 = 1
m = Vector([1, 1, 1])
er = Vector([1, 0, 0])


def f(Y):
    y_0 = Y[0]
    y_1 = Y[1]
    f_0 = ((q / mp) * (mu0 / (4 * pi * y_0[0] ** 3)) * y_1) @ (3 * (m * er) * er - m)
    f_1 = y_0
    return Vector([f_0, f_1])


print(
    methode_de_collocation_de_degre_trois(
        Vector([Vector([1, 1, 1]), Vector([1, 1, 1])]), lambda x: x, 0, 1
    )
)
print(
    methode_de_collocation_de_degre_trois(
        Vector([Vector([1, 1, 1]), Vector([1, 1, 1])]), f, 0, 1
    )
)
