import generate_solution as gs
from utils import Vector
from typing import List, Callable
from math import exp


def f(Y: Vector, t: float) -> Vector:
    return Y


def euler(u_i: List[Vector], f: Callable[[Vector, float], Vector], ti: float, h: float, m: int) -> Vector:
    return u_i[0] + h * f(u_i[0], ti)


solution = gs.compute_solution(euler, f, 1000, -10, 3, Vector([Vector([exp(-10)])]), False, 1)
print(len(solution))
intervall = gs.get_intervall(1000, -10, 3)
gs.plot_x_solution(intervall, solution)

error = 0
for i in range(len(intervall)):
    error += (exp(intervall[i]) - solution[i][0][0]) ** 2

print(error)

exact_solution = []
for ti in intervall:
    exact_solution.append(Vector([Vector([exp(ti)])]))
gs.plot_error(solution, exact_solution, intervall)
