from standard_colloc import integrate_standard_colloc, standard_colloc, f
from generate_solution import compute_solution, plot_3d
from utils import Vector
from math import pi, sin, cos

solution = compute_solution(standard_colloc, f, 1000000000, 0, 1000000, Vector([1, 1, 1]), False, 1), Vector([4, 4, 4]), 0.00001
x, y, z = [], [], []

for i in range(len(solution)):
    r = solution[i][0]
    theta = solution[i][1]
    phi = solution[i][2]
    x.append(r * sin(theta) * cos(phi))
    y.append(r * sin(theta) * sin(phi))
    z.append(r * cos(theta))

postitions = []
for i in range(len(x)):
    postitions.append(Vector([x[i], y[i], z[i]]))
plot_3d(postitions)
