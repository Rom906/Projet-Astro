from verlet import verlet
from normalization import f_normalized
from generate_solution import compute_solution, plot_3d
from utils import Vector
from math import pi, sin, cos

solution = compute_solution(model=verlet, differential_equation=f_normalized, steps=1000, minimum=0, maximum=100, initial_condition=Vector([1, 1, 1]), multiple_steps_method=False, number_of_steps=1)

postitions = []
for i in range(len(solution)):
    postitions.append(soltion[i][1])
plot_3d(postitions)