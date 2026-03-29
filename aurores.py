from utils import Vector
from constants import RT
from scientific_notation import ScientificNotation
from typing import Callable, List, Tuple
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from math import pi, cos, sin
import time
import numpy as np
from random import random
from integration_functions import RK4
from generate_solutions import plot_3d

def collision_test(position: Vector):
    return random * atm_model(position)["n"] > 0.5

def atm_model(position: float):
    n = 1/position[0]
    return {"n": n}

def compute_collisional_trajectory(initial_conditions: Vector, differential_equation, h, max_collisions):
    n_collisions = 0
    solution = [initial_conditions]
    collision_positions = []
    t = 0
    while n_collisions < max_collisions:
        new = RK4(solution[-1], differential_equation, t, h, 1)
        t += h
        if collision_test(new[0]): #assuming spherical coords
            for i in range(3):
                new[1][i] = -new[1][i]
            collision_positions.append(new[0])
        solution.append(new)

    return solution, collision_positions
