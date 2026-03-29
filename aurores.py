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
from Integrate_fonctions import RK4
from generate_solutions import plot_3d

def collision_test(position: Vector, n: float):
    return random * n > 0.5

def atm_model(altitude: float):
    n = 1/altitude
    return n

def compute_collisional_trajectory(initial_conditions: Vector, differential_equation, h, max_collisions):
    n_collisions = 0
    solution = [initial_conditions]
    t = 0
    collision_points = []
    while n_collisions < max_collisions:
        new = RK4(solution[-1], differential_equation, t, h, 1)
        t += h
        if collision_test(new[:3], atm_model(1)):
            collision_points.append(new[0:3])
            for i in range(3, 6):
                new[-1][i] = -new[-1][i]
            
    return solution