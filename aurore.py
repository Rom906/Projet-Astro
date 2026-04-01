from random import random
from math import exp
from utils import Vector
from constants import RT


def collision(position: Vector, velocity: Vector):
    collision_probability = atm_model(position)["n"] * atm_model(position)["eff_sec"]
    return random() > collision_probability


H0 = 8000  # m
p0 = 130025  # Pa
R = 8.314  # J.K-1.mol-1
T0 = 273.15  # K
eff_sec_n2 = 8e-19  # m²


def atm_model(position: float):
    n = (
        p0 * (1 + exp(-(abs(position) - 1) / H0)) / (R * T0)
    )  # assuming spherical coords
    eff_sec = eff_sec_n2 / RT
    return {"n": n, "eff_sec": eff_sec}
