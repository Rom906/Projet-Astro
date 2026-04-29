from utils import Vector
from math import exp, pi
from numpy import random
from constants import RT, kB, e0
from typing import Callable, List, Tuple
from random import uniform
from integration_functions import RK4
from atmospheric_model import concentration_ni, Na, T

molecules_list = ["O", "O2", "H", "HE", "AR", "N2"]

def test_collision(conditions: List[Vector], molecule_index) -> bool:
    n = concentration_ni(abs(conditions[0]), molecule_index)
    s = cross_section(molecules_list[molecule_index], 0)
    v = conditions[1] - draw_maxwell_boltzmann_velocity(mass(molecules_list[molecule_index]), T(abs(conditions[0])))
    collision_rate = n * s * abs(v)
    return uniform(0, 1) < collision_rate.real


def draw_maxwell_boltzmann_velocity(m, T):
    sdev = (kB * T / m) ** (1/2)
    vx = random.normal(loc=0, scale=sdev)
    vy = random.normal(loc=0, scale=sdev)
    vz = random.normal(loc=0, scale=sdev)
    return Vector([vx, vy, vz])


def cross_section(molecule, electron_nrg):
    if molecule == "O":
        return 1.0**(-17) #m² @13.62eV
    elif molecule == "O2":
        return 10**(-19) #m²
    elif molecule == "H":
        return 3.0186 * 10**(-19) #m² @13eV 1s2 -> 1s2p
    elif molecule == "HE":
        return 3.5 * 10**(-17) #m² @20.6eV 1S2 -> 1s2p
    elif molecule == "AR":
        return 2.5 * 10**(-20) #m² @circa 20eV
    elif molecule == "N2":
        return 10**(-20) #m²

def mass(molecule):
    if molecule == "O":
        molar_mass = 15.999 #g/mol
    elif molecule == "O2":
        molar_mass = 29.998 #g//mol
    elif molecule == "H":
        molar_mass = 1.0080 #g/mol
    elif molecule == "HE":
        molar_mass = 4.002602 #g/mol
    elif molecule == "AR":
        molar_mass = 39.95 #g/mol
    elif molecule == "N2":
        molar_mass = 28.014 #g/mol
    return molar_mass * Na 

def field_contribution(q_other, r_self, r_other):
    return (q_other) / (4 * pi * e0 * abs(r_other - r_self)**2) * r_other - r_self
