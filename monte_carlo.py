from utils import Vector
from math import exp, pi
from numpy import random
from constants import RT, kB, e0, qe
from typing import Callable, List, Tuple
from random import uniform
from integration_functions import RK4
from atmospheric_model import concentration_ni, Na, T
import numpy as np
from normalization import convert_to_dimensional, convert_to_normalized

molecules_list = ["O", "O2", "H", "HE", "AR", "N2"]

ENERGY_LOSS = {
    "O": 13.62 * qe,
    "O2": 12.06 * qe, 
    "H": 13.0 * qe,
    "HE": 20.6 * qe,
    "AR": 15.7 * qe,
    "N2": 15.5 * qe
}

def test_collision(conditions: List[Vector], molecule_index) -> bool:
    n = concentration_ni(abs(conditions[0]), molecule_index)
    s = cross_section(molecules_list[molecule_index], 0)
    v = conditions[1] - draw_maxwell_boltzmann_velocity(mass(molecules_list[molecule_index]), T(abs(conditions[0])))
    collision_rate = n * s * abs(v)
    return uniform(0, 1) < collision_rate


def draw_maxwell_boltzmann_velocity(m, T):
    sdev = (kB * T / m) ** (1/2)
    vx = random.normal(loc=0, scale=sdev)
    vy = random.normal(loc=0, scale=sdev)
    vz = random.normal(loc=0, scale=sdev)
    return Vector([vx, vy, vz])


def cross_section(molecule, electron_nrg):
    if molecule == "O":
        return 1.0e-11  # m² @13.62eV (augmenté pour tester)
    elif molecule == "O2":
        return 10**(-13)  # m² (augmenté pour tester)
    elif molecule == "H":
        return 3.0186e-13  # m² @13eV (augmenté pour tester)
    elif molecule == "HE":
        return 3.5e-11  # m² @20.6eV (augmenté pour tester)
    elif molecule == "AR":
        return 2.5e-14  # m² @circa 20eV (augmenté pour tester)
    elif molecule == "N2":
        return 10**(-14)  # m² (augmenté pour tester)

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

def apply_post_collision_velocity(pos_norm: Vector, vel_norm: Vector, molecule_index: int, params) -> Vector:
    """
    Calcule le nouveau vecteur vitesse normalisé après une collision inélastique.
    """
    molecule_name = molecules_list[molecule_index]
    delta_E = ENERGY_LOSS[molecule_name]
    
    # 1. Dénormaliser la vitesse pour faire le calcul d'énergie en unités réelles (m/s)
    # Remarque : on passe pos_norm juste car ta fonction l'exige, mais seule la vitesse nous intéresse ici
    _, vel_real = convert_to_dimensional(pos_norm, vel_norm, params)
    v_mag_real = abs(vel_real)
    
    # 2. Calcul de l'énergie cinétique actuelle (Attention à utiliser la masse de ta particule incidente, ex: me)
    # Je mets la masse utilisée dans tes paramètres de normalisation (qe / ratio)
    masse_particule = qe / params.K1 
    Ec_real = 0.5 * masse_particule * (v_mag_real ** 2)
    
    # 3. Retirer l'énergie du choc
    Ec_new = Ec_real - delta_E
    
    # Si la particule n'a plus d'énergie, elle s'arrête (vitesse nulle)
    if Ec_new <= 0:
        return Vector([0, 0, 0])
        
    # 4. Nouvelle norme de la vitesse
    v_mag_new = np.sqrt(2 * Ec_new / masse_particule)
    
    # 5. Déviation (Scattering isotrope 3D)
    # On tire une nouvelle direction complètement aléatoire sur une sphère
    phi = np.random.uniform(0, 2 * pi)
    costheta = np.random.uniform(-1, 1)
    sintheta = np.sqrt(1 - costheta**2)
    
    vx_new = v_mag_new * sintheta * np.cos(phi)
    vy_new = v_mag_new * sintheta * np.sin(phi)
    vz_new = v_mag_new * costheta
    vel_new_real = Vector([vx_new, vy_new, vz_new])
    
    # 6. Renormaliser la nouvelle vitesse pour la suite de l'intégration RK4
    _, vel_new_norm = convert_to_normalized(pos_norm, vel_new_real, params)
    
    return vel_new_norm
