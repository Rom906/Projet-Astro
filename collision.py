from normalization import NormalizationParameters, convert_to_normalized
from utils import Vector
from numpy import np


def tirage_vitesse_aleatoire(initial_velocity=400000, vitesse_thermique=40000):

    initial_velocity = Vector([initial_velocity, 0, 0])

    vx = np.random.normal(loc=initial_velocity[0], scale=vitesse_thermique)
    vy = np.random.normal(loc=initial_velocity[1], scale=vitesse_thermique)
    vz = np.random.normal(loc=initial_velocity[2], scale=vitesse_thermique)

    vitesse_particule_random = Vector([vx, vy, vz])
    return vitesse_particule_random
