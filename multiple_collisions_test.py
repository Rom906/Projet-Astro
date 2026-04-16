from generate_solutions import compute_solution
from normalization import NormalizationParameters
from integration_functions import RK4
from utils import Vector
from constants import RT, mp, MO, qe, mu


speed = Vector([0.05, 0.05, 0.05])
parameters = NormalizationParameters(1, 1, 1, 1)
