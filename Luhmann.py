from utils import Vector
from normalization import NormalizationParameters
import numpy as np


def magnetic_dipole_field_normalized(u, mu_normalized=np.array([0.0, 0.0, 1.0]),
                                     add_tail: bool = False):
    """
    Normalized magnetic field B_norm with optional Luhmann magnetotail term.

    Formula (dipole only):
        B_norm = (1/u³) * [3(μ·û)û - μ]

    Formula (dipole + Luhmann tail, eq. 1 du papier):
        B_norm = (1/u³) * [3(μ·û)û - μ] + BT * sign(μ·û) * x̂

    Where BT = 0.00015 est directement dans le système normalisé de Luhmann (en RT),
    cohérent avec μ = 0.31 et r en RT.

    Parameters:
    -----------
    u : np.ndarray or Vector
        Normalized position vector (en RT)
    mu_normalized : np.ndarray
        Unit magnetic moment vector (default: z-axis [0, 0, 1])
    add_tail : bool
        If True, adds the Luhmann magnetotail term BT * x̂

    Returns:
    --------
    np.ndarray
        Normalized magnetic field vector
    """
    if isinstance(u, Vector):
        u = np.array(u.coordinates)
    if isinstance(mu_normalized, Vector):
        mu_normalized = np.array(mu_normalized.coordinates)

    u_mag_sq = np.dot(u, u)
    if u_mag_sq < 1e-20:
        return np.array([0.0, 0.0, 0.0])

    u_mag = np.sqrt(u_mag_sq)
    u_cubed = u_mag_sq * u_mag
    u_hat = u / u_mag

    # Terme dipolaire normalisé
    mu_dot_u_hat = np.dot(mu_normalized, u_hat)
    B_field = (1.0 / u_cubed) * (3.0 * mu_dot_u_hat * u_hat - mu_normalized)

    # Terme queue de Luhmann (eq. 1 du papier Luhmann & Friesen 1979)
    if add_tail:
        BT = 0.00015  # dans le système normalisé de Luhmann (RT), sans conversion nécessaire

        # Hémisphère selon la projection sur l'axe du dipôle
        # BT > 0 hémisphère nord (μ·û > 0), BT < 0 hémisphère sud (μ·û < 0)
        hemisphere = np.sign(mu_dot_u_hat)

        x_hat = np.array([1.0, 0.0, 0.0])  # direction vers le Soleil

        B_field = B_field + hemisphere * BT * x_hat

    return B_field
