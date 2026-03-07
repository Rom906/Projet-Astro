"""
Normalized (dimensionless) equations module.

This module provides functions to work with dimensionless equations as derived in the notes.
Variables are normalized with respect to a characteristic length R0 and characteristic velocity u.

Normalization:
- Spatial coordinate: R_dim = R0 * R_norm
- Velocity: V_dim = u * V_norm
- Time: t_dim = (R0 / u) * t_norm
- Magnetic moment: mu_dim = mu0_dim / rho  where rho = m / m_phi
"""

import numpy as np
from utils import Vector
from math import pi


class NormalizationParameters:
    """
    Container for normalization parameters.

    Attributes:
    -----------
    R0 : float
        Characteristic length (typically Earth radius in meters)
    u : float
        Characteristic velocity (typically in m/s)
    rho : float
        Mass ratio = m_particle / m_reference
    mu0_dim : float
        Dimensional magnetic moment (μ0 / 4π)
    mu0_norm : float
        Normalized magnetic moment
    """

    def __init__(self, R0: float, u: float, rho: float, mu0_dim: float):
        """
        Initialize normalization parameters.

        Parameters:
        -----------
        R0 : float
            Characteristic length [meters]
        u : float
            Characteristic velocity [m/s]
        rho : float
            Mass ratio m_particle / m_reference
        mu0_dim : float
            Dimensional constant μ0 / 4π
        """
        self.R0 = R0
        self.u = u
        self.rho = rho
        self.mu0_dim = mu0_dim

        # Normalized magnetic moment: mu_norm = mu_dim * rho / (mu0 * R0^3 * u^2)
        self.mu0_norm = 1.0  # Will be scaled appropriately in B_normalized

    def dimensionalize_position(self, R_norm) -> Vector:
        """Convert normalized position to dimensional. Accepts Vector or array."""
        if isinstance(R_norm, Vector):
            return Vector([R_norm[i] * self.R0 for i in range(3)])
        else:
            return Vector((R_norm * self.R0).tolist())

    def normalize_position(self, R_dim) -> Vector:
        """Convert dimensional position to normalized. Accepts Vector or array."""
        if isinstance(R_dim, Vector):
            return Vector([R_dim[i] / self.R0 for i in range(3)])
        else:
            return Vector((R_dim / self.R0).tolist())

    def dimensionalize_velocity(self, V_norm) -> Vector:
        """Convert normalized velocity to dimensional. Accepts Vector or array."""
        if isinstance(V_norm, Vector):
            return Vector([V_norm[i] * self.u for i in range(3)])
        else:
            return Vector((V_norm * self.u).tolist())

    def normalize_velocity(self, V_dim) -> Vector:
        """Convert dimensional velocity to normalized. Accepts Vector or array."""
        if isinstance(V_dim, Vector):
            return Vector([V_dim[i] / self.u for i in range(3)])
        else:
            return Vector((V_dim / self.u).tolist())

    def dimensionalize_time(self, t_norm: float) -> float:
        """Convert normalized time to dimensional."""
        return t_norm * (self.R0 / self.u)

    def normalize_time(self, t_dim: float) -> float:
        """Convert dimensional time to normalized."""
        return t_dim * (self.u / self.R0)


def B_normalized(R_norm, mu_norm, R0_norm=np.array([0.0, 0.0, 0.0]), mu0_norm=1.0):
    """
    Normalized magnetic dipole field.

    B_norm = (mu0_norm / |R - R0|^3) * [3(mu_norm · n̂)n̂ - mu_norm]

    Parameters:
    -----------
    R_norm : np.ndarray or Vector
        Normalized position vector (3D)
    mu_norm : np.ndarray or Vector
        Normalized magnetic moment vector
    R0_norm : np.ndarray
        Normalized dipole location (default: origin)
    mu0_norm : float
        Normalized magnetic constant

    Returns:
    --------
    B_field : np.ndarray
        Normalized magnetic field
    """
    # Convert Vector to numpy if needed
    if isinstance(R_norm, Vector):
        R_norm = np.array(R_norm.coordinates)
    if isinstance(mu_norm, Vector):
        mu_norm = np.array(mu_norm.coordinates)

    r = R_norm - R0_norm
    rmag = np.linalg.norm(r)

    if rmag < 1e-10:  # Avoid division by zero
        return np.array([0.0, 0.0, 0.0])

    # B_norm = (mu0_norm / r^3) * [3(mu·n̂)n̂ - mu]
    B_field = (mu0_norm / (rmag**3)) * (
        3.0 * r * np.dot(mu_norm, r) / (rmag**2) - mu_norm
    )

    return B_field


def differential_equation_normalized(
    t_norm: float, Y: Vector, params: NormalizationParameters
) -> Vector:
    """
    Normalized differential equation for charged particle motion in magnetic dipole field.

    System:
    dY/dt_norm = [dR/dt_norm, dV/dt_norm]
    dR/dt_norm = V
    dV/dt_norm = (q/mp) * (μ₀_norm / |R|³) * [V × [3(μ·n̂)n̂ - μ]]

    Where:
    - R is normalized position (dimensionless)
    - V is normalized velocity (dimensionless)
    - q/mp is the charge-to-mass ratio (normalized)
    - μ₀_norm is the normalized magnetic constant

    Parameters:
    -----------
    t_norm : float
        Normalized time
    Y : Vector
        State vector [R_norm, V_norm] where each is a Vector of 3D coordinates
    params : NormalizationParameters
        Normalization parameters

    Returns:
    --------
    Vector
        Derivative vector [dR/dt_norm, dV/dt_norm]
    """
    # Extract position and velocity vectors
    R_norm: Vector = Y[0]
    V_norm: Vector = Y[1]

    # Convert to numpy for calculation
    R_array = np.array(R_norm.coordinates)
    V_array = np.array(V_norm.coordinates)

    # Get normalized magnetic field
    B_field = B_normalized(
        R_array, np.array([0.0, 0.0, 1.0])
    )  # Normalized dipole along z
    B_vector = Vector(B_field.tolist())

    # Compute normalized acceleration: a = (q/m_rho) * (V × B)
    # For normalized form, we use normalized parameters
    V_cross_B = V_norm @ B_vector  # Cross product using Vector operator

    # Acceleration in normalized form
    # The factor (q/mp) in normalized form depends on the specific normalization
    # For typical values, we need: a_norm = (q * mu0_norm) / (mp * rho) * (V × B)
    q_over_m_normalized = 0.1  # This would be calculated from actual parameters

    a_normalized = V_cross_B * q_over_m_normalized

    # Return derivatives: [dR/dt, dV/dt]
    return Vector([V_norm, a_normalized])


def create_normalized_differential_equation(params: NormalizationParameters):
    """
    Factory function to create a differential equation function with specific normalization parameters.

    Parameters:
    -----------
    params : NormalizationParameters
        Normalization parameters

    Returns:
    --------
    callable
        Function f(t, Y) -> Vector representing the normalized differential equation
    """

    def f_normalized(t_norm: float, Y: Vector) -> Vector:
        """
        Normalized differential equation with bound parameters.

        Parameters:
        -----------
        t_norm : float
            Normalized time
        Y : Vector
            State vector [R_norm, V_norm]

        Returns:
        --------
        Vector
            Derivative [dR/dt_norm, dV/dt_norm]
        """
        # Extract state (stay in Vector to avoid conversions)
        R_norm: Vector = Y[0]
        V_norm: Vector = Y[1]

        # Compute |R|² and |R| efficiently using Vector operations
        R_squared = R_norm[0] ** 2 + R_norm[1] ** 2 + R_norm[2] ** 2

        if R_squared < 1e-20:  # Avoid division by zero
            a_normalized = Vector([0.0, 0.0, 0.0])
        else:
            # |R|³ for normalization
            R_mag = R_squared**0.5
            R_cubed = R_squared * R_mag

            # Magnetic dipole moment (unit vector along z-axis)
            # μ = [0, 0, 1]
            # So (μ·R̂) = R_z / R_mag
            mu_dot_r_unit = R_norm[2] / R_mag

            # B_norm = (1/R³) * [3(μ·r̂)r̂ - μ]
            # B_norm = (1/R³) * [3(R_z/R)R/R - [0,0,1]]
            coeff = 3.0 * mu_dot_r_unit / R_cubed

            B_x = coeff * R_norm[0]
            B_y = coeff * R_norm[1]
            B_z = (coeff * R_norm[2]) - (1.0 / R_cubed)

            B_norm = Vector([B_x, B_y, B_z])

            # Compute cross product V × B (stays in Vector)
            V_cross_B = V_norm @ B_norm

            # Normalized acceleration: a = ρ * (V × B)
            a_normalized = V_cross_B * params.rho

        # Return derivatives: dR/dt = V, dV/dt = a
        return Vector([V_norm, a_normalized])

    return f_normalized


# Helper function to convert dimensional system to normalized
def convert_to_normalized(
    R_dim: np.ndarray, V_dim: np.ndarray, params: NormalizationParameters
) -> tuple:
    """
    Convert dimensional position and velocity to normalized form.

    Parameters:
    -----------
    R_dim : np.ndarray
        Dimensional position [m]
    V_dim : np.ndarray
        Dimensional velocity [m/s]
    params : NormalizationParameters
        Normalization parameters

    Returns:
    --------
    tuple
        (R_norm, V_norm) - normalized position and velocity
    """
    R_norm = params.normalize_position(R_dim)
    V_norm = params.normalize_velocity(V_dim)
    return R_norm, V_norm


# Function to convert normalized system back to dimensional
def convert_to_dimensional(
    R_norm: np.ndarray, V_norm: np.ndarray, params: NormalizationParameters
) -> tuple:
    """
    Convert normalized position and velocity to dimensional form.

    Parameters:
    -----------
    R_norm : np.ndarray
        Normalized position
    V_norm : np.ndarray
        Normalized velocity
    params : NormalizationParameters
        Normalization parameters

    Returns:
    --------
    tuple
        (R_dim, V_dim) - dimensional position [m] and velocity [m/s]
    """
    R_dim = params.dimensionalize_position(R_norm)
    V_dim = params.dimensionalize_velocity(V_norm)
    return R_dim, V_dim
