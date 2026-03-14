"""
Normalized (Dimensionless) Equations Module

This module implements the normalized differential equation for charged particle motion
in a magnetic dipole field as derived in 'demo_changement_variables_dipole.md'.

NORMALIZED EQUATION:
====================
du'/dτ² = (1/u³) * u' × [3(μ·û)û - μ]

Where:
  - u: normalized position (dimensionless)
  - u' = du/dτ: normalized velocity
  - u'' = d²u/dτ²: normalized acceleration
  - τ: normalized time
  - μ: unit normalized magnetic moment vector (||μ|| = 1)
  - û: unit position vector (u/||u||)
  - u = ||u||: distance from origin

NORMALIZATION SCALES:
=====================
- Length scale: R₀ (Earth radius) 
- Time scale: T = 4πmₚR₀³/(qμ₀m₀)
- Spatial coordinate: r = R₀ * u
- Temporal coordinate: t = T * τ
- Velocity: v = (R₀/T) * u'
"""

import numpy as np
from utils import Vector
from math import pi
from typing import List
from constants import mu


class NormalizationParameters:
    """
    Container for normalization parameters based on Earth's magnetic environment.

    Attributes:
    -----------
    R0 : float
        Length scale - Earth's radius [m]
    T : float
        Time scale = 4πmₚR₀³/(qμ₀m₀) [s]
    q_over_m : float
        Charge-to-mass ratio [C/kg]
    mu0_over_4pi : float
        Normalized magnetic constant μ₀/(4π) [T·m³/A]
    m_oplus : float
        Earth's magnetic moment magnitude [A·m²]
    """

    def __init__(self, R0: float, q_over_m: float, mu0_over_4pi: float, m_oplus: float):
        """
        Initialize normalization from physical parameters.

        Parameters:
        -----------
        R0 : float
            Earth radius [m]
        q_over_m : float
            Charge-to-mass ratio [C/kg]
        mu0_over_4pi : float
            μ₀/(4π) constant [T·m³/A]
        m_oplus : float
            Earth's dipole moment [A·m²]
        """
        self.R0 = R0
        self.q_over_m = q_over_m
        self.mu0_over_4pi = mu0_over_4pi
        self.m_oplus = m_oplus

        # Time scale from normalization: T = 4πmₚR₀³/(qμ₀m₀)
        # Using q/m instead: T = 4πR₀³/((q/m)·μ₀·m₀)
        self.T = 4 * pi * self.R0**3 / (self.q_over_m * 4 * mu0_over_4pi * m_oplus)

    def dimensionalize_position(self, u_norm: Vector | np.ndarray) -> Vector:
        """Convert normalized position u to dimensional r = R₀·u."""
        if isinstance(u_norm, Vector):
            return Vector([u_norm[i] * self.R0 for i in range(3)])
        else:
            return Vector((u_norm * self.R0).tolist())

    def normalize_position(self, r_dim: Vector | np.ndarray) -> Vector:
        """Convert dimensional position r to normalized u = r/R₀."""
        if isinstance(r_dim, Vector):
            return Vector([r_dim[i] / self.R0 for i in range(3)])
        else:
            return Vector((r_dim / self.R0).tolist())

    def dimensionalize_velocity(self, u_prime_norm: Vector | np.ndarray) -> Vector:
        """Convert normalized velocity u' to dimensional v = (R₀/T)·u'."""
        scale = self.R0 / self.T
        if isinstance(u_prime_norm, Vector):
            return Vector([u_prime_norm[i] * scale for i in range(3)])
        else:
            return Vector((u_prime_norm * scale).tolist())

    def dimensionalize_velocity_time_only(self, u_prime_norm: Vector | np.ndarray) -> Vector:
        """Convert normalized velocity u' to dimensional v = (1/T)·u'."""
        scale = 1 / self.T
        if isinstance(u_prime_norm, Vector):
            return Vector([u_prime_norm[i] * scale for i in range(3)])
        else:
            return Vector((u_prime_norm * scale).tolist())

    def normalize_velocity(self, v_dim: Vector | np.ndarray) -> Vector:
        """Convert dimensional velocity v to normalized u' = (T/R₀)·v."""
        scale = self.T / self.R0
        if isinstance(v_dim, Vector):
            return Vector([v_dim[i] * scale for i in range(3)])
        else:
            return Vector((v_dim * scale).tolist())

    def rescale_normalized_time_intervall(self, norm_intervall: List[float]) -> List[float]:
        denormalized_intervall = []
        for i in range(len(norm_intervall)):
            denormalized_intervall.append(norm_intervall[i] * self.T)
        return denormalized_intervall

    def normalize_time_intervall(self, time_intervall: List[float]) -> List[float]:
        norm_intervall = []
        for i in range(len(time_intervall)):
            norm_intervall.append(time_intervall[i] / self.T)
        return norm_intervall


def magnetic_dipole_field_normalized(u, mu_normalized=np.array([0.0, 0.0, 1.0])):
    """
    Normalized magnetic dipole field B_norm.

    Formula from eq. 9 of the derivation:
    B_norm = (1/u³) * [3(μ·û)û - μ]

    Where:
    - u: normalized position vector
    - B_norm: normalized magnetic field
    - μ: unit magnetic moment vector (default: z-axis)
    - û: unit position vector
    - u = ||u||: magnitude of position

    Parameters:
    -----------
    u : np.ndarray or Vector
        Normalized position vector
    mu_normalized : np.ndarray or Vector
        Unit magnetic moment vector (default: [0, 0, 1])

    Returns:
    --------
    np.ndarray
        Normalized magnetic field vector
    """
    # Convert Vector to numpy if needed
    if isinstance(u, Vector):
        u = np.array(u.coordinates)
    if isinstance(mu_normalized, Vector):
        mu_normalized = np.array(mu_normalized.coordinates)

    u_mag_squared = np.dot(u, u)

    if u_mag_squared < 1e-20:  # Avoid division by zero
        return np.array([0.0, 0.0, 0.0])

    u_mag = np.sqrt(u_mag_squared)
    u_cubed = u_mag_squared * u_mag

    # Unit position vector
    u_hat = u / u_mag

    # B_norm = (1/u³) * [3(μ·û)û - μ]
    mu_dot_u_hat = np.dot(mu_normalized, u_hat)
    B_field = (1.0 / u_cubed) * (3.0 * mu_dot_u_hat * u_hat - mu_normalized)

    return B_field


def create_normalized_differential_equation(
    params: NormalizationParameters = None, mu_direction=np.array([0.0, 0.0, 1.0])
):
    """
    Factory function creating the normalized ODE system.

    Returns a function implementing the system:

    du/dτ = v
    dv/dτ = (1/u³) * v × [3(μ·û)û - μ]

    This is the normalized equation from section 9 of the derivation.

    Parameters:
    -----------
    params : NormalizationParameters, optional
        Not used in the normalized system (already dimensionless)
    mu_direction : np.ndarray
        Unit vector for magnetic moment direction (default: z-axis)

    Returns:
    --------
    callable
        Function f(τ, Y) -> Vector where:
        Y = [u, v] is the state vector (position and velocity)
        Returns dY/dτ = [v, acceleration]
    """

    def f_normalized(tau: float, Y: Vector) -> Vector:
        """
        Normalized ODE system from equation 9.

        du/dτ = v
        dv/dτ = (1/u³) * v × [3(μ·û)û - μ]

        Parameters:
        -----------
        tau : float
            Normalized time
        Y : Vector
            State vector [u, v] where u is position, v is velocity
            Both are Vector objects with 3D coordinates

        Returns:
        --------
        Vector
            Derivative [du/dτ, dv/dτ]
        """
        # Extract state components
        u_norm: Vector = Y[0]  # normalized position
        v_norm: Vector = Y[1]  # normalized velocity

        # Convert to numpy for calculations
        u_array = np.array(u_norm.coordinates)
        v_array = np.array(v_norm.coordinates)

        # Compute magnitude and check for singularity
        u_mag_sq = np.dot(u_array, u_array)

        if u_mag_sq < 1e-20:  # Singular point avoided
            acceleration = Vector([0.0, 0.0, 0.0])
        else:
            u_mag = np.sqrt(u_mag_sq)
            u_cubed = u_mag_sq * u_mag

            # Unit position vector: û = u/|||u|||
            u_hat = u_array / u_mag

            # Magnetic field from eq. 9: B = (1/u³)[3(μ·û)û - μ]
            mu_dot_u_hat = np.dot(mu_direction, u_hat)
            B_field = (1.0 / u_cubed) * (3.0 * mu_dot_u_hat * u_hat - mu_direction)

            # Convert to Vector for cross product
            B_vector = Vector(B_field.tolist())

            # Cross product: v × B = (v₁, v₂, v₃) × (B₁, B₂, B₃)
            v_cross_B = v_norm @ B_vector

            # Acceleration is the result (magnitude is already scaled by 1/u³)
            acceleration = v_cross_B

        # Return state derivatives: [du/dτ, dv/dτ]
        return Vector([v_norm, acceleration])

    return f_normalized


def differential_equation_normalized(
    tau: float,
    Y: Vector,
    params: NormalizationParameters = None,
    mu_direction=mu.normalized(),
) -> Vector:
    """
    Direct evaluation of normalized differential equation.

    Implements: du'/dτ² = (1/u³) * u' × [3(μ·û)û - μ]
    As a first-order system:
    du/dτ = v
    dv/dτ = (1/u³) * v × [3(μ·û)û - μ]

    Parameters:
    -----------
    tau : float
        Normalized time
    Y : Vector
        State vector [u, v]
    params : NormalizationParameters, optional
        Normalization parameters (not required for normalized equation)
    mu_direction : np.ndarray
        Magnetic moment direction

    Returns:
    --------
    Vector
        Time derivatives [du/dτ, dv/dτ]
    """
    u_norm: Vector = Y[0]
    v_norm: Vector = Y[1]

    u_array = np.array(u_norm.coordinates)
    v_array = np.array(v_norm.coordinates)

    u_mag_sq = np.dot(u_array, u_array)

    if u_mag_sq < 1e-20:
        return Vector([v_norm, Vector([0.0, 0.0, 0.0])])

    u_mag = np.sqrt(u_mag_sq)
    u_cubed = u_mag_sq * u_mag
    u_hat = u_array / u_mag

    # Compute B_norm
    mu_dot_u_hat = np.dot(mu_direction, u_hat)
    B_field = (1.0 / u_cubed) * (3.0 * mu_dot_u_hat * u_hat - mu_direction)
    B_vector = Vector(B_field.tolist())

    # Compute acceleration
    acceleration = v_norm @ B_vector

    return Vector([v_norm, acceleration])


# ============================================================================
# Coordinate Conversion Utilities
# ============================================================================


def convert_to_normalized(r_dim, v_dim, params: NormalizationParameters):
    """
    Convert dimensional coordinates to normalized form.

    Parameters:
    -----------
    r_dim : np.ndarray or Vector
        Dimensional position [m]
    v_dim : np.ndarray or Vector
        Dimensional velocity [m/s]
    params : NormalizationParameters
        Normalization parameters containing R0 and T

    Returns:
    --------
    tuple
        (u_norm, v_norm) - normalized position and velocity
    """
    u_norm = params.normalize_position(r_dim)
    v_norm = params.normalize_velocity(v_dim)
    return u_norm, v_norm


def convert_to_dimensional(u_norm, v_norm, params: NormalizationParameters):
    """
    Convert normalized coordinates to dimensional form.

    Parameters:
    -----------
    u_norm : np.ndarray or Vector
        Normalized position
    v_norm : np.ndarray or Vector
        Normalized velocity
    params : NormalizationParameters
        Normalization parameters containing R0 and T

    Returns:
    --------
    tuple
        (r_dim, v_dim) - dimensional position [m] and velocity [m/s]
    """
    r_dim = params.dimensionalize_position(u_norm)
    v_dim = params.dimensionalize_velocity(v_norm)
    return r_dim, v_dim


def convert_to_dimensional_time_only(u_norm, v_norm, params: NormalizationParameters):
    """
    Convert normalized coordinates to another normalized form but this time only normalized in space.

    Parameters:
    -----------
    u_norm : np.ndarray or Vector
        Normalized position
    v_norm : np.ndarray or Vector
        Normalized velocity
    params : NormalizationParameters
        Normalization parameters containing R0 and T

    Returns:
    --------
    tuple
        (r_dim, v_dim) - dimensional position [m] and velocity [m/s]
    """
    v_dim = params.dimensionalize_velocity_time_only(v_norm)
    return u_norm, v_dim
