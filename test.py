from numerical_methods import RK4
from fonctions import plot_3d, compute_solution
from normalized_equations import (
    NormalizationParameters,
    create_normalized_differential_equation,
)
from utils import Vector
import numpy as np

# ===== Physical Parameters =====
R0 = 6.371e6  # Earth radius [m]
q_over_m = 1.76e11  # Charge-to-mass ratio for proton [C/kg]
mu0_over_4pi = 1e-7  # μ₀/(4π) [T·m³/A]
m_oplus = 7.91e15  # Earth's magnetic moment [A·m²]

# Initialize normalization parameters
norm_params = NormalizationParameters(R0, q_over_m, mu0_over_4pi, m_oplus)


# ===== Initial Conditions (Dimensional) =====
# Position: multiple of Earth radius
r0_RT = np.array([-4.0, -1.0, -6.0])
r0_meters = r0_RT * R0

# Velocity scaled by some physical velocity
v0_scale = np.array([0.1, 0.1, 0.1])
v0_characteristic = 1e4  # m/s (characteristic velocity)
v0_ms = v0_scale * v0_characteristic

# Convert to normalized coordinates
r0_norm = norm_params.normalize_position(r0_meters)
v0_norm = norm_params.normalize_velocity(v0_ms)

# Create initial state vector [position, velocity]
Y0 = Vector([r0_norm, v0_norm])

f_normalized = create_normalized_differential_equation(
    mu_direction=np.array([0.0, 0.0, 1.0])
)

# ===== Integration Parameters =====
n_steps = 100000  # Number of integration steps
tau_final = 100.0  # Final normalized time
tau_start = 0.0  # Initial normalized time
dt_norm = tau_final / n_steps  # Normalized time step

# ===== Solve the ODE =====
solutions = compute_solution(
    RK4,  # Integration method
    f_normalized,  # ODE function
    n_steps,  # Number of steps
    tau_start,  # Initial time
    tau_final,  # Final time
    Y0,  # Initial state
    False,  # Verbose output?
    1,  # Output every N steps
)

# ===== Extract and Analyze Trajectory =====
ploted_position = []
r_mags = []
v_mags = []

for i, solution in enumerate(solutions):
    u_norm = solution[0]  # Normalized position
    v_norm = solution[1]  # Normalized velocity

    ploted_position.append(u_norm)

    # Compute magnitude of position
    u_mag = np.sqrt(u_norm[0] ** 2 + u_norm[1] ** 2 + u_norm[2] ** 2)
    v_mag = np.sqrt(v_norm[0] ** 2 + v_norm[1] ** 2 + v_norm[2] ** 2)

    r_mags.append(u_mag)
    v_mags.append(v_mag)

# 3D plot
plot_3d(ploted_position)
