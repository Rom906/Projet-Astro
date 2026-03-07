from fonctions_RK4 import RK4
from fonctions import plot_3d, compute_solution
from normalized_equations import NormalizationParameters, create_normalized_differential_equation
from utils import Vector
import numpy as np

# ===== Normalization Parameters =====
RT = 6.371e6  # Earth radius [m]
u = 1e4       # Characteristic velocity [m/s]
rho = 1.0     # Mass ratio
mu0 = 1.0e-7  # μ₀/(4π)

norm_params = NormalizationParameters(RT, u, rho, mu0)

# ===== Initial Conditions =====
# Dimensional initial conditions
r0_RT = np.array([-4.0, -1.0, -6.0])
r0_meters = r0_RT * RT
v0_scale = np.array([0.1, 0.1, 0.1])
v0_ms = v0_scale * u


# Normalized initial conditions using class methods
r0_norm = norm_params.normalize_position(r0_meters)
v0_norm = norm_params.normalize_velocity(v0_ms)


# Create initial condition vector
vector_CI = Vector([r0_norm, v0_norm])

# ===== Differential Equation =====
print(f"\n=== Differential Equation ===")
print(f"dR̃/dt̃ = Ṽ")
print(f"dṼ/dt̃ = (1/|R̃|³) * (Ṽ × [3(μ·r̂)r̂ - μ])")

f_normalized = create_normalized_differential_equation(norm_params)

# ===== Solve =====
T_dim = 5.400*10**6 # seconds
n_steps = 1000000

solutions = compute_solution(
    RK4, 
    f_normalized, 
    n_steps, 
    0, 
    1000000,
    vector_CI, 
    False, 
    1
)

# ===== Extract and analyze trajectory =====
ploted_position = []
r_mags = []

for i, solution in enumerate(solutions):
    R_norm = solution[0]  # Normalized position
    ploted_position.append(R_norm)
    
    # Compute magnitude
    r_mag = np.sqrt(R_norm[0]**2 + R_norm[1]**2 + R_norm[2]**2)
    r_mags.append(r_mag)

plot_3d(ploted_position)

