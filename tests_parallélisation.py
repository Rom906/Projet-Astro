from numba import cuda
import math as m
import numpy as np


phi = 11.70 * m.pi / 180.0  # Magnetic dipole tilt [rad]
theta = 23.5 * m.pi / 180  # Magnetic dipole theta angle
mux, muy, muz = m.cos(phi) * m.sin(theta), m.sin(phi) * m.sin(theta), m.cos(theta)  # Earth's magnetic moment [A m2]


@cuda.jit(device=True)
def cross(ax, ay, az, bx, by, bz):
    cx = ay * bz - by * az
    cy = az * bx - bz * ax
    cz = ax * by - bx * ay
    return cx, cy, cz


@cuda.jit(device=True)
def normalized(ax, ay, az):
    norme = (ax ** 2 + ay ** 2 + az ** 2) ** 0.5
    return ax / norme, ay / norme, az / norme


@cuda.jit(device=True)
def dot(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@cuda.jit(device=True)
def norme(ax, ay, az):
    return (ax ** 2 + ay ** 2 + az ** 2) ** 0.5


@cuda.jit(device=True)
def equa_diff(vx, vy, vz, rx, ry, rz, mux, muy, muz):
    drx = vx
    dry = vy
    drz = vz
    r_hatx, r_haty, r_hatz = normalized(rx, ry, rz)
    r = norme(rx, ry, rz)
    mu_dot_r_hat = dot(mux, muy, muz, r_hatx, r_haty, r_hatz)
    dvx, dvy, dvz = cross((1 / (r ** 3)) * vx, (1 / (r ** 3)) * vy, (1 / (r ** 3)) * vz, 3 * mu_dot_r_hat * r_hatx - mux, 3 * mu_dot_r_hat * r_haty - muy, 3 * mu_dot_r_hat * r_hatz - muz)

    return dvx, dvy, dvz, drx, dry, drz


@cuda.jit(device=True)
def euler(vx, vy, vz, rx, ry, rz, mux, muy, muz, h):
    dvx, dvy, dvz, drx, dry, drz = equa_diff(vx, vy, vz, rx, ry, rz, mux, muy, muz)
    vx_1 = vx + dvx * h
    vy_1 = vy + h * dvy
    vz_1 = vz + h * dvz
    rx_1 = rx + h * drx
    ry_1 = ry + h * dry
    rz_1 = rz + h * drz
    return vx_1, vy_1, vz_1, rx_1, ry_1, rz_1


@cuda.jit
def compute_sol_euler(v, r, traj, h, mux, muy, muz):
    i = cuda.grid(1)
    if i >= r.shape[0]:
        return

    rx, ry, rz = r[i][0], r[i][1], r[i][2]
    vx, vy, vz = v[i][0], v[i][1], v[i][2]
    traj[i][0][0] = vx
    traj[i][0][1] = vy
    traj[i][0][2] = vz
    traj[i][0][3] = rx
    traj[i][0][4] = ry
    traj[i][0][5] = rz
    for s in range(1, traj.shape[1]):
        vx, vy, vz, rx, ry, rz = euler(vx, vy, vz, rx, ry, rz, mux, muy, muz, h)
        traj[i][s][0] = vx
        traj[i][s][1] = vy
        traj[i][s][2] = vz
        traj[i][s][3] = rx
        traj[i][s][4] = ry
        traj[i][s][5] = rz


N_particules = 200
N_steps = 200001
time = 2000

trajectory = np.zeros([N_particules, N_steps, 6], np.float32)
r = np.ones(shape=[N_particules, 3])
v = np.random.uniform(-0.05, 0.05, [N_particules, 3])
threads_per_block = 256
blocks = (N_particules + threads_per_block - 1) // threads_per_block
h = time / (N_steps - 1)

gpu_r = cuda.to_device(r)
gpu_v = cuda.to_device(v)
gpu_traj = cuda.to_device(trajectory)

compute_sol_euler[blocks, threads_per_block](gpu_v, gpu_r, gpu_traj, h, mux, muy, muz)

trajectory = gpu_traj.copy_to_host()
