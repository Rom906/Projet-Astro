import typing

import matplotlib.pyplot as plt

RT = 6371000.0  # Earth radius [m]


def normalize_r_position(rp):
    return rp / RT


def ui_plus_un(
    liste_ui: typing.List["Vector"],
    f: typing.Callable[[float, "Vector"], "Vector"],
    ti: float,
    h: float,
    m: int,
):
    """Calculate the next ui of the list, using the previous ones

    Parameters
    ----------
    liste ui : List['Vector']
        all ui to calculate ui+1, in order (u0,u1,u2 etc...)
    f : function
        ODE (ordinary differential equation) that takes in entry a time and a vector and returns a vector
    ti : float
        temps i avec la relation ti+1 = ti + h
    h : int
        step
    m : int
        number of previous step used"""

    betas = [1, 1 / 2, 1 / 12, 1 / 24]
    alphas = [(1,), (3, -1), (23, -16, 5), (55, -59, 37, -9)]

    beta_choisi = betas[m - 1]
    alpha_choisi = alphas[m - 1]

    sum = 0

    for i in range(1, m + 1):
        alpha_i = alpha_choisi[i]
        u_pred = liste_ui[-i]
        t_pred = ti - (i - 1) * h

        produit = alpha_i * f(t_pred, u_pred)

        sum += produit

    ui_plus_1 = liste_ui[-1] + beta_choisi * h * sum

    return ui_plus_1


# === Trajectory visualization ===


def plot_trajectory_2D(rp):
    plt.plot(rp[:, 0], rp[:, 1])
    plt.xlabel("X (Earth Radii)")
    plt.ylabel("Y (Earth Radii)")
    plt.title("2D Trajectory of Charged Particle")
    plt.grid()
    plt.axis("equal")
    plt.show()


def plot_trajectory_3D(rp):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(rp[:, 0], rp[:, 1], rp[:, 2])
    ax.set_xlabel("X (Earth Radii)")
    ax.set_ylabel("Y (Earth Radii)")
    ax.set_zlabel("Z (Earth Radii)")
    ax.set_title("3D Trajectory of Charged Particle")
    plt.show()
