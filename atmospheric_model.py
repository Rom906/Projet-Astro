from math import e, exp


T_inf = 900  # K, can be modified considering different solar activity
g_a = 9.80665  # m.s-2, gravity
R_a = 6356.766  # Km, Terestrial rayon
z_0 = 120000  # m, reference height
T_0 = 355  # K, température at the reference height z_0
y = (T_inf - 800) / (750 + 1.722 * (10 ** (-4)) * ((T_inf - 800) ** 2))  # a coefficient
S = 0.0291 * exp((-(y ** 2) / 2))  # a coefficient
sigma = S + 1 / (R_a + z_0)  # a coefficient
a = (T_inf - T_0) / T_inf
R = 8.31
Na = 6.022 * (10 ** 23)


O2 = 0
N2 = 1
HE = 2
O = 3
H = 4
AR = 5


def M_i(molecular_index: int) -> float:
    M = [31.9988, 28.0134, 4.0026, 15.9994, 1.00797, 39.948]
    return M[molecular_index]


def alpha_i(molecular_index: int) -> float:
    alpha = [0, 0, -0.38, 0, -0.25, 0]
    return alpha[molecular_index]


def g(z: float) -> float:
    """
    gives gravitationnal force at z altitude used in Jacchia 77's model
    :param z: the altitude
    :type z: float
    :rparam: the gravitationnal force
    :rtype: float
    """
    return (g_a * R_a ** 2) / ((R_a + z) ** 2)


g_0 = g(z_0)


def C(z: float) -> float:
    return (z - z_0) * (R_a + z_0) / (R_a + z)


def T(z: float) -> float:
    return T_inf - (T_inf - T_0) * e ** (-sigma * C(z))


def gamma_i(molecular_index: int) -> float:
    return M_i(molecular_index) * g_0 / (sigma * R * T_inf)


def concentration_ni_z0(molecular_index: int) -> float:
    ni_z0 = [4.43 * (10 ** 16) / Na, 2.83 * (10 ** 17) / Na, 3.48 * (10 ** 13) / Na, 5.51 * (10 ** 16) / Na, 4.77 ** 12 / Na, 1.42 * (10 ** 15) / Na]  # From NRLMSIS 2.0 nominal 05/04/2026 55°/45°
    return ni_z0[molecular_index]


def concentration_ni(z: float, moleculat_index: int) -> float:
    return concentration_ni_z0(moleculat_index) * (((1 - a) / (1 - a * e ** (-sigma * C(z)))) ** (1 + alpha_i(moleculat_index) + gamma_i(moleculat_index))) * e ** (-sigma * gamma_i(moleculat_index) * C(z))


print(concentration_ni(500000, H) * Na)
