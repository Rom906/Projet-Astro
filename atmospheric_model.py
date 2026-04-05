T_inf = 900  # K, can be modified considering different solar activity
g_a = 9.80665  # m.s-2, gravity
R_a = 6356.766  # Km, Terestrial rayon


def g(z: float) -> float:
    """
    gives gravitationnal force at z altitude used in Jacchia 77's model
    :param z: the altitude
    :type z: float
    :rparam: the gravitationnal force
    :rtype: float
    """
    return (g_a * R_a ** 2) / ((R_a + z) ** 2)
