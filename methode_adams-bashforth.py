import typing

# y = alpha f'' + beta f' +  gama f + d
# ti+1 = t0 + h*i = ti + h


def ui_plus_un(
    liste_ui: typing.List['Vector'], f: typing.Callable[[float,'Vector'],'Vector'], ti: float, h: float, m: int
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

    beta = [1, 1 / 2, 1 / 12, 1 / 24]
    alphas = [(1,), (3, -1), (23, -16, 5), (55, -59, 37, -9)]



    ui_plus_1 = list_ui[-1] + beta * h * sum (tout le reste)

    res = u_i + h * beta * sum_ODE

    return res
