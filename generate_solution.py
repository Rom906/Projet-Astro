from utils import Vector
from typing import Callable, List
# import seaborn as sb
# import matplotlib.pyplot as plt

differential_equation_type = Callable[[Vector, float], Vector]
model_type = Callable[[List[Vector], differential_equation_type, float, float, int], Vector]


def get_intervall(steps: int, minimum: float, maximum: float) -> List[float]:
    """
    return a liste of the specified number of values evenly spaced between "minimum" and "maximum" included

    :param steps: number of steps
    :type steps: int
    :param minimum: intervall start
    :type minimum: float
    :param maximum: intervall stop
    :type maximum: float
    :return: a list of steps value evenly spaced between "minimum" and "maximum" included
    :rtype: List[float]
    """
    intervall = []
    h = (maximum - minimum) / (steps - 1)
    for i in range(steps):
        intervall.append(minimum + h * i)
    return intervall


def compute_solution(
    model: model_type,
    differential_equation: differential_equation_type,
    steps: int,
    minimum: float,
    maximum: float,
    initial_conditions: Vector,
    multiple_steps_method: bool = False,
    number_of_steps: int = 1
) -> List[Vector]:
    """
    compute an approximated solution of the given differential equation using the given model between min and max in a specified number of steps

    :param model: function representing the model used to approximate the solution. It needs to take a specified amount of previous steps to calculate the next one
    :type model: model_type
    :param differential_equation: represent the differential equation system to approximate. It is a function which represent the f in the equation y' = f(y, t)
    :type differential_equation: differential_equation_type
    :param steps: number of steps used to approximate the solution
    :type steps: int
    :param minimum: the value where we start to compute the approximate solution of the differential equation
    :type minimum: float
    :param maximum: the value where we stop to compute the approximate solution of the differential equation
    :type maximum: float
    :param initial_conditions: the initial values of the differential equation system
    :type initial_conditions: Vector
    :param multiple_steps_method: if true, means that the model used is using multiple steps to compute the solution
    :type multiple_steps_method: bool
    :param number_of_steps: if the method is using multiple steps, it is the maximum number of step used by it
    :type number_of_step: int
    :return: a list of "steps" approximated value of the differential equation solution
    :rtype: List[Vector]
    """
    solution: List[Vector] = [initial_conditions]
    h = (maximum - minimum) / (steps - 1)
    start = minimum + h

    if multiple_steps_method:
        for i in range(1, number_of_steps):
            vector_list = []
            for j in range(i):
                vector_list.append(solution[-1 - i])
            solution.append(model(vector_list, differential_equation, minimum + h * i, h, i))
        start = minimum - h * number_of_steps

    intervall = get_intervall(steps, start, maximum)

    for ti in intervall:
        vector_list = []
        for i in range(number_of_steps):
            vector_list.append(solution[-1 - i])
        solution.append(model(vector_list, differential_equation, ti, h, number_of_steps))
    return solution
