from utils import Vector
from typing import List, Callable


def euler(liste_ui: List[Vector], f: Callable[[float, Vector], Vector], ti: float, h: float, m: int) -> Vector:
    vector = liste_ui[0]
    ui_plus_un = vector + h * f(ti, vector)
    return ui_plus_un
