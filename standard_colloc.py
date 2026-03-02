from utils import Vector
from constants import *
from math import pi
from numpy import linspace
from numpy.polynomial import Polynomial
from matplotlib.pyplot import plot, show


def f(Y):
    """
    Function of the differential equation studied in the dipolar model
    
    :param Y: Container of speed and acceleration in spherical coordinates
    :type Y: List[Vector]
    """
    y0 = Y[0]
    y1 = Y[1]
    y1_norm = y1.normalized()
    f0 = ((qe / mp) * (MO / (pi * (abs(y1) ** 3))) * y0 @ (3 * ((mu * (y1_norm)) * y1_norm - mu)))
    f1 = y0
    return Vector([f0, f1])

def f_der(Y, i, j):
    """
    Computes the jth coord of the ith vector of the derivative of f with respect to the ith coord of the jth vector of Y
    
    :param Y: Container of speed and acceleration in spherical coordinates
    :type Y: List[Vector]
    :param i: The index of the vector concerned in the container (either 0 or 1)
    :type i: int
    :param j: The coordinate which respect to which the derivative is made (either 0, 1 or 2)
    :type j: int
    """
    y0 = Y[0]
    y1 = Y[1]
    y1_norm = y1.normalized()
    if i == 0 and j == 0:
        return -((qe / mp) * (MO / (pi * (abs(y1) ** 4))) * y0 @ (((mu * (y1_norm)) * y1_norm - mu)))[i]
    else:
        return 0

def lagrange_polynomials(roots):
    """
    Computes the Lagrange interpolation polynomials for the chosen collocation points
    
    :param roots: The times of the collocation points
    :type roots: List[float]
    """
    L = []
    for i in range(len(roots)):
        L_i = 1
        for j in range(len(roots)):
            if j != i:
                L_i *= Polynomial.fromroots([roots[j]]) / (roots[i] - roots[j])
        L.append(L_i)
    L_der = [L[i].deriv() for i in range(len(L))]
    L_dder = [L_der[i].deriv() for i in range(len(L_der))]
    return L, L_der, L_dder

def compute_J_inv(f_der, L_der, t1, t2, p1, p2, i, j):
    """
    Computes the inverse of the Jacobian matrix for the collocation system
    
    :param f_der: The derivative of our main ODE's function
    :type f_der: function
    :param L_der: The derivative of the Lagrange polynomials involved in the system
    :type L_der: np.Polynomial
    :param t1: Intermediate collocation time
    :type t1: float
    :param t2: Final collocation time
    :type t2: float
    :param p1: Intermediate collocation polynomial
    :type p1: List[Vector[np.Polynomial]]
    :param p2: Final collocation polynomial
    :type p2: List[Vector[np.Polynomial]]
    :param i: The index of the concerned Vector (either 0 or 1)
    :type i: int
    :param j: The index of the concerned coordinate (either 0, 1 or 2)
    :type j: int
    """
    j00 = L_der[2](t2).item() - f_der(p2, i, j)
    j11 = L_der[1](t1).item() - f_der(p1, i, j)
    j01 = - L_der[2](t1).item()
    j10 = - L_der[1](t2).item()
    det = j00 * j11 - j01 * j10
    j00 = (1/det) * j00
    j01 = (1/det) * j01
    j10 = (1/det) * j10
    j11 = (1/det) * j11
    return [[j00, j01], [j10, j11]]

def compute_F(ui, f, L_der, t1, t2, p1, p2, i, j):
    """
    Computes the evaluation of the collocation system's overal function
    
    :param ui: Initial condition of the collocation
    :type ui: List[Vector]
    :param f: The main ODE's function
    :type f: function
    :param L_der: The derivative of the Lagrange polynomials involved in the system
    :type L_der: np.Polynomial
    :param t1: Intermediate collocation time
    :type t1: float
    :param t2: Final collocation time
    :type t2: float
    :param p1: Intermediate collocation polynomial
    :type p1: List[Vector[np.Polynomial]]
    :param p2: Final collocation pokynomial
    :type p2: List[Vector[np.Polynomial]]
    :param i: The index of the concerned Vector (either 0 or 1)
    :type i: int
    :param j: The index of the concerned coordinate (either 0, 1 or 2)
    :type j: int
    """
    return [ui[i][j] * L_der[0](t1).item() + p1[i][j] * L_der[1](t1).item() + p2[i][j] * L_der[2](t1).item() - f(p1)[i][j], ui[i][j] * L_der[0](t2).item() + p1[i][j] * L_der[1](t2).item() + p2[i][j] * L_der[2](t2).item() - f(p2)[i][j]]

def Newton(ui, N, f_der, L_der, t1, t2):
    """
    Applies Newton's method to solve the collocatio system
    
    :param ui: Initial condition of the collocation
    :type ui: List[Vector]
    :param f: The main ODE's function
    :type f: function
    :param L_der: The derivative of the Lagrange polynomials involved in the system
    :type L_der: np.Polynomial
    :param t1: Intermediate collocation time
    :type t1: float
    :param t2: Final collocation time
    :type t2: float
    :param p1: Intermediate collocation polynomial
    :type p1: List[Vector[np.Polynomial]]
    :param p2: Final collocation pokynomial
    :type p2: List[Vector[np.Polynomial]]
    :param i: The index of the concerned Vector (either 0 or 1)
    :type i: int
    :param j: The index of the concerned coordinate (either 0, 1 or 2)
    :type j: int
    """
    p1 = ui
    p2 = ui
    for _ in range(N):
        new_p1 = [Vector([0] * 3), Vector([0] * 3)]
        new_p2 = [Vector([0] * 3), Vector([0] * 3)]
        for i in range(2):
            for j in range(3):
                J = compute_J_inv(f_der, L_der, t1, t2, p1, p2, i, j)
                F = compute_F(ui, f, L_der, t1, t2, p1, p2, i, j)
                JF = [J[0][0]*F[0] + J[0][1]*F[1], J[1][0]*F[0] + J[1][1]*F[1]]
                new_p1[i][j] = p1[i][j] - JF[0]
                new_p2[i][j] = p2[i][j] - JF[1]
        p1 = new_p1
        p2 = new_p2
    return p1, p2

def find_segment_polynomial(ti, ui, f, h):
    """
    Finds the polynomial to which approximates a segment of the studied intervall
    :param ti: Initial time
    :type ti: float
    :param ui: Initial condition at ti
    :type ui: List[Vector]
    :param f: The main ODE's function
    :type f: function
    :param h: The width of the segment intervall
    :type h: int
    """
    t0 = ti
    t1 = ti + h / 2
    t2 = ti + h
    L, L_der, L_dder = lagrange_polynomials([t0, t1, t2])
    p1, p2 = Newton(ui, 4, f_der, L_der, t1, t2)
    segment_solution = [[], []]
    for i in range(2):
        for j in range(3):
            segment_solution[i].append(ui[i][j] * L[0] + p1[i][j] * L[1] + p2[i][j] * L[2])
    return segment_solution

def standard_colloc(u0, f, t0, h, m):
    """
    Numerically approximates the value of the speed
    :param ti: Initial time
    :type ti: float
    :param ui: Initial condition at ti
    :type ui: List[Vector]
    :param f: The main ODE's function
    :type f: function
    :param h: The width of the segment intervall
    :type h: int
    """
    segment_polynomials = find_segment_polynomial(t0, [u0[0], u0[0]], f, h)[1]
    
    solution = Vector([segment_polynomials[i](t0 + h).item() for i in range(3)])
    return solution

def integrate_standard_colloc(solution, initial_position, dt):
    position = [initial_position]
    for i in range(len(solution)):
        new_r = position[-1][0] + solution[i][0] * dt
        new_theta = position[-1][1] + solution[i][1] * dt
        new_phi = position[-1][2] + solution[i][2] * dt
        position.append(Vector([new_r, new_theta, new_phi]))
    return position
