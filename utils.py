from __future__ import annotations
from typing import Any, List, Iterator
from math import sqrt, sin, cos
from numbers import Real
import csv


class Vector:
    """
    A vector in any dimension with all arithmetics operators
    """

    def __init__(self, coordinates: List[Any]) -> None:
        """
        Initializing method for Vector. Sets its dimension and its coordinates

        :param coordinates: The vector coordinates in the order of the list
        :type coordinates: List[Any]
        """
        if coordinates is not None:
            self.coordinates = coordinates
        else:
            self.coordinates = []
        self.dimension = len(self.coordinates)  # type: ignore

    def set_coordinates(self, coordinates: List[Any]) -> None:
        """
        Sets the coordinates of the current vector

        :param coordinates: The new vector coordinates in the order of the list
        :type coordinates: List[Any]
        """
        self.coordinates = coordinates
        self.dimension = len(coordinates)

    def __getitem__(self, index: int) -> Any:
        """
        Gives the chosen coordinate of the current vector

        :param index: index of the wanted coordinate
        :type index: int
        :return: return the vector coordinate at the chosen index
        :rtype: Any
        """
        return self.coordinates[index]

    def __setitem__(self, index: int, value: Any) -> None:
        """
        Sets the vector coordinate at the chosen index

        :param index: the index of the coordinate you are changing
        :type index: int
        :param value: the new value of the chosen coordinate
        :type value: Any
        """
        self.coordinates[index] = value

    def __add__(self, other: Vector) -> Vector:
        """
        adding two vectors and returning a new one containing the result of the addition. The two vectors need to have the same dimension

        :param other: the other vector to add
        :type other: Vector
        :return: return the sum of the two vectors
        :rtype: Vector
        """
        if self.dimension != other.dimension:
            raise TypeError("The two vectors have not the same dimension")

        coordinates = []
        for i in range(self.dimension):
            coordinates.append(self[i] + other[i])
        return Vector(coordinates)

    def __sub__(self, other: Vector) -> Vector:
        """
        substrating two vectors and returning the result. The two vectors need to have the same dimension

        :param other: the vector to substract
        :type other: Vector
        :return: the vector made from the result
        :rtype: Vector
        """
        if self.dimension != other.dimension:
            raise TypeError("The two vectors have not the same dimension")

        coordinates = []
        for i in range(self.dimension):
            coordinates.append(self[i] - other[i])
        return Vector(coordinates)

    def __neg__(self) -> Vector:
        """
        return the opposite vector

        :return: The opposite vector
        :rtype: Vector
        """
        coordinates = []
        for i in range(self.dimension):
            coordinates.append(-self[i])
        return Vector(coordinates)

    def __abs__(self) -> float:
        """
        norm two of the vector

        :return: the norm of the vector
        :rtype: float
        """
        sum = 0
        for i in range(self.dimension):
            sum += self[i] ** 2
        return sqrt(sum)

    def __mul__(self, other: float | Vector) -> Vector | float:
        """
        depending on the input multiply th vector by a scalar or the dot product with the input. If the two inputs are a vector, they need to have the same dimension

        :param other: either a scalar or a vector to multiply with
        :type other: int | Vector
        :return: the result of the dot product or the multiplication depending on the input
        """
        if isinstance(other, Real):
            coordinates = []
            for i in range(self.dimension):
                coordinates.append(self[i] * other)
            return Vector(coordinates)

        elif isinstance(other, Vector):
            if self.dimension != other.dimension:
                raise TypeError("The two vectors have not the same dimension")
            result = 0
            for i in range(self.dimension):
                result += self[i] * other[i]
            return result

        else:
            raise TypeError("Wrong type input")

    def __matmul__(self, other: Vector) -> Vector:
        """
        Make the cross product of the two vectors if their dimension is three only

        :param other: the vector on the right of the product
        :type other: Vector
        :return: the result of the cross product between the two vectors
        :rtype: Vector
        """
        if self.dimension != 3 or other.dimension != 3:
            raise TypeError("Inputs don't have the good dimension")
        result = Vector([0, 0, 0])
        result[0] = self[1] * other[2] - other[1] * self[2]
        result[1] = self[2] * other[0] - other[2] * self[0]
        result[2] = self[0] * other[1] - other[0] * self[1]
        return result

    def __repr__(self) -> str:
        return f"Vector({self.coordinates})"

    def __len__(self) -> int:
        """
        return the dimension of the vector

        :param self: Description
        :return: the dimension
        :rtype: int
        """
        return self.dimension

    def __str__(self) -> str:
        return str(self.coordinates)

    def __eq__(self, other: object) -> bool:
        """
        comparate the two vectors

        :param other: the other vector
        :type other: Vector
        :return: True if the two vectors are equals False otherwise
        :rtype: bool
        """
        if isinstance(other, Vector):
            if self.dimension != other.dimension:
                return False
            else:
                for i in range(self.dimension):
                    if self[i] != other[i]:
                        return False
                return True
        else:
            return NotImplemented

    def __iter__(self) -> Iterator[Any]:
        """
        return an iterator to be used in for loops

        :return: an iterator to be used in for loops
        :rtype: Iterator[Any]
        """
        return iter(self.coordinates)

    def __rmul__(self, other: float) -> Vector:
        """
        multiply a constant to the left of a vector : a * vector

        :param other: the constant to multiply
        :type other: float
        :return: the result of the multiplication
        :rtype: Vector
        """
        coordinates = []
        for i in range(self.dimension):
            coordinates.append(other * self[i])
        return Vector(coordinates)

    def __truediv__(self, scalar: float) -> Vector:
        """
        Divide the vector by a scalar

        :param scalar: the scalar you want to divide your vector by
        :type scalar: float
        :return: the result of the division
        :rtype: Vector
        """
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide a vector by zero")
        coordinates = []
        for i in range(self.dimension):
            coordinates.append(self[i] / scalar)
        return Vector(coordinates)

    def normalized(self) -> Vector:
        """
        normalize the vector

        :return: the normalized vector
        :rtype: Vector
        """
        norm = abs(self)
        if norm != 0:
            return self / norm
        else:
            raise ValueError("Cannot normalize a zero vector")

    def copy(self) -> Vector:
        """
        copy the current vector

        :return: return a copy of the vector
        :rtype: Vector
        """
        return Vector(self.coordinates.copy())


def convert_spherical_to_cartesian(sp_vector: Vector) -> Vector:
    """
    take a vector in spherical coordinates and return the same vector but in cartesian
    :param sp_vector: the vector to convert
    :type sp_vector: Vector
    :return: a vector in cartesian coordinates system
    :rtype: Vector
    """
    r = sp_vector[0]
    theta = sp_vector[1]
    phi = sp_vector[2]

    x = r * cos(phi) * sin(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(theta)
    return Vector([x, y, z])


def save_to_csv(vector_list: List[Vector], save_name: str) -> None:
    f = open(save_name, "x", newline="")
    writer = csv.writer(f)
    coordinate_list = []
    for vector in vector_list:
        coordinate = []
        for i in range(vector.dimension):
            coordinate.append(vector[i])
        coordinate_list.append(coordinate)
    writer.writerows(coordinate_list)
    f.close()


def load_from_csv(save_name: str) -> List[Vector]:
    vector_list = []
    f = open(save_name, "r", newline="")
    reader = csv.reader(f)
    for row in reader:
        coordinates = []
        for coordinate in row:
            coordinates.append(float(coordinate))
        vector_list.append(Vector(coordinates))
    f.close()
    return vector_list


def save_time_interval(time_interval: List[float], save_name: str) -> None:
    f = open(save_name, "x", newline="")
    writer = csv.writer(f)
    writer.writerow(time_interval)
    f.close()


def load_time_interval(save_name: str) -> List[float]:
    f = open(save_name, 'r', newline="")
    reader = csv.reader(f)
    time_intervall = []
    for value in reader:
        time_intervall.append(float(value[0]))
    return time_intervall
