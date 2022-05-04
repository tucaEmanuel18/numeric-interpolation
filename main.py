from math import sin, cos
from numpy import poly1d as poly, polyadd, polyval
import copy
import random
import numpy as np

FUNCTION_NUMBER = 0
N = 4


# Function as it is
def real_function(x):
    if FUNCTION_NUMBER == 0:
        return pow(x, 2) - 12 * x + 30
    elif FUNCTION_NUMBER == 1:
        return sin(x) - cos(x)
    else:
        return 2 * pow(x, 3) - 3 * x + 15


def generate_input_values(input_file_path):
    f = open(input_file_path, "r")
    data = f.read().split('\n')
    x0 = float(data[0])
    xn = float(data[1])
    x_set = {x0, xn}
    while len(x_set) < N + 1:
        x_set.add(random.uniform(x0, xn))
    x_list = list(x_set)
    x_list.sort()
    y_list = list()
    for x in x_list:
        y_list.append(real_function(x))

    return x_list, y_list


class Lagrange:
    def __init__(self, input_file_path):
        self.x_list, self.y_list = generate_input_values(input_file_path)
        self.divided_differences = self.compute_divided_differences()
        self.poly_products = self.compute_poly_products()
        self.ln = self.compute_ln()

    def compute_ln(self):
        ln = poly([self.y_list[0]])
        for i in range(1, N + 1):
            ln = polyadd(ln, poly([self.divided_differences[i]]) * self.poly_products[i - 1])
        return ln

    def compute_divided_differences(self):
        dd_list = copy.deepcopy(self.y_list)
        for i in range(1, N + 1):
            for j in reversed(range(i, N + 1)):
                dd_list[j] = (dd_list[j] - dd_list[j - 1]) / (self.x_list[j] - self.x_list[N-i-(N - j)])
        return dd_list

    def compute_poly_products(self):
        pp_list = list()
        pp_list.append(poly([1, -1 * self.x_list[0]]))
        for i in range(1, N):
            pp_list.append(pp_list[i - 1] * poly([1, -1 * self.x_list[i]]))
        return pp_list

    def interpolate(self, value):
        return polyval(self.ln, value)

    def __str__(self):
        return "Lagrange:{\n x_list=" + str(self.x_list) + \
               "\n y_list = " + str(self.y_list) + \
               "\n divided_differences = " + str(self.divided_differences) + \
               "\npoly_products = " + str(self.poly_products) + \
               "\nln = " + str(self.ln) + \
               "\n}"


def lagrange():
    L = Lagrange("./data/input_1.txt")
    print(f"{L}")
    x = 3
    print(f"Interpolate Value for x = {x} -> ln({x}) = {L.interpolate(x)}")
    print(f"Actual function value for x = {x} -> f({x}) = {real_function(x)}")


# Build the X Matrix
def build_x_matrix(m, x_arr):
    output = []
    while len(x_arr) != 0:
        current_x = x_arr.pop(0)
        current_row = []
        for i in range(m, 0, -1):
            current_row.append(pow(current_x, i))
        current_row.append(1)
        output.append(current_row)
    return output


# Compute the a polynom given the formula
def compute_a_polynom(x_matrix, y_arr):
    # Convert to numpy
    np_x_matrix = np.array(x_matrix)
    np_y_arr = np.array(y_arr)
    np_x_matrix_T = np_x_matrix.transpose()
    multiplication = np.dot(np_x_matrix_T, np_x_matrix)
    inverse_multiplication = np.linalg.inv(multiplication)
    first_out_multiplication = np.dot(inverse_multiplication, np_x_matrix_T)
    final_result = np.dot(first_out_multiplication, np_y_arr)
    return final_result


# Horner Scheme for computing the value of x
def horner_apply_polynom(a, x, p):
    a_coeficients = a.tolist()
    current_d = a_coeficients[0]
    for i in range(1, p + 1):
        current_d = current_d * x + a_coeficients[i]
    return current_d


def least_differences():
    # Known Values
    [x1, x2, x4, x5] = [1, 2, 4, 5]
    [y1, y2, y4, y5] = [real_function(x1), real_function(x2), real_function(x4),
                        real_function(x5)]
    # Value for which to Compute the Output
    x3 = 2.25
    # Polynom Grade
    m = 4

    # Compute the arrays out of the Given values
    x_arr = [x1, x2, x4, x5]
    y_arr = [y1, y2, y4, y5]

    # Build the X Matrix
    x_matrix = build_x_matrix(m, x_arr)

    # Build the Polynom aka the A Array Coefficients
    a_polynom = compute_a_polynom(x_matrix, y_arr)
    # Apply Horner Scheme to output the Final Value
    computed_value = horner_apply_polynom(a_polynom, x3, m)
    # Prints
    print(f"Intepolate value = {computed_value}")
    print(f"Real value f({x3}) = {real_function(x3)}")


if __name__ == '__main__':
    lagrange()
    # least_differences()
