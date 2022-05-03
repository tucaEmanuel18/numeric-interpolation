from math import sin, cos
from numpy import poly1d as poly, polyadd, polyval
import copy
import random

N = 4


def generate_input_values(input_file_path):
    def execute_function(function_number, x):
        if function_number == 0:
            return x * x - 12 * x + 30
        elif function_number == 1:
            return sin(x) - cos(x)
        else:
            return 2 * pow(x, 3) - 3 * x + 15

    f = open(input_file_path, "r")
    data = f.read().split('\n')
    x0 = float(data[0])
    xn = float(data[1])
    function_number = int(data[2])
    x_set = {x0, xn}
    while len(x_set) < N + 1:
        x_set.add(random.uniform(x0, xn))
    x_list = list(x_set)
    x_list.sort()
    y_list = list()
    for x in x_list:
        y_list.append(execute_function(function_number, x))

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
                dd_list[j] = (dd_list[j] - dd_list[j - 1]) / (self.x_list[j] - self.x_list[0])
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


if __name__ == '__main__':
    lagrange()
