import numpy as np

import random_matrix.random


def _uniformly_distributed_random_value_function(lower_bound, upper_bound):
    def random_value_function(*shape):
        return np.random.uniform(lower_bound, upper_bound, shape)
    return random_value_function


def matrix(n, m, dense=True, complex_values=False, lower_bound=0, upper_bound=1):
    random_value_function = _uniformly_distributed_random_value_function(lower_bound, upper_bound)
    return random_matrix.random.matrix(n, m, dense=dense, complex_values=complex_values, random_value_function=random_value_function)


def hermitian_matrix(n, dense=True, complex_values=False, lower_bound=0, upper_bound=1):
    random_value_function = _uniformly_distributed_random_value_function(lower_bound, upper_bound)
    return random_matrix.random.hermitian_matrix(n, dense=dense, complex_values=complex_values, random_value_function=random_value_function)


def lower_triangle_matrix(n, dense=True, complex_values=False, lower_bound=0, upper_bound=1):
    random_value_function = _uniformly_distributed_random_value_function(lower_bound, upper_bound)
    return random_matrix.random.lower_triangle_matrix(n, dense=dense, complex_values=complex_values, random_value_function=random_value_function)

