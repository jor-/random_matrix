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


def symmetric_matrix_with_eigenvalues(n, eigenvalue_lower_bound, eigenvalue_upper_bound, at_least_one_negative_eigenvalue=False, at_least_one_positive_eigenvalue=False, at_least_one_zero_eigenvalue=False):
    eigenvalues = np.random.uniform(eigenvalue_lower_bound, eigenvalue_upper_bound, n)
    m = at_least_one_negative_eigenvalue + at_least_one_positive_eigenvalue + at_least_one_zero_eigenvalue
    if m > 0:
        r = np.random.choice(n, size=m, replace=False)
        if at_least_one_zero_eigenvalue:
            eigenvalues[r[0]] = 0
            r = r[1:]
        if at_least_one_negative_eigenvalue:
            if eigenvalue_lower_bound > 0:
                raise ValueError(f'Lower bound {eigenvalue_lower_bound} is positive but at least one negative eigenvalue should be used.')
            eigenvalues[r[0]] = np.random.uniform(eigenvalue_lower_bound, 0, 1)
            r = r[1:]
        if at_least_one_positive_eigenvalue:
            if eigenvalue_upper_bound < 0:
                raise ValueError(f'Upper bound {eigenvalue_upper_bound} is negative but at least one positive eigenvalue should be used.')
            eigenvalues[r[0]] = np.random.uniform(0, eigenvalue_upper_bound, 1)
            r = r[1:]

    assert np.all(eigenvalues >= eigenvalue_lower_bound)
    assert np.all(eigenvalues <= eigenvalue_upper_bound)
    assert not at_least_one_zero_eigenvalue or 0 in eigenvalues
    assert not at_least_one_negative_eigenvalue or np.any(eigenvalues < 0)
    assert not at_least_one_positive_eigenvalue or np.any(eigenvalues > 0)
    return random_matrix.random.symmetric_matrix_with_eigenvalues(eigenvalues)
