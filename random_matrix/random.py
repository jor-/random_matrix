import numpy as np
import scipy.sparse


def matrix(n, m, dense=True, complex_values=False, random_value_function=None):
    if random_value_function is None:
        def random_value_function(*shape):
            return np.random.uniform(-1, 1, shape)

    if dense:
        A = random_value_function(n, m)
        if complex_values:
            A = A + random_value_function(n, m) * 1j
    else:
        density = 0.1
        A = scipy.sparse.rand(n, m, density=density, format='coo', data_rvs=random_value_function)
        if complex_values:
            A.data += random_value_function(len(B.data)) * 1j
    return A


def hermitian_matrix(n, dense=True, complex_values=False, random_value_function=None):
    A = matrix(n, n, dense=dense, complex_values=complex_values,
               random_value_function=random_value_function)
    A = A + A.transpose().conj()

    if complex_values:
        for i in range(n):
            A_ii = A[i, i]
            if A_ii != 0:
                A[i, i] = A_ii.real

    assert np.all(np.isreal(A.diagonal()))
    return A
