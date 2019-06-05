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


def lower_triangle_matrix(n, dense=True, complex_values=False, random_value_function=None):
    A = matrix(n, n, dense=dense, complex_values=complex_values,
               random_value_function=random_value_function)
    if dense:
        A = np.tril(A)
    else:
        A = scipy.sparse.tril(A)
    return A


def orthogonal_matrix(n):
    """
    reference:
    G.W. Stewart,
    The efficient generation of random orthogonal matrices
    with an application to condition estimators,
    SIAM J. Numer. Anal., 17 (1980), 403-409.
    """

    A = np.eye(n)
    d = np.zeros(n)

    def sign_without_zero(v):
        return np.sign(v) + (v == 0)

    for k in range(n - 2, -1, -1):
        # Generate random Householder transformation.
        x = np.random.randn(n - k)
        s = np.linalg.norm(x)
        sgn = sign_without_zero(x[0])
        s = sgn * s
        d[k] = -sgn
        x[0] = x[0] + s
        beta = s * x[0]
        # Apply the transformation to A.
        y = x.T @ A[k:n]
        A[k:n] = A[k:n] - np.outer(x, y / beta)

    for i in range(0, n - 1):
        A[i] = d[i] * A[i]

    sgn = sign_without_zero(np.random.randn(1))
    A[n - 1] = A[n - 1] * sgn

    assert np.allclose(A @ A.T, np.eye(n))
    assert np.allclose(A.T @ A, np.eye(n))
    return A
