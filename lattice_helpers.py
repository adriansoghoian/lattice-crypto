# Efficient nonsingular matrix generation over finite fields
# Based on this paper by Dana Randall: http://www.eecs.berkeley.edu/Pubs/TechRpts/1991/CSD-91-658.pdf

import numpy as np
import random


def gen_rand_matrix(n, m, is_invertible=False, finite_field=521):
    """
    Generates a random n x m matrix, with an optional flag for
    invertible matrices (even though it's rare to randomnly generate
    a singular matrix).
    """
    if is_invertible:
        return gen_nonsingular_matrix(n, finite_field)
    else:
        return np.random.random_integers(0, finite_field, size=(n, m))


def gen_nonsingular_matrix(n, p):
    """
    Generates an nxn random nonsingular matrix over the finite field Z/pZ.

    Works by recursing over the construction of two nxn matrices and returning
    their product.
    """
    A = np.random.random_integers(0, p, size=(n, n))
    T = np.random.random_integers(0, p, size=(n, n))

    def generate(A, T, n, p):
        if n == 1:
            A[0][0] = 1
            T[0][0] = int(random.random()*(p - 1) + 1)
        else:
            is_zero = True
            while is_zero:
                v = np.random.random_integers(0, p, size=(1, n))
                for i, element in enumerate(v[0]):
                    if element != 0:
                        r = i
                        is_zero = False
                        break

            e = np.zeros((1, n), dtype=int)
            e[0][r] = 1

            A[0] = e[0]
            T[r] = v[0]

            for i in range(n):
                if i == r: continue
                T[i][r] = 0

            A = reduce_matrix(A, p)
            T = reduce_matrix(T, p)
            generate(submatrix(A, 0, r), submatrix(T, r, r), n - 1, p)

    generate(A, T, n, p)
    return reduce_matrix(np.dot(A, T), p)


def reduce_vector(V, p):
    """
    Reduces a vector over finite field.
    """
    for i, element in enumerate(V):
        V[i] = element % p

    return V


def reduce_matrix(M, p, is_noise=False):
    """
    Reduces an NxN matrix over finite field Z/pZ.
    """
    for i, row in enumerate(M):
        for j, element in enumerate(row):
            if is_noise:
                if element < 0 and abs(element) < p:
                    continue
            M[i][j] = element % p

    return M


def submatrix(matrix, i, j):
    """
    Returns the submatrix when the ith row and jth columns are removed.
    """
    return matrix[np.array(range(i) + range(i + 1,matrix.shape[0]))[:, np.newaxis],
               np.array(range(j) + range(j + 1,matrix.shape[1]))]


def find_prime(n):
    """
    Finds a prime greater than n. In this case, it finds the first prime
    greater than n.
    """
    primes = [3]
    candidate = 5

    while primes[-1] < n:
        is_prime = True
        for prime in primes:
            if candidate % prime == 0:
                is_prime = False
                continue

        if is_prime: primes.append(candidate)
        candidate += 2

    return primes[-1]


def matrix_minor(matrix, i, j):
    """
    Returns the matrix(i, j) minor: http://en.wikipedia.org/wiki/Minor_%28linear_algebra%29
    """
    reduced = submatrix(matrix, i, j)

    return int(round(np.linalg.det(reduced)))


def adjoint_matrix(matrix):
    """
    Determines the adjoint / adjugate matrix of a given square matrix.

    Adjoint matrices: http://en.wikipedia.org/wiki/Adjugate_matrix
    """
    adjoint_matrix = np.empty(matrix.shape, dtype=int)
    matrix = matrix.T

    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            minor = matrix_minor(matrix, i, j)
            adjoint_matrix[i][j] = minor*(-1)**(i + j)

    return adjoint_matrix


def modular_inverse(a, m):
    """
    Finds the inverse of a mod m.
    """
    gcd, x, y = extended_euclidean(a, m)
    if gcd != 1:
        return None
    else:
        return x % m


def extended_euclidean(a, b):
    """
    Recursive extended Euclidean. Returns a triple (gcd, x, y) of form ax + by = g = gcd(a, b).
    """
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = extended_euclidean(b % a, a)
        return (g, x - (b // a) * y, y)


def matrix_inversion_cofactor(matrix, p):
    """
    Matrix inversion by applying (1/det(A)) to the transpose of the cofactor matrix (adjoint matrix).
    """
    determinant = modular_inverse(int(round(np.linalg.det(matrix))) % p, p)
    adjoint = adjoint_matrix(matrix)

    return reduce_matrix(determinant*adjoint, p)


if __name__ == "__main__":
    n = 2
    p = 521
    orig = gen_nonsingular_matrix(n, p)
    inverse = matrix_inversion_cofactor(orig, 521)

    print orig
    print inverse
    print reduce_matrix(np.dot(orig, inverse), 521)

    print recover_bits(314, 64, 521)
