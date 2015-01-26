# Implementation of lattice-based computationally-efficient PIR scheme as described
# here: https://eprint.iacr.org/2007/446.pdf

# Author: Adrian Soghoian

import math
import numpy as np
import random
from lattice_helpers import gen_rand_matrix, reduce_matrix, reduce_vector, matrix_inversion_cofactor, recover_bits


def gen_db(n=2, l=6):
    """
    Constructs a database where n is the number of elements
    and l is the number of bits per element. Elements are padded from the left to
    ensure consistent length bit-wise (default = 6).
    """
    db = []

    for i in range(n):
        element = str(bin(random.getrandbits(l)))[2:]

        while len(element) < l:
            element = "0" + element
        db.append(element)

    return db


def column_permutation(N):
    """
    Returns a random permutation of column indices for a dimension N.
    """
    indices = range(N)
    random.shuffle(indices)

    return [1, 3, 0, 2]
    return indices


def permute_columns(matrix, indices):
    """
    Randomnly permutes the columns of a matrix. Returns both the updated matrix and the
    new column ordering.
    """
    shuffled_matrix = matrix[:, indices[0]:indices[0] + 1]

    for index in indices[1:]:
        column = matrix[:, index:index + 1]
        shuffled_matrix = np.hstack([shuffled_matrix, column])

    return shuffled_matrix


def gen_finite_field(n, N):
    """
    Produces a suitable finite field (p) over which to generate the
    hidden lattice m.

    Also returns parameters (q, l0).
    """
    l0 = int(math.log(n*N, 2) + 1)
    q = 2**(2*l0)
    p = find_prime(2**(3*l0))

    return l0, q, p


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


def gen_scrambler(n, p):
    """
    Generates the random diagonal nxn scrambler matrix over
    finite field Z/pZ.
    """
    scrambler = np.random.random_integers(0, p, size=(n, n))

    return np.diag(np.diag(scrambler))


def gen_noise(N, hard=False, q=None):
    """
    Generates the hard or soft NxN noise matrices over {-1, 1}.
    """
    d = np.empty([N, N], dtype=int)

    for element in np.nditer(d, op_flags=['readwrite']):
        guess = random.random()
        if guess < 0.33:
            element[...] = 1
        elif guess < 0.67:
            element[...] = -1
        else:
            element[...] = 0

    if hard: np.fill_diagonal(d, q)

    return d


def column_reordering(V, column_permutation):
    """
    Re-orders a vector's columns according to a specific permutation.
    """
    output = np.zeros(V.shape, dtype=int)
    for i in range(V.shape[0]):
        position = column_permutation.index(i)
        output[i] = V[position]

    return output


def gen_request(n, N, i0, l0, q, p):
    """
    Generates query to db at index i0 (of n) with lattice parameter
    dimension N*2.
    """
    A = gen_rand_matrix(N, N, is_invertible=True, finite_field=p)
    B = gen_rand_matrix(N, N, is_invertible=False, finite_field=p)
    M = np.hstack([A, B])
    scrambler = gen_scrambler(N, p)
    new_columns = column_permutation(2*N)
    request = []

    for i in range(n):
        rand_matrix = gen_rand_matrix(N, N, is_invertible=True, finite_field=p)

        M11 = reduce_matrix(np.dot(rand_matrix, M), p)

        if i == i0:
            D = gen_noise(N, hard=True, q=q)
        else:
            D = gen_noise(N, hard=False, q=None)

        noise = reduce_matrix(np.dot(D, scrambler), p, is_noise=True)

        M1 = reduce_matrix(np.hstack([M11[:, :M.shape[1] / 2], M11[:, M.shape[1] / 2:] + noise]), p)
        request.append(permute_columns(reduce_matrix(M1, p), new_columns))

    return np.array(request), new_columns, A, B, scrambler


def server_reply(request, db, l0, p):
    """
    Server-side response to client query.
    """
    N = len(db[0]) / l0
    response = np.zeros([N*len(db), 2*N], dtype=int)

    for i, query in enumerate(request):
        for row in range(N):
            db_substring = int(db[i][row*l0:row*l0 + l0], 2)
            response[i*N + row] = reduce_vector(db_substring*request[i][row], p)

    return reduce_vector(np.sum(response, axis=0), p)


def bit_extraction(response, column_permutation, N, A, B, p, q, scrambler, l0):
    """
    Client-side operation to exact relevant bits from server response to query.
    """
    disturbed_vector = column_reordering(response, column_permutation)
    undisturbed_half = disturbed_vector[:len(disturbed_vector) / 2]

    A_inverse = matrix_inversion_cofactor(A, p)
    undisturbed_vector = np.concatenate([undisturbed_half, reduce_vector(np.dot(np.dot(undisturbed_half, A_inverse), B), p)])
    scrambled_noise = reduce_vector(disturbed_vector - undisturbed_vector, p)[N:]
    unscrambled_noise = reduce_vector(np.dot(scrambled_noise, matrix_inversion_cofactor(scrambler, p)), p)
    response = ""

    for each in unscrambled_noise:
        bitstring = str(bin(recover_bits(each, q, p)))[2:]
        while len(bitstring) < l0:
            bitstring = "0" + bitstring
        response = response + bitstring
    return response


if __name__ == "__main__":
    n, N = 8, 2
    l0, q, p = gen_finite_field(n, N)

    i0 = 3
    db = gen_db(n, l0*2)
    print "Target element: ", db[i0]
    print "*"*10

    request, column_permutation, A, B, scrambler = gen_request(n, N, i0, l0, q, p)
    response = server_reply(request, db, l0, p)

    db_element = bit_extraction(response, column_permutation, N, A, B, p, q, scrambler, l0)
    print "Received element: ", db_element
    print "Was this a success?"
    print db_element == db[i0]
