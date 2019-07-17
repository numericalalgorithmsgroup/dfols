import numpy as np
import math

def make_uniform_sketching_matrix(q, m):
    ones = np.random.choice(range(m), size=q, replace=False)

    S = np.zeros((q,m))

    for index, row in enumerate(S):
        row[ones[index]] = 1

    return S

def make_hashing_sketching_matrix(q, m):
    collisions = np.random.choice(range(q), size=m-q)

    buckets = np.concatenate((range(q), collisions), axis=None)
    buckets = np.random.permutation(buckets)

    S = np.zeros((q,m))

    for index, bucket in enumerate(buckets):
        S[bucket, index] = np.random.choice([1, -1], 1)

    return S

def make_gaussian_sketching_matrix(q, m):
    return np.random.normal(size=(q, m))/math.sqrt(q)

def sketch(r, sketch_dim, method=None):
    m, _ = np.shape(r)

    assert method is not None, 'please supply sketching method'

    if method == 'uniform':
        S = make_uniform_sketching_matrix(sketch_dim, m)
    elif method == 'hashing':
        S = make_hashing_sketching_matrix(sketch_dim, m)
    elif method == 'gaussian':
        S = make_gaussian_sketching_matrix(sketch_dim, m)
    else:
        raise Exception('Invalid sketching method passed to sketcher.')

    return S.dot(r)