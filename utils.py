import numpy as np
import itertools

def fast_search(prob, dtype=np.float32):
    size = len(prob)
    fk = np.zeros((size + 1), dtype=dtype)
    C = np.zeros((size + 1, size + 1), dtype=dtype)
    S = np.empty((2 * size + 1), dtype=dtype)
    S[:] = np.nan
    for k in range(1, 2 * size + 1):
        S[k] = 1./k
    roots = (prob - 1.0) / prob
    for k in range(size, 0, -1):
        poly = np.poly1d(roots[0:k], True)
        factor = np.multiply.reduce(prob[0:k])
        C[k, 0:k+1] = poly.coeffs[::-1]*factor
        for k1 in range(size + 1):
            fk[k] += (1. + 1.) * k1 * C[k, k1]*S[k + k1]
        for i in range(1, 2*(k-1)):
            S[i] = (1. - prob[k-1])*S[i] + prob[k-1]*S[i+1]

    return fk




