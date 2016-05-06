cimport cython
cimport numpy as np
import numpy as np

np.import_array()

from libc.math cimport exp, fabs, log
from numpy.math cimport EULER


@cython.boundscheck(False)
@cython.wraparound(False)
def mean_change(np.ndarray[ndim=1, dtype=np.float64_t] arr_1,
                np.ndarray[ndim=1, dtype=np.float64_t] arr_2):
    """Calculate the mean difference between two arrays.
    Equivalent to np.abs(arr_1 - arr2).mean().
    """

    cdef np.float64_t total, diff
    cdef np.npy_intp i, size

    size = arr_1.shape[0]
    total = 0.0
    for i in range(size):
        diff = fabs(arr_1[i] - arr_2[i])
        total += diff

    return total / size

@cython.boundscheck(False)
@cython.wraparound(False)
def mean_change_2d(np.ndarray[ndim=2, dtype=np.float64_t] arr_1,
                   np.ndarray[ndim=2, dtype=np.float64_t] arr_2):
    """
    Calculate absolute mean change, but for matrices
    """

    cdef np.float64_t total, diff
    cdef np.npy_intp i, j, m, n

    m = arr_1.shape[0]
    n = arr_1.shape[1]

    total = 0.0

    for i in range(m):
        for j in range(n):
            diff = fabs(arr_1[i,j] - arr_2[i,j])
            total += diff

    return total / (m * n)

@cython.boundscheck(False)
@cython.wraparound(False)
def gamma_update(np.ndarray[ndim=1, dtype=np.float64_t] alpha,
                 np.ndarray[ndim=2, dtype=np.float64_t] phi):
    """ Not efficient! Do not Use
    """
    cdef np.npy_intp i, j, n, m
    cdef np.float64_t temp_sum

    n = alpha.shape[0]
    m = phi.shape[1]

    cdef np.ndarray[ndim=1, dtype=np.float64_t] sum = np.empty(n)

    for i in range(n):
        temp_sum = alpha[i]
        for j in range(m):
            temp_sum += phi[i,j]
        sum[i] = temp_sum

    return sum[i]

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def psi_sumpsi(np.ndarray[ndim=1, dtype=np.float64_t] gammad,
#                np.ndarray[ndim=1, dtype=np.float64_t] out):
#
#     cdef np.npy_intp i, n
#     cdef np.float64_t total
#
#     n = gammad.shape[0]
#
#     for i in range(n):
#         total += gammad[i]
